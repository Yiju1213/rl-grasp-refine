from __future__ import annotations

import copy
from pathlib import Path
from threading import Lock

import torch
import yaml

from src.perception.cnnmca_types import CNNMCAInferenceResult, PreparedCNNMCAInputs
from src.perception.sga_gsn_runtime import _adapointr_import_context, _resolve_runtime_path


_RUNTIME_CACHE: dict[tuple[str, str, str, str], "CNNMCAPerceptionRuntime"] = {}
_RUNTIME_CACHE_LOCK = Lock()


def infer_cnnmca_body_feature_dim(runtime_cfg: dict) -> int:
    source_root = runtime_cfg.get("source_root")
    config_path = _resolve_runtime_path(runtime_cfg["config_path"], source_root=source_root)
    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)
    grasp_model_cfg = config_dict.get("grasp_model", {})
    fusion_cfg = grasp_model_cfg.get("fusion_transformer", {})
    return int(fusion_cfg.get("embed_dim", 512))


def get_shared_cnnmca_runtime(runtime_cfg: dict) -> "CNNMCAPerceptionRuntime":
    source_root = str(_resolve_runtime_path(runtime_cfg["source_root"]))
    config_path = str(_resolve_runtime_path(runtime_cfg["config_path"], source_root=source_root))
    checkpoint = str(_resolve_runtime_path(runtime_cfg["checkpoint"], source_root=source_root))
    device = str(runtime_cfg.get("device", "cuda:0"))
    cache_key = (source_root, config_path, checkpoint, device)

    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(cache_key)
        if runtime is None:
            runtime = CNNMCAPerceptionRuntime(runtime_cfg)
            _RUNTIME_CACHE[cache_key] = runtime
        return runtime


class CNNMCAPerceptionRuntime:
    """Shared frozen runtime for AdaPoinTr CNNMCA inference."""

    def __init__(self, runtime_cfg: dict):
        self.runtime_cfg = dict(runtime_cfg)
        self.source_root = _resolve_runtime_path(runtime_cfg["source_root"])
        self.config_path = _resolve_runtime_path(runtime_cfg["config_path"], source_root=self.source_root)
        self.checkpoint = _resolve_runtime_path(runtime_cfg["checkpoint"], source_root=self.source_root)
        self.device = torch.device(str(runtime_cfg.get("device", "cuda:0")))
        if self.device.type != "cuda":
            raise RuntimeError("CNNMCA runtime only supports CUDA devices in this implementation.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CNNMCA runtime, but no CUDA device is available.")

        self.config = None
        self.grasp_model = None
        self.body_feature_dim = infer_cnnmca_body_feature_dim(runtime_cfg)
        self.ignored_checkpoint_keys_count = 0
        self._load_model()

    def _load_model(self) -> None:
        with _adapointr_import_context(self.source_root):
            from tools import builder
            from utils.config import cfg_from_yaml_file

            self.config = cfg_from_yaml_file(str(self.config_path))
            model_cfg = copy.deepcopy(self.config.grasp_model)
            self._disable_imagenet_pretrained_init(model_cfg)
            self.grasp_model = builder.model_builder(model_cfg)
            self._load_checkpoint_filtered(self.grasp_model, self.checkpoint)

        self.grasp_model.to(self.device)
        self.grasp_model.eval()
        for parameter in self.grasp_model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _disable_imagenet_pretrained_init(model_cfg) -> None:
        for key in ("visual_extractor", "tactile_extractor"):
            extractor_cfg = getattr(model_cfg, key, None)
            if extractor_cfg is not None:
                extractor_cfg.pretrained = False

    def _load_checkpoint_filtered(self, model, checkpoint_path: Path) -> None:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if not isinstance(checkpoint, dict) or checkpoint.get("base_model") is None:
            raise RuntimeError(f"CNNMCA checkpoint has no base_model weights: {checkpoint_path}")

        filtered_state = {}
        ignored = 0
        for key, value in checkpoint["base_model"].items():
            stripped_key = str(key).replace("module.", "")
            if stripped_key.endswith("total_ops") or stripped_key.endswith("total_params"):
                ignored += 1
                continue
            filtered_state[stripped_key] = value

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        real_missing = [key for key in missing if not (key.endswith("total_ops") or key.endswith("total_params"))]
        real_unexpected = [
            key for key in unexpected if not (key.endswith("total_ops") or key.endswith("total_params"))
        ]
        if real_missing or real_unexpected:
            raise RuntimeError(
                "CNNMCA checkpoint load mismatch after filtering profiling fields: "
                f"missing={real_missing}, unexpected={real_unexpected}."
            )
        self.ignored_checkpoint_keys_count = int(ignored)

    def infer(self, raw_obs, adapter) -> CNNMCAInferenceResult:
        prepared_inputs = adapter.prepare_inputs(raw_obs)
        return self.run_prepared(prepared_inputs)

    def run_prepared(self, prepared_inputs: PreparedCNNMCAInputs) -> CNNMCAInferenceResult:
        visual_img = torch.from_numpy(prepared_inputs.visual_img).unsqueeze(0).to(self.device)
        tactile_img = torch.from_numpy(prepared_inputs.tactile_img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            visual_features = self.grasp_model.visual_extractor(visual_img)
            tactile_features = self.grasp_model.tactile_extractor(tactile_img)
            body_feature = self.grasp_model.fusion_transformer(visual_features, tactile_features)
            raw_logit = self.grasp_model.classifier(body_feature)

        return CNNMCAInferenceResult(
            prepared_inputs=prepared_inputs,
            body_feature=body_feature.squeeze(0).detach().cpu().numpy(),
            raw_logit=float(raw_logit.squeeze(0).detach().cpu().item()),
        )
