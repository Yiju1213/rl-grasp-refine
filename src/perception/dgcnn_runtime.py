from __future__ import annotations

import copy
from pathlib import Path
from threading import Lock

import torch
import yaml

from src.perception.sga_gsn_runtime import _adapointr_import_context, _resolve_runtime_path
from src.perception.sga_gsn_types import PreparedVTGInputs, SGAGSNInferenceResult


_RUNTIME_CACHE: dict[tuple[str, str, str, str, str], "DGCNNPerceptionRuntime"] = {}
_RUNTIME_CACHE_LOCK = Lock()


def infer_dgcnn_body_feature_dim(runtime_cfg: dict) -> int:
    source_root = runtime_cfg.get("source_root")
    config_path = _resolve_runtime_path(runtime_cfg["config_path"], source_root=source_root)
    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)
    grasp_model_cfg = config_dict.get("grasp_model", {})
    vis_emb_dims = int(grasp_model_cfg.get("VIS_BACKBONE", {}).get("emb_dims", 2048))
    tac_emb_dims = int(grasp_model_cfg.get("TAC_BACKBONE", {}).get("emb_dims", 2048))
    return 2 * vis_emb_dims + 2 * tac_emb_dims


def get_shared_dgcnn_runtime(runtime_cfg: dict) -> "DGCNNPerceptionRuntime":
    source_root = str(_resolve_runtime_path(runtime_cfg["source_root"]))
    config_path = str(_resolve_runtime_path(runtime_cfg["config_path"], source_root=source_root))
    shape_checkpoint = str(_resolve_runtime_path(runtime_cfg["shape_checkpoint"], source_root=source_root))
    grasp_checkpoint = str(_resolve_runtime_path(runtime_cfg["grasp_checkpoint"], source_root=source_root))
    device = str(runtime_cfg.get("device", "cuda:0"))
    cache_key = (source_root, config_path, shape_checkpoint, grasp_checkpoint, device)

    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(cache_key)
        if runtime is None:
            runtime = DGCNNPerceptionRuntime(runtime_cfg)
            _RUNTIME_CACHE[cache_key] = runtime
        return runtime


class DGCNNPerceptionRuntime:
    """Shared frozen runtime for AdaPoinTr VTG-DGCNN inference."""

    def __init__(self, runtime_cfg: dict):
        self.runtime_cfg = dict(runtime_cfg)
        self.source_root = _resolve_runtime_path(runtime_cfg["source_root"])
        self.config_path = _resolve_runtime_path(runtime_cfg["config_path"], source_root=self.source_root)
        self.shape_checkpoint = _resolve_runtime_path(runtime_cfg["shape_checkpoint"], source_root=self.source_root)
        self.grasp_checkpoint = _resolve_runtime_path(runtime_cfg["grasp_checkpoint"], source_root=self.source_root)
        self.device = torch.device(str(runtime_cfg.get("device", "cuda:0")))
        if self.device.type != "cuda":
            raise RuntimeError("DGCNN runtime only supports CUDA devices in this implementation.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for DGCNN runtime, but no CUDA device is available.")

        self.config = None
        self.shape_model = None
        self.grasp_model = None
        self.body_feature_dim = infer_dgcnn_body_feature_dim(runtime_cfg)
        self._load_models()

    def _load_models(self) -> None:
        with _adapointr_import_context(self.source_root):
            from tools import builder
            from utils.config import cfg_from_yaml_file

            self.config = cfg_from_yaml_file(str(self.config_path))
            self.shape_model = builder.model_builder(copy.deepcopy(self.config.shape_model)).to(self.device)
            self.grasp_model = builder.model_builder(copy.deepcopy(self.config.grasp_model)).to(self.device)
            builder.load_model(self.shape_model, str(self.shape_checkpoint))
            builder.load_model(self.grasp_model, str(self.grasp_checkpoint))

        self.shape_model.eval()
        self.grasp_model.eval()
        for model in (self.shape_model, self.grasp_model):
            for parameter in model.parameters():
                parameter.requires_grad = False

    def infer(self, raw_obs, adapter) -> SGAGSNInferenceResult:
        prepared_inputs = adapter.prepare_inputs(raw_obs)
        return self.run_prepared(prepared_inputs)

    def run_prepared(self, prepared_inputs: PreparedVTGInputs) -> SGAGSNInferenceResult:
        sc_input = torch.from_numpy(prepared_inputs.sc_input).unsqueeze(0).to(self.device)
        gs_input = torch.from_numpy(prepared_inputs.gs_input).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if bool(getattr(self.config.ablation, "without_shape_completion", False)):
                raise NotImplementedError("DGCNN without_shape_completion ablation is not supported in this runtime.")
            _, vis_f, vis_coor = self.shape_model(sc_input, return_latent=True)
            body_feature = self._encode_body(vis_f=vis_f, vis_coor=vis_coor, gs_input=gs_input)
            raw_logit = self.grasp_model.predict_head(body_feature)

        return SGAGSNInferenceResult(
            prepared_inputs=prepared_inputs,
            body_feature=body_feature.squeeze(0).detach().cpu().numpy(),
            raw_logit=float(raw_logit.squeeze(0).detach().cpu().item()),
        )

    def _encode_body(self, vis_f: torch.Tensor, vis_coor: torch.Tensor, gs_input: torch.Tensor) -> torch.Tensor:
        model = self.grasp_model
        if vis_coor is None or gs_input is None:
            raise ValueError("DGCNN runtime requires vis_coor and gs_input.")

        _, vis_f = model.vis_grouper(vis_coor, model.vis_proxy_num)
        _, tac_f = model.tac_grouper(gs_input, model.tac_proxy_num)
        vis_points = model._to_channel_first(vis_f, model.vis_grouper.num_features, "vis_f")
        tac_points = model._to_channel_first(tac_f, model.tac_grouper.num_features, "tac_f")

        vis_feat = model.vis_encoder(vis_points)
        tac_feat = model.tac_encoder(tac_points)
        return torch.cat([vis_feat, tac_feat], dim=1)
