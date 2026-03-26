from __future__ import annotations

import copy
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import torch
import yaml

from src.perception.sga_gsn_types import PreparedVTGInputs, SGAGSNInferenceResult


REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_CACHE_KEY = "_sga_gsn_inference_result"

_RUNTIME_CACHE: dict[tuple[str, str, str, str, str], "SGAGSNPerceptionRuntime"] = {}
_RUNTIME_CACHE_LOCK = Lock()


def _resolve_runtime_path(path_value: str | os.PathLike[str], source_root: str | os.PathLike[str] | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    if source_root is not None:
        source_candidate = (Path(source_root).resolve() / path).resolve()
        if source_candidate.exists():
            return source_candidate
    return (REPO_ROOT / path).resolve()


@contextmanager
def _adapointr_import_context(source_root: Path):
    source_root = source_root.resolve()
    source_root_str = str(source_root)
    inserted = False
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)
        inserted = True
    old_cwd = Path.cwd()
    os.chdir(source_root_str)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(source_root_str)
            except ValueError:
                pass


def infer_sga_gsn_body_feature_dim(runtime_cfg: dict) -> int:
    source_root = runtime_cfg.get("source_root")
    config_path = _resolve_runtime_path(runtime_cfg["config_path"], source_root=source_root)
    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)
    grasp_model_cfg = config_dict.get("grasp_model", {})
    embed_dims = grasp_model_cfg.get("feature_fusion", {}).get("embed_dims", 32)
    if isinstance(embed_dims, list):
        return int(embed_dims[-1])
    return int(embed_dims)


def get_shared_sga_gsn_runtime(runtime_cfg: dict) -> "SGAGSNPerceptionRuntime":
    source_root = str(_resolve_runtime_path(runtime_cfg["source_root"]))
    config_path = str(_resolve_runtime_path(runtime_cfg["config_path"], source_root=source_root))
    shape_checkpoint = str(_resolve_runtime_path(runtime_cfg["shape_checkpoint"], source_root=source_root))
    grasp_checkpoint = str(_resolve_runtime_path(runtime_cfg["grasp_checkpoint"], source_root=source_root))
    device = str(runtime_cfg.get("device", "cuda:0"))
    cache_key = (source_root, config_path, shape_checkpoint, grasp_checkpoint, device)

    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(cache_key)
        if runtime is None:
            runtime = SGAGSNPerceptionRuntime(runtime_cfg)
            _RUNTIME_CACHE[cache_key] = runtime
        return runtime


class SGAGSNPerceptionRuntime:
    """Shared frozen runtime for AdaPoinTr + SGSNet inference."""

    def __init__(self, runtime_cfg: dict):
        self.runtime_cfg = dict(runtime_cfg)
        self.source_root = _resolve_runtime_path(runtime_cfg["source_root"])
        self.config_path = _resolve_runtime_path(runtime_cfg["config_path"], source_root=self.source_root)
        self.shape_checkpoint = _resolve_runtime_path(runtime_cfg["shape_checkpoint"], source_root=self.source_root)
        self.grasp_checkpoint = _resolve_runtime_path(runtime_cfg["grasp_checkpoint"], source_root=self.source_root)
        self.device = torch.device(str(runtime_cfg.get("device", "cuda:0")))
        if self.device.type != "cuda":
            raise RuntimeError("SGA-GSN runtime only supports CUDA devices in this implementation.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SGA-GSN runtime, but no CUDA device is available.")

        self.config = None
        self.shape_model = None
        self.grasp_model = None
        self._compute_local_density_knn = None
        self._point_seq_drop_shuf = None
        self.body_feature_dim = infer_sga_gsn_body_feature_dim(runtime_cfg)
        self._load_models()

    def _load_models(self) -> None:
        with _adapointr_import_context(self.source_root):
            from models.GraspStability_PPCT import compute_local_density_knn
            from models.PPCT_utils import point_seq_drop_shuf
            from tools import builder
            from utils.config import cfg_from_yaml_file

            self.config = cfg_from_yaml_file(str(self.config_path))
            self.shape_model = builder.model_builder(copy.deepcopy(self.config.shape_model)).to(self.device)
            self.grasp_model = builder.model_builder(copy.deepcopy(self.config.grasp_model)).to(self.device)
            builder.load_model(self.shape_model, str(self.shape_checkpoint))
            builder.load_model(self.grasp_model, str(self.grasp_checkpoint))
            self._compute_local_density_knn = compute_local_density_knn
            self._point_seq_drop_shuf = point_seq_drop_shuf

        self.shape_model.eval()
        self.grasp_model.eval()
        for model in (self.shape_model, self.grasp_model):
            for parameter in model.parameters():
                parameter.requires_grad = False

    def infer(self, raw_obs, adapter) -> SGAGSNInferenceResult:
        cached = raw_obs.grasp_metadata.get(INFERENCE_CACHE_KEY)
        if isinstance(cached, SGAGSNInferenceResult):
            return cached
        prepared_inputs = adapter.prepare_inputs(raw_obs)
        result = self.run_prepared(prepared_inputs)
        raw_obs.grasp_metadata[INFERENCE_CACHE_KEY] = result
        return result

    def run_prepared(self, prepared_inputs: PreparedVTGInputs) -> SGAGSNInferenceResult:
        sc_input = torch.from_numpy(prepared_inputs.sc_input).unsqueeze(0).to(self.device)
        gs_input = torch.from_numpy(prepared_inputs.gs_input).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            _, vis_f, vis_coor = self.shape_model(sc_input, return_latent=True)
            body_feature = self._encode_body(vis_f=vis_f, vis_coor=vis_coor, tac_xyzc=gs_input)
            raw_logit = self.grasp_model.predict_head(body_feature)

        return SGAGSNInferenceResult(
            prepared_inputs=prepared_inputs,
            body_feature=body_feature.squeeze(0).detach().cpu().numpy(),
            raw_logit=float(raw_logit.squeeze(0).detach().cpu().item()),
        )

    def _encode_body(self, vis_f: torch.Tensor, vis_coor: torch.Tensor, tac_xyzc: torch.Tensor) -> torch.Tensor:
        model = self.grasp_model
        if getattr(model.config, "direct_concat", False):
            raise NotImplementedError("direct_concat ablation is not supported in the bridged runtime.")
        if getattr(model.config, "without_shape_completion", False):
            raise NotImplementedError("without_shape_completion ablation is not supported in this version.")

        if model.use_vis_encoder:
            vis_encoder_input = vis_coor
            if int(model.config.visual_encoder.input_dim) == 4:
                local_density = self._compute_local_density_knn(vis_encoder_input)
                vis_encoder_input = torch.cat([vis_encoder_input, local_density], dim=-1)
            vis_f, vis_coor = model.shape_encoder(vis_encoder_input)

        tac_f, tac_coor = model.contact_encoder(tac_xyzc)

        if getattr(model, "use_vis_seq_shuf_drop", False):
            vis_f, vis_coor, _ = self._point_seq_drop_shuf(
                vis_f,
                vis_coor,
                drop_rate=model.vis_seq_drop_rate,
                training=model.training,
            )
        if getattr(model, "use_tac_seq_shuf_drop", False):
            tac_f, tac_coor, _ = self._point_seq_drop_shuf(
                tac_f,
                tac_coor,
                drop_rate=model.tac_seq_drop_rate,
                training=model.training,
            )

        vis_f = model.vis_mem_link(vis_f)
        tac_f = model.tac_mem_link(tac_f)
        fused = model.feature_fusion(q=tac_f, v=vis_f, q_pos=tac_coor, v_pos=vis_coor)
        return fused.max(dim=1, keepdim=False)[0]
