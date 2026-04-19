from __future__ import annotations

from src.models.backbones.dgcnn_encoder import DGCNNEncoder
from src.models.predictors.stability_head import StabilityHead
from src.perception.adapters import CNNMCAAdapter, DGCNNAdapter, SGAGSNAdapter
from src.perception.cnnmca_runtime import get_shared_cnnmca_runtime, infer_cnnmca_body_feature_dim
from src.perception.contact_semantics import ContactSemanticsExtractor
from src.perception.feature_extractor import FeatureExtractor
from src.perception.sga_gsn_runtime import get_shared_sga_gsn_runtime, infer_sga_gsn_body_feature_dim
from src.perception.stability_predictor import StabilityPredictor


def build_feature_extractor(cfg: dict):
    backbone_cfg = cfg.get("backbone", {})
    adapter_type = cfg.get("adapter_type", "sga_gsn")
    freeze = bool(cfg.get("feature_extractor", {}).get("freeze", True))

    if adapter_type == "dgcnn":
        backbone = DGCNNEncoder(backbone_cfg)
        adapter = DGCNNAdapter()
        runtime = None
    elif adapter_type == "sga_gsn":
        runtime_cfg = cfg.get("sga_gsn", {}).get("runtime", {})
        adapter = SGAGSNAdapter(cfg)
        runtime = get_shared_sga_gsn_runtime(runtime_cfg)
        backbone = None
    elif adapter_type == "cnnmca":
        runtime_cfg = cfg.get("cnnmca", {}).get("runtime", {})
        adapter = CNNMCAAdapter(cfg)
        runtime = get_shared_cnnmca_runtime(runtime_cfg)
        backbone = None
    else:
        raise ValueError(f"Unknown perception adapter_type '{adapter_type}'.")
    return FeatureExtractor(backbone_model=backbone, adapter=adapter, freeze=freeze, runtime=runtime)


def build_stability_predictor(cfg: dict):
    predictor_cfg = cfg.get("predictor", {}).copy()
    adapter_type = cfg.get("adapter_type", "sga_gsn")
    freeze = bool(cfg.get("feature_extractor", {}).get("freeze", True))

    if adapter_type == "dgcnn":
        predictor_cfg.setdefault("latent_dim", cfg.get("backbone", {}).get("latent_dim", 32))
        predictor_model = StabilityHead(predictor_cfg)
    else:
        predictor_model = None
    return StabilityPredictor(predictor_model=predictor_model, freeze=freeze)


def infer_perception_feature_dim(cfg: dict) -> int:
    adapter_type = cfg.get("adapter_type", "sga_gsn")
    if adapter_type == "sga_gsn":
        runtime_cfg = cfg.get("sga_gsn", {}).get("runtime", {})
        return int(infer_sga_gsn_body_feature_dim(runtime_cfg))
    if adapter_type == "cnnmca":
        runtime_cfg = cfg.get("cnnmca", {}).get("runtime", {})
        try:
            return int(infer_cnnmca_body_feature_dim(runtime_cfg))
        except (KeyError, FileNotFoundError):
            return int(cfg.get("backbone", {}).get("latent_dim", 512))
    if adapter_type == "dgcnn":
        return int(cfg.get("backbone", {}).get("latent_dim", 32))
    raise ValueError(f"Unknown perception adapter_type '{adapter_type}'.")


def build_contact_semantics_extractor(cfg: dict):
    return ContactSemanticsExtractor(cfg.get("contact_semantics", {}))


def build_perception_stack(cfg: dict):
    feature_extractor = build_feature_extractor(cfg)
    contact_semantics_extractor = build_contact_semantics_extractor(cfg)
    stability_predictor = build_stability_predictor(cfg)
    return feature_extractor, contact_semantics_extractor, stability_predictor
