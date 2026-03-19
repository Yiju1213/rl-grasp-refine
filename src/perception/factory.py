from __future__ import annotations

from src.models.backbones.dgcnn_encoder import DGCNNEncoder
from src.models.backbones.sga_gsn_encoder import SGAGSNEncoder
from src.models.predictors.stability_head import StabilityHead
from src.perception.adapters import DGCNNAdapter, SGAGSNAdapter
from src.perception.contact_semantics import ContactSemanticsExtractor
from src.perception.feature_extractor import FeatureExtractor
from src.perception.stability_predictor import StabilityPredictor


def build_feature_extractor(cfg: dict):
    backbone_cfg = cfg.get("backbone", {})
    adapter_type = cfg.get("adapter_type", "sga_gsn")
    freeze = bool(cfg.get("feature_extractor", {}).get("freeze", True))

    if adapter_type == "dgcnn":
        backbone = DGCNNEncoder(backbone_cfg)
        adapter = DGCNNAdapter()
    else:
        backbone = SGAGSNEncoder(backbone_cfg)
        adapter = SGAGSNAdapter()
    return FeatureExtractor(backbone_model=backbone, adapter=adapter, freeze=freeze)


def build_stability_predictor(cfg: dict):
    predictor_cfg = cfg.get("predictor", {}).copy()
    predictor_cfg.setdefault("latent_dim", cfg.get("backbone", {}).get("latent_dim", 32))
    adapter_type = cfg.get("adapter_type", "sga_gsn")
    adapter = DGCNNAdapter() if adapter_type == "dgcnn" else SGAGSNAdapter()
    predictor_model = StabilityHead(predictor_cfg)
    freeze = bool(cfg.get("feature_extractor", {}).get("freeze", True))
    return StabilityPredictor(predictor_model=predictor_model, adapter=adapter, freeze=freeze)


def build_contact_semantics_extractor(cfg: dict):
    return ContactSemanticsExtractor(cfg.get("contact_semantics", {}))


def build_perception_stack(cfg: dict):
    feature_extractor = build_feature_extractor(cfg)
    contact_semantics_extractor = build_contact_semantics_extractor(cfg)
    stability_predictor = build_stability_predictor(cfg)
    return feature_extractor, contact_semantics_extractor, stability_predictor
