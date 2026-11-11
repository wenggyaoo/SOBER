"""Load self-contained SOBER release checkpoints for representation extraction."""
import torch

from .biomarker_embedder import BiomarkerEmbedder
from .config import Config
from .model import SOBER


def load_checkpoint(path, device="cpu", weights="student"):
    if weights not in ("student", "teacher"):
        raise ValueError("weights must be student or teacher")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("model_name") != "SOBER":
        raise ValueError("Expected a SOBER release checkpoint with config and biomarker_vocab")
    if weights == "teacher" and not checkpoint["teacher_active"]:
        raise ValueError("This run did not update the teacher; use student weights")
    config = Config(**checkpoint["config"])
    config.device = torch.device(device)
    config._assets = {key: asset["text"] for key, asset in checkpoint["assets"].items()}
    embedder = BiomarkerEmbedder(config, precomputed_esm=config.embedding_method == "esm")
    embedder.build_biomarker_vocab(checkpoint["biomarker_vocab"])
    model = SOBER(config, biomarker_embedder=embedder, is_teacher_model=weights == "teacher")
    model.load_state_dict(checkpoint[f"{weights}_state_dict"], strict=True)
    model.to(config.device).eval()
    model.masking = False
    return model, checkpoint
