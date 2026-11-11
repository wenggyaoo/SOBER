"""Train SOBER from standardized multiplexed cell imaging tables."""
import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from sober import Config, SOBER
from sober.biomarker_embedder import BiomarkerEmbedder
from sober.cell_type_normalize import apply_cell_type_normalization
from sober.data_loader import SpatialBiomarkerDataset
from sober.spatial_sampler import save_neighborhood_sampling_summary
from sober.splits import filter_dataset_by_patient_metadata, make_patient_split
from sober.trainer import Trainer


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing study/region folders")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/sober"), help="New or empty output directory")
    parser.add_argument("--config", type=Path, help="JSON overrides of sober/config.py")
    parser.add_argument("--split-by", choices=("patient", "study", "region"), default="patient")
    parser.add_argument("--patient-metadata", type=Path, help="CSV/TSV or trusted pickle: ACQUISITION_ID, PATIENT_ID")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, help="Training random seed")
    parser.add_argument("--device", help="auto, cpu, or cuda:N")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-subgraphs", type=int, help="Samples per region defining an epoch")
    parser.add_argument("--no-triplet", action="store_true", help="Disable label supervision; keep configured reconstruction/distillation weights")
    parser.add_argument("--no-lazy-sampling", action="store_true")
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--run-name")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error("--data-root must be an existing directory")
    if args.split_by == "patient" and (args.patient_metadata is None or not args.patient_metadata.is_file()):
        parser.error("Patient splitting requires --patient-metadata; use --split-by study/region explicitly if appropriate")
    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be strictly between 0 and 1")
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    if args.save_dir.exists() and any(args.save_dir.iterdir()):
        parser.error("--save-dir must be empty to avoid overwriting an earlier run")
    return args


def build_config(args):
    overrides = json.loads(args.config.read_text()) if args.config else {}
    config = Config(**overrides)
    for source, target in (("epochs", "num_epochs"), ("batch_size", "batch_size"),
                           ("n_subgraphs", "n_subgraphs_per_region"), ("device", "device"), ("seed", "seed")):
        value = getattr(args, source)
        if value is not None:
            setattr(config, target, value)
    if args.no_triplet:
        config.enable_triplet_loss = False
    if not config.enable_triplet_loss:
        config.use_cell_types = False
    if args.no_lazy_sampling:
        config.lazy_subgraph_sampling = False
    if config.device == "auto":
        config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.device = torch.device(config.device)
    if config.device.type not in ("cpu", "cuda"):
        raise ValueError("This release supports CPU and CUDA devices")
    if config.device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; select --device cpu")
    config.model_save_path = str(args.save_dir.resolve())
    config.log_dir = config.model_save_path
    config.wandb_mode, config.wandb_name = args.wandb, args.run_name
    config._tqdm_file = sys.stderr
    config.tqdm_disable = not args.show_progress
    for name in ("num_epochs", "batch_size", "n_subgraphs_per_region", "max_neighbors", "min_cells_per_region"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if config.d_model % 4 or config.d_model % config.nhead:
        raise ValueError("d_model must be divisible by 4 and nhead")
    weights = [config.reconstruction_loss_weight, config.distillation_loss_weight,
               config.triplet_coarse_weight, config.triplet_granular_weight]
    if any(not np.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Loss weights must be finite and non-negative")
    active_weights = weights if config.enable_triplet_loss else weights[:2]
    if sum(active_weights) <= 0:
        raise ValueError("At least one training loss must have a positive weight")
    if config.enable_triplet_loss and not config.use_cell_types:
        raise ValueError("Label supervision requires use_cell_types=true")
    if config.enable_triplet_loss and not config.enable_hierarchical_triplet:
        raise ValueError("The release CLI uses hierarchical label losses; set enable_hierarchical_triplet=true")
    if config.normalize_masked_expression or config.spec_index_recon:
        raise ValueError("Experimental reconstruction/renormalization branches are not implemented")
    if config.distillation_mode not in ("center_mse", "byol"):
        raise ValueError("distillation_mode must be center_mse or byol")
    if config.contrastive_loss not in ("infonce", "triplet"):
        raise ValueError("contrastive_loss must be infonce or triplet")
    if config.neighbor_distance_threshold <= 0 or not np.isfinite(config.neighbor_distance_threshold):
        raise ValueError("neighbor_distance_threshold must be finite and positive")
    return config


def split_dataset(dataset, args):
    patient_filter = None
    if args.split_by == "patient":
        patient_filter = filter_dataset_by_patient_metadata(dataset, str(args.patient_metadata))
        train, validation, _, _ = make_patient_split(dataset, args.train_ratio, args.split_seed)
    else:
        groups = {}
        for index, region in enumerate(dataset.regions):
            key = region["study_name"] if args.split_by == "study" else index
            groups.setdefault(key, []).append(index)
        keys = list(groups)
        if len(keys) < 2:
            raise ValueError(f"{args.split_by} split requires at least two groups")
        np.random.RandomState(args.split_seed).shuffle(keys)
        count = min(len(keys) - 1, max(1, int(len(keys) * args.train_ratio)))
        train = [index for key in keys[:count] for index in groups[key]]
        validation = [index for key in keys[count:] for index in groups[key]]
    def identities(indices):
        return [{"study_name": dataset[i]["study_name"], "region_name": dataset[i]["region_name"],
                 "patient_id": dataset[i].get("patient_id")} for i in indices]
    manifest = {"split_unit": args.split_by, "split_seed": args.split_seed,
                "train_ratio": args.train_ratio, "held_out_partition": "internal_validation",
                "train_region_indices": train, "validation_region_indices": validation,
                "train_regions": identities(train), "validation_regions": identities(validation),
                "patient_filter": patient_filter, "skipped_small_regions": dataset.skipped_regions}
    return train, validation, manifest


def run(args, config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(args.num_threads)
    print(f"SOBER | device={config.device} | output={args.save_dir.resolve()}")
    print(f"Loss weights: reconstruction={config.reconstruction_loss_weight}, distillation={config.distillation_loss_weight}, "
          f"label supervision={config.enable_triplet_loss} ({config.contrastive_loss})")
    dataset = SpatialBiomarkerDataset(args.data_root, config)
    train_indices, validation_indices, split = split_dataset(dataset, args)
    if config.use_cell_types:
        apply_cell_type_normalization(dataset, config.cell_type_mapping_file)
    train_dataset = Subset(dataset, train_indices)
    vocab = sorted({marker for region in train_dataset for marker in region["biomarkers"]})
    write_json(args.save_dir / "split.json", split)
    write_json(args.save_dir / "config.json", config.to_dict())
    write_json(args.save_dir / "biomarker_vocab.json", vocab)
    write_json(args.save_dir / "data_manifest.json", [
        {key: region[key] for key in ("study_name", "region_name", "num_cells", "biomarkers", "cell_type_file_used")}
        for region in dataset.regions])
    packages = {}
    for package in ("torch", "numpy", "pandas", "scipy", "tqdm", "wandb", "fair-esm"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    release_root = Path(__file__).resolve().parent
    code_files = [Path(__file__).resolve(), *sorted((release_root / "sober").rglob("*.py"))]
    write_json(args.save_dir / "run_metadata.json", {
        "model_name": "SOBER", "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_sha256": {str(path.relative_to(release_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files},
    })
    asset_dir = args.save_dir / "assets"
    asset_dir.mkdir()
    assets = {}
    for key in ("biomarker_name_mapping_file", "biomarker_sequence_file", "cell_type_mapping_file", "study_celltype_version_file"):
        path = Path(getattr(config, key))
        if path.is_file():
            shutil.copyfile(path, asset_dir / path.name)
            assets[key] = {"filename": path.name, "text": path.read_text(encoding="utf-8")}
    save_neighborhood_sampling_summary(config, train_dataset, str(args.save_dir), seed=args.seed if args.seed is not None else config.seed)
    embedder = BiomarkerEmbedder(config)
    embedder.build_biomarker_vocab(vocab)
    student = SOBER(config, biomarker_embedder=embedder).to(config.device)
    # Share immutable run settings without copying the progress output stream.
    teacher = copy.deepcopy(student, {id(config): config})
    teacher.is_teacher_model = True
    teacher.masking = False
    teacher.eval()
    trainer = Trainer(config, student, teacher, train_dataset)
    history = []
    try:
        for epoch in range(config.num_epochs):
            recon, distill, contrastive, diagnostics = trainer.train_epoch(epoch)
            row = {"epoch": epoch + 1, "reconstruction_loss": float(recon), "distillation_loss": float(distill),
                   "weighted_label_loss": float(contrastive), "total_loss": trainer.epoch_total_loss,
                   "diagnostics": diagnostics}
            history.append(row)
            write_json(args.save_dir / "training_history.json", history)
            print(f"Epoch {epoch + 1}/{config.num_epochs} | loss={row['total_loss']:.6f} | "
                  f"reconstruction={recon:.6f} | distillation={distill:.6f} | labels={contrastive:.6f}")
            if (epoch + 1 == config.num_epochs or config.save_every_n_epochs <= 0
                    or (epoch + 1) % config.save_every_n_epochs == 0):
                checkpoint = trainer.save_checkpoint(epoch, data_root=str(args.data_root.resolve()), split=split, assets=assets)
        shutil.copyfile(checkpoint, args.save_dir / "checkpoint_final.pth")
    finally:
        trainer.finish()
    print(f"Training complete: {args.save_dir / 'checkpoint_final.pth'}")


def main():
    args = arguments()
    config = build_config(args)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "train.log").open("w", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            run(args, config)


if __name__ == "__main__":
    main()
