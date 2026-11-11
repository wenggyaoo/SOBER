"""SOBER model and training settings."""
from pathlib import Path
import json

import torch


ASSET_DIR = Path(__file__).resolve().parent / "assets"


class Config:
    tqdm_mininterval = 60.0
    tqdm_disable = False
    d_model = 512
    nhead = 8
    num_encoder_layers = 12
    dim_feedforward = 2048
    dropout = 0.1
    positional_embedder = 'rotational'
    use_triplet_projection_head = True
    triplet_projection_dim = 128
    batch_size = 128
    num_epochs = 10
    save_every_n_epochs = 2
    lazy_subgraph_sampling = True
    n_subgraphs_per_region = 500
    lr = 0.0001
    initial_lr = 1e-06
    weight_decay = 0
    warmup_steps = 1000
    ema_momentum = 0.999
    reconstruction_loss_weight = 1
    distillation_loss_weight = 0
    enable_triplet_loss = True
    distillation_mode = 'byol'
    distillation_projection_dim = 256
    distillation_predictor_hidden_dim = 512
    spec_index_recon = False
    contrastive_loss = 'infonce'
    contrastive_temperature = 0.2
    triplet_warmup_steps = 500
    enable_hierarchical_triplet = True
    triplet_coarse_weight = 0.2
    triplet_granular_weight = 0.0
    triplet_loss_weight = 0.1
    triplet_margin = 0.05
    triplet_distance_squared = True
    triplet_mining = 'semi_hard'
    triplet_hard_negative_fallback = False
    triplet_sampling_level = 'coarse'
    triplet_batch_size = 128
    triplet_labels_per_batch = 8
    triplet_studies_per_label = 4
    contrastive_positive_aggregation = 'sum'
    contrastive_queue_size = 4096
    contrastive_queue_warmup_steps = 500
    contrastive_hard_negative_weight = 1.5
    contrastive_regular_negative_weight = 1.0
    max_neighbors = 100
    neighbor_distance_threshold = 300
    microns_per_model_unit = 0.5
    neighborhood_summary_max_centers_per_region = 2000
    normalize_masked_expression = False
    enable_center_intensity_masking = True
    center_intensity_mask_probability = 1
    center_intensity_min_mask_ratio = 0.8
    center_intensity_max_mask_ratio = 0.8
    center_intensity_mask_value = 0.0
    center_mask_strategy = 'random'
    preserve_top_biomarkers = 0
    enable_neighbor_intensity_masking = True
    neighbor_intensity_mask_probability = 1
    neighbor_intensity_min_mask_ratio = 0.2
    neighbor_intensity_max_mask_ratio = 0.8
    neighbor_intensity_mask_value = 0.0
    neighbor_mask_strategy = 'random'
    position_filename = 'cell_data.csv'
    biomarker_filename = 'expression.csv'
    celltype_filename = 'cell_type.csv'
    min_cells_per_region = 100
    use_cell_types = True
    embedding_method = 'learnable'
    biomarker_name_alias_overrides = {'CK': 'PanCK'}
    learnable_embedding_dim = 512

    seed = 42
    device = "auto"
    model_save_path = "checkpoints/sober"
    log_dir = "checkpoints/sober"
    wandb_mode = "disabled"
    wandb_project = "SOBER"
    wandb_name = None
    cell_type_usage_log_path = None
    study_celltype_version_file = str(ASSET_DIR / "study_celltype_final.tsv")
    cell_type_mapping_file = str(ASSET_DIR / "cell_type_mappings.csv")
    biomarker_sequence_file = str(ASSET_DIR / "biomarker_sequences.csv")
    biomarker_name_mapping_file = str(ASSET_DIR / "biomarker_name_mapping.json")

    def __init__(self, **overrides):
        import copy
        defaults = {key: value for key, value in vars(type(self)).items()
                    if not key.startswith("_") and not callable(value)}
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        self.__dict__.update(copy.deepcopy(defaults))
        self.__dict__.update(overrides)

    def to_dict(self):
        return {key: str(value) if isinstance(value, (torch.device, Path)) else value
                for key, value in vars(self).items() if not key.startswith("_")}

    def read_asset(self, name):
        if name in getattr(self, "_assets", {}):
            return self._assets[name]
        return Path(getattr(self, name)).read_text(encoding="utf-8")

    def biomarker_aliases(self):
        mapping = json.loads(self.read_asset("biomarker_name_mapping_file"))
        mapping.update(self.biomarker_name_alias_overrides)
        for name in mapping:
            target, seen = mapping[name], {name}
            while target in mapping and mapping[target] != target:
                if target in seen:
                    raise ValueError(f"Cyclic biomarker alias: {name}")
                seen.add(target)
                target = mapping[target]
            mapping[name] = target
        return mapping
