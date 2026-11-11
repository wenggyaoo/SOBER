import torch
import torch.nn as nn
from .intensity_masker import IntensityMasker
from .positional_embedding import SinusoidalRotationalEmbedding, SinusoidalSpatialEmbedding


class SOBER(nn.Module):
    """Encode spatial cell neighborhoods and reconstruct masked center-cell biomarkers."""

    def __init__(self,
                 config,
                 biomarker_embedder=None,
                 intensity_masker=None,
                 is_teacher_model=False):
        super().__init__()
        self.config = config
        self.is_teacher_model = is_teacher_model
        self.masking = not is_teacher_model

        self.biomarker_embedder = biomarker_embedder
        assert self.biomarker_embedder is not None, "Biomarker embedder must be provided"
        assert self.biomarker_embedder.embedding_dim is not None, "Biomarker embedder is not initialized"
        self.biomarker_dim = self.biomarker_embedder.embedding_dim

        self.intensity_masker = IntensityMasker(config) if intensity_masker is None else intensity_masker

        self.input_norm = nn.LayerNorm(config.d_model)

        self.embedder_projection = nn.Sequential(
            nn.Linear(self.biomarker_dim, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        if config.positional_embedder == "rotational":
            self.pos_embedding = SinusoidalRotationalEmbedding(config.d_model)
        else:
            self.pos_embedding = SinusoidalSpatialEmbedding(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(config.d_model + self.biomarker_dim),
            nn.Linear(config.d_model + self.biomarker_dim, 1024),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 1)
        )

        if getattr(config, "use_triplet_projection_head", False):
            proj_dim = int(getattr(config, "triplet_projection_dim", config.d_model))
            self.celltype_proj = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, proj_dim),
                nn.LayerNorm(proj_dim),
            )
        else:
            self.celltype_proj = None

        self.distillation_mode = str(getattr(config, "distillation_mode", "center_mse")).lower()
        if self.distillation_mode == "byol":
            distill_dim = int(getattr(config, "distillation_projection_dim", 256))
            predictor_hidden = int(getattr(config, "distillation_predictor_hidden_dim", config.d_model))
            self.distill_projector = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, distill_dim),
            )
            self.distill_predictor = nn.Sequential(
                nn.LayerNorm(distill_dim),
                nn.Linear(distill_dim, predictor_hidden),
                nn.ReLU(),
                nn.Linear(predictor_hidden, distill_dim),
            )
        else:
            self.distill_projector = None
            self.distill_predictor = None

        self.initialize_weights()

    def initialize_weights(self):
        # Preserve head order so seeded initializations match existing runs.
        heads = (self.embedder_projection, self.reconstruction_head, self.celltype_proj,
                 self.distill_projector, self.distill_predictor)
        for head in heads:
            if head is None:
                continue
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

        # The encoder uses a smaller initialization gain than the prediction heads.
        for name, param in self.encoder.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_normal_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def create_embeddings(self, batch_data):
        """Pad cell embeddings and coordinates; collect masked center-cell targets."""
        device = self.config.device
        batch_size = len(batch_data)
        max_seq_len = max(len(region['coordinates']) for region in batch_data)

        embeddings = torch.zeros(batch_size, max_seq_len, self.biomarker_dim, device=device)
        coordinates = torch.zeros(batch_size, max_seq_len, 2, device=device)
        pad_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)

        masked_biomarker_expression = []

        for i, sample in enumerate(batch_data):
            seq_len = len(sample['coordinates'])
            pad_mask[i, seq_len:] = True

            biomarkers = sample['biomarkers']
            coordinates[i, :seq_len] = torch.tensor(sample['coordinates'], dtype=torch.float32, device=device)

            if self.masking:
                masked_sample = self.intensity_masker.mask_sample(sample)
                masked_intensities = masked_sample['masked_intensities']
                if self.config.normalize_masked_expression:
                    masked_intensities = self._renormalize_intensities(masked_intensities)
                embeddings[i, :seq_len] = self.biomarker_embedder(biomarkers, masked_intensities)

                masked_biomarker_expression.append(masked_sample['masked_items'])
            else:
                embeddings[i, :seq_len] = self.biomarker_embedder(biomarkers, sample['intensities'])

        return embeddings, coordinates, pad_mask, masked_biomarker_expression

    def _renormalize_intensities(self, masked_intensities):
        raise NotImplementedError("Renormalization not implemented yet")

    def forward(
            self,
            batch_data,
            recon=True,
            return_celltype_logits=False,
            return_token_embeddings=False):
        """Return center_encoded with shape (batch, d_model), plus requested outputs.

        Set recon=False to skip reconstruction. return_token_embeddings includes all
        cell tokens and their padding mask. return_celltype_logits is a legacy no-op.
        """
        embeddings, coordinates, pad_mask, masked_biomarker_expression = \
            self.create_embeddings(batch_data)

        results = {}

        x = self.embedder_projection(embeddings)
        x = x + self.pos_embedding(coordinates)
        x = self.input_norm(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        # Sampling always places the center cell first.
        center_encoded = x[:, 0]
        results['center_encoded'] = center_encoded
        if return_token_embeddings:
            results['token_encoded'] = x
            results['token_padding_mask'] = pad_mask
        if self.celltype_proj is not None:
            results['triplet_projection'] = self.celltype_proj(center_encoded)
        if self.distill_projector is not None:
            distill_projection = self.distill_projector(center_encoded)
            results['distill_projection'] = distill_projection
            if self.distill_predictor is not None:
                results['distill_prediction'] = self.distill_predictor(distill_projection)

        if not self.is_teacher_model and recon:
            if self.config.spec_index_recon:
                results.update(self._perform_spec_batch_reconstruction(
                    center_encoded, batch_data))
            else:
                results.update(self._perform_batch_reconstruction(
                    center_encoded, batch_data, masked_biomarker_expression))

        return results

    def _perform_batch_reconstruction(self, center_encoded, batch_data, masked_biomarker_expression):
        """Predict each masked center-cell intensity from its cell and biomarker embeddings."""

        recon_biomarker_names = []
        recon_sample_indices = []
        recon_targets = []
        target_descs = []

        for sample_idx, (sample_data, masked_items) in enumerate(zip(batch_data, masked_biomarker_expression)):
            for biomarker_name, intensity in masked_items.items():
                recon_biomarker_names.append(biomarker_name)
                recon_sample_indices.append(sample_idx)
                recon_targets.append(intensity)
                target_descs.append((
                    sample_data['study_name'],
                    sample_data['region_name'],
                    sample_data['original_center_idx'],
                    biomarker_name,
                    intensity
                ))

        if len(recon_targets) == 0:
            empty = torch.tensor([], device=self.config.device, dtype=torch.float32)
            return {'recon_target': empty, 'recon_results': empty, 'target_descs': target_descs}

        recon_sample_indices = torch.tensor(recon_sample_indices, device=self.config.device, dtype=torch.long)
        recon_center_encodings = center_encoded[recon_sample_indices]  # (targets, d_model)
        recon_biomarker_embs = self.biomarker_embedder.get_batched_embeddings(recon_biomarker_names)  # (targets, biomarker_dim)

        recon_targets = torch.tensor(recon_targets, dtype=torch.float32, device=self.config.device)
        recon_inputs = torch.cat([recon_center_encodings, recon_biomarker_embs], dim=-1)
        recon_results = self.reconstruction_head(recon_inputs).squeeze(-1)
        return {'recon_target': recon_targets,
                'recon_results': recon_results,
                'target_descs': target_descs}


SpatialBiomarkerTransformer = SOBER
