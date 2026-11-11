"""SOBER optimization, EMA distillation, and optional label supervision."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from .spatial_sampler import SpatialSampler
from .intensity_masker import save_intensity_masking_record, intensity_masking_record


def _tqdm_options(config):
    return {
        "file": getattr(config, "_tqdm_file", None),
        "mininterval": float(getattr(config, "tqdm_mininterval", 60.0)),
        "disable": bool(getattr(config, "tqdm_disable", False)),
        "dynamic_ncols": True,
    }


def _label_key(label, level):
    if label is None:
        return None
    if level == "coarse":
        return label[0] if isinstance(label, tuple) else label
    if level == "granular":
        if not isinstance(label, tuple) or len(label) < 2 or not label[1]:
            return None
        return "%s|%s" % (label[0], label[1])
    raise ValueError("Triplet label level must be coarse or granular")


def _semi_hard_triplet_loss(
    embeddings, labels, studies, level, margin=0.05, squared=True,
    mining="semi_hard", hard_negative_fallback=True
):
    """Mine triplets on unit-normalized embeddings.

    Use the farthest same-label cell from another study as the positive. Negatives
    have different labels; semi-hard negatives satisfy d_ap < d_an < d_ap + margin.
    """
    keys = [_label_key(l, level) for l in labels]
    valid_idx = [i for i, k in enumerate(keys) if k is not None and studies[i]]
    if len(valid_idx) < 3:
        zero = torch.tensor(0.0, device=embeddings.device)
        return zero, {"valid_anchors": 0, "semi_hard": 0}

    emb = F.normalize(embeddings[valid_idx], p=2, dim=1)
    keys = [keys[i] for i in valid_idx]
    studies = [studies[i] for i in valid_idx]
    dist = torch.cdist(emb, emb, p=2)
    if squared:
        dist = dist.pow(2)

    losses, d_ap_used, d_an_used = [], [], []
    easy_triplets, semi_hard_triplets, hard_triplets = 0, 0, 0
    positive_pairs, negative_pairs = 0, 0
    n = len(keys)
    for i in range(n):
        pos_idx = [
            j for j in range(n)
            if j != i and keys[j] == keys[i] and studies[j] != studies[i]
        ]
        neg_idx = [j for j in range(n) if keys[j] != keys[i]]
        if not pos_idx or not neg_idx:
            continue

        pos_d = dist[i, pos_idx]
        neg_d = dist[i, neg_idx]
        positive_pairs += len(pos_idx)
        negative_pairs += len(neg_idx)

        triplet_delta = pos_d[:, None] - neg_d[None, :]
        hard_triplets += int((triplet_delta >= 0).sum().detach().cpu().item())
        semi_hard_triplets += int(((triplet_delta < 0) & (triplet_delta > -margin)).sum().detach().cpu().item())
        easy_triplets += int((triplet_delta <= -margin).sum().detach().cpu().item())

        if mining == "batch_hard":
            p_local = torch.argmax(pos_d)
            n_local = torch.argmin(neg_d)
        else:
            p_local = torch.argmax(pos_d)
            d_ap = pos_d[p_local]
            semi = (neg_d > d_ap) & (neg_d < d_ap + margin)
            if semi.any():
                semi_idx = torch.where(semi)[0]
                n_local = semi_idx[torch.randint(len(semi_idx), (), device=embeddings.device)]
            elif hard_negative_fallback:
                n_local = torch.argmin(neg_d)
            else:
                continue

        d_ap = pos_d[p_local]
        d_an = neg_d[n_local]
        losses.append(F.relu(d_ap - d_an + margin))
        d_ap_used.append(d_ap)
        d_an_used.append(d_an)

    if not losses:
        zero = torch.tensor(0.0, device=embeddings.device)
        return zero, {
            "valid_anchors": len(valid_idx),
            "positive_pairs": positive_pairs,
            "negative_pairs": negative_pairs,
            "easy": easy_triplets,
            "semi_hard": semi_hard_triplets,
            "hard": hard_triplets,
            "active": 0,
            "active_ratio": 0.0,
        }

    loss = torch.stack(losses).mean()
    total_triplets = easy_triplets + semi_hard_triplets + hard_triplets
    active_triplets = semi_hard_triplets + hard_triplets
    stats = {
        "valid_anchors": len(valid_idx),
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "easy": easy_triplets,
        "semi_hard": semi_hard_triplets,
        "hard": hard_triplets,
        "active": active_triplets,
        "active_ratio": float(active_triplets / max(1, total_triplets)),
        "mined_anchors": len(losses),
        "d_ap": float(torch.stack(d_ap_used).mean().detach().cpu().item()),
        "d_an": float(torch.stack(d_an_used).mean().detach().cpu().item()),
        "d_gap": float((torch.stack(d_an_used) - torch.stack(d_ap_used)).mean().detach().cpu().item()),
    }
    return loss, stats


def _supervised_infonce_loss(
    embeddings,
    labels,
    studies,
    level,
    temperature=0.1,
    queue=None,
    hard_negative_weight=1.0,
    regular_negative_weight=1.0,
    positive_aggregation="mean",
):
    """InfoNCE with cross-study positives and different-label negatives.

    Same-study, same-label pairs are excluded. "mean" averages positive log
    probabilities (SupCon); "sum" pools positives in the numerator.
    """
    if temperature <= 0:
        raise ValueError("contrastive_temperature must be positive.")
    if hard_negative_weight <= 0 or regular_negative_weight <= 0:
        raise ValueError("contrastive negative weights must be positive.")
    positive_aggregation = str(positive_aggregation).lower()
    if positive_aggregation not in ("mean", "sum"):
        raise ValueError("contrastive_positive_aggregation must be 'mean' or 'sum'.")

    anchor_keys_all = [_label_key(l, level) for l in labels]
    anchor_valid_idx = [
        i for i, k in enumerate(anchor_keys_all)
        if k is not None and studies[i]
    ]
    current_candidate_valid_idx = anchor_valid_idx
    if not anchor_valid_idx:
        zero = torch.tensor(0.0, device=embeddings.device)
        return zero, {"valid_anchors": 0, "positive_pairs": 0, "negative_pairs": 0}

    device = embeddings.device
    anchor_emb = F.normalize(embeddings[anchor_valid_idx], p=2, dim=1)
    anchor_keys = [anchor_keys_all[i] for i in anchor_valid_idx]
    anchor_studies = [studies[i] for i in anchor_valid_idx]

    candidate_parts = [embeddings[current_candidate_valid_idx]]
    candidate_keys = [anchor_keys_all[i] for i in current_candidate_valid_idx]
    candidate_studies = [studies[i] for i in current_candidate_valid_idx]
    candidate_source_idx = list(current_candidate_valid_idx)
    candidate_is_queue = [False] * len(current_candidate_valid_idx)

    queue_total = 0
    if queue is not None:
        queue_embeddings, queue_labels, queue_studies = queue
        if queue_embeddings is not None and len(queue_labels) > 0:
            queue_total = len(queue_labels)
            queue_keys_all = [_label_key(l, level) for l in queue_labels]
            queue_valid_idx = [
                i for i, k in enumerate(queue_keys_all)
                if k is not None and queue_studies[i]
            ]
            if queue_valid_idx:
                q_idx = torch.tensor(queue_valid_idx, dtype=torch.long, device=queue_embeddings.device)
                q_emb = queue_embeddings.index_select(0, q_idx).to(device)
                candidate_parts.append(q_emb)
                candidate_keys.extend(queue_keys_all[i] for i in queue_valid_idx)
                candidate_studies.extend(queue_studies[i] for i in queue_valid_idx)
                candidate_source_idx.extend([-1] * len(queue_valid_idx))
                candidate_is_queue.extend([True] * len(queue_valid_idx))

    candidate_emb = F.normalize(torch.cat(candidate_parts, dim=0), p=2, dim=1)
    n_candidate = len(candidate_keys)
    if n_candidate < 2:
        zero = torch.tensor(0.0, device=device)
        return zero, {"valid_anchors": 0, "positive_pairs": 0, "negative_pairs": 0}

    label_to_id, study_to_id = {}, {}

    def _ids(values, mapping):
        ids = []
        for value in values:
            if value not in mapping:
                mapping[value] = len(mapping)
            ids.append(mapping[value])
        return ids

    anchor_label_ids = torch.tensor(_ids(anchor_keys, label_to_id), dtype=torch.long, device=device)
    candidate_label_ids = torch.tensor(_ids(candidate_keys, label_to_id), dtype=torch.long, device=device)
    anchor_study_ids = torch.tensor(_ids(anchor_studies, study_to_id), dtype=torch.long, device=device)
    candidate_study_ids = torch.tensor(_ids(candidate_studies, study_to_id), dtype=torch.long, device=device)
    anchor_source_idx = torch.tensor(anchor_valid_idx, dtype=torch.long, device=device)
    candidate_source_idx = torch.tensor(candidate_source_idx, dtype=torch.long, device=device)
    candidate_is_queue = torch.tensor(candidate_is_queue, dtype=torch.bool, device=device)

    same_label = anchor_label_ids[:, None] == candidate_label_ids[None, :]
    same_study = anchor_study_ids[:, None] == candidate_study_ids[None, :]
    not_self = anchor_source_idx[:, None] != candidate_source_idx[None, :]

    positive_mask = not_self & same_label & ~same_study
    negative_mask = not_self & ~same_label
    hard_negative_mask = negative_mask & same_study
    regular_negative_mask = negative_mask & ~same_study
    ignored_same_study_same_type = not_self & same_label & same_study

    pos_counts = positive_mask.sum(dim=1)
    neg_counts = negative_mask.sum(dim=1)
    valid_anchor_mask = (pos_counts > 0) & (neg_counts > 0)
    if not valid_anchor_mask.any():
        zero = torch.tensor(0.0, device=device)
        return zero, {
            "valid_anchors": 0,
            "anchors_with_positive": int((pos_counts > 0).sum().detach().cpu().item()),
            "positive_pairs": int(positive_mask.sum().detach().cpu().item()),
            "negative_pairs": int(negative_mask.sum().detach().cpu().item()),
            "ignored_same_study_same_type_pairs": int(ignored_same_study_same_type.sum().detach().cpu().item()),
            "candidate_count": n_candidate,
            "queue_size": queue_total,
            "queue_candidates": int(candidate_is_queue.sum().detach().cpu().item()),
        }

    sim = anchor_emb @ candidate_emb.t()
    logits = sim / float(temperature)
    denominator_mask = positive_mask | negative_mask
    denominator_weights = torch.ones_like(logits)
    denominator_weights = torch.where(
        hard_negative_mask,
        torch.full_like(denominator_weights, float(hard_negative_weight)),
        denominator_weights,
    )
    denominator_weights = torch.where(
        regular_negative_mask,
        torch.full_like(denominator_weights, float(regular_negative_weight)),
        denominator_weights,
    )
    weighted_logits = logits + torch.log(denominator_weights)
    masked_logits = weighted_logits.masked_fill(~denominator_mask, float("-inf"))
    log_denominator = torch.logsumexp(masked_logits, dim=1)
    log_prob = logits - log_denominator[:, None]

    anchors = torch.where(valid_anchor_mask)[0]
    anchor_positive_mask = positive_mask[anchors].float()
    if positive_aggregation == "sum":
        positive_logits = logits[anchors].masked_fill(~positive_mask[anchors], float("-inf"))
        log_numerator = torch.logsumexp(positive_logits, dim=1)
        loss = -(log_numerator - log_denominator[anchors]).mean()
    else:
        anchor_log_prob = log_prob[anchors]
        anchor_pos_counts = pos_counts[anchors].float()
        mean_log_prob_pos = (anchor_log_prob * anchor_positive_mask).sum(dim=1) / anchor_pos_counts
        loss = -mean_log_prob_pos.mean()

    def _mask_mean(mask):
        if mask.any():
            return float(sim[mask].mean().detach().cpu().item())
        return 0.0

    valid_anchor_count = int(valid_anchor_mask.sum().detach().cpu().item())
    hard_counts = hard_negative_mask.sum(dim=1)
    regular_counts = regular_negative_mask.sum(dim=1)
    queue_positive_mask = positive_mask & candidate_is_queue[None, :]
    queue_negative_mask = negative_mask & candidate_is_queue[None, :]
    queue_hard_negative_mask = hard_negative_mask & candidate_is_queue[None, :]
    queue_regular_negative_mask = regular_negative_mask & candidate_is_queue[None, :]
    stats = {
        "valid_anchors": valid_anchor_count,
        "anchors_with_positive": int((pos_counts > 0).sum().detach().cpu().item()),
        "anchors_with_hard_negatives": int((hard_counts > 0).sum().detach().cpu().item()),
        "positive_pairs": int(positive_mask.sum().detach().cpu().item()),
        "negative_pairs": int(negative_mask.sum().detach().cpu().item()),
        "hard_negative_pairs": int(hard_negative_mask.sum().detach().cpu().item()),
        "regular_negative_pairs": int(regular_negative_mask.sum().detach().cpu().item()),
        "ignored_same_study_same_type_pairs": int(ignored_same_study_same_type.sum().detach().cpu().item()),
        "pos_per_anchor": float(pos_counts[anchors].float().mean().detach().cpu().item()),
        "neg_per_anchor": float(neg_counts[anchors].float().mean().detach().cpu().item()),
        "hard_neg_per_anchor": float(hard_counts[anchors].float().mean().detach().cpu().item()),
        "regular_neg_per_anchor": float(regular_counts[anchors].float().mean().detach().cpu().item()),
        "positive_sim": _mask_mean(positive_mask),
        "hard_negative_sim": _mask_mean(hard_negative_mask),
        "regular_negative_sim": _mask_mean(regular_negative_mask),
        "temperature": float(temperature),
        "candidate_count": n_candidate,
        "queue_size": queue_total,
        "queue_candidates": int(candidate_is_queue.sum().detach().cpu().item()),
        "queue_positive_pairs": int(queue_positive_mask.sum().detach().cpu().item()),
        "queue_negative_pairs": int(queue_negative_mask.sum().detach().cpu().item()),
        "queue_hard_negative_pairs": int(queue_hard_negative_mask.sum().detach().cpu().item()),
        "queue_regular_negative_pairs": int(queue_regular_negative_mask.sum().detach().cpu().item()),
        "hard_negative_weight": float(hard_negative_weight),
        "regular_negative_weight": float(regular_negative_weight),
        "positive_aggregation_is_sum": 1.0 if positive_aggregation == "sum" else 0.0,
    }
    return loss, stats


def _center_labels_for_triplets(batch_regions):
    """Read labels from the first cell in each neighborhood, preferring hierarchy tuples."""
    use_hier = any(
        r.get("cell_type_hierarchy") is not None and len(r.get("cell_type_hierarchy") or []) > 0
        for r in batch_regions
    )
    if use_hier:
        labels = []
        for r in batch_regions:
            ch = r.get("cell_type_hierarchy") or []
            labels.append(ch[0] if ch else None)
        return labels, True
    labels = []
    for r in batch_regions:
        ct = r.get("cell_types") or []
        labels.append(ct[0] if ct else None)
    return labels, False


def _triplet_warmup_scale(config, global_step):
    warmup_steps = int(getattr(config, "triplet_warmup_steps", 0) or 0)
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(global_step) / float(warmup_steps))


def _distillation_loss(config, student_results, teacher_results, mse_loss_fn):
    mode = str(getattr(config, "distillation_mode", "center_mse")).lower()
    if mode in ("center_mse", "mse", "ema_mse"):
        loss = mse_loss_fn(teacher_results["center_encoded"], student_results["center_encoded"])
        with torch.no_grad():
            s = F.normalize(student_results["center_encoded"], p=2, dim=1)
            t = F.normalize(teacher_results["center_encoded"], p=2, dim=1)
            cosine = (s * t).sum(dim=1)
            stats = {
                "mode_is_byol": 0.0,
                "cosine": float(cosine.mean().detach().cpu().item()),
                "student_norm": float(student_results["center_encoded"].norm(dim=1).mean().detach().cpu().item()),
                "teacher_norm": float(teacher_results["center_encoded"].norm(dim=1).mean().detach().cpu().item()),
            }
        return loss, stats

    if mode == "byol":
        if "distill_prediction" not in student_results or "distill_projection" not in teacher_results:
            raise ValueError("distillation_mode='byol' requires distill_projector and student predictor outputs.")
        student_pred = F.normalize(student_results["distill_prediction"], p=2, dim=1)
        teacher_proj = F.normalize(teacher_results["distill_projection"].detach(), p=2, dim=1)
        cosine = (student_pred * teacher_proj).sum(dim=1)
        loss = (2.0 - 2.0 * cosine).mean()
        with torch.no_grad():
            stats = {
                "mode_is_byol": 1.0,
                "cosine": float(cosine.mean().detach().cpu().item()),
                "student_norm": float(student_results["distill_prediction"].norm(dim=1).mean().detach().cpu().item()),
                "teacher_norm": float(teacher_results["distill_projection"].norm(dim=1).mean().detach().cpu().item()),
                "student_projection_std": float(student_results["distill_projection"].std(dim=0, unbiased=False).mean().detach().cpu().item()),
                "teacher_projection_std": float(teacher_results["distill_projection"].std(dim=0, unbiased=False).mean().detach().cpu().item()),
            }
        return loss, stats

    raise ValueError("Unknown distillation_mode: %s" % mode)


class Trainer:
    def __init__(self, config, student_model, teacher_model, train_dataset):
        self.config = config
        self.student_model = student_model.to(config.device)
        self.teacher_model = teacher_model.to(config.device)
        self.train_dataset = train_dataset

        self.sampler = SpatialSampler(config)
        self.recon_loss_fn = nn.MSELoss()
        self.distill_loss_fn = nn.MSELoss()
        self._last_triplet_sampler_stats = {}
        self.contrastive_queue_embeddings = None
        self.contrastive_queue_labels = []
        self.contrastive_queue_studies = []

        self.optimizer = optim.Adam(
            student_model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        self.global_step = 0
        self.warmup_steps = config.warmup_steps

        self.ema_momentum = config.ema_momentum
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        use_triplet = getattr(config, 'enable_triplet_loss', False)
        if hasattr(train_dataset, 'regions'):
            all_regions = train_dataset.regions
            self.labeled_region_indices = [
                i for i, r in enumerate(all_regions)
                if len(r.get('cell_types') or []) > 0
            ]
        else:
            # These indices are local to the Subset.
            base_regions = train_dataset.dataset.regions
            self.labeled_region_indices = [
                local_i for local_i, global_i in enumerate(train_dataset.indices)
                if len(base_regions[global_i].get('cell_types') or []) > 0
            ]
        if use_triplet:
            objective = getattr(config, "contrastive_loss", "triplet")
            print("%s contrastive loss enabled: %d / %d train regions have cell type labels — "
                  "main batch samples all regions; contrastive batch samples labeled regions only." % (
                      objective,
                      len(self.labeled_region_indices), len(train_dataset)))
            if len(self.labeled_region_indices) == 0:
                raise ValueError("enable_triplet_loss=True but no labeled regions found.")
            self._build_triplet_cell_index()
            if objective in ("infonce", "supcon", "supervised_infonce"):
                print(
                    "InfoNCE queue/weights: queue_size=%d, queue_warmup_steps=%d, "
                    "hard_negative_weight=%.3f, regular_negative_weight=%.3f"
                    % (
                        int(getattr(config, "contrastive_queue_size", 0) or 0),
                        int(getattr(config, "contrastive_queue_warmup_steps", 0) or 0),
                        float(getattr(config, "contrastive_hard_negative_weight", 1.0)),
                        float(getattr(config, "contrastive_regular_negative_weight", 1.0)),
                    )
                )
                print(
                    "InfoNCE positive aggregation: %s"
                    % str(getattr(config, "contrastive_positive_aggregation", "mean"))
                )

        _mp = getattr(config, "model_save_path", None) or "."
        _rec_path = save_intensity_masking_record(config, _mp)
        if _rec_path:
            print("Recorded intensity masking settings to %s" % _rec_path)
        _r = intensity_masking_record(config)
        _c, _n = _r["center_cell"], _r["neighbor_cells"]
        print(
            "Intensity masking (training): center enabled=%s P_apply=%.3g ratio[%.3g,%.3g] strategy=%s | "
            "neighbor enabled=%s P_apply=%.3g ratio[%.3g,%.3g] strategy=%s"
            % (
                _c["enable_center_intensity_masking"],
                _c["center_intensity_mask_probability"],
                _c["center_intensity_min_mask_ratio"],
                _c["center_intensity_max_mask_ratio"],
                _c["center_mask_strategy"],
                _n["enable_neighbor_intensity_masking"],
                _n["neighbor_intensity_mask_probability"],
                _n["neighbor_intensity_min_mask_ratio"],
                _n["neighbor_intensity_max_mask_ratio"],
                _n["neighbor_mask_strategy"],
            )
        )

        self.wandb_run = None
        if getattr(config, "wandb_mode", "disabled") != "disabled":
            import wandb
            self.wandb_run = wandb.init(
                project=config.wandb_project, name=config.wandb_name,
                config=config.to_dict(), dir=config.log_dir, mode=config.wandb_mode,
            )

    def log(self, values, step):
        if self.wandb_run is not None:
            self.wandb_run.log(values, step=step)

    def finish(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def ema_update(self):
        with torch.no_grad():
            for student_param, teacher_param in zip(self.student_model.parameters(), self.teacher_model.parameters()):
                teacher_param.data = self.ema_momentum * teacher_param.data + (1.0 - self.ema_momentum) * student_param.data

    def get_lr(self):
        if self.global_step < self.warmup_steps:
            return self.config.initial_lr + (self.config.lr - self.config.initial_lr) * \
                (self.global_step / max(1, self.warmup_steps))
        else:
            return self.config.lr

    def _contrastive_queue_max_size(self):
        return max(0, int(getattr(self.config, "contrastive_queue_size", 0) or 0))

    def _contrastive_queue_active(self):
        if self._contrastive_queue_max_size() <= 0:
            return False
        if self.contrastive_queue_embeddings is None or len(self.contrastive_queue_labels) == 0:
            return False
        warmup_steps = int(getattr(self.config, "contrastive_queue_warmup_steps", 0) or 0)
        return self.global_step > warmup_steps

    def _get_contrastive_queue(self):
        if not self._contrastive_queue_active():
            return None
        return (
            self.contrastive_queue_embeddings,
            self.contrastive_queue_labels,
            self.contrastive_queue_studies,
        )

    def _contrastive_queue_len(self):
        return len(self.contrastive_queue_labels)

    def _enqueue_contrastive_queue(self, embeddings, labels, studies):
        max_size = self._contrastive_queue_max_size()
        if max_size <= 0 or embeddings is None or embeddings.numel() == 0:
            return

        emb = embeddings.detach().to(self.config.device)
        labels = list(labels)
        studies = list(studies)
        if emb.size(0) != len(labels) or emb.size(0) != len(studies):
            raise ValueError("Contrastive queue update received mismatched embeddings/labels/studies.")

        if emb.size(0) > max_size:
            emb = emb[-max_size:]
            labels = labels[-max_size:]
            studies = studies[-max_size:]

        if self.contrastive_queue_embeddings is None:
            self.contrastive_queue_embeddings = emb.clone()
            self.contrastive_queue_labels = labels
            self.contrastive_queue_studies = studies
            return

        self.contrastive_queue_embeddings = torch.cat(
            [self.contrastive_queue_embeddings, emb.clone()],
            dim=0,
        )
        self.contrastive_queue_labels.extend(labels)
        self.contrastive_queue_studies.extend(studies)
        if self.contrastive_queue_embeddings.size(0) > max_size:
            keep = self.contrastive_queue_embeddings.size(0) - max_size
            self.contrastive_queue_embeddings = self.contrastive_queue_embeddings[keep:].contiguous()
            self.contrastive_queue_labels = self.contrastive_queue_labels[keep:]
            self.contrastive_queue_studies = self.contrastive_queue_studies[keep:]

    def build_subgraph_dataset(self, seed=None):
        """Precompute one epoch of neighborhoods when lazy sampling is disabled."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        train_subgraphs = []
        for i in range(len(self.train_dataset)):
            region = self.train_dataset[i]
            for _ in range(self.config.n_subgraphs_per_region):
                subgraph = self.sampler.sample_region(region)
                train_subgraphs.append(subgraph)

        np.random.shuffle(train_subgraphs)
        print(f"Assembled {len(train_subgraphs)} training subgraphs")
        return train_subgraphs

    def _build_triplet_cell_index(self):
        """Index labeled cells by hierarchy level, label, and study."""
        self.triplet_cells = {"coarse": {}, "granular": {}}
        for local_region_idx in self.labeled_region_indices:
            region = self.train_dataset[local_region_idx]
            study = region.get("study_name", "")
            hier = region.get("cell_type_hierarchy") or []
            n_cells = min(region.get("num_cells", 0), len(hier))
            if not study or n_cells == 0:
                continue
            for cell_idx in range(n_cells):
                h = hier[cell_idx]
                if h is None:
                    continue
                coarse = _label_key(h, "coarse")
                granular = _label_key(h, "granular")
                if coarse:
                    self.triplet_cells["coarse"].setdefault(coarse, {}).setdefault(study, []).append(
                        (local_region_idx, cell_idx)
                    )
                if granular:
                    self.triplet_cells["granular"].setdefault(granular, {}).setdefault(study, []).append(
                        (local_region_idx, cell_idx)
                    )

        level = getattr(self.config, "triplet_sampling_level", "coarse")
        self.triplet_study_to_labels = {"coarse": {}, "granular": {}}
        for lvl, label_map in self.triplet_cells.items():
            for label, study_map in label_map.items():
                for study in study_map:
                    self.triplet_study_to_labels[lvl].setdefault(study, set()).add(label)

        label_to_studies = self.triplet_cells.get(level, {})
        self.triplet_eligible_labels = [
            label for label, study_map in label_to_studies.items()
            if len(study_map) >= 2
        ]
        n_cells = sum(
            len(cells)
            for study_map in label_to_studies.values()
            for cells in study_map.values()
        )
        print(
            "Triplet sampler index (%s): %d labels with >=2 studies, %d labeled cells indexed."
            % (level, len(self.triplet_eligible_labels), n_cells)
        )
        if not self.triplet_eligible_labels:
            raise ValueError(
                "Triplet sampling requires at least one cell type label present in >=2 studies."
            )

    def _prune_contrastive_grid(self, labels, studies, label_to_studies):
        """Keep labels with >=2 selected studies and studies with >=2 selected labels."""
        labels = list(dict.fromkeys(labels))
        studies = list(dict.fromkeys(studies))
        changed = True
        while changed:
            old = (len(labels), len(studies))
            labels = [
                label for label in labels
                if sum(study in label_to_studies[label] for study in studies) >= 2
            ]
            studies = [
                study for study in studies
                if sum(study in label_to_studies[label] for label in labels) >= 2
            ]
            changed = old != (len(labels), len(studies))
        return labels, studies

    def _sample_main_batch(self, epoch, batch_idx, batch_size):
        """Sample regions uniformly with replacement for reconstruction and distillation."""
        rng = np.random.RandomState(seed=(self.config.seed + epoch * 131071 + batch_idx) % (2**31))
        n_regions = len(self.train_dataset)
        region_indices = rng.randint(0, n_regions, size=batch_size)
        return [self.sampler.sample_region(self.train_dataset[i]) for i in region_indices]

    def _sample_labeled_batch(self, epoch, batch_idx, batch_size):
        """Balance labeled neighborhoods across cell types and studies.

        Prefer a shared grid that provides cross-study positives and same-study negatives.
        """
        rng = np.random.RandomState(seed=(self.config.seed + epoch * 131071 + batch_idx + 999983) % (2**31))
        level = getattr(self.config, "triplet_sampling_level", "coarse")
        label_to_studies = self.triplet_cells.get(level, {})
        study_to_labels = self.triplet_study_to_labels.get(level, {})

        target_size = int(getattr(self.config, "triplet_batch_size", 0) or batch_size)
        target_labels = min(
            len(self.triplet_eligible_labels),
            max(1, int(getattr(self.config, "triplet_labels_per_batch", 16)))
        )
        max_studies = max(2, int(getattr(self.config, "triplet_studies_per_label", 4)))

        selected_labels, selected_studies, buckets = [], [], []
        shared_grid = False
        study_candidates = [
            study for study, labels in study_to_labels.items()
            if len(labels) >= 2
        ]
        if len(study_candidates) >= 2:
            for _ in range(64):
                n_studies = min(len(study_candidates), max_studies)
                chosen_studies = list(rng.choice(study_candidates, size=n_studies, replace=False))
                label_pool = [
                    label for label in self.triplet_eligible_labels
                    if sum(study in label_to_studies[label] for study in chosen_studies) >= 2
                ]
                if len(label_pool) < 2:
                    continue
                n_labels = min(target_labels, len(label_pool))
                chosen_labels = list(rng.choice(label_pool, size=n_labels, replace=False))
                chosen_labels, chosen_studies = self._prune_contrastive_grid(
                    chosen_labels, chosen_studies, label_to_studies
                )
                if len(chosen_labels) < 2 or len(chosen_studies) < 2:
                    continue
                buckets = [
                    (label, study)
                    for label in chosen_labels
                    for study in chosen_studies
                    if study in label_to_studies[label]
                ]
                if buckets:
                    selected_labels = chosen_labels
                    selected_studies = chosen_studies
                    shared_grid = True
                    break

        if not buckets:
            n_labels = min(target_labels, len(self.triplet_eligible_labels))
            replace_labels = len(self.triplet_eligible_labels) < n_labels
            selected_labels = list(rng.choice(self.triplet_eligible_labels, size=n_labels, replace=replace_labels))
            for label in selected_labels:
                studies = list(label_to_studies[label].keys())
                n_studies = min(len(studies), max_studies)
                chosen_studies = list(rng.choice(studies, size=n_studies, replace=False))
                selected_studies.extend(chosen_studies)
                for study in chosen_studies:
                    buckets.append((label, study))
            selected_studies = list(dict.fromkeys(selected_studies))

        if not buckets:
            self._last_triplet_sampler_stats = {
                "labels": 0, "studies": 0, "study_celltype_buckets": 0,
                "target_size": target_size, "actual_size": 0, "shared_grid": 0,
            }
            return []

        samples = []
        while len(samples) < target_size:
            order = rng.permutation(len(buckets))
            for bucket_i in order:
                label, study = buckets[int(bucket_i)]
                cells = label_to_studies[label][study]
                region_idx, cell_idx = cells[rng.randint(0, len(cells))]
                region = self.train_dataset[region_idx]
                samples.append(self.sampler.sample_region_at(region, cell_idx))
                if len(samples) >= target_size:
                    break

        self._last_triplet_sampler_stats = {
            "labels": len(set(selected_labels)),
            "studies": len(set(selected_studies)),
            "study_celltype_buckets": len(buckets),
            "target_size": target_size,
            "actual_size": len(samples),
            "mean_cells_per_bucket": float(len(samples) / max(1, len(buckets))),
            "grid_density": float(len(buckets) / max(1, len(set(selected_labels)) * len(set(selected_studies)))),
            "shared_grid": 1 if shared_grid else 0,
        }
        return samples

    def train_epoch(self, epoch):
        self.student_model.train()
        self.teacher_model.eval()

        batch_size = self.config.batch_size
        n_regions = len(self.train_dataset)
        n_per_region = self.config.n_subgraphs_per_region
        lazy = getattr(self.config, 'lazy_subgraph_sampling', True)

        if lazy:
            n_batches = max(1, (n_regions * n_per_region) // batch_size)
            print(f"Epoch {epoch+1}: lazy sampling, {n_batches} batches (batch_size={batch_size})")
        else:
            subgraphs = self.build_subgraph_dataset(seed=self.config.seed + epoch)
            n_batches = (len(subgraphs) + batch_size - 1) // batch_size

        recon_losses, distill_losses, total_losses = [], [], []
        triplet_losses_log = []
        triplet_diag_logs = {}
        distill_diag_logs = {}

        pbar = tqdm(
            range(n_batches),
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            **_tqdm_options(self.config),
        )
        for b in pbar:
            self.global_step += 1
            recon_enabled = float(getattr(self.config, 'reconstruction_loss_weight', 0.0)) > 0.0
            distill_enabled = float(getattr(self.config, 'distillation_loss_weight', 0.0)) > 0.0
            main_batch_enabled = recon_enabled or distill_enabled

            if main_batch_enabled:
                if lazy:
                    batch_regions = self._sample_main_batch(epoch, b, batch_size)
                else:
                    start = b * batch_size
                    batch_regions = subgraphs[start : start + batch_size]

            current_lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            self.optimizer.zero_grad()

            if main_batch_enabled:
                student_results = self.student_model(batch_regions, recon=recon_enabled)
                if distill_enabled:
                    with torch.no_grad():
                        teacher_results = self.teacher_model(batch_regions)

            total_loss = 0.0
            if recon_enabled:
                rt, rr = student_results['recon_target'], student_results['recon_results']
                if rt.numel() > 0:
                    recon_loss = self.recon_loss_fn(rt, rr)
                else:
                    if self.global_step <= 1 or self.global_step % 500 == 0:
                        print("Warning: batch has no recon targets (empty masking); using recon_loss=0.")
                    recon_loss = torch.tensor(0.0, device=self.config.device)
                total_loss += self.config.reconstruction_loss_weight * recon_loss
            else:
                recon_loss = torch.tensor(0.0, device=self.config.device)

            distill_log_extra = {}
            if distill_enabled:
                distill_loss, distill_stats = _distillation_loss(
                    self.config, student_results, teacher_results, self.distill_loss_fn)
                total_loss += self.config.distillation_loss_weight * distill_loss
                for key, value in distill_stats.items():
                    distill_log_extra["train/distill_%s" % key] = value
            else:
                distill_loss = torch.tensor(0.0, device=self.config.device)

            # Label supervision uses a separate batch with unmasked intensities.
            triplet_loss_val = 0.0
            triplet_log_extra = {}
            contrastive_queue_update = None
            if getattr(self.config, 'enable_triplet_loss', False) and getattr(self.config, 'use_cell_types', True):
                triplet_batch = self._sample_labeled_batch(epoch, b, batch_size)
                if triplet_batch:
                    prev_masking = self.student_model.masking
                    self.student_model.masking = False
                    try:
                        triplet_results = self.student_model(triplet_batch, recon=False)
                    finally:
                        self.student_model.masking = prev_masking
                    center_S = triplet_results.get('triplet_projection', triplet_results['center_encoded'])
                    objective = str(getattr(self.config, 'contrastive_loss', 'triplet')).lower()
                    margin = getattr(self.config, 'triplet_margin', 0.05)
                    squared = getattr(self.config, 'triplet_distance_squared', True)
                    mining = getattr(self.config, 'triplet_mining', "semi_hard")
                    fallback = getattr(self.config, 'triplet_hard_negative_fallback', True)
                    temperature = float(getattr(self.config, 'contrastive_temperature', 0.1))
                    hard_negative_weight = float(getattr(self.config, 'contrastive_hard_negative_weight', 1.0))
                    regular_negative_weight = float(getattr(self.config, 'contrastive_regular_negative_weight', 1.0))
                    positive_aggregation = str(getattr(self.config, 'contrastive_positive_aggregation', 'mean'))
                    labels, from_hierarchy = _center_labels_for_triplets(triplet_batch)
                    studies = [r.get("study_name", "") for r in triplet_batch]
                    use_hier = (
                        from_hierarchy
                        and getattr(self.config, 'enable_hierarchical_triplet', True)
                    )
                    is_infonce = objective in ("infonce", "supcon", "supervised_infonce")
                    queue_for_loss = self._get_contrastive_queue() if is_infonce else None

                    def _contrastive_loss_for_level(level_name):
                        if is_infonce:
                            return _supervised_infonce_loss(
                                center_S,
                                labels,
                                studies,
                                level_name,
                                temperature=temperature,
                                queue=queue_for_loss,
                                hard_negative_weight=hard_negative_weight,
                                regular_negative_weight=regular_negative_weight,
                                positive_aggregation=positive_aggregation,
                            )
                        if objective == "triplet":
                            return _semi_hard_triplet_loss(
                                center_S, labels, studies, level_name, margin, squared=squared,
                                mining=mining, hard_negative_fallback=fallback)
                        raise ValueError("Unknown contrastive_loss: %s" % objective)

                    triplet_log_extra["train/triplet_is_infonce"] = 1.0 if objective in (
                        "infonce", "supcon", "supervised_infonce"
                    ) else 0.0
                    triplet_log_extra["train/triplet_temperature"] = temperature
                    if is_infonce:
                        contrastive_queue_update = (center_S, labels, studies)
                        triplet_log_extra["train/triplet_queue_len"] = float(self._contrastive_queue_len())
                        triplet_log_extra["train/triplet_queue_active"] = 1.0 if queue_for_loss is not None else 0.0
                    if use_hier:
                        w_c = float(getattr(self.config, 'triplet_coarse_weight', 0.1))
                        w_g = float(getattr(self.config, 'triplet_granular_weight', 0.1))
                        loss_c, stats_c = _contrastive_loss_for_level("coarse")
                        loss_g, stats_g = _contrastive_loss_for_level("granular")
                        triplet_unscaled = w_c * loss_c + w_g * loss_g
                        triplet_scale = _triplet_warmup_scale(self.config, self.global_step)
                        triplet_loss_val = triplet_scale * triplet_unscaled
                        total_loss += triplet_loss_val
                        triplet_log_extra["train/triplet_warmup_scale"] = triplet_scale
                        triplet_log_extra["train/triplet_unscaled"] = float(triplet_unscaled.detach().cpu().item())
                        triplet_log_extra["train/triplet_coarse_raw"] = float(loss_c.detach().cpu().item())
                        triplet_log_extra["train/triplet_granular_raw"] = float(loss_g.detach().cpu().item())
                        for name, stats in (("coarse", stats_c), ("granular", stats_g)):
                            for key, value in stats.items():
                                triplet_log_extra["train/triplet_%s_%s" % (name, key)] = value
                    else:
                        triplet_loss_val, stats = _contrastive_loss_for_level("coarse")
                        triplet_scale = _triplet_warmup_scale(self.config, self.global_step)
                        triplet_weighted = getattr(self.config, 'triplet_loss_weight', 0.1) * triplet_loss_val
                        total_loss += triplet_scale * triplet_weighted
                        triplet_log_extra["train/triplet_warmup_scale"] = triplet_scale
                        triplet_log_extra["train/triplet_unscaled"] = float(triplet_weighted.detach().cpu().item())
                        triplet_log_extra["train/triplet_flat_raw"] = float(triplet_loss_val.detach().cpu().item())
                        for key, value in stats.items():
                            triplet_log_extra["train/triplet_flat_%s" % key] = value
                    for key, value in self._last_triplet_sampler_stats.items():
                        triplet_log_extra["train/triplet_sampler_%s" % key] = value
                    with torch.no_grad():
                        center_norm = center_S.norm(dim=1)
                        center_l2 = F.normalize(center_S, p=2, dim=1)
                        if center_l2.size(0) > 1:
                            d = torch.pdist(center_l2, p=2)
                            triplet_log_extra["train/triplet_center_pairdist_mean"] = float(d.mean().detach().cpu().item())
                        triplet_log_extra["train/triplet_center_norm_mean"] = float(center_norm.mean().detach().cpu().item())
                        triplet_log_extra["train/triplet_center_norm_std"] = float(
                            center_norm.std(unbiased=center_norm.numel() > 1).detach().cpu().item())
            if torch.is_tensor(triplet_loss_val):
                triplet_losses_log.append(float(triplet_loss_val.detach().cpu().item()))
            else:
                triplet_losses_log.append(float(triplet_loss_val))
            for k, v in triplet_log_extra.items():
                short_key = k.replace("train/triplet_", "triplet_")
                triplet_diag_logs.setdefault(short_key, []).append(float(v))
            for k, v in distill_log_extra.items():
                short_key = k.replace("train/distill_", "distill_")
                distill_diag_logs.setdefault(short_key, []).append(float(v))

            if not torch.is_tensor(total_loss):
                raise ValueError(
                    "No active training loss. Set reconstruction_loss_weight, "
                    "distillation_loss_weight, or enable contrastive loss."
                )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if distill_enabled:
                self.ema_update()
            if contrastive_queue_update is not None:
                self._enqueue_contrastive_queue(*contrastive_queue_update)

            recon_losses.append(recon_loss.item())
            distill_losses.append(distill_loss.item())
            total_losses.append(total_loss.item())

            if self.global_step % 50 == 0:
                postfix = {
                    'recon': f'{np.mean(recon_losses[-50:]):.4f}',
                    'distill': f'{np.mean(distill_losses[-50:]):.4f}',
                    'lr': f'{current_lr:.2e}'
                }
                if triplet_losses_log:
                    postfix['contrastive'] = f'{np.mean(triplet_losses_log[-50:]):.4f}'
                pbar.set_postfix(postfix)

            if self.global_step % 10 == 0:
                log_dict = {
                    "train/recon_loss": np.mean(recon_losses[-10:]),
                    "train/distill_loss": np.mean(distill_losses[-10:]),
                    "train/total_loss": np.mean(total_losses[-10:]),
                    "train/lr": current_lr,
                    "train/global_step": self.global_step,
                }
                if triplet_losses_log:
                    log_dict["train/triplet_loss"] = np.mean(triplet_losses_log[-10:])
                    log_dict["train/contrastive_loss"] = np.mean(triplet_losses_log[-10:])
                    if triplet_log_extra:
                        log_dict.update(triplet_log_extra)
                if distill_log_extra:
                    log_dict.update(distill_log_extra)
                self.log(log_dict, step=self.global_step)

        epoch_log = {
            "epoch": epoch + 1,
            "train/epoch_recon_loss": np.mean(recon_losses),
            "train/epoch_distill_loss": np.mean(distill_losses),
            "train/epoch_total_loss": np.mean(total_losses),
        }
        if triplet_losses_log:
            epoch_log["train/epoch_triplet_loss"] = np.mean(triplet_losses_log)
            epoch_log["train/epoch_contrastive_loss"] = np.mean(triplet_losses_log)
        for k, values in triplet_diag_logs.items():
            if values:
                epoch_log["train/epoch_%s" % k] = float(np.mean(values))
        for k, values in distill_diag_logs.items():
            if values:
                epoch_log["train/epoch_%s" % k] = float(np.mean(values))
        self.log(epoch_log, step=self.global_step)

        epoch_triplet = np.mean(triplet_losses_log) if triplet_losses_log else 0.0
        epoch_diag = {k: float(np.mean(v)) for k, v in triplet_diag_logs.items() if v}
        epoch_diag.update({k: float(np.mean(v)) for k, v in distill_diag_logs.items() if v})
        self.epoch_total_loss = float(np.mean(total_losses))
        return np.mean(recon_losses), np.mean(distill_losses), epoch_triplet, epoch_diag

    def save_checkpoint(self, epoch, data_root=None, **extra):
        if "assets" not in extra:
            extra["assets"] = {}
            for key in ("biomarker_name_mapping_file", "biomarker_sequence_file",
                        "cell_type_mapping_file", "study_celltype_version_file"):
                path = getattr(self.config, key)
                if key in getattr(self.config, "_assets", {}) or (path and os.path.isfile(path)):
                    extra["assets"][key] = {"filename": os.path.basename(path),
                                            "text": self.config.read_asset(key)}
        checkpoint = {
            'model_name': 'SOBER',
            'format_version': 1,
            'config': self.config.to_dict(),
            'biomarker_vocab': list(self.student_model.biomarker_embedder.biomarker_to_idx),
            'teacher_active': self.config.distillation_loss_weight > 0,
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'student_state_dict': self.student_model.state_dict(),
            'teacher_state_dict': self.teacher_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if data_root is not None:
            checkpoint['data_root'] = data_root
        for k, v in extra.items():
            checkpoint[k] = v
        filename = f"checkpoint_epoch_{epoch+1}.pth"
        path = os.path.join(self.config.model_save_path, filename)
        torch.save(checkpoint, path)
        return path

    def train(self):
        os.makedirs(self.config.model_save_path, exist_ok=True)

        save_every = getattr(self.config, 'save_every_n_epochs', 1)
        for epoch in range(self.config.num_epochs):
            epoch_losses = self.train_epoch(epoch)
            if save_every <= 0 or (epoch + 1) % save_every == 0 or (epoch + 1) == self.config.num_epochs:
                self.save_checkpoint(epoch)
            print(f"Epoch {epoch+1} | losses: {epoch_losses}")

        self.finish()
