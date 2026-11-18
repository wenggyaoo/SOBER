"""Fixed-mask reconstruction and distance-ranked neighborhood replacement."""
import copy

import numpy as np
import pandas as pd
import torch

from .evaluation import _evaluation


def center_masks(samples, targets, visible=("CD3e",), ratio=0.8, seed=42):
    """Force response markers into the mask, then fill to the requested ratio."""
    if not 0 <= ratio <= 1 or set(targets) & set(visible):
        raise ValueError("Use a mask ratio in [0, 1] and distinct target/visible markers")
    rng = np.random.RandomState(seed)
    masks = []
    for sample in samples:
        markers = sample["biomarkers"]
        target = [i for i, marker in enumerate(markers) if marker in targets]
        available = [i for i, marker in enumerate(markers) if marker not in visible]
        size = min(len(available), max(len(target), int(len(markers) * ratio)))
        extras = [i for i in available if i not in target]
        fill = rng.choice(extras, size=min(size - len(target), len(extras)), replace=False).tolist() if extras else []
        masks.append(np.asarray(sorted(target + fill), dtype=int))
    return masks


def predict_center_markers(model, samples, masks, markers, batch_size=16):
    """Use explicit center masks; leave all neighbor markers visible."""
    if not samples or len(samples) != len(masks) or batch_size < 1:
        raise ValueError("Provide one mask per sample and a positive batch size")
    predictions = []
    with _evaluation(model, masking=False):
        marker_embeddings = model.biomarker_embedder.get_batched_embeddings(list(markers))
        for start in range(0, len(samples), batch_size):
            batch = copy.deepcopy(samples[start:start + batch_size])
            for sample, mask in zip(batch, masks[start:start + batch_size]):
                columns = [sample["biomarkers"].index(marker) for marker in markers]
                if not set(columns) <= set(mask):
                    raise ValueError("Every reported center marker must be hidden")
                sample["intensities"][0, mask] = 0.0
            encoded = model(batch, recon=False)["center_encoded"]
            for center in encoded:
                inputs = torch.cat([center.expand(len(markers), -1), marker_embeddings], dim=-1)
                predictions.append(model.reconstruction_head(inputs).squeeze(-1).cpu().numpy())
    return np.stack(predictions)


def _distances(sample):
    return np.linalg.norm(np.asarray(sample["coordinates"])[1:], axis=1)


def _composition(sample):
    counts = np.zeros(4, dtype=float)
    terms = (("epithelial", "tumor", "cancer", "malignant"),
             ("macrophage", "monocyte", "dendritic", "granulocyte", "myeloid"),
             ("t cell", "b cell", "nk cell", "plasma", "lymph"),
             ("stromal", "endothelial", "fibro"))
    for hierarchy in sample["cell_type_hierarchy"][1:]:
        label = str(hierarchy[0] if hierarchy and hierarchy[0] else "").lower()
        for group, words in enumerate(terms):
            if any(word in label for word in words):
                counts[group] += 1
                break
    annotated = counts.sum()
    return np.r_[counts / max(1, annotated), annotated / max(1, len(sample["intensities"]) - 1)]


def match_donors(samples, recipients, n_donors=2, min_neighbors=10, seed=42):
    """Match within study/panel, excluding overlapping neighborhoods."""
    if n_donors < 1:
        raise ValueError("n_donors must be positive")
    rng = np.random.RandomState(seed)
    compositions = np.stack([_composition(sample) for sample in samples])
    counts = np.asarray([len(sample["intensities"]) - 1 for sample in samples])
    radii = np.asarray([max(_distances(sample), default=0.0) for sample in samples])
    means = {study: compositions[[s["study_name"] == study and counts[i] >= min_neighbors
                                 for i, s in enumerate(samples)]].mean(axis=0)
             for study in {s["study_name"] for s in samples}}
    reuse = np.zeros(len(samples), dtype=int)
    assignments = {}
    for recipient in recipients:
        source = samples[recipient]
        ids = set(map(str, source["cell_ids"]))
        candidates = [i for i, s in enumerate(samples)
                      if i != recipient and counts[i] >= min_neighbors
                      and s["study_name"] == source["study_name"]
                      and s["biomarkers"] == source["biomarkers"]
                      and (s["region_name"] != source["region_name"]
                           or not ids.intersection(map(str, s["cell_ids"])))]
        if len(candidates) < n_donors or counts[recipient] < min_neighbors:
            raise ValueError("Insufficient non-overlapping donors; enlarge the input pool")
        mean = means[source["study_name"]][:4]
        typicality = np.linalg.norm(compositions[recipient, :4] - mean)
        jitter = {i: rng.uniform(0, 1e-6) for i in candidates}

        def rank(i):
            donor_typicality = np.linalg.norm(compositions[i, :4] - mean)
            score = (abs(np.log(counts[i] / counts[recipient]))
                     + abs(np.log(max(radii[i], 1e-6) / max(radii[recipient], 1e-6)))
                     + np.abs(compositions[recipient] - compositions[i]).sum()
                     + abs(typicality - donor_typicality)
                     + 2 * max(0, typicality - donor_typicality))
            return (samples[i]["region_name"] == source["region_name"],
                    score + jitter[i], reuse[i], i)

        chosen = sorted(candidates, key=rank)[:n_donors]
        # Research correlations order each recipient's donors by sample index.
        assignments[recipient] = sorted(chosen)
        reuse[chosen] += 1
    return assignments


def radial_plan(recipient, donor):
    """Pair outer-to-inner distance ranks once for all subsequent boundaries."""
    if recipient["biomarkers"] != donor["biomarkers"]:
        raise ValueError("Recipient and donor must have the same ordered panel")
    target = np.argsort(_distances(recipient))[::-1] + 1
    source = np.argsort(_distances(donor))[::-1] + 1
    if not len(target) or not len(source):
        raise ValueError("Recipient and donor must both contain neighbors")
    positions = np.rint(np.linspace(0, len(source) - 1, len(target))).astype(int)
    return np.column_stack([target, source[positions]])


def replace_outside_radius(recipient, donor, plan, boundary):
    """Replace expression outside the boundary; retain recipient coordinates."""
    if boundary < 0 or not np.isfinite(boundary):
        raise ValueError("boundary must be finite and non-negative")
    plan = np.asarray(plan, dtype=int)
    if plan.ndim != 2 or plan.shape[1] != 2 or np.any(plan < 1):
        raise ValueError("The replacement plan must contain neighbor index pairs")
    selected = plan[_distances(recipient)[plan[:, 0] - 1] > boundary]
    out = copy.deepcopy(recipient)
    if len(selected):
        target, source = selected.T
        out["intensities"][target] = np.asarray(donor["intensities"])[source]
        for key in ("cell_ids", "cell_types", "cell_type_hierarchy"):
            if recipient.get(key) and donor.get(key):
                out[key] = list(out[key])
                for i, j in selected:
                    out[key][i] = donor[key][j]
    return out, len(selected)


def _pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def radial_summary(predictions):
    """Correlate across centers within donor slot, then average in Fisher space."""
    rows = []
    for (marker, boundary), frame in predictions.groupby(["marker", "boundary"]):
        correlations = np.array([_pearson(g.observed, g.predicted)
                                 for _, g in frame.groupby("donor_slot")])
        valid = correlations[np.isfinite(correlations)]
        pearson = np.tanh(np.arctanh(np.clip(valid, -0.999999, 0.999999)).mean()) if len(valid) else np.nan
        rows.append({"marker": marker, "boundary": boundary, "pearson_r": pearson,
                     "n_centers": frame.center_idx.nunique(), "n_valid_donor_slots": len(valid),
                     "mean_replaced_fraction": frame.replaced_fraction.mean()})
    summary = pd.DataFrame(rows).sort_values(["marker", "boundary"], ascending=[True, False])
    baseline = summary.loc[summary.boundary.eq(summary.boundary.max())].set_index("marker").pearson_r
    summary["pearson_drop"] = summary.marker.map(baseline) - summary.pearson_r
    return summary.reset_index(drop=True)
