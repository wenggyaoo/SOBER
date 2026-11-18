"""Small downstream evaluations using the frozen SOBER encoder."""
import copy
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from .spatial_sampler import SpatialSampler
from .checkpoint import file_sha256


# Literal marker exclusions used in the saved reconstruction analysis.
RECONSTRUCTION_EXCLUSIONS = frozenset({
    "CD134", "CD152", "CD47", "CD49f", "CD57", "TMEM16A", "p16",
})


def verify_example_files(data_root, records):
    """Check the exact CSV files recorded for the bundled tutorial regions."""
    for record in records:
        for name, metadata in record["files"].items():
            path = Path(data_root) / record["folder"] / name
            if file_sha256(path) != metadata["sha256"]:
                raise ValueError(f"Example data SHA-256 mismatch: {path}")


@contextmanager
def _evaluation(model, masking):
    was_training, was_masking = model.training, model.masking
    model.eval()
    model.masking = masking
    try:
        with torch.inference_mode():
            yield
    finally:
        model.masking = was_masking
        model.train(was_training)


def sample_neighborhoods(region, config, n_cells, seed=42, labeled_only=False):
    """Select centers without replacement; stratify coarse labels if requested."""
    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    rng = np.random.RandomState(seed)
    eligible = list(range(region["num_cells"]))
    selected = set()
    if labeled_only:
        by_label = {}
        for i, h in enumerate(region["cell_type_hierarchy"]):
            if h and h[0]:
                by_label.setdefault(h[0], []).append(i)
        eligible = [i for indices in by_label.values() for i in indices]
        if len(eligible) > n_cells:
            # Research protocol: one center per class (rarest first), then fill.
            for indices in sorted(by_label.values(), key=len):
                if len(selected) == n_cells:
                    break
                selected.add(int(rng.choice(indices)))
    if not len(eligible):
        raise ValueError("No eligible center cells")
    if len(eligible) <= n_cells:
        indices = sorted(eligible)
    else:
        pool = [i for i in eligible if i not in selected]
        selected.update(rng.choice(pool, n_cells - len(selected), replace=False).tolist())
        indices = sorted(selected)
    sampler = SpatialSampler(config)
    return [sampler.sample_region_at(region, int(i)) for i in indices]


def extract_embeddings(model, samples, batch_size=16):
    """Return unmasked center embeddings in sample order."""
    if not samples or batch_size < 1:
        raise ValueError("Provide samples and a positive batch_size")
    embeddings = []
    with _evaluation(model, masking=False):
        for start in range(0, len(samples), batch_size):
            out = model(samples[start:start + batch_size], recon=False)
            embeddings.append(out["center_encoded"].cpu().numpy())
    return np.concatenate(embeddings)


def sample_reference_neighborhoods(regions, config, n_per_class=64, seed=42):
    """Build a balanced coarse-label reference from training regions only."""
    if not isinstance(n_per_class, (int, np.integer)) or n_per_class < 1:
        raise ValueError("n_per_class must be a positive integer")
    keys = [(r["study_name"], r["region_name"]) for r in regions]
    if len(set(keys)) != len(keys):
        raise ValueError("Reference regions must be unique")
    by_label = {}
    for region_idx, region in enumerate(regions):
        for cell_idx, h in enumerate(region["cell_type_hierarchy"]):
            if h and h[0]:
                by_label.setdefault(h[0], []).append((region_idx, cell_idx))
    if not by_label:
        raise ValueError("No mapped reference labels")
    rng = np.random.RandomState(seed)
    selected = []
    for label, candidates in sorted(by_label.items()):
        indices = rng.choice(len(candidates), min(n_per_class, len(candidates)), replace=False)
        selected.extend(candidates[i] for i in indices)
    sampler = SpatialSampler(config)
    return [sampler.sample_region_at(regions[r], i) for r, i in sorted(selected)]


def reconstruct(model, samples, batch_size=16, seed=42):
    """Predict masked center entries and a mean of the same visible neighbors.

    Masking follows the checkpoint's IntensityMasker. Inputs, model mode and
    NumPy RNG state are restored; changing batch size does not change the masks.
    """
    if not samples or batch_size < 1:
        raise ValueError("Provide samples and a positive batch_size")
    keys = [(s["study_name"], s["region_name"], s["original_center_idx"]) for s in samples]
    if len(set(keys)) != len(keys):
        raise ValueError("Reconstruction centers must be unique within study/region")
    rows = []
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        with _evaluation(model, masking=True):
            for start in range(0, len(samples), batch_size):
                batch = copy.deepcopy(samples[start:start + batch_size])
                out = model(batch, recon=True)
                targets = out["recon_target"].cpu().numpy()
                predictions = out["recon_results"].cpu().numpy()
                descriptions = out["target_descs"]
                if not (len(descriptions) == len(targets) == len(predictions)):
                    raise RuntimeError("Reconstruction targets and predictions are misaligned")
                by_key = dict(zip(keys[start:start + batch_size], batch))
                for desc, target, prediction in zip(descriptions, targets, predictions):
                    study, region, center_idx, marker, _ = desc
                    sample = by_key[(study, region, center_idx)]
                    j = sample["biomarkers"].index(marker)
                    values = np.asarray(sample["masked_intensities"][1:, j], dtype=np.float32)
                    visible = (sample["mask_flags"][1:, j] == 0) & np.isfinite(values)
                    hierarchy = sample.get("cell_type_hierarchy") or []
                    label = hierarchy[0][0] if hierarchy and hierarchy[0] else "unknown"
                    rows.append({
                        "study": study, "region": region, "center_idx": int(center_idx),
                        "cell_id": str(sample["cell_ids"][0]), "cell_type": label,
                        "marker": marker, "observed": float(target), "sober": float(prediction),
                        "neighbor_mean": float(values[visible].mean()) if visible.any() else np.nan,
                        "n_visible_neighbors": int(visible.sum()),
                    })
    finally:
        np.random.set_state(rng_state)
    return pd.DataFrame(rows, columns=[
        "study", "region", "center_idx", "cell_id", "cell_type", "marker",
        "observed", "sober", "neighbor_mean", "n_visible_neighbors",
    ])


def regression_metrics(observed, predicted):
    """Pairwise finite Pearson, Spearman and MSE, matching the research analysis."""
    observed, predicted = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if observed.ndim != 1 or observed.shape != predicted.shape:
        raise ValueError("Expected aligned one-dimensional observations and predictions")
    valid = np.isfinite(observed) & np.isfinite(predicted)
    y, p = observed[valid], predicted[valid]
    result = {"n": len(y), "pearson_r": np.nan, "spearman_r": np.nan, "mse": np.nan}
    if len(y) >= 2:
        result["mse"] = float(np.mean((y - p) ** 2))
        if np.std(y) > 0 and np.std(p) > 0:
            result["pearson_r"] = float(np.corrcoef(y, p)[0, 1])
            result["spearman_r"] = float(spearmanr(y, p).statistic)
    return result


def reconstruction_metrics(predictions, groupby=None, methods=("sober", "neighbor_mean")):
    """Summarize all entries or groups such as marker or (cell_type, marker)."""
    groups = [] if groupby is None else ([groupby] if isinstance(groupby, str) else list(groupby))
    grouped = predictions.groupby(groups, sort=True, dropna=False) if groups else [((), predictions)]
    rows = []
    for key, frame in grouped:
        key = key if isinstance(key, tuple) else (key,)
        for method in methods:
            rows.append({**dict(zip(groups, key)), "method": method,
                         **regression_metrics(frame["observed"], frame[method])})
    return pd.DataFrame(rows, columns=groups + ["method", "n", "pearson_r", "spearman_r", "mse"])


def predict_cell_types(reference_embeddings, reference_labels, query_embeddings, k=10):
    """Transfer reference labels using L2-normalized embeddings and cosine KNN."""
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(reference_embeddings):
        raise ValueError("k must be an integer between 1 and the reference cell count")
    reference = normalize(np.asarray(reference_embeddings), norm="l2")
    query = normalize(np.asarray(query_embeddings), norm="l2")
    classifier = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="uniform")
    classifier.fit(reference, reference_labels)
    return classifier.predict(query)


def annotation_metrics(observed, predicted):
    """Score all query cells, including labels absent from the reference."""
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if observed.ndim != 1 or observed.shape != predicted.shape or not len(observed):
        raise ValueError("Expected nonempty, aligned label vectors")
    return {"n_cells": len(observed), "accuracy": float(accuracy_score(observed, predicted)),
            "f1_weighted": float(f1_score(observed, predicted, average="weighted", zero_division=0))}
