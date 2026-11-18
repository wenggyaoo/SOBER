"""Frozen neighborhood states and patient-level readouts of cached state edits."""
import numpy as np
import pandas as pd
from scipy.special import expit


def assign_states(embeddings, atlas):
    """Apply the saved scaler, PCA and nearest-centroid atlas without fitting."""
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[1] != len(atlas["scaler_mean"]):
        raise ValueError("Embedding dimensions do not match the frozen atlas")
    x = (embeddings.astype(np.float64) - atlas["scaler_mean"]) / atlas["scaler_scale"]
    x = (x - atlas["pca_mean"]) @ atlas["pca_components"].T
    distances = ((x[:, None] - atlas["kmeans_centers"][None]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1)


def patient_abundance(states, groups, n_states):
    states, groups = np.asarray(states), np.asarray(groups, dtype=str)
    if states.shape != groups.shape or states.ndim != 1 or not len(states):
        raise ValueError("Provide aligned, nonempty state and patient vectors")
    if np.any(states < 0) or np.any(states >= n_states) or np.any(states != states.astype(int)):
        raise ValueError("State indices fall outside the frozen atlas")
    counts = pd.crosstab(pd.Series(groups, name="patient"), pd.Series(states, name="state"))
    counts = counts.reindex(columns=range(n_states), fill_value=0)
    abundance = counts.div(counts.sum(axis=1), axis=0)
    abundance.columns = [f"cluster_{i}" for i in range(n_states)]
    return abundance


def update_state_counts(counts, original_states, edited_states):
    """Update edited neighborhoods while preserving the full patient's denominator."""
    counts = np.asarray(counts)
    before, after = np.asarray(original_states), np.asarray(edited_states)
    if counts.ndim != 1 or np.any(counts < 0) or not counts.sum() > 0 or np.any(counts != counts.astype(int)):
        raise ValueError("Use non-negative integer counts for the complete patient")
    if before.shape != after.shape or before.ndim != 1:
        raise ValueError("Each edited neighborhood needs a before/after state")
    for states in (before, after):
        if np.any(states < 0) or np.any(states >= len(counts)) or np.any(states != states.astype(int)):
            raise ValueError("State indices fall outside the frozen atlas")
    removed = np.bincount(before.astype(int), minlength=len(counts))
    if np.any(removed > counts):
        raise ValueError("Edited neighborhoods exceed the original state counts")
    return counts.astype(int) - removed + np.bincount(after.astype(int), minlength=len(counts))


def clinical_score(counts, head):
    """Evaluate a frozen fold-specific scaler and logistic pCR readout."""
    counts = np.asarray(counts)
    if counts.ndim != 1 or len(counts) != len(head["scale_mean"]) or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("Patient state counts do not match the clinical head")
    # Original pipeline passes float32 abundances through StandardScaler.
    x = (counts / counts.sum()).astype(np.float32)
    x -= head["scale_mean"]
    x /= head["scale_scale"]
    logit = float(x @ head["coefficient"] + head["intercept"])
    return {"logit": logit, "probability": float(expit(np.clip(logit, -50, 50)))}
