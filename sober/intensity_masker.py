import json
import os
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import rankdata

from .config import Config


def intensity_masking_record(config: Any) -> Dict[str, Any]:
    """Record masking settings, including how ratios become integer mask counts."""
    return {
        "center_cell": {
            "enable_center_intensity_masking": bool(getattr(config, "enable_center_intensity_masking", False)),
            "center_intensity_mask_probability": float(getattr(config, "center_intensity_mask_probability", 0.0)),
            "center_intensity_min_mask_ratio": float(getattr(config, "center_intensity_min_mask_ratio", 0.0)),
            "center_intensity_max_mask_ratio": float(getattr(config, "center_intensity_max_mask_ratio", 0.0)),
            "center_intensity_mask_value": float(getattr(config, "center_intensity_mask_value", 0.0)),
            "center_mask_strategy": str(getattr(config, "center_mask_strategy", "random")),
            "preserve_top_biomarkers": int(getattr(config, "preserve_top_biomarkers", 0)),
            "implementation_note": (
                "If enabled and Bernoulli(mask_probability) passes: sample count K ~ UniformInteger["
                "min(max(1, floor(n*min_ratio)), K_max), K_max], where K_max=floor(n*max_ratio)."
            ),
        },
        "neighbor_cells": {
            "enable_neighbor_intensity_masking": bool(getattr(config, "enable_neighbor_intensity_masking", False)),
            "neighbor_intensity_mask_probability": float(getattr(config, "neighbor_intensity_mask_probability", 0.0)),
            "neighbor_intensity_min_mask_ratio": float(getattr(config, "neighbor_intensity_min_mask_ratio", 0.0)),
            "neighbor_intensity_max_mask_ratio": float(getattr(config, "neighbor_intensity_max_mask_ratio", 0.0)),
            "neighbor_intensity_mask_value": float(getattr(config, "neighbor_intensity_mask_value", 0.0)),
            "neighbor_mask_strategy": str(getattr(config, "neighbor_mask_strategy", "random")),
            "implementation_note": (
                "Per row: draw K ~ UniformInteger[min(max(1,floor(n*min)), K_max), K_max], K_max=floor(n*max); "
                "if Bernoulli(neighbor_intensity_mask_probability) fails, K=0. Then mask K channels."
            ),
        },
    }


def save_intensity_masking_record(config: Any, save_dir: str) -> Optional[str]:
    """Write intensity_masking.json under save_dir; return path or None on failure."""
    if not save_dir:
        return None
    try:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "intensity_masking.json")
        rec = intensity_masking_record(config)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        return path
    except OSError:
        return None


class IntensityMasker:
    """Mask center and neighbor intensities without renormalization."""

    def __init__(self, config: Config):
        self.config = config
        for group in ("center", "neighbor"):
            low = getattr(config, f"{group}_intensity_min_mask_ratio")
            high = getattr(config, f"{group}_intensity_max_mask_ratio")
            if not 0 <= low <= high <= 1:
                raise ValueError(f"{group} mask ratios must satisfy 0 <= min <= max <= 1")

    def mask_sample(self, sample_data):
        center_intensities = sample_data['intensities'][0:1]
        neighbor_intensities = sample_data['intensities'][1:]

        center_masked, center_mask_flag = self.apply_center_intensity_masking(center_intensities)
        neighbor_masked, neighbor_mask_flag = self.apply_neighbor_intensity_masking(neighbor_intensities)

        masked_items = {sample_data['biomarkers'][i]: center_intensities[0, i]
                        for i in range(len(sample_data['biomarkers'])) if center_mask_flag[0, i] == 1}

        sample_data['masked_intensities'] = np.concatenate([center_masked, neighbor_masked], 0)
        sample_data['mask_flags'] = np.concatenate([center_mask_flag, neighbor_mask_flag], 0)
        sample_data['masked_items'] = masked_items
        return sample_data

    def apply_center_intensity_masking(self, center_intensities):
        """Return masked intensities and a binary mask for the (1, biomarkers) center row."""
        assert center_intensities.ndim == 2 and center_intensities.shape[0] == 1
        if center_intensities.shape[1] == 0:
            raise ValueError("Center cell has no intensities")

        if not self.config.enable_center_intensity_masking:
            return center_intensities, np.zeros_like(center_intensities)

        if np.random.random() > self.config.center_intensity_mask_probability:
            return center_intensities, np.zeros_like(center_intensities)

        num_biomarkers = center_intensities.shape[1]
        max_mask = min(num_biomarkers, int(num_biomarkers * self.config.center_intensity_max_mask_ratio))
        min_mask = min(max_mask, max(1, int(num_biomarkers * self.config.center_intensity_min_mask_ratio)))
        num_to_mask = np.random.randint(min_mask, max_mask + 1, (1,))

        protected_mask = np.zeros_like(center_intensities, dtype=int)
        if self.config.preserve_top_biomarkers > 0:
            protected_mask = self._select_masks(
                center_intensities, np.array([self.config.preserve_top_biomarkers]),
                'highest', np.zeros_like(center_intensities))

        mask_flag = self._select_masks(
            center_intensities,
            num_to_mask,
            self.config.center_mask_strategy,
            protected_mask)

        masked_intensities = np.where(mask_flag, self.config.center_intensity_mask_value, center_intensities)
        return masked_intensities, mask_flag

    def apply_neighbor_intensity_masking(self, neighbor_intensities):
        """Return masked intensities and a binary mask for the neighbor rows."""
        if not self.config.enable_neighbor_intensity_masking:
            return neighbor_intensities, np.zeros_like(neighbor_intensities)

        n_neighbors, num_biomarkers = neighbor_intensities.shape
        max_mask = min(num_biomarkers, int(num_biomarkers * self.config.neighbor_intensity_max_mask_ratio))
        min_mask = min(max_mask, max(1, int(num_biomarkers * self.config.neighbor_intensity_min_mask_ratio)))
        num_to_mask = np.random.randint(min_mask, max_mask + 1, (n_neighbors,))
        num_to_mask = np.where(np.random.rand(n_neighbors) < self.config.neighbor_intensity_mask_probability, num_to_mask, 0)

        mask_flag = self._select_masks(
            neighbor_intensities,
            num_to_mask,
            self.config.neighbor_mask_strategy,
            np.zeros_like(neighbor_intensities, dtype=bool))

        masked_intensities = np.where(mask_flag, self.config.neighbor_intensity_mask_value, neighbor_intensities)
        return masked_intensities, mask_flag

    def _select_masks(self,
                      intensities: np.ndarray,
                      num_to_mask: np.ndarray,
                      strategy: str,
                      protected_mask: np.ndarray) -> np.ndarray:
        """Select a mask for each row, excluding protected entries.

        num_to_mask contains one count per row. Ties use rankdata's average ranks.
        """
        assert intensities.ndim == 2
        assert intensities.shape == protected_mask.shape
        assert num_to_mask.ndim == 1 and num_to_mask.shape[0] == intensities.shape[0]
        available_mask = 1 - (protected_mask * 1)

        if np.sum(available_mask) == 0 or np.sum(num_to_mask) == 0:
            return np.zeros_like(intensities, dtype=int)

        if strategy == 'random':
            random_values = np.random.rand(*intensities.shape)
            random_values = np.where(available_mask, random_values, np.inf)
            random_ranks = rankdata(random_values, axis=1)
            selected_mask = (random_ranks <= num_to_mask[:, None]).astype(int)

        elif strategy == 'highest':
            _intensities = np.where(intensities == intensities, intensities, -np.inf)  # NaNs rank last.
            _intensities = np.where(available_mask, _intensities, -np.inf)
            ranks = rankdata(-_intensities, axis=1)
            selected_mask = (ranks <= num_to_mask[:, None]).astype(int)

        elif strategy == 'lowest':
            _intensities = np.where(intensities == intensities, intensities, np.inf)  # NaNs rank last.
            _intensities = np.where(available_mask, _intensities, np.inf)
            ranks = rankdata(_intensities, axis=1)
            selected_mask = (ranks <= num_to_mask[:, None]).astype(int)

        elif strategy == 'middle':
            medians = np.nanmedian(intensities, axis=1, keepdims=True)
            _intensities = np.where(intensities == intensities, intensities, np.inf)
            _intensities = np.where(available_mask, _intensities, np.inf)
            deviations = np.abs(_intensities - medians)
            ranks = rankdata(deviations, axis=1)
            selected_mask = (ranks <= num_to_mask[:, None]).astype(int)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        selected_mask = ((selected_mask * available_mask) > 0).astype(int)
        return selected_mask
