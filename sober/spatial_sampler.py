import json
import os
import random
from typing import Dict, List

import numpy as np
from scipy.spatial import KDTree


class SpatialSampler:
    """Sample center-relative neighborhoods using a KDTree."""

    def __init__(self, config):
        self.config = config

    def find_neighbors(self, center_idx: int, kdtree: KDTree) -> List[int]:
        """Return up to max_neighbors nearest cells within the radius, excluding the center."""
        center_coord = kdtree.data[center_idx]

        n_to_query = self.config.max_neighbors + 1
        distances, indices = kdtree.query(center_coord, k=n_to_query)
        filtered_neighbors = [
            idx for dist, idx in zip(distances, indices)
            if dist <= self.config.neighbor_distance_threshold and idx != center_idx
        ]
        # A coordinate tie can leave the center outside the k-nearest results.
        return filtered_neighbors[:self.config.max_neighbors]

    def _empty_sample(self):
        return {
            'coordinates': [],
            'intensities': [],
            'cell_ids': [],
            'cell_types': [],
            'cell_type_hierarchy': [],
            'biomarkers': [],
            'original_center_idx': None,
            'original_indices': [],
            'center_x': None,
            'center_y': None,
        }

    def _assemble_sample(self, region_data: Dict, all_indices):
        all_indices = [int(idx) for idx in all_indices]
        if not all_indices:
            return self._empty_sample()
        center_idx = all_indices[0]
        coordinates = region_data['coordinates']
        cell_types = region_data['cell_types']
        cell_type_hierarchy = region_data.get('cell_type_hierarchy')
        center_xy = np.asarray(coordinates[center_idx], dtype=float).reshape(2)
        relative_coords = coordinates[all_indices] - coordinates[center_idx].reshape((1, 2))
        out = {
            'coordinates': relative_coords,
            'intensities': region_data['intensities'][all_indices],
            'cell_ids': [region_data['cell_ids'][i] for i in all_indices],
            'cell_types': [cell_types[i] for i in all_indices] if cell_types else [],
            'biomarkers': region_data['biomarkers'],
            'study_name': region_data['study_name'],
            'region_name': region_data['region_name'],
            'original_center_idx': center_idx,
            'original_indices': all_indices,
            'center_x': float(center_xy[0]),
            'center_y': float(center_xy[1]),
        }
        out['cell_type_hierarchy'] = (
            [cell_type_hierarchy[i] for i in all_indices] if cell_type_hierarchy else []
        )
        return out

    def sample_region(self, region_data: Dict) -> Dict:
        """Choose a center uniformly and return its spatial neighborhood."""
        if region_data['num_cells'] == 0:
            return self._empty_sample()
        center_idx = random.randint(0, len(region_data['coordinates']) - 1)
        neighbor_indices = self.find_neighbors(center_idx, region_data['kdtree'])
        return self._assemble_sample(region_data, [center_idx] + neighbor_indices)

    def sample_region_at(self, region_data: Dict, center_idx: int) -> Dict:
        if region_data['num_cells'] == 0:
            return self._empty_sample()
        neighbor_indices = self.find_neighbors(center_idx, region_data['kdtree'])
        return self._assemble_sample(region_data, [int(center_idx)] + neighbor_indices)

    def sample_region_with_indices(self, region_data: Dict, all_indices) -> Dict:
        """Rebuild a neighborhood from saved indices, with the center first."""
        if region_data['num_cells'] == 0:
            return self._empty_sample()
        all_indices = [int(idx) for idx in all_indices]
        n_cells = int(region_data['num_cells'])
        if not all_indices:
            return self._empty_sample()
        if any(idx < 0 or idx >= n_cells for idx in all_indices):
            raise ValueError(
                "Frozen neighborhood index is out of range for %s/%s"
                % (
                    region_data.get('study_name', 'unknown'),
                    region_data.get('region_name', 'unknown'),
                )
            )
        center_idx = all_indices[0]
        unique = [center_idx]
        seen = {center_idx}
        for idx in all_indices[1:]:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
        return self._assemble_sample(region_data, unique)


def _distribution_summary(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            'n': 0, 'mean': None, 'std': None, 'min': None,
            'p10': None, 'median': None, 'p90': None, 'max': None,
        }
    return {
        'n': int(values.size),
        'mean': float(values.mean()),
        'std': float(values.std()),
        'min': float(values.min()),
        'p10': float(np.percentile(values, 10)),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
        'max': float(values.max()),
    }


def build_neighborhood_sampling_summary(config, dataset, seed=42):
    """Summarize neighborhoods from deterministic samples of the training regions."""
    sampler = SpatialSampler(config)
    rng = np.random.RandomState(seed)
    max_centers = max(
        1, int(getattr(config, 'neighborhood_summary_max_centers_per_region', 2000))
    )
    max_neighbors = int(config.max_neighbors)
    threshold = float(config.neighbor_distance_threshold)
    microns_per_unit = float(getattr(config, 'microns_per_model_unit', 1.0))

    region_rows = []
    study_counts = {}
    all_neighbor_counts = []
    all_furthest_distances = []
    selected_distance_sum = 0.0
    selected_distance_count = 0
    selected_distance_max = 0.0

    for region_idx in range(len(dataset)):
        region = dataset[region_idx]
        coordinates = np.asarray(region['coordinates'], dtype=float)
        n_cells = len(coordinates)
        n_centers = min(n_cells, max_centers)
        center_indices = (
            np.arange(n_cells)
            if n_centers == n_cells
            else np.sort(rng.choice(n_cells, size=n_centers, replace=False))
        )

        neighbor_counts = []
        furthest_distances = []
        for center_idx in center_indices:
            neighbors = sampler.find_neighbors(int(center_idx), region['kdtree'])
            neighbor_counts.append(len(neighbors))
            if neighbors:
                distances = np.linalg.norm(
                    coordinates[np.asarray(neighbors)] - coordinates[int(center_idx)],
                    axis=1,
                )
                furthest_distances.append(float(distances.max()))
                selected_distance_sum += float(distances.sum())
                selected_distance_count += int(distances.size)
                selected_distance_max = max(selected_distance_max, float(distances.max()))

        neighbor_counts = np.asarray(neighbor_counts, dtype=int)
        study = str(region.get('study_name', 'unknown'))
        row = {
            'study_name': study,
            'region_name': str(region.get('region_name', region_idx)),
            'num_cells': int(n_cells),
            'sampled_centers': int(n_centers),
            'sampled_center_fraction': float(n_centers / max(1, n_cells)),
            'neighbor_count': _distribution_summary(neighbor_counts),
            'zero_neighbor_fraction': float(np.mean(neighbor_counts == 0)),
            'max_neighbor_cap_fraction': (
                float(np.mean(neighbor_counts >= max_neighbors))
                if max_neighbors > 0 else 0.0
            ),
            'furthest_neighbor_distance_model_units': _distribution_summary(furthest_distances),
        }
        region_rows.append(row)
        all_neighbor_counts.extend(neighbor_counts.tolist())
        all_furthest_distances.extend(furthest_distances)

        study_row = study_counts.setdefault(study, {
            'n_regions': 0,
            'total_cells': 0,
            'sampled_centers': 0,
            'neighbor_counts': [],
        })
        study_row['n_regions'] += 1
        study_row['total_cells'] += int(n_cells)
        study_row['sampled_centers'] += int(n_centers)
        study_row['neighbor_counts'].extend(neighbor_counts.tolist())

    per_study = {}
    for study, row in sorted(study_counts.items()):
        counts = np.asarray(row.pop('neighbor_counts'), dtype=int)
        row['neighbor_count'] = _distribution_summary(counts)
        row['zero_neighbor_fraction'] = float(np.mean(counts == 0))
        row['max_neighbor_cap_fraction'] = (
            float(np.mean(counts >= max_neighbors)) if max_neighbors > 0 else 0.0
        )
        per_study[study] = row

    all_neighbor_counts = np.asarray(all_neighbor_counts, dtype=int)
    expected_subgraphs = len(dataset) * int(config.n_subgraphs_per_region)
    expected_batches = max(1, expected_subgraphs // int(config.batch_size))
    return {
        'version': 1,
        'scope': 'training_regions',
        'configuration': {
            'neighbor_distance_threshold_model_units': threshold,
            'microns_per_model_unit': microns_per_unit,
            'physical_radius_microns': threshold * microns_per_unit,
            'max_neighbors': max_neighbors,
            'min_cells_per_region': int(config.min_cells_per_region),
            'n_subgraphs_per_region_per_epoch': int(config.n_subgraphs_per_region),
            'batch_size': int(config.batch_size),
        },
        'diagnostic_sampling': {
            'seed': int(seed),
            'max_centers_per_region': max_centers,
            'method': 'all centers when possible; otherwise uniform sampling without replacement',
        },
        'dataset': {
            'n_regions': int(len(dataset)),
            'n_studies': int(len(study_counts)),
            'total_cells': int(sum(row['num_cells'] for row in region_rows)),
            'sampled_centers': int(len(all_neighbor_counts)),
            'expected_subgraphs_per_epoch': int(expected_subgraphs),
            'expected_lazy_batches_per_epoch': int(expected_batches),
            'region_sampling_note': 'Training samples regions uniformly, then samples one center cell uniformly.',
        },
        'aggregate': {
            'neighbor_count': _distribution_summary(all_neighbor_counts),
            'sequence_length_including_center': _distribution_summary(all_neighbor_counts + 1),
            'zero_neighbor_fraction': (
                float(np.mean(all_neighbor_counts == 0)) if all_neighbor_counts.size else None
            ),
            'max_neighbor_cap_fraction': (
                float(np.mean(all_neighbor_counts >= max_neighbors))
                if all_neighbor_counts.size and max_neighbors > 0 else 0.0
            ),
            'furthest_neighbor_distance_model_units': _distribution_summary(
                all_furthest_distances
            ),
            'furthest_neighbor_distance_microns': _distribution_summary(
                np.asarray(all_furthest_distances) * microns_per_unit
            ),
            'selected_neighbor_distance_model_units': {
                'n': int(selected_distance_count),
                'mean': (
                    float(selected_distance_sum / selected_distance_count)
                    if selected_distance_count else None
                ),
                'max': float(selected_distance_max) if selected_distance_count else None,
            },
            'selected_neighbor_distance_microns': {
                'n': int(selected_distance_count),
                'mean': (
                    float(selected_distance_sum / selected_distance_count * microns_per_unit)
                    if selected_distance_count else None
                ),
                'max': (
                    float(selected_distance_max * microns_per_unit)
                    if selected_distance_count else None
                ),
            },
            'region_mean_neighbor_count': _distribution_summary([
                row['neighbor_count']['mean'] for row in region_rows
            ]),
            'region_balanced_zero_neighbor_fraction': (
                float(np.mean([row['zero_neighbor_fraction'] for row in region_rows]))
                if region_rows else None
            ),
            'region_balanced_max_neighbor_cap_fraction': (
                float(np.mean([row['max_neighbor_cap_fraction'] for row in region_rows]))
                if region_rows else None
            ),
        },
        'per_study': per_study,
        'per_region': region_rows,
    }


def save_neighborhood_sampling_summary(config, dataset, save_dir, seed=42):
    summary = build_neighborhood_sampling_summary(config, dataset, seed=seed)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'neighborhood_sampling_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    return path
