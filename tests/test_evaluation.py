import copy
import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import torch
from scipy.spatial import KDTree

from sober.biomarker_embedder import BiomarkerEmbedder
from sober.config import Config
from sober.evaluation import (
    annotation_metrics, extract_embeddings, predict_cell_types,
    reconstruct, reconstruction_metrics, regression_metrics,
    sample_reference_neighborhoods,
)
from sober.model import SOBER


class FixedMasker:
    def mask_sample(self, sample):
        flags = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        sample['mask_flags'] = flags
        sample['masked_intensities'] = np.where(flags, 0, sample['intensities'])
        sample['masked_items'] = dict(zip(sample['biomarkers'], sample['intensities'][0]))
        return sample


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        config = Config(device='cpu', d_model=16, learnable_embedding_dim=16,
                        nhead=2, num_encoder_layers=1, dim_feedforward=32)
        with redirect_stdout(io.StringIO()):
            embedder = BiomarkerEmbedder(config)
            embedder.build_biomarker_vocab(['CD3e', 'CD8', 'CD20'])
            cls.model = SOBER(config, biomarker_embedder=embedder)
        cls.sample = {
            'study_name': 'study', 'region_name': 'region', 'original_center_idx': 0,
            'cell_ids': ['center', 'neighbor1', 'neighbor2'],
            'coordinates': np.array([[0, 0], [1, 0], [0, 1]], dtype=float),
            'biomarkers': ['CD3e', 'CD8', 'CD20'],
            'intensities': np.array([[100, 200, 300], [2, 20, 200], [4, 40, 400]], dtype=float),
        }

    def test_shared_mask_baseline_excludes_center_and_hidden_values(self):
        model = self.model
        original = model.intensity_masker
        model.intensity_masker = FixedMasker()
        try:
            table = reconstruct(model, [self.sample])
        finally:
            model.intensity_masker = original
        np.testing.assert_array_equal(table['observed'], [100, 200, 300])
        np.testing.assert_allclose(table['neighbor_mean'], [3, 40, np.nan], equal_nan=True)
        self.assertEqual(table['n_visible_neighbors'].tolist(), [2, 1, 0])
        self.assertEqual(reconstruction_metrics(table).set_index('method').loc['neighbor_mean', 'n'], 2)
        self.assertNotIn('mask_flags', self.sample)

    def test_seeded_masks_and_state_restoration(self):
        model = self.model
        model.train()
        model.masking = False
        samples = [dict(copy.deepcopy(self.sample), original_center_idx=i) for i in range(3)]
        np.random.seed(123)
        expected_next = np.random.rand()
        np.random.seed(123)
        a = reconstruct(model, samples, batch_size=1, seed=42)
        self.assertEqual(np.random.rand(), expected_next)
        b = reconstruct(model, samples, batch_size=3, seed=42)
        self.assertEqual(a['marker'].tolist(), b['marker'].tolist())
        np.testing.assert_allclose(a['sober'], b['sober'], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(a['neighbor_mean'], b['neighbor_mean'], equal_nan=True)
        embeddings = extract_embeddings(model, samples)
        self.assertEqual(embeddings.shape, (3, 16))
        self.assertTrue(model.training)
        self.assertFalse(model.masking)
        self.assertTrue(all('mask_flags' not in s for s in samples))
        with self.assertRaises(ValueError):
            reconstruct(model, [samples[0], samples[0]])

    def test_finite_and_constant_regression_pairs(self):
        result = regression_metrics([1, 2, np.nan, 5], [1, 4, 3, np.inf])
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['mse'], 2)
        constant = regression_metrics([2, 2], [1, 3])
        self.assertTrue(np.isnan(constant['pearson_r']))
        self.assertTrue(np.isnan(constant['spearman_r']))
        self.assertEqual(constant['mse'], 1)
        self.assertTrue(np.isnan(regression_metrics([1], [2])['mse']))

    def test_cosine_knn_and_unseen_query_label(self):
        reference = np.array([[1., 0], [2, 0], [0, 1], [0, 2]])
        query = np.array([[10., 1], [1, 10], [1, 0]])
        predicted = predict_cell_types(reference, ['A', 'A', 'B', 'B'], query, k=1)
        self.assertEqual(predicted.tolist(), ['A', 'B', 'A'])
        scores = annotation_metrics(['A', 'B', 'C'], predicted)
        self.assertAlmostEqual(scores['accuracy'], 2 / 3)
        self.assertAlmostEqual(scores['f1_weighted'], 5 / 9)
        self.assertEqual(scores['n_cells'], 3)
        with self.assertRaises(ValueError):
            predict_cell_types(reference, ['A', 'A', 'B', 'B'], query, k=10)

    def test_balanced_reference_keeps_regions_and_unique_centers(self):
        regions = []
        for name, labels in [('r1', ['A', 'A', 'B']), ('r2', ['A', 'A', 'C'])]:
            region = dict(copy.deepcopy(self.sample), region_name=name, num_cells=3,
                          cell_types=labels, cell_type_hierarchy=[(label, '') for label in labels])
            region['kdtree'] = KDTree(region['coordinates'])
            regions.append(region)
        samples = sample_reference_neighborhoods(regions, self.model.config, n_per_class=2)
        labels = [s['cell_type_hierarchy'][0][0] for s in samples]
        self.assertEqual(sorted(labels), ['A', 'A', 'B', 'C'])
        keys = [(s['region_name'], s['original_center_idx']) for s in samples]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(len(s['cell_ids']) == 3 for s in samples))
        with self.assertRaises(ValueError):
            sample_reference_neighborhoods([regions[0], regions[0]], self.model.config)


if __name__ == '__main__':
    unittest.main()
