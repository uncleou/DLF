"""Numerical checks requiring only PyTorch: python -m unittest discover -s tests."""

import importlib
import json
import math
from pathlib import Path
import sys
import types
import unittest

import torch


# Load just the losses: trains/__init__.py eagerly imports the full training
# stack, including NVIDIA utilities that are unnecessary for these CPU tests.
ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("_dlf_loss_tests")
package.__path__ = [str(ROOT / "trains" / "singleTask")]
sys.modules[package.__name__] = package
alignment = importlib.import_module("_dlf_loss_tests.shared_alignment")
SharedAlignmentLoss = alignment.SharedAlignmentLoss


def reference_losses(features, labels, sigma, tau):
    """Literal scalar implementation of the agreed formulas, independent of autograd."""
    vectors = []
    for feature in features:
        rows = feature.detach().tolist()
        vectors.append([
            [x / max(math.sqrt(sum(v * v for v in row)), 1e-8) for x in row]
            for row in rows
        ])
    batch, dim = features[0].shape
    atp = sum(
        sum((vectors[m][i][k] - vectors[n][i][k]) ** 2 for k in range(dim))
        for i in range(batch) for m, n in ((0, 1), (0, 2), (1, 2))
    ) / (3 * batch)
    centroids = [[sum(vectors[m][i][k] for m in range(3)) / 3
                  for k in range(dim)] for i in range(batch)]
    y = labels.detach().reshape(-1).tolist()
    kl = 0.0
    if batch >= 3:
        for i in range(batch):
            q = [math.exp(-abs(y[i] - y[j]) / sigma)
                 for j in range(batch) if j != i]
            p = [math.exp(-sum((a - b) ** 2 for a, b in zip(centroids[i], centroids[j])) / tau)
                 for j in range(batch) if j != i]
            q_sum, p_sum = sum(q), sum(p)
            kl += sum((a / q_sum) * math.log((a / q_sum) / (b / p_sum))
                      for a, b in zip(q, p)) / batch
    return atp, kl


class SharedAlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.features = [torch.randn(4, 5, dtype=torch.double, requires_grad=True)
                         for _ in range(3)]
        self.labels = torch.tensor([-2.0, -0.3, 0.7, 2.5], dtype=torch.double)

    def test_matches_scalar_formulas_and_final_weights(self):
        loss = SharedAlignmentLoss("atp_sent", 0.17, 0.31, 0.8, 0.6)
        result = loss(*self.features, self.labels)
        atp, sent = reference_losses(self.features, self.labels, 0.8, 0.6)
        self.assertAlmostEqual(result['atp'].item(), atp, places=12)
        self.assertAlmostEqual(result['sent'].item(), sent, places=12)
        self.assertAlmostEqual(result['total'].item(), 0.17 * atp + 0.31 * sent, places=12)

    def test_atp_sums_features_instead_of_averaging_them(self):
        t = torch.tensor([[1., 0.]])
        a = torch.tensor([[0., 1.]])
        v = torch.tensor([[-1., 0.]])
        result = SharedAlignmentLoss("atp")(t, a, v, torch.tensor([0.]))
        self.assertAlmostEqual(result['atp'].item(), 8 / 3, places=6)

    def test_original_matches_old_triplet_construction(self):
        t, a, v = self.features
        ids, feats = [], []
        for i in range(len(self.labels)):
            for h in (t, v, a):
                feats.append(h[i].view(1, -1))
                ids.append(self.labels[i].view(1, -1))
        old = alignment.HingeLoss()(torch.cat(ids), torch.cat(feats))
        result = SharedAlignmentLoss("original")(*self.features, self.labels)
        torch.testing.assert_close(result['triplet'], old)
        torch.testing.assert_close(result['total'], 0.01 * old)

    def test_all_six_modes(self):
        full = SharedAlignmentLoss("original_atp_sent", 0.17, 0.31)(*self.features, self.labels)
        expected = {
            "original": 0.01 * full['triplet'],
            "none": full['total'].new_zeros(()),
            "atp": 0.17 * full['atp'],
            "sent": 0.31 * full['sent'],
            "atp_sent": 0.17 * full['atp'] + 0.31 * full['sent'],
            "original_atp_sent": full['total'],
        }
        for mode, target in expected.items():
            with self.subTest(mode=mode):
                result = SharedAlignmentLoss(mode, 0.17, 0.31)(*self.features, self.labels)
                torch.testing.assert_close(result['total'], target)

    def test_each_loss_backpropagates_to_all_modalities_but_not_labels(self):
        for mode in ("atp", "sent"):
            features = [x.detach().clone().requires_grad_() for x in self.features]
            labels = self.labels.clone().requires_grad_()
            SharedAlignmentLoss(mode)(*features, labels)['total'].backward()
            for h in features:
                self.assertTrue(torch.isfinite(h.grad).all())
                self.assertGreater(h.grad.abs().sum().item(), 0)
            self.assertIsNone(labels.grad)

    def test_gradients_match_finite_differences(self):
        loss = SharedAlignmentLoss("atp_sent", 0.17, 0.31, 0.8, 0.6)
        self.assertTrue(torch.autograd.gradcheck(
            lambda *h: loss(*h, self.labels)['total'], tuple(self.features)
        ))

    def test_small_batches_return_differentiable_zero_kl(self):
        for batch in (1, 2):
            features = [x[:batch].detach().clone().requires_grad_() for x in self.features]
            result = SharedAlignmentLoss("sent")(*features, self.labels[:batch])
            self.assertEqual(result['sent'].item(), 0)
            result['total'].backward()
            for h in features:
                torch.testing.assert_close(h.grad, torch.zeros_like(h))

    def test_zero_vectors_and_sharp_targets_are_finite(self):
        for mode in ("atp_sent", "original_atp_sent"):
            for zero_vectors in (False, True):
                features = [(torch.zeros_like(x) if zero_vectors else x.detach().clone())
                            .requires_grad_() for x in self.features]
                result = SharedAlignmentLoss(mode, sent_sigma=1e-6, sent_tau=1e-6)(
                    *features, self.labels
                )
                self.assertTrue(torch.isfinite(result['total']))
                result['total'].backward()
                self.assertTrue(all(torch.isfinite(h.grad).all() for h in features))

    def test_equal_labels_and_identical_features_have_zero_losses(self):
        features = [torch.ones(4, 5, dtype=torch.double, requires_grad=True) for _ in range(3)]
        for mode in SharedAlignmentLoss.MODES:
            result = SharedAlignmentLoss(mode)(*features, torch.zeros(4))
            self.assertEqual(result['total'].item(), 0)
            result['total'].backward()

    def test_invariance_to_modality_and_batch_permutations(self):
        loss = SharedAlignmentLoss("atp_sent")
        original = loss(*self.features, self.labels)
        order = torch.tensor([2, 0, 3, 1])
        permuted = loss(*(h[order] for h in reversed(self.features)), self.labels[order])
        for key in original:
            torch.testing.assert_close(original[key], permuted[key])

    def test_inputs_are_not_modified(self):
        before = [h.detach().clone() for h in self.features]
        SharedAlignmentLoss("atp_sent")(*self.features, self.labels)
        for actual, expected in zip(self.features, before):
            torch.testing.assert_close(actual, expected)

    def test_invalid_configuration_and_shapes(self):
        for kwargs in ({"mode": "typo"}, {"atp_weight": -1}, {"sent_weight": float('nan')},
                       {"sent_sigma": 0}, {"sent_tau": -1}, {"sent_tau": float('inf')}):
            with self.assertRaises(ValueError):
                SharedAlignmentLoss(**kwargs)
        with self.assertRaises(ValueError):
            SharedAlignmentLoss()(self.features[0][:, :2], *self.features[1:], self.labels)
        with self.assertRaises(ValueError):
            SharedAlignmentLoss()(*self.features, self.labels[:2])

    def test_repository_config_enables_new_objective(self):
        config = json.loads((ROOT / 'config' / 'config.json').read_text())['DLF']
        for dataset in ('mosi', 'mosei'):
            args = dict(config['commonParams'], **config['datasetParams'][dataset])
            self.assertEqual(args['shared_loss_mode'], 'atp_sent')
            loss = SharedAlignmentLoss(args['shared_loss_mode'], args['atp_weight'],
                                       args['sent_weight'], args['sent_sigma'], args['sent_tau'])
            self.assertTrue(torch.isfinite(loss(*self.features, self.labels)['total']))


if __name__ == '__main__':
    unittest.main()
