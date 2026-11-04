"""Unit tests for the distribution head and HL-Gauss targets (CPU only).

Mirrors the other test modules: loads model/DLF.py with stubbed BERT/Transformer
imports so the full model runs on CPU.
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]

SPEC = importlib.util.spec_from_file_location(
    'dlf_distribution_head', ROOT / 'trains/singleTask/distribution_head.py'
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

make_centers = MODULE.make_centers
bin_edges = MODULE.bin_edges
gaussian_soft_target = MODULE.gaussian_soft_target
hard_target = MODULE.hard_target
DistributionHead = MODULE.DistributionHead
DistributionCriterion = MODULE.DistributionCriterion

BINS, LO, HI, SIGMA = 66, -3.3, 3.3, 0.075
CENTERS = make_centers(BINS, LO, HI)


class _StubTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.last_null_attention = None

    def forward(self, *args, **kwargs):
        return args[0]


class _StubBert(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, text):
        return text


def _load_model_module():
    trains = types.ModuleType('trains')
    trains.__path__ = [str(ROOT / 'trains')]
    single = types.ModuleType('trains.singleTask')
    single.__path__ = [str(ROOT / 'trains' / 'singleTask')]
    model_pkg = types.ModuleType('trains.singleTask.model')
    model_pkg.__path__ = [str(ROOT / 'trains' / 'singleTask' / 'model')]
    subnets = types.ModuleType('trains.subNets')
    subnets.__path__ = []
    subnets.BertTextEncoder = _StubBert
    transformer_pkg = types.ModuleType('trains.subNets.transformers_encoder')
    transformer_pkg.__path__ = []
    transformer_mod = types.ModuleType(
        'trains.subNets.transformers_encoder.transformer'
    )
    transformer_mod.TransformerEncoder = _StubTransformer
    for name, mod in (
        ('trains', trains), ('trains.singleTask', single),
        ('trains.singleTask.model', model_pkg),
        ('trains.subNets', subnets),
        ('trains.subNets.transformers_encoder', transformer_pkg),
        ('trains.subNets.transformers_encoder.transformer', transformer_mod),
    ):
        sys.modules[name] = mod
    # The model imports the head as ``..distribution_head``.
    sys.modules['trains.singleTask.distribution_head'] = MODULE

    path = ROOT / 'trains' / 'singleTask' / 'model' / 'DLF.py'
    spec = importlib.util.spec_from_file_location(
        'trains.singleTask.model.DLF', path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_args(head_mode='original'):
    return types.SimpleNamespace(
        use_bert=False,
        dst_feature_dim_nheads=(8, 2),
        dataset_name='mosi',
        need_data_aligned=True,
        feature_dims=(16, 9, 9),
        nlevels=1,
        attn_dropout=0.0, attn_dropout_a=0.0, attn_dropout_v=0.0,
        relu_dropout=0.0, embed_dropout=0.0, res_dropout=0.0,
        output_dropout=0.0, text_dropout=0.0, attn_mask=False,
        conv1d_kernel_size_l=5, conv1d_kernel_size_a=5, conv1d_kernel_size_v=5,
        lfa_mode='original',
        fusion_gate_mode='none',
        fusion_gate_init=0.9,
        lfa_pool_mode='last',
        lfa_pool_paths=[],
        head_mode=head_mode,
        head_num_bins=BINS, head_bin_min=LO, head_bin_max=HI,
        head_gauss_sigma=SIGMA, head_ce_weight=0.1,
    )


class CentersTest(unittest.TestCase):
    def test_equal_width_and_count(self):
        self.assertEqual(CENTERS.numel(), BINS)
        steps = CENTERS[1:] - CENTERS[:-1]
        torch.testing.assert_close(steps, steps[0].expand_as(steps))

    def test_edges_bracket_range(self):
        edges = bin_edges(CENTERS)
        self.assertEqual(edges.numel(), BINS + 1)
        self.assertAlmostEqual(edges[0].item(), LO, places=5)
        self.assertAlmostEqual(edges[-1].item(), HI, places=5)

    def test_invalid_range_rejected(self):
        with self.assertRaises(ValueError):
            make_centers(1, LO, HI)
        with self.assertRaises(ValueError):
            make_centers(BINS, HI, LO)


class GaussianTargetTest(unittest.TestCase):
    def test_rows_sum_to_one(self):
        y = torch.tensor([-3.0, -0.4, 0.0, 1.3, 3.0]).reshape(-1, 1)
        q = gaussian_soft_target(y, CENTERS, SIGMA)
        self.assertEqual(tuple(q.shape), (5, BINS))
        torch.testing.assert_close(q.sum(dim=-1), torch.ones(5), rtol=0, atol=1e-5)

    def test_expectation_close_to_label(self):
        y = torch.tensor([-2.0, -0.37, 0.0, 1.3, 2.71]).reshape(-1, 1)
        q = gaussian_soft_target(y, CENTERS, SIGMA)
        expected = (q * CENTERS.reshape(1, -1)).sum(dim=-1)
        # Truncation to [-3.3, 3.3] leaves a small, nonzero gap.
        torch.testing.assert_close(expected, y.reshape(-1), rtol=0, atol=0.02)

    def test_mass_concentrates_near_label(self):
        y = torch.tensor([[1.3]])
        q = gaussian_soft_target(y, CENTERS, SIGMA)
        peak = q.argmax().item()
        torch.testing.assert_close(CENTERS[peak], torch.tensor(1.3), rtol=0, atol=0.05)
        # sigma=0.075, bin width 0.1: mass is spread over a handful of bins around
        # the label, so the peak bin holds less than half but the center bins
        # dominate the tails.
        self.assertGreater(q.max().item(), 0.3)
        near = (CENTERS - 1.3).abs() <= 0.25
        self.assertGreater(q[0, near].sum().item(), 0.99)

    def test_small_sigma_approaches_hard_target(self):
        y = torch.tensor([[0.42]])
        soft = gaussian_soft_target(y, CENTERS, sigma=1e-3)
        hard = hard_target(y, CENTERS)
        torch.testing.assert_close(soft, hard, rtol=0, atol=1e-4)

    def test_out_of_range_label_is_clamped_to_edge_bins(self):
        y = torch.tensor([[-5.0, 5.0]]).reshape(-1, 1)
        q = gaussian_soft_target(y, CENTERS, SIGMA)
        torch.testing.assert_close(q.sum(dim=-1), torch.ones(2), rtol=0, atol=1e-5)
        self.assertEqual(q[0].argmax().item(), 0)
        self.assertEqual(q[1].argmax().item(), BINS - 1)

    def test_invalid_sigma_rejected(self):
        with self.assertRaises(ValueError):
            gaussian_soft_target(torch.tensor([[0.0]]), CENTERS, 0.0)


class HardTargetTest(unittest.TestCase):
    def test_picks_nearest_bin(self):
        y = torch.tensor([[1.32], [-0.01], [3.2]]).reshape(-1, 1)
        t = hard_target(y, CENTERS)
        self.assertEqual(tuple(t.shape), (3, BINS))
        torch.testing.assert_close(t.sum(dim=-1), torch.ones(3))
        chosen = CENTERS[t.argmax(dim=-1)]
        gap = (chosen - y.reshape(-1)).abs()
        self.assertTrue(bool((gap <= 0.05 + 1e-6).all()))


class HeadForwardTest(unittest.TestCase):
    def test_expectation_shape_and_probabilities(self):
        torch.manual_seed(0)
        head = DistributionHead(12, BINS, LO, HI)
        z = torch.randn(4, 12)
        y_hat, probs, logits = head(z)
        self.assertEqual(tuple(y_hat.shape), (4, 1))
        self.assertEqual(tuple(probs.shape), (4, BINS))
        self.assertEqual(tuple(logits.shape), (4, BINS))
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(4), rtol=0, atol=1e-6)
        torch.testing.assert_close(
            y_hat.squeeze(-1), (probs * CENTERS.reshape(1, -1)).sum(dim=-1)
        )

    def test_gradients_flow(self):
        head = DistributionHead(12, BINS, LO, HI)
        z = torch.randn(3, 12, requires_grad=True)
        y_hat, _, _ = head(z)
        y_hat.sum().backward()
        self.assertGreater(z.grad.abs().sum().item(), 0.0)
        self.assertGreater(head.proj.weight.grad.abs().sum().item(), 0.0)


class CriterionTest(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(1)
        # Leaf logits so gradients land on a tensor the test can inspect.
        logits = torch.randn(5, BINS, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        y_hat = (probs * CENTERS.reshape(1, -1)).sum(dim=-1, keepdim=True)
        labels = torch.linspace(-1.0, 1.0, 5).reshape(-1, 1)
        return y_hat, logits, labels

    def test_dist_l1_has_no_ce(self):
        y_hat, logits, labels = self._inputs()
        crit = DistributionCriterion('dist_l1', BINS, LO, HI)
        out = crit(y_hat, logits, labels)
        self.assertIsNone(out['ce'])
        torch.testing.assert_close(out['total'], out['l1'])

    def test_dist_gauss_drops_l1_from_total(self):
        y_hat, logits, labels = self._inputs()
        crit = DistributionCriterion('dist_gauss', BINS, LO, HI, SIGMA, 0.1)
        out = crit(y_hat, logits, labels)
        self.assertFalse(crit.use_l1)
        self.assertTrue(crit.use_ce)
        torch.testing.assert_close(out['total'], 0.1 * out['ce'])
        # l1 is still reported for diagnostics.
        self.assertGreater(out['l1'].item(), 0.0)

    def test_full_mode_adds_both_terms(self):
        y_hat, logits, labels = self._inputs()
        crit = DistributionCriterion('dist_gauss_l1', BINS, LO, HI, SIGMA, 0.1)
        out = crit(y_hat, logits, labels)
        torch.testing.assert_close(out['total'], out['l1'] + 0.1 * out['ce'])

    def test_dist_hard_uses_hard_label_ce(self):
        y_hat, logits, labels = self._inputs()
        crit = DistributionCriterion('dist_hard', BINS, LO, HI)
        out = crit(y_hat, logits, labels)
        self.assertIsNotNone(out['ce'])
        self.assertTrue(out['total'].requires_grad)

    def test_ce_terms_backpropagate(self):
        for mode in ('dist_hard', 'dist_gauss', 'dist_gauss_l1'):
            y_hat, logits, labels = self._inputs()
            crit = DistributionCriterion(mode, BINS, LO, HI, SIGMA, 0.1)
            out = crit(y_hat, logits, labels)
            out['total'].backward()
            self.assertIsNotNone(logits.grad)
            self.assertGreater(logits.grad.abs().sum().item(), 0.0, mode)

    def test_criterion_matches_double_precision_inputs(self):
        # float64 logits/labels with the default float32 buffer must not error.
        logits = torch.randn(3, BINS, dtype=torch.float64, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        y_hat = (probs * CENTERS.double().reshape(1, -1)).sum(-1, keepdim=True)
        labels = torch.tensor([[0.3], [-0.2], [1.1]], dtype=torch.float64)
        for mode in ('dist_hard', 'dist_gauss', 'dist_gauss_l1'):
            crit = DistributionCriterion(mode, BINS, LO, HI, SIGMA, 0.1)
            out = crit(y_hat, logits, labels)
            self.assertTrue(torch.isfinite(out['total']), mode)
            self.assertEqual(out['total'].dtype, torch.float64, mode)

    def test_original_mode_rejected(self):
        with self.assertRaises(ValueError):
            DistributionCriterion('original', BINS, LO, HI)


class ModelModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_model_module()

    def _model(self, head_mode):
        torch.manual_seed(0)
        return self.module.DLF(_make_args(head_mode)).eval()

    def test_original_keeps_scalar_head_and_no_distance_head(self):
        model = self._model('original')
        self.assertIsNotNone(model.out_layer)
        self.assertIsNone(model.dist_head)

    def test_dist_modes_replace_scalar_head(self):
        for mode in ('dist_l1', 'dist_hard', 'dist_gauss', 'dist_gauss_l1'):
            model = self._model(mode)
            self.assertIsNone(model.out_layer, mode)
            self.assertIsNotNone(model.dist_head, mode)
            self.assertEqual(model.dist_head.num_bins, BINS, mode)

    def test_forward_contract_all_modes(self):
        text = torch.randn(2, 50, 16)
        audio = torch.randn(2, 50, 9)
        video = torch.randn(2, 50, 9)
        for mode in ('original', 'dist_l1', 'dist_hard', 'dist_gauss', 'dist_gauss_l1'):
            model = self._model(mode)
            model.train()
            out = model(text, audio, video)
            self.assertEqual(tuple(out['output_logit'].shape), (2, 1), mode)
            if mode == 'original':
                self.assertIsNone(out['output_probs'], mode)
                self.assertIsNone(out['output_bin_logits'], mode)
            else:
                self.assertEqual(tuple(out['output_probs'].shape), (2, BINS), mode)
                torch.testing.assert_close(
                    out['output_probs'].sum(dim=-1), torch.ones(2),
                    rtol=0, atol=1e-5,
                )
            out['output_logit'].sum().backward()

    def test_distribution_head_state_dict_shapes(self):
        state = self._model('dist_gauss_l1').state_dict()
        self.assertIn('dist_head.proj.weight', state)
        self.assertIn('dist_head.centers', state)
        self.assertNotIn('out_layer.weight', state)

    def test_invalid_head_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.module.DLF(_make_args('bogus'))


if __name__ == '__main__':
    unittest.main()
