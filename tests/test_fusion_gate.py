"""Unit tests for the scalar audio / video fusion gates (CPU only).

Loads trains/singleTask/model/DLF.py without importing the heavy BERT stack by
stubbing the two relative imports it needs (`trains.subNets` and the Transformer).
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


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

    # Load as a real submodule so the `from ...subNets import ...` relative
    # imports in DLF.py resolve against the stubbed package hierarchy.
    path = ROOT / 'trains' / 'singleTask' / 'model' / 'DLF.py'
    spec = importlib.util.spec_from_file_location(
        'trains.singleTask.model.DLF', path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_args(mode):
    return types.SimpleNamespace(
        use_bert=False,
        dst_feature_dim_nheads=(8, 2),
        dataset_name='mosi',
        need_data_aligned=True,
        # orig dims must differ from dst (8) so the Conv1d projections run and
        # shorten the sequence from 50 to 46, matching the align_* layers.
        feature_dims=(16, 9, 9),
        nlevels=1,
        attn_dropout=0.0, attn_dropout_a=0.0, attn_dropout_v=0.0,
        relu_dropout=0.0, embed_dropout=0.0, res_dropout=0.0,
        output_dropout=0.0, text_dropout=0.0, attn_mask=False,
        conv1d_kernel_size_l=5, conv1d_kernel_size_a=5, conv1d_kernel_size_v=5,
        lfa_mode='original',
        fusion_gate_mode=mode,
        fusion_gate_init=0.9,
    )


class FusionGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_model_module()

    def _model(self, mode):
        torch.manual_seed(0)
        return self.module.DLF(_make_args(mode)).eval()

    def test_none_matches_ungated_path(self):
        model = self._model('fixed')
        h_l = torch.randn(4, 8)
        h_a = torch.randn(4, 8)
        h_v = torch.randn(4, 8)
        with torch.no_grad():
            # 'fixed' with init 0.9 must scale v/a branches by 0.9.
            model.fusion_gate_mode = 'none'
            self.assertIsNone(model._gate_audio(h_l, h_a))
            self.assertIsNone(model._gate_video(h_l, h_v))
            model.fusion_gate_mode = 'fixed'
            g_a = model._gate_audio(h_l, h_a)
            g_v = model._gate_video(h_l, h_v)
        torch.testing.assert_close(g_a, torch.full_like(g_a, 0.9))
        torch.testing.assert_close(g_v, torch.full_like(g_v, 0.9))

    def test_fixed_gate_is_constant(self):
        model = self._model('fixed')
        g = model._gate_audio(torch.randn(5, 8), torch.randn(5, 8))
        self.assertEqual(g.shape, (5, 1))
        torch.testing.assert_close(g, torch.full_like(g, 0.9))

    def test_no_parameters_for_fixed_or_none(self):
        for mode in ('none', 'fixed'):
            model = self._model(mode)
            self.assertIsNone(model.gate_a)
            self.assertIsNone(model.gate_a_global)

    def test_global_gate_is_learnable_and_shared(self):
        model = self._model('global')
        g = model._gate_video(torch.randn(6, 8), torch.randn(6, 8))
        self.assertEqual(g.shape, (6, 1))
        self.assertTrue(torch.allclose(g, g[0].expand_as(g)))
        self.assertAlmostEqual(g[0, 0].item(), 0.9, places=4)
        g.sum().backward()
        self.assertIsNotNone(model.gate_v_global.grad)
        self.assertGreater(model.gate_v_global.grad.abs().item(), 0.0)

    def test_sample_gate_shape_initialisation_and_grad(self):
        model = self._model('sample')
        g_a = model._gate_audio(torch.randn(7, 8), torch.randn(7, 8))
        g_v = model._gate_video(torch.randn(7, 8), torch.randn(7, 8))
        self.assertEqual(g_a.shape, (7, 1))
        self.assertEqual(g_v.shape, (7, 1))
        self.assertTrue(torch.allclose(g_a, g_a[0].expand_as(g_a), atol=1e-6))
        self.assertAlmostEqual(g_a[0, 0].item(), 0.9, places=4)
        (g_a.sum() + g_v.sum()).backward()
        self.assertGreater(model.gate_a.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.gate_v.weight.grad.abs().sum().item(), 0.0)

    def test_sample_gate_varies_with_input(self):
        model = self._model('sample')
        with torch.no_grad():
            model.gate_a.weight.normal_(0.0, 0.5)
        h_l = torch.randn(32, 8)
        h_a = torch.randn(32, 8)
        g_a = model._gate_audio(h_l, h_a)
        self.assertGreater(g_a.std().item(), 1e-4)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.module.DLF(_make_args('bogus'))

    def test_forward_returns_gate_and_gradient_flows(self):
        model = self._model('sample')
        model.train()
        text = torch.randn(2, 50, 16)
        audio = torch.randn(2, 50, 9)
        video = torch.randn(2, 50, 9)
        out = model(text, audio, video)
        self.assertEqual(out['fusion_gate_a'].shape, (2, 1))
        self.assertEqual(out['fusion_gate_v'].shape, (2, 1))
        out['output_logit'].sum().backward()
        self.assertGreater(model.gate_a.weight.grad.abs().sum().item(), 0.0)

    def test_forward_none_has_no_gate_output(self):
        model = self._model('none')
        out = model(torch.randn(2, 50, 16), torch.randn(2, 50, 9),
                    torch.randn(2, 50, 9))
        self.assertIsNone(out['fusion_gate_a'])
        self.assertIsNone(out['fusion_gate_v'])


if __name__ == '__main__':
    unittest.main()
