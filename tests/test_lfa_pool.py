"""Unit tests for LFA sequence pooling (CPU only).

Mirrors test_fusion_gate.py: loads DLF.py with stubbed BERT/Transformer imports so
the model runs on CPU. Covers the 'last' == original, masked 'mean', and masked
'attn' summarize over the private LFA paths.
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

    path = ROOT / 'trains' / 'singleTask' / 'model' / 'DLF.py'
    spec = importlib.util.spec_from_file_location(
        'trains.singleTask.model.DLF', path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_args(pool_mode='last', pool_paths=()):
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
        lfa_pool_mode=pool_mode,
        lfa_pool_paths=list(pool_paths),
    )


class LfaPoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_model_module()

    def _model(self, pool_mode='last', pool_paths=()):
        torch.manual_seed(0)
        return self.module.DLF(_make_args(pool_mode, pool_paths)).eval()

    # -- mask construction -------------------------------------------------
    def test_mask_aligns_to_conv_output_length(self):
        model = self._model('mean', ['text'])
        text = torch.zeros(2, 50, 16)
        text[0, :30] = 1.0     # 30 valid tokens -> conv length 46
        text[1, :46] = 1.0
        mask = model._build_lfa_mask(text)
        self.assertEqual(tuple(mask.shape), (2, 46))
        # token 29 is the last real token for sample 0; stride-1 conv of k=5
        # makes output position t valid iff tokens t..t+4 contain a real token.
        self.assertTrue(mask[0, 25].item())   # 25..29 all real
        self.assertTrue(mask[0, 29].item())   # 29..33 has real token 29
        self.assertFalse(mask[0, 30].item())  # 30..34 all padding
        self.assertTrue(mask[1, 45].item())

    # -- 'last' reproduces the original read -------------------------------
    def test_last_mode_equals_last_position(self):
        model = self._model('last', [])
        h = torch.randn(46, 4, 8)
        mask = torch.ones(4, 46, dtype=torch.bool)
        mask[0, 20:] = False
        out = model._lfa_pool(h, 'text', mask)
        torch.testing.assert_close(out, h[-1])

    def test_unselected_path_keeps_last_position(self):
        model = self._model('mean', ['text'])
        h = torch.randn(46, 4, 8)
        mask = torch.ones(4, 46, dtype=torch.bool)
        # 'video' is not selected -> should still read h[-1], not the mean.
        out = model._lfa_pool(h, 'video', mask)
        torch.testing.assert_close(out, h[-1])

    # -- 'mean' uses the mask ---------------------------------------------
    def test_mean_pool_excludes_padding(self):
        model = self._model('mean', ['text'])
        h = torch.randn(46, 4, 8)
        mask = torch.ones(4, 46, dtype=torch.bool)
        length0 = 10
        mask[0, length0:] = False
        # Poison padding rows; a masked mean must ignore them.
        h[length0:, 0, :] = 1e6
        out = model._lfa_pool(h, 'text', mask)
        expected = h[:length0, 0, :].mean(dim=0)
        torch.testing.assert_close(out[0], expected)

    def test_mean_falls_back_when_no_valid_positions(self):
        model = self._model('mean', ['text'])
        h = torch.randn(46, 3, 8)
        mask = torch.ones(3, 46, dtype=torch.bool)
        mask[2] = False
        out = model._lfa_pool(h, 'text', mask)
        torch.testing.assert_close(out[2], h[-1, 2])
        self.assertFalse(torch.isnan(out).any())

    # -- 'attn' learns weights and respects mask --------------------------
    def test_attn_pool_shape_and_mask(self):
        model = self._model('attn', ['text', 'audio', 'video'])
        h = torch.randn(46, 4, 8)
        mask = torch.ones(4, 46, dtype=torch.bool)
        mask[:, 20:] = False
        h[20:, :, :] = 1e6  # padding must not leak in
        out = model._lfa_pool(h, 'text', mask)
        self.assertEqual(tuple(out.shape), (4, 8))
        self.assertFalse(torch.isnan(out).any())
        self.assertLess(out.abs().max().item(), 1e5)

    def test_attn_pool_has_gradients(self):
        model = self._model('attn', ['text'])
        h = torch.randn(46, 4, 8, requires_grad=True)
        mask = torch.ones(4, 46, dtype=torch.bool)
        out = model._lfa_pool(h, 'text', mask)
        out.sum().backward()
        self.assertGreater(model.lfa_attn['text'][0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(h.grad.abs().sum().item(), 0.0)

    def test_attn_falls_back_when_no_valid_positions(self):
        model = self._model('attn', ['text'])
        h = torch.randn(46, 2, 8)
        mask = torch.ones(2, 46, dtype=torch.bool)
        mask[1] = False
        out = model._lfa_pool(h, 'text', mask)
        torch.testing.assert_close(out[1], h[-1, 1])

    def test_last_and_mean_add_no_parameters(self):
        for mode in ('last', 'mean'):
            model = self._model(mode, ['text', 'audio', 'video'])
            self.assertEqual(len(model.lfa_attn), 0)

    def test_attn_only_builds_selected_paths(self):
        model = self._model('attn', ['text'])
        self.assertIn('text', model.lfa_attn)
        self.assertNotIn('audio', model.lfa_attn)
        self.assertNotIn('video', model.lfa_attn)

    # -- full forward smoke test ------------------------------------------
    def test_forward_smoke_mean(self):
        model = self._model('mean', ['text', 'audio', 'video'])
        model.train()
        text = torch.randn(2, 50, 16)
        audio = torch.randn(2, 50, 9)
        video = torch.randn(2, 50, 9)
        out = model(text, audio, video)
        self.assertEqual(tuple(out['output_logit'].shape), (2, 1))
        out['output_logit'].sum().backward()

    def test_forward_smoke_attn(self):
        model = self._model('attn', ['text', 'audio', 'video'])
        text = torch.randn(2, 50, 16)
        audio = torch.randn(2, 50, 9)
        video = torch.randn(2, 50, 9)
        out = model(text, audio, video)
        self.assertEqual(tuple(out['output_logit'].shape), (2, 1))

    def test_invalid_mode_and_path_rejected(self):
        with self.assertRaises(ValueError):
            self.module.DLF(_make_args('bogus'))
        with self.assertRaises(ValueError):
            self.module.DLF(_make_args('mean', ['nope']))


if __name__ == '__main__':
    unittest.main()
