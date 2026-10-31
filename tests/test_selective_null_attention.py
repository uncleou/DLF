"""Unit tests for the optional selective-null attention path."""

import importlib
import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / 'trains/subNets/transformers_encoder/multihead_attention.py'
)
SPEC = importlib.util.spec_from_file_location('dlf_multihead_attention', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MultiheadAttention = MODULE.MultiheadAttention


class SelectiveNullAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.query = torch.randn(3, 2, 8)
        self.key = torch.randn(5, 2, 8)
        self.value = torch.randn(5, 2, 8)

    def test_original_mode_is_unchanged(self):
        attention = MultiheadAttention(8, 2, attn_dropout=0.0)
        output, weights = attention(self.query, self.key, self.value)
        self.assertEqual(output.shape, (3, 2, 8))
        self.assertEqual(weights.shape, (2, 3, 5))
        self.assertIsNone(attention.null_key)
        self.assertIsNone(attention.last_null_attention)

    def test_selective_null_adds_one_choice_and_learns_only_its_key(self):
        attention = MultiheadAttention(
            8, 2, attn_dropout=0.0, add_selective_null=True
        )
        output, weights = attention(self.query, self.key, self.value)
        self.assertEqual(output.shape, (3, 2, 8))
        self.assertEqual(weights.shape, (2, 3, 6))
        self.assertEqual(attention.last_null_attention.shape, (2, 3))
        self.assertNotIn('null_value', dict(attention.named_parameters()))
        output.square().mean().backward()
        self.assertIsNotNone(attention.null_key.grad)
        self.assertGreater(attention.null_key.grad.abs().sum().item(), 0.0)

    def test_selective_null_extends_attention_mask(self):
        attention = MultiheadAttention(
            8, 2, attn_dropout=0.0, add_selective_null=True
        )
        mask = torch.zeros(3, 5)
        mask[0, 4] = float('-inf')
        output, weights = attention(
            self.query, self.key, self.value, attn_mask=mask
        )
        self.assertEqual(weights.shape, (2, 3, 6))
        self.assertTrue(torch.isfinite(output).all())

    def test_null_probability_is_tracked_when_zero_attention_is_also_enabled(self):
        attention = MultiheadAttention(
            8, 2, attn_dropout=0.0, add_selective_null=True,
            add_zero_attn=True
        )
        _, weights = attention(self.query, self.key, self.value)
        self.assertEqual(weights.shape, (2, 3, 7))
        torch.testing.assert_close(attention.last_null_attention, weights[:, :, 5])

    def test_force_null_puts_all_mass_on_null_and_uses_zero_value(self):
        attention = MultiheadAttention(
            8, 2, attn_dropout=0.0, add_selective_null=True
        )
        output, weights = attention(
            self.query, self.key, self.value, force_null=True
        )
        # Null is the last of the 6 key positions.
        torch.testing.assert_close(
            weights[:, :, 5], torch.ones_like(weights[:, :, 5])
        )
        torch.testing.assert_close(weights[:, :, :5], torch.zeros_like(weights[:, :, :5]))
        # Null value is a fixed zero tensor, so the projected output is bias-only.
        self.assertTrue(torch.isfinite(output).all())

    def test_force_null_matches_explicit_null_only_value_aggregation(self):
        attention = MultiheadAttention(
            8, 2, attn_dropout=0.0, add_selective_null=True
        )
        forced, _ = attention(self.query, self.key, self.value, force_null=True)
        # With all weight on the zero value, attn = out_proj(0) = out_proj bias/zero.
        zero_attn = attention.out_proj(torch.zeros_like(forced))
        torch.testing.assert_close(forced, zero_attn, atol=1e-6, rtol=1e-6)

    def test_force_null_requires_selective_null(self):
        attention = MultiheadAttention(8, 2, attn_dropout=0.0)
        with self.assertRaises(ValueError):
            attention(self.query, self.key, self.value, force_null=True)

    def test_null_bias_is_zero_and_learnable_by_default(self):
        attention = MultiheadAttention(
            8, 2, attn_dropout=0.0, add_selective_null=True
        )
        self.assertTrue(attention.null_gate_open)
        self.assertTrue(attention.null_bias.requires_grad)
        torch.testing.assert_close(attention.null_bias, torch.zeros(2))

    def test_closed_gate_zeroes_null_attention_and_freezes_parameters(self):
        attention = MultiheadAttention(
            8, 2, attn_dropout=0.0, add_selective_null=True
        )
        attention.set_null_gate_open(False)
        self.assertFalse(attention.null_gate_open)
        self.assertFalse(attention.null_bias.requires_grad)
        self.assertFalse(attention.null_key.requires_grad)
        _, weights = attention(self.query, self.key, self.value)
        # Null column is the last key position.
        self.assertLess(weights[:, :, 5].max().item(), 1e-6)

    def test_reopening_gate_resets_bias_and_unfreezes(self):
        attention = MultiheadAttention(
            8, 2, attn_dropout=0.0, add_selective_null=True
        )
        attention.set_null_gate_open(False)
        attention.set_null_gate_open(True)
        self.assertTrue(attention.null_gate_open)
        self.assertTrue(attention.null_bias.requires_grad)
        self.assertTrue(attention.null_key.requires_grad)
        torch.testing.assert_close(attention.null_bias, torch.zeros(2))

    def test_encoder_adds_null_only_to_last_layer(self):
        import sys
        import types
        root = Path(__file__).parents[1]
        pkg_name = '_dlf_encoder_tests'
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(root / 'trains/subNets/transformers_encoder')]
            sys.modules[pkg_name] = pkg
        mod = importlib.import_module(pkg_name + '.transformer')
        enc = mod.TransformerEncoder(8, 2, layers=3, selective_null_attn=True)
        has_null = [layer.self_attn.add_selective_null for layer in enc.layers]
        self.assertEqual(has_null, [False, False, True])

    def test_encoder_can_add_null_to_all_layers(self):
        import sys
        import types
        root = Path(__file__).parents[1]
        pkg_name = '_dlf_encoder_tests2'
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(root / 'trains/subNets/transformers_encoder')]
            sys.modules[pkg_name] = pkg
        mod = importlib.import_module(pkg_name + '.transformer')
        enc = mod.TransformerEncoder(
            8, 2, layers=3, selective_null_attn=True,
            selective_null_last_only=False
        )
        has_null = [layer.self_attn.add_selective_null for layer in enc.layers]
        self.assertEqual(has_null, [True, True, True])


if __name__ == '__main__':
    unittest.main()
