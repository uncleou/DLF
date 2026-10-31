"""Unit tests for the optional selective-null attention path."""

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


if __name__ == '__main__':
    unittest.main()
