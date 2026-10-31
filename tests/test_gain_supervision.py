"""Unit tests for gain supervision helpers (CPU only)."""

import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'trains/singleTask/gain_supervision.py'
SPEC = importlib.util.spec_from_file_location('dlf_gain_supervision', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

GainSupervision = MODULE.GainSupervision
EMATeacher = MODULE.EMATeacher
gain_soft_target = MODULE.gain_soft_target
confidence_weight = MODULE.confidence_weight
gain_weight_for_epoch = MODULE.gain_weight_for_epoch
per_sample_smooth_l1 = MODULE.per_sample_smooth_l1


class GainSupervisionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_positive_gain_means_read_modality(self):
        # Large positive gain -> read prob near 1 -> null target near 0.
        gain = torch.tensor([1.0])
        read = gain_soft_target(gain, margin=0.01, temperature=0.1)
        self.assertGreater(read.item(), 0.99)
        self.assertLess((1.0 - read).item(), 0.01)

    def test_negative_gain_means_increase_null_reading(self):
        gain = torch.tensor([-1.0])
        read = gain_soft_target(gain, margin=0.01, temperature=0.1)
        self.assertLess(read.item(), 0.01)

    def test_confidence_weight_saturates_and_scales(self):
        gain = torch.tensor([0.0, 0.05, 0.2, -0.3])
        weight = confidence_weight(gain, scale=0.1)
        expected = torch.tensor([0.0, 0.5, 1.0, 1.0])
        torch.testing.assert_close(weight, expected)

    def test_per_sample_smooth_l1_shape(self):
        pred = torch.zeros(4, 1)
        target = torch.ones(4, 1) * 0.5
        err = per_sample_smooth_l1(pred, target)
        self.assertEqual(err.shape, (4, 1))

    def test_loss_backpropagates_to_null_probs_only(self):
        null_a = torch.tensor([0.5, 0.2], requires_grad=True)
        null_v = torch.tensor([0.1, 0.9], requires_grad=True)
        gain_a = torch.tensor([1.0, -1.0])
        gain_v = torch.tensor([0.8, -0.8])
        loss = GainSupervision()(null_a, null_v, gain_a, gain_v)
        self.assertTrue(torch.isfinite(loss['total']))
        loss['total'].backward()
        self.assertIsNotNone(null_a.grad)
        self.assertGreater(null_a.grad.abs().sum().item(), 0.0)
        self.assertGreater(null_v.grad.abs().sum().item(), 0.0)

    def test_zero_gain_samples_are_downweighted(self):
        # Sample 0 has zero gain (weight 0), sample 1 has large gain (weight 1).
        null_prob = torch.tensor([0.9, 0.9], requires_grad=True)
        gain = torch.tensor([0.0, 1.0])
        loss = GainSupervision(confidence_scale=0.1)(
            null_prob, null_prob, gain, gain
        )
        loss['total'].backward()
        # Only the second sample should contribute gradient.
        self.assertEqual(null_prob.grad[0].item(), 0.0)
        self.assertNotEqual(null_prob.grad[1].item(), 0.0)

    def test_invalid_temperature_rejected(self):
        with self.assertRaises(ValueError):
            GainSupervision(temperature=0.0)
        with self.assertRaises(ValueError):
            GainSupervision(confidence_scale=-1.0)


class EMATeacherTest(unittest.TestCase):
    def test_ema_tracks_student_without_gradients(self):
        student = nn.Linear(3, 2)
        teacher = EMATeacher(student, decay=0.5)
        before = [p.detach().clone() for p in student.parameters()]
        for p in teacher.module.parameters():
            self.assertFalse(p.requires_grad)
        with torch.no_grad():
            for p in student.parameters():
                p.add_(1.0)
        teacher.update(student)
        # ema = 0.5 * old_ema(=old student) + 0.5 * (old student + 1)
        for ema_param, old in zip(teacher.module.parameters(), before):
            torch.testing.assert_close(ema_param, old + 0.5)

    def test_invalid_decay_rejected(self):
        with self.assertRaises(ValueError):
            EMATeacher(nn.Linear(2, 2), decay=1.0)


class GainScheduleTest(unittest.TestCase):
    def test_linear_ramp(self):
        # warmup 5, ramp 5, max 0.05.
        self.assertEqual(gain_weight_for_epoch(5, 5, 5, 0.05), 0.0)
        self.assertAlmostEqual(gain_weight_for_epoch(7, 5, 5, 0.05), 0.02)
        self.assertAlmostEqual(gain_weight_for_epoch(10, 5, 5, 0.05), 0.05)
        self.assertAlmostEqual(gain_weight_for_epoch(20, 5, 5, 0.05), 0.05)


if __name__ == '__main__':
    unittest.main()
