"""Per-sample modality-gain supervision for the selective-null gate.

Counterfactual question: if a cross-modal path is forced to read Null, does the
prediction error grow? A positive gain means the modality helps, so the gate
should read it (lower null probability). Targets are soft and weighted so that
samples with near-zero gain do not force the gate.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def per_sample_smooth_l1(pred, target, beta=1.0):
    """Per-sample Smooth L1 in the shape [B, 1]; no reduction."""
    return F.smooth_l1_loss(pred, target, beta=beta, reduction='none')


def gain_soft_target(gain, margin, temperature):
    """Probability that the modality should be read: sigmoid((gain - margin) / tau)."""
    return torch.sigmoid((gain - margin) / temperature)


def confidence_weight(gain, scale):
    """min(|gain| / scale, 1) so uncertain samples contribute little."""
    return torch.clamp(gain.detach().abs() / scale, max=1.0)


def null_probability(null_attn):
    """Average null-read probability over text steps -> [B]."""
    return null_attn.mean(dim=-1)


class GainSupervision(nn.Module):
    """Weighted BCE between the gate's null probability and gain-derived targets."""

    def __init__(self, margin=0.01, temperature=0.1, confidence_scale=0.1, beta=1.0):
        super().__init__()
        for name, value in (("margin", margin), ("temperature", temperature),
                            ("confidence_scale", confidence_scale)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if confidence_scale <= 0:
            raise ValueError("confidence_scale must be positive")
        self.margin = margin
        self.temperature = temperature
        self.confidence_scale = confidence_scale
        self.beta = beta

    def forward(self, null_prob_a, null_prob_v, gain_a, gain_v):
        """Return {'total', 'a', 'v', 'target_null_a', 'target_null_v'}.

        ``null_prob_a``/``null_prob_v`` carry gradients (student gate). The gains
        are treated as constants: targets and weights are detached.
        """
        with torch.no_grad():
            read_a = gain_soft_target(gain_a, self.margin, self.temperature)
            read_v = gain_soft_target(gain_v, self.margin, self.temperature)
            target_null_a = 1.0 - read_a
            target_null_v = 1.0 - read_v
            weight_a = confidence_weight(gain_a, self.confidence_scale)
            weight_v = confidence_weight(gain_v, self.confidence_scale)

        loss_a = F.binary_cross_entropy(null_prob_a, target_null_a, reduction='none')
        loss_v = F.binary_cross_entropy(null_prob_v, target_null_v, reduction='none')
        term_a = (weight_a * loss_a).mean()
        term_v = (weight_v * loss_v).mean()
        return {
            'total': term_a + term_v,
            'a': term_a,
            'v': term_v,
            'target_null_a': target_null_a,
            'target_null_v': target_null_v,
        }


class EMATeacher:
    """Exponential moving average copy of a student model (no gradients)."""

    def __init__(self, student, decay=0.995):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.module = copy.deepcopy(student)
        self.module.eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, student):
        decay = self.decay
        for ema_param, param in zip(self.module.parameters(), student.parameters()):
            ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        # Buffers (e.g. BatchNorm statistics) are copied outright.
        for ema_buf, buf in zip(self.module.buffers(), student.buffers()):
            ema_buf.copy_(buf)

    def to(self, device):
        self.module.to(device)
        return self

    def train(self, mode=True):
        # The teacher is only ever used in eval mode for the counterfactual.
        self.module.eval()
        return self


def gain_weight_for_epoch(epoch, warmup, ramp_epochs, max_weight):
    """Linear ramp: 0 during warmup, then 0 -> max_weight over ramp_epochs."""
    if ramp_epochs <= 0:
        return max_weight if epoch > warmup else 0.0
    progress = (epoch - warmup) / ramp_epochs
    return max_weight * min(1.0, max(0.0, progress))
