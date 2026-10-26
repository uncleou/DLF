"""Shared-space alignment losses on DLF's existing [B, d] projections."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .HingeLoss import HingeLoss


def atp_all_loss(z_t, z_a, z_v):
    """Mean of three squared L2 distances; inputs are unit-normalized vectors.

    Sum over features, then average over samples and modality pairs (not MSE
    averaged over features). All three inputs retain their gradients.
    """
    return (
        (z_t - z_a).square().sum(dim=-1)
        + (z_t - z_v).square().sum(dim=-1)
        + (z_a - z_v).square().sum(dim=-1)
    ).mean() / 3.0


def sentiment_centroid_kl(z_t, z_a, z_v, labels, sigma=1.0, tau=1.0):
    """Batch-mean KL(q || p), excluding self-neighbors in both distributions.

    q uses absolute differences of continuous sentiment labels; p uses squared
    distances between the mean of three unit vectors. Centroids are NOT
    normalized again. Only labels are detached, never the representations.
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sent_sigma must be finite and positive")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("sent_tau must be finite and positive")
    centroid = (z_t + z_a + z_v) / 3.0
    batch_size = centroid.size(0)
    if labels.numel() != batch_size:
        raise ValueError("Expected one continuous sentiment label per sample")
    # B=1 has no neighbors; B=2 has one neighbor and KL is identically zero.
    if batch_size < 3:
        return centroid.sum() * 0.0

    y = labels.detach().to(device=centroid.device, dtype=centroid.dtype).reshape(-1)
    label_distance = (y[:, None] - y[None, :]).abs()
    feature_distance = (
        centroid[:, None, :] - centroid[None, :, :]
    ).square().sum(dim=-1)
    neighbors = ~torch.eye(batch_size, dtype=torch.bool, device=centroid.device)
    # Remove the diagonal before log_softmax: no masked 0 * (-inf) in KL.
    target_logits = (-label_distance / sigma)[neighbors].reshape(batch_size, -1)
    model_logits = (-feature_distance / tau)[neighbors].reshape(batch_size, -1)
    log_q = F.log_softmax(target_logits, dim=-1)
    log_p = F.log_softmax(model_logits, dim=-1)
    return (log_q.exp() * (log_q - log_p)).sum(dim=-1).mean()


class SharedAlignmentLoss(nn.Module):
    """Select the original objective or one of five shared-loss ablations.

    Weights here are final coefficients in the total training objective.
    Legacy triplet mode uses unnormalized projections and its original 0.01
    coefficient. No parameters are added to the model or its checkpoints.
    """

    MODES = ("original", "none", "atp", "sent", "atp_sent", "original_atp_sent")

    def __init__(self, mode="original", atp_weight=0.01, sent_weight=0.01,
                 sent_sigma=1.0, sent_tau=1.0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"shared_loss_mode must be one of {self.MODES}, got {mode!r}")
        for name, value in (("atp_weight", atp_weight), ("sent_weight", sent_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (("sent_sigma", sent_sigma), ("sent_tau", sent_tau)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.mode = mode
        self.atp_weight = atp_weight
        self.sent_weight = sent_weight
        self.sent_sigma = sent_sigma
        self.sent_tau = sent_tau
        self.triplet = HingeLoss()

    def forward(self, h_t, h_a, h_v, labels):
        if h_t.ndim != 2 or h_t.shape != h_a.shape or h_t.shape != h_v.shape:
            raise ValueError("Shared projections must have identical [B, d] shapes")
        if h_t.size(0) == 0 or h_t.size(1) == 0 or labels.numel() != h_t.size(0):
            raise ValueError("Expected nonempty projections and one label per sample")
        zero = (h_t.sum() + h_a.sum() + h_v.sum()) * 0.0
        loss_atp, loss_sent, loss_triplet = zero, zero, zero
        use_atp = self.mode in ("atp", "atp_sent", "original_atp_sent")
        use_sent = self.mode in ("sent", "atp_sent", "original_atp_sent")

        if use_atp or use_sent:
            # Separate copies for losses; leave the model's features untouched.
            z_t, z_a, z_v = (
                F.normalize(h, p=2, dim=-1, eps=1e-8) for h in (h_t, h_a, h_v)
            )
            if use_atp:
                loss_atp = atp_all_loss(z_t, z_a, z_v)
            if use_sent:
                loss_sent = sentiment_centroid_kl(
                    z_t, z_a, z_v, labels, self.sent_sigma, self.sent_tau
                )

        if self.mode in ("original", "original_atp_sent"):
            # Preserve original per-sample text/vision/audio order and raw inputs.
            feats = torch.stack((h_t, h_v, h_a), dim=1).reshape(-1, h_t.size(1))
            ids = labels.detach().reshape(-1, 1).repeat_interleave(3, dim=0)
            loss_triplet = self.triplet(ids, feats)
            # Original HingeLoss returns a Python zero if no valid triplet exists.
            if not torch.is_tensor(loss_triplet):
                loss_triplet = zero + loss_triplet

        return {
            "total": (self.atp_weight * loss_atp + self.sent_weight * loss_sent
                      + 0.01 * loss_triplet),
            "atp": loss_atp,
            "sent": loss_sent,
            "triplet": loss_triplet,
        }
