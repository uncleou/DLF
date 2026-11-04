"""Distribution prediction head and HL-Gauss targets for the final DLF head.

The final head predicts a categorical distribution over ``K`` equal-width score
bins and returns the expectation ``sum_k p_k c_k`` as the continuous sentiment
score. Supervision can be plain L1 on the expectation, a hard-label cross
entropy, a Gaussian soft-label (HL-Gauss) cross entropy, or a combination.

Bins are equal width over ``[bin_min, bin_max]``; the two outer bins extend to
+/-inf so every Gaussian target sums to one without renormalization. That leaves
``sum_k q_k c_k ~= y`` (not exactly ``y``) which is why the optional L1 term is
kept in the full variant.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


HEAD_MODES = ("original", "dist_l1", "dist_hard", "dist_gauss", "dist_gauss_l1")
# Modes whose training objective keeps the main-head L1 on the expectation.
_L1_MODES = {"original", "dist_l1", "dist_hard", "dist_gauss_l1"}
# Modes that add a cross-entropy term on the bin logits.
_CE_MODES = {"dist_hard", "dist_gauss", "dist_gauss_l1"}


def make_centers(num_bins, bin_min, bin_max, dtype=torch.float32):
    """Equal-width bin centers c_k for k=0..K-1."""
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    if not bin_max > bin_min:
        raise ValueError("bin_max must be greater than bin_min")
    step = (bin_max - bin_min) / num_bins
    k = torch.arange(num_bins, dtype=dtype)
    return bin_min + (k + 0.5) * step


def bin_edges(centers):
    """K+1 finite bin edges derived from sorted centers."""
    if centers.dim() != 1 or centers.numel() < 2:
        raise ValueError("centers must be a 1-D tensor with >= 2 entries")
    mid = (centers[:-1] + centers[1:]) / 2
    lo = centers[0] - (centers[1] - centers[0]) / 2
    hi = centers[-1] + (centers[-1] - centers[-2]) / 2
    return torch.cat([lo.reshape(1), mid, hi.reshape(1)])


def gaussian_soft_target(labels, centers, sigma):
    """HL-Gauss soft target q(y): Gaussian CDF mass per bin.

    ``labels`` is [B, 1] (or [B]); returns [B, K]. The outer edges are treated as
    +/-inf so rows sum to 1.
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    y = labels.reshape(-1, 1).to(device=centers.device, dtype=centers.dtype)
    edges = bin_edges(centers)
    # q_k = Phi((e_{k+1} - y)/sigma) - Phi((e_k - y)/sigma) for k=0..K-1,
    # with e_0=-inf and e_K=+inf so each row sums to exactly one (including
    # labels outside the supported range, which pile up in the edge bins).
    edges = torch.cat([
        edges.new_full((1,), float("-inf")),
        edges[1:-1],
        edges.new_full((1,), float("inf")),
    ])
    upper = edges[1:].reshape(1, -1)
    lower = edges[:-1].reshape(1, -1)
    q = (torch.special.ndtr((upper - y) / sigma)
         - torch.special.ndtr((lower - y) / sigma))
    return q.clamp_min(0.0)


def hard_target(labels, centers):
    """One-hot target at the bin whose center is nearest to each label: [B, K]."""
    y = labels.reshape(-1, 1).to(device=centers.device, dtype=centers.dtype)
    idx = (y - centers.reshape(1, -1)).abs().argmin(dim=-1)
    target = torch.zeros(idx.numel(), centers.numel(),
                         device=centers.device, dtype=centers.dtype)
    target.scatter_(1, idx.reshape(-1, 1), 1.0)
    return target


class DistributionHead(nn.Module):
    """Linear bin-logit head producing (expected score, probabilities, logits)."""

    def __init__(self, in_dim, num_bins=66, bin_min=-3.3, bin_max=3.3):
        super().__init__()
        self.num_bins = num_bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.proj = nn.Linear(in_dim, num_bins)
        # Persistent so a bare state_dict carries the bin geometry.
        self.register_buffer(
            "centers", make_centers(num_bins, bin_min, bin_max), persistent=True
        )

    def forward(self, z):
        logits = self.proj(z)
        probs = torch.softmax(logits, dim=-1)
        y_hat = (probs * self.centers.reshape(1, -1)).sum(dim=-1, keepdim=True)
        return y_hat, probs, logits


class DistributionCriterion(nn.Module):
    """Main-head loss for one head_mode.

    Returns a dict with 'total', 'l1', 'ce'. 'l1' is always computed for
    diagnostics but only enters 'total' for modes that keep it. 'ce' is None for
    modes without a cross-entropy term.
    """

    def __init__(self, mode, num_bins=66, bin_min=-3.3, bin_max=3.3,
                 sigma=0.075, ce_weight=0.1):
        super().__init__()
        if mode not in HEAD_MODES or mode == "original":
            raise ValueError(
                f"DistributionCriterion does not handle mode={mode!r}; expected "
                f"one of {[m for m in HEAD_MODES if m != 'original']}."
            )
        if not math.isfinite(ce_weight) or ce_weight < 0:
            raise ValueError("ce_weight must be finite and nonnegative")
        self.mode = mode
        self.sigma = sigma
        self.ce_weight = ce_weight
        self.use_l1 = mode in _L1_MODES
        self.use_ce = mode in _CE_MODES
        self.target_kind = None
        if mode == "dist_hard":
            self.target_kind = "hard"
        elif mode in ("dist_gauss", "dist_gauss_l1"):
            self.target_kind = "gauss"
        self.register_buffer(
            "centers", make_centers(num_bins, bin_min, bin_max), persistent=False
        )

    def forward(self, y_hat, logits, labels):
        labels = labels.reshape(-1, 1).to(device=y_hat.device, dtype=y_hat.dtype)
        # The criterion is not part of the model, so its buffer may stay on CPU
        # while logits live on GPU; move it to match.
        centers = self.centers.to(device=logits.device, dtype=logits.dtype)
        l1 = F.l1_loss(y_hat, labels)
        ce = None
        if self.use_ce:
            if self.target_kind == "hard":
                idx = hard_target(labels, centers).argmax(dim=-1)
                ce = F.cross_entropy(logits, idx)
            else:
                q = gaussian_soft_target(labels, centers, self.sigma)
                ce = -(q * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        total = l1.new_zeros(())
        if self.use_l1:
            total = total + l1
        if self.use_ce:
            total = total + self.ce_weight * ce
        return {"total": total, "l1": l1, "ce": ce}
