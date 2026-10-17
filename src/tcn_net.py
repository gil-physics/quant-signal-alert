"""Step 16/18/19: sequence-network definitions for the TCN/CNN risk-model challengers.

This module imports torch at module level and therefore must ONLY be imported
lazily (inside functions) from tcn_model.py -- the repo's 32-bit / dependency-free
runtimes never import it. Architectures: TCNRiskNet (Step 16, Bai et al. 2018
dilated causal Conv1d residual blocks + last-timestep pooling), CNN1dRiskNet
(Step 18, plain Conv1d stack + mask-safe triple pooling) and MultiScaleCNNRiskNet
(Step 19, parallel-kernel multi-scale stages + the same mask-safe pooling); all
fuse the static features in an identical head.

Weight norm (not batch norm) on the convolutions: batch norm would fold padded
zeros and time-of-day distribution shift into batch statistics; weight norm is
per-parameter and mask-safe.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:                                            # torch >= 2.1
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:                             # older torch
    from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Drop the right-side padding so the convolution stays strictly causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size else x


class TemporalBlock(nn.Module):
    """Two weight-normed causal convolutions + residual (Bai et al. 2018)."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int,
                 dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation, padding=pad)),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            weight_norm(nn.Conv1d(c_out, c_out, kernel_size, dilation=dilation, padding=pad)),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout1d(dropout),
        )
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class CNN1dRiskNet(nn.Module):
    """Step 18 (v5_cnn_60m): plain (non-dilated) Conv1d stack + mask-safe pooling
    + the same static fusion head as TCNRiskNet. Performance-seeking challenger,
    NOT a parameter-matched baseline (channels default 64 vs the TCN's 32).

    Requires the pad mask as the LAST input channel (SequenceConfig.add_mask_channel).
    Padding is triple-guarded so the score is invariant to pad content/count:
      1. the input is mask-zeroed BEFORE the first convolution;
      2. every block's activations are re-masked -- without this a padded column
         turns non-zero through the conv bias / kernel smear and leaks back into
         valid positions at the next layer;
      3. pooling is mask-based: concat of masked global max (padded -> -inf),
         masked global mean (mask-weighted sum / clamped count), and the last VALID
         timestep representation (index found via the mask; with the left-padding
         convention this is usually T-1, but the lookup is mask-driven). All-pad
         rows (padding-never-dropping keeps them) pool to zeros and stay finite.

    A future multi-scale variant (parallel kernel stem -> concat -> 1x1 fuse) can be
    added as a sibling class + registry entry; the arch dispatch is already generic.
    """

    def __init__(self, *, in_channels: int = 17, channels: int = 64, levels: int = 4,
                 kernel_size: int = 5, dropout: float = 0.10,
                 static_dim: int = 13, head_hidden: int = 64):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError(f"kernel_size must be odd for same-length padding "
                             f"(got {kernel_size})")
        pad = (int(kernel_size) - 1) // 2
        blocks = []
        c_prev = in_channels
        for level in range(int(levels)):
            layers = [weight_norm(nn.Conv1d(c_prev, channels, kernel_size, padding=pad)),
                      nn.ReLU()]
            if level < int(levels) - 1:
                layers.append(nn.Dropout1d(dropout))
            blocks.append(nn.Sequential(*layers))
            c_prev = channels
        self.blocks = nn.ModuleList(blocks)
        self.static_dim = int(static_dim)
        self.head = nn.Sequential(
            nn.Linear(3 * channels + self.static_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor = None) -> torch.Tensor:
        """x_seq: (B, C, T) with the pad mask as channel C-1; returns (B,) logits."""
        mask = x_seq[:, -1:, :]                            # (B, 1, T), 1=real bar 0=pad
        h = x_seq * mask                                   # (1) zero pads pre-conv
        for block in self.blocks:
            h = block(h) * mask                            # (2) re-mask every block
        m = mask.squeeze(1)                                # (B, T)
        n_valid = m.sum(dim=1, keepdim=True)               # (B, 1)
        has_valid = n_valid > 0
        h_max = h.masked_fill(m.unsqueeze(1) == 0, float("-inf")).amax(dim=2)
        h_max = torch.where(has_valid, h_max, torch.zeros_like(h_max))
        h_avg = h.sum(dim=2) / n_valid.clamp(min=1.0)      # pads are exact zeros
        last_idx = (h.shape[2] - 1) - m.flip(1).argmax(dim=1)   # all-pad -> zeroed below
        h_last = h.gather(2, last_idx.view(-1, 1, 1).expand(-1, h.shape[1], 1)).squeeze(2)
        h_last = torch.where(has_valid, h_last, torch.zeros_like(h_last))
        pooled = torch.cat([h_max, h_avg, h_last], dim=1)
        if self.static_dim:
            if x_static is None:
                raise ValueError("static_dim > 0 but x_static is None")
            pooled = torch.cat([pooled, x_static], dim=1)
        return self.head(pooled).squeeze(-1)


class MultiScaleBlock(nn.Module):
    """Step 19: parallel odd-kernel convolutions -> concat -> 1x1 fuse.

    Every branch reads the SAME input at a different temporal scale; the 1x1
    convolution learns the per-channel mix of scales. ``residual`` adds the block
    input back after the fuse (requires c_in == c_out) so deeper stages refine
    the multi-scale features instead of replacing them. Pad safety is the
    CALLER's job (MultiScaleCNNRiskNet re-masks after every stage): at padded
    positions the conv bias / kernel smear makes this block's output non-zero,
    exactly like any plain Conv1d.
    """

    def __init__(self, c_in: int, c_out: int, *, kernel_sizes, branch_channels: int,
                 dropout: float, residual: bool):
        super().__init__()
        kernel_sizes = tuple(int(k) for k in kernel_sizes)
        if not kernel_sizes:
            raise ValueError("kernel_sizes must be a non-empty tuple of odd ints")
        for k in kernel_sizes:
            if k % 2 != 1:
                raise ValueError(f"kernel sizes must be odd for same-length padding "
                                 f"(got {kernel_sizes})")
        if residual and c_in != c_out:
            raise ValueError(f"residual stage needs c_in == c_out (got {c_in}/{c_out})")
        self.branches = nn.ModuleList(
            weight_norm(nn.Conv1d(c_in, int(branch_channels), k, padding=(k - 1) // 2))
            for k in kernel_sizes)
        self.fuse = weight_norm(nn.Conv1d(int(branch_channels) * len(kernel_sizes), c_out, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(dropout) if dropout else None
        self.residual = bool(residual)

    def forward(self, x):
        h = torch.cat([self.relu(branch(x)) for branch in self.branches], dim=1)
        h = self.relu(self.fuse(h))
        if self.dropout is not None:
            h = self.dropout(h)
        return x + h if self.residual else h


class MultiScaleCNNRiskNet(nn.Module):
    """Step 19 (v6_mscnn_60m): multi-scale 1D-CNN. Each stage runs parallel
    kernel-size branches over the same input and fuses them 1x1; stages after the
    (dimension-changing) stem are residual. The pad triple-guard, mask-safe
    triple pooling and static fusion head are IDENTICAL to CNN1dRiskNet, so the
    A/B against v5 isolates the multi-scale trunk as the only change.

    Kernel ladder rationale (pre-registered, Step 19): (3, 7, 15) bars on the
    60-bar minute window -- a ~2x geometric ladder separating bar-to-bar
    microstructure, short swings and quarter-window trend. k=31 was rejected
    (about 2x the per-stage cost while 4 stacked stages already give the k=15
    path a 57-bar receptive field, and the global pooling sees all 60 bars
    anyway); dilated branches were rejected because that family IS the TCN --
    v4/v5 probabilities correlate ~0.97, so kernel-scale diversity, not
    dilation, is the untested ingredient.
    """

    def __init__(self, *, in_channels: int = 15, channels: int = 64, levels: int = 4,
                 kernel_sizes=(3, 7, 15), branch_channels: int = 32,
                 dropout: float = 0.10, static_dim: int = 13, head_hidden: int = 64):
        super().__init__()
        blocks = []
        for level in range(int(levels)):
            blocks.append(MultiScaleBlock(
                in_channels if level == 0 else channels, channels,
                kernel_sizes=kernel_sizes, branch_channels=branch_channels,
                dropout=dropout if level < int(levels) - 1 else 0.0,
                residual=level > 0))
        self.blocks = nn.ModuleList(blocks)
        self.static_dim = int(static_dim)
        self.head = nn.Sequential(
            nn.Linear(3 * channels + self.static_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor = None) -> torch.Tensor:
        """x_seq: (B, C, T) with the pad mask as channel C-1; returns (B,) logits."""
        mask = x_seq[:, -1:, :]                            # (B, 1, T), 1=real bar 0=pad
        h = x_seq * mask                                   # (1) zero pads pre-conv
        for block in self.blocks:
            h = block(h) * mask                            # (2) re-mask every stage
        m = mask.squeeze(1)                                # (B, T) -- pooling as in v5
        n_valid = m.sum(dim=1, keepdim=True)
        has_valid = n_valid > 0
        h_max = h.masked_fill(m.unsqueeze(1) == 0, float("-inf")).amax(dim=2)
        h_max = torch.where(has_valid, h_max, torch.zeros_like(h_max))
        h_avg = h.sum(dim=2) / n_valid.clamp(min=1.0)      # pads are exact zeros
        last_idx = (h.shape[2] - 1) - m.flip(1).argmax(dim=1)   # all-pad -> zeroed below
        h_last = h.gather(2, last_idx.view(-1, 1, 1).expand(-1, h.shape[1], 1)).squeeze(2)
        h_last = torch.where(has_valid, h_last, torch.zeros_like(h_last))
        pooled = torch.cat([h_max, h_avg, h_last], dim=1)
        if self.static_dim:
            if x_static is None:
                raise ValueError("static_dim > 0 but x_static is None")
            pooled = torch.cat([pooled, x_static], dim=1)
        return self.head(pooled).squeeze(-1)


class TCNRiskNet(nn.Module):
    """Causal TCN over (B, C, T) bar windows + static-feature fusion head.

    Receptive field = 1 + 2*(kernel_size-1)*(2**levels - 1); with the defaults
    (k=3, levels=4) that is 61 >= seq_len 60, so the last timestep (the decision
    bar, never padding) sees the whole window. ``static_dim=0`` disables the
    static head (ablation A: sequence-only).
    """

    def __init__(self, *, in_channels: int = 15, channels: int = 32, levels: int = 4,
                 kernel_size: int = 3, dropout: float = 0.10,
                 static_dim: int = 13, head_hidden: int = 64):
        super().__init__()
        blocks = []
        c_prev = in_channels
        for level in range(int(levels)):
            blocks.append(TemporalBlock(c_prev, channels, kernel_size,
                                        dilation=2 ** level, dropout=dropout))
            c_prev = channels
        self.tcn = nn.Sequential(*blocks)
        self.static_dim = int(static_dim)
        self.head = nn.Sequential(
            nn.Linear(channels + self.static_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    @staticmethod
    def receptive_field(*, levels: int, kernel_size: int) -> int:
        return 1 + 2 * (kernel_size - 1) * (2 ** int(levels) - 1)

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor = None) -> torch.Tensor:
        """x_seq: (B, C, T); x_static: (B, static_dim) or None. Returns (B,) logits."""
        h = self.tcn(x_seq)[:, :, -1]                       # last timestep = decision bar
        if self.static_dim:
            if x_static is None:
                raise ValueError("static_dim > 0 but x_static is None")
            h = torch.cat([h, x_static], dim=1)
        return self.head(h).squeeze(-1)
