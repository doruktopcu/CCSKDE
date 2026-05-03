"""Context-conditioned variant of SeeKer's MADEPartial.

We append a per-segment context tensor C ∈ R^{D_ctx} to the input and assign
those entries MADE degree 0 ("unconditional shortcut"). Properties:
  * Output dim is unchanged (2*D), so SeeKer's training reshape still works.
  * Every hidden unit and every keypoint output may read every C entry.
  * The keypoint-to-keypoint autoregressive constraint is preserved exactly
    (mask construction over the keypoint columns is bit-for-bit identical to
    the upstream MADEPartial — only D_ctx all-ones columns are appended).

The implementation deliberately mirrors `seeker/models/made/made_partial.py`
to keep the diff against the baseline auditable.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    """Linear layer with a frozen 0/1 mask multiplied into the weight matrix."""

    def __init__(self, n_in: int, n_out: int, bias: bool = True) -> None:
        super().__init__(n_in, n_out, bias)
        self.register_buffer("mask", None)

    def initialise_mask(self, mask: Tensor) -> None:
        self.mask = mask

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.mask * self.weight, self.bias)


class MADEPartialContext(nn.Module):
    """SeeKer's MADEPartial with a D_ctx-dim shortcut block appended.

    Input  shape: (B, n_kp + n_ctx)
    Output shape: (B, 2 * n_kp)   — mu and logvar over keypoints only.
    """

    def __init__(
        self,
        n_kp: int,
        n_ctx: int,
        hidden_dims: List[int],
        droppout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_kp = n_kp
        self.n_ctx = n_ctx
        self.n_in = n_kp + n_ctx
        self.n_out = 2 * n_kp
        self.hidden_dims = hidden_dims
        self.gaussian = True

        self._masks: dict[int, np.ndarray] = {}
        self._mask_matrix: list[Tensor] = []
        layers: list[nn.Module] = []

        dim_list = [self.n_in, *hidden_dims, self.n_out]
        for i in range(len(dim_list) - 2):
            layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=droppout))
        layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))

        self.model = nn.Sequential(*layers)
        self._create_masks()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    # ------------------------------------------------------------------ masks
    def _create_masks(self) -> None:
        """Mirrors MADEPartial._create_masks() and appends D_ctx shortcut cols."""
        L = len(self.hidden_dims)
        D = self.n_kp                         # keypoint input dim
        D_context_past = D - 18 * 2           # past-frame portion (kp-only)

        # Degree assignment ON KEYPOINT INPUTS ONLY (identical to baseline).
        self._masks[0] = np.repeat(np.arange(0, D, 2), 2)
        for l in range(L):
            self._masks[l + 1] = np.concatenate(
                ((self.hidden_dims[l]) // D + 1)
                * [np.repeat(np.arange(0, D - 2, 2), 2)]
            )
        self._masks[L + 1] = self._masks[0]   # output keypoint order = input order

        # Hidden masks (use >=); past-frame keypoint degrees forced to 0.
        for i in range(len(self._masks) - 2):
            m = self._masks[i].copy()
            m[m < D_context_past] = 0
            m_next = self._masks[i + 1]
            M_kp = torch.zeros(len(m_next), len(m))
            for j in range(len(m_next)):
                M_kp[j, :] = torch.from_numpy((m_next[j] >= m).astype(int))
            self._mask_matrix.append(M_kp)

        # Output mask (use >); past-frame degrees → -1 so they are reachable.
        i = len(self._masks) - 2
        m = self._masks[i].copy()
        m[m < D_context_past] = -1
        m_next = self._masks[i + 1]
        M_out = torch.zeros(len(m_next), len(m))
        for j in range(len(m_next)):
            M_out[j, :] = torch.from_numpy((m_next[j] > m).astype(int))
        self._mask_matrix.append(M_out)

        # Duplicate the output mask for (mu, logvar).
        last = self._mask_matrix.pop(-1)
        self._mask_matrix.append(torch.cat((last, last), dim=0))

        # Append D_ctx shortcut columns to the *input* layer's mask only.
        # All ones: every hidden unit may read every C entry.
        if self.n_ctx > 0:
            ctx_cols = torch.ones(
                self._mask_matrix[0].shape[0], self.n_ctx, dtype=self._mask_matrix[0].dtype
            )
            self._mask_matrix[0] = torch.cat((self._mask_matrix[0], ctx_cols), dim=1)

        # Bind masks to MaskedLinear modules in order.
        mask_iter = iter(self._mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))


class PartialAutoregressiveContextFC(nn.Module):
    """Drop-in for `PartialAutoregressiveFC` that consumes (x_pose, c_ctx).

    x_pose: (B, ...) — flattened to (B, n_kp).
    c_ctx:  (B, ...) — flattened to (B, n_ctx). Pass None to behave as baseline
            (n_ctx must have been initialised to 0).
    """

    def __init__(
        self,
        dim: int,
        ctx_dim: int,
        hidden_dims: List[int],
        droppout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ctx_dim = ctx_dim
        self.hidden_dims = hidden_dims
        self.model = MADEPartialContext(dim, ctx_dim, hidden_dims, droppout)

    def forward(self, x_pose: Tensor, c_ctx: Tensor | None = None) -> Tensor:
        x = x_pose.flatten(start_dim=1)
        if self.ctx_dim > 0:
            if c_ctx is None:
                raise ValueError("ctx_dim>0 but c_ctx is None")
            c = c_ctx.flatten(start_dim=1)
            x = torch.cat((x, c), dim=1)
        return self.model(x)
