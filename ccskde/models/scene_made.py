"""Generalised MADE for the unified scene skeleton.

SeeKer's MADEPartial hardcodes 18 keypoints per frame. The unified scene model
treats each road agent's keypoints as part of one skeleton of
``N' = 18 + 6*M`` keypoints (pedestrian + M vehicle 6-keypoint skeletons), so we
need the identical autoregressive mask construction with the current-frame
keypoint count parameterised. Everything else (degree scheme, past-frame
handling, mu/logvar doubling) mirrors ``seeker/models/made/made_partial.py``
bit-for-bit, so the diff against the baseline is auditable.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, n_in: int, n_out: int, bias: bool = True) -> None:
        super().__init__(n_in, n_out, bias)
        self.register_buffer("mask", None)

    def initialise_mask(self, mask: Tensor) -> None:
        self.mask = mask

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.mask * self.weight, self.bias)


class SceneMADEPartial(nn.Module):
    """MADEPartial with a configurable current-frame keypoint count.

    Args:
        n_in: total input coords = 2 * n_keypoints * T.
        n_current_kp: number of keypoints in the *current* frame (N').
        hidden_dims, droppout: as in MADEPartial.
    """

    def __init__(self, n_in: int, n_current_kp: int, hidden_dims: List[int],
                 droppout: float = 0.0) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = int(2 * n_in)
        self.n_current_kp = n_current_kp
        self.hidden_dims = hidden_dims
        self.gaussian = True
        self.masks: dict[int, np.ndarray] = {}
        self.mask_matrix: list[Tensor] = []

        dim_list = [self.n_in, *hidden_dims, self.n_out]
        layers: list[nn.Module] = []
        for i in range(len(dim_list) - 2):
            layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=droppout))
        layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))
        self.model = nn.Sequential(*layers)
        self._create_masks()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _create_masks(self) -> None:
        L = len(self.hidden_dims)
        D = self.n_in
        D_context = D - self.n_current_kp * 2     # past-frame coords (generalised)

        self.masks[0] = np.repeat(np.arange(0, D, 2), 2)
        for l in range(L):
            self.masks[l + 1] = np.concatenate(
                ((self.hidden_dims[l]) // D + 1)
                * [np.repeat(np.arange(0, D - 2, 2), 2)]
            )
        self.masks[L + 1] = self.masks[0]

        for i in range(len(self.masks) - 2):
            m = self.masks[i]
            m[m < D_context] = 0
            m_next = self.masks[i + 1]
            M = torch.zeros(len(m_next), len(m))
            for j in range(len(m_next)):
                M[j, :] = torch.from_numpy((m_next[j] >= m).astype(int))
            self.mask_matrix.append(M)

        i = i + 1
        m = self.masks[i]
        m[m < D_context] = -1
        m_next = self.masks[i + 1]
        M = torch.zeros(len(m_next), len(m))
        for j in range(len(m_next)):
            M[j, :] = torch.from_numpy((m_next[j] > m).astype(int))
        self.mask_matrix.append(M)

        m = self.mask_matrix.pop(-1)
        self.mask_matrix.append(torch.cat((m, m), dim=0))

        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))


class PartialAutoregressiveSceneFC(nn.Module):
    """Drop-in scene density model: input (B, T, N', A) -> (B, 2*N'*T)."""

    def __init__(self, dim: int, n_current_kp: int, hidden_dims: List[int],
                 droppout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.n_current_kp = n_current_kp
        self.hidden_dims = hidden_dims
        self.model = SceneMADEPartial(dim, n_current_kp, hidden_dims, droppout)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x.flatten(start_dim=1))
