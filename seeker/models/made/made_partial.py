from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import ReLU

# This implementation of MADE is copied from: https://github.com/e-hulten/made and adapted.

class MaskedLinear(nn.Linear):
    """Linear transformation with masked out elements. y = x.dot(mask*W.T) + b"""

    def __init__(self, n_in: int, n_out: int, bias: bool = True) -> None:
        """
        Args:
            n_in: Size of each input sample.
            n_out:Size of each output sample.
            bias: Whether to include additive bias. Default: True.
        """
        super().__init__(n_in, n_out, bias)
        self.register_buffer('mask', None)

    def initialise_mask(self, mask: Tensor):
        """Internal method to initialise mask."""
        self.mask = mask

    def forward(self, x: Tensor) -> Tensor:
        """Apply masked linear transformation."""
        return F.linear(x, self.mask * self.weight, self.bias)


class MADEPartial(nn.Module):
    def __init__(
        self,
        n_in: int,
        hidden_dims: List[int],
        droppout: float = 0.5
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = int(2 * n_in)
        self.hidden_dims = hidden_dims
        self.gaussian = True
        self.masks = {}
        self.mask_matrix = []
        self.layers = []

        # List of layers sizes.
        dim_list = [self.n_in, *hidden_dims, self.n_out]
        # Make layers and activation functions.
        for i in range(len(dim_list) - 2):
            self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]),)
            self.layers.append(ReLU())
            self.layers.append(torch.nn.Dropout(p=droppout))
        # Hidden layer to output layer.
        self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))
        # Create model.
        self.model = nn.Sequential(*self.layers)
        # Get masks for the masked activations.
        self._create_masks()

    def forward(self, x: Tensor) -> Tensor:
        # self._create_masks()
        """Forward pass."""
        if self.gaussian:
            # If the output is Gaussan, return raw mus and sigmas.
            return self.model(x)
        else:
            # If the output is Bernoulli, run it trough sigmoid to squash p into (0,1).
            return torch.sigmoid(self.model(x))

    def _create_masks(self) -> None:
        """Create masks for the hidden layers."""
        L = len(self.hidden_dims)
        D = self.n_in
        D_context = D - 18*2

        self.masks[0] = np.repeat(np.arange(0, D, 2), 2)


        # Set the connectivity number m for the hidden layers.
        for l in range(L):
            self.masks[l + 1] = np.concatenate(((self.hidden_dims[l])//D+1)*[np.repeat(np.arange(0, D-2, 2), 2)])

        # Add m for output layer. Output order same as input order.
        self.masks[L + 1] = self.masks[0]

        # Create mask matrix for input -> hidden_1 -> ... -> hidden_L.
        for i in range(len(self.masks) - 2):
            m = self.masks[i]
            m[m<D_context] = 0
            m_next = self.masks[i + 1]
            # Initialise mask matrix.
            M = torch.zeros(len(m_next), len(m))
            for j in range(len(m_next)):
                # Use broadcasting to compare m_next[j] to each element in m.
                M[j, :] = torch.from_numpy((m_next[j] >= m).astype(int))
            # Append to mask matrix list.
            self.mask_matrix.append(M)

        i = i+1
        m = self.masks[i]
        m[m<D_context] = -1
        m_next = self.masks[i + 1]
        # Initialise mask matrix.
        M = torch.zeros(len(m_next), len(m))
        for j in range(len(m_next)):
            # Use broadcasting to compare m_next[j] to each element in m.
            M[j, :] = torch.from_numpy((m_next[j] > m).astype(int))
        # Append to mask matrix list.
        self.mask_matrix.append(M)

        

        # If the output is Gaussian, double the number of output units (mu,sigma).
        # Pairwise identical masks.
        m = self.mask_matrix.pop(-1)
        self.mask_matrix.append(torch.cat((m, m), dim=0))

        # Initalise the MaskedLinear layers with weights.
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))
