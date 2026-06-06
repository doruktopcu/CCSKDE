from typing import List, Tuple
import torch.nn as nn
from torch import Tensor
from models.made.made import MADE


class  AutoregressiveFC(nn.Module):
    def __init__(
        self, dim: int, hidden_dims: List[int]
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dims = hidden_dims
        self.model = MADE(dim, hidden_dims)
            
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        x = x.flatten(start_dim=1)
        out = self.model(x)

        return out
