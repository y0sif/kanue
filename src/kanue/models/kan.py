"""KAN-based NNUE model variants.

Replaces the post-accumulator MLP layers with KAN layers that use
trainable activation functions (B-splines) instead of fixed CReLU.

The feature transformer (sparse input -> accumulator) stays identical
to the baseline -- KAN only replaces the activation/output layers.
"""

import torch
import torch.nn as nn

from kanue.models.baseline import sparse_to_dense
from kanue.models.kan_layer import EfficientKANLayer


class KanBoard768(nn.Module):
    """NNUE with KAN post-accumulator layers.

    Architecture:
        ft(768 -> hidden) shared for both perspectives [same as baseline]
        concat(stm, nstm) -> KAN(hidden*2 -> hidden) -> KAN(hidden -> 1) -> sigmoid

    The KAN layers replace both the CReLU activation and the output linear layer.
    Each edge in the KAN has its own trainable B-spline activation function.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        self.ft = nn.Linear(768, hidden_size)
        self.kan1 = EfficientKANLayer(hidden_size * 2, hidden_size, grid_size, spline_order)
        self.kan2 = EfficientKANLayer(hidden_size, 1, grid_size, spline_order)

    def forward(self, stm: torch.Tensor, nstm: torch.Tensor) -> torch.Tensor:
        if stm.shape[-1] != 768:
            stm = sparse_to_dense(stm)
            nstm = sparse_to_dense(nstm)

        stm_hidden = self.ft(stm)
        nstm_hidden = self.ft(nstm)
        hidden = torch.cat([stm_hidden, nstm_hidden], dim=1)
        hidden = self.kan1(hidden)
        output = self.kan2(hidden)
        return torch.sigmoid(output)


class HybridKanBoard768(nn.Module):
    """Hybrid: standard linear feature transformer + KAN output head.

    Uses CReLU after the feature transformer (like baseline) but replaces
    only the final output layer with a KAN layer.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        self.ft = nn.Linear(768, hidden_size)
        self.kan_out = EfficientKANLayer(hidden_size * 2, 1, grid_size, spline_order)

    def forward(self, stm: torch.Tensor, nstm: torch.Tensor) -> torch.Tensor:
        if stm.shape[-1] != 768:
            stm = sparse_to_dense(stm)
            nstm = sparse_to_dense(nstm)

        stm_hidden = self.ft(stm)
        nstm_hidden = self.ft(nstm)
        hidden = torch.cat([stm_hidden, nstm_hidden], dim=1)
        hidden = torch.clamp(hidden, 0.0, 1.0)
        output = self.kan_out(hidden)
        return torch.sigmoid(output)
