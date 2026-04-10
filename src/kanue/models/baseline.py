"""Baseline NNUE model (marlinflow-style NnBoard768).

Standard architecture: sparse 768 inputs -> feature transformer -> CReLU -> output.
This serves as the control for comparing against KAN variants.
"""

import torch
import torch.nn as nn


class NnBoard768(nn.Module):
    """Perspective-net with 768 binary inputs (2 colors x 6 pieces x 64 squares).

    Architecture:
        ft(768 -> hidden) shared for both perspectives
        concat(stm, nstm) -> clamp(0,1) -> out(hidden*2 -> 1) -> sigmoid
    """

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.ft = nn.Linear(768, hidden_size)
        self.out = nn.Linear(hidden_size * 2, 1)

    def forward(
        self,
        stm_indices: torch.Tensor,
        nstm_indices: torch.Tensor,
        stm_values: torch.Tensor,
        nstm_values: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        # Build sparse tensors and convert to dense
        stm_sparse = torch.sparse_coo_tensor(
            stm_indices, stm_values, (batch_size, 768), dtype=torch.float32
        ).to_dense()
        nstm_sparse = torch.sparse_coo_tensor(
            nstm_indices, nstm_values, (batch_size, 768), dtype=torch.float32
        ).to_dense()

        # Feature transformer (shared weights, both perspectives)
        stm_hidden = self.ft(stm_sparse)
        nstm_hidden = self.ft(nstm_sparse)

        # Concat perspectives, CReLU activation, output
        hidden = torch.cat([stm_hidden, nstm_hidden], dim=1)
        hidden = torch.clamp(hidden, 0.0, 1.0)
        return torch.sigmoid(self.out(hidden))


class NnBoard768Dense(nn.Module):
    """Dense-input variant for use with pre-densified batches.

    Same architecture but accepts dense (batch, 768) tensors directly.
    Simpler for prototyping and Colab use.
    """

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.ft = nn.Linear(768, hidden_size)
        self.out = nn.Linear(hidden_size * 2, 1)

    def forward(self, stm: torch.Tensor, nstm: torch.Tensor) -> torch.Tensor:
        stm_hidden = self.ft(stm)
        nstm_hidden = self.ft(nstm)
        hidden = torch.cat([stm_hidden, nstm_hidden], dim=1)
        hidden = torch.clamp(hidden, 0.0, 1.0)
        return torch.sigmoid(self.out(hidden))
