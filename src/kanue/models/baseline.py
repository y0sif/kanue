"""Baseline NNUE model (marlinflow-style NnBoard768).

Standard architecture: sparse 768 inputs -> feature transformer -> CReLU -> output.
This serves as the control for comparing against KAN variants.

Accepts sparse feature indices (N, max_active) with -1 padding,
matching the output format of the bulletformat data loader.
"""

import torch
import torch.nn as nn


def sparse_to_dense(indices: torch.Tensor, size: int = 768) -> torch.Tensor:
    """Convert sparse feature indices to dense binary vectors.

    Args:
        indices: (batch, max_active) int32 tensor with -1 for padding.
        size: Feature dimension (768 for Board768).

    Returns:
        (batch, size) float32 dense tensor.
    """
    batch_size = indices.shape[0]
    dense = torch.zeros(batch_size, size, device=indices.device)
    mask = indices >= 0
    # Build (row, col) pairs for valid indices only
    rows = torch.arange(batch_size, device=indices.device).unsqueeze(1).expand_as(indices)[mask]
    cols = indices[mask].long()
    dense[rows, cols] = 1.0
    return dense


class NnBoard768(nn.Module):
    """Perspective-net with 768 binary inputs.

    Architecture:
        ft(768 -> hidden) shared for both perspectives
        concat(stm, nstm) -> clamp(0,1) -> out(hidden*2 -> 1) -> sigmoid

    Accepts both sparse indices (N, 32) and dense (N, 768) inputs.
    """

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.ft = nn.Linear(768, hidden_size)
        self.out = nn.Linear(hidden_size * 2, 1)

    def forward(self, stm: torch.Tensor, nstm: torch.Tensor) -> torch.Tensor:
        # Auto-detect sparse vs dense input
        if stm.shape[-1] != 768:
            stm = sparse_to_dense(stm)
            nstm = sparse_to_dense(nstm)

        stm_hidden = self.ft(stm)
        nstm_hidden = self.ft(nstm)
        hidden = torch.cat([stm_hidden, nstm_hidden], dim=1)
        hidden = torch.clamp(hidden, 0.0, 1.0)
        return torch.sigmoid(self.out(hidden))
