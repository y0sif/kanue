"""Efficient KAN layer implementation.

Based on the efficient-kan approach: evaluate B-spline basis functions first,
then combine with a standard matrix multiply. This avoids the expensive
(batch, out, in) tensor expansion of the original pykan.

Reference: https://github.com/Blealtan/efficient-kan
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientKANLayer(nn.Module):
    """Single KAN layer with B-spline trainable activations.

    Each edge (i, j) in the layer has a trainable activation function
    parameterized as a linear combination of B-spline basis functions.
    The key optimization: compute basis values first, then use a standard
    linear layer to combine them.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        grid_size: Number of grid intervals for B-splines.
        spline_order: Order of B-spline basis functions.
        scale_noise: Scale of random noise for spline weight initialization.
        scale_base: Scale of the base weight (residual linear connection).
        grid_eps: Fraction of grid range to extend beyond data range.
        grid_range: Range of the grid.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        grid_eps: float = 0.02,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Uniform grid for B-spline evaluation
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
        self.register_buffer("grid", grid.unsqueeze(0).expand(in_features, -1))

        # Trainable weights
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features * (grid_size + spline_order)))

        # Store init params
        self._scale_noise = scale_noise
        self._scale_base = scale_base

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            noise = (torch.rand_like(self.spline_weight) - 0.5) * self._scale_noise
            self.spline_weight.copy_(noise / (self.grid_size + self.spline_order))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis functions on input x.

        Args:
            x: (batch, in_features) input tensor.

        Returns:
            (batch, in_features * (grid_size + spline_order)) basis values.
        """
        x = x.unsqueeze(-1)  # (batch, in, 1)
        grid = self.grid  # (in, grid_points)

        # Cox-de Boor recursion for B-spline basis
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()

        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        # (batch, in, grid_size + spline_order) -> (batch, in * (grid_size + spline_order))
        return bases.reshape(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear component (residual connection, like SiLU-weighted)
        base_output = F.linear(F.silu(x), self.base_weight * self._scale_base)

        # Spline component: evaluate basis, then linear combine
        spline_basis = self.b_splines(x)
        spline_output = F.linear(spline_basis, self.spline_weight)

        return base_output + spline_output
