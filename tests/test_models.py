"""Smoke tests for model forward passes."""

import torch
import pytest


def test_baseline_forward():
    from kanue.models.baseline import NnBoard768Dense

    model = NnBoard768Dense(hidden_size=32)
    stm = torch.randn(4, 768)
    nstm = torch.randn(4, 768)
    out = model(stm, nstm)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0, 1] (sigmoid)"


def test_kan_forward():
    from kanue.models.kan import KanBoard768

    model = KanBoard768(hidden_size=32, grid_size=3, spline_order=3)
    stm = torch.randn(4, 768)
    nstm = torch.randn(4, 768)
    out = model(stm, nstm)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_hybrid_kan_forward():
    from kanue.models.kan import HybridKanBoard768

    model = HybridKanBoard768(hidden_size=32, grid_size=3, spline_order=3)
    stm = torch.randn(4, 768)
    nstm = torch.randn(4, 768)
    out = model(stm, nstm)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_kan_layer_bsplines():
    from kanue.models.kan_layer import EfficientKANLayer

    layer = EfficientKANLayer(16, 8, grid_size=5, spline_order=3)
    x = torch.randn(4, 16)
    basis = layer.b_splines(x)
    # Should be (batch, in_features * (grid_size + spline_order))
    assert basis.shape == (4, 16 * (5 + 3))


def test_board_to_features():
    import chess
    from kanue.data.loader import board_to_features

    board = chess.Board()  # Starting position
    stm, nstm = board_to_features(board)
    assert stm.shape == (768,)
    assert nstm.shape == (768,)
    # Starting position has 32 pieces -> 32 active features per perspective
    assert stm.sum() == 32
    assert nstm.sum() == 32


def test_dataset():
    import numpy as np
    from kanue.data import ChessDataset

    stm = np.random.rand(10, 768).astype(np.float32)
    nstm = np.random.rand(10, 768).astype(np.float32)
    targets = np.random.rand(10, 1).astype(np.float32)
    ds = ChessDataset(stm, nstm, targets)
    assert len(ds) == 10
    s, n, t = ds[0]
    assert s.shape == (768,)
    assert t.shape == (1,)
