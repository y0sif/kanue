"""Smoke tests for model forward passes."""

import torch


def test_baseline_forward_sparse():
    from kanue.models.baseline import NnBoard768

    model = NnBoard768(hidden_size=32)
    # Sparse indices: (batch, max_active) with -1 padding
    stm = torch.randint(0, 768, (4, 16))
    nstm = torch.randint(0, 768, (4, 16))
    out = model(stm, nstm)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0, 1] (sigmoid)"


def test_baseline_forward_dense():
    from kanue.models.baseline import NnBoard768

    model = NnBoard768(hidden_size=32)
    # Dense input: (batch, 768)
    stm = torch.randn(4, 768)
    nstm = torch.randn(4, 768)
    out = model(stm, nstm)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()


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
    assert basis.shape == (4, 16 * (5 + 3))


def test_sparse_to_dense():
    from kanue.models.baseline import sparse_to_dense

    indices = torch.tensor([[0, 5, 767, -1], [3, -1, -1, -1]], dtype=torch.int32)
    dense = sparse_to_dense(indices, 768)
    assert dense.shape == (2, 768)
    assert dense[0, 0] == 1.0
    assert dense[0, 5] == 1.0
    assert dense[0, 767] == 1.0
    assert dense[0].sum() == 3.0
    assert dense[1, 3] == 1.0
    assert dense[1].sum() == 1.0


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
