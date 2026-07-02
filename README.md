# kanue

> Can trainable activation functions improve chess evaluation? Replacing NNUE's fixed activations with Kolmogorov-Arnold Networks.

[![CI](https://github.com/y0sif/kanue/actions/workflows/ci.yml/badge.svg)](https://github.com/y0sif/kanue/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)

**kanue** is a research project that replaces the fixed activation functions (CReLU/SCReLU) in chess NNUE evaluation networks with trainable B-spline activations from Kolmogorov-Arnold Networks (KAN). The goal: determine whether learnable activations improve chess position evaluation quality, and at what cost.

## The paper

This repository is the offline prototype behind the paper **"Trainable Activation Functions for Real-Time Chess Engine Evaluation: Custom CUDA Kernels, Quantized Lookup-Table Inference, and Playing-Strength Validation"** (preprint forthcoming on arXiv). It compares KAN and CReLU NNUE post-accumulator heads under identical training conditions on 8M Stockfish-evaluated positions from [linrock/test77](https://huggingface.co/datasets/linrock/test77).

Headline prototype results (vs the matched CReLU baseline):

- **B-spline KAN**: −22% test loss and +2.1pp outcome-prediction accuracy
- **Hybrid variant** (single KAN layer, at parameter parity): −14% test loss

The production side of the paper lives in two sibling repos:

- [y0sif/bullet](https://github.com/y0sif/bullet) — research fork of the [bullet](https://github.com/jw1912/bullet) trainer: B-spline and ReLU-KAN basis operations as IR nodes with custom CUDA kernels, a fusion pass, and int8 lookup-table export
- [y0sif/akimbo](https://github.com/y0sif/akimbo) — research fork of the [akimbo](https://github.com/jw1912/akimbo) engine: int8 lookup-table KAN evaluation and SPRT playing-strength validation (see the branch map in that repo's README)

## Quick Start

Open any notebook in Google Colab (Colab Pro recommended for GPU access):

| Notebook | Description | Colab |
|---|---|---|
| `01_data_preparation` | Generate training data via Stockfish self-play | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/y0sif/kanue/blob/main/notebooks/01_data_preparation.ipynb) |
| `02_baseline_nnue` | Train standard NNUE (control) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/y0sif/kanue/blob/main/notebooks/02_baseline_nnue.ipynb) |
| `03_kan_nnue` | Train KAN variants and grid size sweep | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/y0sif/kanue/blob/main/notebooks/03_kan_nnue.ipynb) |
| `04_analysis` | Compare results, visualize learned activations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/y0sif/kanue/blob/main/notebooks/04_analysis.ipynb) |

## Background

### What is NNUE?

NNUE (Efficiently Updatable Neural Network) is the neural network architecture used in modern chess engines like Stockfish. It uses:
- Sparse binary input features (768 features: 2 colors x 6 piece types x 64 squares)
- An incrementally-updatable accumulator (the key to real-time speed)
- Fixed activation functions (CReLU, SCReLU) in post-accumulator layers
- Integer quantization for fast CPU inference

### What is KAN?

Kolmogorov-Arnold Networks place **trainable activation functions on edges** instead of using fixed activations on nodes. Each connection learns its own univariate function via B-spline basis functions. This provides:
- More expressive per-connection transformations
- Interpretable learned activations (you can visualize what each edge learned)
- Potentially better approximation of structured low-dimensional functions

### The Hypothesis

Chess evaluation is a structured, low-dimensional function (board state -> scalar eval). KAN's strength is exactly on such functions. By replacing NNUE's fixed CReLU/SCReLU with trainable splines, we might get:
- Better evaluation accuracy with the same number of parameters
- New insights from visualizing what the activation functions learn about chess

## Architecture

```
Standard NNUE:
  768 sparse -> ft(128) -> [CReLU] -> concat(stm, nstm) -> Linear(256->1) -> sigmoid

KAN NNUE (this project):
  768 sparse -> ft(128) -> concat(stm, nstm) -> KAN(256->128) -> KAN(128->1) -> sigmoid
                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                 Trainable B-spline activations
                                                 replace CReLU + Linear
```

The feature transformer (sparse input -> accumulator) is kept identical. KAN only replaces the post-accumulator layers, which is where activation function choice matters most.

## Variants Tested

| Variant | Description | Post-accumulator |
|---|---|---|
| `NnBoard768Dense` | Baseline (marlinflow-style) | Linear + CReLU + Linear |
| `KanBoard768` | Full KAN replacement | KAN(256->128) + KAN(128->1) |
| `HybridKanBoard768` | Minimal KAN | CReLU + KAN(256->1) |

Grid size sweep tests `grid_size={3, 5, 8}` to find the spline resolution sweet spot.

## How It Works

1. **Data**: 8M Stockfish-evaluated positions from [linrock/test77](https://huggingface.co/datasets/linrock/test77) for the paper's experiments (`tools/binpack-to-bullet` converts Stockfish `.binpack` files to bulletformat); the `01` notebook can alternatively generate data via Stockfish self-play
2. **Encoding**: Board768 features (binary, 768-dim) for side-to-move and non-side-to-move perspectives
3. **Training**: Identical hyperparameters (Adam, MSE loss, same LR schedule) across all variants
4. **Comparison**: Loss convergence, winner prediction accuracy, parameter efficiency, training time

All intermediate results (checkpoints, logs, preprocessed data) persist to Google Drive across Colab sessions.

## Project Structure

```
kanue/
  notebooks/
    01_data_preparation.ipynb    # Data generation and preprocessing
    02_baseline_nnue.ipynb       # Standard NNUE training (control)
    03_kan_nnue.ipynb            # KAN variant training and grid sweep
    04_analysis.ipynb            # Comparison and visualization
  src/kanue/
    models/
      baseline.py                # NnBoard768, NnBoard768Dense
      kan.py                     # KanBoard768, HybridKanBoard768
      kan_layer.py               # EfficientKANLayer (B-spline implementation)
    data/
      loader.py                  # Board768 encoding, plaintext data loading
    utils/
      drive.py                   # Google Drive checkpointing
      training.py                # Shared training/eval loops
  crates/
    kanue-parse/                 # Rust cdylib: fast Board768 batch loading from bulletformat data (via ctypes)
  tools/
    binpack-to-bullet/           # Rust CLI: convert Stockfish .binpack (e.g. test77) to bulletformat .data
```

## Local Development

```bash
git clone https://github.com/y0sif/kanue.git
cd kanue
pip install -e ".[dev]"
ruff check src/
pytest
```

## Prior Work

This project extends [rough_hook](https://github.com/y0sif/rough_hook), which tested KAN across three chess domains:
- **Computer vision** (board recognition): KAN+CNN achieved 97.86% accuracy (+1.81% over MLP)
- **Engine evaluation**: Hit a wall due to CUDA/framework constraints (this project picks up here)
- **Cheat detection**: Feature representation was the bottleneck, not architecture

## References

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) (Liu et al., 2024)
- [efficient-kan](https://github.com/Blealtan/efficient-kan) (Blealtan)
- [marlinflow](https://github.com/jnlt3/marlinflow) (NNUE trainer, architecture reference)
- [bullet](https://github.com/jw1912/bullet) (Rust NNUE trainer; the paper's production training uses the [y0sif/bullet](https://github.com/y0sif/bullet) research fork)
- [Stockfish NNUE](https://github.com/official-stockfish/nnue-pytorch)
- [LUT-KAN](https://arxiv.org/abs/2601.03332) (quantization for production inference)

## License

[MIT](LICENSE)
