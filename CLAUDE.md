# CLAUDE.md

## Project Overview

kanue: Research project testing KAN (trainable B-spline activations) as a replacement for fixed CReLU/SCReLU in chess NNUE evaluation networks.

## Build & Run

```bash
# Install
pip install -e ".[dev]"

# Lint
ruff check src/

# Format
ruff format src/

# Test
pytest
```

## Architecture

- `src/kanue/models/baseline.py` -- Standard NNUE (NnBoard768Dense) as control
- `src/kanue/models/kan.py` -- KAN variants (KanBoard768, HybridKanBoard768)
- `src/kanue/models/kan_layer.py` -- Efficient B-spline KAN layer implementation
- `src/kanue/data/loader.py` -- Board768 feature encoding, plaintext data loading
- `src/kanue/utils/drive.py` -- Google Drive persistence for Colab
- `src/kanue/utils/training.py` -- Shared train/eval loops
- `notebooks/01-04` -- Sequential Colab notebooks (data -> baseline -> KAN -> analysis)

## Conventions

- All models accept dense (batch, 768) tensors via `forward(stm, nstm)`
- Feature transformer is shared between perspectives (marlinflow convention)
- Training targets: sigmoid(cp/400) interpolated with WDL
- Checkpoints and logs auto-save to Google Drive under `/content/drive/MyDrive/kanue/`
- Ruff for linting and formatting (line-length=100)

## Key Design Decisions

- Dense data loading (not sparse) for Colab simplicity -- sacrifice some speed for debuggability
- Own KAN layer implementation (not external dep) for full control over B-spline params
- Board768 features only (not HalfKA) to keep the research focused on activation functions
