# Contributing to kanue

## Development Setup

```bash
git clone https://github.com/y0sif/kanue.git
cd kanue
pip install -e ".[dev]"
```

## Before Submitting a PR

1. **Format**: `ruff format src/`
2. **Lint**: `ruff check src/` (zero warnings)
3. **Test**: `pytest`

## Code Style

- Python 3.10+, type hints on public APIs
- Line length: 100 characters
- Ruff handles formatting and linting

## Adding a New Model Variant

1. Create a new class in `src/kanue/models/` following the same interface:
   - `__init__(self, hidden_size, ...)` with sensible defaults
   - `forward(self, stm, nstm) -> torch.Tensor` returning sigmoid output
2. Export from `src/kanue/models/__init__.py`
3. Add a training section in notebook 03 or create a new notebook

## Adding a New KAN Basis Function

1. Subclass or modify `EfficientKANLayer` in `kan_layer.py`
2. Replace `b_splines()` with your basis function evaluation
3. Keep the same forward signature: basis evaluation + linear combine

## Commit Messages

- `feat:` new model, new experiment
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code restructuring
- `data:` data pipeline changes
