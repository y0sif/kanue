from kanue.data.loader import (
    BulletformatBatchDataset,
    BulletformatDataset,
    ChessDataset,
    load_bulletformat,
)

# Try to import native loader (requires compiled libkanue_parse.so)
try:
    from kanue.data.native import NativeBatchLoader
except (ImportError, FileNotFoundError, OSError):
    NativeBatchLoader = None

__all__ = [
    "BulletformatBatchDataset",
    "BulletformatDataset",
    "ChessDataset",
    "NativeBatchLoader",
    "load_bulletformat",
]
