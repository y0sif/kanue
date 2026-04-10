"""Chess position data loading for NNUE training.

Efficient data pipeline inspired by Bullet's approach:
1. Stockfish .binpack files are converted to flat 32-byte bulletformat records
   using the Rust `binpack-to-bullet` tool (runs once)
2. Python memory-maps the flat binary file (zero-copy, instant)
3. Feature extraction is vectorized in numpy/torch (no python-chess per position)

BulletFormat ChessBoard layout (32 bytes, STM-relative):
    occ:     u64    occupancy bitboard
    pcs:     16xU8  nibble-packed pieces (2 per byte), ordered by occ bit scan
    score:   i16    centipawn eval (STM-relative)
    result:  u8     0=loss, 1=draw, 2=win (STM-relative)
    ksq:     u8     our king square
    opp_ksq: u8     opponent king square XOR 56
    extra:   3xU8   spare bytes

Board768 feature encoding:
    STM  own pieces: piece_type * 64 + square        (indices 0..383)
    STM  opp pieces: 384 + piece_type * 64 + square   (indices 384..767)
    NSTM flips own/opp perspective
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

# Structured dtype matching bulletformat's ChessBoard (32 bytes)
CHESSBOARD_DTYPE = np.dtype([
    ("occ", "<u8"),       # u64 as raw bytes (8 bytes)
    ("pcs", "u1", (16,)), # nibble-packed pieces
    ("score", "<i2"),     # i16 centipawn eval
    ("result", "u1"),     # 0=loss, 1=draw, 2=win
    ("ksq", "u1"),        # STM king square
    ("opp_ksq", "u1"),    # NSTM king square ^ 56
    ("extra", "u1", (3,)),
], align=False)

# Compact dtype for reading occ as actual u64
CHESSBOARD_RAW_DTYPE = np.dtype([
    ("raw", "u1", (32,)),
])


def _occ_to_squares(occ_bytes: np.ndarray) -> np.ndarray:
    """Convert occupancy bitboard bytes to list of set bit indices.

    Args:
        occ_bytes: (N, 8) array of bytes representing u64 occupancy bitboards.

    Returns:
        (N, 32) array of square indices (-1 padded for empty slots).
    """
    # Reconstruct u64 from little-endian bytes
    occ = np.zeros(len(occ_bytes), dtype=np.uint64)
    for i in range(8):
        occ |= occ_bytes[:, i].astype(np.uint64) << np.uint64(i * 8)

    # Extract set bit indices (max 32 pieces on board)
    squares = np.full((len(occ), 32), -1, dtype=np.int32)
    for pos_idx in range(len(occ)):
        bits = occ[pos_idx]
        j = 0
        while bits:
            sq = int(bits & np.uint64(-np.int64(bits))).bit_length() - 1
            squares[pos_idx, j] = sq
            bits &= bits - np.uint64(1)
            j += 1

    return squares


def _extract_piece_info(pcs: np.ndarray, n_pieces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract piece types and colors from nibble-packed piece array.

    Args:
        pcs: (N, 16) array of nibble-packed pieces (2 per byte).
        n_pieces: (N,) number of pieces per position.

    Returns:
        (piece_types, colors): each (N, 32) int32 arrays, -1 padded.
    """
    # Unpack nibbles: low nibble first, then high nibble
    low = (pcs & 0x0F).astype(np.int32)   # pieces at even indices
    high = (pcs >> 4).astype(np.int32)     # pieces at odd indices

    # Interleave: position 0 = low[0], position 1 = high[0], position 2 = low[1], ...
    all_nibbles = np.zeros((len(pcs), 32), dtype=np.int32)
    all_nibbles[:, 0::2] = low
    all_nibbles[:, 1::2] = high

    # Decode: color = bit 3, piece_type = bits 0-2
    colors = (all_nibbles >> 3) & 1       # 0=STM, 1=opponent
    piece_types = all_nibbles & 0x07      # 0=pawn..5=king

    # Mask out unused slots
    for i in range(len(pcs)):
        n = n_pieces[i]
        if n < 32:
            piece_types[i, n:] = -1
            colors[i, n:] = -1

    return piece_types, colors


def bulletformat_to_features_batch(
    records: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a batch of bulletformat records to Board768 feature tensors.

    This is the vectorized equivalent of Bullet's on-the-fly feature extraction.

    Args:
        records: structured numpy array with CHESSBOARD_DTYPE.

    Returns:
        (stm_indices, nstm_indices, targets): sparse index tensors.
        stm/nstm are (N, max_active) int32 with -1 padding.
        targets are (N, 1) float32.
    """
    n = len(records)
    occ_bytes = records["occ"].reshape(n, 8) if records["occ"].ndim == 1 else records["occ"]

    # Handle the case where occ is stored as single u64 vs 8 bytes
    if occ_bytes.ndim == 1:
        # It's a flat u64 array, convert to bytes view
        occ_bytes = occ_bytes.view(np.uint8).reshape(n, 8)

    squares = _occ_to_squares(occ_bytes)
    pcs = records["pcs"]

    # Count pieces per position
    n_pieces = (squares >= 0).sum(axis=1).astype(np.int32)

    piece_types, colors = _extract_piece_info(pcs, n_pieces)

    # Compute Board768 feature indices
    # STM perspective: own (color=0) -> piece_type * 64 + square
    #                  opp (color=1) -> 384 + piece_type * 64 + square
    stm_indices = np.full((n, 32), -1, dtype=np.int32)
    nstm_indices = np.full((n, 32), -1, dtype=np.int32)

    valid = squares >= 0
    base_feature = piece_types * 64 + squares

    # STM: own pieces at 0..383, opp pieces at 384..767
    stm_offset = colors * 384  # 0 for own, 384 for opp
    stm_indices[valid] = (base_feature + stm_offset)[valid]

    # NSTM: flip perspectives — opp pieces at 0..383, own at 384..767
    # Also flip squares vertically (XOR 56)
    nstm_squares = squares.copy()
    nstm_squares[valid] = nstm_squares[valid] ^ 56
    nstm_base = piece_types * 64 + nstm_squares
    nstm_offset = (1 - colors) * 384  # flip: 384 for own, 0 for opp
    nstm_indices[valid] = (nstm_base + nstm_offset)[valid]

    # Compute targets: sigmoid(score / 400) blended with WDL
    scores = records["score"].astype(np.float32)
    results = records["result"].astype(np.float32) / 2.0  # 0=loss->0, 1=draw->0.5, 2=win->1

    eval_targets = 1.0 / (1.0 + np.exp(-scores / 400.0))

    # Default: pure eval target (wdl_weight can be applied in training)
    targets = eval_targets.reshape(-1, 1)

    return (
        torch.from_numpy(stm_indices),
        torch.from_numpy(nstm_indices),
        torch.from_numpy(targets),
    )


def load_bulletformat(path: str | Path) -> np.ndarray:
    """Memory-map a flat bulletformat .data file.

    Returns a numpy structured array backed by the file (zero-copy).
    """
    path = Path(path)
    file_size = path.stat().st_size
    n_positions = file_size // 32
    assert file_size % 32 == 0, f"File size {file_size} is not a multiple of 32 bytes"

    return np.memmap(path, dtype=CHESSBOARD_DTYPE, mode="r", shape=(n_positions,))


class BulletformatDataset(Dataset):
    """PyTorch dataset reading directly from bulletformat .data files.

    Memory-maps the file and extracts features on-the-fly per batch,
    mimicking Bullet's efficient data pipeline.
    """

    def __init__(self, path: str | Path, wdl_weight: float = 0.0):
        self.data = load_bulletformat(path)
        self.wdl_weight = wdl_weight

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.data[idx:idx+1]  # Keep as structured array
        stm, nstm, targets = bulletformat_to_features_batch(rec)

        if self.wdl_weight > 0:
            wdl = rec["result"].astype(np.float32) / 2.0
            eval_t = targets[0]
            targets[0] = eval_t * (1.0 - self.wdl_weight) + wdl * self.wdl_weight

        return stm[0], nstm[0], targets[0]


class BulletformatBatchDataset(IterableDataset):
    """Streaming dataset that yields pre-batched feature tensors.

    More efficient than BulletformatDataset for large files because
    it extracts features in large vectorized chunks rather than
    per-position.
    """

    def __init__(
        self,
        path: str | Path,
        batch_size: int = 16384,
        wdl_weight: float = 0.0,
        shuffle: bool = True,
    ):
        self.data = load_bulletformat(path)
        self.batch_size = batch_size
        self.wdl_weight = wdl_weight
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data)
        indices = np.random.permutation(n) if self.shuffle else np.arange(n)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_idx = indices[start:end]
            batch = self.data[batch_idx]

            stm, nstm, targets = bulletformat_to_features_batch(batch)

            if self.wdl_weight > 0:
                wdl = batch["result"].astype(np.float32).reshape(-1, 1) / 2.0
                targets = targets * (1.0 - self.wdl_weight) + wdl * self.wdl_weight

            yield stm, nstm, targets

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size


# -- Legacy support --

class ChessDataset(Dataset):
    """PyTorch dataset wrapping pre-loaded numpy arrays (legacy/testing)."""

    def __init__(self, stm: np.ndarray, nstm: np.ndarray, targets: np.ndarray):
        self.stm = torch.from_numpy(stm)
        self.nstm = torch.from_numpy(nstm)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.stm[idx], self.nstm[idx], self.targets[idx]
