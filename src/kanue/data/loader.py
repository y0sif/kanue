"""Chess position data loading for NNUE training.

Supports loading from plaintext format (FEN | eval | WDL) which is the
interchange format most trainers can export to. For Colab, we convert
binpack -> plaintext once, then work with the simpler format.

Board768 feature encoding: 2 colors x 6 piece types x 64 squares = 768 binary features.
Feature index = color * 384 + piece_type * 64 + square
"""

from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset


PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_features(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    """Convert a chess.Board to Board768 feature vectors for both perspectives.

    Returns:
        (stm_features, nstm_features): each is a (768,) binary float32 array.
    """
    stm_features = np.zeros(768, dtype=np.float32)
    nstm_features = np.zeros(768, dtype=np.float32)

    # Determine perspective flip
    flip = board.turn == chess.BLACK

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        piece_idx = PIECE_TO_INDEX[piece.piece_type]
        sq = square ^ 56 if flip else square  # Flip rank for black's perspective

        # STM perspective: own pieces are color=0, opponent pieces are color=1
        if piece.color == board.turn:
            stm_features[0 * 384 + piece_idx * 64 + sq] = 1.0
        else:
            stm_features[1 * 384 + piece_idx * 64 + sq] = 1.0

        # NSTM perspective: flip colors and board
        nstm_sq = square ^ 56 if not flip else square
        if piece.color != board.turn:
            nstm_features[0 * 384 + piece_idx * 64 + nstm_sq] = 1.0
        else:
            nstm_features[1 * 384 + piece_idx * 64 + nstm_sq] = 1.0

    return stm_features, nstm_features


def sigmoid(x: float, scale: float = 400.0) -> float:
    """Convert centipawn eval to win probability."""
    return 1.0 / (1.0 + np.exp(-x / scale))


def load_positions_from_plaintext(
    path: str | Path,
    max_positions: int | None = None,
    max_cp: int = 3000,
    wdl_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load positions from plaintext format: 'FEN | eval_cp | wdl'.

    Args:
        path: Path to the plaintext file.
        max_positions: Maximum number of positions to load.
        max_cp: Filter out positions with |eval| > max_cp.
        wdl_weight: Interpolation weight between eval and WDL target.
            0.0 = pure eval prediction, 1.0 = pure WDL prediction.

    Returns:
        (stm_features, nstm_features, targets): numpy arrays.
        Features are (N, 768) float32, targets are (N, 1) float32.
    """
    stm_list = []
    nstm_list = []
    target_list = []

    path = Path(path)
    with path.open() as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue

            fen = parts[0].strip()
            cp = int(parts[1].strip())
            wdl = float(parts[2].strip())

            if abs(cp) > max_cp:
                continue

            board = chess.Board(fen)
            stm, nstm = board_to_features(board)

            # Target: interpolate between eval-based and WDL-based
            eval_target = sigmoid(cp)
            target = eval_target * (1.0 - wdl_weight) + wdl * wdl_weight

            stm_list.append(stm)
            nstm_list.append(nstm)
            target_list.append(target)

            if max_positions and len(stm_list) >= max_positions:
                break

    return (
        np.array(stm_list, dtype=np.float32),
        np.array(nstm_list, dtype=np.float32),
        np.array(target_list, dtype=np.float32).reshape(-1, 1),
    )


class ChessDataset(Dataset):
    """PyTorch dataset wrapping pre-loaded chess position arrays."""

    def __init__(
        self,
        stm: np.ndarray,
        nstm: np.ndarray,
        targets: np.ndarray,
    ):
        self.stm = torch.from_numpy(stm)
        self.nstm = torch.from_numpy(nstm)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.stm[idx], self.nstm[idx], self.targets[idx]
