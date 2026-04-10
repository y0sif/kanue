"""Fast native data loader using compiled Rust shared library (kanue-parse).

This replaces the slow Python feature extraction with compiled Rust code,
following the same pattern as marlinflow (Rust cdylib + Python ctypes).

Expected speedup: ~100x (490s -> ~5s per epoch for 8M positions).
"""

import ctypes
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

_LIB = None
_LIB_PATH = None


def _find_lib() -> Path:
    """Search for libkanue_parse.so in standard locations."""
    candidates = [
        # Colab: built from cloned repo
        Path("/content/kanue/crates/kanue-parse/target/release/libkanue_parse.so"),
        # Local dev
        Path(__file__).parent.parent.parent.parent / "crates" / "kanue-parse" / "target" / "release" / "libkanue_parse.so",
        # Current directory
        Path("./libkanue_parse.so"),
        # Google Drive cache
        Path("/content/drive/MyDrive/kanue/lib/libkanue_parse.so"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "libkanue_parse.so not found. Build it with:\n"
        "  cd crates/kanue-parse && cargo build --release"
    )


def _load_lib():
    """Load the native library and set up function signatures."""
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB

    path = _find_lib()
    lib = ctypes.cdll.LoadLibrary(str(path))

    # batch_new(capacity: u32) -> *mut Batch
    lib.batch_new.argtypes = [ctypes.c_uint32]
    lib.batch_new.restype = ctypes.c_void_p

    # batch_drop(batch: *mut Batch)
    lib.batch_drop.argtypes = [ctypes.c_void_p]
    lib.batch_drop.restype = None

    # batch_get_len(batch: *const Batch) -> u32
    lib.batch_get_len.argtypes = [ctypes.c_void_p]
    lib.batch_get_len.restype = ctypes.c_uint32

    # batch_get_stm_ptr(batch: *const Batch) -> *const i32
    lib.batch_get_stm_ptr.argtypes = [ctypes.c_void_p]
    lib.batch_get_stm_ptr.restype = ctypes.POINTER(ctypes.c_int32)

    # batch_get_nstm_ptr(batch: *const Batch) -> *const i32
    lib.batch_get_nstm_ptr.argtypes = [ctypes.c_void_p]
    lib.batch_get_nstm_ptr.restype = ctypes.POINTER(ctypes.c_int32)

    # batch_get_targets_ptr(batch: *const Batch) -> *const f32
    lib.batch_get_targets_ptr.argtypes = [ctypes.c_void_p]
    lib.batch_get_targets_ptr.restype = ctypes.POINTER(ctypes.c_float)

    # batch_fill(batch, data_ptr, count, blend)
    lib.batch_fill.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_float,
    ]
    lib.batch_fill.restype = None

    _LIB = lib
    _LIB_PATH = path
    return lib


class NativeBatch:
    """Wrapper around Rust-allocated batch buffer."""

    def __init__(self, capacity: int):
        self._lib = _load_lib()
        self._ptr = self._lib.batch_new(capacity)
        self.capacity = capacity

    def fill(self, data: np.ndarray, blend: float = 0.0):
        """Fill batch from raw bytes. data must be contiguous uint8 array."""
        count = len(data) // 32
        self._lib.batch_fill(
            self._ptr,
            data.ctypes.data,
            count,
            ctypes.c_float(blend),
        )

    def to_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zero-copy view into Rust memory, then transfer to device."""
        n = self._lib.batch_get_len(self._ptr)
        if n == 0:
            return (
                torch.zeros(0, 32, dtype=torch.int32),
                torch.zeros(0, 32, dtype=torch.int32),
                torch.zeros(0, 1, dtype=torch.float32),
            )

        # Zero-copy numpy views into Rust-owned buffers
        stm_np = np.ctypeslib.as_array(self._lib.batch_get_stm_ptr(self._ptr), shape=(n * 32,))
        nstm_np = np.ctypeslib.as_array(self._lib.batch_get_nstm_ptr(self._ptr), shape=(n * 32,))
        tgt_np = np.ctypeslib.as_array(self._lib.batch_get_targets_ptr(self._ptr), shape=(n,))

        stm_t = torch.from_numpy(stm_np.reshape(n, 32))
        nstm_t = torch.from_numpy(nstm_np.reshape(n, 32))
        tgt_t = torch.from_numpy(tgt_np.reshape(n, 1))

        if device.type == "cuda":
            stm_t = stm_t.clone().pin_memory().to(device, non_blocking=True)
            nstm_t = nstm_t.clone().pin_memory().to(device, non_blocking=True)
            tgt_t = tgt_t.clone().pin_memory().to(device, non_blocking=True)
        elif device.type != "cpu":
            stm_t = stm_t.to(device)
            nstm_t = nstm_t.to(device)
            tgt_t = tgt_t.to(device)

        return stm_t, nstm_t, tgt_t

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr is not None:
            self._lib.batch_drop(self._ptr)
            self._ptr = None


class NativeBatchLoader(IterableDataset):
    """Fast data loader using native Rust feature extraction.

    Drop-in replacement for BulletformatBatchDataset.
    Same interface: yields (stm, nstm, targets) tuples.
    """

    def __init__(
        self,
        path: str | Path,
        batch_size: int = 16384,
        wdl_weight: float = 0.0,
        shuffle: bool = True,
        device: torch.device | None = None,
    ):
        self.path = Path(path)
        self.batch_size = batch_size
        self.wdl_weight = wdl_weight
        self.shuffle = shuffle
        self.device = device or torch.device("cpu")

        # Memory-map the raw file
        file_size = self.path.stat().st_size
        assert file_size % 32 == 0, f"File size {file_size} not a multiple of 32"
        self.n_positions = file_size // 32
        self.data = np.memmap(self.path, dtype=np.uint8, mode="r")

        # Pre-allocate the Rust batch buffer (reused across iterations)
        self._batch = NativeBatch(batch_size)

    def __iter__(self):
        indices = np.random.permutation(self.n_positions) if self.shuffle else np.arange(self.n_positions)
        data_2d = self.data.reshape(-1, 32)

        for start in range(0, self.n_positions, self.batch_size):
            end = min(start + self.batch_size, self.n_positions)
            batch_idx = indices[start:end]

            # Gather batch into contiguous memory (required for pointer passing)
            batch_bytes = np.ascontiguousarray(data_2d[batch_idx])

            self._batch.fill(batch_bytes, blend=self.wdl_weight)
            yield self._batch.to_tensors(self.device)

    def __len__(self) -> int:
        return (self.n_positions + self.batch_size - 1) // self.batch_size
