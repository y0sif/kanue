//! Fast Board768 feature extraction from bulletformat ChessBoard records.
//!
//! Compiled as a cdylib (.so) and called from Python via ctypes.
//! This replaces the slow Python feature extraction loop.

/// Matches bulletformat's ChessBoard exactly: 32 bytes, STM-relative.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ChessBoard {
    occ: u64,       // occupancy bitboard
    pcs: [u8; 16],  // nibble-packed pieces (2 per byte, LSB order of occ)
    score: i16,     // centipawn eval (STM-relative)
    result: u8,     // 0=loss, 1=draw, 2=win (STM-relative)
    ksq: u8,        // STM king square
    opp_ksq: u8,    // NSTM king square ^ 56
    extra: [u8; 3],
}

const MAX_PIECES: usize = 32;

/// Pre-allocated batch buffers for feature extraction results.
pub struct Batch {
    capacity: usize,
    len: usize,
    stm: Vec<i32>,     // capacity * MAX_PIECES, -1 padded
    nstm: Vec<i32>,    // capacity * MAX_PIECES, -1 padded
    targets: Vec<f32>, // capacity
}

#[no_mangle]
pub extern "C" fn batch_new(capacity: u32) -> *mut Batch {
    let cap = capacity as usize;
    let batch = Box::new(Batch {
        capacity: cap,
        len: 0,
        stm: vec![-1i32; cap * MAX_PIECES],
        nstm: vec![-1i32; cap * MAX_PIECES],
        targets: vec![0.0f32; cap],
    });
    Box::into_raw(batch)
}

#[no_mangle]
pub extern "C" fn batch_drop(batch: *mut Batch) {
    if !batch.is_null() {
        unsafe {
            drop(Box::from_raw(batch));
        }
    }
}

#[no_mangle]
pub extern "C" fn batch_get_len(batch: *const Batch) -> u32 {
    unsafe { (*batch).len as u32 }
}

#[no_mangle]
pub extern "C" fn batch_get_stm_ptr(batch: *const Batch) -> *const i32 {
    unsafe { (*batch).stm.as_ptr() }
}

#[no_mangle]
pub extern "C" fn batch_get_nstm_ptr(batch: *const Batch) -> *const i32 {
    unsafe { (*batch).nstm.as_ptr() }
}

#[no_mangle]
pub extern "C" fn batch_get_targets_ptr(batch: *const Batch) -> *const f32 {
    unsafe { (*batch).targets.as_ptr() }
}

/// Extract Board768 features from raw bulletformat bytes into the batch.
///
/// # Arguments
/// * `batch` - Pre-allocated batch buffer
/// * `data_ptr` - Pointer to N contiguous 32-byte ChessBoard records
/// * `count` - Number of records to process (capped at batch capacity)
/// * `blend` - WDL interpolation weight (0.0 = pure eval, 1.0 = pure WDL)
///
/// # Safety
/// `data_ptr` must point to at least `count * 32` valid bytes.
#[no_mangle]
pub extern "C" fn batch_fill(
    batch: *mut Batch,
    data_ptr: *const u8,
    count: u32,
    blend: f32,
) {
    let batch = unsafe { &mut *batch };
    let n = (count as usize).min(batch.capacity);
    let data: &[ChessBoard] =
        unsafe { std::slice::from_raw_parts(data_ptr as *const ChessBoard, n) };

    batch.len = n;

    // Reset stm/nstm to -1 (padding sentinel)
    // Only reset the portion we'll use
    let buf_len = n * MAX_PIECES;
    batch.stm[..buf_len].fill(-1);
    batch.nstm[..buf_len].fill(-1);

    for (i, pos) in data.iter().enumerate() {
        let base = i * MAX_PIECES;
        let occ = pos.occ;
        let mut remaining = occ;
        let mut piece_idx = 0usize;

        while remaining != 0 {
            let sq = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1; // clear LSB

            // Decode nibble-packed piece
            let nibble = if piece_idx % 2 == 0 {
                pos.pcs[piece_idx / 2] & 0x0F
            } else {
                pos.pcs[piece_idx / 2] >> 4
            };

            let color = ((nibble >> 3) & 1) as usize; // 0=STM, 1=opponent
            let piece_type = (nibble & 7) as usize;   // 0=pawn..5=king

            // STM: own pieces at 0..383, opponent at 384..767
            let stm_feat = color * 384 + piece_type * 64 + sq;
            // NSTM: flip own/opp AND flip squares vertically (XOR 56)
            let nstm_feat = (1 - color) * 384 + piece_type * 64 + (sq ^ 56);

            batch.stm[base + piece_idx] = stm_feat as i32;
            batch.nstm[base + piece_idx] = nstm_feat as i32;

            piece_idx += 1;
        }

        // Target: sigmoid(score/400) blended with WDL
        let score = pos.score as f32;
        let eval_target = 1.0 / (1.0 + (-score / 400.0).exp());
        let wdl_target = pos.result as f32 / 2.0;
        batch.targets[i] = eval_target * (1.0 - blend) + wdl_target * blend;
    }
}
