use bulletformat::ChessBoard;
use sfbinpack::CompressedTrainingDataEntryReader;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: binpack-to-bullet <input.binpack> <output.data> [--max-positions N] [--max-score N]"
        );
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    let mut max_positions: usize = usize::MAX;
    let mut max_score: i16 = 3000;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--max-positions" => {
                max_positions = args[i + 1].parse().expect("Invalid max-positions");
                i += 2;
            }
            "--max-score" => {
                max_score = args[i + 1].parse().expect("Invalid max-score");
                i += 2;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    eprintln!(
        "Converting {} -> {} (max_positions={}, max_score={})",
        input_path,
        output_path,
        if max_positions == usize::MAX {
            "unlimited".to_string()
        } else {
            max_positions.to_string()
        },
        max_score
    );

    let input = File::open(Path::new(input_path)).expect("Failed to open input file");
    let output = File::create(Path::new(output_path)).expect("Failed to create output file");
    let mut writer = BufWriter::with_capacity(1024 * 1024, output);

    let mut reader =
        CompressedTrainingDataEntryReader::new(input).expect("Failed to create binpack reader");
    let mut count: usize = 0;
    let mut skipped: usize = 0;

    while reader.has_next() {
        if count >= max_positions {
            break;
        }

        let entry = reader.next();

        // Filter extreme scores
        if entry.score.abs() > max_score {
            skipped += 1;
            continue;
        }

        // Filter positions in check
        let stm = entry.pos.side_to_move();
        if entry.pos.is_checked(stm) {
            skipped += 1;
            continue;
        }

        // Convert: build "FEN | score | wdl" string, parse into ChessBoard
        let fen = match entry.pos.fen() {
            Ok(f) => f,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        // Result: sfbinpack uses -1/0/1 (loss/draw/win STM-relative)
        // ChessBoard::from_str expects 0.0/0.5/1.0
        let wdl = match entry.result {
            -1 => "0.0",
            0 => "0.5",
            1 => "1.0",
            _ => "0.5",
        };

        let line = format!("{} | {} | {}", fen, entry.score, wdl);
        let board: ChessBoard = match line.parse() {
            Ok(b) => b,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        // Write raw 32 bytes
        let bytes: &[u8; 32] = unsafe { &*(std::ptr::from_ref(&board) as *const [u8; 32]) };
        writer.write_all(bytes).expect("Failed to write");

        count += 1;

        if count % 1_000_000 == 0 {
            eprintln!("  {} M positions written...", count / 1_000_000);
        }
    }

    writer.flush().expect("Failed to flush");
    eprintln!("Done: {} positions written, {} skipped", count, skipped);
}
