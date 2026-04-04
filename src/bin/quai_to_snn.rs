//! quai_to_snn — Quai Sync Node Data → NeuromorphicSnapshot Converter
//!
//! Reads Quai node sync telemetry JSONL and converts it to the
//! `NeuromorphicSnapshot` format expected by `train_snn`.
//!
//! The Quai telemetry has 20 fields that map directly to `GpuTelemetry`.
//! Missing fields (kaspa_*, monero_*, solver_*, etc.) default to 0.0 via serde.
//!
//! # Usage
//! ```bash
//! # Convert full dataset to single file
//! cargo run --bin quai_to_snn -- \
//!     --input "/home/raulmc/Spikenaut Sync node data/node_sync_harvest.jsonl" \
//!     --out "/home/raulmc/Spikenaut-Vault/mining/processed/quai_sync.jsonl"
//!
//! # Convert with sample limit for testing
//! cargo run --bin quai_to_snn -- \
//!     --input "/home/raulmc/Spikenaut Sync node data/node_sync_harvest.jsonl" \
//!     --out "/home/raulmc/Spikenaut-Vault/mining/processed/quai_sync.jsonl" \
//!     --limit 1000
//!
//! # Convert chunked (splits into multiple chunk files for train_snn)
//! cargo run --bin quai_to_snn -- \
//!     --input "/home/raulmc/Spikenaut Sync node data/node_sync_harvest.jsonl" \
//!     --out "/home/raulmc/Spikenaut-Vault/mining/processed/" \
//!     --chunk-size 30000
//! ```

use clap::Parser;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

/// Quai Sync Telemetry — raw format from node_sync_harvest.jsonl
#[derive(Debug, Clone, Deserialize)]
struct QuaiSyncRecord {
    #[allow(dead_code)]
    timestamp: String,
    telemetry: QuaiTelemetry,
}

/// Telemetry fields from Quai node sync (20 fields)
/// Maps directly to GpuTelemetry fields used by soma-engine.
#[derive(Debug, Clone, Deserialize)]
struct QuaiTelemetry {
    hashrate_mh: f32,
    power_w: f32,
    gpu_temp_c: f32,
    qubic_tick_trace: f32,
    qubic_epoch_progress: f32,
    reward_hint: f32,
    vddcr_gfx_v: f32,
    vram_temp_c: f32,
    gpu_clock_mhz: f32,
    mem_clock_mhz: f32,
    fan_speed_pct: f32,
    rejected_shares: u32,
    mem_util_pct: f32,
    #[serde(default)]
    ocean_intel: f32,
    #[serde(default)]
    power_z_score: f32,
    #[serde(default)]
    temp_z_score: f32,
    #[serde(default)]
    clock_z_score: f32,
    #[serde(default)]
    clock_mhz: f32,
    #[serde(default)]
    qubic_tick_rate: f32,
    #[serde(default)]
    qu_price_usd: f32,
}

/// Minimal GpuTelemetry-compatible struct for serialization.
/// All soma-engine GpuTelemetry fields present; Quai data fills core fields.
#[derive(Debug, Clone, serde::Serialize)]
struct GpuTelemetryCompat {
    vddcr_gfx_v: f32,
    vram_temp_c: f32,
    gpu_temp_c: f32,
    hashrate_mh: f32,
    power_w: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    solver_steps: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    solver_chips: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    complexity: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    joules_per_step: Option<f32>,
    gpu_clock_mhz: f32,
    mem_clock_mhz: f32,
    fan_speed_pct: f32,
    rejected_shares: u32,
    mem_util_pct: f32,
    #[serde(default)]
    ocean_intel: f32,
    #[serde(default)]
    kaspa_hashrate_mh: f32,
    #[serde(default)]
    kaspa_power_w: f32,
    #[serde(default)]
    kaspa_gpu_temp_c: f32,
    #[serde(default)]
    monero_hashrate_h: f32,
    #[serde(default)]
    monero_power_w: f32,
    #[serde(default)]
    monero_cpu_temp_c: f32,
    #[serde(default)]
    power_z_score: f32,
    #[serde(default)]
    temp_z_score: f32,
    #[serde(default)]
    clock_z_score: f32,
    clock_mhz: f32,
    #[serde(default)]
    qubic_tick_trace: f32,
    #[serde(default)]
    qubic_tick_rate: f32,
    #[serde(default)]
    qubic_epoch_progress: f32,
    #[serde(default)]
    qu_price_usd: f32,
}

impl From<QuaiTelemetry> for GpuTelemetryCompat {
    fn from(q: QuaiTelemetry) -> Self {
        Self {
            vddcr_gfx_v: q.vddcr_gfx_v,
            vram_temp_c: q.vram_temp_c,
            gpu_temp_c: q.gpu_temp_c,
            hashrate_mh: q.hashrate_mh,
            power_w: q.power_w,
            solver_steps: None,
            solver_chips: None,
            complexity: None,
            joules_per_step: None,
            gpu_clock_mhz: q.gpu_clock_mhz,
            mem_clock_mhz: q.mem_clock_mhz,
            fan_speed_pct: q.fan_speed_pct,
            rejected_shares: q.rejected_shares,
            mem_util_pct: q.mem_util_pct,
            ocean_intel: q.ocean_intel,
            kaspa_hashrate_mh: 0.0,
            kaspa_power_w: 0.0,
            kaspa_gpu_temp_c: 0.0,
            monero_hashrate_h: 0.0,
            monero_power_w: 0.0,
            monero_cpu_temp_c: 0.0,
            power_z_score: q.power_z_score,
            temp_z_score: q.temp_z_score,
            clock_z_score: q.clock_z_score,
            clock_mhz: q.clock_mhz,
            qubic_tick_trace: q.qubic_tick_trace,
            qubic_tick_rate: q.qubic_tick_rate,
            qubic_epoch_progress: q.qubic_epoch_progress,
            qu_price_usd: q.qu_price_usd,
        }
    }
}

/// NeuromorphicSnapshot — the format train_snn expects.
/// Wraps GpuTelemetry in the same structure soma_engine::ai::researcher uses.
#[derive(Debug, Clone, serde::Serialize)]
struct NeuromorphicSnapshot {
    telemetry: GpuTelemetryCompat,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Convert Quai sync data to SNN training format")]
struct Args {
    /// Path to Quai sync JSONL input file
    #[arg(long)]
    input: PathBuf,

    /// Output path: either a single JSONL file OR a directory for chunked output
    #[arg(long)]
    out: PathBuf,

    /// Max records to process (0 = all)
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Chunk size for directory output mode (records per chunk)
    #[arg(long, default_value_t = 30000)]
    chunk_size: usize,

    /// Print stats only, don't write output
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════");
    println!("  Quai → SNN Data Converter");
    println!("═══════════════════════════════════════════════════════");
    println!("  Input     : {}", args.input.display());
    println!("  Output    : {}", args.out.display());
    println!("  Limit     : {}", if args.limit > 0 { args.limit.to_string() } else { "all".to_string() });
    println!("  Chunk size: {}", args.chunk_size);
    println!("  Dry run   : {}", args.dry_run);
    println!("═══════════════════════════════════════════════════════");

    if !args.input.exists() {
        eprintln!("\n❌ Input file not found: {}", args.input.display());
        std::process::exit(1);
    }

    let file = File::open(&args.input)?;
    let reader = BufReader::new(file);

    let is_chunked = args.out.is_dir() || args.out.to_string_lossy().ends_with('/');
    let mut total_converted: usize = 0;
    let mut total_skipped: usize = 0;
    let mut chunk_idx: usize = 0;
    let mut chunk_records: usize = 0;
    let mut chunk_writer: Option<BufWriter<File>> = None;

    // If chunked mode, create output directory
    if is_chunked {
        std::fs::create_dir_all(&args.out)?;
    }

    for (line_num, line) in reader.lines().enumerate() {
        if args.limit > 0 && total_converted >= args.limit {
            break;
        }

        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[WARN] Line {} read error: {}", line_num + 1, e);
                total_skipped += 1;
                continue;
            }
        };

        let record: QuaiSyncRecord = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[WARN] Line {} parse error: {}", line_num + 1, e);
                total_skipped += 1;
                continue;
            }
        };

        // Filter out records with zero power (idle/corrupt)
        if record.telemetry.power_w <= 0.0 {
            total_skipped += 1;
            continue;
        }

        let snapshot = NeuromorphicSnapshot {
            telemetry: record.telemetry.into(),
        };

        let json_line = match serde_json::to_string(&snapshot) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("[WARN] Line {} serialize error: {}", line_num + 1, e);
                total_skipped += 1;
                continue;
            }
        };

        if args.dry_run {
            total_converted += 1;
            continue;
        }

        if is_chunked {
            // Chunked output mode
            if chunk_writer.is_none() || chunk_records >= args.chunk_size {
                // Close previous chunk
                if let Some(mut w) = chunk_writer.take() {
                    w.flush()?;
                }

                // Open new chunk
                chunk_idx += 1;
                let chunk_name = format!("quai_sync_chunk_{:02}.jsonl", chunk_idx);
                let chunk_path = args.out.join(&chunk_name);
                let f = File::create(&chunk_path)?;
                chunk_writer = Some(BufWriter::new(f));
                chunk_records = 0;
                eprintln!("  📦 Writing {} ...", chunk_name);
            }

            if let Some(ref mut w) = chunk_writer {
                writeln!(w, "{}", json_line)?;
                chunk_records += 1;
            }
        } else {
            // Single file output
            if chunk_writer.is_none() {
                if let Some(parent) = args.out.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let f = File::create(&args.out)?;
                chunk_writer = Some(BufWriter::new(f));
            }

            if let Some(ref mut w) = chunk_writer {
                writeln!(w, "{}", json_line)?;
            }
        }

        total_converted += 1;
    }

    // Flush final chunk
    if let Some(mut w) = chunk_writer.take() {
        w.flush()?;
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Conversion Complete");
    println!("═══════════════════════════════════════════════════════");
    println!("  ✅ Converted : {} records", total_converted);
    println!("  ⏭️  Skipped  : {} records", total_skipped);
    if is_chunked {
        println!("  📦 Chunks   : {} files", chunk_idx);
        println!("  📁 Output   : {}", args.out.display());
    } else {
        println!("  📄 Output   : {}", args.out.display());
    }
    println!("═══════════════════════════════════════════════════════");

    if total_converted == 0 {
        eprintln!("\n❌ No valid records converted. Check input format.");
        std::process::exit(1);
    }

    let data_path = if is_chunked {
        args.out.display().to_string()
    } else if let Some(parent) = args.out.parent() {
        parent.display().to_string()
    } else {
        args.out.display().to_string()
    };

    println!("\nNext step: train_snn");
    println!(
        "  cargo run -p soma-engine --bin train_snn -- \\\n    --data {} \\\n    --epochs 20 \\\n    --out Vault/models/quai_v1",
        data_path
    );

    Ok(())
}
