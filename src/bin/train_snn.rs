//! train_snn — Eagle-Lander SNN Offline Trainer
//!
//! Reads pre-collected telemetry samples (NeuromorphicSnapshot JSONL),
//! trains the dual-bank LIF/Izhikevich network via reward-modulated STDP
//! extended with E-prop eligibility traces and OTTT presynaptic trace
//! tracking, then exports parameters.mem for Basys3 FPGA deployment.
//!
//! # Algorithm summary
//!
//! Per tick:
//!   1. Encode telemetry → Poisson pre-spikes (per channel)
//!   2. Update OTTT trace:  â_j[t+1] = λ·â_j[t] + s_j[t+1]
//!   3. Forward pass via SpikingInferenceEngine::step() (incl. STDP)
//!   4. Update eligibility: e_{ij}[t+1] = λ·e_{ij}[t] + â_j[t]·z_i[t+1]
//!   5. Weight update:      Δw = reward · e_{ij} · η_eprop
//!   6. L1 synaptic scaling (budget = 1.0 per neuron)
//!
//! # Usage
//! ```
//! cargo run -p soma-engine --bin train_snn -- \
//!     --data research/neuromorphic_data.jsonl \
//!     --epochs 20 \
//!     --out research
//! ```

use soma_engine::learning_trainer::{EpochTrainer, NeuromorphicTrainer};
use soma_engine::market_trainer::MarketTrainer;
use soma_engine::snn::engine::NUM_INPUT_CHANNELS;
use clap::Parser;
use std::path::PathBuf;

/// Eagle-Lander — SNN Offline Trainer (E-prop + OTTT)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to JSONL telemetry dataset (NeuromorphicSnapshot records)
    /// Can be a single file OR a directory containing chunk files (*chunk*)
    #[arg(long, default_value = "/home/raulmc/Spikenaut-Vault/mining/processed/")]
    data: PathBuf,

    /// Number of training epochs over the dataset
    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// Output directory for FPGA .mem files and serialised model
    #[arg(long, default_value = "DATA/research")]
    out: PathBuf,

    /// Training mode: "mining" (hardware telemetry) or "market" (ghost trade log)
    #[arg(long, default_value = "mining")]
    mode: String,

    /// Also run the legacy single-pass STDP trainer and export comparison files
    #[arg(long, default_value_t = false)]
    legacy: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("--- SpikeLens: SNN Training Aperture ---");
    println!("Branding       : Hardware-Aware Deep Learning");
    println!("Model Identity : Spikenaut-v1 (The Navigator)");
    println!("Methodology    : E-prop + Online Temporal Trace (OTTT)");
    println!("Mode           : {}", args.mode);
    println!("Telemetry Path : {}", args.data.display());
    println!("Output  : {}", args.out.display());
    if args.data.is_dir() {
        println!("Data Mode : CHUNKED DIRECTORY (auto-loads *chunk* files)");
    } else {
        println!("Data Mode : SINGLE FILE");
    }

    // ── FPGA scaling analysis ─────────────────────────────────────────
    println!();
    print_fpga_analysis();

    // ── Load telemetry samples ────────────────────────────────────────
    if !args.data.exists() {
        eprintln!(
            "Error: dataset not found at '{}'.\n\
             Generate one with: cargo run --bin live_supervisor",
            args.data.display()
        );
        std::process::exit(1);
    }

    if args.mode == "market" {
        println!("--- Market-Spikenaut Trainer ---");
        let records = MarketTrainer::load_records(&args.data)?;
        let mut trainer = MarketTrainer::new();
        trainer.run_epochs(&records, args.epochs);
        std::fs::create_dir_all(&args.out)?;
        trainer.export(&args.out)?;
        return Ok(());
    }

    let samples = EpochTrainer::load_samples(&args.data)?;
    if samples.is_empty() {
        eprintln!(
            "Error: no valid telemetry samples found in '{}'.\n\
             Samples must have power_w > 0.",
            args.data.display()
        );
        std::process::exit(1);
    }
    println!("Loaded  : {} telemetry samples", samples.len());

    // ── E-prop multi-epoch training ───────────────────────────────────
    println!();
    println!("--- E-prop + OTTT Training ({} epochs) ---", args.epochs);
    let mut trainer = EpochTrainer::new();
    let metrics = trainer.run_epochs(&samples, args.epochs);

    // Print final summary
    if let Some(last) = metrics.last() {
        println!();
        println!("--- Final Epoch Summary ---");
        println!("  avg_reward  : {:.4}", last.avg_reward);
        println!("  spike_rate  : {:.3} spikes/neuron/tick", last.spike_rate);
        println!("  weight mean : {:.4}  std: {:.4}", last.mean_weight, last.std_weight);
        println!("  throughput  : {:.3} ms/tick", last.ms_per_tick);
    }

    // Print learned weight matrix
    println!();
    println!("--- Learned Weight Matrix [neuron × channel] ---");
    let channel_names = ["VDDCR", "Power", "Hash"];
    for (i, neuron) in trainer.engine.neurons.iter().enumerate() {
        let weights_str: Vec<String> = neuron
            .weights
            .iter()
            .take(NUM_INPUT_CHANNELS)
            .enumerate()
            .map(|(ch, w)| {
                format!(
                    "{}:{:.4}",
                    channel_names.get(ch).unwrap_or(&"?"),
                    w
                )
            })
            .collect();
        println!(
            "  N{:>2} thresh={:.4}  [{}]",
            i,
            neuron.threshold,
            weights_str.join(", ")
        );
    }

    // ── Export FPGA parameters ────────────────────────────────────────
    println!();
    println!("--- Exporting FPGA Parameters to {}/ ---", args.out.display());
    std::fs::create_dir_all(&args.out)?;

    trainer.export_fpga(&args.out)?;
    println!(
        "  parameters.mem         — {} threshold values (Q8.8)",
        trainer.engine.neurons.len()
    );
    println!(
        "  parameters_weights.mem — {} weight values (Q8.8, row-major [N×CH])",
        trainer.engine.neurons.len() * NUM_INPUT_CHANNELS
    );
    println!(
        "  parameters_decay.mem   — {} decay-rate values (Q8.8)",
        trainer.engine.neurons.len()
    );

    // Serialise full neuron state as JSON (for inspection / reload)
    trainer.engine.save_parameters(args.out.join("snn_model.json"))?;
    println!("  snn_model.json         — serialised neuron state");

    // Export simulation verification data (stimulus + expected spikes)
    // Re-uses the legacy exporter which replays the dataset through a fresh engine.
    {
        let stim_path = args.out.join("stimulus.mem");
        let exp_path  = args.out.join("expected.mem");
        let mut legacy_for_sim = NeuromorphicTrainer::new();
        if legacy_for_sim
            .export_simulation_data(
                &args.data,
                stim_path.to_str().unwrap_or("DATA/research/stimulus.mem"),
                exp_path.to_str().unwrap_or("DATA/research/expected.mem"),
            )
            .is_ok()
        {
            println!("  stimulus.mem           — packed stimulus vectors (Vcore/Power/Hash)");
            println!("  expected.mem           — expected spike bitmask per step");
        }
    }

    // ── Legacy STDP trainer (optional comparison) ─────────────────────
    if args.legacy {
        println!();
        println!("--- Legacy STDP Trainer (single-pass, for comparison) ---");
        let mut legacy = NeuromorphicTrainer::new();
        let summary = legacy.run_training_session(&args.data)?;
        println!(
            "  Steps: {}  Spikes: {}",
            summary.steps_processed, summary.total_spikes
        );
        legacy.export_to_verilog(args.out.join("parameters_legacy.mem"))?;
        println!("  parameters_legacy.mem  — legacy STDP thresholds (Q8.8)");
    }

    println!();
    println!("SUCCESS: SNN trained and parameters exported for FPGA deployment.");
    Ok(())
}

// ── FPGA scaling analysis ─────────────────────────────────────────────────

fn print_fpga_analysis() {
    println!("--- FPGA Resource Analysis (Artix-7 XC7A35T / Basys3) ---");

    // Values from fpga-project/ship_ssn_logic.runs/impl_1/Basys3_Top_utilization_placed.rpt
    // Tool: Vivado 2025.2, Date: Fri Mar 6 14:06:03 2026, Design: Basys3_Top
    const TOTAL_LUTS:      u32 = 20_800;
    const TOTAL_FFS:       u32 = 41_600;
    const TOTAL_DSPS:      u32 = 90;
    const USED_LUTS:       u32 = 1_074;  // 5.16%
    const USED_FFS:        u32 = 1_091;  // 2.62%
    const CURRENT_NEURONS: u32 = 8;
    const CURRENT_CHANNELS: u32 = 3;

    // Peripheral LUT estimate (UART, 7-seg display, top-level glue ≈ 500 LUTs)
    const PERIPHERAL_LUTS: u32 = 500;
    let snn_luts        = USED_LUTS.saturating_sub(PERIPHERAL_LUTS);
    let luts_per_neuron = (snn_luts / CURRENT_NEURONS).max(1);
    let target_luts     = (TOTAL_LUTS as f32 * 0.80) as u32;
    let target_dsps     = (TOTAL_DSPS as f32 * 0.80) as u32; // 72

    // LUT-limited count (reserve PERIPHERAL_LUTS for top-level logic)
    let lut_limited = (target_luts.saturating_sub(PERIPHERAL_LUTS)) / luts_per_neuron;

    // DSP-limited count: each neuron needs CHANNELS multiplies.
    // With 2-cycle pipeline sharing, effective DSP demand halves.
    let dsp_limited = (target_dsps * 2) / CURRENT_CHANNELS;

    // Conservative recommendation: min of both limits, rounded to a power of 2
    let raw_rec  = lut_limited.min(dsp_limited);
    let recommended = if raw_rec >= 32 { 32 } else if raw_rec >= 16 { 16 } else { 8 };

    println!(
        "  Current (8N): LUTs {}/{} ({:.1}%)  FFs {}/{} ({:.1}%)",
        USED_LUTS, TOTAL_LUTS,
        USED_LUTS as f32 / TOTAL_LUTS as f32 * 100.0,
        USED_FFS,  TOTAL_FFS,
        USED_FFS  as f32 / TOTAL_FFS  as f32 * 100.0,
    );
    println!(
        "  SNN LUT estimate: {} LUTs / {} neurons ≈ {} LUTs/neuron",
        snn_luts, CURRENT_NEURONS, luts_per_neuron
    );
    println!(
        "  80% targets: {} LUTs  {} DSPs (72 available)",
        target_luts, target_dsps
    );
    println!(
        "  Scaling limits: LUT-limited ~{}N  DSP-limited (2-cycle) ~{}N",
        lut_limited, dsp_limited
    );
    println!(
        "  RECOMMENDATION: {} neurons, {} channels, Q8.8 fixed-point",
        recommended, CURRENT_CHANNELS
    );
    println!(
        "    Weight matrix: {}×{} = {} synapses × 2 bytes = {} bytes",
        recommended, CURRENT_CHANNELS,
        recommended * CURRENT_CHANNELS,
        recommended * CURRENT_CHANNELS * 2,
    );
    println!(
        "    Threshold file: {} × 2 bytes = {} bytes",
        recommended, recommended * 2
    );

    // WNS from Vivado timing report
    match read_wns() {
        Some(wns) if wns < 0.0 => println!(
            "  WNS: {:.3} ns  *** TIMING VIOLATION — reduce neurons or add pipeline stages ***",
            wns
        ),
        Some(wns) => println!("  WNS: +{:.3} ns  (timing closure OK)", wns),
        None => println!("  WNS: (timing report not found — run Vivado impl first)"),
    }
    println!();
}

/// Parse WNS from the Vivado timing summary report.
fn read_wns() -> Option<f32> {
    let text = std::fs::read_to_string(
        "fpga-project/ship_ssn_logic.runs/impl_1/Basys3_Top_timing_summary_routed.rpt",
    )
    .ok()?;
    let mut found_header = false;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("WNS(ns)") {
            found_header = true;
            continue;
        }
        if found_header && !t.is_empty() {
            return t.split_whitespace().next()?.parse().ok();
        }
    }
    None
}
