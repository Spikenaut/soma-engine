//! Market-Spikenaut Offline Trainer
//!
//! Replays `research/ghost_market_log.jsonl` through the 16-neuron SNN with
//! directional price-prediction rewards for all 7 mining chains.
//!
//! # Usage
//! ```
//! # Default: 20 epochs, auto-export to research/
//! cargo run -p soma-engine --bin train_market
//!
//! # Custom: 50 epochs, explicit data path
//! cargo run -p soma-engine --bin train_market -- --epochs 50 --data research/ghost_market_log.jsonl
//!
//! # Resume from a previous checkpoint
//! cargo run -p soma-engine --bin train_market -- --resume research/snn_model_market.json
//! ```

use std::path::Path;
use soma_engine::market_trainer::MarketTrainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // Parse flags
    let data_path = args.iter().position(|a| a == "--data")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("DATA/research/ghost_market_log.jsonl");

    let epochs: usize = args.iter().position(|a| a == "--epochs")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let resume_path = args.iter().position(|a| a == "--resume")
        .and_then(|i| args.get(i + 1));

    let out_dir = args.iter().position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("research");

    // Validate input
    if !Path::new(data_path).exists() {
        eprintln!("Error: {} not found.", data_path);
        eprintln!("Run `cargo run -p soma-engine --bin market_pilot` first to collect data.");
        std::process::exit(1);
    }

    // Load records
    println!("[train_market] Loading records from {}...", data_path);
    let records = MarketTrainer::load_records(data_path)?;
    println!("[train_market] Loaded {} ticks", records.len());

    if records.len() < 100 {
        eprintln!("[train_market] WARNING: Very few records ({}). Need 500+ for meaningful training.", records.len());
    }

    // Initialize trainer
    let mut trainer = MarketTrainer::new();

    // Resume from checkpoint if requested
    if let Some(checkpoint) = resume_path {
        if Path::new(checkpoint).exists() {
            trainer.engine.load_parameters(checkpoint)?;
            println!("[train_market] Resumed from checkpoint: {}", checkpoint);
        } else {
            eprintln!("[train_market] Checkpoint not found: {} — starting fresh", checkpoint);
        }
    }

    // Run training
    println!("[train_market] Starting {} epochs over {} ticks...", epochs, records.len());
    println!("─────────────────────────────────────────────────────────────");
    let metrics = trainer.run_epochs(&records, epochs);
    println!("─────────────────────────────────────────────────────────────");

    // Final summary
    if let Some(last) = metrics.last() {
        println!("\n[train_market] Final accuracy: {:.1}% | Mean weight: {:.4} | Reward: {:+.4}",
            last.accuracy * 100.0, last.mean_weight, last.avg_reward);
    }

    // Export
    println!("\n[train_market] Exporting model to {}/...", out_dir);
    trainer.export(out_dir)?;
    // trainer.export_training_log(&metrics, out_dir)?;

    println!("[train_market] Outputs:");
    println!("  - {}/snn_model_market.json     (neuron weights + thresholds)", out_dir);
    println!("  - {}/parameters.mem            (FPGA thresholds, Q8.8)", out_dir);
    println!("  - {}/parameters_weights.mem    (FPGA weight matrix, Q8.8)", out_dir);
    println!("  - {}/parameters_decay.mem      (FPGA decay rates, Q8.8)", out_dir);
    println!("  - {}/training_log.jsonl        (per-epoch metrics w/ chain breakdown)", out_dir);
    println!("\n[train_market] Done. Load into FPGA via: source research/load_brain.tcl");

    Ok(())
}
