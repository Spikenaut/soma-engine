//! Legacy Data to SNN Training Data Converter
//!
//! Converts various legacy datasets into spiking neural network training format.
//! Supports mind telemetry, market data, tutor context, and other historical data.
//!
//! Usage: cargo run --bin legacy_data_converter [dataset_type]

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use serde_json::Value;
use serde::{Serialize, Deserialize};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Type of legacy data to convert
    #[arg(value_enum)]
    dataset_type: DatasetType,
    
    /// Output file for converted training data
    #[arg(short, long)]
    output: Option<String>,
    
    /// Number of samples to process (0 = all)
    #[arg(short, long, default_value = "0")]
    samples: usize,
}

#[derive(clap::ValueEnum, Debug, Clone)]
enum DatasetType {
    MindTelemetry,
    MarketData,
    TutorContext,
    EngineeringChemistry,
    All,
}

#[derive(Debug, Clone)]
struct SnnTrainingSample {
    timestamp: String,
    stimuli: [f32; 16],
    expected_spikes: [f32; 16],
    metadata: TrainingMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingMetadata {
    source: String,
    context: String,
    reward_signal: Option<f32>,
    error_magnitude: Option<f32>,
}

impl SnnTrainingSample {
    fn new(timestamp: String, stimuli: [f32; 16], expected_spikes: [f32; 16], metadata: TrainingMetadata) -> Self {
        Self {
            timestamp,
            stimuli,
            expected_spikes,
            metadata,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    match args.dataset_type {
        DatasetType::MindTelemetry => convert_mind_telemetry(&args)?,
        DatasetType::MarketData => convert_market_data(&args)?,
        DatasetType::TutorContext => convert_tutor_context(&args)?,
        DatasetType::EngineeringChemistry => convert_engineering_chemistry(&args)?,
        DatasetType::All => convert_all_datasets(&args)?,
    }
    
    Ok(())
}

fn convert_mind_telemetry(args: &Args) -> anyhow::Result<()> {
    println!("🧠 Converting Mind Telemetry data...");
    
    let file = File::open("DATA/research/mind_telemetry.jsonl")?;
    let reader = BufReader::new(file);
    
    let mut training_samples = Vec::new();
    let mut processed = 0;
    
    for line in reader.lines() {
        if args.samples > 0 && processed >= args.samples {
            break;
        }
        
        let line = line?;
        if let Ok(json) = serde_json::from_str::<Value>(&line) {
            let sample = convert_mind_sample(json)?;
            training_samples.push(sample);
            processed += 1;
        }
    }
    
    let output_path = args.output.as_deref().unwrap_or("DATA/research/snn_training_mind.jsonl");
    write_training_samples(training_samples, output_path)?;
    
    println!("✅ Converted {} mind telemetry samples to {}", processed, output_path);
    Ok(())
}

fn convert_market_data(args: &Args) -> anyhow::Result<()> {
    println!("📈 Converting Market Data...");
    
    let file = File::open("DATA/research/ghost_market_log.jsonl")?;
    let reader = BufReader::new(file);
    
    let mut training_samples = Vec::new();
    let mut processed = 0;
    
    for line in reader.lines() {
        if args.samples > 0 && processed >= args.samples {
            break;
        }
        
        let line = line?;
        if let Ok(json) = serde_json::from_str::<Value>(&line) {
            let sample = convert_market_sample(json)?;
            training_samples.push(sample);
            processed += 1;
        }
    }
    
    let output_path = args.output.as_deref().unwrap_or("DATA/research/snn_training_market.jsonl");
    write_training_samples(training_samples, output_path)?;
    
    println!("✅ Converted {} market data samples to {}", processed, output_path);
    Ok(())
}

fn convert_tutor_context(args: &Args) -> anyhow::Result<()> {
    println!("🎓 Converting Tutor Context data...");
    
    let file = File::open("DATA/research/tutor_context.jsonl")?;
    let reader = BufReader::new(file);
    
    let mut training_samples = Vec::new();
    let mut processed = 0;
    
    for line in reader.lines() {
        if args.samples > 0 && processed >= args.samples {
            break;
        }
        
        let line = line?;
        if let Ok(json) = serde_json::from_str::<Value>(&line) {
            let sample = convert_tutor_sample(json)?;
            training_samples.push(sample);
            processed += 1;
        }
    }
    
    let output_path = args.output.as_deref().unwrap_or("DATA/research/snn_training_tutor.jsonl");
    write_training_samples(training_samples, output_path)?;
    
    println!("✅ Converted {} tutor context samples to {}", processed, output_path);
    Ok(())
}

fn convert_engineering_chemistry(args: &Args) -> anyhow::Result<()> {
    println!("⚗️ Converting Engineering Chemistry data...");
    
    let file = File::open("DATA/research/Engineering Chemistry/tutor_context.jsonl")?;
    let reader = BufReader::new(file);
    
    let mut training_samples = Vec::new();
    let mut processed = 0;
    
    for line in reader.lines() {
        if args.samples > 0 && processed >= args.samples {
            break;
        }
        
        let line = line?;
        if let Ok(json) = serde_json::from_str::<Value>(&line) {
            let sample = convert_chemistry_sample(json)?;
            training_samples.push(sample);
            processed += 1;
        }
    }
    
    let output_path = args.output.as_deref().unwrap_or("DATA/research/snn_training_chemistry.jsonl");
    write_training_samples(training_samples, output_path)?;
    
    println!("✅ Converted {} chemistry samples to {}", processed, output_path);
    Ok(())
}

fn convert_all_datasets(args: &Args) -> anyhow::Result<()> {
    println!("🔄 Converting all legacy datasets...");
    
    let mut all_samples = Vec::new();
    
    // Convert each dataset
    all_samples.extend(convert_mind_samples_to_vec(args.samples));
    all_samples.extend(convert_market_samples_to_vec(args.samples));
    all_samples.extend(convert_tutor_samples_to_vec(args.samples));
    all_samples.extend(convert_chemistry_samples_to_vec(args.samples));
    
    // Sort by timestamp
    all_samples.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    
    let output_path = args.output.as_deref().unwrap_or("DATA/research/snn_training_all.jsonl");
    let sample_count = all_samples.len();
    write_training_samples(all_samples, output_path)?;
    
    println!("✅ Converted {} total samples to {}", sample_count, output_path);
    Ok(())
}

// Individual conversion functions
fn convert_mind_sample(json: Value) -> anyhow::Result<SnnTrainingSample> {
    let timestamp = json["timestamp"].as_str().unwrap_or("").to_string();
    let mood = json["mood"].as_str().unwrap_or("neutral");
    let focus_level = json["focus_level"].as_u64().unwrap_or(5) as f32;
    
    // Convert mood and focus to SNN stimuli
    let mut stimuli = [0.0f32; 16];
    
    // Channels 0-6: Emotional states
    match mood {
        "focused" => { stimuli[0] = 0.8; stimuli[1] = 0.2; }
        "relaxed" => { stimuli[0] = 0.2; stimuli[1] = 0.8; }
        "stressed" => { stimuli[0] = 0.1; stimuli[1] = 0.9; }
        _ => { stimuli[0] = 0.5; stimuli[1] = 0.5; }
    }
    
    // Channel 7: Focus level (normalized 0-1)
    stimuli[7] = focus_level / 10.0;
    
    // Expected spikes based on focus
    let mut expected_spikes = [0.0f32; 16];
    if focus_level > 7.0 {
        expected_spikes[0] = 1.0; // High focus -> spike
        expected_spikes[7] = 1.0;
    }
    
    let metadata = TrainingMetadata {
        source: "mind_telemetry".to_string(),
        context: format!("mood:{}, focus:{}", mood, focus_level),
        reward_signal: Some(focus_level / 10.0),
        error_magnitude: None,
    };
    
    Ok(SnnTrainingSample::new(timestamp, stimuli, expected_spikes, metadata))
}

fn convert_market_sample(json: Value) -> anyhow::Result<SnnTrainingSample> {
    let timestamp = json["timestamp"].as_str().unwrap_or("").to_string();
    let price_usd = json["price_usd"].as_f64().unwrap_or(0.0) as f32;
    let portfolio_value = json["portfolio_value"].as_f64().unwrap_or(0.0) as f32;
    let gpu_temp = json["gpu_temp_c"].as_f64().unwrap_or(0.0) as f32;
    let gpu_power = json["gpu_power_w"].as_f64().unwrap_or(0.0) as f32;
    
    let mut stimuli = [0.0f32; 16];
    
    // Channels 0-6: Market data (normalized)
    stimuli[0] = (price_usd / 100000.0).clamp(0.0, 1.0); // BTC price normalized
    stimuli[1] = (portfolio_value / 1000.0).clamp(0.0, 1.0); // Portfolio value
    
    // Channel 7: Hashrate
    stimuli[7] = json["dnx_hashrate_mh"].as_f64().unwrap_or(0.0) as f32 / 0.02;
    
    // Channels 12-15: GPU telemetry
    stimuli[12] = ((json["vddcr_gfx_v"].as_f64().unwrap_or(0.7) as f32 - 1.0).abs() * 2.0).clamp(0.0, 1.0);
    stimuli[13] = (gpu_power / 400.0).clamp(0.0, 1.0);
    stimuli[14] = ((gpu_temp - 40.0) / 40.0).clamp(0.0, 1.0);
    stimuli[15] = (json["fan_speed_pct"].as_f64().unwrap_or(0.0) as f32 / 100.0).clamp(0.0, 1.0);
    
    // Expected spikes for profitable conditions
    let mut expected_spikes = [0.0f32; 16];
    if portfolio_value > 500.0 {
        expected_spikes[0] = 1.0; // Profitable -> spike
    }
    if gpu_temp > 70.0 {
        expected_spikes[14] = 1.0; // High temp -> spike
    }
    
    let metadata = TrainingMetadata {
        source: "market_data".to_string(),
        context: format!("price:${:.2}, portfolio:${:.2}", price_usd, portfolio_value),
        reward_signal: Some((portfolio_value - 500.0) / 100.0),
        error_magnitude: None,
    };
    
    Ok(SnnTrainingSample::new(timestamp, stimuli, expected_spikes, metadata))
}

fn convert_tutor_sample(json: Value) -> anyhow::Result<SnnTrainingSample> {
    let timestamp = json["timestamp"].as_str().unwrap_or("").to_string();
    
    let mut stimuli = [0.0f32; 16];
    
    // Extract learning state
    if let Some(snn_state) = json.get("snn_learning_state") {
        let empty_vec = vec![];
        let predictions = snn_state["latest_predictions"].as_array().unwrap_or(&empty_vec);
        for (i, pred) in predictions.iter().take(8).enumerate() {
            stimuli[i] = pred.as_f64().unwrap_or(0.0) as f32;
        }
        
        // Reward signals
        if let Some(reward) = snn_state.get("reward_injected") {
            stimuli[8] = reward["dopamine_ltp"].as_f64().unwrap_or(0.0) as f32;
            stimuli[9] = reward["cortisol_ltd"].as_f64().unwrap_or(0.0) as f32;
            stimuli[10] = reward["error_magnitude"].as_f64().unwrap_or(0.0) as f32;
        }
    }
    
    // Expected spikes based on learning progress
    let mut expected_spikes = [0.0f32; 16];
    if stimuli[10] > 0.5 { // High error -> learning spike
        expected_spikes[8] = 1.0;
        expected_spikes[9] = 1.0;
    }
    
    let metadata = TrainingMetadata {
        source: "tutor_context".to_string(),
        context: "learning_state".to_string(),
        reward_signal: Some(stimuli[8]),
        error_magnitude: Some(stimuli[10]),
    };
    
    Ok(SnnTrainingSample::new(timestamp, stimuli, expected_spikes, metadata))
}

fn convert_chemistry_sample(json: Value) -> anyhow::Result<SnnTrainingSample> {
    let timestamp = json["timestamp"].as_str().unwrap_or("").to_string();
    
    let mut stimuli = [0.0f32; 16];
    
    // Extract experiment data
    if let Some(experiment) = json.get("experiment_data") {
        let state = &experiment["state"];
        
        // Normalize physical quantities
        stimuli[0] = (state["P_theoretical"].as_f64().unwrap_or(0.0) as f32 / 10.0).clamp(0.0, 1.0);
        stimuli[1] = (state["T"].as_f64().unwrap_or(0.0) as f32 / 500.0).clamp(0.0, 1.0);
        stimuli[2] = (state["V"].as_f64().unwrap_or(0.0) as f32 / 10.0).clamp(0.0, 1.0);
        stimuli[3] = (state["n"].as_f64().unwrap_or(0.0) as f32 / 2.0).clamp(0.0, 1.0);
    }
    
    // Extract SNN learning state
    if let Some(snn_state) = json.get("snn_learning_state") {
        let empty_vec = vec![];
        let predictions = snn_state["latest_predictions"].as_array().unwrap_or(&empty_vec);
        for (i, pred) in predictions.iter().take(4).enumerate() {
            stimuli[8 + i] = pred.as_f64().unwrap_or(0.0) as f32;
        }
        
        if let Some(reward) = snn_state.get("reward_injected") {
            stimuli[12] = reward["dopamine_ltp"].as_f64().unwrap_or(0.0) as f32;
            stimuli[13] = reward["cortisol_ltd"].as_f64().unwrap_or(0.0) as f32;
            stimuli[14] = reward["error_magnitude"].as_f64().unwrap_or(0.0) as f32;
        }
    }
    
    // Expected spikes for correct predictions
    let mut expected_spikes = [0.0f32; 16];
    if stimuli[14] < 0.1 { // Low error -> correct prediction
        expected_spikes[12] = 1.0; // Dopamine spike
    }
    
    let metadata = TrainingMetadata {
        source: "engineering_chemistry".to_string(),
        context: "ideal_gas_law_experiment".to_string(),
        reward_signal: Some(stimuli[12]),
        error_magnitude: Some(stimuli[14]),
    };
    
    Ok(SnnTrainingSample::new(timestamp, stimuli, expected_spikes, metadata))
}

// Helper functions for batch processing
fn convert_mind_samples_to_vec(limit: usize) -> Vec<SnnTrainingSample> {
    let file = File::open("DATA/research/mind_telemetry.jsonl").unwrap();
    let reader = BufReader::new(file);
    let mut samples = Vec::new();
    
    for (i, line) in reader.lines().enumerate() {
        if limit > 0 && i >= limit {
            break;
        }
        
        if let Ok(line) = line {
            if let Ok(json) = serde_json::from_str::<Value>(&line) {
                if let Ok(sample) = convert_mind_sample(json) {
                    samples.push(sample);
                }
            }
        }
    }
    samples
}

fn convert_market_samples_to_vec(limit: usize) -> Vec<SnnTrainingSample> {
    let file = File::open("DATA/research/ghost_market_log.jsonl").unwrap();
    let reader = BufReader::new(file);
    let mut samples = Vec::new();
    
    for (i, line) in reader.lines().enumerate() {
        if limit > 0 && i >= limit {
            break;
        }
        
        if let Ok(line) = line {
            if let Ok(json) = serde_json::from_str::<Value>(&line) {
                if let Ok(sample) = convert_market_sample(json) {
                    samples.push(sample);
                }
            }
        }
    }
    samples
}

fn convert_tutor_samples_to_vec(limit: usize) -> Vec<SnnTrainingSample> {
    let file = File::open("DATA/research/tutor_context.jsonl").unwrap();
    let reader = BufReader::new(file);
    let mut samples = Vec::new();
    
    for (i, line) in reader.lines().enumerate() {
        if limit > 0 && i >= limit {
            break;
        }
        
        if let Ok(line) = line {
            if let Ok(json) = serde_json::from_str::<Value>(&line) {
                if let Ok(sample) = convert_tutor_sample(json) {
                    samples.push(sample);
                }
            }
        }
    }
    samples
}

fn convert_chemistry_samples_to_vec(limit: usize) -> Vec<SnnTrainingSample> {
    let file = File::open("DATA/research/Engineering Chemistry/tutor_context.jsonl").unwrap();
    let reader = BufReader::new(file);
    let mut samples = Vec::new();
    
    for (i, line) in reader.lines().enumerate() {
        if limit > 0 && i >= limit {
            break;
        }
        
        if let Ok(line) = line {
            if let Ok(json) = serde_json::from_str::<Value>(&line) {
                if let Ok(sample) = convert_chemistry_sample(json) {
                    samples.push(sample);
                }
            }
        }
    }
    samples
}

fn write_training_samples(samples: Vec<SnnTrainingSample>, output_path: &str) -> anyhow::Result<()> {
    let mut file = File::create(output_path)?;
    
    for sample in samples {
        let json = serde_json::json!({
            "timestamp": sample.timestamp,
            "stimuli": sample.stimuli,
            "expected_spikes": sample.expected_spikes,
            "metadata": sample.metadata
        });
        
        writeln!(file, "{}", json)?;
    }
    
    Ok(())
}
