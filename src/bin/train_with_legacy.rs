//! Train SNN with Legacy Data Example
//!
//! Demonstrates how to use converted legacy data for SNN training.

use std::fs::File;
use std::io::{BufRead, BufReader};
use serde_json::Value;

#[derive(Debug, Clone)]
struct TrainingSample {
    stimuli: [f32; 16],
    expected_spikes: [f32; 16],
    reward_signal: Option<f32>,
}

fn main() -> anyhow::Result<()> {
    println!("🧠 Training SNN with Legacy Data");
    
    // Load converted training data
    let training_samples = load_training_data("DATA/research/snn_training_all.jsonl")?;
    println!("📚 Loaded {} training samples", training_samples.len());
    
    // Analyze data distribution
    analyze_training_data(&training_samples)?;
    
    // Simple training simulation
    simulate_training(&training_samples)?;
    
    println!("✅ Legacy data training demonstration complete!");
    Ok(())
}

fn load_training_data(file_path: &str) -> anyhow::Result<Vec<TrainingSample>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        if let Ok(json) = serde_json::from_str::<Value>(&line) {
            let stimuli: Vec<f32> = json["stimuli"].as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            
            let expected_spikes: Vec<f32> = json["expected_spikes"].as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            
            let reward_signal = json["metadata"]["reward_signal"].as_f64();
            
            // Convert to fixed-size arrays
            let mut stimuli_array = [0.0f32; 16];
            let mut spikes_array = [0.0f32; 16];
            
            for (i, val) in stimuli.iter().take(16).enumerate() {
                stimuli_array[i] = *val;
            }
            
            for (i, val) in expected_spikes.iter().take(16).enumerate() {
                spikes_array[i] = *val;
            }
            
            samples.push(TrainingSample {
                stimuli: stimuli_array,
                expected_spikes: spikes_array,
                reward_signal: reward_signal.map(|r| r as f32),
            });
        }
    }
    
    Ok(samples)
}

fn analyze_training_data(samples: &[TrainingSample]) -> anyhow::Result<()> {
    println!("\n📊 Training Data Analysis:");
    
    // Channel statistics
    for channel in 0..16 {
        let mut stimulus_sum = 0.0;
        let mut spike_sum = 0.0;
        let mut stimulus_count = 0;
        let mut spike_count = 0;
        
        for sample in samples {
            stimulus_sum += sample.stimuli[channel];
            spike_sum += sample.expected_spikes[channel];
            
            if sample.stimuli[channel] > 0.1 {
                stimulus_count += 1;
            }
            if sample.expected_spikes[channel] > 0.5 {
                spike_count += 1;
            }
        }
        
        let avg_stimulus = stimulus_sum / samples.len() as f32;
        let avg_spike = spike_sum / samples.len() as f32;
        let stimulus_activity = stimulus_count as f32 / samples.len() as f32;
        let spike_activity = spike_count as f32 / samples.len() as f32;
        
        println!("  Channel {}: Stimulus={:.3} (activity={:.1}%), Spikes={:.3} (activity={:.1}%)", 
            channel, avg_stimulus, stimulus_activity * 100.0, avg_spike, spike_activity * 100.0);
    }
    
    // Reward signal analysis
    let reward_samples: Vec<_> = samples.iter()
        .filter(|s| s.reward_signal.is_some())
        .collect();
    
    if !reward_samples.is_empty() {
        let avg_reward: f32 = reward_samples.iter()
            .map(|s| s.reward_signal.unwrap_or(0.0))
            .sum::<f32>() / reward_samples.len() as f32;
        
        println!("\n💰 Reward Analysis:");
        println!("  Samples with rewards: {}", reward_samples.len());
        println!("  Average reward: {:.3}", avg_reward);
    }
    
    Ok(())
}

fn simulate_training(samples: &[TrainingSample]) -> anyhow::Result<()> {
    println!("\n🎯 Simulating SNN Training:");
    
    // Simple neural network simulation
    let mut weights = [0.5f32; 16]; // Initial weights
    let learning_rate = 0.1;
    
    for epoch in 1..=5 {
        let mut total_error = 0.0;
        let mut correct_predictions = 0;
        
        for sample in samples {
            // Forward pass (simple weighted sum)
            let mut output = [0.0f32; 16];
            for i in 0..16 {
                output[i] = sample.stimuli[i] * weights[i];
            }
            
            // Calculate error
            let mut error = 0.0;
            for i in 0..16 {
                let expected = sample.expected_spikes[i];
                let actual = if output[i] > 0.5 { 1.0 } else { 0.0 };
                error += (expected - actual).powi(2);
                
                if (expected > 0.5) == (actual > 0.5) {
                    correct_predictions += 1;
                }
            }
            total_error += error;
            
            // Backward pass (simple weight update)
            for i in 0..16 {
                let target = sample.expected_spikes[i];
                let actual = output[i];
                let delta = (target - actual) * learning_rate;
                weights[i] += delta * sample.stimuli[i];
                
                // Apply reward signal if available
                if let Some(reward) = sample.reward_signal {
                    if reward > 0.0 && actual > 0.5 {
                        weights[i] += reward * learning_rate * 0.1;
                    }
                }
            }
        }
        
        let avg_error = total_error / (samples.len() * 16) as f32;
        let accuracy = correct_predictions as f32 / (samples.len() * 16) as f32;
        
        println!("  Epoch {}: Error={:.4}, Accuracy={:.4}", epoch, avg_error, accuracy);
    }
    
    println!("\n🔧 Final Weights:");
    for i in 0..16 {
        println!("  Channel {}: {:.3}", i, weights[i]);
    }
    
    Ok(())
}
