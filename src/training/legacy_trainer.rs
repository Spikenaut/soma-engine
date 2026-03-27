//! Legacy Data Trainer - SNN Training with Historical Data
//!
//! Uses converted legacy data to train spiking neural networks through
//! supervised learning and reinforcement learning approaches.

use std::fs::File;
use std::io::{BufRead, BufReader};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use crate::snn::SpikingNetwork;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyTrainingSample {
    pub timestamp: String,
    pub stimuli: [f32; 16],
    pub expected_spikes: [f32; 16],
    pub metadata: TrainingMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    pub source: String,
    pub context: String,
    pub reward_signal: Option<f32>,
    pub error_magnitude: Option<f32>,
}

pub struct LegacyDataTrainer {
    pub samples: Vec<LegacyTrainingSample>,
    pub network: SpikingNetwork,
    pub learning_rate: f32,
    pub batch_size: usize,
}

impl LegacyDataTrainer {
    pub fn new(learning_rate: f32, batch_size: usize) -> Self {
        Self {
            samples: Vec::new(),
            network: SpikingNetwork::new(),
            learning_rate,
            batch_size,
        }
    }
    
    /// Load training data from converted legacy files
    pub fn load_data(&mut self, file_path: &str) -> anyhow::Result<usize> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut loaded = 0;
        
        for line in reader.lines() {
            let line = line?;
            if let Ok(sample) = serde_json::from_str::<LegacyTrainingSample>(&line) {
                self.samples.push(sample);
                loaded += 1;
            }
        }
        
        println!("📚 Loaded {} training samples from {}", loaded, file_path);
        Ok(loaded)
    }
    
    /// Train the SNN using supervised learning with expected spikes
    pub fn train_supervised(&mut self, epochs: usize) -> anyhow::Result<TrainingMetrics> {
        println!("🎓 Starting supervised SNN training...");
        
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let mut samples_processed = 0;
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut epoch_accuracy = 0.0;
            
            // Shuffle samples for each epoch
            let mut indices: Vec<usize> = (0..self.samples.len()).collect();
            indices.shuffle(&mut rand::thread_rng());
            
            for batch_start in (0..self.samples.len()).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(self.samples.len());
                let batch_indices = &indices[batch_start..batch_end];
                
                let (batch_loss, batch_accuracy) = self.train_supervised_batch(batch_indices)?;
                epoch_loss += batch_loss;
                epoch_accuracy += batch_accuracy;
            }
            
            epoch_loss /= self.samples.len() as f32;
            epoch_accuracy /= self.samples.len() as f32;
            
            total_loss += epoch_loss;
            total_accuracy += epoch_accuracy;
            samples_processed += self.samples.len();
            
            if epoch % 10 == 0 {
                println!("Epoch {}: Loss={:.4}, Accuracy={:.4}", epoch, epoch_loss, epoch_accuracy);
            }
        }
        
        let metrics = TrainingMetrics {
            total_loss: total_loss / epochs as f32,
            accuracy: total_accuracy / epochs as f32,
            samples_processed,
            epochs,
        };
        
        println!("✅ Training completed: Loss={:.4}, Accuracy={:.4}", metrics.total_loss, metrics.accuracy);
        Ok(metrics)
    }
    
    /// Train the SNN using reinforcement learning with reward signals
    pub fn train_reinforcement(&mut self, episodes: usize) -> anyhow::Result<TrainingMetrics> {
        println!("🎮 Starting reinforcement learning training...");
        
        let mut total_reward = 0.0;
        let mut episodes_processed = 0;
        
        for episode in 0..episodes {
            let mut episode_reward = 0.0;
            let mut steps = 0;
            
            for sample in &self.samples {
                // Process stimuli through network
                let actual_spikes = self.network.process_stimuli(&sample.stimuli);
                
                // Calculate reward based on metadata
                let reward = sample.metadata.reward_signal.unwrap_or(0.0);
                
                // Apply STDP learning based on reward
                if reward > 0.0 {
                    self.apply_reward_learning(&sample.stimuli, &actual_spikes, reward)?;
                }
                
                episode_reward += reward;
                steps += 1;
            }
            
            total_reward += episode_reward / steps as f32;
            episodes_processed += 1;
            
            if episode % 10 == 0 {
                println!("Episode {}: Avg Reward={:.4}", episode, episode_reward / steps as f32);
            }
        }
        
        let metrics = TrainingMetrics {
            total_loss: -total_reward / episodes_processed as f32, // Negative reward as loss
            accuracy: (total_reward / episodes_processed as f32).clamp(0.0, 1.0),
            samples_processed: self.samples.len() * episodes_processed,
            epochs: episodes,
        };
        
        println!("✅ RL Training completed: Avg Reward={:.4}", -metrics.total_loss);
        Ok(metrics)
    }
    
    /// Train on specific data source for domain adaptation
    pub fn train_domain_specific(&mut self, domain: &str, epochs: usize) -> anyhow::Result<TrainingMetrics> {
        println!("🎯 Training on domain: {}", domain);
        
        // Filter samples by domain
        let domain_samples: Vec<_> = self.samples.iter()
            .filter(|s| s.metadata.source == domain)
            .collect();
        
        if domain_samples.is_empty() {
            println!("⚠️ No samples found for domain: {}", domain);
            return Ok(TrainingMetrics::default());
        }
        
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut epoch_accuracy = 0.0;
            
            for sample in &domain_samples {
                let actual_spikes = self.network.process_stimuli(&sample.stimuli);
                let (loss, accuracy) = self.calculate_supervised_metrics(&actual_spikes, &sample.expected_spikes);
                
                // Apply weight updates
                self.apply_supervised_learning(&sample.stimuli, &actual_spikes, &sample.expected_spikes, loss);
                
                epoch_loss += loss;
                epoch_accuracy += accuracy;
            }
            
            epoch_loss /= domain_samples.len() as f32;
            epoch_accuracy /= domain_samples.len() as f32;
            
            total_loss += epoch_loss;
            total_accuracy += epoch_accuracy;
            
            if epoch % 10 == 0 {
                println!("Epoch {} ({}): Loss={:.4}, Accuracy={:.4}", epoch, domain, epoch_loss, epoch_accuracy);
            }
        }
        
        let metrics = TrainingMetrics {
            total_loss: total_loss / epochs as f32,
            accuracy: total_accuracy / epochs as f32,
            samples_processed: domain_samples.len() * epochs,
            epochs,
        };
        
        println!("✅ Domain training completed: Loss={:.4}, Accuracy={:.4}", metrics.total_loss, metrics.accuracy);
        Ok(metrics)
    }
    
    /// Evaluate the trained network on test data
    pub fn evaluate(&self, test_samples: &[LegacyTrainingSample]) -> anyhow::Result<EvaluationMetrics> {
        println!("📊 Evaluating trained network...");
        
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let mut domain_performance = std::collections::HashMap::new();
        
        for sample in test_samples {
            let actual_spikes = self.network.process_stimuli(&sample.stimuli);
            let (loss, accuracy) = self.calculate_supervised_metrics(&actual_spikes, &sample.expected_spikes);
            
            total_loss += loss;
            total_accuracy += accuracy;
            
            // Track performance by domain
            let domain = &sample.metadata.source;
            let entry = domain_performance.entry(domain.clone()).or_insert((0.0, 0.0));
            entry.0 += accuracy;
            entry.1 += 1.0;
        }
        
        // Calculate domain averages
        for (_, (acc, count)) in domain_performance.iter_mut() {
            *acc /= *count;
        }
        
        let metrics = EvaluationMetrics {
            average_loss: total_loss / test_samples.len() as f32,
            average_accuracy: total_accuracy / test_samples.len() as f32,
            domain_performance,
            total_samples: test_samples.len(),
        };
        
        println!("📈 Evaluation Results:");
        println!("  Average Loss: {:.4}", metrics.average_loss);
        println!("  Average Accuracy: {:.4}", metrics.average_accuracy);
        for (domain, (accuracy, _)) in &metrics.domain_performance {
            println!("  {}: {:.4}", domain, accuracy);
        }
        
        Ok(metrics)
    }
    
    // Private helper methods
    fn train_supervised_batch(&mut self, batch_indices: &[usize]) -> anyhow::Result<(f32, f32)> {
        let mut batch_loss = 0.0;
        let mut batch_accuracy = 0.0;
        
        for &idx in batch_indices {
            let sample = &self.samples[idx];
            let actual_spikes = self.network.process_stimuli(&sample.stimuli);
            let (loss, accuracy) = self.calculate_supervised_metrics(&actual_spikes, &sample.expected_spikes);
            
            // Apply weight updates
            self.apply_supervised_learning(&sample.stimuli, &actual_spikes, &sample.expected_spikes, loss);
            
            batch_loss += loss;
            batch_accuracy += accuracy;
        }
        
        Ok((batch_loss / batch_indices.len() as f32, batch_accuracy / batch_indices.len() as f32))
    }
    
    fn calculate_supervised_metrics(&self, actual: &[f32], expected: &[f32]) -> (f32, f32) {
        let mut loss = 0.0;
        let mut correct = 0;
        
        for (a, e) in actual.iter().zip(expected.iter()) {
            // Mean squared error for loss
            loss += (a - e).powi(2);
            
            // Binary accuracy (spike or no spike)
            let predicted_spike = *a > 0.5;
            let expected_spike = *e > 0.5;
            if predicted_spike == expected_spike {
                correct += 1;
            }
        }
        
        loss /= actual.len() as f32;
        let accuracy = correct as f32 / actual.len() as f32;
        
        (loss, accuracy)
    }
    
    fn apply_supervised_learning(&mut self, stimuli: &[f32], actual: &[f32], expected: &[f32], loss: f32) {
        // Simplified supervised learning - adjust weights based on error
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let error = e - a;
            if error.abs() > 0.1 {
                // Adjust synaptic weights for neurons that should have spiked
                if *e > 0.5 && *a <= 0.5 {
                    self.network.increase_synaptic_strength(i, self.learning_rate * error);
                }
                // Decrease strength for neurons that shouldn't have spiked
                else if *e <= 0.5 && *a > 0.5 {
                    self.network.decrease_synaptic_strength(i, self.learning_rate * error.abs());
                }
            }
        }
    }
    
    fn apply_reward_learning(&mut self, stimuli: &[f32], spikes: &[f32], reward: f32) -> anyhow::Result<()> {
        // Reinforcement learning - strengthen pathways that led to reward
        for (i, spike) in spikes.iter().enumerate() {
            if *spike > 0.5 {
                self.network.increase_synaptic_strength(i, self.learning_rate * reward);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub total_loss: f32,
    pub accuracy: f32,
    pub samples_processed: usize,
    pub epochs: usize,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            total_loss: 0.0,
            accuracy: 0.0,
            samples_processed: 0,
            epochs: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub average_loss: f32,
    pub average_accuracy: f32,
    pub domain_performance: std::collections::HashMap<String, (f32, f32)>,
    pub total_samples: usize,
}
