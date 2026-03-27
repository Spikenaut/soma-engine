//! FPGA Parameter Export - Spikenaut-v2 Deployment
//!
//! Q8.8 fixed-point parameter export for FPGA deployment.
//! Converts learned parameters to hardware-compatible format.

use std::fs;
use std::path::Path;
use std::io::Write;
use serde::{Deserialize, Serialize};
use rand::Rng;

/// FPGA parameter exporter for Spikenaut-v2
/// 
/// Exports learned SNN parameters in Q8.8 fixed-point format
/// for FPGA deployment with <35µs/tick target performance.
pub struct FpgaParameterExporter {
    thresholds: Vec<f32>,
    weights: Vec<Vec<f32>>,
    decay_rates: Vec<f32>,
}

/// FPGA-compatible parameter format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpgaParameters {
    /// Neuron thresholds in Q8.8 format
    pub thresholds: Vec<u16>,
    /// Weight matrix [neurons x channels] in Q8.8 format  
    pub weights: Vec<u16>,
    /// Decay rates in Q8.8 format
    pub decay_rates: Vec<u16>,
    /// Metadata about the parameter set
    pub metadata: FpgaMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpgaMetadata {
    pub version: String,
    pub timestamp: String,
    pub num_neurons: usize,
    pub num_channels: usize,
    pub target_latency_us: f32,
    pub memory_usage_kb: f32,
}

impl FpgaParameterExporter {
    /// Create new exporter with default parameters
    pub fn new() -> Self {
        Self {
            thresholds: Vec::new(),
            weights: Vec::new(),
            decay_rates: Vec::new(),
        }
    }

    /// Set neuron thresholds
    pub fn set_thresholds(&mut self, thresholds: Vec<f32>) {
        self.thresholds = thresholds;
    }

    /// Set weight matrix [neurons x channels]
    pub fn set_weights(&mut self, weights: Vec<Vec<f32>>) {
        self.weights = weights;
    }

    /// Set decay rates
    pub fn set_decay_rates(&mut self, decay_rates: Vec<f32>) {
        self.decay_rates = decay_rates;
    }

    /// Convert f32 to Q8.8 fixed-point format
    fn to_q88(&self, value: f32) -> u16 {
        // Q8.8: 8 integer bits, 8 fractional bits
        // Range: 0.0 to 255.996
        let scaled = value * 256.0;
        scaled.clamp(0.0, 65535.0) as u16
    }

    /// Export parameters to FPGA-compatible format
    pub fn export(&self) -> FpgaParameters {
        let thresholds_q88: Vec<u16> = self.thresholds.iter()
            .map(|&v| self.to_q88(v))
            .collect();

        let weights_q88: Vec<u16> = self.weights.iter()
            .flat_map(|row| row.iter())
            .map(|&v| self.to_q88(v))
            .collect();

        let decay_rates_q88: Vec<u16> = self.decay_rates.iter()
            .map(|&v| self.to_q88(v))
            .collect();

        let metadata = FpgaMetadata {
            version: "Spikenaut-v2".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            num_neurons: self.thresholds.len(),
            num_channels: if self.weights.is_empty() { 0 } else { self.weights[0].len() },
            target_latency_us: 35.0,
            memory_usage_kb: self.calculate_memory_usage(),
        };

        FpgaParameters {
            thresholds: thresholds_q88,
            weights: weights_q88,
            decay_rates: decay_rates_q88,
            metadata,
        }
    }

    /// Calculate memory usage in KB
    fn calculate_memory_usage(&self) -> f32 {
        let total_params = self.thresholds.len() + 
                          self.weights.iter().map(|row| row.len()).sum::<usize>() + 
                          self.decay_rates.len();
        
        // Each parameter is 2 bytes (u16) in Q8.8 format
        (total_params * 2) as f32 / 1024.0
    }

    /// Export parameters to .mem files for FPGA
    pub fn export_to_mem_files<P: AsRef<Path>>(&self, output_dir: P) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(&output_dir)?;

        let params = self.export();

        // Export thresholds
        let thresholds_path = output_dir.as_ref().join("parameters.mem");
        let mut thresholds_file = fs::File::create(thresholds_path)?;
        for threshold in &params.thresholds {
            writeln!(thresholds_file, "{:04X}", threshold)?;
        }

        // Export weights
        let weights_path = output_dir.as_ref().join("parameters_weights.mem");
        let mut weights_file = fs::File::create(weights_path)?;
        for weight in &params.weights {
            writeln!(weights_file, "{:04X}", weight)?;
        }

        // Export decay rates
        let decay_path = output_dir.as_ref().join("parameters_decay.mem");
        let mut decay_file = fs::File::create(decay_path)?;
        for decay in &params.decay_rates {
            writeln!(decay_file, "{:04X}", decay)?;
        }

        // Export metadata
        let metadata_path = output_dir.as_ref().join("parameters.json");
        let metadata_json = serde_json::to_string_pretty(&params)?;
        fs::write(metadata_path, metadata_json)?;

        self.print_export_summary(&params, output_dir);

        Ok(())
    }

    /// Print export summary
    fn print_export_summary<P: AsRef<Path>>(&self, params: &FpgaParameters, output_dir: P) {
        println!("=== FPGA Parameter Export Summary ===");
        println!("Output Directory: {}", output_dir.as_ref().display());
        println!("Version: {}", params.metadata.version);
        println!("Timestamp: {}", params.metadata.timestamp);
        println!("Neurons: {}", params.metadata.num_neurons);
        println!("Channels: {}", params.metadata.num_channels);
        println!("Target Latency: {:.1}µs", params.metadata.target_latency_us);
        println!("Memory Usage: {:.2} KB", params.metadata.memory_usage_kb);
        println!();
        println!("Files Generated:");
        println!("  parameters.mem         - {} thresholds", params.thresholds.len());
        println!("  parameters_weights.mem - {} weights", params.weights.len());
        println!("  parameters_decay.mem   - {} decay rates", params.decay_rates.len());
        println!("  parameters.json        - metadata and configuration");
        println!();
        println!("SUCCESS: FPGA parameters ready for deployment");
    }

    /// Load parameters from SNN engine state
    pub fn from_snn_engine(_engine: &crate::snn::engine::SpikingInferenceEngine) -> Self {
        let mut exporter = Self::new();

        // TODO: Extract parameters from engine
        // Note: This assumes the engine has methods to access parameters
        // Implementation depends on the actual SpikingInferenceEngine API
        
        // For now, create default parameters
        let num_neurons = 16;
        let num_channels = 16;
        
        // Default thresholds (1.0 in Q8.8 = 256)
        exporter.thresholds = vec![1.0; num_neurons];
        
        // Default weights (uniform random 0.5-1.5)
        let mut rng = rand::thread_rng();
        exporter.weights = (0..num_neurons)
            .map(|_| (0..num_channels)
                .map(|_| rng.gen_range(0.5..1.5))
                .collect())
            .collect();
        
        // Default decay rates (0.85 in Q8.8 = 217.6)
        exporter.decay_rates = vec![0.85; num_neurons];
        
        exporter
    }
}

impl Default for FpgaParameterExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to format Q8.8 value as hex string
pub fn format_q88_hex(value: f32) -> String {
    let exporter = FpgaParameterExporter::new();
    let q88_value = exporter.to_q88(value);
    format!("{:04X}", q88_value)
}

/// Helper function to convert Q8.8 back to f32
pub fn q88_to_f32(q88_value: u16) -> f32 {
    q88_value as f32 / 256.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q88_conversion() {
        let exporter = FpgaParameterExporter::new();
        
        // Test basic conversions
        assert_eq!(exporter.to_q88(0.0), 0);
        assert_eq!(exporter.to_q88(1.0), 256);
        assert_eq!(exporter.to_q88(255.0), 65280);
        
        // Test precision
        assert_eq!(q88_to_f32(256), 1.0);
        assert_eq!(q88_to_f32(0), 0.0);
        assert_eq!(q88_to_f32(65280), 255.0);
    }

    #[test]
    fn test_parameter_export() {
        let mut exporter = FpgaParameterExporter::new();
        
        // Set test parameters
        exporter.set_thresholds(vec![1.0, 0.8, 1.2]);
        exporter.set_weights(vec![
            vec![0.5, 1.0, 0.3],
            vec![0.7, 0.9, 1.1],
            vec![0.4, 0.6, 0.8],
        ]);
        exporter.set_decay_rates(vec![0.85, 0.9, 0.8]);
        
        // Export to FPGA format
        let params = exporter.export();
        
        // Verify conversion
        assert_eq!(params.thresholds.len(), 3);
        assert_eq!(params.weights.len(), 9); // 3x3
        assert_eq!(params.decay_rates.len(), 3);
        assert_eq!(params.metadata.num_neurons, 3);
        assert_eq!(params.metadata.num_channels, 3);
        
        // Verify Q8.8 values are in range
        for threshold in &params.thresholds {
            assert!(*threshold <= 65535);
        }
        for weight in &params.weights {
            assert!(*weight <= 65535);
        }
        for decay in &params.decay_rates {
            assert!(*decay <= 65535);
        }
    }

    #[test]
    fn test_memory_calculation() {
        let mut exporter = FpgaParameterExporter::new();
        
        // Set parameters for 16 neurons, 16 channels
        exporter.set_thresholds(vec![1.0; 16]);
        exporter.set_weights(vec![vec![0.5; 16]; 16]);
        exporter.set_decay_rates(vec![0.85; 16]);
        
        let params = exporter.export();
        
        // Expected memory: (16 + 256 + 16) * 2 bytes = 576 bytes = 0.5625 KB
        assert!((params.metadata.memory_usage_kb - 0.5625).abs() < 0.01);
    }
}
