//! Neuromod Sensory Encoder - Spikenaut-v2 Poisson Encoding
//!
//! Replaces basic telemetry mapping with neuromodulator-driven Poisson sensory encoding.
//! Integrates all 7 neuromodulators for adaptive firing rate modulation.

use crate::snn::modulators::NeuroModulators;
use crate::telemetry::gpu_telemetry::GpuTelemetry;

/// Neuromod Sensory Encoder for Spikenaut-v2
/// 
/// Provides 16-channel Poisson encoding driven by neuromodulator state.
/// Replaces basic telemetry mapping with adaptive sensory processing.
pub struct NeuromodSensoryEncoder {
    neuromodulators: NeuroModulators,
    channel_gains: [f32; 16],
    channel_biases: [f32; 16],
    adaptation_rates: [f32; 16],
}

impl NeuromodSensoryEncoder {
    pub fn new() -> Self {
        Self {
            neuromodulators: NeuroModulators::default(),
            channel_gains: [1.0; 16],
            channel_biases: [0.0; 16],
            adaptation_rates: [0.01; 16],
        }
    }

    /// Update neuromodulator state from telemetry
    pub fn update_neuromodulators(&mut self, telemetry: &GpuTelemetry) {
        self.neuromodulators = NeuroModulators::from_telemetry(telemetry);
    }

    /// Set market volatility (injected by market_pilot)
    pub fn set_market_volatility(&mut self, volatility: f32) {
        self.neuromodulators.market_volatility = volatility.clamp(0.0, 1.0);
    }

    /// Set mining dopamine (injected by mining supervisor)
    pub fn set_mining_dopamine(&mut self, mining_dopamine: f32) {
        self.neuromodulators.mining_dopamine = mining_dopamine.clamp(-0.8, 0.8);
    }

    /// Set FPGA stress (injected by hardware monitor)
    pub fn set_fpga_stress(&mut self, fpga_stress: f32) {
        self.neuromodulators.fpga_stress = fpga_stress.clamp(0.0, 1.0);
    }

    /// Encode 8-channel market data into 16-channel Poisson stimuli
    /// 
    /// # Arguments
    /// * `market_inputs` - 8-channel normalized market data (Z-scores)
    /// * `hardware_inputs` - 8-channel hardware telemetry (optional)
    /// 
    /// # Returns
    /// * `[f32; 16]` - 16-channel Poisson firing rates (0.0-1.0)
    pub fn encode_poisson_stimuli(&mut self, market_inputs: &[f32; 8]) -> [f32; 16] {
        let mut stimuli = [0.0f32; 16];

        // Neuromodulator modulation factors
        let learning_rate = self.compute_learning_rate_modulation();
        let stress_inhibition = self.compute_stress_inhibition();
        let focus_gain = self.compute_focus_gain();
        let tempo_modulation = self.compute_tempo_modulation();

        // Encode each market channel into bear/bull pair
        for i in 0..8 {
            let market_value = market_inputs[i];
            
            // Apply neuromodulator modulation
            let modulated_value = market_value 
                * learning_rate 
                * focus_gain 
                * tempo_modulation
                - stress_inhibition;

            // Poisson encoding with neuromodulator-driven firing rates
            let (bear_rate, bull_rate) = self.poisson_encode_channel(modulated_value, i);
            
            stimuli[i * 2] = bear_rate;     // Bear channel (negative signals)
            stimuli[i * 2 + 1] = bull_rate; // Bull channel (positive signals)
        }

        // Update adaptation based on recent activity
        self.update_adaptation(&stimuli);

        stimuli
    }

    /// Compute learning rate modulation from dopamine systems
    fn compute_learning_rate_modulation(&self) -> f32 {
        // Combine reward dopamine and mining dopamine
        let total_dopamine = (self.neuromodulators.dopamine + self.neuromodulators.mining_dopamine.max(0.0)).clamp(0.0, 1.0);
        
        // Learning rate scales with dopamine (0.1 to 2.0)
        0.1 + total_dopamine * 1.9
    }

    /// Compute stress inhibition from cortisol and market volatility
    fn compute_stress_inhibition(&self) -> f32 {
        // Combine hardware stress and market stress
        let total_stress = (self.neuromodulators.cortisol + self.neuromodulators.market_volatility).clamp(0.0, 1.0);
        
        // Stress inhibition reduces firing rates (0.0 to 0.8)
        total_stress * 0.8
    }

    /// Compute focus gain from acetylcholine
    fn compute_focus_gain(&self) -> f32 {
        // Acetylcholine enhances signal-to-noise ratio (0.5 to 1.5)
        0.5 + self.neuromodulators.acetylcholine * 1.0
    }

    /// Compute tempo modulation from clock speed and FPGA stress
    fn compute_tempo_modulation(&self) -> f32 {
        // Tempo affects temporal resolution (0.7 to 1.3)
        let base_tempo = self.neuromodulators.tempo.clamp(0.7, 1.3);
        
        // FPGA stress reduces tempo (timing violations)
        let stress_penalty = 1.0 - self.neuromodulators.fpga_stress * 0.3;
        
        base_tempo * stress_penalty
    }

    /// Poisson encoding for a single channel
    fn poisson_encode_channel(&self, value: f32, channel_idx: usize) -> (f32, f32) {
        let gain = self.channel_gains[channel_idx];
        let bias = self.channel_biases[channel_idx];
        
        // Apply channel-specific gain and bias
        let scaled_value = (value * gain + bias).clamp(-3.0, 3.0);
        
        if scaled_value >= 0.0 {
            // Positive signal -> bull channel active
            let bull_rate = (scaled_value / 3.0).clamp(0.0, 1.0);
            let bear_rate = 0.01; // Minimal background activity
            (bear_rate, bull_rate)
        } else {
            // Negative signal -> bear channel active
            let bear_rate = (-scaled_value / 3.0).clamp(0.0, 1.0);
            let bull_rate = 0.01; // Minimal background activity
            (bear_rate, bull_rate)
        }
    }

    /// Update channel adaptation based on recent activity
    fn update_adaptation(&mut self, stimuli: &[f32; 16]) {
        for i in 0..16 {
            let rate = self.adaptation_rates[i];
            let activity = stimuli[i];
            
            // Homeostatic plasticity: maintain optimal firing rates
            let target_rate = 0.1; // Target average firing rate
            let error = activity - target_rate;
            
            // Adjust gain to maintain target activity
            self.channel_gains[i] -= rate * error * 0.1;
            self.channel_gains[i] = self.channel_gains[i].clamp(0.1, 3.0);
            
            // Adjust bias to prevent systematic drift
            if activity < 0.05 {
                self.channel_biases[i] += rate * 0.01;
            } else if activity > 0.5 {
                self.channel_biases[i] -= rate * 0.01;
            }
            self.channel_biases[i] = self.channel_biases[i].clamp(-0.5, 0.5);
        }
    }

    /// Get current neuromodulator state
    pub fn get_neuromodulators(&self) -> &NeuroModulators {
        &self.neuromodulators
    }

    /// Get channel statistics for debugging
    pub fn get_channel_stats(&self) -> Vec<ChannelStats> {
        (0..16).map(|i| ChannelStats {
            channel: i,
            gain: self.channel_gains[i],
            bias: self.channel_biases[i],
            adaptation_rate: self.adaptation_rates[i],
        }).collect()
    }

    /// Reset encoder to initial state
    pub fn reset(&mut self) {
        self.neuromodulators = NeuroModulators::default();
        self.channel_gains = [1.0; 16];
        self.channel_biases = [0.0; 16];
        self.adaptation_rates = [0.01; 16];
    }
}

impl Default for NeuromodSensoryEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Channel statistics for debugging and monitoring
#[derive(Debug, Clone)]
pub struct ChannelStats {
    pub channel: usize,
    pub gain: f32,
    pub bias: f32,
    pub adaptation_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromod_encoder_basic() {
        let mut encoder = NeuromodSensoryEncoder::new();
        
        // Test with simple market inputs
        let market_inputs = [0.5, -0.3, 0.0, 0.8, -0.2, 0.1, -0.7, 0.4];
        let stimuli = encoder.encode_poisson_stimuli(&market_inputs);
        
        // Verify output is 16 channels
        assert_eq!(stimuli.len(), 16);
        
        // Verify rates are in valid range
        for rate in &stimuli {
            assert!(*rate >= 0.0 && *rate <= 1.0);
        }
        
        // Verify bear/bull pairing
        for i in 0..8 {
            let bear_idx = i * 2;
            let bull_idx = i * 2 + 1;
            
            if market_inputs[i] > 0.0 {
                assert!(stimuli[bull_idx] > stimuli[bear_idx]);
            } else if market_inputs[i] < 0.0 {
                assert!(stimuli[bear_idx] > stimuli[bull_idx]);
            }
        }
    }

    #[test]
    fn test_neuromodulator_modulation() {
        let mut encoder = NeuromodSensoryEncoder::new();
        
        // Set high dopamine (should increase learning rate)
        encoder.neuromodulators.dopamine = 0.9;
        encoder.neuromodulators.mining_dopamine = 0.5;
        
        let learning_rate = encoder.compute_learning_rate_modulation();
        assert!(learning_rate > 1.0);
        
        // Set high cortisol (should increase stress inhibition)
        encoder.neuromodulators.cortisol = 0.8;
        encoder.neuromodulators.market_volatility = 0.6;
        
        let stress_inhibition = encoder.compute_stress_inhibition();
        assert!(stress_inhibition > 0.5);
        
        // Set high acetylcholine (should increase focus)
        encoder.neuromodulators.acetylcholine = 0.9;
        
        let focus_gain = encoder.compute_focus_gain();
        assert!(focus_gain > 1.0);
    }
}
