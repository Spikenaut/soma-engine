use rand::Rng;
use serde::{Deserialize, Serialize};

use super::lif::LifNeuron;
use super::izhikevich::IzhikevichNeuron;
use super::stdp::*;
use super::modulators::NeuroModulators;
use super::mining_reward::MiningRewardState;
use crate::spine::fpga_readback::FpgaBridge;

/// L1 synaptic weight budget per neuron (total weight sum target).
const WEIGHT_BUDGET: f32 = 2.0;  // Increased budget to allow stronger synapses

/// Number of input channels feeding each LIF neuron.
/// CH0-7: Asset deltas and Telemetry.
pub const NUM_INPUT_CHANNELS: usize = 16;

/// FPGA synthesis and implementation metrics parsed from Vivado reports.
///
/// Parsed from `Basys3_Top_timing_summary_routed.rpt` in ship_ssn_logic/runs/impl_1/.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FpgaMetrics {
    /// Worst Negative Slack in nanoseconds.
    /// Negative value = timing violation. Positive = margin.
    pub wns_ns: f32,
    /// LUT resource utilization (0.0–1.0)
    pub lut_utilization: f32,
    /// `true` if the last synthesis/implementation run completed without errors
    pub synthesis_ok: bool,
}

impl FpgaMetrics {
    /// Parse the WNS from a Vivado timing summary report text.
    ///
    /// Looks for the `WNS(ns)` column header row and extracts the first value.
    /// Returns `None` if the file format is not recognized.
    pub fn parse_from_report(report_text: &str) -> Option<f32> {
        // The Vivado timing summary has a line like:
        // "  WNS(ns)      TNS(ns)  ..."
        // followed by a data row with the actual values.
        let mut found_header = false;
        for line in report_text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("WNS(ns)") {
                found_header = true;
                continue;
            }
            if found_header && !trimmed.is_empty() {
                // First token of the data row is WNS
                if let Some(wns_str) = trimmed.split_whitespace().next() {
                    return wns_str.parse::<f32>().ok();
                }
                break;
            }
        }
        None
    }

    /// Attempt to load metrics from the canonical implementation report path.
    pub fn load_from_project() -> Option<Self> {
        let report_path = "fpga-project/ship_ssn_logic.runs/impl_1/Basys3_Top_timing_summary_routed.rpt";
        let text = std::fs::read_to_string(report_path).ok()?;
        let wns = Self::parse_from_report(&text)?;
        Some(Self {
            wns_ns: wns,
            lut_utilization: 0.0, // future enhancement
            synthesis_ok: true,
        })
    }
}

#[derive(Default)]
pub struct SpikingInferenceEngine {
    // Bank 1: LIF Neurons (Fast, Reactive)
    pub neurons: Vec<LifNeuron>,
    // Bank 2: Izhikevich Neurons (Complex, Adaptive)
    pub iz_neurons: Vec<IzhikevichNeuron>,
    // Global Neuromodulators
    pub modulators: NeuroModulators,
    /// Global step counter for STDP timing
    pub global_step: i64,
    /// Pre-synaptic spike times for each input channel (for STDP)
    pub input_spike_times: Vec<i64>,
    /// Most recently loaded FPGA metrics (optional — None if no FPGA project)
    pub fpga_metrics: Option<FpgaMetrics>,
    /// FPGA bridge for hardware spike readback
    pub fpga_bridge: Option<FpgaBridge>,
    /// Whether to use FPGA spikes instead of software simulation
    pub use_fpga: bool,
    /// Per-channel exponential moving average of input stimuli.
    ///
    /// Updated every `step()` call. Used for Predictive Error Coding:
    /// the surprise signal (`|actual − predicted|`) is blended into the
    /// synaptic current so neurons respond proactively to abrupt changes
    /// (voltage sags, hash crashes) rather than purely reactively.
    pub predictive_state: [f32; NUM_INPUT_CHANNELS],
    /// Mining-efficiency reward computer (EMA-smoothed, zero-alloc).
    ///
    /// Computes the `mining_dopamine` signal from hardware telemetry each
    /// tick.  The signal gates STDP learning alongside the event-driven
    /// `dopamine` field — positive values validate neural patterns,
    /// negative values suppress plasticity during thermal/power stress.
    pub mining_reward: MiningRewardState,
    /// Ring buffer of recent trading-neuron spike events for coincidence detection (N14).
    /// Each entry: (neuron_idx, global_step). Evicted when older than COINCIDENCE_WINDOW_STEPS.
    coincidence_window: Vec<(usize, i64)>,
}

impl SpikingInferenceEngine {
    pub fn new() -> Self {
        // Try to initialize FPGA bridge
        let fpga_bridge = match FpgaBridge::new() {
            Ok(bridge) => {
                println!("[engine] FPGA bridge initialized - hardware neurons active");
                Some(bridge)
            }
            Err(e) => {
                println!("[engine] FPGA not available: {} - using software neurons", e);
                None
            }
        };

        let use_fpga = fpga_bridge.is_some();

        // Create neurons with initial weights
        let mut neurons: Vec<LifNeuron> = (0..16).map(|_| {
            let mut n = LifNeuron::new();
            n.weights = vec![1.5; NUM_INPUT_CHANNELS]; // Aggressively strong synapses
            n.last_spike_time = -1;
            n
        }).collect();

        let mut rng = rand::thread_rng();

        // 7 trading pairs (N0-N13) + N14 (coincidence detector) + N15 (global inhibitor)
        // N0/1: DNX, N2/3: QUAI, N4/5: QUBIC, N6/7: KAS, N8/9: XMR, N10/11: OCEAN, N12/13: VERUS
        for i in 0..14 {
            let ch = i / 2;
            let neuron = &mut neurons[i];

            // Set primary channel weight (Q8.8 parity: 1.0)
            neuron.weights[ch] = 0.8 + (rng.gen::<f32>() * 0.4);

            // Differentiated thresholds
            if i % 2 == 0 {
                // Bear neurons: conservative threshold
                neuron.threshold = 0.10 + (rng.gen::<f32>() * 0.04);
            } else {
                // Bull neurons: sensitive threshold
                neuron.threshold = 0.06 + (rng.gen::<f32>() * 0.04);
            }
            neuron.base_threshold = neuron.threshold;
        }
        // N14: Coincidence Detector — fires when ≥3 chains spike within temporal window
        neurons[14].threshold = 0.50;
        neurons[14].base_threshold = 0.50;
        // N15: Global Inhibitory Interneuron — driven by N14, not by external stimuli
        neurons[15].threshold = 0.50;
        neurons[15].base_threshold = 0.50;

        Self {
            neurons,
            // 5 Izhikevich neurons: 4 hardware channels + IZ[4] for FPGA timing stress
            iz_neurons: vec![IzhikevichNeuron::new_regular_spiking(); 5],
            // Initialize modulators (baseline)
            modulators: NeuroModulators::default(),
            global_step: 0,
            input_spike_times: vec![-1; NUM_INPUT_CHANNELS],
            fpga_metrics: FpgaMetrics::load_from_project(),
            fpga_bridge,
            use_fpga,
            predictive_state: [0.0; NUM_INPUT_CHANNELS],
            mining_reward: MiningRewardState::new(),
            coincidence_window: Vec::new(),
        }
    }

    /// HEARTBEAT (LSM mode): Steps the readout layer using signals from the Julia reservoir.
    pub fn step_with_lsm(&mut self, lsm_currents: &[f32; 16], telem: &crate::telemetry::gpu_telemetry::GpuTelemetry) {
        self.global_step += 1;

        // 0. Update Neurochemistry (Still reacts to telemetry)
        let saved_market_vol = self.modulators.market_volatility;
        let saved_mining_da = self.modulators.mining_dopamine;
        self.modulators = NeuroModulators::from_telemetry(telem);
        self.modulators.market_volatility = saved_market_vol;
        self.modulators.mining_dopamine = saved_mining_da;

        let combined_stress = (self.modulators.cortisol + saved_market_vol * 0.3).min(1.0);
        let stress_multiplier = (1.0 - combined_stress).max(0.1);

        // Blend mining-efficiency signal into STDP learning rate (same as step()).
        let mining_contrib = (self.modulators.mining_dopamine + 0.8) / 1.6;
        let blended_da = (self.modulators.dopamine * 0.6 + mining_contrib * 0.4).clamp(0.0, 1.0);
        let learning_rate = 0.5 * blended_da;

        // Apply modulation to LIF neurons
        for neuron in &mut self.neurons {
            let target_decay = 0.15 - (0.05 * self.modulators.acetylcholine);
            neuron.decay_rate = target_decay;

            let global_target = 0.20 - (0.05 * self.modulators.dopamine) + (0.15 * self.modulators.cortisol);
            let local_adjustment = if neuron.last_spike { 0.005 } else { -0.001 };
            let target_threshold = (global_target + local_adjustment).clamp(0.05, 0.50);
            neuron.threshold += (target_threshold - neuron.threshold) * learning_rate;
            neuron.threshold = neuron.threshold.clamp(0.05, 0.50);
        }

        // 1. Integrate LSM currents (The Readout Layer)
        let mut spike_ids: Vec<usize> = Vec::new();
        let step = self.global_step;
        if self.use_fpga {
            // Send stimuli to FPGA and read back spikes
            if let Some(ref mut bridge) = self.fpga_bridge {
                match bridge.process_stimuli(lsm_currents) {
                    Ok((potentials, spikes)) => {
                        // Update software neurons with FPGA potentials for logging
                        for (i, &potential) in potentials.iter().enumerate() {
                            self.neurons[i].membrane_potential = potential;
                        }
                        // Set spike flags from FPGA
                        for (i, &spiked) in spikes.iter().enumerate() {
                            self.neurons[i].last_spike = spiked;
                            if spiked {
                                self.neurons[i].last_spike_time = step;
                                spike_ids.push(i);
                            }
                        }
                        if !spike_ids.is_empty() {
                            println!("[fpga] Hardware spikes detected: {:?}", spike_ids);
                        }
                    }
                    Err(e) => {
                        println!("[fpga] Communication error: {} - falling back to software", e);
                        self.use_fpga = false; // Disable FPGA on error
                        // Continue with software simulation
                        for i in 0..16 {
                            self.neurons[i].integrate(lsm_currents[i] * stress_multiplier);
                        }
                    }
                }
            }
        } else {
            // Software simulation
            for i in 0..16 {
                self.neurons[i].integrate(lsm_currents[i] * stress_multiplier);
            }
        }

        // 2. Evaluation: Pulsing the actual Action Potentials
        // Skip if FPGA already provided spikes
        if !self.use_fpga {
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                if let Some(_) = neuron.check_fire() {
                    neuron.last_spike = true;
                    neuron.last_spike_time = step;
                    spike_ids.push(i);
                } else {
                    neuron.last_spike = false;
                }
            }
        }

        // Log neuron activity for debugging
        if self.global_step % 10 == 0 {
            let mut active_neurons = Vec::new();
            for (i, neuron) in self.neurons.iter().enumerate() {
                if neuron.membrane_potential > 0.01 {
                    active_neurons.push((i, neuron.membrane_potential, neuron.threshold));
                }
            }
            if !active_neurons.is_empty() {
                let mode = if self.use_fpga { "FPGA" } else { "Software" };
                println!("[neurons-{}] Active: {:?}", mode, active_neurons);
            }
        }

        // 3. Lateral & Competitive Inhibition
        if !spike_ids.is_empty() {
            const INHIBITION_STRENGTH: f32 = 0.05;
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                if !spike_ids.contains(&i) {
                    neuron.membrane_potential = (neuron.membrane_potential - INHIBITION_STRENGTH).max(0.0);
                }
            }

            const COMPETITIVE_INHIBITION: f32 = 0.15;
            for pair in 0..8 {
                let bear_idx = pair * 2;
                let bull_idx = pair * 2 + 1;
                let bear_spiked = spike_ids.contains(&bear_idx);
                let bull_spiked = spike_ids.contains(&bull_idx);
                
                if bear_spiked && !bull_spiked {
                    self.neurons[bull_idx].membrane_potential = (self.neurons[bull_idx].membrane_potential - COMPETITIVE_INHIBITION).max(0.0);
                } else if bull_spiked && !bear_spiked {
                    self.neurons[bear_idx].membrane_potential = (self.neurons[bear_idx].membrane_potential - COMPETITIVE_INHIBITION).max(0.0);
                } else if bear_spiked && bull_spiked {
                    self.neurons[bear_idx].membrane_potential = 0.0;
                    self.neurons[bull_idx].membrane_potential = 0.0;
                }
            }
        }
    }

    /// HEARTBEAT: Steps the entire neural network using real-world telemetry.
    /// 
    /// MAPPINGS (input channels → stimuli):
    /// - Channel 0: 12V Rail Stability (Stimulated if voltage sag)
    /// - Channel 1: VDDCR Core Voltage (Stimulated if voltage low)
    /// - Channel 2: Board Power Draw (Stimulated by high wattage)
    /// - Channel 3: Hashrate Performance (Consistency check)
    ///
    /// Each neuron now computes its input as a weighted sum across all channels,
    /// learned via STDP.
    pub fn step(&mut self, stimuli: &[f32; NUM_INPUT_CHANNELS], telem: &crate::telemetry::gpu_telemetry::GpuTelemetry) {
        self.global_step += 1;

        // Store stimuli for cross-coupling and STDP
        let raw_stimuli = *stimuli;
        let stimuli_scale = 1.0;
        let stim_ocean = stimuli[6]; // Ocean channel (index 6)

        // 0. Update Neurochemistry
        let saved_market_vol = self.modulators.market_volatility;
        let saved_mining_da = self.modulators.mining_dopamine;
        self.modulators = NeuroModulators::from_telemetry(telem);
        self.modulators.market_volatility = saved_market_vol;
        self.modulators.mining_dopamine = saved_mining_da;

        let combined_stress = (self.modulators.cortisol + saved_market_vol * 0.3).min(1.0);
        let stress_multiplier = (1.0 - combined_stress).max(0.1);

        // Blend mining-efficiency signal into the STDP learning rate.
        // mining_dopamine ∈ [-0.8, 0.8] → mining_contrib ∈ [0.0, 1.0].
        // The mining signal acts as a survival gate: positive values validate
        // current neural patterns, negative values suppress plasticity.
        let mining_contrib = (self.modulators.mining_dopamine + 0.8) / 1.6;
        let blended_da = (self.modulators.dopamine * 0.6 + mining_contrib * 0.4).clamp(0.0, 1.0);
        let learning_rate = 0.5 * blended_da;

        for neuron in &mut self.neurons {
            let target_decay = 0.15 - (0.05 * self.modulators.acetylcholine);
            neuron.decay_rate = target_decay;

            let global_target = 0.20
                - (0.05 * self.modulators.dopamine)
                + (0.15 * self.modulators.cortisol);

            let target_threshold = (global_target + if neuron.last_spike { 0.005 } else { -0.001 }).clamp(0.05, 0.50);
            neuron.threshold += (target_threshold - neuron.threshold) * learning_rate;
            neuron.threshold = neuron.threshold.clamp(0.05, 0.50);
        }

        // 1. Predictive Error Coding (Surprise)
        const PRED_ALPHA: f32 = 0.1;
        const PRED_ERR_WEIGHT: f32 = 0.5;
        let mut pred_errors = [0.0_f32; NUM_INPUT_CHANNELS];
        for ch in 0..NUM_INPUT_CHANNELS {
            let s = stimuli[ch].abs().clamp(0.0, 1.0);
            pred_errors[ch] = (s - self.predictive_state[ch]).abs();
            self.predictive_state[ch] = PRED_ALPHA * s + (1.0 - PRED_ALPHA) * self.predictive_state[ch];
        }

        let mut rng = rand::thread_rng();
        for (ch, &s) in stimuli.iter().enumerate() {
            let abs_s = s.abs().clamp(0.0, 1.0);
            if abs_s > 0.01 && rng.gen::<f32>() < abs_s {
                self.input_spike_times[ch] = self.global_step;
            }
        }

        // 2. Integrate into 16-Neuron Cluster
        // Polarity awareness: Bull neurons (odd) respond to positive, Bear (even) to negative.
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let ch = i % NUM_INPUT_CHANNELS; // Spread neurons across all 16 channels
            let is_bull = i % 2 == 1;
            let delta = stimuli[ch];
            let polarity_match = if is_bull { delta > 0.0 } else { delta < 0.0 };
            
            let mut total_current = 0.0;
            if polarity_match {
                let stim = delta.abs().clamp(0.0, 1.0);
                let surprise = PRED_ERR_WEIGHT * pred_errors[ch];
                total_current = neuron.weights[ch] * (stim + surprise) * 0.45 * stress_multiplier;
            }
            
            // Small cross-coupling from other channels (non-polarized)
            for (other_ch, &other_stim) in raw_stimuli.iter().enumerate() {
                if other_ch != ch {
                    total_current += neuron.weights[other_ch] * other_stim * stimuli_scale * 0.1;
                }
            }
            
            neuron.integrate(total_current);
        }

        // ── Cross-Modal Plasticity Boost ─────────────────────────────────────
        // When Ocean Predictoor delivers a high-confidence signal (> 0.8 bull or
        // < 0.2 bear), transiently lower the firing threshold of neurons 3–15 by 15%.
        // This lets market-sensitive neurons respond earlier to the incoming context.
        // Thresholds are restored after the fire check (stack-allocated, zero-alloc).
        let ocean_high_confidence = stim_ocean > 0.01 && (stim_ocean > 0.8 || stim_ocean < 0.2);
        let n_lif = self.neurons.len().min(16);
        let mut saved_thresholds = [0.0_f32; 16];
        if ocean_high_confidence {
            for i in 3..n_lif {
                saved_thresholds[i] = self.neurons[i].threshold;
                self.neurons[i].threshold = (self.neurons[i].threshold * 0.85).max(0.05);
            }
        }

        // 3. Evaluation: Pulsing the actual Action Potentials
        // This is where integration meets breakdown voltage (Threshold)
        let step = self.global_step;
        let mut spike_ids = Vec::new();
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if let Some(_peak_v) = neuron.check_fire() {
                neuron.last_spike = true;
                neuron.last_spike_time = step;
                spike_ids.push(i); // <-- Save who spiked

                // Log spike information for debugging
                println!("[SNN] Neuron {} spiked at step {} | Threshold: {:.4} | Peak_V: {:.4} | Membrane_Potential: {:.4}",
                    i, step, neuron.threshold, _peak_v, neuron.membrane_potential);
            } else {
                neuron.last_spike = false;
                // Log membrane potential for neurons that didn't spike (for debugging)
                if step % 100 == 0 { // Log every 100 steps to avoid spam
                    println!("[SNN] Neuron {} silent at step {} | Threshold: {:.4} | Membrane_Potential: {:.4}",
                        i, step, neuron.threshold, neuron.membrane_potential);
                }
            }
        }

        // Restore transiently lowered thresholds after fire evaluation
        if ocean_high_confidence {
            for i in 3..n_lif {
                self.neurons[i].threshold = saved_thresholds[i];
            }
        }

        // ── Coincidence Detection (N14) ──────────────────────────────────────
        // Hippocampal place cell analog: N14 fires when ≥3 distinct mining
        // chains spike within a temporal window, signaling a macro shock event.
        const COINCIDENCE_WINDOW_STEPS: i64 = 500; // ~500ms at 1ms/step
        const COINCIDENCE_THRESHOLD: usize = 3;

        // Record trading neuron spikes (N0-N13 only)
        for &sid in &spike_ids {
            if sid < 14 {
                self.coincidence_window.push((sid, self.global_step));
            }
        }
        // Evict old entries outside the temporal window
        let cutoff = self.global_step - COINCIDENCE_WINDOW_STEPS;
        self.coincidence_window.retain(|&(_, t)| t >= cutoff);

        // Count distinct chains (chain = neuron_idx / 2) active in window
        let mut chain_mask: u8 = 0;
        for &(nid, _) in &self.coincidence_window {
            chain_mask |= 1 << (nid / 2);
        }
        let active_chains = chain_mask.count_ones() as usize;

        // N14 fires if ≥COINCIDENCE_THRESHOLD distinct chains spiked recently
        if active_chains >= COINCIDENCE_THRESHOLD && self.neurons.len() > 14 {
            self.neurons[14].last_spike = true;
            self.neurons[14].last_spike_time = self.global_step;
            self.neurons[14].membrane_potential = 0.0;
            if !spike_ids.contains(&14) {
                spike_ids.push(14);
            }
        } else if self.neurons.len() > 14 {
            self.neurons[14].last_spike = false;
        }

        // ── Global Inhibitory Interneuron (N15) ──────────────────────────────
        // GABAergic analog: when N14 fires (macro shock detected), N15 raises
        // all trading neuron thresholds via Vth(t) = base_threshold + W_INHIB.
        // When N14 is quiet, thresholds exponentially decay back to base.
        const W_INHIB: f32 = 0.5;
        const INHIB_DECAY: f32 = 0.95;

        if self.neurons.len() > 15 {
            if self.neurons[14].last_spike {
                // N15 fires in response to N14
                self.neurons[15].last_spike = true;
                self.neurons[15].last_spike_time = self.global_step;
                self.neurons[15].membrane_potential = 0.0;
                if !spike_ids.contains(&15) {
                    spike_ids.push(15);
                }
                // Raise all trading neuron thresholds
                for i in 0..14 {
                    self.neurons[i].threshold = self.neurons[i].base_threshold + W_INHIB;
                }
            } else {
                self.neurons[15].last_spike = false;
                // Exponential decay back toward base threshold
                for i in 0..14 {
                    let excess = self.neurons[i].threshold - self.neurons[i].base_threshold;
                    if excess > 0.001 {
                        self.neurons[i].threshold = self.neurons[i].base_threshold + excess * INHIB_DECAY;
                    } else {
                        self.neurons[i].threshold = self.neurons[i].base_threshold;
                    }
                }
            }
        }

        // LATERAL INHIBITION: The "Shush" Rule
        // If anyone spiked, suppress the neighbors to force specialization
        if !spike_ids.is_empty() {
            const INHIBITION_STRENGTH: f32 = 0.05;
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                // If this neuron did NOT spike, subtract potential
                if !spike_ids.contains(&i) {
                    neuron.membrane_potential = (neuron.membrane_potential - INHIBITION_STRENGTH).max(0.0);
                }
            }

            // ── Bull/Bear competitive inhibition ──────────────────────────────
        // Force each trading pair to commit to one direction per tick.
        // Only pairs 0..7 (N0-N13). N14/N15 are meta-regulatory, not bear/bull.
        const COMPETITIVE_INHIBITION: f32 = 0.15;
        for pair in 0..7 {
            let bear_idx = pair * 2;
            let bull_idx = pair * 2 + 1;
            let bear_spiked = spike_ids.contains(&bear_idx);
            let bull_spiked = spike_ids.contains(&bull_idx);
            
            if bear_spiked && !bull_spiked {
                self.neurons[bull_idx].membrane_potential = 
                    (self.neurons[bull_idx].membrane_potential - COMPETITIVE_INHIBITION).max(0.0);
            } else if bull_spiked && !bear_spiked {
                self.neurons[bear_idx].membrane_potential =
                    (self.neurons[bear_idx].membrane_potential - COMPETITIVE_INHIBITION).max(0.0);
            } else if bear_spiked && bull_spiked {
                self.neurons[bear_idx].membrane_potential = 0.0;
                self.neurons[bull_idx].membrane_potential = 0.0;
            }
        }
        }

        // 4. STDP Weight Update (Reward-Modulated)
        //
        // ANALOGY: This is the "experience" step from StdpOutline.sv,
        // implemented in software. Pre→Post = LTP, Post→Pre = LTD.
        // The Dopamine level scales the learning — high reward = faster wiring.
        let dopamine_lr = learning_rate; // Already = 0.01 * dopamine
        self.apply_stdp(&raw_stimuli, dopamine_lr);

        // 4b. SYNAPTIC SCALING (L1 Weight Normalization)
        //
        // ANALOGY: This is the "budget" rule. Each neuron has a fixed total amount
        // of synaptic "ink" to spend across all channels. Reinforcing one weight
        // requires borrowing from the others — preventing any single neuron from
        // monopolizing all incoming signal (the "winner-take-all" runaway).
        //
        // Implementation: After STDP, rescale each neuron's weight vector so
        // the L1 norm (sum of weights) equals 1.0. Target budget is 1.0 so that
        // the weighted sum of a unit input still produces a unit stimulus.

        for neuron in &mut self.neurons {
            let total: f32 = neuron.weights.iter().sum();
            if total > 1e-6 {
                let scale = WEIGHT_BUDGET / total;
                for w in &mut neuron.weights {
                    *w *= scale;
                    // Re-clamp after scaling to respect hard bounds
                    *w = w.clamp(STDP_W_MIN, STDP_W_MAX);
                }
            }
        }

        // 5. Step Izhikevich Neurons (Bank 2)
        // These provide the "slow" dynamics for Chemistry/Math analogies (u-variable)
        // The tempo floor (.max(0.5)) prevents zero-multiplication when GPU clock reads fail.
        let iz_tempo = self.modulators.tempo.max(0.5);
        if self.iz_neurons.len() >= 4 {
            // Neuron 0: Tracks Vcore Stability (VDDCR_GFX)
            // Stimulate proportionally to how far Vcore sags below nominal (1.05V)
            let i_stable = (1.05 - telem.vddcr_gfx_v).max(0.0) * 20.0 * stress_multiplier * iz_tempo;
            self.iz_neurons[0].step(i_stable);

            // Neuron 1: Tracks Power Dynamics (Accumulation)
            let i_power = (telem.power_w / 50.0).clamp(0.0, 20.0) * stress_multiplier * iz_tempo;
            self.iz_neurons[1].step(i_power);

            // Neuron 2: Tracks GPU Temperature (Thermal Fatigue)
            // Higher temps = more current, modeling thermal runaway risk
            let i_thermal = ((telem.gpu_temp_c - 40.0) / 10.0).clamp(0.0, 10.0) * stress_multiplier * iz_tempo;
            self.iz_neurons[2].step(i_thermal);

            // Neuron 3: Tracks Hashrate (Work Effort / Efficiency)
            // Normalized to expected RTX 5080 Dynex hashrate range
            let i_hash = (telem.hashrate_mh / 0.05).clamp(0.0, 15.0) * stress_multiplier * iz_tempo;
            self.iz_neurons[3].step(i_hash);
        }

        // IZ[4]: FPGA Timing Stress Neuron
        // When WNS < 0 (timing violation), fires a stress spike into cortisol.
        // WNS is a slow-changing value so we re-use self.fpga_metrics (loaded at startup).
        if self.iz_neurons.len() >= 5 {
            let fpga_stress = self.modulators.fpga_stress;
            // Scale: 0.0 fpga_stress → 0 current; 1.0 → 15.0 (enough to drive spikes)
            let i_fpga = fpga_stress * 15.0 * iz_tempo;
            let spiked = self.iz_neurons[4].step(i_fpga);
            if spiked {
                // FPGA timing violation → propagate to cortisol
                self.modulators.cortisol = (self.modulators.cortisol + 0.15).min(1.0);
            }
        }

        // Update fpga_stress from loaded metrics
        if let Some(ref m) = self.fpga_metrics {
            self.modulators.fpga_stress = if m.wns_ns < 0.0 {
                // Normalize: WNS = -5ns → full stress; -0.1ns → mild stress
                (-m.wns_ns / 5.0).clamp(0.0, 1.0)
            } else {
                0.0 // Positive slack = no FPGA stress
            };
        }
    }

    /// STDP: Spike-Timing-Dependent Plasticity weight update.
    ///
    /// Implements the classic exponential STDP window:
    ///   Δw = A+ · exp(-Δt/τ+)  if pre fires before post (LTP — strengthen)
    ///   Δw = -A- · exp(Δt/τ-)  if post fires before pre (LTD — weaken)
    ///
    /// Modulated by `dopamine_lr` so reward scales learning.
    ///
    /// REFERENCE: StdpOutline.sv (research/StdpOutline.sv)
    fn apply_stdp(&mut self, _raw_stimuli: &[f32; NUM_INPUT_CHANNELS], dopamine_lr: f32) {
        if dopamine_lr < 1e-6 {
            return; // No reward → no learning (saves cycles)
        }

        let input_times = self.input_spike_times.clone();

        for neuron in &mut self.neurons {
            if neuron.last_spike_time < 0 {
                continue; // Neuron has never fired — skip
            }

            for (ch, &pre_time) in input_times.iter().enumerate() {
                if ch >= neuron.weights.len() || pre_time < 0 {
                    continue;
                }

                let post_time = neuron.last_spike_time;
                if post_time < 0 {
                    continue;
                }

                let delta_t = (post_time - pre_time) as f32;

                let dw = if delta_t >= 0.0 {
                    // Pre fired BEFORE Post (or simultaneously) → LTP (potentiate)
                    STDP_A_PLUS * (-delta_t / STDP_TAU_PLUS).exp()
                } else {
                    // Post fired BEFORE Pre → LTD (depress)
                    -STDP_A_MINUS * (delta_t / STDP_TAU_MINUS).exp()
                };


                // Apply dopamine-modulated weight change
                neuron.weights[ch] = (neuron.weights[ch] + dw * dopamine_lr)
                    .clamp(STDP_W_MIN, STDP_W_MAX);
            }
        }
    }

    // ── SNN ↔ LLM Feedback Loop ─────────────────────────────────────────

    /// Called after the LLM finishes a streaming response.
    ///
    /// Updates `tempo` based on observed tokens-per-second so the SNN reflects
    /// actual generation speed. Also boosts acetylcholine for longer responses
    /// (>200 tokens → richer detail → more "focused" signal).
    pub fn on_llm_response(&mut self, tokens: u32, duration_ms: u64) {
        if duration_ms == 0 {
            return;
        }
        let tokens_per_sec = tokens as f32 / (duration_ms as f32 / 1000.0);

        // RTX 5080 peaks ~30 tok/s on 32B — normalise to that ceiling
        self.modulators.tempo = (tokens_per_sec / 30.0).clamp(0.1, 2.0);

        // Long detailed responses → acetylcholine boost (more focused signal)
        if tokens > 200 {
            self.modulators.acetylcholine = (self.modulators.acetylcholine + 0.2).min(1.0);
        }
    }

    /// Called when the student gives explicit feedback on an AI response.
    ///
    /// Thumbs up  → dopamine spike → LTP (potentiate neurons that were active)
    /// Thumbs down → cortisol spike → mild LTD (depress active neurons)
    pub fn on_student_feedback(&mut self, positive: bool) {
        if positive {
            self.modulators.dopamine = (self.modulators.dopamine + 0.3).min(1.0);

            // Reward-modulated LTP: strengthen recently-firing neurons
            let da = self.modulators.dopamine;
            for neuron in &mut self.neurons {
                if neuron.last_spike {
                    for w in &mut neuron.weights {
                        *w = (*w + STDP_A_PLUS * da).clamp(STDP_W_MIN, STDP_W_MAX);
                    }
                }
            }
        } else {
            self.modulators.cortisol = (self.modulators.cortisol + 0.2).min(1.0);

            // Mild LTD: gently depress active neurons
            let cort = self.modulators.cortisol;
            for neuron in &mut self.neurons {
                if neuron.last_spike {
                    for w in &mut neuron.weights {
                        *w = (*w - STDP_A_MINUS * cort * 0.5).clamp(STDP_W_MIN, STDP_W_MAX);
                    }
                }
            }
        }

        // Synaptic scaling after manual feedback (same budget rule as normal step)

        for neuron in &mut self.neurons {
            let total: f32 = neuron.weights.iter().sum();
            if total > 1e-6 {
                let scale = WEIGHT_BUDGET / total;
                for w in &mut neuron.weights {
                    *w = (*w * scale).clamp(STDP_W_MIN, STDP_W_MAX);
                }
            }
        }
    }

    /// External IPC Injection: called by the UDP listener when the terminal AI
    /// explicitly sends a dopamine/cortisol spike.
    pub fn inject_learning_reward(&mut self, dopamine: f32, cortisol: f32) {
        if dopamine > 0.0 {
            self.modulators.dopamine = (self.modulators.dopamine + dopamine).clamp(0.0, 1.0);
            
            // Immediately apply LTP to neurons that recently fired
            let da = self.modulators.dopamine;
            for neuron in &mut self.neurons {
                if neuron.last_spike_time >= self.global_step - 100 { // Window of 100 steps
                    for w in &mut neuron.weights {
                        *w = (*w + STDP_A_PLUS * da).clamp(STDP_W_MIN, STDP_W_MAX);
                    }
                }
            }
        }
        
        if cortisol > 0.0 {
            self.modulators.cortisol = (self.modulators.cortisol + cortisol).clamp(0.0, 1.0);
            
            // Immediately apply LTD to neurons that recently fired
            let cort = self.modulators.cortisol;
            for neuron in &mut self.neurons {
                if neuron.last_spike_time >= self.global_step - 100 {
                    for w in &mut neuron.weights {
                        *w = (*w - STDP_A_MINUS * cort).clamp(STDP_W_MIN, STDP_W_MAX);
                    }
                }
            }
        }

        // L1 Synaptic scaling

        for neuron in &mut self.neurons {
            let total: f32 = neuron.weights.iter().sum();
            if total > 1e-6 {
                let scale = WEIGHT_BUDGET / total;
                for w in &mut neuron.weights {
                    *w = (*w * scale).clamp(STDP_W_MIN, STDP_W_MAX);
                }
            }
        }
    }

    /// DIAGNOSTIC LOGIC: Processes telemetry for fault class detection.
    pub fn infer_class(&self, rails: &[(String, f32)], fw_ok: bool) -> crate::diagnostics::fault_class::FaultClass {
        let v_gfx = rails.iter().find(|(n, _)| n == "VDDCR_GFX").map(|(_, v)| *v).unwrap_or(0.0);

        // If GFX rail is missing (< 0.05V), trigger a spike
        if v_gfx < 0.05 && fw_ok {
            return crate::diagnostics::fault_class::FaultClass::GpuDieDead; 
        }

        if !fw_ok {
            return crate::diagnostics::fault_class::FaultClass::FirmwareStateBad;
        }

        crate::diagnostics::fault_class::FaultClass::HealthyStandby
    }

    /// CHEMISTRY LOGIC: Specialized function for CHEM 1335.
    /// 
    /// Maps a Molarity value to a visual spike train.
    /// Useful for visualizing reaction rates or concentration gradients.
    pub fn analyze_chemistry(&self, molarity: f32, max_molarity: f32) -> Vec<u8> {
        // Normalize molarity to a 0.0 - 1.0 probability range
        let normalized = (molarity / max_molarity.max(1e-9)).clamp(0.0, 1.0);
        
        // Create a temporary encoder for this analysis frame
        // 20 steps is enough for a smooth UI visualization
        let encoder = super::lif::PoissonEncoder::new(20); 
        encoder.encode(normalized)
    }

    /// FPGA EXPORT: Writes the current learned parameters to Vivado-compatible `.mem` files.
    ///
    /// Three files are written alongside `base_path`:
    /// - `parameters.mem`         — LIF thresholds (Q8.8, 1 value per neuron)
    /// - `parameters_weights.mem` — Synaptic weight matrix (Q8.8, 8×channels, row-major)
    /// - `parameters_decay.mem`   — Decay rates (Q8.8, 1 value per neuron)
    ///
    /// Q8.8 encoding: `(float * 256.0) as u16` formatted as 4-char uppercase hex.
    /// Files are `$readmemh`-compatible — each line is one memory word.
    ///
    /// The FPGA BRAM is static; call `source research/load_brain.tcl` in the Vivado
    /// TCL console to hot-reload these values into the hardware between mining sessions.
    pub fn export_fpga_mem<P: AsRef<std::path::Path>>(&self, base_dir: P) -> std::io::Result<()> {
        let dir = base_dir.as_ref();
        std::fs::create_dir_all(dir)?;

        // — parameters.mem: one threshold per LIF neuron —
        let mut thresh_content = String::new();
        for n in &self.neurons {
            let q88 = (n.threshold * 256.0).clamp(0.0, 65535.0) as u16;
            thresh_content.push_str(&format!("{:04X}\n", q88));
        }
        std::fs::write(dir.join("parameters.mem"), thresh_content)?;

        // — parameters_weights.mem: weight matrix, row-major [neuron][channel] —
        let mut w_content = String::new();
        for n in &self.neurons {
            for &w in &n.weights {
                let q88 = (w * 256.0).clamp(0.0, 65535.0) as u16;
                w_content.push_str(&format!("{:04X}\n", q88));
            }
        }
        std::fs::write(dir.join("parameters_weights.mem"), w_content)?;

        // — parameters_decay.mem: leak/decay rate per neuron —
        let mut d_content = String::new();
        for n in &self.neurons {
            let q88 = (n.decay_rate * 256.0).clamp(0.0, 65535.0) as u16;
            d_content.push_str(&format!("{:04X}\n", q88));
        }
        std::fs::write(dir.join("parameters_decay.mem"), d_content)?;

        Ok(())
    }

    /// PERSISTENCE: Saves the learned thresholds, weights, and decay rates to a student model file.
    pub fn save_parameters<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let serialized = serde_json::to_string_pretty(&self.neurons)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, serialized)
    }

    /// PERSISTENCE: Loads previously learned parameters (including weights) from a student model file.
    pub fn load_parameters<P: AsRef<std::path::Path>>(&mut self, path: P) -> std::io::Result<()> {
        let data = std::fs::read_to_string(path)?;
        let loaded_neurons: Vec<LifNeuron> = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if loaded_neurons.len() == self.neurons.len() {
            self.neurons = loaded_neurons;
            // Ensure all neurons have the correct number of weights
            for neuron in &mut self.neurons {
                if neuron.weights.len() != NUM_INPUT_CHANNELS {
                    neuron.weights.resize(NUM_INPUT_CHANNELS, 1.0);
                }
            }
        }
        Ok(())
    }

    // ── Module 8: LLM → SNN Feedback Injection ──────────────────────────

    /// Called when the LLM response sentiment analysis is complete.
    ///
    /// Injects a virtual chemical spike into the SNN based on what the AI
    /// just said — praise raises dopamine, warnings raise cortisol.
    /// This closes the bidirectional SNN↔LLM feedback loop.
    ///
    /// The event is also archived to `research/neuromorphic_data.jsonl`
    /// so the AI's mood history is visible to the researcher module.
    ///
    /// ANALOGY: Like the brain's reward/threat circuitry — the prefrontal
    /// cortex (LLM analysis) signals the amygdala (SNN modulators) with
    /// a chemical pulse after each significant cognitive event.
    pub fn on_llm_feedback(&mut self, dopamine_delta: f32, cortisol_delta: f32, reason: &str) {
        // Apply deltas — clamp to valid range
        self.modulators.dopamine = (self.modulators.dopamine + dopamine_delta).clamp(0.0, 1.0);
        self.modulators.cortisol = (self.modulators.cortisol + cortisol_delta).clamp(0.0, 1.0);

        // Log the spike event to the research JSONL stream (non-blocking append)
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let event_line = serde_json::json!({
            "event_type": "llm_spike",
            "timestamp": ts,
            "reason": reason,
            "dopamine_delta": (dopamine_delta as f64 * 10000.0).round() / 10000.0,
            "cortisol_delta": (cortisol_delta as f64 * 10000.0).round() / 10000.0,
            "state_after": {
                "dopamine": (self.modulators.dopamine as f64 * 10000.0).round() / 10000.0,
                "cortisol": (self.modulators.cortisol as f64 * 10000.0).round() / 10000.0,
                "acetylcholine": (self.modulators.acetylcholine as f64 * 10000.0).round() / 10000.0,
            }
        });

        use std::io::Write as _;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("DATA/research/neuromorphic_data.jsonl")
        {
            let _ = writeln!(file, "{}", event_line);
        }
    }

    /// Returns the flattened weight matrix [num_neurons × NUM_INPUT_CHANNELS] for GPU upload.
    pub fn flatten_weights(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.neurons.len() * NUM_INPUT_CHANNELS);
        for neuron in &self.neurons {
            for ch in 0..NUM_INPUT_CHANNELS {
                flat.push(if ch < neuron.weights.len() { neuron.weights[ch] } else { 0.0 });
            }
        }
        flat
    }
}
