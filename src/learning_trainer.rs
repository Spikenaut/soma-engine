//! Neuromorphic Learning Trainer
//! 
//! This module provides the infrastructure for "Off-line Training" (Path B).
//! It replays recorded telemetry data from the Gold Dataset 
//! into the Spiking Inference Engine to evolve thresholds, weights,
//! and connectivity via STDP.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
// use serde::{Deserialize, Serialize};
// use crate::hardware_bridge::GpuTelemetry;
use crate::snn::engine::SpikingInferenceEngine;
use crate::ai::researcher::NeuromorphicSnapshot;

/// The Trainer manages the evolution of the SNN parameters.
/// 
/// ANALOGY:
/// If the `SpikingInferenceEngine` is the Brain, this Trainer is the "Experience".
/// It exposes the brain to past electrical events so it can learn patterns (Long-Term Potentiation).
pub struct NeuromorphicTrainer {
    pub engine: SpikingInferenceEngine,
}

impl NeuromorphicTrainer {
    pub fn new() -> Self {
        Self {
            engine: SpikingInferenceEngine::new(),
        }
    }

    /// Replays the research/neuromorphic_data.jsonl Gold Dataset.
    /// 
    /// This is the primary way the AI "dreams" about hardware data 
    /// to refine its internal models before being used for live control.
    /// Now evolves both thresholds AND synaptic weights via STDP.
    pub fn run_training_session<P: AsRef<Path>>(&mut self, data_path: P) -> Result<TrainingSummary, Box<dyn std::error::Error>> {
        let file = File::open(data_path)?;
        let reader = BufReader::new(file);

        let mut summary = TrainingSummary::default();
        let mut initial_thresholds = Vec::new();
        let mut initial_weights = Vec::new();

        for n in &self.engine.neurons {
            initial_thresholds.push(n.threshold);
            initial_weights.push(n.weights.clone());
            // One counter slot per neuron, initialized to zero
            summary.per_neuron_spikes.push(0);
        }

        for line in reader.lines() {
            let line = line?;
            if let Ok(entry) = serde_json::from_str::<NeuromorphicSnapshot>(&line) {
                // 1. Advance the neural state based on recorded electricity
                let stimuli = entry.telemetry.to_stimuli();
                self.engine.step(&stimuli, &entry.telemetry);
                
                // 2. Monitor Learning Progress
                summary.steps_processed += 1;

                // Track total and per-neuron activity
                for (idx, neuron) in self.engine.neurons.iter().enumerate() {
                    if neuron.last_spike {
                        summary.total_spikes += 1;
                        if let Some(count) = summary.per_neuron_spikes.get_mut(idx) {
                            *count += 1;
                        }
                    }
                }
            }
        }

        // Calculate deltas (Learning Progress)
        for (i, n) in self.engine.neurons.iter().enumerate() {
            // Threshold drift
            let delta = n.threshold - initial_thresholds[i];
            summary.threshold_drifts.push(delta);

            // Weight drift (per channel)
            let mut w_deltas = Vec::new();
            for (ch, &w) in n.weights.iter().enumerate() {
                let initial_w = initial_weights.get(i)
                    .and_then(|ws: &Vec<f32>| ws.get(ch))
                    .copied()
                    .unwrap_or(1.0);
                w_deltas.push(w - initial_w);
            }
            summary.weight_drifts.push(w_deltas);
        }

        // PERSIST the learned brain to a file
        let _ = self.engine.save_parameters("DATA/research/snn_model.json");

        Ok(summary)
    }

    /// EXPORT: Converts the learned thresholds AND weights into Verilog-compatible hex memory files.
    /// This allows the FPGA to "download" the learning from this Rust session.
    ///
    /// Format:
    ///   parameters.mem      — thresholds in Q8.8 fixed point
    ///   parameters_weights.mem — weight matrix in Q8.8 fixed point
    pub fn export_to_verilog<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        // Export thresholds
        let mut content = String::new();
        for neuron in &self.engine.neurons {
            // Convert 0.0-2.0 float range to 16-bit fixed point (Q8.8)
            // 1.0 = 256 (0x0100)
            let fixed_val = (neuron.threshold * 256.0).clamp(0.0, 65535.0) as u16;
            content.push_str(&format!("{:04X}\n", fixed_val));
        }
        std::fs::write(&path, content)?;

        // Export weight matrix (same Q8.8 format, row-major)
        let weight_path = path.as_ref().with_file_name("parameters_weights.mem");
        let mut w_content = String::new();
        for neuron in &self.engine.neurons {
            for &w in &neuron.weights {
                let fixed_w = (w * 256.0).clamp(0.0, 65535.0) as u16;
                w_content.push_str(&format!("{:04X}\n", fixed_w));
            }
        }
        std::fs::write(weight_path, w_content)?;

        Ok(())
    }

    /// SIMULATION: Exports stimulus and expected spikes in 16-bit packed hex format 
    /// based on a replay of the provided data path.
    pub fn export_simulation_data<P: AsRef<Path>>(&mut self, data_path: P, stim_file: &str, exp_file: &str) -> std::io::Result<()> {
        let file = File::open(data_path)?;
        let reader = BufReader::new(file);

        let mut stim_content = String::new();
        let mut exp_content = String::new();

        for line in reader.lines() {
            let line = line.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            if let Ok(entry) = serde_json::from_str::<NeuromorphicSnapshot>(&line) {
                // 1. Pack Stimulus (Vcore, Power, Hashrate)
                // Format: [Vcore_8.8][Power_12.4][Hash_8.8]
                let vgfx = (entry.telemetry.vddcr_gfx_v * 256.0) as u16;
                let pwr = (entry.telemetry.power_w * 16.0) as u16;
                let hsh = (entry.telemetry.hashrate_mh * 256.0) as u16;
                
                stim_content.push_str(&format!("{:04X}{:04X}{:04X}\n", vgfx, pwr, hsh));

                // 2. Step engine and pack spikes
                self.engine.step(&entry.telemetry.to_stimuli(), &entry.telemetry);
                let mut spike_byte: u8 = 0;
                for (i, neuron) in self.engine.neurons.iter().enumerate() {
                    if i < 8 && neuron.last_spike {
                        spike_byte |= 1 << i;
                    }
                }
                exp_content.push_str(&format!("{:02X}\n", spike_byte));
            }
        }

        std::fs::write(stim_file, stim_content)?;
        std::fs::write(exp_file, exp_content)?;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct TrainingSummary {
    pub steps_processed: usize,
    pub total_spikes: u64,
    pub threshold_drifts: Vec<f32>,
    /// Per-neuron, per-channel weight deltas: weight_drifts[neuron][channel]
    pub weight_drifts: Vec<Vec<f32>>,
    /// Individual spike count per neuron across the full training session
    pub per_neuron_spikes: Vec<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_replay() {
        let mut trainer = NeuromorphicTrainer::new();
        let path = "DATA/research/neuromorphic_data.jsonl";

        // Only run if the file exists (prevents CI failure)
        if Path::new(path).exists() {
            let result = trainer.run_training_session(path);
            assert!(result.is_ok());
            let summary = result.unwrap();
            println!("Summary: {:?}", summary);
            assert!(summary.steps_processed > 0);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  E-prop / OTTT Epoch Trainer
//
//  Extends reward-modulated STDP with eligibility traces and
//  Online Temporal Trace Training (OTTT) presynaptic traces.
//
//  Algorithm (per tick t):
//
//    OTTT presynaptic trace (O(1) per channel, no spike history stored):
//      â_j[t+1] = λ · â_j[t] + s_j[t+1]
//
//    E-prop eligibility trace (O(1) per synapse):
//      e_{ij}[t+1] = λ · e_{ij}[t] + â_j[t] · z_i[t+1]
//
//    Reward-modulated weight update:
//      Δw_{ij} = R[t] · e_{ij}[t+1] · η_eprop
//      w_{ij}  = clamp(w_{ij} + Δw_{ij}, W_MIN, W_MAX)
//
//  where:
//    λ  = TRACE_LAMBDA = 1 - LIF_decay_rate = 0.85
//    R  = composite reward from telemetry delta
//    η_eprop = EPROP_LR (additive to the engine's own instantaneous STDP)
//
//  All trace buffers are pre-allocated at construction. The tick()
//  method is zero-alloc after init.
// ════════════════════════════════════════════════════════════════════

/// Number of LIF neurons in Bank 1 (matches SpikingInferenceEngine::new()).
const NUM_LIF_NEURONS: usize = 16;

/// Trace decay constant: survival factor of the LIF membrane leak.
/// λ = 1.0 - decay_rate = 1.0 - 0.15 = 0.85.
const TRACE_LAMBDA: f32 = 0.85;

/// E-prop learning rate (additive to the engine's reward-modulated STDP).
/// Kept smaller than STDP_A_PLUS (0.01) to avoid weight instability.
const EPROP_LR: f32 = 0.002;

/// Normalisation ceiling for hashrate reward (RTX 5080 Dynex target).
const MAX_HASHRATE_MH: f32 = 0.015;

/// Normalisation ceiling for power reward (RTX 5080 TDP).
const MAX_POWER_W: f32 = 400.0;

/// Fast-Sigmoid surrogate gradient for the LIF hard-threshold.
///
/// Replaces the Heaviside step derivative (zero everywhere, undefined at θ)
/// with a smooth bell centred on the threshold:
///
///   f′(v) = 1 / (1 + |10 · (v − θ)|)²
///
/// Properties:
///   · f′(θ)      = 1.0    — maximum gradient at exact threshold
///   · f′(θ±0.1) ≈ 0.50   — half-maximum at ±0.1 V from threshold
///   · f′(θ±0.3) ≈ 0.10   — rapidly fades for distant potentials
///   · No `pow` / `exp`   — two multiplies, one abs, one divide
///
/// Used in `EpochTrainer::tick()` to propagate eligibility-trace credit
/// through neurons that came close to firing but did not cross θ.
#[inline(always)]
fn fast_sigmoid_grad(v: f32, theta: f32) -> f32 {
    let x = 10.0 * (v - theta);
    let denom = 1.0 + x.abs();
    1.0 / (denom * denom)
}

/// Per-epoch training metrics.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch:       usize,
    /// Mean reward signal across all ticks in the epoch.
    pub avg_reward:  f32,
    /// Mean per-neuron spike rate (spikes / neuron / tick).
    pub spike_rate:  f32,
    /// Mean weight magnitude across the full weight matrix.
    pub mean_weight: f32,
    /// Standard deviation of weight magnitudes.
    pub std_weight:  f32,
    /// Wall-clock milliseconds per training tick (averaged over the epoch).
    pub ms_per_tick: f32,
}

/// Online E-prop trainer with OTTT presynaptic traces.
///
/// All trace buffers are pre-allocated in `new()`.
/// The `tick()` hot path is zero-alloc.
pub struct EpochTrainer {
    pub engine: SpikingInferenceEngine,
    /// OTTT presynaptic traces: one per input channel.
    pre_traces: [f32; crate::snn::engine::NUM_INPUT_CHANNELS],
    /// E-prop eligibility traces: [NUM_LIF_NEURONS × NUM_INPUT_CHANNELS], row-major.
    eligibility: Vec<f32>,
    // Previous-step state for reward delta computation.
    prev_hashrate_mh: f32,
    prev_power_w:     f32,
    prev_gpu_temp_c:  f32,
    /// Previous Qubic tick rate for stability delta (Triple Node Sync).
    prev_qubic_tick_rate: f32,
}

impl EpochTrainer {
    /// Construct a trainer with pre-allocated trace buffers.
    pub fn new() -> Self {
        use crate::snn::engine::NUM_INPUT_CHANNELS;
        Self {
            engine:           SpikingInferenceEngine::new(),
            pre_traces:       [0.0; NUM_INPUT_CHANNELS],
            eligibility:      vec![0.0_f32; NUM_LIF_NEURONS * NUM_INPUT_CHANNELS],
            prev_hashrate_mh: 0.0,
            prev_power_w:     0.0,
            prev_gpu_temp_c:  40.0,
            prev_qubic_tick_rate: 0.0,
        }
    }

    /// Composite reward signal from telemetry (0.0 – 1.0).
    ///
    /// R = 0.40 · hashrate_stability
    ///   + 0.25 · power_efficiency
    ///   + 0.20 · thermal_margin
    ///   + 0.15 · qubic_stability
    ///
    /// Weight rationale:
    ///   Hardware homeostasis (power + thermal) = 0.45 — always dominant.
    ///   Hashrate = 0.40 — primary work signal.
    ///   Qubic = 0.15 — modulatory only, cannot cause reward explosion.
    ///
    /// The Qubic term rewards a smooth, steady tick ingestion rate.
    /// When the Qubic node is offline (tick_rate = 0, prev = 0), the delta
    /// is zero and qubic_stability = 1.0, so the term gracefully degrades
    /// to a constant +0.15 baseline rather than penalising the system.
    fn compute_reward(&self, telem: &crate::telemetry::gpu_telemetry::GpuTelemetry) -> f32 {
        // 1. Hashrate stability: reward steady or rising hashrate
        let hash_norm = (telem.hashrate_mh / MAX_HASHRATE_MH).clamp(0.0, 1.0);
        let prev_norm = (self.prev_hashrate_mh / MAX_HASHRATE_MH).clamp(0.0, 1.0);
        let hashrate_stability = 1.0 - (hash_norm - prev_norm).abs();

        // 2. Power efficiency: normalised hashrate per normalised watt
        let power_efficiency = if telem.power_w > 1.0 {
            (hash_norm / (telem.power_w / MAX_POWER_W)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 3. Thermal margin: 40 °C → 1.0, 100 °C → 0.0
        let thermal_margin = (1.0 - (telem.gpu_temp_c - 40.0) / 60.0).clamp(0.0, 1.0);

        // 4. Qubic tick-rate stability: reward smooth ingestion.
        //    Delta is bounded [0, 1] because both values are clamped [0, 1].
        //    When node is offline: both = 0 → delta = 0 → stability = 1.0 (benign).
        //    When node hiccups: rate spike → delta > 0 → stability < 1.0 (penalty).
        let qubic_rate = telem.qubic_tick_rate.clamp(0.0, 1.0);
        let qubic_delta = (qubic_rate - self.prev_qubic_tick_rate).abs();
        let qubic_stability = 1.0 - qubic_delta;

        (0.40 * hashrate_stability
       + 0.25 * power_efficiency
       + 0.20 * thermal_margin
       + 0.15 * qubic_stability)
            .clamp(0.0, 1.0)
    }

    /// One training tick.
    ///
    /// 1. Compute composite reward from telemetry delta.
    /// 2. Encode telemetry to Poisson pre-spikes.
    /// 3. Update OTTT presynaptic traces in-place.
    /// 4. Forward pass via `engine.step()` (includes engine's instantaneous STDP).
    /// 5. Update E-prop eligibility traces.
    /// 6. Apply E-prop weight update (additive to STDP).
    /// 7. L1 synaptic scaling (weight budget = 1.0 per neuron).
    ///
    /// Returns `(reward, post_spike_count)`. Zero heap allocation.
    pub fn tick(
        &mut self,
        telem: &crate::telemetry::gpu_telemetry::GpuTelemetry,
    ) -> (f32, usize) {
        use rand::Rng as _;
        use crate::snn::engine::NUM_INPUT_CHANNELS;
        use crate::snn::{STDP_W_MIN, STDP_W_MAX};

        // 1. Reward from previous vs current telemetry
        let reward = self.compute_reward(telem);

        // 2. Normalise telemetry to spike probabilities (same mapping as engine.step)
        let stim_vddcr  = ((telem.vddcr_gfx_v - 1.0).abs() * 2.0).clamp(0.0, 1.0);
        let stim_power  = (telem.power_w / MAX_POWER_W).clamp(0.0, 1.0);
        let stim_hash   = (telem.hashrate_mh / MAX_HASHRATE_MH).clamp(0.0, 1.0);
        let stim_ocean  = telem.ocean_intel.clamp(0.0, 1.0);
        let raw_stimuli = [stim_vddcr, stim_power, stim_hash, stim_ocean];

        // 3. Poisson spike encoding: each channel fires with probability = stimulus
        let mut rng = rand::thread_rng();
        let mut pre_spikes = [0.0_f32; NUM_INPUT_CHANNELS];
        for ch in 0..NUM_INPUT_CHANNELS {
            if raw_stimuli[ch] > 0.01 && rng.gen::<f32>() < raw_stimuli[ch] {
                pre_spikes[ch] = 1.0;
            }
        }

        // 4. OTTT presynaptic trace update (in-place, zero-alloc)
        //    â_j[t+1] = λ · â_j[t] + s_j[t+1]
        for ch in 0..NUM_INPUT_CHANNELS {
            self.pre_traces[ch] = TRACE_LAMBDA * self.pre_traces[ch] + pre_spikes[ch];
        }
        // Snapshot pre_traces before engine borrow (array is Copy)
        let pre_traces_snap = self.pre_traces;

        // 5. Forward pass
        self.engine.step(&telem.to_stimuli(), &telem);

        // 6a. Snapshot post-spike flags, membrane potentials, and thresholds.
        //
        // All three arrays live on the stack (zero-alloc).  For neurons that
        // spiked, `membrane_potential` is already 0.0 after the hard reset in
        // `check_fire()`; the eligibility step below handles that case by
        // substituting the full gradient (1.0) instead of the surrogate.
        let n_neurons = self.engine.neurons.len().min(NUM_LIF_NEURONS);
        let mut post_spikes_snap = [0.0_f32; NUM_LIF_NEURONS];
        let mut v_snap           = [0.0_f32; NUM_LIF_NEURONS];
        let mut theta_snap       = [0.0_f32; NUM_LIF_NEURONS];
        for i in 0..n_neurons {
            let n = &self.engine.neurons[i];
            if n.last_spike {
                post_spikes_snap[i] = 1.0;
            }
            v_snap[i]     = n.membrane_potential;
            theta_snap[i] = n.threshold;
        }
        let spike_count = post_spikes_snap.iter().filter(|&&s| s > 0.0).count();

        // 6b. Update eligibility traces with Fast-Sigmoid surrogate gradient.
        //
        //   pseudo_dz_i = 1.0              if neuron i fired  (binary spike wins)
        //               = f′(v_i, θ_i)    otherwise          (near-miss credit)
        //
        //   e_{ij}[t+1] = λ · e_{ij}[t] + â_j[t] · pseudo_dz_i
        //
        // Neurons whose potential was close to θ but did not cross it receive
        // a proportional gradient credit via `fast_sigmoid_grad`.  This allows
        // the model to learn from near-misses — the key advantage over the
        // binary OTTT formulation where silent neurons contribute nothing.
        for i in 0..n_neurons {
            let pseudo_dz = if post_spikes_snap[i] > 0.0 {
                1.0_f32 // Spiked: full gradient (v was ≥ θ by definition)
            } else {
                fast_sigmoid_grad(v_snap[i], theta_snap[i])
            };
            for ch in 0..NUM_INPUT_CHANNELS {
                let idx = i * NUM_INPUT_CHANNELS + ch;
                self.eligibility[idx] =
                    TRACE_LAMBDA * self.eligibility[idx]
                    + pre_traces_snap[ch] * pseudo_dz;
            }
        }

        // 6c. Apply E-prop weight update: Δw = reward · e · η_eprop
        for i in 0..n_neurons {
            let n_weights = self.engine.neurons[i].weights.len().min(NUM_INPUT_CHANNELS);
            for ch in 0..n_weights {
                let idx = i * NUM_INPUT_CHANNELS + ch;
                let e   = self.eligibility[idx];
                let dw  = reward * e * EPROP_LR;
                self.engine.neurons[i].weights[ch] =
                    (self.engine.neurons[i].weights[ch] + dw)
                        .clamp(STDP_W_MIN, STDP_W_MAX);
            }
        }

        // 7. L1 synaptic scaling — total weight budget = 1.0 per neuron
        for i in 0..n_neurons {
            let total: f32 = self.engine.neurons[i].weights.iter().sum();
            if total > 1e-6 {
                let scale = 1.0 / total;
                for w in &mut self.engine.neurons[i].weights {
                    *w = (*w * scale).clamp(STDP_W_MIN, STDP_W_MAX);
                }
            }
        }

        // 8. Advance previous-step state
        self.prev_hashrate_mh    = telem.hashrate_mh;
        self.prev_power_w        = telem.power_w;
        self.prev_gpu_temp_c     = telem.gpu_temp_c;
        self.prev_qubic_tick_rate = telem.qubic_tick_rate;

        (reward, spike_count)
    }

    /// Load and validate telemetry samples from a JSONL file.
    ///
    /// Accepts any record that deserialises to `NeuromorphicSnapshot` with
    /// `power_w > 0` (skips idle/corrupt entries silently).
    /// Returns `Err` only on I/O failures.
    pub fn load_samples(
        path: impl AsRef<Path>,
    ) -> std::io::Result<Vec<NeuromorphicSnapshot>> {
        let file   = File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut samples = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if let Ok(snap) =
                serde_json::from_str::<NeuromorphicSnapshot>(&line)
            {
                if snap.telemetry.power_w > 0.0 {
                    samples.push(snap);
                }
            }
        }
        Ok(samples)
    }

    /// Run `epochs` passes over `samples` with E-prop online learning.
    ///
    /// Membrane potentials reset at each epoch start for a clean forward pass.
    /// Pre-traces and eligibility traces persist across epochs to carry
    /// temporal credit accumulated at epoch boundaries.
    ///
    /// Returns per-epoch metrics (length = `epochs`).
    pub fn run_epochs(
        &mut self,
        samples: &[NeuromorphicSnapshot],
        epochs:  usize,
    ) -> Vec<EpochMetrics> {
        use std::time::Instant;
        use crate::snn::engine::NUM_INPUT_CHANNELS;

        let mut log = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let t0 = Instant::now();
            let mut total_reward  = 0.0_f32;
            let mut total_spikes: u64 = 0;

            // Reset membrane potentials; traces persist intentionally.
            for neuron in &mut self.engine.neurons {
                neuron.membrane_potential = 0.0;
            }
            // Reset prev-step telemetry to avoid reward aliasing at epoch boundary.
            self.prev_hashrate_mh    = 0.0;
            self.prev_power_w        = 0.0;
            self.prev_gpu_temp_c     = 40.0;
            self.prev_qubic_tick_rate = 0.0;

            for sample in samples.iter() {
                let (r, spikes) = self.tick(&sample.telemetry);
                total_reward  += r;
                total_spikes  += spikes as u64;
            }

            let n           = samples.len().max(1) as f32;
            let elapsed_ms  = t0.elapsed().as_secs_f32() * 1000.0;

            // Weight matrix statistics
            let total_synapses = NUM_LIF_NEURONS * NUM_INPUT_CHANNELS;
            let mut sum_w  = 0.0_f32;
            let mut sum_w2 = 0.0_f32;
            let mut count  = 0usize;
            for neuron in &self.engine.neurons {
                for &w in &neuron.weights {
                    sum_w  += w;
                    sum_w2 += w * w;
                    count  += 1;
                }
            }
            let count_f   = count.max(1) as f32;
            let mean_w    = sum_w / count_f;
            let std_w     = ((sum_w2 / count_f) - mean_w * mean_w).max(0.0).sqrt();
            let _ = total_synapses; // suppress unused warning if count differs

            let m = EpochMetrics {
                epoch,
                avg_reward:  total_reward / n,
                spike_rate:  total_spikes as f32 / (n * NUM_LIF_NEURONS as f32),
                mean_weight: mean_w,
                std_weight:  std_w,
                ms_per_tick: elapsed_ms / n,
            };

            println!(
                "Epoch {:>3}/{} | reward={:.4} | spike_rate={:.3} | \
                 w={:.4}±{:.4} | {:.3}ms/tick",
                epoch + 1, epochs,
                m.avg_reward, m.spike_rate,
                m.mean_weight, m.std_weight,
                m.ms_per_tick,
            );

            log.push(m);
        }

        log
    }

    /// Export learned parameters to FPGA-compatible `.mem` files.
    ///
    /// Writes three files to `out_dir/`:
    /// - `parameters.mem`         — 8 threshold values, Q8.8 hex
    /// - `parameters_weights.mem` — 24 weight values (8 × 3), Q8.8 hex, row-major
    /// - `parameters_decay.mem`   — 8 decay-rate values, Q8.8 hex
    ///
    /// Format: one 4-character uppercase hex value per line (`$readmemh` compatible).
    /// Q8.8: value × 256, saturating to u16. (1.0 → 0x0100, 0.15 → 0x0026)
    pub fn export_fpga(&self, out_dir: impl AsRef<Path>) -> std::io::Result<()> {
        use crate::snn::engine::NUM_INPUT_CHANNELS;
        let dir = out_dir.as_ref();
        std::fs::create_dir_all(dir)?;

        // parameters.mem — thresholds in Q8.8
        let mut thresholds = String::new();
        for neuron in &self.engine.neurons {
            let fixed = (neuron.threshold * 256.0).clamp(0.0, 65535.0) as u16;
            thresholds.push_str(&format!("{:04X}\n", fixed));
        }
        std::fs::write(dir.join("parameters.mem"), thresholds)?;

        // parameters_weights.mem — weight matrix in Q8.8, row-major [neuron][channel]
        let mut weights = String::new();
        for neuron in &self.engine.neurons {
            for ch in 0..NUM_INPUT_CHANNELS {
                let w     = if ch < neuron.weights.len() { neuron.weights[ch] } else { 0.0 };
                let fixed = (w * 256.0).clamp(0.0, 65535.0) as u16;
                weights.push_str(&format!("{:04X}\n", fixed));
            }
        }
        std::fs::write(dir.join("parameters_weights.mem"), weights)?;

        // parameters_decay.mem — decay rates in Q8.8 (for RTL consumption)
        let mut decay = String::new();
        for neuron in &self.engine.neurons {
            let fixed = (neuron.decay_rate * 256.0).clamp(0.0, 65535.0) as u16;
            decay.push_str(&format!("{:04X}\n", fixed));
        }
        std::fs::write(dir.join("parameters_decay.mem"), decay)?;

        Ok(())
    }
}
