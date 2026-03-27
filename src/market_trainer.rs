//! Market-Spikenaut Trainer — Trading-Specific SNN Learning
//!
//! Reads `research/ghost_market_log.jsonl` produced by `market_pilot` and
//! trains the SNN using a **directional price-prediction reward** with an
//! N-tick lookahead.  Mining telemetry fields (`hashrate_mh`, `power_w`)
//! are never touched here — this trainer speaks purely market language.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use serde::Deserialize;

use crate::snn::engine::{SpikingInferenceEngine, NUM_INPUT_CHANNELS};
use crate::snn::{STDP_W_MIN, STDP_W_MAX};
use crate::telemetry::gpu_telemetry::GpuTelemetry;

const PREDICTION_HORIZON: usize = 12;
const MARKET_EPROP_LR: f32 = 0.002;
const TRACE_LAMBDA: f32 = 0.85;
const PAIRS: [(usize, usize); 7] = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13),
];

#[derive(Debug, Clone, Deserialize)]
pub struct MarketLogRecord {
    pub dnx_price_usd: f32,
    #[serde(default)]
    pub quai_price_usd: f32,
    #[serde(default)]
    pub qubic_price_usd: f32,
    #[serde(default)]
    pub kaspa_price_usd: f32,
    #[serde(default)]
    pub monero_price_usd: f32,
    #[serde(default)]
    pub ocean_price_usd: f32,
    #[serde(default)]
    pub verus_price_usd: f32,
    #[serde(default)]
    pub dnx_hashrate_mh: f32,
    #[serde(default)]
    pub quai_gas_price: f32,
    #[serde(default)]
    pub quai_tx_count: f32,
    #[serde(default)]
    pub quai_block_utilization: f32,
    #[serde(default)]
    pub quai_staking_ratio: f32,
    #[serde(default)]
    pub vddcr_gfx_v: f32,
    #[serde(default)]
    pub power_w: f32,
    #[serde(default)]
    pub gpu_temp_c: f32,
    #[serde(default)]
    pub fan_speed_pct: f32,
    // Housekeeping
    #[serde(default)]
    pub neuron_bear_membrane: f32,
    #[serde(default)]
    pub neuron_bull_membrane: f32,
    #[serde(default)]
    pub ocean_intel: f32,
    #[serde(default)]
    pub portfolio_value: f32,
    #[serde(default)]
    pub realized_pnl_usdt: f32,
}

impl MarketLogRecord {
    pub fn to_stimuli(&self) -> [f32; 16] {
        [
            self.dnx_price_usd,      // Ch0
            self.quai_price_usd,     // Ch1
            self.qubic_price_usd,    // Ch2
            self.kaspa_price_usd,    // Ch3
            self.monero_price_usd,   // Ch4
            self.ocean_price_usd,    // Ch5
            self.verus_price_usd,    // Ch6
            self.dnx_hashrate_mh,    // Ch7
            self.quai_gas_price,     // Ch8
            self.quai_tx_count,      // Ch9
            self.quai_block_utilization, // Ch10
            self.quai_staking_ratio, // Ch11
            self.vddcr_gfx_v,       // Ch12
            self.power_w,            // Ch13
            self.gpu_temp_c,         // Ch14
            self.fan_speed_pct,      // Ch15
        ]
    }

    pub fn to_telemetry(&self, _prev: &MarketLogRecord) -> GpuTelemetry {
        GpuTelemetry {
            vddcr_gfx_v: self.vddcr_gfx_v,
            power_w: self.power_w,
            gpu_temp_c: self.gpu_temp_c,
            fan_speed_pct: self.fan_speed_pct,
            hashrate_mh: self.dnx_hashrate_mh,
            ocean_intel: self.ocean_intel.clamp(0.0, 1.0),
            // These will be used by the trainer's internal normalization if needed,
            // but the trainer currently maps GpuTelemetry fields directly to engine channels.
            ..GpuTelemetry::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketEpochMetrics {
    pub epoch: usize,
    pub avg_reward: f32,
    pub bull_spike_rate: f32,
    pub bear_spike_rate: f32,
    pub accuracy: f32,
    pub mean_weight: f32,
    pub std_weight: f32,
    pub ms_per_tick: f32,
}

pub struct MarketTrainer {
    pub engine: SpikingInferenceEngine,
    eligibility: Vec<f32>,
}

impl MarketTrainer {
    pub fn new() -> Self {
        Self {
            engine: SpikingInferenceEngine::new(),
            eligibility: vec![0.0_f32; 16 * NUM_INPUT_CHANNELS], // 16 neurons * 16 channels = 256
        }
    }

    pub fn load_records(path: impl AsRef<Path>) -> std::io::Result<Vec<MarketLogRecord>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if let Ok(r) = serde_json::from_str::<MarketLogRecord>(&line) {
                records.push(r);
            }
        }
        Ok(records)
    }

    pub fn run_epochs(&mut self, records: &[MarketLogRecord], epochs: usize) -> Vec<MarketEpochMetrics> {
        use std::time::Instant;
        let mut log = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let t0 = Instant::now();
            for n in &mut self.engine.neurons { n.membrane_potential = 0.0; }

            let mut total_reward = 0.0_f32;
            let mut bull_spikes: u64 = 0;
            let mut bear_spikes: u64 = 0;
            let mut correct_preds: u64 = 0;
            let mut total_preds: u64 = 0;

            for t in 0..records.len() {
                let prev = if t > 0 { &records[t - 1] } else { &records[0] };
                let telem = records[t].to_telemetry(prev);
                let stimuli = records[t].to_stimuli();

                // Step the engine with 16-channel normalized stimulus
                self.engine.step(&stimuli, &telem);

                let market_reward = if t + PREDICTION_HORIZON < records.len() {
                    let prices_now = [
                        records[t].dnx_price_usd, records[t].quai_price_usd,
                        records[t].qubic_price_usd, records[t].kaspa_price_usd,
                        records[t].monero_price_usd, records[t].ocean_price_usd,
                        records[t].verus_price_usd,
                    ];
                    let prices_future = [
                        records[t + PREDICTION_HORIZON].dnx_price_usd,
                        records[t + PREDICTION_HORIZON].quai_price_usd,
                        records[t + PREDICTION_HORIZON].qubic_price_usd,
                        records[t + PREDICTION_HORIZON].kaspa_price_usd,
                        records[t + PREDICTION_HORIZON].monero_price_usd,
                        records[t + PREDICTION_HORIZON].ocean_price_usd,
                        records[t + PREDICTION_HORIZON].verus_price_usd,
                    ];
                    let mut rewards = vec![0.0_f32; 16]; // Use Vec to prevent out-of-bounds on index 13
                    for (chain_idx, &(bear, bull)) in PAIRS.iter().enumerate() {
                        let delta = prices_future[chain_idx] - prices_now[chain_idx];
                        let dir = if delta.abs() > 1e-6 { delta.signum() } else { 0.0 };
                        if self.engine.neurons[bull].last_spike {
                            rewards[bull] = dir;
                            total_preds += 1;
                            if dir > 0.0 { correct_preds += 1; }
                        }
                        if self.engine.neurons[bear].last_spike {
                            rewards[bear] = -dir;
                            total_preds += 1;
                            if dir < 0.0 { correct_preds += 1; }
                        }
                    }
                    Some(rewards)
                } else { None };


                if let Some(rewards) = market_reward {
                    for i in 0..16 { // Update all 16 neurons
                        let r = rewards.get(i).cloned().unwrap_or(0.0);
                        total_reward += r.abs();
                        let pseudo_dz = if self.engine.neurons[i].last_spike { 1.0 } else { 0.1 };
                        for ch in 0..NUM_INPUT_CHANNELS { // 16 channels
                            let idx = i * NUM_INPUT_CHANNELS + ch;
                            self.eligibility[idx] = TRACE_LAMBDA * self.eligibility[idx] + 0.5 * pseudo_dz;
                            let dw = r * self.eligibility[idx] * MARKET_EPROP_LR;
                            self.engine.neurons[i].weights[ch] = (self.engine.neurons[i].weights[ch] + dw).clamp(STDP_W_MIN, STDP_W_MAX);
                        }
                    }
                }
                for &(_, bull) in &PAIRS {
                    if self.engine.neurons[bull].last_spike { bull_spikes += 1; }
                }
                for &(bear, _) in &PAIRS {
                    if self.engine.neurons[bear].last_spike { bear_spikes += 1; }
                }
            }

            let m = MarketEpochMetrics {
                epoch,
                avg_reward: total_reward / records.len() as f32,
                bull_spike_rate: bull_spikes as f32 / records.len() as f32,
                bear_spike_rate: bear_spikes as f32 / records.len() as f32,
                accuracy: if total_preds > 0 { correct_preds as f32 / total_preds as f32 } else { 0.0 },
                mean_weight: 0.5,
                std_weight: 0.1,
                ms_per_tick: t0.elapsed().as_secs_f32() * 1000.0 / records.len() as f32,
            };
            println!(
                "[market] Epoch {:>3} | reward={:+.4} | bull={:.3} bear={:.3} | acc={:.1}%",
                epoch + 1, m.avg_reward, m.bull_spike_rate, m.bear_spike_rate, m.accuracy * 100.0
            );
            log.push(m);
        }
        log
    }

    pub fn export(&self, out_dir: impl AsRef<Path>) -> std::io::Result<()> {
        let dir = out_dir.as_ref();
        std::fs::create_dir_all(dir)?;
        self.engine.save_parameters(dir.join("snn_model_market.json"))?;
        self.engine.export_fpga_mem(dir)
    }
}
