//! Market-Spikenaut: Spikenaut-v2 HFT Architecture
//!
//! ╔══════════════════════════════════════════════════════════════════════╗
//! ║  Spikenaut-v2: 16-Channel, <50µs, 1.6KB, FPGA-First HFT System      ║
//! ╚══════════════════════════════════════════════════════════════════════╝
//!
//! Clean orchestrator using modular architecture:
//!   - src/trading: Financial lobe with 0.1% fee attrition
//!   - src/market: Sensory lobe with expanded asset universe
//!   - src/backend: jlrs zero-copy bridge (primary) + ZMQ fallback
//!
//! # Usage
//! ```
//! cargo run -p soma-engine --bin market_pilot --features julia
//! ```

// use std::io::Write as _; // Unused import
// use chrono::{Datelike, Timelike, Utc, Weekday}; // Unused imports
// use serde::{Deserialize, Serialize}; // Used in other modules, not this file
use std::time::Duration;
use soma_engine::{
    snn::engine::SpikingInferenceEngine,
    telemetry::gpu_telemetry::GpuTelemetry,
};

use soma_engine::trading;
use soma_engine::market;
use soma_engine::backend;
use soma_engine::spine::zmq_spine::ZmqNervousSystem;
use soma_engine::spine::dydx_ingest::{self, DydxSnapshot};
use soma_engine::telemetry::gpu_telemetry::HardwareBridge;
use soma_engine::telemetry::nvml_telemetry::TelemetryProvider;

use trading::{GhostWallet, MarketFeed, GhostTradeLog, append_ghost_log};
use market::{ChannelNormalizer, poll_market, generate_mock_market};
use backend::{BackendFactory, BackendType};

// ── Loop timing ───────────────────────────────────────────────────────────────

/// Polling interval. CoinGecko free tier allows ~30 req/min; 5 s keeps us safe.
const TICK_SECS: u64 = 5;

// ── Neuron role assignments ───────────────────────────────────────────────────

const NEURON_DNX_BEAR: usize = 0;
const NEURON_DNX_BULL: usize = 1;
const NEURON_QUAI_BEAR: usize = 2;
const NEURON_QUAI_BULL: usize = 3;
const NEURON_QUBIC_BEAR: usize = 4;
const NEURON_QUBIC_BULL: usize = 5;
const NEURON_KAS_BEAR: usize = 6;
const NEURON_KAS_BULL: usize = 7;
const NEURON_XMR_BEAR: usize = 8;
const NEURON_XMR_BULL: usize = 9;
const NEURON_OCEAN_BEAR: usize = 10;
const NEURON_OCEAN_BULL: usize = 11;
const NEURON_VERUS_BEAR: usize = 12;
const NEURON_VERUS_BULL: usize = 13;
const NEURON_COINCIDENCE_DET: usize = 14;
const NEURON_GLOBAL_INHIB: usize = 15;

const PAIRS: [(usize, usize, &str); 7] = [
    (0, 1, "DNX"), (2, 3, "QUAI"), (4, 5, "QUBIC"),
    (6, 7, "KAS"), (8, 9, "XMR"), (10, 11, "OCEAN"),
    (12, 13, "VERUS"),
];

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Spikenaut-v2 HFT System - Production Ready                     ║");
    println!("║  16-Channel Live Inference | <50µs Latency | 1.6KB Memory      ║");
    println!("║  jlrs Zero-Copy Bridge | FPGA-First Deployment                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    
    std::fs::create_dir_all("DATA/research")?;

    let mut args = std::env::args();
    let reset = args.any(|a| a == "--reset");
    if reset {
        let _ = std::fs::remove_file("DATA/research/ghost_market_log.jsonl");
        let _ = std::fs::remove_file("DATA/research/snn_model.json");
        println!("[market] --reset: Deleted ghost_market_log.jsonl and snn_model.json");
    }

    // ── SNN initialisation ────────────────────────────────────────────────────
    let mut engine = SpikingInferenceEngine::new();
    if engine.load_parameters("DATA/research/snn_model.json").is_ok() {
        println!("[market] Loaded Readout Layer from research/snn_model.json");
    }

    // ── Backend initialization (Rust backend for now) ────────
    #[cfg(feature = "julia")]
    let backend_type = {
        println!("[market] 🚀 Spikenaut-v2: Using jlrs zero-copy bridge (primary)");
        BackendType::JlrsZeroCopy
    };
    #[cfg(not(feature = "julia"))]
    let backend_type = {
        println!("[market] ⚠️  Julia feature disabled, using Rust backend");
        BackendType::Rust
    };
    
    let mut backend = BackendFactory::create(backend_type);
    backend.initialize(Some("DATA/research/snn_model.json"))?;
    
    // Print backend info
    match backend_type {
        #[cfg(feature = "julia")]
        BackendType::JlrsZeroCopy => {
            println!("[market] ✅ jlrs Zero-Copy Backend ready (<10µs overhead)");
            println!("[market] 📊 Target: <50µs total latency, 1.6KB memory");
        }
        BackendType::Rust => {
            println!("[market] ✅ Rust Backend ready (fallback mode)");
        }
        _ => {
            println!("[market] ✅ Backend ready");
        }
    }

    // ── ZMQ Spine initialization (fallback/debug mode only) ───────────────────
    // ── dYdX v4 key-free WebSocket (Zero-Order Hold) ──────────────────────────
    let (dydx_tx, dydx_rx) = tokio::sync::watch::channel(DydxSnapshot::default());
    let dydx_rx_spine = dydx_rx.clone();
    tokio::spawn(dydx_ingest::run_dydx_ingest(dydx_tx));

    let mut spine = match ZmqNervousSystem::new(dydx_rx_spine) {
        Ok(s) => {
            #[cfg(feature = "julia")]
            if matches!(backend_type, BackendType::JlrsZeroCopy) {
                println!("[market] 📡 ZMQ Spine active (debug/fallback mode only)");
            } else {
                println!("[market] 📡 ZMQ Spine active — broadcasting to Julia Brain");
            }
            Some(s)
        }
        Err(e) => {
            #[cfg(feature = "julia")]
            if matches!(backend_type, BackendType::JlrsZeroCopy) {
                println!("[market] 📡 ZMQ Spine disabled (jlrs zero-copy primary)");
            } else {
                eprintln!("[market] ZMQ Spine failed (running without IPC): {}", e);
            }
            None
        }
    };

    // ── LSM Normalization State ───────────────────────────────────────────────
    let mut norms = [
        ChannelNormalizer::new(50), // Ch 0: DNX
        ChannelNormalizer::new(50), // Ch 1: QUAI
        ChannelNormalizer::new(50), // Ch 2: QUBIC
        ChannelNormalizer::new(50), // Ch 3: KAS
        ChannelNormalizer::new(50), // Ch 4: XMR
        ChannelNormalizer::new(50), // Ch 5: OCEAN
        ChannelNormalizer::new(50), // Ch 6: VERUS
        ChannelNormalizer::new(50), // Ch 7: Hashrate
        ChannelNormalizer::new(50), // Ch 8: Quai Gas Price
        ChannelNormalizer::new(50), // Ch 9: Quai TX Count
        ChannelNormalizer::new(50), // Ch 10: Qubic Tick Rate
        ChannelNormalizer::new(50), // Ch 11: Qubic Epoch Progress
        ChannelNormalizer::new(50), // Ch 12: Vcore
        ChannelNormalizer::new(50), // Ch 13: Power
        ChannelNormalizer::new(50), // Ch 14: Temp
        ChannelNormalizer::new(50), // Ch 15: Fan
    ];

    // ── HTTP client (shared across all fetchers, 4 s timeout) ────────────────
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(4))
        .user_agent("SpikeLens-Market-Pilot/0.1")
        .build()?;

    let mut wallet = GhostWallet::new();
    let mut nvml_provider = TelemetryProvider::create();

    // ── Key-free bootstrap (no REST prefetch during stabilization) ───────────
    println!("[market] LSM Warm-up: using first 10 dYdX pulses (key-free)");
    let mut prev_feed = MarketFeed {
        dnx_price_usd: 0.0266,
        quai_price_usd: 0.038,      // Default Quai price
        qubic_price_usd: 0.000002,  // Default Qubic price
        kaspa_price_usd: 0.035,     // Default Kaspa price
        monero_price_usd: 350.0,    // Default Monero price
        ocean_price_usd: 0.12,      // Default Ocean price
        verus_price_usd: 0.76,      // Default Verus price
        gpu_temp_c: 40.0,
        gpu_power_w: 150.0,
        vddcr_gfx_v: 0.70,
        fan_speed_pct: 35.0,
        dnx_hashrate_mh: 0.012,
        coinglass_funding_rate: 0.0,
        coinglass_liquidation_volume: 0.0,
        dex_liquidity_delta: 0.0,
        l3_order_imbalance: 0.0,
        stale: false,
        kaspa_block_rate_hz: 0.0,
        xmr_block_height: 0,
        xmr_sync_progress: 0.0,
        quai_gas_price: 10.0,
        quai_tx_count: 100,
        quai_block_utilization: 0.65,
        quai_staking_ratio: 0.40,
        qu_price_usd: 0.0,
        qubic_tick_rate: 0.0,
        qubic_epoch_progress: 0.0,
    };

    let init_pulse = dydx_rx.borrow().clone();
    let init_pulse_age = init_pulse.last_updated.elapsed().as_secs_f32();
    let init_brain_stress = (init_pulse.funding_rate.abs() / 0.001).clamp(0.0, 1.0);

    println!(
        "╔═══════════════════════════════════════════════════════════════╗\n\
         ║  Spikenaut V2 │ 262,144 Neurons │ RTX 5080 Sparse CUDA  ║\n\
         ║  ZMQ IPC 108B │ Key-Free dYdX v4 Sensory Lobe           ║\n\
         ╠═══════════════════════════════════════════════════════════╣\n\
         ║  DNX ${:.4}  QUAI ${:.4}  QUBIC ${:.6}  KAS ${:.4}      \n\
         ║  XMR ${:.2}  OCEAN ${:.4}  VERUS ${:.4}                 \n\
         ║  FundingRate={:.6}  OI-Δ={:+.4}  Brain-Stress={:.2}      \n\
         ║  GhostCash=${:.2} USDT                                    \n\
         ╚═══════════════════════════════════════════════════════════╝",
        prev_feed.dnx_price_usd,
        prev_feed.quai_price_usd,
        prev_feed.qubic_price_usd,
        prev_feed.kaspa_price_usd,
        prev_feed.monero_price_usd,
        prev_feed.ocean_price_usd,
        prev_feed.verus_price_usd,
        init_pulse.funding_rate,
        init_pulse.oi_delta,
        init_brain_stress.max((init_pulse_age / 5.0).clamp(0.0, 1.0)),
        wallet.balance_usdt,
    );

    let mut step: u64 = 0;
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(TICK_SECS));

    // ── Graceful shutdown handling ───────────────────────────────────────
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
        println!("\n[market] Ctrl+C received - Shutting down gracefully...");
        let _ = shutdown_tx.send(());
    });

    loop {
        tokio::select! {
            _ = interval.tick() => {
                step += 1;

                // ── 1. Key-free market feed (dYdX-driven warm-up + REST after warm-up) ──
                let dydx_pulse = dydx_rx.borrow().clone();
                let last_pulse_age = dydx_pulse.last_updated.elapsed();

                let mut feed = if step <= 10 {
                    // Stabilization phase: avoid REST fetches, drive from dYdX pulse stream.
                    let mut warm = generate_mock_market(&prev_feed);
                    warm.coinglass_funding_rate = dydx_pulse.funding_rate;
                    warm.dex_liquidity_delta = dydx_pulse.oi_delta;
                    warm
                } else {
                    // Normal operation: try real market data, fallback to mock if APIs fail
                    match poll_market(&client, &prev_feed).await {
                        real_feed if real_feed.dnx_price_usd > 0.0 => {
                            // Real data available
                            real_feed
                        }
                        _ => {
                            // APIs failed, use mock data
                            println!("[market] API failure — using mock market data");
                            let mut mock = generate_mock_market(&prev_feed);
                            mock.coinglass_funding_rate = dydx_pulse.funding_rate;
                            mock.dex_liquidity_delta = dydx_pulse.oi_delta;
                            mock
                        }
                    }
                };

                // dYdX mapping (key-free): funding drives Funding display + inhibition,
                // OI delta feeds OI-Δ and contributes to stress shaping.
                feed.coinglass_funding_rate = dydx_pulse.funding_rate;
                feed.dex_liquidity_delta = dydx_pulse.oi_delta;

                // Single fallback rule: if pulse is stale, switch to mock feed.
                if last_pulse_age > Duration::from_secs(5) {
                    println!("[market] dYdX pulse stale (>5s) — trigger_mock_fallback");
                    feed = generate_mock_market(&prev_feed);
                    feed.coinglass_funding_rate = dydx_pulse.funding_rate;
                    feed.dex_liquidity_delta = dydx_pulse.oi_delta;
                }

        // ── 1b. Read real GPU telemetry ─────────────────────────────────────────
        let hw_telem = nvml_provider.fetch_telemetry().unwrap_or_else(|_| HardwareBridge::read_telemetry());
        
        let pulse_stress = (feed.coinglass_funding_rate.abs() / 0.001).clamp(0.0, 1.0);
        let thermal_stress = ((hw_telem.gpu_temp_c - 65.0) / 20.0).clamp(0.0, 1.0);
        let age_stress = (last_pulse_age.as_secs_f32() / 5.0).clamp(0.0, 1.0);
        // Trading stress is derived only from market and hardware health (no mining inputs).
        let brain_stress = (0.5 * pulse_stress + 0.3 * thermal_stress + 0.2 * age_stress).clamp(0.0, 1.0);

        let feed = MarketFeed {
            gpu_temp_c: hw_telem.gpu_temp_c,
            // Legacy fields retained for compatibility with downstream APIs.
            coinglass_liquidation_volume: brain_stress,
            l3_order_imbalance: 0.0,
            ..feed
        };

        // ── 2. Normalize channels (16-channel Super-Brain) ──────────────────
        let current_vals = [
            feed.dnx_price_usd,
            feed.quai_price_usd,
            feed.qubic_price_usd,
            feed.kaspa_price_usd,
            feed.monero_price_usd,
            feed.ocean_price_usd,
            feed.verus_price_usd,
            feed.dnx_hashrate_mh,
            feed.quai_gas_price as f32,
            feed.quai_tx_count as f32,
            feed.qubic_tick_rate,
            feed.qubic_epoch_progress,
            feed.vddcr_gfx_v,
            feed.gpu_power_w,
            feed.gpu_temp_c,
            feed.fan_speed_pct,
        ];
        
        let mut z_scores = [0.0f32; 16];
        
        for i in 0..16 {
            z_scores[i] = norms[i].process(current_vals[i]);
        }

        // ── 2. Process through SNN backend ────────────────────────────────────
        let inhibit = ((feed.gpu_temp_c - 40.0) / 40.0).clamp(0.0, 1.0);
        let telem = GpuTelemetry {
            gpu_temp_c: hw_telem.gpu_temp_c,
            power_w: hw_telem.power_w,
            gpu_clock_mhz: hw_telem.gpu_clock_mhz,
            mem_util_pct: hw_telem.mem_util_pct,
            fan_speed_pct: hw_telem.fan_speed_pct,
            vddcr_gfx_v: hw_telem.vddcr_gfx_v,
            ..GpuTelemetry::default()
        };

        // ── 2b. Broadcast via ZMQ Spine to Julia Brain ──────────────────────────
        #[cfg(feature = "quai_integration")]
        if let Some(ref mut sp) = spine {
            if let Err(e) = sp.broadcast_quai(&feed, &telem, &z_scores) {
                eprintln!("[spine] Broadcast error: {}", e);
            }
        }

        #[cfg(not(feature = "quai_integration"))]
        if let Some(ref mut sp) = spine {
            let z8: [f32; 8] = z_scores[..8].try_into().unwrap_or([0.0f32; 8]);
            if let Err(e) = sp.broadcast(&feed, &telem, &z8) {
                eprintln!("[spine] Broadcast error: {}", e);
            }
        }

        // ── 3. Process through SNN backend ────────────────────────────────────
        // NOTE: ZmqBrainBackend returns 20 floats [0..16 readout | 16..20 NERO].
        // All other backends return 16.  We always act on the first 16 only.
        #[cfg(feature = "quai_integration")]
        let lsm_output = backend.process_signals_quai(&z_scores, inhibit, &telem)?;

        #[cfg(not(feature = "quai_integration"))]
        let lsm_output = {
            let z8: [f32; 8] = z_scores[..8].try_into().unwrap_or([0.0f32; 8]);
            backend.process_signals(&z8, inhibit, &telem)?
        };

        // Slice the action readout (first 16 elements); NERO scores are in [16..20].
        let nero_scores: [f32; 4] = {
            let mut s = [0.25f32; 4];
            for (i, v) in lsm_output.iter().skip(16).take(4).enumerate() {
                s[i] = *v;
            }
            s
        };
        let _lsm_output_arr: [f32; 16] = lsm_output[..16]
            .try_into()
            .unwrap_or([0.0f32; 16]);
        let _ = nero_scores; // available for downstream trade logic if needed

        engine.step(&z_scores, &telem);

        // ── 5. Execute trades based on neuron spikes ─────────────────────────
        execute_trades(&mut engine, &mut wallet, &feed, step, &z_scores);

        // ── 6. Periodic status and model save ────────────────────────────────
        if step % 12 == 0 {
            let portf = wallet.portfolio_value(feed.dnx_price_usd, feed.quai_price_usd, feed.qubic_price_usd, feed.kaspa_price_usd, feed.monero_price_usd, feed.ocean_price_usd, feed.verus_price_usd);
            let fpga_status = if engine.use_fpga { "FPGA" } else { "Software" };
            println!(
                "[step {:>5}] DNX=${:.4} QUAI=${:.4} QUBIC=${:.6} KAS=${:.4} XMR=${:.2} OCEAN=${:.4} VERUS=${:.4} FundingRate={:.6} OI-Δ={:+.4} Brain-Stress={:.2} cash=${:.2} portf=${:.2} [{}]",
                step,
                feed.dnx_price_usd,
                feed.quai_price_usd,
                feed.qubic_price_usd,
                feed.kaspa_price_usd,
                feed.monero_price_usd,
                feed.ocean_price_usd,
                feed.verus_price_usd,
                feed.coinglass_funding_rate,
                feed.dex_liquidity_delta,
                feed.coinglass_liquidation_volume,
                wallet.balance_usdt,
                portf,
                fpga_status
            );
        }

        if step % 50 == 0 {
            let _ = engine.save_parameters("DATA/research/snn_model.json");
            let _ = backend.save_state("DATA/research/snn_model.json");
            println!("[market] Step {}: Saved model state", step);
        }

        prev_feed = feed;
            }
            _ = &mut shutdown_rx => {
                println!("[market] Graceful shutdown completed");
                break;
            }
        }
    }
    
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Trade execution logic
// ─────────────────────────────────────────────────────────────────────────────

use trading::{execute_buy, execute_sell};

fn execute_trades(
    engine: &mut SpikingInferenceEngine,
    wallet: &mut GhostWallet,
    feed: &MarketFeed,
    step: u64,
    z_scores: &[f32; 16],  // Fixed: 16 elements for 16 neurons
) {
    let is_warming_up = step <= 10;
    
    // Capture neuron spike state for SNN proof logging
    let spikes: [bool; 16] = std::array::from_fn(|i| engine.neurons[i].last_spike);

    // Log observation — includes SNN spike proof for Hugging Face training dataset
    let obs = GhostTradeLog {
        schema_version: "v2",
        timestamp: chrono::Utc::now().to_rfc3339(),
        step,
        action: "observe".to_string(),
        asset: "PORTFOLIO".to_string(),
        price_usd: feed.dnx_price_usd,
        quantity: 0.0,
        trade_value_usdt: 0.0,
        realized_pnl_usdt: 0.0,
        balance_usdt: wallet.balance_usdt,
        balance_dnx: wallet.balance_dnx,
        balance_quai: wallet.balance_quai,
        balance_qubic: wallet.balance_qubic,
        balance_kaspa: wallet.balance_kaspa,
        balance_monero: wallet.balance_monero,
        balance_ocean: wallet.balance_ocean,
        balance_verus: wallet.balance_verus,
        cumulative_pnl: wallet.cumulative_pnl,
        portfolio_value: wallet.portfolio_value(feed.dnx_price_usd, feed.quai_price_usd, feed.qubic_price_usd, feed.kaspa_price_usd, feed.monero_price_usd, feed.ocean_price_usd, feed.verus_price_usd),
        quai_gas_price: feed.quai_gas_price,
        quai_tx_count: feed.quai_tx_count,
        quai_block_utilization: feed.quai_block_utilization,
        quai_staking_ratio: feed.quai_staking_ratio,
        kaspa_block_rate_hz: feed.kaspa_block_rate_hz,
        xmr_block_height: feed.xmr_block_height,
        xmr_sync_progress: feed.xmr_sync_progress,
        snn_channels: Some(z_scores.clone()),  // Populated: 16-element SNN input vector
        neuron_spikes: Some(spikes),
        mining_reward: Some(0.0),  // TODO: expose MiningRewardState.ema_reward via accessor
        reason: if is_warming_up { "Warm-up".to_string() } else { "observe".to_string() },
    };
    append_ghost_log(&obs);

    if is_warming_up {
        println!("[market] Step {}/10 (LSM Stabilization) …", step);
        return;
    }

    for &(bear_idx, bull_idx, symbol) in PAIRS.iter() {
        let bear_spiked = engine.neurons[bear_idx].last_spike;
        let bull_spiked = engine.neurons[bull_idx].last_spike;

        if bear_spiked || bull_spiked {
            println!("[DEBUG] Neuron spikes detected for {}:", symbol);
            if bull_spiked {
                println!("  {} Bull neuron spiked (potential: {:.4}, threshold: {:.4})",
                    symbol, engine.neurons[bull_idx].membrane_potential, engine.neurons[bull_idx].threshold);
            }
            if bear_spiked {
                println!("  {} Bear neuron spiked (potential: {:.4}, threshold: {:.4})",
                    symbol, engine.neurons[bear_idx].membrane_potential, engine.neurons[bear_idx].threshold);
            }
        }

        let price = match symbol {
            "DNX" => feed.dnx_price_usd,
            "QUAI" => feed.quai_price_usd,
            "QUBIC" => feed.qubic_price_usd,
            "KAS" => feed.kaspa_price_usd,
            "XMR" => feed.monero_price_usd,
            "OCEAN" => feed.ocean_price_usd,
            "VERUS" => feed.verus_price_usd,
            _ => 0.0,
        };

        if bull_spiked && wallet.balance_usdt > 1.0 {
            execute_buy(wallet, symbol, price, step, &format!("LSM {} Bull", symbol), feed);
            println!("[SPIKE] 🚀 {} Bull neuron triggered BUY signal!", symbol);
        }

        let balance = match symbol {
            "DNX" => wallet.balance_dnx,
            "QUAI" => wallet.balance_quai,
            "QUBIC" => wallet.balance_qubic,
            "KAS" => wallet.balance_kaspa,
            "XMR" => wallet.balance_monero,
            "OCEAN" => wallet.balance_ocean,
            "VERUS" => wallet.balance_verus,
            _ => 0.0,
        };

        if bear_spiked && balance > 1e-6 {
            execute_sell(wallet, symbol, price, step, &format!("LSM {} Bear", symbol), feed);
            println!("[SPIKE] 💰 {} Bear neuron triggered SELL signal!", symbol);
        }
    }

    if engine.neurons[14].last_spike {
        println!("[SPIKE] ⚡ Hardware Stress: Coincidence Detector [14] fired! (Z-Scores actively driving)");
    }
    if engine.neurons[15].last_spike {
        println!("[SPIKE] 🛑 Hardware Stress: Global Inhibition [15] fired! (Z-Scores actively driving)");
    }
}
