use soma_engine::telemetry::gpu_telemetry::{HardwareBridge, GpuTelemetry, QubicTraceState};
use soma_engine::ingest::triple_bridge::{spawn_triple_bridge, TripleSnapshot};

/// Qubic HTTP API at localhost:8099 (from qubic-http container in docker-compose.triple-node.yml).
const QUBIC_API_BASE: &str = "http://127.0.0.1:8099";
use synapse_link::fpga_bridge::FpgaBridge;
use soma_engine::snn::SpikingInferenceEngine;
use soma_engine::snn::mining_reward::reward_to_q8_8;
use soma_engine::market::poll_market;
use soma_engine::trading::MarketFeed;
use std::time::Duration;
use std::io::{self, Write};

/// SHIP OF THESEUS - LIVE SUPERVISOR v2.2 (Hyper-Debug)
/// 
/// If this doesn't print, then the binary is CRASHING during startup 
/// (likely a dynamic linker issue or global constructor panic).

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Instance Protection (Lockfile) ───────────────────────────
    struct LockGuard(&'static str);
    impl Drop for LockGuard {
        fn drop(&mut self) {
            // SAFETY: The lockfile is only used for instance protection and is safe to remove on shutdown.
            let _ = std::fs::remove_file(self.0);
        }
    }

    let lock_path = "/tmp/ship_supervisor.lock";
    
    if std::path::Path::new(lock_path).exists() {
        // SAFETY: Reading the lockfile is necessary to check for active instances.
        if let Ok(content) = std::fs::read_to_string(lock_path) {
            if let Ok(old_pid) = content.trim().parse::<u32>() {
                // Check if the process is still alive on Linux
                if std::path::Path::new(&format!("/proc/{}", old_pid)).exists() {
                    eprintln!("[supervisor] FATAL: Another brain instance is already active (PID: {}).", old_pid);
                    eprintln!("[supervisor] Kill it first or run 'rm {}' if you are sure.", lock_path);
                    std::process::exit(1);
                } else {
                    println!("[supervisor] Found stale lock (Owner PID {} is dead). Reclaiming...", old_pid);
                }
            }
        }
    }
    
    // SAFETY: Writing the PID to the lockfile is standard for instance protection.
    std::fs::write(lock_path, std::process::id().to_string())?;
    let _lock_guard = LockGuard(lock_path);

    // FORCE FLUSH at every step
    let _ = io::stdout().write_all(b"[supervisor] STARTING BINARY...\n");
    io::stdout().flush()?;
    
    println!("[supervisor] --- Eagle-Lander: Live Hardware Supervisor ---");
    io::stdout().flush()?;
    
    // Auto-detect port
    let ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2"];
    let mut bridge = None;

    println!("[supervisor] Scanning for Artix-7 FPGA on USB-UART...");
    io::stdout().flush()?;

    for p in &ports {
        print!("[supervisor] Testing {}... ", p);
        io::stdout().flush()?;
        match FpgaBridge::new(p) {
            Ok(b) => {
                println!("SUCCESS!");
                io::stdout().flush()?;
                bridge = Some(b);
                break;
            }
            Err(e) => {
                let hint = if e.to_string().to_lowercase().contains("busy") {
                    " — zombie process? run: sudo fuser -k /dev/ttyUSB* && sudo chmod 666 /dev/ttyUSB*"
                } else if e.to_string().to_lowercase().contains("permission") {
                    " — run: sudo chmod 666 /dev/ttyUSB*"
                } else {
                    ""
                };
                println!("FAILED ({}{})", e, hint);
                io::stdout().flush()?;
            }
        }
    }

    // If no FPGA is connected, run in software-only mode rather than exiting.
    // Telemetry, SNN learning, and IPC all continue normally.
    // FPGA pulses are skipped; the dashboard shows "--" for hardware readings.
    let fpga_connected = bridge.is_some();
    if !fpga_connected {
        eprintln!("[supervisor] WARNING: No Artix-7 found — running in software-only mode.");
        eprintln!("[supervisor] Ensure board is ON and run 'sudo chmod 666 /dev/ttyUSB*' to enable FPGA.");
        eprintln!("[supervisor] SNN learning and telemetry will continue without hardware pulses.");
        io::stderr().flush()?;
    }
    // Safe to unwrap: loop guards on fpga_connected before calling bridge methods.
    let mut bridge: Option<FpgaBridge> = bridge;

    println!("[supervisor] Loop established. Monitoring GPU stability...");
    io::stdout().flush()?;

    // ── Neuromorphic Context Engines ─────────────────────────────
    let mut engine = SpikingInferenceEngine::new();
    let research_path = "DATA/research/neuromorphic_data.jsonl";
    let model_path = "DATA/research/snn_model.json";

    if std::path::Path::new(model_path).exists() {
        println!("[supervisor] Found existing brain state. Loading parameters...");
        let _ = engine.load_parameters(model_path);
    }

    let researcher = soma_engine::ai::researcher::NeuromorphicResearcher::new(research_path);
    // SAFETY: Canonicalizing the path is safe here as it's used for logging startup info.
    let abs_path = std::fs::canonicalize(research_path).unwrap_or_else(|_| std::path::PathBuf::from(research_path));
    println!("[supervisor] Logging research to: {:?}", abs_path);

    // ── Market Feed (Quai Integration) ───────────────────────────
    let client = reqwest::Client::new();
    let mut market_feed = MarketFeed::default();
    let mut market_step = 0u64;
    const MARKET_UPDATE_INTERVAL: u64 = 50; // Update market data every 50 steps (5 seconds)

    // ── Qubic Triple Node Sync ──────────────────────────────────────
    let mut qubic_trace = QubicTraceState::new();
    const QUBIC_POLL_INTERVAL: u64 = 10; // Poll Qubic RPC every 10 steps (1 second)
    let triple_bridge_rx = spawn_triple_bridge();

    let mut step_count = 0u64;
    // Tick at 100 ms → 600 ticks = 60 s between FPGA re-scan attempts
    const RESCAN_INTERVAL_STEPS: u64 = 600;
    let mut last_rescan_step: u64 = 0;

    // ── IPC Server (Zero-Block UDP) ──────────────────────────────
    let udp_socket = std::net::UdpSocket::bind("127.0.0.1:9898").expect("FATAL: Failed to bind IPC socket port 9898");
    udp_socket.set_nonblocking(true).expect("FATAL: Failed to set non-blocking UDP");
    println!("[supervisor] IPC Server listening on UDP 127.0.0.1:9898");

    println!("[supervisor] Heartbeat active. Starting telemetry loop...");
    io::stdout().flush()?;

    loop {
        step_count += 1;
        // 0. Visual Heartbeat (Let the user know we're cycling)
        let heartbeat = match step_count % 4 {
            0 => "/",
            1 => "-",
            2 => "\\",
            _ => "|",
        };
        print!("\r[supervisor] Pulse {} | Step: {} ", heartbeat, step_count);
        let _ = io::stdout().flush();

        let bridge_snap = triple_bridge_rx.borrow().clone();

        // 1. Read real sensors
        print!("[DEBUG] Reading sensors... "); io::stdout().flush()?;
        let mut telem = HardwareBridge::read_telemetry();
        println!("DONE. Power: {}W", telem.power_w); io::stdout().flush()?;

        // 1b. Qubic Temporal Trace: decay every step, poll RPC periodically.
        //     This prevents channel aliasing (2-5s ticks sampled at 10Hz).
        qubic_trace.decay();
        if bridge_snap.qubic_tick_number > 0 {
            let tick_in_epoch =
                (bridge_snap.qubic_epoch_progress.clamp(0.0, 1.0) * qubic_trace.estimated_ticks_per_epoch as f32)
                    as u64;
            let _ = qubic_trace.on_new_tick(
                bridge_snap.qubic_tick_number,
                bridge_snap.qubic_epoch,
                tick_in_epoch,
            );
            if bridge_snap.qu_price_usd > 0.0 {
                qubic_trace.qu_price_usd = bridge_snap.qu_price_usd;
            }
        } else if step_count % QUBIC_POLL_INTERVAL == 0 {
            poll_qubic(&client, &mut qubic_trace).await;
        }
        // Stamp Qubic-derived channels onto telemetry before SNN step.
        qubic_trace.btc_price_usd = market_feed.dnx_price_usd;
        qubic_trace.stamp_telemetry(&mut telem);
        // Blend Neuraxon-derived chemistry into live modulators so parsed
        // log telemetry affects reward/stress/focus dynamics, not just display.
        blend_neuraxon_modulators(&mut engine, &bridge_snap);

        // 1c. Mining Efficiency Dopamine — compute before engine.step() so the
        //     blended learning rate reflects this tick's thermal/power/efficiency
        //     state.  CPU temp passed as None when SystemTelemetry is unavailable.
        let mining_da = engine.mining_reward.compute(&telem, None);
        engine.modulators.mining_dopamine = mining_da;

        // 2. Update Market Data (Quai + Crypto)
        market_step += 1;
        if market_step % MARKET_UPDATE_INTERVAL == 0 {
            print!("[DEBUG] Updating market data... "); io::stdout().flush()?;
            market_feed = poll_market(&client, &market_feed).await;
            let tick_number = bridge_snap.qubic_tick_number;
            let epoch = bridge_snap.qubic_epoch;
            println!("[supervisor] Qubic tick {} (epoch {}) — QU/BTC trace updated. Qubic Rate: {:.2} Hz | Gas: {} gwei | DNX Price: {:.2}", 
                tick_number, epoch, qubic_trace.tick_rate_raw, 
                     market_feed.quai_gas_price, market_feed.dnx_price_usd); 
            io::stdout().flush()?;
        }
        
        // 3. Map Telemetry + Market Data to Stimuli using learned weights
        print!("[DEBUG] Encoding stimuli... "); io::stdout().flush()?;
        let stimuli = encode_telemetry_with_market(&telem, &market_feed, &engine);
        println!("DONE."); io::stdout().flush()?;

        // 2.5 Process IPC JSON messages (Zero-Block)
        let mut buf = [0; 4096]; // Increased buffer size for larger JSON messages
        while let Ok((amt, src)) = udp_socket.recv_from(&mut buf) {
            if let Ok(msg) = std::str::from_utf8(&buf[..amt]) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(msg) {
                    match json["type"].as_str() {
                        Some("LearningReward") => {
                            let dopamine = json["dopamine_delta"].as_f64().unwrap_or(0.0) as f32;
                            let cortisol = json["cortisol_delta"].as_f64().unwrap_or(0.0) as f32;
                            let reason = json["reason"].as_str().unwrap_or("unknown");
                            print!("\n[IPC] Terminal AI injected reward (DA: +{:.3}, CORT: +{:.3}) | Reason: {}\n", dopamine, cortisol, reason);
                            let _ = io::stdout().flush();
                            engine.inject_learning_reward(dopamine, cortisol);
                        }
                        Some("GetNeuroState") => {
                            let state_json = serde_json::json!({
                                "dopamine": engine.modulators.dopamine,
                                "mining_dopamine": engine.modulators.mining_dopamine,
                                "cortisol": engine.modulators.cortisol,
                                "acetylcholine": engine.modulators.acetylcholine,
                                "tempo": engine.modulators.tempo,
                                "fpga_stress": engine.modulators.fpga_stress,
                                "global_step": engine.global_step,
                                "lif_spike_count": engine.neurons.iter().filter(|n| n.last_spike).count() as u8,
                                "iz_spike_count": engine.iz_neurons.iter().filter(|n| n.v > -50.0).count() as u8,
                                "neuraxon_tick_number": bridge_snap.neuraxon_tick_number,
                                "neuraxon_epoch": bridge_snap.neuraxon_epoch,
                                "neuraxon_its": bridge_snap.neuraxon_its,
                                "neuraxon_dopamine": bridge_snap.neuraxon_dopamine,
                                "neuraxon_serotonin": bridge_snap.neuraxon_serotonin,
                                "neuraxon_action_potential_threshold": bridge_snap.neuraxon_action_potential_threshold
                            });
                            if let Ok(encoded) = serde_json::to_string(&state_json) {
                                let _ = udp_socket.send_to(encoded.as_bytes(), src);
                            }
                        }
                        Some(unknown_type) => {
                            eprintln!("[IPC] Unknown message type received: {}", unknown_type);
                            eprintln!("[IPC] Full message: {}", msg);
                        }
                        None => {
                            eprintln!("[IPC] Missing 'type' field in message");
                            eprintln!("[IPC] Full message: {}", msg);
                        }
                    }
                } else {
                    eprintln!("[IPC] Failed to parse JSON message: {}", msg);
                }
            } else {
                eprintln!("[IPC] Failed to decode UTF-8 message from buffer");
            }
        }

        // 3.5. Update Local Simulation & Archive Telemetry
        // This MUST happen regardless of FPGA status so we don't lose history!
        let combined_stimuli = combine_telemetry_and_market(&telem, &market_feed);
        engine.step(&combined_stimuli, &telem);
        
        // Archive every step for deep verification
        match researcher.archive_snapshot(&telem, &engine, Some(&bridge_snap)) {
            Ok(_) => { 
                print!("[DEBUG] Snapshot archived. "); 
                let _ = io::stdout().flush();
            },
            Err(e) => eprintln!("\n[researcher] ARCHIVE FAILED: {}", e),
        }
        
        if step_count % 10 == 0 {
            print!("[DEBUG] Exporting continue context... ");
            let _ = io::stdout().flush();
            let _ = researcher.export_continue_context(&telem, &engine, Some(&bridge_snap));
            println!("DONE.");
        }

        // Every 100 steps write updated FPGA .mem files so that the latest
        // learned weights are available for a Vivado hot-reload at any time.
        // Run `source research/load_brain.tcl` in the Vivado TCL console to
        // commit these weights to the FPGA BRAM between mining sessions.
        if step_count % 100 == 0 {
            match engine.export_fpga_mem("research") {
                Ok(()) => {
                    print!("\n[brain] ✓ FPGA .mem updated (step {}) — reload via: source research/load_brain.tcl\n", step_count);
                }
                Err(e) => {
                    eprintln!("\n[brain] WARN: Could not write FPGA .mem files: {}", e);
                }
            }
        }
        let _ = io::stdout().flush();

        // 4. Pulse the FPGA Brain (Hardware Layer)
        // Skipped gracefully when running in software-only mode (no Basys3 connected).
        if let Some(ref mut hw_bridge) = bridge {
            print!("[DEBUG] Pulsing FPGA... "); io::stdout().flush()?;

            match hw_bridge.step_cluster(&stimuli) {
                Ok((potentials, spikes, switches)) => {
                    println!("DONE (Spikes: {:04X}, SW: {:04X})", spikes, switches);
                    
                    // 4a. Update 7-Segment Display based on mode (SW[15:14])
                    let mode = (switches >> 14) & 0x03;
                    match mode {
                        0 => { // NVDA Stock Price (nXX.X)
                            // Mock price for now; in production this would come from MarketFeed
                            let price = 135.24; 
                            let display_price = if price >= 100.0 { price - 100.0 } else { price };
                            let scaled = (display_price * 10.0) as u32;
                            let bcd = (0xE << 12) | // 'n'
                                      ((((scaled / 100) % 10) as u16) << 8) |
                                      ((((scaled / 10) % 10) as u16) << 4) |
                                      ((scaled % 10) as u16);
                            let _ = hw_bridge.set_display(bcd, 0x02); // nXX.X
                        }
                        1 => { // GPU Thermal Stress (XX.XC)
                            let temp = telem.gpu_temp_c; // e.g., 75.2
                            let scaled = (temp * 10.0) as u32;
                            let bcd = ((((scaled / 100) % 10) as u16) << 12) |
                                      ((((scaled / 10) % 10) as u16) << 8) |
                                      (((scaled % 10) as u16) << 4) |
                                      (0xC << 0); // 'C'
                            let _ = hw_bridge.set_display(bcd, 0x04); // XX.XC
                        }
                        2 => { // SNN Performance Error (PXX.X)
                            let error = 0.05; 
                            let scaled = (error * 1000.0) as u32; // 0.05 -> 05.0
                            let bcd = (0xF << 12) | // 'P'
                                      ((((scaled / 100) % 10) as u16) << 8) |
                                      ((((scaled / 10) % 10) as u16) << 4) |
                                      ((scaled % 10) as u16);
                            let _ = hw_bridge.set_display(bcd, 0x02); // PXX.X
                        }
                        3 | _ => { // Step counter / 10 (XXXX)
                            let val = (step_count / 10) as u32;
                            let bcd = ((((val / 1000) % 10) as u16) << 12) |
                                      ((((val / 100) % 10) as u16) << 8) |
                                      ((((val / 10) % 10) as u16) << 4) |
                                      ((val % 10) as u16);
                            let _ = hw_bridge.set_display(bcd, 0x00); // XXXX
                        }
                    }

                    // 4b. Send mining dopamine reward to FPGA STDP controller.
                    //     Blend event dopamine (0.3–1.0) with mining_dopamine
                    //     (clamp negatives to 0 — FPGA reward_scalar is unsigned).
                    //     Convert to Q8.8 via strict overflow-safe helper.
                    {
                        let mining_gate = (engine.modulators.mining_dopamine + 0.8) / 1.6; // [0, 1]
                        let combined_reward = (engine.modulators.dopamine * mining_gate).clamp(0.0, 1.0);
                        let reward_q8_8 = reward_to_q8_8(combined_reward);
                        let _ = hw_bridge.send_reward(reward_q8_8);
                    }

                    // 5. Closed-Loop Logic
                    if spikes == 0xFFFF {
                        println!("\n[!] FULL CLUSTER SPIKE on FPGA! Triggering Emergency Brake...");
                        match HardwareBridge::apply_emergency_brake(0.6) {
                            Ok(_) => println!("[supervisor] Brake applied successfully."),
                            Err(e) => eprintln!("[supervisor] BRAKE FAILURE: {}", e),
                        }
                        io::stdout().flush()?;
                    }
                    // 6. Visual Dashboard
                    print_dashboard(&telem, &potentials, spikes, heartbeat, &engine, Some(&bridge_snap));
                }
                Err(e) => {
                    eprintln!("\n[supervisor] FPGA Comms Lag: {} — dropping to software-only mode.", e);
                    eprintln!("[supervisor] Likely causes: (1) FPGA not programmed / power-cycled; \
                               (2) Vivado serial monitor still open (close it); \
                               (3) baud rate mismatch (firmware must be 115200). \
                               Run 'source research/load_brain.tcl' in Vivado to reload firmware.");
                    bridge = None; // Mark link as lost; re-scan will attempt recovery
                    last_rescan_step = step_count; // Reset timer so next scan is 60 s away
                    print_dashboard(&telem, &[0u16; 16], 0, heartbeat, &engine, Some(&bridge_snap));
                }
            }
        } else {
            // Software-only mode: show dashboard with zeroed hardware readings.
            print_dashboard(&telem, &[0u16; 16], 0, heartbeat, &engine, Some(&bridge_snap));

            // ── Periodic FPGA Re-scan ──────────────────────────────────
            // When the bridge is absent (never connected, or link was lost),
            // try to re-detect the Artix-7 every 60 seconds so a board that
            // was plugged in after startup gets picked up automatically.
            if step_count.saturating_sub(last_rescan_step) >= RESCAN_INTERVAL_STEPS {
                last_rescan_step = step_count;
                eprintln!("\n[supervisor] Attempting FPGA re-scan...");
                let ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2"];
                for p in &ports {
                    match FpgaBridge::new(p) {
                        Ok(b) => {
                            eprintln!("[supervisor] FPGA re-scan: found board on {} — resuming hardware mode.", p);
                            bridge = Some(b);
                            break;
                        }
                        Err(_) => {} // Not on this port — keep trying
                    }
                }
                if bridge.is_none() {
                    eprintln!("[supervisor] FPGA re-scan: no board found — next scan in 60 s.");
                }
                io::stderr().flush().ok();
            }
        }

        tokio::time::sleep(Duration::from_millis(100)).await; 
    }
}

fn encode_telemetry_with_market(t: &GpuTelemetry, market: &MarketFeed, engine: &SpikingInferenceEngine) -> [u16; 16] {
    let telem_stimuli = t.to_stimuli();
    let market_stimuli: [f32; 16] = [
        market.dnx_price_usd, market.quai_price_usd, market.qubic_price_usd,
        market.kaspa_price_usd, market.monero_price_usd, market.ocean_price_usd,
        market.verus_price_usd,
        market.dnx_hashrate_mh,
        market.quai_gas_price, market.quai_tx_count as f32,
        0.0, // Ch10: Qubic tick trace already in telem_stimuli
        0.0, // Ch11: Qubic epoch progress already in telem_stimuli
        market.vddcr_gfx_v, market.gpu_power_w, market.gpu_temp_c, market.fan_speed_pct,
    ];
    
    // Combine telemetry and market stimuli (weighted average)
    let mut combined_stimuli = [0.0f32; 16];
    for i in 0..16 {
        combined_stimuli[i] = telem_stimuli[i] * 0.7 + market_stimuli[i] * 0.3;
    }
    
    let stress_multiplier = (1.0_f32 - engine.modulators.cortisol).max(0.1_f32);
    let stimuli_scale = 0.45 * stress_multiplier * engine.modulators.tempo.max(0.5);

    let mut stims = [0u16; 16];
    for (i, neuron) in engine.neurons.iter().enumerate() {
        let ch = i % 16;
        let delta = combined_stimuli[ch];
        let surprise = 0.0; // Supervisor uses raw stimuli for hardware bridge
        let stim = delta.abs().clamp(0.0, 1.0);
        let current = neuron.weights[ch] * (stim + surprise) * stimuli_scale;
        
        stims[i] = (current * 256.0) as u16;
    }
    stims
}

fn combine_telemetry_and_market(t: &GpuTelemetry, market: &MarketFeed) -> [f32; 16] {
    let telem_stimuli = t.to_stimuli();
    let market_stimuli: [f32; 16] = [
        market.dnx_price_usd, market.quai_price_usd, market.qubic_price_usd,
        market.kaspa_price_usd, market.monero_price_usd, market.ocean_price_usd,
        market.verus_price_usd,
        market.dnx_hashrate_mh,
        market.quai_gas_price, market.quai_tx_count as f32,
        0.0, // Ch10: Qubic tick trace — driven by QubicTraceState
        0.0, // Ch11: Qubic epoch progress — driven by QubicTraceState
        market.vddcr_gfx_v, market.gpu_power_w, market.gpu_temp_c, market.fan_speed_pct,
    ];
    
    let mut combined = [0.0f32; 16];
    for i in 0..16 {
        combined[i] = telem_stimuli[i] * 0.7 + market_stimuli[i] * 0.3;
    }
    combined
}

fn print_dashboard(
    t: &GpuTelemetry,
    _pots: &[u16],
    spikes: u16,
    hb: &str,
    engine: &SpikingInferenceEngine,
    bridge_snap: Option<&TripleSnapshot>,
) {
    let hash_display = if t.hashrate_mh > 0.0 && t.hashrate_mh < 1.0 {
        format!("{:5.2}K", t.hashrate_mh * 1000.0)
    } else {
        format!("{:5.2}M", t.hashrate_mh)
    };

    let m = &engine.modulators;
    let qb = if t.qubic_tick_trace > 0.01 {
        format!("QT:{:.2}", t.qubic_tick_trace)
    } else {
        "QT:--".to_string()
    };

    let nx = bridge_snap
        .filter(|s| s.neuraxon_tick_number > 0)
        .map(|s| {
            format!(
                " NX:T{} DA:{:.3} 5HT:{:.3}",
                s.neuraxon_tick_number, s.neuraxon_dopamine, s.neuraxon_serotonin
            )
        })
        .unwrap_or_default();

    print!("\r[Live {}] Hash: {} | Pwr: {:5.1}W | Spikes: {:04X} | DA: {:.2} MDA: {:+.2} CORT: {:.2} ACh: {:.2} Tmp: {:.2} {}{} ",
           hb, hash_display, t.power_w, spikes, m.dopamine, m.mining_dopamine, m.cortisol, m.acetylcholine, m.tempo, qb, nx);
    let _ = io::stdout().flush();
}

fn blend_neuraxon_modulators(engine: &mut SpikingInferenceEngine, snap: &TripleSnapshot) {
    if snap.neuraxon_tick_number == 0 {
        return;
    }

    let alpha = 0.08_f32;
    let nx_dopa = snap.neuraxon_dopamine.clamp(0.0, 1.0);
    let nx_serotonin = snap.neuraxon_serotonin.clamp(0.0, 1.0);
    let nx_its = (snap.neuraxon_its / 2000.0).clamp(0.0, 1.0);
    let ap_target_tempo = if snap.neuraxon_action_potential_threshold > 0.0 {
        (1.0 + (0.5 - snap.neuraxon_action_potential_threshold.clamp(0.0, 1.0)) * 0.25)
            .clamp(0.75, 1.25)
    } else {
        (0.85 + nx_its * 0.4).clamp(0.75, 1.25)
    };

    let m = &mut engine.modulators;
    m.dopamine = ((1.0 - alpha) * m.dopamine + alpha * nx_dopa).clamp(0.0, 1.0);
    // Higher serotonin softly reduces stress target.
    let cortisol_target = ((1.0 - nx_serotonin) * 0.6).clamp(0.0, 1.0);
    m.cortisol = ((1.0 - alpha) * m.cortisol + alpha * cortisol_target).clamp(0.0, 1.0);
    m.acetylcholine = ((1.0 - alpha) * m.acetylcholine + alpha * nx_its).clamp(0.0, 1.0);
    m.tempo = ((1.0 - alpha) * m.tempo + alpha * ap_target_tempo).clamp(0.5, 2.0);
}

/// Poll the Qubic HTTP API for current tick/epoch status.
///
/// Called every QUBIC_POLL_INTERVAL steps (1 second). Non-blocking:
/// if the API is unreachable, the trace simply continues decaying
/// and the SNN gracefully degrades (qubic_stability → 1.0 baseline).
///
/// Endpoint: GET /v1/status → { "lastProcessedTick": { "tickNumber": N }, "epoch": E }
///
/// This function is zero-alloc in the hot path (reqwest reuses the client).
async fn poll_qubic(client: &reqwest::Client, trace: &mut QubicTraceState) {
    let url = format!("{}/v1/status", QUBIC_API_BASE);

    // Fire-and-forget with 500ms timeout — must not block the 100ms loop.
    let resp = match tokio::time::timeout(
        Duration::from_millis(500),
        client.get(&url).send(),
    ).await {
        Ok(Ok(r)) => r,
        Ok(Err(_)) | Err(_) => return, // Network error or timeout — silent degradation
    };

    let json: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(_) => return,
    };

    // Parse tick number and epoch from the Qubic HTTP API response.
    // The exact JSON shape depends on the qubic-http version; we handle
    // both flat and nested formats gracefully.
    let tick_number = json["lastProcessedTick"]["tickNumber"]
        .as_u64()
        .or_else(|| json["currentTick"].as_u64())
        .unwrap_or(0);
    let epoch = json["epoch"]
        .as_u64()
        .or_else(|| json["lastProcessedTick"]["epoch"].as_u64())
        .unwrap_or(0) as u32;

    if tick_number > 0 {
        // Estimate tick-in-epoch: Qubic epochs reset the tick counter,
        // but the API may return an absolute tick. Use modular arithmetic
        // against the estimated epoch length.
        let tick_in_epoch = if trace.estimated_ticks_per_epoch > 0 {
            tick_number % trace.estimated_ticks_per_epoch
        } else {
            tick_number
        };
        trace.on_new_tick(tick_number, epoch, tick_in_epoch);
    }
}
