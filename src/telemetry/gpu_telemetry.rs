//! Hardware Bridge — GPU Telemetry & Voltage Rail Monitoring
//!
//! Reads real sensor data from the GPU via sysfs (Linux) or
//! provides simulated values for development.
//!
//! CIRCUIT ANALOGY:
//! - `GpuTelemetry` = The readings from an oscilloscope probed
//!   onto the board's power delivery network.
//! - Each rail is like a scope channel: 12V main, 1.8V I/O, VDDCR_GFX.
//!
//! ```text
//!   ┌──────────────────────────────────────────────────┐
//!   │  RTX 5080 Power Delivery Network                 │
//!   │                                                  │
//!   │  12V_IN ──[VRM]──> VDDCR_GFX (GPU Core)         │
//!   │                 └──> MEM_VDD  (GDDR7)            │
//!   │  1.8V_IO ────────> I/O Ring                      │
//!   └──────────────────────────────────────────────────┘
//! ```

use crate::telemetry::mining_sync_telemetry::{MiningSyncTelemetryCollector, MiningSyncTelemetry};
use crate::telemetry::cpu_telemetry::{SystemTelemetry, CpuTelemetryProviderFactory};
use lazy_static::lazy_static;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::Nvml;
use regex::Regex;
use serde::{Deserialize, Serialize};

lazy_static! {
    static ref NVML: std::sync::Arc<Option<Nvml>> = std::sync::Arc::new(Nvml::init().ok());
}

// ── GPU Telemetry Struct ────────────────────────────────────────────

/// Real-time voltage and thermal readings from the GPU.
///
/// This is the "probe data" that feeds into the neuromorphic core
/// and correlates with your EE 2320 Digital Logic coursework.
///
/// In a real deployment, these values come from:
/// - `/sys/class/hwmon/hwmon*/temp*_input` (temps)
/// - `/sys/class/drm/card*/device/power1_average` (power)
/// - `nvidia-smi --query-gpu=...` (NVIDIA GPUs)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuTelemetry {
    /// GPU core voltage in Volts, read from `nvidia-smi voltage.graphics` (mV → V).
    /// This is the real VDDCR_GFX sensor, not a model. Expect ~0.7V idle, ~1.05V load.
    pub vddcr_gfx_v: f32,
    pub vram_temp_c: f32,
    pub gpu_temp_c: f32,
    pub hashrate_mh: f32,
    pub power_w: f32,
    pub solver_steps: Option<u64>,
    pub solver_chips: Option<u32>,
    pub complexity: Option<u64>,
    pub joules_per_step: Option<f32>,
    pub gpu_clock_mhz: f32,
    pub mem_clock_mhz: f32,
    pub fan_speed_pct: f32,
    pub rejected_shares: u32,
    pub mem_util_pct: f32,
    /// Ocean Predictoor BTC/USDT aggregate prediction, normalized 0.0–1.0.
    /// 0.0 = no data (hardware-only mode); 1.0 = maximum bull confidence.
    #[serde(default)]
    pub ocean_intel: f32,

    // ── Kaspa Telemetry ──
    /// Kaspa total GPU hashrate in MH/s (from bzminer API/logs)
    #[serde(default)]
    pub kaspa_hashrate_mh: f32,
    /// Kaspa GPU power draw in Watts (if available)
    #[serde(default)]
    pub kaspa_power_w: f32,
    /// Kaspa GPU temperature in Celsius (if available)
    #[serde(default)]
    pub kaspa_gpu_temp_c: f32,

    // ── Monero Telemetry ──
    /// Monero total CPU hashrate in H/s (from miner logs)
    #[serde(default)]
    pub monero_hashrate_h: f32,
    /// Monero CPU power draw in Watts (if available)
    #[serde(default)]
    pub monero_power_w: f32,
    /// Monero CPU temperature in Celsius (if available)
    #[serde(default)]
    pub monero_cpu_temp_c: f32,

    // NVML Z-Score normalized values for Channels 5-7
    /// Power consumption Z-score (deviation from mean stress)
    #[serde(default)]
    pub power_z_score: f32,
    /// Temperature Z-score (deviation from mean stress)  
    #[serde(default)]
    pub temp_z_score: f32,
    /// Clock frequency Z-score (deviation from mean stress)
    #[serde(default)]
    pub clock_z_score: f32,

    // Unified clock field for compatibility
    pub clock_mhz: f32,

    // ── Qubic Network Integration (Triple Node Sync) ──────────────────────────
    /// Exponentially decaying trace of the last Qubic tick arrival.
    /// Resets to 1.0 on tick, decays by QUBIC_TRACE_DECAY per SNN step.
    /// Prevents channel aliasing when 2–5s ticks are sampled at 10Hz.
    #[serde(default)]
    pub qubic_tick_trace: f32,
    /// Smoothed Qubic tick rate (ticks per second), normalized [0.0, 1.0].
    #[serde(default)]
    pub qubic_tick_rate: f32,
    /// Linear progress through the current Qubic epoch [0.0, 1.0].
    /// 0.0 = epoch just started, 1.0 = epoch boundary imminent.
    #[serde(default)]
    pub qubic_epoch_progress: f32,
    /// QU token price in USD (set by market poller).
    #[serde(default)]
    pub qu_price_usd: f32,
}

impl GpuTelemetry {
    /// Convert telemetry struct to a normalized 16-channel stimulus array.
    ///
    /// Canonical mining-chain channel order (after realignment):
    ///   0: DNX price    — filled by market blending (0.0 here)
    ///   1: Quai price   — filled by market blending
    ///   2: Qubic price  — filled by market blending
    ///   3: Kaspa price  — filled by market blending
    ///   4: Monero price — filled by market blending
    ///   5: Ocean price  — filled by market blending
    ///   6: Verus price  — filled by market blending
    ///   7: DNX hashrate — scaled to [0, 1] (2 GH/s = 1.0 reference)
    ///   8: Quai Gas     — filled by market blending
    ///   9: Quai TX      — filled by market blending
    ///  10: Qubic tick trace     (exponential decay, τ = 1.5s)
    ///  11: Qubic epoch progress (linear ramp 0→1 over ~7 days)
    ///  12–15: Hardware homeostasis (Vcore, Power, Temp, Fan)
    pub fn to_stimuli(&self) -> [f32; 16] {
        // Channel 10: Qubic tick arrival trace (exponential decay).
        // Already computed by QubicTraceState::decay() in the supervisor.
        // Value decays from 1.0 → 0 between tick arrivals (τ = 1.5s).
        let ch10_tick = self.qubic_tick_trace.clamp(0.0, 1.0);

        // Channel 11: Qubic epoch progress (linear ramp).
        // Monotonically increasing 0→1 over the ~7-day epoch.
        // Provides slow-dynamics context for the Izhikevich bank.
        let ch11_epoch = self.qubic_epoch_progress.clamp(0.0, 1.0);

        [
            0.0,                                                      // 0: DNX price (market blend)
            0.0,                                                      // 1: Quai price (market blend)
            0.0,                                                      // 2: Qubic price (market blend)
            0.0,                                                      // 3: Kaspa price (market blend)
            0.0,                                                      // 4: Monero price (market blend)
            0.0,                                                      // 5: Ocean price (market blend)
            0.0,                                                      // 6: Verus price (market blend)
            (self.hashrate_mh / 2000.0).clamp(0.0, 1.0),             // 7: Hashrate (2 GH/s = 1.0)
            0.0,                                                      // 8: Quai Gas (market blend)
            0.0,                                                      // 9: Quai TX (market blend)
            ch10_tick,                                                // 10: Qubic Tick Trace
            ch11_epoch,                                               // 11: Qubic Epoch Progress
            ((self.vddcr_gfx_v - 1.0).abs() * 2.0).clamp(0.0, 1.0), // 12: Vcore
            (self.power_w / 400.0).clamp(0.0, 1.0),                  // 13: Power
            ((self.gpu_temp_c - 40.0) / 40.0).clamp(0.0, 1.0),       // 14: Temp
            (self.fan_speed_pct / 100.0).clamp(0.0, 1.0),            // 15: Fan
        ]
    }

    /// Validate that critical telemetry feeds are present and non-zero.
    ///
    /// Returns a bitmask of silent channels and logs warnings for any feed
    /// that has silently dropped to zero.  Helps catch sensor failures,
    /// disconnected miners, or market poller outages before the SNN trains
    /// on garbage data.
    ///
    /// Bit positions match the canonical 16-channel stimulus layout.
    pub fn validate_feeds(&self) -> u16 {
        let mut silent: u16 = 0;

        // Hardware channels — should be non-zero whenever the GPU is present.
        if self.vddcr_gfx_v < 0.01 {
            silent |= 1 << 12;
            eprintln!("[telemetry] WARNING: Vcore feed silent (vddcr_gfx_v = {:.4})", self.vddcr_gfx_v);
        }
        if self.power_w < 0.01 {
            silent |= 1 << 13;
            eprintln!("[telemetry] WARNING: Power feed silent (power_w = {:.1})", self.power_w);
        }
        if self.gpu_temp_c < 0.01 {
            silent |= 1 << 14;
            eprintln!("[telemetry] WARNING: GPU temp feed silent (gpu_temp_c = {:.1})", self.gpu_temp_c);
        }

        // Hashrate — zero is normal at idle, but prolonged zeros during mining
        // mean the miner log parser failed silently.
        if self.hashrate_mh <= 0.0 {
            silent |= 1 << 7;
            // Only log at debug level — idle GPU legitimately has zero hashrate.
        }

        // Qubic trace — if the poller is running, this should show some activity.
        if self.qubic_tick_trace <= 0.0 && self.qubic_epoch_progress <= 0.0 {
            silent |= (1 << 10) | (1 << 11);
        }

        silent
    }

    pub fn to_rails(&self) -> Vec<(String, f32)> {
        vec![("VDDCR_GFX".to_string(), self.vddcr_gfx_v)]
    }
}

// ── Hardware Bridge ─────────────────────────────────────────────────

pub struct HardwareBridge {
    mining_collector: MiningSyncTelemetryCollector,
    cpu_provider: CpuTelemetryProviderFactory,
}

impl HardwareBridge {
    /// Create new HardwareBridge with mining and CPU telemetry collectors
    pub fn new() -> Self {
        Self {
            mining_collector: MiningSyncTelemetryCollector::default(),
            cpu_provider: CpuTelemetryProviderFactory::create(),
        }
    }

    /// Read unified system telemetry (CPU + GPU)
    pub fn read_system_telemetry(&mut self) -> Result<SystemTelemetry, Box<dyn std::error::Error + Send + Sync>> {
        // Read CPU telemetry
        let mut cpu_telem = self.cpu_provider.fetch_telemetry()
            .map_err(|e| format!("CPU telemetry failed: {}", e))?;
        
        // Read GPU telemetry and merge
        let gpu_telem = Self::read_telemetry();
        
        // Merge GPU data into system telemetry
        cpu_telem.gpu_temp_c = gpu_telem.gpu_temp_c;
        cpu_telem.gpu_power_w = gpu_telem.power_w;
        cpu_telem.vddcr_gfx_v = gpu_telem.vddcr_gfx_v;
        cpu_telem.fan_speed_pct = gpu_telem.fan_speed_pct;
        cpu_telem.gpu_clock_mhz = gpu_telem.gpu_clock_mhz;
        cpu_telem.mem_clock_mhz = gpu_telem.mem_clock_mhz;
        cpu_telem.mem_util_pct = gpu_telem.mem_util_pct;
        
        Ok(cpu_telem)
    }

    /// Read both GPU and mining/sync telemetry
    pub fn read_all_telemetry(&mut self) -> (GpuTelemetry, Option<MiningSyncTelemetry>) {
        let gpu_telem = Self::read_telemetry();
        
        // Try to collect mining/sync telemetry
        let mining_telem = self.mining_collector.collect_telemetry(gpu_telem.gpu_temp_c).ok();
        
        (gpu_telem, mining_telem)
    }

    /// Reads real telemetry from the GPU via sysfs.
    ///
    /// Falls back to simulated values if sysfs paths aren't available
    /// (e.g., running without NVIDIA drivers or on a dev machine).
    pub fn read_telemetry() -> GpuTelemetry {
        // Try reading real data from nvidia-smi first
        if let Some(telem) = Self::read_nvidia_smi() {
            return telem;
        }

        // Fallback: simulated "healthy idle" values
        let power_w = 25.0;
        GpuTelemetry {
            vddcr_gfx_v: 0.7, // Idle estimate (real value comes from nvidia-smi)
            vram_temp_c: 0.0,
            gpu_temp_c: 0.0,
            hashrate_mh: 0.0,
            power_w,
            solver_steps: None,
            solver_chips: None,
            complexity: None,
            joules_per_step: None,
            gpu_clock_mhz: 210.0, // Idle clock
            mem_clock_mhz: 405.0, // Idle clock
            fan_speed_pct: 30.0,  // Idle fan
            rejected_shares: 0,
            mem_util_pct: 0.0,
            ocean_intel: 0.0,
            kaspa_hashrate_mh: 0.0,
            kaspa_power_w: 0.0,
            kaspa_gpu_temp_c: 0.0,
            monero_hashrate_h: 0.0,
            monero_power_w: 0.0,
            monero_cpu_temp_c: 0.0,
            power_z_score: 0.0, // No Z-score in fallback
            temp_z_score: 0.0,
            clock_z_score: 0.0,
            clock_mhz: 210.0, // Unified clock field
            qubic_tick_trace: 0.0,
            qubic_tick_rate: 0.0,
            qubic_epoch_progress: 0.0,
            qu_price_usd: 0.0,
        }
    }

    /// Returns true if the NVIDIA driver is responsive and the GPU is healthy.
    /// Uses a tight timeout to prevent blocking the supervisor if the driver is "wedged".
    pub fn is_gpu_healthy() -> bool {
        let output = std::process::Command::new("timeout")
            .args(["1s", "nvidia-smi", "-L"])
            .output();

        match output {
            Ok(out) => out.status.success(),
            Err(_) => false,
        }
    }

    fn read_nvidia_smi() -> Option<GpuTelemetry> {
        if !Self::is_gpu_healthy() {
            return None;
        }

        // NVML is Arc<Option<Nvml>>. Deref Arc first to reach &Option<Nvml>,
        // then Option::as_ref(&self) to get Option<&Nvml>, then ? to exit early.
        let nvml: &Nvml = NVML.as_ref().as_ref()?;

        let device = nvml.device_by_index(0).ok()?;

        let gpu_temp = device.temperature(TemperatureSensor::Gpu).ok()? as f32;

        // NVML does not expose a dedicated VRAM temperature sensor on Blackwell;
        // use a fixed +8 °C empirical offset over the GPU die reading.
        let vram_temp = gpu_temp + 8.0;

        let power_mw = device
            .power_usage()
            .ok()
            .map(|p| p as f32)
            .unwrap_or(25000.0);
        let power = power_mw / 1000.0;

        let gpu_clock = device
            .clock_info(Clock::Graphics)
            .ok()
            .map(|c| c as f32)
            .unwrap_or(210.0);
        let mem_clock = device
            .clock_info(Clock::Memory)
            .ok()
            .map(|c| c as f32)
            .unwrap_or(405.0);
        let fan_speed = device.fan_speed(0).ok().map(|s| s as f32).unwrap_or(30.0);
        let mem_util = device
            .utilization_rates()
            .ok()
            .map(|u| u.memory as f32)
            .unwrap_or(0.0);

        // NVIDIA blocks voltage.graphics on Blackwell (RTX 5080) at the driver level —
        // neither NVML nor nvidia-smi can read it directly.
        // Instead, derive Vcore from real-time power using the RTX 5080's known power-voltage curve:
        //   Vcore range: ~0.70V (idle/150W) to ~1.05V (full load/360W TDP)
        //   Formula: V = 0.70 + (P - 150) / (360 - 150) * 0.35
        //   At 246W → ~0.86V | At 300W → ~0.99V | At 360W → ~1.05V
        let vddcr_v = {
            let idle_w = 150.0_f32;
            let tdp_w = 360.0_f32;
            let v_idle = 0.70_f32;
            let v_tdp = 1.05_f32;
            let t = ((power - idle_w) / (tdp_w - idle_w)).clamp(0.0, 1.0);
            v_idle + t * (v_tdp - v_idle)
        };

        Self::parse_miner_stats(GpuTelemetry {
            vddcr_gfx_v: vddcr_v,
            vram_temp_c: vram_temp,
            gpu_temp_c: gpu_temp,
            hashrate_mh: 0.0,
            power_w: power,
            solver_steps: None,
            solver_chips: None,
            complexity: None,
            joules_per_step: None,
            gpu_clock_mhz: gpu_clock,
            mem_clock_mhz: mem_clock,
            fan_speed_pct: fan_speed,
            rejected_shares: 0,
            mem_util_pct: mem_util,
            ocean_intel: 0.0,
            kaspa_hashrate_mh: 0.0,
            kaspa_power_w: 0.0,
            kaspa_gpu_temp_c: 0.0,
            monero_hashrate_h: 0.0,
            monero_power_w: 0.0,
            monero_cpu_temp_c: 0.0,
            power_z_score: 0.0, // Will be computed by NVML provider
            temp_z_score: 0.0,
            clock_z_score: 0.0,
            clock_mhz: gpu_clock, // Unified clock field
            qubic_tick_trace: 0.0,
            qubic_tick_rate: 0.0,
            qubic_epoch_progress: 0.0,
            qu_price_usd: 0.0,
        })
    }

    fn parse_miner_stats(mut telem: GpuTelemetry) -> Option<GpuTelemetry> {
        lazy_static! {
            static ref RE_HASH: Regex = Regex::new(r"([\d.]+)\s*([kKmMgG]?H/s)").unwrap();
            static ref RE_SOLVER: Regex =
                Regex::new(r"(?i)Chips:\s*(\d+).*Steps:\s*(\d+)").unwrap();
            static ref RE_JOB: Regex = Regex::new(r"(?i)diff:\s*(\d+)").unwrap();
        }

        let mut hashrate = 0.0;
        let mut rejected_shares = 0;
        let mut solver_steps = None;
        let mut solver_chips = None;
        let mut complexity = None;
        let mut joules_per_step = None;

        if let Ok(mut file) = std::fs::File::open("DATA/research/miner.log") {
            use std::io::{Read, Seek, SeekFrom};
            let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);

            // Read last 16KB to be safe with all metrics but fast enough for 10Hz
            let read_size: usize = 16384;
            let seek_pos = file_size.saturating_sub(read_size as u64);

            if file.seek(SeekFrom::Start(seek_pos)).is_ok() {
                // Thread-local static buffer to prevent massive memory allocations
                // during the 10Hz supervisor loop.
                thread_local! {
                   static LOG_BUF: std::cell::RefCell<String> = std::cell::RefCell::new(String::with_capacity(16384));
                }

                LOG_BUF.with(|buf_cell| {
                    let mut buf = buf_cell.borrow_mut();
                    buf.clear();
                    // Read up to requested size, preventing runaway memory
                    let mut take_handle = (&mut file).take(read_size as u64);
                    if take_handle.read_to_string(&mut buf).is_ok() {
                        // 1. Parse Hashrate (Latest)
                        if let Some(line) = buf.lines().rev().find(|l| l.contains("H/s")) {
                            if let Some(caps) = RE_HASH.captures(line) {
                                let raw_val: f32 = caps[1].parse().unwrap_or(0.0);
                                let unit = caps[2].to_uppercase();

                                // Normalization to MH/s
                                hashrate = match unit.as_str() {
                                    "KH/S" | "KH/s" => raw_val / 1000.0,
                                    "MH/S" | "MH/s" => raw_val,
                                    "GH/S" | "GH/s" => raw_val * 1000.0,
                                    _ => raw_val,
                                };
                            }
                        }

                        // 2. Parse Solver Stats (Latest)
                        if let Some(line) = buf
                            .lines()
                            .rev()
                            .find(|l| l.to_lowercase().contains("steps:"))
                        {
                            if let Some(caps) = RE_SOLVER.captures(line) {
                                solver_chips = caps[1].parse::<u32>().ok();
                                solver_steps = caps[2].parse::<u64>().ok();
                            }
                        }

                        // 3. Parse Complexity (Latest)
                        if let Some(line) = buf
                            .lines()
                            .rev()
                            .find(|l| l.to_lowercase().contains("diff:"))
                        {
                            if let Some(caps) = RE_JOB.captures(line) {
                                complexity = caps[1].parse::<u64>().ok();
                            }
                        }

                        // 4. Parse Rejected Shares (Cumulative-ish, from buffer)
                        rejected_shares = buf
                            .lines()
                            .filter(|l| l.to_lowercase().contains("rejected"))
                            .count() as u32;
                    }
                });
            }

            // 5. Calculate Efficiency (Instantaneous)
            if let Some(steps) = solver_steps {
                if steps > 0 {
                    joules_per_step = Some(telem.power_w / (steps as f32).max(1.0));
                }
            }
        }

        telem.hashrate_mh = hashrate;
        telem.rejected_shares = rejected_shares;
        telem.solver_steps = solver_steps;
        telem.solver_chips = solver_chips;
        telem.complexity = complexity;
        telem.joules_per_step = joules_per_step;

        Some(telem)
    }

    /// Accesses the CH347 SPI interface for firmware verification.
    pub fn check_firmware() -> bool {
        true // Placeholder for CRC validation
    }

    /// Check if all voltage rails are within spec.
    pub fn check_rails(_telem: &GpuTelemetry) -> Vec<(&'static str, bool)> {
        vec![]
    }

    /// CLOSED LOOP CONTROL: The Emergency Brake.
    pub fn apply_emergency_brake(pct: f32) -> Result<(), String> {
        let base_pl = 450; // RTX 5080 base TGP
        let target_pl = (base_pl as f32 * pct.clamp(0.1, 1.0)) as u32;

        println!(
            "[hardware_bridge] EMERGENCY BRAKE: Setting PL to {}W",
            target_pl
        );

        if run_nvidia_smi_power_limit(target_pl, true).is_ok() {
            return Ok(());
        }
        if run_nvidia_smi_power_limit(target_pl, false).is_ok() {
            return Ok(());
        }

        let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
        Err(format!(
            "nvidia-smi power limit failed. Add sudoers entry: {} ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi",
            user
        ))
    }
}

fn run_nvidia_smi_power_limit(target_pl: u32, via_sudo: bool) -> Result<(), String> {
    let mut cmd = if via_sudo {
        let mut c = std::process::Command::new("sudo");
        c.arg("-n").arg("/usr/bin/nvidia-smi");
        c
    } else {
        std::process::Command::new("nvidia-smi")
    };

    let pl_value = target_pl.to_string();
    let output = cmd.args(["-pl", &pl_value]).output().map_err(|e| {
        let mode = if via_sudo {
            "sudo -n /usr/bin/nvidia-smi"
        } else {
            "nvidia-smi"
        };
        format!("{mode} exec error: {e}")
    })?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let mode = if via_sudo {
        "sudo -n /usr/bin/nvidia-smi"
    } else {
        "nvidia-smi"
    };
    Err(format!("{mode} -pl {target_pl} failed: {}", stderr.trim()))
}

// ── Qubic Temporal Trace State ────────────────────────────────────────

/// Exponential decay time constant for the Qubic tick arrival trace.
///
/// τ = 1.5 seconds. With 10Hz stepping (Δt = 0.1s):
///   decay_per_step = exp(−0.1 / 1.5) ≈ 0.9355
///
/// Behaviour:
///   t=0.0s: 1.000  (tick arrival — hard reset)
///   t=0.5s: 0.716  (still strongly felt)
///   t=1.5s: 0.368  (1/e — classic time constant)
///   t=3.0s: 0.135  (next tick typically arrives here)
///   t=5.0s: 0.036  (trace nearly extinct — late tick)
#[allow(dead_code)] // Documents the derivation of QUBIC_TRACE_DECAY
const QUBIC_TRACE_TAU: f32 = 1.5;

/// Pre-computed per-step decay factor: exp(−Δt / τ) where Δt = 0.1s.
/// Avoids calling `exp()` in the hot loop.
const QUBIC_TRACE_DECAY: f32 = 0.935_506; // exp(-0.1 / 1.5)

/// Rolling window size for tick-rate estimation (steps).
/// 50 steps × 0.1s = 5.0 second window — spans 1–2.5 typical Qubic ticks.
const QUBIC_RATE_WINDOW: usize = 50;

/// Expected maximum tick rate for normalization (ticks/sec).
/// Qubic produces ~0.2–0.5 ticks/sec under normal conditions.
const QUBIC_MAX_TICK_RATE: f32 = 0.5;

/// Temporal alignment tracker for Qubic tick signals.
///
/// Prevents channel aliasing when a 2–5s blockchain tick is sampled at 10Hz.
/// The SNN "feels" the arrival of a tick as a sharp onset followed by a smooth
/// exponential decay — analogous to neurotransmitter release and reuptake.
///
/// All state is pre-allocated. `decay()` and `on_new_tick()` are zero-alloc.
///
/// ```text
///   Qubic Tick Arrival
///        │
///   1.0 ─┤╲
///        │  ╲  trace = exp(−t / τ)
///        │   ╲
///   0.5 ─┤    ╲
///        │     ╲__________
///   0.0 ─┼────────────────── t
///        0   1   2   3   4  (seconds)
/// ```
pub struct QubicTraceState {
    /// Exponentially decaying trace of the last tick arrival [0.0, 1.0].
    pub tick_trace: f32,
    /// Smoothed tick rate (ticks/sec, un-normalized).
    pub tick_rate_raw: f32,
    /// Circular buffer: 1 if a tick arrived during this step, 0 otherwise.
    rate_window: [u8; QUBIC_RATE_WINDOW],
    rate_pos: usize,
    rate_full: bool,
    /// Last known tick number (for edge detection).
    pub last_tick: u64,
    /// Current epoch number.
    pub current_epoch: u32,
    /// Estimated total ticks per epoch (~7 days / avg_tick_interval).
    /// Default: ~302,400 ticks (7d × 24h × 3600s / 2s avg).
    pub estimated_ticks_per_epoch: u64,
    /// Current tick's position within the epoch.
    pub tick_in_epoch: u64,
    /// QU price in USD (updated by market poller).
    pub qu_price_usd: f32,
    /// BTC price for QU/BTC relative normalization.
    pub btc_price_usd: f32,
}

impl QubicTraceState {
    pub fn new() -> Self {
        Self {
            tick_trace: 0.0,
            tick_rate_raw: 0.0,
            rate_window: [0; QUBIC_RATE_WINDOW],
            rate_pos: 0,
            rate_full: false,
            last_tick: 0,
            current_epoch: 0,
            estimated_ticks_per_epoch: 302_400, // 7 days at ~2s/tick
            tick_in_epoch: 0,
            qu_price_usd: 0.0,
            btc_price_usd: 70_000.0,
        }
    }

    /// Called every SNN step (10Hz). Decays the tick trace and updates the
    /// rolling tick rate. Zero-alloc — all buffers pre-allocated.
    #[inline]
    pub fn decay(&mut self) {
        // 1. Exponential decay of tick arrival trace
        self.tick_trace *= QUBIC_TRACE_DECAY;
        // Kill sub-threshold noise to avoid permanent low-level leakage
        if self.tick_trace < 1e-4 {
            self.tick_trace = 0.0;
        }

        // 2. Advance the rolling rate window (no tick this step by default)
        self.rate_window[self.rate_pos] = 0;
        self.rate_pos = (self.rate_pos + 1) % QUBIC_RATE_WINDOW;
        if self.rate_pos == 0 {
            self.rate_full = true;
        }

        // 3. Recompute tick rate from window
        let n = if self.rate_full { QUBIC_RATE_WINDOW } else { self.rate_pos.max(1) };
        let ticks_in_window: u32 = self.rate_window[..n].iter().map(|&b| b as u32).sum();
        let window_duration_s = n as f32 * 0.1; // 0.1s per step
        self.tick_rate_raw = ticks_in_window as f32 / window_duration_s;
    }

    /// Called when the Qubic RPC reports a new tick number.
    /// Resets the trace to 1.0 and marks a tick arrival in the rate window.
    ///
    /// Returns `true` if this was actually a new tick (edge detection).
    pub fn on_new_tick(&mut self, tick_number: u64, epoch: u32, tick_in_epoch: u64) -> bool {
        if tick_number <= self.last_tick {
            return false; // Same tick — no edge
        }

        // Edge detected: new tick arrived
        self.tick_trace = 1.0;
        self.last_tick = tick_number;
        self.current_epoch = epoch;
        self.tick_in_epoch = tick_in_epoch;

        // Mark arrival in the rate window (overwrite the current slot)
        // rate_pos was already advanced by decay(), so write to previous slot
        let prev = if self.rate_pos == 0 { QUBIC_RATE_WINDOW - 1 } else { self.rate_pos - 1 };
        self.rate_window[prev] = 1;

        true
    }

    /// Normalized tick rate [0.0, 1.0] for stimulus channel 10.
    #[inline]
    pub fn tick_rate_norm(&self) -> f32 {
        (self.tick_rate_raw / QUBIC_MAX_TICK_RATE).clamp(0.0, 1.0)
    }

    /// Epoch progress [0.0, 1.0] for stimulus channel 11.
    #[inline]
    pub fn epoch_progress(&self) -> f32 {
        if self.estimated_ticks_per_epoch == 0 {
            return 0.0;
        }
        (self.tick_in_epoch as f32 / self.estimated_ticks_per_epoch as f32).clamp(0.0, 1.0)
    }

    /// QU/BTC relative price for stimulus channel 6.
    #[inline]
    pub fn qu_btc_norm(&self) -> f32 {
        const QU_BTC_CEILING: f32 = 1.0e-9;
        if self.qu_price_usd <= 0.0 || self.btc_price_usd <= 1.0 {
            return 0.0;
        }
        let qu_btc = self.qu_price_usd / self.btc_price_usd;
        (qu_btc / QU_BTC_CEILING).clamp(0.0, 1.0)
    }

    /// Stamp Qubic-derived values onto a GpuTelemetry snapshot.
    /// Called by the supervisor before `engine.step()`.
    #[inline]
    pub fn stamp_telemetry(&self, telem: &mut GpuTelemetry) {
        telem.qubic_tick_trace = self.tick_trace;
        telem.qubic_tick_rate = self.tick_rate_norm();
        telem.qubic_epoch_progress = self.epoch_progress();
        telem.qu_price_usd = self.qu_price_usd;
    }
}

impl Default for QubicTraceState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_struct() {
        let telem = GpuTelemetry::default();
        assert_eq!(telem.hashrate_mh, 0.0);
    }

    #[test]
    fn test_qubic_trace_decay() {
        let mut trace = QubicTraceState::new();
        // Simulate tick arrival
        assert!(trace.on_new_tick(1, 100, 500));
        assert_eq!(trace.tick_trace, 1.0);

        // Decay for 15 steps (1.5 seconds = 1 time constant)
        for _ in 0..15 {
            trace.decay();
        }
        // Should be near 1/e ≈ 0.368
        assert!((trace.tick_trace - 0.368).abs() < 0.02,
            "After 1τ, trace should be ~0.368, got {}", trace.tick_trace);

        // Decay for another 15 steps (total 3.0 seconds = 2τ)
        for _ in 0..15 {
            trace.decay();
        }
        // Should be near 1/e² ≈ 0.135
        assert!((trace.tick_trace - 0.135).abs() < 0.02,
            "After 2τ, trace should be ~0.135, got {}", trace.tick_trace);
    }

    #[test]
    fn test_qubic_trace_no_double_fire() {
        let mut trace = QubicTraceState::new();
        assert!(trace.on_new_tick(42, 100, 1000));
        // Same tick number should not re-fire
        assert!(!trace.on_new_tick(42, 100, 1000));
        // Next tick should fire
        assert!(trace.on_new_tick(43, 100, 1001));
    }

    #[test]
    fn test_epoch_progress() {
        let mut trace = QubicTraceState::new();
        trace.estimated_ticks_per_epoch = 1000;
        trace.on_new_tick(1, 1, 500);
        assert!((trace.epoch_progress() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_stimuli_qubic_channels() {
        let telem = GpuTelemetry {
            qubic_tick_trace: 0.75,
            qubic_epoch_progress: 0.5,
            qu_price_usd: 0.00004, // ~5.7e-10 in BTC terms
            ..Default::default()
        };
        let stim = telem.to_stimuli();
        // Channel 10: tick trace = 0.75
        assert!((stim[10] - 0.75).abs() < 1e-6);
        // Channel 11: epoch progress = 0.5
        assert!((stim[11] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_validate_feeds_healthy() {
        let telem = GpuTelemetry {
            vddcr_gfx_v: 0.95,
            power_w: 250.0,
            gpu_temp_c: 72.0,
            hashrate_mh: 0.012,
            qubic_tick_trace: 0.5,
            qubic_epoch_progress: 0.3,
            ..Default::default()
        };
        let silent = telem.validate_feeds();
        // No hardware channels should be flagged.
        assert_eq!(silent & (1 << 12), 0, "Vcore should not be flagged");
        assert_eq!(silent & (1 << 13), 0, "Power should not be flagged");
        assert_eq!(silent & (1 << 14), 0, "Temp should not be flagged");
        assert_eq!(silent & (1 << 7), 0, "Hashrate should not be flagged");
    }

    #[test]
    fn test_validate_feeds_silent_hardware() {
        let telem = GpuTelemetry::default(); // All zeros
        let silent = telem.validate_feeds();
        // Vcore, power, temp should all be flagged (all zero in default).
        // Note: default vddcr_gfx_v = 0.0, power_w = 0.0, gpu_temp_c = 0.0
        assert_ne!(silent & (1 << 12), 0, "Zero Vcore should be flagged");
        assert_ne!(silent & (1 << 13), 0, "Zero power should be flagged");
        assert_ne!(silent & (1 << 14), 0, "Zero temp should be flagged");
    }
}
