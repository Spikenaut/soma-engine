//! Mining Telemetry Integration
//! 
//! Unified telemetry system for Quai and Qubic mining operations
//! Integrates mining stats into SNN trading system

use std::process::Command;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynexStats {
    /// Reported hashrate in MH/s (onezerominer output, normalised).
    pub hashrate_mh_s: f64,
    pub shares_accepted: u64,
    pub shares_rejected: u64,
    pub uptime_seconds: u64,
    pub is_active: bool,
    pub solver_steps: Option<u64>,
    pub solver_chips: Option<u32>,
    pub complexity: Option<u64>,
    pub joules_per_step: Option<f32>,
}

impl Default for DynexStats {
    fn default() -> Self {
        Self {
            hashrate_mh_s: 0.0,
            shares_accepted: 0,
            shares_rejected: 0,
            uptime_seconds: 0,
            is_active: false,
            solver_steps: None,
            solver_chips: None,
            complexity: None,
            joules_per_step: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MiningStats {
    pub dynex: DynexStats,
    pub quai: QuaiStats,
    pub qubic: QubicStats,
    pub monero: MoneroStats,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuaiStats {
    pub hashrate_mh_s: f64,
    pub shares_accepted: u64,
    pub shares_rejected: u64,
    pub difficulty: f64,
    pub pool_difficulty: f64,
    pub uptime_seconds: u64,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QubicStats {
    pub hashrate_kh_s: f64,
    pub solutions_found: u64,
    pub current_tick: u32,
    pub peers_connected: u16,
    pub uptime_seconds: u64,
    pub is_active: bool,
    pub epoch_progress: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoneroStats {
    pub hashrate_h_s: f64,
    pub shares_accepted: u64,
    pub shares_rejected: u64,
    pub uptime_seconds: u64,
    pub is_active: bool,
}

impl Default for MoneroStats {
    fn default() -> Self {
        Self {
            hashrate_h_s: 0.0,
            shares_accepted: 0,
            shares_rejected: 0,
            uptime_seconds: 0,
            is_active: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MiningTelemetry {
    #[serde(skip)]
    pub last_update: Instant,
    pub stats: MiningStats,
}

impl MiningTelemetry {
    pub fn new() -> Self {
        Self {
            last_update: Instant::now(),
            stats: MiningStats {
                dynex: DynexStats::default(),
                quai: QuaiStats {
                    hashrate_mh_s: 0.0,
                    shares_accepted: 0,
                    shares_rejected: 0,
                    difficulty: 0.0,
                    pool_difficulty: 0.0,
                    uptime_seconds: 0,
                    is_active: false,
                },
                qubic: QubicStats {
                    hashrate_kh_s: 0.0,
                    solutions_found: 0,
                    current_tick: 0,
                    peers_connected: 0,
                    uptime_seconds: 0,
                    is_active: false,
                    epoch_progress: 0.0,
                },
                monero: MoneroStats::default(),
                timestamp: chrono::Utc::now(),
            },
        }
    }

    pub fn update(&mut self) -> Result<()> {
        // Update every 2 seconds (was 5s — too slow to detect miner death)
        if self.last_update.elapsed() < Duration::from_secs(2) {
            return Ok(());
        }

        self.stats.quai = self.get_quai_stats()?;
        self.stats.dynex = self.get_dynex_stats()?;
        self.stats.qubic = self.get_qubic_stats()?;
        self.stats.monero = self.get_monero_stats()?;
        self.stats.timestamp = chrono::Utc::now();
        self.last_update = Instant::now();

        Ok(())
    }

    fn get_quai_stats(&self) -> Result<QuaiStats> {
        // Check if rigel process is running
        let output = Command::new("pgrep")
            .args(&["-f", "rigel.*quai"])
            .output()
            .ok();

        let is_active = output.map(|o| !o.stdout.is_empty()).unwrap_or(false);

        if !is_active {
            return Ok(QuaiStats {
                hashrate_mh_s: 0.0,
                shares_accepted: 0,
                shares_rejected: 0,
                difficulty: 0.0,
                pool_difficulty: 0.0,
                uptime_seconds: 0,
                is_active: false,
            });
        }

        // Parse mining log for stats
        let base = std::env::var("SHIP_WORK_DIR")
            .unwrap_or_else(|_| format!("{}/ship_of_theseus_rs", std::env::var("HOME").unwrap_or_default()));
        let log_path = format!("{}/BLOCKCHAIN/mining/logs/quai_mining.log", base);
        let log_content = std::fs::read_to_string(&log_path).unwrap_or_default();

        let mut hashrate = 0.0f64;
        let mut shares_accepted = 0u64;
        let mut shares_rejected = 0u64;

        for line in log_content.lines().rev().take(50) {
            if hashrate == 0.0 && line.contains("hashrate") {
                if let Some(rate_str) = line.split_whitespace()
                    .find(|s| s.contains("MH/s") || s.contains("mh/s")) {
                    if let Ok(rate) = rate_str.trim_end_matches("MH/s").trim_end_matches("mh/s").parse::<f64>() {
                        hashrate = rate;
                    }
                }
            }
            if line.contains("share accepted") {
                shares_accepted += 1;
            }
            if line.contains("share rejected") {
                shares_rejected += 1;
            }
        }

        Ok(QuaiStats {
            hashrate_mh_s: hashrate,
            shares_accepted,
            shares_rejected,
            difficulty: 0.0,
            pool_difficulty: 0.0,
            uptime_seconds: 0,
            // Only active if we actually parsed a non-zero hashrate
            is_active: hashrate > 0.0,
        })
    }

    fn get_qubic_stats(&self) -> Result<QubicStats> {
        // Check if qubic-core process is running
        let output = Command::new("pgrep")
            .args(&["-f", "qubic-core"])
            .output()
            .ok();

        let is_active = output.map(|o| !o.stdout.is_empty()).unwrap_or(false);

        if !is_active {
            return Ok(QubicStats {
                hashrate_kh_s: 0.0,
                solutions_found: 0,
                current_tick: 0,
                peers_connected: 0,
                uptime_seconds: 0,
                is_active: false,
                epoch_progress: 0.0,
            });
        }

        // Parse Qubic log for stats
        let base = std::env::var("SHIP_WORK_DIR")
            .unwrap_or_else(|_| format!("{}/ship_of_theseus_rs", std::env::var("HOME").unwrap_or_default()));
        let log_path = format!("{}/BLOCKCHAIN/mining/nodes/Qubic/qubic_core.log", base);
        let log_content = std::fs::read_to_string(&log_path).unwrap_or_default();

        let mut hashrate = 0.0f64;
        let mut solutions = 0u64;
        let mut current_tick: u32 = 0;
        let mut peers = 0u16;
        let mut epoch_progress = 0.0f32;

        for line in log_content.lines().rev().take(50) {
            if hashrate == 0.0 && line.contains("hashrate") {
                if let Some(rate_str) = line.split_whitespace()
                    .find(|s| s.contains("kH/s") || s.contains("KH/s")) {
                    let clean = rate_str.trim_end_matches("kH/s").trim_end_matches("KH/s");
                    if let Ok(rate) = clean.parse::<f64>() {
                        hashrate = rate;
                    }
                }
            }
            if line.contains("solution") || line.contains("Solution") {
                solutions += 1;
            }
            // rev().take(50) scans newest-first; break on first tick match to get most recent value
            if current_tick == 0 && line.contains("tick") {
                if let Some(tick_str) = line.split_whitespace()
                    .find(|s| s.parse::<u32>().is_ok()) {
                    if let Ok(tick) = tick_str.parse() {
                        current_tick = tick;
                        // Calculate epoch progress based on current_tick (assuming 302_400 ticks per epoch)
                        epoch_progress = (current_tick as f32 / 302_400.0).clamp(0.0, 1.0);
                    }
                }
            }
            if peers == 0 && (line.contains("peers") || line.contains("connected")) {
                if let Some(peer_str) = line.split_whitespace()
                    .find(|s| s.parse::<u16>().is_ok()) {
                    if let Ok(peer_count) = peer_str.parse() {
                        peers = peer_count;
                    }
                }
            }
        }

        Ok(QubicStats {
            hashrate_kh_s: hashrate,
            solutions_found: solutions,
            current_tick,
            peers_connected: peers,
            uptime_seconds: 0,
            // Only active if we actually parsed a non-zero hashrate
            is_active: hashrate > 0.0,
            epoch_progress,
        })
    }

    fn get_dynex_stats(&self) -> Result<DynexStats> {
        let output = Command::new("pgrep")
            .args(&["-f", "onezerominer"])
            .output()
            .ok();

        let is_active = output.map(|o| !o.stdout.is_empty()).unwrap_or(false);

        if !is_active {
            return Ok(DynexStats {
                hashrate_mh_s: 0.0,
                shares_accepted: 0,
                shares_rejected: 0,
                uptime_seconds: 0,
                is_active: false,
                solver_steps: None,
                solver_chips: None,
                complexity: None,
                joules_per_step: None,
            });
        }

        let base = std::env::var("SHIP_WORK_DIR")
            .unwrap_or_else(|_| format!("{}/ship_of_theseus_rs", std::env::var("HOME").unwrap_or_default()));
        let log_path = format!("{}/BLOCKCHAIN/mining/logs/dynex_mining.log", base);
        let log_content = std::fs::read_to_string(&log_path).unwrap_or_default();

        let mut hashrate = 0.0f64;
        let mut shares_accepted = 0u64;
        let mut shares_rejected = 0u64;
        let mut solver_steps = None;
        let mut solver_chips = None;
        let mut complexity = None;
        let mut joules_per_step = None;

        for line in log_content.lines().rev().take(50) {
            if hashrate == 0.0 && line.contains("hashrate") {
                if let Some(rate_str) = line.split_whitespace()
                    .find(|s| s.contains("MH/s") || s.contains("mh/s")) {
                    if let Ok(rate) = rate_str.trim_end_matches("MH/s").trim_end_matches("mh/s").parse::<f64>() {
                        hashrate = rate;
                    }
                }
            }
            if line.contains("share accepted") {
                shares_accepted += 1;
            }
            if line.contains("share rejected") {
                shares_rejected += 1;
            }
            if solver_steps.is_none() && line.contains("Solver Steps:") {
                if let Some(steps_str) = line.split("Solver Steps:").nth(1).and_then(|s| s.trim().split_whitespace().next()) {
                    if let Ok(steps) = steps_str.parse::<u64>() {
                        solver_steps = Some(steps);
                    }
                }
            }
            if solver_chips.is_none() && line.contains("Solver Chips:") {
                if let Some(chips_str) = line.split("Solver Chips:").nth(1).and_then(|s| s.trim().split_whitespace().next()) {
                    if let Ok(chips) = chips_str.parse::<u32>() {
                        solver_chips = Some(chips);
                    }
                }
            }
            if complexity.is_none() && line.contains("Complexity:") {
                if let Some(comp_str) = line.split("Complexity:").nth(1).and_then(|s| s.trim().split_whitespace().next()) {
                    if let Ok(comp) = comp_str.parse::<u64>() {
                        complexity = Some(comp);
                    }
                }
            }
            if joules_per_step.is_none() && line.contains("Joules/Step:") {
                if let Some(joules_str) = line.split("Joules/Step:").nth(1).and_then(|s| s.trim().split_whitespace().next()) {
                    if let Ok(joules) = joules_str.parse::<f32>() {
                        joules_per_step = Some(joules);
                    }
                }
            }
        }

        Ok(DynexStats {
            hashrate_mh_s: hashrate,
            shares_accepted,
            shares_rejected,
            uptime_seconds: 0,
            is_active: hashrate > 0.0,
            solver_steps,
            solver_chips,
            complexity,
            joules_per_step,
        })
    }

    fn get_monero_stats(&self) -> Result<MoneroStats> {
        // Check if XMRig process (part of qli-Client) is running
        let output = Command::new("pgrep")
            .args(&["-f", "xmrig"])
            .output()
            .ok();

        let is_active = output.map(|o| !o.stdout.is_empty()).unwrap_or(false);

        if !is_active {
            return Ok(MoneroStats {
                hashrate_h_s: 0.0,
                shares_accepted: 0,
                shares_rejected: 0,
                uptime_seconds: 0,
                is_active: false,
            });
        }

        // Parse XMRig log for stats
        let base = std::env::var("SHIP_WORK_DIR")
            .unwrap_or_else(|_| format!("{}/ship_of_theseus_rs", std::env::var("HOME").unwrap_or_default()));
        let log_path = format!("{}/BLOCKCHAIN/mining/logs/monero_mining.log", base);
        let log_content = std::fs::read_to_string(&log_path).unwrap_or_default();

        let mut hashrate = 0.0f64;
        let mut shares_accepted = 0u64;
        let mut shares_rejected = 0u64;

        for line in log_content.lines().rev().take(50) {
            // Example log line: "[2026-03-26 20:00:44.996] [INFO] hashrate: 1234.5 H/s, shares: 100/0, uptime: 120s"
            if hashrate == 0.0 && line.contains("hashrate") {
                if let Some(rate_str) = line.split("hashrate:").nth(1).and_then(|s| s.split_whitespace().next()) {
                    if let Ok(rate) = rate_str.parse::<f64>() {
                        hashrate = rate;
                    }
                }
            }
            if line.contains("shares:") {
                if let Some(shares_str) = line.split("shares:").nth(1).and_then(|s| s.split_whitespace().next()) {
                    let parts: Vec<&str> = shares_str.split('/').collect();
                    if parts.len() == 2 {
                        if let (Ok(accepted), Ok(rejected)) = (parts[0].parse::<u64>(), parts[1].parse::<u64>()) {
                            shares_accepted = accepted;
                            shares_rejected = rejected;
                        }
                    }
                }
            }
        }

        Ok(MoneroStats {
            hashrate_h_s: hashrate,
            shares_accepted,
            shares_rejected,
            uptime_seconds: 0, // XMRig logs don't typically provide uptime directly in this format
            is_active: hashrate > 0.0,
        })
    }

    pub fn get_stats(&self) -> &MiningStats {
        &self.stats
    }

    pub fn get_mining_power_signal(&self) -> f64 {
        // RTX 5080 typical peaks: Dynex ~1.5 MH/s (neuromorphic), Quai ~50 MH/s, Qubic ~200 kH/s (CPU)
        const DYNEX_MAX_MHS: f64 = 1.5;
        const QUAI_MAX_MHS: f64  = 50.0;
        const QUBIC_MAX_KHS: f64 = 200.0;

        let dynex_power = if self.stats.dynex.is_active {
            (self.stats.dynex.hashrate_mh_s / DYNEX_MAX_MHS).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let quai_power = if self.stats.quai.is_active {
            (self.stats.quai.hashrate_mh_s / QUAI_MAX_MHS).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let qubic_power = if self.stats.qubic.is_active {
            (self.stats.qubic.hashrate_kh_s / QUBIC_MAX_KHS).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Dynex = GPU (weight 0.5), Quai = GPU (weight 0.3), Qubic = CPU (weight 0.2)
        (0.5 * dynex_power + 0.3 * quai_power + 0.2 * qubic_power).clamp(0.0, 1.0)
    }

    pub fn log_mining_status(&self) {
        let signal = self.get_mining_power_signal();
        eprintln!(
            "[market] mining: dynex={:.3}MH/s active={} quai={:.1}MH/s active={} qubic={:.0}kH/s active={}  signal={:.3}",
            self.stats.dynex.hashrate_mh_s,
            self.stats.dynex.is_active,
            self.stats.quai.hashrate_mh_s,
            self.stats.quai.is_active,
            self.stats.qubic.hashrate_kh_s,
            self.stats.qubic.is_active,
            signal,
        );
    }
}

impl Default for MiningTelemetry {
    fn default() -> Self {
        Self::new()
    }
}
