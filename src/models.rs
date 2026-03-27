// models — data types that flow through the Ship's nervous system.

pub mod miner_bridge;

pub use crate::telemetry::mining_sync_telemetry::MiningSyncTelemetry;
pub use crate::telemetry::mining_telemetry::MiningTelemetry;

use serde::{Deserialize, Serialize};

/// NERO manifold snapshot: 4 neuromodulator scores from Julia brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeroManifoldSnapshot {
    pub tick: i64,
    pub dopamine: f32,
    pub cortisol: f32,
    pub acetylcholine: f32,
    pub tempo: f32,
}

impl NeroManifoldSnapshot {
    pub fn from_scores(tick: i64, scores: &[f32; 4]) -> Self {
        Self {
            tick,
            dopamine:       scores[0],
            cortisol:       scores[1],
            acetylcholine:  scores[2],
            tempo:          scores[3],
        }
    }
}

/// GPU scheduler state transition event.
#[derive(Debug, Clone, Serialize)]
pub struct GpuSchedulerEvent {
    /// Decision: "allowed" | "paused:llm" | "paused:vram" | "paused:thermal" | "throttled"
    pub decision: String,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub gpu_temp_c: f32,
    pub power_w: f32,
    /// Total state transitions since startup.
    pub transition_count: u64,
}

/// Algorithm rotation event.
#[derive(Debug, Clone, Serialize)]
pub struct RotationEvent {
    /// Kind: "evaluated" | "rotated" | "skipped:cooldown" | "skipped:stale" | "skipped:threshold" | "rollback"
    pub kind: String,
    pub from_algo: Option<String>,
    pub to_algo: Option<String>,
    /// (algo_name, revenue_usd_hr, confidence)
    pub revenues: Vec<(String, f64, f64)>,
    pub market_age_secs: f64,
}

/// Top-level IPC message type sent from background threads to SupervisorApp.
#[derive(Debug, Clone)]
pub enum WireMsg {
    /// Julia brain emitted a NERO readout packet.
    NeroUpdate(NeroManifoldSnapshot),
    /// Mining telemetry data
    MiningTelem(MiningTelemetry),
    /// Status message
    Status(String),
    /// GPU scheduler state transition or heartbeat.
    GpuSchedulerEvent(GpuSchedulerEvent),
    /// Algorithm rotation evaluation or switch.
    RotationEvent(RotationEvent),
}
