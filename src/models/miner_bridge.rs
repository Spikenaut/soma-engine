// models/miner_bridge — mining pool events that modulate the SNN.

/// Events emitted by the mining pool client.
#[derive(Debug, Clone)]
pub enum PoolEvent {
    /// A submitted share was accepted by the pool.
    ShareAccepted { latency_ms: u64 },
    /// A block was found (rare — strongest dopamine burst).
    BlockFound { block_height: u64, reward_dnx: f64 },
    /// Pool connection switched (cortisol spike).
    PoolSwitch { reason: String },
    /// A share was rejected (mild cortisol).
    ShareRejected { reason: String },
}
