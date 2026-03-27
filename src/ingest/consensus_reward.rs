//! Consensus Reward — converts blockchain "successful solve" events into
//! dopamine spikes for the SNN reward-modulated learning system.
//!
//! The key insight: when YOUR local node validates a solution (Dynex share,
//! Quai block, Qubic computation), that is the strongest possible reward
//! signal — it means the GPU's electrical state at that moment was
//! *provably optimal*. This is far more valuable than the synthetic
//! `compute_reward()` in `learning_trainer.rs`.
//!
//! Reward magnitude hierarchy (biological analogy):
//!   - Qubic solution validated:  1.0  (rare, high-value — like finding food)
//!   - Quai block mined:          0.8  (medium-rare — like successful hunt)
//!   - Dynex share accepted:      0.3  (frequent — like foraging success)
//!
//! The dopamine spike decays exponentially (τ = 0.5s = 5 steps at 10Hz)
//! so the E-prop eligibility trace captures a temporal credit window
//! of ~1.5s around the solve event. This teaches the SNN which
//! electrical states (power, temp, voltage) preceded success.
//!
//! Integration with `EpochTrainer::compute_reward()`:
//!   R_total = R_synthetic + dopamine_boost
//!   R_total = clamp(R_total, 0.0, 1.5)  // allow transient overshoot
//!
//! The 1.5 ceiling prevents runaway weight growth while still making
//! consensus events significantly more impactful than steady-state reward.

use std::time::Instant;

use super::triple_bridge::TripleSnapshot;

/// Dopamine decay constant: τ = 0.5s.
/// At 10Hz (Δt = 0.1s): decay = exp(-0.1 / 0.5) = 0.8187
const DOPAMINE_DECAY: f32 = 0.8187;

/// Maximum combined reward (synthetic + dopamine). Allows transient
/// overshoot to make consensus events salient without runaway growth.
pub const REWARD_CEILING: f32 = 1.5;

/// Reward magnitudes for each consensus event type.
const DYNEX_SHARE_REWARD: f32 = 0.3;
const QUAI_BLOCK_REWARD: f32 = 0.8;
const QUBIC_SOLUTION_REWARD: f32 = 1.0;

/// Tracks dopamine level from consensus reward events.
///
/// Zero-alloc, lives on the stack in the supervisor loop.
/// Call `update()` every 10Hz step with the latest TripleSnapshot.
pub struct ConsensusRewardTracker {
    /// Current dopamine level [0.0, 1.0]. Decays exponentially.
    dopamine: f32,
    /// Timestamp of last event (for diagnostics).
    last_event: Option<Instant>,
    /// Cumulative event counters (for logging).
    pub dynex_shares: u64,
    pub quai_blocks: u64,
    pub qubic_solutions: u64,
}

impl ConsensusRewardTracker {
    pub fn new() -> Self {
        Self {
            dopamine: 0.0,
            last_event: None,
            dynex_shares: 0,
            quai_blocks: 0,
            qubic_solutions: 0,
        }
    }

    /// Process one 10Hz step. Checks for consensus events and decays dopamine.
    ///
    /// Returns the current dopamine boost to add to `compute_reward()`.
    /// Zero-alloc, branchless decay.
    pub fn update(&mut self, snap: &TripleSnapshot) -> f32 {
        // Check for new consensus events (edge-triggered by triple_bridge)
        if snap.qubic_solution_found {
            self.dopamine = QUBIC_SOLUTION_REWARD;
            self.last_event = Some(Instant::now());
            self.qubic_solutions += 1;
            eprintln!(
                "[dopamine] QUBIC SOLUTION! dopamine=1.0 (total solutions: {})",
                self.qubic_solutions
            );
        } else if snap.quai_block_mined {
            // Don't override a stronger signal that's still active
            self.dopamine = self.dopamine.max(QUAI_BLOCK_REWARD);
            self.last_event = Some(Instant::now());
            self.quai_blocks += 1;
            eprintln!(
                "[dopamine] QUAI BLOCK MINED! dopamine={:.2} (total blocks: {})",
                self.dopamine, self.quai_blocks
            );
        } else if snap.dynex_share_found {
            self.dopamine = self.dopamine.max(DYNEX_SHARE_REWARD);
            self.last_event = Some(Instant::now());
            self.dynex_shares += 1;
            if self.dynex_shares % 10 == 0 {
                eprintln!(
                    "[dopamine] Dynex share #{} | dopamine={:.2}",
                    self.dynex_shares, self.dopamine
                );
            }
        }

        // Exponential decay
        self.dopamine *= DOPAMINE_DECAY;

        // Floor to zero when negligible (avoid denormals)
        if self.dopamine < 1e-4 {
            self.dopamine = 0.0;
        }

        self.dopamine
    }

    /// Current dopamine level without advancing state.
    pub fn dopamine(&self) -> f32 {
        self.dopamine
    }

    /// Apply the consensus dopamine boost to a synthetic reward.
    ///
    /// R_total = clamp(R_synthetic + dopamine, 0.0, REWARD_CEILING)
    ///
    /// Use this in `EpochTrainer::tick()` or the live supervisor's
    /// reward computation to integrate real consensus feedback.
    pub fn boost_reward(&self, synthetic_reward: f32) -> f32 {
        (synthetic_reward + self.dopamine).clamp(0.0, REWARD_CEILING)
    }

    /// Summary string for dashboard display.
    pub fn status_line(&self) -> String {
        format!(
            "DA:{:.2} shares:{} blocks:{} sols:{}",
            self.dopamine, self.dynex_shares, self.quai_blocks, self.qubic_solutions
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dopamine_decay() {
        let mut tracker = ConsensusRewardTracker::new();
        let mut snap = TripleSnapshot::default();

        // Trigger a Dynex share
        snap.dynex_share_found = true;
        let d = tracker.update(&snap);
        assert!((d - DYNEX_SHARE_REWARD * DOPAMINE_DECAY).abs() < 0.01);

        // Clear the edge trigger
        snap.dynex_share_found = false;

        // Decay for 50 steps (5 seconds) — should approach zero
        for _ in 0..50 {
            tracker.update(&snap);
        }
        assert!(tracker.dopamine() < 0.001, "should decay to ~0, got {}", tracker.dopamine());
    }

    #[test]
    fn test_qubic_overrides_dynex() {
        let mut tracker = ConsensusRewardTracker::new();
        let mut snap = TripleSnapshot::default();

        // Dynex share first
        snap.dynex_share_found = true;
        tracker.update(&snap);
        snap.dynex_share_found = false;

        // Qubic solution should override
        snap.qubic_solution_found = true;
        tracker.update(&snap);
        assert!(tracker.dopamine() > 0.8, "Qubic should dominate");
    }

    #[test]
    fn test_boost_reward_ceiling() {
        let mut tracker = ConsensusRewardTracker::new();
        let mut snap = TripleSnapshot::default();

        snap.qubic_solution_found = true;
        tracker.update(&snap);

        // Synthetic reward of 0.9 + dopamine ~1.0 should cap at 1.5
        let boosted = tracker.boost_reward(0.9);
        assert!(boosted <= REWARD_CEILING);
        assert!(boosted > 1.0, "should be boosted above 1.0");
    }

    #[test]
    fn test_event_counters() {
        let mut tracker = ConsensusRewardTracker::new();
        let mut snap = TripleSnapshot::default();

        snap.dynex_share_found = true;
        tracker.update(&snap);
        snap.dynex_share_found = false;
        snap.dynex_share_found = true;
        tracker.update(&snap);
        assert_eq!(tracker.dynex_shares, 2);

        snap.dynex_share_found = false;
        snap.quai_block_mined = true;
        tracker.update(&snap);
        assert_eq!(tracker.quai_blocks, 1);
    }
}
