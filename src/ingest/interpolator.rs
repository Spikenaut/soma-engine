//! State-Space Interpolator — upsamples slow blockchain signals to 10Hz.
//!
//! Problem: Blockchain data arrives at wildly different rates:
//!   - Dynex miner stats:  ~1 Hz (nvidia-smi polling)
//!   - Qubic ticks:        ~0.2-0.5 Hz (2-5 second intervals)
//!   - Quai blocks:        ~0.08 Hz (12-second block time)
//!
//! The SNN supervisor runs at 10Hz. Naive sample-and-hold causes "stutter":
//! the SNN sees identical values for 20-120 consecutive steps, then a sharp
//! discontinuity. This creates phantom spikes that drown real signal.
//!
//! Solution: First-order exponential interpolation (State-Space Model).
//!
//! State equation:
//!   x[k+1] = α · x[k] + (1 - α) · u[k]
//!
//! where:
//!   x[k]  = interpolated output at 10Hz step k
//!   u[k]  = latest raw observation (Zero-Order Hold of last RPC response)
//!   α     = exp(-Δt / τ), smoothing factor
//!   τ     = time constant (tuned per signal class)
//!   Δt    = 0.1s (10Hz step interval)
//!
//! Properties:
//!   - Converges to true value within ~3τ of a step change
//!   - Zero heap allocation (all state on stack)
//!   - Monotone: no overshoot or ringing (unlike Kalman or spline)
//!   - Graceful degradation: if no new observation arrives, state
//!     exponentially decays toward the last known value (no divergence)

/// Time constant classes for different signal dynamics.
///
/// Faster τ = more responsive to new data, more stutter.
/// Slower τ = smoother output, more latency.
#[derive(Debug, Clone, Copy)]
pub enum SignalClass {
    /// Fast hardware signals (power, temp, hashrate): τ = 0.3s
    /// Converges in ~1 second. Tracks GPU transients without aliasing.
    Hardware,
    /// Medium blockchain signals (Qubic ticks, gas price): τ = 1.0s
    /// Converges in ~3 seconds. Smooths 2-5s tick jitter.
    Blockchain,
    /// Slow epoch signals (Quai blocks, epoch progress): τ = 3.0s
    /// Converges in ~9 seconds. Matches 12s block cadence.
    SlowChain,
}

impl SignalClass {
    /// Smoothing factor α = exp(-Δt / τ) for 10Hz (Δt = 0.1s).
    /// Pre-computed to avoid `exp()` in the hot path.
    #[cfg(test)]
    fn alpha(self) -> f32 {
        match self {
            // exp(-0.1 / 0.3) = 0.7165
            SignalClass::Hardware => 0.7165,
            // exp(-0.1 / 1.0) = 0.9048
            SignalClass::Blockchain => 0.9048,
            // exp(-0.1 / 3.0) = 0.9672
            SignalClass::SlowChain => 0.9672,
        }
    }
}

/// Single-channel state-space interpolator.
///
/// Zero-alloc, Copy-able, designed to live inside a fixed-size array.
#[derive(Debug, Clone, Copy)]
pub struct ChannelInterpolator {
    /// Current interpolated state.
    state: f32,
    /// Last raw observation (Zero-Order Hold).
    observation: f32,
    /// Smoothing factor α (pre-computed from SignalClass).
    alpha: f32,
    /// True after at least one observation has been received.
    initialized: bool,
}

impl ChannelInterpolator {
    /// Create a new interpolator for the given signal class.
    pub const fn new(class: SignalClass) -> Self {
        // Can't call methods in const fn, so inline the alpha values.
        let alpha = match class {
            SignalClass::Hardware => 0.7165,
            SignalClass::Blockchain => 0.9048,
            SignalClass::SlowChain => 0.9672,
        };
        Self {
            state: 0.0,
            observation: 0.0,
            alpha,
            initialized: false,
        }
    }

    /// Feed a new raw observation from an RPC response.
    ///
    /// Call this whenever new blockchain data arrives (irregular cadence).
    /// Between observations, call `step()` at 10Hz to get smooth output.
    pub fn observe(&mut self, value: f32) {
        self.observation = value;
        if !self.initialized {
            // First observation: snap state to avoid cold-start ramp.
            self.state = value;
            self.initialized = true;
        }
    }

    /// Advance one 10Hz step. Returns the interpolated value.
    ///
    /// x[k+1] = α · x[k] + (1 - α) · u[k]
    ///
    /// Zero-alloc, branchless after initialization.
    #[inline(always)]
    pub fn step(&mut self) -> f32 {
        if !self.initialized {
            return 0.0;
        }
        self.state = self.alpha * self.state + (1.0 - self.alpha) * self.observation;
        self.state
    }

    /// Current interpolated value without advancing state.
    pub fn value(&self) -> f32 {
        self.state
    }
}

// ── Multi-channel interpolator bank ──────────────────────────────────────────

/// Number of interpolated channels from the triple bridge.
///
/// Layout:
///   0: Dynex hashrate (MH/s)         — Hardware
///   1: Dynex power (W)               — Hardware
///   2: Dynex GPU temp (°C)           — Hardware
///   3: Qubic tick rate (ticks/s)     — Blockchain
///   4: Qubic epoch progress [0,1]    — SlowChain
///   5: QU price (USD)                — Blockchain
///   6: Quai gas price (gwei)         — SlowChain
///   7: Quai tx count                 — SlowChain
///   8: Quai block utilization [0,1]  — SlowChain
///   9: Neuraxon dopamine [0,1]       — Blockchain
///   10: Neuraxon serotonin [0,1]     — Blockchain
///   11: Neuraxon ITS (normalized)    — Blockchain
pub const NUM_BRIDGE_CHANNELS: usize = 12;

/// Multi-channel interpolator bank for all triple-bridge signals.
///
/// Each channel has its own time constant matched to signal dynamics.
/// The entire bank fits in ~144 bytes (9 × 16 bytes) on the stack.
pub struct InterpolatorBank {
    channels: [ChannelInterpolator; NUM_BRIDGE_CHANNELS],
}

impl InterpolatorBank {
    pub fn new() -> Self {
        Self {
            channels: [
                // 0: Dynex hashrate
                ChannelInterpolator::new(SignalClass::Hardware),
                // 1: Dynex power
                ChannelInterpolator::new(SignalClass::Hardware),
                // 2: Dynex GPU temp
                ChannelInterpolator::new(SignalClass::Hardware),
                // 3: Qubic tick rate
                ChannelInterpolator::new(SignalClass::Blockchain),
                // 4: Qubic epoch progress
                ChannelInterpolator::new(SignalClass::SlowChain),
                // 5: QU price
                ChannelInterpolator::new(SignalClass::Blockchain),
                // 6: Quai gas price
                ChannelInterpolator::new(SignalClass::SlowChain),
                // 7: Quai tx count
                ChannelInterpolator::new(SignalClass::SlowChain),
                // 8: Quai block utilization
                ChannelInterpolator::new(SignalClass::SlowChain),
                // 9: Neuraxon dopamine
                ChannelInterpolator::new(SignalClass::Blockchain),
                // 10: Neuraxon serotonin
                ChannelInterpolator::new(SignalClass::Blockchain),
                // 11: Neuraxon ITS (normalized)
                ChannelInterpolator::new(SignalClass::Blockchain),
            ],
        }
    }

    /// Feed raw observations from a TripleSnapshot.
    /// Call once per RPC response (irregular cadence).
    pub fn observe(&mut self, snap: &super::triple_bridge::TripleSnapshot) {
        self.channels[0].observe(snap.dynex_hashrate_mh);
        self.channels[1].observe(snap.dynex_power_w);
        self.channels[2].observe(snap.dynex_gpu_temp_c);
        self.channels[3].observe(snap.qubic_tick_rate);
        self.channels[4].observe(snap.qubic_epoch_progress);
        self.channels[5].observe(snap.qu_price_usd);
        self.channels[6].observe(snap.quai_gas_price);
        self.channels[7].observe(snap.quai_tx_count as f32);
        self.channels[8].observe(snap.quai_block_utilization);
        self.channels[9].observe(snap.neuraxon_dopamine.clamp(0.0, 1.0));
        self.channels[10].observe(snap.neuraxon_serotonin.clamp(0.0, 1.0));
        // ITS magnitude is source-specific; this keeps high values bounded.
        self.channels[11].observe((snap.neuraxon_its / 2000.0).clamp(0.0, 1.0));
    }

    /// Advance all channels one 10Hz step. Returns interpolated values.
    ///
    /// Call this every 100ms in the supervisor loop.
    pub fn step(&mut self) -> [f32; NUM_BRIDGE_CHANNELS] {
        let mut out = [0.0_f32; NUM_BRIDGE_CHANNELS];
        for (i, ch) in self.channels.iter_mut().enumerate() {
            out[i] = ch.step();
        }
        out
    }

    /// Read current interpolated values without advancing.
    pub fn values(&self) -> [f32; NUM_BRIDGE_CHANNELS] {
        let mut out = [0.0_f32; NUM_BRIDGE_CHANNELS];
        for (i, ch) in self.channels.iter().enumerate() {
            out[i] = ch.value();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolator_snap_on_first_observe() {
        let mut ch = ChannelInterpolator::new(SignalClass::Hardware);
        assert_eq!(ch.step(), 0.0); // no observation yet
        ch.observe(100.0);
        assert_eq!(ch.value(), 100.0); // snaps immediately
    }

    #[test]
    fn test_interpolator_convergence() {
        let mut ch = ChannelInterpolator::new(SignalClass::Hardware);
        ch.observe(0.0);
        // Step change to 1.0
        ch.observe(1.0);
        // Hardware τ=0.3s → converges in ~10 steps (1s)
        for _ in 0..30 {
            ch.step();
        }
        assert!((ch.value() - 1.0).abs() < 0.01, "should converge to 1.0, got {}", ch.value());
    }

    #[test]
    fn test_interpolator_no_overshoot() {
        let mut ch = ChannelInterpolator::new(SignalClass::Blockchain);
        ch.observe(0.0);
        ch.observe(1.0);
        let mut prev = 0.0;
        for _ in 0..50 {
            let v = ch.step();
            assert!(v >= prev, "must be monotone: {} < {}", v, prev);
            assert!(v <= 1.0, "must not overshoot: {}", v);
            prev = v;
        }
    }

    #[test]
    fn test_slow_chain_lag() {
        let mut ch = ChannelInterpolator::new(SignalClass::SlowChain);
        ch.observe(0.0);
        ch.observe(1.0);
        // After 10 steps (1s), SlowChain should still be lagging
        for _ in 0..10 {
            ch.step();
        }
        assert!(ch.value() < 0.5, "SlowChain should lag at 1s, got {}", ch.value());
    }

    #[test]
    fn test_signal_class_alpha_values() {
        // Verify pre-computed alphas match exp(-0.1/τ)
        let hw = (-0.1_f64 / 0.3).exp() as f32;
        assert!((SignalClass::Hardware.alpha() - hw).abs() < 0.001);
        let bc = (-0.1_f64 / 1.0).exp() as f32;
        assert!((SignalClass::Blockchain.alpha() - bc).abs() < 0.001);
        let sc = (-0.1_f64 / 3.0).exp() as f32;
        assert!((SignalClass::SlowChain.alpha() - sc).abs() < 0.001);
    }
}
