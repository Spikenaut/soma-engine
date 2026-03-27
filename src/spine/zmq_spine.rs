//! ZMQ Nervous System — The IPC spine connecting Rust → Julia
//!
//! Binds a ZMQ PUB socket to `ipc:///tmp/spikenaut.ipc` and broadcasts
//! packed MarketPulse messages at tick frequency.
//!
//! The Julia Brain subscribes to this socket and reads the packed
//! struct via zero-copy pointer casting — no deserialization overhead.
//!
//! ```text
//!   ┌─────────────────┐  120-byte packed   ┌──────────────────────┐
//!   │  Rust Nervous    │──────────────────▶│  Julia CUDA Brain     │
//!   │  System (PUB)    │  ipc:///tmp/      │  65,536 neurons       │
//!   │  + Qubic Pulse   │  spikenaut.ipc    │  Sparse OU-SDE + STDP │
//!   └─────────────────┘                    └──────────────────────┘
//! ```

use std::path::Path;
use std::time::Instant;

use crate::spine::dydx_ingest::DydxSnapshot;
use crate::telemetry::gpu_telemetry::{GpuTelemetry, QubicTraceState};
use crate::trading::MarketFeed;

/// IPC socket path — verified at runtime for existence and permissions.
const IPC_ENDPOINT: &str = "ipc:///tmp/spikenaut.ipc";

/// Rolling volatility window for each asset channel.
const VOLATILITY_WINDOW: usize = 50;

/// The ZMQ Nervous System: zero-allocation market data broadcaster.
pub struct ZmqNervousSystem {
    #[allow(dead_code)]
    socket: zmq::Socket,
    _context: zmq::Context,
    epoch: Instant,
    /// Per-channel volatility estimators (RMS of absolute log-returns)
    #[allow(dead_code)]
    vol_windows: [VolEstimator; 7],
    /// Tick counter for diagnostics
    #[allow(dead_code)]
    pub ticks_sent: u64,
    /// Last FPGA buffer load reading
    #[allow(dead_code)]
    pub basys_buffer_load: f32,
    /// Zero-Order Hold receiver for dYdX v4 key-free market data.
    #[allow(dead_code)]
    dydx_rx: tokio::sync::watch::Receiver<DydxSnapshot>,
}

/// Simple rolling RMS volatility estimator.
struct VolEstimator {
    buf: Vec<f32>,
    pos: usize,
    full: bool,
    cap: usize,
}

impl VolEstimator {
    fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            pos: 0,
            full: false,
            cap: capacity,
        }
    }

    fn push(&mut self, abs_log_return: f32) {
        self.buf[self.pos] = abs_log_return;
        self.pos += 1;
        if self.pos >= self.cap {
            self.pos = 0;
            self.full = true;
        }
    }

    /// RMS volatility: sqrt(mean(r^2)) over the window, clamped [0, 1].
    #[allow(dead_code)]
    fn rms(&self) -> f32 {
        let n = if self.full { self.cap } else { self.pos };
        if n == 0 { return 0.0; }
        let sum_sq: f32 = self.buf[..n].iter().map(|r| r * r).sum();
        (sum_sq / n as f32).sqrt().clamp(0.0, 1.0)
    }
}

impl ZmqNervousSystem {
    /// Create and bind the ZMQ PUB socket.
    ///
    /// Verifies the IPC socket path is writable.  Removes stale socket
    /// files from previous runs to avoid EADDRINUSE.
    pub fn new(dydx_rx: tokio::sync::watch::Receiver<DydxSnapshot>) -> Result<Self, String> {
        // Clean up stale socket file
        let ipc_path = "/tmp/spikenaut.ipc";
        if Path::new(ipc_path).exists() {
            let _ = std::fs::remove_file(ipc_path);
            println!("[spine] Removed stale IPC socket: {}", ipc_path);
        }

        let context = zmq::Context::new();
        let socket = context.socket(zmq::PUB)
            .map_err(|e| format!("ZMQ socket creation failed: {}", e))?;

        // Set high-water mark to prevent slow-subscriber backpressure
        socket.set_sndhwm(64)
            .map_err(|e| format!("ZMQ set_sndhwm failed: {}", e))?;

        // Bind the publisher
        socket.bind(IPC_ENDPOINT)
            .map_err(|e| format!("ZMQ bind to {} failed: {}", IPC_ENDPOINT, e))?;

        println!(
            "[spine] ZMQ PUB bound to {} (HWM=64, packed=120 bytes, Triple Node Sync)",
            IPC_ENDPOINT
        );

        Ok(Self {
            socket,
            _context: context,
            epoch: Instant::now(),
            vol_windows: std::array::from_fn(|_| VolEstimator::new(VOLATILITY_WINDOW)),
            ticks_sent: 0,
            basys_buffer_load: 0.0,
            dydx_rx,
        })
    }

    /// Set the FPGA UART buffer load (called from the bridge module).
    pub fn set_basys_buffer_load(&mut self, load: f32) {
        self.basys_buffer_load = load.clamp(0.0, 1.0);
    }

    /// Broadcast one tick of market data + hardware telemetry.
    ///
    /// Converts the `MarketFeed` + `GpuTelemetry` + normalizer Z-scores + dYdX ZOH
    /// into a packed `MarketPulseData` and sends it over ZMQ IPC.
    ///
    /// This is the "axon firing" of the Rust nervous system.
    pub fn broadcast(
        &mut self,
        feed: &MarketFeed,
        telem: &GpuTelemetry,
        z_scores: &[f32; 8],
    ) -> Result<(), String> {
        use crate::spine::market_pulse::{MarketPulseData, AssetTick};

        let now_ns = self.epoch.elapsed().as_nanos() as u64;

        // Update volatility estimators with absolute Z-scores
        // Channels: 0=DNX, 1=QUAI, 2=QUBIC, 3=KAS, 4=XMR, 5=OCEAN, 6=VERUS
        for i in 0..7 {
            self.vol_windows[i].push(z_scores[i].abs());
        }

        let pulse = MarketPulseData {
            timestamp_ns: now_ns,
            dnx:    AssetTick::new(z_scores[0], self.vol_windows[0].rms()),
            quai:   AssetTick::new(z_scores[1], self.vol_windows[1].rms()),
            qubic:  AssetTick::new(z_scores[2], self.vol_windows[2].rms()),
            kaspa:  AssetTick::new(z_scores[3], self.vol_windows[3].rms()),
            monero: AssetTick::new(z_scores[4], self.vol_windows[4].rms()),
            ocean:  AssetTick::new(z_scores[5], self.vol_windows[5].rms()),
            verus:  AssetTick::new(z_scores[6], self.vol_windows[6].rms()),
            ocean_intel_confidence: telem.ocean_intel,
            coinglass_funding_rate: feed.coinglass_funding_rate,
            coinglass_liquidation_volume: feed.coinglass_liquidation_volume,
            dex_liquidity_delta: feed.dex_liquidity_delta,
            l3_order_imbalance: feed.l3_order_imbalance,
            gpu_temp_c: telem.gpu_temp_c,
            gpu_power_w: telem.power_w,
            gpu_util_pct: telem.mem_util_pct,
            basys_uart_buffer_load: self.basys_buffer_load,
            dydx_oi_delta: self.dydx_rx.borrow().oi_delta,
            dydx_funding_rate: self.dydx_rx.borrow().funding_rate,
            // Qubic fields zeroed in non-triple mode
            qubic_tick_trace: 0.0,
            qubic_tick_rate: 0.0,
            qubic_epoch_progress: 0.0,
        };

        let packed = pulse.serialize_packed();

        self.socket.send(packed.as_slice(), 0)
            .map_err(|e| format!("ZMQ send failed: {}", e))?;

        self.ticks_sent += 1;

        if self.ticks_sent % 100 == 0 {
            println!(
                "[spine] Tick {} | gpu={:.0}°C {:.0}W | basys_buf={:.2} | {} bytes/tick",
                self.ticks_sent, telem.gpu_temp_c, telem.power_w,
                self.basys_buffer_load, packed.len()
            );
        }

        Ok(())
    }

    /// Broadcast with Qubic Global Computing Pulse (Triple Node Sync).
    ///
    /// Extends the standard broadcast with three additional f32 fields
    /// from the `QubicTraceState`, bringing the packed message from 108 → 120 bytes.
    /// The Julia Brain receives:
    ///   - `qubic_tick_trace`:     exponentially decaying tick arrival (τ=1.5s)
    ///   - `qubic_tick_rate`:      smoothed ticks/sec, normalized [0,1]
    ///   - `qubic_epoch_progress`: linear ramp over ~7-day epoch [0,1]
    ///
    /// Zero-alloc: all buffers are stack-allocated.
    pub fn broadcast_triple(
        &mut self,
        feed: &MarketFeed,
        telem: &GpuTelemetry,
        z_scores: &[f32; 8],
        qubic: &QubicTraceState,
    ) -> Result<(), String> {
        use crate::spine::market_pulse::{MarketPulseData, AssetTick};

        let now_ns = self.epoch.elapsed().as_nanos() as u64;

        // Update volatility estimators
        for i in 0..7 {
            self.vol_windows[i].push(z_scores[i].abs());
        }

        let pulse = MarketPulseData {
            timestamp_ns: now_ns,
            dnx:    AssetTick::new(z_scores[0], self.vol_windows[0].rms()),
            quai:   AssetTick::new(z_scores[1], self.vol_windows[1].rms()),
            qubic:  AssetTick::new(z_scores[2], self.vol_windows[2].rms()),
            kaspa:  AssetTick::new(z_scores[3], self.vol_windows[3].rms()),
            monero: AssetTick::new(z_scores[4], self.vol_windows[4].rms()),
            ocean:  AssetTick::new(z_scores[5], self.vol_windows[5].rms()),
            verus:  AssetTick::new(z_scores[6], self.vol_windows[6].rms()),
            ocean_intel_confidence: telem.ocean_intel,
            coinglass_funding_rate: feed.coinglass_funding_rate,
            coinglass_liquidation_volume: feed.coinglass_liquidation_volume,
            dex_liquidity_delta: feed.dex_liquidity_delta,
            l3_order_imbalance: feed.l3_order_imbalance,
            gpu_temp_c: telem.gpu_temp_c,
            gpu_power_w: telem.power_w,
            gpu_util_pct: telem.mem_util_pct,
            basys_uart_buffer_load: self.basys_buffer_load,
            dydx_oi_delta: self.dydx_rx.borrow().oi_delta,
            dydx_funding_rate: self.dydx_rx.borrow().funding_rate,
            // Qubic Global Computing Pulse
            qubic_tick_trace: qubic.tick_trace,
            qubic_tick_rate: qubic.tick_rate_norm(),
            qubic_epoch_progress: qubic.epoch_progress(),
        };

        let packed = pulse.serialize_packed();

        self.socket.send(packed.as_slice(), 0)
            .map_err(|e| format!("ZMQ send failed: {}", e))?;

        self.ticks_sent += 1;

        if self.ticks_sent % 100 == 0 {
            println!(
                "[spine] Triple Tick {} | gpu={:.0}°C {:.0}W | qubic_trace={:.2} rate={:.2} epoch={:.1}% | {} bytes",
                self.ticks_sent, telem.gpu_temp_c, telem.power_w,
                qubic.tick_trace, qubic.tick_rate_norm(), qubic.epoch_progress() * 100.0,
                packed.len()
            );
        }

        Ok(())
    }

    #[cfg(feature = "quai_integration")]
    pub fn broadcast_quai(
        &mut self,
        feed: &MarketFeed,
        telem: &GpuTelemetry,
        z_scores: &[f32; 12],
    ) -> Result<(), String> {
        use crate::spine::market_pulse::{AssetTick, MarketPulseData};

        let now_ns = self.epoch.elapsed().as_nanos() as u64;

        // Update volatility estimators with absolute Z-scores
        // Channels: 0=DNX, 1=QUAI, 2=QUBIC, 3=KAS, 4=XMR, 5=OCEAN, 6=VERUS, 7=GPU, 8=Quai Gas, 9=Quai TX, 10=Quai Util, 11=Quai Stake
        for i in 0..7 {
            self.vol_windows[i].push(z_scores[i].abs());
        }

        let pulse = MarketPulseData {
            timestamp_ns: now_ns,
            dnx:    AssetTick::new(z_scores[0], self.vol_windows[0].rms()),
            quai:   AssetTick::new(z_scores[1], self.vol_windows[1].rms()),
            qubic:  AssetTick::new(z_scores[2], self.vol_windows[2].rms()),
            kaspa:  AssetTick::new(z_scores[3], self.vol_windows[3].rms()),
            monero: AssetTick::new(z_scores[4], self.vol_windows[4].rms()),
            ocean:  AssetTick::new(z_scores[5], self.vol_windows[5].rms()),
            verus:  AssetTick::new(z_scores[6], self.vol_windows[6].rms()),
            ocean_intel_confidence: telem.ocean_intel,
            // Institutional sensors — forwarded from MarketFeed
            coinglass_funding_rate: feed.coinglass_funding_rate,
            coinglass_liquidation_volume: feed.coinglass_liquidation_volume,
            dex_liquidity_delta: feed.dex_liquidity_delta,
            l3_order_imbalance: feed.l3_order_imbalance,
            // Hardware proprioception
            gpu_temp_c: telem.gpu_temp_c,
            gpu_power_w: telem.power_w,
            gpu_util_pct: telem.mem_util_pct, // NVML util rate
            basys_uart_buffer_load: self.basys_buffer_load,
            // dYdX v4 key-free signals (Zero-Order Hold — last known value if WS is slow)
            dydx_oi_delta: self.dydx_rx.borrow().oi_delta,
            dydx_funding_rate: self.dydx_rx.borrow().funding_rate,
            // Qubic fields not used in quai-only broadcast path.
            qubic_tick_trace: 0.0,
            qubic_tick_rate: 0.0,
            qubic_epoch_progress: 0.0,
        };

        let packed = pulse.serialize_packed();

        self.socket.send(packed.as_slice(), 0)
            .map_err(|e| format!("ZMQ send failed: {}", e))?;

        self.ticks_sent += 1;

        if self.ticks_sent % 100 == 0 {
            println!(
                "[spine] Quai Tick {} | gpu={:.0}°C {:.0}W | basys_buf={:.2} | {} bytes/tick",
                self.ticks_sent, telem.gpu_temp_c, telem.power_w,
                self.basys_buffer_load, packed.len()
            );
        }

        Ok(())
    }
}

impl Drop for ZmqNervousSystem {
    fn drop(&mut self) {
        println!("[spine] Shutting down ZMQ PUB ({} ticks sent)", self.ticks_sent);
        // ZMQ context drop handles cleanup; also remove the IPC file
        let _ = std::fs::remove_file("/tmp/spikenaut.ipc");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vol_estimator() {
        let mut v = VolEstimator::new(3);
        v.push(0.1);
        v.push(0.2);
        v.push(0.3);
        let rms = v.rms();
        // RMS of [0.1, 0.2, 0.3] = sqrt((0.01+0.04+0.09)/3) ≈ 0.2160
        assert!((rms - 0.2160).abs() < 0.01);
    }

    #[test]
    fn test_vol_estimator_empty() {
        let v = VolEstimator::new(10);
        assert_eq!(v.rms(), 0.0);
    }
}
