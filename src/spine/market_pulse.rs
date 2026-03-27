//! MarketPulse — the binary nerve impulse between Rust and Julia.
//!
//! Wire format: 108-byte packed little-endian struct.
//! Contract documented in `proto/spikenaut_tick.fbs`.
//!
//! Layout:
//!   [0..8]    timestamp_ns: u64
//!   [8..16]   dnx: AssetTick (price_norm f32, volatility f32)
//!   [16..24]  quai: AssetTick
//!   [24..32]  qubic: AssetTick
//!   [32..40]  kaspa: AssetTick
//!   [40..48]  monero: AssetTick
//!   [48..56]  ocean: AssetTick
//!   [56..64]  verus: AssetTick
//!   [64..68]  ocean_intel_confidence: f32
//!   [68..72]  coinglass_funding_rate: f32
//!   [72..76]  coinglass_liquidation_volume: f32
//!   [76..80]  dex_liquidity_delta: f32
//!   [80..84]  l3_order_imbalance: f32
//!   [84..88]  gpu_temp_c: f32
//!   [88..92]  gpu_power_w: f32
//!   [92..96]  gpu_util_pct: f32
//!   [96..100] basys_uart_buffer_load: f32
//!   [100..104] dydx_oi_delta: f32         (dYdX v4 key-free WebSocket)
//!   [104..108] dydx_funding_rate: f32     (dYdX v4 key-free WebSocket)
//!   ── Qubic Global Computing Pulse (Triple Node Sync) ──
//!   [108..112] qubic_tick_trace: f32      (exponential decay, τ=1.5s)
//!   [112..116] qubic_tick_rate: f32       (normalized ticks/sec [0,1])
//!   [116..120] qubic_epoch_progress: f32  (linear ramp [0,1] over ~7d)
//!   Total: 120 bytes

// ── AssetTick (inline struct, 8 bytes) ──────────────────────────────────────

/// Fixed-size asset tick — maps 1:1 to the FBS `struct AssetTick`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AssetTick {
    pub price_norm: f32,
    pub volatility: f32,
}

impl AssetTick {
    pub fn new(price_norm: f32, volatility: f32) -> Self {
        Self { price_norm, volatility }
    }

    /// Serialize to 8 bytes (little-endian).
    pub fn to_le_bytes(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        buf[0..4].copy_from_slice(&self.price_norm.to_le_bytes());
        buf[4..8].copy_from_slice(&self.volatility.to_le_bytes());
        buf
    }
}

// ── MarketPulse (packed binary) ─────────────────────────────────────────────

/// All fields needed to build one MarketPulse message.
#[derive(Debug, Clone, Default)]
pub struct MarketPulseData {
    pub timestamp_ns: u64,
    pub dnx: AssetTick,
    pub quai: AssetTick,
    pub qubic: AssetTick,
    pub kaspa: AssetTick,
    pub monero: AssetTick,
    pub ocean: AssetTick,
    pub verus: AssetTick,
    pub ocean_intel_confidence: f32,
    // ── Institutional Sensors ──────────────────────────────────────────────
    pub coinglass_funding_rate: f32,
    pub coinglass_liquidation_volume: f32,
    pub dex_liquidity_delta: f32,
    pub l3_order_imbalance: f32,
    // ── Hardware Proprioception ────────────────────────────────────────────
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub gpu_util_pct: f32,
    pub basys_uart_buffer_load: f32,
    // ── dYdX v4 Key-Free Signals ────────────────────────────────────────────
    pub dydx_oi_delta: f32,        // [100..104] BTC-USD OI normalised delta (+ve = accumulation)
    pub dydx_funding_rate: f32,    // [104..108] BTC-USD next funding rate
    // ── Qubic Global Computing Pulse (Triple Node Sync) ──────────────────
    /// Exponentially decaying trace of the last Qubic tick arrival.
    /// τ = 1.5s; reset to 1.0 on tick, decays per SNN step.
    pub qubic_tick_trace: f32,     // [108..112]
    /// Smoothed tick ingestion rate, normalized [0.0, 1.0].
    pub qubic_tick_rate: f32,      // [112..116]
    /// Linear progress through the ~7-day Qubic epoch [0.0, 1.0].
    pub qubic_epoch_progress: f32, // [116..120]
}

impl MarketPulseData {
    /// Serialize to 120-byte packed little-endian binary.
    ///
    /// Julia reads this via `reinterpret(Float32, buf[9:120])` —
    /// zero deserialization overhead (direct pointer cast).
    ///
    /// Layout v2 (Triple Node Sync): 108 + 12 bytes Qubic pulse = 120 bytes.
    pub fn serialize_packed(&self) -> [u8; 120] {
        let mut buf = [0u8; 120];
        buf[0..8].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        buf[8..16].copy_from_slice(&self.dnx.to_le_bytes());
        buf[16..24].copy_from_slice(&self.quai.to_le_bytes());
        buf[24..32].copy_from_slice(&self.qubic.to_le_bytes());
        buf[32..40].copy_from_slice(&self.kaspa.to_le_bytes());
        buf[40..48].copy_from_slice(&self.monero.to_le_bytes());
        buf[48..56].copy_from_slice(&self.ocean.to_le_bytes());
        buf[56..64].copy_from_slice(&self.verus.to_le_bytes());
        buf[64..68].copy_from_slice(&self.ocean_intel_confidence.to_le_bytes());
        // Institutional sensors
        buf[68..72].copy_from_slice(&self.coinglass_funding_rate.to_le_bytes());
        buf[72..76].copy_from_slice(&self.coinglass_liquidation_volume.to_le_bytes());
        buf[76..80].copy_from_slice(&self.dex_liquidity_delta.to_le_bytes());
        buf[80..84].copy_from_slice(&self.l3_order_imbalance.to_le_bytes());
        // Hardware proprioception
        buf[84..88].copy_from_slice(&self.gpu_temp_c.to_le_bytes());
        buf[88..92].copy_from_slice(&self.gpu_power_w.to_le_bytes());
        buf[92..96].copy_from_slice(&self.gpu_util_pct.to_le_bytes());
        buf[96..100].copy_from_slice(&self.basys_uart_buffer_load.to_le_bytes());
        // dYdX v4 key-free signals
        buf[100..104].copy_from_slice(&self.dydx_oi_delta.to_le_bytes());
        buf[104..108].copy_from_slice(&self.dydx_funding_rate.to_le_bytes());
        // Qubic Global Computing Pulse (Triple Node Sync)
        buf[108..112].copy_from_slice(&self.qubic_tick_trace.to_le_bytes());
        buf[112..116].copy_from_slice(&self.qubic_tick_rate.to_le_bytes());
        buf[116..120].copy_from_slice(&self.qubic_epoch_progress.to_le_bytes());
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_roundtrip() {
        let pulse = MarketPulseData {
            timestamp_ns: 1234567890,
            dnx: AssetTick::new(0.5, 0.3),
            quai: AssetTick::new(-0.2, 0.1),
            qubic: AssetTick::new(0.0, 0.0),
            kaspa: AssetTick::new(1.0, 0.9),
            monero: AssetTick::new(0.1, 0.05),
            ocean: AssetTick::new(-0.3, 0.2),
            verus: AssetTick::new(0.4, 0.15),
            ocean_intel_confidence: 0.75,
            coinglass_funding_rate: 0.0008,
            coinglass_liquidation_volume: 50_000.0,
            dex_liquidity_delta: -0.15,
            l3_order_imbalance: 0.35,
            gpu_temp_c: 65.0,
            gpu_power_w: 280.0,
            gpu_util_pct: 85.0,
            basys_uart_buffer_load: 0.3,
            dydx_oi_delta: 0.02,
            dydx_funding_rate: 0.0005,
            qubic_tick_trace: 0.75,
            qubic_tick_rate: 0.35,
            qubic_epoch_progress: 0.62,
        };

        let packed = pulse.serialize_packed();
        assert_eq!(packed.len(), 120);

        // Verify timestamp
        let ts = u64::from_le_bytes(packed[0..8].try_into().unwrap());
        assert_eq!(ts, 1234567890);

        // Verify btc price_norm
        let btc_price = f32::from_le_bytes(packed[8..12].try_into().unwrap());
        assert!((btc_price - 0.5).abs() < 1e-6);

        // Verify coinglass_funding_rate at offset 68
        let funding = f32::from_le_bytes(packed[68..72].try_into().unwrap());
        assert!((funding - 0.0008).abs() < 1e-6);

        // Verify gpu_temp_c at offset 84
        let temp = f32::from_le_bytes(packed[84..88].try_into().unwrap());
        assert!((temp - 65.0).abs() < 1e-6);

        // Verify dydx_oi_delta at offset 100
        let oi = f32::from_le_bytes(packed[100..104].try_into().unwrap());
        assert!((oi - 0.02).abs() < 1e-6);

        // Verify dydx_funding_rate at offset 104
        let fr = f32::from_le_bytes(packed[104..108].try_into().unwrap());
        assert!((fr - 0.0005).abs() < 1e-6);

        // Verify Qubic tick trace at offset 108
        let tick_trace = f32::from_le_bytes(packed[108..112].try_into().unwrap());
        assert!((tick_trace - 0.75).abs() < 1e-6);

        // Verify Qubic tick rate at offset 112
        let tick_rate = f32::from_le_bytes(packed[112..116].try_into().unwrap());
        assert!((tick_rate - 0.35).abs() < 1e-6);

        // Verify Qubic epoch progress at offset 116
        let epoch_prog = f32::from_le_bytes(packed[116..120].try_into().unwrap());
        assert!((epoch_prog - 0.62).abs() < 1e-6);
    }

    #[test]
    fn test_asset_tick_size() {
        assert_eq!(std::mem::size_of::<AssetTick>(), 8);
    }
}
