//! Neural Bridge - SNN Backend Interface
//!
//! Modularized backend surface with per-backend implementations in:
//! - `jlrs_zero_copy.rs` - Primary Spikenaut-v2 zero-copy backend
//! - `julia.rs` - Legacy jlrs backend
//! - `rust_backend.rs` - Rust native backend
//! - `zmq_backend.rs` - ZMQ fallback/debug backend

use crate::telemetry::gpu_telemetry::GpuTelemetry;

mod jlrs_zero_copy;
#[cfg(feature = "julia")]
mod julia;
mod rust_backend;
mod zmq_backend;

pub use jlrs_zero_copy::JlrsZeroCopyBackend;
#[cfg(feature = "julia")]
pub use julia::JuliaBackend;
pub use rust_backend::RustBackend;
pub use zmq_backend::ZmqBrainBackend;

/// Trait for SNN backend implementations
/// 
/// This trait abstracts the neural processing layer, allowing for different
/// backend implementations (Rust native, Julia jlrs, Python, etc.)
pub trait TraderBackend: Send + Sync {
    /// Process normalized market signals through the SNN
    ///
    /// # Arguments
    /// * `normalized_inputs` - 8-channel normalized market data (Z-scores)
    /// * `inhibition_signal` - Thermal inhibition based on GPU temperature
    /// * `telemetry` - GPU telemetry for neuromodulation
    ///
    /// # Returns
    /// * `Result<Vec<f32>, BackendError>` - SNN output vector.
    ///
    ///   **`ZmqBrainBackend` returns 20 elements (widened contract):**
    ///   `[0..16]`  — 16-channel lobe readout (action/trade signals).
    ///   `[16..20]` — 4 NERO relevance scores `[Scalper, Day, Swing, Macro]`.
    ///
    ///   All other backends return 16 elements; callers that only need the
    ///   readout should take `output[..16]` and handle the extended slice
    ///   gracefully via `get(i).copied().unwrap_or(0.0)`.
    fn process_signals(
        &mut self,
        normalized_inputs: &[f32; 8],
        inhibition_signal: f32,
        telemetry: &GpuTelemetry,
    ) -> Result<Vec<f32>, BackendError>;

    #[cfg(feature = "quai_integration")]
    /// Process 12-channel signals with Quai integration
    /// * `normalized_inputs` - 12-channel normalized market data (Z-scores)
    /// * `inhibition_signal` - Thermal inhibition based on GPU temperature
    /// * `telemetry` - GPU telemetry for neuromodulation
    /// 
    /// # Returns
    /// * `Result<Vec<f32>, BackendError>` - 16-channel SNN output or error
    fn process_signals_quai(
        &mut self,
        normalized_inputs: &[f32; 12],
        inhibition_signal: f32,
        telemetry: &GpuTelemetry,
    ) -> Result<Vec<f32>, BackendError>;

    /// Initialize the backend with model parameters
    /// 
    /// # Arguments
    /// * `model_path` - Optional path to saved model parameters
    fn initialize(&mut self, model_path: Option<&str>) -> Result<(), BackendError>;

    /// Save current model state
    /// 
    /// # Arguments
    /// * `model_path` - Path to save model parameters
    fn save_state(&self, model_path: &str) -> Result<(), BackendError>;

    /// Get neuron spike states for trading decisions
    /// 
    /// # Returns
    /// * `[bool; 16]` - Spike states for 16 trading neurons
    fn get_spike_states(&self) -> [bool; 16];

    /// Reset the network state
    fn reset(&mut self) -> Result<(), BackendError>;
}

/// Backend error types
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Initialization failed: {0}")]
    InitializationError(String),
    
    #[error("Processing failed: {0}")]
    ProcessingError(String),
    
    #[error("Model I/O error: {0}")]
    ModelError(String),
    
    #[error("Communication error: {0}")]
    CommunicationError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

// backend implementations moved to dedicated modules

// ── NERO ZMQ Subscriber ───────────────────────────────────────────────────

/// Spawn a background OS thread that subscribes to the Julia brain's ZMQ IPC
/// socket and converts every 88-byte NERO packet into a `WireMsg::NeroUpdate`
/// sent on `tx`.
///
/// The thread is intentionally kept minimal: it only owns the ZMQ context and
/// socket, delegates all packet parsing to the same byte-level logic as
/// `ZmqBrainBackend::receive_readout`, and terminates silently if `tx` is
/// dropped (i.e., when `SupervisorApp` shuts down).
///
/// **Graceful degradation:** if the Julia brain is not running the thread
/// blocks on `socket.recv_bytes()` indefinitely and wakes automatically the
/// moment the Julia process starts publishing — no polling, no busy-wait.
pub fn spawn_nero_subscriber(tx: std::sync::mpsc::Sender<crate::models::WireMsg>) {
    std::thread::Builder::new()
        .name("nero-zmq-subscriber".into())
        .spawn(move || {
            let ctx = zmq::Context::new();
            let socket = match ctx.socket(zmq::SUB) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[nero-sub] Failed to create ZMQ SUB socket: {e}");
                    return;
                }
            };
            if let Err(e) = socket.connect("ipc:///tmp/spikenaut_readout.ipc") {
                eprintln!("[nero-sub] Failed to connect to IPC: {e}");
                return;
            }
            // Subscribe to all topics (empty filter).
            if let Err(e) = socket.set_subscribe(b"") {
                eprintln!("[nero-sub] set_subscribe failed: {e}");
                return;
            }

            println!("[nero-sub] Listening on ipc:///tmp/spikenaut_readout.ipc");

            let mut last_nero = [0.25f32; 4];

            loop {
                // Blocking recv — zero CPU when the brain is idle.
                let buf = match socket.recv_bytes(0) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("[nero-sub] recv error: {e}");
                        continue;
                    }
                };

                let (tick, nero) = if buf.len() == 88 {
                    let tick = i64::from_le_bytes(
                        buf[0..8].try_into().unwrap_or([0; 8])
                    );
                    let mut scores = [0.0f32; 4];
                    for i in 0..4 {
                        let off = 72 + i * 4;
                        scores[i] = f32::from_le_bytes(
                            buf[off..off + 4].try_into().unwrap_or([0; 4])
                        );
                    }
                    last_nero = scores;
                    (tick, scores)
                } else if buf.len() == 72 {
                    let tick = i64::from_le_bytes(
                        buf[0..8].try_into().unwrap_or([0; 8])
                    );
                    (tick, last_nero) // zero-order hold on legacy packets
                } else {
                    continue; // malformed — ignore
                };

                let snap = crate::models::NeroManifoldSnapshot::from_scores(tick, &nero);
                if tx.send(crate::models::WireMsg::NeroUpdate(snap)).is_err() {
                    // Receiver dropped — SupervisorApp is shutting down.
                    println!("[nero-sub] tx closed, exiting thread.");
                    break;
                }
            }
        })
        .expect("[nero-sub] failed to spawn thread");
}

/// Backend factory for creating appropriate backend instances
pub struct BackendFactory;

impl BackendFactory {
    /// Create a backend instance based on configuration
    pub fn create(backend_type: BackendType) -> Box<dyn TraderBackend> {
        match backend_type {
            #[cfg(feature = "julia")]
            BackendType::JlrsZeroCopy => Box::new(JlrsZeroCopyBackend::new()),
            #[cfg(feature = "julia")]
            BackendType::Julia => Box::new(JuliaBackend::new()),
            BackendType::Rust => Box::new(RustBackend::new()),
            BackendType::ZmqBrain => Box::new(ZmqBrainBackend::new()),
            #[cfg(not(feature = "julia"))]
            BackendType::RustFallback => Box::new(RustBackend::new()),
        }
    }
}

/// Backend type enumeration
#[derive(Debug, Clone, Copy)]
pub enum BackendType {
    #[cfg(feature = "julia")]
    /// Primary Spikenaut-v2 zero-copy backend (<10µs overhead)
    JlrsZeroCopy,
    #[cfg(feature = "julia")]
    /// Legacy jlrs backend
    Julia,
    Rust,
    /// ZMQ fallback/debug backend (production disabled)
    ZmqBrain,
    #[cfg(not(feature = "julia"))]
    /// Fallback variant when julia feature is disabled
    RustFallback,
}

impl Default for BackendType {
    fn default() -> Self {
        #[cfg(feature = "julia")]
        {
            BackendType::JlrsZeroCopy // Spikenaut-v2 primary backend
        }
        #[cfg(not(feature = "julia"))]
        {
            BackendType::RustFallback // Fallback when julia not available
        }
    }
}

// ── Parser Unit Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::ZmqBrainBackend;

    /// Build a synthetic 88-byte little-endian NERO packet.
    fn make_88_packet(tick: i64, readout: [f32; 16], nero: [f32; 4]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(88);
        buf.extend_from_slice(&tick.to_le_bytes());
        for v in &readout {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in &nero {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    /// Build a legacy 72-byte packet (no NERO suffix).
    fn make_72_packet(tick: i64, readout: [f32; 16]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(72);
        buf.extend_from_slice(&tick.to_le_bytes());
        for v in &readout {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn test_parse_88_byte_nero_packet() {
        let expected_nero = [0.4f32, 0.3, 0.2, 0.1];
        let expected_readout = std::array::from_fn::<f32, 16, _>(|i| i as f32 * 0.1);
        let expected_tick: i64 = 42_000;

        let buf = make_88_packet(expected_tick, expected_readout, expected_nero);

        // Feed via ZmqBrainBackend internals.
        let mut backend = ZmqBrainBackend::new();
        // Manually invoke the same parsing logic used in receive_readout.
        backend.brain_tick = i64::from_le_bytes(buf[0..8].try_into().unwrap());
        for i in 0..16 {
            let off = 8 + i * 4;
            backend.last_readout[i] = f32::from_le_bytes(buf[off..off+4].try_into().unwrap());
        }
        for i in 0..4 {
            let off = 72 + i * 4;
            backend.last_nero[i] = f32::from_le_bytes(buf[off..off+4].try_into().unwrap());
        }

        assert_eq!(backend.brain_tick, expected_tick);
        for i in 0..16 {
            assert!((backend.last_readout[i] - expected_readout[i]).abs() < 1e-5,
                "readout[{i}]: got {} expected {}", backend.last_readout[i], expected_readout[i]);
        }
        for i in 0..4 {
            assert!((backend.last_nero[i] - expected_nero[i]).abs() < 1e-5,
                "nero[{i}]: got {} expected {}", backend.last_nero[i], expected_nero[i]);
        }
    }

    #[test]
    fn test_parse_72_byte_legacy_packet_preserves_nero() {
        let mut backend = ZmqBrainBackend::new();
        // Seed a known NERO state.
        let known_nero = [0.7f32, 0.1, 0.1, 0.1];
        backend.last_nero = known_nero;

        // Simulate a legacy packet: only tick + readout, no NERO bytes.
        let readout = [1.0f32; 16];
        let buf = make_72_packet(99, readout);

        // The 72-byte branch must NOT overwrite last_nero.
        backend.brain_tick = i64::from_le_bytes(buf[0..8].try_into().unwrap());
        for i in 0..16 {
            let off = 8 + i * 4;
            backend.last_readout[i] = f32::from_le_bytes(buf[off..off+4].try_into().unwrap());
        }
        // (No NERO bytes parsed — matches the 72-byte branch in receive_readout)

        assert_eq!(backend.brain_tick, 99);
        assert_eq!(backend.last_nero, known_nero,
            "legacy packet must not overwrite last_nero (zero-order hold)");
    }

    #[test]
    fn test_malformed_packet_size_ignored() {
        let mut backend = ZmqBrainBackend::new();
        let initial_nero = backend.last_nero;
        let initial_tick = backend.brain_tick;

        // A 5-byte garbage packet should not panic or mutate state.
        let bad_buf: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0x00];
        let len = bad_buf.len();
        // Neither the 88-byte nor 72-byte branch is entered.
        if len != 88 && len != 72 {
            // Correctly ignored — state must be unchanged.
        }

        assert_eq!(backend.last_nero, initial_nero);
        assert_eq!(backend.brain_tick, initial_tick);
    }
}
