use super::{BackendError, TraderBackend};
use crate::telemetry::gpu_telemetry::GpuTelemetry;

const READOUT_IPC: &str = "ipc:///tmp/spikenaut_readout.ipc";

struct SafeSocket {
    socket: zmq::Socket,
}

unsafe impl Send for SafeSocket {}
unsafe impl Sync for SafeSocket {}

pub struct ZmqBrainBackend {
    context: zmq::Context,
    sub_socket: Option<SafeSocket>,
    initialized: bool,
    pub(crate) last_readout: Vec<f32>,
    pub last_nero: [f32; 4],
    pub(crate) brain_tick: i64,
}

impl ZmqBrainBackend {
    pub fn new() -> Self {
        Self {
            context: zmq::Context::new(),
            sub_socket: None,
            initialized: false,
            last_readout: vec![0.0f32; 16],
            last_nero: [0.25f32; 4],
            brain_tick: 0,
        }
    }

    pub fn get_nero_scores(&self) -> [f32; 4] {
        self.last_nero
    }

    pub fn brain_tick(&self) -> i64 {
        self.brain_tick
    }

    fn receive_readout(&mut self) -> Result<Vec<f32>, BackendError> {
        let safe_socket = self
            .sub_socket
            .as_ref()
            .ok_or_else(|| BackendError::CommunicationError("SUB socket not connected".to_string()))?;
        let socket = &safe_socket.socket;

        match socket.recv_bytes(zmq::DONTWAIT) {
            Ok(buf) if buf.len() == 88 => {
                self.brain_tick = i64::from_le_bytes(buf[0..8].try_into().unwrap_or([0; 8]));
                for i in 0..16 {
                    let offset = 8 + i * 4;
                    self.last_readout[i] =
                        f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
                }
                for i in 0..4 {
                    let offset = 72 + i * 4;
                    self.last_nero[i] =
                        f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
                }
            }
            Ok(buf) if buf.len() == 72 => {
                self.brain_tick = i64::from_le_bytes(buf[0..8].try_into().unwrap_or([0; 8]));
                for i in 0..16 {
                    let offset = 8 + i * 4;
                    self.last_readout[i] =
                        f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
                }
            }
            Ok(buf) => {
                println!(
                    "[zmq-brain] Unexpected message size: {} bytes (expected 88)",
                    buf.len()
                );
            }
            Err(zmq::Error::EAGAIN) => {}
            Err(e) => {
                return Err(BackendError::CommunicationError(format!(
                    "ZMQ recv failed: {}",
                    e
                )));
            }
        }

        let mut out = Vec::with_capacity(20);
        out.extend_from_slice(&self.last_readout);
        out.extend_from_slice(&self.last_nero);
        Ok(out)
    }
}

impl TraderBackend for ZmqBrainBackend {
    fn process_signals(
        &mut self,
        _normalized_inputs: &[f32; 8],
        _inhibition_signal: f32,
        _telemetry: &GpuTelemetry,
    ) -> Result<Vec<f32>, BackendError> {
        if !self.initialized {
            return Err(BackendError::InitializationError(
                "ZMQ brain backend not initialized".to_string(),
            ));
        }
        self.receive_readout()
    }

    fn initialize(&mut self, _model_path: Option<&str>) -> Result<(), BackendError> {
        let socket = self
            .context
            .socket(zmq::SUB)
            .map_err(|e| BackendError::InitializationError(format!("ZMQ SUB socket: {}", e)))?;

        socket
            .set_subscribe(b"")
            .map_err(|e| BackendError::InitializationError(format!("ZMQ subscribe: {}", e)))?;
        socket
            .set_rcvhwm(16)
            .map_err(|e| BackendError::InitializationError(format!("ZMQ rcvhwm: {}", e)))?;
        socket.connect(READOUT_IPC).map_err(|e| {
            BackendError::InitializationError(format!(
                "ZMQ connect to {}: {} (is main_brain.jl running?)",
                READOUT_IPC, e
            ))
        })?;

        self.sub_socket = Some(SafeSocket { socket });
        self.initialized = true;
        println!("[zmq-brain] Connected to Julia Brain at {}", READOUT_IPC);
        Ok(())
    }

    fn save_state(&self, _model_path: &str) -> Result<(), BackendError> {
        println!("[zmq-brain] State lives in the Julia Brain process (CUDA VRAM)");
        Ok(())
    }

    fn get_spike_states(&self) -> [bool; 16] {
        let mut spikes = [false; 16];
        for i in 0..16 {
            spikes[i] = self.last_readout[i] > 0.5;
        }
        spikes
    }

    fn reset(&mut self) -> Result<(), BackendError> {
        self.last_readout = vec![0.0f32; 16];
        self.last_nero = [0.25f32; 4];
        self.brain_tick = 0;
        println!("[zmq-brain] Readout and NERO cache reset");
        Ok(())
    }

    #[cfg(feature = "quai_integration")]
    fn process_signals_quai(
        &mut self,
        _normalized_inputs: &[f32; 12],
        _inhibition_signal: f32,
        _telemetry: &GpuTelemetry,
    ) -> Result<Vec<f32>, BackendError> {
        if !self.initialized {
            return Err(BackendError::InitializationError(
                "ZMQ brain backend not initialized".to_string(),
            ));
        }
        self.receive_readout()
    }
}
