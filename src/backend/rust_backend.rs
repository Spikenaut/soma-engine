use super::{BackendError, TraderBackend};
use crate::telemetry::gpu_telemetry::GpuTelemetry;

pub struct RustBackend {
    initialized: bool,
}

impl RustBackend {
    pub fn new() -> Self {
        Self { initialized: false }
    }
}

impl TraderBackend for RustBackend {
    fn process_signals(
        &mut self,
        normalized_inputs: &[f32; 8],
        _inhibition_signal: f32,
        _telemetry: &GpuTelemetry,
    ) -> Result<Vec<f32>, BackendError> {
        if !self.initialized {
            return Err(BackendError::InitializationError(
                "Rust backend not initialized".to_string(),
            ));
        }

        let mut output = vec![0.0f32; 16];
        for i in 0..8 {
            let val = normalized_inputs[i];
            if val > 0.0 {
                output[i * 2] = val;
                output[i * 2 + 1] = 0.0;
            } else {
                output[i * 2] = 0.0;
                output[i * 2 + 1] = val.abs();
            }
        }
        Ok(output)
    }

    #[cfg(feature = "quai_integration")]
    fn process_signals_quai(
        &mut self,
        normalized_inputs: &[f32; 12],
        _inhibition_signal: f32,
        _telemetry: &GpuTelemetry,
    ) -> Result<Vec<f32>, BackendError> {
        if !self.initialized {
            return Err(BackendError::InitializationError(
                "Rust backend not initialized".to_string(),
            ));
        }

        let mut output = vec![0.0f32; 24];
        for i in 0..12 {
            let val = normalized_inputs[i];
            if val > 0.0 {
                output[i * 2] = val;
                output[i * 2 + 1] = 0.0;
            } else {
                output[i * 2] = 0.0;
                output[i * 2 + 1] = val.abs();
            }
        }
        Ok(output)
    }

    fn initialize(&mut self, _model_path: Option<&str>) -> Result<(), BackendError> {
        self.initialized = true;
        Ok(())
    }

    fn save_state(&self, _model_path: &str) -> Result<(), BackendError> {
        Ok(())
    }

    fn get_spike_states(&self) -> [bool; 16] {
        [false; 16]
    }

    fn reset(&mut self) -> Result<(), BackendError> {
        Ok(())
    }
}
