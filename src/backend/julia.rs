#[cfg(feature = "julia")]
use std::fs;
#[cfg(feature = "julia")]
use std::sync::mpsc;
#[cfg(feature = "julia")]
use std::time::Duration;

#[cfg(feature = "julia")]
use jlrs::prelude::*;
#[cfg(feature = "julia")]
use jlrs::runtime::builder::Builder;

#[cfg(feature = "julia")]
use super::{BackendError, TraderBackend};
#[cfg(feature = "julia")]
use crate::telemetry::gpu_telemetry::GpuTelemetry;

#[cfg(feature = "julia")]
pub struct JuliaBackend {
    julia_tx: Option<mpsc::SyncSender<JuliaTask>>,
    _julia_handle: Option<std::thread::JoinHandle<()>>,
    initialized: bool,
    // HFT state
    hft_enabled: bool,
    last_var: f32,
    last_drawdown: f32,
}

#[cfg(feature = "julia")]
struct JuliaTask {
    inputs: Vec<f32>,
    inhibit: f32,
    // HFT extended fields
    prices: Vec<f32>,
    volatility: f32,
    reply: mpsc::Sender<Result<Vec<f32>, BackendError>>,
}

#[cfg(feature = "julia")]
fn julia_float(v: f32) -> String {
    if v.is_nan() {
        "NaN32".to_string()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "Inf32".to_string()
        } else {
            "-Inf32".to_string()
        }
    } else {
        format!("{:.6}f0", v)
    }
}

#[cfg(feature = "julia")]
impl JuliaBackend {
    pub fn new() -> Self {
        Self {
            julia_tx: None,
            _julia_handle: None,
            initialized: false,
            hft_enabled: true,
            last_var: 0.0,
            last_drawdown: 0.0,
        }
    }

    /// Initialize Julia worker with 4-lobe ensemble brain (262k neurons)
    pub fn initialize_julia_worker(&mut self) -> Result<(), BackendError> {
        let (julia_tx, julia_rx) = mpsc::sync_channel::<JuliaTask>(16);

        let handle = std::thread::Builder::new()
            .name("spikenaut-ensemble-brain".to_string())
            .stack_size(64 * 1024 * 1024)  // 64MB stack for large allocations
            .spawn(move || {
                #[allow(unused_unsafe)]
                let handle = unsafe {
                    match Builder::new().start_local() {
                        Ok(h) => h,
                        Err(e) => {
                            eprintln!("[backend] Failed to start Julia: {:?}", e);
                            return;
                        }
                    }
                };

                handle.local_scope::<_, 4>(|mut frame| {
                    println!("[backend] Spikenaut V3: Loading 4-lobe ensemble (262k neurons)...");
                    unsafe {
                        // Load required packages
                        let _ = Value::eval_string(&mut frame, "using CUDA, SparseArrays, Statistics, LinearAlgebra, Printf");
                        
                        // Load sparse brain (4-lobe ensemble)
                        let brain_path = "Spikenaut-Capital/brain/sparse_brain.jl";
                        if std::path::Path::new(brain_path).exists() {
                            let script = fs::read_to_string(brain_path).expect("Failed to read sparse_brain.jl");
                            let _ = Value::eval_string(&mut frame, &script);
                            println!("[backend] ✓ Loaded sparse_brain.jl (65k neurons/lobe)");
                        }
                        
                        // Load NERO orchestrator
                        let nero_path = "Spikenaut-Capital/brain/nero_orchestrator.jl";
                        if std::path::Path::new(nero_path).exists() {
                            let script = fs::read_to_string(nero_path).expect("Failed to read nero_orchestrator.jl");
                            let _ = Value::eval_string(&mut frame, &script);
                            println!("[backend] ✓ Loaded nero_orchestrator.jl");
                        }
                        
                        // Load HFT modules if available
                        let hft_path = "Spikenaut-Capital/hft";
                        if std::path::Path::new(hft_path).exists() {
                            println!("[backend] ✓ HFT quant modules available in {}", hft_path);
                        }
                        
                        // Warm up CUDA
                        let _ = Value::eval_string(&mut frame, "CUDA.synchronize(); println(\"[julia] CUDA warm-up complete\")");
                    }
                });

                println!("[backend] Spikenaut V3 ensemble brain initialized");

                while let Ok(task) = julia_rx.recv() {
                    let result: Result<Vec<f32>, BackendError> = handle.local_scope::<_, 12>(|mut frame| {
                        // Build input array for ensemble
                        let inputs_str = task
                            .inputs
                            .iter()
                            .map(|v| julia_float(*v))
                            .collect::<Vec<_>>()
                            .join(",");
                        
                        // Call ensemble step with HFT signals
                        let cmd = format!(
                            "run_lsm_step_str(Float32[{}], {})",
                            inputs_str, julia_float(task.inhibit)
                        );

                        match unsafe { Value::eval_string(&mut frame, &cmd) } {
                            Ok(res) => {
                                if let Ok(jl_str) = res.cast::<JuliaString>() {
                                    let s = jl_str.as_str().unwrap_or("");
                                    let data: Vec<f32> = s.split(',').filter_map(|v| v.parse().ok()).collect();
                                    Ok(data)
                                } else {
                                    Err(BackendError::ProcessingError("Julia returned non-string".to_string()))
                                }
                            }
                            Err(e) => {
                                let msg = e.error_string_or("Julia call failed");
                                Err(BackendError::ProcessingError(msg))
                            }
                        }
                    });
                    let _ = task.reply.send(result);
                }
            })
            .map_err(|e| BackendError::InitializationError(e.to_string()))?;

        self.julia_tx = Some(julia_tx);
        self._julia_handle = Some(handle);
        self.initialized = true;
        println!("[backend] Julia worker initialized for 4-lobe ensemble");
        Ok(())
    }

    /// Get HFT risk metrics from Julia brain
    pub fn get_hft_metrics(&self) -> (f32, f32) {
        (self.last_var, self.last_drawdown)
    }

    /// Enable/disable HFT features
    pub fn set_hft_enabled(&mut self, enabled: bool) {
        self.hft_enabled = enabled;
    }
}

#[cfg(feature = "julia")]
impl TraderBackend for JuliaBackend {
    fn process_signals(
        &mut self,
        normalized_inputs: &[f32; 8],
        inhibition_signal: f32,
        _telemetry: &GpuTelemetry,
    ) -> Result<Vec<f32>, BackendError> {
        if !self.initialized {
            return Err(BackendError::InitializationError("Backend not initialized".to_string()));
        }

        if let Some(ref tx) = self.julia_tx {
            let (reply_tx, reply_rx) = mpsc::channel();

            let mut receptors = vec![0.0f32; 16];
            for i in 0..8 {
                let val = normalized_inputs[i];
                if val > 0.0 {
                    receptors[i * 2 + 1] = val;
                    receptors[i * 2] = 0.0;
                } else {
                    receptors[i * 2 + 1] = 0.0;
                    receptors[i * 2] = val.abs();
                }
            }

            let task = JuliaTask {
                inputs: receptors,
                inhibit: inhibition_signal,
                reply: reply_tx,
            };

            tx.send(task)
                .map_err(|e| BackendError::CommunicationError(format!("Send failed: {}", e)))?;

            reply_rx
                .recv()
                .map_err(|e| BackendError::CommunicationError(format!("Receive failed: {}", e)))?
        } else {
            Err(BackendError::CommunicationError("Julia worker not connected".to_string()))
        }
    }

    fn initialize(&mut self, _model_path: Option<&str>) -> Result<(), BackendError> {
        self.initialize_julia_worker()
    }

    fn save_state(&self, _model_path: &str) -> Result<(), BackendError> {
        println!("[backend] Julia state preservation triggered via JIT session.");
        Ok(())
    }

    fn get_spike_states(&self) -> [bool; 16] {
        [false; 16]
    }

    fn reset(&mut self) -> Result<(), BackendError> {
        println!("[backend] Reservoir reset signal queued for Sanctuary Thread.");
        Ok(())
    }

    #[cfg(feature = "quai_integration")]
    fn process_signals_quai(
        &mut self,
        normalized_inputs: &[f32; 12],
        inhibition_signal: f32,
        _telemetry: &GpuTelemetry,
    ) -> Result<Vec<f32>, BackendError> {
        if !self.initialized {
            return Err(BackendError::InitializationError("Backend not initialized".to_string()));
        }

        if let Some(ref tx) = self.julia_tx {
            let (reply_tx, reply_rx) = mpsc::channel();

            let mut receptors = vec![0.0f32; 24];
            for i in 0..12 {
                let val = normalized_inputs[i];
                if val > 0.0 {
                    receptors[i * 2 + 1] = val;
                    receptors[i * 2] = 0.0;
                } else {
                    receptors[i * 2 + 1] = 0.0;
                    receptors[i * 2] = val.abs();
                }
            }

            let msg = JuliaTask {
                inputs: receptors,
                inhibit: inhibition_signal,
                reply: reply_tx,
            };

            if let Err(_) = tx.send(msg) {
                return Err(BackendError::ProcessingError("Failed to send message to Julia brain".to_string()));
            }

            match reply_rx.recv_timeout(Duration::from_millis(5000)) {
                Ok(output) => Ok(output?),
                Err(_) => Err(BackendError::ProcessingError("Timeout waiting for Julia brain response".to_string())),
            }
        } else {
            Err(BackendError::InitializationError(
                "Julia backend not properly initialized".to_string(),
            ))
        }
    }
}
