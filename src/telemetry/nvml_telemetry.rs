//! NVML Hardware Telemetry Integration
//! 
//! Integrates raw NVML telemetry for Channels 5-7 (Power, Temp, Clock)
//! Standardizes GpuTelemetry struct to use Z-Score normalization

use std::collections::VecDeque;
use rand::Rng;
use crate::telemetry::gpu_telemetry::GpuTelemetry;
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::error::NvmlError as NvmlWrapperError;

/// NVML telemetry provider with Z-Score normalization
pub struct NvmlTelemetryProvider {
    // Rolling statistics for Z-score normalization
    power_stats: RollingStats,
    temp_stats: RollingStats,
    clock_stats: RollingStats,
    
    // NVML instance
    nvml: Option<Nvml>,
    
    // Telemetry state
    last_telemetry: Option<GpuTelemetry>,
}

impl NvmlTelemetryProvider {
    /// Create new NVML telemetry provider
    pub fn new() -> Result<Self, NvmlError> {
        let mut provider = Self {
            power_stats: RollingStats::new(100), // 100-sample window
            temp_stats: RollingStats::new(100),
            clock_stats: RollingStats::new(100),
            nvml: None,
            last_telemetry: None,
        };
        
        provider.initialize_nvml()?;
        Ok(provider)
    }
    
    /// Initialize NVML library and device
    fn initialize_nvml(&mut self) -> Result<(), NvmlError> {
        // Initialize NVML
        let nvml = Nvml::init()?;
        self.nvml = Some(nvml);
        Ok(())
    }
    
    /// Fetch current NVML telemetry and normalize to Z-scores
    pub fn fetch_normalized_telemetry(&mut self) -> Result<GpuTelemetry, NvmlError> {
        let nvml = self.nvml.as_ref().ok_or(NvmlError::DeviceNotInitialized)?;
        let device = nvml.device_by_index(0)?;
        
        // Fetch raw NVML data
        let power_usage = device.power_usage()? as f32 / 1000.0; // Convert mW to W
        let temperature = device.temperature(TemperatureSensor::Gpu)? as f32;
        let clock_graphics = device.clock_info(Clock::Graphics)? as f32;
        
        // Normalize to Z-scores
        let power_z = self.power_stats.z_score(power_usage);
        let temp_z = self.temp_stats.z_score(temperature);
        let clock_z = self.clock_stats.z_score(clock_graphics);
        
        // Update rolling statistics
        self.power_stats.add(power_usage);
        self.temp_stats.add(temperature);
        self.clock_stats.add(clock_graphics);
        
        // Create standardized telemetry with Z-scores
        let telemetry = GpuTelemetry {
            gpu_temp_c: temperature,
            power_w: power_usage,
            clock_mhz: clock_graphics,
            // Additional channels for normalized values
            power_z_score: power_z,
            temp_z_score: temp_z,
            clock_z_score: clock_z,
            ..GpuTelemetry::default()
        };
        
        self.last_telemetry = Some(telemetry.clone());
        Ok(telemetry)
    }
    
    /// Get last known telemetry (fallback if NVML fails)
    pub fn get_last_telemetry(&self) -> Option<GpuTelemetry> {
        self.last_telemetry.clone()
    }
    
    /// Get stress level based on Z-score deviations
    pub fn compute_stress_level(&self) -> f32 {
        if let Some(ref telemetry) = self.last_telemetry {
            // Combine absolute Z-scores for overall stress metric
            let power_stress = telemetry.power_z_score.abs();
            let temp_stress = telemetry.temp_z_score.abs();
            let clock_stress = telemetry.clock_z_score.abs();
            
            // Weight temperature more heavily (thermal stress is critical)
            (power_stress * 0.3 + temp_stress * 0.5 + clock_stress * 0.2)
                .clamp(0.0, 3.0) // Cap at 3 standard deviations
        } else {
            0.0
        }
    }
}

/// Rolling statistics for Z-score computation
#[derive(Debug, Clone)]
pub struct RollingStats {
    window: VecDeque<f32>,
    max_size: usize,
    mean: f32,
    variance: f32,
    computed: bool,
}

impl RollingStats {
    pub fn new(max_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(max_size),
            max_size,
            mean: 0.0,
            variance: 0.0,
            computed: false,
        }
    }
    
    pub fn add(&mut self, val: f32) {
        if self.window.len() >= self.max_size {
            if let Some(old) = self.window.pop_front() {
                // Remove old value from running statistics
                self.remove_from_stats(old);
            }
        }
        
        self.window.push_back(val);
        self.add_to_stats(val);
        self.computed = false;
    }
    
    pub fn z_score(&mut self, val: f32) -> f32 {
        if !self.computed {
            self.compute_stats();
        }
        
        if self.window.len() < 2 {
            return 0.0;
        }
        
        let std = self.variance.sqrt();
        if std < 1e-6 {
            0.0
        } else {
            (val - self.mean) / std
        }
    }
    
    fn add_to_stats(&mut self, val: f32) {
        let n = self.window.len() as f32;
        if n == 1.0 {
            self.mean = val;
            self.variance = 0.0;
        } else {
            let old_mean = self.mean;
            self.mean += (val - old_mean) / n;
            self.variance += (val - old_mean) * (val - self.mean);
        }
    }
    
    fn remove_from_stats(&mut self, val: f32) {
        let n = self.window.len() as f32;
        if n > 1.0 {
            let old_mean = self.mean;
            self.mean -= (val - old_mean) / n;
            self.variance -= (val - old_mean) * (val - self.mean);
        }
    }
    
    fn compute_stats(&mut self) {
        let n = self.window.len() as f32;
        if n > 1.0 {
            self.variance /= n - 1.0; // Sample variance
        }
        self.computed = true;
    }
    
    pub fn mean(&self) -> f32 {
        self.mean
    }
    
    pub fn std(&self) -> f32 {
        self.variance.sqrt()
    }
}

/// NVML error types
#[derive(Debug, thiserror::Error)]
pub enum NvmlError {
    #[error("NVML initialization failed: {0}")]
    InitializationError(String),
    
    #[error("Device not initialized")]
    DeviceNotInitialized,
    
    #[error("NVML error: {0}")]
    NvmlError(#[from] NvmlWrapperError),
    
    #[error("Telemetry fetch failed: {0}")]
    FetchError(String),
}

/// Fallback telemetry provider when NVML is unavailable
pub struct FallbackTelemetryProvider {
    simulated_power: f32,
    simulated_temp: f32,
    simulated_clock: f32,
    noise_factor: f32,
}

impl FallbackTelemetryProvider {
    pub fn new() -> Self {
        Self {
            simulated_power: 250.0, // Typical RTX 5080 power
            simulated_temp: 65.0,  // Typical temperature
            simulated_clock: 2200.0, // Typical boost clock
            noise_factor: 0.1,
        }
    }
    
    pub fn fetch_telemetry(&mut self) -> GpuTelemetry {
        // Add some realistic noise
        let mut rng = rand::thread_rng();
        
        let power = self.simulated_power + rng.gen_range(-self.noise_factor..self.noise_factor) * self.simulated_power;
        let temp = self.simulated_temp + rng.gen_range(-2.0..2.0);
        let clock = self.simulated_clock + rng.gen_range(-50.0..50.0);
        
        // Simulate thermal throttling at high temps
        let adjusted_clock = if temp > 80.0 {
            clock * (1.0 - (temp - 80.0) * 0.01) // Reduce clock by 1% per degree above 80°C
        } else {
            clock
        };
        
        GpuTelemetry {
            gpu_temp_c: temp,
            power_w: power,
            clock_mhz: adjusted_clock,
            power_z_score: 0.0, // No Z-score computation in fallback
            temp_z_score: 0.0,
            clock_z_score: 0.0,
            ..GpuTelemetry::default()
        }
    }
}

impl Default for FallbackTelemetryProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Telemetry provider factory
pub enum TelemetryProvider {
    Nvml(NvmlTelemetryProvider),
    Fallback(FallbackTelemetryProvider),
}

impl TelemetryProvider {
    /// Create appropriate telemetry provider based on system capabilities
    pub fn create() -> Self {
        match NvmlTelemetryProvider::new() {
            Ok(nvml_provider) => TelemetryProvider::Nvml(nvml_provider),
            Err(_) => {
                println!("[telemetry] NVML unavailable, using fallback telemetry");
                TelemetryProvider::Fallback(FallbackTelemetryProvider::new())
            }
        }
    }
    
    /// Fetch current telemetry (normalized if available)
    pub fn fetch_telemetry(&mut self) -> Result<GpuTelemetry, NvmlError> {
        match self {
            TelemetryProvider::Nvml(provider) => provider.fetch_normalized_telemetry(),
            TelemetryProvider::Fallback(provider) => Ok(provider.fetch_telemetry()),
        }
    }
    
    /// Get current stress level
    pub fn get_stress_level(&self) -> f32 {
        match self {
            TelemetryProvider::Nvml(provider) => provider.compute_stress_level(),
            TelemetryProvider::Fallback(_) => 0.5, // Moderate stress for fallback
        }
    }
}
