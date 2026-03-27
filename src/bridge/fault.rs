//! Fault detection and classification for mining operations

#[derive(Debug, Clone, PartialEq)]
pub enum MiningFaultClass {
    /// No fault detected
    None,
    /// Hardware-related fault
    Hardware,
    /// Network connectivity fault
    Network,
    /// Performance degradation
    Performance,
    /// Thermal issues
    Thermal,
    /// Power supply issues
    Power,
    /// Software/configuration error
    Software,
    /// Unknown fault type
    Unknown,
}

impl Default for MiningFaultClass {
    fn default() -> Self {
        Self::None
    }
}
