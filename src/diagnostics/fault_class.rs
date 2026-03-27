// diagnostics/fault_class — knowledge base and GPU fault classification.
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFragment {
    pub doc_id: String,
    pub source_path: String,
    pub location: String,
    pub content: String,
    pub hash: String,
    pub vector: Vec<f32>,
    pub course_tag: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct KnowledgeBase {
    pub fragments: Vec<MemoryFragment>,
}

impl KnowledgeBase {
    pub fn load_or_new(path: &str) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

/// GPU health / firmware fault classification used by the SNN engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultClass {
    /// GPU hardware fault — die is unresponsive.
    GpuDieDead,
    /// Firmware state machine is in a bad state.
    FirmwareStateBad,
    /// No fault detected — GPU is in healthy standby.
    HealthyStandby,
    /// Generic sensor fault.
    SensorFault,
    /// Generic actuator fault.
    ActuatorFault,
    /// Unknown fault class.
    Unknown,
}
