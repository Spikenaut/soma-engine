// ai/researcher — NeuromorphicSnapshot deserialized from JSONL training data.
use serde::{Serialize, Deserialize};
use crate::telemetry::gpu_telemetry::GpuTelemetry;
use crate::snn::SpikingInferenceEngine;
use crate::ingest::triple_bridge::TripleSnapshot;
use std::path::PathBuf;

/// One timestamped snapshot from the live GPU/mining telemetry JSONL stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicSnapshot {
    pub telemetry: GpuTelemetry,
}

/// Neuromorphic researcher for handling research data and analysis
pub struct NeuromorphicResearcher {
    research_path: PathBuf,
}

impl NeuromorphicResearcher {
    pub fn new(research_path: &str) -> Self {
        let path = PathBuf::from(research_path);
        let resolved = if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
            path.parent().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
        } else {
            path
        };

        Self {
            research_path: resolved,
        }
    }

    /// Append a single timestamped snapshot to the JSONL archive on disk.
    pub fn archive_snapshot(
        &self,
        telem: &GpuTelemetry,
        _engine: &SpikingInferenceEngine,
        _bridge: Option<&TripleSnapshot>,
    ) -> anyhow::Result<()> {
        use std::io::Write;
        let snap = NeuromorphicSnapshot { telemetry: telem.clone() };
        let line = serde_json::to_string(&snap)?;
        let path = self.research_path.join("neuromorphic_data.jsonl");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(f, "{}", line)?;
        Ok(())
    }

    /// Write a JSON context file for session continuity.
    pub fn export_continue_context(
        &self,
        telem: &GpuTelemetry,
        _engine: &SpikingInferenceEngine,
        _bridge: Option<&TripleSnapshot>,
    ) -> anyhow::Result<()> {
        let snap = NeuromorphicSnapshot { telemetry: telem.clone() };
        let path = self.research_path.join("continue_context.json");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(&snap)?)?;
        Ok(())
    }
}
