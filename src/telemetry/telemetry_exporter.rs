//! Telemetry Exporter - CSV and JSON export for mining/sync telemetry
//!
//! This module handles exporting hardware telemetry data to CSV and JSONL formats
//! for SNN training and analysis.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use serde_json;
use crate::telemetry::mining_sync_telemetry::MiningSyncTelemetry;
use crate::telemetry::cpu_telemetry::SystemTelemetry;

/// Telemetry exporter supporting CSV and JSONL formats
pub struct TelemetryExporter {
    csv_writer: Option<BufWriter<std::fs::File>>,
    jsonl_writer: Option<BufWriter<std::fs::File>>,
    csv_header_written: bool,
    system_csv_header_written: bool, // Separate header flag for system telemetry
}

impl TelemetryExporter {
    /// Create new telemetry exporter
    pub fn new(csv_path: Option<&str>, jsonl_path: Option<&str>) -> Result<Self, Box<dyn std::error::Error>> {
        let csv_writer = if let Some(path) = csv_path {
            Some(Self::create_writer(path)?)
        } else {
            None
        };
        
        let jsonl_writer = if let Some(path) = jsonl_path {
            Some(Self::create_writer(path)?)
        } else {
            None
        };
        
        Ok(Self {
            csv_writer,
            jsonl_writer,
            csv_header_written: false,
            system_csv_header_written: false,
        })
    }
    
    /// Create buffered file writer
    fn create_writer(path: &str) -> Result<BufWriter<std::fs::File>, Box<dyn std::error::Error>> {
        // Ensure directory exists
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
            
        Ok(BufWriter::new(file))
    }
    
    /// Write CSV header
    fn write_csv_header(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut writer) = self.csv_writer {
            if !self.csv_header_written {
                let header = "timestamp,monero_pid,kaspa_pid,monero_cpu_percent,kaspa_cpu_percent,monero_memory_mb,kaspa_memory_mb,system_io_wait_percent,network_tx_mbps,network_rx_mbps,cpu_package_temp_c,gpu_temp_c,system_load_average\n";
                writer.write_all(header.as_bytes())?;
                writer.flush()?;
                self.csv_header_written = true;
            }
        }
        Ok(())
    }
    
    /// Export unified system telemetry (CPU + GPU) to configured formats
    pub fn export_system_telemetry(&mut self, telemetry: &SystemTelemetry) -> Result<(), Box<dyn std::error::Error>> {
        // Write CSV
        if let Some(ref mut writer) = self.csv_writer {
            if !self.system_csv_header_written {
                let header = "timestamp_ms,cpu_tctl_c,cpu_ccd1_c,cpu_ccd2_c,cpu_package_power_w,gpu_temp_c,gpu_power_w,vddcr_gfx_v,fan_speed_pct,gpu_clock_mhz,mem_clock_mhz,mem_util_pct\n";
                writer.write_all(header.as_bytes())?;
                writer.flush()?;
                self.system_csv_header_written = true;
            }
            
            let csv_line = format!(
                "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.3},{:.1},{:.0},{:.0},{:.1}\n",
                telemetry.timestamp_ms,
                telemetry.cpu_tctl_c,
                telemetry.cpu_ccd1_c,
                telemetry.cpu_ccd2_c,
                telemetry.cpu_package_power_w,
                telemetry.gpu_temp_c,
                telemetry.gpu_power_w,
                telemetry.vddcr_gfx_v,
                telemetry.fan_speed_pct,
                telemetry.gpu_clock_mhz,
                telemetry.mem_clock_mhz,
                telemetry.mem_util_pct
            );
            
            writer.write_all(csv_line.as_bytes())?;
        }
        
        // Write JSONL
        if let Some(ref mut writer) = self.jsonl_writer {
            let json_line = serde_json::to_string(telemetry)?;
            writer.write_all(json_line.as_bytes())?;
            writer.write_all(b"\n")?;
        }
        
        Ok(())
    }

    /// Export telemetry data to configured formats
    pub fn export_telemetry(&mut self, telemetry: &MiningSyncTelemetry) -> Result<(), Box<dyn std::error::Error>> {
        self.write_csv_header()?;
        
        // Write CSV
        if let Some(ref mut writer) = self.csv_writer {
            
            let csv_line = format!(
                "{},{},{},{:.2},{:.2},{:.1},{:.1},{:.2},{:.3},{:.3},{:.1},{:.1},{:.2}\n",
                telemetry.timestamp,
                telemetry.monero_pid.map_or("".to_string(), |p| p.to_string()),
                telemetry.kaspa_pid.map_or("".to_string(), |p| p.to_string()),
                telemetry.monero_cpu_percent,
                telemetry.kaspa_cpu_percent,
                telemetry.monero_memory_mb,
                telemetry.kaspa_memory_mb,
                telemetry.system_io_wait_percent,
                telemetry.network_tx_mbps,
                telemetry.network_rx_mbps,
                telemetry.cpu_package_temp_c,
                telemetry.gpu_temp_c,
                telemetry.system_load_average
            );
            
            writer.write_all(csv_line.as_bytes())?;
        }
        
        // Write JSONL
        if let Some(ref mut writer) = self.jsonl_writer {
            let json_line = serde_json::to_string(telemetry)?;
            writer.write_all(json_line.as_bytes())?;
            writer.write_all(b"\n")?;
        }
        
        // Flush both writers periodically
        if let Some(ref mut writer) = self.csv_writer {
            writer.flush()?;
        }
        if let Some(ref mut writer) = self.jsonl_writer {
            writer.flush()?;
        }
        
        Ok(())
    }
    
    /// Flush all pending writes
    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut writer) = self.csv_writer {
            writer.flush()?;
        }
        if let Some(ref mut writer) = self.jsonl_writer {
            writer.flush()?;
        }
        Ok(())
    }
}

impl Drop for TelemetryExporter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::mining_sync_telemetry::MiningSyncTelemetry;
    
    #[test]
    fn test_telemetry_exporter() {
        let exporter = TelemetryExporter::new(
            Some("/tmp/test_hardware_telemetry.csv"),
            Some("/tmp/test_hardware_telemetry.jsonl")
        ).unwrap();
        
        let telemetry = MiningSyncTelemetry {
            timestamp: "2026-03-19T22:30:00Z".to_string(),
            monero_pid: Some(1713319),
            kaspa_pid: Some(1714456),
            monero_cpu_percent: 45.6,
            kaspa_cpu_percent: 23.4,
            monero_memory_mb: 1024.0,
            kaspa_memory_mb: 512.0,
            system_io_wait_percent: 12.5,
            network_tx_mbps: 2.1,
            network_rx_mbps: 1.8,
            cpu_package_temp_c: 67.2,
            gpu_temp_c: 75.3,
            system_load_average: 2.8,
        };
        
        // Note: This would write to /tmp/ for testing
        // In real usage, files would be created in research/ directory
    }
}
