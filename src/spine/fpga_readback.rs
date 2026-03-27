/*
 * SHIP OF THESEUS - FPGA SPIKE READBACK MODULE
 * 
 * Handles UART communication with Basys3 FPGA to read spike states
 * and integrate them into the Rust trading system.
 */

use std::time::Duration;
use serialport::{SerialPort, SerialPortInfo};
use std::io::{Read, Write};

pub struct FpgaBridge {
    port: Box<dyn SerialPort>,
    active: bool,
}

impl FpgaBridge {
    /// Try to open FPGA connection on available USB ports
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Try common USB ports for Basys3
        let ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2"];
        
        for port_name in &ports {
            match serialport::new(*port_name, 115_200)
                .timeout(Duration::from_millis(100))
                .open() {
                Ok(port) => {
                    println!("[fpga] Connected to FPGA on {}", port_name);
                    return Ok(FpgaBridge {
                        port,
                        active: true,
                    });
                }
                Err(_) => continue,
            }
        }
        
        Err("FPGA not found on any USB port".into())
    }
    
    /// Send market stimuli to FPGA and read back spike states.
    ///
    /// Protocol (16-neuron SiliconBridge v3.0):
    ///   TX: 0xAA + 32 bytes (16 × Q8.8 stimuli)
    ///   RX: 32 bytes (16 × Q8.8 potentials) + 2 bytes (spike flags) + 2 bytes (switches)
    pub fn process_stimuli(&mut self, stimuli: &[f32; 16]) -> Result<(Vec<f32>, Vec<bool>), Box<dyn std::error::Error>> {
        if !self.active {
            return Err("FPGA bridge not active".into());
        }

        // Convert stimuli to Q8.8 format (16-bit fixed point)
        let mut tx_data = vec![0xAAu8]; // Sync byte
        for &s in stimuli {
            let q8_8 = (s.clamp(-255.0, 255.0) * 256.0) as i16;
            tx_data.extend_from_slice(&q8_8.to_be_bytes());
        }

        // Send to FPGA
        self.port.write_all(&tx_data)?;
        self.port.flush()?;

        // Read response: 32 bytes potentials + 2 bytes spike flags + 2 bytes switches
        let mut rx_data = vec![0u8; 36];
        self.port.read_exact(&mut rx_data)?;

        // Parse potentials (Q8.8 back to f32)
        let mut potentials = Vec::with_capacity(16);
        for i in 0..16 {
            let raw = i16::from_be_bytes([rx_data[i*2], rx_data[i*2+1]]);
            potentials.push(raw as f32 / 256.0);
        }

        // Parse spike flags (16-bit, 1 per neuron)
        let spike_word = u16::from_be_bytes([rx_data[32], rx_data[33]]);
        let spikes = (0..16).map(|i| (spike_word & (1 << i)) != 0).collect();
        // rx_data[34..36] = switch state (available but unused here)

        Ok((potentials, spikes))
    }
    
    /// Check if FPGA is responsive
    pub fn ping(&mut self) -> bool {
        let test_stimuli = [0.1; 16];
        match self.process_stimuli(&test_stimuli) {
            Ok(_) => true,
            Err(_) => {
                self.active = false;
                false
            }
        }
    }
    
    /// Get connection status
    pub fn is_active(&self) -> bool {
        self.active
    }
}

/// Find FPGA ports on the system
pub fn find_fpga_ports() -> Vec<SerialPortInfo> {
    match serialport::available_ports() {
        Ok(ports) => ports.into_iter()
            .filter(|p| p.port_name.contains("ttyUSB"))
            .collect(),
        Err(_) => Vec::new(),
    }
}
