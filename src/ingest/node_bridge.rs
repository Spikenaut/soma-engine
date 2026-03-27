//! Cross-Process Node Bridge - IPC for live_supervisor ↔ market_pilot
//!
//! Provides high-performance crossbeam channels for sharing TripleSnapshot
//! between processes without bloating the main trading loop.

use crossbeam_channel::{self, Receiver, Sender, TryRecvError, RecvTimeoutError};
use crate::ingest::triple_bridge::{TripleSnapshot, spawn_triple_bridge};
use tokio::sync::watch;
use std::thread;
use std::time::Duration;

/// Cross-process bridge for node telemetry data
pub struct NodeBridge {
    /// Sender for publishing snapshots (used by live_supervisor)
    pub tx: Sender<TripleSnapshot>,
    /// Receiver for consuming snapshots (used by market_pilot)
    pub rx: Receiver<TripleSnapshot>,
    /// Buffer size for channel (trade-off between latency and memory)
    buffer_size: usize,
}

impl NodeBridge {
    /// Create new node bridge with specified buffer size
    pub fn new(buffer_size: usize) -> Self {
        let (tx, rx) = crossbeam_channel::bounded(buffer_size);
        
        Self {
            tx,
            rx,
            buffer_size,
        }
    }
    
    /// Create bridge with optimal defaults for 50Hz telemetry
    pub fn with_optimal_defaults() -> Self {
        // Buffer for ~2 seconds of data at 50Hz with some headroom
        Self::new(200)
    }
    
    /// Start the bridge worker that forwards from watch channel to crossbeam
    pub fn start_forwarder(self, watch_rx: watch::Receiver<TripleSnapshot>) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let mut last_snap = TripleSnapshot::default();
            let mut rx = watch_rx;
            
            loop {
                // Poll for changes without async (sync thread context)
                if rx.has_changed().unwrap_or(false) {
                    let current_snap = rx.borrow_and_update().clone();

                    // Only forward if data actually changed (reduces channel traffic)
                    if self.should_forward(&last_snap, &current_snap) {
                        match self.tx.send(current_snap.clone()) {
                            Ok(_) => {
                                last_snap = current_snap;
                            }
                            Err(_) => {
                                // Channel disconnected (market_pilot shutdown)
                                eprintln!("[node_bridge] Channel disconnected, stopping forwarder");
                                break;
                            }
                        }
                    }
                }

                // Small sleep to prevent busy-waiting
                thread::sleep(Duration::from_millis(10));
            }
        })
    }
    
    /// Determine if snapshot should be forwarded (reduces unnecessary traffic)
    fn should_forward(&self, last: &TripleSnapshot, current: &TripleSnapshot) -> bool {
        // Forward if any significant field changed
        last.dynex_hashrate_mh != current.dynex_hashrate_mh ||
        last.qubic_tick_number != current.qubic_tick_number ||
        last.quai_block_number != current.quai_block_number ||
        last.kaspa_block_count != current.kaspa_block_count ||
        last.xmr_block_height != current.xmr_block_height ||
        // Always forward if rewards changed (important for trading)
        last.dynex_share_found != current.dynex_share_found ||
        last.quai_block_mined != current.quai_block_mined ||
        last.qubic_solution_found != current.qubic_solution_found
    }
    
    /// Receive latest snapshot with timeout
    pub fn recv_timeout(&self, timeout: Duration) -> Result<TripleSnapshot, RecvTimeoutError> {
        self.rx.recv_timeout(timeout)
    }
    
    /// Try to receive without blocking
    pub fn try_recv(&self) -> Result<TripleSnapshot, TryRecvError> {
        self.rx.try_recv()
    }
    
    /// Get current buffer statistics
    pub fn stats(&self) -> BridgeStats {
        BridgeStats {
            buffer_size: self.buffer_size,
            current_len: self.rx.len(),
            is_full: self.rx.is_full(),
            is_empty: self.rx.is_empty(),
        }
    }
}

/// Bridge performance statistics
#[derive(Debug, Clone)]
pub struct BridgeStats {
    pub buffer_size: usize,
    pub current_len: usize,
    pub is_full: bool,
    pub is_empty: bool,
}

impl NodeBridge {
    /// Convenience function to create bridge and start forwarder in one call
    pub fn spawn_with_forwarder(buffer_size: usize) -> (Self, thread::JoinHandle<()>) {
        let bridge = Self::new(buffer_size);
        let watch_rx = spawn_triple_bridge();
        let forwarder = bridge.clone().start_forwarder(watch_rx);
        (bridge, forwarder)
    }
}

// Implement Clone for sharing across threads
impl Clone for NodeBridge {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            rx: self.rx.clone(),
            buffer_size: self.buffer_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_bridge_creation() {
        let bridge = NodeBridge::with_optimal_defaults();
        let stats = bridge.stats();
        
        assert_eq!(stats.buffer_size, 200);
        assert!(stats.is_empty);
        assert!(!stats.is_full);
    }
    
    #[test]
    fn test_should_forward_logic() {
        let bridge = NodeBridge::new(100);
        
        let mut snap1 = TripleSnapshot::default();
        snap1.dynex_hashrate_mh = 10.0;
        
        let mut snap2 = TripleSnapshot::default();
        snap2.dynex_hashrate_mh = 10.0; // Same value
        
        let mut snap3 = TripleSnapshot::default();
        snap3.dynex_hashrate_mh = 15.0; // Different value
        
        assert!(!bridge.should_forward(&snap1, &snap2)); // Should not forward
        assert!(bridge.should_forward(&snap1, &snap3));  // Should forward
    }
    
    #[test]
    fn test_bridge_communication() {
        let bridge = NodeBridge::new(10);
        
        // Test send/receive
        let test_snap = TripleSnapshot::default();
        bridge.tx.send(test_snap.clone()).unwrap();
        
        let received = bridge.try_recv().unwrap();
        assert_eq!(received.dynex_hashrate_mh, test_snap.dynex_hashrate_mh);
    }
}
