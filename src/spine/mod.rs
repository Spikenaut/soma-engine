//! Spine — ZeroMQ IPC Nervous System
//!
//! The Rust "Nervous System" that broadcasts zero-copy FlatBuffer market pulses
//! to the Julia "Brain" over `ipc:///tmp/spikenaut.ipc`.
//!
//! Architecture:
//! ```text
//!   WebSockets ──▶ Rust Ingestion ──▶ FlatBuffer ──▶ ZMQ PUB ──▶ Julia Brain
//!   (Ocean, BTC,    (zero-alloc)      (AssetTick     ipc://      (65,536-neuron
//!    RENDER, NVDA)                      structs)     /tmp/        CUDA LSM)
//! ```

pub mod market_pulse;
pub mod zmq_spine;
pub mod dydx_ingest;
pub mod fpga_readback;
