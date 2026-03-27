//! Bridge Module
//!
//! Connects the high-level application logic with low-level mining operations.

pub mod fault;
pub mod monitor;

// Re-export commonly used bridge functions
pub use fault::MiningFaultClass;
