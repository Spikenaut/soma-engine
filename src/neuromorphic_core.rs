//! Neuromorphic Computing Core
//!
//! This module provides the fundamental building blocks for neuromorphic computing,
//! including spike encoders, neuron models, and inference engines.

// Re-export the main neuromorphic core from the snn module
// pub use crate::snn::neuromorphic_core::*; // Temporarily commented - unused import

// Re-export commonly used types for convenience
pub use crate::snn::neuromorphic_core::{
    PoissonEncoder, 
    LifNeuron, 
    IzhikevichNeuron, 
    SpikingInferenceEngine,
    FpgaMetrics
};
