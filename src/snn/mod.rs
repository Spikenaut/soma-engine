// snn/mod.rs

pub mod lif;
pub mod izhikevich;
pub mod stdp;
pub mod modulators;
pub mod mining_reward;
pub mod neuromorphic_core;
pub mod neuromod_encoder;
pub mod engine;
pub mod ahl_router;

// Re-export common types for ease of use
pub use lif::{LifNeuron, PoissonEncoder};
pub use izhikevich::IzhikevichNeuron;
pub use stdp::*;
pub use modulators::NeuroModulators;
pub use mining_reward::MiningRewardState;
pub use neuromod_encoder::{NeuromodSensoryEncoder, ChannelStats};
pub use engine::{SpikingInferenceEngine, FpgaMetrics, NUM_INPUT_CHANNELS};
pub use ahl_router::{AhlRouter, VerificationDomain, RoutingDecision, DomainSignals};
