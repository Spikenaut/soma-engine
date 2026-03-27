pub mod completion;
pub mod cuda_bridge;
pub mod model;
pub mod neuro_bridge;
pub mod persona;
pub mod embeddings;
pub mod vision;

/// Convenience re-exports for downstream crates (e.g. the root native_ai_worker).
pub mod prelude {
    pub use super::model::ShipModel;
    pub use super::persona::AgentPersona;
    pub use super::neuro_bridge::{NeuroState, send_reward_to_supervisor};
    pub use super::embeddings::ShipEmbedder;
    pub use super::vision::VisionEmbedder;
}
