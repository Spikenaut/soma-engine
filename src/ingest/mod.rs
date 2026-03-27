//! Ingest — Triple Node Data Lake
//!
//! Simultaneous RPC bridge to Dynex, Quai, and Qubic local nodes.
//! Upsamples slow blockchain signals to 10Hz via State-Space Interpolation,
//! and emits consensus reward events (dopamine spikes) on successful solves.
//!
//! ```text
//!   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
//!   │  Dynex Miner  │   │  Quai Node   │   │  Qubic Core  │
//!   │  (GPU stats)  │   │  (go-quai)   │   │  (HTTP API)  │
//!   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
//!          │ 100ms            │ ~12s             │ 2-5s
//!          ▼                  ▼                  ▼
//!   ┌─────────────────────────────────────────────────────┐
//!   │              TripleBridge (async worker)             │
//!   │  State-Space Interpolator → 10Hz unified output     │
//!   │  Consensus reward → dopamine spike channel          │
//!   └─────────────────────┬─────────────────────┬─────────┘
//!                         │ watch::Receiver<TripleSnapshot>
//!                         ▼
//!                   live_supervisor.rs
//! ```

pub mod kaspa_grpc;
pub mod triple_bridge;
pub mod interpolator;
pub mod consensus_reward;
pub mod neuraxon_log_parser;
pub mod node_bridge;
#[cfg(feature = "julia")]
pub mod julia_bridge;
