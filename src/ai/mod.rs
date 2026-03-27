//! AI subsystem — RAG, reranking, and prompt engineering.
//!
//! ARCHITECTURE (signal flow):
//! ```text
//!   ┌─────────────┐     ┌────────────┐     ┌──────────────────┐
//!   │ prompts.rs   │────>│ dispatcher │────>│  native-ai-worker│
//!   │ (identity)   │     │ (routing)  │     │  (Candle/GGUF)   │
//!   └─────────────┘     └────────────┘     └──────────────────┘
//!                              ^
//!   ┌─────────────┐           │
//!   │  rag.rs     │───────────┘
//!   │ (ingestion) │
//!   └─────────────┘
//!   ┌─────────────┐
//!   │ reranker.rs │  (bi-encoder re-scoring, no external calls)
//!   │ (Stage 2)   │
//!   └─────────────┘
//! ```
//!
//! Each file has ONE job — like dedicated ICs on a board.

pub mod prompts;
pub mod reranker;
#[cfg(feature = "image-gen")]
pub mod image_gen;
pub mod researcher;
pub mod mind_researcher;
