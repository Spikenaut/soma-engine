use std::path::PathBuf;
use soma_engine::diagnostics::fault_class::KnowledgeBase;
use soma_engine::rag_ingest::ingester;
use soma_engine::config::{MEMORY_FILE, load_env};
use soma_engine::inference::embeddings::ShipEmbedder;

fn main() -> anyhow::Result<()> {
    load_env();
    println!("Eagle-Lander: Standalone Lecture Ingester");

    let mut kb = KnowledgeBase::load_or_new(MEMORY_FILE);
    println!("Loading native BERT embedder...");
    let embedder = ShipEmbedder::new(true)?;

    let mut roots = vec![PathBuf::from("research")];
    for r in soma_engine::config::lecture_roots() {
        roots.push(PathBuf::from(r));
    }

    println!("Scanning roots for Digital Logic and Chemistry content...");

    let count = ingester::ingest_roots(roots, &mut kb, &embedder, |status| {
        println!("{}", status);
    })?;

    println!("\n✅ Ingestion complete. Scanned {} files.", count);
    
    kb.save(MEMORY_FILE)?;
    println!("💾 Knowledge Base saved to {}", MEMORY_FILE);

    Ok(())
}
