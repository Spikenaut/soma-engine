use soma_engine::rag_ingest::ingester::ingest_roots;
use soma_engine::diagnostics::fault_class::KnowledgeBase;
use soma_engine::inference::prelude::ShipEmbedder;
use soma_engine::config::MEMORY_FILE;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("--- Ship-of-Theseus Native Ingester (CLI) ---");
    let mut kb = KnowledgeBase::load_or_new(MEMORY_FILE);
    
    println!("Creating native embedder (BERT)...");
    let embedder = ShipEmbedder::new(true)?;

    let roots = vec![
        PathBuf::from("/home/raulmc/ship_of_theseus_rs/research/Engineering Chemistry"),
        PathBuf::from("/home/raulmc/ship_of_theseus_rs/research/Math 2417"),
        PathBuf::from("/home/raulmc/ship_of_theseus_rs/research/Digital Logic"),
        // Add existing lecture roots too for safety
        PathBuf::from("/home/raulmc/Downloads/TXST Spring 2026 HW/EE.2320"),
    ];

    println!("Starting ingestion from: {:?}", roots);
    let count = ingest_roots(roots, &mut kb, &embedder, |msg| {
        println!("{}", msg);
    })?;

    println!("Saving knowledge base...");
    kb.save(MEMORY_FILE)?;

    println!("Done! Ingested {} files stored in memory.", count);
    Ok(())
}
