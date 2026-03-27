//! RAG Ingestion Engine — Hardware-Accelerated Knowledge Ingestion
//!
//! This module provides the logic for scanning directories, reading documents,
//! chunking text semantically, and generating embeddings via native Candle/BERT.
//!
//! By living in `soma-engine`, it can be used by both the headless 
//! `ingest_lectures` CLI and the graphical AI Tutor.

pub mod ingester {
    use std::path::{Path, PathBuf};
    use std::fs;
    use walkdir::WalkDir;
    use crate::inference::embeddings::ShipEmbedder;
    use tokio::runtime::Runtime;
    use crate::diagnostics::fault_class::{MemoryFragment, KnowledgeBase};
    use crate::utils::{hash_text, chunk_text_semantic};

    /// Ingests all supported documents from a list of root directories.
    pub fn ingest_roots(
        roots: Vec<PathBuf>, 
        kb: &mut KnowledgeBase, 
        embedder: &ShipEmbedder,
        status_callback: impl Fn(&str)
    ) -> anyhow::Result<usize> {
        let _rt = Runtime::new()?;
        let mut file_count = 0;
        let mut seen_hashes = std::collections::HashSet::new();
        
        // Initial pass to populate seen hashes
        for frag in &kb.fragments {
            seen_hashes.insert(frag.hash.clone());
        }

        for root in roots {
            if !root.exists() { continue; }
            status_callback(&format!("Scanning: {:?}", root));

            for entry in WalkDir::new(&root).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_dir() { continue; }

                let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                let supported = ["rs", "jl", "py", "sv", "csv", "pdf", "md", "json", "jsonl", "cu", "cuh", "xdc", "rpt", "tcl", "docx", "txt", "tex"];
                
                if !supported.contains(&ext.as_str()) { continue; }
                if path.to_string_lossy().contains("/target/") || path.to_string_lossy().contains("/.") {
                    continue;
                }

                file_count += 1;
                let file_name = path.file_name().unwrap_or_default().to_string_lossy();
                let mut status = String::with_capacity(64 + file_name.len());
                status.push_str("  [+] Ingesting: ");
                status.push_str(&file_name);
                status_callback(&status);

                if let Some(content) = read_document_text(path, &ext) {
                    let h = hash_text(&content);
                    if !seen_hashes.contains(&h) {
                        let course_tag = infer_course_tag(path, &ext);
                        let chunks = chunk_content(&content, &ext);
                        let doc_id = path.to_string_lossy().into_owned();
                        let source_path = doc_id.clone();

                        for (i, chunk) in chunks.into_iter().enumerate() {
                            if let Ok(vec) = embedder.embed(&chunk) {
                                let frag = MemoryFragment {
                                    doc_id: doc_id.clone(),
                                    source_path: source_path.clone(),
                                    location: i.to_string(),
                                    content: chunk,
                                    hash: h.clone(),
                                    vector: vec,
                                    course_tag: course_tag.clone(),
                                };
                                kb.fragments.push(frag);
                            }
                        }
                        seen_hashes.insert(h);
                    }
                }
            }
        }
        Ok(file_count)
    }

    fn chunk_content(content: &str, ext: &str) -> Vec<String> {
        if ext == "jsonl" {
            content.lines().filter(|l| !l.trim().is_empty()).map(|l| l.to_string()).collect()
        } else {
            let chunk_size = match ext {
                "rs" | "cu" | "cuh" | "sv" | "v" => 700,
                "tex" | "md" => 900,
                "py" | "jl" => 600,
                _ => 500,
            };
            chunk_text_semantic(content, ext, chunk_size)
        }
    }

    fn read_document_text(path: &Path, ext: &str) -> Option<String> {
        match ext {
            "pdf" => {
                match pdf_extract::extract_text(path) {
                    Ok(t) => Some(t),
                    Err(e) => {
                        eprintln!(
                            "[rag_ingest] pdf_extract failed for {:?}: {}",
                            path, e
                        );
                        None
                    }
                }
            }
            "docx" | "doc" => {
                use std::io::Read;
                let file = std::fs::File::open(path).ok()?;
                let mut archive = zip::ZipArchive::new(file).ok()?;
                let mut xml_entry = archive.by_name("word/document.xml").ok()?;
                let mut xml = String::new();
                xml_entry.read_to_string(&mut xml).ok()?;
                let mut result = String::with_capacity(xml.len() / 2);
                let mut in_tag = false;
                for ch in xml.chars() {
                    match ch {
                        '<' => in_tag = true,
                        '>' => in_tag = false,
                        c if !in_tag => result.push(c),
                        _ => {}
                    }
                }
                Some(result.trim().to_string())
            }
            _ => fs::read_to_string(path).ok(),
        }
    }

    fn infer_course_tag(path: &Path, ext: &str) -> Option<String> {
        let path_str = path.to_string_lossy().to_lowercase();
        let file_name = path.file_stem().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();

        if path_str.contains("ee.2320") || path_str.contains("ee2320") || file_name.contains("dlogic") {
            return Some("dlogic".to_string());
        }
        if path_str.contains("math.2417") || path_str.contains("math2417")
            || file_name.contains("math2417") || file_name.contains("precal")
        {
            return Some("math2417".to_string());
        }
        if path_str.contains("chem") {
            return Some("chem".to_string());
        }
        if matches!(ext, "cu" | "cuh" | "sv" | "xdc") {
            return Some("hw".to_string());
        }
        None
    }
}
