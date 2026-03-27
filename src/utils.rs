// utils — text hashing and chunking utilities.
use sha2::{Sha256, Digest};

/// SHA-256 hex digest of the text content.
pub fn hash_text(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Split text into semantic chunks of approximately `chunk_size` characters.
/// Splits on paragraph boundaries when possible, falls back to word boundaries.
pub fn chunk_text_semantic(content: &str, _ext: &str, chunk_size: usize) -> Vec<String> {
    if content.len() <= chunk_size {
        return vec![content.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < content.len() {
        let end = (start + chunk_size).min(content.len());
        // Try to find a paragraph break near `end`
        let split = content[start..end]
            .rfind("\n\n")
            .or_else(|| content[start..end].rfind('\n'))
            .or_else(|| content[start..end].rfind(' '))
            .map(|p| start + p + 1)
            .unwrap_or(end);

        chunks.push(content[start..split].trim().to_string());
        start = split;
    }

    chunks.into_iter().filter(|c| !c.is_empty()).collect()
}
