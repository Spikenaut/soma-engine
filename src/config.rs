pub const MEMORY_FILE: &str = "/home/raulmc/.ship_of_theseus/knowledge_base.json";

pub fn load_env() {
    // Load .env if present; ignore errors (file may not exist in all environments).
    if let Ok(path) = std::env::var("SHIP_ENV_FILE") {
        let _ = std::fs::read_to_string(&path).map(|contents| {
            for line in contents.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    std::env::set_var(k.trim(), v.trim());
                }
            }
        });
    }
}

pub fn lecture_roots() -> Vec<&'static str> {
    vec![
        "/home/raulmc/ship_of_theseus_rs/research/Engineering Chemistry",
        "/home/raulmc/ship_of_theseus_rs/research/Math 2417",
        "/home/raulmc/ship_of_theseus_rs/research/Digital Logic",
        "/home/raulmc/Downloads/TXST Spring 2026 HW/EE.2320",
    ]
}
