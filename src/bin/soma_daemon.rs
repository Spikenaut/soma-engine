use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use serde::Deserialize;
use tokio::signal;
use tokio::time;
use tracing::{error, info};

/// CLI arguments
#[derive(Parser, Debug)]
#[command(version, about = "Soma Spiking Network Daemon", long_about = None)]
struct Cli {
    /// Override configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,
}

/// Daemon configuration loaded from TOML
#[derive(Debug, Deserialize, Clone)]
struct DaemonConfig {
    tick_rate_hz: u32,
    log_level: String,
    spine_sub_port: u16,
    spine_pub_port: u16,
    model_path: PathBuf,
    network_size: usize,
}

impl DaemonConfig {
    fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let data = fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&data)?;
        Ok(cfg)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse CLI
    let cli = Cli::parse();

    // Resolve config path
    let default_path = dirs::config_dir()
        .unwrap_or(PathBuf::from("."))
        .join("soma/daemon.toml");
    let config_path = cli.config.unwrap_or(default_path);

    // Load configuration
    let cfg = match DaemonConfig::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load config {}: {e}", config_path.display());
            std::process::exit(1);
        }
    };

    // Initialize tracing
    std::env::set_var("RUST_LOG", &cfg.log_level);
    tracing_subscriber::fmt::init();

    info!("Loaded config from {}", config_path.display());

    // Prepare tick interval
    let tick_duration = Duration::from_micros(1_000_000 / cfg.tick_rate_hz as u64);
    let mut ticker = time::interval(tick_duration);

    // TODO: initialize SpikingNetwork from neuromod crate
    // For now, we stub with a placeholder struct.
    let mut network = neuromod::SpikingNetwork::new(cfg.network_size);

    // TODO: initialize ZMQ sockets (using existing spikenaut_spine helpers if available)
    // Placeholder: not implemented

    // Main loop
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                // Step the network
                network.step();
                // TODO: publish spikes
            }
            _ = signal::ctrl_c() => {
                info!("Termination signal received, shutting down");
                break;
            }
        }
    }

    Ok(())
}

// ----- Stub neuromod for compilation when the real crate is absent -----
#[allow(dead_code)]
mod neuromod {
    pub struct SpikingNetwork {
        size: usize,
    }

    impl SpikingNetwork {
        pub fn new(size: usize) -> Self {
            Self { size }
        }
        pub fn step(&mut self) {
            // placeholder implementation
        }
    }
}
