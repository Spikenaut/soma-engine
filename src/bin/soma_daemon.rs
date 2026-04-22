use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use corpus_ipc::{NeuralBackend, SpikeBatch, SpikeEvent, SpineMessage, ZmqBrainBackend};
use neuromod::{NeuroModulators, SpikingNetwork};
use serde::Deserialize;
use tokio::signal;
use tokio::time;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

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
    lif_count: usize,
    izh_count: usize,
    channels: usize,
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
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_new(cfg.log_level.clone()).unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("Loaded config from {}", config_path.display());

    if cfg.tick_rate_hz == 0 {
        anyhow::bail!("tick_rate_hz must be > 0");
    }

    // Prepare tick interval
    let tick_duration = Duration::from_micros(1_000_000 / cfg.tick_rate_hz as u64);
    let mut ticker = time::interval(tick_duration);

    // Initialize dynamic neuromod network (v0.4.0 API)
    let mut network = SpikingNetwork::with_dimensions(cfg.lif_count, cfg.izh_count, cfg.channels);

    // Initialize corpus-ipc ZMQ SUB backend for incoming stimuli/modulators.
    let mut ingress = ZmqBrainBackend::new();
    let readout_endpoint = format!("tcp://127.0.0.1:{}", cfg.spine_sub_port);
    // SAFETY: this daemon is single-threaded during initialization and no other
    // threads read env vars here.
    unsafe {
        std::env::set_var("SPIKENAUT_ZMQ_READOUT_IPC", &readout_endpoint);
    }
    ingress.initialize(Some(&cfg.model_path.to_string_lossy()))?;

    // ZeroMQ PUB socket for outbound spike events.
    let zmq_context = zmq::Context::new();
    let pub_socket = zmq_context.socket(zmq::PUB)?;
    pub_socket.bind(&format!("tcp://*:{}", cfg.spine_pub_port))?;
    info!(
        "Ingress SUB {} / Egress PUB tcp://*:{}",
        readout_endpoint, cfg.spine_pub_port
    );

    // Main loop
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let readout = match ingress.process_signals(&[]) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("Failed to receive from corpus-ipc backend: {e}");
                        continue;
                    }
                };

                let (stimuli, modulators) = decode_inputs(&readout, cfg.channels);
                let spike_ids = match network.step(&stimuli, &modulators) {
                    Ok(spikes) => spikes,
                    Err(e) => {
                        error!("Network step failed: {e:?}");
                        continue;
                    }
                };

                if let Err(e) = publish_spikes(&pub_socket, &spike_ids) {
                    warn!("Failed to publish spikes: {e}");
                }
            }
            _ = signal::ctrl_c() => {
                info!("Termination signal received, shutting down");
                break;
            }
        }
    }

    Ok(())
}

fn decode_inputs(readout: &[f32], channels: usize) -> (Vec<f32>, NeuroModulators) {
    let mut stimuli = vec![0.0; channels];
    let upto = readout.len().min(channels);
    stimuli[..upto].copy_from_slice(&readout[..upto]);

    let modulators = if readout.len() >= channels + 4 {
        NeuroModulators {
            dopamine: readout[channels],
            cortisol: readout[channels + 1],
            acetylcholine: readout[channels + 2],
            tempo: readout[channels + 3],
            aux_dopamine: 0.0,
        }
    } else {
        NeuroModulators::default()
    };

    (stimuli, modulators)
}

fn publish_spikes(pub_socket: &zmq::Socket, spike_ids: &[usize]) -> anyhow::Result<()> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let tick = now.as_millis() as u64;

    let spikes = spike_ids
        .iter()
        .filter_map(|&idx| u16::try_from(idx).ok())
        .map(|channel| SpikeEvent {
            channel,
            time: (tick & u32::MAX as u64) as u32,
            strength: 1.0,
        })
        .collect();

    let msg = SpineMessage::Spikes(SpikeBatch {
        session_id: None,
        batch_id: tick,
        timestamp: now.as_nanos() as u64,
        spikes,
        metadata: None,
    });
    let payload = serde_json::to_vec(&msg)?;
    pub_socket.send(payload, 0)?;
    Ok(())
}
