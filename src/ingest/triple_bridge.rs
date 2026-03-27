//! Triple Bridge — simultaneous async RPC to Dynex, Quai, and Qubic nodes.
//!
//! Spawns a single Tokio task that polls all three local nodes in parallel
//! using `tokio::join!`. Results are published via a `watch` channel so the
//! supervisor can read the latest snapshot non-blockingly (Zero-Order Hold).
//!
//! Resource budget:
//!   - 1 Tokio task (no dedicated thread)
//!   - ~3 concurrent HTTP requests per cycle
//!   - ~2 KB heap for reqwest client + JSON buffers
//!   - Cycle time: max(Dynex, Quai, Qubic) RPC latency, typically <100ms local

use std::io::{BufRead as _, Seek as _, SeekFrom, Write as _};
use std::time::{Duration, Instant};
use serde::Deserialize;
use tokio::sync::watch;
use tokio::time::interval;
use crate::ingest::neuraxon_log_parser::{
    parse_action_potential,
    parse_qubic_heartbeat,
    ActionPotentialEvent,
    NeuraxonTelemetryAccumulator,
    NeuraxonTelemetryPoint,
    QubicHeartbeat,
};
use crate::ingest::kaspa_grpc::{KaspaGrpcClient, check_kaspa_node};

// ── RPC Endpoints (all localhost, set by docker-compose.triple-node.yml) ─────

/// Qubic HTTP API (binary-to-REST wrapper, localhost only).
const QUBIC_API: &str = "http://127.0.0.1:8099/tick-info";

/// Quai Node JSON-RPC — go-quai v2+ (colosseum) binds on 9001 (quai namespace).
/// Verified: quai_blockNumber works; eth_blockNumber does not exist.
const QUAI_RPC: &str = "http://127.0.0.1:9001";

/// Dynex miner telemetry — read from nvidia-smi via the existing
/// `HardwareBridge::read_nvidia_smi()`. No separate RPC needed;
/// the supervisor already polls NVML. We re-read here for the
/// unified snapshot only.
///
/// If a dedicated Dynex status API is available in future, set it here.
const _DYNEX_STATUS: &str = "http://127.0.0.1:18080/status";

/// Append-only JSONL log of confirmed Qubic ticks for SNN replay validation.
const QUBIC_TICK_LOG: &str = "/home/raulmc/ship_of_theseus_rs/qubic_ticks.jsonl";

/// Monero JSON-RPC endpoint
const MONERO_RPC: &str = "http://127.0.0.1:18081/json_rpc";

/// Optional line-oriented Neuraxon/Qubic log path.
///
/// Set via `NEURAXON_LOG_PATH` to stream parser outputs into `TripleSnapshot`.
const NEURAXON_LOG_PATH_ENV: &str = "NEURAXON_LOG_PATH";

/// Polling interval for the triple bridge worker.
const POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Per-RPC timeout. Keeps the bridge responsive even if one node hangs.
const RPC_TIMEOUT: Duration = Duration::from_millis(500);

// ── Unified Snapshot ─────────────────────────────────────────────────────────

/// Unified snapshot from all three nodes, published at ~1Hz.
///
/// All fields default to 0.0/0 so that missing data is benign.
/// The `InterpolatorBank` smooths these to 10Hz for the SNN.
#[derive(Debug, Clone)]
pub struct TripleSnapshot {
    // ── Dynex (from NVML, always available) ──────────────────────
    pub dynex_hashrate_mh: f32,
    pub dynex_power_w: f32,
    pub dynex_gpu_temp_c: f32,

    // ── Qubic (from HTTP API, when synced) ───────────────────────
    pub qubic_tick_number: u64,
    pub qubic_epoch: u32,
    pub qubic_tick_rate: f32,
    pub qubic_epoch_progress: f32,
    pub qu_price_usd: f32,

    // ── Quai (from JSON-RPC, when synced) ────────────────────────
    pub quai_gas_price: f32,
    pub quai_tx_count: u32,
    pub quai_block_utilization: f32,
    pub quai_block_number: u64,

    // ── Kaspa (from gRPC, real-time) ────────────────────────────
    pub kaspa_block_count: u64,
    pub kaspa_header_count: u64,
    pub kaspa_block_rate_hz: f32,
    pub kaspa_sync_progress: f32,

    // ── Monero (from JSON-RPC) ───────────────────────────────────
    pub xmr_block_height: u64,
    pub xmr_target_height: u64,
    pub xmr_sync_progress: f32,

    // ── Consensus Rewards ────────────────────────────────────────
    /// Set to true for exactly one cycle when a Dynex share is accepted.
    pub dynex_share_found: bool,
    /// Set to true for exactly one cycle when a Quai block is mined.
    pub quai_block_mined: bool,
    /// Set to true for exactly one cycle when a Qubic solution is validated.
    pub qubic_solution_found: bool,

    // ── Neuraxon log-derived telemetry (optional) ───────────────────────
    pub neuraxon_tick_number: u64,
    pub neuraxon_epoch: u32,
    pub neuraxon_its: f32,
    pub neuraxon_dopamine: f32,
    pub neuraxon_serotonin: f32,
    pub neuraxon_state_excitatory: u32,
    pub neuraxon_state_inhibitory: u32,
    pub neuraxon_state_neutral: u32,
    pub neuraxon_state_total: u32,
    pub neuraxon_action_potential_core: u32,
    pub neuraxon_action_potential_threshold: f32,
    pub neuraxon_heartbeat_tick: u64,
    pub neuraxon_heartbeat_sub_tick: u32,

    /// Wall-clock timestamp of this snapshot.
    pub timestamp: Instant,
}

impl Default for TripleSnapshot {
    fn default() -> Self {
        Self {
            dynex_hashrate_mh: 0.0,
            dynex_power_w: 0.0,
            dynex_gpu_temp_c: 0.0,
            qubic_tick_number: 0,
            qubic_epoch: 0,
            qubic_tick_rate: 0.0,
            qubic_epoch_progress: 0.0,
            qu_price_usd: 0.0,
            quai_gas_price: 0.0,
            quai_tx_count: 0,
            quai_block_utilization: 0.0,
            quai_block_number: 0,
            kaspa_block_count: 0,
            kaspa_header_count: 0,
            kaspa_block_rate_hz: 0.0,
            kaspa_sync_progress: 0.0,
            xmr_block_height: 0,
            xmr_target_height: 0,
            xmr_sync_progress: 0.0,
            dynex_share_found: false,
            quai_block_mined: false,
            qubic_solution_found: false,
            neuraxon_tick_number: 0,
            neuraxon_epoch: 0,
            neuraxon_its: 0.0,
            neuraxon_dopamine: 0.0,
            neuraxon_serotonin: 0.0,
            neuraxon_state_excitatory: 0,
            neuraxon_state_inhibitory: 0,
            neuraxon_state_neutral: 0,
            neuraxon_state_total: 0,
            neuraxon_action_potential_core: 0,
            neuraxon_action_potential_threshold: 0.0,
            neuraxon_heartbeat_tick: 0,
            neuraxon_heartbeat_sub_tick: 0,
            timestamp: Instant::now(),
        }
    }
}

#[derive(Debug, Default)]
struct NeuraxonTailState {
    offset: u64,
    accum: NeuraxonTelemetryAccumulator,
    last_telemetry: Option<NeuraxonTelemetryPoint>,
    last_heartbeat: Option<QubicHeartbeat>,
    last_action_potential: Option<ActionPotentialEvent>,
}

impl NeuraxonTailState {
    fn poll(&mut self, path: &std::path::Path) {
        let mut file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return,
        };

        let len = match file.metadata() {
            Ok(meta) => meta.len(),
            Err(_) => return,
        };

        // Handle truncation/rotation by restarting from beginning.
        if len < self.offset {
            self.offset = 0;
        }

        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return;
        }

        let mut reader = std::io::BufReader::new(file);
        let mut line = String::new();
        loop {
            line.clear();
            let read = match reader.read_line(&mut line) {
                Ok(n) => n,
                Err(_) => break,
            };
            if read == 0 {
                break;
            }

            if let Ok(bytes) = u64::try_from(read) {
                self.offset = self.offset.saturating_add(bytes);
            }

            let trimmed = line.trim_end_matches(['\n', '\r']);

            if let Some(tp) = self.accum.ingest_line(trimmed) {
                self.last_telemetry = Some(tp);
            }
            if let Some(hb) = parse_qubic_heartbeat(trimmed) {
                self.last_heartbeat = Some(hb);
            }
            if let Some(ap) = parse_action_potential(trimmed) {
                self.last_action_potential = Some(ap);
            }
        }
    }

    fn stamp_snapshot(&self, snap: &mut TripleSnapshot) {
        if let Some(tp) = self.last_telemetry {
            snap.neuraxon_tick_number = tp.tick;
            snap.neuraxon_epoch = tp.epoch;
            snap.neuraxon_its = tp.its;
            snap.neuraxon_dopamine = tp.dopamine;
            snap.neuraxon_serotonin = tp.serotonin;
            if let Some(state) = tp.state {
                snap.neuraxon_state_excitatory = state.excitatory;
                snap.neuraxon_state_inhibitory = state.inhibitory;
                snap.neuraxon_state_neutral = state.neutral;
                snap.neuraxon_state_total = state.total;
            }
        }

        if let Some(hb) = self.last_heartbeat {
            snap.neuraxon_heartbeat_tick = hb.tick;
            snap.neuraxon_heartbeat_sub_tick = hb.sub_tick;
        }

        if let Some(ap) = self.last_action_potential {
            snap.neuraxon_action_potential_core = ap.core;
            snap.neuraxon_action_potential_threshold = ap.threshold;
        }
    }
}

// ── RPC Response Shapes ──────────────────────────────────────────────────────

/// Qubic HTTP API `/tick-info` response (partial).
#[derive(Deserialize)]
struct QubicStatus {
    #[serde(rename = "tickInfo")]
    tick_info: Option<QubicTick>,
}

#[derive(Deserialize)]
struct QubicTick {
    tick: u64,
    #[serde(default)]
    epoch: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)] // TODO: Use duration for timing analysis
    duration: Option<u32>,
    #[serde(rename = "initialTick")]
    initial_tick: Option<u64>,
}

/// Quai JSON-RPC response wrapper.
#[derive(Deserialize)]
struct JsonRpcResponse {
    result: Option<serde_json::Value>,
}

/// Monero JSON-RPC response wrapper.
#[derive(Deserialize)]
struct MoneroRpcResponse {
    result: Option<MoneroInfo>,
}

#[derive(Deserialize)]
struct MoneroInfo {
    height: u64,
    target_height: u64,
    #[allow(dead_code)]
    synchronized: Option<bool>,
}

// ── Bridge Worker ────────────────────────────────────────────────────────────

/// Spawn the triple bridge worker. Returns a `watch::Receiver` for the
/// supervisor to read the latest snapshot.
///
/// The worker runs forever, polling all three nodes in parallel every
/// `POLL_INTERVAL`. Failures are logged and silently retried next cycle.
pub fn spawn_triple_bridge() -> watch::Receiver<TripleSnapshot> {
    let (tx, rx) = watch::channel(TripleSnapshot::default());

    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(RPC_TIMEOUT)
            .build()
            .expect("reqwest client");

        let mut ticker = interval(POLL_INTERVAL);
        let mut prev_qubic_tick: u64 = 0;
        let mut prev_quai_block: u64 = 0;
        let mut tick_count_window: [Instant; 16] = [Instant::now(); 16];
        let mut tick_window_pos: usize = 0;
        let mut cycle: u64 = 0;
        let mut prev_persist_tick: u64 = 0;
        let neuraxon_log_path = std::env::var(NEURAXON_LOG_PATH_ENV)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(std::path::PathBuf::from);
        let mut neuraxon_tail = neuraxon_log_path.as_ref().map(|_| NeuraxonTailState::default());
        
        // Kaspa gRPC client (lazy initialization)
        let mut kaspa_client: Option<KaspaGrpcClient> = None;

        if let Some(path) = neuraxon_log_path.as_ref() {
            eprintln!(
                "[triple_bridge] neuraxon log tail enabled: env={} path={}",
                NEURAXON_LOG_PATH_ENV,
                path.display()
            );
        }

        loop {
            ticker.tick().await;
            cycle += 1;

            // ── Parallel RPC to all nodes ──────────────────────────
            let (qubic_res, quai_res, monero_res) = tokio::join!(
                poll_qubic(&client),
                poll_quai(&client),
                poll_monero(&client),
            );

            // Dynex telemetry comes from NVML (already polled by supervisor).
            // Read it here via the existing HardwareBridge for the unified snapshot.
            let dynex = read_dynex_telemetry();

            // Kaspa gRPC polling (lazy init + graceful fallback)
            let kaspa_info = if check_kaspa_node() {
                if kaspa_client.is_none() {
                    match KaspaGrpcClient::new("127.0.0.1:16110").await {
                        Ok(client) => {
                            kaspa_client = Some(client);
                            eprintln!("[triple_bridge] Kaspa gRPC client connected");
                        }
                        Err(e) => {
                            eprintln!("[triple_bridge] Failed to connect Kaspa gRPC: {}", e);
                        }
                    }
                }
                
                if let Some(ref client) = kaspa_client {
                    match client.get_block_info().await {
                        Ok(info) => Some(info),
                        Err(e) => {
                            eprintln!("[triple_bridge] Kaspa gRPC error: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let mut snap = TripleSnapshot {
                dynex_hashrate_mh: dynex.0,
                dynex_power_w: dynex.1,
                dynex_gpu_temp_c: dynex.2,
                timestamp: Instant::now(),
                ..Default::default()
            };

            // ── Qubic ────────────────────────────────────────────────────
            if let Some((tick_num, epoch, epoch_progress)) = qubic_res {
                // Edge detection: new tick arrived?
                if tick_num > prev_qubic_tick && prev_qubic_tick > 0 {
                    tick_count_window[tick_window_pos] = Instant::now();
                    tick_window_pos = (tick_window_pos + 1) % tick_count_window.len();
                }
                prev_qubic_tick = tick_num;

                // Compute tick rate from rolling window
                let now = Instant::now();
                let recent: Vec<&Instant> = tick_count_window.iter()
                    .filter(|t| now.duration_since(**t).as_secs() < 30)
                    .collect();
                let tick_rate = if recent.len() > 1 {
                    recent.len() as f32 / 30.0
                } else {
                    0.0
                };

                snap.qubic_tick_number = tick_num;
                snap.qubic_epoch = epoch;
                snap.qubic_tick_rate = tick_rate;
                snap.qubic_epoch_progress = epoch_progress;
            }

            // ── Quai ─────────────────────────────────────────────────────
            if let Some(quai) = quai_res {
                if quai.block_number > prev_quai_block && prev_quai_block > 0 {
                    snap.quai_block_mined = true; // Edge: new block
                }
                prev_quai_block = quai.block_number;
                snap.quai_gas_price = quai.gas_price;
                snap.quai_tx_count = quai.tx_count;
                snap.quai_block_utilization = quai.block_utilization;
                snap.quai_block_number = quai.block_number;
            }

            // ── Kaspa (gRPC real-time) ───────────────────────────────
            if let Some(info) = kaspa_info {
                snap.kaspa_block_count = info.block_count;
                snap.kaspa_header_count = info.header_count;
                snap.kaspa_block_rate_hz = info.block_rate_hz;
                snap.kaspa_sync_progress = info.sync_progress;
            }

            // ── Monero (JSON-RPC) ───────────────────────────────────
            if let Some((height, target_height, sync_progress)) = monero_res {
                snap.xmr_block_height = height;
                snap.xmr_target_height = target_height;
                snap.xmr_sync_progress = sync_progress;
            }

            // ── JSONL tick persistence (append only, one entry per new tick) ──
            if snap.qubic_tick_number > 0 && snap.qubic_tick_number != prev_persist_tick {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let entry = format!(
                    "{{\"ts\":{ts},\"tick\":{tick},\"epoch\":{epoch},\"tick_rate\":{rate:.4},\"epoch_progress\":{prog:.4}}}\n",
                    ts = ts,
                    tick = snap.qubic_tick_number,
                    epoch = snap.qubic_epoch,
                    rate = snap.qubic_tick_rate,
                    prog = snap.qubic_epoch_progress,
                );
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(QUBIC_TICK_LOG)
                    .and_then(|mut f| f.write_all(entry.as_bytes()));
                prev_persist_tick = snap.qubic_tick_number;
            }

            // ── Neuraxon/Qubic log tail parser (optional) ──────────────────
            if let (Some(path), Some(tail)) = (neuraxon_log_path.as_ref(), neuraxon_tail.as_mut()) {
                tail.poll(path);
                tail.stamp_snapshot(&mut snap);
            }

            // Log before sending (snap is moved by send)
            if cycle % 60 == 0 {
                eprintln!(
                    "[triple_bridge] cycle={} | qubic_tick={} quai_block={} dynex_hash={:.4} MH/s kaspa_blocks={} xmr_height={}",
                    cycle, prev_qubic_tick, prev_quai_block, dynex.0,
                    snap.kaspa_block_count, snap.xmr_block_height
                );
            }

            // Publish
            let _ = tx.send(snap);
        }
    });

    rx
}

// ── Individual RPC Pollers ───────────────────────────────────────────────────

/// Poll Qubic HTTP API. Returns (tick_number, epoch, epoch_progress).
async fn poll_qubic(client: &reqwest::Client) -> Option<(u64, u32, f32)> {
    let resp = tokio::time::timeout(
        RPC_TIMEOUT,
        client.get(QUBIC_API).send(),
    )
    .await
    .ok()?
    .ok()?;

    let status: QubicStatus = resp.json().await.ok()?;
    let tick = status.tick_info?;
    let epoch = tick.epoch.unwrap_or(0);

    // /tick-info exposes the epoch anchor directly; prefer that when present.
    let ticks_per_epoch: u64 = 676;
    let tick_in_epoch = tick
        .initial_tick
        .map(|initial_tick| tick.tick.saturating_sub(initial_tick))
        .unwrap_or_else(|| tick.tick % ticks_per_epoch);
    let epoch_progress = (tick_in_epoch as f32 / ticks_per_epoch as f32).clamp(0.0, 1.0);

    Some((tick.tick, epoch, epoch_progress))
}

struct QuaiBlock {
    block_number: u64,
    gas_price: f32,
    tx_count: u32,
    block_utilization: f32,
}

/// Poll Quai JSON-RPC for latest block data.
/// Step 1: quai_blockNumber (verified working on port 9001, quai namespace).
/// Step 2: quai_getBlockByNumber with explicit hex number for gas/tx metrics.
async fn poll_quai(client: &reqwest::Client) -> Option<QuaiBlock> {
    // Step 1: current block number via native quai namespace.
    let num_body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "quai_blockNumber",
        "params": [],
        "id": 1
    });

    let num_resp = tokio::time::timeout(
        RPC_TIMEOUT,
        client.post(QUAI_RPC).json(&num_body).send(),
    )
    .await
    .ok()?
    .ok()?;

    let num_rpc: JsonRpcResponse = num_resp.json().await.ok()?;
    let block_hex = num_rpc.result?.as_str()?.to_owned();
    let block_number = parse_hex_u64(&block_hex)?;

    // Step 2: fetch block details by explicit number ("latest" not accepted).
    let blk_body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "quai_getBlockByNumber",
        "params": [block_hex, false],
        "id": 2
    });

    let blk_resp = tokio::time::timeout(
        RPC_TIMEOUT,
        client.post(QUAI_RPC).json(&blk_body).send(),
    )
    .await
    .ok()?
    .ok()?;

    let blk_rpc: JsonRpcResponse = blk_resp.json().await.ok()?;
    let result = blk_rpc.result?;

    let gas_used = parse_hex_u64(result.get("gasUsed")?.as_str()?).unwrap_or(0);
    let gas_limit = parse_hex_u64(result.get("gasLimit")?.as_str()?).unwrap_or(1);
    let gas_price_wei = parse_hex_u64(
        result.get("baseFeePerGas")
            .and_then(|v| v.as_str())
            .unwrap_or("0x0")
    ).unwrap_or(0);

    let tx_count = result.get("transactions")
        .and_then(|v| v.as_array())
        .map(|a| a.len() as u32)
        .unwrap_or(0);

    let block_utilization = if gas_limit > 0 {
        (gas_used as f32 / gas_limit as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Convert wei to gwei (1 gwei = 1e9 wei)
    let gas_price_gwei = gas_price_wei as f32 / 1e9;

    Some(QuaiBlock {
        block_number,
        gas_price: gas_price_gwei,
        tx_count,
        block_utilization,
    })
}

/// Read Dynex miner telemetry from NVML (same data as HardwareBridge).
/// Returns (hashrate_mh, power_w, gpu_temp_c).
fn read_dynex_telemetry() -> (f32, f32, f32) {
    let telem = crate::telemetry::gpu_telemetry::HardwareBridge::read_telemetry();
    (telem.hashrate_mh, telem.power_w, telem.gpu_temp_c)
}

/// Poll Monero JSON-RPC for blockchain info
async fn poll_monero(client: &reqwest::Client) -> Option<(u64, u64, f32)> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "get_info",
        "params": [],
        "id": 1
    });

    let resp = tokio::time::timeout(
        RPC_TIMEOUT,
        client.post(MONERO_RPC).json(&body).send(),
    )
    .await
    .ok()?
    .ok()?;

    let monero_resp: MoneroRpcResponse = resp.json().await.ok()?;
    let info = monero_resp.result?;

    let sync_progress = if info.target_height > 0 {
        info.height as f32 / info.target_height as f32
    } else {
        0.0
    };

    Some((info.height, info.target_height, sync_progress))
}

/// Parse a "0x..." hex string to u64.
fn parse_hex_u64(s: &str) -> Option<u64> {
    let stripped = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(stripped, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex_u64() {
        assert_eq!(parse_hex_u64("0x1a"), Some(26));
        assert_eq!(parse_hex_u64("0xff"), Some(255));
        assert_eq!(parse_hex_u64("0x0"), Some(0));
        assert_eq!(parse_hex_u64("ff"), Some(255));
        assert_eq!(parse_hex_u64("not_hex"), None);
    }

    #[test]
    fn test_triple_snapshot_default() {
        let snap = TripleSnapshot::default();
        assert_eq!(snap.dynex_hashrate_mh, 0.0);
        assert_eq!(snap.qubic_tick_number, 0);
        assert_eq!(snap.quai_block_number, 0);
        assert!(!snap.dynex_share_found);
        assert!(!snap.quai_block_mined);
        assert!(!snap.qubic_solution_found);
        assert_eq!(snap.neuraxon_tick_number, 0);
        assert_eq!(snap.neuraxon_epoch, 0);
        assert_eq!(snap.neuraxon_heartbeat_tick, 0);
    }

    #[test]
    fn neuraxon_tail_updates_snapshot_from_split_lines() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("neuraxon_tail_test_{unique}.log"));

        std::fs::write(
            &path,
            "[14:05:01] Tick: 7100026 | Epoch: 121 | Neuraxon State: [+31 -12 *6 /44]\n\
[14:05:01] ITS: 1250.45 | Dopamine: 0.158 | Serotonin: 0.142\n\
[14:05:01] ⚡ Action Potential Triggered on Core 2 [Threshold: 0.40]\n\
230726140501 A- 000:000(000). 7100026.67 [+26 -0 *129 /38] 52|12 30/39 Dynamic\n",
        )
        .expect("write test log");

        let mut tail = NeuraxonTailState::default();
        tail.poll(&path);

        let mut snap = TripleSnapshot::default();
        tail.stamp_snapshot(&mut snap);

        assert_eq!(snap.neuraxon_tick_number, 7_100_026);
        assert_eq!(snap.neuraxon_epoch, 121);
        assert_eq!(snap.neuraxon_state_total, 44);
        assert!((snap.neuraxon_its - 1250.45).abs() < f32::EPSILON);
        assert!((snap.neuraxon_dopamine - 0.158).abs() < f32::EPSILON);
        assert!((snap.neuraxon_serotonin - 0.142).abs() < f32::EPSILON);
        assert_eq!(snap.neuraxon_action_potential_core, 2);
        assert!((snap.neuraxon_action_potential_threshold - 0.40).abs() < f32::EPSILON);
        assert_eq!(snap.neuraxon_heartbeat_tick, 7_100_026);
        assert_eq!(snap.neuraxon_heartbeat_sub_tick, 67);

        let _ = std::fs::remove_file(path);
    }
}
