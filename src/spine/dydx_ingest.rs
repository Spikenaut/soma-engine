//! dYdX v4 sensory lobe — key-free WebSocket ingestion of Open Interest
//! and Funding Rate for BTC-USD.
//!
//! Publishes a `DydxSnapshot` via a `tokio::sync::watch` channel so that
//! `zmq_spine.rs` can read the latest value non-blockingly (Zero-Order Hold).
//! If the WebSocket disconnects or is slow the last known value persists.

use std::time::Instant;

use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::time::{sleep, Duration};
use tokio_tungstenite::{connect_async, tungstenite::Message};

const DYDX_WS: &str = "wss://indexer.dydx.trade/v4/ws";
const SUBSCRIBE_MSGS: [&str; 2] = [
    r#"{"type":"subscribe","channel":"v4_markets","id":"BTC-USD"}"#,
    r#"{"type":"subscribe","channel":"v4_markets"}"#,
];

/// Latest dYdX BTC-USD values — consumed by the ZMQ spine via watch::Receiver.
#[derive(Debug, Clone)]
pub struct DydxSnapshot {
    /// Normalised OI change vs previous observation: (cur - prev) / prev.max(1).
    pub oi_delta: f32,
    /// Raw `nextFundingRate` from dYdX (e.g. 0.0005 = 0.05%).
    pub funding_rate: f32,
    pub last_updated: Instant,
}

impl Default for DydxSnapshot {
    fn default() -> Self {
        Self {
            oi_delta: 0.0,
            funding_rate: 0.0,
            last_updated: Instant::now(),
        }
    }
}

// ─── helpers ────────────────────────────────────────────────────────────────

fn as_f64(v: &Value) -> Option<f64> {
    if let Some(n) = v.as_f64() {
        return Some(n);
    }
    v.as_str()?.parse::<f64>().ok()
}

/// Try to pull (oi_or_price, funding_rate) out of a single market object.
///
/// Returns Some if the object contains:
///   • both `openInterest` + `nextFundingRate`  →  (oi, fr)
///   • `oraclePrice`                             →  (price, 0.0)
///
/// NOTE: never use `?` inside the `if let` tuple — that would early-return
/// from this function when `openInterest` is absent, bypassing the
/// oraclePrice fallback entirely.
fn try_extract_market_obj(v: &Value) -> Option<(f64, f64)> {
    let oi = v.get("openInterest").and_then(as_f64);
    let fr = v.get("nextFundingRate").and_then(as_f64);
    if let (Some(oi), Some(fr)) = (oi, fr) {
        return Some((oi, fr));
    }

    if let Some(price) = v.get("oraclePrice").and_then(as_f64) {
        return Some((price, 0.0));
    }

    None
}

/// Walk `v` looking for an `oraclePrices` map and extract the first asset's price.
fn search_for_any_oracle_price(v: &Value) -> Option<(f64, f64)> {
    match v {
        Value::Object(map) => {
            if let Some(Value::Object(assets)) = map.get("oraclePrices") {
                for (asset, data) in assets {
                    if let Some(pair) = try_extract_market_obj(data) {
                        eprintln!("[dydx_ingest] oracle pulse  asset={asset}  price={:.6}", pair.0);
                        return Some(pair);
                    }
                }
            }
            if let Some(pair) = try_extract_market_obj(v) {
                return Some(pair);
            }
            for child in map.values() {
                if let Some(pair) = search_for_any_oracle_price(child) {
                    return Some(pair);
                }
            }
            None
        }
        Value::Array(items) => items.iter().find_map(search_for_any_oracle_price),
        _ => None,
    }
}

/// Walk `v` preferring BTC-USD data but accepting any market object.
fn search_for_btc_market(v: &Value) -> Option<(f64, f64)> {
    match v {
        Value::Object(map) => {
            // Direct BTC-USD key
            if let Some(btc) = map.get("BTC-USD") {
                if let Some(pair) = try_extract_market_obj(btc) {
                    return Some(pair);
                }
            }

            // oraclePrices → BTC-USD
            if let Some(oracle_prices) = map.get("oraclePrices") {
                if let Some(btc_oracle) = oracle_prices.get("BTC-USD") {
                    if let Some(pair) = try_extract_market_obj(btc_oracle) {
                        return Some(pair);
                    }
                }
            }

            // Flat style: { "market": "BTC-USD", "openInterest": ... }
            let market_name = map
                .get("market")
                .and_then(Value::as_str)
                .or_else(|| map.get("ticker").and_then(Value::as_str));
            if market_name == Some("BTC-USD") {
                if let Some(pair) = try_extract_market_obj(v) {
                    return Some(pair);
                }
            }

            // Generic object check
            if let Some(pair) = try_extract_market_obj(v) {
                return Some(pair);
            }

            for child in map.values() {
                if let Some(pair) = search_for_btc_market(child) {
                    return Some(pair);
                }
            }
            None
        }
        Value::Array(items) => items.iter().find_map(search_for_btc_market),
        _ => None,
    }
}

/// Top-level extractor: tries BTC-specific data first, then any oracle price.
fn extract_market_fields(v: &Value) -> Option<(f64, f64)> {
    let target = v.get("contents").unwrap_or(v);
    search_for_btc_market(target).or_else(|| search_for_any_oracle_price(target))
}

/// True when a message comes from the v4_markets channel (any known subtype).
fn is_v4_markets_msg(v: &Value) -> bool {
    let msg_type = v.get("type").and_then(Value::as_str).unwrap_or("");
    let channel  = v.get("channel").and_then(Value::as_str).unwrap_or("");
    let known_type = matches!(
        msg_type,
        "channel_data" | "channel_batch_data" | "subscribed" | "channel_heartbeat"
    );
    let known_chan = matches!(channel, "v4_markets" | "markets");
    known_type && known_chan
}


// ─── main loop ──────────────────────────────────────────────────────────────

pub async fn run_dydx_ingest(tx: tokio::sync::watch::Sender<DydxSnapshot>) {
    let mut retry_delay_secs: u64 = 5;
    let mut prev_oi: f64 = 0.0;

    loop {
        match connect_and_stream(&tx, &mut prev_oi).await {
            Ok(()) => {
                retry_delay_secs = 5;
            }
            Err(e) => {
                eprintln!("[dydx_ingest] WS error: {e}. Reconnecting in {retry_delay_secs}s");
            }
        }
        sleep(Duration::from_secs(retry_delay_secs)).await;
        retry_delay_secs = (retry_delay_secs * 2).min(120);
    }
}

async fn connect_and_stream(
    tx: &tokio::sync::watch::Sender<DydxSnapshot>,
    prev_oi: &mut f64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (mut ws, _) = connect_async(DYDX_WS).await?;
    eprintln!("[dydx_ingest] Connected to {DYDX_WS}");

    for msg in SUBSCRIBE_MSGS {
        ws.send(Message::Text(msg.into())).await?;
    }

    let mut parse_miss_count: u64 = 0;
    let mut last_funding_rate: f32 = 0.0;

    while let Some(msg) = ws.next().await {
        let msg = msg?;
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Binary(b) => match String::from_utf8(b.to_vec()) {
                Ok(s) => s,
                Err(_) => continue,
            },
            Message::Ping(p) => {
                ws.send(Message::Pong(p)).await?;
                continue;
            }
            Message::Close(_) => break,
            _ => continue,
        };

        let parsed: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Handle websocket control messages and keep the snapshot fresh.
        if let Some(msg_type) = parsed.get("type").and_then(Value::as_str) {
            if matches!(msg_type, "connected" | "subscribed" | "channel_heartbeat") {
                eprintln!("[dydx_ingest] WebSocket connection established");
                let _ = tx.send(DydxSnapshot {
                    oi_delta: 0.0,
                    funding_rate: last_funding_rate,
                    last_updated: Instant::now(),
                });
                continue;
            }
        }

        let from_markets = is_v4_markets_msg(&parsed);

        // Always print shape of v4_markets messages for easy diagnosis.
        if from_markets {
            // Removed debug spam - log_contents_shape(&parsed);
            // Keep snapshot timestamp alive so stale-gate doesn't trigger.
            let _ = tx.send(DydxSnapshot {
                oi_delta: 0.0,
                funding_rate: last_funding_rate,
                last_updated: Instant::now(),
            });
        }

        if let Some((value, fr)) = extract_market_fields(&parsed) {
            let oi_delta = if *prev_oi == 0.0 {
                0.0
            } else {
                ((value - *prev_oi) / prev_oi.max(1.0)) as f64
            };
            *prev_oi = value;
            last_funding_rate = fr as f32;

            let _ = tx.send(DydxSnapshot {
                oi_delta: oi_delta as f32,
                funding_rate: fr as f32,
                last_updated: Instant::now(),
            });
            parse_miss_count = 0;
        } else if from_markets {
            // Known v4_markets message (e.g. "trading" stats update) but no numeric
            // OI/oracle data we care about.  Heartbeat already sent above; silence log.
            parse_miss_count = 0;
        } else {
            parse_miss_count += 1;
            if parse_miss_count == 1 || parse_miss_count % 50 == 0 {
                let preview: String = text.chars().take(220).collect();
                eprintln!(
                    "[dydx_ingest] Unparsed message #{parse_miss_count}; preview={preview}"
                );
            }
        }
    }
    Ok(())
}
