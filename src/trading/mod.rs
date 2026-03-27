//! Financial Lobe - Ghost Trading Engine
//!
//! Implements virtual trading with 0.1% fee attrition for Survival Mode.
//! Supports 7 mining chains: DNX, QUAI, QUBIC, KAS, XMR, OCEAN, VERUS with USDT base.

use std::collections::VecDeque;
use std::io::Write as _;
use chrono::Utc;
use serde::Serialize;

// ── Ghost wallet parameters ───────────────────────────────────────────────────

const GHOST_INITIAL_USDT: f32 = 500.0; // Survival Mode: $500
/// Fraction of the available balance committed per trade signal.
/// High-frequency trading: 8% per trade for more frequent execution
const GHOST_TRADE_FRACTION: f32 = 0.08;
// const LOG_PATH: &str = "DATA/research/spikenaut_market_v2.jsonl";  // Deprecated

/// Financial friction: 0.1% fee on all trades
const TRADING_FEE_RATE: f32 = 0.001;

// ─────────────────────────────────────────────────────────────────────────────
// Domain types
// ─────────────────────────────────────────────────────────────────────────────

/// One polling cycle's worth of market data.
#[derive(Clone)]
pub struct MarketFeed {
    pub dnx_price_usd:    f32,
    pub quai_price_usd:   f32,
    pub qubic_price_usd:  f32,
    pub kaspa_price_usd:  f32,
    pub monero_price_usd: f32,
    pub ocean_price_usd:  f32,
    pub verus_price_usd:  f32,
    pub gpu_temp_c:       f32,
    pub gpu_power_w:      f32,
    pub vddcr_gfx_v:      f32,
    pub fan_speed_pct:    f32,
    pub dnx_hashrate_mh:  f32,
    // ── Institutional Sensors (zeroed — mining coins lack deep perp/DEX markets) ──
    /// CoinGlass BTC perpetual funding rate (e.g. 0.0001 = 0.01%).
    /// Maps to Global Inhibition: high funding → raise V_thresh across all lobes.
    pub coinglass_funding_rate: f32,
    /// CoinGlass aggregate liquidation volume (USD). Stress signal.
    pub coinglass_liquidation_volume: f32,
    /// On-chain liquidity delta. Negative = liquidity withdrawal (risk).
    pub dex_liquidity_delta: f32,
    /// Order book imbalance [-1, +1]. +1 = strong bid pressure.
    pub l3_order_imbalance: f32,
    /// True if any channel fell back to previous values this tick.
    pub stale:            bool,
    // ── Full-Node Signals (Kaspa + Monero) ───────────────────────────────────
    /// Kaspa blocks ingested per second from local full node (ch3 auxiliary).
    pub kaspa_block_rate_hz: f32,
    /// Monero daemon sync height from local full node (progress signal).
    pub xmr_block_height: u64,
    /// Monero sync fraction [0.0, 1.0] — 1.0 = fully synced.
    pub xmr_sync_progress: f32,
    // ── Quai Blockchain Integration ───────────────────────────────────────────────
    /// Quai Network gas price in gwei
    pub quai_gas_price: f32,
    /// Quai Network transaction count per block
    pub quai_tx_count: u32,
    /// Quai Network block utilization percentage [0.0, 1.0]
    pub quai_block_utilization: f32,
    /// Quai Network staking ratio percentage [0.0, 1.0]
    pub quai_staking_ratio: f32,
    // ── Qubic Network Integration ─────────────────────────────────────────────────
    /// QU token price in USD (from CoinGecko or Qubic HTTP API).
    pub qu_price_usd: f32,
    /// Qubic tick rate (ticks/sec), normalized [0.0, 1.0].
    pub qubic_tick_rate: f32,
    /// Qubic epoch progress [0.0, 1.0].
    pub qubic_epoch_progress: f32,
}

impl Default for MarketFeed {
    fn default() -> Self {
        Self {
            dnx_price_usd:    0.0266,
            quai_price_usd:   0.02,
            qubic_price_usd:  0.000002,
            kaspa_price_usd:  0.08,
            monero_price_usd: 180.0,
            ocean_price_usd:  0.35,
            verus_price_usd:  0.45,
            gpu_temp_c: 40.0,
            gpu_power_w: 150.0,
            vddcr_gfx_v: 0.70,
            fan_speed_pct: 35.0,
            dnx_hashrate_mh: 0.012,
            coinglass_funding_rate: 0.0,
            coinglass_liquidation_volume: 0.0,
            dex_liquidity_delta: 0.0,
            l3_order_imbalance: 0.0,
            stale: false,
            kaspa_block_rate_hz: 0.0,
            xmr_block_height: 0,
            xmr_sync_progress: 0.0,
            quai_gas_price: 10.0,
            quai_tx_count: 100,
            quai_block_utilization: 0.65,
            quai_staking_ratio: 0.40,
            qu_price_usd: 0.0,
            qubic_tick_rate: 0.0,
            qubic_epoch_progress: 0.0,
        }
    }
}

/// Virtual ghost-trading wallet. 7 mining chains (DNX, QUAI, QUBIC, KAS, XMR, OCEAN, VERUS) with USDT base.
pub struct GhostWallet {
    pub balance_usdt:    f32,
    pub balance_dnx:     f32,
    pub balance_quai:    f32,
    pub balance_qubic:   f32,
    pub balance_kaspa:   f32,
    pub balance_monero:  f32,
    pub balance_ocean:   f32,
    pub balance_verus:   f32,
    /// Weighted-average cost basis of the current positions.
    pub entry_price_dnx:    f32,
    pub entry_price_quai:   f32,
    pub entry_price_qubic:  f32,
    pub entry_price_kaspa:  f32,
    pub entry_price_monero: f32,
    pub entry_price_ocean:  f32,
    pub entry_price_verus:  f32,
    pub cumulative_pnl: f32,
    pub trade_count:    u64,
    // ── Kelly position-sizing state ──────────────────────────────────────────
    pub win_count:    u64,
    pub loss_count:   u64,
    pub total_win:    f32,
    pub total_loss:   f32,
    pub trade_fraction: f32,
    pub price_history: VecDeque<f32>,
}

impl GhostWallet {
    pub fn new() -> Self {
        Self {
            balance_usdt:     GHOST_INITIAL_USDT,
            balance_dnx:      50.0,       // ~$1.33 worth at $0.0266
            balance_quai:     100.0,      // ~$2.00 worth at $0.02
            balance_qubic:    500000.0,   // ~$1.00 worth at $0.000002
            balance_kaspa:    25.0,       // ~$2.00 worth at $0.08
            balance_monero:   0.01,       // ~$1.80 worth at $180
            balance_ocean:    5.0,        // ~$1.75 worth at $0.35
            balance_verus:    5.0,        // ~$2.25 worth at $0.45
            entry_price_dnx:    0.0266,
            entry_price_quai:   0.02,
            entry_price_qubic:  0.000002,
            entry_price_kaspa:  0.08,
            entry_price_monero: 180.0,
            entry_price_ocean:  0.35,
            entry_price_verus:  0.45,
            cumulative_pnl:   0.0,
            trade_count:      0,
            win_count:        0,
            loss_count:       0,
            total_win:        0.0,
            total_loss:       0.0,
            trade_fraction:   GHOST_TRADE_FRACTION,
            price_history:    VecDeque::with_capacity(50),
        }
    }

    pub fn portfolio_value(
        &self,
        dnx_price: f32, quai_price: f32, qubic_price: f32, kaspa_price: f32,
        monero_price: f32, ocean_price: f32, verus_price: f32,
    ) -> f32 {
        self.balance_usdt
            + self.balance_dnx    * dnx_price
            + self.balance_quai   * quai_price
            + self.balance_qubic  * qubic_price
            + self.balance_kaspa  * kaspa_price
            + self.balance_monero * monero_price
            + self.balance_ocean  * ocean_price
            + self.balance_verus  * verus_price
    }

    /// Record a closed trade's PnL and update Kelly fraction via Julia.
    pub fn record_pnl_and_update_kelly(&mut self, pnl: f32) {
        if pnl > 0.0 {
            self.win_count  += 1;
            self.total_win  += pnl;
        } else if pnl < 0.0 {
            self.loss_count += 1;
            self.total_loss += pnl.abs();
        }

        let closed = self.win_count + self.loss_count;
        if closed < 10 { return; } // need minimum sample before trusting Kelly

        let win_rate = self.win_count as f64 / closed as f64;
        let avg_win  = if self.win_count  > 0 { self.total_win  as f64 / self.win_count  as f64 } else { 0.0 };
        let avg_loss = if self.loss_count > 0 { self.total_loss as f64 / self.loss_count as f64 } else { 0.0 };

        if avg_win < 1e-6 || avg_loss < 1e-6 { return; }

        let julia_bin = std::env::var("JULIA_BIN")
            .unwrap_or_else(|_| "julia".to_string());
        let script = "eagle-neuro/spikenaut-capital/execution/market_kelly.jl";

        let output = std::process::Command::new(&julia_bin)
            .arg(script)
            .env("SHIP_WIN_RATE", format!("{:.6}", win_rate))
            .env("SHIP_AVG_WIN",  format!("{:.6}", avg_win))
            .env("SHIP_AVG_LOSS", format!("{:.6}", avg_loss))
            .output();

        if let Ok(out) = output {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(f) = s.trim().parse::<f32>() {
                    self.trade_fraction = f;
                    println!(
                        "[kelly] win={:.0}% avg_win=${:.2} avg_loss=${:.2} → fraction={:.1}%",
                        win_rate * 100.0, avg_win, avg_loss, f * 100.0
                    );
                }
            }
        }
    }
}

/// JSONL record appended to `research/spikenaut_market_v2.jsonl`.
///
/// Schema v2 — adds SNN channel inputs, neuron spike outputs, and full-node
/// signals from Kaspa + Monero.  These fields are the training proof for the
/// Spikenaut Hugging Face dataset: every record shows *what the SNN saw* and
/// *what it decided*, alongside the financial outcome.
#[derive(Serialize)]
pub struct GhostTradeLog {
    /// Always "v2" — lets downstream tools gate on schema.
    pub schema_version:     &'static str,
    pub timestamp:          String,
    pub step:               u64,
    pub action:             String,
    pub asset:              String,
    pub price_usd:          f32,
    pub quantity:           f32,
    pub trade_value_usdt:   f32,
    pub realized_pnl_usdt:  f32,
    pub balance_usdt:       f32,
    pub balance_dnx:        f32,
    pub balance_quai:       f32,
    pub balance_qubic:      f32,
    pub balance_kaspa:      f32,
    pub balance_monero:     f32,
    pub balance_ocean:      f32,
    pub balance_verus:      f32,
    pub cumulative_pnl:     f32,
    pub portfolio_value:    f32,
    // ── Quai on-chain ──────────────────────────────────────────────────────
    pub quai_gas_price:         f32,
    pub quai_tx_count:          u32,
    pub quai_block_utilization: f32,
    pub quai_staking_ratio:     f32,
    // ── Full-node signals ──────────────────────────────────────────────────
    pub kaspa_block_rate_hz: f32,
    pub xmr_block_height:    u64,
    pub xmr_sync_progress:   f32,
    // ── SNN proof fields (Spikenaut training evidence) ─────────────────────
    /// Normalized 16-channel input vector fed into the SNN this tick.
    /// Indices match snn_config_baseline.json channel_mapping.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snn_channels:   Option<[f32; 16]>,
    /// Which of the 16 LIF neurons fired this tick (bear/bull/aux pairs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_spikes:  Option<[bool; 16]>,
    /// Blended dopamine reward signal used for STDP weight update.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mining_reward:  Option<f32>,
    pub reason:         String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Ghost trade execution and logging
// ─────────────────────────────────────────────────────────────────────────────

pub fn execute_buy(
    wallet: &mut GhostWallet,
    asset: &str,
    price: f32,
    step: u64,
    reason: &str,
    feed: &MarketFeed,
) {
    let spend_usdt = wallet.balance_usdt * wallet.trade_fraction;
    if spend_usdt < 0.01 { return; }

    // Apply 0.1% trading fee
    let fee = spend_usdt * TRADING_FEE_RATE;
    let net_spend = spend_usdt - fee;

    let qty = net_spend / price.max(1e-9);

    match asset {
        "DNX" => {
            let prev_cost = wallet.entry_price_dnx * wallet.balance_dnx;
            wallet.balance_dnx += qty;
            wallet.entry_price_dnx = (prev_cost + net_spend) / wallet.balance_dnx;
        }
        "QUAI" => {
            let prev_cost = wallet.entry_price_quai * wallet.balance_quai;
            wallet.balance_quai += qty;
            wallet.entry_price_quai = (prev_cost + net_spend) / wallet.balance_quai;
        }
        "QUBIC" => {
            let prev_cost = wallet.entry_price_qubic * wallet.balance_qubic;
            wallet.balance_qubic += qty;
            wallet.entry_price_qubic = (prev_cost + net_spend) / wallet.balance_qubic;
        }
        "KAS" => {
            let prev_cost = wallet.entry_price_kaspa * wallet.balance_kaspa;
            wallet.balance_kaspa += qty;
            wallet.entry_price_kaspa = (prev_cost + net_spend) / wallet.balance_kaspa;
        }
        "XMR" => {
            let prev_cost = wallet.entry_price_monero * wallet.balance_monero;
            wallet.balance_monero += qty;
            wallet.entry_price_monero = (prev_cost + net_spend) / wallet.balance_monero;
        }
        "OCEAN" => {
            let prev_cost = wallet.entry_price_ocean * wallet.balance_ocean;
            wallet.balance_ocean += qty;
            wallet.entry_price_ocean = (prev_cost + net_spend) / wallet.balance_ocean;
        }
        "VERUS" => {
            let prev_cost = wallet.entry_price_verus * wallet.balance_verus;
            wallet.balance_verus += qty;
            wallet.entry_price_verus = (prev_cost + net_spend) / wallet.balance_verus;
        }
        _ => return,
    }

    wallet.balance_usdt -= spend_usdt;
    wallet.trade_count += 1;

    let record = GhostTradeLog {
        schema_version: "v2",
        timestamp: Utc::now().to_rfc3339(),
        step,
        action: "buy".to_string(),
        asset: asset.to_string(),
        price_usd: price,
        quantity: qty,
        trade_value_usdt: spend_usdt,
        realized_pnl_usdt: -fee,
        balance_usdt: wallet.balance_usdt,
        balance_dnx: wallet.balance_dnx,
        balance_quai: wallet.balance_quai,
        balance_qubic: wallet.balance_qubic,
        balance_kaspa: wallet.balance_kaspa,
        balance_monero: wallet.balance_monero,
        balance_ocean: wallet.balance_ocean,
        balance_verus: wallet.balance_verus,
        cumulative_pnl: wallet.cumulative_pnl,
        portfolio_value: wallet.portfolio_value(
            feed.dnx_price_usd, feed.quai_price_usd, feed.qubic_price_usd,
            feed.kaspa_price_usd, feed.monero_price_usd, feed.ocean_price_usd,
            feed.verus_price_usd,
        ),
        quai_gas_price: feed.quai_gas_price,
        quai_tx_count: feed.quai_tx_count,
        quai_block_utilization: feed.quai_block_utilization,
        quai_staking_ratio: feed.quai_staking_ratio,
        kaspa_block_rate_hz: feed.kaspa_block_rate_hz,
        xmr_block_height: feed.xmr_block_height,
        xmr_sync_progress: feed.xmr_sync_progress,
        snn_channels: None,
        neuron_spikes: None,
        mining_reward: None,
        reason: reason.to_string(),
    };
    append_ghost_log(&record);
    println!(
        "[ghost BUY ] step={:>5}  asset={:<6} qty={:.4} @ ${:.4}  fee=${:.4}  portf=${:.2}",
        step, asset, qty, price, fee, record.portfolio_value
    );
}

pub fn execute_sell(
    wallet: &mut GhostWallet,
    asset: &str,
    price: f32,
    step: u64,
    reason: &str,
    feed: &MarketFeed,
) {
    let (qty, entry_price) = match asset {
        "DNX"   => (wallet.balance_dnx    * wallet.trade_fraction, wallet.entry_price_dnx),
        "QUAI"  => (wallet.balance_quai   * wallet.trade_fraction, wallet.entry_price_quai),
        "QUBIC" => (wallet.balance_qubic  * wallet.trade_fraction, wallet.entry_price_qubic),
        "KAS"   => (wallet.balance_kaspa  * wallet.trade_fraction, wallet.entry_price_kaspa),
        "XMR"   => (wallet.balance_monero * wallet.trade_fraction, wallet.entry_price_monero),
        "OCEAN" => (wallet.balance_ocean  * wallet.trade_fraction, wallet.entry_price_ocean),
        "VERUS" => (wallet.balance_verus  * wallet.trade_fraction, wallet.entry_price_verus),
        _ => return,
    };

    if qty < 1e-9 { return; }

    let proceeds = qty * price;
    let fee = proceeds * TRADING_FEE_RATE;
    let net_proceeds = proceeds - fee;

    let pnl = (price - entry_price) * qty - fee;

    match asset {
        "DNX"   => wallet.balance_dnx    -= qty,
        "QUAI"  => wallet.balance_quai   -= qty,
        "QUBIC" => wallet.balance_qubic  -= qty,
        "KAS"   => wallet.balance_kaspa  -= qty,
        "XMR"   => wallet.balance_monero -= qty,
        "OCEAN" => wallet.balance_ocean  -= qty,
        "VERUS" => wallet.balance_verus  -= qty,
        _ => return,
    }

    wallet.balance_usdt += net_proceeds;
    wallet.cumulative_pnl += pnl;
    wallet.trade_count += 1;
    wallet.record_pnl_and_update_kelly(pnl);

    let record = GhostTradeLog {
        schema_version: "v2",
        timestamp: Utc::now().to_rfc3339(),
        step,
        action: "sell".to_string(),
        asset: asset.to_string(),
        price_usd: price,
        quantity: qty,
        trade_value_usdt: proceeds,
        realized_pnl_usdt: pnl,
        balance_usdt: wallet.balance_usdt,
        balance_dnx: wallet.balance_dnx,
        balance_quai: wallet.balance_quai,
        balance_qubic: wallet.balance_qubic,
        balance_kaspa: wallet.balance_kaspa,
        balance_monero: wallet.balance_monero,
        balance_ocean: wallet.balance_ocean,
        balance_verus: wallet.balance_verus,
        cumulative_pnl: wallet.cumulative_pnl,
        portfolio_value: wallet.portfolio_value(
            feed.dnx_price_usd, feed.quai_price_usd, feed.qubic_price_usd,
            feed.kaspa_price_usd, feed.monero_price_usd, feed.ocean_price_usd,
            feed.verus_price_usd,
        ),
        quai_gas_price: feed.quai_gas_price,
        quai_tx_count: feed.quai_tx_count,
        quai_block_utilization: feed.quai_block_utilization,
        quai_staking_ratio: feed.quai_staking_ratio,
        kaspa_block_rate_hz: feed.kaspa_block_rate_hz,
        xmr_block_height: feed.xmr_block_height,
        xmr_sync_progress: feed.xmr_sync_progress,
        snn_channels: None,
        neuron_spikes: None,
        mining_reward: None,
        reason: reason.to_string(),
    };
    append_ghost_log(&record);
    println!(
        "[ghost SELL] step={:>5}  asset={:<6} qty={:.4} @ ${:.4}  fee=${:.4}  PnL={:+.2}  portf=${:.2}",
        step, asset, qty, price, fee, pnl, record.portfolio_value
    );
}

/// Append-only JSONL logger for ghost trading records (v2 format)
pub fn append_ghost_log(record: &GhostTradeLog) {
    let path = "DATA/research/spikenaut_market_v2.jsonl";
    let Ok(line) = serde_json::to_string(record) else { return };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{}", line);
    }
}
