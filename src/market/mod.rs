//! Sensory Lobe - Market Data Fetching & Normalization
//! 
//! Expands asset universe to 7 chains
//! Implements log returns normalization: R_t = ln(P_t / P_{t-1})

use std::collections::VecDeque;
use rand::Rng;
use serde::Deserialize;
use crate::trading::MarketFeed;

#[cfg(feature = "quai_integration")]
pub mod quai;

const COINGECKO_PRICE_URL: &str =
    "https://api.coingecko.com/api/v3/simple/price\
     ?ids=dynex,quai-network,qubic-network,kaspa,monero,ocean-protocol,verus-coin&vs_currencies=usd";

const DEFAULT_DNX_USD: f32 = 0.0266;
const DEFAULT_QUAI_USD: f32 = 0.02;
const DEFAULT_QUBIC_USD: f32 = 0.000002;
const DEFAULT_KASPA_USD: f32 = 0.08;
const DEFAULT_MONERO_USD: f32 = 180.0;
const DEFAULT_OCEAN_USD: f32 = 0.35;
const DEFAULT_VERUS_USD: f32 = 0.45;

#[derive(Deserialize)]
pub struct CoinGeckoSimple {
    pub dynex:  Option<CoinUsd>,
    #[serde(rename = "quai-network")]
    pub quai: Option<CoinUsd>,
    #[serde(rename = "qubic-network")]
    pub qubic: Option<CoinUsd>,
    pub kaspa: Option<CoinUsd>,
    pub monero: Option<CoinUsd>,
    #[serde(rename = "ocean-protocol")]
    pub ocean: Option<CoinUsd>,
    #[serde(rename = "verus-coin")]
    pub verus: Option<CoinUsd>,
}

#[derive(Deserialize)]
pub struct CoinUsd {
    pub usd: f32,
}

pub struct RollingStats {
    window: VecDeque<f32>,
    max_size: usize,
}

impl RollingStats {
    pub fn new(max_size: usize) -> Self {
        Self { window: VecDeque::with_capacity(max_size), max_size }
    }
    
    pub fn add(&mut self, val: f32) {
        if self.window.len() >= self.max_size { self.window.pop_front(); }
        self.window.push_back(val);
    }
    
    pub fn z_score(&self, val: f32) -> f32 {
        if self.window.len() < 2 { return 0.0; }
        let mean = self.window.iter().sum::<f32>() / self.window.len() as f32;
        let var = self.window.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.window.len() as f32;
        let std = var.sqrt();
        if !std.is_finite() || std < 1e-6 { 0.0 } else { (val - mean) / std }
    }
}

pub struct ChannelNormalizer {
    stats: RollingStats,
    last_val: Option<f32>,
}

impl ChannelNormalizer {
    pub fn new(window_size: usize) -> Self {
        Self { stats: RollingStats::new(window_size), last_val: None }
    }
    
    pub fn process(&mut self, val: f32) -> f32 {
        let log_ret = if let Some(last) = self.last_val {
            let ratio = val / last.max(1e-9);
            if ratio <= 0.0 { 0.0 } else { ratio.ln() }
        } else {
            0.0
        };
        self.last_val = Some(val);
        self.stats.add(log_ret);
        self.stats.z_score(log_ret)
    }
}

pub fn compute_log_returns(prices: &[f32]) -> Vec<f32> {
    if prices.len() < 2 {
        return Vec::new();
    }
    
    prices.windows(2)
        .map(|window| (window[1] / window[0].max(1e-9)).ln())
        .collect()
}

pub fn compute_volatility(dnx_abs: f32, quai_abs: f32, qbc_abs: f32, ocean_abs: f32) -> f32 {
    ((dnx_abs * dnx_abs + quai_abs * quai_abs + qbc_abs * qbc_abs + ocean_abs * ocean_abs) / 4.0)
        .sqrt()
        .clamp(0.0, 1.0)
}

pub async fn fetch_coingecko_prices(
    client: &reqwest::Client,
) -> Option<CoinGeckoSimple> {
    client
        .get(COINGECKO_PRICE_URL)
        .header("User-Agent", "Mozilla/5.0")
        .header("Accept", "application/json")
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json::<CoinGeckoSimple>()
        .await
        .ok()
}

fn fallback_price(prev: f32, default: f32) -> f32 {
    if prev > 0.0 { prev } else { default }
}

pub fn generate_mock_market(prev: &MarketFeed) -> MarketFeed {
    let mut rng = rand::thread_rng();
    
    let dnx = prev.dnx_price_usd * (1.0 + rng.gen_range(-0.005..0.005));
    let quai = prev.quai_price_usd * (1.0 + rng.gen_range(-0.005..0.005));
    let qubic = prev.qubic_price_usd * (1.0 + rng.gen_range(-0.005..0.005));
    let kas = prev.kaspa_price_usd * (1.0 + rng.gen_range(-0.005..0.005));
    let xmr = prev.monero_price_usd * (1.0 + rng.gen_range(-0.005..0.005));
    let oc = prev.ocean_price_usd * (1.0 + rng.gen_range(-0.005..0.005));
    let ver = prev.verus_price_usd * (1.0 + rng.gen_range(-0.005..0.005));
    
    let dex_delta = rng.gen_range(-0.2..0.2);
    
    MarketFeed {
        dnx_price_usd: dnx,
        quai_price_usd: quai,
        qubic_price_usd: qubic,
        kaspa_price_usd: kas,
        monero_price_usd: xmr,
        ocean_price_usd: oc,
        verus_price_usd: ver,
        gpu_temp_c: prev.gpu_temp_c,
        gpu_power_w: prev.gpu_power_w,
        vddcr_gfx_v: prev.vddcr_gfx_v,
        fan_speed_pct: prev.fan_speed_pct,
        dnx_hashrate_mh: prev.dnx_hashrate_mh,
        coinglass_funding_rate: 0.0,
        coinglass_liquidation_volume: 0.0,
        dex_liquidity_delta: dex_delta,
        l3_order_imbalance: 0.0,
        stale: false,
        kaspa_block_rate_hz: prev.kaspa_block_rate_hz,
        xmr_block_height: prev.xmr_block_height,
        xmr_sync_progress: prev.xmr_sync_progress,
        quai_gas_price: prev.quai_gas_price,
        quai_tx_count: prev.quai_tx_count,
        quai_block_utilization: prev.quai_block_utilization,
        quai_staking_ratio: prev.quai_staking_ratio,
        qu_price_usd: prev.qu_price_usd,
        qubic_tick_rate: prev.qubic_tick_rate,
        qubic_epoch_progress: prev.qubic_epoch_progress,
    }
}

pub async fn poll_market(client: &reqwest::Client, prev: &MarketFeed) -> MarketFeed {
    #[cfg(feature = "quai_integration")]
    let (cg, quai) = tokio::join!(
        fetch_coingecko_prices(client),
        quai::fetch_quai_data(client)
    );

    #[cfg(not(feature = "quai_integration"))]
    let cg = fetch_coingecko_prices(client).await;

    let dnx = cg.as_ref().and_then(|c| c.dynex.as_ref()).map(|u| u.usd)
        .unwrap_or_else(|| fallback_price(prev.dnx_price_usd, DEFAULT_DNX_USD));

    let quai_price = cg.as_ref().and_then(|c| c.quai.as_ref()).map(|u| u.usd)
        .unwrap_or_else(|| fallback_price(prev.quai_price_usd, DEFAULT_QUAI_USD));

    let qubic = cg.as_ref().and_then(|c| c.qubic.as_ref()).map(|u| u.usd)
        .unwrap_or_else(|| fallback_price(prev.qubic_price_usd, DEFAULT_QUBIC_USD));

    let kas = cg.as_ref().and_then(|c| c.kaspa.as_ref()).map(|u| u.usd)
        .unwrap_or_else(|| fallback_price(prev.kaspa_price_usd, DEFAULT_KASPA_USD));

    let xmr = cg.as_ref().and_then(|c| c.monero.as_ref()).map(|u| u.usd)
        .unwrap_or_else(|| fallback_price(prev.monero_price_usd, DEFAULT_MONERO_USD));

    let oc = cg.as_ref().and_then(|c| c.ocean.as_ref()).map(|u| u.usd)
        .unwrap_or_else(|| fallback_price(prev.ocean_price_usd, DEFAULT_OCEAN_USD));

    let ver = cg.as_ref().and_then(|c| c.verus.as_ref()).map(|u| u.usd)
        .unwrap_or_else(|| fallback_price(prev.verus_price_usd, DEFAULT_VERUS_USD));

    let gpu_temp = 40.0; 
    let funding = 0.0f32;
    let liq_vol = 0.0f32;
    let dex_delta = 0.0f32;
    let l3_imbalance = 0.0f32;

    #[cfg(feature = "quai_integration")]
    let quai_data = if let Some(q) = quai {
        q
    } else {
        quai::QuaiData {
            gas_price: prev.quai_gas_price,
            tx_count: prev.quai_tx_count,
            block_utilization: prev.quai_block_utilization,
            staking_ratio: prev.quai_staking_ratio,
        }
    };

    #[cfg(feature = "quai_integration")]
    let (quai_gas, quai_tx, quai_util, quai_stake) = (
        quai_data.gas_price, quai_data.tx_count,
        quai_data.block_utilization, quai_data.staking_ratio,
    );
    #[cfg(not(feature = "quai_integration"))]
    let (quai_gas, quai_tx, quai_util, quai_stake) = (
        prev.quai_gas_price, prev.quai_tx_count,
        prev.quai_block_utilization, prev.quai_staking_ratio,
    );

    MarketFeed {
        dnx_price_usd: dnx,
        quai_price_usd: quai_price,
        qubic_price_usd: qubic,
        kaspa_price_usd: kas,
        monero_price_usd: xmr,
        ocean_price_usd: oc,
        verus_price_usd: ver,
        gpu_temp_c: gpu_temp,
        gpu_power_w: 0.0,
        vddcr_gfx_v: 0.0,
        fan_speed_pct: 0.0,
        dnx_hashrate_mh: 0.0,
        coinglass_funding_rate: funding,
        coinglass_liquidation_volume: liq_vol,
        dex_liquidity_delta: dex_delta,
        l3_order_imbalance: l3_imbalance,
        stale: false,
        kaspa_block_rate_hz: prev.kaspa_block_rate_hz,
        xmr_block_height: prev.xmr_block_height,
        xmr_sync_progress: prev.xmr_sync_progress,
        quai_gas_price: quai_gas,
        quai_tx_count: quai_tx,
        quai_block_utilization: quai_util,
        quai_staking_ratio: quai_stake,
        qu_price_usd: prev.qu_price_usd,
        qubic_tick_rate: prev.qubic_tick_rate,
        qubic_epoch_progress: prev.qubic_epoch_progress,
    }
}
