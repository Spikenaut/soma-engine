//! Quai Network Blockchain Integration
//!
//! Isolated module for Quai blockchain data fetching and processing.
//! This module is feature-gated to prevent any impact on existing trading functionality.

use serde::Deserialize;

/// Quai blockchain data structure for multi-chain metrics
#[derive(Deserialize, Debug, Clone)]
pub struct QuaiData {
    pub gas_price: f32,        // Gas price in gwei
    pub tx_count: u32,         // Transaction count per block
    pub block_utilization: f32, // Block utilization percentage [0.0, 1.0]
    pub staking_ratio: f32,    // Staking ratio percentage [0.0, 1.0]
}

impl Default for QuaiData {
    fn default() -> Self {
        Self {
            gas_price: 10.0,           // 10 gwei default
            tx_count: 100,              // 100 transactions default
            block_utilization: 0.65,   // 65% utilization default
            staking_ratio: 0.40,       // 40% staking default
        }
    }
}

/// Quai RPC response structure
#[derive(Deserialize, Debug, Clone)]
pub struct QuaiResponse {
    pub result: Option<QuaiResult>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct QuaiResult {
    pub gasPrice: Option<String>,      // Hex string like "0x1e8480"
    pub transactionCount: Option<String>, // Hex string like "0x64"
    pub blockUtilization: Option<f32>,
    pub stakingRatio: Option<f32>,
}

/// Quai Network RPC endpoints
pub mod endpoints {
    pub const MAINNET_RPC: &str = "https://rpc.qu.ai";
    pub const TESTNET_RPC: &str = "https://orchard.rpc.quai.network/cyprus1";
    
    // Default values for fallback
    pub const DEFAULT_GAS_PRICE: f32 = 10.0;      // gwei
    pub const DEFAULT_TX_COUNT: u32 = 100;
    pub const DEFAULT_BLOCK_UTIL: f32 = 0.65;     // 65%
    pub const DEFAULT_STAKING_RATIO: f32 = 0.40; // 40%
}

/// Parse hex string to f32 gwei value
/// Handles hex strings like "0x1e8480" -> 2000000 (gwei)
pub fn parse_hex_gwei(hex_str: &str) -> Option<f32> {
    if !hex_str.starts_with("0x") {
        return None;
    }
    
    let hex_part = &hex_str[2..];
    match u64::from_str_radix(hex_part, 16) {
        Ok(wei_value) => Some((wei_value as f32) / 1_000_000_000.0), // Convert wei to gwei
        Err(_) => None,
    }
}

/// Parse hex string to u32
/// Handles hex strings like "0x64" -> 100
pub fn parse_hex_u32(hex_str: &str) -> Option<u32> {
    if !hex_str.starts_with("0x") {
        return None;
    }
    
    let hex_part = &hex_str[2..];
    u32::from_str_radix(hex_part, 16).ok()
}

/// Fetch Quai blockchain data via JSON-RPC
/// This function is designed to be safe and never impact existing functionality
pub async fn fetch_quai_data(client: &reqwest::Client) -> Option<QuaiData> {
    // Check if Quai integration is enabled
    let quai_enabled = std::env::var("QUAI_ENABLED")
        .unwrap_or("false".to_string()) == "true";
    
    if !quai_enabled {
        return None; // Silently return None if disabled
    }
    
    // Use testnet if in development mode
    let rpc_url = if std::env::var("TRADING_MODE").unwrap_or_default() == "development" {
        endpoints::TESTNET_RPC
    } else {
        endpoints::MAINNET_RPC
    };
    
    let request_body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "quai_getBlockStats",
        "params": ["latest"],
        "id": 1
    });
    
    let response = client
        .post(rpc_url)
        .header("Content-Type", "application/json")
        .header("User-Agent", "SpikeLens-Market-Pilot/0.1")
        .json(&request_body)
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json::<QuaiResponse>()
        .await
        .ok()?;
    
    let result = response.result?;
    
    Some(QuaiData {
        gas_price: result.gasPrice
            .as_ref()
            .and_then(|s| parse_hex_gwei(s))
            .unwrap_or(endpoints::DEFAULT_GAS_PRICE),
        tx_count: result.transactionCount
            .as_ref()
            .and_then(|s| parse_hex_u32(s))
            .unwrap_or(endpoints::DEFAULT_TX_COUNT),
        block_utilization: result.blockUtilization
            .unwrap_or(endpoints::DEFAULT_BLOCK_UTIL),
        staking_ratio: result.stakingRatio
            .unwrap_or(endpoints::DEFAULT_STAKING_RATIO),
    })
}

/// Generate mock Quai data for testing
/// This function is used when QUAI_TESTING=true to avoid real API calls
pub fn generate_mock_quai_data(step: u64) -> QuaiData {
    let base_gas = 10.0;
    let base_tx = 100;
    let base_util = 0.65;
    let base_stake = 0.40;
    
    // Add realistic variations based on step
    let gas_variation = (step as f32 * 0.1).sin() * 5.0; // ±5 gwei variation
    let tx_variation = (step % 50) as i32 - 25; // ±25 transaction variation
    let util_variation = (step as f32 * 0.01).cos() * 0.1; // ±10% utilization variation
    let stake_variation = (step as f32 * 0.005).sin() * 0.05; // ±5% staking variation
    
    QuaiData {
        gas_price: (base_gas + gas_variation).max(1.0),
        tx_count: (base_tx as i32 + tx_variation).max(10) as u32,
        block_utilization: (base_util + util_variation).clamp(0.1, 0.95),
        staking_ratio: (base_stake + stake_variation).clamp(0.1, 0.8),
    }
}

/// Check if Quai integration should be active
pub fn is_quai_integration_enabled() -> bool {
    // Multiple safety checks
    let env_enabled = std::env::var("QUAI_ENABLED")
        .unwrap_or("false".to_string()) == "true";
    
    let not_production = std::env::var("TRADING_MODE")
        .unwrap_or_default() != "production";
    
    let testing_mode = std::env::var("QUAI_TESTING")
        .unwrap_or("false".to_string()) == "true";
    
    // Enable if explicitly enabled, or if in testing mode
    env_enabled || (testing_mode && not_production)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_hex_gwei() {
        assert_eq!(parse_hex_gwei("0x1e8480"), Some(2.0)); // 2000000000 wei = 2 gwei
        assert_eq!(parse_hex_gwei("0x5F5E100"), Some(159.254876)); // 159254876000 wei
        assert_eq!(parse_hex_gwei("invalid"), None);
        assert_eq!(parse_hex_gwei("0xinvalid"), None);
    }
    
    #[test]
    fn test_parse_hex_u32() {
        assert_eq!(parse_hex_u32("0x64"), Some(100));
        assert_eq!(parse_hex_u32("0xFF"), Some(255));
        assert_eq!(parse_hex_u32("invalid"), None);
        assert_eq!(parse_hex_u32("0xinvalid"), None);
    }
    
    #[test]
    fn test_mock_data_generation() {
        let data1 = generate_mock_quai_data(0);
        let data2 = generate_mock_quai_data(100);
        
        // Should generate different data
        assert_ne!(data1.gas_price, data2.gas_price);
        assert_ne!(data1.tx_count, data2.tx_count);
        
        // Should stay within reasonable bounds
        assert!(data1.gas_price > 0.0);
        assert!(data1.block_utilization >= 0.1 && data1.block_utilization <= 0.95);
        assert!(data1.staking_ratio >= 0.1 && data1.staking_ratio <= 0.8);
    }
    
    #[test]
    fn test_quai_data_default() {
        let default = QuaiData::default();
        assert_eq!(default.gas_price, 10.0);
        assert_eq!(default.tx_count, 100);
        assert_eq!(default.block_utilization, 0.65);
        assert_eq!(default.staking_ratio, 0.40);
    }
}
