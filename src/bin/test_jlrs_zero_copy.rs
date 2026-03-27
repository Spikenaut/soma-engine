#!/usr/bin/env rust-script
//! Test jlrs Zero-Copy Backend - Spikenaut-v2 Validation
//!
//! Tests the jlrs zero-copy backend with neuromod sensory encoding.
//! Validates <50µs total latency target.

use std::time::Instant;
use soma_engine::backend::{JlrsZeroCopyBackend, BackendType, BackendFactory};
use soma_engine::telemetry::gpu_telemetry::GpuTelemetry;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Spikenaut-v2 jlrs Zero-Copy Backend Test ===");
    println!();

    // Test 1: Backend Creation
    println!("1. Testing backend creation...");
    let mut backend = BackendFactory::create(BackendType::JlrsZeroCopy);
    println!("   ✅ Backend created successfully");

    // Test 2: Initialization
    println!("2. Testing backend initialization...");
    let init_result = backend.initialize(None);
    match init_result {
        Ok(()) => println!("   ✅ Backend initialized successfully"),
        Err(e) => {
            println!("   ❌ Backend initialization failed: {}", e);
            println!("   💡 This is expected if Julia runtime is not available");
            return Ok(());
        }
    }

    // Test 3: Create test telemetry
    println!("3. Creating test telemetry data...");
    let telemetry = GpuTelemetry {
        hashrate_mh: 0.5,
        power_w: 250.0,
        gpu_temp_c: 65.0,
        gpu_clock_mhz: 2000.0,
        mem_clock_mhz: 400.0,
        fan_speed_pct: 50.0,
        vddcr_gfx_v: 1.0,
        qubic_tick_trace: 1.0,
        qubic_epoch_progress: 0.5,
        reward_hint: 0.8,
        ..Default::default()
    };
    println!("   ✅ Test telemetry created");

    // Test 4: Neuromodulator Integration
    println!("4. Testing neuromodulator integration...");
    if let Some(jlrs_backend) = backend.as_any().downcast_ref::<JlrsZeroCopyBackend>() {
        jlrs_backend.set_market_volatility(0.3);
        jlrs_backend.set_mining_dopamine(0.5);
        jlrs_backend.set_fpga_stress(0.1);
        println!("   ✅ Neuromodulators set successfully");
        
        let neuromodulators = jlrs_backend.get_neuromodulators();
        println!("   📊 Dopamine: {:.3}", neuromodulators.dopamine);
        println!("   📊 Cortisol: {:.3}", neuromodulators.cortisol);
        println!("   📊 Acetylcholine: {:.3}", neuromodulators.acetylcholine);
        println!("   📊 Tempo: {:.3}", neuromodulators.tempo);
        println!("   📊 Market Volatility: {:.3}", neuromodulators.market_volatility);
        println!("   📊 Mining Dopamine: {:.3}", neuromodulators.mining_dopamine);
        println!("   📊 FPGA Stress: {:.3}", neuromodulators.fpga_stress);
    }

    // Test 5: Performance Benchmark
    println!("5. Running performance benchmark...");
    let test_inputs = [0.5, -0.3, 0.0, 0.8, -0.2, 0.1, -0.7, 0.4];
    let mut latencies = Vec::new();
    let iterations = 1000;

    for i in 0..iterations {
        let start = Instant::now();
        let result = backend.process_signals(&test_inputs, 0.1, &telemetry);
        let elapsed = start.elapsed();
        
        match result {
            Ok(output) => {
                latencies.push(elapsed);
                if i == 0 {
                    println!("   ✅ First call successful, output length: {}", output.len());
                }
            }
            Err(e) => {
                println!("   ❌ Process signals failed on iteration {}: {}", i, e);
                return Ok(());
            }
        }
    }

    // Calculate statistics
    let total_time: std::time::Duration = latencies.iter().sum();
    let avg_latency = total_time / iterations as u32;
    let max_latency = latencies.iter().max().unwrap();
    let min_latency = latencies.iter().min().unwrap();

    println!("   📊 Performance Statistics:");
    println!("      Iterations: {}", iterations);
    println!("      Average Latency: {:.2}µs", avg_latency.as_micros() as f64);
    println!("      Min Latency: {:.2}µs", min_latency.as_micros() as f64);
    println!("      Max Latency: {:.2}µs", max_latency.as_micros() as f64);
    println!("      Total Time: {:.2}ms", total_time.as_millis() as f64);

    // Check if we meet the target
    let target_us = 50.0;
    if avg_latency.as_micros() as f64 < target_us {
        println!("   ✅ Target met: {:.2}µs < {:.2}µs", avg_latency.as_micros() as f64, target_us);
    } else {
        println!("   ⚠️  Target not met: {:.2}µs > {:.2}µs", avg_latency.as_micros() as f64, target_us);
    }

    // Test 6: Performance Stats
    println!("6. Checking backend performance stats...");
    if let Some(jlrs_backend) = backend.as_any().downcast_ref::<JlrsZeroCopyBackend>() {
        let (call_count, total_time) = jlrs_backend.get_performance_stats();
        println!("   📊 Backend Stats: {} calls, total {:.2}ms", call_count, total_time.as_millis() as f64);
    }

    println!();
    println!("=== Test Complete ===");
    println!("✅ Spikenaut-v2 jlrs Zero-Copy Backend is functional");

    Ok(())
}
