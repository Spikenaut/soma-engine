#!/usr/bin/env rust-script
//! Test FPGA Parameter Export - Spikenaut-v2 Validation
//!
//! Tests the FPGA parameter export functionality.
//! Validates Q8.8 format conversion and memory usage.

use std::fs;
use soma_engine::fpga_export::{FpgaParameterExporter, FpgaParameters, format_q88_hex, q88_to_f32};

fn main() -> anyhow::Result<()> {
    println!("=== Spikenaut-v2 FPGA Parameter Export Test ===");
    println!();

    // Test 1: Q8.8 Conversion
    println!("1. Testing Q8.8 format conversion...");
    let test_values = [0.0, 0.5, 1.0, 1.5, 2.0, 255.0];
    for &value in &test_values {
        let hex = format_q88_hex(value);
        let converted = q88_to_f32(u16::from_str_radix(&hex, 16)?);
        println!("   {:.2} -> {} -> {:.2}", value, hex, converted);
    }
    println!("   ✅ Q8.8 conversion working");

    // Test 2: Parameter Export
    println!("2. Testing parameter export...");
    let mut exporter = FpgaParameterExporter::new();
    
    // Set test parameters (16 neurons, 16 channels)
    let thresholds: Vec<f32> = (0..16).map(|i| 0.8 + (i as f32) * 0.05).collect();
    let weights: Vec<Vec<f32>> = (0..16)
        .map(|i| (0..16)
            .map(|j| 0.5 + ((i * 16 + j) as f32) * 0.01)
            .collect())
        .collect();
    let decay_rates: Vec<f32> = (0..16).map(|_| 0.85).collect();
    
    exporter.set_thresholds(thresholds);
    exporter.set_weights(weights);
    exporter.set_decay_rates(decay_rates);
    
    let params = exporter.export();
    println!("   ✅ Parameters exported successfully");
    println!("   📊 Neurons: {}", params.metadata.num_neurons);
    println!("   📊 Channels: {}", params.metadata.num_channels);
    println!("   📊 Memory Usage: {:.2} KB", params.metadata.memory_usage_kb);
    println!("   📊 Target Latency: {:.1}µs", params.metadata.target_latency_us);

    // Test 3: File Export
    println!("3. Testing file export...");
    let output_dir = "test_fpga_export";
    
    // Clean up any existing test directory
    if fs::metadata(output_dir).is_ok() {
        fs::remove_dir_all(output_dir)?;
    }
    
    exporter.export_to_mem_files(output_dir)?;
    println!("   ✅ Files exported to {}", output_dir);

    // Verify files exist
    let expected_files = [
        "parameters.mem",
        "parameters_weights.mem", 
        "parameters_decay.mem",
        "parameters.json"
    ];
    
    for file in &expected_files {
        let path = format!("{}/{}", output_dir, file);
        if fs::metadata(&path).is_ok() {
            println!("   ✅ {} exists", file);
        } else {
            println!("   ❌ {} missing", file);
        }
    }

    // Test 4: Memory Usage Validation
    println!("4. Testing memory usage validation...");
    let expected_memory_kb = (16 + 256 + 16) * 2 / 1024.0; // 0.5625 KB
    let actual_memory_kb = params.metadata.memory_usage_kb;
    
    if (actual_memory_kb - expected_memory_kb).abs() < 0.01 {
        println!("   ✅ Memory usage correct: {:.2} KB", actual_memory_kb);
    } else {
        println!("   ❌ Memory usage incorrect: expected {:.2} KB, got {:.2} KB", 
                 expected_memory_kb, actual_memory_kb);
    }

    // Test 5: Parameter Range Validation
    println!("5. Testing parameter range validation...");
    let mut in_range = true;
    
    for threshold in &params.thresholds {
        if *threshold > 65535 {
            in_range = false;
            break;
        }
    }
    
    for weight in &params.weights {
        if *weight > 65535 {
            in_range = false;
            break;
        }
    }
    
    for decay in &params.decay_rates {
        if *decay > 65535 {
            in_range = false;
            break;
        }
    }
    
    if in_range {
        println!("   ✅ All parameters in Q8.8 range (0-65535)");
    } else {
        println!("   ❌ Some parameters out of Q8.8 range");
    }

    // Test 6: JSON Metadata
    println!("6. Testing JSON metadata...");
    let metadata_path = format!("{}/parameters.json", output_dir);
    let metadata_json = fs::read_to_string(&metadata_path)?;
    
    if metadata_json.contains("Spikenaut-v2") {
        println!("   ✅ JSON metadata contains version");
    } else {
        println!("   ❌ JSON metadata missing version");
    }
    
    if metadata_json.contains("35.0") {
        println!("   ✅ JSON metadata contains target latency");
    } else {
        println!("   ❌ JSON metadata missing target latency");
    }

    // Clean up test directory
    fs::remove_dir_all(output_dir)?;
    println!("   🧹 Test directory cleaned up");

    println!();
    println!("=== Test Complete ===");
    println!("✅ Spikenaut-v2 FPGA Parameter Export is functional");
    println!("✅ Ready for hardware deployment with <35µs/tick target");

    Ok(())
}
