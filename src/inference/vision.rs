use anyhow::{Result, Context};
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, Module};
use candle_transformers::models::siglip::{VisionConfig, VisionModel};
use std::path::Path;
use image::DynamicImage;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct SiglipConfig {
    vision_config: VisionConfig,
}

/// Native Image Embedding Engine using Candle and SigLIP 2.
/// 
/// This module provides the "eyes" for the Eagle-Lander local-first RAG.
/// It takes raw image data and maps it into the same high-dimensional
/// vector space used by the text models, enabling cross-modal retrieval.
pub struct VisionEmbedder {
    model: VisionModel,
    config: VisionConfig,
    device: Device,
}

impl VisionEmbedder {
    /// Load a SigLIP 2 vision model onto the specified device.
    /// 
    /// Expects model weights in SafeTensors format and a HuggingFace-style config.json.
    pub fn new(model_path: &Path, config_path: &Path, device: &Device) -> Result<Self> {
        let config_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read SigLIP config at {}", config_path.display()))?;
        
        let config_full: SiglipConfig = serde_json::from_str(&config_str)
            .with_context(|| "Failed to parse SigLIP config JSON")?;
            
        let config = config_full.vision_config;
        
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)
                .with_context(|| "Failed to mmap SigLIP weights")?
        };
        
        // SigLIP 2 weights in HF usually have "vision_model" prefix
        let model = VisionModel::new(&config, true, vb.pp("vision_model"))
            .with_context(|| "Failed to initialize SigLIP vision model")?;
        
        Ok(Self {
            model,
            config,
            device: device.clone(),
        })
    }

    /// Generate an embedding vector for a given image.
    /// 
    /// Resizes, normalizes, and runs the forward pass on the Blackwell GPU.
    pub fn embed_image(&self, img: &DynamicImage) -> Result<Vec<f32>> {
        let img = img.resize_exact(
            self.config.image_size as u32,
            self.config.image_size as u32,
            image::imageops::FilterType::Triangle,
        );
        let rgb = img.to_rgb8();
        let pixels = rgb.as_raw();
        
        // Load into GPU tensor [3, H, W]
        let data = Tensor::from_vec(pixels.to_vec(), (self.config.image_size, self.config.image_size, 3), &self.device)?
            .permute((2, 0, 1))? // HWC -> CHW
            .to_dtype(DType::F32)?
            .unsqueeze(0)?;      // B=1, C=3, H, W
        
        // Normalization: (x/255.0 - 0.5) / 0.5
        // This maps the 0..255 range to -1.0..1.0 centered at 0.
        let data = (((data / 255.0)? - 0.5)? / 0.5)?;
        
        // Forward pass
        let features = self.model.forward(&data)?;
        
        // Squeeze batch and return as Vec<f32>
        let vector = features.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vector)
    }
}
