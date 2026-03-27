//! Native Embedding Engine using Candle
//! 
//! Implements a BERT-based embedding pipeline for RAG.
//! Defaults to 'sentence-transformers/all-MiniLM-L6-v2' for efficiency.

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct ShipEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl ShipEmbedder {
    pub fn new(use_gpu: bool) -> Result<Self> {
        let device = if use_gpu && candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };

        let api = Api::new()?;
        let repo = api.repo(Repo::new("sentence-transformers/all-MiniLM-L6-v2".to_string(), RepoType::Model));

        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)?
        };
        let model = BertModel::load(vb, &config)?;

        Ok(Self { model, tokenizer, device })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?;
        let token_ids = tokens.get_ids();
        let token_type_ids = tokens.get_type_ids();
        
        let token_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(token_type_ids, &self.device)?.unsqueeze(0)?;

        // Forward pass
        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;
        
        // Mean pooling: [1, seq_len, 384] -> [384]
        // (For simplicity, we just mean over the sequence dimension)
        let (_batch_size, seq_len, _dim) = embeddings.dims3()?;
        let mean_embedding = (embeddings.sum(1)? / (seq_len as f64))?;
        let vector = mean_embedding.squeeze(0)?.to_vec1::<f32>()?;

        Ok(vector)
    }
}
