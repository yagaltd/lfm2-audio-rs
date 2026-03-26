//! KV-cache management for autoregressive generation
//! Reference: hand-voice-racer/audio-model.js:500-600 (initializeCache, updateCache)

use ndarray::{Array3, Array4};
use ort::session::SessionOutputs;
use std::collections::HashMap;

use crate::config::{LFM2Config, LayerType};
use crate::error::{LFM2Error, Result};

/// Cache for attention layers (keys and values)
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Key cache: [batch=1, num_kv_heads, seq_len, head_dim]
    pub key: Array4<f32>,
    /// Value cache: [batch=1, num_kv_heads, seq_len, head_dim]
    pub value: Array4<f32>,
}

/// Cache for convolutional layers
#[derive(Debug, Clone)]
pub struct ConvCache {
    /// State: [batch=1, hidden_size, cache_len=3]
    pub state: Array3<f32>,
}

/// Complete generation cache for LFM2 model
#[derive(Debug)]
pub struct GenerationCache {
    /// KV caches for attention layers, keyed by layer index
    pub kv_caches: HashMap<usize, KVCache>,
    /// Conv caches for convolutional layers, keyed by layer index
    pub conv_caches: HashMap<usize, ConvCache>,
    /// Current sequence length
    pub seq_len: usize,
    /// Layer types to know which cache type per layer
    pub layer_types: Vec<LayerType>,
    /// Model configuration
    pub config: LFM2CacheConfig,
}

/// Cache-related config extracted from LFM2Config
#[derive(Debug, Clone)]
pub struct LFM2CacheConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub conv_cache_len: usize,
}

impl From<&LFM2Config> for LFM2CacheConfig {
    fn from(config: &LFM2Config) -> Self {
        Self {
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            num_layers: config.num_hidden_layers,
            conv_cache_len: config.conv_l_cache,
        }
    }
}

impl GenerationCache {
    /// Initialize empty cache from model config
    pub fn new(config: &LFM2Config) -> Result<Self> {
        let cache_config = LFM2CacheConfig::from(config);
        let mut kv_caches = HashMap::new();
        let mut conv_caches = HashMap::new();

        // Parse layer types and initialize appropriate caches
        for (layer_idx, layer_type_str) in config.layer_types.iter().enumerate() {
            let layer_type = match layer_type_str.as_str() {
                "conv" => LayerType::Conv,
                "full_attention" => LayerType::FullAttention,
                _ => {
                    return Err(LFM2Error::Cache(format!(
                        "Unknown layer type: {}",
                        layer_type_str
                    )))
                }
            };

            match layer_type {
                LayerType::Conv => {
                    // Conv cache: [1, hidden_size, 3]
                    let state = Array3::<f32>::zeros((
                        1,
                        cache_config.hidden_size,
                        cache_config.conv_cache_len,
                    ));
                    conv_caches.insert(layer_idx, ConvCache { state });
                }
                LayerType::FullAttention => {
                    // Attention cache: start with seq_len=0 (empty) matching JS implementation
                    let key = Array4::<f32>::zeros((
                        1,
                        cache_config.num_key_value_heads,
                        0,
                        cache_config.head_dim,
                    ));
                    let value = Array4::<f32>::zeros((
                        1,
                        cache_config.num_key_value_heads,
                        0,
                        cache_config.head_dim,
                    ));
                    kv_caches.insert(layer_idx, KVCache { key, value });
                }
            }
        }

        Ok(Self {
            kv_caches,
            conv_caches,
            seq_len: 0,
            layer_types: config
                .layer_types
                .iter()
                .map(|s| match s.as_str() {
                    "conv" => LayerType::Conv,
                    _ => LayerType::FullAttention,
                })
                .collect(),
            config: cache_config,
        })
    }

    /// Prepare cache inputs for decoder
    /// Returns values in the exact order expected by the model
    pub fn prepare_cache_inputs(&self) -> Vec<(String, ort::value::DynValue)> {
        let mut feeds = Vec::new();

        // Iterate through ALL layers in order
        let num_layers = self.config.num_layers;

        for layer_idx in 0..num_layers {
            if let Some(cache) = self.conv_caches.get(&layer_idx) {
                // Conv layer
                let name = format!("past_conv.{}", layer_idx);
                let contiguous = cache.state.as_standard_layout().to_owned();
                if let Ok(tensor) = ort::value::Tensor::from_array(contiguous) {
                    feeds.push((name, tensor.into_dyn()));
                }
            } else if let Some(cache) = self.kv_caches.get(&layer_idx) {
                // Attention layer - add key then value
                let key_name = format!("past_key_values.{}.key", layer_idx);
                let value_name = format!("past_key_values.{}.value", layer_idx);

                let key_contig = cache.key.as_standard_layout().to_owned();
                let value_contig = cache.value.as_standard_layout().to_owned();

                if let Ok(key_tensor) = ort::value::Tensor::from_array(key_contig) {
                    feeds.push((key_name, key_tensor.into_dyn()));
                }
                if let Ok(value_tensor) = ort::value::Tensor::from_array(value_contig) {
                    feeds.push((value_name, value_tensor.into_dyn()));
                }
            }
        }

        feeds
    }

    /// Update cache from decoder outputs
    pub fn update(&mut self, outputs: &SessionOutputs) -> Result<()> {
        // Update conv caches
        for layer_idx in self.conv_caches.keys().copied().collect::<Vec<_>>() {
            let present_name = format!("present_conv.{}", layer_idx);
            if let Some(output) = outputs.get(&present_name) {
                if let Ok(view) = output.try_extract_array::<f32>() {
                    let array = view.to_owned();
                    if array.ndim() == 3 {
                        if let Some(cache) = self.conv_caches.get_mut(&layer_idx) {
                            cache.state = array
                                .into_dimensionality()
                                .map_err(|e| LFM2Error::Cache(format!("Shape error: {:?}", e)))?;
                        }
                    }
                }
            }
        }

        // Update KV caches
        for layer_idx in self.kv_caches.keys().copied().collect::<Vec<_>>() {
            let key_name = format!("present.{}.key", layer_idx);
            let value_name = format!("present.{}.value", layer_idx);

            if let Some(output) = outputs.get(&key_name) {
                if let Ok(view) = output.try_extract_array::<f32>() {
                    let array = view.to_owned();
                    if array.ndim() == 4 {
                        if let Some(cache) = self.kv_caches.get_mut(&layer_idx) {
                            cache.key = array
                                .into_dimensionality()
                                .map_err(|e| LFM2Error::Cache(format!("Shape error: {:?}", e)))?;
                        }
                    }
                }
            }

            if let Some(output) = outputs.get(&value_name) {
                if let Ok(view) = output.try_extract_array::<f32>() {
                    let array = view.to_owned();
                    if array.ndim() == 4 {
                        if let Some(cache) = self.kv_caches.get_mut(&layer_idx) {
                            cache.value = array
                                .into_dimensionality()
                                .map_err(|e| LFM2Error::Cache(format!("Shape error: {:?}", e)))?;
                        }
                    }
                }
            }
        }

        // Update sequence length
        if let Some((_, cache)) = self.kv_caches.iter().next() {
            self.seq_len = cache.key.shape()[2];
        }

        Ok(())
    }

    /// Get current sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Reset cache to empty state
    pub fn reset(&mut self) -> Result<()> {
        *self = Self::new(&self.to_lfm2_config()?)?;
        Ok(())
    }

    /// Helper to reconstruct LFM2Config for reset
    fn to_lfm2_config(&self) -> Result<LFM2Config> {
        Ok(LFM2Config {
            hidden_size: self.config.hidden_size,
            num_attention_heads: self.config.num_attention_heads,
            num_key_value_heads: self.config.num_key_value_heads,
            num_hidden_layers: self.config.num_layers,
            conv_l_cache: self.config.conv_cache_len,
            layer_types: self
                .layer_types
                .iter()
                .map(|lt| match lt {
                    LayerType::Conv => "conv".to_string(),
                    LayerType::FullAttention => "full_attention".to_string(),
                })
                .collect(),
            // Default values for fields we don't track
            name_or_path: "".to_string(),
            architectures: vec![],
            block_auto_adjust_ff_dim: false,
            block_dim: 0,
            block_ff_dim: 0,
            block_ffn_dim_multiplier: 0,
            block_mlp_init_scale: 0,
            block_multiple_of: 0,
            block_norm_eps: 1e-5,
            block_out_init_scale: 0,
            block_use_swiglu: false,
            block_use_xavier_init: false,
            conv_bias: false,
            conv_dim: 0,
            conv_dim_out: 0,
            conv_use_xavier_init: false,
            eos_token_id: 0,
            initializer_range: 0.0,
            intermediate_size: 0,
            max_position_embeddings: 0,
            model_type: "".to_string(),
            norm_eps: 1e-5,
            num_heads: 0,
            rope_theta: 0.0,
            torch_dtype: "".to_string(),
            use_cache: false,
            use_pos_enc: false,
            vocab_size: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> LFM2Config {
        LFM2Config {
            hidden_size: 2048,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_hidden_layers: 16,
            conv_l_cache: 3,
            layer_types: vec![
                "conv".to_string(),
                "conv".to_string(),
                "full_attention".to_string(),
                "conv".to_string(),
                "conv".to_string(),
                "full_attention".to_string(),
            ],
            name_or_path: "".to_string(),
            architectures: vec![],
            block_auto_adjust_ff_dim: false,
            block_dim: 0,
            block_ff_dim: 0,
            block_ffn_dim_multiplier: 0,
            block_mlp_init_scale: 0,
            block_multiple_of: 0,
            block_norm_eps: 1e-5,
            block_out_init_scale: 0,
            block_use_swiglu: false,
            block_use_xavier_init: false,
            conv_bias: false,
            conv_dim: 0,
            conv_dim_out: 0,
            conv_use_xavier_init: false,
            eos_token_id: 0,
            initializer_range: 0.0,
            intermediate_size: 0,
            max_position_embeddings: 0,
            model_type: "".to_string(),
            norm_eps: 1e-5,
            num_heads: 0,
            rope_theta: 0.0,
            torch_dtype: "".to_string(),
            use_cache: false,
            use_pos_enc: false,
            vocab_size: 0,
        }
    }

    #[test]
    fn test_cache_initialization() {
        let config = create_test_config();
        let cache = GenerationCache::new(&config).unwrap();

        assert_eq!(cache.conv_caches.len(), 4);
        assert_eq!(cache.kv_caches.len(), 2);
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn test_cache_input_order() {
        let config = create_test_config();
        let cache = GenerationCache::new(&config).unwrap();

        let inputs = cache.prepare_cache_inputs();
        let names: Vec<_> = inputs.iter().map(|(n, _)| n.clone()).collect();

        assert_eq!(names[0], "past_conv.0");
        assert_eq!(names[1], "past_conv.1");
        assert_eq!(names[2], "past_key_values.2.key");
        assert_eq!(names[3], "past_key_values.2.value");
        assert_eq!(names[4], "past_conv.3");
        assert_eq!(names[5], "past_conv.4");
        assert_eq!(names[6], "past_key_values.5.key");
        assert_eq!(names[7], "past_key_values.5.value");
    }
}
