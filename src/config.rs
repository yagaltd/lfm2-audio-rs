//! Model configuration from config.json

use serde::Deserialize;
use std::path::Path;

/// Model configuration loaded from config.json
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub codebooks: usize,
    #[serde(rename = "tie_audio_embeddings")]
    pub tie_audio_embeddings: bool,
    #[serde(rename = "semantic_codebook_factor")]
    pub semantic_codebook_factor: f32,
    #[serde(rename = "codebook_weight")]
    pub codebook_weight: String,
    #[serde(rename = "interleaved_n_text")]
    pub interleaved_n_text: usize,
    #[serde(rename = "interleaved_n_audio")]
    pub interleaved_n_audio: usize,
    pub preprocessor: PreprocessorConfig,
    pub encoder: ConformerEncoderConfig,
    pub lfm: LFM2Config,
    pub depthformer: DepthformerConfig,
}

impl ModelConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    #[serde(rename = "sample_rate")]
    pub sample_rate: u32,
    pub normalize: String,
    #[serde(rename = "window_size")]
    pub window_size: f64,
    #[serde(rename = "window_stride")]
    pub window_stride: f64,
    pub window: String,
    pub features: usize,
    #[serde(rename = "n_fft")]
    pub n_fft: usize,
    pub log: bool,
    #[serde(rename = "frame_splicing")]
    pub frame_splicing: usize,
    pub dither: f64,
    #[serde(rename = "pad_to")]
    pub pad_to: usize,
    #[serde(rename = "pad_value")]
    pub pad_value: f64,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            normalize: "per_feature".to_string(),
            window_size: 0.025,
            window_stride: 0.01,
            window: "hann".to_string(),
            features: 128,
            n_fft: 512,
            log: true,
            frame_splicing: 1,
            dither: 1.0e-05,
            pad_to: 0,
            pad_value: 0.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConformerEncoderConfig {
    #[serde(rename = "feat_in")]
    pub feat_in: usize,
    #[serde(rename = "feat_out")]
    pub feat_out: i32,
    #[serde(rename = "n_layers")]
    pub n_layers: usize,
    #[serde(rename = "d_model")]
    pub d_model: usize,
    pub subsampling: String,
    #[serde(rename = "subsampling_factor")]
    pub subsampling_factor: usize,
    #[serde(rename = "subsampling_conv_channels")]
    pub subsampling_conv_channels: usize,
    #[serde(rename = "causal_downsampling")]
    pub causal_downsampling: bool,
    pub reduction: Option<serde_json::Value>,
    #[serde(rename = "reduction_position")]
    pub reduction_position: Option<serde_json::Value>,
    #[serde(rename = "reduction_factor")]
    pub reduction_factor: usize,
    #[serde(rename = "ff_expansion_factor")]
    pub ff_expansion_factor: usize,
    #[serde(rename = "self_attention_model")]
    pub self_attention_model: String,
    #[serde(rename = "n_heads")]
    pub n_heads: usize,
    #[serde(rename = "att_context_size")]
    pub att_context_size: Vec<i32>,
    pub xscaling: bool,
    #[serde(rename = "untie_biases")]
    pub untie_biases: bool,
    #[serde(rename = "pos_emb_max_len")]
    pub pos_emb_max_len: usize,
    #[serde(rename = "conv_kernel_size")]
    pub conv_kernel_size: usize,
    #[serde(rename = "conv_norm_type")]
    pub conv_norm_type: String,
    #[serde(rename = "conv_context_size")]
    pub conv_context_size: Option<serde_json::Value>,
    pub dropout: f64,
    #[serde(rename = "dropout_pre_encoder")]
    pub dropout_pre_encoder: f64,
    #[serde(rename = "dropout_emb")]
    pub dropout_emb: f64,
    #[serde(rename = "dropout_att")]
    pub dropout_att: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LFM2Config {
    #[serde(rename = "_name_or_path")]
    pub name_or_path: String,
    pub architectures: Vec<String>,
    #[serde(rename = "block_auto_adjust_ff_dim")]
    pub block_auto_adjust_ff_dim: bool,
    #[serde(rename = "block_dim")]
    pub block_dim: usize,
    #[serde(rename = "block_ff_dim")]
    pub block_ff_dim: usize,
    #[serde(rename = "block_ffn_dim_multiplier")]
    pub block_ffn_dim_multiplier: usize,
    #[serde(rename = "block_mlp_init_scale")]
    pub block_mlp_init_scale: usize,
    #[serde(rename = "block_multiple_of")]
    pub block_multiple_of: usize,
    #[serde(rename = "block_norm_eps")]
    pub block_norm_eps: f64,
    #[serde(rename = "block_out_init_scale")]
    pub block_out_init_scale: usize,
    #[serde(rename = "block_use_swiglu")]
    pub block_use_swiglu: bool,
    #[serde(rename = "block_use_xavier_init")]
    pub block_use_xavier_init: bool,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(rename = "conv_bias")]
    pub conv_bias: bool,
    #[serde(rename = "conv_dim")]
    pub conv_dim: usize,
    #[serde(rename = "conv_dim_out")]
    pub conv_dim_out: usize,
    #[serde(rename = "conv_use_xavier_init")]
    pub conv_use_xavier_init: bool,
    #[serde(rename = "eos_token_id")]
    pub eos_token_id: usize,
    #[serde(rename = "hidden_size")]
    pub hidden_size: usize,
    #[serde(rename = "initializer_range")]
    pub initializer_range: f64,
    #[serde(rename = "intermediate_size")]
    pub intermediate_size: usize,
    #[serde(rename = "layer_types")]
    pub layer_types: Vec<String>,
    #[serde(rename = "max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(rename = "model_type")]
    pub model_type: String,
    #[serde(rename = "norm_eps")]
    pub norm_eps: f64,
    #[serde(rename = "num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(rename = "num_heads")]
    pub num_heads: usize,
    #[serde(rename = "num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(rename = "num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(rename = "rope_theta")]
    pub rope_theta: f64,
    #[serde(rename = "torch_dtype")]
    pub torch_dtype: String,
    #[serde(rename = "use_cache")]
    pub use_cache: bool,
    #[serde(rename = "use_pos_enc")]
    pub use_pos_enc: bool,
    #[serde(rename = "vocab_size")]
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DepthformerConfig {
    pub layers: usize,
    pub dim: usize,
    #[serde(rename = "tie")]
    pub tie: bool,
}

/// Model precision variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP32,
    FP16,
    Q4,
    Q8,
}

impl Precision {
    pub fn suffix(&self) -> &'static str {
        match self {
            Precision::FP32 => "",
            Precision::FP16 => "_fp16",
            Precision::Q4 => "_q4",
            Precision::Q8 => "_q8",
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            Precision::FP32 => "fp32",
            Precision::FP16 => "fp16",
            Precision::Q4 => "q4",
            Precision::Q8 => "q8",
        }
    }
}

impl std::str::FromStr for Precision {
    type Err = String;
    
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "fp32" => Ok(Precision::FP32),
            "fp16" => Ok(Precision::FP16),
            "q4" => Ok(Precision::Q4),
            "q8" => Ok(Precision::Q8),
            _ => Err(format!("Unknown precision: {}", s)),
        }
    }
}

/// Device selection for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU,
    Cuda,
    CoreML,
    DirectML,
    TensorRT,
}

impl Device {
    #[allow(deprecated)]
    pub fn execution_providers(&self) -> Vec<ort::execution_providers::ExecutionProviderDispatch> {
        use ort::execution_providers::*;
        
        let mut eps = Vec::new();
        
        match self {
            Device::CPU => {
                // CPU is always available as fallback
            }
            Device::Cuda => {
                #[cfg(feature = "cuda")]
                eps.push(CUDAExecutionProvider::default().build());
            }
            Device::CoreML => {
                #[cfg(feature = "coreml")]
                eps.push(CoreMLExecutionProvider::default().build());
            }
            Device::DirectML => {
                #[cfg(feature = "directml")]
                eps.push(DirectMLExecutionProvider::default().build());
            }
            Device::TensorRT => {
                #[cfg(feature = "tensorrt")]
                eps.push(TensorRTExecutionProvider::default().build());
            }
        }
        
        // Always add CPU as fallback
        eps.push(CPUExecutionProvider::default().build());
        
        eps
    }
}
/// Layer types in LFM2 model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Conv,
    FullAttention,
}
