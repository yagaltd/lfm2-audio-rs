use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use moshi::{candle, mimi, StreamTensor};

use crate::error::{LFM2Error, Result};

const MIMI_CHECKPOINT_FILENAME: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
const LIQUID_MIMI_REPO_CACHE_DIR: &str = "models--LiquidAI--LFM2.5-Audio-1.5B";

fn find_snapshot_checkpoint(snapshot_root: &Path) -> Option<PathBuf> {
    let mut snapshot_dirs: Vec<PathBuf> = fs::read_dir(snapshot_root)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect();
    snapshot_dirs.sort();
    snapshot_dirs.reverse();
    snapshot_dirs
        .into_iter()
        .map(|dir| dir.join(MIMI_CHECKPOINT_FILENAME))
        .find(|path| path.is_file())
}

pub fn discover_mimi_checkpoint_in_hf_root(hf_root: &Path) -> Result<PathBuf> {
    let snapshot_root = hf_root
        .join(LIQUID_MIMI_REPO_CACHE_DIR)
        .join("snapshots");
    find_snapshot_checkpoint(&snapshot_root).ok_or_else(|| {
        LFM2Error::ModelNotFound(format!(
            "Mimi checkpoint '{}' not found under {}",
            MIMI_CHECKPOINT_FILENAME,
            snapshot_root.display()
        ))
    })
}

pub fn discover_default_mimi_checkpoint() -> Result<PathBuf> {
    if let Ok(explicit) = env::var("LFM2_MIMI_PATH") {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Ok(path);
        }
        return Err(LFM2Error::ModelNotFound(format!(
            "Configured Mimi checkpoint not found: {}",
            path.display()
        )));
    }

    if let Ok(hf_home) = env::var("HF_HOME") {
        let path = PathBuf::from(hf_home).join("hub");
        if path.is_dir() {
            if let Ok(found) = discover_mimi_checkpoint_in_hf_root(&path) {
                return Ok(found);
            }
        }
    }

    let home = env::var("HOME")
        .map(PathBuf::from)
        .map_err(|_| LFM2Error::ModelNotFound("HOME is not set; cannot discover Mimi checkpoint".to_string()))?;
    discover_mimi_checkpoint_in_hf_root(&home.join(".cache").join("huggingface").join("hub"))
}

fn flatten_pcm_tensor(pcm: &candle::Tensor) -> Result<Vec<f32>> {
    let pcm = pcm
        .to_dtype(candle::DType::F32)
        .map_err(|err| LFM2Error::Generation(format!("Mimi PCM dtype conversion failed: {err}")))?;
    let pcm = pcm
        .flatten_all()
        .map_err(|err| LFM2Error::Generation(format!("Mimi PCM flatten failed: {err}")))?;
    pcm.to_vec1::<f32>()
        .map_err(|err| LFM2Error::Generation(format!("Mimi PCM extraction failed: {err}")))
}

fn frame_tensor(device: &candle::Device, frame: [u16; 8]) -> Result<candle::Tensor> {
    let values: Vec<u32> = frame.into_iter().map(u32::from).collect();
    candle::Tensor::from_vec(values, (1, 8, 1), device)
        .map_err(|err| LFM2Error::Generation(format!("Failed to build Mimi frame tensor: {err}")))
}

fn frames_tensor(device: &candle::Device, codes: &[[u16; 8]]) -> Result<candle::Tensor> {
    let mut values = Vec::with_capacity(codes.len() * 8);
    for codebook in 0..8 {
        for frame in codes {
            values.push(u32::from(frame[codebook]));
        }
    }
    candle::Tensor::from_vec(values, (1, 8, codes.len()), device)
        .map_err(|err| LFM2Error::Generation(format!("Failed to build Mimi codes tensor: {err}")))
}

/// Template for creating Mimi streaming decoders.
/// 
/// Uses a Mutex to ensure thread-safe access to the shared Mimi instance.
/// The key insight is that `moshi::Mimi::clone()` performs shallow tensor cloning
/// via Arc, which means KV cache state is shared. We must reset the template's
/// state BEFORE cloning to ensure fresh state for each new decoder.
pub struct MimiDecoderTemplate {
    /// The shared Mimi instance, protected by a Mutex for thread safety.
    /// We reset its state before cloning to ensure fresh KV cache for each decoder.
    mimi: Mutex<mimi::Mimi>,
    device: candle::Device,
    checkpoint_path: PathBuf,
}

impl MimiDecoderTemplate {
    pub fn from_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let checkpoint_path = path.as_ref().to_path_buf();
        let device = candle::Device::Cpu;
        let mimi = mimi::load(
            checkpoint_path.to_str().ok_or_else(|| {
                LFM2Error::ModelNotFound(format!(
                    "Invalid Mimi checkpoint path: {}",
                    checkpoint_path.display()
                ))
            })?,
            Some(8),
            &device,
        )
        .map_err(|err| {
            LFM2Error::ModelNotFound(format!(
                "Failed to load Mimi checkpoint {}: {err}",
                checkpoint_path.display()
            ))
        })?;
        Ok(Self {
            mimi: Mutex::new(mimi),
            device,
            checkpoint_path,
        })
    }

    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }

    /// Create a new streaming decoder with fresh state.
    /// 
    /// This method locks the template, resets its state (to clear any accumulated
    /// KV cache from previous uses), clones it (now with fresh state), and returns
    /// the clone. This ensures each decoder starts with a clean slate.
    pub fn new_streaming_decoder(&self) -> MimiStreamingDecoder {
        let mut mimi = self.mimi.lock().expect("Mimi lock poisoned");
        
        // CRITICAL: Reset the template's state before cloning!
        // Without this, the clone shares the KV cache tensors via Arc,
        // and any accumulated state from previous sessions leaks through.
        mimi.reset_state();
        
        // Now clone - the clone will have fresh (None) KV cache
        let mut cloned = mimi.clone();
        
        // Reset again on the clone for good measure
        cloned.reset_state();
        
        MimiStreamingDecoder {
            mimi: cloned,
            device: self.device.clone(),
        }
    }

    pub fn decode_all(&self, codes: &[[u16; 8]]) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Ok(Vec::new());
        }
        let mut mimi = self.mimi.lock().expect("Mimi lock poisoned");
        mimi.reset_state();
        let codes = frames_tensor(&self.device, codes)?;
        let pcm = mimi
            .decode(&codes)
            .map_err(|err| LFM2Error::Generation(format!("Mimi batch decode failed: {err}")))?;
        flatten_pcm_tensor(&pcm)
    }
}

pub struct MimiStreamingDecoder {
    mimi: mimi::Mimi,
    device: candle::Device,
}

impl MimiStreamingDecoder {
    pub fn from_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        Ok(MimiDecoderTemplate::from_checkpoint(path)?.new_streaming_decoder())
    }

    pub fn decode_all(path: impl AsRef<Path>, codes: &[[u16; 8]]) -> Result<Vec<f32>> {
        MimiDecoderTemplate::from_checkpoint(path)?.decode_all(codes)
    }

    pub fn push_frame(&mut self, frame: [u16; 8]) -> Result<Vec<f32>> {
        let codes = frame_tensor(&self.device, frame)?;
        let pcm = self
            .mimi
            .decode_step(&StreamTensor::from_tensor(codes), &().into())
            .map_err(|err| LFM2Error::Generation(format!("Mimi step decode failed: {err}")))?;
        match pcm.as_option() {
            Some(pcm) => flatten_pcm_tensor(pcm),
            None => Ok(Vec::new()),
        }
    }

    pub fn reset(&mut self) {
        self.mimi.reset_state();
    }
}
