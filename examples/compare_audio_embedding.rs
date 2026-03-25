use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Parser;
use ort::value::Value;
use serde::Deserialize;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX/onnx")]
    onnx_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct AudioEmbeddingMeta {
    num_codebooks: usize,
    codebook_vocab: usize,
    hidden_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let meta: AudioEmbeddingMeta =
        serde_json::from_slice(&std::fs::read(args.onnx_dir.join("audio_embedding.json"))?)?;
    let raw = std::fs::read(args.onnx_dir.join("audio_embedding.bin"))?;
    let weights: Vec<f32> = raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let test_frames = vec![
        [1049i64, 477, 1626, 142, 1335, 555, 976, 1648],
        [127, 1056, 1697, 290, 481, 1443, 825, 1744],
        [1880, 1056, 1178, 290, 1335, 1443, 976, 2008],
    ];

    let mut session = ort::session::Session::builder()?
        .commit_from_file(args.onnx_dir.join("audio_embedding.onnx"))?;

    for (frame_idx, frame) in test_frames.into_iter().enumerate() {
        let offset_tokens: Vec<i64> = frame
            .iter()
            .enumerate()
            .map(|(cb, code)| cb as i64 * meta.codebook_vocab as i64 + code)
            .collect();
        let input = ndarray::Array2::from_shape_vec((1, meta.num_codebooks), offset_tokens)?;
        let outputs = session.run(ort::inputs! {
            "audio_codes" => Value::from_array(input.as_standard_layout().to_owned())?,
        })?;
        let output = outputs
            .get("audio_embeds")
            .ok_or_else(|| anyhow::anyhow!("audio_embeds output missing"))?;
        let view = output.try_extract_array::<f32>()?;
        let shape = view.shape().to_vec();
        if shape.len() != 3 || shape[1] != meta.num_codebooks || shape[2] != meta.hidden_size {
            bail!("unexpected audio_embeds shape: {:?}", shape);
        }

        let mut onnx_sum = vec![0.0f32; meta.hidden_size];
        for cb in 0..meta.num_codebooks {
            for h in 0..meta.hidden_size {
                onnx_sum[h] += view[[0, cb, h]];
            }
        }

        let mut bin_sum = vec![0.0f32; meta.hidden_size];
        for (cb, &code) in frame.iter().enumerate() {
            let token_idx = cb * meta.codebook_vocab + code as usize;
            let offset = token_idx * meta.hidden_size;
            for h in 0..meta.hidden_size {
                bin_sum[h] += weights[offset + h];
            }
        }

        let mut max_abs_diff = 0.0f32;
        let mut mean_abs_diff = 0.0f32;
        for h in 0..meta.hidden_size {
            let diff = (bin_sum[h] - onnx_sum[h]).abs();
            max_abs_diff = max_abs_diff.max(diff);
            mean_abs_diff += diff;
        }
        mean_abs_diff /= meta.hidden_size as f32;

        println!(
            "frame[{frame_idx}] max_abs_diff={max_abs_diff:.8} mean_abs_diff={mean_abs_diff:.8}"
        );
    }

    Ok(())
}
