use std::path::PathBuf;

use lfm2_audio::{
    discover_mimi_checkpoint_in_hf_root, Device, InterleavedEvent, InterleavedOptions, LFM2Audio,
    MimiStreamingDecoder, Precision,
};
use tempfile::tempdir;

fn get_model_path() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("tests/models/LFM2.5-Audio-1.5B-ONNX"),
        PathBuf::from("/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX"),
    ];

    candidates.into_iter().find(|path| path.exists())
}

fn load_model() -> LFM2Audio {
    let model_path = get_model_path().expect("Model not found. Skipping test.");
    LFM2Audio::from_pretrained(&model_path, Precision::Q4, Device::CPU)
        .expect("Failed to load model")
}

#[test]
#[ignore = "requires model files and is slow"]
fn test_streaming_chat_emits_text_or_audio_events() {
    let model = load_model();
    let mut session = model.chat();
    session.add_user_text("Say hello briefly and respond with audio.");

    let mut saw_text = false;
    let mut saw_audio = false;

    let response = session
        .generate_streaming(|event| {
            match event {
                InterleavedEvent::TextUpdated(text) => {
                    if !text.trim().is_empty() {
                        saw_text = true;
                    }
                }
                InterleavedEvent::AudioFrame(frame) => {
                    if frame.iter().any(|&code| code < 2048) {
                        saw_audio = true;
                    }
                }
            }
            Ok(())
        })
        .expect("streaming generation should succeed");

    assert!(
        saw_text || saw_audio,
        "streaming generation should emit text and/or audio events"
    );
    assert!(
        !response.text.trim().is_empty()
            || response
                .audio
                .as_ref()
                .is_some_and(|audio| !audio.is_empty()),
        "final response should contain text and/or audio"
    );
}

#[test]
#[ignore = "requires model files and is slow"]
fn test_streaming_chat_skips_final_audio_decode() {
    let model = load_model();
    let mut session = model.chat_with_options(InterleavedOptions {
        max_new_tokens: 160,
        text_temperature: 0.0,
        audio_temperature: 0.0,
        audio_top_k: 1,
        ..Default::default()
    });
    session.add_user_text("Say hello briefly and respond with audio.");

    let mut saw_audio = false;
    let response = session
        .generate_streaming(|event| {
            if let InterleavedEvent::AudioFrame(frame) = event {
                if frame.iter().any(|&code| code < 2048) {
                    saw_audio = true;
                }
            }
            Ok(())
        })
        .expect("streaming generation should succeed");

    assert!(saw_audio, "streaming generation should emit audio frames");
    assert!(
        response.audio.is_none(),
        "streaming chat should not perform a final full audio decode"
    );
}

#[test]
fn test_discover_mimi_checkpoint_in_hf_root_finds_expected_snapshot_file() {
    let temp = tempdir().expect("tempdir should be created");
    let checkpoint = temp
        .path()
        .join("models--LiquidAI--LFM2.5-Audio-1.5B")
        .join("snapshots")
        .join("snapshot-a")
        .join("tokenizer-e351c8d8-checkpoint125.safetensors");
    std::fs::create_dir_all(
        checkpoint
            .parent()
            .expect("checkpoint path should have a parent directory"),
    )
    .expect("snapshot directory should be created");
    std::fs::write(&checkpoint, b"stub").expect("checkpoint file should be created");

    let discovered = discover_mimi_checkpoint_in_hf_root(temp.path())
        .expect("checkpoint should be discovered");

    assert_eq!(discovered, checkpoint);
}

#[test]
#[ignore = "requires model files and Mimi checkpoint and is slow"]
fn test_mimi_streaming_decoder_matches_batch_decode_on_lfm2_frames() {
    let model = load_model();
    let options = lfm2_audio::TTSOptions {
        max_new_tokens: 48,
        text_temperature: 0.0,
        audio_temperature: 0.0,
        audio_top_k: 1,
        ..Default::default()
    };
    let debug = model
        .tts()
        .synthesize_debug("Hello there.", &options)
        .expect("failed to synthesize reference audio");
    let audio_codes: Vec<[u16; 8]> = debug.audio_codes.iter().take(8).copied().collect();
    assert!(
        !audio_codes.is_empty(),
        "need generated audio codes to exercise Mimi streaming decode"
    );

    let mimi_path = discover_mimi_checkpoint_in_hf_root(PathBuf::from("/home/aurel/.cache/huggingface/hub").as_path())
        .expect("Mimi checkpoint should be discoverable in the local HF cache");

    let batch_audio = MimiStreamingDecoder::decode_all(&mimi_path, &audio_codes)
        .expect("native Mimi batch decode should succeed");

    let mut decoder =
        MimiStreamingDecoder::from_checkpoint(&mimi_path).expect("Mimi decoder should load");
    let mut step_audio = Vec::new();
    for frame in &audio_codes {
        let chunk = decoder
            .push_frame(*frame)
            .expect("step decode should succeed");
        assert_eq!(
            chunk.len(),
            1920,
            "one LFM2 audio frame should decode to 1920 samples"
        );
        step_audio.extend_from_slice(&chunk);
    }

    assert_eq!(
        step_audio.len(),
        batch_audio.len(),
        "step decode should match batch decode sample count"
    );

    let correlation = {
        let mean = |values: &[f32]| values.iter().copied().sum::<f32>() / values.len() as f32;
        let ax = mean(&batch_audio);
        let ay = mean(&step_audio);
        let mut num = 0.0f64;
        let mut den_x = 0.0f64;
        let mut den_y = 0.0f64;
        for (&x, &y) in batch_audio.iter().zip(step_audio.iter()) {
            let dx = (x - ax) as f64;
            let dy = (y - ay) as f64;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }
        num / (den_x.sqrt() * den_y.sqrt())
    };
    assert!(
        correlation > 0.9999,
        "native Mimi step decode should closely match native Mimi batch decode"
    );
}
