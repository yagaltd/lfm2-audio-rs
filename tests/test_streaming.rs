use std::path::PathBuf;

use lfm2_audio::{Device, InterleavedEvent, InterleavedOptions, LFM2Audio, Precision};

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