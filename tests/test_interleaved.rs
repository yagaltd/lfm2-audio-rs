use std::path::PathBuf;
use std::process::Command;

use lfm2_audio::{Device, LFM2Audio, Precision, TTSOptions};
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
fn test_interleaved_text_response_generates_real_output() {
    let model = load_model();

    let response = model
        .interleaved()
        .respond_to_text("Say hello in a friendly way")
        .expect("interleaved text generation should succeed");

    assert_ne!(response.text, "Not yet implemented");
    assert!(
        !response.text.trim().is_empty() || !response.audio_codes.is_empty(),
        "interleaved should produce text and/or audio"
    );
    if !response.audio_codes.is_empty() {
        assert!(!response.audio.is_empty(), "audio waveform should be decoded");
        assert!(response.audio.iter().all(|sample| sample.is_finite()));
        assert!(
            response
                .audio_codes
                .iter()
                .all(|frame| frame.iter().all(|&code| code < 2048)),
            "audio codes must be valid detokenizer inputs"
        );
    }
}

#[test]
#[ignore = "requires model files and is slow"]
fn test_interleaved_audio_response_generates_real_output() {
    let model = load_model();

    let prompt_audio = model
        .tts()
        .synthesize(
            "Hello there.",
            &TTSOptions {
                max_new_tokens: 48,
                text_temperature: 0.0,
                audio_temperature: 0.0,
                audio_top_k: 1,
                ..Default::default()
            },
        )
        .expect("failed to synthesize prompt audio");

    let response = model
        .interleaved()
        .respond_to_audio(&prompt_audio, 24_000)
        .expect("interleaved audio generation should succeed");

    assert_ne!(response.text, "Not yet implemented");
    assert!(
        !response.text.trim().is_empty() || !response.audio_codes.is_empty(),
        "interleaved should produce text and/or audio"
    );
    if !response.audio_codes.is_empty() {
        assert!(!response.audio.is_empty(), "audio waveform should be decoded");
        assert!(response.audio.iter().all(|sample| sample.is_finite()));
    }
}

#[test]
#[ignore = "requires model files and is slow"]
fn test_chat_multi_turn_generates_stateful_responses() {
    let model = load_model();
    let mut session = model.chat();

    session.add_user_text("Say hello in a friendly way");
    let first = session.generate().expect("first chat response should succeed");
    assert_ne!(first.text, "Not yet implemented");
    assert!(
        !first.text.trim().is_empty() || first.audio.as_ref().is_some_and(|audio| !audio.is_empty()),
        "first chat response should contain text and/or audio"
    );

    session.add_user_text("Now answer with a different short follow up");
    let second = session.generate().expect("second chat response should succeed");
    assert_ne!(second.text, "Not yet implemented");
    assert!(
        !second.text.trim().is_empty() || second.audio.as_ref().is_some_and(|audio| !audio.is_empty()),
        "second chat response should contain text and/or audio"
    );

    assert_eq!(session.history().len(), 4, "chat history should contain two user turns and two assistant turns");
}

#[test]
#[ignore = "requires model files and is slow"]
fn test_chat_with_custom_system_prompt_generates_real_output() {
    let model = load_model();
    let mut session = model.chat_with_options(lfm2_audio::InterleavedOptions {
        system_prompt: "Respond as a concise pirate captain with interleaved text and audio.".to_string(),
        max_new_tokens: 80,
        text_temperature: 0.0,
        audio_temperature: 0.0,
        audio_top_k: 1,
        interleaved_n_text: None,
        interleaved_n_audio: None,
    });

    session.add_user_text("Greet the crew briefly.");
    let response = session
        .generate()
        .expect("chat with custom system prompt should succeed");

    assert!(
        !response.text.trim().is_empty() || response.audio.as_ref().is_some_and(|audio| !audio.is_empty()),
        "custom chat response should contain text and/or audio"
    );
}

#[test]
#[ignore = "requires model files and is slow"]
fn test_interleaved_cli_text_prompt_generates_output_file() {
    let model_path = get_model_path().expect("Model not found. Skipping test.");
    let temp_dir = tempdir().expect("failed to create temp dir");
    let output_path = temp_dir.path().join("interleaved.wav");

    let status = Command::new(env!("CARGO_BIN_EXE_lfm2-audio"))
        .arg("--model")
        .arg(&model_path)
        .arg("--precision")
        .arg("q4")
        .arg("--device")
        .arg("cpu")
        .arg("interleaved")
        .arg("--prompt")
        .arg("Say hello in a friendly way")
        .arg("--output")
        .arg(&output_path)
        .status()
        .expect("failed to launch interleaved CLI");

    assert!(status.success(), "CLI should exit successfully");
    assert!(output_path.exists(), "CLI should write audio output when audio is generated");
}
