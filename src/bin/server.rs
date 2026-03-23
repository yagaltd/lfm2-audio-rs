//! LFM2-Audio API Server
//! OpenAI-compatible API endpoints

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use lfm2_audio::{Device, LFM2Audio, Precision};

struct AppState {
    model: Arc<Mutex<LFM2Audio>>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/transcriptions", post(transcribe))
        .route("/v1/audio/speech", post(synthesize));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    tracing::info!("Server listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "model": "lfm2.5-audio"
    }))
}

async fn list_models() -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "lfm2.5-audio",
                "object": "model",
                "created": 1700000000,
                "owned_by": "liquid-ai"
            }
        ]
    }))
}

#[derive(Deserialize)]
struct TranscriptionRequest {
    file: String, // base64 audio
    model: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
}

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
}

async fn transcribe(Json(_req): Json<TranscriptionRequest>) -> impl IntoResponse {
    // TODO: Implement
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "Not yet implemented"
        }))
    )
}

#[derive(Deserialize)]
struct SpeechRequest {
    model: String,
    input: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
}

async fn synthesize(Json(_req): Json<SpeechRequest>) -> impl IntoResponse {
    // TODO: Implement
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "Not yet implemented"
        }))
    )
}