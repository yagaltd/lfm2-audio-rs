//! LFM2-Audio demo server with browser-friendly HTTP and WebSocket APIs.

use anyhow::{anyhow, Context, Result};
use axum::{
    body::Bytes,
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    env,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        mpsc as std_mpsc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};
use tower_http::{
    cors::CorsLayer,
    services::{ServeDir, ServeFile},
};
use tracing::{info, warn};

use lfm2_audio::{
    decode_wav_bytes, encode_wav_bytes,
    sessions::load_audio_detokenizer_session_with_threads, ASROptions, Device,
    InterleavedOptions, LFM2Audio, Precision, TTSOptions,
};

const DEFAULT_MODEL_DIR: &str = "/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX";
const DEFAULT_BIND_ADDR: &str = "127.0.0.1:8080";
const DEFAULT_SYSTEM_PROMPT_INTERLEAVED: &str = "Respond with interleaved text and audio.";
const MAX_BINARY_AUDIO_BYTES: usize = 8 * 1024 * 1024;
const STREAM_EVENT_CHANNEL_CAPACITY: usize = 64;
const STREAM_DECODE_BATCH_FRAMES: usize = 2;
const STREAM_DECODE_CONTEXT_FRAMES: usize = 0;
const STREAM_OUTPUT_QUEUE_CHUNKS: usize = 1;
const STREAM_SAMPLES_PER_FRAME: usize = 1_920;
const STREAM_DETOKENIZER_MAX_INFLIGHT_BATCHES: usize = 1;
const STREAM_DETOKENIZER_INTRA_THREADS: usize = 1;

#[derive(Clone)]
struct AppState {
    workers: Arc<Vec<mpsc::Sender<ModelCommand>>>,
    next_session_id: Arc<AtomicU64>,
    next_worker: Arc<AtomicUsize>,
    session_workers: Arc<AsyncMutex<HashMap<u64, usize>>>,
    interleaved_trace: Option<Arc<InterleavedTraceLogger>>,
}

#[derive(Debug, Clone)]
struct ServerConfig {
    model_path: PathBuf,
    static_dir: PathBuf,
    bind_addr: SocketAddr,
    precision: Precision,
    device_preference: DevicePreference,
    cpu_workers: usize,
    interleaved_n_text: Option<usize>,
    interleaved_n_audio: Option<usize>,
    stream_decode: StreamingDecodeConfig,
    interleaved_log_path: Option<PathBuf>,
}

struct InterleavedTraceLogger {
    writer: std::sync::Mutex<BufWriter<File>>,
}

#[derive(Debug, Serialize)]
struct InterleavedTraceEvent<'a> {
    ts_ms: u128,
    session_id: u64,
    #[serde(flatten)]
    payload: InterleavedTracePayload<'a>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum InterleavedTracePayload<'a> {
    SessionStarted {
        system_prompt: &'a str,
    },
    UserText {
        text: &'a str,
    },
    UserAudio {
        sample_rate: u32,
        input_samples: usize,
        input_duration_ms: u64,
        text_prompt: Option<&'a str>,
        include_transcript: bool,
    },
    AssistantTurn {
        user_transcript: Option<&'a str>,
        text: &'a str,
        stream: Option<InterleavedTraceStreamSummary>,
    },
    SessionReset,
    SessionClosed,
}

#[derive(Debug, Serialize, Clone, Copy)]
struct InterleavedTraceStreamSummary {
    saw_stream_event: bool,
    sent_audio: bool,
    chunk_count: usize,
    audio_samples: usize,
    first_frame_elapsed_ms: Option<u64>,
    first_decode_elapsed_ms: Option<u64>,
    first_detokenizer_wait_ms: Option<u64>,
    max_frame_gap_ms: u64,
    max_decode_elapsed_ms: u64,
    max_detokenizer_wait_ms: u64,
    max_queue_wait_ms: u64,
}

impl InterleavedTraceLogger {
    fn new(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        Ok(Self {
            writer: std::sync::Mutex::new(BufWriter::new(file)),
        })
    }

    fn log<'a>(&self, session_id: u64, payload: InterleavedTracePayload<'a>) {
        let event = InterleavedTraceEvent {
            ts_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or(0),
            session_id,
            payload,
        };
        let Ok(serialized) = serde_json::to_string(&event) else {
            warn!(session_id, "failed to serialize interleaved trace event");
            return;
        };
        let mut writer = match self.writer.lock() {
            Ok(writer) => writer,
            Err(_) => {
                warn!(session_id, "interleaved trace logger lock poisoned");
                return;
            }
        };
        if let Err(err) = writeln!(writer, "{}", serialized).and_then(|_| writer.flush()) {
            warn!(session_id, error = %err, "failed to write interleaved trace event");
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct InterleavedOverrides {
    n_text: Option<usize>,
    n_audio: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct StreamingDecodeConfig {
    batch_frames: usize,
    context_frames: usize,
}

impl StreamingDecodeConfig {
    fn new(batch_frames: usize, context_frames: usize) -> Self {
        Self {
            batch_frames: batch_frames.max(1),
            context_frames: context_frames,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DevicePreference {
    Auto,
    Specific(Device),
}

#[derive(Debug, Clone, Copy)]
struct AutoDetectRuntime {
    nvidia_device_present: bool,
    is_macos: bool,
    is_windows: bool,
}

impl AutoDetectRuntime {
    fn current() -> Self {
        Self {
            nvidia_device_present: Path::new("/dev/nvidia0").exists()
                || Path::new("/dev/nvidiactl").exists(),
            is_macos: cfg!(target_os = "macos"),
            is_windows: cfg!(target_os = "windows"),
        }
    }
}

impl ServerConfig {
    fn from_env_and_args() -> Result<Self> {
        let mut model_path = PathBuf::from(
            env::var("LFM2_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string()),
        );
        let mut static_dir = env::current_dir()
            .context("failed to read current directory")?
            .join("static");
        let mut bind_addr = env::var("LFM2_BIND_ADDR")
            .unwrap_or_else(|_| DEFAULT_BIND_ADDR.to_string())
            .parse()
            .context("invalid bind address")?;
        let mut precision = env::var("LFM2_PRECISION")
            .ok()
            .map(|value| value.parse())
            .transpose()
            .map_err(|err: String| anyhow!(err))?
            .unwrap_or(Precision::Q4);
        let mut device_preference = env::var("LFM2_DEVICE")
            .ok()
            .map(|value| parse_device_preference(&value))
            .transpose()?
            .unwrap_or(DevicePreference::Auto);
        let mut cpu_workers = env::var("LFM2_CPU_WORKERS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1usize);
        let mut stream_batch_frames = env::var("LFM2_STREAM_BATCH_FRAMES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(STREAM_DECODE_BATCH_FRAMES);
        let mut stream_context_frames = env::var("LFM2_STREAM_CONTEXT_FRAMES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(STREAM_DECODE_CONTEXT_FRAMES);
        let mut interleaved_n_text = env::var("LFM2_INTERLEAVED_N_TEXT")
            .ok()
            .and_then(|value| value.parse().ok());
        let mut interleaved_n_audio = env::var("LFM2_INTERLEAVED_N_AUDIO")
            .ok()
            .and_then(|value| value.parse().ok());
        let mut interleaved_log_path = env::var("LFM2_INTERLEAVED_LOG_PATH")
            .ok()
            .map(PathBuf::from);
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => {
                    model_path = PathBuf::from(
                        args.next()
                            .ok_or_else(|| anyhow!("missing value for --model"))?,
                    );
                }
                "--static-dir" => {
                    static_dir = PathBuf::from(
                        args.next()
                            .ok_or_else(|| anyhow!("missing value for --static-dir"))?,
                    );
                }
                "--bind" => {
                    bind_addr = args
                        .next()
                        .ok_or_else(|| anyhow!("missing value for --bind"))?
                        .parse()
                        .context("invalid --bind socket address")?;
                }
                "--precision" => {
                    precision = args
                        .next()
                        .ok_or_else(|| anyhow!("missing value for --precision"))?
                        .parse()
                        .map_err(|err: String| anyhow!(err))?;
                }
                "--device" => {
                    device_preference = parse_device_preference(
                        &args
                            .next()
                            .ok_or_else(|| anyhow!("missing value for --device"))?,
                    )?;
                }
                "--cpu-workers" => {
                    cpu_workers = args
                        .next()
                        .ok_or_else(|| anyhow!("missing value for --cpu-workers"))?
                        .parse()
                        .context("invalid --cpu-workers value")?;
                }
                "--stream-batch-frames" => {
                    stream_batch_frames = args
                        .next()
                        .ok_or_else(|| anyhow!("missing value for --stream-batch-frames"))?
                        .parse()
                        .context("invalid --stream-batch-frames value")?;
                }
                "--stream-context-frames" => {
                    stream_context_frames = args
                        .next()
                        .ok_or_else(|| anyhow!("missing value for --stream-context-frames"))?
                        .parse()
                        .context("invalid --stream-context-frames value")?;
                }
                "--interleaved-n-text" => {
                    interleaved_n_text = Some(
                        args.next()
                            .ok_or_else(|| anyhow!("missing value for --interleaved-n-text"))?
                            .parse()
                            .context("invalid --interleaved-n-text value")?,
                    );
                }
                "--interleaved-n-audio" => {
                    interleaved_n_audio = Some(
                        args.next()
                            .ok_or_else(|| anyhow!("missing value for --interleaved-n-audio"))?
                            .parse()
                            .context("invalid --interleaved-n-audio value")?,
                    );
                }
                "--interleaved-log" => {
                    interleaved_log_path = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| anyhow!("missing value for --interleaved-log"))?,
                    ));
                }
                other => return Err(anyhow!("unknown argument: {}", other)),
            }
        }

        Ok(Self {
            model_path,
            static_dir,
            bind_addr,
            precision,
            device_preference,
            cpu_workers,
            interleaved_n_text,
            interleaved_n_audio,
            stream_decode: StreamingDecodeConfig::new(
                stream_batch_frames,
                stream_context_frames,
            ),
            interleaved_log_path,
        })
    }
}

fn parse_device_preference(value: &str) -> Result<DevicePreference> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(DevicePreference::Auto),
        "cpu" => Ok(DevicePreference::Specific(Device::CPU)),
        "cuda" => Ok(DevicePreference::Specific(Device::Cuda)),
        "coreml" => Ok(DevicePreference::Specific(Device::CoreML)),
        "directml" => Ok(DevicePreference::Specific(Device::DirectML)),
        "tensorrt" => Ok(DevicePreference::Specific(Device::TensorRT)),
        other => Err(anyhow!("unknown device: {}", other)),
    }
}

fn apply_interleaved_overrides(
    system_prompt: String,
    overrides: InterleavedOverrides,
) -> InterleavedOptions {
    InterleavedOptions {
        system_prompt,
        interleaved_n_text: overrides.n_text,
        interleaved_n_audio: overrides.n_audio,
        ..Default::default()
    }
}

fn auto_detect_device(runtime: AutoDetectRuntime) -> Device {
    let _ = (
        runtime.nvidia_device_present,
        runtime.is_macos,
        runtime.is_windows,
    );

    #[cfg(feature = "cuda")]
    if runtime.nvidia_device_present {
        return Device::Cuda;
    }

    #[cfg(feature = "coreml")]
    if runtime.is_macos {
        return Device::CoreML;
    }

    #[cfg(feature = "directml")]
    if runtime.is_windows {
        return Device::DirectML;
    }

    Device::CPU
}

fn resolve_device(preference: DevicePreference, runtime: AutoDetectRuntime) -> Device {
    match preference {
        DevicePreference::Auto => auto_detect_device(runtime),
        DevicePreference::Specific(device) => device,
    }
}

fn effective_worker_count(device: Device, requested: usize) -> usize {
    match device {
        Device::CPU => requested.max(1),
        _ => 1,
    }
}

#[derive(Debug, Serialize, Clone)]
struct ErrorBody {
    error: String,
}

#[derive(Debug, Clone)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

type ApiResult<T> = std::result::Result<T, ApiError>;

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    model: &'static str,
}

#[derive(Debug, Serialize)]
struct AsrResponse {
    text: String,
    sample_rate: u32,
    duration_ms: u64,
}

#[derive(Debug, Deserialize)]
struct TtsRequest {
    text: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    text_temperature: Option<f32>,
    #[serde(default)]
    audio_temperature: Option<f32>,
    #[serde(default)]
    audio_top_k: Option<usize>,
}

#[derive(Debug)]
struct TtsResponse {
    wav_bytes: Vec<u8>,
}

#[derive(Debug)]
struct AssistantTurn {
    user_transcript: Option<String>,
    text: String,
}

#[derive(Debug)]
enum AssistantStreamEvent {
    TextUpdated(String),
    AudioChunk {
        bytes: Vec<u8>,
        chunk_index: usize,
        frame_index: usize,
        frame_elapsed_ms: u64,
        frame_gap_ms: u64,
        decode_elapsed_ms: u64,
        detokenizer_wait_ms: u64,
        backend: &'static str,
        produced_at: Instant,
    },
}

#[derive(Debug)]
struct DecodedAudioChunk {
    bytes: Vec<u8>,
    decode_index: usize,
    emitted_samples: usize,
    decode_elapsed_ms: u64,
    detokenizer_wait_ms: u64,
    backend: &'static str,
}

type DetokenizeResult = ApiResult<DetokenizeCompletion>;

struct DetokenizeRequest {
    decode_index: usize,
    codes: Vec<[u16; 8]>,
    submitted_at: Instant,
    reply_tx: std_mpsc::Sender<DetokenizeResult>,
}

#[derive(Debug, Clone)]
struct DetokenizeCompletion {
    decode_index: usize,
    waveform: Vec<f32>,
    decode_elapsed_ms: u64,
    detokenizer_wait_ms: u64,
    backend: &'static str,
}

impl DetokenizeCompletion {
    #[cfg(test)]
    fn from_waveform(
        decode_index: usize,
        waveform: Vec<f32>,
        decode_elapsed_ms: u64,
        detokenizer_wait_ms: u64,
        backend: &'static str,
    ) -> Self {
        Self {
            decode_index,
            waveform,
            decode_elapsed_ms,
            detokenizer_wait_ms,
            backend,
        }
    }
}

#[derive(Clone)]
struct DetokenizerWorkerHandle {
    request_tx: std_mpsc::Sender<DetokenizeRequest>,
}

trait DetokenizerClient {
    fn submit(&mut self, request: DetokenizeRequest) -> ApiResult<()>;
    fn try_complete(&mut self) -> ApiResult<Option<DetokenizeCompletion>>;
    fn recv_complete(&mut self) -> ApiResult<DetokenizeCompletion>;
}

struct WorkerDetokenizerClient {
    request_tx: std_mpsc::Sender<DetokenizeRequest>,
    result_tx: std_mpsc::Sender<DetokenizeResult>,
    result_rx: std_mpsc::Receiver<DetokenizeResult>,
}

impl WorkerDetokenizerClient {
    fn new(handle: DetokenizerWorkerHandle) -> Self {
        let (result_tx, result_rx) = std_mpsc::channel();
        Self {
            request_tx: handle.request_tx,
            result_tx,
            result_rx,
        }
    }
}

impl DetokenizerClient for WorkerDetokenizerClient {
    fn submit(&mut self, mut request: DetokenizeRequest) -> ApiResult<()> {
        request.reply_tx = self.result_tx.clone();
        self.request_tx
            .send(request)
            .map_err(|_| ApiError::internal("dedicated detokenizer worker is unavailable"))
    }

    fn try_complete(&mut self) -> ApiResult<Option<DetokenizeCompletion>> {
        match self.result_rx.try_recv() {
            Ok(result) => result.map(Some),
            Err(std_mpsc::TryRecvError::Empty) => Ok(None),
            Err(std_mpsc::TryRecvError::Disconnected) => Err(ApiError::internal(
                "dedicated detokenizer worker disconnected",
            )),
        }
    }

    fn recv_complete(&mut self) -> ApiResult<DetokenizeCompletion> {
        self.result_rx
            .recv()
            .map_err(|_| ApiError::internal("dedicated detokenizer worker disconnected"))?
    }
}

enum ModelCommand {
    Asr {
        audio: Vec<f32>,
        sample_rate: u32,
        system_prompt: Option<String>,
        reply: oneshot::Sender<ApiResult<AsrResponse>>,
    },
    Tts {
        request: TtsRequest,
        reply: oneshot::Sender<ApiResult<TtsResponse>>,
    },
    SessionStart {
        session_id: u64,
        system_prompt: String,
        reply: oneshot::Sender<ApiResult<()>>,
    },
    SessionText {
        session_id: u64,
        text: String,
        stream: mpsc::Sender<AssistantStreamEvent>,
        reply: oneshot::Sender<ApiResult<AssistantTurn>>,
    },
    SessionAudio {
        session_id: u64,
        audio: Vec<f32>,
        sample_rate: u32,
        text_prompt: Option<String>,
        stream: mpsc::Sender<AssistantStreamEvent>,
        reply: oneshot::Sender<ApiResult<AssistantTurn>>,
    },
    SessionReset {
        session_id: u64,
        reply: oneshot::Sender<ApiResult<()>>,
    },
    SessionClose {
        session_id: u64,
    },
}

struct InflightDecode {
    decode_index: usize,
    submitted_frame_count: usize,
    pending_frames: usize,
    flushed_frames: usize,
    zero_context: bool,
}

struct StreamingAudioDecoder {
    client: Box<dyn DetokenizerClient + Send>,
    all_codes: Vec<[u16; 8]>,
    flushed_frames: usize,
    batch_frames: usize,
    context_frames: usize,
    session_id: u64,
    decode_index: usize,
    last_emitted_samples: usize,
    inflight: VecDeque<InflightDecode>,
    ready_chunks: VecDeque<DecodedAudioChunk>,
}

fn split_new_waveform_tail(
    waveform: Vec<f32>,
    last_emitted_samples: usize,
) -> ApiResult<(Vec<f32>, usize)> {
    if last_emitted_samples > waveform.len() {
        return Err(ApiError::internal(
            "streaming detokenizer produced no new waveform tail",
        ));
    }

    let next_emitted_samples = waveform.len();
    if last_emitted_samples == 0 {
        return Ok((waveform, next_emitted_samples));
    }

    Ok((waveform[last_emitted_samples..].to_vec(), next_emitted_samples))
}

impl StreamingAudioDecoder {
    fn new(
        detokenizer: DetokenizerWorkerHandle,
        config: StreamingDecodeConfig,
        session_id: u64,
    ) -> ApiResult<Self> {
        Ok(Self::new_with_client(
            Box::new(WorkerDetokenizerClient::new(detokenizer)),
            config,
            session_id,
        ))
    }

    fn new_with_client(
        client: Box<dyn DetokenizerClient + Send>,
        config: StreamingDecodeConfig,
        session_id: u64,
    ) -> Self {
        Self {
            client,
            all_codes: Vec::new(),
            flushed_frames: 0,
            batch_frames: config.batch_frames,
            context_frames: config.context_frames,
            session_id,
            decode_index: 0,
            last_emitted_samples: 0,
            inflight: VecDeque::new(),
            ready_chunks: VecDeque::new(),
        }
    }

    fn push_frame(&mut self, frame: [u16; 8]) -> ApiResult<Option<DecodedAudioChunk>> {
        self.all_codes.push(frame);
        if self.context_frames == 0 {
            self.submit_zero_context_batches(false)?;
            self.cap_zero_context_backlog()?;
        } else {
            self.submit_pending_if_idle(false)?;
        }
        Ok(self.ready_chunks.pop_front())
    }

    fn drain_ready(&mut self) -> ApiResult<Vec<DecodedAudioChunk>> {
        let mut ready = self.ready_chunks.drain(..).collect::<Vec<_>>();

        loop {
            if self.context_frames == 0 {
                self.submit_zero_context_batches(false)?;
            } else {
                self.submit_pending_if_idle(false)?;
            }
            let Some(completion) = self.client.try_complete()? else {
                break;
            };
            if let Some(chunk) = self.process_completion(completion)? {
                ready.push(chunk);
            }
        }

        Ok(ready)
    }

    fn finish(&mut self) -> ApiResult<Vec<DecodedAudioChunk>> {
        let mut drained = self.drain_ready()?;

        if self.context_frames == 0 {
            self.submit_zero_context_batches(true)?;
        }

        while !self.inflight.is_empty() {
            let completion = self.client.recv_complete()?;
            if let Some(chunk) = self.process_completion(completion)? {
                drained.push(chunk);
            }
            if self.context_frames == 0 {
                self.submit_zero_context_batches(true)?;
            } else {
                self.submit_pending_if_idle(true)?;
            }
        }

        if self.context_frames > 0 && self.pending_frames() > 0 {
            self.submit_pending_if_idle(true)?;
            while !self.inflight.is_empty() {
                let completion = self.client.recv_complete()?;
                if let Some(chunk) = self.process_completion(completion)? {
                    drained.push(chunk);
                }
                self.submit_pending_if_idle(true)?;
            }
        }

        Ok(drained)
    }

    fn submit_pending_if_idle(&mut self, allow_partial: bool) -> ApiResult<()> {
        if !self.inflight.is_empty() {
            return Ok(());
        }

        let pending_frames = self.pending_frames();
        if pending_frames == 0 || (!allow_partial && pending_frames < self.batch_frames) {
            return Ok(());
        }

        let decode_index = self.decode_index + 1;
        let submitted_frame_count = self.all_codes.len();
        self.client.submit(DetokenizeRequest {
            decode_index,
            codes: self.all_codes.clone(),
            submitted_at: Instant::now(),
            reply_tx: std_mpsc::channel().0,
        })?;
        self.decode_index = decode_index;
        self.inflight.push_back(InflightDecode {
            decode_index,
            submitted_frame_count,
            pending_frames,
            flushed_frames: self.flushed_frames,
            zero_context: false,
        });

        Ok(())
    }

    fn submit_zero_context_batches(&mut self, allow_partial: bool) -> ApiResult<()> {
        while self.all_codes.len() >= self.batch_frames {
            self.submit_zero_context_batch(self.batch_frames)?;
        }

        if allow_partial && !self.all_codes.is_empty() {
            self.submit_zero_context_batch(self.all_codes.len())?;
        }

        Ok(())
    }

    fn submit_zero_context_batch(&mut self, frame_count: usize) -> ApiResult<()> {
        let decode_index = self.decode_index + 1;
        let codes = self.all_codes.drain(..frame_count).collect::<Vec<_>>();
        self.client.submit(DetokenizeRequest {
            decode_index,
            codes,
            submitted_at: Instant::now(),
            reply_tx: std_mpsc::channel().0,
        })?;
        self.decode_index = decode_index;
        self.inflight.push_back(InflightDecode {
            decode_index,
            submitted_frame_count: frame_count,
            pending_frames: frame_count,
            flushed_frames: 0,
            zero_context: true,
        });
        Ok(())
    }

    fn cap_zero_context_backlog(&mut self) -> ApiResult<()> {
        while self.inflight.len() > STREAM_DETOKENIZER_MAX_INFLIGHT_BATCHES {
            let completion = self.client.recv_complete()?;
            if let Some(chunk) = self.process_completion(completion)? {
                self.ready_chunks.push_back(chunk);
            }
        }
        Ok(())
    }

    fn process_completion(
        &mut self,
        completion: DetokenizeCompletion,
    ) -> ApiResult<Option<DecodedAudioChunk>> {
        let inflight = self
            .inflight
            .pop_front()
            .ok_or_else(|| ApiError::internal("received detokenizer completion without in-flight request"))?;
        if inflight.decode_index != completion.decode_index {
            return Err(ApiError::internal(format!(
                "detokenizer completion out of order: expected {}, got {}",
                inflight.decode_index, completion.decode_index
            )));
        }

        if inflight.zero_context {
            info!(
                session_id = self.session_id,
                decode_index = completion.decode_index,
                pending_frames = inflight.pending_frames,
                window_frames = inflight.submitted_frame_count,
                flushed_frames = inflight.flushed_frames,
                batch_frames = self.batch_frames,
                context_frames = self.context_frames,
                emitted_samples = completion.waveform.len(),
                full_decode_ms = completion.decode_elapsed_ms,
                total_decode_ms = completion.decode_elapsed_ms,
                detokenizer_wait_ms = completion.detokenizer_wait_ms,
                retained_frames = self.all_codes.len(),
                "streaming detokenizer flush"
            );
            return Ok(Some(DecodedAudioChunk {
                bytes: encode_pcm_s16le_bytes(&completion.waveform),
                decode_index: completion.decode_index,
                emitted_samples: completion.waveform.len(),
                decode_elapsed_ms: completion.decode_elapsed_ms,
                detokenizer_wait_ms: completion.detokenizer_wait_ms,
                backend: completion.backend,
            }));
        }

        let window_frames = inflight.submitted_frame_count;
        let pending_frames = inflight.pending_frames;
        let flushed_frames = inflight.flushed_frames;
        let (new_waveform, next_emitted_samples) =
            split_new_waveform_tail(completion.waveform, self.last_emitted_samples)?;
        self.last_emitted_samples = next_emitted_samples;
        self.retain_recent_context_after_decode(window_frames)?;
        let context_decode_ms = 0;

        if new_waveform.is_empty() {
            return Ok(None);
        }

        info!(
            session_id = self.session_id,
            decode_index = completion.decode_index,
            pending_frames,
            window_frames,
            flushed_frames,
            batch_frames = self.batch_frames,
            context_frames = self.context_frames,
            emitted_samples = new_waveform.len(),
            full_decode_ms = completion.decode_elapsed_ms,
            context_decode_ms,
            total_decode_ms = completion.decode_elapsed_ms,
            detokenizer_wait_ms = completion.detokenizer_wait_ms,
            retained_frames = self.all_codes.len(),
            "streaming detokenizer flush"
        );
        Ok(Some(DecodedAudioChunk {
            bytes: encode_pcm_s16le_bytes(&new_waveform),
            decode_index: completion.decode_index,
            emitted_samples: new_waveform.len(),
            decode_elapsed_ms: completion.decode_elapsed_ms,
            detokenizer_wait_ms: completion.detokenizer_wait_ms,
            backend: completion.backend,
        }))
    }

    fn pending_frames(&self) -> usize {
        self.all_codes.len().saturating_sub(self.flushed_frames)
    }

    fn retain_recent_context_after_decode(
        &mut self,
        completed_window_frames: usize,
    ) -> ApiResult<()> {
        if completed_window_frames > self.all_codes.len() {
            return Err(ApiError::internal(
                "detokenizer completed more frames than are buffered",
            ));
        }

        let keep_start = completed_window_frames.saturating_sub(self.context_frames);
        if keep_start > 0 {
            let drained_samples = keep_start * STREAM_SAMPLES_PER_FRAME;
            self.last_emitted_samples = self.last_emitted_samples.saturating_sub(drained_samples);
            self.all_codes.drain(..keep_start);
        }
        self.flushed_frames = completed_window_frames.saturating_sub(keep_start);
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ClientWsMessage {
    #[serde(rename = "session.start")]
    SessionStart {
        #[serde(default)]
        system_prompt: Option<String>,
    },
    #[serde(rename = "user.text")]
    UserText { text: String },
    #[serde(rename = "user.audio.start")]
    UserAudioStart {
        sample_rate: u32,
        format: String,
        channels: u16,
    },
    #[serde(rename = "user.audio.end")]
    UserAudioEnd {
        #[serde(default)]
        text_prompt: Option<String>,
        #[serde(default)]
        include_transcript: bool,
    },
    #[serde(rename = "session.reset")]
    SessionReset,
    #[serde(rename = "ping")]
    Ping,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ServerWsMessage<'a> {
    #[serde(rename = "status")]
    Status { phase: &'a str },
    #[serde(rename = "assistant.text.delta")]
    AssistantTextDelta { text: &'a str },
    #[serde(rename = "assistant.text.done")]
    AssistantTextDone { text: &'a str },
    #[serde(rename = "user.transcript")]
    UserTranscript { text: &'a str },
    #[serde(rename = "assistant.audio.start")]
    AssistantAudioStart {
        sample_rate: u32,
        format: &'a str,
        channels: u16,
        chunk_samples: usize,
    },
    #[serde(rename = "assistant.audio.chunk")]
    AssistantAudioChunk {
        chunk_index: usize,
        frame_index: usize,
        frame_elapsed_ms: u64,
        frame_gap_ms: u64,
        decode_elapsed_ms: u64,
        detokenizer_wait_ms: u64,
        queue_wait_ms: u64,
        chunk_samples: usize,
        backend: &'a str,
    },
    #[serde(rename = "assistant.audio.end")]
    AssistantAudioEnd,
    #[serde(rename = "session.started")]
    SessionStarted,
    #[serde(rename = "session.reset")]
    SessionReset,
    #[serde(rename = "pong")]
    Pong,
    #[serde(rename = "error")]
    Error { message: &'a str },
}

#[derive(Debug)]
struct PendingAudioTurn {
    sample_rate: u32,
    bytes: Vec<u8>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let config = ServerConfig::from_env_and_args()?;
    let runtime = AutoDetectRuntime::current();
    let device = resolve_device(config.device_preference, runtime);
    let worker_count = effective_worker_count(device, config.cpu_workers);

    if !config.static_dir.exists() {
        return Err(anyhow!(
            "static directory not found: {}",
            config.static_dir.display()
        ));
    }

    info!(
        "Loading model from {} with precision {:?} on {:?} using {} worker(s), interleaved_n_text={:?}, interleaved_n_audio={:?}, stream_batch_frames={}, stream_context_frames={}, interleaved_log_path={:?}",
        config.model_path.display(),
        config.precision,
        device,
        worker_count,
        config.interleaved_n_text,
        config.interleaved_n_audio,
        config.stream_decode.batch_frames,
        config.stream_decode.context_frames,
        config.interleaved_log_path,
    );

    let mut workers = Vec::with_capacity(worker_count);
    for worker_index in 0..worker_count {
        let model_path = config.model_path.clone();
        let precision = config.precision;
        let stream_decode = config.stream_decode;
        let interleaved_overrides = InterleavedOverrides {
            n_text: config.interleaved_n_text,
            n_audio: config.interleaved_n_audio,
        };
        let (tx, rx) = mpsc::channel(32);
        std::thread::Builder::new()
            .name(format!("lfm2-worker-{}", worker_index))
            .spawn(move || {
                let model = Box::leak(Box::new(
                    LFM2Audio::from_pretrained(&model_path, precision, device)
                        .expect("failed to load worker model"),
                ));
                let detokenizer = spawn_detokenizer_worker(
                    model_path.clone(),
                    precision,
                    device,
                    worker_index,
                )
                .expect("failed to spawn dedicated detokenizer worker");
                model_worker(
                    model,
                    rx,
                    stream_decode,
                    interleaved_overrides,
                    detokenizer,
                );
            })
            .map_err(|err| anyhow!("failed to spawn worker thread: {}", err))?;
        workers.push(tx);
    }

    let interleaved_trace = config
        .interleaved_log_path
        .as_deref()
        .map(InterleavedTraceLogger::new)
        .transpose()?
        .map(Arc::new);

    let state = AppState {
        workers: Arc::new(workers),
        next_session_id: Arc::new(AtomicU64::new(1)),
        next_worker: Arc::new(AtomicUsize::new(0)),
        session_workers: Arc::new(AsyncMutex::new(HashMap::new())),
        interleaved_trace,
    };

    let app = build_router(&config, state);
    let listener = tokio::net::TcpListener::bind(config.bind_addr).await?;
    info!("Server listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

fn build_router(config: &ServerConfig, state: AppState) -> Router {
    let static_dir = config.static_dir.clone();
    Router::new()
        .route("/health", get(health_check))
        .route("/api/asr", post(transcribe_raw))
        .route("/api/tts", post(synthesize))
        .route("/ws/interleaved", get(interleaved_socket))
        .route("/v1/audio/transcriptions", post(transcribe_raw))
        .route("/v1/audio/speech", post(synthesize))
        .route_service("/", ServeFile::new(static_dir.join("index.html")))
        .nest_service("/static", ServeDir::new(static_dir))
        .layer(middleware::map_response(no_store_response))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn no_store_response(mut response: Response) -> Response {
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, no-cache, must-revalidate"),
    );
    response
}

fn next_worker_index(state: &AppState) -> usize {
    state.next_worker.fetch_add(1, Ordering::Relaxed) % state.workers.len()
}

fn background_worker_index(primary_worker: usize, worker_count: usize) -> usize {
    if worker_count <= 1 {
        primary_worker
    } else {
        (primary_worker + 1) % worker_count
    }
}

async fn session_worker_index(state: &AppState, session_id: u64) -> usize {
    if let Some(existing) = state.session_workers.lock().await.get(&session_id).copied() {
        return existing;
    }

    let worker_index = next_worker_index(state);
    state
        .session_workers
        .lock()
        .await
        .insert(session_id, worker_index);
    worker_index
}

async fn close_session_worker(state: &AppState, session_id: u64) -> Option<usize> {
    state.session_workers.lock().await.remove(&session_id)
}

fn spawn_detokenizer_worker(
    model_path: PathBuf,
    precision: Precision,
    device: Device,
    worker_index: usize,
) -> Result<DetokenizerWorkerHandle> {
    let (request_tx, request_rx) = std_mpsc::channel::<DetokenizeRequest>();
    std::thread::Builder::new()
        .name(format!("lfm2-detokenizer-{}", worker_index))
        .spawn(move || {
            let mut detokenizer = load_audio_detokenizer_session_with_threads(
                &model_path,
                precision,
                device,
                STREAM_DETOKENIZER_INTRA_THREADS,
            )
                .expect("failed to load dedicated detokenizer session");

            while let Ok(request) = request_rx.recv() {
                let detokenizer_wait_ms = request.submitted_at.elapsed().as_millis() as u64;
                let decode_started_at = Instant::now();
                let result = lfm2_audio::decode_audio_codes_standalone(&mut detokenizer, &request.codes)
                    .map(|waveform| DetokenizeCompletion {
                        decode_index: request.decode_index,
                        waveform,
                        decode_elapsed_ms: decode_started_at.elapsed().as_millis() as u64,
                        detokenizer_wait_ms,
                        backend: "onnx",
                    })
                    .map_err(|err| ApiError::internal(err.to_string()));
                let _ = request.reply_tx.send(result);
            }
        })
        .map_err(|err| anyhow!("failed to spawn dedicated detokenizer worker: {}", err))?;
    Ok(DetokenizerWorkerHandle { request_tx })
}

fn emit_streaming_audio_chunk(
    stream: &mpsc::Sender<AssistantStreamEvent>,
    chunk: DecodedAudioChunk,
    emitted_audio_chunks: &mut usize,
    emitted_audio_bytes: &mut usize,
    first_audio_chunk_at_ms: &mut Option<u64>,
    session_id: u64,
    audio_frame_index: usize,
    frame_elapsed_ms: u64,
    frame_gap_ms: u64,
    stream_started_at: Instant,
    final_flush: bool,
) -> ApiResult<()> {
    *emitted_audio_chunks += 1;
    *emitted_audio_bytes += chunk.bytes.len();
    let elapsed_ms = stream_started_at.elapsed().as_millis() as u64;
    first_audio_chunk_at_ms.get_or_insert(elapsed_ms);
    info!(
        session_id,
        chunk_index = *emitted_audio_chunks,
        frame_index = audio_frame_index,
        chunk_bytes = chunk.bytes.len(),
        chunk_samples = chunk.emitted_samples,
        decode_index = chunk.decode_index,
        decode_elapsed_ms = chunk.decode_elapsed_ms,
        detokenizer_wait_ms = chunk.detokenizer_wait_ms,
        backend = chunk.backend,
        elapsed_ms,
        final_flush,
        "streaming audio chunk ready"
    );
    stream
        .blocking_send(AssistantStreamEvent::AudioChunk {
            bytes: chunk.bytes,
            chunk_index: *emitted_audio_chunks,
            frame_index: audio_frame_index,
            frame_elapsed_ms,
            frame_gap_ms,
            decode_elapsed_ms: chunk.decode_elapsed_ms,
            detokenizer_wait_ms: chunk.detokenizer_wait_ms,
            backend: chunk.backend,
            produced_at: Instant::now(),
        })
        .map_err(|_| ApiError::internal("stream receiver dropped"))
}

fn model_worker(
    model: &'static LFM2Audio,
    mut rx: mpsc::Receiver<ModelCommand>,
    stream_decode: StreamingDecodeConfig,
    interleaved_overrides: InterleavedOverrides,
    detokenizer: DetokenizerWorkerHandle,
) {
    let mut sessions: HashMap<u64, lfm2_audio::ChatSession<'static>> = HashMap::new();

    while let Some(command) = rx.blocking_recv() {
        match command {
            ModelCommand::Asr {
                audio,
                sample_rate,
                system_prompt,
                reply,
            } => {
                let options = ASROptions {
                    system_prompt: system_prompt.unwrap_or_else(|| "Perform ASR.".to_string()),
                    ..Default::default()
                };
                let result = model
                    .asr()
                    .transcribe(&audio, sample_rate, &options)
                    .map(|text| AsrResponse {
                        duration_ms: ((audio.len() as f64 / sample_rate as f64) * 1000.0) as u64,
                        sample_rate,
                        text,
                    })
                    .map_err(|err| ApiError::internal(err.to_string()));
                let _ = reply.send(result);
            }
            ModelCommand::Tts { request, reply } => {
                let options = TTSOptions {
                    system_prompt: resolve_tts_system_prompt(
                        request.system_prompt.as_deref(),
                        request.voice.as_deref(),
                    ),
                    max_new_tokens: request.max_tokens.unwrap_or(1024),
                    text_temperature: request.text_temperature.unwrap_or(1.0),
                    audio_temperature: request.audio_temperature.unwrap_or(0.8),
                    audio_top_k: request.audio_top_k.unwrap_or(64),
                };
                let result = model
                    .tts()
                    .synthesize(&request.text, &options)
                    .and_then(|audio| encode_wav_bytes(&audio, 24_000))
                    .map(|wav_bytes| TtsResponse { wav_bytes })
                    .map_err(|err| ApiError::internal(err.to_string()));
                let _ = reply.send(result);
            }
            ModelCommand::SessionStart {
                session_id,
                system_prompt,
                reply,
            } => {
                sessions.insert(
                    session_id,
                    model.chat_with_options(apply_interleaved_overrides(
                        system_prompt,
                        interleaved_overrides,
                    )),
                );
                let _ = reply.send(Ok(()));
            }
            ModelCommand::SessionText {
                session_id,
                text,
                stream,
                reply,
            } => {
                info!(
                    session_id,
                    text_len = text.len(),
                    "streaming text turn started"
                );
                let session = sessions.entry(session_id).or_insert_with(|| {
                    model.chat_with_options(apply_interleaved_overrides(
                        DEFAULT_SYSTEM_PROMPT_INTERLEAVED.to_string(),
                        interleaved_overrides,
                    ))
                });
                session.add_user_text(&text);
                let mut audio_decoder =
                    match StreamingAudioDecoder::new(detokenizer.clone(), stream_decode, session_id) {
                        Ok(decoder) => decoder,
                        Err(err) => {
                            let _ = reply.send(Err(err));
                            continue;
                        }
                    };
                let stream_started_at = Instant::now();
                let mut emitted_audio_chunks = 0usize;
                let mut emitted_audio_bytes = 0usize;
                let mut first_audio_chunk_at_ms = None;
                let mut audio_frame_index = 0usize;
                let mut last_audio_frame_at = None;
                let result = session
                    .generate_streaming(|event| {
                        match event {
                            lfm2_audio::InterleavedEvent::TextUpdated(text) => stream
                                .blocking_send(AssistantStreamEvent::TextUpdated(text))
                                .map_err(|_| {
                                    lfm2_audio::LFM2Error::Generation(
                                        "stream receiver dropped".to_string(),
                                    )
                                })?,
                            lfm2_audio::InterleavedEvent::AudioFrame(frame) => {
                                audio_frame_index += 1;
                                let frame_elapsed_ms =
                                    stream_started_at.elapsed().as_millis() as u64;
                                let frame_gap_ms = last_audio_frame_at
                                    .replace(Instant::now())
                                    .map(|instant| instant.elapsed().as_millis() as u64)
                                    .unwrap_or(0);
                                info!(
                                    session_id,
                                    frame_index = audio_frame_index,
                                    frame_elapsed_ms,
                                    frame_gap_ms,
                                    "audio frame generated"
                                );
                                if let Some(chunk) = audio_decoder
                                    .push_frame(frame)
                                    .map_err(|err| lfm2_audio::LFM2Error::Generation(err.message))?
                                {
                                    emit_streaming_audio_chunk(
                                        &stream,
                                        chunk,
                                        &mut emitted_audio_chunks,
                                        &mut emitted_audio_bytes,
                                        &mut first_audio_chunk_at_ms,
                                        session_id,
                                        audio_frame_index,
                                        frame_elapsed_ms,
                                        frame_gap_ms,
                                        stream_started_at,
                                        false,
                                    )
                                    .map_err(|err| lfm2_audio::LFM2Error::Generation(err.message))?;
                                }
                                for chunk in audio_decoder
                                    .drain_ready()
                                    .map_err(|err| lfm2_audio::LFM2Error::Generation(err.message))?
                                {
                                    emit_streaming_audio_chunk(
                                        &stream,
                                        chunk,
                                        &mut emitted_audio_chunks,
                                        &mut emitted_audio_bytes,
                                        &mut first_audio_chunk_at_ms,
                                        session_id,
                                        audio_frame_index,
                                        frame_elapsed_ms,
                                        frame_gap_ms,
                                        stream_started_at,
                                        false,
                                    )
                                    .map_err(|err| lfm2_audio::LFM2Error::Generation(err.message))?;
                                }
                            }
                        }
                        Ok(())
                    })
                    .map_err(|err| ApiError::internal(err.to_string()))
                    .and_then(|response| {
                        for chunk in audio_decoder
                            .finish()
                            .map_err(|err| ApiError::internal(err.message))?
                        {
                            let elapsed_ms = stream_started_at.elapsed().as_millis() as u64;
                            emit_streaming_audio_chunk(
                                &stream,
                                chunk,
                                &mut emitted_audio_chunks,
                                &mut emitted_audio_bytes,
                                &mut first_audio_chunk_at_ms,
                                session_id,
                                audio_frame_index,
                                elapsed_ms,
                                0,
                                stream_started_at,
                                true,
                            )?;
                        }
                        assistant_turn_from_response(response, None)
                    });
                info!(
                    session_id,
                    total_audio_chunks = emitted_audio_chunks,
                    total_audio_bytes = emitted_audio_bytes,
                    first_audio_chunk_at_ms = first_audio_chunk_at_ms.unwrap_or(0),
                    total_elapsed_ms = stream_started_at.elapsed().as_millis() as u64,
                    "streaming text turn finished"
                );
                let _ = reply.send(result);
            }
            ModelCommand::SessionAudio {
                session_id,
                audio,
                sample_rate,
                text_prompt,
                stream,
                reply,
            } => {
                info!(
                    session_id,
                    input_sample_rate = sample_rate,
                    input_samples = audio.len(),
                    input_duration_ms = ((audio.len() as f64 / sample_rate as f64) * 1000.0)
                        .round() as u64,
                    has_text_prompt = text_prompt.is_some(),
                    "streaming audio turn started"
                );
                let session = sessions.entry(session_id).or_insert_with(|| {
                    model.chat_with_options(apply_interleaved_overrides(
                        DEFAULT_SYSTEM_PROMPT_INTERLEAVED.to_string(),
                        interleaved_overrides,
                    ))
                });
                let add_result =
                    session.add_user_audio_with_text(&audio, sample_rate, text_prompt.as_deref());
                let mut audio_decoder =
                    match StreamingAudioDecoder::new(detokenizer.clone(), stream_decode, session_id) {
                        Ok(decoder) => decoder,
                        Err(err) => {
                            let _ = reply.send(Err(err));
                            continue;
                        }
                    };
                let stream_started_at = Instant::now();
                let mut emitted_audio_chunks = 0usize;
                let mut emitted_audio_bytes = 0usize;
                let mut first_audio_chunk_at_ms = None;
                let mut audio_frame_index = 0usize;
                let mut last_audio_frame_at = None;
                let result = add_result
                    .map_err(|err| ApiError::internal(err.to_string()))
                    .and_then(|_| {
                        session
                            .generate_streaming(|event| {
                                match event {
                                    lfm2_audio::InterleavedEvent::TextUpdated(text) => stream
                                        .blocking_send(AssistantStreamEvent::TextUpdated(text))
                                        .map_err(|_| {
                                            lfm2_audio::LFM2Error::Generation(
                                                "stream receiver dropped".to_string(),
                                            )
                                        })?,
                                    lfm2_audio::InterleavedEvent::AudioFrame(frame) => {
                                        audio_frame_index += 1;
                                        let frame_elapsed_ms =
                                            stream_started_at.elapsed().as_millis() as u64;
                                        let frame_gap_ms = last_audio_frame_at
                                            .replace(Instant::now())
                                            .map(|instant| instant.elapsed().as_millis() as u64)
                                            .unwrap_or(0);
                                        info!(
                                            session_id,
                                            frame_index = audio_frame_index,
                                            frame_elapsed_ms,
                                            frame_gap_ms,
                                            "audio frame generated"
                                        );
                                        if let Some(chunk) = audio_decoder
                                            .push_frame(frame)
                                            .map_err(|err| lfm2_audio::LFM2Error::Generation(err.message))?
                                        {
                                            emit_streaming_audio_chunk(
                                                &stream,
                                                chunk,
                                                &mut emitted_audio_chunks,
                                                &mut emitted_audio_bytes,
                                                &mut first_audio_chunk_at_ms,
                                                session_id,
                                                audio_frame_index,
                                                frame_elapsed_ms,
                                                frame_gap_ms,
                                                stream_started_at,
                                                false,
                                            )
                                            .map_err(|err| lfm2_audio::LFM2Error::Generation(err.message))?;
                                        }
                                        for chunk in audio_decoder
                                            .drain_ready()
                                            .map_err(|err| lfm2_audio::LFM2Error::Generation(err.message))?
                                        {
                                            emit_streaming_audio_chunk(
                                                &stream,
                                                chunk,
                                                &mut emitted_audio_chunks,
                                                &mut emitted_audio_bytes,
                                                &mut first_audio_chunk_at_ms,
                                                session_id,
                                                audio_frame_index,
                                                frame_elapsed_ms,
                                                frame_gap_ms,
                                                stream_started_at,
                                                false,
                                            )
                                            .map_err(|err| lfm2_audio::LFM2Error::Generation(err.message))?;
                                        }
                                    }
                                }
                                Ok(())
                            })
                            .map_err(|err| ApiError::internal(err.to_string()))
                            .and_then(|response| {
                                for chunk in audio_decoder
                                    .finish()
                                    .map_err(|err| ApiError::internal(err.message))?
                                {
                                    let elapsed_ms = stream_started_at.elapsed().as_millis() as u64;
                                    emit_streaming_audio_chunk(
                                        &stream,
                                        chunk,
                                        &mut emitted_audio_chunks,
                                        &mut emitted_audio_bytes,
                                        &mut first_audio_chunk_at_ms,
                                        session_id,
                                        audio_frame_index,
                                        elapsed_ms,
                                        0,
                                        stream_started_at,
                                        true,
                                    )?;
                                }
                                assistant_turn_from_response(response, None)
                            })
                    });
                info!(
                    session_id,
                    total_audio_chunks = emitted_audio_chunks,
                    total_audio_bytes = emitted_audio_bytes,
                    first_audio_chunk_at_ms = first_audio_chunk_at_ms.unwrap_or(0),
                    total_elapsed_ms = stream_started_at.elapsed().as_millis() as u64,
                    "streaming audio turn finished"
                );
                let _ = reply.send(result);
            }
            ModelCommand::SessionReset { session_id, reply } => {
                let session = sessions.entry(session_id).or_insert_with(|| {
                    model.chat_with_options(InterleavedOptions {
                        system_prompt: DEFAULT_SYSTEM_PROMPT_INTERLEAVED.to_string(),
                        ..Default::default()
                    })
                });
                let result = session
                    .reset()
                    .map_err(|err| ApiError::internal(err.to_string()));
                let _ = reply.send(result);
            }
            ModelCommand::SessionClose { session_id } => {
                sessions.remove(&session_id);
            }
        }
    }
}

fn assistant_turn_from_response(
    response: lfm2_audio::AssistantResponse,
    user_transcript: Option<String>,
) -> ApiResult<AssistantTurn> {
    Ok(AssistantTurn {
        user_transcript,
        text: response.text,
    })
}

async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        model: "lfm2.5-audio",
    })
}

async fn transcribe_raw(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult<Json<AsrResponse>> {
    validate_wav_content_type(&headers)?;
    let (audio, spec) =
        decode_wav_bytes(&body).map_err(|err| ApiError::bad_request(err.to_string()))?;
    let system_prompt = headers
        .get("x-system-prompt")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);

    let (reply_tx, reply_rx) = oneshot::channel();
    let worker_index = next_worker_index(&state);
    state.workers[worker_index]
        .send(ModelCommand::Asr {
            audio,
            sample_rate: spec.sample_rate,
            system_prompt,
            reply: reply_tx,
        })
        .await
        .map_err(|_| ApiError::internal("model worker unavailable"))?;

    let response = reply_rx
        .await
        .map_err(|_| ApiError::internal("model worker dropped ASR reply"))??;
    Ok(Json(response))
}

/// Given the TTS request, returns WAV bytes.
async fn synthesize(
    State(state): State<AppState>,
    Json(request): Json<TtsRequest>,
) -> ApiResult<Response> {
    if request.text.trim().is_empty() {
        return Err(ApiError::bad_request("text must not be empty"));
    }
    let (reply_tx, reply_rx) = oneshot::channel();
    let worker_index = next_worker_index(&state);
    state.workers[worker_index]
        .send(ModelCommand::Tts {
            request,
            reply: reply_tx,
        })
        .await
        .map_err(|_| ApiError::internal("model worker unavailable"))?;

    let response = reply_rx
        .await
        .map_err(|_| ApiError::internal("model worker dropped TTS reply"))??;

    Ok(([(header::CONTENT_TYPE, "audio/wav")], response.wav_bytes).into_response())
}

async fn interleaved_socket(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    let session_id = state.next_session_id.fetch_add(1, Ordering::Relaxed);
    let mut pending_audio: Option<PendingAudioTurn> = None;

    if let Err(err) =
        send_ws_json(&mut socket, &ServerWsMessage::Status { phase: "listening" }).await
    {
        warn!(
            "failed to send initial status to websocket {}: {}",
            session_id, err.message
        );
        return;
    }

    while let Some(message) = socket.next().await {
        let message = match message {
            Ok(message) => message,
            Err(err) => {
                warn!(
                    "websocket receive error for session {}: {}",
                    session_id, err
                );
                break;
            }
        };

        let result = match message {
            Message::Text(text) => match serde_json::from_str::<ClientWsMessage>(&text) {
                Ok(payload) => {
                    handle_client_message(
                        &mut socket,
                        &state,
                        session_id,
                        &mut pending_audio,
                        payload,
                    )
                    .await
                }
                Err(err) => Err(ApiError::bad_request(format!(
                    "invalid websocket JSON: {}",
                    err
                ))),
            },
            Message::Binary(bytes) => handle_binary_audio(&mut pending_audio, bytes),
            Message::Close(_) => break,
            Message::Ping(payload) => socket
                .send(Message::Pong(payload))
                .await
                .map_err(|err| ApiError::internal(err.to_string())),
            Message::Pong(_) => Ok(()),
        };

        if let Err(err) = result {
            let _ = send_ws_json(
                &mut socket,
                &ServerWsMessage::Error {
                    message: &err.message,
                },
            )
            .await;
        }
    }

    if let Some(worker_index) = close_session_worker(&state, session_id).await {
        let _ = state.workers[worker_index]
            .send(ModelCommand::SessionClose { session_id })
            .await;
    }
    log_interleaved_event(&state, session_id, InterleavedTracePayload::SessionClosed);
}

async fn handle_client_message(
    socket: &mut WebSocket,
    state: &AppState,
    session_id: u64,
    pending_audio: &mut Option<PendingAudioTurn>,
    payload: ClientWsMessage,
) -> ApiResult<()> {
    match payload {
        ClientWsMessage::SessionStart { system_prompt } => {
            let prompt =
                system_prompt.unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT_INTERLEAVED.to_string());
            let worker_index = session_worker_index(state, session_id).await;
            let (reply_tx, reply_rx) = oneshot::channel();
            state.workers[worker_index]
                .send(ModelCommand::SessionStart {
                    session_id,
                    system_prompt: prompt.clone(),
                    reply: reply_tx,
                })
                .await
                .map_err(|_| ApiError::internal("model worker unavailable"))?;
            reply_rx
                .await
                .map_err(|_| ApiError::internal("session start reply dropped"))??;
            log_interleaved_event(
                state,
                session_id,
                InterleavedTracePayload::SessionStarted {
                    system_prompt: &prompt,
                },
            );
            send_ws_json(socket, &ServerWsMessage::SessionStarted).await?;
            send_ws_json(socket, &ServerWsMessage::Status { phase: "listening" }).await?;
            Ok(())
        }
        ClientWsMessage::UserText { text } => {
            if text.trim().is_empty() {
                return Err(ApiError::bad_request("text turn must not be empty"));
            }
            log_interleaved_event(
                state,
                session_id,
                InterleavedTracePayload::UserText { text: &text },
            );
            send_ws_json(
                socket,
                &ServerWsMessage::Status {
                    phase: "processing",
                },
            )
            .await?;
            let worker_index = session_worker_index(state, session_id).await;
            let (reply_tx, reply_rx) = oneshot::channel();
            let (stream_tx, mut stream_rx) = mpsc::channel(STREAM_EVENT_CHANNEL_CAPACITY);

            state.workers[worker_index]
                .send(ModelCommand::SessionText {
                    session_id,
                    text,
                    stream: stream_tx,
                    reply: reply_tx,
                })
                .await
                .map_err(|_| ApiError::internal("model worker unavailable"))?;
            let stream_summary = send_streaming_assistant_events(socket, &mut stream_rx).await?;
            let response = reply_rx
                .await
                .map_err(|_| ApiError::internal("text reply dropped"))??;

            send_assistant_turn(socket, &response).await?;
            log_interleaved_event(
                state,
                session_id,
                InterleavedTracePayload::AssistantTurn {
                    user_transcript: response.user_transcript.as_deref(),
                    text: &response.text,
                    stream: Some(InterleavedTraceStreamSummary {
                        saw_stream_event: stream_summary.saw_stream_event,
                        sent_audio: stream_summary.sent_audio,
                        chunk_count: stream_summary.chunk_count,
                        audio_samples: stream_summary.audio_samples,
                        first_frame_elapsed_ms: stream_summary.first_frame_elapsed_ms,
                        first_decode_elapsed_ms: stream_summary.first_decode_elapsed_ms,
                        first_detokenizer_wait_ms: stream_summary.first_detokenizer_wait_ms,
                        max_frame_gap_ms: stream_summary.max_frame_gap_ms,
                        max_decode_elapsed_ms: stream_summary.max_decode_elapsed_ms,
                        max_detokenizer_wait_ms: stream_summary.max_detokenizer_wait_ms,
                        max_queue_wait_ms: stream_summary.max_queue_wait_ms,
                    }),
                },
            );
            send_ws_json(socket, &ServerWsMessage::Status { phase: "listening" }).await?;
            Ok(())
        }
        ClientWsMessage::UserAudioStart {
            sample_rate,
            format,
            channels,
        } => {
            if format != "pcm_s16le" {
                return Err(ApiError::unsupported(
                    "only pcm_s16le is supported for websocket audio",
                ));
            }
            if channels != 1 {
                return Err(ApiError::unsupported(
                    "only mono websocket audio is supported",
                ));
            }
            *pending_audio = Some(PendingAudioTurn {
                sample_rate,
                bytes: Vec::new(),
            });
            send_ws_json(socket, &ServerWsMessage::Status { phase: "recording" }).await?;
            Ok(())
        }
        ClientWsMessage::UserAudioEnd {
            text_prompt,
            include_transcript,
        } => {
            let pending = pending_audio.take().ok_or_else(|| {
                ApiError::bad_request("received user.audio.end without user.audio.start")
            })?;
            if pending.bytes.is_empty() {
                return Err(ApiError::bad_request("audio turn was empty"));
            }

            let audio = pcm_s16le_bytes_to_f32(&pending.bytes)?;
            log_interleaved_event(
                state,
                session_id,
                InterleavedTracePayload::UserAudio {
                    sample_rate: pending.sample_rate,
                    input_samples: audio.len(),
                    input_duration_ms: ((audio.len() as f64 / pending.sample_rate as f64) * 1000.0)
                        .round() as u64,
                    text_prompt: text_prompt.as_deref(),
                    include_transcript,
                },
            );
            send_ws_json(
                socket,
                &ServerWsMessage::Status {
                    phase: "processing",
                },
            )
            .await?;
            let worker_index = session_worker_index(state, session_id).await;
            let chat_audio = audio.clone();
            let chat_text_prompt = text_prompt.clone();
            let (chat_tx, chat_rx) = oneshot::channel();
            let (stream_tx, mut stream_rx) = mpsc::channel(STREAM_EVENT_CHANNEL_CAPACITY);

            state.workers[worker_index]
                .send(ModelCommand::SessionAudio {
                    session_id,
                    audio: chat_audio,
                    sample_rate: pending.sample_rate,
                    text_prompt: chat_text_prompt,
                    stream: stream_tx,
                    reply: chat_tx,
                })
                .await
                .map_err(|_| ApiError::internal("model worker unavailable"))?;

            let transcript_rx = if include_transcript {
                let transcript_worker = background_worker_index(worker_index, state.workers.len());
                let (transcript_tx, transcript_rx) = oneshot::channel();
                state.workers[transcript_worker]
                    .send(ModelCommand::Asr {
                        audio,
                        sample_rate: pending.sample_rate,
                        system_prompt: None,
                        reply: transcript_tx,
                    })
                    .await
                    .map_err(|_| ApiError::internal("transcript worker unavailable"))?;
                Some(transcript_rx)
            } else {
                None
            };

            let stream_summary = send_streaming_assistant_events(socket, &mut stream_rx).await?;

            let mut response = chat_rx
                .await
                .map_err(|_| ApiError::internal("audio reply dropped"))??;

            if let Some(transcript_rx) = transcript_rx {
                match transcript_rx.await {
                    Ok(Ok(transcript)) if !transcript.text.trim().is_empty() => {
                        response.user_transcript = Some(transcript.text);
                    }
                    Ok(Ok(_)) => {}
                    Ok(Err(err)) => {
                        warn!(
                            "transcript worker failed for session {}: {}",
                            session_id, err.message
                        );
                    }
                    Err(_) => {
                        warn!("transcript worker dropped reply for session {}", session_id);
                    }
                }
            }

            send_assistant_turn(socket, &response).await?;
            log_interleaved_event(
                state,
                session_id,
                InterleavedTracePayload::AssistantTurn {
                    user_transcript: response.user_transcript.as_deref(),
                    text: &response.text,
                    stream: Some(InterleavedTraceStreamSummary {
                        saw_stream_event: stream_summary.saw_stream_event,
                        sent_audio: stream_summary.sent_audio,
                        chunk_count: stream_summary.chunk_count,
                        audio_samples: stream_summary.audio_samples,
                        first_frame_elapsed_ms: stream_summary.first_frame_elapsed_ms,
                        first_decode_elapsed_ms: stream_summary.first_decode_elapsed_ms,
                        first_detokenizer_wait_ms: stream_summary.first_detokenizer_wait_ms,
                        max_frame_gap_ms: stream_summary.max_frame_gap_ms,
                        max_decode_elapsed_ms: stream_summary.max_decode_elapsed_ms,
                        max_detokenizer_wait_ms: stream_summary.max_detokenizer_wait_ms,
                        max_queue_wait_ms: stream_summary.max_queue_wait_ms,
                    }),
                },
            );
            send_ws_json(socket, &ServerWsMessage::Status { phase: "listening" }).await?;
            Ok(())
        }
        ClientWsMessage::SessionReset => {
            *pending_audio = None;
            let worker_index = session_worker_index(state, session_id).await;
            let (reply_tx, reply_rx) = oneshot::channel();
            state.workers[worker_index]
                .send(ModelCommand::SessionReset {
                    session_id,
                    reply: reply_tx,
                })
                .await
                .map_err(|_| ApiError::internal("model worker unavailable"))?;
            reply_rx
                .await
                .map_err(|_| ApiError::internal("session reset reply dropped"))??;
            log_interleaved_event(state, session_id, InterleavedTracePayload::SessionReset);
            send_ws_json(socket, &ServerWsMessage::SessionReset).await?;
            send_ws_json(socket, &ServerWsMessage::Status { phase: "listening" }).await?;
            Ok(())
        }
        ClientWsMessage::Ping => {
            send_ws_json(socket, &ServerWsMessage::Pong).await?;
            Ok(())
        }
    }
}

fn handle_binary_audio(
    pending_audio: &mut Option<PendingAudioTurn>,
    bytes: Bytes,
) -> ApiResult<()> {
    let pending = pending_audio
        .as_mut()
        .ok_or_else(|| ApiError::bad_request("received binary audio before user.audio.start"))?;
    if pending.bytes.len() + bytes.len() > MAX_BINARY_AUDIO_BYTES {
        return Err(ApiError::bad_request("audio turn exceeds maximum size"));
    }
    pending.bytes.extend_from_slice(&bytes);
    Ok(())
}

async fn send_assistant_turn(socket: &mut WebSocket, response: &AssistantTurn) -> ApiResult<()> {
    if let Some(transcript) = response
        .user_transcript
        .as_deref()
        .filter(|text| !text.trim().is_empty())
    {
        send_ws_json(
            socket,
            &ServerWsMessage::UserTranscript { text: transcript },
        )
        .await?;
    }

    send_ws_json(
        socket,
        &ServerWsMessage::AssistantTextDone {
            text: response.text.as_str(),
        },
    )
    .await?;

    Ok(())
}

fn log_interleaved_event<'a>(
    state: &AppState,
    session_id: u64,
    payload: InterleavedTracePayload<'a>,
) {
    if let Some(logger) = &state.interleaved_trace {
        logger.log(session_id, payload);
    }
}

#[derive(Debug, Clone, Copy)]
struct OutputQueueConfig {
    startup_chunks: usize,
}

impl OutputQueueConfig {
    fn new(startup_chunks: usize) -> Self {
        Self {
            startup_chunks: startup_chunks.max(1),
        }
    }
}

#[derive(Debug)]
struct AudioOutputPacer {
    config: OutputQueueConfig,
    pending_chunks: usize,
    started: bool,
    next_release_at: Option<Instant>,
}

impl AudioOutputPacer {
    fn new(config: OutputQueueConfig) -> Self {
        Self {
            config,
            pending_chunks: 0,
            started: false,
            next_release_at: None,
        }
    }

    fn enqueue(&mut self, now: Instant) {
        self.pending_chunks += 1;
        if self.started && self.next_release_at.is_none() {
            self.next_release_at = Some(now);
            return;
        }

        if !self.started && self.pending_chunks >= self.config.startup_chunks {
            self.started = true;
            self.next_release_at = Some(now);
        }
    }

    fn finish_input(&mut self, now: Instant) {
        if self.pending_chunks > 0 && self.next_release_at.is_none() {
            self.started = true;
            self.next_release_at = Some(now);
        }
    }

    fn next_wake_delay(&self, now: Instant) -> Option<Duration> {
        self.next_release_at
            .map(|deadline| deadline.saturating_duration_since(now))
    }

    fn release_one(&mut self, now: Instant, chunk_duration: Duration) {
        self.pending_chunks = self.pending_chunks.saturating_sub(1);

        if self.pending_chunks == 0 {
            self.next_release_at = None;
            return;
        }

        let scheduled_from = self
            .next_release_at
            .map(|deadline| deadline.max(now))
            .unwrap_or(now);
        self.next_release_at = Some(scheduled_from + chunk_duration);
    }
}

#[derive(Debug)]
struct QueuedAudioChunk {
    bytes: Vec<u8>,
    chunk_index: usize,
    frame_index: usize,
    frame_elapsed_ms: u64,
    frame_gap_ms: u64,
    decode_elapsed_ms: u64,
    detokenizer_wait_ms: u64,
    backend: &'static str,
    produced_at: Instant,
}

impl QueuedAudioChunk {
    fn chunk_samples(&self) -> usize {
        self.bytes.len() / 2
    }
}

fn chunk_playback_duration(chunk_samples: usize) -> Duration {
    let seconds = chunk_samples as f64 / 24_000.0;
    Duration::from_secs_f64(seconds)
}

#[derive(Debug, Clone, Copy, Default)]
struct StreamingAssistantSummary {
    saw_stream_event: bool,
    sent_audio: bool,
    chunk_count: usize,
    audio_samples: usize,
    first_frame_elapsed_ms: Option<u64>,
    first_decode_elapsed_ms: Option<u64>,
    first_detokenizer_wait_ms: Option<u64>,
    max_frame_gap_ms: u64,
    max_decode_elapsed_ms: u64,
    max_detokenizer_wait_ms: u64,
    max_queue_wait_ms: u64,
}

async fn send_streaming_assistant_events(
    socket: &mut WebSocket,
    stream_rx: &mut mpsc::Receiver<AssistantStreamEvent>,
) -> ApiResult<StreamingAssistantSummary> {
    let mut sent_audio_start = false;
    let mut saw_event = false;
    let mut stream_closed = false;
    let mut audio_queue = VecDeque::new();
    let mut pacer = AudioOutputPacer::new(OutputQueueConfig::new(STREAM_OUTPUT_QUEUE_CHUNKS));
    let mut summary = StreamingAssistantSummary::default();

    loop {
        if stream_closed && audio_queue.is_empty() {
            break;
        }

        let next_wake_delay = pacer.next_wake_delay(Instant::now());

        tokio::select! {
            event = stream_rx.recv(), if !stream_closed => {
                match event {
                    Some(event) => {
                        if !saw_event {
                            send_ws_json(
                                socket,
                                &ServerWsMessage::Status {
                                    phase: "responding",
                                },
                            )
                            .await?;
                            saw_event = true;
                            summary.saw_stream_event = true;
                        }

                        match event {
                            AssistantStreamEvent::TextUpdated(text) => {
                                send_ws_json(socket, &ServerWsMessage::AssistantTextDelta { text: &text }).await?;
                            }
                            AssistantStreamEvent::AudioChunk {
                                bytes,
                                chunk_index,
                                frame_index,
                                frame_elapsed_ms,
                                frame_gap_ms,
                                decode_elapsed_ms,
                                detokenizer_wait_ms,
                                backend,
                                produced_at,
                            } => {
                                audio_queue.push_back(QueuedAudioChunk {
                                    bytes,
                                    chunk_index,
                                    frame_index,
                                    frame_elapsed_ms,
                                    frame_gap_ms,
                                    decode_elapsed_ms,
                                    detokenizer_wait_ms,
                                    backend,
                                    produced_at,
                                });
                                pacer.enqueue(Instant::now());
                            }
                        }
                    }
                    None => {
                        stream_closed = true;
                        pacer.finish_input(Instant::now());
                    }
                }
            }
            _ = tokio::time::sleep(next_wake_delay.unwrap_or(Duration::from_secs(86400))), if next_wake_delay.is_some() => {
                if let Some(chunk) = audio_queue.pop_front() {
                    let chunk_samples = chunk.chunk_samples();
                    if !sent_audio_start {
                        send_ws_json(
                            socket,
                            &ServerWsMessage::AssistantAudioStart {
                                sample_rate: 24_000,
                                format: "pcm_s16le",
                                channels: 1,
                                chunk_samples,
                            },
                        )
                        .await?;
                        sent_audio_start = true;
                    }
                    let queue_wait_ms = chunk.produced_at.elapsed().as_millis() as u64;
                    let send_started_at = Instant::now();
                    send_ws_json(
                        socket,
                        &ServerWsMessage::AssistantAudioChunk {
                            chunk_index: chunk.chunk_index,
                            frame_index: chunk.frame_index,
                            frame_elapsed_ms: chunk.frame_elapsed_ms,
                            frame_gap_ms: chunk.frame_gap_ms,
                            decode_elapsed_ms: chunk.decode_elapsed_ms,
                            detokenizer_wait_ms: chunk.detokenizer_wait_ms,
                            queue_wait_ms,
                            chunk_samples,
                            backend: chunk.backend,
                        },
                    )
                    .await?;
                    socket
                        .send(Message::Binary(chunk.bytes.into()))
                        .await
                        .map_err(|err| ApiError::internal(err.to_string()))?;
                    summary.sent_audio = true;
                    summary.chunk_count += 1;
                    summary.audio_samples += chunk_samples;
                    summary.first_frame_elapsed_ms
                        .get_or_insert(chunk.frame_elapsed_ms);
                    summary.first_decode_elapsed_ms
                        .get_or_insert(chunk.decode_elapsed_ms);
                    summary.first_detokenizer_wait_ms
                        .get_or_insert(chunk.detokenizer_wait_ms);
                    summary.max_frame_gap_ms = summary.max_frame_gap_ms.max(chunk.frame_gap_ms);
                    summary.max_decode_elapsed_ms =
                        summary.max_decode_elapsed_ms.max(chunk.decode_elapsed_ms);
                    summary.max_detokenizer_wait_ms = summary
                        .max_detokenizer_wait_ms
                        .max(chunk.detokenizer_wait_ms);
                    summary.max_queue_wait_ms =
                        summary.max_queue_wait_ms.max(queue_wait_ms);
                    info!(
                        chunk_index = chunk.chunk_index,
                        decode_elapsed_ms = chunk.decode_elapsed_ms,
                        detokenizer_wait_ms = chunk.detokenizer_wait_ms,
                        queue_wait_ms,
                        ws_send_ms = send_started_at.elapsed().as_millis() as u64,
                        output_queue_chunks = audio_queue.len(),
                        "streaming audio chunk sent"
                    );
                    pacer.release_one(Instant::now(), chunk_playback_duration(chunk_samples));
                }
            }
        }
    }

    if sent_audio_start {
        send_ws_json(socket, &ServerWsMessage::AssistantAudioEnd).await?;
    }

    Ok(summary)
}

async fn send_ws_json<T: Serialize>(socket: &mut WebSocket, payload: &T) -> ApiResult<()> {
    let text = serde_json::to_string(payload).map_err(|err| {
        ApiError::internal(format!("failed to serialize websocket message: {}", err))
    })?;
    socket
        .send(Message::Text(text.into()))
        .await
        .map_err(|err| ApiError::internal(err.to_string()))
}

#[cfg(test)]
#[derive(Clone)]
struct FakeDetokenizerRecorder {
    state: Arc<std::sync::Mutex<FakeDetokenizerState>>,
}

#[cfg(test)]
impl FakeDetokenizerRecorder {
    fn submitted_window_frames(&self) -> Vec<usize> {
        self.state
            .lock()
            .expect("fake detokenizer state should lock")
            .submitted_window_frames
            .clone()
    }

    fn complete_next(&self, completion: DetokenizeCompletion) {
        self.state
            .lock()
            .expect("fake detokenizer state should lock")
            .ready_results
            .push_back(Ok(completion));
    }

    fn set_blocking_waveform_len(&self, waveform_len: usize) {
        self.state
            .lock()
            .expect("fake detokenizer state should lock")
            .blocking_waveform_len = Some(waveform_len);
    }
}

#[cfg(test)]
struct FakeDetokenizerClient {
    state: Arc<std::sync::Mutex<FakeDetokenizerState>>,
}

#[cfg(test)]
impl FakeDetokenizerClient {
    fn new() -> Self {
        Self {
            state: Arc::new(std::sync::Mutex::new(FakeDetokenizerState::default())),
        }
    }

    fn recorder(&self) -> FakeDetokenizerRecorder {
        FakeDetokenizerRecorder {
            state: self.state.clone(),
        }
    }
}

#[cfg(test)]
impl DetokenizerClient for FakeDetokenizerClient {
    fn submit(&mut self, request: DetokenizeRequest) -> ApiResult<()> {
        let mut state = self.state.lock().expect("fake detokenizer state should lock");
        state.submitted_window_frames.push(request.codes.len());
        state.pending_requests.push_back((request.decode_index, request.codes.len()));
        Ok(())
    }

    fn try_complete(&mut self) -> ApiResult<Option<DetokenizeCompletion>> {
        let mut state = self.state.lock().expect("fake detokenizer state should lock");
        let Some(result) = state.ready_results.pop_front() else {
            return Ok(None);
        };
        state.pending_requests.pop_front();
        result.map(Some)
    }

    fn recv_complete(&mut self) -> ApiResult<DetokenizeCompletion> {
        let mut state = self.state.lock().expect("fake detokenizer state should lock");
        if let Some(result) = state.ready_results.pop_front() {
            state.pending_requests.pop_front();
            return result;
        }

        let (decode_index, window_frames) = state
            .pending_requests
            .pop_front()
            .ok_or_else(|| ApiError::internal("fake detokenizer had no pending request"))?;
        let waveform_len = state
            .blocking_waveform_len
            .take()
            .unwrap_or(window_frames * STREAM_SAMPLES_PER_FRAME);
        Ok(DetokenizeCompletion {
            decode_index,
            waveform: vec![0.0; waveform_len],
            decode_elapsed_ms: 0,
            detokenizer_wait_ms: 0,
            backend: "onnx",
        })
    }
}

#[cfg(test)]
#[derive(Default)]
struct FakeDetokenizerState {
    submitted_window_frames: Vec<usize>,
    pending_requests: VecDeque<(usize, usize)>,
    ready_results: VecDeque<DetokenizeResult>,
    blocking_waveform_len: Option<usize>,
}

fn validate_wav_content_type(headers: &HeaderMap) -> ApiResult<()> {
    let Some(content_type) = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return Ok(());
    };

    if content_type.starts_with("audio/wav")
        || content_type.starts_with("audio/x-wav")
        || content_type.starts_with("application/octet-stream")
    {
        Ok(())
    } else {
        Err(ApiError::unsupported(format!(
            "unsupported content type: {}",
            content_type
        )))
    }
}

fn pcm_s16le_bytes_to_f32(bytes: &[u8]) -> ApiResult<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        return Err(ApiError::bad_request(
            "pcm_s16le audio must have an even number of bytes",
        ));
    }

    let samples = bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
        .collect();
    Ok(samples)
}

fn encode_pcm_s16le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int16 = if clamped < 0.0 {
            (clamped * 32768.0).round() as i16
        } else {
            (clamped * 32767.0).round() as i16
        };
        bytes.extend_from_slice(&int16.to_le_bytes());
    }
    bytes
}

fn resolve_tts_system_prompt(system_prompt: Option<&str>, voice: Option<&str>) -> String {
    if let Some(prompt) = system_prompt.filter(|prompt| !prompt.trim().is_empty()) {
        return prompt.to_string();
    }

    match voice
        .unwrap_or("uk_female")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "uk_female" | "uk female" => "Perform TTS. Use the UK female voice.".to_string(),
        "uk_male" | "uk male" => "Perform TTS. Use the UK male voice.".to_string(),
        "us_female" | "us female" => "Perform TTS. Use the US female voice.".to_string(),
        "us_male" | "us male" => "Perform TTS. Use the US male voice.".to_string(),
        other => format!("Perform TTS. Use the {} voice.", other),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        background_worker_index, chunk_playback_duration, effective_worker_count,
        parse_device_preference, split_new_waveform_tail, DetokenizeCompletion,
        Device, DevicePreference, FakeDetokenizerClient, OutputQueueConfig,
        STREAM_DECODE_BATCH_FRAMES, STREAM_DECODE_CONTEXT_FRAMES, STREAM_OUTPUT_QUEUE_CHUNKS,
        StreamingAudioDecoder, AudioOutputPacer, spawn_detokenizer_worker,
    };
    use lfm2_audio::{LFM2Audio, Precision, TTSOptions};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    #[test]
    fn cpu_workers_follow_requested_count() {
        assert_eq!(effective_worker_count(Device::CPU, 4), 4);
    }

    #[test]
    fn gpu_workers_are_forced_to_one() {
        assert_eq!(effective_worker_count(Device::Cuda, 4), 1);
    }

    #[test]
    fn parse_device_preference_accepts_auto() {
        assert!(matches!(
            parse_device_preference("auto").expect("auto should parse"),
            DevicePreference::Auto
        ));
    }

    #[test]
    fn background_worker_prefers_a_different_worker_when_available() {
        assert_eq!(background_worker_index(0, 2), 1);
        assert_eq!(background_worker_index(1, 2), 0);
        assert_eq!(background_worker_index(0, 1), 0);
    }

    #[test]
    fn audio_output_pacer_waits_for_startup_queue_before_first_release() {
        let start = Instant::now();
        let mut pacer = AudioOutputPacer::new(OutputQueueConfig::new(4));

        pacer.enqueue(start);
        pacer.enqueue(start);
        pacer.enqueue(start);
        assert_eq!(pacer.next_wake_delay(start), None);

        pacer.enqueue(start);
        assert_eq!(pacer.next_wake_delay(start), Some(Duration::ZERO));

        pacer.release_one(start, chunk_playback_duration(1920));
        assert_eq!(
            pacer.next_wake_delay(start),
            Some(chunk_playback_duration(1920))
        );
    }

    #[test]
    fn audio_output_pacer_flushes_immediately_when_stream_ends_early() {
        let start = Instant::now();
        let mut pacer = AudioOutputPacer::new(OutputQueueConfig::new(4));

        pacer.enqueue(start);
        pacer.enqueue(start);
        assert_eq!(pacer.next_wake_delay(start), None);

        pacer.finish_input(start);
        assert_eq!(pacer.next_wake_delay(start), Some(Duration::ZERO));
    }

    #[test]
    fn audio_output_pacer_restarts_after_queue_drains() {
        let start = Instant::now();
        let mut pacer = AudioOutputPacer::new(OutputQueueConfig::new(4));

        for _ in 0..4 {
            pacer.enqueue(start);
        }
        pacer.release_one(start, chunk_playback_duration(1920));
        pacer.release_one(start, chunk_playback_duration(1920));
        pacer.release_one(start, chunk_playback_duration(1920));
        pacer.release_one(start, chunk_playback_duration(1920));
        assert_eq!(pacer.next_wake_delay(start), None);

        pacer.enqueue(start);
        assert_eq!(pacer.next_wake_delay(start), Some(Duration::ZERO));
    }

    #[test]
    fn default_stream_output_queue_starts_after_one_chunk() {
        assert_eq!(STREAM_OUTPUT_QUEUE_CHUNKS, 1);
    }

    #[test]
    fn default_stream_decode_batch_is_two_frames() {
        assert_eq!(STREAM_DECODE_BATCH_FRAMES, 2);
    }

    #[test]
    fn default_stream_decode_context_is_zero_frames() {
        assert_eq!(STREAM_DECODE_CONTEXT_FRAMES, 0);
    }

    #[test]
    fn waveform_tail_tracking_advances_after_first_emit() {
        let waveform = vec![0.0; 7_680];
        let (new_waveform, next_emitted_samples) =
            split_new_waveform_tail(waveform, 0).expect("first emit should succeed");

        assert_eq!(new_waveform.len(), 7_680);
        assert_eq!(next_emitted_samples, 7_680);
    }

    #[test]
    fn waveform_tail_tracking_only_returns_new_samples_after_previous_emit() {
        let waveform = vec![0.0; 15_360];
        let (new_waveform, next_emitted_samples) = split_new_waveform_tail(waveform, 7_680)
            .expect("subsequent emit should slice only the new tail");

        assert_eq!(new_waveform.len(), 7_680);
        assert_eq!(next_emitted_samples, 15_360);
    }

    #[test]
    fn streaming_audio_decoder_keeps_one_decode_in_flight_until_ready_chunk_is_drained() {
        let client = FakeDetokenizerClient::new();
        let recorder = client.recorder();
        let mut decoder = StreamingAudioDecoder::new_with_client(
            Box::new(client),
            super::StreamingDecodeConfig::new(2, 2),
            7,
        );

        assert!(decoder.push_frame([1; 8]).expect("push should succeed").is_none());
        assert!(decoder.push_frame([2; 8]).expect("push should succeed").is_none());
        assert_eq!(recorder.submitted_window_frames(), vec![2]);

        assert!(decoder.push_frame([3; 8]).expect("push should succeed").is_none());
        assert_eq!(
            recorder.submitted_window_frames(),
            vec![2],
            "decoder must not submit a second job while the first is still in flight"
        );

        recorder.complete_next(DetokenizeCompletion::from_waveform(
            1,
            vec![0.0; 3_840],
            11,
            4,
            "onnx",
        ));

        let ready = decoder.drain_ready().expect("drain should succeed");
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].emitted_samples, 3_840);
        assert_eq!(ready[0].detokenizer_wait_ms, 4);

        assert!(decoder.push_frame([4; 8]).expect("push should succeed").is_none());
        assert_eq!(recorder.submitted_window_frames(), vec![2, 4]);
    }

    #[test]
    fn streaming_audio_decoder_finish_drains_inflight_and_final_pending_frames() {
        let client = FakeDetokenizerClient::new();
        let recorder = client.recorder();
        let mut decoder = StreamingAudioDecoder::new_with_client(
            Box::new(client),
            super::StreamingDecodeConfig::new(2, 0),
            9,
        );

        assert!(decoder.push_frame([1; 8]).expect("push should succeed").is_none());
        assert!(decoder.push_frame([2; 8]).expect("push should succeed").is_none());
        assert!(decoder.push_frame([3; 8]).expect("push should succeed").is_none());
        assert_eq!(recorder.submitted_window_frames(), vec![2]);

        recorder.complete_next(DetokenizeCompletion::from_waveform(
            1,
            vec![0.0; 3_840],
            9,
            2,
            "onnx",
        ));
        recorder.set_blocking_waveform_len(1_920);

        let drained = decoder.finish().expect("finish should succeed");
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].emitted_samples, 3_840);
        assert_eq!(drained[1].emitted_samples, 1_920);
        assert_eq!(recorder.submitted_window_frames(), vec![2, 1]);
    }

    #[test]
    fn streaming_audio_decoder_limits_zero_context_backlog_to_one_batch() {
        let client = FakeDetokenizerClient::new();
        let recorder = client.recorder();
        let mut decoder = StreamingAudioDecoder::new_with_client(
            Box::new(client),
            super::StreamingDecodeConfig::new(2, 0),
            11,
        );

        assert!(decoder.push_frame([1; 8]).expect("push should succeed").is_none());
        assert!(decoder.push_frame([2; 8]).expect("push should succeed").is_none());
        assert!(decoder.push_frame([3; 8]).expect("push should succeed").is_none());

        recorder.complete_next(DetokenizeCompletion::from_waveform(
            1,
            vec![0.0; 3_840],
            6,
            1,
            "onnx",
        ));

        let ready = decoder.push_frame([4; 8]).expect("push should succeed");
        assert!(ready.is_some(), "pushing the next batch should surface the completed prior batch");

        assert_eq!(
            recorder.submitted_window_frames(),
            vec![2, 2],
            "zero-context streaming should only allow one batch of decode backlog"
        );
    }

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
    fn streaming_audio_decoder_onnx_produces_audio() {
        let model = load_model();
        let options = TTSOptions {
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

        assert!(
            debug.audio_codes.len() >= 8,
            "need enough generated frames to exercise streaming batches"
        );
        let audio_codes: Vec<[u16; 8]> = debug.audio_codes.iter().take(8).copied().collect();

        let detokenizer = spawn_detokenizer_worker(
            get_model_path().expect("model path should exist"),
            Precision::Q4,
            Device::CPU,
            99,
        )
        .expect("dedicated detokenizer worker should load");
        let mut decoder = StreamingAudioDecoder::new(
            detokenizer,
            super::StreamingDecodeConfig::new(4, 16),
            0,
        )
        .expect("ONNX streaming decoder should load");
        let mut streamed_bytes = Vec::new();
        for frame in &audio_codes {
            decoder.push_frame(*frame).expect("push_frame should succeed");
            for chunk in decoder.drain_ready().expect("drain_ready should succeed") {
                streamed_bytes.extend_from_slice(&chunk.bytes);
            }
        }

        for chunk in decoder.finish().expect("finish should succeed") {
            streamed_bytes.extend_from_slice(&chunk.bytes);
        }

        assert!(
            !streamed_bytes.is_empty(),
            "ONNX streaming decoder should produce audio"
        );
    }
}
