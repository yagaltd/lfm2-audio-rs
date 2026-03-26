# Autoresearch Ideas

- If pacing and flush-size tuning stall, split generation and detokenization into separate producer/consumer stages so audio decode no longer blocks frame generation callbacks.
- If CPU decode remains the bottleneck after correctness fixes, test a dedicated detokenizer session per streaming request or a tiny detokenizer pool to reduce contention.
- If first-playable latency stays high after pacing improvements, experiment with text/audio interleaving ratios (`interleaved_n_text`, `interleaved_n_audio`) on holdout prompts and reject quality regressions.
- If websocket timing is acceptable but browser playback still sounds rough, add a client-side millisecond-based startup target in the worklet path rather than relying on chunk count semantics.
- If all simple paths are exhausted, compare a smaller context window or tail-only decode strategy against subjective continuity checks before keeping it.
