# Autoresearch Ideas

- If revisiting async decode, preserve the current 2-frame chunking semantics instead of switching to ordered single-frame output. The single-frame async prototype improved first audio but produced multi-second starvation gaps even after the final-drain bug was fixed.
- If CPU decode remains the bottleneck after correctness fixes, test a dedicated detokenizer session per streaming request or a tiny detokenizer pool to reduce contention.
- If websocket timing is acceptable but browser playback still sounds rough, add a client-side millisecond-based startup target in the worklet path rather than relying on chunk count semantics.
