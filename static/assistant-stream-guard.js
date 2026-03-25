export function acceptAssistantAudioChunk(lastChunkIndex, meta) {
  if (!meta || !Number.isInteger(meta.chunk_index)) {
    return {
      accept: false,
      nextChunkIndex: lastChunkIndex,
      reason: "missing-meta",
    };
  }

  if (lastChunkIndex == null) {
    return {
      accept: true,
      nextChunkIndex: meta.chunk_index,
      reason: null,
    };
  }

  if (meta.chunk_index === lastChunkIndex) {
    return {
      accept: false,
      nextChunkIndex: lastChunkIndex,
      reason: "duplicate",
    };
  }

  if (meta.chunk_index < lastChunkIndex) {
    return {
      accept: false,
      nextChunkIndex: lastChunkIndex,
      reason: "non-monotonic",
    };
  }

  return {
    accept: true,
    nextChunkIndex: meta.chunk_index,
    reason: null,
  };
}
