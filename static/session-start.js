export function buildSessionStartMessage(systemPrompt) {
  return {
    type: "session.start",
    system_prompt: systemPrompt,
  };
}
