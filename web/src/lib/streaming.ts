import type { ChatMessage } from "../api/types";

export function streamMessageId(runId: string): string {
  return `stream-${runId}`;
}

export function applyAnswerDelta(
  messages: ChatMessage[],
  runId: string,
  delta: string,
): ChatMessage[] {
  if (!runId || !delta) return messages;
  const id = streamMessageId(runId);
  const last = messages[messages.length - 1];
  if (last?.role === "assistant" && last.id === id) {
    return [
      ...messages.slice(0, -1),
      { ...last, content: `${last.content}${delta}` },
    ];
  }
  return [...messages, { id, role: "assistant", content: delta }];
}

export function applyAnswerDone(
  messages: ChatMessage[],
  runId: string,
  answer: string,
  extra: Partial<ChatMessage> = {},
): ChatMessage[] {
  const id = runId ? streamMessageId(runId) : "";
  const last = messages[messages.length - 1];
  const next: ChatMessage = {
    id: last?.role === "assistant" && last.id === id ? id : extra.id || id || "assistant",
    role: "assistant",
    content: answer,
    ...extra,
  };
  if (last?.role === "assistant" && last.id === id) {
    return [...messages.slice(0, -1), next];
  }
  return [...messages, next];
}

export function markStreamError(
  messages: ChatMessage[],
  runId: string,
): ChatMessage[] {
  const id = streamMessageId(runId);
  const last = messages[messages.length - 1];
  if (last?.role === "assistant" && last.id === id) {
    return [...messages.slice(0, -1), { ...last, error: true }];
  }
  return messages;
}
