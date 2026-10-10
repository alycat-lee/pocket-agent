/** 缓存命中率展示：不谎报 100%（仅真满命中）。 */
export function formatCacheHitRate(rate: number | null | undefined): string {
  if (rate == null || Number.isNaN(rate)) return "—";
  if (rate >= 1) return "100%";
  if (rate <= 0) return "0%";
  // floor 到 1 位小数，避免 0.995 → 100%
  const pct = Math.floor(rate * 1000) / 10;
  return `${pct}%`;
}

export function formatMs(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "不可用";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function formatChars(n: number): string {
  if (n >= 1000) return `≈${(n / 1000).toFixed(1)}k`;
  return `≈${n}`;
}

export function formatElapsed(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}:${String(r).padStart(2, "0")}` : `${r}s`;
}
