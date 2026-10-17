import type { Bundle, Dir, Filter, ReplayEvent } from "../types";

export interface PEvent extends ReplayEvent {
  /** bar index the event appears on (the replay clock position) */
  at: number;
  /** bar index from which each outcome may be shown (never before its window closes) */
  revealAt: { h60?: number; eod?: number; touch?: number };
  /** meaning-based directions of the alerts on this line (drives colour + filters) */
  dirs: Dir[];
}

export interface Prepared {
  bundle: Bundle;
  labels: string[];
  events: PEvent[];
  /** events that are alerts (no per-minute status lines) */
  alerts: PEvent[];
  last: number;
}

export function prepare(bundle: Bundle): Prepared {
  const labels = bundle.bars.map((b) => b[0]);
  const idx = new Map(labels.map((l, i) => [l, i]));
  const last = labels.length - 1;
  const pos = (label: string | undefined) =>
    label === undefined ? undefined : (idx.get(label) ?? last);

  const events: PEvent[] = bundle.events.map((e) => {
    const at = idx.get(e.bar) ?? 0;
    const o = e.outcome;
    const dirs: Dir[] =
      e.kind === "touch"
        ? [e.dir ?? "neutral"]
        : (e.alerts ?? []).map((a) => a.dir);
    return {
      ...e,
      at,
      dirs,
      revealAt: {
        h60: o?.h60 ? pos(o.h60.reveal) : undefined,
        eod: o?.eod ? pos(o.eod.reveal) : undefined,
        touch: o?.touch ? pos(o.touch.reveal) : undefined,
      },
    };
  });
  return { bundle, labels, events, alerts: events.filter((e) => e.kind !== "status"), last };
}

export function matchesFilter(e: PEvent, filter: Filter): boolean {
  if (e.kind === "status") return false;
  switch (filter) {
    case "all":
      return true;
    case "touch":
      return e.kind === "touch";
    default:
      return e.kind !== "touch" && e.dirs.includes(filter);
  }
}

// ---------------------------------------------------------------- formatting
const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];

export function dateKo(date: string, withWeekday = true): string {
  const [y, m, d] = date.split("-").map(Number);
  const wd = WEEKDAYS[new Date(Date.UTC(y, m - 1, d)).getUTCDay()];
  return withWeekday ? `${m}월 ${d}일 (${wd})` : `${m}월 ${d}일`;
}

export function fmtPrice(v: number): string {
  return Math.round(v).toLocaleString("ko-KR");
}

export function fmtPct(v: number | null | undefined, digits = 1, sign = true): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const p = v * 100;
  const s = p.toFixed(digits);
  return sign && p > 0 ? `+${s}%` : `${s}%`.replace("-", "−");
}

export const DIR_ICON: Record<Dir, string> = { up: "↑", down: "↓", neutral: "~" };
export const TOUCH_ICON = "✦";

export const BAND_KO: Record<string, string> = {
  HIGH: "높음",
  MED: "보통",
  LOW: "낮음",
  NA: "없음",
  "": "없음",
};

export const SPEEDS = [1, 4, 16] as const;
export type Speed = (typeof SPEEDS)[number];
