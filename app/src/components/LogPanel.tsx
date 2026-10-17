import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Alert, Dir } from "../types";
import { BAND_KO, DIR_ICON, TOUCH_ICON, matchesFilter, type PEvent } from "../lib/prepare";
import { useReplay } from "../store";

const DIR_TEXT: Record<Dir, string> = {
  up: "text-up",
  down: "text-down",
  neutral: "text-neutral",
};

const HINT_KEY = "quant-replay-hint-dismissed";

function readHint() {
  try {
    return localStorage.getItem(HINT_KEY) === "1";
  } catch {
    return false;
  }
}

export default function LogPanel({ onEnd }: { onEnd: React.ReactNode }) {
  const data = useReplay((s) => s.data)!;
  const idx = useReplay((s) => s.idx);
  const filter = useReplay((s) => s.filter);
  const raw = useReplay((s) => s.raw);
  const showStatus = useReplay((s) => s.showStatus);
  const setRaw = useReplay((s) => s.setRaw);
  const setShowStatus = useReplay((s) => s.setShowStatus);
  const select = useReplay((s) => s.select);

  const lines = useMemo(() => {
    const out: PEvent[] = [];
    for (const e of data.events) {
      if (e.at > idx) break;
      if (e.kind === "status" ? showStatus : matchesFilter(e, filter)) out.push(e);
    }
    return out;
  }, [data, idx, filter, showStatus]);

  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const [behind, setBehind] = useState(false);
  const [hintDismissed, setHintDismissed] = useState(readHint);

  // terminal behaviour: follow new lines unless the reader scrolled up
  const prevLen = useRef(0);
  const ended = !!onEnd;
  useLayoutEffect(() => {
    const s = scroller.current;
    if (!s) return;
    const grew = lines.length > prevLen.current;
    prevLen.current = lines.length;
    if (stick.current) s.scrollTop = s.scrollHeight;
    else if (grew) setBehind(true);
  }, [lines.length, ended]);

  // switching views or filters re-flows every line: jump back to the newest one
  useLayoutEffect(() => {
    const s = scroller.current;
    if (!s) return;
    stick.current = true;
    setBehind(false);
    s.scrollTop = s.scrollHeight;
  }, [raw, filter, showStatus]);

  useEffect(() => {
    if (idx <= 0) {
      stick.current = true;
      setBehind(false);
    }
  }, [idx]);

  const onScroll = () => {
    const s = scroller.current;
    if (!s) return;
    const atBottom = s.scrollHeight - s.scrollTop - s.clientHeight < 32;
    stick.current = atBottom;
    if (atBottom) setBehind(false);
  };

  const jumpToLatest = () => {
    const s = scroller.current;
    if (!s) return;
    s.scrollTo({ top: s.scrollHeight, behavior: "smooth" });
    stick.current = true;
    setBehind(false);
  };

  const dismissHint = () => {
    setHintDismissed(true);
    try {
      localStorage.setItem(HINT_KEY, "1");
    } catch {
      /* private mode: the hint just comes back next time */
    }
  };

  const alertCount = lines.filter((l) => l.kind !== "status").length;

  return (
    <section className="relative flex min-h-0 flex-1 flex-col bg-term text-term-ink" aria-label="알림 로그">
      <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-term-line px-3">
        <p className="min-w-0 truncate text-[13px] text-term-muted">
          <span className="font-semibold text-term-ink">터미널 로그</span>
          <span className="tabular"> · 알림 {alertCount}</span>
        </p>
        <div className="flex shrink-0 items-center gap-1.5">
          <button
            onClick={() => setShowStatus(!showStatus)}
            aria-pressed={showStatus}
            className={`h-8 rounded-lg px-2.5 text-[12px] font-semibold ${
              showStatus ? "bg-term-ink text-term" : "text-term-muted ring-1 ring-term-line"
            }`}
          >
            매분 상태
          </button>
          <div className="flex h-8 rounded-lg p-0.5 ring-1 ring-term-line" role="group" aria-label="보기">
            {[
              { v: false, t: "쉽게" },
              { v: true, t: "원본" },
            ].map((o) => (
              <button
                key={o.t}
                onClick={() => setRaw(o.v)}
                aria-pressed={raw === o.v}
                className={`rounded-md px-2.5 text-[12px] font-semibold ${
                  raw === o.v ? "bg-term-ink text-term" : "text-term-muted"
                }`}
              >
                {o.t}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div
        ref={scroller}
        onScroll={onScroll}
        className={`min-h-0 flex-1 overflow-y-auto overscroll-contain ${raw ? "overflow-x-auto" : ""}`}
      >
        {!hintDismissed && (
          <div className="m-3 flex items-start gap-3 rounded-xl bg-term-hover p-3 text-[13px] leading-relaxed">
            <p className="min-w-0 flex-1">
              알림이 1분마다 <b className="text-term-ink">아래에 쌓여요.</b> 줄을 누르면 무슨 뜻인지와
              <b className="text-term-ink"> 그 뒤에 실제로 어떻게 됐는지</b> 볼 수 있어요.
            </p>
            <button onClick={dismissHint} aria-label="안내 닫기" className="-m-2 h-10 w-10 shrink-0 text-term-muted">
              ✕
            </button>
          </div>
        )}
        {idx < 0 && <p className="px-4 py-6 text-[14px] text-term-muted">09:00 장이 곧 열립니다…</p>}
        {raw ? (
          <div className="w-max min-w-full py-1 font-mono text-[12px] leading-6">
            {lines.map((e) => (
              <RawLine key={e.id} e={e} fresh={e.at === idx} onTap={select} />
            ))}
          </div>
        ) : (
          <ul className="pb-1">
            {lines.map((e, i) => {
              const hour = e.bar.slice(0, 2);
              const newHour = i === 0 || lines[i - 1].bar.slice(0, 2) !== hour;
              const row =
                e.kind === "status" ? (
                  <StatusLine key={e.id} e={e} fresh={e.at === idx} />
                ) : (
                  <EasyLine key={e.id} e={e} fresh={e.at === idx} onTap={select} />
                );
              return newHour ? [<HourMark key={`h${hour}`} hour={hour} />, row] : row;
            })}
          </ul>
        )}
        {onEnd}
        <div className="h-3" />
      </div>

      {behind && (
        <button
          onClick={jumpToLatest}
          className="absolute bottom-3 left-1/2 h-10 -translate-x-1/2 rounded-full bg-term-ink px-4 text-[13px] font-semibold text-term shadow-lg"
        >
          새 알림 ↓
        </button>
      )}
    </section>
  );
}

// ---------------------------------------------------------------- rows
const DIR_VAR: Record<Dir, string> = { up: "var(--up)", down: "var(--down)", neutral: "var(--neutral)" };
const TIME_TONE: Record<Dir | "touch", string> = {
  up: "bg-up/15 text-up",
  down: "bg-down/15 text-down",
  neutral: "bg-neutral/15 text-term-ink",
  touch: "bg-touch/15 text-touch",
};

/** the line's lead colour: the first alert that points somewhere, else neutral */
function leadTone(e: PEvent): Dir | "touch" {
  if (e.kind === "touch") return "touch";
  return e.dirs.find((d) => d !== "neutral") ?? "neutral";
}

/** left edge colour bar; split top/bottom when two alerts share the minute */
function stripe(e: PEvent): string {
  if (e.kind === "touch") return "var(--touch)";
  const cols = e.dirs.map((d) => DIR_VAR[d]);
  if (cols.length < 2) return cols[0] ?? DIR_VAR.neutral;
  return `linear-gradient(to bottom, ${cols[0]} 50%, ${cols[1]} 50%)`;
}

function HourMark({ hour }: { hour: string }) {
  return (
    <li className="sticky top-0 z-10 flex items-center gap-2 bg-term/95 px-3 pt-3 pb-1.5 backdrop-blur-sm" aria-hidden="true">
      <span className="font-mono text-[11.5px] font-semibold text-term-muted">{Number(hour)}시</span>
      <span className="h-px flex-1 bg-term-line" />
    </li>
  );
}

const EasyLine = memo(function EasyLine({
  e,
  fresh,
  onTap,
}: {
  e: PEvent;
  fresh: boolean;
  onTap: (id: number) => void;
}) {
  return (
    <li className={`relative border-b border-term-line last:border-b-0 ${fresh ? "line-in" : ""}`}>
      <span
        className="absolute inset-y-2 left-0 w-[3px] rounded-r-full"
        style={{ background: stripe(e) }}
        aria-hidden="true"
      />
      <button
        onClick={() => onTap(e.id)}
        className="flex w-full items-start gap-3 py-3 pr-3 pl-4 text-left active:bg-term-hover"
      >
        <span
          className={`tabular shrink-0 rounded-md px-1.5 py-0.5 font-mono text-[12.5px] font-semibold ${TIME_TONE[leadTone(e)]}`}
        >
          {e.bar}
        </span>
        <div className="min-w-0 flex-1 space-y-2">
          {e.kind === "touch" ? <TouchBody e={e} /> : e.alerts!.map((a, i) => <AlertBody key={i} a={a} />)}
        </div>
        <span className="pt-0.5 text-term-muted" aria-hidden="true">
          ›
        </span>
      </button>
    </li>
  );
});

function Icon({ dir, touch }: { dir: Dir; touch?: boolean }) {
  if (touch)
    return <span className="w-4 shrink-0 text-center font-bold text-touch">{TOUCH_ICON}</span>;
  return <span className={`w-4 shrink-0 text-center font-bold ${DIR_TEXT[dir]}`}>{DIR_ICON[dir]}</span>;
}

function riskChip(a: Alert) {
  const label = a.side === "long" ? "모델 위험" : "모델 반등위험";
  const tone =
    a.band === "HIGH"
      ? "text-risk-high ring-risk-high/50"
      : a.band === "LOW"
        ? "text-risk-low ring-risk-low/50"
        : "text-term-muted ring-term-line";
  return (
    <span className={`rounded-md px-1.5 py-px text-[11.5px] font-semibold ring-1 ${tone}`}>
      {label} {BAND_KO[a.band] ?? a.band}
    </span>
  );
}

function AlertBody({ a }: { a: Alert }) {
  const showGrade = a.grade === "A" || a.grade === "B" || a.grade === "C";
  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <Icon dir={a.dir} />
        <span className="text-[15px] font-semibold">{a.type_ko}</span>
        {showGrade && (
          <span className="rounded bg-term-hover px-1.5 font-mono text-[12px] text-term-ink">{a.grade}</span>
        )}
        {riskChip(a)}
      </div>
      {a.reasons.length > 0 && (
        <p className="mt-0.5 pl-6 text-[13px] text-term-muted">{a.reasons.join(" · ")}</p>
      )}
    </div>
  );
}

function TouchBody({ e }: { e: PEvent }) {
  const t = e.touch!;
  const up = t.direction > 0;
  const bandKo = { near: "약함", mid: "중간", far: "강함" }[t.band];
  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <Icon dir={up ? "up" : "down"} touch />
        <span className="text-[15px] font-semibold">{up ? "이평 상향 돌파" : "이평 하향 돌파"}</span>
        {t.grade ? (
          <span className="rounded-md bg-touch/15 px-1.5 py-px text-[11.5px] font-semibold text-touch">
            등급 {t.grade} · 과거 {Math.round((t.grade_rate ?? 0) * 100)}%
          </span>
        ) : (
          <span className="rounded-md px-1.5 py-px text-[11.5px] text-term-muted ring-1 ring-term-line">
            등급 없음
          </span>
        )}
        {t.whipsaw && (
          <span className="rounded-md px-1.5 py-px text-[11.5px] font-semibold text-touch ring-1 ring-touch/50">
            휩쏘
          </span>
        )}
      </div>
      <p className="mt-0.5 pl-6 text-[13px] text-term-muted">
        5분선이 20분선을 {up ? "위로" : "아래로"} 넘음 · 돌파 거리 {bandKo}
      </p>
    </div>
  );
}

const StatusLine = memo(function StatusLine({ e, fresh }: { e: PEvent; fresh: boolean }) {
  const l = e.status?.long;
  const s = e.status?.short;
  return (
    <li className={`flex gap-3 border-b border-term-line/60 py-1.5 pr-3 pl-4 text-[12px] text-term-muted ${fresh ? "line-in" : ""}`}>
      <span className="tabular shrink-0 px-1.5 font-mono">{e.bar}</span>
      <span className="min-w-0">
        {l ? `롱 위험 ${l.prob?.toFixed(2) ?? "–"} ${BAND_KO[l.band] ?? ""}` : "롱 준비 중"}
        {s ? ` · 하락 점수 ${s.dn ?? "–"} · 상승 점수 ${s.up ?? "–"}` : ""}
      </span>
    </li>
  );
});

const RawLine = memo(function RawLine({
  e,
  fresh,
  onTap,
}: {
  e: PEvent;
  fresh: boolean;
  onTap: (id: number) => void;
}) {
  // the original output, uncoloured except that per-minute status lines are dimmed
  const tone = e.kind === "status" ? "text-term-muted" : "text-term-ink";
  const body = <span className="whitespace-pre">{e.raw}</span>;
  if (e.kind === "status")
    return <div className={`px-3 ${tone} ${fresh ? "line-in" : ""}`}>{body}</div>;
  return (
    <button
      onClick={() => onTap(e.id)}
      className={`block w-full px-3 text-left active:bg-term-hover ${tone} ${fresh ? "line-in" : ""}`}
    >
      {body}
    </button>
  );
});
