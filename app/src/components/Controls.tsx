import { useMemo } from "react";
import { matchesFilter } from "../lib/prepare";
import { useReplay } from "../store";

const TICK_TONE = { up: "bg-up", down: "bg-down", neutral: "bg-neutral" } as const;

export default function Controls() {
  const data = useReplay((s) => s.data)!;
  const idx = useReplay((s) => s.idx);
  const playing = useReplay((s) => s.playing);
  const speed = useReplay((s) => s.speed);
  const filter = useReplay((s) => s.filter);
  const togglePlay = useReplay((s) => s.togglePlay);
  const cycleSpeed = useReplay((s) => s.cycleSpeed);
  const setIdx = useReplay((s) => s.setIdx);
  const restart = useReplay((s) => s.restart);
  const nextAlert = useReplay((s) => s.nextAlert);

  // one tick per alert minute on the scrubber, coloured by what the alert means
  const ticks = useMemo(() => {
    const seen = new Map<number, string>();
    for (const e of data.alerts) {
      if (!matchesFilter(e, filter) || seen.has(e.at)) continue;
      seen.set(e.at, e.kind === "touch" ? "bg-touch" : TICK_TONE[e.dirs[0] ?? "neutral"]);
    }
    return [...seen.entries()];
  }, [data, filter]);

  const last = data.last;
  const pos = (i: number) => `calc(11px + (100% - 22px) * ${i / last})`;
  const cur = idx >= 0 ? data.labels[idx] : "09:00";

  return (
    <div
      className="shrink-0 border-t border-line bg-surface px-4"
      style={{ paddingBottom: "max(env(safe-area-inset-bottom), 10px)" }}
    >
      <div className="relative">
        <div className="pointer-events-none absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-sunk">
          <div
            className="h-full rounded-full bg-ink/25"
            style={{ width: idx < 0 ? 0 : pos(idx) }}
          />
        </div>
        {ticks.map(([at, tone]) => (
          <span
            key={at}
            className={`pointer-events-none absolute top-1/2 h-3 w-[2px] -translate-x-1/2 -translate-y-1/2 rounded-full ${tone} ${
              at > idx ? "opacity-35" : ""
            }`}
            style={{ left: pos(at) }}
          />
        ))}
        <input
          type="range"
          className="scrub relative"
          min={0}
          max={last}
          step={1}
          value={Math.max(idx, 0)}
          onChange={(ev) => setIdx(Number(ev.target.value))}
          aria-label="재생 위치"
          aria-valuetext={cur}
        />
      </div>
      <div className="-mt-1.5 flex justify-between text-[11px] text-muted tabular">
        <span>09:00</span>
        <span className="font-semibold text-ink">{cur}</span>
        <span>15:30</span>
      </div>

      <div className="mt-1 flex items-center justify-between gap-2">
        <button
          onClick={restart}
          aria-label="처음부터"
          className="flex h-12 w-12 items-center justify-center rounded-full text-muted active:bg-sunk"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 12a8 8 0 1 0 2.4-5.7M4 4v4.5h4.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
        <button
          onClick={cycleSpeed}
          aria-label={`재생 속도 ${speed}배, 눌러서 바꾸기`}
          className="tabular h-12 min-w-14 rounded-full bg-sunk px-3 text-[15px] font-bold active:scale-95"
        >
          {speed}×
        </button>
        <button
          onClick={togglePlay}
          aria-label={playing ? "일시정지" : "재생"}
          className="flex h-14 w-14 items-center justify-center rounded-full bg-ink text-surface shadow-md transition active:scale-95"
        >
          {playing ? (
            <svg width="26" height="26" viewBox="0 0 24 24" aria-hidden="true">
              <rect x="6" y="5" width="4" height="14" rx="1.2" fill="currentColor" />
              <rect x="14" y="5" width="4" height="14" rx="1.2" fill="currentColor" />
            </svg>
          ) : (
            <svg width="28" height="28" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.4-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" fill="currentColor" />
            </svg>
          )}
        </button>
        <button
          onClick={nextAlert}
          className="flex h-12 items-center gap-1 rounded-full bg-sunk pl-3.5 pr-3 text-[14px] font-semibold active:scale-95"
        >
          다음 알림
          <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6 6l7 6-7 6zM16 6v12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
    </div>
  );
}
