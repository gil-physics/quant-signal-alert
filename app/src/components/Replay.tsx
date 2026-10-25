import { useEffect, useMemo, useState } from "react";
import type { Filter } from "../types";
import { dateKo, fmtPct, fmtPrice, matchesFilter } from "../lib/prepare";
import { useReplay } from "../store";
import Chart from "./Chart";
import LogPanel from "./LogPanel";
import Controls from "./Controls";
import DetailSheet from "./DetailSheet";
import AboutSheet from "./AboutSheet";

const FILTERS: { key: Filter; icon: string; label: string; tone: string; title: string }[] = [
  { key: "all", icon: "", label: "전체", tone: "", title: "모든 알림" },
  { key: "up", icon: "↑", label: "상방", tone: "text-up", title: "상방 쪽 알림" },
  { key: "down", icon: "↓", label: "하방", tone: "text-down", title: "하방 쪽 알림" },
  { key: "neutral", icon: "~", label: "약화", tone: "text-neutral", title: "추세 약화·변동성 확대" },
  { key: "touch", icon: "✦", label: "돌파", tone: "text-touch", title: "이평 돌파" },
];

export default function Replay({
  ready,
  error,
  onRetry,
  onBack,
  label,
  date,
  name,
  datePickerOpen,
  onChooseDate,
}: {
  ready: boolean;
  error: string | null;
  onRetry?: () => void;
  onBack: () => void;
  label: string;
  date: string;
  name: string;
  datePickerOpen: boolean;
  onChooseDate?: () => void;
}) {
  const [about, setAbout] = useState(false);
  const data = useReplay((s) => s.data);
  const idx = useReplay((s) => s.idx);
  const playing = useReplay((s) => s.playing);
  const speed = useReplay((s) => s.speed);
  const filter = useReplay((s) => s.filter);
  const selected = useReplay((s) => s.selected);
  const step = useReplay((s) => s.step);

  // the replay clock: one bar per tick, 1x = one minute per second
  useEffect(() => {
    if (!playing || !ready || datePickerOpen) return;
    const id = window.setInterval(step, 1000 / speed);
    return () => window.clearInterval(id);
  }, [playing, speed, ready, datePickerOpen, step]);

  useKeyboard(ready && !datePickerOpen && !about);

  if (!ready || !data) {
    return (
      <Shell onBack={onBack} onAbout={() => setAbout(true)} onChooseDate={onChooseDate}
        title={`${name ? `${name} · ` : ""}${dateKo(date)}`} sub={label}>
        <div className="flex flex-1 flex-col items-center justify-center gap-4 bg-term p-6 text-center text-term-muted">
          {error ? (
            <>
              <p className="text-[15px]">{error}</p>
              {onRetry && <button onClick={onRetry} className="h-11 rounded-xl bg-term-ink px-5 font-semibold text-term">
                다시 시도
              </button>}
            </>
          ) : (
            <p className="text-[15px]">그날 장을 불러오는 중…</p>
          )}
        </div>
        <AboutSheet open={about} onClose={() => setAbout(false)} />
      </Shell>
    );
  }

  const meta = data.bundle.meta;
  const bar = idx >= 0 ? data.bundle.bars[idx] : null;
  const price = bar ? bar[4] : meta.prev_close ?? meta.open;
  const chg = meta.prev_close && bar ? bar[4] / meta.prev_close - 1 : null;
  const tone = chg === null ? "text-muted" : chg > 0 ? "text-up" : chg < 0 ? "text-down" : "text-muted";

  return (
    <Shell
      onBack={onBack}
      onAbout={() => setAbout(true)}
      onChooseDate={onChooseDate}
      title={`${meta.name} · ${dateKo(meta.date)}`}
      sub={label}
    >
      {!!meta.touch_grade_misses && (
        <p className="bg-sunk px-4 py-2 text-[12px] leading-relaxed text-muted">
          일부 돌파 알림은 등급 데이터가 없습니다. 신호는 그대로 표시합니다.
        </p>
      )}
      <div className="flex shrink-0 items-end justify-between gap-3 bg-surface px-4 pb-2">
        <div>
          <p className="tabular font-mono text-[30px] leading-none font-semibold tracking-tight" aria-live="off">
            {bar ? bar[0] : "09:00"}
          </p>
          <p className="mt-1 text-[12px] text-muted">{bar ? (idx === data.last ? "장 마감" : "재생 중인 시각") : "개장 전"}</p>
        </div>
        <div className="text-right">
          <p className="tabular text-[20px] font-bold leading-none">{fmtPrice(price)}</p>
          <p className={`tabular mt-1 text-[13px] font-semibold ${tone}`}>
            {chg === null ? "전일 종가" : `전일 대비 ${fmtPct(chg, 2)}`}
          </p>
        </div>
      </div>

      <div className="h-[25dvh] min-h-[150px] max-h-[320px] shrink-0 bg-surface px-1">
        <Chart data={data} idx={idx} filter={filter} selected={selected} />
      </div>

      <FilterBar />

      <LogPanel onEnd={idx === data.last ? <EndCard onBack={onChooseDate ?? onBack} /> : null} />

      <Controls />
      <DetailSheet />
      <AboutSheet open={about} onClose={() => setAbout(false)} />
    </Shell>
  );
}

function Shell({
  children,
  onBack,
  onAbout,
  onChooseDate,
  title,
  sub,
}: {
  children: React.ReactNode;
  onBack: () => void;
  onAbout: () => void;
  onChooseDate?: () => void;
  title: string;
  sub: string;
}) {
  return (
    <div className="mx-auto flex h-dvh max-w-2xl flex-col overflow-hidden bg-surface sm:border-x sm:border-line">
      <header
        className="flex shrink-0 items-center gap-1 bg-surface px-1"
        style={{ paddingTop: "env(safe-area-inset-top)" }}
      >
        <button onClick={onBack} aria-label="날짜 목록으로" className="flex h-12 w-12 items-center justify-center">
          <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M15 5l-7 7 7 7" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
        <button
          onClick={onChooseDate}
          disabled={!onChooseDate}
          aria-label={`${title} · 날짜 선택`}
          className="min-w-0 flex-1 rounded-lg py-2 text-left"
        >
          <p className="flex items-center gap-1 text-[15px] font-bold">
            <span className="truncate">{title || " "}</span>
            <span aria-hidden="true" className="text-muted">⌄</span>
          </p>
          <p className="truncate text-[12px] text-muted">{sub || "날짜를 눌러 다른 날 보기"}</p>
        </button>
        <button onClick={onAbout} aria-label="이 앱이 보여 주는 것" className="flex h-12 w-12 items-center justify-center text-muted">
          <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="1.8" />
            <path d="M12 11v6M12 7.5v.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </button>
      </header>
      {children}
    </div>
  );
}

function FilterBar() {
  const data = useReplay((s) => s.data)!;
  const idx = useReplay((s) => s.idx);
  const filter = useReplay((s) => s.filter);
  const setFilter = useReplay((s) => s.setFilter);
  const counts = useMemo(() => {
    const c: Record<Filter, number> = { all: 0, up: 0, down: 0, neutral: 0, touch: 0 };
    for (const e of data.alerts) {
      if (e.at > idx) break;
      for (const f of FILTERS) if (matchesFilter(e, f.key)) c[f.key] += 1;
    }
    return c;
  }, [data, idx]);
  return (
    <div className="grid shrink-0 grid-cols-5 gap-1.5 bg-surface px-3 py-2" role="tablist" aria-label="알림 종류">
      {FILTERS.map((f) => {
        const on = filter === f.key;
        return (
          <button
            key={f.key}
            role="tab"
            aria-selected={on}
            title={f.title}
            onClick={() => setFilter(f.key)}
            className={`flex h-10 items-center justify-center gap-1 rounded-xl transition ${
              on ? "bg-ink text-surface" : "bg-sunk text-ink"
            }`}
          >
            <span className="text-[13px] font-semibold whitespace-nowrap">
              {f.icon && <span className={on ? "" : f.tone}>{f.icon} </span>}
              {f.label}
            </span>
            <span className={`tabular text-[11px] ${on ? "opacity-70" : "text-muted"}`}>{counts[f.key]}</span>
          </button>
        );
      })}
    </div>
  );
}

function EndCard({ onBack }: { onBack: () => void }) {
  const data = useReplay((s) => s.data)!;
  const restart = useReplay((s) => s.restart);
  const c = data.bundle.meta.counts;
  return (
    <div className="m-3 rounded-2xl bg-term-hover p-4">
      <p className="text-[16px] font-bold text-term-ink">15:30 장 마감</p>
      <p className="tabular mt-1 text-[14px] text-term-muted">
        오늘 알림 롱 {c.long} · 하방 {c.down} · 이평 돌파 {c.touch}
      </p>
      <p className="mt-2 text-[13px] leading-relaxed text-term-muted">
        이제 모든 알림의 결과가 열렸습니다. 지나간 줄을 눌러 그 뒤에 어떻게 되었는지 확인해 보십시오.
      </p>
      <div className="mt-3 flex gap-2">
        <button onClick={restart} className="h-11 flex-1 rounded-xl bg-term-ink font-semibold text-term">
          처음부터 다시
        </button>
        <button onClick={onBack} className="h-11 flex-1 rounded-xl font-semibold text-term-ink ring-1 ring-term-line">
          다른 날짜
        </button>
      </div>
    </div>
  );
}

function useKeyboard(ready: boolean) {
  useEffect(() => {
    if (!ready) return;
    const onKey = (ev: KeyboardEvent) => {
      const s = useReplay.getState();
      if (s.selected !== null || ev.metaKey || ev.ctrlKey || ev.altKey) return;
      const tag = (ev.target as HTMLElement)?.tagName;
      if (tag === "BUTTON" && ev.key === " ") return;
      if (tag === "INPUT" && ev.key !== " ") return;
      if (ev.key === " ") {
        ev.preventDefault();
        s.togglePlay();
      } else if (ev.key === "ArrowRight") s.setIdx(s.idx + 1);
      else if (ev.key === "ArrowLeft") s.setIdx(s.idx - 1);
      else if (ev.key === "n" || ev.key === "N") s.nextAlert();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ready]);
}
