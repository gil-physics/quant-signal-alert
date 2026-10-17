import { useState } from "react";
import type { IndexEntry, ReplayIndex } from "../types";
import { dateKo, fmtPct } from "../lib/prepare";
import { openDate } from "../lib/nav";
import Spark from "./Spark";
import AboutSheet from "./AboutSheet";

export default function Home({
  index,
  error,
  onRetry,
}: {
  index: ReplayIndex | null;
  error: boolean;
  onRetry: () => void;
}) {
  const [about, setAbout] = useState(false);
  return (
    <div className="min-h-full bg-bg">
      <div
        className="mx-auto max-w-3xl px-4"
        style={{
          paddingTop: "max(env(safe-area-inset-top), 28px)",
          paddingBottom: "max(env(safe-area-inset-bottom), 28px)",
        }}
      >
        <header className="mb-6">
          <p className="text-[13px] font-medium text-muted">
            {index ? `${index.name} ${index.symbol}` : " "}
          </p>
          <h1 className="mt-1 text-[28px] leading-tight font-bold tracking-tight text-balance">
            quant 리플레이
          </h1>
          <p className="mt-2 text-[15px] leading-relaxed text-muted text-pretty">
            지나간 하루 장을 1분씩 다시 재생합니다. 날짜를 누르면 그날 알림이 터미널 로그처럼
            쌓이는 모습을 볼 수 있어요.
          </p>
        </header>

        {error && (
          <div className="rounded-2xl border border-line bg-surface p-5 text-[15px]">
            <p>날짜 목록을 불러오지 못했어요. 연결을 확인하고 다시 시도해 주세요.</p>
            <button
              onClick={onRetry}
              className="mt-3 h-11 rounded-xl bg-ink px-4 font-semibold text-surface"
            >
              다시 시도
            </button>
          </div>
        )}

        <ul className="grid gap-3 sm:grid-cols-2">
          {index
            ? index.dates.map((d) => <DateCard key={d.date} entry={d} />)
            : !error && [0, 1, 2, 3].map((i) => <SkeletonCard key={i} />)}
        </ul>

        <footer className="mt-8 space-y-2 text-[13px] leading-relaxed text-muted">
          <p>
            모든 날짜는 모델이 학습하지 않은 기간입니다. 화면의 알림과 점수는 실제 알림 프로그램을
            그날 그대로 다시 돌린 결과예요.
          </p>
          <p>교육용 리플레이입니다. 주문을 내지 않으며 투자 권유가 아닙니다.</p>
          <button
            onClick={() => setAbout(true)}
            className="-ml-2 h-11 rounded-lg px-2 font-semibold text-ink underline underline-offset-4"
          >
            이 앱이 보여 주는 것
          </button>
        </footer>
      </div>
      <AboutSheet open={about} onClose={() => setAbout(false)} />
    </div>
  );
}

function DateCard({ entry }: { entry: IndexEntry }) {
  const chg = entry.change_pct ?? 0;
  const tone = chg > 0 ? "text-up" : chg < 0 ? "text-down" : "text-muted";
  const total = entry.counts.long + entry.counts.down + entry.counts.touch;
  return (
    <li>
      <button
        onClick={() => openDate(entry.date)}
        className="group w-full rounded-2xl border border-line bg-surface p-4 text-left transition active:scale-[0.99]"
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[17px] font-bold">{dateKo(entry.date)}</p>
            <p className="mt-0.5 truncate text-[14px] text-muted">{entry.label || " "}</p>
          </div>
          <div className="text-right">
            <p className={`tabular text-[17px] font-bold ${tone}`}>{fmtPct(entry.change_pct)}</p>
            <p className="text-[12px] text-muted">전일 대비</p>
          </div>
        </div>
        <Spark values={entry.spark} prev={entry.prev_close} className={`mt-3 h-12 w-full ${tone}`} />
        <div className="mt-3 flex items-center justify-between text-[13px] text-muted">
          <span className="tabular">
            알림 {total}개 · 롱 {entry.counts.long} · 하방 {entry.counts.down} · 돌파{" "}
            {entry.counts.touch}
          </span>
          <span className="flex items-center gap-1 font-semibold text-ink">
            재생
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
              <path d="M5 3.5v9l7-4.5z" fill="currentColor" />
            </svg>
          </span>
        </div>
      </button>
    </li>
  );
}

function SkeletonCard() {
  return (
    <li className="h-[150px] animate-pulse rounded-2xl border border-line bg-surface p-4">
      <div className="h-5 w-28 rounded bg-sunk" />
      <div className="mt-2 h-4 w-20 rounded bg-sunk" />
      <div className="mt-4 h-12 rounded bg-sunk" />
    </li>
  );
}
