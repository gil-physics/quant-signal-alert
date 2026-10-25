import { Suspense, lazy, useEffect, useState } from "react";
import type { Bundle, ReplayIndex } from "./types";
import { prepare } from "./lib/prepare";
import { useReplay } from "./store";
import Home from "./components/Home";
import DateSheet from "./components/DateSheet";
import { dateFromHash, goHome, openDate } from "./lib/nav";

// the replay screen carries the chart library; load it only when a day is opened
const Replay = lazy(() => import("./components/Replay"));

export default function App() {
  const [index, setIndex] = useState<ReplayIndex | null>(null);
  const [indexError, setIndexError] = useState(false);
  const [indexAttempt, setIndexAttempt] = useState(0);
  const [date, setDate] = useState<string | null>(dateFromHash);
  const [dayError, setDayError] = useState(false);
  const [dayAttempt, setDayAttempt] = useState(0);
  const [choosingDate, setChoosingDate] = useState(false);
  const load = useReplay((s) => s.load);
  const unload = useReplay((s) => s.unload);
  const data = useReplay((s) => s.data);
  const available = index?.availableDates ?? index?.dates ?? [];
  const entry = available.find((d) => d.date === date);
  const featured = index?.dates.find((d) => d.date === date);
  const missingDate = !!index && !!date && !entry;

  useEffect(() => {
    const controller = new AbortController();
    setIndexError(false);
    // A new key avoids the old service worker returning a six-day catalog.
    fetch("/data/index.json?v=2", { signal: controller.signal, cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((j: ReplayIndex) => setIndex(j))
      .catch(() => !controller.signal.aborted && setIndexError(true));
    return () => controller.abort();
  }, [indexAttempt]);

  useEffect(() => {
    const onHash = () => {
      setDate(dateFromHash());
      setChoosingDate(false);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    unload();
    setDayError(false);
    if (!date || !entry) return;
    const controller = new AbortController();
    fetch(`/${entry.file}`, { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((bundle: Bundle) => {
        if (controller.signal.aborted) return;
        if (bundle.meta.date !== date || bundle.meta.symbol !== index?.symbol) {
          throw new Error("Replay does not match the selected date/symbol");
        }
        load(prepare(bundle));
      })
      .catch(() => !controller.signal.aborted && setDayError(true));
    return () => controller.abort();
  }, [date, entry, index?.symbol, dayAttempt, load, unload]);

  const retry = () => {
    if (indexError) setIndexAttempt((n) => n + 1);
    else setDayAttempt((n) => n + 1);
  };

  return (
    <>
      {date ? (
        <Suspense fallback={<div className="h-dvh bg-term" />}>
          <Replay
            ready={!!data && data.bundle.meta.date === date}
            error={missingDate
              ? "이 날짜에는 준비된 리플레이가 없습니다. 상단에서 다른 날짜를 선택해 주십시오."
              : indexError || dayError
                ? "이 날짜를 불러오지 못했습니다. 연결을 확인한 뒤 다시 시도해 주십시오."
                : null}
            onRetry={missingDate ? undefined : retry}
            label={featured?.label ?? ""}
            date={date}
            name={index?.name ?? ""}
            datePickerOpen={choosingDate}
            onChooseDate={available.length ? () => setChoosingDate(true) : undefined}
            onBack={goHome}
          />
        </Suspense>
      ) : (
        <Home
          index={index}
          error={indexError}
          onRetry={() => setIndexAttempt((n) => n + 1)}
          onChooseDate={() => setChoosingDate(true)}
        />
      )}
      {choosingDate && available.length > 0 && (
        <DateSheet
          dates={available}
          selected={date}
          onClose={() => setChoosingDate(false)}
          onSelect={(nextDate) => {
            setChoosingDate(false);
            if (nextDate !== date) openDate(nextDate);
          }}
        />
      )}
    </>
  );
}
