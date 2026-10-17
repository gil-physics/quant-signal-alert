import { Suspense, lazy, useCallback, useEffect, useState } from "react";
import type { Bundle, ReplayIndex } from "./types";
import { prepare } from "./lib/prepare";
import { useReplay } from "./store";
import Home from "./components/Home";
import { dateFromHash, goHome } from "./lib/nav";

// the replay screen carries the chart library; load it only when a day is opened
const Replay = lazy(() => import("./components/Replay"));

export default function App() {
  const [index, setIndex] = useState<ReplayIndex | null>(null);
  const [indexError, setIndexError] = useState(false);
  const [date, setDate] = useState<string | null>(dateFromHash);
  const [dayError, setDayError] = useState(false);
  const load = useReplay((s) => s.load);
  const unload = useReplay((s) => s.unload);
  const data = useReplay((s) => s.data);

  const fetchIndex = useCallback(() => {
    setIndexError(false);
    fetch("/data/index.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((j: ReplayIndex) => setIndex(j))
      .catch(() => setIndexError(true));
  }, []);

  useEffect(fetchIndex, [fetchIndex]);

  useEffect(() => {
    const onHash = () => setDate(dateFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const fetchDay = useCallback(() => {
    if (!date || !index) return;
    const entry = index.dates.find((d) => d.date === date);
    if (!entry) {
      window.location.hash = "#/";
      return;
    }
    setDayError(false);
    let cancelled = false;
    fetch(`/${entry.file}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((b: Bundle) => {
        if (!cancelled) load(prepare(b));
      })
      .catch(() => !cancelled && setDayError(true));
    return () => {
      cancelled = true;
    };
  }, [date, index, load]);

  useEffect(() => {
    if (!date) {
      unload();
      return;
    }
    return fetchDay();
  }, [date, fetchDay, unload]);

  // warm the cache with the first day so the first tap starts instantly
  useEffect(() => {
    if (index?.dates[0]) fetch(`/${index.dates[0].file}`).catch(() => {});
  }, [index]);

  if (date) {
    const entry = index?.dates.find((d) => d.date === date);
    return (
      <Suspense fallback={<div className="h-dvh bg-term" />}>
        <Replay
          ready={!!data && data.bundle.meta.date === date}
          error={dayError}
          onRetry={fetchDay}
          label={entry?.label ?? ""}
          onBack={goHome}
        />
      </Suspense>
    );
  }
  return <Home index={index} error={indexError} onRetry={fetchIndex} />;
}
