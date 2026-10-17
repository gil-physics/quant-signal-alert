import { useEffect, useMemo, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  LineSeries,
  LineStyle,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import type { Filter } from "../types";
import { matchesFilter, type Prepared } from "../lib/prepare";

function css(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function palette() {
  return {
    up: css("--up"),
    down: css("--down"),
    neutral: css("--neutral"),
    touch: css("--touch"),
    ink: css("--ink"),
    muted: css("--muted"),
    line: css("--line"),
  };
}

const WINDOW = 90; // bars in view

const pad = (n: number) => String(n).padStart(2, "0");
const hhmm = (t: Time) => {
  const d = new Date((t as number) * 1000);
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
};

export default function Chart({
  data,
  idx,
  filter,
  selected,
}: {
  data: Prepared;
  idx: number;
  filter: Filter;
  selected: number | null;
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const candles = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const ma5 = useRef<ISeriesApi<"Line"> | null>(null);
  const ma20 = useRef<ISeriesApi<"Line"> | null>(null);
  const markers = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const drawn = useRef(-1);

  // bar label -> chart time (UTC seconds so the axis shows the exchange clock as-is)
  const times = useMemo(() => {
    const [y, m, d] = data.bundle.meta.date.split("-").map(Number);
    return data.labels.map((l) => {
      const [h, mi] = l.split(":").map(Number);
      return (Date.UTC(y, m - 1, d, h, mi) / 1000) as UTCTimestamp;
    });
  }, [data]);

  useEffect(() => {
    if (!el.current) return;
    const p = palette();
    const c = createChart(el.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: p.muted,
        fontFamily: "'Pretendard Variable', system-ui, sans-serif",
        fontSize: 11,
        attributionLogo: true,
      },
      grid: { vertLines: { visible: false }, horzLines: { color: p.line, style: LineStyle.Dotted } },
      rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.12, bottom: 0.08 } },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
        shiftVisibleRangeOnNewBar: false,
        tickMarkFormatter: (t: Time) => hhmm(t),
      },
      localization: {
        timeFormatter: (t: Time) => hhmm(t),
        priceFormatter: (v: number) => Math.round(v).toLocaleString("ko-KR"),
      },
      crosshair: { mode: CrosshairMode.Hidden },
      handleScroll: false,
      handleScale: false,
    });
    candles.current = c.addSeries(CandlestickSeries, {
      upColor: p.up,
      downColor: p.down,
      wickUpColor: p.up,
      wickDownColor: p.down,
      borderVisible: false,
      priceLineVisible: true,
      priceLineStyle: LineStyle.Dashed,
    });
    ma5.current = c.addSeries(LineSeries, {
      color: p.touch,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    ma20.current = c.addSeries(LineSeries, {
      color: p.muted,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    markers.current = createSeriesMarkers(candles.current, []);
    chart.current = c;
    drawn.current = -1;

    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const retheme = () => {
      const q = palette();
      c.applyOptions({ layout: { textColor: q.muted }, grid: { horzLines: { color: q.line } } });
      candles.current?.applyOptions({
        upColor: q.up,
        downColor: q.down,
        wickUpColor: q.up,
        wickDownColor: q.down,
      });
      ma5.current?.applyOptions({ color: q.touch });
      ma20.current?.applyOptions({ color: q.muted });
    };
    mq.addEventListener("change", retheme);
    return () => {
      mq.removeEventListener("change", retheme);
      c.remove();
      chart.current = null;
    };
  }, [data]);

  // bars: append one at a time while playing, redraw on any jump
  useEffect(() => {
    const cs = candles.current;
    if (!cs || !ma5.current || !ma20.current || !chart.current) return;
    const bars = data.bundle.bars;
    const { ma5: m5, ma20: m20 } = data.bundle.ma;
    const candle = (i: number) => ({
      time: times[i],
      open: bars[i][1],
      high: bars[i][2],
      low: bars[i][3],
      close: bars[i][4],
    });
    const line = (arr: (number | null)[], i: number) =>
      arr[i] == null ? { time: times[i] } : { time: times[i], value: arr[i] as number };
    if (idx === drawn.current + 1 && idx >= 0) {
      cs.update(candle(idx));
      ma5.current.update(line(m5, idx));
      ma20.current.update(line(m20, idx));
    } else if (idx !== drawn.current) {
      const n = idx + 1;
      cs.setData(Array.from({ length: n }, (_, i) => candle(i)));
      ma5.current.setData(Array.from({ length: n }, (_, i) => line(m5, i)));
      ma20.current.setData(Array.from({ length: n }, (_, i) => line(m20, i)));
    }
    drawn.current = idx;
    // a 90-minute window (like the runner's live chart): it fills left to right for
    // the first 90 minutes, then scrolls with the clock; the scrubber shows the day
    const right = Math.max(idx, WINDOW - 1);
    chart.current.timeScale().setVisibleLogicalRange({ from: right - WINDOW + 0.5, to: right + 1.5 });
  }, [idx, data, times]);

  // markers for the alerts seen so far (following the active filter)
  useEffect(() => {
    if (!markers.current) return;
    const p = palette();
    const list: SeriesMarker<Time>[] = [];
    for (const e of data.alerts) {
      if (e.at > idx) break;
      if (!matchesFilter(e, filter)) continue;
      const big = e.id === selected;
      const size = big ? 2 : 0.8;
      if (e.kind === "touch") {
        list.push({
          time: times[e.at],
          position: e.dir === "up" ? "belowBar" : "aboveBar",
          shape: "square",
          color: p.touch,
          size,
        });
        continue;
      }
      for (const d of e.dirs) {
        list.push({
          time: times[e.at],
          position: d === "up" ? "belowBar" : "aboveBar",
          shape: d === "up" ? "arrowUp" : d === "down" ? "arrowDown" : "circle",
          color: d === "up" ? p.up : d === "down" ? p.down : p.neutral,
          size,
        });
      }
    }
    markers.current.setMarkers(list);
  }, [idx, data, filter, selected, times]);

  return <div ref={el} className="h-full w-full" />;
}
