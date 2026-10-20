// Shapes written by scripts/export_replay_bundle.py (BUNDLE_VERSION 1).

export type Dir = "up" | "down" | "neutral";
export type Kind = "long" | "down" | "both" | "touch" | "status";
export type Band = "HIGH" | "MED" | "LOW" | "NA" | "";

/** [HH:MM, open, high, low, close, volume] */
export type Bar = [string, number, number, number, number, number];

export interface Alert {
  side: "long" | "down";
  type: string;
  type_ko: string;
  dir: Dir;
  grade: string;
  prob: number | null;
  band: Band;
  reasons: string[];
  message: string;
}

export interface Window {
  hi: number | null;
  lo: number | null;
  reveal: string;
  full?: boolean;
}

export interface TouchInfo {
  direction: 1 | -1;
  band: "near" | "mid" | "far";
  dist_pct: number | null;
  whipsaw: boolean;
  grade: string | null;
  score: number | null;
  grade_rate: number | null;
  grade_n: number | null;
  status: string;
}

export interface Outcome {
  h60?: Window | null;
  eod?: Window | null;
  touch?: { hit: boolean; mae: number | null; reveal: string };
}

export interface StatusInfo {
  long?: { type: string; grade: string; prob: number | null; band: Band };
  short?: {
    type: string;
    grade: string;
    dn: number | null;
    up: number | null;
    dir: number | null;
    prob: number | null;
    band: Band;
  };
}

export interface ReplayEvent {
  id: number;
  bar: string;
  decision_time: string;
  kind: Kind;
  raw: string;
  close: number;
  dir?: Dir;
  alerts?: Alert[];
  touch?: TouchInfo;
  outcome?: Outcome;
  status?: StatusInfo;
}

export interface Bundle {
  version: number;
  meta: {
    date: string;
    symbol: string;
    name: string;
    open: number;
    close: number;
    prev_close: number | null;
    change_pct: number | null;
    high: number;
    low: number;
    counts: { long: number; down: number; touch: number; touch_graded: number };
    universe: string[];
    models: { long: string; down: string; touch: string };
    touch_grade_misses?: number;
  };
  bars: Bar[];
  ma: { ma5: (number | null)[]; ma20: (number | null)[] };
  events: ReplayEvent[];
}

export interface AvailableDate {
  date: string;
  file: string;
}

export interface IndexEntry extends AvailableDate {
  date: string;
  label: string;
  change_pct: number | null;
  counts: Bundle["meta"]["counts"];
  open: number;
  close: number;
  prev_close: number | null;
  spark: number[];
  file: string;
}

export interface ReplayIndex {
  version: number;
  symbol: string;
  name: string;
  dates: IndexEntry[];
  /** Lightweight availability list. Older cached catalogs contain featured dates only. */
  availableDates?: AvailableDate[];
}

export type Filter = "all" | "up" | "down" | "neutral" | "touch";
