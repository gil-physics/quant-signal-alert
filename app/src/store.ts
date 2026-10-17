import { create } from "zustand";
import type { Filter } from "./types";
import type { Prepared, Speed } from "./lib/prepare";
import { SPEEDS, matchesFilter } from "./lib/prepare";

interface ReplayState {
  data: Prepared | null;
  /** current bar index; -1 = before the first bar */
  idx: number;
  playing: boolean;
  speed: Speed;
  filter: Filter;
  raw: boolean;
  showStatus: boolean;
  selected: number | null;
  resumeAfterSheet: boolean;

  load: (data: Prepared) => void;
  unload: () => void;
  setIdx: (idx: number) => void;
  step: () => void;
  togglePlay: () => void;
  cycleSpeed: () => void;
  setFilter: (f: Filter) => void;
  setRaw: (raw: boolean) => void;
  setShowStatus: (v: boolean) => void;
  select: (id: number | null) => void;
  restart: () => void;
  /** jump to the next alert matching the filter (end of day when none) */
  nextAlert: () => void;
}

export const useReplay = create<ReplayState>((set, get) => ({
  data: null,
  idx: -1,
  playing: false,
  speed: 1,
  filter: "all",
  raw: false,
  showStatus: false,
  selected: null,
  resumeAfterSheet: false,

  load: (data) => set({ data, idx: -1, playing: true, selected: null }),
  unload: () => set({ data: null, idx: -1, playing: false, selected: null }),
  setIdx: (idx) => {
    const { data } = get();
    if (!data) return;
    set({ idx: Math.max(-1, Math.min(idx, data.last)) });
  },
  step: () => {
    const { data, idx } = get();
    if (!data) return;
    if (idx >= data.last) {
      set({ playing: false });
      return;
    }
    set({ idx: idx + 1, playing: idx + 1 < data.last ? get().playing : false });
  },
  togglePlay: () => {
    const { data, idx, playing } = get();
    if (!data) return;
    // pressing play at the end of the day starts the day over
    if (!playing && idx >= data.last) set({ idx: -1, playing: true });
    else set({ playing: !playing });
  },
  cycleSpeed: () => {
    const i = SPEEDS.indexOf(get().speed);
    set({ speed: SPEEDS[(i + 1) % SPEEDS.length] });
  },
  setFilter: (filter) => set({ filter }),
  setRaw: (raw) => set({ raw }),
  setShowStatus: (showStatus) => set({ showStatus }),
  // opening an alert pauses the replay so its "after" section doesn't change under
  // the reader; closing it resumes if it was playing
  select: (id) => {
    const { playing, resumeAfterSheet } = get();
    if (id !== null) set({ selected: id, playing: false, resumeAfterSheet: playing });
    else set({ selected: null, playing: resumeAfterSheet, resumeAfterSheet: false });
  },
  restart: () => set({ idx: -1, playing: true, selected: null }),
  nextAlert: () => {
    const { data, idx, filter } = get();
    if (!data) return;
    const next = data.alerts.find((e) => e.at > idx && matchesFilter(e, filter));
    set({ idx: next ? next.at : data.last });
  },
}));
