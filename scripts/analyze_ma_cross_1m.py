"""MA5 x MA20 1-minute crossover raw-statistics study (no ML).

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. Pure historical
aggregation on bars_1m; nothing here touches serving or the model lineages.

What it measures (user-decided conventions; see src/ma_cross_study.py docstring):
  * cases 1-3: every MA5x20 cross, split by |open - MA5| proximity bands
    (near <= 0.10% / mid / far >= 0.30%, CLI-adjustable) plus an ATR-normalized
    auxiliary distance; outcomes over timestamp windows [T, T+10/15min): +0.7%
    directional close-touch rate, high/low extreme-touch rate, endpoint return,
    max adverse excursion (MAE) distribution.
  * case 4: whipsaw = opposite cross within 5 wall-clock minutes -- rate,
    time-to-whipsaw, exit return at the reversal bar, and the original-direction
    resume rate within 15 minutes.
  * phases: P1 (open-MA, causal), P2i (close-MA chart, same-bar open entry --
    LOOK-AHEAD diagnostic, crossing-bar body quantified), P2r (close-MA, next-open
    entry -- causal).
  * warmup variants: regular / preopen / carry (premarket + previous-day carry);
    preopen & carry signals may fire in the premarket (session-split in every table).
  * baselines: unconditional (every bar as a pseudo-entry) and +-30min matched
    non-cross bars around each event (local-regime control).

Run:
  ..\\Python311-64\\python.exe scripts\\analyze_ma_cross_1m.py --main-symbols <code1,code2>
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]  # noqa: E402 (path bootstrap below)
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

import ma_cross_study as mcs
from config import DB_PATH, PROCESSED_DIR, ensure_directories
from settings import load_active_symbols, load_data_config
from utils import now_kst_text

QUANTS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def parse_args():
    p = argparse.ArgumentParser(description="MA5x20 1m crossover raw-statistics study.")
    p.add_argument("--start", default="2025-06-02")
    p.add_argument("--end", default="", help="inclusive; default = all available")
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--raw-source", default="al_raw",
                   help="bar source carrying premarket bars (preopen/carry variants).")
    p.add_argument("--variants", default=",".join(mcs.VARIANTS))
    p.add_argument("--windows", default="10,15")
    p.add_argument("--profit-threshold", type=float, default=mcs.PROFIT_THRESHOLD)
    p.add_argument("--whipsaw-minutes", type=float, default=5)
    p.add_argument("--resume-minutes", type=float, default=15)
    p.add_argument("--cluster-minutes", type=float, default=15)
    p.add_argument("--near", type=float, default=mcs.NEAR_MAX)
    p.add_argument("--far", type=float, default=mcs.FAR_MIN)
    p.add_argument("--baseline-halfwidth", type=float, default=30)
    p.add_argument("--main-symbols", default="",
                   help="comma-separated codes highlighted separately in the symbol "
                        "table (must be active symbols; never hard-coded here).")
    p.add_argument("--out-prefix", default="ma_cross_1m")
    args = p.parse_args()
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in args.variants:
        if v not in mcs.VARIANTS:
            p.error(f"unknown variant {v!r}; choices={mcs.VARIANTS}")
    args.windows = [int(w) for w in str(args.windows).split(",") if w.strip()]
    args.main_symbols = [s.strip() for s in args.main_symbols.split(",") if s.strip()]
    return args


def load_bars(args, symbols):
    conn = sqlite3.connect(args.db)
    try:
        q = ("SELECT symbol, bar_start, open, high, low, close FROM bars_1m "
             "WHERE source = ? AND substr(bar_start,1,10) >= ?")
        params = [args.raw_source, args.start]
        if args.end:
            q += " AND substr(bar_start,1,10) <= ?"
            params.append(args.end)
        bars = pd.read_sql_query(q, conn, params=params)
        n_reg = conn.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE source = 'al_regular' "
            "AND substr(bar_start,1,10) >= ?" + (" AND substr(bar_start,1,10) <= ?" if args.end else ""),
            params[1:]).fetchone()[0]
    finally:
        conn.close()
    bars = bars[bars["symbol"].isin(symbols)].reset_index(drop=True)
    return bars, n_reg


PRIM_BASE = ("fwd_max_close", "fwd_min_close", "fwd_max_high", "fwd_min_low",
             "endpoint_close", "complete")


def prim_cols(windows):
    return [f"{b}_w{int(w)}" for w in windows for b in PRIM_BASE]


def metric_cols(windows):
    out = []
    for w in windows:
        w = int(w)
        out += [f"w{w}_close_touch", f"w{w}_extreme_touch", f"w{w}_endpoint_ret",
                f"w{w}_mae", f"w{w}_mfe", f"w{w}_complete"]
    return out


def build_events(args, bars, per_bar):
    """One long frame: a row per (variant, phase, cross)."""
    windows = args.windows
    anchor_keep = ["symbol", "bar_start", "open", "pos", "day_n"] + prim_cols(windows)
    a0 = per_bar[anchor_keep]
    a1 = per_bar[["symbol", "date", "pos", "open", "bar_start"] + prim_cols(windows)].copy()
    a1["pos"] = a1["pos"] - 1          # joining events pos -> next bar's row
    rev_open = per_bar[["symbol", "bar_start", "open", "date", "pos"]].rename(
        columns={"bar_start": "rev_bar_start", "open": "rev_open",
                 "pos": "rev_pos"})
    frames = []
    mask_counts = []
    for variant in args.variants:
        vf, gcols = mcs.prepare_variant(bars, variant)
        for ma_source in ("open", "close"):
            mf = mcs.add_mas_and_atr(vf, gcols, price_col=ma_source)
            mask_counts.append({"variant": variant, "ma_source": ma_source,
                                "n_bars": len(mf),
                                "n_invalid_bars": int(mf["bar_invalid"].sum())})
            cf = mcs.detect_crossovers(mf, gcols,
                                       signal_start=mcs.variant_signal_start(variant))
            ev = cf[(cf["cross"] != 0) & cf["eligible"]].copy()
            if ev.empty:
                continue
            ev["direction"] = ev["cross"].astype(int)
            open_t = ev["open"].astype(float)
            ev["dist_pct"] = (open_t - ev["ma_fast"]).abs() / ev["ma_fast"]
            ev["dist_atr"] = (open_t - ev["ma_fast"]).abs() / ev["atr"]
            ev["band"] = mcs.classify_bands(ev["dist_pct"], near_max=args.near,
                                            far_min=args.far)
            ev["atr_band"] = mcs.classify_bands(
                ev["dist_atr"], near_max=mcs.ATR_NEAR_MAX, far_min=mcs.ATR_FAR_MIN)
            ev["month"] = ev["date"].str.slice(0, 7)
            base_cols = ["symbol", "date", "month", "bar_start", "time", "session",
                         "direction", "dist_pct", "dist_atr", "band", "atr_band",
                         "open", "close"]
            ev = ev[base_cols].reset_index(drop=True)
            ev = mcs.attach_whipsaw_and_resume(
                ev, whipsaw_minutes=args.whipsaw_minutes,
                resume_minutes=args.resume_minutes)
            ev = mcs.mark_cluster_first(ev, cluster_minutes=args.cluster_minutes)
            # anchors on the outcome stream
            ev = ev.merge(a0.rename(columns={"open": "open_a0"}),
                          on=["symbol", "bar_start"], how="left", validate="m:1")
            ev = ev.merge(
                a1.rename(columns={c: f"{c}_next" for c in a1.columns
                                   if c not in ("symbol", "date", "pos")}),
                on=["symbol", "date", "pos"], how="left", validate="m:1")
            ev = ev.merge(rev_open, on=["symbol", "rev_bar_start"], how="left",
                          suffixes=("", "_revjoin"))
            rev_next = per_bar[["symbol", "date", "pos", "open"]].rename(
                columns={"pos": "rev_pos_next_key", "open": "rev_next_open"})
            ev["rev_pos_next_key"] = ev["rev_pos"] + 1
            ev = ev.merge(rev_next, on=["symbol", "date", "rev_pos_next_key"],
                          how="left")
            d = ev["direction"].astype(float)
            phase_rows = []
            phases = mcs.MA_SOURCE_PHASES[ma_source]
            for phase in phases:
                pr = ev.copy()
                pr["variant"] = variant
                pr["ma_source"] = ma_source
                pr["phase"] = phase
                if phase in ("P1", "P2i"):
                    anchor = pr[["open"] + prim_cols(windows)].copy()
                    entry = pr["open"].astype(float)
                    pr["exit_ret"] = d * (pr["rev_open"] / entry - 1)
                else:                          # P2r: next-open entry, window t+1..
                    src_cols = [f"{c}_next" for c in ["open"] + prim_cols(windows)]
                    anchor = pr[src_cols].copy()
                    anchor.columns = ["open"] + prim_cols(windows)
                    for w in windows:
                        anchor[f"complete_w{int(w)}"] = (
                            anchor[f"complete_w{int(w)}"].fillna(False).astype(bool))
                    entry = anchor["open"].astype(float)
                    pr["exit_ret"] = d * (pr["rev_next_open"] / entry - 1)
                outc = mcs.directional_outcomes(
                    anchor, d.to_numpy(), windows,
                    threshold=args.profit_threshold, entry=entry.to_numpy())
                pr["entry"] = entry.to_numpy()
                pr = pd.concat([pr, outc], axis=1)
                pr["body_ret"] = (d * (pr["close"] / pr["open"].astype(float) - 1)
                                  if phase == "P2i" else np.nan)
                keep = (["variant", "ma_source", "phase", "symbol", "date", "month",
                         "bar_start", "time", "session", "direction", "entry",
                         "dist_pct", "dist_atr", "band", "atr_band", "whipsaw",
                         "tt_whipsaw_min", "resume", "tt_resume_min", "exit_ret",
                         "cluster_first", "body_ret", "pos"]
                        + metric_cols(windows))
                phase_rows.append(pr[keep])
            frames.append(pd.concat(phase_rows, ignore_index=True))
    events = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame())
    return events, pd.DataFrame(mask_counts)


def summarize(df, keys, windows, *, label):
    """Wide aggregate rows per key combination."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for key_vals, g in df.groupby(keys, dropna=False, observed=True):
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        row = {"section": label, **dict(zip(keys, key_vals)), "n": int(len(g))}
        if "whipsaw" in g.columns and "whipsaw" not in keys:
            nws = int(g["whipsaw"].sum())
            row["whipsaw_rate"] = nws / len(g) if len(g) else np.nan
            row["tt_whipsaw_mean"] = float(g["tt_whipsaw_min"].mean())
            row["tt_whipsaw_med"] = float(g["tt_whipsaw_min"].median())
            row["exit_ret_mean"] = float(g["exit_ret"].mean())
            row["exit_ret_med"] = float(g["exit_ret"].median())
            row["resume_rate"] = (float(g.loc[g["whipsaw"], "resume"].mean())
                                  if nws else np.nan)
            row["tt_resume_med"] = float(g["tt_resume_min"].median())
        for w in windows:
            w = int(w)
            comp = g[g[f"w{w}_complete"] == True]  # noqa: E712
            row[f"w{w}_n"] = int(len(comp))
            row[f"w{w}_n_truncated"] = int(len(g) - len(comp))
            if not len(comp):
                continue
            row[f"w{w}_close_touch"] = float(comp[f"w{w}_close_touch"].mean())
            row[f"w{w}_extreme_touch"] = float(comp[f"w{w}_extreme_touch"].mean())
            row[f"w{w}_endpoint_mean"] = float(comp[f"w{w}_endpoint_ret"].mean())
            row[f"w{w}_endpoint_med"] = float(comp[f"w{w}_endpoint_ret"].median())
            row[f"w{w}_mae_mean"] = float(comp[f"w{w}_mae"].mean())
            row[f"w{w}_mae_med"] = float(comp[f"w{w}_mae"].median())
            row[f"w{w}_mae_q10"] = float(comp[f"w{w}_mae"].quantile(0.10))
            row[f"w{w}_mae_q90"] = float(comp[f"w{w}_mae"].quantile(0.90))
            for mark in mcs.MAE_MARKS:
                pct = int(round(abs(mark) * 1000))
                row[f"w{w}_p_mae_le_{pct}"] = float((comp[f"w{w}_mae"] <= mark).mean())
            row[f"w{w}_mfe_med"] = float(comp[f"w{w}_mfe"].median())
        if "body_ret" in g.columns:
            b = g["body_ret"].dropna()
            if len(b):
                row["body_ret_mean"] = float(b.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def baselines(args, events, per_bar):
    """Unconditional per-bar control + matched +-30min non-cross control."""
    windows = args.windows
    out = []
    prim = per_bar[["open", "session"] + prim_cols(windows)]
    for direction in (1, -1):
        m = mcs.directional_outcomes(prim, float(direction), windows,
                                     threshold=args.profit_threshold)
        m["session"] = prim["session"].to_numpy()
        m["direction"] = direction
        out.append(summarize(m, ["session", "direction"], windows,
                             label="baseline_unconditional"))
    pos_by_key = {(s, b): i for i, (s, b) in
                  enumerate(zip(per_bar["symbol"].to_numpy(),
                                per_bar["bar_start"].to_numpy()))}
    matched = []
    for (variant, ma_source), ev in events[events["phase"].isin(["P1", "P2i"])] \
            .groupby(["variant", "ma_source"], sort=False):
        ev = ev.reset_index(drop=True)
        cross_positions = [pos_by_key[k] for k in
                           zip(ev["symbol"], ev["bar_start"]) if k in pos_by_key]
        ev_rows, bar_rows = mcs.matched_baseline_index(
            ev, per_bar, halfwidth_minutes=args.baseline_halfwidth,
            cross_positions=cross_positions)
        if not len(bar_rows):
            continue
        anchor = per_bar.iloc[bar_rows][["open"] + prim_cols(windows)].reset_index(drop=True)
        dirs = ev["direction"].to_numpy()[ev_rows]
        m = mcs.directional_outcomes(anchor, dirs, windows,
                                     threshold=args.profit_threshold)
        m["variant"] = variant
        m["ma_source"] = ma_source
        m["direction"] = dirs
        m["band"] = ev["band"].to_numpy()[ev_rows]
        matched.append(m)
    if matched:
        mdf = pd.concat(matched, ignore_index=True)
        out.append(summarize(mdf, ["variant", "ma_source", "direction", "band"],
                             windows, label="baseline_matched"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def dist_quantile_rows(events):
    rows = []
    for (variant, ma_source), g in events[events["phase"] != "P2r"].groupby(
            ["variant", "ma_source"], sort=False):
        for col in ("dist_pct", "dist_atr"):
            x = g[col].dropna()
            row = {"section": "dist_quantiles", "variant": variant,
                   "ma_source": ma_source, "metric": col, "n": int(len(x))}
            for q in QUANTS:
                row[f"q{int(q * 100):02d}"] = float(x.quantile(q)) if len(x) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def symbol_rows(events, windows, main_symbols):
    ev = events
    out = summarize(ev, ["variant", "phase", "direction", "symbol"], windows,
                    label="by_symbol")
    if len(out):
        out["is_main"] = out["symbol"].isin(main_symbols)
    return out


def fmt(df, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_string(index=False,
                              float_format=lambda v: f"{v:,.4f}")


def main():
    args = parse_args()
    ensure_directories()
    model_source = load_data_config().model_source
    symbols = load_active_symbols()
    bad_main = [s for s in args.main_symbols if s not in symbols]
    if bad_main:
        raise SystemExit(f"--main-symbols not in active symbols: {bad_main}")
    print(f"[{now_kst_text()}] MA5x20 cross study source={args.raw_source} "
          f"(model_source={model_source}) symbols={len(symbols)} "
          f"variants={args.variants} windows={args.windows} "
          f"thr={args.profit_threshold} near<={args.near} far>={args.far} "
          f"main={args.main_symbols or '(none)'}")

    bars, n_reg = load_bars(args, symbols)
    print(f"  bars({args.raw_source})={len(bars):,}  al_regular rows in range={n_reg:,}")
    if bars.empty:
        raise SystemExit("no bars loaded - check --raw-source/--start/--end")

    stream = mcs.build_outcome_stream(bars)
    n_regular_slice = int((stream["session"] == "regular").sum())
    print(f"  outcome stream: bars={len(stream):,} regular={n_regular_slice:,} "
          f"premarket={len(stream) - n_regular_slice:,}")
    per_bar = mcs.per_bar_forward_stats(stream, args.windows)

    events, mask_counts = build_events(args, bars, per_bar)
    if events.empty:
        raise SystemExit("no crossover events found")
    print(f"  events (variant x phase rows): {len(events):,}")

    windows = args.windows
    sections = []
    pop = summarize(events, ["variant", "phase", "session"], windows, label="population")
    share = (events.groupby(["variant", "phase", "band"], observed=True).size()
             / events.groupby(["variant", "phase"], observed=True).size())
    share = share.rename("band_share").reset_index()
    share["section"] = "band_share"
    sections += [pop, share,
                 summarize(events, ["variant", "phase", "direction", "band"],
                           windows, label="core"),
                 summarize(events, ["variant", "phase", "session", "direction", "band"],
                           windows, label="core_by_session"),
                 summarize(events, ["variant", "phase", "direction", "band", "whipsaw"],
                           windows, label="whipsaw_split"),
                 summarize(events[events["cluster_first"]],
                           ["variant", "phase", "direction", "band"],
                           windows, label="core_cluster_first"),
                 summarize(events, ["variant", "phase", "direction", "month"],
                           windows, label="monthly"),
                 symbol_rows(events, windows, args.main_symbols),
                 dist_quantile_rows(events),
                 baselines(args, events, per_bar)]
    mask_counts["section"] = "invalid_bars"
    sections.append(mask_counts)
    aggregate = pd.concat(sections, ignore_index=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    events_path = PROCESSED_DIR / f"{args.out_prefix}_events.parquet"
    csv_path = PROCESSED_DIR / f"{args.out_prefix}_study_{ts}.csv"
    events.to_parquet(events_path, index=False)
    aggregate.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ---- console pivots (full detail lives in the CSV) ----
    w0 = int(windows[0])
    core = aggregate[aggregate["section"] == "core"]
    print("\n== population (variant x phase x session)")
    print(fmt(pop, ["variant", "phase", "session", "n", "whipsaw_rate",
                    f"w{w0}_n", f"w{w0}_n_truncated"]))
    print("\n== band shares (near/mid/far)")
    piv = share.pivot_table(index=["variant", "phase"], columns="band",
                            values="band_share")
    print(piv.to_string(float_format=lambda v: f"{v:.1%}"))
    print(f"\n== core (variant=regular, W={w0}) — touch/MAE by phase x direction x band")
    sel = core[core["variant"] == "regular"] if (core["variant"] == "regular").any() else core
    print(fmt(sel, ["phase", "direction", "band", "n", f"w{w0}_n",
                    f"w{w0}_close_touch", f"w{w0}_extreme_touch",
                    f"w{w0}_endpoint_mean", f"w{w0}_mae_med",
                    f"w{w0}_p_mae_le_7", "whipsaw_rate"]))
    print("\n== whipsaw analysis (case 4)")
    ws = summarize(events, ["variant", "phase", "direction"], windows, label="ws")
    print(fmt(ws, ["variant", "phase", "direction", "n", "whipsaw_rate",
                   "tt_whipsaw_mean", "tt_whipsaw_med", "exit_ret_mean",
                   "exit_ret_med", "resume_rate", "tt_resume_med"]))
    base = aggregate[aggregate["section"] == "baseline_unconditional"]
    if len(base):
        print("\n== unconditional baseline (all bars pseudo-entry)")
        print(fmt(base, ["session", "direction", f"w{w0}_n", f"w{w0}_close_touch",
                         f"w{w0}_extreme_touch", f"w{w0}_mae_med"]))
    if args.main_symbols and "is_main" in aggregate.columns:
        sym = aggregate[(aggregate["section"] == "by_symbol")
                        & (aggregate["is_main"] == True)]  # noqa: E712 (NaN-safe)
        if len(sym):
            print("\n== main symbols")
            print(fmt(sym, ["variant", "phase", "direction", "symbol", "n",
                            f"w{w0}_close_touch", f"w{w0}_mae_med", "whipsaw_rate"]))

    print(f"\nevents -> {events_path}")
    print(f"aggregate -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
