"""Entry/selection variants compared against the conservative top1/top2 baseline.

Built on Step 11/12 (rule_signals + labels + daily_topk_backtest). The conservative
threshold's daily top1/top2 is the fixed baseline; variants modify either selection
or entry timing and are re-evaluated:

  baseline         conservative top-k, enter at decision close, evaluate forward.
  entry_delay_Nm   same selection, but enter N minutes after the decision close.
  upper_shadow     candidate pool also requires upper_shadow_1m <= limit (weak filter).
  preopen_penalty  preopen weakness is a SOFT score penalty (not a hard filter) that
                   re-ranks the daily top-k.

Selection/penalty use only pre-decision info; labels are never used for selection.
Evaluation (MFE/MAE/hits/stop_first) is recomputed from the actual entry price.
"""

from __future__ import annotations

PREOPEN_START = "08:00:00"
PREOPEN_END = "09:00:00"          # [start, end)
UPPER_SHADOW_LIMIT = 0.005        # upper_shadow_1m <= 0.5%
ENTRY_DELAY_BARS = 3              # entry_delay_3m = enter 3 minutes after the decision close
PREOPEN_PENALTY = 1.0            # subtracted from the selection margin when preopen is weak
TARGET_PCT = 0.05
STOP_PCT = 0.015

SORT_COLS = ["_sel_margin", "up_score", "rank_ret_20m_1m", "volume_z_20m_1m", "_decision_dt"]
SORT_ASCENDING = [False, False, True, False, True]


def build_preopen_context(raw_bars):
    """Per (symbol, date) pre-open (08:00-09:00) context from al_raw bars.

    preopen_ok = preopen_ret >= 0 AND preopen_vwap_distance >= 0 (False when missing).
    """
    import pandas as pd

    raw = raw_bars.copy()
    raw["date"] = raw["bar_start"].str.slice(0, 10)
    raw["time"] = raw["bar_start"].str.slice(11, 19)
    pre = raw[(raw["time"] >= PREOPEN_START) & (raw["time"] < PREOPEN_END)]

    rows = []
    for (symbol, date), g in pre.groupby(["symbol", "date"], sort=False):
        g = g.sort_values("bar_start")
        vol = float(g["volume"].sum())
        typical = (g["high"] + g["low"] + g["close"]) / 3
        vwap = float((typical * g["volume"]).sum() / vol) if vol else float("nan")
        first_open = float(g.iloc[0]["open"])
        last_close = float(g.iloc[-1]["close"])
        rows.append({
            "symbol": symbol,
            "date": date,
            "preopen_ret": last_close / first_open - 1 if first_open else float("nan"),
            "preopen_vwap_distance": last_close / vwap - 1 if (vwap == vwap and vwap) else float("nan"),
            "preopen_bar_count": int(len(g)),
        })
    ctx = pd.DataFrame(rows, columns=["symbol", "date", "preopen_ret", "preopen_vwap_distance", "preopen_bar_count"])
    ctx["preopen_ok"] = (ctx["preopen_ret"] >= 0) & (ctx["preopen_vwap_distance"] >= 0)
    return ctx


def bars_by_day(bars):
    import pandas as pd

    b = bars.copy()
    b["_dt"] = pd.to_datetime(b["bar_start"])
    b["_date"] = b["bar_start"].str.slice(0, 10)
    return {
        (symbol, date): g.sort_values("_dt").reset_index(drop=True)
        for (symbol, date), g in b.groupby(["symbol", "_date"], sort=False)
    }


def add_selection_margin(candidates, *, preopen_penalty=0.0):
    """Add `_sel_margin` = score_margin minus a penalty when preopen is weak.

    preopen_penalty=0 -> baseline ranking (just score_margin). The candidates frame
    may carry a `preopen_ok` column (missing/NaN treated as not-ok).
    """
    df = candidates.copy()
    margin = df["score_margin"].astype(float)
    if preopen_penalty and "preopen_ok" in df.columns:
        not_ok = ~df["preopen_ok"].fillna(False).astype(bool)
        margin = margin - preopen_penalty * not_ok.astype(float)
    df["_sel_margin"] = margin
    return df


def select_topk(candidates, *, top_k=2, cooldown_minutes=60):
    """Daily top-k by `_sel_margin` (then up_score, rank, volume_z, time) with cooldown."""
    import pandas as pd

    if candidates is None or len(candidates) == 0:
        return pd.DataFrame()
    rows = []
    for _, day in candidates.groupby("date", sort=True):
        ordered = day.sort_values(SORT_COLS, ascending=SORT_ASCENDING).reset_index(drop=True)
        chosen = []
        by_symbol = {}
        for _, row in ordered.iterrows():
            if len(chosen) >= top_k:
                break
            prior = by_symbol.get(row["symbol"], [])
            if any(abs((row["_decision_dt"] - t).total_seconds()) < cooldown_minutes * 60 for t in prior):
                continue
            out = row.to_dict()
            out["daily_rank"] = len(chosen) + 1
            chosen.append(out)
            by_symbol.setdefault(row["symbol"], []).append(row["_decision_dt"])
        rows.extend(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def evaluate_entry(day, decision_dt, close_t, *, delay_bars=0, target_pct=TARGET_PCT, stop_pct=STOP_PCT):
    """Evaluate a single signal from its actual entry.

    ``future`` is the bars at/after the decision; future.iloc[0] is the decision
    bar and its close is the decision close, so the close ``delay_bars`` minutes
    later is future.iloc[delay_bars].

    delay_bars=0 -> baseline: enter at the decision close (close_t); MFE/MAE/returns
        span the decision bar and every bar after it (the Step 12 convention).
    delay_bars=N -> enter N minutes after the decision close, at future.iloc[N]'s
        close; MFE/MAE/returns span strictly the bars AFTER that entry bar (no
        look-ahead on the entry bar itself), so the delay comparison is, if
        anything, conservative against the delay.
    Returns None when the entry bar or the post-entry window is missing.
    """
    import pandas as pd

    future = day[day["_dt"] >= decision_dt].reset_index(drop=True)
    if delay_bars == 0:
        if future.empty:
            return None
        entry_price = float(close_t)
        entry_dt = decision_dt
        eval_bars = future
    else:
        if len(future) <= delay_bars:
            return None
        entry_bar = future.iloc[delay_bars]
        entry_price = float(entry_bar["close"])
        entry_dt = entry_bar["_dt"]
        eval_bars = future.iloc[delay_bars + 1:].reset_index(drop=True)
    if eval_bars is None or len(eval_bars) == 0:
        return None

    mfe = float(eval_bars["high"].max() / entry_price - 1)
    mae = float(eval_bars["low"].min() / entry_price - 1)
    out = {"entry_price": entry_price, "future_bar_count": int(len(eval_bars)), "mfe": mfe, "mae": mae}
    for label, minutes in {"30m": 30, "60m": 60, "120m": 120}.items():
        end = entry_dt + pd.Timedelta(minutes=minutes)
        sub = eval_bars[eval_bars["_dt"] < end]
        out[f"return_{label}"] = float(sub.iloc[-1]["close"] / entry_price - 1) if len(sub) else float("nan")
    out["return_eod"] = float(eval_bars.iloc[-1]["close"] / entry_price - 1)
    out["hit_up2"] = int(mfe >= 0.02)
    out["hit_up3"] = int(mfe >= 0.03)
    out["hit_up5"] = int(mfe >= 0.05)
    target = entry_price * (1 + target_pct)
    stop = entry_price * (1 - stop_pct)
    ht = eval_bars.index[eval_bars["high"] >= target]
    hs = eval_bars.index[eval_bars["low"] <= stop]
    first_t = ht[0] if len(ht) else None
    first_s = hs[0] if len(hs) else None
    out["stop_first"] = int(first_s is not None and (first_t is None or first_s <= first_t))
    return out


def evaluate_variant(selected, by_day, *, delay_bars=0):
    import pandas as pd

    if selected is None or len(selected) == 0:
        return pd.DataFrame()
    rows = []
    for _, row in selected.iterrows():
        day = by_day.get((row["symbol"], row["date"]))
        if day is None:
            continue
        ev = evaluate_entry(day, row["_decision_dt"], float(row["close"]), delay_bars=delay_bars)
        if ev is None:
            continue
        out = row.to_dict()
        out.update(ev)
        rows.append(out)
    return pd.DataFrame(rows)


def summarize(df, *, name=""):
    if df is None or len(df) == 0:
        base = {k: 0.0 for k in ("hit2", "hit3", "hit5", "mfe", "mae", "stop_first", "ret30", "ret60", "ret120", "eod")}
        return {"variant": name, "n": 0, "symbol_concentration": 0.0, **base}
    n = int(len(df))
    return {
        "variant": name,
        "n": n,
        "hit2": round(float(df["hit_up2"].mean()), 6),
        "hit3": round(float(df["hit_up3"].mean()), 6),
        "hit5": round(float(df["hit_up5"].mean()), 6),
        "mfe": round(float(df["mfe"].mean()), 6),
        "mae": round(float(df["mae"].mean()), 6),
        "stop_first": round(float(df["stop_first"].mean()), 6),
        "ret30": round(float(df["return_30m"].mean()), 6),
        "ret60": round(float(df["return_60m"].mean()), 6),
        "ret120": round(float(df["return_120m"].mean()), 6),
        "eod": round(float(df["return_eod"].mean()), 6),
        "symbol_concentration": round(float(df["symbol"].value_counts().max() / n), 6),
    }
