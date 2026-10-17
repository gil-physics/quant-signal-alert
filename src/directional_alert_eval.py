"""Step 13_6: directional-alert quality report helpers.

This module keeps Step 13_5 files intact and adds an evaluation/reporting layer.
Alert generation uses only rule-signal and current-time feature columns. Labels
are accepted only by the evaluation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bracket_eval import day_arrays, prepare_entry
from daily_topk_backtest import apply_base_filters
from directional_alert import (
    ALERT_TYPES,
    AlertThresholds,
    alert_fire_gate,
    compute_directional_alerts,
    select_online_alerts,
    select_topk_diagnostic,
)


MICRO_COLUMNS = [
    "upper_shadow_1m",
    "vwap_distance_1m",
    "distance_from_day_high_1m",
    "return_from_open_1m",
    "range_1m",
    "dist_ma5_intraday",
]

LABEL_COLUMNS = [
    "up2_60m",
    "up2_120m",
    "up3_120m",
    "down2_60m",
    "up5_event",
]

ALERT_OUTPUT_COLUMNS = [
    "symbol",
    "decision_time",
    "close",
    "alert_type",
    "alert_grade",
    "alert_score",
    "direction_score",
    "risk_score",
    "up_score",
    "down_score",
    "reason_1",
    "reason_2",
    "reason_3",
    "source",
    "created_at",
    "rank_ret_20m_1m",
    "relative_ret_20m_to_universe",
    "volume_z_20m_1m",
    "vwap_distance_1m",
    "universe_breadth",
    "date",
    "time",
]

EVAL_COLUMNS = [
    "future_return_30m",
    "future_return_60m",
    "future_return_120m",
    "future_return_eod",
    "direction_hit_30m",
    "direction_hit_60m",
    "direction_hit_120m",
    "MFE",
    "MAE",
    "hit_up2_60m",
    "hit_up2_120m",
    "hit_up3_120m",
    "down2_60m",
    "UP_FAIL",
    "stop_first",
]

RISK_TYPES = {"RISK_ALERT"}
RISK_GRADES = {"RISK"}
UP_TARGET_FOR_STOP_FIRST = 0.015
DOWN_STOP_FOR_STOP_FIRST = 0.015


@dataclass(frozen=True)
class ReportConfig:
    name: str = "baseline"
    min_time: str = "09:10:00"
    max_time: str = "15:00:00"
    daily_alert_limit: int = 5
    cooldown_minutes: int = 30
    entry_delay_minutes: int = 3
    top_k: int = 2


def _ensure_date_time(frame):
    df = frame.copy()
    if "date" not in df.columns:
        df["date"] = df["decision_time"].str.slice(0, 10)
    if "time" not in df.columns:
        df["time"] = df["decision_time"].str.slice(11, 19)
    if "_decision_dt" not in df.columns:
        df["_decision_dt"] = pd.to_datetime(df["decision_time"])
    return df


def build_alert_catalog(
    rule_signals,
    micro_features=None,
    *,
    created_at="",
    min_time="09:10:00",
    max_time="15:00:00",
    fail_quality_pairs=None,
    outlier_pairs=None,
):
    """Build the Step 13_6 alert catalogue without reading labels.

    The output applies time/quality filters, excludes both-side rule candidates,
    and keeps only rows that the baseline fire gate considers noteworthy.
    """
    if rule_signals is None or len(rule_signals) == 0:
        return pd.DataFrame(columns=ALERT_OUTPUT_COLUMNS), {
            "raw_rows": 0,
            "after_time_filter": 0,
            "after_quality_filter": 0,
            "both_excluded_count": 0,
            "after_both_exclusion": 0,
            "after_fire_gate": 0,
        }

    frame = rule_signals.copy()
    if micro_features is not None and len(micro_features):
        cols = ["symbol", "decision_time"] + [c for c in MICRO_COLUMNS if c in micro_features.columns]
        frame = frame.merge(
            micro_features[cols],
            on=["symbol", "decision_time"],
            how="left",
            validate="one_to_one",
        )

    alerts = compute_directional_alerts(frame)
    alerts = _ensure_date_time(alerts)
    base, stats = apply_base_filters(
        alerts,
        min_time=min_time,
        max_time=max_time,
        fail_quality_pairs=fail_quality_pairs,
        outlier_pairs=outlier_pairs,
    )

    both = (base.get("is_up_candidate", 0) == 1) & (base.get("is_down_candidate", 0) == 1)
    base = base.loc[~both].reset_index(drop=True)
    fired = base.loc[alert_fire_gate(base, AlertThresholds())].copy()
    fired["created_at"] = created_at

    out_cols = [c for c in ALERT_OUTPUT_COLUMNS if c in fired.columns]
    out = fired[out_cols].reset_index(drop=True)
    stats.update({
        "both_excluded_count": int(both.sum()),
        "after_both_exclusion": int(len(base)),
        "after_fire_gate": int(len(out)),
    })
    return out, stats


def select_for_evaluation(alerts, *, config=ReportConfig()):
    """Return online and daily-topK selected alerts, separated by selection_mode."""
    if alerts is None or len(alerts) == 0:
        return pd.DataFrame()

    df = _ensure_date_time(alerts)
    online = select_online_alerts(
        df,
        daily_alert_limit=config.daily_alert_limit,
        cooldown_minutes=config.cooldown_minutes,
    ).copy()
    if len(online):
        online["selection_mode"] = "online"
        online["alert_rank"] = online["alert_seq"]

    topk = select_topk_diagnostic(df, top_k=config.top_k).copy()
    if len(topk):
        topk["selection_mode"] = "daily_topk"
        topk["alert_rank"] = topk["daily_rank"]

    frames = [f for f in (online, topk) if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _window_return(entry_price, closes, offsets, minutes):
    window = closes[offsets <= minutes]
    return float(window[-1] / entry_price - 1) if len(window) else np.nan


def _hit_up(entry_price, highs, offsets, level, minutes):
    window = highs[offsets <= minutes]
    return int((window >= entry_price * (1 + level)).any()) if len(window) else 0


def _hit_down(entry_price, lows, offsets, level, minutes):
    window = lows[offsets <= minutes]
    return int((window <= entry_price * (1 - level)).any()) if len(window) else 0


def _stop_first(entry_price, highs, lows):
    up_hits = np.nonzero(highs >= entry_price * (1 + UP_TARGET_FOR_STOP_FIRST))[0]
    dn_hits = np.nonzero(lows <= entry_price * (1 - DOWN_STOP_FOR_STOP_FIRST))[0]
    up_i = int(up_hits[0]) if up_hits.size else None
    dn_i = int(dn_hits[0]) if dn_hits.size else None
    return int(dn_i is not None and (up_i is None or dn_i <= up_i))


def _semantic_direction_hit(row, minutes):
    value = row[f"future_return_{minutes}m"]
    if pd.isna(value):
        return 0
    is_risk = row.get("alert_type") in RISK_TYPES or row.get("alert_grade") in RISK_GRADES
    return int(value < 0) if is_risk else int(value > 0)


def evaluate_alerts(selected_alerts, bars, labels=None, *, entry_delay_minutes=3):
    """Evaluate selected alerts from future bars; labels are eval-only."""
    if selected_alerts is None or len(selected_alerts) == 0:
        return pd.DataFrame(columns=list(selected_alerts.columns) + EVAL_COLUMNS if selected_alerts is not None else EVAL_COLUMNS)

    selected = _ensure_date_time(selected_alerts)
    by_day = day_arrays(bars)
    label_map = {}
    if labels is not None and len(labels):
        label_cols = [c for c in LABEL_COLUMNS if c in labels.columns]
        label_map = {
            (r["symbol"], r["decision_time"]): {c: r[c] for c in label_cols}
            for r in labels[["symbol", "decision_time"] + label_cols].to_dict("records")
        }

    rows = []
    for row in selected.to_dict("records"):
        arr = by_day.get((row["symbol"], row["date"]))
        if arr is None:
            continue
        prep = prepare_entry(
            arr,
            pd.Timestamp(row["decision_time"]),
            float(row["close"]),
            delay_bars=entry_delay_minutes,
        )
        if prep is None:
            continue
        entry_price, highs, lows, closes, offsets = prep
        out = dict(row)
        out["entry_delay_minutes"] = int(entry_delay_minutes)
        out["entry_price"] = float(entry_price)
        out["future_return_30m"] = _window_return(entry_price, closes, offsets, 30)
        out["future_return_60m"] = _window_return(entry_price, closes, offsets, 60)
        out["future_return_120m"] = _window_return(entry_price, closes, offsets, 120)
        out["future_return_eod"] = float(closes[-1] / entry_price - 1) if len(closes) else np.nan
        out["MFE"] = float(highs.max() / entry_price - 1) if len(highs) else np.nan
        out["MAE"] = float(lows.min() / entry_price - 1) if len(lows) else np.nan
        out["hit_up2_60m"] = _hit_up(entry_price, highs, offsets, 0.02, 60)
        out["hit_up2_120m"] = _hit_up(entry_price, highs, offsets, 0.02, 120)
        out["hit_up3_120m"] = _hit_up(entry_price, highs, offsets, 0.03, 120)
        out["down2_60m"] = _hit_down(entry_price, lows, offsets, 0.02, 60)
        out["stop_first"] = _stop_first(entry_price, highs, lows)

        label_values = label_map.get((row["symbol"], row["decision_time"]), {})
        out["UP_FAIL"] = int(label_values.get("up5_event") == "UP_FAIL")
        for col, value in label_values.items():
            out[f"label_{col}"] = value

        for minutes in (30, 60, 120):
            out[f"direction_hit_{minutes}m"] = _semantic_direction_hit(out, minutes)
        rows.append(out)

    return pd.DataFrame(rows)


def _rate(frame, col):
    if frame is None or len(frame) == 0 or col not in frame.columns:
        return 0.0
    return float(frame[col].mean())


def _mean(frame, col):
    if frame is None or len(frame) == 0 or col not in frame.columns:
        return 0.0
    value = frame[col].mean()
    return float(value) if pd.notna(value) else 0.0


def summarize_alerts(evaluated, *, by):
    """Summarize evaluated alerts by columns such as selection_mode+grade/type."""
    columns = list(by) if isinstance(by, (list, tuple)) else [by]
    if evaluated is None or len(evaluated) == 0:
        return pd.DataFrame()

    rows = []
    for keys, group in evaluated.groupby(columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(columns, keys)}
        n = int(len(group))
        row.update({
            "n": n,
            "return_30m": round(_mean(group, "future_return_30m"), 6),
            "return_60m": round(_mean(group, "future_return_60m"), 6),
            "return_120m": round(_mean(group, "future_return_120m"), 6),
            "return_eod": round(_mean(group, "future_return_eod"), 6),
            "direction_hit_30m": round(_rate(group, "direction_hit_30m"), 6),
            "direction_hit_60m": round(_rate(group, "direction_hit_60m"), 6),
            "direction_hit_120m": round(_rate(group, "direction_hit_120m"), 6),
            "MFE": round(_mean(group, "MFE"), 6),
            "MAE": round(_mean(group, "MAE"), 6),
            "hit_up2_60m_rate": round(_rate(group, "hit_up2_60m"), 6),
            "hit_up2_120m_rate": round(_rate(group, "hit_up2_120m"), 6),
            "hit_up3_120m_rate": round(_rate(group, "hit_up3_120m"), 6),
            "down2_60m_rate": round(_rate(group, "down2_60m"), 6),
            "UP_FAIL_rate": round(_rate(group, "UP_FAIL"), 6),
            "stop_first_rate": round(_rate(group, "stop_first"), 6),
            "symbol_concentration": round(float(group["symbol"].value_counts().max() / n), 6) if n else 0.0,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_quality_checks(grade_summary, type_summary):
    """Small pass/fail table for the Step 13_6 report criteria."""
    rows = []
    online_grades = grade_summary[grade_summary.get("selection_mode") == "online"] if len(grade_summary) else pd.DataFrame()

    def grade_value(grade, col):
        if online_grades.empty:
            return np.nan
        hit = online_grades[online_grades["alert_grade"] == grade]
        return float(hit.iloc[0][col]) if len(hit) else np.nan

    a60 = grade_value("A", "direction_hit_60m")
    b60 = grade_value("B", "direction_hit_60m")
    c60 = grade_value("C", "direction_hit_60m")
    ar60 = grade_value("A", "return_60m")
    br60 = grade_value("B", "return_60m")
    cr60 = grade_value("C", "return_60m")
    amae = grade_value("A", "MAE")
    cmae = grade_value("C", "MAE")
    rmae = grade_value("RISK", "MAE")
    rdown = grade_value("RISK", "down2_60m_rate")
    rupfail = grade_value("RISK", "UP_FAIL_rate")

    rows.append({"check": "A_direction_hit_60m_gt_B_C", "passed": int(a60 > b60 and a60 > c60), "value": f"A={a60:.4f}, B={b60:.4f}, C={c60:.4f}"})
    rows.append({"check": "A_return_60m_gt_B_C", "passed": int(ar60 > br60 and ar60 > cr60), "value": f"A={ar60:.6f}, B={br60:.6f}, C={cr60:.6f}"})
    rows.append({"check": "A_MAE_less_bad_than_C_RISK", "passed": int(amae > cmae and amae > rmae), "value": f"A={amae:.6f}, C={cmae:.6f}, RISK={rmae:.6f}"})
    rows.append({"check": "RISK_has_downside_or_upfail", "passed": int((rdown >= max(grade_value('A', 'down2_60m_rate'), grade_value('B', 'down2_60m_rate'))) or (rupfail >= max(grade_value('A', 'UP_FAIL_rate'), grade_value('B', 'UP_FAIL_rate')))), "value": f"RISK_down2={rdown:.4f}, RISK_UP_FAIL={rupfail:.4f}"})

    if type_summary is not None and len(type_summary):
        online_types = type_summary[type_summary["selection_mode"] == "online"]
        up = online_types[online_types["alert_type"].isin(["UP_ALERT", "UP_CONTINUATION"])]
        c_ref = c60
        up_hit = float(up["direction_hit_60m"].mean()) if len(up) else np.nan
        rows.append({"check": "UP_types_direction_hit_gt_C_grade", "passed": int(up_hit > c_ref), "value": f"UP_types={up_hit:.4f}, C={c_ref:.4f}"})
    return pd.DataFrame(rows)
