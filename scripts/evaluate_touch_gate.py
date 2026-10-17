"""Touch-study gate: CNN/TCN challengers vs the B2 LightGBM champion (+B0/B1).

Research / evaluation only -- no orders, no Kiwoom, no SendOrder. Parquet-only: every
model's fold predictions (written by scripts/train_touch_model.py, labels + prob +
prob_b0/prob_b1 included) are stitched and judged; nothing is retrained here.

Pre-registered decision structure (plan v3, USER modifications #4/#5 applied):
  1. Architecture winner: the higher recent (2026-05+06) stitched AUC among the
     challengers, using SELECTION folds 1-4 (2026-03..06) ONLY.
  2. GO gate (winner vs B2, folds 1-4 stitched): recent AUC delta >= +0.010 AND
     worst recent-month delta > -0.005 AND two-sided buckets (top-decile touch rate
     >= B2 - 0.010 AND bottom-20% touch rate <= B2 + 0.005; eps sized from the
     recent window's binomial SE) AND months-up > 1.
  3. Confirmation (fold 5 = 2026-07, a PARTIAL month never used for selection):
     the winner's July AUC delta vs B2 must stay > -0.005, else adoption is held.
  4. 000660 fine-tune: on the FT symbol's validation rows pooled across all folds,
     adopt FT iff AUC(FT) - AUC(pooled winner) >= +0.010 (scratch arm reported).

Run:
  ..\\Python311-64\\python.exe scripts\\evaluate_touch_gate.py --ft-symbol <code>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

import risk_model as rm
import touch_model as tmm
from config import PROCESSED_DIR
from utils import now_kst_text

CHALLENGERS = ("cnn", "tcn")
CHAMPION = "lgbm"
GATE_RECENT_AUC_MIN = 0.010
GATE_WORST_MONTH_FLOOR = -0.005
TOP_DECILE_EPS = 0.010      # ~1.4 binomial SE at the recent window's top-decile N
BOTTOM_EPS = 0.005
CONFIRM_FLOOR = -0.005      # fold-5 (July) confirmation floor
FT_ADOPT_MIN = 0.010


def parse_args():
    p = argparse.ArgumentParser(description="Touch-study walk-forward gate.")
    p.add_argument("--ft-symbol", default="",
                   help="the fine-tune symbol (names the _ft/_scratch parquets and "
                        "the per-symbol section).")
    p.add_argument("--out-suffix", default="",
                   help="read the artifacts of a diagnostic namespace (smoke).")
    return p.parse_args()


def _pred_path(model, fold, *, mode="pretrain", ft_symbol="", out_suffix=""):
    mode_sfx = {"pretrain": "", "finetune": f"_ft_{ft_symbol}",
                "scratch": f"_scratch_{ft_symbol}"}[mode]
    return PROCESSED_DIR / f"touch_predictions_{model}{mode_sfx}{out_suffix}_fold{fold}.parquet"


def load_fold(model, fold, *, mode="pretrain", ft_symbol="", out_suffix="",
              required=True):
    path = _pred_path(model, fold, mode=mode, ft_symbol=ft_symbol,
                      out_suffix=out_suffix)
    if not path.exists():
        if required:
            raise SystemExit(f"missing predictions: {path.name} -- run "
                             f"scripts/train_touch_model.py --model {model}"
                             + (f" --mode {mode}" if mode != "pretrain" else "")
                             + " first.")
        return None, None
    df = pd.read_parquet(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
    df["fold"] = fold
    return df, meta


def assert_aligned(champ, chall, model, fold):
    """Same rows, same labels -- the parquets must describe one shared population."""
    if len(champ) != len(chall):
        raise SystemExit(f"fold {fold}: row count differs (lgbm={len(champ)}, "
                         f"{model}={len(chall)})")
    m = champ[["symbol", "decision_time", tmm.TOUCH_TARGET]].merge(
        chall[["symbol", "decision_time", tmm.TOUCH_TARGET]],
        on=["symbol", "decision_time"], validate="one_to_one",
        suffixes=("", "_c"))
    if len(m) != len(champ) or not bool(
            (m[tmm.TOUCH_TARGET] == m[f"{tmm.TOUCH_TARGET}_c"]).all()):
        raise SystemExit(f"fold {fold}: label/population mismatch vs {model}")


def auc(df, prob_col):
    y = df[tmm.TOUCH_TARGET].astype(int).to_numpy()
    return rm.roc_auc_score_np(y, df[prob_col].to_numpy(dtype=float))


def bucket_rates(df, prob_col):
    p = df[prob_col].to_numpy(dtype=float)
    y = df[tmm.TOUCH_TARGET].astype(int).to_numpy()
    top = rm.top_fraction_mask(p, 0.10)
    bottom = p <= np.quantile(p, 0.20)
    return {"top_decile_rate": float(y[top].mean()),
            "bottom20_rate": float(y[bottom].mean()),
            "top_decile_n": int(top.sum())}


def monthly_auc(df, prob_col):
    return {m: auc(g, prob_col) for m, g in df.groupby("month", sort=True)}


def metrics(df, prob_col):
    out = {"n": int(len(df)), "auc": auc(df, prob_col),
           "recent_auc": auc(df[df["month"].isin(tmm.RECENT_MONTHS)], prob_col),
           "monthly": monthly_auc(df, prob_col), **bucket_rates(df, prob_col)}
    aux = df.copy()
    y7 = aux[tmm.TOUCH_TARGET_AUX].astype(int).to_numpy()
    out["auc_aux_07"] = rm.roc_auc_score_np(y7, aux[prob_col].to_numpy(dtype=float))
    out["auc_early"] = auc(df[df["early_flag"] == 1], prob_col) \
        if int((df["early_flag"] == 1).sum()) > 50 else float("nan")
    out["auc_rest"] = auc(df[df["early_flag"] != 1], prob_col)
    out["auc_cluster_first"] = auc(df[df["cluster_first"] == True], prob_col)  # noqa: E712
    out["band_auc"] = {b: auc(g, prob_col) for b, g in df.groupby("band", observed=True)
                       if len(g) > 200}
    return out


def decide(base, variant):
    """The pre-registered GO decision on the folds-1..4 stitched frame."""
    d_recent = variant["recent_auc"] - base["recent_auc"]
    deltas = {m: variant["monthly"][m] - base["monthly"][m]
              for m in variant["monthly"] if m in base["monthly"]}
    recent_deltas = {m: d for m, d in deltas.items() if m in tmm.RECENT_MONTHS}
    worst_recent = min(recent_deltas.values()) if recent_deltas else float("nan")
    n_months_up = sum(1 for d in deltas.values() if d > 0)
    top_ok = variant["top_decile_rate"] >= base["top_decile_rate"] - TOP_DECILE_EPS
    bottom_ok = variant["bottom20_rate"] <= base["bottom20_rate"] + BOTTOM_EPS
    go = (d_recent >= GATE_RECENT_AUC_MIN and worst_recent > GATE_WORST_MONTH_FLOOR
          and top_ok and bottom_ok and n_months_up > 1)
    reasons = []
    if d_recent < GATE_RECENT_AUC_MIN:
        reasons.append(f"recent AUC Δ {d_recent:+.4f} < +{GATE_RECENT_AUC_MIN:.3f}")
    if not (worst_recent > GATE_WORST_MONTH_FLOOR):
        reasons.append(f"worst recent month Δ {worst_recent:+.4f} <= {GATE_WORST_MONTH_FLOOR:+.3f}")
    if not top_ok:
        reasons.append("top-decile touch rate worse than B2 beyond eps")
    if not bottom_ok:
        reasons.append("bottom-20% touch rate worse than B2 beyond eps")
    if n_months_up <= 1:
        reasons.append(f"improvement not broad ({n_months_up} month(s) up)")
    verdict = "GO" if go else ("PARTIAL" if d_recent > 0 and (top_ok or bottom_ok)
                               else "NO-GO")
    return {"verdict": verdict, "d_recent": d_recent, "worst_recent": worst_recent,
            "n_months_up": n_months_up, "top_ok": top_ok, "bottom_ok": bottom_ok,
            "deltas": deltas, "reasons": reasons}


def main() -> int:
    args = parse_args()
    print(f"[{now_kst_text()}] touch gate: champion=B2(lgbm) challengers={CHALLENGERS} "
          f"selection_folds={tmm.SELECTION_FOLDS} confirmation_fold={tmm.CONFIRMATION_FOLD} "
          f"recent={tmm.RECENT_MONTHS} ft_symbol={args.ft_symbol or '(none)'}")

    frames = {}
    metas = {}
    for model in (CHAMPION,) + CHALLENGERS:
        per_fold = {}
        for fold in range(1, len(tmm.TOUCH_FOLDS) + 1):
            df, meta = load_fold(model, fold, ft_symbol=args.ft_symbol,
                                 out_suffix=args.out_suffix)
            per_fold[fold] = df
            metas[(model, fold)] = meta
        frames[model] = per_fold
    # cross-artifact seals: shared population/labels + identical FT-tail exclusion
    for fold in range(1, len(tmm.TOUCH_FOLDS) + 1):
        champ_meta = metas[(CHAMPION, fold)]
        for model in CHALLENGERS:
            assert_aligned(frames[CHAMPION][fold], frames[model][fold], model, fold)
            m = metas[(model, fold)]
            for key in ("target", "threshold", "ft_symbol", "ft_tail_dates"):
                if m.get(key) != champ_meta.get(key):
                    raise SystemExit(f"fold {fold}: sidecar {key} mismatch between "
                                     f"lgbm and {model} -- populations differ.")

    def stitch(model, folds):
        return pd.concat([frames[model][f] for f in folds], ignore_index=True)

    sel_folds = list(tmm.SELECTION_FOLDS)
    champ_sel = stitch(CHAMPION, sel_folds)
    results = {"b0": metrics(champ_sel, "prob_b0"), "b1": metrics(champ_sel, "prob_b1"),
               "b2": metrics(champ_sel, "prob")}
    for model in CHALLENGERS:
        results[model] = metrics(stitch(model, sel_folds), "prob")

    # 1. architecture winner (selection folds only)
    winner = max(CHALLENGERS, key=lambda m: results[m]["recent_auc"])
    # 2. GO gate winner vs champion
    verdict = decide(results["b2"], results[winner])
    # 3. fold-5 confirmation (partial July; never used above)
    july = {m: auc(frames[m][tmm.CONFIRMATION_FOLD], "prob")
            for m in (CHAMPION,) + CHALLENGERS}
    july_delta = july[winner] - july[CHAMPION]
    confirmed = july_delta > CONFIRM_FLOOR
    # 4. FT section (pooled across every fold's validation, FT-symbol rows)
    ft_section = None
    if args.ft_symbol:
        pooled_all = pd.concat([frames[winner][f] for f in frames[winner]],
                               ignore_index=True)
        sym = pooled_all[pooled_all["symbol"] == args.ft_symbol]
        champ_all = pd.concat([frames[CHAMPION][f] for f in frames[CHAMPION]],
                              ignore_index=True)
        champ_sym = champ_all[champ_all["symbol"] == args.ft_symbol]
        ft_frames, sc_frames = [], []
        for fold in range(1, len(tmm.TOUCH_FOLDS) + 1):
            f_df, _ = load_fold(winner, fold, mode="finetune",
                                ft_symbol=args.ft_symbol,
                                out_suffix=args.out_suffix, required=False)
            s_df, _ = load_fold(winner, fold, mode="scratch",
                                ft_symbol=args.ft_symbol,
                                out_suffix=args.out_suffix, required=False)
            if f_df is not None:
                ft_frames.append(f_df[f_df["symbol"] == args.ft_symbol])
            if s_df is not None:
                sc_frames.append(s_df[s_df["symbol"] == args.ft_symbol])
        ft_section = {
            "n": int(len(sym)),
            "auc_b2": auc(champ_sym, "prob"),
            "auc_pooled": auc(sym, "prob"),
            "auc_ft": (auc(pd.concat(ft_frames, ignore_index=True), "prob")
                       if ft_frames else float("nan")),
            "auc_scratch": (auc(pd.concat(sc_frames, ignore_index=True), "prob")
                            if sc_frames else float("nan")),
            "ft_folds": len(ft_frames), "scratch_folds": len(sc_frames),
        }
        ft_section["ft_delta"] = ft_section["auc_ft"] - ft_section["auc_pooled"]
        ft_section["ft_adopt"] = (ft_section["ft_delta"] >= FT_ADOPT_MIN
                                  if ft_section["ft_delta"] == ft_section["ft_delta"]
                                  else False)

    # ---- report -----------------------------------------------------------------
    months = sorted(results["b2"]["monthly"])
    lines = [f"# touch model gate — B2(LightGBM) vs {'/'.join(CHALLENGERS)} "
             f"(P2r +{tmm.TOUCH_THRESHOLD:.1%} close-touch, {tmm.TOUCH_WINDOW_MIN}m)\n"]
    lines.append(f"**Winner: {winner} — verdict {verdict['verdict']}; July confirmation: "
                 f"{'CONFIRMED' if confirmed else 'NOT CONFIRMED (adoption held)'}"
                 + (f"; FT adopt: {ft_section['ft_adopt']}" if ft_section else "") + "**\n")
    lines.append("Selection = folds 1-4 (2026-03..06); fold 5 (2026-07, PARTIAL month) "
                 "is confirmation-only. Probabilities are class-weighted and "
                 "UNCALIBRATED — ranks only.\n")
    lines.append("## Stitched metrics (selection folds)\n")
    lines.append("| model | pooled AUC | recent(05+06) | " + " | ".join(months)
                 + " | top-decile touch | bottom-20% touch | early AUC | rest AUC |")
    lines.append("|" + "---|" * (7 + len(months)))
    for name in ("b0", "b1", "b2") + CHALLENGERS:
        r = results[name]
        lines.append(
            f"| {name} | {r['auc']:.4f} | {r['recent_auc']:.4f} | "
            + " | ".join(f"{r['monthly'].get(m, float('nan')):.4f}" for m in months)
            + f" | {r['top_decile_rate']:.3f} | {r['bottom20_rate']:.3f} | "
            + (f"{r['auc_early']:.4f}" if r["auc_early"] == r["auc_early"] else "n/a")
            + f" | {r['auc_rest']:.4f} |")
    lines.append("")
    lines.append(f"- lift sanity: B1−B0 recent Δ {results['b1']['recent_auc'] - results['b0']['recent_auc']:+.4f}, "
                 f"B2−B1 recent Δ {results['b2']['recent_auc'] - results['b1']['recent_auc']:+.4f} "
                 "(B2가 밴드표 이상을 배우는지)")
    for model in CHALLENGERS:
        lines.append(f"- {model} band-wise AUC (dist를 알고도 남는 판별): "
                     + ", ".join(f"{b}={v:.4f}" for b, v in
                                 sorted(results[model]["band_auc"].items())))
        lines.append(f"- {model} cluster-first-only AUC {results[model]['auc_cluster_first']:.4f}; "
                     f"aux +0.7% label AUC {results[model]['auc_aux_07']:.4f}")
    lines.append("")
    lines.append("## Gate (winner vs B2, pre-registered)\n")
    lines.append(f"| winner | recent Δ | worst recent-month Δ | months up | top ok | bottom ok | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| **{winner}** | {verdict['d_recent']:+.4f} | {verdict['worst_recent']:+.4f} | "
                 f"{verdict['n_months_up']} | {verdict['top_ok']} | {verdict['bottom_ok']} | "
                 f"**{verdict['verdict']}** |")
    if verdict["reasons"]:
        lines.append("\n- " + "; ".join(verdict["reasons"]))
    lines.append("")
    lines.append("## Fold-5 confirmation (2026-07, partial month — diagnostic floor)\n")
    lines.append("| model | July AUC |")
    lines.append("|---|---|")
    for m in (CHAMPION,) + CHALLENGERS:
        lines.append(f"| {m} | {july[m]:.4f} |")
    lines.append(f"\n- winner Δ vs B2 = {july_delta:+.4f} (floor {CONFIRM_FLOOR:+.3f}) → "
                 + ("CONFIRMED" if confirmed else "NOT CONFIRMED — adoption held"))
    if ft_section:
        lines.append("")
        lines.append(f"## {args.ft_symbol} fine-tune (validation rows pooled over all folds)\n")
        lines.append("| slice | n | B2 | pooled winner | FT | scratch |")
        lines.append("|---|---|---|---|---|---|")
        f = ft_section
        lines.append(f"| {args.ft_symbol} | {f['n']:,} | {f['auc_b2']:.4f} | "
                     f"{f['auc_pooled']:.4f} | "
                     + (f"{f['auc_ft']:.4f}" if f["auc_ft"] == f["auc_ft"] else "n/a")
                     + " | "
                     + (f"{f['auc_scratch']:.4f}" if f["auc_scratch"] == f["auc_scratch"] else "n/a")
                     + " |")
        lines.append(f"\n- FT Δ vs pooled = "
                     + (f"{f['ft_delta']:+.4f}" if f["ft_delta"] == f["ft_delta"] else "n/a")
                     + f" (adopt ≥ +{FT_ADOPT_MIN:.3f}) → adopt={f['ft_adopt']} "
                     f"(FT folds present: {f['ft_folds']}/{len(tmm.TOUCH_FOLDS)})")
    lines.append("")
    report_path = PROCESSED_DIR / f"touch_gate_report{args.out_suffix}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    rows = []
    for name in ("b0", "b1", "b2") + CHALLENGERS:
        r = results[name]
        rows.append({"model": name, "pooled_auc": r["auc"], "recent_auc": r["recent_auc"],
                     "top_decile_rate": r["top_decile_rate"],
                     "bottom20_rate": r["bottom20_rate"],
                     "july_auc": july.get("lgbm" if name == "b2" else name),
                     "verdict": verdict["verdict"] if name == winner else ""})
    csv_path = PROCESSED_DIR / f"touch_gate{args.out_suffix}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\nwinner={winner} verdict={verdict['verdict']} "
          f"recentΔ={verdict['d_recent']:+.4f} worstΔ={verdict['worst_recent']:+.4f} "
          f"julyΔ={july_delta:+.4f} ({'CONFIRMED' if confirmed else 'HELD'})")
    if ft_section:
        print(f"FT: pooled={ft_section['auc_pooled']:.4f} ft="
              + (f"{ft_section['auc_ft']:.4f}" if ft_section['auc_ft'] == ft_section['auc_ft'] else "n/a")
              + f" adopt={ft_section['ft_adopt']}")
    print(f"report -> {report_path}")
    print(f"csv -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
