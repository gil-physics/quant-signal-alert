# SHORT_ENTRY gate — holdout GO/NO-GO report (Step 15)

**Verdict: NO-GO — no combination passed the holdout gate.** SHORT_ENTRY defaults stay UNVALIDATED (`DEFAULTS_VALIDATED=False`); the runner remains `--experimental` only. Mean-reversion history (Step 13_5) stands.

The top 5 train-ranked combinations from the threshold sweep (`C:\OpenAPI\quant\data\processed\short_gate_sweep_summary_20260703_150935.csv`) were re-gated on the pre-gate replay candidates (`db:short_replay_candidates`, 28,994 rows) through the same offline path the sweep uses (rule gate → model veto → session window → causal cooldown/daily-limit selection) and judged **on the holdout window only** (2026-03-01 → 2026-06-26; train ranking window ends 2026-02-28). Forward metrics follow the project execution model: entry = close of the decision + delay bar; raw close returns, so **negative = short profit**.

Baselines the gate must beat:

1. **Time-of-day-matched** — unconditional entry at the same HH:MM:SS on every other holdout day (paired per fire; edge = alert − baseline, negative = short edge).
2. **Frozen RISK_ALERT-as-short** (Step 13_5, alert_type_summary.csv) — fwd60 +0.13%, dir-hit 39.6%: down alerts historically mean-reverted.

## 1. Top train-ranked combinations (train ≤ 2026-02-28)

| rank | combo | train n | avg fwd60 | dir-hit60 | stop-first |
|---|---|---|---|---|---|
| 1 | `down6_up2_m5_rankNone_veto0.45` | 47 | -0.03% | 44.7% | 14.9% |
| 2 | `down6_up2_m5_rank7_veto0.45` | 47 | -0.03% | 44.7% | 14.9% |
| 3 | `down6_up2_m5_rank5_veto0.45` | 45 | -0.02% | 44.4% | 15.6% |
| 4 | `down4_up2_m5_rankNone_veto0.45` | 75 | -0.02% | 44.0% | 12.0% |
| 5 | `down4_up2_m5_rank7_veto0.45` | 75 | -0.02% | 44.0% | 12.0% |

## 2. Holdout metrics (2026-03-01 → 2026-06-26)

| combo | n | days | avg fwd60 | baseline fwd60 | paired edge | t | dir-hit60 | stop-first | avg MAE | down2_60m | +months |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `down6_up2_m5_rankNone_veto0.45` | 21 | 16 | +0.08% | -0.07% | +0.15% | +0.69 | 33.3% | 28.6% | +1.08% | 9.5% | 2 |
| `down6_up2_m5_rank7_veto0.45` | 21 | 16 | +0.08% | -0.07% | +0.15% | +0.69 | 33.3% | 28.6% | +1.08% | 9.5% | 2 |
| `down6_up2_m5_rank5_veto0.45` | 20 | 15 | +0.15% | -0.07% | +0.21% | +1.00 | 30.0% | 30.0% | +1.09% | 10.0% | 2 |
| `down4_up2_m5_rankNone_veto0.45` | 27 | 19 | +0.03% | -0.05% | +0.08% | +0.46 | 37.0% | 22.2% | +0.99% | 11.1% | 2 |
| `down4_up2_m5_rank7_veto0.45` | 27 | 19 | +0.03% | -0.05% | +0.08% | +0.46 | 37.0% | 22.2% | +0.99% | 11.1% | 2 |

## 3. Gate criteria (ALL must hold on holdout for GO)

- n ≥ 40 evaluable fires on ≥ 20 distinct days
- avg fwd60 ≤ -0.10% AND paired edge ≤ -0.15% AND paired t ≤ -2
- dir-hit60 ≥ 55.0%
- stop-first ≤ 35.0% AND avg short_mae ≤ +1.50%
- down2_60m ≥ 8.0%
- train avg fwd60 < 0 (sign consistency) AND ≤ 1 positive holdout month(s)

PARTIAL = beats BOTH baselines while exactly one of {t_stat, n, stop_first} fails (every other criterion passes).

## 4. Verdicts

| combo | verdict | beats ToD baseline | beats RISK_ALERT baseline | failed criteria |
|---|---|---|---|---|
| `down6_up2_m5_rankNone_veto0.45` | **NO-GO** | False | False | n, avg_fwd60, paired_edge, t_stat, dir_hit60, monthly |
| `down6_up2_m5_rank7_veto0.45` | **NO-GO** | False | False | n, avg_fwd60, paired_edge, t_stat, dir_hit60, monthly |
| `down6_up2_m5_rank5_veto0.45` | **NO-GO** | False | False | n, avg_fwd60, paired_edge, t_stat, dir_hit60, monthly |
| `down4_up2_m5_rankNone_veto0.45` | **NO-GO** | False | False | n, avg_fwd60, paired_edge, t_stat, dir_hit60, monthly |
| `down4_up2_m5_rank7_veto0.45` | **NO-GO** | False | False | n, avg_fwd60, paired_edge, t_stat, dir_hit60, monthly |

## Status

- `short_entry.DEFAULTS_VALIDATED` stays `False`; the realtime runner keeps refusing to fire without `--experimental` + explicit thresholds.
- Project convention: the failed configuration is documented here and NOT wired live.
- Generated 2026-07-03 15:10:19 by scripts/evaluate_short_entry_gate.py (research only, no orders).
