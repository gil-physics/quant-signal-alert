# risk_model v2.0 vs v1 — A/B report & GO/NO-GO verdict

**Verdict: NO-GO. Keep v1 (served via v1.1 real cross-symbol features).**

v2.0 = v1's 8 cross-symbol features + 5 richer universe features
(`universe_disp_ret_20m`, `universe_breadth_5m`, `universe_breadth_60m`,
`ret_20m_gap_to_best`, `universe_disp_volume_z`). Direction (rule alert) unchanged.
Both models scored on the IDENTICAL validation frame (145,647 rows, 2026-03→06,
positive_rate 0.348). Model file kept at `models/risk_model_v2_0_lgbm.txt`.

## 1. Discrimination (validation AUC; higher = better)

| | pooled | 2026-03 | 2026-04 | 2026-05 | 2026-06 |
|---|---|---|---|---|---|
| v1 | 0.6611 | 0.6814 | 0.6944 | 0.6716 | 0.5920 |
| v2.0 | 0.6638 | 0.6992 | 0.6939 | 0.6645 | 0.5835 |
| Δ | **+0.0027** | +0.0178 | −0.0005 | **−0.0071** | **−0.0085** |

The small pooled gain is driven ENTIRELY by the oldest month (March). On the
recent regime (May/June) v2.0 REGRESSES.

## 2-3. Risk-bucket separation + MAE (essentially flat)

| model | top10% mae_risk_rate | top10% avgMAE | bottom20% mae_risk_rate | bottom20% avgMAE |
|---|---|---|---|---|
| v1 | 0.533 | −0.0225 | 0.146 | −0.0083 |
| v2.0 | 0.539 | −0.0227 | 0.146 | −0.0083 |

## 4. A-grade low-risk quality (the trading payoff — no improvement)

Among grade-A alerts the model marks low risk_prob (median split, n≈18,900):

| model | avg ret60 | avg MAE | win-rate(ret60>0) | mae_risk_rate |
|---|---|---|---|---|
| v1 | −0.0004 | −0.0112 | 0.438 | 0.233 |
| v2.0 | −0.0003 | −0.0109 | 0.435 | 0.233 |

## Gate (decided up front, recent-weighted)

- recent-month (2026-05/06) mean AUC Δ = **−0.0078** — gate ≥ +0.010 → **FAIL**
- worst of last 3 months AUC Δ = **−0.0085** — gate > −0.005 → **FAIL**
- risk-bucket / MAE / A-grade-low-risk quality → **flat (no gain)**
- feature importance: `universe_disp_ret_20m` ranks #9 (67.0) — the dispersion idea
  has *some* signal, but `gap_to_best`/`breadth_60m` are negligible (~6), and the
  net effect overfits the older regime rather than helping recent data.

## Conclusion

The richer universe features mostly re-express information v1 already captures and
do not survive the documented regime decay (AUC 0.69→0.58 into June). Per the gate,
**v1 stays the model; v1.1 (real cross-symbol serving) is the adopted live config.**
v2.0 is archived for the record.
