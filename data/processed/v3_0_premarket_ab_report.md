# risk_model v3.0a / v3.0b vs v1.1 — premarket-context A/B report & GO/NO-GO

**Verdict: NO-GO for both v3.0a and v3.0b. Keep v1.1 as the live config.**

v3.0 keeps the **v1.1 baseline** (the v1 model's 8 cross-symbol features served on the REAL cross-section) and ATTACHES per-day **premarket (08:00–09:00)** context to every same-day regular-session row (joined by trade_date; the premarket window closes before the open → causal). Labels and the direction rule are unchanged.

- **v3.0a** (`v3_0a_premarket_000660`) — 000660 premarket only (10 premarket features).
- **v3.0b** (`v3_0b_premarket_universe`) — 000660 + universe-wide premarket breadth/relative/rank (23 premarket features).

All models scored on the IDENTICAL validation frame (146,674 rows, 2026-03-01→2026-06-26, positive_rate 0.348). "v1.1" = the v1 model file scored on real cross-symbol features — the adopted live config.

## 1. Discrimination (validation AUC; higher = better)

| | pooled | recent(05+06) | 2026-03 | 2026-04 | 2026-05 | 2026-06 |
|---|---|---|---|---|---|---|
| v1.1 | 0.6614 | 0.6327 | 0.6814 | 0.6944 | 0.6716 | 0.5963 |
| v3_0a_premarket_000660 | 0.6303 | 0.6344 | 0.5772 | 0.6375 | 0.6193 | 0.6491 |
| v3_0b_premarket_universe | 0.6180 | 0.5753 | 0.6324 | 0.6450 | 0.5615 | 0.5990 |
| Δ v3_0a_premarket_000660 vs v1.1 | -0.0311 | +0.0018 | -0.1042 | -0.0569 | -0.0523 | +0.0528 |
| Δ v3_0b_premarket_universe vs v1.1 | -0.0433 | -0.0573 | -0.0490 | -0.0494 | -0.1101 | +0.0027 |

## 2–3. Risk-bucket separation + MAE

| model | top10% mae_risk_rate | top10% avgMAE | bottom20% mae_risk_rate | decile monotonicity |
|---|---|---|---|---|
| v1.1 | 0.534 | -0.0225 | 0.145 | 1.00 |
| v3_0a_premarket_000660 | 0.518 | -0.0215 | 0.189 | 1.00 |
| v3_0b_premarket_universe | 0.453 | -0.0194 | 0.184 | 0.78 |

## 4. A-grade low-risk quality (the trading payoff)

Among grade-A alerts the model marks low risk_prob (median split):

| model | n | avg ret60 | avg MAE | win-rate | mae_risk_rate |
|---|---|---|---|---|---|
| v1.1 | 19017 | -0.0004 | -0.0112 | 0.436 | 0.233 |
| v3_0a_premarket_000660 | 19017 | -0.0007 | -0.0119 | 0.431 | 0.254 |
| v3_0b_premarket_universe | 19017 | -0.0006 | -0.0121 | 0.430 | 0.265 |

## 5. Premarket-specific diagnostics

| variant | pm features | top pm importance (feature: rank) | missing-pm rate | AUC days-with-pm | AUC days-without-pm | days <30 pm bars |
|---|---|---|---|---|---|---|
| v3_0a_premarket_000660 | 10 | pm_000660_last_5m_ret: #1, pm_000660_close_to_vwap: #2, pm_000660_range: #3 | 0.0000 | 0.6303 | n/a | 0 |
| v3_0b_premarket_universe | 23 | pm_000660_last_5m_ret: #1, pm_universe_std_ret: #2, pm_universe_mean_ret: #3 | 0.0000 | 0.6180 | n/a | 0 |

_The 2026-03→06 validation window has full target premarket coverage, so the with/without-premarket AUC split is uninformative here (the 11 missing-data days fall earlier in the sample). Premarket features dominate the importance ranks yet **degrade** generalisation — a day-broadcast feature (constant within a day) lets the model fit day-regime noise, which is why recent-month AUC and the low-risk bucket get worse, not better._

## Gate (same recent-weighted gate as v2.0/v2.1)

GO requires: recent-month(05+06) AUC Δ ≥ +0.010 vs v1.1 AND no recent month regresses past -0.005 AND risk-bucket separation preserved AND A-grade low-risk MAE not worse AND the gain is broad (>1 month up).

| variant | recent Δ | worst recent-month Δ | months up | bucket ok | A-grade MAE ok | verdict |
|---|---|---|---|---|---|---|
| **v3_0a_premarket_000660** | +0.0018 | -0.0523 | 1 | False | False | **NO-GO** |
| **v3_0b_premarket_universe** | -0.0573 | -0.1101 | 1 | False | False | **NO-GO** |

- **v3_0a_premarket_000660** (NO-GO): recent-month AUC Δ +0.0018 < +0.010; worst recent month Δ -0.0523 <= -0.005; risk-bucket separation not preserved; A-grade low-risk MAE worse; improvement not broad (1 month(s) up).
- **v3_0b_premarket_universe** (NO-GO): recent-month AUC Δ -0.0573 < +0.010; worst recent month Δ -0.1101 <= -0.005; risk-bucket separation not preserved; A-grade low-risk MAE worse; improvement not broad (1 month(s) up).

## Live serving status

- **v1.1 remains the adopted live config.**
- v3.0 variants are research-only until they pass the gate.
- If a v3.0 variant ever passes, realtime serving must compute the SAME premarket features (08:00–09:00, same definitions) before the open and join them by date (`requires_premarket=True` in the model metadata) — this is **not wired** yet.
- v3.0c / v3.0d (005930 / 402340 premarket anchors) are not built in this round.
