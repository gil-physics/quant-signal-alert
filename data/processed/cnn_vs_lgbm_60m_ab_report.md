# risk_model v5_cnn_60m (TCN, walk-forward) vs v1_60m — A/B report & GO/NO-GO

**Verdict: GO — subject to the Phase-2 serving-parity work before any live wiring.**

> ⚠ **fold1: DIAGNOSTIC predictions: parquet seed columns=(42, 43, 44, 45, 46), sidecar submission_run=False, sidecar seeds=[42, 43, 44, 45, 46] (pre-registered submission = (42,), submission_run=True). The verdict below is a pipeline check, NOT an adoption decision — rerun the full pre-registered training with the default config before treating any GO as real. | fold2: DIAGNOSTIC predictions: parquet seed columns=(42, 43, 44, 45, 46), sidecar submission_run=False, sidecar seeds=[42, 43, 44, 45, 46] (pre-registered submission = (42,), submission_run=True). The verdict below is a pipeline check, NOT an adoption decision — rerun the full pre-registered training with the default config before treating any GO as real. | fold3: DIAGNOSTIC predictions: parquet seed columns=(42, 43, 44, 45, 46), sidecar submission_run=False, sidecar seeds=[42, 43, 44, 45, 46] (pre-registered submission = (42,), submission_run=True). The verdict below is a pipeline check, NOT an adoption decision — rerun the full pre-registered training with the default config before treating any GO as real. | fold4: DIAGNOSTIC predictions: parquet seed columns=(42, 43, 44, 45, 46), sidecar submission_run=False, sidecar seeds=[42, 43, 44, 45, 46] (pre-registered submission = (42,), submission_run=True). The verdict below is a pipeline check, NOT an adoption decision — rerun the full pre-registered training with the default config before treating any GO as real.**

Step 17 lineage: fixed-horizon label `target_mae_risk_60m` = next-60-minute window min/max (main population = decision time-of-day <= 14:30:00, i.e. always exactly 60 minutes observed; later rows are the late-session diagnostic below, never gated). Champion = `v1_60m` (v1 features + minutes_since_open); challenger = the TCN with its time-of-day channel KEPT — both models hold the clock, so the gate measures architecture value on top of clock access.

**Walk-forward protocol (user-decided):** expanding train windows, one validation month per fold, ONLY the periods differ between folds; the TCN's pre-registered rule is seed(s) (42, 43, 44, 45, 46) per fold (no seed selection). Pooled/recent metrics below STITCH the four out-of-sample months — every row is scored by a model that never saw its month.

## 0. Per-fold out-of-sample AUC (the regime-robustness read)

| fold | train window | val month | n (main) | pos_rate | v1_60m AUC | v5_cnn_60m AUC | Δ |
|---|---|---|---|---|---|---|---|
| 1 | 2025-06-02→2026-02-28 | 2026-03 | 35,769 | 0.224 | 0.7377 | 0.7466 | +0.0089 |
| 2 | 2025-06-02→2026-03-31 | 2026-04 | 33,290 | 0.094 | 0.7336 | 0.7587 | +0.0252 |
| 3 | 2025-06-02→2026-04-30 | 2026-05 | 35,395 | 0.205 | 0.7168 | 0.7202 | +0.0034 |
| 4 | 2025-06-02→2026-05-31 | 2026-06 | 28,689 | 0.224 | 0.6430 | 0.6730 | +0.0300 |

## 1. Discrimination (stitched walk-forward validation AUC)

| | pooled | recent(05+06) | 2026-03 | 2026-04 | 2026-05 | 2026-06 |
|---|---|---|---|---|---|---|
| v1_60m | 0.7066 | 0.6656 | 0.7377 | 0.7336 | 0.7168 | 0.6430 |
| v5_cnn_60m | 0.7304 | 0.6971 | 0.7466 | 0.7587 | 0.7202 | 0.6730 |
| Δ | +0.0237 | +0.0316 | +0.0089 | +0.0251 | +0.0034 | +0.0300 |

## 2–3. Risk-bucket separation + MAE (stitched)

| model | top10% rate | top10% avg future_MAE_60m | bottom20% rate | decile monotonicity |
|---|---|---|---|---|
| v1_60m | 0.416 | -0.0162 | 0.052 | 1.00 |
| v5_cnn_60m | 0.461 | -0.0171 | 0.047 | 1.00 |

## 4. A-grade low-risk quality (the trading payoff)

MAE margin for the gate's mae_ok check: **0.0003** (user-decided; scale-matched to the tighter 60m label — 60m/EOD future_MAE std ratio ≈ 0.59 × the legacy 0.0005).

| model | n_A | n (low half) | avg ret60 | avg future_MAE_60m | win-rate | risk-rate |
|---|---|---|---|---|---|---|
| v1_60m | 34672 | 17336 | -0.0003 | -0.0070 | 0.438 | 0.089 |
| v5_cnn_60m | 34672 | 17336 | -0.0002 | -0.0068 | 0.444 | 0.083 |

## 5. Late-session diagnostic (time > cutoff; truncated-horizon label — never gated)

| scope | n | pos_rate | v1_60m AUC | v5_cnn_60m AUC |
|---|---|---|---|---|
| pooled | 13,531 | 0.128 | 0.6694 | 0.7287 |
| fold 1 | 3,772 | 0.213 | 0.6798 | 0.7160 |
| fold 2 | 3,263 | 0.018 | 0.7545 | 0.7951 |
| fold 3 | 3,438 | 0.110 | 0.7839 | 0.8307 |
| fold 4 | 3,058 | 0.161 | 0.6993 | 0.7151 |

## 6. Calibration (reported, never gated)

| model | brier | logloss | unique-score ratio |
|---|---|---|---|
| v1_60m | 0.2525 | 0.7036 | 0.817 |
| v5_cnn_60m | 0.2718 | 0.7581 | 0.999 |

## 7. Ablations (single fold each, diagnostic only — never gated)

| ablation | fold | n (main) | AUC | champion same-fold AUC | Δ |
|---|---|---|---|---|---|
| tcn_5seed_fold1 | 1 | 35,769 | 0.7314 | 0.7377 | -0.0063 |
| tcn_5seed_fold2 | 2 | 33,290 | 0.7617 | 0.7336 | +0.0282 |
| tcn_5seed_fold3 | 3 | 35,395 | 0.7188 | 0.7168 | +0.0020 |
| tcn_5seed_fold4 | 4 | 28,689 | 0.6729 | 0.6430 | +0.0300 |

## Gate (same recent-weighted math as v2.x/v3.0; walk-forward stitched)

GO requires: recent-month(05+06) AUC Δ ≥ +0.010 vs v1_60m AND no recent month regresses past -0.005 AND risk-bucket separation preserved AND A-grade low-risk MAE not worse than -0.0003 AND the gain is broad (>1 month up).

| variant | recent Δ | worst recent-month Δ | months up | bucket ok | A-grade MAE ok | verdict |
|---|---|---|---|---|---|---|
| **v5_cnn_60m** | +0.0316 | +0.0034 | 4 | True | True | **GO** |


## Live serving status

- **v1.1 remains the adopted live config** (the 60m lineage is research-only; even its champion v1_60m is not wired live).
- GO does not mean live: Phase 2 (sequence serving, parity replay, band re-thresholding) must pass before any wiring.
