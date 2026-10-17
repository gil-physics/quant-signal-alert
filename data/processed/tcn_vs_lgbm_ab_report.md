# risk_model v4_tcn (TCN seed ensemble) vs v1.1 — A/B report & GO/NO-GO

**Verdict: GO — subject to the Phase-2 serving-parity work before any live wiring.**

v4_tcn is an ARCHITECTURE challenger: same rows (directional-alert candidates), same `target_mae_risk` label, same date split as v1.1 — but instead of the 49 flat aggregated features it consumes a causal window of the last 60 one-minute bars' per-bar features (15 channels incl. time-of-day + pad mask) through dilated causal convolutions (Bai et al. TCN), fusing the v1 static features (5 alert + 8 cross-symbol) in the head. decision_time = the CONFIRMED feature bar's bar_end; the window's newest element is that bar and the label's forward window starts at bar_start >= decision_time, so window and label never overlap.

**Pre-registered seed rule** (fixed in src/tcn_model.py BEFORE any validation scoring): seeds (42, 43, 44, 45, 46), submission = arithmetic MEAN of the per-seed probabilities. No seed selection of any kind; per-seed dispersion is reported as a diagnostic below.

Both models scored on the IDENTICAL validation frame (146,674 rows, 2026-03-01→2026-06-26, positive_rate 0.348). "v1.1" = the v1 model file scored on real cross-symbol features — the adopted live config.

## 1. Discrimination (validation AUC; higher = better)

| | pooled | recent(05+06) | 2026-03 | 2026-04 | 2026-05 | 2026-06 |
|---|---|---|---|---|---|---|
| v1.1 | 0.6614 | 0.6327 | 0.6814 | 0.6944 | 0.6716 | 0.5963 |
| v4_tcn (ensemble) | 0.6986 | 0.6698 | 0.7140 | 0.7423 | 0.7007 | 0.6490 |
| Δ v4_tcn vs v1.1 | +0.0373 | +0.0372 | +0.0326 | +0.0479 | +0.0291 | +0.0527 |

## 2–3. Risk-bucket separation + MAE

| model | top10% mae_risk_rate | top10% avgMAE | bottom20% mae_risk_rate | decile monotonicity |
|---|---|---|---|---|
| v1.1 | 0.534 | -0.0225 | 0.145 | 1.00 |
| v4_tcn (ensemble) | 0.600 | -0.0260 | 0.110 | 1.00 |

## 4. A-grade low-risk quality (the trading payoff)

Among grade-A alerts the model marks low risk_prob (each model's OWN median split; `n` asymmetry vs half of n_A signals heavy score ties):

| model | n_A | n (low half) | avg ret60 | avg MAE | win-rate | mae_risk_rate |
|---|---|---|---|---|---|---|
| v1.1 | 38034 | 19017 | -0.0004 | -0.0112 | 0.436 | 0.233 |
| v4_tcn (ensemble) | 38034 | 19017 | +0.0000 | -0.0103 | 0.452 | 0.207 |

## 5. TCN-specific diagnostics (informational — the gate math is unchanged)

### Seed dispersion (pre-registered ensemble; no seed was selected)

| seed | pooled AUC | recent AUC | Δ recent vs v1.1 |
|---|---|---|---|
| 42 | 0.7010 | 0.6694 | +0.0368 |
| 43 | 0.6965 | 0.6648 | +0.0321 |
| 44 | 0.7027 | 0.6705 | +0.0378 |
| 45 | 0.6812 | 0.6636 | +0.0309 |
| 46 | 0.6948 | 0.6641 | +0.0315 |

- seed majority sign: **5/5 seeds have Δ recent > 0**
- recent-AUC std across seeds: 0.0029
- median-seed Δ recent: +0.0321

### Calibration (reported, never gated — both models are class-weighted and emit uncalibrated probabilities)

| model | brier | logloss | unique-score ratio | score deciles (q0..q10) |
|---|---|---|---|---|
| v1.1 | 0.2431 | 0.6797 | 0.934 | 0.18 0.33 0.41 0.46 0.50 0.54 0.57 0.61 0.65 0.70 0.83 |
| v4_tcn | 0.2431 | 0.6852 | 0.999 | 0.01 0.17 0.30 0.42 0.52 0.58 0.64 0.69 0.74 0.81 1.00 |

Reliability by own-score decile (mean predicted prob vs observed positive rate):

| decile | v1.1 pred | v1.1 obs | v4_tcn pred | v4_tcn obs |
|---|---|---|---|---|
| 1 | 0.2812 | 0.1003 | 0.0995 | 0.0704 |
| 2 | 0.3725 | 0.1906 | 0.2355 | 0.1501 |
| 3 | 0.4337 | 0.2515 | 0.3624 | 0.2309 |
| 4 | 0.4828 | 0.3051 | 0.4711 | 0.2955 |
| 5 | 0.5220 | 0.3678 | 0.5513 | 0.3408 |
| 6 | 0.5559 | 0.3911 | 0.6110 | 0.3897 |
| 7 | 0.5913 | 0.4121 | 0.6621 | 0.4271 |
| 8 | 0.6320 | 0.4546 | 0.7135 | 0.4596 |
| 9 | 0.6758 | 0.4706 | 0.7730 | 0.5129 |
| 10 | 0.7425 | 0.5336 | 0.8724 | 0.6001 |

_The serving bands RISK_HIGH=0.60 / RISK_MED=0.45 are tuned to the LightGBM score distribution and do NOT transfer to the TCN (see the score deciles above); on a GO, Phase 2 re-derives TCN thresholds by matching v1.1's band occupancy on this validation frame._

### Ablations (single seed, diagnostic only — never gated)

| ablation | pooled AUC | recent AUC | Δ recent vs v1.1 | top10% rate | bottom20% rate | monotonicity |
|---|---|---|---|---|---|---|
| ablA_no_static | 0.7028 | 0.6740 | +0.0414 | 0.6266 | 0.1118 | 1.0000 |
| ablB_seq120 | 0.7030 | 0.6742 | +0.0415 | 0.6012 | 0.1100 | 1.0000 |
| ablC_no_tod | 0.6735 | 0.6396 | +0.0069 | 0.5831 | 0.1407 | 1.0000 |

_ablA (no static head) isolates how much of the TCN's discrimination comes from the sequence itself vs riding the v1 static features; ablB (seq_len 120) checks whether the 60-bar window is the bottleneck._

## Gate (same recent-weighted gate as v2.x/v3.0)

GO requires: recent-month(05+06) AUC Δ ≥ +0.010 vs v1.1 AND no recent month regresses past -0.005 AND risk-bucket separation preserved AND A-grade low-risk MAE not worse AND the gain is broad (>1 month up).

| variant | recent Δ | worst recent-month Δ | months up | bucket ok | A-grade MAE ok | verdict |
|---|---|---|---|---|---|---|
| **v4_tcn** | +0.0372 | +0.0291 | 4 | True | True | **GO** |


## Live serving status

- **v1.1 remains the adopted live config.**
- The TCN challenger is research-only until it passes the gate; the configuration evaluated here is documented above and NOT wired live.
- GO does not mean live: Phase 2 (sequence-serving builder parallel to score_serving, a score-parity replay check, and band re-thresholding by v1.1-occupancy matching) must pass before any wiring.
