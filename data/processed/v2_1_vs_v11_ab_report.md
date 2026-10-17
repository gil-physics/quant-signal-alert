# risk_model v2.1a / v2.1b vs v1.1 — A/B report & GO/NO-GO verdict

**Verdict: NO-GO for both v2.1a and v2.1b. Keep v1.1 (v1 model served on real
cross-symbol features) as the live config.** v2.1b (402340 leader) is the strongest v2
variant to date but still fails the recent-month gate.

v2.1x are SINGLE-leader lead-lag hypotheses, each built on the **v1.1 baseline** (v1's 8
cross-symbol features + ONE leader's 2 lead-lag columns), NOT on v2.0's rejected
universe-wide set. A "leader's lead-lag" = that symbol's already-closed 5m/20m return as
of decision_time D, broadcast onto every symbol's row at D (contemporaneous →
leakage-safe). Leaders come from `config.relative.leaders` (label→code), never
hard-coded; columns are `leader_<label>_ret_5m_1m` / `_20m_1m`.

- **v2.1a** — leader **a = 005930 (삼성전자)**, the broad semiconductor leader.
- **v2.1b** — leader **b = 402340 (SK스퀘어)**, which holds a large SK하이닉스 (000660)
  stake, so its price tracks 000660 directly — a stronger structural lead-lag candidate.

Direction (rule alert) unchanged. All models scored on the IDENTICAL validation frame
(145,647 rows, 2026-03→06, positive_rate 0.348). "v1.1" = the v1 model file
(`risk_model_lgbm.txt`) scored on REAL cross-symbol features — the adopted live config.
Archived: `models/risk_model_v2_1a_lgbm.txt`, `models/risk_model_v2_1b_lgbm.txt`.

## 1. Discrimination (validation AUC; higher = better)

| | pooled | 2026-03 | 2026-04 | 2026-05 | 2026-06 |
|---|---|---|---|---|---|
| v1.1  | 0.6611 | 0.6814 | 0.6944 | 0.6716 | 0.5920 |
| v2.1a | 0.6582 | 0.6766 | 0.6945 | 0.6708 | 0.5861 |
| v2.1b | 0.6623 | 0.6832 | 0.6964 | 0.6729 | 0.5864 |
| Δ v2.1a vs v1.1 | −0.0029 | −0.0048 | +0.0001 | −0.0008 | **−0.0059** |
| Δ v2.1b vs v1.1 | +0.0012 | +0.0018 | +0.0020 | +0.0013 | **−0.0056** |

v2.1a (Samsung) adds essentially nothing. v2.1b (SK Square) is the most encouraging v2
result so far — it improves **three of four months** (Mar/Apr/May) and regresses only in
June; that single month sinks both gates.

## 2-3. Risk-bucket separation + MAE

| model | top10% mae_risk_rate | top10% avgMAE | bottom20% mae_risk_rate | bottom20% avgMAE |
|---|---|---|---|---|
| v1.1  | 0.533 | −0.0225 | 0.146 | −0.0083 |
| v2.1a | 0.532 | −0.0224 | 0.147 | −0.0083 |
| v2.1b | 0.543 | −0.0229 | **0.150** | −0.0085 |

v2.1b's top-10% high-risk bucket is sharper (0.543 vs 0.533) but its bottom-20% "low
risk" bucket is **worse** (0.150 vs 0.146 — more blow-ups slipped into the low-risk
picks), so the separation gain is not clean.

## 4. A-grade low-risk quality (the trading payoff — flat)

Among grade-A alerts the model marks low risk_prob (median split, n≈18,900):

| model | avg ret60 | avg MAE | win-rate(ret60>0) | mae_risk_rate |
|---|---|---|---|---|
| v1.1  | −0.0004 | −0.0112 | 0.438 | 0.233 |
| v2.1a | −0.0004 | −0.0112 | 0.435 | 0.235 |
| v2.1b | −0.0003 | −0.0110 | 0.438 | 0.232 |

## Gate (same recent-weighted gate as v2.0)

| | recent-month (05/06) AUC Δ | worst of last 3 months Δ | pooled Δ | verdict |
|---|---|---|---|---|
| gate | ≥ +0.010 | > −0.005 | — | |
| **v2.1a** | −0.0033 → FAIL | −0.0059 → FAIL | −0.0029 | **NO-GO** |
| **v2.1b** | **−0.0021** → FAIL | **−0.0055** → FAIL | +0.0012 | **NO-GO** |

Feature importance: `leader_b_ret_20m_1m` (SK Square) ranks **#10 (importance 41)** — a
genuine, non-trivial signal — whereas `leader_a_ret_20m_1m` (Samsung) ranks ~#20 (14),
largely redundant with `relative_ret_20m_to_universe`. The 5m leg of both leaders is
negligible.

## Conclusion

The **005930 lead-lag (v2.1a) is dead** — Samsung's move adds nothing v1 doesn't already
encode. The **402340 lead-lag (v2.1b) is the most promising cross-symbol idea tried**
(real feature importance, positive in 3/4 months, pooled +0.0012), consistent with SK
Square's structural link to 000660 — but it still **does not clear the gate**: the
recent-month AUC Δ is −0.0021 (needs ≥ +0.010) and June regresses past the −0.005 floor,
the same regime-overfit pattern as v2.0/v2.1a, just milder. Risk-bucket and A-grade
low-risk quality are not convincingly better. **v1.1 stays the live model; v2.1a and
v2.1b are archived, neither wired to live.** If this thread is revisited, 402340 (not
universe-wide features, not 005930) is the lead worth pursuing — e.g. a shorter forward
horizon or regime-conditioned use that survives the June drawdown regime.
