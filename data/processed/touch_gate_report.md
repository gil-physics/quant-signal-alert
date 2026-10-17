# touch model gate — B2(LightGBM) vs cnn/tcn (P2r +0.5% close-touch, 10m)

**Winner: cnn — verdict GO; July confirmation: CONFIRMED; FT adopt: False**

Selection = folds 1-4 (2026-03..06); fold 5 (2026-07, PARTIAL month) is confirmation-only. Probabilities are class-weighted and UNCALIBRATED — ranks only.

## Stitched metrics (selection folds)

| model | pooled AUC | recent(05+06) | 2026-03 | 2026-04 | 2026-05 | 2026-06 | top-decile touch | bottom-20% touch | early AUC | rest AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| b0 | 0.5392 | 0.5355 | 0.4999 | 0.4996 | 0.5265 | 0.5148 | 0.218 | 0.152 | 0.5467 | 0.5381 |
| b1 | 0.6318 | 0.6012 | 0.6133 | 0.6460 | 0.6057 | 0.5853 | 0.296 | 0.084 | 0.5924 | 0.6198 |
| b2 | 0.7149 | 0.6607 | 0.7349 | 0.7384 | 0.7388 | 0.6242 | 0.347 | 0.038 | 0.6286 | 0.7173 |
| cnn | 0.7502 | 0.7128 | 0.7382 | 0.7725 | 0.7465 | 0.6800 | 0.390 | 0.022 | 0.6784 | 0.7515 |
| tcn | 0.7496 | 0.7097 | 0.7402 | 0.7675 | 0.7434 | 0.6807 | 0.398 | 0.024 | 0.6850 | 0.7505 |

- lift sanity: B1−B0 recent Δ +0.0657, B2−B1 recent Δ +0.0596 (B2가 밴드표 이상을 배우는지)
- cnn band-wise AUC (dist를 알고도 남는 판별): far=0.6495, mid=0.7029, near=0.7565
- cnn cluster-first-only AUC 0.7512; aux +0.7% label AUC 0.7916
- tcn band-wise AUC (dist를 알고도 남는 판별): far=0.6600, mid=0.7027, near=0.7553
- tcn cluster-first-only AUC 0.7507; aux +0.7% label AUC 0.7904

## Gate (winner vs B2, pre-registered)

| winner | recent Δ | worst recent-month Δ | months up | top ok | bottom ok | verdict |
|---|---|---|---|---|---|---|
| **cnn** | +0.0521 | +0.0077 | 4 | True | True | **GO** |

## Fold-5 confirmation (2026-07, partial month — diagnostic floor)

| model | July AUC |
|---|---|
| lgbm | 0.5336 |
| cnn | 0.6433 |
| tcn | 0.6395 |

- winner Δ vs B2 = +0.1097 (floor -0.005) → CONFIRMED

## 000660 fine-tune (validation rows pooled over all folds)

| slice | n | B2 | pooled winner | FT | scratch |
|---|---|---|---|---|---|
| 000660 | 1,746 | 0.6278 | 0.7429 | 0.7456 | 0.7349 |

- FT Δ vs pooled = +0.0027 (adopt ≥ +0.010) → adopt=False (FT folds present: 4/5)
