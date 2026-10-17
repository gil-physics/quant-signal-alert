# Morning 🔔 signal → afternoon extreme validation (000660)

- **Date range:** 2025-06-02 → 2026-06-26
- **Trading days (with bars):** 261
- **Days with a valid morning signal:** 232
- **Selection mode:** `first`   **Alerts source:** `realtime_replay`
- **Afternoon window:** 13:00–15:30   **Signal cutoff:** before 11:00

Afternoon high/low is hindsight — this measures predictive POTENTIAL, not a tradable exit.

## Main UP result
- n = 104 signal-days
- avg afternoon_high_return = +1.02% (median +0.88%)
- hit ≥0.5% / 1.0% / 1.5% / 2.0% = 0.606 / 0.462 / 0.375 / 0.308
- avg afternoon_close_return = +0.14%
- avg pre-13 MFE = +1.62%, avg pre-13 MAE = -1.60% (adverse move before 13:00)
- avg full-day MFE after signal = +1.96%, MAE = -1.95%

## Main DOWN result  _(RISK_ALERT treated as SHORT — `--risk-as-short`)_
- n = 128 signal-days
- avg short_potential = +0.68% (median +0.51%); baseline last_bar_before_11 = +1.10%
- hit ≥0.5%/1.0%/1.5%/2.0% = 0.500 / 0.359 / 0.258 / 0.211
- avg pre-13 adverse (price rose vs short) = +1.76%

## Baseline comparison (avg afternoon_high_return)
| source | n | avg high-return | median | avg pre-13 MAE |
|---|---|---|---|---|
| **🔔 UP signal** | 104 | +1.02% | +0.88% | -1.60% |
| last_bar_before_11 | 261 | +1.01% | +0.74% | -0.98% |
| first_bar_after_10_30 | 261 | +1.10% | +0.74% | -1.11% |
| random_morning_bar | 261 | +1.22% | +0.85% | -1.25% |
| all_morning_bars_sample | 1305 | +1.18% | +0.85% | -1.31% |

- **UP signals beat baseline (last_bar_before_11) on afternoon_high_return?** YES
- **DOWN signals beat baseline on short_potential?** see CSV
- **Pre-13 adverse move tolerable?** avg pre-13 MAE = -1.60%
- **Rows with missing-data notes:** 0

## Verdict (UP / long): **FAIL**

- UP avg afternoon_high_return +0.0102 vs baseline +0.0101 (edge +0.0001)
- UP median +0.0088 vs baseline median +0.0074
- avg pre-13 MAE -0.0160 (tolerable >= -0.012)
- UP signal days n=104 (min 20)

## Verdict (DOWN / short): **FAIL**

- DOWN avg short_potential +0.0068 vs baseline +0.0110 (edge -0.0042)
- DOWN median +0.0051 vs baseline median +0.0073
- avg pre-13 MAE +0.0176 (tolerable <= +0.012)
- DOWN signal days n=128 (min 20)

_PASS = beats baseline on mean+median favorable return, enough days, pre-13 adverse tolerable. PARTIAL = potential exists but one criterion fails. FAIL = does not beat baseline._
