# quant

Korean-equities intraday data pipeline. The canonical store for all downstream
work is the SQLite table **`bars_1m`** in `db/market_data.sqlite`.

## Pipeline overview

```
download_kiwoom_minute.py ──► minute_data/<code>[_AL]_1min.csv ──┐
   (Kiwoom opt10080, 32-bit)                                     │  (option A: CSV)
                                                                 ▼
                                          import_minute_csv.py / collect_historical_1m.py ──► bars_1m
                                                                 ▲
   (Kiwoom opt10080, 32-bit) ─────────────────────────────────► │  (option B: direct, no CSV)
                                          collect_kiwoom_to_bars.py
```

Every `bars_1m` row carries a `symbol` and a `source`. Three source series per
symbol (see Step 2):

| source | meaning | window |
|--------|---------|--------|
| `al_raw` | integrated quote (통합시세), raw storage | 08:00–19:59 |
| `al_regular` | integrated quote filtered to the regular session — **model input** | 09:00–15:30 |
| `single_regular` | single-exchange regular file — comparison only | regular file as-is |

### Runtimes
- `..\Python311-32` — Kiwoom OpenAPI+ work (32-bit ActiveX, requires Windows + login).
- `..\Python311-64` — everything else (imports, tests, analysis).
- Dependencies: [requirements.txt](requirements.txt) (PyQt5, pandas 2.0.3, numpy<2, pyarrow, python-dateutil, PyYAML). Keep NumPy < 2 in the 32-bit runtime.

---

## Step 0: Project baseline

Fix the project standards in one place so every later step reads from the same
source of truth — stored in [config/config.yaml](config/config.yaml).

| Item | Value |
|------|-------|
| Source data | 1-minute OHLCV bars |
| Derived data | 3-minute, 5-minute bars |
| Training window | 09:00:00 – 15:30:00 (regular session) |
| Raw storage window | 08:00:00 – 20:00:00 |
| Model input | regular-session clean data only |
| Orders | no automatic ordering |
| Alerts | at most 1–2 signals per day |

`config.yaml` sections: `session` (times), `signals` (alert limits), and
`data` (source classification: `model_source: al_regular`, `raw_source: al_raw`,
`comparison_source: single_regular`).

```powershell
..\Python311-64\python.exe scripts\show_config.py
```

## Step 1: Fixed symbol list

Pin traded symbols in [config/symbols_test.yaml](config/symbols_test.yaml)
instead of computing market cap or hard-coding codes in source. ~20 candidates
are listed; **005930 (삼성전자)** and **000660 (SK하이닉스)** are `active: true`.
Activate more by setting `active: true`.

Never hard-code a symbol code — load it:

```python
from settings import load_symbols, active_symbols, default_symbol, load_active_symbols

load_symbols()          # all symbols, typed (codes are stripped of whitespace)
active_symbols()        # only active: true
load_active_symbols()   # active symbol codes (list[str]) — the multi-symbol iteration target
default_symbol()        # first active code, e.g. "005930"
```

Notes:
- YAML loading prefers PyYAML and falls back to a built-in parser, so the 32-bit Kiwoom runtime keeps working.
- A guard test fails if any 6-digit symbol code is hard-coded in `src/` or `scripts/`.

## Step 2: Unified bar tables + CSV import

Standardize all data into symbol-keyed tables **`bars_1m` / `bars_3m` / `bars_5m`**
with `PRIMARY KEY (symbol, bar_start, source)`. (The plan's `(symbol, bar_start)`
key is extended with `source` so the three overlapping series coexist; the
completion query below is unaffected.)

- `datetime` (YYYYMMDDHHMMSS) → `bar_start` (`YYYY-MM-DD HH:MM:SS`), `bar_end` = +1 minute.
- Raw CSVs live in `..\minute_data\<code>_1min.csv` (single) and `<code>_AL_1min.csv` (integrated).

Import one symbol's CSVs into `bars_1m` (idempotent; re-running inserts nothing new):

```powershell
..\Python311-64\python.exe scripts\import_minute_csv.py --symbol 000660
# subsets / explicit file:
..\Python311-64\python.exe scripts\import_minute_csv.py --symbol 000660 --jobs al_raw,al_regular
..\Python311-64\python.exe scripts\import_minute_csv.py --csv path\to\file.csv --symbol 000660 --source al_raw
```

Completion check (from step2.md):

```sql
SELECT symbol, COUNT(*) FROM bars_1m GROUP BY symbol;
```

`bars_3m` / `bars_5m` are populated by Step 5 (resampling from 1m).

## Step 3: Multi-symbol historical collection

Iterate active symbols and store each symbol's history into `bars_1m`. Every row
gets `symbol`; existing `(symbol, bar_start, source)` is never duplicated; a
per-symbol failure is logged and skipped; START/DONE lines go to
`logs/collect_1m.log`; a missing/coverage report is written to
`data/processed/`.

```python
def collect_1m(symbol, start_date=None, end_date=None): ...   # src/historical_collector.py
for symbol in load_active_symbols():
    collect_1m(symbol, start_date, end_date)
```

### Option A — from CSV (input = minute_data CSVs)

```powershell
..\Python311-64\python.exe scripts\collect_historical_1m.py                       # active symbols
..\Python311-64\python.exe scripts\collect_historical_1m.py --symbols 373220,028260
..\Python311-64\python.exe scripts\collect_historical_1m.py --symbols 000270 --start-date 2025-06-01 --end-date 2026-06-19
```

### Option B — direct from Kiwoom, no CSV

Reuses the downloader in `..\download_kiwoom_minute.py`. Per symbol it fetches
`<code>_AL` → `al_raw` + `al_regular`, and `<code>` → `single_regular`, writing
straight into `bars_1m`. **Requires 32-bit Python and Kiwoom login.**

```powershell
..\Python311-32\python.exe scripts\collect_kiwoom_to_bars.py --symbols 373220,028260
..\Python311-32\python.exe scripts\collect_kiwoom_to_bars.py --symbols 000270 --no-single   # skip comparison fetch
..\Python311-32\python.exe scripts\collect_kiwoom_to_bars.py --symbols 000270 --from-date 20250602 --to-date 20260619
```

Options: `--al-suffix` (default `_AL` for the integrated code), `--max-pages`,
`--delay-sec`, `--adjusted`, `--no-single`, `--db`. Fetching all symbols × both
codes × ~1 year is many pages — bound it with `--from-date` / `--no-single`.

## Step 4: Source policy + al_regular validation

Finalize the source policy and validate the model data — **no large-scale
transform**, just policy + a report.

- **`config.data.model_source = al_regular`** is the fixed model source.
- Source roles are documented in [src/bars_reader.py](src/bars_reader.py): `al_raw` = raw storage, `al_regular` = model input, `single_regular` = comparison only.
- Downstream steps read the model series by default:
  ```python
  from bars_reader import load_bars, load_model_bars
  df = load_bars("000660")        # source defaults to config.data.model_source (al_regular)
  df = load_bars("000660", source="single_regular", start_date="2025-06-01")  # explicit override
  ```
- Anomaly report over `al_regular` (records issues; **does not drop** — exclusion is decided in Step 6):

  ```powershell
  ..\Python311-64\python.exe scripts\validate_model_data.py
  ..\Python311-64\python.exe scripts\validate_model_data.py --symbols 000660 --spike-pct 0.05
  ```

  Checks per symbol: NULLs, OHLC consistency, out-of-session bars,
  `bar_end == bar_start + 1min`, duplicate `(symbol, bar_start)` (all **structural →
  SEVERE**), plus per-minute close spikes and intrabar range outliers above
  thresholds (within a trade day, so overnight gaps don't count → **REVIEW**).
  Outputs `data/processed/al_regular_validation_<ts>.csv` (per-symbol summary +
  severity + observed max move/range) and `..._details_<ts>.csv` (worst example
  rows); log at `logs/validate.log`.

## Step 5: 1m → 3m / 5m resampling

Build 3-minute and 5-minute bars from the 1-minute model series (`al_regular`),
aligned to the session open (09:00): 3m = 09:00–09:03, 09:03–09:06, …; 5m =
09:00–09:05, … Aggregation per bucket: `open`=first, `high`=max, `low`=min,
`close`=last, `volume`=sum. Only **complete** buckets are kept (all N one-minute
bars present), so the trailing partial bucket is dropped.

```python
from resampler import resample_ohlcv
df_3m = resample_ohlcv(df_1m, 3)
df_5m = resample_ohlcv(df_1m, 5)
```

Generate and store into `bars_3m` / `bars_5m` (source encodes the base, e.g.
`resampled_from_al_regular`, idempotent):

```powershell
..\Python311-64\python.exe scripts\resample_bars.py                  # all symbols present in al_regular
..\Python311-64\python.exe scripts\resample_bars.py --symbols 000660 --intervals 3,5
```

Per-day counts (consistent across symbols): 1m ≈ 380, **3m ≈ 126, 5m ≈ 76**.
These are slightly below the naive 390/130/78 because the continuous session
effectively runs 09:00–15:19 — the 15:20–15:29 **closing call auction** prints no
1-minute trade bars and 15:30 is the closing single price (an incomplete bucket,
dropped). This is expected market microstructure, not missing data.

## Step 6: Data-quality report

Automated quality checks over `bars_1m` (source `al_regular`) and `bars_3m` /
`bars_5m` (source `resampled_from_al_regular`), restricted to **active** symbols.
Reports only — never deletes or edits data.

```powershell
..\Python311-64\python.exe scripts\quality_check.py
..\Python311-64\python.exe scripts\quality_check.py --symbols 005930 --spike-pct 0.05
```

Per `(symbol, date, timeframe)` it records actual vs a **data-driven** expected
range (median daily count ± tolerance — no hardcoded 390/130/78), plus:

| status | triggers |
|--------|----------|
| **FAIL** | duplicate `bar_start`, OHLC relation error, price ≤ 0, negative volume, `bar_end` interval error, out-of-session bar, source-mixing (rows under an unexpected source) |
| **WARN** | missing bars / actual outside the expected range, abnormal close spikes, range outliers |
| **OK** | none of the above |

Output: `data/processed/quality_report_<ts>.csv`; log at `logs/quality.log`. The
completion criteria (no duplicates, no OHLC errors, valid session times/`bar_end`,
no source-mixing, no price≤0 / negative volume, missing within tolerance) are the
**FAIL** conditions — any FAIL day/symbol should be excluded or corrected before
training. Note: late-open days (e.g. CSAT/수능, the first trading day) legitimately
produce fewer bars and surface as WARN, not FAIL.

## Step 7: Per-timeframe features

Build `features_1m`, `features_3m`, `features_5m` separately (decision-time merge
is Step 8). Feature logic lives in three split modules — [src/features_1m.py](src/features_1m.py),
[src/features_3m.py](src/features_3m.py), [src/features_5m.py](src/features_5m.py) — over a shared
[src/feature_lib.py](src/feature_lib.py).

- `features_1m` ← `bars_1m` source `al_regular`; `features_3m`/`features_5m` ← `bars_3m`/`bars_5m` source `resampled_from_al_regular` (active symbols only).
- Every feature carries a **timeframe suffix** (`ret_5m_1m`, `ret_1bar_3m`, `volume_z_20bar_5m`, …) and each table keeps `bar_end_<tf>` (the feature's usable time); `features_1m` also keeps `decision_time` (= `bar_end_1m`) and `close`.
- **No leakage**: every rolling/shift/expanding op is grouped by `(symbol, trade date)` and looks strictly backward — no day-boundary crossing, no future bars.
- Initial-window rolling NaNs are reported then dropped (output has 0 NaN).
- **5-day moving averages** (two variants, for comparison) — both use only the **prior 5 trading days** (today excluded), so they are leakage-safe:
  - `ma5_daily` — mean of the prior 5 daily closes (last regular bar of each day); **constant across the trading day**. `dist_ma5_daily = close/ma5_daily − 1`.
  - `ma5_intraday` — mean of the **same minute-of-day** close on the prior 5 trading days (e.g. at 10:23, the average of D-5…D-1 at 10:23); **varies by minute**. `dist_ma5_intraday = close/ma5_intraday − 1`.
  - `ma5_daily_n` / `ma5_intraday_n` report how many prior days contributed (5 once warmed up).
  - Unlike the other features, these cross day boundaries and so are **NaN for the first ~5 trading days**; they are deliberately kept **out of `FEATURE_COLUMNS`** so the warmup-row drop never removes a row on their account (the v0 row set is unchanged). They flow through to `features_merged`; `features_relative`/the v0 rule are untouched. The breakout signal that consumes them is a separate rule version (v1).

```powershell
..\Python311-64\python.exe scripts\build_features.py                 # active symbols
..\Python311-64\python.exe scripts\build_features.py --symbols 005930
```

Outputs: SQLite tables `features_1m/3m/5m` + `data/processed/features_<tf>.parquet`; log `logs/features_build.log`. Each table records a provenance `source` column.

Locked invariants (run after building; exits non-zero on violation):

```powershell
..\Python311-64\python.exe scripts\check_feature_invariants.py
```

- `features_1m.source = al_regular`; `features_3m`/`features_5m.source = resampled_from_al_regular`
- `features_1m.decision_time == features_1m.bar_end_1m` (exactly equal)

## Step 8: Leakage-safe feature merge

Merge `features_1m`/`3m`/`5m` onto the 1-minute `decision_time` grid so that no
not-yet-complete higher-timeframe bar leaks into the present. For each
`(symbol, decision_time)` the **most recent** 3m and 5m feature with
`bar_end <= decision_time` is attached (backward as-of join, **per symbol and per
day**, equality allowed). The per-day key prevents borrowing the previous day's
last feature for a new day's early decision times.

```powershell
..\Python311-64\python.exe scripts\merge_features.py
..\Python311-64\python.exe scripts\merge_features.py --drop-incomplete   # drop rows still missing 3m/5m
```

- Active symbols only; `features_1m` source `al_regular`, `features_3m`/`5m` source `resampled_from_al_regular`.
- After merge a **leakage check** asserts every attached `bar_end_3m/5m <= decision_time` (run aborts on any violation).
- Residual NaN is reported (early-day decision times before a day's first 3m/5m feature legitimately have none — availability, not leakage).
- Output: SQLite `features_merged` + `data/processed/features_merged.parquet`; log `logs/merge.log`.

## Step 9: Cross-symbol relative-strength features

From `features_merged`, compare the active symbols against each other **at the same
`decision_time`** (cross-sectional — no future data, only contemporaneous merged
features). One row per `(symbol, decision_time)`:

```powershell
..\Python311-64\python.exe scripts\build_relative_features.py
..\Python311-64\python.exe scripts\build_relative_features.py --drop-partial   # drop times lacking the full universe
```

- `rank_ret_5m_1m`, `rank_ret_20m_1m`, `rank_volume_z_20m_1m` — rank within the decision_time, **1 = highest** (competition ranking; ties share rank 1).
- `relative_ret_20m_to_universe`, `relative_volume_z_to_universe` — value minus the universe (same-time) mean.
- `universe_mean_ret_20m`, `universe_breadth` (share with `ret_20m_1m > 0`), `active_symbol_count` (symbols present at that time).
- Works with 2 symbols; a `decision_time` lacking the full active universe is reported as WARN (or dropped with `--drop-partial`).
- Output: SQLite `features_relative` + `data/processed/features_relative.parquet`; log `logs/relative.log`.

## Step 10: Forward-looking labels

For each `(symbol, decision_time, close_t)` from `features_relative`, look at FUTURE
bars (`bar_start >= decision_time`, current bar excluded) from `bars_1m al_regular`
and build labels. Labels live in their **own** `labels` table — no future value ever
enters a feature.

```powershell
..\Python311-64\python.exe scripts\make_labels.py
```

- **Binary** `up{2,3,5}_{60m,120m,180m,eod}` / `down{...}` — `future_high >= close_t*(1+t%)` / `future_low <= close_t*(1-t%)`.
- **has_full_60m/120m/180m** — whether the full horizon is observable within the same trading day (use only `has_full=1` rows for fixed-horizon training).
- **up5_event** (path-dependent long): `UP5_SUCCESS` (+5% before the −1.5% stop), `UP_FAIL` (stop first; same-bar tie → FAIL), `NO_EVENT`.
- Everything is within a single trading day (no cross-day lookahead). Output: SQLite `labels` + `data/processed/labels.parquet`; a distribution report CSV; log `logs/labels.log`.
- On the current data UP5 is sparse (≈0.07% at 60m … 1.1% at 180m), so models should start from UP2/UP3.

## Step 11: Rule-based signal v0

Build human-readable score-based signal candidates from Step 9
`features_relative`. This is not ML and it does **not** read `labels`; labels are
only for later backtesting/validation.

```powershell
..\Python311-64\python.exe scripts\build_rule_signals.py
..\Python311-64\python.exe scripts\build_rule_signals.py --up-threshold 3.5 --down-threshold 3.5
```

- Active symbols only; input rows come from `features_relative.source = al_regular`.
- Scores use only current-time features present in `features_relative`: short/20m
  returns, volume z-score, cross-sectional ranks, relative return/volume, and
  universe breadth.
- Per row outputs: `up_score`, `down_score`, `is_up_candidate`,
  `is_down_candidate`, `primary_side`, `reason_1..3`, plus full `up_reasons` /
  `down_reasons`.
- Default thresholds are deliberately low (`3.0`) so v0 produces enough
  candidates for Step 12-style backtesting; tune thresholds after seeing
  precision/recall.
- Output: SQLite `rule_signals` + `data/processed/rule_signals.parquet`; log
  `logs/rule_signal.log`.

## Step 12: Daily top1/top2 backtest

Evaluate only the 1–2 signals that would actually be selected per day, rather
than minute-level prediction accuracy. This step uses the already-built
`rule_signals` and `labels`; labels are used only for evaluation, never for
selection.

```powershell
..\Python311-64\python.exe scripts\daily_topk_backtest.py
..\Python311-64\python.exe scripts\daily_topk_backtest.py --threshold-set conservative
```

- Inputs: `rule_signals`, `labels`, and the existing `bars_1m` `al_regular`
  prices for MFE/MAE and horizon returns. It does not collect or request new
  market data.
- Basic filters: `09:10:00 <= decision_time < 15:00:00`, Step 6 FAIL dates
  excluded, outlier dates excluded from the quality report, same-symbol
  60-minute cooldown, daily max 2.
- Long filters are compared across:
  `conservative` (`up_score >= 4`, `down_score <= 1`, margin `>= 3`),
  `medium` (`up_score >= 4`, margin `>= 2`), and
  `aggressive` (`up_score >= 3`, margin `>= 1`).
- Both-side candidates (`is_up_candidate = 1` and `is_down_candidate = 1`) are
  excluded in v0 and counted in the threshold summary.
- Ranking: score margin desc, up score desc, `rank_ret_20m_1m` asc (rank 1 is
  strongest), volume z desc, earlier decision time first.
- The threshold summary includes the selection funnel:
  `raw_rows`, `after_time_filter`, `after_quality_filter`,
  `after_both_exclusion`, `after_score_filter`, `after_cooldown`,
  `final_topk`.
- Top1 and top2 outcomes are reported separately (`top1_hit_*`,
  `top2_hit_*`, MFE/MAE/stop/eod), in addition to combined top-k metrics.
- Output: SQLite `backtest_signals`, `backtest_daily_summary`,
  `backtest_threshold_summary`; `data/processed/backtest_signals.parquet`;
  daily and threshold summary CSVs; log `logs/backtest.log`.

### Step 12 entry/selection variants

`scripts/compare_entry_variants.py` (logic in `src/entry_variants.py`) fixes the
**conservative top1/top2** as the baseline and compares variants that change only
*selection* or *entry timing*. All four are scored with one evaluation method
(MFE/MAE/hits, target +5% / stop −1.5%, `stop_first` recomputed path-dependently
from each variant's actual entry price), so differences are attributable to the
variant alone. Selection and the pre-open penalty use only pre-decision info.

```powershell
..\Python311-64\python.exe scripts\compare_entry_variants.py   # cwd = quant
```

- `baseline` — conservative top-k, enter at the decision-bar close.
- `entry_delay_3m` — identical selection, but enter 3 minutes after the decision
  close (`future.iloc[3]`); forward MFE/MAE/returns are measured strictly after the
  entry bar, so the delay comparison is conservative against the delay.
- `upper_shadow` — candidate pool also requires `upper_shadow_1m <= 0.5%` (weak
  over-extension filter), then re-select top-k.
- `preopen_penalty` — pre-open (08:00–09:00 `al_raw`) weakness is a **soft** score
  penalty (`score_margin − 1.0` when `preopen_ret < 0` or `preopen_vwap_distance < 0`),
  not a hard filter; it re-ranks the daily top-k.

Result on the 15 active symbols (all top-k, `data/processed/entry_variant_summary.csv`
also carries top1/top2 breakdowns):

| variant | n | hit2 | hit3 | hit5 | mfe | mae | stop_first | eod |
|---|---|---|---|---|---|---|---|---|
| baseline | 514 | 19.26% | 11.48% | 3.11% | 1.33% | −1.54% | 37.74% | −0.19% |
| entry_delay_3m | 514 | 20.04% | 10.89% | 3.31% | 1.35% | −1.44% | 35.02% | −0.10% |
| upper_shadow | 514 | 19.46% | 11.48% | 3.11% | 1.33% | −1.53% | 37.55% | −0.18% |
| preopen_penalty | 514 | 15.95% | 8.75% | 1.75% | 1.17% | −1.51% | 36.19% | −0.22% |

- **entry_delay_3m** modestly helps: lower `stop_first` (37.7%→35.0%) and a less
  negative eod (−0.19%→−0.10%), i.e. waiting out the immediate post-signal wick
  avoids some overheated fills.
- **upper_shadow ≤0.5%** is nearly inert (only 4 of the selected slots change) —
  the strongest conservative candidates already have small upper shadows.
- **preopen_penalty** re-selects ~43% of slots yet degrades every metric, so
  pre-open strength is not predictive for this universe; keep pre-open as context,
  not as a selection signal.

The baseline row reproduces the Step 12 `conservative` numbers exactly, confirming
the variant harness is faithful to the main backtest. Re-run after expanding the
active set from 15 to 20 symbols.

## Step 13: Rule threshold sweep

Sweep rule thresholds to decide which score level is worth sending as an alert.
The default baseline is **conservative + entry_delay_3m**, with top1 as the main
view and top2 as support. Labels are joined only for evaluation; threshold
selection reads only `rule_signals` score/current-time fields.

```powershell
..\Python311-64\python.exe scripts\threshold_sweep.py
..\Python311-64\python.exe scripts\threshold_sweep.py --score-grid-only
..\Python311-64\python.exe scripts\threshold_sweep.py --percentiles-only
```

- Inputs: `rule_signals`, `labels`, and `bars_1m` `al_regular`.
- Score grid: `up_threshold` 4/5/6, `down_risk_limit` 0/1/2,
  `score_margin` 2/3/4/5, `top_k` 1/2, `entry_delay` 0/3.
- Percentile grid: conservative pool plus top 10%/5%/2%/1% by `up_score` and
  by `score_margin`; percentile cutoffs are fitted on train and applied to
  validation.
- Split defaults: train `2025-06-02` to `2026-02-28`, validation `2026-03-01`
  to `2026-06-19`.
- Delayed-entry MFE/MAE/hit/return/stop metrics are recomputed from actual
  `bars_1m` prices, so `entry_delay_3m` is evaluated from the delayed fill.
- Output: SQLite `threshold_sweep_summary` / `threshold_sweep_signals`,
  `data/processed/threshold_sweep_summary_<ts>.csv`,
  `threshold_sweep_signals_<ts>.parquet`, and a compact stable train/validation
  CSV.

## Step 13_5: Directional alert system (not auto-trading)

Prior steps showed the rule has weak auto-trading edge, so Step 13_5 **reframes
`rule_signals` as a directional ALERT feed**, not entry signals. Each row gets an
`alert_type`, an `alert_grade`, three reasons, and three scores
(`alert_score` / `risk_score` / `direction_score`) — all from features known at
`decision_time`. **No label or future price enters alert generation**; labels are
merged only for evaluation. Logic in `src/directional_alert.py`; backtest in
`scripts/online_alert_backtest.py`.

```powershell
..\Python311-64\python.exe scripts\online_alert_backtest.py   # cwd = quant
```

- **Types** (one per row, `""` = not noteworthy): `UP_ALERT` 상방 가능성,
  `UP_CONTINUATION` 상방 유지, `RISK_ALERT` 하방 위험, `WEAKENING_ALERT` 추세 약화,
  `VOLATILITY_ALERT` 변동성 확대.
- **Grades**: `A`/`B`/`C` rank a (mostly bullish) alert's quality; `RISK` flags
  `risk_score ≥ 3` (downside risk dominant) regardless of the bull case. `risk_score`
  adds microstructure risk (upper-shadow rejection, VWAP over-extension, at-day-high
  exhaustion, volume spike) to `down_score`, so a stretched-but-bullish row still
  fires and is then graded `RISK`.
- **Online firing** (`select_online_alerts`): walk each day in time order, emit
  alerts as conditions are met, cap with `daily_alert_limit` and a per-symbol
  `cooldown_minutes`. No future row affects an earlier firing decision. Up-types are
  gated by the swept `AlertThresholds` (`up_threshold`, `score_margin`,
  `down_risk_limit`, `current_rank_limit`); RISK/WEAKENING/VOLATILITY use fixed
  selective gates so the catalogue stays noteworthy.
- **Forward eval** (eval-only, from the `entry_delay`-shifted entry): 30/60/120m +
  eod returns, MFE, MAE, up2/up3 reach, and `stop_first` (−1.5% touched before
  +1.5%, tie→down). Label hits (`up2_60m`, …, `up5_event`) are merged as a
  delay-0 cross-check.
- **Daily top-k is diagnostic only** (`select_topk_diagnostic`, look-ahead) — never
  the traded/alerted stream.
- Threshold sweep is one-knob-at-a-time around the baseline
  (`up_threshold`/`score_margin`/`down_risk_limit`/`current_rank_limit`/
  `daily_alert_limit`/`cooldown_minutes`/`entry_delay_minutes`).
- Output: SQLite `directional_alerts` (alert catalogue) and `online_alert_results`
  (fired alerts × configs + forward performance); `data/processed/
  alert_grade_summary_<ts>.csv` and `alert_threshold_compare_<ts>.csv`.

Result on the 20 active symbols (257 trading days, baseline ≈ 5.0 alerts/day):

| bucket | n | ret60m | dir60% | MFE | MAE | stop1st |
|---|---|---|---|---|---|---|
| grade A | 339 | +0.052% | 45.1% | +1.71% | −1.49% | 34.5% |
| grade B | 209 | −0.081% | 39.2% | +1.59% | −1.82% | 42.1% |
| grade C | 65 | −0.170% | 43.1% | +1.50% | −1.92% | 43.1% |
| RISK_ALERT type | 637 | +0.129% | 52.4% | +1.77% | −1.40% | 28.4% |

- **A > B > C holds** on `ret60m`/MAE/`stop_first` — the bullish scoring captures a
  real *relative* quality gradient. But the absolute edge is tiny (≤0.05%, inside
  the ~0.2% cost floor), so the grades rank alerts, they don't make a trade.
- **The downside semantics invert**: `RISK_ALERT` (down-leading) rows are followed by
  a *bounce* — higher `ret60m`, higher `dir_hit`, **lower** `stop_first` (28%) than the
  UP alerts (39–43%). Consistent with the intraday **mean reversion** seen throughout:
  buying confirmed strength fades, post-weakness bounces. So "하방 위험" as a
  momentum-direction label does not predict further downside on this universe/horizon.
- The online stream even **beats the look-ahead daily top-k** (+0.063% vs top1
  −0.017%), because top-k picks the highest-score (most over-extended) names, which
  revert hardest.

## Tests

```powershell
..\Python311-64\python.exe -m unittest tests.test_settings tests.test_collector tests.test_validation tests.test_resampler tests.test_quality tests.test_features tests.test_merge tests.test_relative tests.test_labeling tests.test_rule_signal tests.test_daily_topk_backtest tests.test_entry_variants tests.test_threshold_sweep tests.test_directional_alert tests.test_directional_alert_eval tests.test_risk_model tests.test_realtime_features   # cwd = quant
```

Covers config/symbol loading, the no-hardcode guard, CSV + record ingest,
session/date filtering, idempotency, missing-file handling, the coverage report,
the al_regular anomaly checks, 1m→3m/5m resampling, the quality checks, the
per-timeframe features, the leakage-safe merge, the cross-symbol relative features,
the forward-looking labels (current-bar exclusion, horizon high/low, has_full
flags, path SUCCESS/FAIL/NO_EVENT incl. same-bar tie, date isolation), and the
rule-signal score/candidate logic including the guard that labels are ignored,
plus the daily top-k backtest filters, both-side exclusion, cooldown, MFE/MAE,
and the guard that labels do not affect signal selection, and the entry/selection
variants (pre-open context and `preopen_ok`, the soft selection penalty and top-k
re-ranking, cooldown, delayed-entry price/eval window, `stop_first` path-dependence,
and the summary roll-up).
Step 13 tests cover label-independent selection, train-fitted percentile cutoffs,
and delayed-entry evaluation.
Step 13_5 tests (`tests/test_directional_alert.py`) cover the alert type/grade
classification (A/B/C/RISK, all five types), the guard that future label columns
do not affect alert generation, the swept up-gate and per-type fire gate, and the
causal online selection (time order, daily limit, per-symbol cooldown) versus the
look-ahead top-k diagnostic.
Step 14 tests (`tests/test_risk_model.py`) cover the leakage guard on
`FEATURE_COLUMNS`, the date-based split, AUC/bucket reporting, and the NumPy model;
`tests/test_realtime_features.py` covers serving/train feature parity, neutral
cross-sectional fill, strict causality (truncation invariance), and `risk_prob` +
alert scoring with the dependency-free model (so the suite stays portable to the
32-bit runtime, which has no lightgbm/sklearn/matplotlib).

## Step 14: Risk model 1

Step 14 trains the first ML model as a **risk filter**, not a direction or entry
model. The modelling universe is the Step 13_6 `directional_alerts` candidate
catalogue, and the primary target is `target_mae_risk`:

```text
future_MAE <= -1.5%
```

Labels are used only to create/evaluate the targets (`target_up_fail`,
`target_down2_60m`) and are excluded from `FEATURE_COLUMNS`; price-path MAE and
60m return are recomputed from `bars_1m` `al_regular`. The split is date-based
(`2025-06-02` to `2026-02-28` train, `2026-03-01` to `2026-06-19` validation).

```powershell
..\Python311-64\python.exe scripts\train_risk_model.py
..\Python311-64\python.exe scripts\train_risk_model.py --target target_up_fail
..\Python311-64\python.exe scripts\train_risk_model.py --target target_down2_60m
```

Backend chain (`--model auto`): **LightGBM** (primary, `class_weight="balanced"`,
early-stopped on a chronological holdout carved from the **tail of train** so the
validation set stays pristine) → sklearn `HistGradientBoostingClassifier` →
`LogisticRegression` → a dependency-free NumPy logistic (keeps the 32-bit runtime
working). The model file is always written:

```text
models/risk_model_lgbm.txt          # + .meta.json (preprocessor + metadata)
```

Outputs:

- SQLite: `risk_model_predictions`, `risk_model_summary`,
  `risk_prob_decile_report`, `feature_importance`
- Files: `data/processed/risk_model_predictions.parquet`,
  `risk_model_summary.csv`, `risk_prob_decile_report.csv`,
  `feature_importance.csv`, `risk_prob_monthly_auc.csv`

Model 1 validation result on 20 active symbols (LightGBM, `target_mae_risk`):

| metric | value |
|---|---:|
| validation AUC | 0.660 |
| PR-AUC | 0.474 |
| precision@top10% | 53.1% |
| recall@top10% | 15.3% |
| top10% MAE-risk rate | 53.3% |
| bottom20% MAE-risk rate | 14.6% |
| existing RISK-grade MAE-risk rate | 35.0% |

`risk_prob` deciles are monotone (top10% 53.3% → top20% 49.8% → mid40% 36.9% →
bottom20% 14.6%), so the model passes the Step 14 risk-separation check and the
LightGBM top-10% is **+18pp** more risky than the old `RISK` grade — which barely
separates (35.0% vs 34.4% non-RISK), exactly the Step 13_5 problem this step fixes.
Importances are interpretable (extension/volatility: `dist_ma5_intraday`,
`vwap_distance_1m`, `volume_z_20bar_5m`, `range_1bar_5m`, `distance_from_day_high_1m`).

An adversarial multi-agent audit confirmed the model is **leakage-free** (the
strongest single feature only reaches AUC 0.64 < the model's 0.66 — a weak-signal
problem, the opposite of a leak; `risk_score`/`down_score` are near-random vs the
target, so no tautology) and the metrics reproduce against sklearn. Two caveats it
surfaced, both now handled:

- **Target timing**: `decision_time` is the feature bar's `bar_end` (see
  `feature_lib.build_minute_features`), so `add_future_risk_targets` uses
  `searchsorted(side="left")` — the first forward bar is the one starting at
  `decision_time`; the feature bar is excluded (matches `labeling.py`).
- **Regime decay (the key caveat)**: validation AUC by month is 0.681 / 0.694 /
  0.672 / **0.592** (Mar→Jun 2026) as the drawdown base rate roughly doubles from
  train (18.9%) to validation (34.8%). Report the rolling/most-recent-month AUC
  (~0.59–0.61), not the pooled 0.66, and retrain periodically. `risk_prob_monthly_auc.csv`
  carries this breakdown.

## Real-time trend-alert capstone

Puts the whole pipeline online for **one symbol**: receive live 1-minute bars,
reproduce the exact feature pipeline at serving time, and emit a **trend alert**
whose direction comes from the Step 13_5 alert (type/grade) and whose **risk is
corrected by the Step 14 `risk_prob`**.

Architecture is split by the 32/64-bit constraint (Kiwoom OpenAPI+ needs 32-bit
Python; LightGBM/sklearn need 64-bit), bridged through the `realtime_minute_bars`
table:

```text
[32-bit] scripts/realtime_alert_collector.py   Kiwoom ticks -> 1m bars -> realtime_minute_bars (one or ALL symbols)
[64-bit] scripts/realtime_trend_alert.py       reads bars -> features -> alert -> risk_prob -> live matplotlib chart
[64-bit] scripts/realtime_trend_signal.py       same engine, NO chart: a streaming terminal signal feed
[either] scripts/check_realtime_bars.py        inspect realtime_minute_bars: per-symbol accumulation + alignment grid
```

- **Serving parity** (`src/realtime_features.py`): rebuilds `features_merged` with
  the same `feature_lib`/`resampler`/`feature_merge` functions, so 1m **and** 3m/5m
  features match the trained pipeline exactly (verified to 0.0 vs stored). The
  cross-sectional Step 9 features (rank/relative/breadth) need the whole universe,
  which a single-symbol feed lacks, so they are **neutral-filled** — a documented
  degradation; the risk model's strongest features are all within-symbol. Strictly
  causal (only bars up to each `decision_time`).
- **Trend alert**: fires on a non-empty alert type (09:10–15:00, per-symbol
  cooldown). The message states the direction (상방/상방 유지/하방 위험/추세 약화/변동성)
  and the model's risk band (`risk_prob` ≥0.60 HIGH / ≥0.45 MED / else LOW), flagging
  e.g. `⚠ 상방이지만 고위험` when a bullish alert carries high model risk — the risk
  filter in action.
- **Replay mode** (`--mode replay --replay-date YYYY-MM-DD`) streams a historical day
  from `bars_1m` as if live, so the engine is fully testable without a Kiwoom session.
- **Live startup**: `realtime_trend_signal.py --mode live` **replays the day's backlog**
  (market open → now) by default, so the morning's signals — per-minute lines, 🔔 fires,
  and 📈/📉 crossovers — print on startup before it streams one new line per minute.
  Replayed backlog fires are printed and CSV-ledgered but **not** re-written to
  `realtime_alert_log` (a restart would otherwise re-log the whole day); only live
  (current-minute) fires persist to the DB table. `--no-backlog` restores the old fast
  path: the engine `prime()`s the backlog in a single recompute (~0.5s) and emits only the
  latest line. (The chart engine `realtime_trend_alert.py` still primes silently — its
  candle history already shows the morning visually.)
- **Default `realtime_trend_signal.py` run = a single-stock trend-judgment feed**: target
  `000660` (override with `--code`), one per-minute signal line **every minute** (+🔔 on a
  fired alert). `--alerts-only` reverts to the print-only-alert-minutes view. Every other
  capability is opt-in via a flag.
- **5-minute MA crossover** (opt-in: `--tick-cross`): on top of the per-minute feed, a 📈/📉
  line fires when a finished bar's **close** crosses the 5-minute MA line (mean of the last 5
  completed 1m closes — the chart's MA5), once per cross (deduped), both up (golden) and down
  (dead). The 64-bit engine never sees Kiwoom ticks directly, so the check is close-based in
  **both** modes (the same close-cross rule as `rule_signal_ma`): *live* → the finished-bar
  close; *replay* → confirmed close-to-close cross by default, or `--replay-ticks ohlc` to
  approximate an intra-bar cross from each bar's OHLC path. `--ma5-window` sets the MA
  length; `--cross-eps-bp` adds a hysteresis band to suppress chatter at the line.
- **Cross-symbol serving** (opt-in: `--cross-symbol`, the **adopted "v1.1"** config): judge
  000660 with the REAL universe cross-section (rank/relative/breadth/dispersion across all
  active symbols via `build_relative_features`) instead of the single-symbol NEUTRAL fill —
  fixing a train/serve skew (the model was trained on the real cross-section). Strictly causal;
  the replay path computes the day's cross-section once and caches it; live rebuilds it each
  poll (needs the 32-bit collector running `--all-symbols`, else it degrades to neutral). Served
  values match the offline `features_relative` table exactly (`scripts/check_cross_symbol_parity.py`).
- **Risk-model versioning** (`--risk-model-version`, `src/risk_model.py:MODEL_VERSIONS`):
  `v1` (default, 8 cross-symbol features); `v2.0` (+5 richer universe features:
  `universe_disp_ret_20m`, multi-horizon breadth, `ret_20m_gap_to_best`, dispersion of
  `volume_z`); `v2.1a` / `v2.1b` — each v1 + ONE **leader lead-lag** pair (a leader's
  already-closed 5m/20m return, broadcast onto every symbol's row at the same
  decision_time; leaders are label→code in `config.relative.leaders`, columns
  `leader_<label>_ret_*`). v2.1a's leader is **005930** (Samsung), v2.1b's is **402340**
  (SK Square, holds a large 000660 stake). **All v2 variants were A/B-evaluated against
  v1.1 and NONE cleared the recent-month gate** — recent-month (May/Jun) AUC Δ: v2.0 ≈
  −0.008, v2.1a ≈ −0.003, **v2.1b ≈ −0.002 (closest)**; any pooled gain concentrates in
  the oldest validation month and reverses in June. v2.1b (402340) is the strongest
  attempt — its leader feature ranks #10 in importance and it beats v1.1 in 3 of 4 months
  — but June (drawdown regime) sinks the gate. See `data/processed/v2_0_vs_v1_ab_report.md`
  and `v2_1_vs_v11_ab_report.md`. **v1 stays the model; v1.1 (`--cross-symbol`) is the
  adopted live config; v2.0/v2.1a/v2.1b are archived, none wired to live.** A v2.x model is
  refused on neutral-fill serving (must use `--cross-symbol`). `v1_60m` (Step 17) is the
  fixed-horizon research champion — v1 features + `minutes_since_open`, trained on
  `target_mae_risk_60m` per walk-forward fold (`--fold`, files `risk_model_v1_60m_fold<k>_lgbm.txt`);
  never served.
- **Premarket context (v3.0, `src/premarket_features.py`)** — attaches per-day
  **premarket (08:00–09:00)** features to each same-day regular-session row (joined by
  `trade_date`; the window closes before the open → strictly causal; the only cross-day
  input is the *previous* regular close via a backward as-of lookup). Built from `al_raw`
  by `scripts/build_premarket_features.py` → `premarket_features_daily`. Two variants
  ride the v1.1 baseline: `v3_0a_premarket_000660` (target overnight gap / range / VWAP /
  volume-z, 10 features) and `v3_0b_premarket_universe` (+ universe breadth / dispersion /
  rank / relative, 23 features). v3.0c/v3.0d (005930 / 402340 premarket anchors) are
  scaffolded (`build_premarket_features.py --with-anchors`, `pm_anchor_columns`) but **not
  built** in this round. **Both v3.0 variants are NO-GO** (`scripts/evaluate_v3_premarket_gate.py`
  → `v3_0_premarket_ab_report.md`): on the identical 2026-03→06 frame they *underperform*
  v1.1 (pooled AUC 0.66 → v3.0a 0.63, v3.0b 0.62) and **worsen** the bottom-20% low-risk
  bucket (0.145 → ~0.19, more blow-ups slip into "low risk"). v3.0a's recent-pooled AUC Δ
  is ≈ 0 only by accident — its monthly Δ swings −0.10/−0.06/−0.05/+0.05, the classic
  regime-overfit signature (worst recent month −0.05); v3.0b is flatly worse (recent Δ
  ≈ −0.06). Premarket features top the importance ranks yet hurt generalisation: a
  day-broadcast value is constant within a day, so the model overfits day-regime noise. **v1.1 stays the live model; v3.0
  is research-only and not wired to live** (a passing v3 variant would need the same
  premarket features computed pre-open and joined at serving — `requires_premarket=True`).
- **TCN challenger (v4_tcn, Step 16)** — an ARCHITECTURE challenger (dilated causal
  convolutions over the raw 60-bar features_1m window instead of the 49 flat aggregates).
  Lives in its OWN registry (`src/tcn_model.py:TCN_VERSIONS`, torch artifacts
  `models/risk_model_v4_tcn_seed<k>.pt`), deliberately NOT in `MODEL_VERSIONS` (the
  LightGBM trainer/loader must never route onto a torch file). Same rows/label/split as
  v1.1; gated by `scripts/evaluate_tcn_gate.py` (same recent-weighted gate). See Step 16.

```powershell
# replay (no Kiwoom): chart version (--no-window saves a headless PNG) or chartless feed
..\Python311-64\python.exe scripts\realtime_trend_alert.py  --mode replay --replay-date 2026-06-19
..\Python311-64\python.exe scripts\realtime_trend_signal.py --mode replay --replay-date 2026-06-19                # 000660 per-minute trend feed
..\Python311-64\python.exe scripts\realtime_trend_signal.py --mode replay --replay-date 2026-06-19 --tick-cross   # + 5m MA crossover
..\Python311-64\python.exe scripts\realtime_trend_signal.py --mode replay --replay-date 2026-06-19 --cross-symbol # v1.1: real universe features
```

Live smoke-test procedure (during market hours, 32-bit Kiwoom login):

```powershell
# 1. collector: ALL active symbols (enables the same-timestamp alignment check)
..\Python311-32\python.exe scripts\realtime_alert_collector.py --all-symbols
# 2 & 6. confirm bars accumulate + symbols align on the same ts_kst
..\Python311-64\python.exe scripts\check_realtime_bars.py --watch 30
# 3-7. chartless signal feed for one symbol (prints per minute, persists fired alerts).
#       --cross-symbol = adopted v1.1 config (real universe features; needs --all-symbols above).
..\Python311-64\python.exe scripts\realtime_trend_signal.py --mode live --code 000660 --cross-symbol
```

Outputs: `data/processed/realtime_trend_chart.png` (chart), `realtime_alert_ledger.csv`
/ `realtime_signal_ledger.csv`, and the `realtime_alert_log` DB table (live fired alerts).
The same regime-decay caveat applies — `risk_prob` is least reliable in the most recent
regime. Note: single-symbol serving neutral-fills the cross-sectional features, so the
multi-symbol collector mainly proves the data layer aligns (step 6) for a future
universe-aware engine.

### Position exit-warning monitor (manual-trade support)

`scripts/realtime_position_monitor.py` is a **separate layer** for an **already-open
manual position**: the trend-signal feed answers "what is the market doing?"; this
answers "I bought 000660 — hold, or sell warning?". It **reuses the same
`TrendAlertEngine`** (no duplicated feature/model code) for the trend/risk snapshot and
layers `src/position_exit.py`'s pure, deterministic exit rules on the position's price
excursions since entry. It is **decision-support only — it never sends an order** (no
order API is imported or called).

States (weakest→strongest): `HOLD` / `WEAK_SELL_WARNING` / `SELL_WARNING` /
`STRONG_SELL_WARNING`. Rules: hard stop, profit giveback to break-even, trailing
drawdown, MA5/VWAP break (both = SELL, one = WEAK), risk + weakening trend, opposite/down
signal, and a no-progress time stop — strongest state wins, all matched reasons logged.
All thresholds are CLI-overridable (`--hard-stop`, `--trailing-drawdown`, …). Long-only
for now (structured for a later short side).

```powershell
# replay (no Kiwoom): monitor a hypothetical entry against a historical day
..\Python311-64\python.exe scripts\realtime_position_monitor.py --mode replay --replay-date 2026-06-19 ^
    --code 000660 --side long --entry-price 2856000 --entry-time 10:14 --cross-symbol
# live: start AFTER you manually buy (needs the 32-bit collector running --all-symbols)
..\Python311-64\python.exe scripts\realtime_position_monitor.py --mode live --code 000660 ^
    --side long --entry-price <MY_PRICE> --entry-time <HH:MM> --cross-symbol
```

Outputs: `data/processed/realtime_position_ledger.csv` (every monitored bar) and the
`position_exit_log` DB table (warning rows). By default it prints **only when the exit
state changes** (HOLD → WEAK → SELL → STRONG and back) — the actionable transitions, with
the triggering reason on each line; `--all-bars` prints every bar instead. E.g.:

```text
10:15 000660 LONG HOLD ret=+0.00% MFE=+0.07% MAE=-0.21% DD=-0.07% trend=UP risk=MED
10:18 000660 LONG WEAK_SELL_WARNING reason=risk_elevated ret=+0.56% MFE=+0.60% MAE=-0.21% DD=-0.03% trend=UP risk=HIGH
11:19 000660 LONG STRONG_SELL_WARNING reason=hard_stop ret=-1.05% MFE=+1.23% MAE=-1.05% DD=-2.25% trend=RISK risk=HIGH
```

### Morning-signal → afternoon-extreme validation (research only)

`scripts/validate_morning_signal_afternoon_extreme.py` (logic in
`src/morning_signal_validation.py`) is a **research/validation** study — no orders, no
live wiring. It tests: *does a morning 🔔 signal before 11:00 predict a useful afternoon
(13:00–15:30) high/low vs the alert price?* It selects one morning signal per day —
**first / last / strongest** (the three core modes), plus `lowest-risk-up` /
`highest-risk-down` / `all` — measures the afternoon extreme (hindsight = potential, not a
tradable exit), and compares against morning-price baselines (`last_bar_before_11`,
`first_bar_after_10_30`, `random_morning_bar`, `all_morning_bars_sample`).

**Alert feed (`--alerts-source`, `auto` prefers `realtime_replay`):** use
**`realtime_replay`** — the realtime engine's fired 🔔 replayed over history
(`scripts/build_realtime_replay_alerts.py` → `realtime_replay_alerts`). It is
**09:10-capable** because the engine seeds `HISTORY_DAYS=8` prior days, so it reflects the
TRUE first morning 🔔. Do **not** use `directional_alerts` for a "first 🔔" question: that
offline feed is built from within-day features (60-minute warmup) so it only starts at
**10:11** and is blind to 09:00–10:00. The fast per-day capture (build features once over
seeded history + the day, then replay the engine's cooldown gate) was verified **identical**
to bar-by-bar streaming and leakage-free. `risk_prob` is the engine's (single-symbol by
default; the annotation may differ from the live v1.1 cross-symbol config, but
direction/price/returns are rule/bar-derived and config-invariant).

```powershell
..\Python311-64\python.exe scripts\build_realtime_replay_alerts.py --code 000660            # build the 09:xx-capable feed
..\Python311-64\python.exe scripts\validate_morning_signal_afternoon_extreme.py --code 000660 --selection first --risk-as-short
```

Outputs (to `data/processed/`): `_trades.csv` (per signal), `_summary.csv` (by
direction/grade/risk_bucket with hit-rates), `_baseline.csv`, `_daily.csv` (**one row per
day** — the day's direction is set by its single selected signal, so a day is long XOR
short, never both; signal_return vs same-direction baseline_return + edge), and `_report.md`
with a **PASS / PARTIAL / FAIL** verdict (UP + DOWN). `--risk-as-short` treats RISK_ALERT
(하방 위험) as a SHORT signal so the short side is validated too.

**Finding (000660, `realtime_replay` feed, `first` + `--risk-as-short`, 232 signal-days =
104 long + 128 short, 188 of them firing 09:xx):** the first morning **directional** 🔔 does
**NOT** beat baseline. LONG afternoon-high +1.02% vs same-day baseline +0.83% → paired edge
only **+0.19% (t≈1.1, not significant)**, and *worse* than a random morning bar (+1.18–1.22%);
SHORT short_potential +0.68% < baseline +1.10%. Both **FAIL**. ⚠️ This **overturns** the
earlier result ("first beats baseline, +1.11% vs +1.01%"), which was a **warmup-truncation
artifact** of the old 10:11-floor `directional_alerts` feed — selecting later, more-confirmed
signals. (Note: "first DIRECTIONAL signal" — on ~66% of days the literal first 🔔 is a
non-directional WEAKENING/VOLATILITY alert that the study skips.)

---

## Step 15: conservative DOWNTREND trend-alert system

Decision-support **downtrend trend-alert** system — the symmetric mirror of the long-side
directional-alert taxonomy, for the user's manual short-side/dip judgement. Being a trend
feed (like the long capstone), it carries **no GO/NO-GO gate**: it classifies what the DOWN
side is doing and shows typed alerts. **No order API is imported or called** (guarded by
tests). All alert logic lives in NEW files; the long engine/alert modules are untouched, and
`src/risk_model.py` / `scripts/train_risk_model.py` received **additive-only** changes
(default behavior byte-identical; the pre-existing test suite passes unmodified).

**Alert taxonomy** (`src/short_entry.py`, sign-flipped mirror of `directional_alert`'s
`RISK > WEAKENING > UP_CONTINUATION > UP_ALERT > VOLATILITY` priority):

| type | 한국어 | mirror of |
|---|---|---|
| `SHORT_RISK_ALERT` | 상승 위험 | RISK_ALERT (upside/bounce risk dominant) |
| `DOWN_WEAKENING_ALERT` | 하락 추세 약화 | WEAKENING_ALERT |
| `DOWN_CONTINUATION` | 하방 유지 가능성 | UP_CONTINUATION |
| `DOWN_ALERT` | 하방 가능성 | UP_ALERT |
| `VOLATILITY_ALERT` | 변동성 확대 | VOLATILITY_ALERT |

- **`src/short_entry.py`** — pure module: `compute_short_entry_columns` (assigns `alert_type`
  via the mirrored `np.select` priority + grades A/B/C/`SHORT_RISK`), `short_entry_gate` (the
  DOWN-type rule gate: `down>=4, up<=1, margin>=3, bottom-rank 5`), `short_alert_fire_gate`
  (DOWN types via the rule gate; `SHORT_RISK`/`WEAKENING`/`VOLATILITY` via fixed constants —
  mirror of `alert_fire_gate`), stateful `ShortFireGate` (session 09:10–15:00, **per-type**
  cooldown 15m so a different type fires immediately, daily limit 5), `score_serving_short`
  (own serving path — `engine.latest` exposes RESULT_COLUMNS only), `short_risk_band`,
  `short_message`. The v1_short model is **annotation-first**: `short_adverse_prob` +
  `short_risk_band` are shown on every alert; DOWN types add ✓ 저위험 하방 (LOW) / ⚠ 모델
  상승위험 (HIGH). `short_adverse_prob_max` defaults to **None** (annotate, don't veto); set a
  float to veto DOWN types only.
- **`scripts/realtime_short_trend_alert.py`** — chartless replay/live runner composing the
  UNMODIFIED `TrendAlertEngine` for long context (`engine.enable_persistence` never called;
  cooldown state fully separate). Emits **one line per bar**, labeled like the chart candle
  (bar start — same convention as `realtime_trend_signal.py`; the decision itself happens at
  that bar's close, and the ledger/DB keep `decision_time`): leading markers `🔻` = a DOWN-side
  alert fired / blue `▲` = the LONG-side engine fired that bar (both can show); then the fired
  down message or the compact status (`dn`=down_score, `up`=up_score, `dir`=up−down direction,
  `bounce`=upside_risk_score, `adverse`=short_adverse_prob band); then `| long=…` (the long
  engine's current classification — its own fire adds the ▲). Live down fires log to
  `short_alert_log` + CSV ledger `data/processed/short_entry_ledger.csv` (long fires are
  console-only). `daily_alert_limit` is **off by default** (the per-type cooldown, kept at
  15 min, is the throttle; pass `--daily-alert-limit N` to cap). Without `--cross-symbol` the
  bottom-rank gate auto-disables (single-symbol serving neutral-fills ranks). Starts unconditionally.
- **`scripts/realtime_long_short_trend_alert.py`** — unified runner (copy-derivative of
  `realtime_short_trend_alert.py`, which stays available): the LONG uptrend alerts become
  **first-class fires on equal footing** with the down alerts. The unmodified engine runs with
  `cooldown_min=0` (its internal cooldown bypassed — `_maybe_fire` gates only on session window
  + typed row, never the `fires` column) and is re-gated in the runner, so **both sides share
  one cooldown policy**: `--cooldown-scope per-type` (default: 같은 알림만 15분 억제, 타입별
  독립) or `single-slot` (the engine's historical rule: 다른 알림이 울리면 15분 전이라도
  재울림); `--cooldown-min` applies to both. Every alert line (🔻 short / ▲ long / 🔻▲ both)
  ends with the shared `‖ dn/up/dir/bounce/adverse` score tail. `--strict` raises standards on
  both sides (short: `down_threshold+1`/`score_margin+1` + bounce≥5&dir≥+4 / dn≥4 / volz≥3.0
  for the informational types; long: grade C suppressed, UP up≥5&dir≥+4, RISK risk≥4&dir≤−3,
  WEAKENING up≥4 — rules printed in the banner, provenance tag `*_strict`). One CSV ledger
  `data/processed/long_short_trend_ledger.csv` with a leading `side` column (`down`/`long`);
  live down fires go to `short_alert_log` as before, **long fires never write
  `realtime_alert_log`** (console+CSV only — that table belongs to a concurrently running
  long runner).
- **`scripts/build_short_replay_alerts.py`** — would-have-seen replay feed (same
  `ShortFireGate`; runner↔builder parity verified bar-for-bar on 2026-02-19): tables
  `short_replay_alerts` (all five types) + pre-gate `short_replay_candidates` (down_score>=3).
- **`scripts/short_gate_sweep.py`** / **`scripts/evaluate_short_entry_gate.py`** — an
  **optional, historical** study of whether the two entry-flavoured types (`DOWN_ALERT`/
  `DOWN_CONTINUATION`) additionally carry short-side forward edge. Not a gate on the runner.
- Tests: `tests/test_short_entry.py`, `tests/test_short_risk_model.py`,
  `tests/test_short_entry_eval.py` (full suite green).

**v1_short short-adverse model** (`--target target_up_risk --risk-model-version v1_short`,
label = rest-of-day upside MFE >= +1.5%, the exact mirror of `target_mae_risk`):
`models/short_risk_model_lgbm.txt`. Val AUC **0.684** (long model: 0.661), monthly AUC
0.68/0.66/0.66/0.66 (no regime decay), buckets strictly monotone 59.9/55.5/34.0/14.5%
(top10−bottom20 spread 45.3pp). `short_adverse_prob` = P(the down case gets an adverse ≥+1.5%
upside move) — high prob ⇒ bounce likely ⇒ the ⚠ 상승위험 annotation.

**Replay feed** (265 days, `--cross-symbol`, veto off, `daily_alert_limit` off — the current
default): 6,088 fires (~23/day) — `SHORT_RISK_ALERT` 2,518, `DOWN_ALERT` 1,965,
`DOWN_WEAKENING_ALERT` 1,285, `DOWN_CONTINUATION` 319, `VOLATILITY_ALERT` 1 (loses the
priority race); grades SHORT_RISK 2,629 / A 1,862 / B 1,374 / C 223; bands MED 4,756 /
LOW 1,212 / HIGH 120. With no daily cap the per-type 15-min cooldown is the only throttle,
so the feed is *informative* rather than *sparse*; pass `--daily-alert-limit N` for a quieter
feed (capping at 5/day yields 1,325 fires over 175 firing days) or use the runner's `--strict`.

**Optional entry-edge study (historical): DOWN types show NO short-entry edge.** Sweeping the
`DOWN_ALERT`/`DOWN_CONTINUATION` thresholds as if they were short-entry signals, the best
train-ranked combos collapse on the holdout (fwd60 +0.03%..+0.15%, direction hit 30–37%),
consistent with the Step 13_5 mean-reversion of down signals (RISK_ALERT +0.13% fwd60) and
the morning-study SHORT FAIL. So the DOWN types stay **informational trend alerts** (their
intended role); they are not wired as an entry timer. Report:
`data/processed/short_entry_gate_report.md`.

```powershell
# runbook (cwd = quant, 64-bit)
..\Python311-64\python.exe scripts\train_risk_model.py --target target_up_risk --risk-model-version v1_short
..\Python311-64\python.exe scripts\build_short_replay_alerts.py --cross-symbol   # typed alert feed
# runner (starts unconditionally; --short-adverse-prob-max <float> turns the model into a DOWN-type veto):
..\Python311-64\python.exe scripts\realtime_short_trend_alert.py --mode replay --replay-date 2026-02-19 --cross-symbol
..\Python311-64\python.exe scripts\realtime_short_trend_alert.py --mode live --cross-symbol
# unified long+short runner (equal-footing ▲ long fires; add --cooldown-scope single-slot / --strict as needed):
..\Python311-64\python.exe scripts\realtime_long_short_trend_alert.py --mode replay --replay-date 2026-02-19 --cross-symbol
# optional historical entry-edge study:
..\Python311-64\python.exe scripts\short_gate_sweep.py
..\Python311-64\python.exe scripts\evaluate_short_entry_gate.py
```

---

## Step 16: TCN challenger vs LightGBM risk model (A/B gate)

**Research / evaluation only — no orders, no Kiwoom, no SendOrder; nothing is wired to
live serving.** `src/risk_model.py`, `scripts/train_risk_model.py` and
`src/realtime_features.py` are byte-untouched; all TCN code is in NEW files and the
pre-existing test suite passes unmodified.

An **architecture** challenger to the long risk model: same rows (directional-alert
candidates), same `target_mae_risk` label (rest-of-day MAE ≤ −1.5%), same date split as
v1.1 — but instead of the 49 flat aggregated features it consumes a **causal window of the
last 60 one-minute bars' per-bar features** (13 features_1m channels — multi-window returns
excluded as collinear partial sums of `ret_1m` — plus a time-of-day channel and a pad-mask
channel) through **dilated causal Conv1d residual blocks** (Bai et al. TCN: 4 levels,
d=1/2/4/8, k=3, weight norm, receptive field 61 ≥ 60, ~30k params), pooling the **last
timestep** (= the decision bar) and fusing the v1 static features (5 alert + 8 cross-symbol)
in the head. `decision_time` is the CONFIRMED feature bar's `bar_end` (full timestamp —
the anchor join hard-fails on date-less values); the window's newest element is that bar,
the label's forward window starts at `bar_start >= decision_time`, windows never cross a
day boundary, and short/unanchored rows are **padded, never dropped** (the challenger
scores the exact frozen v1.1 validation frame 1:1).

- **`src/tcn_data.py`** — torch-free sequence dataset: shared sorted bar matrix +
  vectorized window gather (`idx = j − offsets` cannot express a future bar), day-start
  clamp, `SequencePreprocessor` (median+z-score fit on **train-range bars only**,
  asserted). All leakage invariants unit-tested without torch.
- **`src/tcn_net.py`** — the only module with a top-level `import torch`; itself imported
  ONLY lazily from `tcn_model.py` (lightgbm/sklearn lazy-import precedent — torch is NOT
  added to requirements.txt; install per runbook below).
- **`src/tcn_model.py`** — `TCN_VERSIONS` registry (own registry; `"v4_tcn" not in
  rm.MODEL_VERSIONS` is a test), `fit_tcn` (BCE with pos_weight = the champion's
  `class_weight="balanced"` analog; AdamW; early stopping on the SAME
  `rm._time_ordered_holdout` train-tail 15% — `fit_tcn`'s signature cannot even receive the
  validation frame), `predict_tcn`, `.pt` + `.meta.json` artifacts.
- **Pre-registered seed protocol**: `SUBMISSION_SEEDS = (42, 43, 44, 45, 46)` fixed in
  code before any validation scoring; the submitted score is the arithmetic **mean of the
  per-seed probabilities** — no seed selection of any kind (per-seed dispersion, majority
  sign and a fragility flag are report diagnostics only).
- **`scripts/train_tcn_risk_model.py`** — mirrors `train_risk_model.py` (same input SQL +
  `features_1m`), writes the standard `vsfx=_v4_tcn` artifact family (predictions parquet
  carries `risk_prob` = ensemble + `risk_prob_seed<k>` columns + a `.parquet.meta.json`
  provenance sidecar with the run config and a validation-frame fingerprint; no
  feature_importance — not meaningful for a TCN, ablations replace it). The default
  `_v4_tcn` suffix is **reserved for the pre-registered submission** (all knobs at
  default) — any diagnostic/smoke/ablation run must pass its own `--out-suffix` or the
  trainer refuses, so a cheap run can never clobber the submission's weights/parquet.
- **`scripts/evaluate_tcn_gate.py`** — the A/B gate, torch-free (champion rescored
  in-process as in the v3 gate; challenger read from the parquet with hard 1:1
  row/label/coverage asserts). `decide_verdict` is copied VERBATIM from
  `evaluate_v3_premarket_gate.py`: GO ⇔ recent(05+06) AUC Δ ≥ +0.010 ∧ worst recent month
  > −0.005 ∧ buckets preserved ∧ A-grade low-risk MAE not worse ∧ >1 month up. Extra
  TCN-only diagnostics (never gated): seed dispersion/majority sign, calibration
  reliability + Brier (both models are class-weighted ⇒ uncalibrated; the serving bands
  0.60/0.45 do NOT transfer — a GO would re-derive TCN bands by v1.1-occupancy matching),
  and single-seed ablations (`--ablation-predictions NAME=PATH`).
- Tests: `tests/test_tcn_model.py` (registry separation, v1 freeze re-asserts, poison/
  truncation-invariance/cross-day/anchor-key/padding causality, train-only stats, torch
  determinism + save/load round-trip, lazy-torch + no-order + no-hardcoded-symbol scans).
- **Status: awaiting the 5-seed submission run** (the 3-epoch smoke run only validated the
  end-to-end schema; its gate output is NOT adoption-grade). Verdict + results table land in
  `data/processed/tcn_vs_lgbm_ab_report.md`; **v1.1 remains the adopted live config** and a
  GO would additionally require Phase-2 serving work (sequence-serving builder parallel to
  `score_serving`, a score-parity replay check, band re-thresholding) before any wiring.

```powershell
# runbook (cwd = quant, 64-bit; torch CPU wheel once):
..\Python311-64\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# smoke (fast end-to-end schema check; the default _v4_tcn suffix is RESERVED for the
# submission, so a diagnostic run MUST use its own suffix -- the trainer refuses otherwise):
..\Python311-64\python.exe scripts\train_tcn_risk_model.py --single-seed 42 --max-epochs 3 --out-suffix _v4_tcn_smoke
..\Python311-64\python.exe scripts\evaluate_tcn_gate.py --tcn-predictions data\processed\risk_model_predictions_v4_tcn_smoke.parquet
# submission run (pre-registered 5 seeds, all knobs at default; CPU, roughly 2.5-4 h):
..\Python311-64\python.exe scripts\train_tcn_risk_model.py
# ablations (single seed, diagnostic only; C isolates the time-of-day channel -- the
# rest-of-day label's base rate depends on clock time, so tod alone lifts AUC.
# --seq-len 120 auto-deepens the stack (levels 4->5) so the extra bars are actually seen):
..\Python311-64\python.exe scripts\train_tcn_risk_model.py --single-seed 42 --no-static --out-suffix _v4_tcn_ablA
..\Python311-64\python.exe scripts\train_tcn_risk_model.py --single-seed 42 --seq-len 120 --out-suffix _v4_tcn_ablB
..\Python311-64\python.exe scripts\train_tcn_risk_model.py --single-seed 42 --no-tod --out-suffix _v4_tcn_ablC
# gate + report (defaults to the submission parquet; adds the ablations as diagnostics):
..\Python311-64\python.exe scripts\evaluate_tcn_gate.py ^
    --ablation-predictions ablA_no_static=risk_model_predictions_v4_tcn_ablA.parquet ^
    --ablation-predictions ablB_seq120=risk_model_predictions_v4_tcn_ablB.parquet ^
    --ablation-predictions ablC_no_tod=risk_model_predictions_v4_tcn_ablC.parquet
# tests:
..\Python311-64\python.exe -m unittest tests.test_tcn_model
```

---

## Step 17: fixed-horizon (60m) label lineage — v1_60m vs v4_tcn_60m walk-forward gate

**Motivation.** The Step 16 ablations showed ~81% of the TCN's GO-clearing edge was the
single time-of-day channel: the legacy label (rest-of-day `future_MAE` ≤ −1.5%) has a
mechanically time-dependent base rate (less day left → less risk left), so a clock
feature lifts AUC without any cross-sectional skill. Step 17 removes that confound in
the LABEL: `future_MAE_60m` / `future_MFE_60m` = min(low)/max(high) over the **next
60 minutes** (`src/risk_model.py:add_future_risk_targets(horizon_minutes=60)`), and the
whole champion/challenger comparison is re-run as a new lineage. The legacy label math,
constants, and artifacts stay frozen (additive only).

**Label conventions** (pinned by `tests/test_label_60m.py`):
- Window = bars with `bar_start` in `[decision_time, decision_time + 60m)` — same
  half-open end as `future_return_60m`; first forward bar included, decision bar
  excluded; identical to `labeling.py`'s `down2_60m` window (cross-checked on real data:
  gap 0.0000).
- **Main population** = decisions with time-of-day ≤ `HORIZON_60M_MAIN_CUTOFF`
  (`14:30:00`, inclusive) — always exactly 60 minutes observed. Later decisions carry a
  truncated-horizon label (== rest-of-day; verified 100.0000%) and form a separate
  **late-session diagnostic**, excluded from training and from the gate.
- **NaN propagation**: no bars in the window → `future_MAE_60m` NaN → `target_mae_risk_60m`
  NaN (float column; training drops it) — deliberately unlike the legacy NaN→0.
- **Thresholds** (USER-decided after `scripts/survey_60m_label_thresholds.py`):
  `MAE_RISK_THRESHOLD_60M = -0.015`, `UP_RISK_THRESHOLD_60M = +0.015` (survey: 18.6% of
  validation / 7.2% of train main-population rows at −1.5%; hour-of-day profile shows the
  legacy monotone decay 47%→21% replaced by flat intraday seasonality ≈16–24%).
- **Clock parity**: BOTH lineage models receive the clock — `v1_60m` = v1's 49 features +
  `minutes_since_open` (`FEATURE_COLUMNS_60M`, 50), `v4_tcn_60m` keeps the tod channel —
  so the gate measures architecture value on top of clock access.

**Walk-forward protocol** (USER-decided; `risk_model.WALK_FORWARD_FOLDS_60M`): four
expanding-window folds — train always starts 2025-06-02; train ends 2026-02-28 /
03-31 / 04-30 / 05-31; validation = the following single month (2026-03 / 04 / 05 / 06).
ONLY the periods differ between folds. Champion `v1_60m` (LightGBM) and challenger
`v4_tcn_60m` (TCN, pre-registered **single seed 42** per fold; 5-seed only if later
justified) are trained per fold; the gate
(`scripts/evaluate_tcn_gate.py --tcn-version v4_tcn_60m`) rescore-stitches the four
genuinely out-of-sample months into one walk-forward frame and runs the **unchanged**
recent-weighted gate math on it (per-fold Δ table reported alongside). The A-grade
low-risk MAE margin is **0.0003** (USER-decided; scale-matched: 60m/EOD `future_MAE`
std ratio ≈ 0.59 × legacy 0.0005). Cross-lineage seals: parquet sidecars carry
target/fold/version — a target or fold mismatch is a hard refusal.

```powershell
# survey that fixed thresholds/margin (research-only, already run):
..\Python311-64\python.exe scripts\survey_60m_label_thresholds.py
# champion: one LightGBM per fold (fast)
..\Python311-64\python.exe scripts\train_risk_model.py --risk-model-version v1_60m --target target_mae_risk_60m --fold 1
#   ... --fold 2 / --fold 3 / --fold 4
# challenger: all four folds, seed 42 each (one command; ~5-8 min/fold on CPU)
..\Python311-64\python.exe scripts\train_tcn_risk_model.py --tcn-version v4_tcn_60m
# gate + report -> data\processed\tcn_vs_lgbm_60m_ab_report.md + risk_model_tcn_60m_gate.csv
..\Python311-64\python.exe scripts\evaluate_tcn_gate.py --tcn-version v4_tcn_60m
# optional ablation (diagnostic; residual tod attribution on one fold):
..\Python311-64\python.exe scripts\train_tcn_risk_model.py --tcn-version v4_tcn_60m --fold 4 --single-seed 42 --no-tod --out-suffix _v4_tcn_60m_ablC
..\Python311-64\python.exe scripts\evaluate_tcn_gate.py --tcn-version v4_tcn_60m --ablation-predictions ablC_no_tod=risk_model_predictions_v4_tcn_60m_ablC_fold4.parquet
# tests:
..\Python311-64\python.exe -m unittest tests.test_label_60m tests.test_tcn_model
```

**GPU-server training.** TCN training accepts `--device cuda` / `cuda:0` / `cuda:1`
(default `cpu`; recorded in every artifact meta + provenance sidecar; seeded per
device but NOT bit-identical across devices — the gate's frame fingerprint still
verifies data identity). `--deterministic auto` (default) keeps bit-reproducible
algorithms on cpu but disables them on cuda: cuDNN's deterministic dilated-conv
backward kernels are orders of magnitude slower (observed hours/seed vs minutes).
Ship the inputs as a **frame cache** — `--export-frame-cache db\frame_cache` builds
the modelling frame (labels included) + the features_1m slice once, locally, into two
parquets (~140 MB total); copy them with `src/`, `scripts/train_tcn_risk_model.py`,
`config/` and train on the server with `--frame-cache db/frame_cache --device cuda:0`
(no sqlite, no label math; verified bit-identical to the DB path). Copy back ONLY
`models/risk_model_<sfx>_seed*.pt(+.meta.json)` and `data/processed/
risk_model_predictions_<sfx>*.parquet(+.meta.json)` — the gate runs locally against
the local DB and champion models. Do not rebuild the local features/labels tables
between exporting and gating, or the fingerprint drift guard fires (by design).
(`scripts\export_tcn_training_db.py` remains the sqlite-shipping alternative.)

The 60m lineage is research-only end to end: even on a GO, **v1.1 remains the live
config** until a deliberate adoption decision plus Phase-2 serving work.

---

## Step 18: v5_cnn_60m — 1D-CNN challenger (performance search vs the TCN)

Performance-seeking challenger (NOT a parameter-matched baseline): a plain 1D-CNN on
the same 60m lineage — same folds (`WALK_FORWARD_FOLDS_60M`), label, features, tod,
preprocessing, gate — to see whether a different conv family beats the TCN.
Architecture (`src/tcn_net.py:CNN1dRiskNet`, registry `TCN_VERSIONS["v5_cnn_60m"]`):
4 × Conv1d(k=5, channels 64) with a **triple pad-guard** — input mask-zeroed before
the first conv, activations re-masked after every block (conv bias/kernel smear would
otherwise leak pad columns back into valid positions), and mask-based pooling — then
concat of **masked global max + masked global mean + last-VALID-timestep** (~80k
params; single-row CPU inference is ms-scale, honoring the local-CPU serving
constraint). Internal early-stopping holdout is split by **whole trading days**
(`holdout_by_day`), and every seed×fold summary row records the overfit watch:
train/holdout loss, holdout ROC/PR-AUC, best epoch, external AUC/PR-AUC, Brier,
top10/bottom20 rates. Reuses the shipped v4_tcn_60m frame cache (`frame_cache_alias`).

**Pre-registered protocol**: single seed 42 per fold. **Verdict structure**: Primary =
CNN s42 vs TCN s42 (fold-by-fold + stitched recent); Secondary = CNN s42 vs v1_60m
(the unchanged gate, report `cnn_vs_lgbm_60m_ab_report.md`); Reference = the TCN
5-seed ensemble. **5-seed escalation criteria (fixed before results)**: run the 5-seed
confirmation iff ① stitched recent AUC CNN s42 ≥ TCN s42 − 0.005, ② CNN s42 vs v1_60m
recent Δ ≥ +0.010, ③ each recent fold (3, 4) vs v1_60m Δ > −0.005. After 5 seeds:
median-seed recent Δ ≥ +0.010 AND CNN ensemble recent ≥ TCN ensemble → "CNN
leads/ties" (adoption-candidate switch review); otherwise the TCN stands. A future
multi-scale CNN (parallel-kernel stem, e.g. k∈{3,5,9} → concat → 1×1 fuse) slots in
as a sibling net class + registry entry (v6) — the arch dispatch is already generic.

```powershell
# server (single seed 42, all 4 folds; reuses the existing frame cache):
#   python scripts/train_tcn_risk_model.py --tcn-version v5_cnn_60m --device cuda:0 --frame-cache db/frame_cache
# local gate + TCN s42 side-by-side rows:
..\Python311-64\python.exe scripts\evaluate_tcn_gate.py --tcn-version v5_cnn_60m ^
    --ablation-predictions tcn_s42_fold1=risk_model_predictions_v4_tcn_60m_fold1.parquet ^
    --ablation-predictions tcn_s42_fold2=risk_model_predictions_v4_tcn_60m_fold2.parquet ^
    --ablation-predictions tcn_s42_fold3=risk_model_predictions_v4_tcn_60m_fold3.parquet ^
    --ablation-predictions tcn_s42_fold4=risk_model_predictions_v4_tcn_60m_fold4.parquet
```

**Results (2026-07-16).** Secondary: **GO** — per-fold Δ vs v1_60m +0.0089 / +0.0198 /
+0.0019 / +0.0416, stitched recent Δ +0.0352 (report archived as
`cnn_vs_lgbm_60m_ab_report_s42.md`; the live report file holds the 5-seed rerun).
Primary: statistical tie — CNN s42 vs TCN s42 splits the folds 2–2 (CNN wins March
+0.0051 and the hard June month +0.0057; TCN wins April/May), stitched recent 0.7007
vs 0.7034 (−0.0026, inside the −0.005 tolerance), pooled 0.7315 vs 0.7319. All three
escalation criteria passed → 5-seed confirmation ran (`--seeds 42,43,44,45,46
--out-suffix _v5_cnn_60m_5seed`): per-seed recent Δ +0.0204…+0.0362 (median **+0.0247**,
all seeds ≥ 2× the +0.010 bar), CNN ensemble recent **0.6971** ≥ TCN ensemble 0.6962 →
both pre-registered checks pass, so the CNN formally reaches **adoption-candidate
status alongside the TCN** (neither dominates; TCN keeps the better median seed
+0.0301 and calibration, CNN the better June/late-session and pooled ensemble 0.7304
vs 0.7232). Overfit watch: best epoch 1–3 on every seed×fold (fast saturation, ES
working), holdout↔val AUC gap ≈ 0 on folds 1–3; fold 4's +0.08 gap is the June regime
shift every model shares, not memorization. Diagnostic (not pre-registered): the two
architectures are ~0.97 prob-correlated; a 50/50 TCN+CNN blend adds nothing (recent
0.6977), so a heterogeneous ensemble is dead — improvement must come from a better
single architecture (the declared v6 multi-scale path).

---

## Step 19: v6_mscnn_60m — multi-scale CNN (the tie-breaker attempt)

Step 18 ended in a statistical tie (TCN ≈ CNN, prob corr ~0.97, blend dead), so the
next performance search changes the one untested ingredient: **parallel kernel
scales**. `tcn_net.MultiScaleCNNRiskNet` (registry `TCN_VERSIONS["v6_mscnn_60m"]`):
4 stages, each running k ∈ **(3, 7, 15)** branches × 32ch over the same input →
concat → 1×1 fuse to 64 (weight_norm everywhere); stages after the stem are
residual. The pad **triple-guard, masked 3-pooling and head are v5-verbatim**, so
any delta is attributable to the multi-scale trunk (~204k params; single-row CPU
inference stays ms-scale, honoring the local-CPU serving constraint). Same folds /
label / features / tod / day-holdout / overfit-watch records / frame-cache alias.

Kernel-ladder rationale (fixed before results): 3/7/15 bars ≈ bar-to-bar
microstructure / short swing / quarter-window trend — a ~2× geometric ladder with
wider scale separation than the illustrative {3,5,9}. k=31 rejected: ~2× per-stage
cost while 4 stacked stages already give the k=15 path a 57-bar receptive field and
the global pooling sees all 60 bars anyway. Dilated branches rejected: that family
IS the TCN, and Step 18 showed dilation vs plain conv changes almost nothing.

**Pre-registered protocol (Step 18 structure, incumbents updated)**: single seed 42
per fold. Primary = v6 s42 vs **TCN s42 (recent 0.7034)** and **CNN s42 (0.7007)**
side-by-side; Secondary = v6 s42 vs v1_60m (gate verbatim, report
`mscnn_vs_lgbm_60m_ab_report.md`); Reference = the 5-seed ensembles (TCN 0.6962 /
CNN 0.6971). **5-seed escalation iff** ① stitched recent AUC v6 s42 ≥ 0.7034 − 0.005
(best incumbent single-seed), ② vs v1_60m recent Δ ≥ +0.010, ③ each recent fold
(3, 4) vs v1_60m Δ > −0.005. **After 5 seeds**: median-seed recent Δ ≥ +0.010 AND
v6 ensemble recent ≥ 0.6971 (best incumbent ensemble) → adoption candidate; the tie
counts as BROKEN only if v6 ensemble recent ≥ **0.7021** (+0.005 over the best
incumbent) — anything between is another tie and the lineage stops there.

```powershell
# server (single seed 42, all 4 folds; reuses the existing frame cache):
#   python scripts/train_tcn_risk_model.py --tcn-version v6_mscnn_60m --device cuda:0 --frame-cache db/frame_cache
# local gate + both incumbents side-by-side (8 ablation slots):
..\Python311-64\python.exe scripts\evaluate_tcn_gate.py --tcn-version v6_mscnn_60m ^
    --ablation-predictions tcn_s42_fold1=risk_model_predictions_v4_tcn_60m_fold1.parquet ^
    --ablation-predictions tcn_s42_fold2=risk_model_predictions_v4_tcn_60m_fold2.parquet ^
    --ablation-predictions tcn_s42_fold3=risk_model_predictions_v4_tcn_60m_fold3.parquet ^
    --ablation-predictions tcn_s42_fold4=risk_model_predictions_v4_tcn_60m_fold4.parquet ^
    --ablation-predictions cnn_s42_fold1=risk_model_predictions_v5_cnn_60m_fold1.parquet ^
    --ablation-predictions cnn_s42_fold2=risk_model_predictions_v5_cnn_60m_fold2.parquet ^
    --ablation-predictions cnn_s42_fold3=risk_model_predictions_v5_cnn_60m_fold3.parquet ^
    --ablation-predictions cnn_s42_fold4=risk_model_predictions_v5_cnn_60m_fold4.parquet
```

**Results (2026-07-16) — lineage CLOSED at the pre-registered line.** Secondary: GO-quality
vs v1_60m (folds Δ +0.0071/+0.0269/+0.0068/+0.0328, stitched recent Δ +0.0323, worst
recent month +0.0068). Primary: v6 s42 takes folds 2 and 3 outright (0.7605 / 0.7236,
best of the three architectures) but is the worst of the three on the decisive June
fold (0.6757 vs TCN 0.6789 / CNN 0.6846), leaving stitched recent **0.6978** vs the
escalation bar 0.6984 (TCN s42 − 0.005) — **criterion ① missed by 0.0005**, ②③ passed;
per pre-registration no 5-seed run and the lineage stops (artifacts archived). Pooled
is a three-way dead heat (v6 0.73153 / TCN 0.73187 / CNN 0.73145). Overfit watch:
best epoch 1 on every fold with final train loss 0.34–0.36 (v5: 0.63–0.79) yet
holdout AUCs identical to v5's — the 2.5× capacity memorized train noise faster and
bought zero generalization. Read: three conv families (dilated, plain, multi-scale)
land within ±0.0004 pooled AUC of each other — the 60m label on these features looks
**information-ceiling-bound (~0.73 pooled / ~0.70 recent)**, so further architecture
search inside the conv family is a dead end; the next lever is data/features/label
(e.g. July fold 5 + production train per Step 17, or new feature families), not nets.

---

## Step 20: production run — train through June, July sanity window, pick TCN vs CNN

Resumes the Step 17 production plan (USER-decided): retrain the two adoption
candidates on `rm.PRODUCTION_SPLIT_60M` — train 2025-06-02→**2026-06-30** (the models
finally see the June drawdown) — and sanity-check on **2026-07-01→07-10**, then ship
whichever is better. The frozen 4-fold walk-forward tuple is untouched; production
uses its own registry entries so no sealed artifact can be clobbered:
`v1_60m_prod` (champion floor, LightGBM, local) and `v4_tcn_60m_prod` /
`v5_cnn_60m_prod` (candidates, GPU server, **pre-registered 5-seed mean** — a single
seed's regime luck must not pick the deployable weights; registry `split` pins the
dates and the CLI refuses explicit ones). Research-only: serving wiring stays
Phase-2 work after selection.

**Data backfill (2026-07-16)**: `bars_1m` already reached 07-13 (collectors current);
the derived pipeline (`build_features` → `merge_features` → `build_relative_features`
→ `make_labels` → `build_rule_signals` → `quality_check` → `online_alert_backtest` →
`evaluate_directional_alerts`) was rerun to extend features/labels/alert catalogue
from 06-22 to 07-13, and the frame cache re-exported (`--export-frame-cache`) — the
server needs the TWO refreshed cache parquets re-shipped.

**Pre-registered selection rule** (`scripts/select_prod_60m.py`, fixed before any
July row was scored — the ~8-day window is small enough to eyeball-hack): main
population only (time ≤ 14:30); ensemble = asserted mean-of-5-seeds; sanity floor =
candidate AUC ≥ v1_60m_prod AUC (both below → NO-SHIP, investigate July); pick the
higher-AUC candidate iff |ΔAUC| ≥ **0.005**, else the tie goes to the **CNN** on the
frozen walk-forward evidence (ens recent 0.6971 > 0.6962, best June fold, best
late-session) — July never re-litigates the tie-break.

```powershell
# 1. local: champion floor + refreshed frame cache
..\Python311-64\python.exe scripts\train_risk_model.py --risk-model-version v1_60m_prod --target target_mae_risk_60m
..\Python311-64\python.exe scripts\train_tcn_risk_model.py --tcn-version v4_tcn_60m --export-frame-cache db\frame_cache
# 2. server (ship db/frame_cache/*.parquet + the updated src/scripts files first):
#    python scripts/train_tcn_risk_model.py --tcn-version v4_tcn_60m_prod --device cuda:0 --frame-cache db/frame_cache
#    python scripts/train_tcn_risk_model.py --tcn-version v5_cnn_60m_prod --device cuda:0 --frame-cache db/frame_cache
# 3. local: pre-registered selection (needs all three predictions parquets back)
..\Python311-64\python.exe scripts\select_prod_60m.py
```

**Results (2026-07-16) — SELECT `v5_cnn_60m_prod`.** July 01–10 (n_main=16,298,
pos_rate 37% — a hostile regime): champion floor v1_60m_prod AUC **0.5558** with
BROKEN buckets (top10 0.320 < bottom20 0.322); both June-trained candidates clear it
by ~+0.10 AND restore bucket separation — TCN 0.6509 (top10 0.542/bottom20 0.197),
CNN **0.6555** (0.567/0.192, every seed 0.6453–0.6609). ΔAUC = 0.0046 < the 0.005
margin → the pre-registered tie-break fired (→ CNN; the raw ranking agreed anyway).
Per-day: candidates beat the champion 7/8 days; 07-02/07-08 were hard for everyone.
Overfit watch: best epoch 1 on all 10 runs, holdout AUCs 0.68 (consistent with the
research folds). Read: training through the June drawdown is what saved the July
window — the LightGBM collapsed there while both sequence models kept separating.
**`models/risk_model_v5_cnn_60m_prod_seed{42..46}.pt` (5-seed mean) is the selected
production candidate. Still research-only: Phase-2 serving (sequence-window serving
parity + band thresholds) must pass before any live wiring; v1.1 stays live.**

---

## Replay app (app/)

A mobile-first web app that re-plays a past trading day minute by minute and shows the
alerts the realtime runners would have printed, terminal-log style (long+short trend
lines and MA5x20 touch lines for 000660), with each alert's outcome revealed only after
its window has passed. Nothing is recomputed in the browser:
`scripts/export_replay_bundle.py` runs the runners' own replay code (captured
`_process_minute` lines; TouchAlertRunner records with out-of-fold grades), pins the
cross-section to the 20-symbol training universe, and writes `app/public/data/`. Every
replay date is out-of-sample for the models it shows. Parity: exported raw lines match
the runner's stdout line for line (2026-06-19: 361/361 trend, 22/22 touch).
Logic in `src/replay_export.py` (tests: `tests/test_replay_export.py`); app, build and
Firebase Hosting deploy steps in [app/README.md](app/README.md).

```powershell
..\Python311-64\python.exe scripts\export_replay_bundle.py     # every date in app/replay_dates.json
```

---

## Legacy single-symbol components

These predate the unified `bars_1m` store and write to the **older** single-symbol
tables (`minute_bars`, `realtime_minute_bars`, `minute_features`). Kept for
reference; not part of the `bars_1m` pipeline above.

- **Historical (opt10080 → `minute_bars` + parquet):**
  `..\Python311-32\python.exe scripts\collect_000660_1m.py --symbol 000660 --tick 1 --max-pages 10`
- **Realtime builder + comparison (→ `realtime_minute_bars`):**
  `..\Python311-32\python.exe scripts\run_realtime_000660.py --symbol 000660`
  then `..\Python311-64\python.exe scripts\compare_realtime_vs_opt10080.py --symbol 000660 --date YYYY-MM-DD`
- **Feature builder v0 (→ `minute_features`, `data/processed/features_<symbol>_1m.parquet`):**
  `..\Python311-64\python.exe scripts\build_features_000660.py --symbol 000660`

Notes for the legacy collectors: no orders are ever placed (market data only);
Kiwoom requires Windows + 32-bit Python + login; features cover 09:00–15:30 and
handle invalid prices, negative volume, duplicate `symbol + ts_kst`, and infinities.
