# quant 신호 알림 프로젝트 개요

> 국내 주식 1분봉 기반 **장중 신호·알림 시스템**. 키움 1분봉을 SQLite에 쌓고 → 피처 → 규칙 신호 → 방향 알림 → 리스크 모델로 가공한 뒤, 같은 파이프라인을 실시간으로 재현해 **터미널 알림과 로그**를 남긴다.
> **주문은 하지 않는다.** 모든 러너는 판단 보조용이며 주문 API를 import하거나 호출하지 않는다(테스트로 가드).

기준일: 2026-10-10 (데이터·아티팩트는 2026-07-23 시점) · 상세 런북: [README.md](README.md)

---

## 1. 한눈에 보기

| 항목 | 현재 값 |
|---|---|
| 데이터 | `bars_1m` 2025-06-02 ~ 2026-07-22, 21종목 × 3개 source (al_raw 364만 행, al_regular 218만 행) |
| 유니버스 | `config/symbols_test.yaml`의 `active: true` 21종목 (2026-07-22에 396500 TIGER 반도체TOP10 추가) |
| 코드 | `src/` 모듈 49개(약 11.7k줄), `scripts/` 진입점 65개, `tests/` 32개 파일(테스트 519개) |
| DB | `db/market_data.sqlite` 테이블 112개 |
| 실전 리스크 모델 | **v1 LightGBM + `--cross-symbol` (v1.1)** |
| 실전 후보(미연결) | `v5_cnn_60m_prod` 5시드 CNN (Step 20 선택, Phase-2 서빙 대기) |
| 터치 알림 | MA5x20 돌파 + CNN 등급 (Step 22). **현재 등급 비활성**(아래 6절) |

---

## 2. 실행 환경: 32비트 / 64비트 분리

키움 OpenAPI+는 32비트 ActiveX라 32비트 Python에서만 돌고, LightGBM·torch는 64비트가 필요하다. 두 프로세스는 **SQLite 테이블로만** 연결된다.

```
[32-bit  ..\Python311-32]                       [64-bit  ..\Python311-64]
키움 로그인 + 실시간 틱                            pandas / LightGBM / torch
 realtime_alert_collector.py ──► realtime_minute_bars ──► 실시간 러너 (1분 폴링)
 collect_kiwoom_to_bars.py   ──► bars_1m              ──► 오프라인 파이프라인·학습·평가
                   (db/market_data.sqlite 하나가 브리지)
```

- 수집기는 틱을 1분봉으로 묶고 **완성된 봉만** `realtime_minute_bars`에 쓴다. DB의 심볼은 6자리 기본 코드(`000660`)로 저장한다(`000660_AL` 구독 시에도).
- 크로스심볼 피처·터치 알림은 전 종목 봉이 필요하므로 수집기를 `--all-symbols`로 띄운다.

---

## 3. 공통 원칙

- **모델 입력 = `al_regular`** (통합시세, 09:00–15:30). `al_raw`(08:00–20:00)는 원본 보관과 장전 피처, `single_regular`는 비교용.
- **엄격한 인과성**: 모든 피처는 `decision_time` 이전 봉만 본다. `decision_time` = 피처 봉의 `bar_end`.
- **라벨은 평가 전용**: 미래 값은 `labels` 테이블과 학습 타깃에만 존재. 신호 생성 코드가 라벨을 읽지 않는지 테스트로 가드.
- **종목코드 하드코딩 금지**: `load_active_symbols()`로만 읽는다. `src/`·`scripts/`의 6자리 코드를 스캔하는 가드 테스트가 있다.
- **사전등록 게이트**: 모델 도전자는 결과를 보기 전에 시드·판정 기준을 코드에 고정하고, 같은 판정 함수로 GO / NO-GO를 낸다.
- **additive-only**: 기존 모델·서빙 경로는 바이트 단위로 유지하고 새 기능은 새 파일·새 레지스트리로 추가.

---

## 4. 오프라인 파이프라인 (Step 0–13_5)

| Step | 스크립트 | 결과 | 비고 |
|---|---|---|---|
| 0–1 | `show_config.py` | `config/config.yaml`, `symbols_test.yaml` | 세션 시간·신호 한도·종목 고정 |
| 2–3 | `import_minute_csv.py`, `collect_kiwoom_to_bars.py` | `bars_1m` | PK `(symbol, bar_start, source)`, 멱등 |
| 4 | `validate_model_data.py` | 리포트 | 이상치 기록만, 삭제 안 함 |
| 5 | `resample_bars.py` | `bars_3m`, `bars_5m` | 09:00 기준 정렬, 완전한 버킷만 |
| 6 | `quality_check.py` | 품질 리포트 | FAIL / WARN / OK |
| 7 | `build_features.py` | `features_1m/3m/5m` | 타임프레임 접미어, 일 경계 미교차 |
| 8 | `merge_features.py` | `features_merged` | backward as-of 병합 + 누수 검사 |
| 9 | `build_relative_features.py` | `features_relative` | 같은 시각 종목 간 순위·상대값·breadth |
| 10 | `make_labels.py` | `labels` | up/down 2·3·5%, 60m~eod, has_full |
| 11 | `build_rule_signals.py` | `rule_signals` | 사람이 읽는 up/down 점수 + 사유 |
| 12 | `daily_topk_backtest.py`, `compare_entry_variants.py` | 백테스트 테이블 | 하루 top1/top2, 진입 변형 비교 |
| 13 | `threshold_sweep.py` | 스윕 요약 | conservative + 3분 지연 진입 기준 |
| 13_5 | `online_alert_backtest.py` | `directional_alerts` | 5개 알림 타입, A/B/C/RISK 등급 |

**Step 12–13에서 얻은 결론**: 규칙 신호로 자동매매할 만한 엣지는 약하다(평균 효과가 비용 바닥 ~0.2% 안). 그래서 Step 13_5에서 `rule_signals`를 **진입 신호가 아닌 방향 알림 피드**로 재정의했다.

**방향 알림 타입** (`src/directional_alert.py`)

| 타입 | 의미 |
|---|---|
| `UP_ALERT` | 상방 가능성 |
| `UP_CONTINUATION` | 상방 유지 |
| `RISK_ALERT` | 하방 위험 |
| `WEAKENING_ALERT` | 추세 약화 |
| `VOLATILITY_ALERT` | 변동성 확대 |

등급 A > B > C의 순서는 실제로 성립하지만 절대 엣지는 작다. `RISK_ALERT` 이후에는 오히려 반등이 나온다(평균 회귀). 등급은 알림의 **상대 품질**을 나타낼 뿐 매매 신호가 아니다.

---

## 5. 모델 계층

### 5.1 롱 리스크 모델 (타깃: 이후 MAE ≤ −1.5%)

| 버전 | 내용 | 결과 | 상태 |
|---|---|---|---|
| v1 (Step 14) | LightGBM, 49개 피처, 단일 종목 neutral fill | AUC 0.66, 월별 감쇠(6월 0.59) | 기반 모델 |
| **v1.1** | v1 + 실제 유니버스 크로스섹션 피처(`--cross-symbol`) | 학습·서빙 불일치 해소 | **실전** |
| v2.0 / v2.1a / v2.1b | 유니버스 피처, 선도 종목(005930, 402340) 추가 | 최근 월 게이트 미통과 | NO-GO, 보관 |
| v3.0a / v3.0b | 장전(08–09시) 피처 | 일 단위 피처가 레짐에 과적합 | NO-GO, 보관 |

### 5.2 60분 라벨 계열 (Step 16–20, 연구용)

| Step | 모델 | 결과 |
|---|---|---|
| 16 | v4 TCN (60봉 시퀀스) | 5시드 GO. 단 엣지의 상당 부분이 시각(time-of-day) 채널에서 나옴 |
| 17 | 60분 고정 라벨로 교체, 4폴드 walk-forward | TCN이 4폴드 모두 승리(최근 Δ+0.038) → 아키텍처 가치 확인 |
| 18 | v5 1D-CNN | 5시드 판정 통과, TCN과 통계적 동률(앙상블 0.6971 vs 0.6962) |
| 19 | v6 멀티스케일 CNN | 0.0005 차이로 탈락. conv 계열 정보 상한 ~0.73 |
| 20 | 6월까지 학습한 prod 재학습, 7월로 검증 | **v5_cnn_60m_prod 선택**: 7월 AUC 0.6555 (TCN 0.6509, LightGBM 0.5558) |

`models/risk_model_v5_cnn_60m_prod_seed{42..46}.pt`가 실전 후보다. 시퀀스 서빙 파리티와 위험 구간 재설정(Phase-2)을 통과하기 전까지는 연결하지 않는다.

### 5.3 기타 모델

| 모델 | 용도 | 상태 |
|---|---|---|
| v1_short (LightGBM) | 하방 알림의 `short_adverse_prob` (이후 MFE ≥ +1.5%) | AUC 0.684, 알림에 표시만(거부 안 함) |
| touch CNN prod (5시드) | MA5x20 돌파 후 10분 안에 +0.5% 터치 여부 | 게이트 GO: 최근 AUC 0.713 vs LightGBM 0.661, 7월 0.643 vs 0.534 |

---

## 6. 실시간 알림 계층

모든 러너는 64비트, `--mode replay --replay-date YYYY-MM-DD`(키움 없이 과거 하루 재생)와 `--mode live`를 지원한다.

| 러너 | 답하는 질문 | 출력 / 기록 |
|---|---|---|
| `realtime_trend_signal.py` | 이 종목(기본 000660) 추세가 지금 어떤가 | 1분마다 한 줄 + 🔔 알림, `realtime_alert_log` |
| `realtime_trend_alert.py` | 위와 같음, 차트 포함 | matplotlib 캔들 차트 |
| `realtime_short_trend_alert.py` | 하방 쪽 추세 | 🔻 알림, `short_alert_log` |
| `realtime_long_short_trend_alert.py` | 롱·숏을 같은 기준으로 | ▲ / 🔻 알림, 쿨다운 공유 |
| `realtime_touch_alert.py` | 돌파가 나왔고 그 등급은 무엇인가 | 1단 돌파(무조건) + 2단 CNN 등급, `touch_alert_log` |
| `realtime_position_monitor.py` | 산 포지션을 계속 들고 있어도 되나 | HOLD → STRONG_SELL_WARNING, `position_exit_log` |
| `check_realtime_bars.py` | 봉이 잘 쌓이고 종목 간 시각이 맞나 | 누적·정렬 그리드 |

### 6.1 추세 알림 1분 루프 (`TrendAlertEngine`)

```
수집기가 완성된 1분봉 기록 → realtime_minute_bars
 → 러너가 폴링, 새 봉만 처리 (장 시작 전 봉 제외)
 → build_serving_features : 학습과 같은 feature_lib / resampler / merge (서빙 파리티 0.0 검증)
    └ --cross-symbol 이면 유니버스 전체로 순위·상대값·breadth 재계산
 → rule_signal → directional_alert (타입·등급) → risk model (risk_prob)
 → 위험 구간: ≥0.60 HIGH / ≥0.45 MED / 그 외 LOW
 → _maybe_fire : 09:10–15:00 시간대 + 같은 타입 쿨다운
 → 출력 · DB · CSV 원장
```

- 라이브 시작 시 당일 백로그(장 시작~현재)를 먼저 재생해 아침 신호도 출력한다. 백로그 알림은 DB에 다시 쓰지 않는다.
- `--tick-cross`를 주면 5분 이동평균 종가 교차(📈/📉)를 추가로 표시한다.
- 상방 알림인데 risk가 HIGH이면 `⚠ 상방이지만 고위험`으로 표시한다. 리스크 모델의 역할이 이것이다.

### 6.2 터치 알림 2단 구조 (`TouchAlertRunner`)

- **1단(돌파)**: 정규장에서 확정된 MA5x20 종가 교차는 모두 알린다. 쿨다운 없음. 직전 반대 교차가 가까우면 휩쏘 재교차로 표시만 한다.
- **2단(등급)**: 피처가 있는 돌파(`features_1m`이 약 10:01부터 생김)이면서 10분 결과 창이 정규장 안에 들어오는 경우(15:21 이전)에만 붙는다. CNN 점수를 고정된 OOF 등급맵(A/B/C/D)에 매핑하고, 해당 등급의 **과거 OOF 터치율**을 함께 표시한다. 점수는 보정되지 않았으므로 확률로 표시하지 않는다.
- 새 봉이 있을 때만 전 종목을 한 번에 재계산하고, CNN은 새 돌파에만 돌린다.
- 등급 산출이 실패해도 1단 돌파 알림은 계속 나온다.

| 등급 | OOF 표본 | 과거 터치율 |
|---|---|---|
| A (상위 10%) | 3,730 | 41.7% |
| B (70–90%) | 7,459 | 27.0% |
| C (20–70%) | 18,648 | 12.4% |
| D (하위 20%) | 7,460 | 2.5% |

### 6.3 포지션 청산 경고 (`realtime_position_monitor.py`)

수동으로 매수한 포지션에 대해 `TrendAlertEngine` 스냅샷과 `src/position_exit.py` 규칙을 결합한다. 규칙: 손절, 수익 반납, 트레일링, MA5/VWAP 이탈, 위험 + 약화, 반대 신호, 시간 손절. 롱 전용이며 가장 강한 상태가 우선한다.

---

## 7. 연구로 확인한 것 (실전 미연결)

| 연구 | 결론 |
|---|---|
| 아침 🔔 → 오후 고가/저가 (000660) | 첫 아침 신호는 기준선을 이기지 못함(롱 +0.19%, 유의하지 않음). 이전 결과는 워밍업 절단 착시였음 |
| 하방 알림의 숏 진입 엣지 | 없음. 하방 타입은 정보용 알림으로만 유지 |
| MA5x20 1분 교차 | 교차 진입 자체는 엣지 없음. 휩쏘 재교차는 유효한 청산 신호(MAE ≤ −0.7% 확률 ~80%) |
| 진입 변형 (Step 12) | 3분 지연 진입이 약간 유리, 장전 약세 페널티는 모든 지표를 악화 |

---

## 8. 폴더 구조

```
quant/
├─ config/          config.yaml(세션·신호 한도·선도 종목), symbols_test.yaml(종목)
├─ src/             순수 로직: 피처·라벨·신호·알림·모델·서빙 어댑터
│   ├─ feature_lib, features_1m/3m/5m, feature_merge, relative_features   (피처)
│   ├─ labeling, rule_signal, directional_alert, short_entry              (신호·알림)
│   ├─ risk_model, tcn_data, tcn_net, tcn_model, touch_model              (모델)
│   ├─ realtime_features, touch_serving, position_exit                    (서빙)
│   └─ kiwoom_client, minute_builder, realtime_receiver                   (32비트 수집)
├─ scripts/         실행 진입점: build_* / train_* / evaluate_* / realtime_* / check_*
├─ tests/           unittest (누수·인과성·파리티·주문 API 부재·하드코딩 가드)
├─ models/          *.txt(LightGBM), *.pt(torch) + *.meta.json, touch_grade_map.json
├─ db/              market_data.sqlite, frame_cache/(GPU 서버용 parquet)
├─ data/processed/  리포트(.md/.csv), 알림 원장 CSV, 예측 parquet
└─ logs/            단계별 실행 로그
```

---

## 9. 현재 상태와 다음 과제

1. **터치 알림 등급 비활성**: 2026-07-22에 396500을 활성화하면서 유니버스가 20 → 21종목이 됐다. 터치 prod 모델은 종목 one-hot을 쓰기 때문에 등급 산출을 거부하고 있고, 러너는 돌파 전용 모드로 동작한다(원장 558행이 모두 이 상태). 21종목으로 prod 재학습과 등급맵 재생성을 하거나, 396500을 다시 비활성화해야 복구된다.
2. **실전 리스크 모델 교체 대기**: v1.1 LightGBM은 7월에 AUC 0.556으로 무너졌고, v5_cnn_60m_prod는 0.656을 냈다. Phase-2 서빙(시퀀스 서빙 파리티, 위험 구간 재설정)이 남아 있다.
3. **유니버스 변경 영향 점검**: v1.1의 크로스섹션 피처(순위·breadth)도 21종목 기준으로 바뀌었다. 하드 실패는 아니지만 분포 이동을 확인할 필요가 있다.
4. **README 갱신**: Step 21–22(터치 예측·터치 알림 서빙) 섹션이 아직 없다.

---

## 10. 자주 쓰는 명령 (cwd = quant)

```powershell
# 실시간 (장중, 32비트 키움 로그인 필요)
..\Python311-32\python.exe scripts\realtime_alert_collector.py --all-symbols
..\Python311-64\python.exe scripts\check_realtime_bars.py --watch 30
..\Python311-64\python.exe scripts\realtime_trend_signal.py --mode live --code 000660 --cross-symbol
..\Python311-64\python.exe scripts\realtime_long_short_trend_alert.py --mode live --cross-symbol
..\Python311-64\python.exe scripts\realtime_touch_alert.py --mode live

# 리플레이 (키움 없이)
..\Python311-64\python.exe scripts\realtime_trend_signal.py --mode replay --replay-date 2026-06-19 --cross-symbol
..\Python311-64\python.exe scripts\realtime_touch_alert.py --mode replay --replay-date 2026-07-10

# 테스트
..\Python311-64\python.exe -m unittest discover -s tests
```
