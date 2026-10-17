# QnA — 실시간 알림/수집 파이프라인 동작 확인

> 코드를 읽고 확인한 내용만 정리한 문서입니다. 구현 파일은 수정하지 않았습니다.

**파트 1 기준 파일**: `scripts/realtime_short_trend_alert.py`, `src/short_entry.py`,
`scripts/realtime_trend_alert.py`(재사용되는 LONG 엔진), `src/directional_alert.py`,
`src/realtime_features.py`

**파트 2 기준 파일**: `scripts/realtime_alert_collector.py`, `src/kiwoom_client.py`,
`src/realtime_receiver.py`, `src/minute_builder.py`, `src/minute_collector.py`,
`src/config.py`, `src/settings.py`, `src/historical_collector.py`, `config/config.yaml`
(+ 실제 `db/market_data.sqlite` 스키마/데이터 대조 확인)

**파트 6 기준 파일**: `README.md`(프로젝트 공식 스펙 문서) + `src/`/`scripts/`/`tests/` 전체 목록 대조

---

# 파트 1 — SHORT/LONG 트렌드 알림 러너 (`realtime_short_trend_alert.py`)

---

## Q1. "fire-gate pass"가 뜨는 건 직전 알림이랑 같은 타입인데 cooldown 걸린 상태인 거지?

**기본 설정 기준으로는 맞습니다.** 좀 더 정확히 말하면:

`*fire-gate pass (not fired)` 노트는 `_status_body()`(`scripts/realtime_short_trend_alert.py:151-164`)에서
`srow["short_fires_rule"]`(= `short_alert_fire_gate()` 결과, "이 봉은 규칙상 알림감이다")가 **True인데도**
`ShortFireGate.maybe_fire()`가 실제로는 `None`을 반환해서 **발동은 안 된** 경우에만 표시됩니다
(`_process_minute()`가 `down_fired`가 False일 때만 `_status_body()`의 note를 사용 — line 278-281).

`ShortFireGate.maybe_fire()`의 체크 순서는 (`src/short_entry.py:429-466`):

1. 세션 시간(`min_time`~`max_time`, 기본 `09:10:00`~`15:00:00`) 안인지
2. `short_fires_rule`이 True인지 (여기가 위 note의 전제 조건)
3. `alert_type`이 비어있지 않은지
4. (옵션) 모델 veto: `short_adverse_prob_max`가 설정된 경우, DOWN 타입(`DOWN_ALERT`/`DOWN_CONTINUATION`)만 확률 체크
5. **같은 alert_type의 직전 발동 시각으로부터 `cooldown_minutes`(기본 15분) 이내인지 — per-type 쿨다운**
6. (옵션) `daily_alert_limit` 도달 여부

러너 기본값은 `short_adverse_prob_max=None`(veto 꺼짐), `daily_alert_limit=None`(일일 한도 꺼짐)이므로,
**3번(빈 타입)은 애초에 `short_fires_rule=True`면 성립 불가**하고, 4·6번은 기본 설정에서 아예 작동하지
않습니다. 따라서 기본 실행에서 "규칙은 통과했는데 안 뜬" 경우의 사실상 유일한 원인은 **5번 — 같은
타입이 쿨다운(기본 15분) 안에 이미 발동한 상태**입니다.

- "직전 알림"은 반드시 *바로 이전 줄*일 필요는 없습니다. 쿨다운은 **타입별로 독립**(`_last_fire_at`이
  `{alert_type: 마지막 발동 시각}` 딕셔너리, `src/short_entry.py:419`)이라서, 그 사이에 다른 타입 상태
  줄이 여러 번 찍혀도 상관없이 **"그 alert_type 자체의 가장 최근 발동"** 기준으로 15분을 셉니다.
- 엄밀히 예외 케이스 2가지도 있습니다 (기본 실행에서는 거의 안 보임):
  - 세션 시간 밖(장 시작 직후/마감 직전)에서도 `short_fires_rule`은 시간 제약 없이 계산되므로,
    이론상 그 구간에서도 note가 뜰 수 있는데 이때 원인은 쿨다운이 아니라 세션 게이트입니다.
  - `--short-adverse-prob-max`나 `--daily-alert-limit`을 직접 켠 경우, 그게 원인일 수도 있습니다.

### Q1 보충 — "모델 veto"(`short_adverse_prob_max`)란 정확히 무엇인가

v1_short 리스크 모델의 예측 확률이 **알림 발동 자체를 막을 수 있게 하는 옵션** 기능. 기본 꺼짐.

- **기본(`None`)**: veto 꺼짐 — `short_adverse_prob`는 메시지에 참고 표시(`✓ 저위험 하방` /
  `⚠ 모델 상승위험: 하방이지만 반등 주의`)만 되고 발동 여부엔 영향 없음(annotation-first).
- **`--short-adverse-prob-max <값>`을 켜면**: `DOWN_ALERT`/`DOWN_CONTINUATION` 딱 이 2개 타입만,
  `short_adverse_prob >= 설정값`이면 규칙(`short_fires_rule`)이 통과했어도 발동을 막음
  (`src/short_entry.py:455-458`: `if not (prob < thr.short_adverse_prob_max): return None`).
- **왜 이 2개 타입만**: `DOWN_TYPES = {"DOWN_ALERT", "DOWN_CONTINUATION"}`(`short_entry.py:63`)만
  veto 대상. 나머지 3개(`SHORT_RISK_ALERT`/`DOWN_WEAKENING_ALERT`/`VOLATILITY_ALERT`)는 애초에
  "위험/반전 경고" 자체가 목적인 정보성 알림이라 확률로 숨기면 존재 이유가 사라지므로 대상 아님.
- **NaN 처리(보수적)**: 모델 미사용(`--no-short-model`)이나 워밍업 중이라 확률이 없으면 NaN인데,
  `NaN < 값`은 항상 `False`라서 `not(...)`이 `True`가 되어 **점수를 모르는 행도 veto당함**(주석:
  "NaN fails the `<` comparison too, so an unscored DOWN row is vetoed") — "모르면 일단 막는다".
- **Q1과의 연결**: veto를 켜면 `*fire-gate pass (not fired)` 노트의 원인이 **쿨다운 OR 모델veto**
  둘 중 하나가 됨. `short_alert_fire_gate`(규칙 판정)는 veto를 아예 모르고 계산되므로
  (`short_entry.py:315-317`) 화면에 구분 표시가 따로 없음 — 같은 줄의 `adverse=X.XX[밴드]`를
  본인이 설정한 `--short-adverse-prob-max`와 직접 비교해야 어느 쪽인지 알 수 있음.
- **참고**: 과거 스윕 결과 DOWN 타입 자체가 숏 진입 엣지가 없었음(Q3 마지막, `short_entry_gate_report.md`)
  — 이 veto는 "모델이 걸러주면 더 나을 것"이라는 아직 백테스트로 증명 안 된 실험적 옵션.

---

## Q2. LONG alert(▲)는 언제, 어떤 것이 뜨는가

`realtime_short_trend_alert.py`는 LONG 판단을 **직접 계산하지 않고** 기존
`scripts/realtime_trend_alert.py`의 `TrendAlertEngine`을 그대로 재사용합니다(`import realtime_trend_alert as rta`).
매 분봉마다 `engine.on_new_bar(bar)` → 내부적으로 `directional_alert.compute_directional_alerts()`로
분류 → `TrendAlertEngine._maybe_fire()`가 발동 여부 결정, 순서입니다.

### 5개 타입 분류 조건 (`src/directional_alert.py:119-241`, 우선순위 순 — 먼저 매치되는 것이 채택)

| 타입 | 한글 | 조건(요약) |
|---|---|---|
| `RISK_ALERT` | 하방 위험 | `risk_score >= 3.0` 이고 `risk_score > alert_score` |
| `WEAKENING_ALERT` | 추세 약화 | (RISK 아니고) `ret_20m>0 & ret_5m<0` 이거나 `up_score>=3.0 & 윗꼬리(0.3%+)` |
| `UP_CONTINUATION` | 상방 유지 가능성 | `alert_score>=4.0` & MA5 위 & 고점 0.5% 이내 & `ret_5m>0` |
| `UP_ALERT` | 상방 가능성 | `direction_score>=1.5` & `up_score>=3.0` |
| `VOLATILITY_ALERT` | 변동성 확대 | `volume_z_20m>=2.0` 또는 `range_1m>=0.6%`, 단 `|direction_score|<1.5`(방향 불분명) |
| (없음) | — | 위 5개 다 해당 없으면 `alert_type=""` → 대부분의 봉이 여기 |

- `alert_score = up_score + (MA5 위면 +1)`, `risk_score = down_score + 윗꼬리(1.5) + VWAP상단이탈(1.0,+1%) + 고점exhaustion(1.0) + 거래량스파이크(0.5)`
- 등급(`alert_grade`): `risk_score>=3.0` 또는 타입이 RISK면 `RISK`; `alert_score>=4.5 & risk_score<=1.5 & direction>=3.0 & 상대강도>0`면 `A`; `alert_score>=3.5 & risk_score<=2.5 & direction>=1.5`면 `B`; 나머지 `C`; 타입 자체가 없으면 등급도 빈 문자열.

### 실제로 화면에 "뜨는"(▲ 발동) 조건 — `TrendAlertEngine._maybe_fire()` (`scripts/realtime_trend_alert.py:257-279`)

1. 시각이 `09:10:00 ~ 15:00:00`(모듈 상수 `MIN_TIME`/`MAX_TIME`, 하드코딩)
2. `alert_type`이 비어있지 않음 (위 5개 중 하나로 분류됨)
3. **직전에 발동됐던 타입과 같고, 그로부터 `cooldown`(기본 15분, `DEFAULT_COOLDOWN_MIN`) 이내면 억제**

**중요하게 확인된 사실**: `AlertThresholds`(코드에 `up_threshold=4.0, score_margin=3.0, down_risk_limit=2.0,
current_rank_limit=5, cooldown_minutes=30, daily_alert_limit=5` 등 필드가 존재)와 그걸로 계산하는
`up_alert_gate`/`alert_fire_gate`는 `realtime_features.score_serving()`이 `alerts["fires"]` 컬럼을 만드는
데만 쓰이고 (`src/realtime_features.py:212-215`), **`TrendAlertEngine._maybe_fire()`는 이 `fires` 값을
어디서도 읽지 않습니다.** 즉 지금 이 실시간 엔진 기준으로는:

- LONG 알림은 **분류만 되면(5타입 중 하나) + 세션시간 + 쿨다운**만으로 뜹니다. `up_score>=4.0`, 순위
  top5, `down_score<=2.0` 같은 `AlertThresholds`의 진입 스크리닝 조건은 **온라인 발동에 관여하지
  않습니다** (Step 13_5에서 "자동 진입 신호"가 아니라 "사람이 보는 알림 피드"로 재정의된 설계와 일치).
- 쿨다운도 **타입별이 아니라 엔진 전체에 스칼라 1개**(`_last_fire_at`/`_last_fire_type`, dict 아님,
  `scripts/realtime_trend_alert.py:112-113`)만 기억합니다. 그래서 "다른 타입이 중간에 한 번 뜨면", 원래
  타입의 15분 타이머가 사실상 리셋되어 **더 일찍 다시 뜰 수 있습니다** (SHORT 쪽과의 핵심 차이, Q4 참고).
  - 예: 10:00 `RISK_ALERT` 발동 → 10:05 `UP_ALERT` 발동(다른 타입이라 즉시 발동, "마지막 타입"이
    `UP_ALERT`로 교체) → 10:07 다시 `RISK_ALERT` 조건 충족 시, 직전 발동 타입(`UP_ALERT`)과 다르므로
    **10:00 이후 7분밖에 안 지났어도 즉시 발동**합니다.

### 메시지 예시 (`trend_message()`, `scripts/realtime_trend_alert.py:78-95`)
```
상방 가능성 [UP_ALERT B] risk_prob=0.32(LOW)  ✓ 저위험 상방  reasons=uptrend_20m/above_5ma
상방 유지 가능성 [UP_CONTINUATION A] risk_prob=0.62(HIGH)  ⚠ 모델 고위험: 상방이지만 위험 (risk filter)  reasons=...
하방 위험 [RISK_ALERT RISK] risk_prob=0.71(HIGH)  ⚠ 하방+고위험 일치  reasons=...
```

---

## Q3. short_alert(🔻)는 언제, 어떤 것이 뜨는가

`src/short_entry.py`가 담당하며, LONG 쪽의 부호를 뒤집은 완전 대칭 구조입니다
(`down_score`↔`up_score`, MA5 아래/VWAP 아래/저점 등).

### 5개 타입 분류 조건 (`src/short_entry.py:177-306`, 우선순위 순)

| 타입 | 한글 | 조건(요약) |
|---|---|---|
| `SHORT_RISK_ALERT` | 상승 위험 | `upside_risk_score>=3.0` 이고 `upside_risk_score > short_score` |
| `DOWN_WEAKENING_ALERT` | 하락 추세 약화 | (SHORT_RISK 아니고) `ret_20m<0 & ret_5m>0` 이거나 `down_score>=3.0 & 아랫꼬리(0.3%+)` |
| `DOWN_CONTINUATION` | 하방 유지 가능성 | `short_score>=4.0` & MA5 아래 & 저점 0.5% 이내 & `ret_5m<0` |
| `DOWN_ALERT` | 하방 가능성 | `direction_score<=-1.5` & `down_score>=3.0` |
| `VOLATILITY_ALERT` | 변동성 확대 | `volume_z_20m>=2.0` 또는 `range_1m>=0.6%`, `|direction_score|<1.5` |
| (없음) | — | 대부분의 봉 |

- `short_score = down_score + (MA5 아래면 +1)`, `upside_risk_score = up_score + 아랫꼬리지지(1.5) + VWAP1%하단이탈(1.0) + 저점capitulation(1.0) + 거래량스파이크(0.5)`
- 등급(`short_grade`): `upside_risk_score>=3.0`이면 `SHORT_RISK`; `short_score>=4.5 & upside_risk_score<=1.5 & direction<=-3.0 & 상대약세`면 `A`; `short_score>=3.5 & upside_risk_score<=2.5 & direction<=-1.5`면 `B`; 나머지 `C`.

### 실제로 화면에 "뜨는"(🔻 발동) 조건 — 두 단계

**1단계 — `short_fires_rule` (= `short_alert_fire_gate()`, `src/short_entry.py:338-360`)**: 타입별로 별도 발동 조건
- `DOWN_ALERT`/`DOWN_CONTINUATION`: `short_entry_gate()` 통과 필요 — `down_score>=4.0`, `up_score<=1.0`,
  `direction_score<=-3.0` (기본값; `--cross-symbol` 없으면 순위 조건은 자동 비활성화)
- `SHORT_RISK_ALERT`: `upside_risk_score>=4.0` **그리고** `direction_score>=3.0`(위쪽으로 강하게 쏠릴 때만 — "하방인데 진짜 반등 위험"을 노이즈 없이 잡기 위함)
- `DOWN_WEAKENING_ALERT`: `down_score>=3.0`
- `VOLATILITY_ALERT`: `volume_z_20m>=2.5` & `range_1m>=0.6%`

**2단계 — `ShortFireGate.maybe_fire()`** (`src/short_entry.py:429-486`, Q1 참고): 세션시간(09:10-15:00) →
`short_fires_rule` True → (옵션)모델 veto → **타입별 쿨다운(기본 15분, per-type dict)** → (옵션)일일 한도.

LONG과 달리 SHORT는 **엔진이 스스로 만든 `short_fires_rule` 플래그를 실제로 체크**하므로, "규칙상
알림감"과 "실제 발동"이 분리되어 있고 그 간극이 Q1의 `*fire-gate pass` 노트로 보입니다.

### `short_adverse_prob` — 주석(annotation) 전용, 기본은 게이트 아님
v1_short 모델이 "이 하방 상황에서 역행(반등) 확률"을 매 알림/상태줄마다 표시(`short_risk_band`:
HIGH>=0.60 / MED>=0.45 / LOW). `--short-adverse-prob-max`를 명시적으로 줘야 DOWN 타입에 한해 veto로
작동하며, 기본(`None`)은 순수 참고용입니다.

### 메시지 예시 (`short_message()`, `src/short_entry.py:373-392`)
```
하방 가능성 [DOWN_ALERT B] short_adverse_prob=0.28(LOW)  ✓ 저위험 하방  reasons=downtrend_20m/momentum_falling_5m
하방 유지 가능성 [DOWN_CONTINUATION A] short_adverse_prob=0.65(HIGH)  ⚠ 모델 상승위험: 하방이지만 반등 주의  reasons=...
상승 위험 [SHORT_RISK_ALERT SHORT_RISK] short_adverse_prob=0.70(HIGH)  reasons=...
```

---

## Q4. LONG vs SHORT 핵심 차이 요약

| 항목 | LONG (`TrendAlertEngine`) | SHORT (`ShortFireGate`) |
|---|---|---|
| 분류 로직 | `compute_directional_alerts` (5타입) | `compute_short_entry_columns` (대칭 5타입) |
| 온라인 발동 게이트 | **분류(빈 타입 아님)만 체크** — `AlertThresholds`(up_threshold 등)는 계산만 되고 `_maybe_fire`에서 미사용 | **`short_fires_rule`(타입별 실제 임계값)을 체크** — `ShortThresholds`가 실제로 게이트 |
| 쿨다운 구조 | 엔진 전체 스칼라 1개(마지막 발동 타입/시각) — 다른 타입이 끼어들면 사실상 리셋 | 타입별 dict — 서로 완전히 독립 |
| 쿨다운 기본값 | 15분 (`DEFAULT_COOLDOWN_MIN`, `--cooldown-min`로 조절) | 15분 (`ShortThresholds.cooldown_minutes`, `--cooldown-min`로 조절) — **서로 다른 상태, 공유 안 함** |
| 일일 한도 | `AlertThresholds.daily_alert_limit=5` 필드는 있으나 미사용(위와 동일 이유) | `daily_alert_limit` 기본 `None`(off); 정수 지정 시 실제로 작동 |
| 모델 확률 | `risk_prob` — 상방 알림 옆에서 "위험도" 보정 표시(annotation) | `short_adverse_prob` — 하방 알림 옆에서 "반등 확률" 표시(annotation), veto는 옵션 |
| 세션 창 | `09:10:00~15:00:00` 하드코딩 (`MIN_TIME`/`MAX_TIME` 모듈 상수) | `09:10:00~15:00:00` 기본값이지만 `ShortThresholds.min_time/max_time`으로 CLI 조절 가능 |
| 발동 시 표시 | `▲`(파란색, TTY에서만 색) | `🔻` |
| 검증된 엔트리 엣지 | RISK/WEAKENING이 대체로 정보성 (평균회귀 알림) | DOWN 타입 모두 숏 진입 엣지 없음(과거 스윕 결과, `short_entry_gate_report.md`) — 트렌드 정보 피드일 뿐 |

---

## Q5. 파란 화살표(▲)가 떴는데 "상방 가능성" 같은 문구가 안 보이는 이유

LONG이 발동해도 이 러너는 LONG 쪽의 한글 문구(`trend_message()` 형태)를 **애초에 출력하지 않습니다.**
`_process_minute()`(`scripts/realtime_short_trend_alert.py:278-283`)에서 한 줄의 본문(`body`)은 **SHORT
발동 여부로만** 결정됩니다:

```python
if down_fired:
    body, note = alert["message"], ""     # SHORT가 발동했을 때만 "메시지" 사용
else:
    body, note = _status_body(srow)        # 그 외엔 항상 SHORT 상태줄
```

LONG이 이번 봉에 발동해도(`▲` 마커는 정상 표시) SHORT가 안 떴으면 `body`는 여전히 SHORT 압축
상태줄이고, LONG 정보는 `_fmt_long(ctx)`가 만드는 **압축 코드** `| long=UP_ALERT(B) risk=0.32[LOW]`
로만 붙습니다. "상방 가능성"이라는 한글 문구 + `reasons=...`는 `trend_message()`
(`scripts/realtime_trend_alert.py:78-95`)에만 있고 이 스크립트엔 연결되어 있지 않습니다.

`long=` 뒤 코드 ↔ 한글 문구 매핑:

| 코드 | 한글 문구 |
|---|---|
| `UP_ALERT` | 상방 가능성 |
| `UP_CONTINUATION` | 상방 유지 가능성 |
| `RISK_ALERT` | 하방 위험 |
| `WEAKENING_ALERT` | 추세 약화 |
| `VOLATILITY_ALERT` | 변동성 확대 |

즉 `▲` + `long=UP_ALERT(B)`가 보였다면 그게 "상방 가능성 B등급" 발동이 맞습니다 — 문구가 빠진 게
아니라 코드로만 축약돼서 나오는 것입니다. 전체 문구+근거까지 보려면 `scripts/realtime_trend_alert.py`
또는 `scripts/realtime_trend_signal.py`를 같은 날짜/모드로 별도 실행해서 `🔔 ALERT ... 상방 가능성
[UP_ALERT B] ...` 줄을 확인해야 합니다.

---

## Q6. SHORT를 우선으로 잡고 LONG은 그냥 곁다리로 넣어둔 건가? (둘이 대등한 줄 알았는데)

**맞습니다 — 대등하지 않고, SHORT가 메인 / LONG은 참고용 컨텍스트입니다.** 근거:

1. **본문(body) 결정권이 SHORT에만 있음**: `_process_minute()`의 `body`는 오직 `down_fired`(SHORT 발동
   여부)로만 갈립니다. LONG이 발동해도 SHORT가 안 뜨면 본문은 여전히 SHORT 상태줄입니다(Q5).
2. **LONG은 코드로만 축약 표시**: LONG 발동은 `▲` + `| long=UP_ALERT(B) risk=0.32[LOW]`처럼 타입/등급/
   risk_prob만 붙고, SHORT처럼 한글 문구 + `reasons=...`가 담긴 완전한 메시지는 없습니다.
3. **저장(로깅)도 비대칭**: SHORT 발동만 `short_alert_log` DB 테이블 + `short_entry_ledger.csv`에
   남습니다. LONG 발동은 콘솔 출력뿐, 이 스크립트에서는 어디에도 저장되지 않습니다 — 모듈 docstring:
   *"Long fires are console-only (down fires own short_alert_log + ledger)"*.
4. **설계 의도로 명시되어 있음**: 파일 상단 docstring — *"The engine supplies the per-minute LONG context
   ... for display only"*. Step 14의 기존 `TrendAlertEngine`을 **수정 없이 재사용**해서 참고 컨텍스트로만
   곁들이는 것이 원래 의도이고, Step 15(SHORT)가 이 스크립트의 실제 주인공입니다.

→ (2026-07-04 확인) 사용자 확인: 구조 이해만 목적, 변경 불필요 — 현재 코드 유지.

---

## 부록 — 화면 한 줄의 구조 (`_process_minute`, `scripts/realtime_short_trend_alert.py:254-290`)

```
{🔻/▲/공백 마커} {시:분} {심볼}  close={종가}  {발동 메시지 또는 상태줄}{note}  | long={LONG타입}({등급}) risk={risk_prob}[{밴드}]
```
- 마커: `🔻`=이번 봉에서 SHORT 쪽이 발동, `▲`=이번 봉에서 LONG 엔진이 발동. 동시에 뜨면 `🔻▲`.
- 본문: SHORT가 발동했으면 `short_message()` 결과, 아니면 `_status_body()`의 상태줄(+ 필요 시 Q1의 fire-gate pass note).
- `| long=...`: LONG 엔진의 "현재" 분류/리스크(발동 여부와 무관하게 항상 표시); LONG이 이번 봉에 발동했다는 사실은 `▲` 마커로만 알 수 있음.

---

## Q14. `realtime_short_trend_alert.py` CLI 옵션 전체 (`parse_args`, line 399-457)

`--all-symbols`는 이 파일엔 없음 — 그건 32bit 수집기(`realtime_alert_collector.py`) 전용 옵션(파트2 Q7).

### A. 실행 기본
| 옵션 | 기본값 | 기능 |
|---|---|---|
| `--code` | `config.yaml`의 `realtime.default_symbol`(000660) | 대상 6자리 종목코드 |
| `--mode` | `replay` | `replay`(과거 재생) / `live`(실시간 폴링) |
| `--replay-date` | 없으면 그 종목의 `bars_1m` 최신 거래일 자동 선택 | replay할 날짜(`YYYY-MM-DD`) |
| `--today` | 없으면 `realtime_minute_bars`의 최신 날짜 자동 선택 | live 모드에서 쓸 거래일 |
| `--poll-sec` | `5.0` | live 모드에서 DB를 몇 초마다 다시 읽을지 |
| `--alerts-only` | 꺼짐 | 매 분 상태줄 대신 **발동(🔻/▲)한 줄만** 출력 |

### B. LONG 컨텍스트 모델(표시 전용, Q2·파트3)
| 옵션 | 기본값 | 기능 |
|---|---|---|
| `--risk-model-version` | `v1` | LONG `risk_prob` 계산용 모델 버전(v2.x는 `--cross-symbol` 필요; 파트3에서 전부 NO-GO 확인) |
| `--model-path` | 버전별 기본 경로 | 그 모델 파일 경로 수동 지정 |

### C. SHORT 모델(v1_short, 파트3 Q10)
| 옵션 | 기본값 | 기능 |
|---|---|---|
| `--short-model-path` | `models/short_risk_model_lgbm.txt` | `short_adverse_prob` 계산용 모델 파일 경로 |
| `--no-short-model` | 꺼짐 | 모델 아예 안 씀 — `short_adverse_prob`=NaN 전행, veto도 강제로 꺼짐(rule-only) |

### D. SHORT 발동 임계값 — `ShortThresholds` 오버라이드 (Q1·Q3)
| 옵션 | 기본값 | 기능 |
|---|---|---|
| `--thresholds-name` | 자동(기본이면 baseline명, 뭔가 바꾸면 `cli_override`) | 발동 알림에 기록되는 태그 |
| `--down-threshold` | `4.0` | `DOWN_ALERT`/`DOWN_CONTINUATION` 발동 최소 `down_score` |
| `--up-risk-limit` | `1.0` | `up_score` 상한(`"none"`으로 비활성) |
| `--score-margin` | `3.0` | 발동 조건 `direction_score <= -margin` |
| `--bottom-rank-limit` | `5` | 하위 랭크 버킷 크기(`"none"`으로 비활성; `--cross-symbol` 없으면 자동 비활성 — Q3) |
| `--require-below-ma5` | 꺼짐 | 추가 확인: `dist_ma5_intraday < 0` |
| `--require-rel-weakness` | 꺼짐 | 추가 확인: `relative_ret_20m_to_universe < 0` |
| `--short-adverse-prob-max` | `None`(꺼짐) | **모델 veto**(Q1 보충) — 값을 주면 DOWN 타입만 그 확률 이상일 때 발동 억제 |
| `--daily-alert-limit` | `None`(꺼짐) | 하루 발동 총량 상한(끄면 쿨다운만이 유일한 억제 — Q1) |
| `--cooldown-min` | `15` | **SHORT 전용** 타입별 쿨다운 분(LONG 엔진 쿨다운은 별도 `rta.DEFAULT_COOLDOWN_MIN=15` 고정 — Q4) |
| `--min-time`/`--max-time` | `09:10:00`/`15:00:00` | 발동 허용 세션 시간대 |

### E. 공용
| 옵션 | 기본값 | 기능 |
|---|---|---|
| `--cross-symbol` | 꺼짐 | LONG 컨텍스트+SHORT 게이트 둘 다에 진짜 종목군 cross-symbol 피처 사용(파트5 Q12 보충). 켜야 `--bottom-rank-limit`도 의미 생김 |

`--up-risk-limit`/`--bottom-rank-limit`/`--short-adverse-prob-max`/`--daily-alert-limit`은 문자열
`"none"`/`"off"`로 조건 자체를 끌 수 있음(`_float_or_none`/`_int_or_none` 헬퍼).

---

# 파트 2 — 1분봉 수집기 (`scripts/realtime_alert_collector.py`)

## Q7. 1분봉을 수집하는 파일은 뭐고, Kiwoom에 어떻게 요청해서 받은 뒤 SQL로 저장하는지

**파일**: `scripts/realtime_alert_collector.py` — 32비트 전용 Kiwoom OpenAPI+ 수집기. Kiwoom OCX가
32bit 전용이라, LightGBM/sklearn을 쓰는 64bit 엔진(`realtime_trend_alert.py --mode live`)과 프로세스를
분리하고 sqlite DB(`realtime_minute_bars`)를 다리로 씁니다. **`src/kiwoom_client.py`는 전혀 쓰지
않고**, 이 파일 안에 자체 Kiwoom OCX 래퍼(`class RealtimeCollector(QAxWidget)`,
`realtime_alert_collector.py:99`)를 직접 구현해뒀습니다.

### 흐름 (`main()`, `realtime_alert_collector.py:275-293`)

1. **로그인** (`login`, 145-154): `dynamicCall("CommConnect()")` → `OnEventConnect` 콜백까지 `QEventLoop`로 블로킹.
2. **오늘 백필(seed)** (`seed_today_snapshots`, 156-179): 심볼당 1회 `opt10080`(주식분봉차트조회) TR을
   `SetInputValue`+`CommRqData`로 요청 → `OnReceiveTrData`(210-225)에서 `GetRepeatCnt`/`GetCommData`로
   행을 뽑아 **오늘 날짜인지만**(`dt14[:8] != today`) 걸러서 하나씩 저장. **시간대 필터는 없음.**
3. **실시간 구독** (`subscribe`, 186-193): 모든 코드를 `;`로 합쳐 **`SetRealReg` 한 번**으로 등록.
   FID는 `"20;10;15;13"`(체결시간;현재가;거래량;누적거래량) — 누적거래량(13)은 등록만 하고 안 읽음.
4. **틱 처리** (`_on_receive_real_data`, 227-245): 틱마다 `GetCommRealData`로 현재가/체결시간/거래량을
   읽어 메모리 상의 `MinuteBar`(84-96) 하나를 갱신. **다음 틱의 "분"이 바뀌는 순간** 직전 봉을
   "완성"으로 보고 즉시 DB에 씀 — 타이머가 아니라 다음 틱 도착이 트리거.
5. **종목코드**: Kiwoom에는 그대로(`000660_AL` 등 통합시세 코드) 보내지만, **DB에는 항상 6자리 베이스
   코드만 저장**(`self.db_symbol = {c: normalize_symbol_code(c.split("_")[0]) for c in self.codes}`, 110).
6. **종료** (`stop`, 195-202): `DisconnectRealData` + 미완성 마지막 봉 강제 flush.

### SQL 저장

```sql
-- src/config.py:105-118 (REALTIME_MINUTE_BARS_DDL)
CREATE TABLE IF NOT EXISTS realtime_minute_bars (
    symbol TEXT NOT NULL, ts_kst TEXT NOT NULL,
    open INTEGER NOT NULL, high INTEGER NOT NULL, low INTEGER NOT NULL, close INTEGER NOT NULL,
    volume INTEGER NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (symbol, ts_kst)
);
```
`_write_bar`(`realtime_alert_collector.py:129-142`)가 매번 새 sqlite 커넥션을 열어 **DELETE 후
INSERT**(업서트) — `connect()`(`src/config.py:22-40`)는 WAL 모드 + `busy_timeout=30000`으로 열어서
32bit 라이터(이 스크립트)와 64bit 리더(엔진)가 같은 파일을 동시에 건드려도 안 막히게 튜닝돼 있음.
쓰기는 **완성된 봉 단위로만** 발생(틱마다 X, 타이머 X) — seed 백필 루프 / 틱 핸들러의 분-롤오버 /
`stop()`의 마지막 flush, 이 3곳뿐. `source` 컬럼은 항상 고정 문자열 `"realtime"`.

(실제 DB 대조 확인: `db/market_data.sqlite`의 `realtime_minute_bars`는 61,116행이고 전량
`source='realtime'` — 스키마·INSERT문 모두 코드와 정확히 일치.)

## Q8. raw, regular 등으로 나눠서 정리하나?

**아니요 — 이 수집기는 raw/regular를 나누지 않습니다.** 장전/정규장을 구분하는 코드가 전혀 없어서,
Kiwoom이 장전(통합시세 `_AL`은 08:00~부터 옴) 틱을 주면 그대로 1분봉이 되어 같은 테이블에 들어갑니다.
seed도 날짜만 거르지 시간은 안 거릅니다. `source` 컬럼이 있지만 이건 raw/regular 구분용이 아니라
**"어느 수집기가 썼는지"를 나타내는 고정 문자열**이고, PK가 `(symbol, ts_kst)`뿐이라 애초에 같은
시각에 raw 버전/regular 버전이 공존하는 구조 자체가 아닙니다.

**raw/regular 3분류는 완전히 다른, 별도의 배치(historical) 파이프라인에만 존재**합니다 —
`src/historical_collector.py` → `bars_1m`/`bars_3m`/`bars_5m` 테이블. 거기서는 PK가
`(symbol, bar_start, source)`라서 같은 시각에 3개 버전이 공존 가능:

| source 값 | 정의 (`src/config.py:157-166,191-194`) | 용도 |
|---|---|---|
| `al_raw` | 통합시세(_AL) 전체, 08:00~19:59, 필터 없음 | 원본 보관 |
| `al_regular` | 같은 통합시세를 정규장(09:00~15:30, `config.yaml`의 `session.train_start/end`)만 필터 | **실제 모델 학습/서빙에 쓰는 버전** (`load_data_config().model_source`) |
| `single_regular` | 단일거래소 정규장 파일 | 비교용 (모델엔 안 씀) |

세션 필터링은 `src/csv_importer.py`의 `_ingest`가 `session_window` 파라미터로 시각 범위를 체크해서
구현(`historical_collector.py`에서만 호출됨). 정리하면: **장전 데이터가 섞이는지 여부는 이 실시간
수집기 단계가 아니라 다운스트림 소비자**(`realtime_trend_alert.py` 등이 `ts_kst[11:19] >= "09:00:00"`으로
거르는 것) **에서 처리**되고, raw/regular라는 이름 자체는 historical 파이프라인 전용 용어입니다.

## Q9 (부가 발견) — 안 쓰이는 이중 실시간 경로

같은 일(Kiwoom 실시간 구독 → 1분봉 조립 → `realtime_minute_bars` 저장)을 하는 **또 다른 별도 구현**이
프로젝트에 존재합니다. 로그 파일까지 대조해서 구체적으로 확인한 내용:

### 구성 (역할이 3개 파일로 분리됨)

- **`src/realtime_receiver.py`의 `RealtimeReceiver`**: Kiwoom OCX 래퍼. `register(symbol=...)`가
  **종목 1개만** 받음(`realtime_alert_collector.py`는 여러 종목을 한 번의 `SetRealReg`로 묶어 등록하는
  것과 다름). FID **9개**(체결시간·현재가·누적거래량·누적거래대금·시가·고가·저가·**매도호가·매수호가**)를
  구독 — collector.py가 안 받는 **매도/매수호가(best_ask/best_bid)까지 받는다는 점은 오히려 더 풍부함**.
  `real_type`이 정확히 `"주식체결"`인 틱만 걸러내고, 거래량도 raw 값이 아니라 **누적거래량을
  직전값과 diff**해서 증분을 계산(더 방어적인 방식). 자체 저장은 안 하고 파싱한 틱 dict를 생성자에
  주입된 `on_tick` 콜백으로 넘기기만 하는 순수 어댑터.
- **`src/minute_builder.py`의 `MinuteBuilder`**: 그 `on_tick` 콜백의 실제 구현. 틱을 모아 봉을
  갱신하다가 분이 바뀌면 `flush_current()`로 **`INSERT OR IGNORE`**(이미 있으면 무시 — collector.py는
  DELETE 후 INSERT라 항상 최신값으로 덮어씀, 미묘하게 다른 시맨틱)로 `realtime_minute_bars`에
  `source="kiwoom_realtime"`(`config.py`의 `REALTIME_SOURCE` 상수) 저장.
- **`scripts/run_realtime_000660.py`**: 위 둘을 실제로 연결해서 돌리는 CLI. 로그인→`register(symbol)`→
  Ctrl+C/SIGTERM 시 마지막 미완성 봉까지 `flush_current(reason="shutdown")`로 강제 저장 후 요약 출력.
  **오늘자 백필(seed) 단계가 없음** — 시작하자마자 바로 실시간만 받아서, 장중에 켜면 그 전까지의
  봉은 영영 비게 됨(`realtime_alert_collector.py`의 `seed_today_snapshots()`가 없는 버전).

### 안 쓰인다는 증거 — 실제 실행 로그로 확인

`logs/realtime.log` 전체 내용, 딱 4줄:
```
2026-06-21 12:30:00 [INFO] login start
2026-06-21 12:30:12 [INFO] login success
2026-06-21 12:30:12 [INFO] SetRealReg success screen_no=2000 symbol=000660 fids=20;10;13;14;16;17;18;27;28
2026-06-21 12:30:49 [INFO] DisconnectRealData screen_no=2000
```
로그인·구독(`symbol=000660`)까지는 성공했지만, `MinuteBuilder.on_tick`이 틱마다 남기는 "minute bar
started/updated" 로그가 **한 줄도 없이** 37초 만에 접속 해제 — **틱을 한 번도 못 받은 연결 테스트**로
끝난 것으로 보임. DB 조회 결과(`realtime_minute_bars` 61,116행 전량 `source='realtime'`,
`'kiwoom_realtime'` 0건)와 정확히 들어맞음 — 이 경로로 저장된 봉이 지금까지 단 하나도 없음.

### 지금 구조상 이 경로로는 어차피 부족함

다시 쓰려고 해도 현재 요구사항을 못 채움: (1) **다종목 동시 수집이 안 됨** —
`register()`가 종목 1개만 받아서 파트 5(Q12)의 `--cross-symbol`(20종목 동시 실시간 필요)을 지원하려면
프로세스를 20개 띄워야 함, (2) **오늘자 백필이 없음** — 장중에 시작하면 그날 이전 분봉이 통째로 빔.
이 파일엔 이 프로젝트의 다른 모든 파일에 있는 `"""Step N: ..."""` 독스트링도 없어서, 활성 심볼
목록(20종목)이 생기기 전 **단일 종목(000660)만 다루던 초기 단계의 프로토타입**이었을 가능성이 높아
보이고, 이후 다종목·백필이 필요해지면서 `realtime_alert_collector.py`로 새로 만들어 대체된 것으로
추정됨(확정하려면 git 이력 확인 필요).

또한 `src/config.py`에는 `realtime_minute_bars`와 완전히 동일한 스키마를 가진 **더 오래된 레거시
테이블 `minute_bars`**(`MINUTE_BARS_DDL`, `src/minute_collector.py`/`src/features.py`가 사용)도
존재하는데, 실제 DB에서 **0행** — 이것도 죽은 경로로 보입니다.

---

# 파트 3 — LONG/SHORT 리스크 예측 모델 학습 (`src/risk_model.py`, `scripts/train_risk_model.py`)

## Q10. 리스크 모델은 어떤 인풋/아웃풋으로, 어떤 의미로 학습했는지

LONG(v1)과 SHORT(v1_short)는 **피처·전처리·하이퍼파라미터가 100% 동일**하고 레이블 부호만 반전된
완전 대칭 모델입니다.

### 학습 후보군(인풋의 모집단) — `directional_alerts` 테이블

전체 1분봉이 아니라 **Step 13_5 방향성 알림 타입으로 분류되고 BASELINE 발동 조건(fire-gate)까지
통과한 행**만 후보로 씁니다 (`scripts/online_alert_backtest.py:260`:
`base[alert_fire_gate(base, BASELINE)]`). 여기에 시간대 필터 + 데이터 품질 이상치 제외까지 거친
뒤 `directional_alerts` 테이블로 저장됩니다. `build_model_frame`(`src/risk_model.py:411-435`)이
여기에 `features_merged`(1m+3m/5m 기술 피처) / `features_relative`(cross-symbol 순위·상대수익률)
/ `bars_1m`(레이블 계산용 실제 OHLC) / `labels`(보조 타겟)를 조인합니다. 데이터는
`model_source="al_regular"`(파트 2 Q8의 정규장 필터 버전)로 한정.

### 실제로 몇 개 행이 학습에 쓰이는지 — DB 직접 조회로 확인 (2026-07-05 기준)

`db/market_data.sqlite`를 읽기 전용으로 직접 조회한 결과 (기간 2025-06-02~2026-06-22, 활성 20종목,
`source=al_regular`):

| 단계 | 행 수 | 원본 대비 |
|---|---|---|
| 원본 1분봉(`bars_1m`) | 1,947,910 | 100% |
| 워밍업 통과(`features_merged`) | 1,640,530 | 84.2% |
| rule_signal 스코어링 성공(`rule_signals`) | 1,637,121 | 84.0% |
| **5타입 분류 + 발동조건 통과 → `directional_alerts`** | **421,425** | **21.6%** |

이 421,425개(v1/v1_short 공용 후보군) 중 `SplitConfig` 기준 학습(2025-06-02~2026-02-28)
**274,751개**, 검증(2026-03-01~2026-06-19) **145,647개**, 나머지 1,027개는 둘 다 아님.

**타입별 분해** — 21.6%가 생각보다 커 보이는 이유는 `RISK_ALERT`(순위 조건 없이 `risk_score≥4 &
direction≤-3`만 있으면 발동, 상대적으로 널널함) 혼자 절반을 차지하기 때문:

| alert_type | 개수 | 비중 |
|---|---|---|
| RISK_ALERT | 215,172 | 51.1% |
| UP_ALERT | 114,906 | 27.3% |
| WEAKENING_ALERT | 51,638 | 12.3% |
| UP_CONTINUATION | 39,616 | 9.4% |
| VOLATILITY_ALERT | 93 | 0.02% |

진짜 "까다로운 진입형" 조건(`up_score≥4, direction≥3, 순위 top5, down_score≤2`를 다 통과해야 하는
`UP_ALERT`+`UP_CONTINUATION`만)만 따로 세면 **154,522개 = 원본 대비 7.9%**로 확 줄어듦 — 체감상
"까다롭다"는 인상은 이쪽이 더 정확함.

**종목별 분해** — 파트 5(Q12)에서 설명한 "20종목이 동등하게 기여" 주장을 수치로도 확인: 000660은
20개 중 3위(25,557개, 전체의 6.1%)일 뿐, 1위는 402340 SK스퀘어(34,280개, 8.1%)이고 최소는
207940 삼성바이오로직스(10,452개, 2.5%) — 000660이 특별 취급되지 않고 20종목 모두 비슷한
규모(1만~3.4만 개)로 기여함.

### 인풋 — 피처 49개 (`FEATURE_COLUMNS`, `src/risk_model.py:30-96`)

| 그룹 | 개수 | 내용 |
|---|---|---|
| 알림 점수 | 5 | `up_score, down_score, alert_score, direction_score, risk_score` |
| 상대(cross-symbol) | 8 | `active_symbol_count, rank_ret_5m_1m, rank_ret_20m_1m, rank_volume_z_20m_1m, universe_mean_ret_20m, relative_ret_20m_to_universe, relative_volume_z_to_universe, universe_breadth` — 단일종목 서빙 시 NEUTRAL, `--cross-symbol`일 때만 실제값 |
| 병합 기술 피처 | 34 | 1/3/5/10/20/60분 수익률, range/body/윗꼬리/아랫꼬리, 거래량 z-score(20m/60m·3m/5m봉 10봉/20봉 창), VWAP거리, 시가대비수익률, 당일고가/저가와의 거리, 5MA(일간/장중) 거리 등 |
| 파생 플래그 | 2 | `above_ma5_intraday`, `above_ma5_daily` |

전부 decision_time(=피처 봉의 bar_end) 시점에 이미 아는 값만 사용 — `FORBIDDEN_FEATURE_TOKENS`
(`target`/`future`/`label_`/`up_fail`/`down2`/`event` 등)로 라벨/미래 정보가 피처에 섞이는 걸
`available_feature_columns()`(`risk_model.py:232-242`)가 강제로 막습니다. LONG/SHORT 피처셋은
완전히 동일(`MODEL_VERSIONS`의 `v1_short`이 `v1`과 같은 `FEATURE_COLUMNS` 사용, `risk_model.py:190-193`).

### 아웃풋(타겟) — 부호만 반전된 대칭 레이블 (`add_future_risk_targets`, `risk_model.py:336-408`)

- `future_MAE` = 결정봉 **다음 봉부터 장마감까지 저가의 누적 최솟값** ÷ 결정봉 종가 − 1
- `future_MFE` = 같은 방식의 고가 누적 최댓값 버전 (SHORT용 미러)
- **`target_mae_risk`(LONG, v1)** = `future_MAE <= -1.5%` → *"지금 이 알림을 봤을 때, 오늘 남은
  장중에 종가 대비 -1.5% 이상 밀릴 확률"*
- **`target_up_risk`(SHORT, v1_short)** = `future_MFE >= +1.5%` → *"지금 이 알림을 봤을 때, 오늘
  남은 장중에 종가 대비 +1.5% 이상 반등할 확률"* — LONG 타겟의 정확한 부호 반전 미러
  (`UP_RISK_THRESHOLD=0.015` / `MAE_RISK_THRESHOLD=-0.015`)

즉 방향을 맞추는 모델이 아니라 **"이미 나온 방향성 알림에 대해 반대 방향으로 역행할 위험을 예측하는
보정 모델"**입니다. `target_up_fail`/`target_down2_60m`도 계산되지만 이건 평가용 보조 지표일 뿐
v1/v1_short의 실제 학습 타겟이 아닙니다.

### 학습 방식

- **분할**: 완전 시간순, 셔플 없음 — train `2025-06-02~2026-02-28`, validation
  `2026-03-01~2026-06-19/30`(`SplitConfig`, `risk_model.py:19-22`).
- **전처리**(`FeaturePreprocessor`, 444-483): train 중앙값으로 결측 대체 → 그 뒤 평균/표준편차로
  표준화. 통계량은 모델과 함께 저장돼 서빙 시 동일 적용.
- **백엔드**(`--model auto`): LightGBM → sklearn HistGradientBoosting → sklearn LogisticRegression
  → 순수 NumPy 로지스틱(32bit 폴백) 순.
- **LightGBM 하이퍼파라미터**(623-656): `n_estimators=800, learning_rate=0.03, num_leaves=31,
  min_child_samples=60, subsample=0.85(freq=1), colsample_bytree=0.85, reg_lambda=1.0,
  class_weight="balanced", random_state=42`.
- **클래스 불균형**: 모든 백엔드에서 balanced 가중치(소수 클래스인 리스크 이벤트 보정).
- **조기 종료**: 진짜 validation은 학습 중 안 보여줌 — train **뒤쪽 15%를 시간순으로 자른** 내부
  홀드아웃에서 AUC 기준 60라운드 patience로 early stopping. 최종 validation 지표는 리키지 없음.
- **버전 잠금**: `--target target_up_risk`는 반드시 `--risk-model-version v1_short`와 짝을 이뤄야
  함(`train_risk_model.py:74-84`) — LONG 기준 모델 파일이 SHORT 학습으로 덮어써지는 걸 방지.

### Q10 보충 — 학습 기술 디테일

**1) 백엔드 자동 선택** (`resolve_model_kind`, `risk_model.py:574-584`)
```python
if _has_module("lightgbm"): return "lightgbm"
if _has_module("sklearn"):  return "sklearn_hgb"
return "numpy"
```
`_has_module`은 `importlib.util.find_spec(name) is not None` — 실제 import 없이 설치 여부만 가볍게
체크. lightgbm 있으면 무조건 lightgbm, 없으면 sklearn(HGB), 그것도 없으면 numpy. 32bit 환경(Kiwoom
때문에 강제)엔 lightgbm/sklearn이 아예 없는 경우가 흔해서, 학습은 64bit+lightgbm으로 하고 32bit
스코어링 상황까지 대비한 폴백 체인.

**2) LightGBM 하이퍼파라미터의 역할** (`_fit_lightgbm`, `risk_model.py:623-656`)

| 파라미터 | 값 | 역할 |
|---|---|---|
| `n_estimators` | 800 | 트리 개수 상한(실제로는 early stopping이 더 일찍 끊음) |
| `learning_rate` | 0.03 | 낮게 잡아 800개까지 천천히·보수적으로 학습(과적합 방지) |
| `num_leaves` | 31 | 트리 하나당 리프 수 상한(leaf-wise 성장의 복잡도 제어) |
| `min_child_samples` | 60 | 리프 하나에 최소 60개 샘플 필요 — 수백만 행 규모의 과적합 방지 |
| `subsample`/`subsample_freq` | 0.85 / 1 | 매 트리마다 행의 85%만 샘플링(bagging, 분산 감소) |
| `colsample_bytree` | 0.85 | 트리마다 피처(49개)의 85%만 사용(트리 간 상관 감소) |
| `reg_lambda` | 1.0 | L2 정규화(리프 가중치 과대화 방지) |
| `class_weight` | "balanced" | 클래스 불균형 보정 |

**3) Early stopping — 진짜 validation을 절대 안 건드리는 내부 홀드아웃** (`_time_ordered_holdout`,
`risk_model.py:600-613`)
```python
order = np.argsort(frame["decision_time"].to_numpy(dtype=str), kind="stable")
cut = int(len(order) * (1.0 - fraction))   # fraction 기본 0.15
return order[:cut], order[cut:]            # 앞 85% fit용, 뒤 15% 조기종료 감시용
```
**train 기간 안에서만** 시간순 뒤쪽 15%를 잘라 조기종료 감시용으로 씀(진짜 validation은 이 함수에
안 들어감). `lgb.early_stopping(60, verbose=False)`로 AUC가 60라운드 연속 안 좋아지면 멈추고,
`best_iteration_used`에 실제 트리 수 기록(학습 스크립트가 "early-stopped at N trees" 출력). 이 15%
안에 양/음성 클래스가 둘 다 있어야만 조기종료가 켜짐, 아니면 800개 전부 학습.

흥미로운 디테일: `fit_binary_model(train, valid, ...)`의 `valid` 파라미터는 시그니처에만 있고
**본문에서 전혀 참조되지 않음**. 실제 validation 스코어링(`predict_with_model`,
`classification_summary` 등)은 `train_risk_model.py`의 `main()`에서 학습이 끝난 *뒤에* 별도 수행 —
validation이 학습 과정에 한 톨도 못 섞이게 함수 구조 자체가 강제하는 설계로 보임. (참고:
`sklearn_hgb` 폴백은 `validation_fraction=0.15`로 **자체 내부 랜덤 분할**을 쓰기 때문에, lightgbm이
없어서 이쪽으로 폴백되면 조기종료 방식이 시간순이 아니게 되어 미묘하게 덜 엄격해짐.)

**4) `FeaturePreprocessor` — 전처리 순서가 중요함** (`risk_model.py:444-466`)
```python
# fit
median = x.median(axis=0).fillna(0.0)      # ① 컬럼별 중앙값(전체 NaN 컬럼 방어로 0)
filled = x.fillna(median)                   # ② 중앙값으로 결측 채움
mean = filled.mean(axis=0)                  # ③ 채운 뒤 데이터로 평균 계산
scale = filled.std(axis=0).replace(0,1.0)   # ④ 표준편차(상수 컬럼은 1로 방지)
# transform — 매번 새로 안 구하고 저장해둔 통계량 재사용
x = frame[cols].fillna(self.median_)
return (x - self.mean_) / self.scale_
```
핵심은 ③ — 평균을 "NaN 무시"가 아니라 **중앙값으로 채운 뒤의 데이터**로 계산. `transform`은 서빙
때마다 통계를 새로 안 구하고 학습 시점에 저장해둔 `median_`/`mean_`/`scale_`을 그대로 재사용(재현성).

**5) 클래스 불균형 보정 — 정확한 공식** (`_balanced_sample_weight`, `risk_model.py:587-597`)
```python
w[y==1] = n / (2.0 * pos)
w[y==0] = n / (2.0 * neg)
```
sklearn `class_weight="balanced"`와 동일 공식(`n_samples/(n_classes×해당클래스개수)`). 양성 비율
10%면 양성 가중치 5배, 음성 약 0.56배 — "전부 안전"이라고만 찍어도 손실이 낮아지는 걸 막음.
lightgbm은 `class_weight="balanced"`로 내장 처리, sklearn HGB는 `class_weight` 파라미터가 없어서
이 함수로 `sample_weight`를 직접 계산해 넘김.

**6) `NumpyLogisticRiskModel` — sklearn/lightgbm이 아예 없을 때의 수제 폴백**
(`risk_model.py:486-562`) 의존성 없이 numpy만으로 짠 배치 경사하강법 로지스틱 회귀:
```python
for _ in range(self.max_iter):   # 기본 120회
    pred = sigmoid(x @ w + b)
    err = (pred - y) * sample_weight           # balanced 가중치 적용
    grad_w = (x.T @ err) / denom + self.l2 * w   # L2(기본 1e-3) 포함
    w -= self.learning_rate * grad_w             # 기본 0.08
```
매 반복 전체 데이터로 한 번에 그래디언트 계산(미니배치 아님) + L2 + balanced 가중치까지 직접 구현.
저장도 sqlite/pickle 아닌 순수 JSON(`save()`가 계수·절편·전처리기 통계량을 통째로 직렬화)이라
32bit 환경에서도 안전하게 주고받을 수 있게 설계됨.

**7) 모델 저장/로드 — 백엔드별 형식이 다름, `load_model()`이 자동 판별** (`save_model`/`load_model`,
`risk_model.py:868-946`)

| 백엔드 | 저장 형식 | 판별 방법 |
|---|---|---|
| lightgbm | `.txt`(부스터 포맷) + `.meta.json` | `.meta.json`+원본파일 둘 다 있으면 |
| sklearn_hgb/logistic | `.pkl`(pickle) + `.meta.json` | `.meta.json`+`.pkl` 둘 다 있으면 |
| numpy | 순수 JSON 파일 하나 | 나머지 경우 |

`load_model(path)` 하나로 세 백엔드를 전부 투명하게 복원 — 서빙 스크립트는 실제 학습에 뭐가
쓰였는지 신경 안 쓰고 그냥 불러오기만 하면 됨.

### 서빙에서의 의미

| | LONG (v1) | SHORT (v1_short) |
|---|---|---|
| 타겟 | `target_mae_risk`(−1.5%↓) | `target_up_risk`(+1.5%↑) |
| 출력 확률명 | `risk_prob` | `short_adverse_prob` |
| 밴드 | HIGH≥0.60 / MED≥0.45 / LOW (동일) | HIGH≥0.60 / MED≥0.45 / LOW |
| 쓰이는 곳 | UP_ALERT/UP_CONTINUATION에 "상방이지만 고위험" | DOWN_ALERT/DOWN_CONTINUATION에 "하방이지만 반등주의" |
| 모델 파일 | `models/risk_model_lgbm.txt` | `models/short_risk_model_lgbm.txt` |

기록된 검증 성능([[step14-risk-model-and-realtime]] 메모 기준): LONG(v1) 검증 AUC≈0.66이지만 월별로
3월 0.68→6월 0.59로 **레짐 디케이**(하락 이벤트 기저율이 train 대비 거의 2배로 증가). SHORT(v1_short)는
AUC≈0.684로 더 높고 월별 0.68/0.66/0.66/0.66으로 **디케이 없이 안정적** — 동일한 방식으로 만든
대칭 모델인데 방향에 따라 안정성이 다르게 나타난 사례.

---

# 파트 4 — 피처 계산 전체 파이프라인과 알림 생성 (`src/feature_lib.py` 외)

## Q11. 현재 어떤 feature들이 있고 어떻게 계산하는지, 그걸로 alert를 어떻게 만드는지

전체 흐름: `bars_1m` → (`feature_lib.build_minute_features`) → `features_1m` / (`resampler.resample_ohlcv`
→ `feature_lib.build_bar_features`) → `features_3m`/`features_5m` → (`feature_merge.merge_asof_features`)
→ `features_merged` → (`relative_features.build_relative_features`) → `features_relative` →
(`rule_signal.build_rule_signals`) → `up_score`/`down_score` → (`directional_alert.py` LONG /
`short_entry.py` SHORT) → 최종 알림.

### 1) 원시(1m/3m/5m) 피처 — `src/feature_lib.py`

`features_1m.py`/`features_3m.py`/`features_5m.py`는 이 파일의 `build_minute_features`/
`build_bar_features`를 그대로 호출하는 ~15줄짜리 얇은 래퍼일 뿐, 자체 계산 로직이 없음(전량
`scripts/build_features.py`가 능동 호출, `tests/test_features.py`로 커버). 모든 계산은
`(symbol, 거래일)`로 그룹핑 후 **과거만 참조**(`shift`/`cummax`/`cummin`/`rolling` 전부 백워드).

```python
# src/feature_lib.py:93-111 (1분봉)
for w in (1, 3, 5, 10, 20, 60):
    out[f"ret_{w}m_1m"] = df["close"] / close_g.shift(w) - 1
out["range_1m"] = df["high"] / df["low"] - 1
out["body_1m"] = df["close"] / df["open"] - 1
out["upper_shadow_1m"] = df["high"] / df[["open", "close"]].max(axis=1) - 1
out["lower_shadow_1m"] = df[["open", "close"]].min(axis=1) / df["low"] - 1
out["volume_z_20m_1m"] = _volume_z(df, group, 20)   # (volume - rollmean)/rollstd
out["volume_z_60m_1m"] = _volume_z(df, group, 60)
typical = (df["high"] + df["low"] + df["close"]) / 3
vwap = (typical*df["volume"]).cumsum() / df["volume"].cumsum()   # 당일 누적, 종목·일자별
out["vwap_distance_1m"] = df["close"] / vwap - 1
out["return_from_open_1m"] = df["close"] / group["open"].transform("first") - 1
out["distance_from_day_high_1m"] = df["close"] / group["high"].cummax() - 1
out["distance_from_day_low_1m"] = df["close"] / group["low"].cummin() - 1
```

**MA5 (일간/장중, 5거래일 이동평균) — 크로스데이 피처, leakage-safe** (`feature_lib.py:113-138`):
- `ma5_intraday` = **같은 시각(분)**의 종가를 `shift(1)` 후 **직전 5거래일**로 rolling 평균 (당일 값은
  절대 자기 자신의 평균에 안 들어감)
- `ma5_daily` = 일자별 마지막 종가를 `shift(1)` 후 **직전 5거래일** rolling 평균, 당일 내내 상수로 broadcast
- `dist_ma5_intraday/daily` = `close / ma5_* - 1`, `ma5_*_n` = 실제 반영된 과거일수(워밍업 구간 체크용)

3m/5m 피처(`build_bar_features`, `feature_lib.py:144-175`)는 같은 패턴을 봉 단위(`ret_{1,2,3}bar_{suffix}`,
`range/body_1bar_{suffix}`, `volume_z_{10,20}bar_{suffix}`)로 반복. `decision_time = bar_end`(그 봉이
끝난 시각)이 전체 인과성의 기준.

3m/5m 원본 봉 자체는 `src/resampler.py:resample_ohlcv`가 만듦: 09:00 세션 오픈에 정렬해서
`(_minutes // interval) * interval`로 버킷 스냅 → `open=첫봉시가, high=최고, low=최저, close=마지막봉종가,
volume=합` → **1분봉이 정확히 interval개 다 있는 버킷만** 채택(`n == interval_minutes`), 마감의
단독 15:30봉처럼 불완전한 버킷은 버림. 날짜별로 앵커를 다시 잡아서 봉이 이틀에 걸치지 않음.

### 2) 타임프레임 병합 — `src/feature_merge.py`

```python
# src/feature_merge.py:45-49
return pd.merge_asof(
    base, right, on="_dt", by=["symbol", "_date"],
    direction="backward",       # 그 시점 이전에 이미 닫힌 마지막 3m/5m봉
    allow_exact_matches=True,   # bar_end == decision_time 도 허용
)
```
1분봉의 `decision_time`마다 `bar_end_3m <= decision_time`인 가장 최근 3m/5m 피처를 붙임 —
`by=["symbol","_date"]`라 종목이 섞이거나 전날 마지막 봉을 빌려오는 게 원천 차단됨. 아직 3m/5m봉이
안 나온 워밍업 구간은 자연스럽게 NaN(`merge_asof`가 우측에 해당 키 없으면 NaN).

### 3) 종목군 상대지표 — `src/relative_features.py` (Step 9)

같은 `decision_time`을 공유하는 활성 종목들끼리만 비교(`groupby("decision_time")`, 다른 시각과 절대
안 섞임):

```python
# src/relative_features.py:84-95
out["active_symbol_count"] = g["symbol"].transform("size")
out["rank_ret_5m_1m"]  = g["ret_5m_1m"].rank(ascending=False, method="min")   # 1=최고, 동점은 낮은순위 공유
out["rank_ret_20m_1m"] = g["ret_20m_1m"].rank(ascending=False, method="min")
out["universe_mean_ret_20m"] = g["ret_20m_1m"].transform("mean")
out["relative_ret_20m_to_universe"] = df["ret_20m_1m"] - out["universe_mean_ret_20m"]   # 평균 대비
out["relative_volume_z_to_universe"] = df["volume_z_20m_1m"] - g["volume_z_20m_1m"].transform("mean")
out["universe_breadth"] = (df["ret_20m_1m"] > 0).astype(float).groupby(df["decision_time"]).transform("mean")
```
v2.0(확산도·다중시간대 시장폭·1위와의 거리)·v2.1(리더 종목 후행지표)도 같은 파일에서 계산되지만
실제 서빙되는 v1/v1_short 모델(파트 3)엔 안 들어감 — 코드는 살아있지만 채택 안 된 실험 컬럼.

### 4) Rule Score — `src/rule_signal.py` (up_score / down_score, 알림의 진짜 기초)

8개 독립 규칙이 각각 가산(상호 배타 아님, 전부 더해짐):

| 규칙 | 조건 → 가점 |
|---|---|
| 20분 수익률 | `ret_20m≥0.8%`→up+2 / `>0.3%`→up+1 · `≤-0.8%`→down+2 / `<-0.3%`→down+1 |
| 5분 수익률 부호 | `ret_5m>0`→up+0.75 · `<0`→down+0.75 |
| 거래량 z (양쪽 다 가산) | `volz≥2.0`→up+1&down+1 · `≥1.0`→up+0.5&down+0.5 |
| 상대수익률 | `rel_ret20>0`→up+1 · `<0`→down+1 |
| 20분 순위 버킷 | 상위 1/3→up+1 · 하위 1/3→down+1 (`ceil(active_count/3)`, 최소 1) |
| 5분 순위 버킷 | 상위 1/3→up+0.5 · 하위 1/3→down+0.5 |
| 상대거래량 (양쪽 다 가산) | 순위상위 or `rel_vol>0`→up+0.25&down+0.25 |
| 시장폭 | `breadth>0.5`→up+0.5 · `<0.5`→down+0.5 |

`is_up_candidate=up_score≥3.0`, `is_down_candidate=down_score≥3.0`(둘 다 파라미터, 기본 3.0);
둘 다 넘으면 점수가 큰 쪽이 `primary_side`, 동점이면 `MIXED`. **주의**: `src/rule_signal_ma.py`라는
별도 모듈도 같은 이름(`up_score`/`down_score`)을 이동평균 기반 다른 공식으로 만듦 — LONG/SHORT
알림 파이프라인은 확실히 `rule_signal.py`(규칙기반) 쪽만 쓰고, `rule_signal_ma.py`는
`realtime_trend_signal.py`의 `--tick-cross`(5분 MA 교차, 📈/📉) 전용 — 이름은 같지만 별개 용도.

### 5~6) 알림 생성 — LONG(`directional_alert.py`) / SHORT(`short_entry.py`)

`up_score`/`down_score` + 선택적 미세구조 피처(`upper/lower_shadow_1m`, `vwap_distance_1m`,
`distance_from_day_high/low_1m`, `dist_ma5_intraday`, `range_1m`, `return_from_open_1m`)를 받아:

- **LONG**: `alert_score = up_score + (MA5위+1)`, `risk_score = down_score + 윗꼬리(1.5)+VWAP상단1%이탈(1.0)+
  고점소진(1.0)+거래량스파이크(0.5)`, `direction_score = up_score-down_score` → 우선순위
  **RISK_ALERT > WEAKENING_ALERT > UP_CONTINUATION > UP_ALERT > VOLATILITY_ALERT**로 5타입 분류,
  A/B/C/RISK 등급.
- **SHORT**: 완전 부호 반전 미러 — `short_score = down_score+(MA5아래+1)`, `upside_risk_score =
  up_score + 아랫꼬리(1.5)+VWAP하단1%이탈(1.0)+저점capitulation(1.0)+스파이크(0.5)` → 우선순위
  **SHORT_RISK_ALERT > DOWN_WEAKENING_ALERT > DOWN_CONTINUATION > DOWN_ALERT > VOLATILITY_ALERT**,
  A/B/C/SHORT_RISK 등급.

정확한 타입별 발동 임계값·쿨다운·메시지 포맷은 [파트 1](#파트-1--shortlong-트렌드-알림-러너-realtime_short_trend_alertpy) Q2/Q3에 이미 정리됨.

### 검증(cross-check) 메모
`directional_alert.py`/`short_entry.py`의 REQUIRED/OPTIONAL 컬럼과 `risk_model.py`의
`MERGED_FEATURE_COLUMNS`(34)/`RELATIVE_FEATURE_COLUMNS`(8), 총 44개 고유 컬럼명 전부를 실제
소스에서 grep으로 재확인 — **전부 정확한 생성 지점이 존재**하고 이름 불일치·생성 누락 없음(단,
위에 적은 `rule_signal_ma.py`의 `up_score`/`down_score` 이름 충돌은 참고용으로 별도 표시).

---

# 파트 5 — 활성 20종목 데이터의 활용 (`config/symbols_test.yaml`)

## Q12. 상위 20종목 데이터를 000660 추세판단/모델학습에 어떻게 활용하는지

### "상위 20종목"의 정체 — 시가총액 계산이 아니라 고정 핀박힌 워치리스트

`config/symbols_test.yaml`에 20개 종목이 전부 `active: true`로 고정돼 있음 (005930 삼성전자,
000660 SK하이닉스, 402340 SK스퀘어, 005935 삼성전자우, 009150 삼성전기, 005380 현대차, 032830
삼성생명, 373220 LG에너지솔루션, 028260 삼성물산, 329180 HD현대중공업, 207940 삼성바이오로직스,
034020 두산에너빌리티, 000270 기아, 012450 한화에어로스페이스, 105560 KB금융, 012330 현대모비스,
034730 SK, 055550 신한지주, 006400 삼성SDI, 042660 한화오션). 파일 헤더 주석이 명시: *"Step 1:
fixed symbol list (**no market-cap calculation**; codes are pinned here)"* — 매일 재계산되는
"진짜 시가총액 상위 20"이 아니라 **수동 고정 대형주 워치리스트**([[active-symbols-expanding]]
메모 갱신: 이전 "15→20 목표"였던 게 지금은 20개 전부 활성화 완료). `load_active_symbols()`
(`src/settings.py:252-254`)가 이 파일을 읽어 전 파이프라인의 "활성 종목 유니버스"로 씀.

### 1) 실시간 추세 판단 — `--cross-symbol` 옵션, 기본은 꺼져 있음

1. 32비트 수집기를 `--all-symbols`로 돌려야 20종목 전부의 실시간 1분봉이 동시에 쌓임.
2. 매 분 `TrendAlertEngine`이 20종목 전체의 그 시각 데이터로 순위/상대수익률/시장폭(8개 cross-symbol
   피처)을 계산해서 **타깃 종목(기본 000660) 행에만** 붙임(`build_serving_features_cross`).
3. 이 값들이 `rule_signal.py`의 규칙(순위 상/하위 버킷, 상대수익률 부호, 시장폭)에 들어가 000660의
   `up_score`/`down_score` → 알림 타입/등급에 직접 반영됨 — "다른 19종목 대비 얼마나 튀는지"가 반영.
4. **다른 19종목 자체의 알림은 안 뜸** — 엔진 인스턴스 하나는 지정한 타깃 하나만 스코어링.
5. `--cross-symbol` 없이 기본 실행하면 이 8개 값은 전부 중립값(순위=중간, 상대값=0, 시장폭=0.5)으로
   채워져 **다른 종목 정보가 전혀 안 들어감** — 이 활용 방식 자체가 옵션.

### 2) 모델 학습 — "000660 보조 데이터"가 아니라 "20종목 공동 기여 + 단일 공용 모델"

짐작과 다를 수 있는 부분: 학습은 000660 전용이 아니라 **20종목을 전부 모아 하나의 공용 모델**
(`risk_model_lgbm.txt`/`short_risk_model_lgbm.txt`, 파트 3)을 만듭니다.

```python
# scripts/online_alert_backtest.py:216-238
symbols = load_active_symbols()          # 20개 전부
...
for f in (rs, mg, lab, bars):
    f.drop(f.index[~f["symbol"].isin(symbols)], inplace=True)   # 20종목 전부 남김, 000660만이 아님
```
- `directional_alerts`(학습 후보 카탈로그)는 **20종목 각각의** 과거 알림 후보를 담음.
- `scripts/train_risk_model.py`도 이 20종목 필터를 그대로 이어받아 **하나로 합쳐서** 학습 — 종목별
  모델이 아니라 파일 하나(`resolve_model_path(version)`).
- 각 행의 8개 cross-symbol 피처는 `scripts/build_relative_features.py`(`symbols=load_active_symbols()`,
  `build_relative_features.py:48-49`)가 배치로 **진짜(neutral 아님) cross-section**을 계산해 넣음
  — 배치라 20종목 데이터가 이미 다 있어서 실시간과 달리 중립값이 필요 없음.

**정리**: "19종목이 000660 전용 모델을 도와준다"가 아니라, **"20종목이 동등하게 학습 행을 제공해서
하나의 범용 리스크 모델을 만들고, 서빙 시점에 어느 종목을 타깃으로 지정하느냐(기본 000660,
`config.yaml`의 `realtime.default_symbol`)에 따라 그 모델을 적용한다"**가 정확한 그림. 000660은
그 20개 중 하나일 뿐, 특별 취급되지 않음.

(참고: 005930/402340을 "리더"로 삼아 후행지표를 붙이는 v2.1a/v2.1b 실험도 `config.yaml`의
`relative.leaders`에 설정은 있지만, 파트 3에서 확인했듯 둘 다 NO-GO라 실제 서빙 모델엔 안 들어감.)

### Q12 보충 — 기본 실행에서 "다른 종목 추세를 안 본다"는 게 정확히 무슨 뜻인가

`--cross-symbol` 없이 기본 실행하면 8개 cross-symbol 컬럼이 **다른 종목 데이터와 무관한 고정값**으로
채워집니다 (`realtime_features._fill_neutral_relative`):

```python
NEUTRAL_RELATIVE = {
    "universe_mean_ret_20m": 0.0, "relative_ret_20m_to_universe": 0.0,
    "relative_volume_z_to_universe": 0.0, "universe_breadth": 0.5,
}
mid_rank = ceil(active_count / 2)   # rank_ret_5m/20m, rank_volume_z 전부 "딱 중간 순위"로 고정
```
즉 000660이 실제로 20종목 중 1등이든 20등이든 **항상 "정확히 평균/정확히 중간 순위"로 가정**하고
계산합니다. 예외적으로 `active_symbol_count`만 진짜 값(20)이 들어가지만, 순위 자체가 가짜 중간값
으로 고정돼 있어 실질적 의미는 없음(예: Q3의 `bottom_rank_limit`은 이 상태에서 절대 안 걸림).

**이 8개 컬럼은 두 시스템 모두에 영향**: (1) `rule_signal.py`의 규칙(상대수익률 부호·순위 버킷·
시장폭)에 들어가 `up_score`/`down_score` → **알림 타입/등급 분류 자체**에 반영, (2) LightGBM 49개
피처 중 8개로도 사용(`risk_prob`/`short_adverse_prob`). 기본 실행에서는 **둘 다** 가짜 값을 받음.

**학습-서빙 불일치**: 파트 3에서 DB로 확인했듯 학습은 `scripts/build_relative_features.py`가
배치로 20종목 전체의 **진짜** cross-section을 계산해 넣었음. 서빙(기본 실행)은 가짜 중립값을
씀 — 모델이 학습 때 본 입력 분포와 실전 입력 분포가 8/49개 피처에서 다름. 이게 project 메모의
"v1.1"(서빙 파리티 수정, `--cross-symbol`로 서빙에도 진짜 cross-section 반영)이 고치려던 정확히
그 문제 — 다만 메모 기록상 켰을 때 실측 영향은 크지 않았음(risk_prob 평균 0.682→0.688, 361개 봉
중 239개에서 값 변화 — 방향이 뒤집힐 정도는 아니었음).

**켜는 법**: `--cross-symbol` 플래그 + 32비트 수집기를 `--all-symbols`로 돌려서 20종목 실시간
봉을 동시에 쌓아야 함 — 둘 다 갖춰야 진짜 데이터가 들어감.

---

# 파트 6 — 프로젝트 전체 구조 (`README.md` 기준)

## Q13. 현재 프로젝트 구조가 어떤지 전체적으로 설명

`README.md`가 이 프로젝트의 공식 스펙 문서이고, **Step 0~15**로 번호가 매겨진 순차 파이프라인
전체를 문서화해뒀습니다. 지금까지 파트 1~5에서 깊이 판 내용들이 이 파이프라인의 어느 지점인지
정확히 지도로 그릴 수 있습니다.

### 큰 그림 — 두 개의 축

1. **32bit(Kiwoom) ↔ 64bit(ML) 분리**: Kiwoom OpenAPI+가 32bit 전용이라, 데이터 수집은 32bit
   프로세스(`..\Python311-32`), 피처 계산·LightGBM·차트는 64bit 프로세스(`..\Python311-64`)로
   완전히 나뉘고, sqlite DB(`db/market_data.sqlite`)가 둘을 잇는 다리 역할을 합니다(WAL 모드로
   동시 접근 지원, 파트2 Q7에서 확인).
2. **배치 파이프라인(Step 0~13) ↔ 실시간 캡스톤(Step 14~15)**: 과거 데이터로 피처→규칙→모델을
   만드는 배치 라인과, 그걸 그대로 실시간에 재현하는 두 개의 라이브 시스템(LONG=Step14,
   SHORT=Step15)으로 나뉩니다. 실시간 캡스톤은 배치 파이프라인의 함수를 그대로 재사용하도록
   설계돼 있음(`src/realtime_features.py`가 `feature_lib`/`resampler`/`feature_merge`를 그대로
   호출 — 파트4 Q11에서 확인한 그대로).

### Step 0~15 파이프라인 전체 지도

| Step | 내용 | 주요 파일 | 산출물(테이블) | 이 문서에서 다룬 부분 |
|---|---|---|---|---|
| 0 | 프로젝트 기준값(세션시간·알림한도 등) | `config/config.yaml` | - | 파트2 Q8, 파트5 |
| 1 | 종목 리스트 고정(시총 계산 아님, 수동 pin) | `config/symbols_test.yaml`, `settings.py` | - | 파트5 Q12 |
| 2 | `bars_1m/3m/5m` 통합 테이블화, CSV 임포트 | `csv_importer.py` | `bars_1m`(PK에 `source` 포함) | 파트2 Q8 |
| 3 | 다종목 히스토리 수집(CSV 또는 Kiwoom TR 직접) | `historical_collector.py` | `bars_1m` | 파트2 |
| 4 | 소스 정책 확정(`al_regular`=모델 입력) + 이상치 리포트 | `bars_reader.py` | 검증 CSV | 파트2 Q8 |
| 5 | 1m→3m/5m 리샘플(완성된 버킷만) | `resampler.py` | `bars_3m/5m` | 파트4 Q11 |
| 6 | 데이터 품질 리포트(FAIL=구조적 오류/WARN=통계적 이상) | `quality.py` | `quality_report_*.csv` | - |
| 7 | 타임프레임별 피처(리키지 안전, backward-only) | `feature_lib.py`+`features_1m/3m/5m.py` | `features_1m/3m/5m` | 파트4 Q11 |
| 8 | 리키지 없는 시간축 병합(merge_asof, backward) | `feature_merge.py` | `features_merged` | 파트4 Q11 |
| 9 | 종목군 상대지표(같은 decision_time끼리만 비교) | `relative_features.py` | `features_relative` | 파트4·5 |
| 10 | 미래 라벨(정답) 생성 — 평가 전용, 피처엔 절대 안 섞임 | `labeling.py` | `labels` | 파트3(future_MAE 연장선) |
| 11 | 규칙기반 시그널 v0(up_score/down_score) | `rule_signal.py` | `rule_signals` | 파트4 Q11 |
| 12 | 일일 top1/top2 백테스트 + 진입시점/필터 변형 비교 | `daily_topk_backtest.py`, `entry_variants.py` | `backtest_*` | - |
| 13 | 규칙 임계값 스윕(그리드/퍼센타일) | `threshold_sweep.py` | `threshold_sweep_*` | - |
| **13_5** | **방향성 알림 시스템**(진입신호 아닌 "알림 피드"로 재정의) | `directional_alert.py` | `directional_alerts` | **파트1, 파트3 Q10, 파트4** |
| **14** | **리스크모델 1**(하락위험 예측) + 실시간 LONG 캡스톤 | `risk_model.py`, `realtime_trend_alert.py`, `realtime_alert_collector.py` | `risk_model_lgbm.txt`, `realtime_minute_bars` | **파트1~4 전체** |
| **15** | **하방 트렌드 알림**(SHORT, LONG의 완전 대칭 미러) | `short_entry.py`, `realtime_short_trend_alert.py` | `short_risk_model_lgbm.txt`, `short_alert_log` | **파트1, 파트3** |

부가 연구/보조 기능(번호 없이 Step 14~15 사이에 얹힌 것들): **포지션 청산 경고 모니터**
(`realtime_position_monitor.py`+`position_exit.py`, 이미 보유한 수동 포지션의 매도경고 — 주문 전송 안 함),
**아침신호→오후극값 검증 연구**(`morning_signal_validation.py`, "아침 첫 신호가 오후 고점/저점을
예측하나?" — 결론 FAIL), **바운스-롱 연구**(`bounce_long_sweep.py`, SHORT의 `short_adverse_prob`을
반대로 LONG 반등 신호로 못 쓰나 — 아직 실데이터 스윕 전, 파우즈 상태).

### 디렉터리 구조

```
quant/
├── src/          38개 모듈 — 순수 로직(피처/규칙/모델/알림 계산). DB·Kiwoom API 직접 안 건드리는 게 원칙
├── scripts/      50개 CLI 러너 — src/의 함수를 호출해서 실제로 배치를 돌리거나 실시간을 구동
├── tests/        27개 unittest 모듈 — Step별로 1개 이상, `python -m unittest tests.test_X ...`로 실행
├── config/       config.yaml(세션/데이터소스/실시간기본값/리더종목), symbols_test.yaml(활성종목)
├── data/
│   ├── raw/, processed/   각 단계 산출 CSV/parquet/리포트(.md 포함)
│   └── labels/
├── models/       학습된 모델 파일(risk_model_lgbm.txt, short_risk_model_lgbm.txt, .meta.json 등)
├── db/           market_data.sqlite (모든 테이블의 단일 저장소, ~4.1GB)
└── logs/         스크립트별 실행 로그(collector.log, realtime.log, backtest.log 등)
```

### "Legacy 단일종목 컴포넌트" — README가 명시적으로 분리해둔 구버전 그룹

README 맨 끝에 **"Legacy single-symbol components"** 섹션이 따로 있고, 정확히 파트2 Q9에서 발견한
그 파일들을 "구버전, `bars_1m` 파이프라인이 아님"이라고 **공식적으로 명시**하고 있습니다:
- `scripts/collect_000660_1m.py` + `src/features.py` → 레거시 `minute_bars`/`minute_features` 테이블
- `scripts/run_realtime_000660.py` + `src/realtime_receiver.py` + `src/minute_builder.py` → 레거시
  `realtime_minute_bars`(source=`kiwoom_realtime`) — **Q9에서 실행 로그로 직접 확인한 그 미사용 경로**
- `scripts/compare_realtime_vs_opt10080.py` + `src/realtime_compare.py` → 위 실시간 빌더가 만든 봉을
  opt10080 스냅샷과 대조 검증하던 짝꿍 스크립트(Q9에서는 못 찾았던 존재 이유가 이걸로 설명됨 — 실시간
  틱 기반 집계가 믿을 만한지 확인하는 **검증용 프로토타입**이었던 것으로 보임)

즉 Q9의 발견이 추측이 아니라 **README가 이미 공식적으로 "레거시"라고 선언해둔 그룹**이었다는 게
이번에 확인됐습니다.

### 테스트

`python -m unittest tests.test_settings tests.test_collector ... tests.test_short_entry`처럼
Step별로 매핑된 27개 테스트 모듈을 한 번에 돌리는 구조(README:512). 각 Step 도입 시 그 Step의
불변식(리키지 없음, 인과성, 게이트 로직 등)을 굳히는 회귀 테스트가 같이 추가되는 패턴입니다.
