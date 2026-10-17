# Step 13_5. 방향성 알림 시스템 v0로 재정의

## 할 일

기존 룰 기반 시그널을 자동매매 신호가 아니라, 방향성 알림으로 재정의합니다.

목표는 “바로 매수하면 수익이 나는 신호”를 찾는 것이 아니라, 아래 정보를 사용자에게 알려주는 것입니다.

```text
상방 가능성 알림
상방 유지 가능성 알림
하방 위험 알림
추세 약화 알림
변동성 확대 알림
```

즉, Step 13_5부터는 `rule_signals`를 매매 진입 신호가 아니라 **알림 후보와 등급**으로 봅니다.

---

## 기본 방향

기존 결과상 현재 룰은 자동매매 전략으로는 edge가 약합니다.

따라서 Step 13_5에서는 수익률 최적화보다 아래를 평가합니다.

```text
알림 후 30분 방향
알림 후 60분 방향
알림 후 120분 방향
알림 후 MFE
알림 후 MAE
알림 후 stop_first 위험
알림 등급별 성과 차이
```

UP5는 주 목표로 두지 않고 보조 지표로만 봅니다.

주 평가 대상은 아래로 둡니다.

```text
up2_60m
up2_120m
up3_120m
future_return_30m > 0
future_return_60m > 0
future_return_120m > 0
```

---

## 알림 등급

각 row마다 알림 등급을 부여합니다.

예:

```text
A등급:
상방 점수 높음
하방 위험 낮음
상대강도 우수
MFE 기대 높음

B등급:
상방 가능성은 있으나 위험도 존재

C등급:
방향성 약함 또는 변동성만 큼

RISK:
상방 점수와 무관하게 하방 위험이 큰 상태
```

알림 등급은 매수/매도 지시가 아니라, 사용자가 판단할 수 있는 참고 정보입니다.

---

## 기본 입력

입력은 기존 파이프라인 결과를 사용합니다.

```text
features_relative
rule_signals
labels
bars_1m source = al_regular
quality report
```

라벨은 평가에만 사용합니다.

알림 등급 생성에는 미래 라벨을 사용하지 않습니다.

---

## 기본 알림 조건

기존 baseline은 유지하되 자동매매가 아니라 알림 후보로 사용합니다.

```text
conservative rule score
entry_delay_3m
both 후보 제외
09:10 이전 제외
15:00 이후 신규 알림 제외
동일 종목 cooldown 적용
FAIL 품질 날짜 제외
```

단, 하루 top1/top2를 고정 매수 대상으로 보지 않습니다.

대신 실시간 알림처럼 시간 순서대로 평가합니다.

---

## Online 알림 백테스트

하루가 끝난 뒤 top1/top2를 고르는 방식과 별도로, 실시간 상황에 가까운 online sequential 백테스트를 만듭니다.

방식:

```text
09:10부터 15:00까지 시간 순서대로 진행
각 decision_time마다 활성 종목 후보 확인
조건을 만족하면 알림 발생
daily limit과 cooldown 적용
이후 30분/60분/120분 성과 평가
```

필수 조건:

```text
미래 후보를 보고 현재 알림을 고르지 않는다
같은 decision_time에 이미 완성된 피처만 사용한다
라벨은 알림 생성에 사용하지 않는다
```

---

## 지시사항

1. `directional_alert.py`를 만든다.

2. `rule_signals`를 입력으로 받아 알림 후보를 만든다.

3. 알림 후보마다 아래 값을 계산한다.

```text
alert_score
risk_score
direction_score
alert_grade
alert_type
reason_1
reason_2
reason_3
```

4. `alert_type`은 아래 중 하나로 둔다.

```text
UP_ALERT
UP_CONTINUATION
RISK_ALERT
WEAKENING_ALERT
VOLATILITY_ALERT
```

5. `alert_grade`는 아래 중 하나로 둔다.

```text
A
B
C
RISK
```

6. `online_alert_backtest.py`를 만든다.

7. 시간 순서대로 알림을 발생시키는 online sequential 백테스트를 구현한다.

8. daily top1/top2 방식은 랭킹 진단용으로만 유지한다.

9. online 방식에서는 threshold가 실제 알림 발생 시점과 빈도를 결정하도록 한다.

10. 아래 threshold를 비교한다.

```text
up_threshold
score_margin
down_risk_limit
current_rank_limit
daily_alert_limit
cooldown_minutes
entry_delay_minutes
```

11. 알림 후 성과를 계산한다.

```text
return_30m
return_60m
return_120m
return_eod
MFE
MAE
hit_up2_60m
hit_up2_120m
hit_up3_120m
stop_first
```

12. 알림 등급별 성과를 비교한다.

```text
A등급 성과
B등급 성과
C등급 성과
RISK 알림 이후 하락 위험
```

13. 결과를 테이블과 CSV로 저장한다.

```text
directional_alerts
online_alert_results
alert_grade_summary
```

---

## 평가 지표

기존 자동매매식 지표보다 방향성 알림 지표를 우선합니다.

```text
alert_count
trading_days_with_alert
avg_alerts_per_day
grade_distribution

return_30m_mean
return_60m_mean
return_120m_mean

direction_hit_30m
direction_hit_60m
direction_hit_120m

average_MFE
average_MAE
stop_first_rate

A_grade_vs_B_grade
B_grade_vs_C_grade
RISK_alert_down_rate
```

추가로 기존 지표도 보조로 유지합니다.

```text
hit_up2_rate
hit_up3_rate
hit_up5_rate
return_eod
symbol_concentration
```

---

## 완료 기준

아래 결과가 나와야 합니다.

```text
directional_alerts 테이블 생성
online_alert_results 테이블 생성
alert_grade_summary CSV 생성
알림 등급별 성과 리포트 생성
라벨 누수 없음
시간 순서 online 백테스트 구현
daily topK와 online threshold 결과 분리
```

좋은 알림 시스템의 기준은 아래입니다.

```text
A등급이 B등급보다 성과가 좋아야 함
B등급이 C등급보다 성과가 좋아야 함
RISK 알림 이후 하락/MAE 위험이 높아야 함
알림 후 30분/60분 방향성이 무작위보다 좋아야 함
stop_first가 높은 알림을 위험 등급으로 분리할 수 있어야 함
하루 평균 알림 수가 과도하지 않아야 함
```

예상 출력 예:

```text
기간: 2025-06-02 ~ 2026-06-19
활성 종목 수: 20
알림 수: 1,240
하루 평균 알림 수: 4.8

A등급:
return_60m +0.18%
direction_hit_60m 56%
MFE +1.10%
MAE -0.72%

B등급:
return_60m +0.04%
direction_hit_60m 51%
MFE +0.82%
MAE -0.88%

C등급:
return_60m -0.06%
direction_hit_60m 48%
MFE +0.61%
MAE -0.95%

RISK:
return_60m -0.22%
MAE -1.25%
```

Step 13_5의 목표는 수익형 자동매매 전략 확정이 아닙니다.

목표는 “어떤 알림이 상대적으로 더 의미 있는지”, “어떤 알림이 위험한지”, “A/B/C 등급이 실제로 구분되는지”를 검증하는 것입니다.
