# Step 13_6. 방향성 알림 평가 리포트 만들기

## 할 일

기존 룰 기반 시그널을 자동매매 신호가 아니라 **방향성 알림 후보**로 재정의하고, 알림 이후의 움직임을 평가합니다.

목표는 아래를 확인하는 것입니다.

```text
A등급 알림이 B/C등급보다 실제 방향성이 좋은가?
RISK 알림 이후 실제 하락 위험이나 MAE가 큰가?
UP_ALERT 이후 30분/60분/120분 방향성이 무작위보다 나은가?
알림 등급이 실제로 의미 있게 구분되는가?
```

이 단계에서는 매수/매도 수익률 최적화가 아니라, **알림 품질 평가**에 집중합니다.

---

## 입력 데이터

아래 테이블을 사용합니다.

```text
features_relative
rule_signals
labels
bars_1m source = al_regular
quality report
```

라벨은 평가에만 사용합니다.

알림 생성에는 미래 라벨을 절대 사용하지 않습니다.

---

## 알림 후보 정의

기존 `rule_signals`를 기반으로 알림 후보를 만듭니다.

기본 필터는 아래를 사용합니다.

```text
09:10 이전 제외
15:00 이후 신규 알림 제외
both 후보 제외
FAIL 품질 날짜 제외
동일 종목 cooldown 적용
active 종목만 사용
```

기존 `up_score`, `down_score`, `relative strength`, `VWAP`, `MA/VWAP 상태` 등을 사용해 알림 후보를 분류합니다.

---

## 알림 타입

각 알림 후보에 `alert_type`을 부여합니다.

```text
UP_ALERT
UP_CONTINUATION
RISK_ALERT
WEAKENING_ALERT
VOLATILITY_ALERT
```

의미는 아래와 같습니다.

```text
UP_ALERT:
상방 전환 또는 상방 압력이 강해지는 상태

UP_CONTINUATION:
이미 상방 흐름이 있고, 추세가 유지되는 상태

RISK_ALERT:
상방 점수가 있어도 하방 위험이나 변동성 위험이 큰 상태

WEAKENING_ALERT:
기존 상방 흐름이 약해지는 상태

VOLATILITY_ALERT:
방향성보다 변동성 확대가 주된 상태
```

---

## 알림 등급

각 알림 후보에 `alert_grade`를 부여합니다.

```text
A
B
C
RISK
```

예시 기준:

```text
A:
상방 점수 높음
하방 위험 낮음
상대강도 상위
VWAP/MA 기준 양호

B:
상방 가능성은 있으나 일부 위험 존재

C:
방향성 약함 또는 신호 품질 낮음

RISK:
상방 가능성과 무관하게 하방 위험 또는 변동성 위험이 큼
```

등급은 매수 지시가 아니라, 사용자가 장중 판단할 때 참고할 **상태 요약**입니다.

---

## 생성 컬럼

`directional_alerts` 테이블에는 최소 아래 컬럼을 둡니다.

```text
symbol
decision_time
alert_type
alert_grade
alert_score
direction_score
risk_score
up_score
down_score
reason_1
reason_2
reason_3
source
created_at
```

가능하면 아래 컬럼도 포함합니다.

```text
rank_ret_20m_1m
relative_ret_20m_to_universe
volume_z_20m_1m
vwap_distance_1m
universe_breadth
```

---

## 평가 방식

알림 이후 30분, 60분, 120분, 장마감까지의 움직임을 평가합니다.

계산 항목:

```text
future_return_30m
future_return_60m
future_return_120m
future_return_eod

direction_hit_30m
direction_hit_60m
direction_hit_120m

MFE
MAE

hit_up2_60m
hit_up2_120m
hit_up3_120m

down2_60m
UP_FAIL
stop_first
```

`direction_hit_*`는 상방 알림의 경우 미래 수익률이 0보다 크면 1로 계산합니다.

```text
direction_hit_60m = future_return_60m > 0
```

RISK 알림은 반대로 하방 위험이 실제로 커졌는지를 봅니다.

```text
RISK 평가:
future_return_60m < 0
MAE가 큼
down2_60m 발생
UP_FAIL 발생
```

---

## 지시사항

1. `directional_alert.py`를 만든다.

2. `rule_signals`를 입력으로 받아 알림 후보를 생성한다.

3. 알림 후보마다 `alert_type`, `alert_grade`, `alert_score`, `direction_score`, `risk_score`를 계산한다.

4. 알림 생성에는 `labels`를 사용하지 않는다.

5. `labels`는 알림 후 성과 평가에만 사용한다.

6. `directional_alerts` 테이블과 parquet을 저장한다.

7. `evaluate_directional_alerts.py`를 만든다.

8. 알림 이후 30분/60분/120분/EOD 성과를 계산한다.

9. 알림 타입별 성과를 집계한다.

10. 알림 등급별 성과를 집계한다.

11. 결과를 CSV로 저장한다.

```text
data/processed/directional_alerts.parquet
data/processed/alert_evaluation_summary.csv
data/processed/alert_grade_summary.csv
data/processed/alert_type_summary.csv
```

12. daily topK 방식과 online 방식은 구분해서 저장한다.

13. 이번 Step 13_6에서는 우선 알림 등급이 의미 있게 구분되는지만 확인한다.

---

## 리포트 예

알림 등급별 리포트:

```text
grade,n,return_30m,return_60m,return_120m,direction_hit_60m,MFE,MAE,down2_60m_rate,UP_FAIL_rate
A,320,+0.08%,+0.14%,+0.21%,55.6%,+0.92%,-0.61%,4.2%,8.1%
B,610,+0.02%,+0.04%,+0.08%,51.8%,+0.75%,-0.78%,6.7%,11.4%
C,980,-0.03%,-0.06%,-0.02%,48.9%,+0.60%,-0.85%,8.9%,13.2%
RISK,450,-0.10%,-0.18%,-0.25%,43.1%,+0.55%,-1.12%,13.5%,19.8%
```

알림 타입별 리포트:

```text
alert_type,n,return_60m,direction_hit_60m,MFE,MAE,UP_FAIL_rate
UP_ALERT,500,+0.12%,54.8%,+0.91%,-0.70%,9.5%
UP_CONTINUATION,420,+0.06%,52.1%,+0.82%,-0.76%,10.2%
WEAKENING_ALERT,390,-0.07%,46.0%,+0.58%,-0.93%,14.8%
RISK_ALERT,300,-0.16%,42.7%,+0.51%,-1.18%,20.1%
VOLATILITY_ALERT,700,+0.01%,50.5%,+1.05%,-1.10%,17.5%
```

---

## 완료 기준

아래 결과가 나와야 합니다.

```text
directional_alerts 테이블 생성
directional_alerts parquet 저장
alert_evaluation_summary.csv 생성
alert_grade_summary.csv 생성
alert_type_summary.csv 생성
라벨 누수 없음
active 종목만 포함
FAIL 품질 날짜 제외
```

좋은 방향성 알림이라면 아래 조건을 만족해야 합니다.

```text
A등급의 direction_hit_60m이 B/C보다 높음
A등급의 return_60m이 B/C보다 높음
A등급의 MAE가 C/RISK보다 작음
RISK 등급의 MAE 또는 down2_60m_rate가 높음
RISK 등급의 UP_FAIL_rate가 높음
UP_ALERT와 UP_CONTINUATION이 C등급보다 방향성이 좋음
```

반대로 아래처럼 나오면 현재 등급화는 실패로 봅니다.

```text
A/B/C 등급 간 direction_hit 차이가 없음
RISK 등급이 오히려 상승 방향성이 더 좋음
RISK 등급의 MAE가 크지 않음
등급별 성과 순서가 불안정함
```

이 경우 Step 13_6의 결론은 “현재 룰 기반 등급은 추세 상태를 충분히 구분하지 못함”으로 기록하고, 다음 단계에서 trend_state 전용 모듈로 이동합니다.

---

## 이번 단계의 핵심

Step 13_6의 목표는 수익형 자동매매 전략을 만드는 것이 아닙니다.

목표는 아래입니다.

```text
알림 등급이 실제 미래 방향성과 위험을 구분하는지 검증한다.
A/B/C/RISK 등급이 의미 있는 순서성을 갖는지 확인한다.
RISK 알림이 실제 위험 알림으로 작동하는지 확인한다.
```

즉, Step 13_6은 방향성 알림 시스템의 첫 번째 품질검사 단계입니다.
