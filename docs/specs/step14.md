# Step 14. LightGBM 위험 모델 만들기

## 할 일

Step 13_6에서 RISK 등급이 실제 하방 위험을 잘 구분하지 못했습니다.

따라서 Step 14에서는 자동매매 모델이 아니라, **위험 알림을 보정하는 ML 모델**을 먼저 만듭니다.

목표는 아래를 예측하는 것입니다.

```text id="ib9zqa"
이 알림이 실제로 위험한가?
알림 후 MAE가 크게 나오는가?
UP_FAIL 또는 down2가 발생할 가능성이 높은가?
```

즉, Step 14의 목표는 “매수 신호 만들기”가 아니라 **위험한 알림을 걸러내는 risk model**입니다.

---

## 기본 방향

첫 번째 모델은 방향성 모델이 아니라 위험 모델로 만듭니다.

```text id="ix5ueu"
Model 1: Risk model
target = UP_FAIL 또는 future_MAE < -1.5% 또는 down2_60m
```

방향성 모델은 이후 단계에서 만듭니다.

```text id="2bimpc"
Model 2: Direction model
target = up3_120m 또는 future_return_60m > 0
```

이번 Step 14에서는 Model 1만 구현합니다.

---

## 입력 데이터

입력은 기존 파이프라인 결과를 사용합니다.

```text id="caluwu"
features_relative
rule_signals
directional_alerts
labels
bars_1m source = al_regular
quality report
```

학습 feature에는 미래 라벨을 절대 넣지 않습니다.

라벨은 target 생성과 평가에만 사용합니다.

---

## 학습 대상

우선 `directional_alerts`에 포함된 알림 후보를 대상으로 학습합니다.

즉, 전체 1분봉 row 전체가 아니라, Step 13_6에서 생성된 알림 후보 중에서 위험한 알림을 구분합니다.

기본 target 후보:

```text id="j4vykd"
target_up_fail:
UP_FAIL이면 1, 아니면 0

target_mae_risk:
future_MAE <= -1.5% 이면 1, 아니면 0

target_down2_60m:
down2_60m이면 1, 아니면 0
```

처음에는 아래 target을 우선 사용합니다.

```text id="3d2q0o"
primary target = target_mae_risk
secondary target = target_up_fail
```

---

## 사용할 feature

미래 정보가 없는 현재 시점 feature만 사용합니다.

예:

```text id="m5tmsp"
up_score
down_score
alert_score
direction_score
risk_score

ret_5m_1m
ret_20m_1m
ret_60m_1m

volume_z_20m_1m
volume_z_60m_1m

vwap_distance_1m
range_1m
body_1m
upper_shadow_1m
lower_shadow_1m

rank_ret_20m_1m
rank_volume_z_20m_1m
relative_ret_20m_to_universe
relative_volume_z_to_universe
universe_breadth
universe_mean_ret_20m

ret_1bar_3m
ret_3bar_3m
volume_z_20bar_3m

ret_1bar_5m
ret_3bar_5m
trend_strength_5m
```

가능하면 MA/VWAP 관련 feature도 포함합니다.

```text id="yjg7np"
ma5_prev_distance
above_ma5_prev
cross_up_ma5_prev
cross_down_ma5_prev
ma5_slope
```

단, 이 feature들이 아직 없다면 Step 14에서는 optional로 둡니다.

---

## 데이터 분할

시간 기준으로 train/validation을 나눕니다.

랜덤 split은 사용하지 않습니다.

예:

```text id="gedawz"
train:
2025-06-02 ~ 2026-02-28

validation:
2026-03-01 ~ 2026-06-19
```

또는 전체 기간의 앞 70%를 train, 뒤 30%를 validation으로 사용합니다.

같은 날짜의 row가 train과 validation에 섞이지 않게 합니다.

---

## 모델

처음에는 LightGBM을 사용합니다.

```text id="4gmkff"
LightGBM binary classifier
objective = binary
target = target_mae_risk
```

LightGBM이 설치되어 있지 않으면 Logistic Regression 또는 sklearn HistGradientBoostingClassifier로 대체할 수 있습니다.

이번 단계에서는 딥러닝을 사용하지 않습니다.

---

## 지시사항

1. `risk_model.py`를 만든다.

2. `train_risk_model.py`를 만든다.

3. 입력은 `directional_alerts`와 기존 feature/label 테이블을 결합해서 만든다.

4. feature 생성 시 미래 라벨 컬럼을 제외한다.

5. 시간 기준 train/validation split을 사용한다.

6. target은 우선 `target_mae_risk`를 사용한다.

7. 보조 target으로 `target_up_fail`, `target_down2_60m`도 실험할 수 있게 만든다.

8. LightGBM binary classifier를 학습한다.

9. validation에서 예측 확률 `risk_prob`를 생성한다.

10. `risk_prob`를 분위수별로 나눠 실제 위험률을 검증한다.

예:

```text id="yo69rl"
risk_prob 상위 10%
risk_prob 상위 20%
risk_prob 중간 40%
risk_prob 하위 20%
```

11. 기존 RISK 등급보다 위험 구분력이 좋아졌는지 비교한다.

12. 결과를 테이블과 CSV로 저장한다.

```text id="kci6vp"
risk_model_predictions
risk_model_summary.csv
risk_prob_decile_report.csv
feature_importance.csv
```

13. 모델 파일도 저장한다.

```text id="nv6clo"
models/risk_model_lgbm.txt
```

---

## 평가 지표

classification 지표:

```text id="8u0jq5"
AUC
PR-AUC
logloss
precision@top10%
recall@top10%
```

알림 시스템 관점 지표:

```text id="mnzrck"
risk_prob 상위 구간의 실제 MAE 위험률
risk_prob 하위 구간의 실제 MAE 위험률
risk_prob 상위 구간의 UP_FAIL rate
risk_prob 상위 구간의 down2_60m rate
```

중요한 비교:

```text id="ykrq5a"
기존 RISK 등급의 실제 위험률
vs
LightGBM risk_prob 상위 10%의 실제 위험률
```

---

## 리포트 예

```text id="uzht4x"
risk_bucket,n,mae_risk_rate,up_fail_rate,down2_60m_rate,avg_MAE,avg_return_60m
top_10%,500,31.2%,22.5%,18.1%,-1.42%,-0.21%
top_20%,1000,27.4%,19.8%,15.3%,-1.25%,-0.16%
middle_40%,2000,17.8%,11.2%,8.7%,-0.84%,-0.03%
bottom_20%,1000,9.1%,5.4%,4.0%,-0.48%,+0.05%
```

좋은 risk model이면 아래 구조가 나와야 합니다.

```text id="8j5e56"
risk_prob 높음 → 실제 MAE 위험률 높음
risk_prob 높음 → UP_FAIL rate 높음
risk_prob 낮음 → 실제 MAE 작음
risk_prob 낮음 → down2_60m 낮음
```

---

## 완료 기준

아래 결과가 나와야 합니다.

```text id="kdoz8c"
risk_model.py 생성
train_risk_model.py 생성
risk_model_predictions 테이블 생성
risk_model_summary.csv 생성
risk_prob_decile_report.csv 생성
feature_importance.csv 생성
모델 파일 저장
라벨 누수 없음
시간 기준 train/validation split 사용
```

성공 기준:

```text id="cwzyto"
validation AUC가 0.5보다 의미 있게 높음
risk_prob 상위 구간의 실제 위험률이 하위 구간보다 높음
기존 RISK 등급보다 위험 구분력이 개선됨
feature importance가 해석 가능함
```

실패 기준:

```text id="vtyqk4"
AUC가 0.5 근처
risk_prob 상위/하위 구간의 실제 위험률 차이가 거의 없음
validation에서만 성능이 무너짐
feature importance가 불안정함
```

Step 14의 목표는 자동매매 모델을 만드는 것이 아닙니다.

목표는 **방향성 알림 중 위험한 알림을 더 잘 구분해서 A/B/C/RISK 등급을 보정할 수 있는지 확인하는 것**입니다.
