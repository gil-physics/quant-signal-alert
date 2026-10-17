# Step 11. 룰 기반 시그널 v0 만들기

## 할 일

딥러닝이나 ML 전에 사람이 이해할 수 있는 점수 기반 룰을 만듭니다.

입력은 Step 9의 `features_relative`를 사용합니다.

라벨은 점수 계산에 사용하지 않고, 이후 백테스트/검증에서만 사용합니다.

---

## 상승 점수 예

```text
UP_SCORE =
    최근 20분 수익률 점수
  + 거래량 z-score 점수
  + VWAP 상회 점수
  + 당일 고점 근처 점수
  + 3분봉 상승 연속성 점수
  + 5분봉 추세 유지 점수
  + 종목 간 상대강도 점수
```

---

## 상승 후보 조건 예

```text
ret_20m_1m > +0.8%
volume_z_20m_1m > 2.0
vwap_distance_1m > 0
distance_from_day_high_1m이 작음
ret_3bar_3m > 0
trend_strength_5m > 0
활성 종목 중 상대강도 상위권
```

---

## 하락 점수 예

```text
DOWN_SCORE =
    최근 20분 하락률 점수
  + 거래량 z-score 점수
  + VWAP 하회 점수
  + 당일 저점 근처 점수
  + 3분봉 약세 점수
  + 5분봉 하락 추세 점수
  + 종목 간 상대약세 점수
```

---

## 하락 후보 조건 예

```text
ret_20m_1m < -0.8%
volume_z_20m_1m > 2.0
vwap_distance_1m < 0
distance_from_day_low_1m이 작음
ret_3bar_3m < 0
trend_strength_5m < 0
활성 종목 중 상대강도 하위권
```

---

## 지시사항

1. `rule_signal.py`를 만든다.

2. 입력은 `features_relative`를 사용한다.

3. 검사 대상 종목은 `symbols_test.yaml`의 `active: true` 종목으로 제한한다.

4. 각 row마다 `up_score`, `down_score`를 계산한다.

5. 점수 계산 근거도 같이 저장한다.

예:

```text
reason_1
reason_2
reason_3
```

6. 처음에는 threshold를 낮게 잡고 후보를 많이 생성한다.

7. 이후 백테스트에서 threshold를 조절한다.

8. 라벨 컬럼은 점수 계산에 사용하지 않는다.

9. 결과를 `rule_signals` 테이블과 parquet으로 저장한다.

10. 후보 여부를 구분하는 컬럼을 만든다.

예:

```text
is_up_candidate
is_down_candidate
```

---

## 완료 기준

아래 형태의 결과가 나와야 합니다.

```text
symbol
decision_time
up_score
down_score
is_up_candidate
is_down_candidate
reason_1
reason_2
reason_3
```

추가 완료 기준:

```text
rule_signals 테이블 생성
rule_signals parquet 저장
active 종목만 포함
up_score/down_score 결측치 없음
점수 계산에 미래 데이터와 라벨을 사용하지 않음
초기 threshold 기준 후보가 너무 적지 않게 생성됨
```
