# Step 12. 하루 top1/top2 백테스트 만들기

## 할 일

분 단위 전체 예측 정확도가 아니라, 하루에 실제로 선택할 1~2개 신호만 평가합니다.

입력은 `rule_signals`와 `labels`를 결합한 테이블을 사용합니다.

롱 기준 백테스트에서는 상승 점수만 보지 않고, 하락 위험 점수를 강한 필터로 사용합니다.

---

## 백테스트 흐름

1. 모든 종목, 모든 `decision_time`에 점수 계산 결과를 불러온다.
2. 기본 필터를 적용한다.
3. 보수적 롱 필터를 적용한다.
4. 동일 종목 cooldown을 적용한다.
5. 하루 전체 후보 중 top1/top2를 선택한다.
6. 선택된 시점 이후 MFE/MAE를 계산한다.
7. +2%, +3%, +5% 도달 여부를 계산한다.
8. 손절선이 먼저 도달했는지 확인한다.

---

## 기본 필터

```text id="uzsmd7"
09:10 이전 제외
15:00 이후 신규 신호 제외
동일 종목 60분 cooldown
하루 최대 2개
outlier 발생 직후 신호 제외
FAIL 품질 날짜 제외
```

---

## 보수적 롱 필터

처음에는 하락 위험을 더 무겁게 봅니다.

```text id="zpsppp"
is_up_candidate = true
is_down_candidate = false
up_score >= up_threshold
down_score <= down_risk_limit
up_score - down_score >= score_margin
```

초기값 예:

```text id="r04ak5"
up_threshold = 4
down_risk_limit = 1
score_margin = 3
```

`is_up_candidate`와 `is_down_candidate`가 둘 다 true인 row는 v0에서는 제외합니다.

```text id="znbx7d"
both 후보 제외
```

이후 백테스트에서 아래 조건을 비교합니다.

```text id="l8ttlw"
보수적: up_score >= 4, down_score <= 1, margin >= 3
중간:   up_score >= 4, margin >= 2
공격적: up_score >= 3, margin >= 1
```

---

## top1/top2 선택 기준

하루 후보 중 아래 기준으로 정렬합니다.

```text id="m2en8p"
1순위: up_score - down_score 높은 순
2순위: up_score 높은 순
3순위: rank_ret_20m_1m 높은 순
4순위: volume_z_20m_1m 높은 순
5순위: decision_time 빠른 순
```

하루 최대 2개만 선택합니다.

동일 종목은 60분 cooldown을 적용합니다.

---

## 평가 지표

```text id="q3sltp"
signal_count
daily_top1_count
daily_top2_count

hit_up2_rate
hit_up3_rate
hit_up5_rate

average_MFE
average_MAE

stop_first_rate
return_30m
return_60m
return_120m
return_eod

symbol_concentration
both_excluded_count
```

추가로 threshold별 결과를 비교합니다.

```text id="lizrer"
conservative
medium
aggressive
```

---

## 지시사항

1. `daily_topk_backtest.py`를 만든다.

2. 입력은 feature+score+label 테이블이다.

3. `rule_signals`와 `labels`를 `symbol`, `decision_time` 기준으로 결합한다.

4. 라벨은 평가에만 사용하고, 신호 선택 조건에는 사용하지 않는다.

5. 기본 필터와 보수적 롱 필터를 적용한다.

6. `both` 후보는 v0에서 제외한다.

7. 동일 종목 60분 cooldown을 적용한다.

8. 하루 전체 후보 중 top1/top2를 선택한다.

9. 출력은 신호별 결과 테이블이다.

10. 날짜별 요약 리포트를 만든다.

11. 종목별 편중도도 본다.

12. threshold 조합별 결과를 비교할 수 있게 저장한다.

---

## 완료 기준

예를 들어 이런 결과가 나와야 합니다.

```text id="pcfboo"
기간: 2025-01-01 ~ 2025-03-31
종목 수: 3
거래일 수: 60
신호 수: 42
하루 평균 신호 수: 0.70

+2% 도달률: 45%
+3% 도달률: 24%
+5% 도달률: 8%
평균 MFE: +2.3%
평균 MAE: -0.9%
손절 먼저 도달률: 18%
```

추가 완료 기준:

```text id="egscs4"
backtest_signals 테이블 생성
daily_summary 리포트 생성
threshold별 요약 리포트 생성
라벨을 신호 선택에 사용하지 않음
both 후보 제외 여부 기록
cooldown 적용 여부 검증
하루 top1/top2 개수 검증
symbol_concentration 계산
```
