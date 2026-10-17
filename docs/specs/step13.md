# Step 13. 룰 기반 threshold 조정

## 할 일

점수가 몇 점 이상일 때 알림을 보낼지 정합니다.

기본 베이스라인은 아래로 둡니다.

```text id="avz5cu"
conservative
entry_delay_3m
top1 중심 평가
top2 보조 평가
```

UP5만 목표로 보지 않고, UP2/UP3를 주 평가 대상으로 둡니다.

---

## 지시사항

1. `threshold_sweep.py`를 만든다.

2. 입력은 `rule_signals`, `labels`, `bars_1m al_regular`을 사용한다.

3. 라벨은 평가에만 사용하고, threshold 선택 조건에는 사용하지 않는다.

4. 기본 조건은 `conservative + entry_delay_3m`으로 한다.

5. `up_score` 또는 `up_score - down_score` 기준으로 상위 구간을 나눠 성과를 본다.

```text id="r4onkw"
상위 10%
상위 5%
상위 2%
상위 1%
```

6. score threshold 조합도 같이 비교한다.

```text id="wa9tq3"
up_threshold: 4, 5, 6
down_risk_limit: 0, 1, 2
score_margin: 2, 3, 4, 5
top_k: 1, 2
entry_delay: 0m, 3m
```

7. 각 구간의 +2%, +3%, +5% 도달률을 비교한다.

```text id="zbakyq"
hit_up2_rate
hit_up3_rate
hit_up5_rate
```

8. 리스크 지표도 같이 비교한다.

```text id="s0hbda"
average_MFE
average_MAE
stop_first_rate
return_30m
return_60m
return_eod
```

9. 신호 수가 너무 많으면 threshold를 올린다.

10. 신호가 너무 적으면 UP5 대신 UP3 목표로 조정한다.

11. threshold 선택은 전체 기간 하나로만 하지 않고 train/validation을 나눠서 본다.

예:

```text id="zrckx0"
train:
2025-06-02 ~ 2026-02-28

validation:
2026-03-01 ~ 2026-06-19
```

12. train에서 좋아 보인 threshold가 validation에서도 유지되는지 확인한다.

13. 결과를 threshold별 summary CSV로 저장한다.

---

## 평가 기준

주 평가 대상:

```text id="msyc7k"
up2_120m
up3_120m
up2_180m
up3_180m
```

보조 평가 대상:

```text id="y7w449"
up5_180m
up5_eod
```

리스크 평가 대상:

```text id="k219zt"
down2_120m
down2_180m
stop_first
average_MAE
return_30m
return_60m
```

---

## 완료 기준

아래 조건을 만족하는 threshold를 찾습니다.

```text id="qu8z3s"
하루 평균 신호 수: 0~2개
평균 MFE가 MAE보다 충분히 큼
+3% 도달률이 의미 있음
+5% 도달 신호가 일부 존재
stop_first가 기존 baseline보다 낮음
return_30m 또는 return_60m이 개선됨
특정 종목에 과도하게 편중되지 않음
validation 구간에서도 성능이 유지됨
```

예상 출력:

```text id="xu5xu5"
threshold_name
top_percentile
up_threshold
down_risk_limit
score_margin
top_k
entry_delay
signal_count
avg_signals_per_day
hit_up2_rate
hit_up3_rate
hit_up5_rate
average_MFE
average_MAE
stop_first_rate
return_30m
return_60m
return_eod
symbol_concentration
train_or_validation
```

최종 선택은 `hit_up5_rate`가 가장 높은 조합이 아니라, 아래 균형이 가장 좋은 조합으로 합니다.

```text id="xxtdwe"
UP2/UP3 도달률 유지
+ stop_first 감소
+ MAE 감소
+ ret30/ret60 개선
+ 신호 수 충분
```

UP5가 너무 적으면 모델과 룰의 1차 목표는 UP3부터 시작합니다.
