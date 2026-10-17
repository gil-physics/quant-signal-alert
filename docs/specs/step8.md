# Step 8. 피처 병합 시 누수 방지

## 할 일

현재 시점에서 아직 완성되지 않은 3분봉/5분봉 피처를 사용하지 않게 합니다.

Step 7에서 만든 아래 테이블을 1분 단위 `decision_time` 기준으로 병합합니다.

```text id="x8t57o"
features_1m
features_3m
features_5m
```

---

## 예시

현재 의사결정 시점:

```text id="sf1w9q"
decision_time = 09:07:00
```

사용 가능:

```text id="m7q4p0"
1분봉 피처: 09:06~09:07, bar_end_1m = 09:07
3분봉 피처: 09:03~09:06, bar_end_3m = 09:06
5분봉 피처: 09:00~09:05, bar_end_5m = 09:05
```

사용 금지:

```text id="v6jg6q"
3분봉 피처: 09:06~09:09, bar_end_3m = 09:09
5분봉 피처: 09:05~09:10, bar_end_5m = 09:10
```

---

## 지시사항

1. 모든 피처 테이블에 `bar_end`를 남긴다.

```text id="b9mgk1"
features_1m: bar_end_1m
features_3m: bar_end_3m
features_5m: bar_end_5m
```

2. 병합 기준은 `features_1m.decision_time`으로 한다.

3. 병합할 때 아래 조건을 강제한다.

```text id="g9j6fi"
bar_end_3m <= decision_time
bar_end_5m <= decision_time
```

4. 3분봉/5분봉은 asof join 방식으로 붙인다.

```text id="kgacvw"
같은 symbol에서
decision_time 이하인 가장 최근 bar_end_3m
decision_time 이하인 가장 최근 bar_end_5m
```

5. 병합 대상 종목은 `symbols_test.yaml`의 `active: true` 종목으로 제한한다.

6. `features_1m`은 `source = al_regular` 기준 데이터만 사용한다.

7. `features_3m`, `features_5m`은 `source = resampled_from_al_regular` 기준 데이터만 사용한다.

8. 병합 후 누수 검사를 한다.

검사 조건:

```python id="xtzbr9"
assert feature_3m_bar_end <= decision_time
assert feature_5m_bar_end <= decision_time
```

9. 병합 결과를 `features_merged` 테이블과 parquet으로 저장한다.

---

## 완료 기준

누수 검사에서 오류 row가 0개여야 합니다.

추가 완료 기준:

```text id="w27u0l"
features_1m/3m/5m 병합 완료
active 종목만 병합됨
3m/5m는 asof join으로 병합됨
bar_end_3m <= decision_time 위반 0건
bar_end_5m <= decision_time 위반 0건
병합 후 잔여 NaN 확인
features_merged 테이블 생성
features_merged parquet 저장
```
