# Step 7. 1m/3m/5m 피처 생성

## 할 일

각 timeframe별 피처를 따로 만듭니다.

1분 단위 decision time 병합은 Step 8에서 처리합니다.

Step 7에서는 `features_1m`, `features_3m`, `features_5m`를 각각 만들고, 각 피처의 사용 가능 시각인 `bar_end`를 반드시 남깁니다.

---

## 7-1. 1분봉 피처

`bars_1m`의 `source = al_regular`에서 생성합니다.

```text
ret_1m_1m
ret_3m_1m
ret_5m_1m
ret_10m_1m
ret_20m_1m
ret_60m_1m

range_1m
body_1m
upper_shadow_1m
lower_shadow_1m

volume_z_20m_1m
volume_z_60m_1m

vwap_distance_1m
return_from_open_1m
distance_from_day_high_1m
distance_from_day_low_1m
```

추가로 병합과 검증을 위해 아래 컬럼을 둡니다.

```text
symbol
decision_time
close
bar_end_1m
```

---

## 7-2. 3분봉 피처

`bars_3m`의 `source = resampled_from_al_regular`에서 생성합니다.

```text
ret_1bar_3m
ret_2bar_3m
ret_3bar_3m

range_1bar_3m
body_1bar_3m

volume_z_10bar_3m
volume_z_20bar_3m
```

추가로 병합과 검증을 위해 아래 컬럼을 둡니다.

```text
symbol
bar_end_3m
```

---

## 7-3. 5분봉 피처

`bars_5m`의 `source = resampled_from_al_regular`에서 생성합니다.

구현방식은 3분봉과 동일. 3m -> 5m
---

## 지시사항

1. `features_1m.py`, `features_3m.py`, `features_5m.py`를 나눈다.

2. 모든 피처에 timeframe suffix를 붙인다.

예:

```text
ret_5m_1m
ret_1bar_3m
ret_1bar_5m
volume_z_20m_1m
volume_z_20bar_3m
```

3. 피처 생성 후 결측치를 확인한다.

4. 초기 구간의 rolling 결측치는 버린다.

5. 모든 rolling/shift 계산은 종목별, 날짜별로 수행한다.

6. 날짜 경계를 넘어 전일 데이터를 참조하지 않는다.

7. 피처 계산에는 미래 데이터를 사용하지 않는다.

8. 검사 대상 종목은 `symbols_test.yaml`의 `active: true` 종목으로 제한한다.

9. 기본 model source는 config의 `data.model_source = al_regular`를 사용한다.

10. 3m/5m 피처는 Step 5에서 생성된 `resampled_from_al_regular` 데이터만 사용한다.

11. Step 7에서는 1m/3m/5m 피처를 아직 병합하지 않는다.

12. 1m 기준 decision time에 3m/5m 피처를 붙이는 작업은 Step 8에서 한다.

---

## 완료 기준

아래와 같은 피처 테이블이 나와야 합니다.

```text
symbol
decision_time
close
bar_end_1m
ret_1m_1m
ret_5m_1m
ret_1bar_3m
ret_1bar_5m
volume_z_20m_1m
vwap_distance_1m
...
```

Step 7에서는 실제로 아래 3개 결과가 생성되어야 합니다.

```text
features_1m
features_3m
features_5m
```

추가 완료 기준:

```text
active 종목만 피처 생성됨
features_1m은 source = al_regular 기준
features_3m/features_5m은 source = resampled_from_al_regular 기준
각 피처 테이블에 bar_end 컬럼이 있음
날짜 경계 누수 없음
미래 데이터 누수 없음
초기 rolling 결측치 처리됨
```
