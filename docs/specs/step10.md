# Step 10. 라벨 생성

## 할 일

각 종목, 각 `decision_time`마다 미래 가격을 보고 라벨을 만듭니다.

입력 row는 Step 9의 `features_relative`를 기준으로 합니다.

라벨 계산용 가격은 `bars_1m`의 `source = al_regular`을 사용합니다.

처음부터 UP5만 보지 말고 UP2, UP3, UP5를 모두 만듭니다.

---

## 라벨 정의

현재가:

```text id="q03al5"
close_t
```

미래 구간:

```text id="2eugjr"
향후 60분
향후 120분
향후 180분
장마감까지
```

미래 가격 계산 시 현재 봉은 제외합니다.

```text id="pjj9ni"
future bars:
bar_start >= decision_time
```

라벨:

```text id="3z75fu"
UP2:
future_high >= close_t * 1.02

DOWN2:
future_low <= close_t * 0.98

UP3:
future_high >= close_t * 1.03

DOWN3:
future_low <= close_t * 0.97

UP5:
future_high >= close_t * 1.05

DOWN5:
future_low <= close_t * 0.95
```

고정 horizon 라벨은 미래 60분/120분/180분을 충분히 관측할 수 있는 row만 학습에 사용합니다.

장마감까지 라벨은 같은 거래일의 남은 구간만 사용합니다.

---

## 실전형 라벨

롱 기준:

```text id="f4tehz"
UP5_SUCCESS:
+5%에 먼저 도달

UP_FAIL:
+5% 도달 전에 -1.5% 손절선 먼저 도달

NO_EVENT:
둘 다 아님
```

손절선:

```text id="s1gbac"
stop_loss = close_t * 0.985
```

`UP5_SUCCESS`와 `UP_FAIL`은 미래 high/low의 도달 여부뿐 아니라, 어느 쪽이 먼저 도달했는지를 기준으로 계산합니다.

---

## 지시사항

1. `make_labels.py`를 만든다.

2. 종목별로 라벨을 계산한다.

3. 입력은 `features_relative`의 `symbol`, `decision_time`, `close`를 사용한다.

4. 라벨 계산용 미래 가격은 `bars_1m`의 `source = al_regular`을 사용한다.

5. 검사 대상 종목은 `symbols_test.yaml`의 `active: true` 종목으로 제한한다.

6. 라벨 계산은 미래 high/low를 사용하되, 피처에는 절대 미래를 넣지 않는다.

7. 여러 라벨을 동시에 만든다.

예:

```text id="oblwhi"
up2_60m
up3_120m
up5_180m
up5_eod
down2_60m
down3_120m
down5_180m
down5_eod
```

8. horizon이 부족한 row는 학습 제외할 수 있도록 flag를 남긴다.

예:

```text id="j0jcrx"
has_full_60m
has_full_120m
has_full_180m
```

9. 결과를 `labels` 테이블과 parquet으로 저장한다.

10. 라벨 분포 리포트를 CSV로 저장한다.

---

## 완료 기준

라벨 분포 리포트가 나와야 합니다.

예:

```text id="s6n219"
전체 row: 120,000
UP2_120m: 4.8%
UP3_120m: 1.7%
UP5_180m: 0.3%
DOWN2_120m: 4.2%
DOWN3_120m: 1.5%
DOWN5_180m: 0.2%
```

추가 완료 기준:

```text id="ug3ebf"
labels 테이블 생성
labels parquet 저장
active 종목만 포함
features_relative의 decision_time 기준으로 라벨 생성
bars_1m source = al_regular 기준으로 미래 high/low 계산
현재 봉 제외
미래 피처 누수 없음
has_full_60m/120m/180m flag 생성
라벨별 양성 비율 리포트 생성
```

UP5가 너무 적으면 모델은 UP3부터 시작합니다.
