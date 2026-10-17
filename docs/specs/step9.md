# Step 9. 종목 간 상대강도 피처 추가

## 할 일

2종목부터라도 종목 간 비교 피처를 만듭니다. (현재 활성되어있는 종목 사용)

Step 8에서 만든 `features_merged`를 입력으로 사용합니다.

같은 `decision_time`에 존재하는 활성 종목들을 모아서, 종목 간 순위와 평균 대비 상대값을 계산합니다.

---

## 피처 예

```text id="bxrcdc"
rank_ret_5m_1m
rank_ret_20m_1m
rank_volume_z_20m_1m
relative_ret_20m_to_universe
relative_volume_z_to_universe
universe_mean_ret_20m
universe_breadth
```

`rank`는 값이 클수록 좋은 것으로 보고 1등이 가장 강한 종목입니다.

예:

```text id="psmyxq"
ret_20m_1m이 가장 높은 종목 → rank_ret_20m_1m = 1
ret_20m_1m이 두 번째로 높은 종목 → rank_ret_20m_1m = 2
```

---

## 계산 예

삼성전자 최근 20분 수익률: +0.3%
SK하이닉스 최근 20분 수익률: +1.2%
현대차 최근 20분 수익률: -0.1%

```text id="xgjvv6"
SK하이닉스 rank_ret_20m_1m = 1
삼성전자 rank_ret_20m_1m = 2
현대차 rank_ret_20m_1m = 3
```

평균 대비 상대값 예:

```text id="tdhv1d"
universe_mean_ret_20m = 활성 종목 ret_20m_1m 평균
relative_ret_20m_to_universe = ret_20m_1m - universe_mean_ret_20m
```

`universe_breadth`는 같은 시점에서 상승 종목 비율로 계산합니다.

```text id="yw3yp6"
universe_breadth = ret_20m_1m > 0 인 종목 수 / 활성 종목 수
```

---

## 지시사항

1. 같은 `decision_time`에 활성 종목들을 모은다.

2. 입력은 Step 8 결과인 `features_merged`를 사용한다.

3. 검사 대상 종목은 `symbols_test.yaml`의 `active: true` 종목으로 제한한다.

4. 수익률, 거래량 z-score 기준으로 rank를 계산한다.

```text id="kwnlpv"
ret_5m_1m
ret_20m_1m
volume_z_20m_1m
```

5. 평균 대비 상대값을 만든다.

```text id="d42yos"
relative_ret_20m_to_universe
relative_volume_z_to_universe
```

6. universe 공통 피처를 만든다.

```text id="zrk5k0"
universe_mean_ret_20m
universe_breadth
active_symbol_count
```

7. 종목이 2개뿐이어도 동일하게 계산한다.

8. 한 `decision_time`에 활성 종목 수보다 row가 부족하면 해당 시점은 제외하거나 `WARN`으로 리포트한다.

9. 미래 데이터는 사용하지 않는다. 같은 `decision_time`에 이미 완성되어 병합된 피처만 사용한다.

10. 결과를 `features_relative` 테이블과 parquet으로 저장한다.

---

## 완료 기준

각 `decision_time`마다 활성 종목 수만큼 row가 있어야 합니다.

예:

종목 3개라면

```text id="6fvw13"
09:10 row 3개
09:11 row 3개
09:12 row 3개
```

종목 2개라면

```text id="2cp4w6"
09:10 row 2개
09:11 row 2개
09:12 row 2개
```

추가 완료 기준:

```text id="mj0w1m"
features_relative 테이블 생성
features_relative parquet 저장
active 종목만 포함
각 decision_time별 active_symbol_count 계산
rank_ret_20m_1m 범위가 1~active_symbol_count 안에 있음
relative_ret_20m_to_universe 계산 완료
universe_mean_ret_20m 계산 완료
universe_breadth 계산 완료
결측치 확인 완료
```
