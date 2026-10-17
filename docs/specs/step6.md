# Step 6. 데이터 품질 리포트 만들기

## 할 일

수집/생성된 봉 데이터가 정상인지 자동 점검합니다.

검사 기준 source는 모델용 데이터인 `al_regular`입니다.

## 체크 항목

```text
중복 bar_start
누락 bar_start
open/high/low/close 관계 오류
거래량 음수
가격 0
정규장 외 데이터 혼입
비정상 급등락 봉
outlier 포함 여부
bar_end 간격 오류
source 혼입 여부
```

## 리포트 예

기존 예시의 390/130/78은 연속 정규장 기준의 이상적인 개수입니다.

현재 AL 데이터는 실제로 하루 약 380개 1분봉이 나오고, 3분봉/5분봉도 약 126개/76개가 정상일 수 있으므로 고정값으로 강제하지 않습니다.

```text
symbol,date,timeframe,expected,actual,missing,duplicate,ohlc_error,bar_end_error,status
005930,2026-06-21,1m,range,380,0,0,0,0,OK
005930,2026-06-21,3m,range,126,0,0,0,0,OK
005930,2026-06-21,5m,range,76,0,0,0,0,OK
000660,2026-06-21,1m,range,379,1,0,0,0,WARN
```

## 지시사항

1. `quality_check.py`를 만든다.
2. `bars_1m`, `bars_3m`, `bars_5m`을 모두 검사한다.
3. `bars_1m`은 `source = al_regular`만 검사한다.
4. `bars_3m`, `bars_5m`은 `source = resampled_from_1m`만 검사한다.
5. 검사 대상 종목은 `symbols_test.yaml`의 `active: true` 종목으로 제한한다.
6. 결과를 CSV로 저장한다.
7. 오류가 있는 날짜/종목은 학습에서 제외하거나 보정할 수 있도록 `OK/WARN/FAIL` 상태를 남긴다.
8. Step 6에서는 데이터를 직접 삭제하지 않고 리포트만 만든다.
9. 누락 bar 또는 expected 범위 이탈은 우선 `WARN`으로 처리한다.
10. 중복, OHLC 오류, 가격 0 이하, 거래량 음수, bar_end 간격 오류는 `FAIL`로 처리한다.

## 완료 기준

학습에 넣는 데이터는 아래 조건을 만족해야 합니다.

```text
중복 없음
OHLC 오류 없음
정규장 기준 시간 정상
bar_end 간격 정상
source 혼입 없음
가격 0 이하 없음
거래량 음수 없음
누락률 허용 범위 이내
```

추가 완료 기준:

```text
quality_check.py 실행 성공
bars_1m, bars_3m, bars_5m 모두 검사됨
active 종목만 검사됨
결과 CSV 생성됨
390/130/78 고정 기대값을 강제하지 않음
```
