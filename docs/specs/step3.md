Step 3. 과거 1분봉 수집기 멀티종목화
할 일

기존 1종목 기준 과거 1분봉 수집/import 흐름을 여러 종목 반복 구조로 확장한다.

Step 3의 목표는 추가 종목들의 과거 1분봉을 bars_1m 표준 테이블에 저장하는 것이다.
config/config.yaml 데이터 분류 업데이트 확인.
모델용 데이터 기준 source는 al_regular로 고정한다.

후속 파이프라인의 기준 저장소는 SQLite bars_1m이다.
config/symbols_test.yaml에 코드 업데이트해둠. 종목코드는 하드코딩이 아니라 symbol에서 읽는 것.

함수 구조
def collect_1m(symbol: str, start_date: str, end_date: str):
    """
    symbol 하나의 과거 1분봉을 수집해서 bars_1m에 저장
    """
    pass

멀티종목 실행:

symbols = load_active_symbols()

for symbol in symbols:
    collect_1m(symbol, start_date, end_date)
지시사항
collect_1m() 함수는 종목코드를 인자로 받게 만든다.
저장할 때 모든 row에 symbol을 넣는다.
이미 저장된 symbol + bar_start는 중복 저장하지 않는다.
수집 실패 시 다음 종목으로 넘어가도록 한다.
수집 로그를 남긴다.

로그 예:

[2026-06-21 21:00:01] START collect 005930
[2026-06-21 21:02:13] DONE collect 005930 rows=39120
[2026-06-21 21:02:14] START collect 000660
완료 기준

처음에는 2종목 기준으로:

005930 1분봉 저장 완료
000660 1분봉 저장 완료
중복 없음
누락 리포트 생성 가능