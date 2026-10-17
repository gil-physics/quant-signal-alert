Step 2. 데이터 테이블 구조 통일

다음구조로 변경할게.
raw:
  000660_AL_1min.csv 전체 08:00~19:59 보관
  source = al_raw
regular 후보:
  AL에서 정규장만 필터링
  source = al_regular
비교용:
  단일 거래소 정규장 파일
  source = single_regular

할 일

모든 데이터에 반드시 symbol을 넣습니다.

1분봉 테이블
CREATE TABLE bars_1m (
    symbol TEXT NOT NULL,
    bar_start TEXT NOT NULL,
    bar_end TEXT NOT NULL,
    open INTEGER NOT NULL,
    high INTEGER NOT NULL,
    low INTEGER NOT NULL,
    close INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (symbol, bar_start)
);
3분봉/5분봉 테이블
CREATE TABLE bars_3m (
    symbol TEXT NOT NULL,
    bar_start TEXT NOT NULL,
    bar_end TEXT NOT NULL,
    open INTEGER NOT NULL,
    high INTEGER NOT NULL,
    low INTEGER NOT NULL,
    close INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (symbol, bar_start)
);

CREATE TABLE bars_5m (
    symbol TEXT NOT NULL,
    bar_start TEXT NOT NULL,
    bar_end TEXT NOT NULL,
    open INTEGER NOT NULL,
    high INTEGER NOT NULL,
    low INTEGER NOT NULL,
    close INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (symbol, bar_start)
);
지시사항
기존 단일 종목용 테이블이 있다면 백업한다.
새 테이블은 symbol + bar_start를 primary key로 만든다.
bar_start, bar_end를 반드시 둔다.
source에는 데이터 출처를 넣는다.

datetime → bar_start
bar_end  → bar_start + 1분
open     → open
high     → high
low      → low
close    → close
volume   → volume
value    → amount 또는 value
symbol   → 파일명/사용자 입력으로 추가
source   → csv_import
created_at → import 시각

예:

kiwoom_opt10080
realtime_rebuilt
resampled_from_1m
완료 기준

아래 쿼리가 정상 동작해야 합니다.

SELECT symbol, COUNT(*)
FROM bars_1m
GROUP BY symbol;