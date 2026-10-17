Step 0. 프로젝트 기준 확정
할 일

먼저 프로젝트의 기준을 고정합니다.

원천 데이터: 1분봉
파생 데이터: 3분봉, 5분봉
학습 시간: 09:00~15:30 정규장
raw 저장: 가능하면 08:00~20:00 전체 저장
모델 입력: 정규장 clean 데이터
주문: 자동주문 없음
알림: 하루 최대 1~2개
완료 기준

아래 기준을 문서나 config.yaml에 적어둡니다.

session:
  train_start: "09:00:00"
  train_end: "15:30:00"
  raw_start: "08:00:00"
  raw_end: "20:00:00"

signals:
  max_signals_per_day: 2
  cooldown_minutes_per_symbol: 60
  min_minutes_after_open: 10
  no_new_signal_minutes_before_close: 30
Step 1. 고정 종목 리스트 만들기
할 일

시총 계산은 하지 않고, 직접 종목코드를 고정합니다.

처음에는 2~3개만 쓰세요.

예:

symbols:
  - code: "005930"
    name: "삼성전자"
    sector: "semiconductor"
    active: true

  - code: "000660"
    name: "SK하이닉스"
    sector: "semiconductor"
    active: true

  - code: "005380"
    name: "현대차"
    sector: "auto"
    active: true

파일명:

config/symbols_test.yaml
지시사항
config/ 폴더를 만든다.
symbols_test.yaml을 만든다.
처음에는 2종목만 active: true로 둔다.
3종목 테스트 때 세 번째 종목을 켠다.
20종목 확장 때도 같은 파일 구조를 유지한다.
완료 기준

코드 어디에도 종목코드를 직접 박아두지 않습니다.

나쁜 예:

symbol = "000660"

좋은 예:

symbols = load_symbols("config/symbols_test.yaml")