# quant 리플레이 앱

지나간 하루 장을 1분씩 다시 재생하며, 실시간 알림 프로그램이 그날 찍었을 로그를 휴대폰에서 보는 앱.
설명은 발표자가 하고, 앱은 직접 눌러 보는 체험을 맡는다.

- **홈:** 대표 6일의 날짜 카드(별명, 등락률, 스파크라인, 알림 수). 누르면 바로 재생.
- **날짜 선택:** 홈의 버튼 또는 리플레이 상단 날짜를 눌러 달력을 연다. 검증한 **2026-03-03~2026-07-16, 94거래일** 중 선택할 수 있다. 선택한 날짜가 속한 달 또는 데이터의 마지막 달부터 열리고, 데이터가 없는 날짜는 비활성화된다. 달력이 열린 동안 재생은 멈추며, 같은 날을 선택하거나 닫으면 이전 재생 상태로 돌아간다.
- **리플레이:** 큰 시계, 90분 캔들 차트(MA5·MA20, 알림 마커), 종류 필터, 터미널 로그, 재생 컨트롤(속도 1×/4×/16×, 다음 알림, 진행 막대).
- **로그 두 보기:** `쉽게`(한국어 요약, 의미 기준 색)와 `원본`(러너가 실제로 찍은 줄 그대로).
- **알림 상세:** 줄을 누르면 재생이 멈추고 뜻, 모델 점수, 원본 줄이 나온다. "그 뒤 결과"는 재생 시각이 그 결과 창을 지나야 열린다.
- **PWA:** 홈 화면에 추가하면 앱처럼 열린다. 최근 조회한 최대 60일의 데이터는 브라우저 캐시에 남아 있는 동안 오프라인에서도 재생된다. 처음 여는 날짜에는 연결이 필요하다.

## 데이터

신호·모델 점수는 브라우저나 Hosting 서버에서 다시 계산하지 않는다. `scripts/export_replay_bundle.py`가 실제 러너의 리플레이 코드를 돌려
결과를 `public/data/`에 JSON으로 저장한다.

- 롱·숏 줄: `realtime_long_short_trend_alert.py --cross-symbol`과 같은 객체·같은 `_process_minute` 호출을 캡처.
- 이평 돌파 줄: `realtime_touch_alert.py`의 `TouchAlertRunner` 기록 + 표본 외(OOF) 5시드 예측으로 매긴 등급.
- 종목 간 피처는 모델이 학습한 20종목으로 고정한다(나중에 추가된 종목이 과거 피처를 바꾸지 않게).
- 대표 날짜와 별명은 `replay_dates.json`. 모든 날짜는 모델이 학습하지 않은 기간이어야 한다.
- `public/data/index.json`의 `dates`에는 대표 카드 6개, `availableDates`에는 전체 94일의 날짜·파일 경로만 담는다. 초기 화면은 목록만, 날짜 선택 후에는 해당 하루 JSON만 요청한다. 전체 JSON을 PWA 사전 캐시에 넣지 않는다.
- 94일은 단기 움직임 검증에 사용한 `data/processed/short_horizon_replay/manifest.json`의 날짜와 정확히 같다. 게시 준비 시 SHA-256을 검사하며, 기존 6일의 bars/ma/events가 다르면 덮어쓰지 않는다. 신호 재계산 없이 표시용 종목명만 정리한다.
- 일부 날짜의 CNN OOF 등급 누락은 그대로 유지한다. 해당 날짜에는 안내를 표시하고, 등급이 없는 돌파 신호도 재생한다.

```powershell
# cwd = quant, 64-bit
..\Python311-64\python.exe scripts\export_replay_bundle.py                       # replay_dates.json의 모든 날짜
..\Python311-64\python.exe scripts\export_replay_bundle.py --dates 2026-06-19    # 한 날짜만

# 이미 검증한 94일을 앱 정적 파일로 준비 (표준 라이브러리만 사용; 추론·배포 없음)
python scripts\publish_validation_replays.py
```

파리티 확인: `--universe active --out-dir <임시 폴더>`로 내보낸 원본 줄이
`realtime_long_short_trend_alert.py --mode replay --replay-date <날짜> --cross-symbol` 출력과 줄 단위로 같아야 한다
(2026-06-19: 361/361 일치, 돌파 22/22 일치).

## 개발과 배포

```powershell
cd app
npm install
npm run dev        # http://localhost:5173
npm run build      # app/dist (타입 검사 포함)
```

배포는 Firebase Hosting(Google Cloud). 설정은 저장소 루트의 `firebase.json`(`public: app/dist`).

```powershell
npm i -g firebase-tools
firebase login                                   # 본인 계정으로 직접
firebase use --add <프로젝트 ID>                  # 저장소 루트에서
firebase hosting:channel:deploy preview          # 미리보기 URL
firebase deploy --only hosting                   # 본 배포
```

교육용 리플레이다. 주문을 내지 않으며 투자 권유가 아니다.

## 서버 직접 계산 구상 — 미구현

현재 앱에는 계산 API, 작업 큐, 서버 추론 코드 또는 클라우드 리소스를 추가하지 않았다.
향후에는 화면과 날짜 선택을 유지하면서 데이터 공급 경로를 아래처럼 바꾸는 것이 적절하다.

1. **입력 동기화:** 기존 수집기가 만든 1분봉을 서버 저장소로 옮긴다. 기존 엔진은 선택한 종목의 당일 봉만으로 재현되지 않는다. 학습 당시 고정한 20종목의 단면 입력, 이전 8거래일 피처 이력, 돌파 MA에 필요한 이전 2일의 `al_raw` 및 당일 장전 입력도 확보해야 한다. 거래일·누락 봉 검사를 통과한 날짜만 목록에 노출한다.
2. **Python 계산 작업:** 기존 exporter/러너의 계산 부분을 재사용하는 컨테이너 작업을 둔다. 모델·전처리기를 작업 프로세스에서 재사용하고, 봉 종료 시각까지의 입력으로 순차 계산한다. 결과는 지금 앱과 같은 JSON 규격으로 저장한다. API는 작업을 요청하고 상태를 조회하는 역할을 맡는다.
3. **캐시 우선 응답:** `종목 + 날짜 + 원본 데이터 버전 + 모델/유니버스 버전 + 계산 코드 버전`으로 결과를 찾는다. 있으면 즉시 반환하고, 없으면 중복되지 않게 작업을 큐에 넣어 준비 중 상태를 반환한다. 완료 후 앱이 JSON을 받게 한다. 처음에는 장 종료 후 일괄 생성부터 시작하고, 필요해지면 미생성 날짜의 요청 시 계산을 추가한다.
4. **과거 검증과 최신 추론 구분:** 94일의 CNN 등급은 OOF 예측이다. 이 기간을 학습한 최신 production 모델로 다시 계산하면 기존 검증과 다른 결과가 된다. 기존 검증 재현에는 저장한 OOF 또는 해당 fold 모델을 사용하고, 학습 종료일 이후의 새로운 날짜만 적절한 production 모델로 계산한다. OOF 누락을 최신 모델로 메우지 않는다. 미래 결과 계산은 신호 생성 뒤 별도로 수행한다.
5. **서비스 연결:** 현재 Firebase Hosting에서 `/api/` 요청을 별도 Python Cloud Run 서비스로 연결할 수 있다. [Firebase의 Cloud Run 연동 문서](https://firebase.google.com/docs/hosting/cloud-run). 결과 JSON은 객체 저장소에 보관하고, API는 허용 종목·날짜, 동일 요청 중복 방지, 작업 상태·실패 재시도를 관리한다. 배포 전 94일의 bars/ma/events가 기존 정적 파일과 일치하는지 비교한다.

이 구상은 과거 날짜를 서버에서 생성하는 단계다. 실시간 장중 계산은 봉 수집 지연,
확정/수정 봉 처리, 지속 실행 상태와 재시작 복구까지 별도로 설계해야 한다.
