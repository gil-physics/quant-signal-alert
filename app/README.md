# quant 리플레이 앱

지나간 하루 장을 1분씩 다시 재생하며, 실시간 알림 프로그램이 그날 찍었을 로그를 휴대폰에서 보는 앱.
설명은 발표자가 하고, 앱은 직접 눌러 보는 체험을 맡는다.

- **홈:** 날짜 카드(별명, 등락률, 스파크라인, 알림 수). 누르면 바로 재생.
- **리플레이:** 큰 시계, 90분 캔들 차트(MA5·MA20, 알림 마커), 종류 필터, 터미널 로그, 재생 컨트롤(속도 1×/4×/16×, 다음 알림, 진행 막대).
- **로그 두 보기:** `쉽게`(한국어 요약, 의미 기준 색)와 `원본`(러너가 실제로 찍은 줄 그대로).
- **알림 상세:** 줄을 누르면 재생이 멈추고 뜻, 모델 점수, 원본 줄이 나온다. "그 뒤 결과"는 재생 시각이 그 결과 창을 지나야 열린다.
- **PWA:** 홈 화면에 추가하면 앱처럼 열리고, 한 번 본 날짜는 오프라인에서도 재생된다.

## 데이터

브라우저는 아무것도 다시 계산하지 않는다. `scripts/export_replay_bundle.py`가 실제 러너의 리플레이 코드를 돌려
결과를 `public/data/`에 JSON으로 저장한다.

- 롱·숏 줄: `realtime_long_short_trend_alert.py --cross-symbol`과 같은 객체·같은 `_process_minute` 호출을 캡처.
- 이평 돌파 줄: `realtime_touch_alert.py`의 `TouchAlertRunner` 기록 + 표본 외(OOF) 5시드 예측으로 매긴 등급.
- 종목 간 피처는 모델이 학습한 20종목으로 고정한다(나중에 추가된 종목이 과거 피처를 바꾸지 않게).
- 날짜와 별명은 `replay_dates.json`. 모든 날짜는 모델이 학습하지 않은 기간이어야 한다.

```powershell
# cwd = quant, 64-bit
..\Python311-64\python.exe scripts\export_replay_bundle.py                       # replay_dates.json의 모든 날짜
..\Python311-64\python.exe scripts\export_replay_bundle.py --dates 2026-06-19    # 한 날짜만
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
