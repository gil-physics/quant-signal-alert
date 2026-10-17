Step 4 = source 정책 확정 + al_regular 모델용 데이터 검증

완료 기준:

config.data.model_source = al_regular
al_raw/al_regular/single_regular 역할 문서화
al_regular만 후속 단계에서 읽도록 함수 기본값 설정
al_regular의 NULL/OHLC/시간범위/급등락 이상치 리포트 생성
심각한 이상치가 있으면 report에 남기고 학습 제외 여부는 Step 6에서 판단

즉 Step 4에서 데이터를 또 대규모로 변환할 필요는 없습니다.