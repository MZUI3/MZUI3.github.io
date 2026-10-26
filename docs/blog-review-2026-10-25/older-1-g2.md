# 블로그 교정·내용 확인 (older-1-g2)

대상: `_posts/2026-10-20-rt-1.md`, `_posts/2026-10-20-gato.md`, `_posts/2026-10-20-act-aloha.md`, `_posts/2026-10-19-vision-language-action-models.md`

네 글 모두 문장이 이미 잘 다듬어져 있습니다. 맞춤법·띄어쓰기·조사 오류는 찾지 못했습니다. 애매한 곳은 고치지 않는다는 원칙에 따라 교정 수정은 하지 않았습니다. 예를 들어 gato의 "DQN이 Atari 게임마다 따로 학습되는 것처럼요."는 해요체지만 말투로 보고 그대로 두었습니다. 모든 `{% post_url %}` 대상(24개)이 `_posts/`에 있는 것을 확인했습니다.

## 2026-10-20-rt-1.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 데이터: 로봇 13대, 17개월, 13만 개 에피소드, 지시 744개(표 합계 130+337+16+6+84+162+9 = 744), 처음 보는 지시 21개
- TokenLearner 2.4배, 토큰 재사용 1.7배 속도 향상, 지연 280ms(최대 관찰 지연), 3Hz / 100ms 예산, 사람 작업 시간 2~4초
- 표 "일반화와 강건성"의 성공률과 추론 시간(Gato 129ms, BC-Z 5.3ms, BC-Z XL 5.9ms, RT-1 15ms), Gato 원래 크기 1.2B의 1.9초, 축소판 37M
- 현실 시나리오 70/30/45/55%, 시뮬레이션 92→90, 23→87, 7→33, Kuka 20만 9천 개, 22→39, Kuka만 0%
- SayCan 계획 87%, 실행 67/67, Gato 0%, BC-Z 13%, 최대 50단계
- 데이터 다양성 표와 모델 ablation 표의 모든 수치·추론 시간
- 파라미터 16M / 19M / 8층 / 35M, 평가 3,000회 이상

## 2026-10-20-gato.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- "행동은 이미 [-1, 1] 범위라 이 변환(mu-law)을 거치지 않습니다"가 논문과 맞는지
- prompt 25%, 그중 절반은 에피소드 끝부분
- 제어 데이터 596개 작업, 6,300만 에피소드, 1.5조 토큰, 85.3% / 14.7%, 전문가 return 80% 필터
- 모델 크기(24층, 2048, FFN 8196), TPU v3 16×16, 1M step, 배치 512, 1024 토큰, 약 4일
- Atari 23개 / 11개, 전용 모델 44개, BabyAI BossLevel 75%, Meta-World 44/45, 35개, DM Control 21/30
- Skill Generalization 50.2% vs BC-IMP 49%, fine-tuning 10개 에피소드, 파란→초록 500개 시연(2시간 45분) 60% vs BC 0.5%(1/200)
- RTX 3090에서 20Hz 주기를 약 0.01초 초과, context 1
- 숙제 2의 계산: 128×128 카메라 두 대 → timestep당 이미지 토큰 128개 → 1,024 토큰에 8 timestep도 다 들어가지 않는다면, 볼 수 있는 과거는 0.4초보다 짧습니다. 그런데 본문은 "0.4초 남짓"(0.4초를 조금 넘는다는 뜻)이라고 써서 앞의 계산과 맞지 않습니다.

## 2026-10-20-act-aloha.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- ViperX 약 5,600달러, WidowX 약 3,300달러, 전체 약 1만 8천 달러(옵션 포함 2만 달러), 2시간 안에 조립
- 10배 넘게 비싼 상용 시스템 시연 15개 중 14개 재현, 고무줄로 30분 넘게 조작
- 결과 표(86/50, 32/20, 88, 96)와 Open Cup 84, Put On Shoe 92, Prep Tape 64, Thread Velcro 20(92 → 40 → 20)
- Thread Velcro를 "단계마다 성공률이 반씩 줄었다"고 썼지만, 92 → 40은 절반보다 더 많이 줄어든 값입니다(근사 표현).
- chunk 1% → 44%, temporal ensembling ACT 3.3%p, BC-ConvMLP 4%p, VINN 하락, CVAE 35.3% → 2%
- 사용자 연구: 6명, 33초 → 20초, 16초 → 10초, 62%, p < 0.001
- 사탕 포장 벗기기 0/10
- 하이퍼파라미터: 80M, RTX 2080 Ti 5시간, 0.01초, lr 1e-5, 배치 8, 512, 8 head, encoder 4층·decoder 7층, β = 10, z 32차원, m = 0.01

## 2026-10-19-vision-language-action-models.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 2~3장 Gato·RT-1·ACT 수치(개별 글과 같은 수치이므로 위의 검증 결과로 함께 확인)
- π0: RTX 4090에서 14 / 32 / 27 / 73ms, action expert 300M, 3.3B, H = 50, 10,000시간, 로봇 구성 7가지, 작업 68개, beta 분포, τ 표기
- Helix: 7B, 7~9Hz, 80M, 200Hz, 35자유도, 약 500시간, 5% 미만, 임베디드 GPU 2개
- GR00T N1: 10Hz / 120Hz, 12번째 층, 16 step 63.9ms(L40), 2.2B / 1.34B, 6,500시간 / 11시간, 88 → 827시간
- GR00T N1.5: 46.6% → 93.3%, 43.3% → 83.0%, VLM 고정, Eagle 2.5, FLARE, 36시간 vs 3개월
- SmolVLA: 450M / 100M, 앞 16개 층, 64 토큰, 10 step, 481개 데이터셋 / 22.9K / 10.6M, Qwen2.5-VL-3B, 13.75초 → 9.7초, 9 → 19
- Gemini Robotics 1.0: 160ms 미만, 약 250ms, 50Hz, ALOHA 2로 12개월 동안 수천 시간 수집, 시연 100개
- WALL-OSS: Qwen2.5-VL-3B, 정적 라우팅 FFN, path-drop, Inspiration / Integration, 1만 시간 이상
- π0.6 / π\*0.6: Gemma 3 4B, 860M, 50Hz, KI, 가치 함수 670M(Gemma 3), advantage 조건 30% 제거, AWR·PPO 비교, 처리량 2배 이상, 13시간, 2시간 넘게, 공장 상자

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| rt-1 | 데이터 규모(13대, 17개월, 13만, 700개 이상 지시, 처음 보는 지시 21개) | 문제 없음 | https://arxiv.org/html/2212.06817v2 | - |
| rt-1 | 2.4배 / 1.7배, 280ms, 100ms, 2~4초, 3Hz | 문제 없음 | https://arxiv.org/html/2212.06817v2 | - (HTML판에서 "33 Hz"로 보이는 것은 LaTeX 렌더링 중복이고 원문은 3Hz) |
| rt-1 | 일반화 표 성공률과 추론 시간(129 / 5.3 / 5.9 / 15ms), 1.2B 1.9초, 37M | 문제 없음 | https://arxiv.org/html/2212.06817v2 (Table 2, Table 13) | - |
| rt-1 | 현실 시나리오 70 / 30 / 45 / 55 | 문제 없음 | https://arxiv.org/html/2212.06817v2 (Table 3) | - |
| rt-1 | 시뮬레이션(92→90, 23→87, 7→33), Kuka(209k, 22→39, 0%) | 문제 없음 | https://arxiv.org/html/2212.06817v2 (Table 4, 5) | - |
| rt-1 | SayCan 87%, 67 / 67, Gato 0, BC-Z 13, 50단계 | 문제 없음 | https://arxiv.org/html/2212.06817v2 (Table 6) | - |
| rt-1 | 데이터 다양성 표 | 문제 없음 | https://arxiv.org/html/2212.06817v2 (Table 7) | - |
| rt-1 | 모델 ablation 표(성공률, 추론 시간) | 문제 없음 | https://arxiv.org/html/2212.06817v2 (Table 13) | - |
| rt-1 | 16M / 19M / 8층 / 35M, 3,000회 이상 | 문제 없음 | https://arxiv.org/html/2212.06817v2 | - |
| gato | 행동은 [-1, 1]이라 mu-law를 적용하지 않음 | 문제 없음 | https://arxiv.org/html/2205.06175v3 (Appendix B) | - |
| gato | prompt 25%, 절반은 에피소드 끝 | 문제 없음 | https://arxiv.org/html/2205.06175v3 | - |
| gato | 596개 작업, 63M, 1.5T, 85.3 / 14.7%, 80% 필터 | 문제 없음 | https://arxiv.org/html/2205.06175v3 | - |
| gato | 24층, 2048, 8196, TPU v3 16×16, 1M step, 512, 1024, 4일 | 문제 없음 | https://arxiv.org/html/2205.06175v3 | - (8196은 논문 표기 그대로) |
| gato | Atari, BabyAI, Meta-World, DM Control, 전용 모델 44개 | 문제 없음 | https://arxiv.org/html/2205.06175v3 | - |
| gato | 50.2% vs 49%, 10개 에피소드, 500개 시연(2시간 45분), 60% vs 0.5% | 문제 없음 | https://arxiv.org/html/2205.06175v3 | - |
| gato | RTX 3090, 약 0.01초 초과, context 1, 행동 토큰을 한 번에 | 문제 없음 | https://arxiv.org/html/2205.06175v3 (Appendix G) | - |
| gato | 숙제 2의 "0.4초 남짓"(128×128 카메라 두 대 → 8 timestep 미만) | 수정함 | https://arxiv.org/html/2205.06175v3 (카메라 두 대, 128×128, 20Hz 확인) + 본문 재계산 | "0.4초 남짓한 과거" → "0.4초도 안 되는 과거" |
| act-aloha | 부품 가격, 약 1만 8천 / 2만 달러, 2시간 조립, 15개 중 14개, 10배, 30분 | 문제 없음 | https://arxiv.org/html/2304.13705 | - |
| act-aloha | 결과 표와 실제 작업 4개, Thread Velcro 92 / 40 / 20 | 문제 없음 | https://arxiv.org/html/2304.13705 (Table I, II) | - ("반씩"은 근사 표현이라 그대로 둠) |
| act-aloha | chunk 1→44%, TE 3.3 / 4%p, VINN 하락, CVAE 35.3→2% | 문제 없음 | https://arxiv.org/html/2304.13705 | - |
| act-aloha | 사용자 연구(6명, 33→20초, 16→10초, 62%, p < 0.001) | 문제 없음 | https://arxiv.org/html/2304.13705 | - |
| act-aloha | 사탕 포장 0/10 | 문제 없음 | https://arxiv.org/html/2304.13705 (Appendix) | - (예비 실험 10회 결과와 일치. 논문은 이후 사탕 5개 실험에서 3/5 성공도 보고함) |
| act-aloha | 하이퍼파라미터, 80M, 5시간, 0.01초, w_0과 m의 의미 | 문제 없음 | https://arxiv.org/html/2304.13705 (Table III), https://github.com/tonyzhaozh/act | - |
| vision-language-action-models | Gato·RT-1·ACT 부분 수치 | 문제 없음 | 위 세 논문 URL | - |
| vision-language-action-models | π0 시간(14 / 32 / 27 / 73ms), 300M / 3.3B, 10,000시간 / 7 / 68, beta, τ 표기 | 문제 없음 | https://arxiv.org/html/2410.24164 | - |
| vision-language-action-models | Helix 수치 | 문제 없음 | https://www.figure.ai/news/helix | - |
| vision-language-action-models | GR00T N1 수치 | 문제 없음 | https://arxiv.org/html/2503.14734 | - |
| vision-language-action-models | GR00T N1.5 성공률, VLM 고정, Eagle 2.5, FLARE | 문제 없음 | https://research.nvidia.com/labs/gear/gr00t-n1_5/ | - |
| vision-language-action-models | GR00T-Dreams 36시간 vs 3개월 가까이 | 문제 없음 | https://tech.everyeye.it/notizie/gr00t-nvidia-evolve-sviluppato-36-ore-3-ia-801310.html (2차 보도, NVIDIA 공식 페이지는 찾지 못함) | - |
| vision-language-action-models | SmolVLA 수치 | 문제 없음 | https://arxiv.org/html/2506.01844 | - |
| vision-language-action-models | Gemini Robotics 1.0 지연·데이터·적응 | 문제 없음 | https://arxiv.org/html/2503.20020 | - |
| vision-language-action-models | WALL-OSS 구조·학습·1만 시간 이상 | 문제 없음 | https://arxiv.org/html/2509.11766 | - (논문에 "10,000시간 이상"과 "수만 시간"이 함께 있으나 "1만 시간 이상"은 둘 다와 맞음) |
| vision-language-action-models | π0.6 / π\*0.6 수치(4B, 860M, 670M, 30%, 13시간 등) | 문제 없음 | https://arxiv.org/html/2511.14759v2 | - |
