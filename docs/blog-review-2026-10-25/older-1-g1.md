# 블로그 교정·사실 확인 (older-1-g1)

대상: `_posts/` 아래 5개 글 (2026-10-20 ~ 2026-10-21, VLA 시리즈). 전체적으로 문장이 이미 깔끔해 교정 항목은 적습니다. 코드 블록의 출력(```text)은 직접 다시 계산해 본문과 맞는지 확인했습니다(π0 계산량 2.42e+13 → 2.55e+12, 9.5배 / OpenVLA LoRA 262,144 / 17,039,360 = 1.5% / GR00T 몸별 파라미터 23,084·4,103·16,928, 공유 526,336 / RT-2 bin 변환 결과, 모두 일치). `{% post_url %}` 대상 12개는 모두 `_posts/`에 있습니다.

## 2026-10-21-pi0.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 다만 행동 expert의 기울기는 | 다만 action expert의 기울기는 | 통일 |

### 내용 확인 필요

- RTX 4090 추론 시간 내역(이미지 인코딩 14ms, 관측 처리 32ms, flow 10 step 27ms, 합계 73ms)
- "모든 작업에서 최대 점수의 50% 이상", "사전학습한 π0가 처음부터 학습한 것보다 최대 2배"
- 데이터 수치(9억 300만 timestep = 단팔 1억 600만 + 양팔 7억 9,700만, 공개 데이터 9.1%, 68개 작업, 7가지 로봇 구성, $$n^{0.43}$$), 학습 step(700k, 비교용 160k), π0-small 470M, action expert 폭 1024, 상태·행동 18차원과 구성

## 2026-10-21-openvla.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- "OpenVLA가 7배 작은 RT-2-X보다 16.5%p 높습니다": RT-2-X(55B)가 OpenVLA(7B)보다 7배 크므로 크기 관계가 거꾸로 쓰인 것으로 보임
- 비전 인코더 고정 vs fine-tuning 부록 수치(85% → 95%, 25% → 90%)
- "무엇이 성능을 만들었나" 표(76.3 / 45.6 / 40.6)와 30%p·5%p 차이
- 8bit 1.2Hz, 4bit 3Hz(A5000), backbone 비교(LLaVA가 IDEFICS-1보다 35%p, Prismatic이 약 10%p), 양자화 표 수치

## 2026-10-21-gr00t-n1.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 도입부 "논문이 없는 후속작 N1.5는 6장에서 함께 다룹니다": N1.5는 실제로 "## 5. 이후: GR00T N1.5"(5장)에서 다룸 (본문 내부 불일치)
- GR-1 원격조작 장비(VIVE 트래커, Xsens 장갑)와 20Hz, 수도꼭지 작업 11.8% vs 42.2%, 실제 GR-1 표의 수치와 평균
- 생성 영상 co-training 향상폭(RoboCasa 4.2~8.8%p, 실제 GR-1 5.8%p), 데이터 피라미드 수치(54만, 78만 궤적, 6,500시간, 11시간, 88 → 827시간, L40 3,600장, 약 10만 5천 GPU-시간, H100 약 5만 GPU-시간)
- N1.5 수치(grounding IoU 40.4 vs 35.5, FLARE 가중치 0.2, H100 1천 장·25만 step·배치 16,384, N1 → N1.5 결과 표, 처음 보는 물체 0% → 15% → 55%), DreamGen "36시간/3개월" 발표

## 2026-10-21-gemini-robotics.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- ERQA 표 수치(400문제, 28% 여러 이미지)
- 코드 생성 27% → 53%(7개 작업), in-context learning 51%/65%(실제 로봇 65%), 옷 접기 0%, "논문 그림 12"
- 지연(160ms 미만, 약 250ms, 50Hz), 특화 79%·도시락 100%·그림 6개 중 4개, 빠른 적응 8개 중 7개 70% 이상·2개 100%, 양팔 Franka 63%, 일반화 85개 작업 구성 비율, 편향 질의 거부율 20% → 96%

## 2026-10-20-rt-2.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 당시 closed-loop 로봇 제어에 쓰인 모델 중 10배 이상 큰 모델이었습니다 | 당시 closed-loop 로봇 제어에 쓰인 어떤 모델보다도 10배 이상 큰 모델이었습니다 | 비문 |

### 내용 확인 필요

- "크기와 학습 방식" 표의 55B co-fine-tuning 새 상황 평균 63과 "일반화" 표의 RT-2-PaLI-X-55B 평균 62가 다름
- 수학 지시에서 PaLM-E 35% vs PaLI-X 25%
- 55B 1~3Hz, 5B 약 5Hz, Language-Table 90%(기존 72~77%), 약 6,000번 평가, chain-of-thought "몇백 step"

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| 2026-10-21-pi0.md | RTX 4090 추론 시간 14/32/27ms, 합계 73ms | 문제 없음 | https://arxiv.org/html/2410.24164 | - (논문: 온보드 73ms, 오프보드 86ms) |
| 2026-10-21-pi0.md | 50% 이상, 최대 2배 | 문제 없음 | https://arxiv.org/html/2410.24164 | - ("more than 50% of the maximum score across all of the tasks", "as much as 2x") |
| 2026-10-21-pi0.md | 데이터·학습 수치, π0-small, action expert 폭, 18차원 | 문제 없음 | https://arxiv.org/html/2410.24164 | - |
| 2026-10-21-openvla.md | "7배 작은 RT-2-X" | 수정함 | https://arxiv.org/html/2406.09246 | 논문은 OpenVLA가 RT-2-X(55B)보다 "7x fewer parameters"라고 씀. "7배 작은 RT-2-X보다" → "7배 큰 RT-2-X보다" |
| 2026-10-21-openvla.md | 비전 인코더 고정 vs fine-tuning 85→95, 25→90 | 문제 없음 | https://arxiv.org/html/2406.09246 | - (부록 Table 10, SigLIP ViT-SO와 LLaVA v1.5 7B 재현) |
| 2026-10-21-openvla.md | 76.3 / 45.6 / 40.6, 30%p·5%p | 문제 없음 | https://arxiv.org/html/2406.09246 | - (부록 Table 9; 76.3−45.6=30.7, 45.6−40.6=5.0) |
| 2026-10-21-openvla.md | 1.2Hz/3Hz, backbone 35%p·약 10%p, 양자화 표 | 문제 없음 | https://arxiv.org/html/2406.09246 | - |
| 2026-10-21-gr00t-n1.md | N1.5를 "6장"에서 다룬다는 도입부 | 수정함 | (본문 내부: N1.5 절은 "## 5.") | "6장에서" → "5장에서" |
| 2026-10-21-gr00t-n1.md | 원격조작 장비·20Hz, 11.8% vs 42.2%, 실제 GR-1 표 | 문제 없음 | https://arxiv.org/html/2503.14734 | - |
| 2026-10-21-gr00t-n1.md | co-training 향상폭, 데이터 피라미드 수치, 사전학습 GPU-시간 | 문제 없음 | https://arxiv.org/html/2503.14734 | - (RoboCasa +4.2/+8.8/+6.8%, GR-1 +5.8%) |
| 2026-10-21-gr00t-n1.md | N1.5 수치(IoU, FLARE 0.2, 학습 설정, 결과 표, 0→15→55%) | 문제 없음 | https://research.nvidia.com/labs/gear/gr00t-n1_5/ | - (페이지 날짜 2025-06-11) |
| 2026-10-21-gr00t-n1.md | DreamGen "3개월 가까이 걸릴 데이터를 36시간" | 문제 없음 | https://www.androidheadlines.com/2025/05/best-of-computex-2025-nvidia-gr00t-n1-5-humanoid-robots.html | - (Computex 2025 발표 보도) |
| 2026-10-21-gemini-robotics.md | ERQA 표, 400문제·28% | 문제 없음 | https://arxiv.org/html/2503.20020 | - |
| 2026-10-21-gemini-robotics.md | 코드 생성 27→53%, ICL 51/65%, 옷 접기 0%, 그림 12 | 문제 없음 | https://arxiv.org/html/2503.20020 | - (그림 12: perception·control API 개요) |
| 2026-10-21-gemini-robotics.md | 지연·특화·적응·일반화 비율·거부율 수치 | 문제 없음 | https://arxiv.org/html/2503.20020 | - |
| 2026-10-20-rt-2.md | 55B co-fine-tuning 63 vs 일반화 표 62 | 문제 없음 | https://arxiv.org/html/2307.15818 | - (논문의 두 표 자체가 62와 63으로 다름. 글은 각 표를 그대로 옮김) |
| 2026-10-20-rt-2.md | 수학 지시 35% vs 25% | 문제 없음 | https://arxiv.org/html/2307.15818 | - |
| 2026-10-20-rt-2.md | 1~3Hz/약 5Hz, Language-Table 90%(72~77%), 약 6,000번, 몇백 step, "10배 이상 큰 모델" | 문제 없음 | https://arxiv.org/html/2307.15818 | - ("largest ever, by over an order of magnitude") |
