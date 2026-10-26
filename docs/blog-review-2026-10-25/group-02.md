# 블로그 교정 보고서: 그룹 2

대상 9개 파일, 수정 총 6건. 수정은 파일에 직접 반영했습니다.

## 2026-10-23-mappo.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| epoch을 줄이고, mini-batch를 | epoch를 줄이고, mini-batch를 | 통일 |

(글의 다른 곳에서는 모두 "epoch를/epoch는"으로 써서 조사를 맞췄습니다.)

### 내용 확인 필요

- 3절 SMAC 표 아래 "RODE와 같은 step 수에서 14개 맵 중 10개에서 대등 이상"이라고 썼는데, 바로 앞에서는 23개 맵이라고 했습니다. RODE가 보고한 맵이 14개인지 확인해 보세요.
- SMAC 표에서 RODE의 27m vs 30m과 3s5z vs 3s6z 값이 둘 다 96.8입니다. 옮겨 적다가 실수한 것은 아닌지 확인해 보세요.

## 2026-10-23-moonshine.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 오차가 1.26% 낮은 | 오차가 1.26%p 낮은 | 통일 |

(오차의 차이를 글의 다른 곳에서는 모두 %p로 적었습니다.)

### 내용 확인 필요

- 2절 "attention transfer가 KD보다 일관되게 좋았습니다", 5절 "모든 경우 AT로 학습한 학생이 … KD보다도 좋습니다", 7절 "AT가 KD를 일관되게 이겼고"와 표가 맞지 않습니다. 40-2 G(N)의 CIFAR-100은 KD 29.99, AT 30.06으로 AT가 더 나쁩니다.

## 2026-10-23-online-speculative-decoding.md

### 수정 목록

없음

### 내용 확인 필요

- 3절 "논문은 버리지 않고 갱신 때마다 비웠습니다"는 앞뒤가 모순되어 읽힙니다. "선별해서 버리지 않고 모두 쓴 뒤 비웠다"는 뜻이라면 그렇게 풀어 쓰는 것이 좋겠습니다.
- 3절에서 서빙에 "초당 약 5.5 GFLOPs만 쓴다"고 한 단위(GFLOPs)를 확인해 보세요. A100 8장의 용량(약 2,500 TFLOPs)과 비교하면 활용률이 1%보다 훨씬 낮은 값이 됩니다.

## 2026-10-23-paradigms.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 100 step DiffusionPolicy가 | 100 step Diffusion Policy가 | 통일 |

### 내용 확인 필요

- 서문의 DDPM 링크는 latent-diffusion-models 글로, DDIM·DPM-Solver 링크는 latent-consistency-model 글로 연결됩니다. 의도한 연결인지 확인해 보세요.
- 3절 "5행의 drift 계산 … 5~6행만 바꾸면 됩니다"의 행 번호가 위 의사코드와 맞지 않습니다. 의사코드에서 drift 계산은 4행입니다. 논문 Algorithm 1의 행 번호를 그대로 쓴 것일 수 있습니다.
- SD v2 표에서 DDIM 25의 모델 호출(25 → 93)이 DDIM 50(50 → 91)보다 많습니다. 옮겨 적은 값이 맞는지 확인해 보세요.
- 7절 "같은 저자들의 관심(Diffusion Policy, Stanford)": Diffusion Policy는 Chi et al.의 논문이라 ParaDiGMS의 저자와 다릅니다. 표현을 확인해 보세요.

## 2026-10-23-phasic-policy-gradient.md

### 수정 목록

없음

### 내용 확인 필요

없음

## 2026-10-23-pi-star-0-6.md

### 수정 목록

없음 (전체 374행을 모두 읽었습니다)

### 내용 확인 필요

- 3절 "off-policy로 학습하는 Q 함수보다 이론상 덜 최적이지만"에서 "덜 최적"이라는 표현이 어색합니다. 뜻을 바꿀 수 있어 손대지 않았습니다.

## 2026-10-23-rainbow.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 게임에 따라 도움도 해도 됨 | 게임에 따라 도움도 되고 해도 됨 | 비문 |

### 내용 확인 필요

- 서문의 DQN 링크가 ml-overview 글로 연결됩니다. 의도한 연결인지 확인해 보세요.

## 2026-10-23-self-attention-distillation.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| TuSimple과 BDD100K에서 넘고 CULane에서 | TuSimple과 BDD100K에서 앞서고 CULane에서 | 비문 |

("넘고"의 목적어가 빠져 있어서 고쳤습니다.)

### 내용 확인 필요

- 4절 "도입 시점: 언제 켜든 최종 성능은 거의 같지만 수렴 속도가 다릅니다"와 5절 "타이밍이 품질을 정합니다 … 절반 학습 뒤에 켜야 한다"가 어긋납니다. SRDL 글 5절도 이 결론을 인용합니다.
- 5절에서 EL-GAN의 수치를 가져왔다고 했지만, TuSimple 표에는 EL-GAN 행이 없습니다.

## 2026-10-23-self-referenced-deep-learning.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| (없으면 58% vs 72%) | (없으면 58% 대 72%) | 통일 |

(본문의 다른 곳에서는 모두 "대"로 썼습니다.)

### 내용 확인 필요

- 서문과 KD 표의 "157배": 표의 값으로 계산하면 12.70 / 0.08 ≈ 159입니다. 반올림한 값이라면 그대로 두어도 됩니다.

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거 | 수정 내용 |
|---|---|---|---|---|
| 2026-10-23-mappo.md | RODE "14개 맵 중 10개" vs 23개 맵 | 문제 없음 | 논문 Table 1에서 RODE는 23개 중 14개 맵만 보고("/"가 9개)하고, 본문에 "comparable or superior to RODE's in 10 of 14 maps"라고 씀. https://arxiv.org/html/2103.01955v4 | 없음 |
| 2026-10-23-mappo.md | RODE 27m vs 30m과 3s5z vs 3s6z가 둘 다 96.8 | 문제 없음 | 논문 Table 1/Table 5에 27m_vs_30m 96.8(1.5), 3s5z_vs_3s6z 96.8(25.11)로 실제로 같은 값. https://arxiv.org/html/2103.01955v4 | 없음 |
| 2026-10-23-moonshine.md | "AT가 KD보다 일관되게/모든 경우 좋다"와 40-2 G(N) CIFAR-100(KD 29.99, AT 30.06) 불일치 | 수정함 | 논문 Table 2에서도 G(N) CIFAR-100은 KD 29.99, AT 30.06으로 AT가 더 나쁨. 논문의 "In all cases"는 Scr 대비 주장이고 KD 대비는 "substantially better"라고만 함. https://arxiv.org/abs/1711.02613 (PDF) | 2절 "일관되게" → "거의 모든 경우", 5절에 예외 한 건(G(N) CIFAR-100) 명시, 7절 "일관되게 이겼고" → "거의 언제나 이겼고" |
| 2026-10-23-online-speculative-decoding.md | "버리지 않고 갱신 때마다 비웠습니다" 모순처럼 읽힘 | 수정함 | 논문: "we refrain from evicting any pairs and release Q after each model update". 뜻은 맞지만 문장이 모호함. https://arxiv.org/html/2310.07177 | "중간에 아무 쌍도 내보내지(evict) 않고 모두 쌓아 두었다가 갱신할 때마다 버퍼를 통째로 비웠습니다"로 풀어 씀 |
| 2026-10-23-online-speculative-decoding.md | "초당 약 5.5 GFLOPs" 단위 | 문제 없음 | 논문 부록 A.4가 5.5×10⁹ FLOPs(5.5 GFLOPs)와 8×312 TFLOPs를 그대로 제시하고, 4.2.2절에 활용률 1% 미만이라고 씀. 글은 논문 수치를 옮긴 것. https://arxiv.org/html/2310.07177 | 없음 |
| 2026-10-23-paradigms.md | DDPM 링크(latent-diffusion-models), DDIM·DPM-Solver 링크(latent-consistency-model) | 문제 없음 | DDPM 전용 글은 _posts에 없음. LDM 글은 픽셀 공간 diffusion과 Ho et al.(DDPM)을 다루고, LCM 글은 DDIM·DPM-Solver를 "더 좋은 ODE solver"로 설명하므로 연결이 성립함. 2026-10-24-diffusion-survey가 DDPM을 더 직접 다루니 바꿀지는 선택 사항 | 없음 |
| 2026-10-23-paradigms.md | "5행의 drift 계산 … 5~6행만 바꾸면" 행 번호 | 수정함 | 논문 Algorithm 1에서는 drift가 5행, Euler 이산화가 5~6행("swapping out the Euler discretization in Lines 5-6")이지만, 글의 축약 의사코드에서는 4행(drift), 5행(Picard 갱신). https://arxiv.org/html/2305.16317 | "4행의 drift 계산", "4~5행(논문 Algorithm 1의 5~6행)" |
| 2026-10-23-paradigms.md | SD v2 표에서 DDIM 25 호출(25 → 93)이 DDIM 50(50 → 91)보다 많음 | 문제 없음 | 논문 표에 DDIM 50은 91회/17 iters(τ=5e-2), DDIM 25는 93회/17 iters(τ=1e-2)로 그대로 나옴. DDIM 25에 더 엄격한 허용 오차를 써서 생긴 차이. https://arxiv.org/html/2305.16317 | 없음 |
| 2026-10-23-paradigms.md | "같은 저자들의 관심(Diffusion Policy, Stanford)" | 수정함 | Diffusion Policy는 Chi, Xu, Feng, Cousineau, Du, Burchfiel, Tedrake, Song(Columbia·TRI·MIT)의 논문이고, ParaDiGMS 저자(Shih, Belkhale, Ermon, Sadigh, Anari)는 모두 Stanford. https://arxiv.org/abs/2303.04137 , https://arxiv.org/html/2305.16317 | "로봇 학습을 연구하는 저자들(Stanford)이 다른 연구진의 Diffusion Policy로 실험한 이유도 여기에 있습니다" |
| 2026-10-23-pi-star-0-6.md | "이론상 덜 최적이지만" 표현 | 문제 없음 | 논문 원문이 "less optimal than a more classic off-policy Q-function estimator"라 내용은 원문과 일치. 표현만 어색하고 사실 오류는 아니어서 그대로 둠. https://arxiv.org/html/2511.14759 | 없음 |
| 2026-10-23-rainbow.md | 서문 DQN 링크가 ml-overview로 연결 | 수정함 | DQN 전용 글 _posts/2026-10-24-dqn.md가 있음(Rainbow 글보다 나중에 작성됨). ml-overview는 개요 글 | `2026-05-05-ml-overview` → `2026-10-24-dqn` |
| 2026-10-23-self-attention-distillation.md | 4절 "도입 시점은 최종 성능에 거의 무관, 수렴 속도만 다름" vs 5절 "타이밍이 품질을 정합니다 … 절반 학습 뒤에 켜야 한다" | 수정함 | 논문 4.2절 "When to add SAD": 도입 시점이 달라도 최종 성능은 거의 같고 수렴 속도에만 영향을 줌. 4절이 맞고 5절이 과장. https://arxiv.org/abs/1908.00821 (PDF) | 5절 3항을 "타이밍이 수렴 속도를 정합니다. 최종 성능은 비슷해도, 깊은 층이 어느 정도 학습된 뒤(논문은 절반 학습)에 켜야 효과가 빨리 난다는 관찰은 …"로 고침. 이 결론을 인용하는 SRDL 글 5절 문장도 맞춰 고침 |
| 2026-10-23-self-attention-distillation.md | EL-GAN 수치를 가져왔다는데 TuSimple 표에 EL-GAN 행이 없음 | 수정함 | 논문 Table 2에 EL-GAN 96.39% / FP 0.0412 / FN 0.0336이 있고, "LaneNet과 EL-GAN은 코드가 없어 논문 수치를 썼다"는 문장도 있음. 글의 표에서 행이 빠진 것. https://arxiv.org/abs/1908.00821 (PDF) | TuSimple 표에 `EL-GAN \| 96.39% \| 0.0412 \| 0.0336` 행 추가 |
| 2026-10-23-self-referenced-deep-learning.md | "157배" vs 12.70/0.08 ≈ 159 | 수정함 | 논문은 "at the price of extra 157×"라고 쓰고, 계산을 12.70/0.08 − 1로 밝힘. 157은 총 비율이 아니라 추가 비용 배수(정확히는 157.75). https://arxiv.org/html/1811.07598v1 | 표 "(157배)" → "(추가 157배)", 표 아래 문장 "비용이 157배" → "추가 비용이 157배입니다(논문의 계산은 12.70/0.08 − 1)". 서문의 "157배 늘어납니다"는 추가 비용이라는 뜻으로 읽혀 그대로 둠 |
