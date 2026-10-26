# 블로그 교정 보고서: 그룹 3

대상 9개 파일, 수정 총 6건. 9개 파일 모두 교정을 끝냈습니다.

## 2026-10-23-speculative-decoding.md

### 수정 목록

수정 없음

### 내용 확인 필요

- 7절 해석 4번: "수락률의 상한이 $$1/(1 - \alpha)$$라는 점을 생각하면": $$1/(1-\alpha)$$는 수락률의 상한이 아니라 호출당 토큰 수(=$$c = 0$$일 때 속도 향상)의 상한입니다. 4절에서도 "상한은 $$1/(1 - \alpha) \approx 4$$배"라고 썼으므로 "속도 향상의 상한"을 뜻한 것으로 보입니다.
- 8절 정리표 "T5-XXL에서 2.6~3.4배(번역)": 5절 표의 EnDe 행에는 T5-large 1.7배, T5-base 2.8배도 있어 범위가 1.7~3.4배입니다. T5-small(온도 0/1)만 기준으로 잡은 범위라면 그렇다고 밝히는 편이 좋겠습니다.

## 2026-10-23-speculative-sampling.md

### 수정 목록

수정 없음

### 내용 확인 필요

- "SpS"라는 약어가 1절 끝("SpS의 초당 토큰 수는")에서 정의 없이 처음 나옵니다. "ArS"도 5절 표에만 나오고 정의가 없습니다(SpS = Speculative Sampling, ArS = autoregressive sampling으로 보입니다).

## 2026-10-23-speed-is-all-you-need.md

### 수정 목록

수정 없음

### 내용 확인 필요

- 4절 결과: "가장 큰 단일 효과는 Winograd(S23에서 660→525ms)": 표로 계산하면 Winograd는 -135ms인데, Opt. Softmax는 1098→908(-190ms), S-GN/GELU는 908→755(-153ms)로 더 큽니다. iPhone에서도 Winograd가 -266ms로 가장 크긴 하지만 S23 기준 문장으로는 맞지 않습니다.
- 3절 코드 설명: "논문 Table 2의 2×2 행과 정확히 같은 숫자": 코드 출력은 1.78배, 본문 표의 2×2 가중치 메모리는 1.77배입니다(16/9 = 1.777…를 반올림했느냐 버렸느냐의 차이).

## 2026-10-23-stealing-production-lm.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 실제로는 더 많은 질의와 낮은 정밀도를 냈습니다. | 실제로는 질의가 더 많이 들고 정밀도도 낮았습니다. | 어색한 표현 |

### 내용 확인 필요

- 절 번호가 섞여 있습니다. 1절 표("모든 logit (4절)", "(5절)", "(6절)"), 4절 표("logprob-4 (5절)"), 코드 주석("4절의 '모든 logit' API", "5절의 실제에 가까운 API")은 논문의 절 번호를 쓰는데, 이 글에서 해당 내용은 2·3·4절입니다. 반면 3절의 "2절 공격을 그대로 적용"은 이 글의 절 번호입니다. 코드 블록과 표 안 표기여서 고치지 않았습니다.
- 4절 Hyperrectangle: "all-pairs shortest path로 바꿀 수 있어 Bellman-Ford로 $$O(N^3)$$에 풉니다": all-pairs 최단 경로를 $$O(N^3)$$에 푸는 대표 알고리즘은 Floyd-Warshall입니다. 논문 원문 표기를 확인해 볼 만합니다.

## 2026-10-23-t-stitch.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| DPM-Solver++ 모두 초기 구간에서 일관된 이득. | DPM-Solver++ 모두 초기 구간에서 일관된 이득을 보였습니다. | 통일 |

### 내용 확인 필요

- 4절 DiT: "약 10초 예산에서 FID 9.21(원래 9.19)": 바로 위 표에서는 DiT-XL 단독(0%) FID가 9.20입니다.
- 4절 DiT: "DiT-S(… 8장에 1.6초)"인데 표의 100% 행 시간은 1.7초입니다(사소한 차이).
- 소속 표기 순서가 description("NVIDIA·Monash")과 본문 첫 문단("Monash·NVIDIA")에서 다릅니다(front matter라 고치지 않음).

## 2026-10-23-watermark-reliability.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 텍스트 길이 $$T$$가 결정적이므로 | 텍스트 길이 $$T$$가 결과를 좌우하므로 | 어색한 표현 |

### 내용 확인 필요

- Copy-Paste 결과가 앞뒤로 어긋납니다. 4절에서는 "25%가 남은 경우 … 200토큰에서 AUC 0.7 미만, 600토큰에서 0.85 미만"이라고 했는데, 9절 정리표에는 "CP 25%는 AUC 0.95 이상"이라고 적혀 있습니다.
- 4절 사람 실험: "약 800토큰(500단어)"과 "약 250토큰(200단어)"의 토큰:단어 비율이 다릅니다(1.6 대 1.25). 논문 수치를 확인해 볼 만합니다.

## 2026-10-24-adain.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 디코더에 정규화를 빼는 선택이 | 디코더에서 정규화를 빼는 선택이 | 비문(조사) |
| 짧은 변을 512로 리사이즈 후 256 크롭, Adam, 배치 8쌍. | 짧은 변을 512로 리사이즈 후 256 크롭, Adam, 배치 8쌍으로 학습합니다. | 비문 |

### 내용 확인 필요

- 2절 런타임 제어: 네 가지 제어 중 "색 보존"만 굵게 표시되지 않았습니다(마크다운 서식이라 고치지 않음).

## 2026-10-24-adaptive-alternate-analog-test.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 생기므로, 필터가 꼭 필요함을 보여 줍니다. | 생기므로, 필터가 꼭 필요함을 알 수 있습니다. | 비문 |

### 내용 확인 필요

- 2절 Epanechnikov 커널 수식: `t^\top t < 1 \ 0 & \text{otherwise}`에서 cases 줄바꿈 `\\`가 `\ `(공백)로 되어 있어, 두 경우가 한 줄로 렌더링될 것으로 보입니다. 수식이라 고치지 않았습니다.
- 4절 회로와 측정: 대체 테스트는 "480kHz로 샘플링", 표준 EVM 테스트는 "100kHz 샘플링이 필요"라고 되어 있습니다. 표준 테스트가 더 비싸다는 맥락과 맞지 않아 단위(예: MHz)를 확인해 볼 만합니다.
- 4절 선택 예: HD 비율이 86.11% + 5.28% + 8.5% = 99.89%로 100%가 되지 않습니다(8.61%의 오기일 수 있음).

## 2026-10-24-albert.md

### 수정 목록

수정 없음

### 내용 확인 필요

- 5절 해석 2번: "같은 파라미터 수에서 공유 없는 모델보다 성능이 낮은 것도 사실입니다(80.1 대 81.6)": 비교 대상인 공유 없는 ALBERT-base는 약 89M, 전체 공유는 12M이어서 파라미터 수가 같지 않습니다. "같은 구성(H, 층 수)에서"를 뜻한 것으로 보입니다.

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거 | 수정 내용 |
|---|---|---|---|---|
| 2026-10-23-speculative-decoding.md | 7절 해석 4번 "수락률의 상한 1/(1-α)" | 수정함 | 논문은 oracle γ일 때 기대 생성 토큰 수가 1/(1-α)라고 씀. 수락률이 아니라 호출당 토큰 수(c = 0일 때 속도 향상)의 상한. https://arxiv.org/abs/2211.17192 (PDF 3.3절) | "수락률의 상한" → "속도 향상의 상한" |
| 2026-10-23-speculative-decoding.md | 8절 정리표 "T5-XXL에서 2.6~3.4배(번역), 2.3~3.1배(요약)" | 수정함 | 논문 Table 2에서 이 범위는 T5-small(★, 최적) 행의 값이고, T5-large는 1.4~2.2배까지 내려감. https://arxiv.org/abs/2211.17192 | "T5-XXL에서" → "T5-XXL에 T5-small을 쓸 때" |
| 2026-10-23-speculative-sampling.md | SpS·ArS 약어 정의 없음 | 수정함 | 논문이 "speculative sampling (SpS)", "auto-regressive sampling (ArS)"로 정의해 씀. https://arxiv.org/abs/2302.01318 | 1절에 "autoregressive 샘플링(ArS)", "speculative sampling(SpS)"로 첫 등장 시 정의 |
| 2026-10-23-speed-is-all-you-need.md | 4절 "가장 큰 단일 효과는 Winograd(S23 660→525ms)" | 수정함 | Table 1 기준 S23에서는 Opt. Softmax(-190ms)가 가장 크고, iPhone에서는 Winograd(-266ms)가 가장 큼. https://arxiv.org/html/2304.11267 | 기기별로 가장 큰 단계를 밝히고 Winograd의 S23 감소폭은 별도 문장으로 유지 |
| 2026-10-23-speed-is-all-you-need.md | 3절 "Table 2의 2×2 행과 정확히 같은 숫자"(1.78 대 1.77) | 수정함 | 논문 Table 2의 2×2 가중치 메모리는 1.77x(16/9 = 1.777…). 코드 출력 1.78배는 그대로 둠. https://arxiv.org/html/2304.11267 | "정확히 같은 숫자" → "같은 값입니다(논문은 16/9 = 1.777…을 1.77배로 적었습니다)" |
| 2026-10-23-stealing-production-lm.md | 절 번호 혼용(논문 절 번호 4·5·6절) | 수정함 | 논문 4절 logit-vector, 5절 logit-bias, 6절 logprob-free = 이 글의 2·3·4절. https://arxiv.org/html/2403.06634 | 1절 표의 (4절)(5절)(6절) → (2절)(3절)(4절), 4절 표의 "logprob-4 (5절)" → "(3절)". 코드 주석의 "4절", "5절"은 코드 블록이라 그대로 둠 |
| 2026-10-23-stealing-production-lm.md | Hyperrectangle "Bellman-Ford로 O(N^3)" | 문제 없음 | 논문 부록 F.2.1(Lemma F.3) 원문이 "using the Bellman-Ford algorithm in O(N^3) time"이라고 씀. https://arxiv.org/html/2403.06634 | 없음 |
| 2026-10-23-t-stitch.md | 4절 "FID 9.21(원래 9.19)" 대 표의 9.20 | 문제 없음 | 4.1절 원문이 "comparable FID (9.21 vs. 9.19)", IS 243.82 vs. 245.73이라고 씀. Table 9의 9.20과는 다른 비교에서 나온 논문 자체 수치. https://arxiv.org/html/2402.14167 | 없음 |
| 2026-10-23-t-stitch.md | DiT-S "8장에 1.6초" 대 표 100% 행 1.7초 | 문제 없음 | 본문의 1.6초는 논문 Table 3의 DiT-S time cost(1.6s)와 일치. 표 시간 행(1.7초)은 논문 표에 없어 출처를 따로 확인하지는 못함. https://arxiv.org/html/2402.14167 | 없음 |
| 2026-10-23-t-stitch.md | 소속 순서 description "NVIDIA·Monash" 대 본문 "Monash·NVIDIA" | 수정함 | 제1저자 Zizheng Pan과 Bohan Zhuang이 Monash, 그다음이 NVIDIA. https://arxiv.org/html/2402.14167 | description을 "Monash·NVIDIA"로 통일(콜론 없음, 따옴표 불필요) |
| 2026-10-23-watermark-reliability.md | CP 25%: 4절 "600토큰에서 0.85 미만" 대 9절 "AUC 0.95 이상" | 확인 불가 | 논문 자체가 다름. 서론은 600토큰 중 150토큰 삽입 시 "AUC for detection is above 0.95", 4.2절은 25% 남은 경우 "below 0.7 for 200 tokens and below 0.85 for 600 tokens". 어느 설정·검출기(WinMax 여부)의 수치인지 본문에 밝혀져 있지 않음. https://arxiv.org/html/2306.04634 | 없음 |
| 2026-10-23-watermark-reliability.md | 사람 실험 "800토큰(500단어)", "250토큰(200단어)" 비율 | 문제 없음 | 부록 A.13.1 원문이 "about 800 tokens, i.e. about 500 words", "about 250 tokens, or 200 words"로 씀. https://arxiv.org/html/2306.04634 | 없음 |
| 2026-10-24-adain.md | 2절 런타임 제어에서 "색 보존"만 굵게 표시 안 됨 | 수정함 | 나머지 세 제어와 서식 통일(내용은 논문과 일치: 스타일 이미지의 색을 콘텐츠에 맞춤). https://arxiv.org/abs/1703.06868 | "색 보존" → "**색 보존**" |
| 2026-10-24-adaptive-alternate-analog-test.md | Epanechnikov 커널 cases의 `\ ` 줄바꿈 | 수정함 | cases 안의 줄바꿈은 `\`여야 함(다른 글의 cases 표기와도 같음). 글 자체 확인 | `t^\top t < 1 \ 0` → `t^\top t < 1 \ 0` |
| 2026-10-24-adaptive-alternate-analog-test.md | 대체 테스트 480kHz 대 표준 EVM 100kHz 샘플링 | 확인 불가 | 원문(IEEE D&T 2012) 전문을 열지 못함. HAL(https://hal.science/hal-00743573) 접근 차단, ResearchGate 403 | 없음 |
| 2026-10-24-adaptive-alternate-analog-test.md | HD 86.11% + 5.28% + 8.5% = 99.89% | 확인 불가 | 글 안에서 다시 계산하면 100%가 안 되고(EPV는 3.11 + 20.78 + 76.11 = 100), 8.61%의 오기일 가능성이 큼. 다만 원문을 열지 못해 어느 숫자가 틀렸는지 확정할 수 없음. https://hal.science/hal-00743573 | 없음 |
| 2026-10-24-albert.md | 5절 해석 2번 "같은 파라미터 수에서 … (80.1 대 81.6)" | 수정함 | Table 4(ALBERT-base, E = 128): all-shared 12M 80.1, not-shared 89M 81.6. 파라미터 수가 다름. https://arxiv.org/abs/1909.11942 | "같은 파라미터 수에서" → "같은 구성(ALBERT-base, E = 128)에서", 괄호에 파라미터 수(12M, 89M) 추가 |
