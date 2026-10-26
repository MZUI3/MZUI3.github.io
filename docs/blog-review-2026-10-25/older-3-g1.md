# 블로그 교정 보고서: older-3-g1

대상 5개 파일(_posts/). 교정 원칙: 맞춤법·띄어쓰기·문장부호·비문·번역투·같은 글 안의 표기 통일만 고치고, 내용·수치·순서·소제목·말투·코드 블록·인라인 코드·수식·링크·Liquid·front matter·기술 용어·고유명사는 그대로 둡니다. 애매하면 고치지 않고, 내용이 의심스러운 부분은 "내용 확인 필요"에 적은 뒤 원문(논문·공식 문서)과 대조해 확인된 오류만 최소 수정했습니다. 포스트 안 코드(PatchCore, FQ-ViT, AdderNet)는 로컬(torch 2.13 CPU)에서 다시 실행해 주석의 출력값과 일치함을 확인했습니다.

## 2026-07-30-patchcore.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 패치별 점수를 모으면 \*\*결함 위치 지도(anomaly map)\*\*, 그중 최댓값이 | 패치별 점수를 모으면 \*\*결함 위치 지도(anomaly map)\*\*가 되고, 그중 최댓값이 | 비문 |
| 메모리 뱅크에 저장된 정상 패치와 \*\*최소 거리\*\*가 멀수록 | 메모리 뱅크에 저장된 정상 패치와의 \*\*최소 거리\*\*가 멀수록 | 비문(조사) |

### 내용 확인 필요

- 이미지 점수 re-weighting 설명 "주변에 다른 정상 패치가 드문 경우는 원래 드문 정상 패턴일 수 있기 때문입니다."가 점수를 낮추는 쪽으로 읽힘. 논문의 보정 방향 확인 필요.
- "이미지 200장, 이미지당 28×28 패치라면 약 15만 개" (200×784 = 156,800).
- 이미지 단위 AUROC 99.6%, few-shot 설정에서도 강하다는 서술.

## 2026-07-28-asip-posit.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- References의 "ASIP Designer 제품 소개" 링크(mips.com)가 실제 ASIP Designer 페이지인지.

## 2026-07-26-latent-consistency-model.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- Skipping-step 예시 $$k = 20$$, 학습 비용 "A100 GPU 약 32시간".
- 실험: LAION-5B-Aesthetics, 512·768 해상도, 1·2·4 step에서 DDIM·DPM-Solver·DPM-Solver++·Guided-Distill 대비 FID·CLIP score 우위, LCF의 포켓몬·심슨 데이터셋.
- CFG 식 $$(1+\omega)\epsilon_c - \omega\epsilon_\varnothing$$이 LCM 논문 표기인지.
- "LCM 파이프라인은 diffusers 0.22 이후 버전에 포함".

## 2026-07-24-addernet.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- ImageNet ResNet-50 top-1 74.9%(CNN 76.2%), BNN보다 훨씬 높다는 서술.
- Adaptive learning rate 식 $$\alpha_l = \eta\sqrt{k}/\lVert\Delta L(F_l)\rVert_2$$와 $$k$$의 의미, HardTanh 범위.
- SCALE-Sim "(2018, 현재 V2)": 이후 버전이 나왔는지.
- "알고리즘과 API를 `.cu`에, 배열 처리를 `.cpp`에" 역할 분담(PyTorch 튜토리얼의 일반적 구성과 다름).

## 2026-07-21-fq-vit.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| \*\*Softmax를 지나려면 다시 FP32로 역양자화\*\*했다가 다시 양자화해야 | \*\*Softmax를 지나려면 FP32로 역양자화\*\*했다가 다시 양자화해야 | 어색한 표현(반복) |
| 정확도 손실이 대부분 \*\*1%p 안팎\*\*으로 거의 손실이 없었습니다. | 정확도 손실이 대부분 \*\*1%p 안팎\*\*으로 거의 없었습니다. | 어색한 표현(반복) |

### 내용 확인 필요

- "Transformer 계열은 채널 간 범위가 \*\*최대 약 40배\*\*까지 차이 남": 논문의 비교 기준(최솟값 대비인지 중앙값 대비인지).
- ViT-L attention 값의 98.8%가 1/16 미만, DeiT-T 4bit 균일 양자화 시 8.69%(약 63%p 하락).
- 완전 양자화 시 "DeiT는 비교적 덜, ViT는 크게, Swin은 중간" 하락.
- W8/A8/Attn4 결과의 "대부분 1%p 안팎" 손실, COCO 검출 실험 유무, IJCAI 2022.

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| 2026-07-30-patchcore.md | re-weighting 방향 | 수정함 | https://arxiv.org/abs/2106.08265 (Eq. 7: 최근접 메모리 특징이 이웃과 멀어 이미 드문 정상 패턴이면 "we increase the anomaly score") | "주변에 다른 정상 패치가 드문 경우는 원래 드문 정상 패턴일 수 있기 때문입니다." → "주변에 다른 정상 패치가 드물면, 즉 그 정상 패치 자체가 이미 드문 정상 패턴이면 점수를 높입니다." |
| 2026-07-30-patchcore.md | 200장×784 패치 ≈ 15만 | 문제 없음 | 본문 재계산(156,800, 어림값) | - |
| 2026-07-30-patchcore.md | AUROC 99.6%, few-shot | 문제 없음 | https://arxiv.org/abs/2106.08265 (abstract: "up to 99.6%", "competitive results in the few samples regime") | - |
| 2026-07-30-patchcore.md | 코드 출력 | 문제 없음 | 로컬 재실행: torch.Size([156, 64]) (10, 5) / 48.59 9.93 | - |
| 2026-07-28-asip-posit.md | mips.com 링크 | 문제 없음 | https://mips.com/processor-solutions/asips-tools/ ("ASIP Designer™ is the leading tool solution for creating ASIPs", © MIPS) | - |
| 2026-07-26-latent-consistency-model.md | k=20, 32 A100 GPU hours | 문제 없음 | https://arxiv.org/abs/2310.04378 ("we set k=20", "only 32 A100 GPU hours") | - |
| 2026-07-26-latent-consistency-model.md | 실험 설정·결과, LCF 데이터셋 | 문제 없음 | https://arxiv.org/abs/2310.04378 (512·768, DDIM/DPM/DPM++/Guided-Distill, Table 1 caption "significantly surpasses baselines in the 1-4 step region", Pokemon·Simpsons) | - |
| 2026-07-26-latent-consistency-model.md | CFG 식 | 문제 없음 | https://arxiv.org/abs/2310.04378 (ε̃ = (1+ω)ε(z_t,c,t) − ωε(z_t,∅,t)) | - |
| 2026-07-26-latent-consistency-model.md | diffusers 0.22 | 문제 없음 | https://newreleases.io/project/pypi/diffusers/release/0.22.0 (v0.22.0에 LCM 추가) | - |
| 2026-07-24-addernet.md | ResNet-50 74.9%/76.2%, BNN | 문제 없음 | https://arxiv.org/abs/1912.13200 (AdderNet 74.9, CNN 76.2, BNN 55.8) | - |
| 2026-07-24-addernet.md | adaptive LR 식, HardTanh | 문제 없음 | https://arxiv.org/abs/1912.13200 (Eq. 13, "k denotes the number of elements in F_l"; HT는 [-1, 1]) | - |
| 2026-07-24-addernet.md | SCALE-Sim "현재 V2" | 수정함 | https://arxiv.org/abs/2504.15377 (SCALE-Sim v3, 2025년 4월, ISPASS 2025) | "(2018, 현재 V2)" → "(2018, 이후 V2를 거쳐 2025년 V3 공개)" |
| 2026-07-24-addernet.md | `.cu`/`.cpp` 역할 분담 | 문제 없음 | https://pytorch.org/tutorials/advanced/cpp_extension.html ("구성할 수 있습니다"라는 선택지 서술이라 오류로 보지 않음) | - |
| 2026-07-21-fq-vit.md | 채널 범위 40배의 기준 | 수정함 | https://arxiv.org/abs/2111.13824 ("some channel ranges even exceed 40× of the median"; Swin-B 최대/중앙 범위 622.5/15.5) | "채널 간 범위가 \*\*최대 약 40배\*\*까지 차이 남" → "일부 채널의 범위가 중앙값의 \*\*약 40배\*\*를 넘을 만큼 차이 남" |
| 2026-07-21-fq-vit.md | 98.8%, 8.69%, 63%p | 문제 없음 | https://arxiv.org/abs/2111.13824 ("about 98.8% of values are smaller than 1/16", "8.69% ... decreasing 63.05% from 8-bit case") | - |
| 2026-07-21-fq-vit.md | DeiT/ViT/Swin 하락 정도 | 문제 없음 | https://arxiv.org/abs/2111.13824 (Table 1 MinMax 8/8/8: DeiT 70.94~78.02, ViT 23.64/3.37, Swin 64.38~25.58) | - |
| 2026-07-21-fq-vit.md | W8/A8/Attn4 손실, COCO, IJCAI 2022 | 문제 없음 | https://arxiv.org/abs/2111.13824 (Table 1 손실 0.73~1.85%p, Table 2 Swin-S Mask/Cascade Mask R-CNN, "Accepted by IJCAI 2022") | - |
| 2026-07-21-fq-vit.md | 코드 출력 | 문제 없음 | 로컬 재실행: [0.0, 0.0, 0.0, 5.0] / [0.2584 0.2648 0.2681 0.2617] / [0.0079 0.008 0.0081 0.2682] | - |
