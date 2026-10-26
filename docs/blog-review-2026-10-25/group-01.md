# 블로그 교정 보고서: 그룹 1

대상 10개 파일(_posts/). 교정 원칙: 맞춤법·띄어쓰기·문장부호·비문·어색한 표현·표기 통일만 고치고, 내용·수치·코드·수식·링크·front matter는 건드리지 않았습니다. 애매하면 고치지 않았습니다.

## 2026-10-22-gemini-robotics-1-5.md

### 수정 목록

수정 없음.

### 내용 확인 필요

없음.

## 2026-10-22-pi05.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 고수준 추론에서 크게 중요했습니다. | 고수준 추론에서 매우 중요했습니다. | 어색한 표현 |

### 내용 확인 필요

없음.

## 2026-10-22-smolvla.md

### 수정 목록

수정 없음.

### 내용 확인 필요

없음.

## 2026-10-22-wall-oss.md

### 수정 목록

수정 없음.

### 내용 확인 필요

없음.

## 2026-10-23-adversarial-diffusion-distillation.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| StyleGAN-T++은 저자들이 | StyleGAN-T++는 저자들이 | 맞춤법 |

### 내용 확인 필요

- 2장: "$$N = 4$$개의 정해진 집합 $$T_{\text{student}} = \{\tau_1, \ldots, \tau_n\}$$"에서 개수는 대문자 $$N$$, 첨자는 소문자 $$n$$($$\tau_n = 1000$$)으로 표기가 섞여 있습니다(수식이라 그대로 둠).
- 4장 ablation은 "ADD-M(SD 2.1 backbone, 860M)"인데, 5장 증류 방법 비교 표는 "SD 1.5 기반"이면서 같은 이름 ADD-M을 씁니다. 두 ADD-M의 backbone이 같은 모델인지 확인이 필요합니다.

## 2026-10-23-auto-kd.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 같은 평가 횟수에서 뒤집니다. | 같은 평가 횟수에서 뒤처집니다. | 맞춤법 |

### 내용 확인 필요

- 4장 시뮬레이션 설명은 보상을 "CKA **유사도**를 합친 것"이라고 하지만, 식은 $$1 - (\mathcal{L}_{CE} + \mathcal{L}_{CKA})$$이고 10장 정리는 "CKA 거리"라고 씁니다. 유사도인지 거리(손실)인지 표현을 맞출 필요가 있습니다.
- 8장 ViT 결과 "2~13% 향상": 제시된 세 수치는 DeiT-Ti +13.5, T2T-ViT-7 +9.25, PVT-Ti +8.26이라 하한 2%가 본문 수치로는 나오지 않습니다(논문의 다른 결과를 포함한 범위인지 확인).

## 2026-10-23-be-your-own-teacher.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| FitNet처럼 합성곱 한 층 대신 bottleneck 구조를 씁니다. | FitNet처럼 합성곱 한 층을 쓰는 대신 bottleneck 구조를 씁니다. | 어색한 표현 |

### 내용 확인 필요

- 4장 CIFAR-100: "모든 네트워크의 4/4 분류기가 ... 평균 2.65%p(ResNeXt 0.61 ~ VGG19 4.07) 향상"과 "ResNet101 +4.05, ResNet18 +2.58"은 표의 **앙상블** 열과 baseline의 차이와 일치합니다(예: ResNeXt 81.90 − 81.29 = 0.61). 4/4 분류기 기준이면 ResNeXt +0.22, VGG19 +3.26, ResNet101 +3.25입니다. 본문이 4/4 분류기 향상으로 서술해 기준이 어긋나 보입니다.
- 4장 "3/4 분류기가 baseline을 넘는 경우가 많아(10개 중 7개)": 표로 세면 ResNet50(75.23 < 77.68)과 PyramidNet(80.98 < 81.12)을 뺀 8개입니다.
- 6장 해석 2 "4/4 분류기도 baseline보다 2~4%p 좋습니다": 표에서는 ResNeXt +0.22, WideResNet20-8 +1.16 등 2%p 미만도 있습니다.
- ImageNet "평균 2.02%p 향상": 표의 4/4 기준 평균은 약 1.83%p, 앙상블 기준은 약 1.55%p라 표 수치로는 2.02가 나오지 않습니다.

## 2026-10-23-decision-transformer.md

### 수정 목록

수정 없음.

### 내용 확인 필요

없음.

## 2026-10-23-diffusion-policy.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 최고 성능 대부분이 이 쪽이지만 | 최고 성능 대부분이 이쪽이지만 | 띄어쓰기 |

### 내용 확인 필요

- 7장 해석 1 "위 예제에서 70줄의 코드로": 4장의 예제 코드는 약 48줄입니다.

## 2026-10-23-dist-kd.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 10배 넘게 차이\*\*납니다 | 10배 넘게 차이\*\* 납니다 | 띄어쓰기 |

### 내용 확인 필요

- 같은 ResNet-34 → ResNet-18 KD 결과가 "ImageNet 기본 설정" 표에서는 70.66, "더 큰 교사" 표에서는 71.21로 다릅니다(출처가 원 논문 인용 값과 재현 값으로 다른 것인지 확인).
- "더 큰 교사" 아래 "DIST는 계속 오르고": 표의 DIST는 72.07 → 72.12 → 72.08 → 72.24로 ResNet-101에서 잠시 내려갑니다.
- 6장 정리 "Swin-T 82.3%(KD +1%)": 표에서 KD 대비는 +0.8(81.5 → 82.3), 단독 학습 대비가 +1.0입니다.

## 내용 확인 결과 (2026-10-25)

gemini-robotics-1-5, pi05, smolvla, wall-oss, decision-transformer는 확인할 항목이 없었습니다.

| 파일 | 항목 | 판정 | 근거 | 수정 내용 |
|---|---|---|---|---|
| 2026-10-23-adversarial-diffusion-distillation.md | 2장 $$N = 4$$와 $$\{\tau_1, \ldots, \tau_n\}$$의 대소문자 혼용 | 문제 없음 | 원 논문도 "a set T_student = {τ1,…,τn} of N chosen student timesteps", "we choose N=4"로 똑같이 섞어 씁니다. 수식 오류가 아니라 원문 표기입니다. https://arxiv.org/html/2311.17042 | 없음 |
| 2026-10-23-adversarial-diffusion-distillation.md | 4장 ADD-M(SD 2.1) vs 5장 표 ADD-M(SD 1.5) | 문제 없음 | 논문에서 ablation은 "For ablating ADD-M, we use a Stable Diffusion (SD) 2.1 backbone"이고, 증류 방법 비교 표 2 캡션은 "All models are based on SD1.5"입니다. 같은 이름의 ADD-M을 두 backbone으로 학습한 것이 맞고, 글도 두 곳에 backbone을 각각 밝혀 두었습니다. https://arxiv.org/html/2311.17042 | 없음 |
| 2026-10-23-auto-kd.md | 보상의 CKA 항이 "유사도"인지 "거리"인지 | 문제 없음 | 논문 식 (3) 설명은 "the similarity between the teacher and student", "CKA metric for representation similarity"이고, 표 9 ablation 설명은 "CKA distance"입니다. 논문 자체가 두 표현을 함께 쓰므로 글의 서술(4장 유사도, 10장 거리)은 원문을 따른 것입니다. https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Automated_Knowledge_Distillation_via_Monte_Carlo_Tree_Search_ICCV_2023_paper.pdf | 없음 |
| 2026-10-23-auto-kd.md | 8장 ViT "2~13% 향상"의 하한 | 문제 없음 | 논문 표 5에는 글이 인용한 세 모델 외에 PiT-Ti(73.58 → 78.51, +4.93)와 PVTv2-B0(77.44 → 79.37, +1.93)가 있고, 본문도 "2% ∼ 13% margins"라고 씁니다. 하한 2%는 PVTv2-B0에서 나옵니다. 위 ICCV Open Access PDF | 없음 |
| 2026-10-23-be-your-own-teacher.md | 4장 CIFAR-100 "4/4 분류기 ... 평균 2.65%p", "ResNet101 +4.05, ResNet18 +2.58"의 기준 | 수정함 | 표 1의 앙상블 − baseline 10개 평균이 2.646, ResNeXt 0.61, VGG19 4.07, ResNet101 4.05, ResNet18 2.58로 모두 앙상블 기준입니다(4/4 기준이면 ResNeXt +0.22). 논문은 기준을 밝히지 않고 "increment of 2.65%"라고만 씁니다. https://arxiv.org/abs/1905.08094 | 두 문장에 "앙상블 기준"을 넣음(4/4가 모두 baseline보다 좋다는 앞부분은 표와 맞아 유지) |
| 2026-10-23-be-your-own-teacher.md | 4장 "3/4 분류기가 baseline을 넘는 경우 10개 중 7개" | 수정함 | 표 1(논문 표와 동일)에서 3/4 < baseline은 ResNet50(75.23 < 77.68), PyramidNet(80.98 < 81.12) 둘뿐이라 8개입니다. https://arxiv.org/abs/1905.08094 | 7개 → 8개 |
| 2026-10-23-be-your-own-teacher.md | 6장 해석 2 "4/4 분류기도 baseline보다 2~4%p" | 수정함 | 논문에 "2~4%" 표현은 없습니다. DSN 비교 대상인 ResNet18~152의 4/4 − baseline은 +1.55, +2.88, +3.25, +2.40입니다. https://arxiv.org/abs/1905.08094 | "2~4%p" → "1.5~3.3%p(ResNet18~152)" |
| 2026-10-23-be-your-own-teacher.md | ImageNet "평균 2.02%p 향상" | 문제 없음 | 논문 본문이 "2.02% in ImageNet on average"라고 씁니다. 표 2 수치로 다시 계산하면 4/4 기준 1.83, 앙상블 기준 1.55, 네트워크별 최고 분류기 기준 약 2.03으로 정확히 일치하지는 않지만, 글은 논문의 값과 표를 그대로 옮겼습니다. https://arxiv.org/abs/1905.08094 | 없음 |
| 2026-10-23-diffusion-policy.md | 7장 "70줄의 코드로" | 수정함 | 4장 예제 코드 블록은 47줄입니다(글 안에서 직접 셈). | "70줄의 코드로" → "50줄이 안 되는 코드로" |
| 2026-10-23-dist-kd.md | ResNet-34 → ResNet-18 KD가 표 2는 70.66, 표 3은 71.21 | 문제 없음 | 논문 표 2(B1 baseline 비교)는 KD 70.66, 표 3(교사 크기)와 ablation 표는 KD 71.21로, 원 논문 자체에 두 값이 따로 있습니다. 글은 두 표를 그대로 옮겼습니다. https://arxiv.org/abs/2205.10536 | 없음 |
| 2026-10-23-dist-kd.md | "더 큰 교사"에서 "DIST는 계속 오르고" | 수정함 | 논문은 "shows an upward trend"라고 하지만 표 3 수치는 72.07 → 72.12 → 72.08 → 72.24로 ResNet-101에서 내려갑니다. https://arxiv.org/abs/2205.10536 | "계속 오르고" → "ResNet-101에서 잠시 주춤할 뿐 대체로 오르고" |
| 2026-10-23-dist-kd.md | 6장 정리 "Swin-T 82.3%(KD +1%)" | 수정함 | 논문 서론은 "improving KD by 1%"라고 하지만, 표 4는 KD 81.5, DIST 82.3(+0.8)이고 본문은 Swin-T를 1.0% 올렸다고 씁니다(단독 학습 81.3 대비). 글의 표와도 맞지 않아 표 기준으로 고쳤습니다. https://arxiv.org/abs/2205.10536 | "(KD +1%)" → "(단독 학습 +1.0%, KD +0.8%)" |
