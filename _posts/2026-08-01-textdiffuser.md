---
title: "[Generative AI] TextDiffuser: 이미지 속에 글자를 정확하게 그리는 Diffusion"
date: 2026-08-01 16:05:00 +0900
categories: [Generative AI, Diffusion]
tags: [diffusion, textdiffuser, text-rendering, layout-generation, character-aware-loss, inpainting, mario-10m]
description: Diffusion 모델이 이미지 속 글자를 잘 못 그리는 이유와, 레이아웃 생성과 글자 단위 마스크·character-aware loss로 이를 해결한 TextDiffuser의 2단계 구조와 MARIO-10M 데이터셋을 정리합니다.
math: true
---

> Stable Diffusion에 "**'OPEN'**이라고 쓰인 카페 간판"을 그려 달라고 하면, 간판은 그럴듯한데 글자는 알아볼 수 없는 경우가 많습니다. **TextDiffuser**(Chen et al., NeurIPS 2023)는 [Latent Diffusion Model]({% post_url 2026-07-03-latent-diffusion-models %})을 fine-tuning해서 **이미지 속 글자를 정확하게 그리는** 문제를 다룹니다.

## 1. 문제: Diffusion은 글자에 약하다

Diffusion 모델은 사진 같은 이미지 생성에서는 뛰어난 성능을 보이지만, **글자(text rendering)**에서는 효율이 낮습니다.

- 글자 모양이 뭉개지거나, 존재하지 않는 글자가 생김
- 철자가 틀리거나 글자가 반복됨
- 글자를 어디에 얼마나 크게 둘지(레이아웃)를 제어하기 어려움

포스터, 책 표지, 간판, 광고 이미지처럼 **글자가 핵심인 이미지**를 만들 때 큰 문제입니다.

### 왜 어려울까

- 텍스트 인코더(CLIP)는 문장의 **의미**는 잘 담지만, 단어를 이루는 **글자 하나하나의 모양**은 잘 담지 못합니다. 대부분의 텍스트 인코더는 단어를 글자가 아니라 **subword 토큰**(예: `hello` 전체가 토큰 하나)으로 쪼개서 받기 때문에, 모델 입장에서는 그 토큰이 h-e-l-l-o로 이루어졌다는 정보가 처음부터 없습니다. Liu et al.(2023)은 이런 인코더를 "character-blind"라고 부르고, 글자 단위로 토큰을 받는 인코더(ByT5 등)를 쓰면 글자 렌더링이 크게 좋아진다는 것을 보였습니다.
- 학습 데이터의 캡션에 이미지 속 글자 내용이 정확히 적혀 있지 않은 경우가 많습니다.
- 일반적인 denoising loss는 이미지 전체의 픽셀 차이를 보기 때문에, 작은 영역인 **글자 획이 틀려도 loss가 거의 늘지 않습니다.**

## 2. TextDiffuser의 2단계 구조

TextDiffuser는 "**어디에 어떤 글자를 쓸지 먼저 정하고(Layout), 그 계획에 맞춰 그린다(Image)**"는 두 단계로 나눕니다.

사용자는 프롬프트에서 그릴 글자를 **따옴표로 표시**합니다.

> A poster of a movie named **'Hello World'** with a dark background

### Stage 1. Layout Generation

1. 프롬프트를 CLIP 텍스트 인코더로 임베딩하고, 따옴표로 표시된 **키워드**(`Hello`, `World`)를 찾습니다.
2. **Transformer**가 키워드마다 **bounding box**(위치와 크기)를 **autoregressive**하게 하나씩 생성합니다. 키워드 길이(글자 수) 정보를 함께 넣어 박스 너비가 글자 수와 맞게 합니다.
3. 생성된 박스 안에 키워드를 폰트로 렌더링해서, **글자 단위 segmentation mask**를 만듭니다. 마스크의 각 픽셀은 "배경" 또는 "어떤 글자(a~z, 0~9 등)"의 클래스 값을 가집니다.

이 마스크가 다음 단계에서 "**이 위치에 이 글자를 이런 모양으로**"라는 정밀한 설계도 역할을 합니다.

### Stage 2. Image Generation

Stable Diffusion의 U-Net을 fine-tuning합니다. 입력 채널을 늘려 원래의 noisy latent에 다음 정보를 **채널 방향으로 이어 붙입니다.**

| 입력 | 역할 |
|---|---|
| Noisy latent $$z_t$$ | 원래 diffusion 입력 |
| **Character-level segmentation mask** | 글자의 위치와 모양 (Stage 1 결과) |
| Feature mask | 새로 그릴 영역 표시 (inpainting용) |
| Masked feature | 유지할 영역의 원본 latent (inpainting용) |

학습할 때는 두 가지 모드를 섞습니다.

- **Whole-image generation**: 이미지 전체를 새로 생성 (feature mask가 전부 1)
- **Part-image generation**: 이미지 일부(글자 영역)만 다시 그림 → **텍스트 inpainting**

## 3. Character-Aware Loss

일반 denoising loss만으로는 글자 획의 정확도를 잡기 어렵습니다. 그래서 **글자에 특화된 loss**를 추가합니다.

1. 사전학습된 **글자 segmentation 네트워크**(latent 공간에서 동작하는 작은 U-Net)를 준비하고 고정합니다.
2. Diffusion이 denoising한 latent를 이 네트워크에 넣어 **글자 segmentation을 예측**합니다.
3. 예측과 정답 글자 마스크 사이의 **cross-entropy**를 loss로 씁니다.

$$
\mathcal{L} = \mathcal{L}_{denoising} + \lambda_{char}\,\mathcal{L}_{char}
$$

$$
\mathcal{L}_{denoising} = \mathbb{E}\big[\lVert\epsilon - \epsilon_\theta(z_t, t, c, \text{mask})\rVert_2^2\big],\qquad
\mathcal{L}_{char} = \text{CE}\big(\text{Seg}(\hat{z}_0),\ \text{CharMask}\big)
$$

글자 모양이 조금만 틀려도 segmentation 결과가 달라지므로, 모델은 **글자 하나하나의 형태**를 정확히 그리도록 학습됩니다. Ablation에서 이 loss를 쓰면 OCR 정확도가 올라갔고, 가중치가 너무 크면 이미지 전체 품질이 떨어져 적절한 균형이 필요했습니다. 논문은 $$\lambda_{char}$$를 0, 0.001, 0.01, 0.1, 1로 바꿔 실험했고, **0.01**일 때 가장 좋았습니다($$\lambda_{char} = 0$$ 대비 9.8% 향상).

> 이 loss가 latent 공간에서 계산된다는 점이 실용적인 포인트입니다. 픽셀 공간에서 OCR 모델로 감독하려면 step마다 VAE decoder로 이미지를 복원해야 해서 비싸지만, latent에서 바로 동작하는 작은 segmenter를 쓰면 그 비용이 없습니다.
{: .prompt-info }

## 4. 데이터셋: MARIO-10M

글자 렌더링을 학습하려면 **이미지 속 글자 위치와 내용이 표시된** 대규모 데이터가 필요합니다. 논문은 **MARIO-10M**을 새로 구축했습니다.

- 약 **1,000만 개**의 이미지–텍스트 쌍
- LAION, 영화 포스터(TMDB), 책 표지(Open Library) 등 글자가 많은 이미지에서 수집
- OCR로 **글자 내용, bounding box, 글자 단위 segmentation mask**를 자동으로 주석
- 글자가 너무 많거나 작은 이미지 등은 필터링

함께 공개한 **MARIO-Eval** 벤치마크로 글자 렌더링 품질을 평가합니다.

## 5. 실험

### 평가 지표

| 지표 | 무엇을 보나 |
|---|---|
| FID | 이미지 전체의 품질·사실성 |
| CLIP Score | 프롬프트와 이미지의 일치도 |
| **OCR Accuracy / Precision / Recall / F1** | 생성 이미지에서 OCR로 읽은 글자가 의도한 글자와 맞는가 |

### 결과

- Stable Diffusion, ControlNet, DeepFloyd IF 등과 비교해 **OCR 지표가 크게 높았고**, FID·CLIP score는 비슷한 수준을 유지했습니다. 이미지 품질을 해치지 않고 글자 정확도를 올린 것입니다.
- 사람 평가(user study)에서도 글자 품질과 이미지–텍스트 일치도에서 선호도가 높았습니다.

### 활용 방식

- **Text-to-Image**: 프롬프트만으로 레이아웃과 이미지를 모두 생성
- **Template 기반 생성**: 기존 이미지의 글자 배치를 템플릿으로 써서 레이아웃 고정
- **Text Inpainting**: 이미지에서 특정 영역의 글자만 다른 글자로 바꾸기

## 6. 한계

- 글자가 **매우 작거나 많으면** 여전히 부정확
- 학습 데이터 특성상 주로 **영어(라틴 문자)**를 대상으로 함. 한글처럼 글자 수가 많은 문자는 별도 연구가 필요
- Layout Transformer가 만든 박스가 어색하면 최종 이미지도 어색해짐 (2단계 구조의 오류 전파)

후속 연구 **TextDiffuser-2**는 레이아웃 생성을 **언어 모델**에 맡겨 더 유연한 배치와 긴 문장을 다룹니다.

## 7. 정리

| 요소 | 핵심 |
|---|---|
| Layout Generation | Transformer가 키워드별 bounding box 생성 → 글자 단위 마스크 |
| Image Generation | LDM U-Net에 글자 마스크·inpainting 마스크를 채널로 추가해 fine-tuning |
| Character-aware loss | latent 글자 segmenter로 글자 모양 정확도를 직접 감독 |
| MARIO-10M | OCR 주석이 달린 1,000만 쌍 데이터셋 |
| 결과 | 이미지 품질 유지 + OCR 정확도 대폭 향상 |

## References

- Chen et al., "TextDiffuser: Diffusion Models as Text Painters," NeurIPS 2023. [arXiv:2305.10855](https://arxiv.org/abs/2305.10855)
- Chen et al., "TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering," 2023.
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022.
- Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)," ICCV 2023.
- Liu et al., "Character-Aware Models Improve Visual Text Rendering," ACL 2023. [arXiv:2212.10562](https://arxiv.org/abs/2212.10562)
