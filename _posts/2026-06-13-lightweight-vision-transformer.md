---
title: "[AI 기초 11] 경량 Vision Transformer 동향: 왜 무겁고, 어떻게 가볍게 만드나"
date: 2026-06-13 17:36:00 +0900
categories: [Transformer, Vision Transformer]
tags: [vision-transformer, vit, lightweight, mobilevit, efficientformer, beit, inductive-bias, backbone-neck-head]
description: Vision Transformer가 CNN과 무엇이 다르고 왜 크고 데이터가 많이 필요한지(inductive bias), 이를 가볍게 만든 BEiT, MobileViT, EfficientFormer, MobileOne의 접근을 정리합니다.
math: true
---

> 2022년 무렵부터 Image Classification과 Semantic Segmentation 벤치마크 상위권을 Transformer 기반 모델이 차지하기 시작했습니다. 하지만 Transformer는 **크고, 데이터가 많이 필요하고, 최적화가 어렵습니다.** 이 글에서는 Vision Transformer(ViT)가 CNN과 어떻게 다르고 왜 무거운지, 그리고 이를 가볍게 만들려는 연구들(BEiT, MobileViT, EfficientFormer)을 정리합니다.
>
> Transformer 자체의 구조는 [Attention Is All You Need 글]({% post_url 2026-06-10-attention-is-all-you-need %})을 먼저 보시면 좋습니다.

## 1. 최근 비전 모델의 구조: Backbone, Neck, Head

Detection, Segmentation 모델은 보통 세 부분으로 나눠 설계합니다([9편 Object Detection]({% post_url 2026-06-06-object-detection %})의 YOLOv4 구조가 대표적).

| 부분 | 역할 | 예시 |
|---|---|---|
| **Backbone** | 입력 이미지에서 특징을 담은 **feature map 추출** | VGG, ResNet, MobileNet, **ViT** |
| **Neck** | Backbone의 feature map을 **정제(refinement)·재구성(reconfiguration)** | FPN, PAN, BiFPN |
| **Head** | 정제된 feature map으로 **실제 예측**(Detection, Classification) | YOLO head, Mask head |

Neck 없이 shortcut으로 바로 연결하는 구조도 있지만, 대부분 **FPN**을 씁니다. FPN은 top-down 피라미드 구조로 **하나의 입력에서 여러 scale의 feature map**을 뽑아, 크기가 다른 물체를 각각 알맞은 해상도에서 찾게 합니다.

Transformer가 주목받는 이유는 **충분한 데이터로 사전학습했을 때** Backbone을 CNN에서 ViT 계열로 바꾸면 성능이 크게 오르기 때문입니다.

## 2. Attention은 Convolution, Dense와 무엇이 다른가

세 가지 층을 "입력 $$x_1, \dots, x_4$$를 어떻게 연결하고 어떤 가중치를 쓰는가"로 비교하면 차이가 분명해집니다.

| 층 | 연결 범위 | 가중치 |
|---|---|---|
| **Convolution** | **지역적**(locally connected) — 주변 몇 개만 | 모든 **위치가 같은 필터를 공유** (학습 후 고정) |
| **Dense (FC)** | **전체**(fully connected) | **연결마다 다른 가중치** (학습 후 고정) |
| **Attention** | **전체**(fully connected) | **입력 값에 따라 매번 새로 계산** |

Convolution과 Dense의 가중치는 학습이 끝나면 **어떤 이미지가 들어와도 같은 값**입니다. Attention만 입력에 따라 가중치가 달라집니다.

Attention의 가중치는 학습된 파라미터가 아니라 **입력끼리의 유사도로 그때그때 계산**됩니다. "이 이미지에서는 이 패치와 저 패치가 관련 있다"를 입력을 보고 판단하는 것입니다.

### SENet과의 공통점

입력에 따라 가중치가 바뀐다는 아이디어는 CNN에도 있었습니다. [SENet]({% post_url 2026-06-02-cnn-architectures %})은 각 채널을 pooling해서 뽑은 값으로 채널 가중치를 만들어 입력에 곱합니다. 이것도 일종의 **채널 attention**입니다. 비유하자면 Self-Attention은 이 "입력 의존 가중치"를 채널 단위가 아니라 **모든 위치 쌍**으로 넓힌 것입니다. (SE는 채널마다 숫자 하나를 곱할 뿐 위치끼리 정보를 섞지 않으므로, 둘이 같은 연산은 아닙니다.)

### Vision에서의 Self-Attention

1. 이미지를 16×16 **패치**로 잘라 각 패치를 토큰으로 만듭니다.
2. 각 토큰에서 Query, Key, Value를 만듭니다.
3. **각 패치마다 다른 모든 패치와의 유사도**를 계산하고 Softmax로 attention map을 만듭니다.
4. Attention map과 Value를 곱해 **self-attention feature map**을 얻습니다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

## 3. Transformer의 약점

### (1) 모델이 크다

| 모델 | 파라미터 |
|---|---|
| MobileNetV2 | 약 3.5M |
| ResNet-50 | 약 25M |
| **ViT-B/16** | **약 86M** |
| ViT-L/16 | 약 307M |
| ViT-H/14 | 약 632M |

또 Self-Attention은 토큰 수 $$N$$에 대해 **$$O(N^2)$$** 연산이 필요합니다. 해상도가 2배가 되면 토큰은 4배, attention 연산은 16배가 됩니다.

### (2) Dense Prediction에 무거운 Decoder가 필요하다

ViT는 하나의 해상도(1/16)의 토큰만 출력합니다. Detection, Segmentation 같은 **dense prediction** 작업에 쓰려면 여러 해상도의 feature를 만들어 줄 **무거운 decoder**가 따로 필요합니다.

이 문제와 $$O(N^2)$$ 문제를 함께 다룬 대표 연구가 **Swin Transformer**(Liu et al., ICCV 2021)입니다. Attention을 작은 window 안에서만 계산해 연산량을 토큰 수에 비례하도록 줄이고, 층마다 window를 조금씩 밀어(shifted window) 이웃 window끼리 정보를 주고받게 했습니다. 또 CNN처럼 stage마다 해상도를 줄여 1/4, 1/8, 1/16, 1/32의 **계층적 feature map**을 내므로 FPN 같은 기존 Neck을 그대로 붙일 수 있습니다. 결과적으로 "지역성"이라는 CNN의 가정을 다시 들여온 셈입니다.

### (3) 최적화가 어렵다

- 학습률, optimizer 설정에 민감하고 수렴이 느립니다.
- 성능을 내려면 **강한 data augmentation과 L2 regularization**이 필요합니다.
- 연구에 따르면 첫 patch embedding을 **몇 개의 conv 층으로 바꾸기만 해도** 학습이 훨씬 안정됩니다(*Early Convolutions Help Transformers See Better*, 2021).

### 근본 원인: Inductive Bias의 부족

**Inductive bias**란 모델이 한 번도 본 적 없는 상황을 예측하기 위해 쓰는 **추가적인 가정**입니다. CNN에는 강한 가정이 두 개 있습니다.

- **Locality**: 가까운 픽셀끼리 의미 있는 패턴을 만든다.
- **Translation invariance/equivariance**: 같은 패턴은 위치와 상관없이 같은 필터로 찾을 수 있다.

Transformer는 모든 토큰이 모든 토큰과 연결되는(fully connected) 구조라 이런 가정이 없습니다. 그래서 이 성질을 **데이터로부터 스스로 배워야** 하고, 그 때문에 학습에 **많은 데이터와 시간**이 필요합니다. ViT 논문에서도 ImageNet-1k(약 130만 장)만으로 학습하면 비슷한 크기의 ResNet보다 성능이 낮았고, ImageNet-21k(약 1,400만 장)에서는 비슷해졌으며, JFT-300M(3억 장)으로 사전학습하자 CNN을 확실히 넘었습니다.

다만 이것이 "ViT는 큰 데이터 없이는 안 된다"는 뜻은 아닙니다. 곧이어 나온 **DeiT**(Touvron et al., ICML 2021)는 강한 data augmentation, regularization, CNN teacher로부터의 distillation을 써서 ImageNet-1k만으로도 ResNet과 비슷하거나 나은 성능을 냈습니다. 데이터가 부족하면 **학습 기법으로 inductive bias를 보충**할 수 있다는 것입니다.

## 4. 경량화 연구들

### BEiT: BERT Pre-Training of Image Transformers (2021)

데이터 부족 문제를 **자기지도 사전학습**으로 해결합니다. NLP의 BERT처럼 이미지 패치 일부를 가리고(masking) 맞히게 합니다.

1. 미리 학습된 tokenizer(dVAE)로 이미지를 **이산적인 visual token**으로 바꿉니다.
2. 이미지 패치의 약 40%를 가린 채 Transformer에 넣습니다.
3. 가려진 위치의 **visual token을 예측**하도록 학습합니다.

Label 없는 이미지로 사전학습할 수 있어, 적은 labeled 데이터로도 좋은 성능을 냅니다. 이후 MAE(Masked Autoencoder) 같은 연구로 이어졌습니다.

### MobileViT (2022)

**CNN과 Transformer를 섞어서** 양쪽의 장점을 취합니다.

- 기본 뼈대는 **MobileNetV2 블록**(inverted residual)
- 중간중간 **MobileViT 블록**을 넣음:
  1. 3×3 conv로 **지역적 특징**을 먼저 추출 (CNN의 inductive bias)
  2. feature map을 작은 패치로 펼쳐 Transformer로 **전역적 관계** 학습
  3. 다시 접어서 원래 feature map과 합침
- 약 5~6M 파라미터로 같은 크기의 CNN, ViT보다 높은 정확도

Transformer를 "convolution처럼 쓰는" 구조라, ViT보다 훨씬 적은 데이터와 단순한 augmentation으로도 학습됩니다.

### EfficientFormer: Vision Transformers at MobileNet Speed (2022)

"파라미터·FLOPs가 적다"가 아니라 **실제 모바일 기기에서의 지연 시간(latency)**을 기준으로 설계합니다. iPhone에서 각 연산의 실제 속도를 측정해 병목을 찾았습니다.

- **Patch embedding을 큰 커널·stride 대신 3×3 conv(stride 2) 여러 개**로 대체 (모바일 컴파일러가 잘 최적화하는 연산)
- **Reshape 연산 최소화**: 4D(CNN용, B×C×H×W)와 3D(Transformer용, B×N×C) 텐서를 자주 바꾸면 느립니다. 그래서 앞 stage는 4D 블록만, 마지막 stage만 3D 블록을 쓰는 **dimension-consistent** 구조
- 결과적으로 MobileNet과 비슷한 속도로 더 높은 정확도

### 비슷한 시기의 CNN 쪽 연구: MobileOne (2022)

MobileViT를 발표한 Apple은 **재매개변수화**([RepVGG]({% post_url 2026-06-02-cnn-architectures %}))를 모바일에 맞게 적용한 MobileOne도 발표했습니다. 학습 때는 여러 갈래의 depthwise/pointwise 블록을 쓰고, 추론 때는 하나로 합쳐 iPhone에서 1ms 이하의 지연 시간을 냅니다. "실제 기기 latency 기준 설계"라는 관점은 EfficientFormer와 같습니다.

## 5. 정리: 가벼운 Transformer를 위한 방향

| 문제 | 방향 |
|---|---|
| Inductive bias 부족 | **CNN 구조를 직·간접적으로 추가**하거나 다른 제약 조건을 둠 (MobileViT, conv stem) |
| 데이터 부족 | 자기지도 사전학습 (BEiT, MAE) |
| 모델 크기 | 기존 MobileNet류 CNN에 **Transformer 블록을 일부만** 추가 |
| 실제 속도 | Latency 기준 설계, reshape 최소화 (EfficientFormer) |
| 온디바이스 학습 | 파라미터 수 제약, activation을 **ReLU 계열**로(메모리 절약, [TinyTL]({% post_url 2026-06-19-edge-ai-hardware %})) |
| 더 극단적 경량화 | Transformer를 직접 이진화하기보다, **추가하는 CNN 구조를 BNN으로** 만들어 결합 ([Bi-Real Net]({% post_url 2026-06-17-binary-neural-networks %}) 등) |

NLP Transformer에서 먼저 나온 개선 사항(사전학습, 효율적 attention, 양자화)을 Vision Transformer에 옮겨 오는 것도 흔한 접근입니다. 또 Transformer 자체에 shortcut을 만들어 넣고 CNN 부분의 크기를 줄이는 방식도 시도되고 있습니다.

다음 글에서는 경량화의 극단인 **이진 신경망(BNN)**을 XNOR-Net부터 ReActNet, RB-Net까지 한 번에 정리합니다.

## References

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Touvron et al., "Training data-efficient image transformers & distillation through attention (DeiT)," ICML 2021. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
- Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," ICCV 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- Xiao et al., "Early Convolutions Help Transformers See Better," NeurIPS 2021.
- Bao et al., "BEiT: BERT Pre-Training of Image Transformers," ICLR 2022.
- Mehta & Rastegari, "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer," ICLR 2022.
- Li et al., "EfficientFormer: Vision Transformers at MobileNet Speed," NeurIPS 2022.
- Vasu et al., "MobileOne: An Improved One millisecond Mobile Backbone," CVPR 2023.
- Lin et al., "Feature Pyramid Networks for Object Detection," CVPR 2017.
