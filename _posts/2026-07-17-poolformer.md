---
title: "[Transformer] PoolFormer: Attention 대신 Pooling이어도 될까? (MetaFormer)"
date: 2026-07-17 23:17:00 +0900
categories: [Transformer, Vision Transformer]
tags: [vision-transformer, poolformer, metaformer, token-mixer, mlp-mixer, fnet, pooling]
description: Self-attention을 학습 파라미터가 없는 평균 풀링으로 바꿔도 성능이 유지된다는 PoolFormer 실험으로, Transformer 성능의 핵심이 attention이 아니라 MetaFormer라는 전체 구조라는 주장을 정리합니다.
math: true
---

> Vision Transformer가 잘 되는 이유는 흔히 **Self-Attention** 덕분이라고 생각합니다. *MetaFormer Is Actually What You Need for Vision*(Yu et al., CVPR 2022)은 이 생각에 의문을 던집니다. Attention을 **아무 학습 파라미터도 없는 평균 풀링**으로 바꿔도 성능이 거의 유지된다면, 진짜 핵심은 attention이 아니라 **Transformer의 전체 구조(MetaFormer)**라는 주장입니다.

## 1. Transformer 블록 다시 보기

[Transformer]({% post_url 2026-06-14-attention-is-all-you-need %}) 인코더 블록은 이렇게 생겼습니다.

$$
Y = \text{Attention}\big(\text{Norm}(X)\big) + X,\qquad Z = \text{MLP}\big(\text{Norm}(Y)\big) + Y
$$

이를 두 부분으로 나눠 볼 수 있습니다.

1. **토큰 사이의 정보를 섞는 부분** (Self-Attention) → **Token Mixer**
2. **그 나머지** (Normalization, Shortcut, Channel MLP)

논문은 Token Mixer를 **구체적으로 정하지 않은** 이 일반 구조를 **MetaFormer**라고 부릅니다.

$$
Y = \text{TokenMixer}\big(\text{Norm}(X)\big) + X
$$

$$
Z = \sigma\big(\text{Norm}(Y)\,W_1\big)W_2 + Y
$$

- 입력 이미지를 패치로 나눠 $$N$$개의 토큰, 각 $$C$$차원으로 임베딩
- Norm은 Layer Normalization 또는 Batch Normalization 등
- Channel MLP는 채널을 $$C \to rC \to C$$로 늘렸다 줄임 ($$W_1, W_2$$는 학습 파라미터, $$\sigma$$는 GELU나 ReLU)

## 2. 기존의 생각: Attention이 핵심이다

ViT 이후 많은 연구가 **attention을 더 잘 만드는** 데 집중했습니다. Attention의 유연성과 용량을 활용하려는 방향입니다.

- ConViT (convolution의 inductive bias를 attention에 섞음)
- PVT (피라미드 구조)
- Transformer in Transformer
- Swin Transformer (shifted window attention)

한편 최근에는 **attention 같은 입력 의존적(token dependent) 모듈이 비효율적**이라는 연구들도 나왔습니다. 고정된 attention 패턴이나 하드코딩된 가우시안 attention으로도 번역 성능이 크게 떨어지지 않는다는 결과들입니다.

### 다른 Token Mixer들

- **FNet**: Token Mixer 자리에 **푸리에 변환**을 넣어 토큰을 섞음. 학습 파라미터 없이 추론이 크게 빨라지고 정확도 손실은 작았습니다.
- **MLP-Mixer**: 모든 층을 **MLP로만** 구성. 토큰 축으로 MLP를 적용해 토큰을 섞습니다. MetaFormer 구조를 그대로 지키면서 **Token Mixer만 바꾼** 예이고, ViT에 근접한 성능을 냈습니다.

Token Mixer를 바꿔도 잘 된다면, **공통된 나머지 구조**가 중요한 것이 아닐까요?

## 3. PoolFormer: 가장 단순한 Token Mixer

이를 확인하려고 논문은 Token Mixer를 **극단적으로 단순한 연산**, 평균 풀링으로 바꿉니다.

$$
T' = \text{AvgPool}(T) - T
$$

- 입력은 $$C \times H \times W$$ 형태의 feature map
- 풀링 크기 $$K$$(기본 3)의 **stride 1** average pooling으로 주변 토큰 정보를 섞음
- **원래 값 $$T$$를 빼는** 이유: 바로 뒤에 residual shortcut으로 $$T$$가 다시 더해지므로, 빼지 않으면 자기 자신이 두 번 들어감

**학습 파라미터가 하나도 없고**, 연산량은 토큰 수에 **선형**입니다. Attention은 $$O(N^2)$$입니다.

```python
import torch.nn as nn

class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):                        # x: (B, C, H, W)
        return self.pool(x) - x                  # shortcut에서 x가 다시 더해짐

class PoolFormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, pool_size=3):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)        # 채널·공간 전체 정규화 (LayerNorm 변형)
        self.token_mixer = Pooling(pool_size)
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(nn.Conv2d(dim, dim * mlp_ratio, 1), nn.GELU(),
                                 nn.Conv2d(dim * mlp_ratio, dim, 1))     # Channel MLP (1x1 conv)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))

import torch

block = PoolFormerBlock(64)
x = torch.randn(2, 64, 56, 56)
print(block(x).shape)                                        # torch.Size([2, 64, 56, 56])
print(sum(p.numel() for p in block.token_mixer.parameters()))   # 0 → token mixer는 파라미터 없음
print(sum(p.numel() for p in block.parameters()))               # 33344 → 전부 Norm과 Channel MLP
```

블록의 파라미터는 모두 Norm과 Channel MLP에 있고, 토큰을 섞는 부분에는 하나도 없습니다. 그런데도 잘 동작한다는 것이 이 논문의 핵심 주장입니다.

> 다만 "attention이 필요 없다"로 확대 해석하면 안 됩니다. 이 결과는 ImageNet 분류처럼 **지역 정보가 중요한 비전 과제**에서 나온 것이고, 아래 ablation에서도 뒤쪽 단계에 attention을 섞으면 성능이 더 올랐습니다. 긴 거리의 관계가 중요한 언어 모델에서는 여전히 attention이 핵심입니다.
{: .prompt-info }

### 전체 구조

- CNN처럼 **4단계 계층 구조**: 해상도를 $$\frac{H}{4} \to \frac{H}{8} \to \frac{H}{16} \to \frac{H}{32}$$로 줄이며 채널을 늘림. 단계 사이에서 patch embedding으로 downsampling
- 크기별 변형: **PoolFormer-S12, S24, S36, M36, M48** (숫자는 전체 블록 수, S/M은 채널 폭)
- 계층 구조라 Detection, Segmentation의 backbone으로 바로 쓸 수 있음

## 4. 결과

### ImageNet 분류

정확도 대비 연산량(MACs)과 파라미터 수에서, PoolFormer는 같은 규모의 ViT·MLP 계열 모델과 비슷하거나 더 좋았습니다.

| 모델 | Top-1 | 파라미터 | Token Mixer |
|---|---|---|---|
| DeiT-B | 81.8% | 86M | Attention |
| **PoolFormer-M48** | **82.5%** | **73M** | **Average Pooling** |

학습 파라미터가 없는 pooling으로, **더 적은 파라미터**에서 attention 기반 모델 이상의 정확도를 냈습니다.

### Ablation

- Pooling 대신 **아무것도 하지 않는 identity**를 Token Mixer로 써도(S12) ImageNet top-1이 **약 74%**나 나왔습니다. 토큰을 섞지 않아도 MetaFormer 구조만으로 상당한 성능이 나온다는 뜻입니다.
- 앞쪽 단계는 pooling, 뒤쪽 단계만 attention을 쓰는 **혼합 구조**는 pooling만 쓴 것보다 더 좋아, 두 연산을 조합할 여지도 보여 줬습니다.

### 다른 과제

COCO 객체 검출·인스턴스 분할, ADE20K 의미 분할에서도 ResNet backbone보다 좋은 성능을 보였습니다.

## 5. Contribution: Attention is not all you need

- Transformer 계열 모델의 성능 원인은 attention이 아니라 **MetaFormer라는 일반 구조**에 있다.
- MetaFormer 형태의 모델 자체가 분류·검출·분할 등 **컴퓨터 비전 전반에서 핵심**이다.
- PoolFormer 자체가 최고 성능 모델은 아니지만, **평균 풀링만으로도** 높은 성능을 보인 만큼 MetaFormer 구조에 대한 연구를 촉진했다.

이후 같은 저자들은 다양한 Token Mixer를 MetaFormer 틀에서 비교하는 후속 연구로 이어갔고, [경량 ViT]({% post_url 2026-06-20-lightweight-vision-transformer %}) 설계에서도 "attention을 일부만 쓰는" 방향의 근거가 됐습니다.

## 6. 정리

| 구성 | 역할 | PoolFormer에서 |
|---|---|---|
| Token Mixer | 토큰 간 정보 교환 | **AvgPool − identity** (파라미터 0) |
| Norm | 안정화 | GroupNorm(1 그룹) |
| Shortcut | 정보 보존 | 그대로 |
| Channel MLP | 채널 간 변환 | 1×1 conv 두 개 |

## References

- Yu et al., "MetaFormer Is Actually What You Need for Vision," CVPR 2022. [arXiv:2111.11418](https://arxiv.org/abs/2111.11418)
- Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms," NAACL 2022.
- Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for Vision," NeurIPS 2021.
- Tay et al., "Synthesizer: Rethinking Self-Attention for Transformer Models," ICML 2021.
- You, Sun & Iyyer, "Hard-Coded Gaussian Attention for Neural Machine Translation," ACL 2020.
- Raganato et al., "Fixed Encoder Self-Attention Patterns in Transformer-Based Machine Translation," 2020.
