---
title: "[Transformer] Mobile-Former: MobileNet과 Transformer를 양방향으로 잇다"
date: 2026-07-13 21:01:00 +0900
categories: [Transformer, Vision Transformer]
tags: [vision-transformer, mobile-former, mobilenet, cross-attention, dynamic-relu, lightweight, inverted-residual]
description: MobileNet과 Transformer를 직렬로 섞지 않고 병렬로 두어 6개의 global token과 가벼운 양방향 cross-attention으로 잇는 Mobile-Former의 구조와, 300M FLOPs 이하에서 MobileNetV3를 넘은 이유를 정리합니다.
math: true
---

> [경량 Vision Transformer 동향]({% post_url 2026-06-13-lightweight-vision-transformer %})에서 "CNN과 Transformer를 섞는 것"이 경량화의 핵심 방향이라고 정리했습니다. **Mobile-Former**(Chen et al., CVPR 2022)는 둘을 **앞뒤로 쌓지 않고 나란히(병렬로)** 두고, 그 사이를 가벼운 cross-attention 다리로 양방향 연결합니다.

## 1. 배경: 작은 연산량에서는 아직 MobileNet이 강하다

ViT, DeiT 같은 Vision Transformer는 큰 모델에서 CNN을 넘어섰습니다. 하지만 **300M FLOPs 이하**처럼 연산량이 작은 영역에서 순위를 매기면 여전히 **MobileNet 계열**(MobileNetV2/V3, GhostNet, MicroNet)이 우세합니다. 특히 MobileNetV2의 **Inverted Residual Block**을 쓰는 모델들이 강합니다.

두 구조의 장점은 서로 보완적입니다.

| | CNN (MobileNet) | Transformer |
|---|---|---|
| 강점 | **Local** 정보 처리, 압도적인 연산 효율 | **Global** 상호작용 |
| 약점 | 멀리 떨어진 위치 사이의 관계 | 연산량 ($$O(N^2)$$), 데이터 요구량 |

그래서 둘을 합치는 연구가 많았지만, 대부분은 **직렬** 결합이었습니다.

- Transformer 앞에 convolution을 전처리처럼 두기 (*Early Convolutions Help Transformers See Better*)
- 중간중간 섞어 쓰기 (MobileViT 등)

## 2. Mobile-Former의 아이디어: 병렬 구조와 양방향 다리

Mobile-Former는 두 갈래를 **동시에** 돌립니다.

- **Mobile**: 이미지 feature map $$X \in \mathbb{R}^{HW \times C}$$를 처리. MobileNetV2 기반 inverted residual block으로 **local feature** 추출
- **Former**: 소수의 **global token** $$Z \in \mathbb{R}^{M \times d}$$를 처리. Multi-Head Attention과 FFN으로 **global** 연산

핵심은 Former가 이미지 패치 전체가 아니라 **학습 가능한 토큰 6개**($$M = 6$$, $$d = 192$$)만 다룬다는 점입니다. 토큰은 처음에 무작위로 초기화되어 학습됩니다. 토큰 수가 적으니 self-attention 비용이 거의 들지 않습니다.

두 갈래 사이는 **Mobile → Former**, **Former → Mobile** 두 방향의 가벼운 cross-attention으로 연결됩니다. 이 네 부분을 묶은 것이 **Mobile-Former Block**입니다.

$$
(X, Z) \;\xrightarrow{\text{Mobile-Former Block}}\; (X', Z')
$$

출력 $$X', Z'$$는 다음 블록의 입력이 됩니다.

### Inverted Residual Block 복습

- **ResNet bottleneck**: Wide → Narrow → Wide. 가운데 좁은 부분이 병목(bottleneck)
- **MobileNetV2 inverted residual**: Narrow → **Wide** → Narrow. 가운데를 넓혀 depthwise conv를 하고, 입출력이 좁아 메모리 효율적

자세한 내용은 [MobileNet 글]({% post_url 2026-05-30-mobilenet %})에 있습니다.

## 3. Mobile-Former Block의 네 단계

### (1) Mobile → Former: 이미지 정보를 토큰으로 모으기

Global token이 **Query**, 이미지 feature가 **Key와 Value**가 되는 cross-attention입니다.

$$
Z^{out} = Z + \Big[\text{Attn}\big(\tilde{z}_i W_i^Q,\ \tilde{x}_i,\ \tilde{x}_i\big)\Big]_{i=1:h} W^O
$$

- $$\tilde{z}_i, \tilde{x}_i$$: $$i$$번째 head로 나눈 토큰과 feature
- $$W^O$$: 여러 head의 출력을 합칠 때 쓰는 가중치

**경량화 포인트**: 일반 attention이라면 $$X$$에도 $$W^K, W^V$$를 곱해야 하지만, 이미지 쪽은 위치 수 $$HW$$가 커서 비쌉니다. 그래서 **Key와 Value에 projection을 하지 않고 $$X$$를 그대로** 씁니다. 연산은 토큰 쪽(크기 $$M$$)에만 몰려서, 비용이 $$O(M \cdot HW \cdot C)$$로 **이미지 크기에 선형**입니다.

또 이 연산은 inverted residual 블록의 **좁은 쪽(bottleneck)** 채널에서 이루어져 비용이 더 줄어듭니다.

### (2) Former: 토큰끼리 전역 정보 교환

토큰 6개에 대해 일반 Transformer처럼 **Multi-Head Self-Attention → FFN**을 수행합니다. 토큰이 이미지 전체에서 모은 정보를 서로 교환하는 단계입니다. 토큰 수가 매우 적어 이 부분의 연산량은 전체에서 아주 작습니다.

### (3) Mobile: Dynamic ReLU를 쓴 inverted residual

입력 $$X$$에 inverted residual 블록(1×1 확장 → depthwise 3×3 → 1×1 축소)을 적용합니다. 다른 점은 활성 함수로 **Dynamic ReLU**를 쓴다는 것입니다.

일반 ReLU는 모양이 고정되어 있고, PReLU·Leaky ReLU는 음수 쪽 기울기 하나만 바뀝니다. Dynamic ReLU는 **입력에 따라 음수·양수 양쪽의 기울기와 절편을 모두** 바꿉니다.

$$
y_c = \max_{k}\big(a_c^k(\cdot)\,x_c + b_c^k(\cdot)\big)
$$

Mobile-Former에서는 이 계수 $$a, b$$를 **Former가 출력한 global token**(첫 번째 토큰)으로부터 만듭니다. 이미지 전체 맥락을 본 토큰이 "이 입력에서는 활성 함수를 어떤 모양으로 할지"를 정해 주는 셈입니다. [SENet]({% post_url 2026-06-02-cnn-architectures %})이 채널 가중치를 입력에 따라 바꾸는 것과 비슷한 발상입니다.

### (4) Former → Mobile: 전역 정보를 이미지로 돌려주기

이번에는 반대로 이미지 feature가 **Query**, 갱신된 토큰 $$Z'$$이 **Key와 Value**입니다.

$$
X^{out} = X + \Big[\text{Attn}\big(\tilde{x}_i,\ \tilde{z}'_i W_i^K,\ \tilde{z}'_i W_i^V\big)\Big]_{i=1:h}
$$

이번에도 이미지 쪽(Query)에는 projection을 하지 않습니다. 각 위치가 "전역 토큰 중 나와 관련 있는 정보"를 가져와 **local feature와 global token을 융합**합니다.

이미 계산된 출력을 재사용하면서 attention을 적용하기 때문에, 추가 메모리 사용량도 최소화됩니다.

```python
import torch
import torch.nn as nn

class Mobile2Former(nn.Module):
    """Query: token(Z), Key/Value: feature map(X) — X 쪽 projection 없음"""
    def __init__(self, dim_tok, dim_x, heads=2):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim_tok, dim_x)           # W^Q (토큰 쪽만)
        self.o = nn.Linear(dim_x, dim_tok)           # W^O

    def forward(self, x, z):                         # x: (B, HW, C), z: (B, M, d)
        B, N, C = x.shape
        h = self.heads
        q = self.q(z).view(B, -1, h, C // h).transpose(1, 2)          # (B, h, M, C/h)
        kv = x.view(B, N, h, C // h).transpose(1, 2)                  # (B, h, HW, C/h)
        attn = (q @ kv.transpose(-2, -1) / (C // h) ** 0.5).softmax(-1)
        out = (attn @ kv).transpose(1, 2).reshape(B, -1, C)            # (B, M, C)
        return z + self.o(out)

class Former2Mobile(nn.Module):
    """Query: feature map(X) — projection 없음, Key/Value: token(Z')"""
    def __init__(self, dim_tok, dim_x, heads=2):
        super().__init__()
        self.heads = heads
        self.k = nn.Linear(dim_tok, dim_x)           # W^K
        self.v = nn.Linear(dim_tok, dim_x)           # W^V

    def forward(self, x, z):
        B, N, C = x.shape
        h = self.heads
        q = x.view(B, N, h, C // h).transpose(1, 2)
        k = self.k(z).view(B, -1, h, C // h).transpose(1, 2)
        v = self.v(z).view(B, -1, h, C // h).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1) / (C // h) ** 0.5).softmax(-1)  # (B, h, HW, M)
        return x + (attn @ v).transpose(1, 2).reshape(B, N, C)

x, z = torch.randn(2, 56 * 56, 24), torch.randn(2, 6, 192)
z = Mobile2Former(192, 24)(x, z)
print(Former2Mobile(192, 24)(x, z).shape)    # torch.Size([2, 3136, 24])
```

Attention 행렬 크기가 $$(HW \times M)$$이라, $$M = 6$$이면 일반 self-attention의 $$(HW \times HW)$$보다 훨씬 작습니다. 위 예시의 56×56 feature map이라면 $$3136 \times 6 = 18{,}816$$개 대 $$3136 \times 3136 \approx 980$$만 개로, 약 520배 차이입니다. 두 다리 모듈의 파라미터도 각각 9,432개, 9,264개에 불과합니다(위 코드 기준, `sum(p.numel() for p in m.parameters())`로 확인).

> 위 코드는 구조를 보여 주기 위한 단순화 버전입니다. 논문 구현에는 LayerNorm, 다리 모듈 안의 채널 차원 맞추기 등 세부 처리가 더 있습니다. 또 위 코드의 토큰 수 $$M$$은 입력 `z`의 크기로 정해지며, 실제 모델에서는 `nn.Parameter`로 선언해 학습합니다.
{: .prompt-info }

## 4. 전체 구조

ImageNet 분류용 **Mobile-Former-294M**(294M FLOPs) 기준입니다.

- Stem: 3×3 conv + 가벼운 bottleneck
- **Mobile-Former Block 11개**, 모든 블록이 **192차원 global token 6개**를 공유하며 전달
- 이미지 크기에 따라 여러 stage로 구성되고, stage마다 해상도를 절반으로 줄이며 채널을 늘림. 해상도를 줄이는 블록은 Mobile 쪽에 stride 2 depthwise conv를 둔 downsampling 버전
- **Head**: 마지막 local feature를 pooling·flatten한 뒤 **첫 번째 global token과 concat**하고, FC 두 개를 거쳐 분류

연산량 예산에 맞춰 채널 수와 블록 구성을 바꾼 26M ~ 508M FLOPs의 여러 버전이 있습니다.

## 5. 결과

- **ImageNet 분류**: Mobile-Former-294M이 top-1 **77.9%**로, MobileNetV3보다 **1.3%p 높으면서 연산량은 약 17% 적습니다.** 다른 연산량 구간에서도 같은 FLOPs의 MobileNet 계열과 ViT 계열 경량 모델보다 높은 정확도를 보였습니다.
- **연산량이 작을수록 이득이 큼**: 매우 작은 연산량 영역에서 특히 효과적이었습니다. 적은 수의 토큰만으로도 전역 정보를 효율적으로 전달할 수 있다는 뜻입니다.
- **Object Detection**: RetinaNet의 backbone으로 썼을 때 MobileNetV3 backbone보다 AP가 크게 높았고, 이 토큰 구조를 DETR 형태의 detector에도 확장했습니다.

## 6. 정리

| 요소 | 역할 | 경량화 포인트 |
|---|---|---|
| Mobile (inverted residual) | Local feature | MobileNet의 효율 그대로 |
| Former (6개 토큰) | Global 상호작용 | 토큰 수가 적어 self-attention 비용 미미 |
| Mobile → Former | 이미지 정보를 토큰으로 수집 | 이미지 쪽 Key/Value projection 제거 |
| Former → Mobile | 전역 정보를 이미지로 전달 | 이미지 쪽 Query projection 제거 |
| Dynamic ReLU | 토큰이 활성 함수 모양을 결정 | 전역 맥락을 local 연산에 반영 |

Mobile-Former는 CNN과 Transformer를 **직렬로 섞는 대신 병렬로 두고 얇은 다리로 잇는** 설계로, 작은 연산량 영역에서 MobileNet을 넘어선 사례입니다.

## References

- Chen et al., "Mobile-Former: Bridging MobileNet and Transformer," CVPR 2022. [arXiv:2108.05895](https://arxiv.org/abs/2108.05895)
- Chen et al., "Dynamic ReLU," ECCV 2020.
- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR 2018.
- Howard et al., "Searching for MobileNetV3," ICCV 2019.
- Xiao et al., "Early Convolutions Help Transformers See Better," NeurIPS 2021.
- Mehta & Rastegari, "MobileViT," ICLR 2022.
