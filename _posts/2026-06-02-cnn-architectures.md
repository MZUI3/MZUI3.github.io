---
title: "[AI 기초 8] CNN 아키텍처 계보: DenseNet, SENet, ShuffleNet, RepVGG"
date: 2026-06-02 19:09:00 +0900
categories: [Neural Network, CNN]
tags: [cnn, densenet, senet, squeeze-excitation, shufflenet, group-convolution, channel-shuffle, repvgg, re-parameterization]
description: ResNet 이후 CNN이 발전한 네 방향을 DenseNet(concat), SENet(채널 attention), ShuffleNet(group conv와 channel shuffle), RepVGG(구조적 재매개변수화)로 정리하고 PyTorch로 직접 확인합니다.
math: true
---

> [ResNet]({% post_url 2026-05-23-resnet %})과 [MobileNet]({% post_url 2026-05-31-mobilenet %}) 이후 CNN 구조는 크게 네 방향으로 발전했습니다. 이 글에서는 각 방향의 대표 모델을 묶어 정리합니다.
>
> - **연결을 더 촘촘하게** → DenseNet
> - **채널마다 중요도를 다르게** → SENet
> - **더 가볍게** → ShuffleNet
> - **학습할 때와 추론할 때 구조를 다르게** → RepVGG

## 한눈에 보기

| 모델 | 연도 | 핵심 아이디어 | 한 줄 요약 |
|---|---|---|---|
| ResNet | 2015 | Residual (덧셈) shortcut | 깊은 네트워크 학습 가능 |
| DenseNet | 2017 | 모든 이전 층을 **concat** | 특징 재사용 극대화 |
| SENet | 2017 | 채널 attention (SE block) | 어떤 채널에 집중할지 학습 |
| ShuffleNet | 2017 | Group conv + **Channel shuffle** | 1×1 conv 비용 절감 |
| RepVGG | 2021 | 구조적 재매개변수화 | 학습은 multi-branch, 추론은 3×3 conv만 |

---

## 1. DenseNet: 모든 층을 서로 연결하기

### ResNet과의 차이

ResNet은 이전 출력을 현재 출력에 **더합니다(summation)**. DenseNet(Huang et al., CVPR 2017)은 **이전의 모든 층 출력을 채널 방향으로 이어 붙여(concatenation)** 입력으로 씁니다.

$$
\text{ResNet: } x_\ell = H_\ell(x_{\ell-1}) + x_{\ell-1}
$$

$$
\text{DenseNet: } x_\ell = H_\ell\big([x_0, x_1, \dots, x_{\ell-1}]\big)
$$

더하기는 정보를 섞어 버리지만, concat은 **이전 층의 특징을 그대로 보존**합니다. 이렇게 모든 층이 연결된 묶음을 **Dense Block**이라고 합니다. $$L$$개 층이면 연결이 $$L(L+1)/2$$개가 됩니다.

### Composition Layer

Dense Block 안의 각 층 $$H_\ell$$은 **BN → ReLU → 3×3 Conv**로 구성됩니다. (Conv 뒤가 아니라 앞에 BN·ReLU를 두는 pre-activation 구조)

### Growth Rate $$k$$

각 층이 **새로 만드는 feature map의 수**입니다. $$\ell$$번째 층의 입력 채널 수는

$$
k_0 + k \times (\ell - 1)
$$

($$k_0$$: Dense Block 입력의 채널 수)

$$k = 4$$라면 층을 지날 때마다 채널이 4개씩 늘어납니다. 모든 층이 앞의 특징을 다 볼 수 있으니, 각 층은 **조금만(k개) 새로 기여하면** 됩니다. 그래서 $$k = 12 \sim 32$$ 정도의 아주 얇은 층으로도 충분하고, 이것이 DenseNet의 파라미터 효율의 비결입니다.

### Bottleneck Layer (DenseNet-B)

층이 깊어지면 입력 채널이 계속 늘어나 3×3 conv가 비싸집니다. 그래서 앞에 1×1 conv를 둬서 채널을 $$4k$$로 줄인 뒤 3×3 conv를 합니다.

$$
\text{BN} \to \text{ReLU} \to \text{1×1 Conv}(4k) \to \text{BN} \to \text{ReLU} \to \text{3×3 Conv}(k)
$$

ResNet의 bottleneck은 입력 채널 수만큼 출력을 만들지만, DenseNet은 **항상 $$k$$개만** 만든다는 차이가 있습니다.

### Transition Layer (DenseNet-C)

Concat하려면 feature map 크기가 같아야 하므로, 크기 축소는 Dense Block **사이**에서만 합니다.

$$
\text{BN} \to \text{1×1 Conv} \to \text{2×2 Average Pooling}
$$

이때 **compression factor** $$\theta$$ ($$0 < \theta \le 1$$)로 채널도 줄입니다. 입력이 $$m$$채널이면 출력은 $$\lfloor\theta m\rfloor$$채널이고, 보통 $$\theta = 0.5$$를 씁니다. Bottleneck과 compression을 모두 쓰면 **DenseNet-BC**입니다.

### DenseNet-121은 왜 121층일까

ImageNet용 DenseNet-121은 4개의 Dense Block에 각각 6, 12, 24, 16개의 bottleneck 층이 있습니다. bottleneck 층 하나에 conv가 2개이므로

$$
2 \times (6 + 12 + 24 + 16) = 116
$$

여기에 첫 7×7 conv 1개, transition layer의 1×1 conv 3개, 마지막 FC 1개를 더해 **121층**입니다.

```python
import torch, torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 4 * growth, 1, bias=False),
            nn.BatchNorm2d(4 * growth), nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth, growth, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.net(x)], dim=1)     # 덧셈이 아니라 concat

class DenseBlock(nn.Sequential):
    def __init__(self, n_layers, in_ch, growth):
        super().__init__(*[DenseLayer(in_ch + i * growth, growth) for i in range(n_layers)])

def transition(in_ch, theta=0.5):
    out = int(in_ch * theta)
    return nn.Sequential(nn.BatchNorm2d(in_ch), nn.Conv2d(in_ch, out, 1, bias=False),
                         nn.AvgPool2d(2)), out

block = DenseBlock(6, 64, growth=32)
print(block(torch.randn(1, 64, 56, 56)).shape)    # (1, 64 + 6*32 = 256, 56, 56)
```

### 장점

1. **Information flow**: 앞 층의 특징이 바뀌지 않고 직접 모든 뒤 층에 전달됩니다.
2. **Gradient flow**: 손실에서 각 층까지 경로가 짧아 기울기가 잘 전달됩니다(implicit deep supervision).
3. **파라미터 효율**: DenseNet-121은 약 8M 파라미터로, 약 25M인 ResNet-50과 비슷한 정확도를 냅니다.
4. **다양한 특징**: 각 층이 이전 모든 층을 입력으로 받아 더 다양하고 풍부한 패턴을 학습합니다.

**단점**: concat 때문에 중간 feature map을 모두 메모리에 들고 있어야 해서, 파라미터는 적어도 **학습 메모리 사용량이 큽니다.** 단순하게 구현하면 층마다 `torch.cat`이 새 텐서를 만들어 메모리가 층 수의 제곱에 비례해 늘어납니다. 그래서 공유 메모리 버퍼를 쓰거나 BN·concat 결과를 역전파 때 다시 계산하는 **memory-efficient DenseNet**(Pleiss et al., 2017) 구현이 따로 있습니다. torchvision의 `densenet121(memory_efficient=True)` 옵션이 이 방식입니다.

---

## 2. SENet: 채널마다 중요도를 다르게

SENet(Hu et al., 2017)은 **ILSVRC 2017 분류 부문 1위**(top-5 error 2.251%)로, 전년도 우승 대비 오차를 약 25% 줄였습니다. 새 네트워크라기보다 **기존 네트워크(VGG, Inception, ResNet)에 붙이는 블록**입니다.

Convolution은 공간적으로 가까운 정보를 잘 섞지만, **채널 사이의 관계**는 명시적으로 모델링하지 않습니다. SE block은 "지금 입력에서는 어떤 채널이 중요한가"를 학습해 채널별 가중치를 곱해 줍니다.

### Squeeze: 채널 정보 압축

Feature map $$U \in \mathbb{R}^{H \times W \times C}$$를 **Global Average Pooling**으로 $$1 \times 1 \times C$$로 압축합니다.

$$
z_c = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W}u_c(i, j)
$$

각 채널의 "전체적인 반응 정도"를 숫자 하나로 요약한 것입니다.

### Excitation: 채널 중요도 계산

두 개의 FC layer로 채널 사이 관계를 학습합니다.

$$
s = \sigma\big(W_2\,\text{ReLU}(W_1 z)\big),\qquad W_1 \in \mathbb{R}^{\frac{C}{r} \times C},\ W_2 \in \mathbb{R}^{C \times \frac{C}{r}}
$$

1. FC로 $$C \to C/r$$로 줄이고 ReLU (**bottleneck**, reduction ratio $$r$$)
2. FC로 $$C/r \to C$$로 되돌림
3. Sigmoid로 0~1 사이의 채널별 가중치 $$s$$ 출력

Bottleneck을 두는 이유는 다음과 같습니다.

- 파라미터 수가 줄어 계산이 가벼워짐
- 일반화에 도움
- ReLU로 비선형성을 줘서 채널 간 **비선형 관계**를 학습

### Scale: 원래 feature map에 곱하기

$$
\tilde{x}_c = s_c \cdot u_c
$$

중요한 채널은 키우고 덜 중요한 채널은 줄여, **어떤 채널에 집중할지 골라 줍니다.** Sigmoid를 쓰기 때문에 하나만 고르는 softmax와 달리 **여러 채널을 동시에 강조**할 수 있습니다.

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)                 # Squeeze
        self.fc = nn.Sequential(                            # Excitation
            nn.Linear(channels, channels // r), nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * s                                        # Scale

se = SEBlock(64)
print(se(torch.randn(2, 64, 8, 8)).shape)   # torch.Size([2, 64, 8, 8]) 크기는 그대로, 채널별로 스케일만 바뀜
```

ResNet에 붙일 때는 residual branch의 출력에 SE를 적용한 뒤 shortcut과 더합니다(SE-ResNet).

### 실험에서 확인된 것들

- **적은 비용, 큰 효과**: SE-ResNet-50은 ResNet-50 대비 연산량이 약 0.26%만 늘지만(3.86 → 3.87 GFLOPs) top-1 error가 약 1.5%p 줄어 ResNet-101에 가까운 성능을 냅니다.
- **Reduction ratio $$r$$**: $$r$$에 따라 성능이 단조롭게 변하지 않습니다. 너무 작으면 파라미터만 늘고, 너무 크면 표현력이 떨어집니다. 논문은 균형점으로 $$r = 16$$을 기본값으로 씁니다.
- **Squeeze의 역할**: GAP를 빼고 FC를 1×1 conv로 바꾼(NoSqueeze) 버전은 성능 향상이 줄어듭니다. **전역 정보**를 보는 것이 중요하다는 뜻입니다.
- **Excitation의 역할**: 서로 다른 클래스 이미지를 넣어 SE 출력을 비교하면, **얕은 층에서는 클래스와 무관하게 비슷한** 분포가 나옵니다(선, 원 같은 기초 특징은 모든 클래스에 필요하기 때문). **깊은 층으로 갈수록 클래스마다 다른** 분포가 나타나 채널 선택이 클래스에 특화됩니다. 단, 마지막 stage에서는 거의 모든 채널이 1에 가까워져 효과가 미미하므로, 이 부분의 SE를 빼서 파라미터를 아낄 수도 있습니다.

### 장점과 주의점

- 기존 네트워크에 **붙이기만 하면** 바로 적용 가능 (MobileNetV3, EfficientNet, RegNet 등에서 사용)
- 적은 연산량 추가로 성능 향상
- 채널 간 **비선형** 상호작용을 학습할 수 있어야 하고, **여러 채널을 동시에** 강조할 수 있어야 함 (그래서 ReLU와 Sigmoid를 사용)

이 "입력에 따라 가중치를 동적으로 바꾼다"는 아이디어는 [Attention]({% post_url 2026-06-20-lightweight-vision-transformer %})과 같은 계열입니다.

---

## 3. ShuffleNet: 1×1 Conv까지 가볍게

### 문제: 1×1 Conv가 너무 비싸다

MobileNet은 depthwise separable conv로 연산량을 크게 줄였지만, 그 결과 **연산량의 대부분이 1×1 pointwise conv**에 몰리게 됐습니다(MobileNet에서 약 95%). ShuffleNet(Zhang et al., CVPR 2018)은 이 1×1 conv를 줄이는 데 집중합니다.

### Pointwise Group Convolution

입력 채널을 $$g$$개 그룹으로 나누고, 1×1 conv도 **그룹 안에서만** 합니다. 연산량이 $$1/g$$로 줄어듭니다.

하지만 문제가 생깁니다. 그룹 conv를 여러 번 쌓으면 **각 그룹의 출력은 자기 그룹의 입력에서만** 나옵니다. 그룹끼리 정보가 섞이지 않아 표현력이 떨어집니다. 각 필터 그룹이 제한된 feature map만 학습하게 되는 것입니다.

### Channel Shuffle

그룹 conv 뒤에 **채널 순서를 섞어서**, 다음 그룹 conv의 각 그룹이 이전의 **모든 그룹에서 온 채널**을 받도록 합니다.

무작위로 섞는 것이 아니라, 모든 그룹의 채널이 고르게 퍼지도록 **정해진 방식**으로 섞습니다. 구현은 reshape와 transpose만으로 간단합니다. 학습할 파라미터도 없고 미분도 가능합니다.

$$
(g \times n) \xrightarrow{\text{reshape}} (g, n) \xrightarrow{\text{transpose}} (n, g) \xrightarrow{\text{flatten}} (n \times g)
$$

```python
import torch

def channel_shuffle(x, groups):
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)    # (g, n)
    x = x.transpose(1, 2).contiguous()          # (n, g)
    return x.view(b, c, h, w)

x = torch.arange(6).float().view(1, 6, 1, 1)   # 채널 [0 1 2 | 3 4 5], g=2
print(channel_shuffle(x, 2).flatten())         # [0, 3, 1, 4, 2, 5]
```

### ShuffleNet Unit

ResNet bottleneck을 바탕으로 만듭니다.

1. **1×1 Group Conv** (채널 축소) → BN → ReLU
2. **Channel Shuffle**
3. **3×3 Depthwise Conv** → BN (ReLU 없음)
4. **1×1 Group Conv** (채널 복원) → BN
5. Shortcut과 **더하기** → ReLU

크기를 줄이는(stride 2) unit에서는 shortcut에 **3×3 average pooling**을 두고, 더하기 대신 **concat**해서 적은 연산으로 채널을 늘립니다.

### 결과

- **Group 수**: 같은 연산량에서 $$g$$를 늘리면 더 많은 채널을 쓸 수 있어 대체로 정확도가 오릅니다. 특히 작은 모델일수록 효과가 큽니다.
- **Shuffle의 효과**: 같은 구조에서 shuffle을 넣으면 모든 $$g$$에서 error가 줄고, $$g$$가 클수록 차이가 커집니다.
- **MobileNet과 비교**: 비슷한 연산량(약 500 MFLOPs)에서 ShuffleNet 2×가 MobileNet 1.0보다 top-1 error가 약 3%p 낮습니다. 실제 ARM 모바일 기기에서 AlexNet보다 약 13배 빠르면서 비슷한 정확도를 냈습니다.

### 후속: ShuffleNetV2

V2(2018)는 **FLOPs가 적다고 실제로 빠른 것은 아니다**라는 점을 지적합니다. 메모리 접근 비용(MAC)과 병렬성을 고려해 네 가지 설계 원칙을 제시했습니다.

1. 입력과 출력 채널 수가 같을 때 메모리 접근이 최소
2. Group conv를 과하게 쓰면 메모리 접근이 늘어남
3. 네트워크를 여러 갈래로 쪼개면(fragmentation) 병렬성이 떨어짐
4. 원소별 연산(ReLU, 덧셈)도 무시할 수 없음

그래서 V2는 group conv 대신 **channel split**(채널을 반으로 나눠 한쪽만 연산)을 씁니다.

---

## 4. RepVGG: 학습과 추론의 구조를 분리하기

### 다시 VGG로?

VGG-16은 3×3 conv와 max pooling만 반복하는 단순한 구조입니다.

- 224×224×3 입력 → 5개의 conv 단계 → 7×7×512 feature map
- Flatten → 25,088 → FC 4096 → FC 4096 → FC 1000 → Softmax

이후 ResNet, Inception 같은 **multi-branch** 모델이 정확도를 크게 올렸지만 대가가 있었습니다.

- 구조가 복잡해 구현과 커스터마이징이 어렵고
- shortcut을 위해 중간 결과를 **메모리에 들고 있어야** 하며
- 여러 갈래를 기다려야 해서 **추론 속도**가 느려집니다.

RepVGG(Ding et al., CVPR 2021)의 해법은 **"multi-branch의 장점은 학습할 때만 쓰자"**입니다.

### 학습 vs 추론 구조

| | 학습 (Training) | 추론 (Inference) |
|---|---|---|
| 블록 구조 | 3×3 Conv+BN, 1×1 Conv+BN, Identity+BN **세 갈래를 더함** | **3×3 Conv 하나** + ReLU |
| 장점 | ResNet처럼 잘 학습됨 | VGG처럼 단순, 빠름, 메모리 효율적 |

### VGG와 다른 점

- Max pooling을 쓰지 않고 각 stage의 **첫 3×3 conv에 stride 2**로 down-sampling
- 5개 stage, 마지막은 **Global Average Pooling** + FC
- 파라미터를 줄이기 위해 일부 층에 **group-wise 3×3 conv** 사용
- 채널 폭을 배율 $$a$$, $$b$$로 조절해 여러 크기의 모델 구성: **RepVGG-A**는 ResNet-18/34/50급 경량·중형 모델과, **RepVGG-B**는 RegNet, ResNeXt급 대형 모델과 경쟁

### Re-parameterization: 세 갈래를 3×3 하나로

**Step 1. Conv + BN → Conv + bias**

BN은 추론 때 고정된 선형 변환이므로 conv에 흡수할 수 있습니다.

$$
\text{BN}(W * x) = \gamma\frac{W * x - \mu}{\sigma} + \beta = \underbrace{\frac{\gamma}{\sigma}W}_{W'} * x + \underbrace{\left(\beta - \frac{\gamma\mu}{\sigma}\right)}_{b'}
$$

**Step 2. 1×1 conv와 identity를 3×3으로**

- **1×1 conv**: 3×3 커널의 **가운데에만** 값을 두고 나머지를 0으로 채우면(zero-padding) 같은 연산입니다.
- **Identity**: 자기 채널의 가운데만 1이고 나머지는 0인 3×3 커널과 같습니다(단위 행렬).

**Step 3. 세 커널을 더하기**

convolution은 선형이므로 세 갈래의 커널과 bias를 **그냥 더하면** 하나의 3×3 conv가 됩니다.

$$
W_{rep} = W'_{3\times3} + \text{pad}(W'_{1\times1}) + W'_{id},\qquad b_{rep} = b'_{3\times3} + b'_{1\times1} + b'_{id}
$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def fuse_conv_bn(kernel, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    return kernel * t, bn.bias - bn.running_mean * bn.weight / std

class RepVGGBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.bn3 = nn.BatchNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, 1, bias=False);            self.bn1 = nn.BatchNorm2d(ch)
        self.bn_id = nn.BatchNorm2d(ch)
        self.fused = None

    def forward(self, x):
        if self.fused is not None:                      # 추론: 3x3 conv 하나
            return F.relu(self.fused(x))
        return F.relu(self.bn3(self.conv3(x)) + self.bn1(self.conv1(x)) + self.bn_id(x))

    @torch.no_grad()
    def reparameterize(self):
        k3, b3 = fuse_conv_bn(self.conv3.weight, self.bn3)
        k1, b1 = fuse_conv_bn(F.pad(self.conv1.weight, [1, 1, 1, 1]), self.bn1)
        eye = torch.zeros(self.ch, self.ch, 3, 3)
        eye[range(self.ch), range(self.ch), 1, 1] = 1   # identity를 3x3 커널로
        kid, bid = fuse_conv_bn(eye, self.bn_id)
        self.fused = nn.Conv2d(self.ch, self.ch, 3, padding=1)
        self.fused.weight.copy_(k3 + k1 + kid)
        self.fused.bias.copy_(b3 + b1 + bid)

torch.manual_seed(0)
blk = RepVGGBlock(8)
for _ in range(20):                                 # BN 통계가 기본값(평균 0, 분산 1)에서 벗어나도록
    blk(torch.randn(4, 8, 16, 16) * 3 + 1)          # 학습 모드로 몇 번 통과시킴
blk.eval()

x = torch.randn(1, 8, 16, 16)
y1 = blk(x); blk.reparameterize(); y2 = blk(x)
print(torch.allclose(y1, y2, atol=1e-5))            # True
print((y1 - y2).abs().max().item())                 # 약 6e-07, float32 반올림 오차 수준
```

`eval()`을 꼭 호출해야 합니다. 학습 모드의 BN은 배치마다 평균·분산을 새로 계산하므로 고정된 conv로 바꿀 수 없고, 추론 모드에서 저장된 running 통계를 쓸 때만 위의 융합 식이 성립합니다. 또 identity branch는 **입력과 출력 채널 수가 같고 stride가 1인 블록에만** 있습니다. 각 stage의 첫 블록(stride 2)은 3×3과 1×1 두 갈래만 씁니다.

### 결과와 영향

- branch를 하나라도 빼면 정확도가 떨어져, **학습 시 multi-branch가 실제로 중요**하다는 것이 확인됐습니다.
- 추론 구조가 3×3 conv만으로 되어 있어 GPU에서 같은 정확도의 ResNet보다 빠릅니다.
- 이후 YOLOv6/v7의 **RepConv**, 모바일용 **MobileOne** 등 재매개변수화 계열 연구로 이어졌습니다. 저전력 하드웨어로 옮길 때도 구조가 단순해서 유리합니다.
- SE block을 붙인 변형(RepVGG-D2se 등)도 있습니다.

---

## 5. 정리

| 모델 | 해결한 문제 | 핵심 연산 | 대가 |
|---|---|---|---|
| DenseNet | 특징 재사용, 기울기 흐름 | Concat, growth rate | 학습 메모리 큼 |
| SENet | 채널 간 관계 미반영 | GAP → FC → Sigmoid → Scale | 연산량 소폭 증가 |
| ShuffleNet | 1×1 conv 비용 | Group conv + Channel shuffle | 실제 속도는 메모리 접근에 좌우 |
| RepVGG | multi-branch의 추론 비효율 | 학습 후 3×3으로 병합 | 학습 시 메모리 증가 |

다음 글에서는 이런 CNN backbone 위에 **물체의 위치까지 찾는** Object Detection을 다룹니다.

## References

- Huang et al., "Densely Connected Convolutional Networks," CVPR 2017. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
- Hu et al., "Squeeze-and-Excitation Networks," CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
- Zhang et al., "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices," CVPR 2018.
- Ma et al., "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design," ECCV 2018.
- Ding et al., "RepVGG: Making VGG-style ConvNets Great Again," CVPR 2021. [arXiv:2101.03697](https://arxiv.org/abs/2101.03697)
- Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR 2015.
