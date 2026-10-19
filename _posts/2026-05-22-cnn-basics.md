---
title: "[AI 기초 7] CNN 기초: Convolution, Padding, Stride, Pooling부터 변형 Convolution까지"
date: 2026-05-22 19:22:00 +0900
categories: [Neural Network, CNN]
tags: [cnn, convolution, padding, stride, pooling, dilated-convolution, transposed-convolution, depthwise-separable, im2col]
description: CNN이 MLP보다 이미지에 강한 이유(지역성, 가중치 공유)부터 Convolution·Padding·Stride·Pooling의 출력 크기 계산, Dilated·Transposed·Depthwise Separable Convolution, im2col까지 PyTorch로 확인하며 정리합니다.
math: true
---

> Part 2의 첫 글입니다. 이미지 처리의 기본인 **합성곱 신경망(CNN)**이 왜 필요한지부터, Convolution·Padding·Stride·Pooling의 동작과 출력 크기 계산, 그리고 Dilated·Transposed·Separable·Depthwise 같은 **변형 Convolution**까지 한 번에 정리합니다.

## 1. Training과 Inference

CNN을 보기 전에 두 용어를 구분해 둡니다.

- **Training (학습)**: 많은 입력 데이터로 가중치를 업데이트하며 모델을 만드는 과정. Forward + Backward를 반복합니다.
- **Inference (추론)**: 학습된 모델에 새 데이터를 넣어 결과를 얻는 과정. **Forward만** 하고 가중치는 바뀌지 않습니다.

온디바이스 AI에서 "추론만 하느냐, 학습까지 하느냐"는 필요한 메모리가 몇 배씩 차이 나는 큰 문제입니다([13편]({% post_url 2026-06-28-edge-ai-hardware %})).

## 2. 왜 CNN인가

MLP(Fully Connected 신경망)로 이미지를 다루면 두 가지 문제가 생깁니다.

1. **공간 정보 손실**: 2차원 이미지를 1차원으로 펼치면(flatten) 위·아래·옆 픽셀이 이웃이라는 정보가 사라집니다.
2. **비효율**: 224×224×3 이미지를 펼치면 입력이 15만 개입니다. 은닉 뉴런 1000개만 둬도 가중치가 1.5억 개가 되고, 학습이 오래 걸립니다.

CNN은 사람의 시각 피질 구조에서 착안해, **작은 필터를 이미지 위에서 움직이며** 지역적인 패턴을 찾습니다. 필터의 값은 사람이 정하지 않고 **학습으로 자동으로 정해집니다.**

CNN이 가진 두 가지 가정(inductive bias)이 핵심입니다.

- **Locality (지역성)**: 의미 있는 패턴은 가까운 픽셀끼리 만든다.
- **Weight sharing / Translation equivariance**: 같은 패턴은 이미지 어디에 있든 같은 필터로 찾을 수 있다. 입력이 옆으로 밀리면 출력 feature map도 똑같이 밀립니다(equivariance). "위치가 바뀌어도 출력이 아예 같다"는 불변성(invariance)과는 다른 말이고, 불변성은 뒤에서 볼 pooling이 어느 정도 더해 줍니다.

덕분에 파라미터가 크게 줄고, 적은 데이터로도 잘 학습됩니다. **Inductive bias**란 데이터를 보기 전에 모델 구조에 미리 들어 있는 가정을 말합니다. 가정이 문제와 잘 맞으면 적은 데이터로도 빨리 배우고, 맞지 않으면 오히려 한계가 됩니다. 이미지에 [Transformer]({% post_url 2026-06-14-attention-is-all-you-need %})를 적용한 ViT가 CNN보다 많은 학습 데이터를 필요로 하는 것도 주로 이 가정이 약하기 때문입니다.

## 3. Convolution

필터(커널)를 입력 위에서 한 칸씩 움직이며, 겹치는 부분끼리 곱해서 더합니다. 결과값이 크면 **그 위치에 필터와 비슷한 패턴이 있다**는 뜻입니다.

예를 들어 4×4 입력에 3×3 필터를 적용하면 2×2 출력이 나옵니다.

```
입력 (4x4)            필터 (3x3)       출력 (2x2)
1 2 0 1               1 0 1
0 1 3 1      *        0 1 0     =     [4  9]
2 1 0 2               1 0 1           [6  5]
1 0 1 3
```

왼쪽 위 값은 입력의 왼쪽 위 3×3 영역과 필터를 원소끼리 곱해 더한 것입니다.

$$
1\cdot1 + 2\cdot0 + 0\cdot1 + 0\cdot0 + 1\cdot1 + 3\cdot0 + 2\cdot1 + 1\cdot0 + 0\cdot1 = 4
$$

필터를 오른쪽으로 한 칸 옮겨 같은 계산을 하면 9가 나옵니다.

입력 채널이 여러 개(RGB)라면 필터도 같은 채널 수를 가지며, **모든 채널의 결과를 더해 출력 채널 하나**를 만듭니다. 필터를 $$N$$개 쓰면 출력 채널이 $$N$$개가 됩니다.

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1, 2, 0, 1],
                  [0, 1, 3, 1],
                  [2, 1, 0, 2],
                  [1, 0, 1, 3]], dtype=torch.float32).view(1, 1, 4, 4)
k = torch.tensor([[1, 0, 1],
                  [0, 1, 0],
                  [1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
print(F.conv2d(x, k))
# tensor([[[[4., 9.],
#           [6., 5.]]]])
```

> 딥러닝의 "convolution"은 수학적으로는 필터를 뒤집지 않는 **cross-correlation**입니다. 필터 값을 학습하기 때문에 뒤집든 안 뒤집든 결과에 차이가 없어서 그냥 convolution이라고 부릅니다.

## 4. Padding

Convolution을 하면 출력이 입력보다 작아집니다. 층을 쌓을수록 계속 줄어들고, **가장자리 픽셀은 필터가 덜 지나가서** 정보가 덜 반영됩니다.

입력 주변을 **0으로 둘러싸면(zero padding)** 출력 크기를 입력과 같게 유지할 수 있습니다. 3×3 필터에 padding 1이면 크기가 그대로입니다(`padding="same"`).

## 5. Stride

필터가 **한 번에 몇 칸씩 이동하는지**입니다. Stride 2면 한 칸 건너뛰며 계산하므로 출력 크기가 약 절반이 됩니다. Pooling 대신 stride 2 convolution으로 크기를 줄이는 경우도 많습니다(ResNet, MobileNet).

## 6. 출력 크기 공식

입력 크기 $$W$$, 필터 크기 $$K$$, padding $$P$$, stride $$S$$일 때

$$
W_{out} = \left\lfloor\frac{W - K + 2P}{S}\right\rfloor + 1
$$

| 입력 | K | P | S | 출력 |
|---|---|---|---|---|
| 32 | 3 | 0 | 1 | 30 |
| 32 | 3 | 1 | 1 | 32 |
| 32 | 3 | 1 | 2 | 16 |
| 224 | 7 | 3 | 2 | 112 |

### 파라미터 수

$$
\text{params} = (K \times K \times C_{in} + 1) \times C_{out}
$$

3×3, 입력 64채널, 출력 128채널이면 $$(9 \times 64 + 1) \times 128 = 73{,}856$$개입니다. **입력 이미지 크기와는 무관**하다는 점이 MLP와의 큰 차이입니다.

```python
conv = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
print(sum(p.numel() for p in conv.parameters()))   # 73856
print(conv(torch.randn(1, 64, 32, 32)).shape)      # torch.Size([1, 128, 32, 32])
```

## 7. Pooling

여러 필터가 만든 feature map의 크기를 줄입니다. 정해진 영역(pool size)에서 **최댓값(Max Pooling)** 또는 **평균값(Average Pooling)**만 남깁니다.

- 학습할 파라미터가 없음
- 연산량과 메모리 감소
- 작은 위치 변화에 둔감해짐 (작은 이동 불변성)

Feature map의 각 값은 이미 필터가 찾은 "그 위치의 특징"이므로, 크기를 줄여도 핵심 정보는 크게 손실되지 않습니다.

**Global Average Pooling (GAP)**은 채널마다 전체 공간을 평균 내서 (C, H, W)를 (C, 1, 1)로 만듭니다. 마지막 FC 층의 파라미터를 크게 줄여 줘서 현대 CNN의 분류 head 표준이 됐고, [SENet]({% post_url 2026-06-02-cnn-architectures %})의 Squeeze 단계에도 쓰입니다.

## 8. CNN의 전체 흐름

1. **Convolution + Activation (+ Pooling)** — 1차: 선, 모서리 같은 **단순한 패턴** 검출
2. **Convolution + Activation (+ Pooling)** — 2차: 1차 결과(선)를 조합해 **면, 질감** 검출
3. 층이 깊어질수록 **더 고차원적인 패턴**: 눈·코·입 → 얼굴
4. **Flatten → Fully Connected Layer**: 추출한 특징으로 분류 (MLP와 같은 과정)
5. **Softmax**: 클래스별 확률 (합이 1)

### 조절할 수 있는 hyper-parameter

- Convolution: 필터 개수, 필터 크기, padding 여부, stride, pooling 종류
- Fully connected: 층 수, 뉴런 수

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 32 → 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 16 → 8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                           # GAP
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

model = SimpleCNN()
print(model(torch.randn(8, 3, 32, 32)).shape)          # torch.Size([8, 10])  CIFAR-10 크기 입력
print(sum(p.numel() for p in model.parameters()))     # 94986
```

### Receptive Field

출력의 한 픽셀이 **입력에서 바라보는 영역의 크기**입니다. 3×3 conv를 두 번 쌓으면 receptive field는 5×5, 세 번이면 7×7이 됩니다. VGG가 큰 필터 대신 3×3을 여러 번 쌓은 이유가 이것입니다. 같은 receptive field를 더 적은 파라미터와 더 많은 비선형성으로 얻을 수 있습니다.

---

## 9. 변형 Convolution

### Dilated (Atrous) Convolution

필터 원소 **사이에 빈칸(0)을 넣어** 필터를 넓게 펼칩니다. 빈칸 간격을 **dilation rate**라고 합니다.

- 3×3 필터에 dilation 2 → 5×5와 같은 receptive field
- 실제로 계산하는 값은 9개 그대로라 **연산량이 거의 늘지 않음**
- 일반 CNN은 conv + pooling으로 해상도를 줄여 가며 전체 정보에 집중하지만, dilated conv는 **해상도를 유지한 채** 넓은 영역을 봅니다.

넓은 시야가 필요하면서 해상도도 중요한 **Semantic Segmentation**(DeepLab), 실시간 처리에서 많이 씁니다. 단, dilation을 크게 반복하면 바둑판처럼 일부 픽셀만 보는 **gridding 문제**가 생길 수 있습니다.

```python
nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)   # 크기 유지, RF 5x5
```

### Transposed Convolution

일반 conv와 반대로 **입력보다 큰 출력**을 만듭니다(upsampling). 입력의 각 값에 필터를 곱해서 출력에 뿌리고, **겹치는 부분은 더합니다.**

- Segmentation의 decoder, GAN의 generator, Super-resolution에서 사용
- Deconvolution이라고도 부르지만, 수학적인 역연산(deconvolution)과는 다른 연산입니다.

**Checkerboard artifact**: 필터 크기가 stride로 나누어떨어지지 않으면 겹치는 횟수가 위치마다 달라져 **격자 무늬**가 생깁니다. 필터 크기를 stride의 배수로(예: stride 2에 4×4 필터) 맞추면 줄어들지만, 학습된 가중치가 고르지 않으면 여전히 생길 수 있습니다. 그래서 Odena et al.(2016)은 **일반 upsampling(nearest/bilinear) 후 conv**(resize-convolution)를 권장합니다.

```python
nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 16x16 → 32x32
```

### Separable Convolution (Spatially Separable)

2차원 필터를 **두 개의 1차원 필터로 쪼갭니다.** 3×3 필터를 3×1과 1×3으로 나누면 곱셈이 9번에서 6번으로 줄어듭니다.

대표 예가 윤곽선 검출에 쓰는 **Sobel 필터**입니다. x 방향 Sobel 필터는 다음처럼 분해됩니다.

$$
\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}
= \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}
\begin{bmatrix} -1 & 0 & 1 \end{bmatrix}
$$

가운데 0을 기준으로 이전 값과 다음 값의 차이로 경계를 판별하고, x·y 방향 결과의 절댓값을 합쳐 윤곽선을 얻습니다. 다만 모든 필터가 이렇게 분해되지는 않아서 딥러닝에서는 제한적으로 씁니다.

### Depthwise Convolution

일반 conv는 모든 입력 채널을 섞어 하나의 출력을 만들지만, depthwise conv는 **채널마다 따로 필터를 하나씩** 적용해 **입력 채널 수만큼** 출력을 만듭니다. RGB 입력이라면 R, G, B를 각각 따로 처리합니다.

```python
nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)   # groups = 채널 수
```

### Pointwise Convolution (1×1 Conv)

1×1 필터로 **채널끼리만** 섞습니다. 공간 크기는 그대로 두고 채널 수를 늘리거나 줄일 때 씁니다(ResNet bottleneck, Inception).

### Depthwise Separable Convolution

**Depthwise conv(공간) → Pointwise conv(채널)** 순서로 일반 conv를 대체합니다. 결과는 일반 conv와 비슷하지만 파라미터와 연산량은 훨씬 적습니다.

$$
\frac{D_K^2 M D_F^2 + M N D_F^2}{D_K^2 M N D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}
$$

3×3 필터면 연산량이 약 **1/9**로 줄어듭니다. MobileNet은 이 연산을 쌓아서 기존 CNN보다 훨씬 적은 연산량으로 비슷한 정확도를 냈습니다. 유도 과정은 [MobileNet 글]({% post_url 2026-05-31-mobilenet %})에 자세히 정리했습니다.

### Group Convolution

입력 채널을 $$G$$개 그룹으로 나눠 그룹끼리만 conv를 합니다. 연산량이 $$1/G$$로 줄어듭니다. AlexNet이 GPU 두 대에 나눠 학습하려고 처음 썼고, ShuffleNet이 이를 적극적으로 활용합니다([8편]({% post_url 2026-06-02-cnn-architectures %})). `groups=C`인 극단적인 경우가 depthwise conv입니다.

## 10. Im2col: Convolution을 행렬 곱으로

Convolution을 정의대로 for문 여러 겹으로 구현하면 매우 느립니다. **Im2col**은 입력에서 필터가 지나가는 각 영역을 **한 열로 펼쳐** 2차원 행렬로 만들고, 필터도 행으로 펼쳐서 **행렬 곱 한 번**으로 convolution을 계산합니다.

- 겹치는 영역의 값이 중복 저장되어 **메모리를 더 쓰지만**
- 고도로 최적화된 행렬 곱(BLAS, cuBLAS)을 쓸 수 있어 **훨씬 빠릅니다**(구현에 따라 수십~수백 배).

실제 cuDNN 같은 라이브러리는 im2col 행렬을 메모리에 다 만들지 않고 필요한 부분만 그때그때 꺼내 쓰는 **implicit GEMM**, 3×3 conv의 곱셈 수를 줄이는 **Winograd** 알고리즘, 큰 커널용 **FFT** 방식 중 입력 크기에 맞는 것을 골라 씁니다. PyTorch의 `torch.backends.cudnn.benchmark = True`는 이 선택을 처음 몇 번 실행해 보고 가장 빠른 것으로 고정하는 옵션입니다.

```python
import torch
import torch.nn.functional as F

x = torch.randn(1, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)                       # (C_out, C_in, K, K)

cols = F.unfold(x, kernel_size=3, padding=1)       # im2col: (1, 3*3*3, 8*8)
out = w.view(16, -1) @ cols                        # (16, 27) @ (27, 64)
out = out.view(1, 16, 8, 8)

print(torch.allclose(out, F.conv2d(x, w, padding=1), atol=1e-5))   # True
```

## 11. 정리

| 개념 | 핵심 |
|---|---|
| Convolution | 작은 필터로 지역 패턴 검출, 가중치 공유 |
| Padding | 출력 크기 유지, 가장자리 정보 보존 |
| Stride | 필터 이동 간격, 크기 축소 |
| Pooling | Max/Avg로 크기 축소, 파라미터 없음 |
| 출력 크기 | $$\lfloor (W - K + 2P)/S \rfloor + 1$$ |
| Dilated | 연산량 그대로 receptive field 확장 |
| Transposed | Upsampling, checkerboard 주의 |
| Depthwise Separable | 공간·채널 분리로 연산량 약 1/9 |
| Im2col | Convolution을 행렬 곱으로 바꿔 가속 |

다음 글에서는 이 연산들을 조합해 만든 대표 CNN 구조들, **DenseNet, SENet, ShuffleNet, RepVGG**를 정리합니다.

## References

- LeCun et al., "Gradient-Based Learning Applied to Document Recognition," 1998.
- Dumoulin & Visin, "A guide to convolution arithmetic for deep learning," 2016. [arXiv:1603.07285](https://arxiv.org/abs/1603.07285)
- Yu & Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions," ICLR 2016.
- Odena et al., "Deconvolution and Checkerboard Artifacts," Distill 2016. <https://distill.pub/2016/deconv-checkerboard/>
- Howard et al., "MobileNets," 2017.
- Chellapilla et al., "High Performance Convolutional Neural Networks for Document Processing," 2006. (im2col)
