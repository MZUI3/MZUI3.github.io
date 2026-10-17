---
title: "[AI 기초 12] 이진 신경망(BNN) 총정리: XNOR-Net부터 Bi-Real, ReActNet, RB-Net까지"
date: 2026-06-25 21:42:00 +0900
categories: [Neural Network, Lightweight]
tags: [bnn, binary-neural-network, xnor-net, bi-real-net, real-to-binary, reactnet, sa-bnn, rb-net, ste, quantization, lightweight]
description: 가중치와 활성값을 ±1로 표현하는 이진 신경망(BNN)이 XNOR·popcount로 빨라지는 원리와, XNOR-Net부터 Bi-Real Net, ReActNet, RB-Net까지 정확도 격차를 줄여 온 연구 흐름을 정리합니다.
math: true
---

> [Quantization 글]({% post_url 2026-06-23-pruning-and-quantization %})에서 FP32를 INT8로 줄였다면, **BNN(Binary Neural Network)**은 그 끝까지 가서 가중치와 활성값을 **+1과 -1, 단 1bit**로 표현합니다. 메모리는 32배 줄고 곱셈은 비트 연산으로 바뀌지만, 정확도가 크게 떨어집니다. 이 글은 그 정확도 차이를 줄여 온 연구들을 흐름대로 묶어 정리합니다.

## 0. 연구 흐름 한눈에 보기

| 방향 | 연구 | 핵심 |
|---|---|---|
| 기초 | BinaryConnect, **BinaryNet** (2015–16) | 가중치·활성값 이진화, STE |
| 근사 오차 줄이기 | **XNOR-Net** (2016), **XNOR-Net++** (2019) | Scaling factor로 실수 conv에 근사 |
| 표현력 늘리기 | **Bi-Real Net** (2018) | 1-bit conv마다 실수 shortcut |
| 학습 방법 | **Real-to-Binary** (2020) | Teacher–student, attention matching, gating |
| 활성 분포 조절 | **ReActNet** (2020) | RSign, RPReLU, distributional loss |
| 기울기 개선 | **SA-BNN** (2021), Bop (2019) | 상태별 기울기, weight flip 제어 |
| 연산량 추가 감소 | **RB-Net** (2022) | Reshaped point-wise conv, Balanced activation |
| 온디바이스 학습 | **BNN Training on the Edge** (2021) | 학습 메모리 절감 |

## 1. 왜 이진화하나

FP32 가중치 하나는 32bit입니다. 이를 1bit로 바꾸면

- **메모리 32배 절약**
- 곱셈이 사라짐: $$(\pm 1) \times (\pm 1)$$은 부호만 바꾸면 됨
- 가중치와 입력이 **모두** ±1이면, 내적이 **XNOR + popcount(bitcount)**라는 비트 연산 두 개로 바뀝니다.

### XNOR와 popcount로 내적 계산하기

+1을 비트 1, -1을 비트 0으로 저장하면, 두 값의 곱이 +1인 경우(부호가 같음)는 XNOR가 1인 경우와 정확히 일치합니다. 길이 $$n$$인 두 벡터의 내적은

$$
\mathbf{a}\cdot\mathbf{b} = (\text{같은 부호 개수}) - (\text{다른 부호 개수}) = 2\cdot\text{popcount}\big(\text{XNOR}(\mathbf{a}, \mathbf{b})\big) - n
$$

```python
a = [+1, -1, +1, +1]
b = [+1, +1, +1, -1]
print(sum(x * y for x, y in zip(a, b)))          # 0

# 비트로 저장: +1 → 1, -1 → 0
abits, bbits = 0b1011, 0b1110
xnor = ~(abits ^ bbits) & 0b1111                  # 같은 자리가 1
print(2 * bin(xnor).count("1") - 4)               # 0, 같은 결과
```

64bit CPU라면 XNOR 명령 하나와 popcount 명령 하나로 64개의 곱셈-덧셈을 처리하는 셈입니다. 이 연산은 메모리 안에서 직접 하기도 좋아서 SRAM 기반 **In-Memory Computing** 가속기로도 이어집니다([13편]({% post_url 2026-06-28-edge-ai-hardware %})).

> "메모리 32배, 수십 배 빠름"은 **모든 층이 이진**이고 **비트 연산 전용 커널**로 실행할 때의 이론값입니다. 실제 BNN은 정확도를 위해 첫 conv와 마지막 FC를 실수로 두고, BN·scaling factor도 실수라서 압축률이 더 낮습니다. 또 아래의 PyTorch 코드처럼 학습할 때는 ±1 값을 **float 텐서에 담아 일반 conv로 계산**하므로 전혀 빨라지지 않습니다. 실제 속도 이득은 비트를 64개씩 묶어(bit-packing) XNOR·popcount로 계산하는 추론 엔진(예: Larq Compute Engine, daBNN)으로 배포할 때 생깁니다.
{: .prompt-warning }

### 대가

정확도가 크게 떨어집니다. AlexNet 기준 ImageNet top-1 정확도가 실수 모델 **56.6%**에서 가중치·활성값을 모두 이진화한 BinaryNet은 **27.9%**로 절반 이하가 됩니다. CIFAR-10, MNIST 같은 작은 데이터셋에서는 차이가 작지만, **데이터가 크고 다양할수록, 네트워크가 깊을수록** 실수 활성값과 이진 활성값의 오차가 쌓여 차이가 커집니다.

---

## 2. 기초: 이진화와 STE

### 이진화 함수

$$
x^b = \text{sign}(x) = \begin{cases} +1 & x \ge 0 \\ -1 & x < 0 \end{cases}
$$

- **Deterministic**: 위처럼 부호로 결정 (대부분 이것을 씀)
- **Stochastic**: $$\sigma(x)$$ 확률로 +1. 이론적으로 더 좋지만, 하드웨어에서 난수를 생성해야 해서 구현이 복잡합니다.

> 주의: 이진화는 **{0, 1}이 아니라 {-1, +1}**이어야 합니다. 그래야 곱셈이 XNOR로 바뀝니다.

### 문제: sign은 미분할 수 없다

sign의 미분은 0을 제외한 모든 곳에서 0이고(0에서는 임펄스 형태) 역전파하면 기울기가 전부 사라집니다. 그래서 [STE(Straight-Through Estimator)]({% post_url 2026-05-13-activation-functions %})를 씁니다. Forward는 sign, backward는 **Hard-tanh(clip)**의 미분을 씁니다.

$$
\frac{\partial x^b}{\partial x} \approx \mathbf{1}_{\lvert x\rvert \le 1}
$$

$$\lvert x\rvert$$가 너무 크면 기울기를 0으로 막아, 이미 확실히 +1/-1인 값은 더 밀지 않습니다.

### 실수 가중치(latent weight)는 유지한다

이진 가중치로 기울기를 계산해도, 그 기울기는 **부호를 바꿀 만큼 크지 않은** 경우가 대부분입니다. 그래서 학습 중에는 **실수 가중치를 따로 저장**해 두고 여기에 기울기를 누적한 뒤, forward 때마다 sign을 씌워 이진 가중치를 만듭니다. 추론 때는 이진 가중치만 남깁니다.

### PyTorch로 이진화 구현하기

`torch.clamp`로 값을 [-1, 1]로 제한하는 것은 간단하지만, 범위와 멀리 떨어진 값은 왜곡이 큽니다. 그래서 평균 크기 같은 **scaling factor**를 함께 씁니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryActivation(nn.Module):
    """Forward: sign(x), Backward: ApproxSign(Bi-Real)의 미분 (ReActNet 구현 방식)"""
    def forward(self, x):
        out_forward = torch.sign(x)
        mask1, mask2, mask3 = x < -1, x < 0, x < 1
        out1 = (-1) * mask1.float() + (x * x + 2 * x) * (1 - mask1.float())
        out2 = out1 * mask2.float() + (-x * x + 2 * x) * (1 - mask2.float())
        out3 = out2 * mask3.float() + 1 * (1 - mask3.float())
        # 값은 sign, 기울기는 out3의 것 (detach 트릭)
        return out_forward.detach() - out3.detach() + out3

class BinaryConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        alpha = w.abs().mean(dim=(1, 2, 3), keepdim=True).detach()   # 출력 채널별 scaling
        w_bin = alpha * torch.sign(w)
        w_bin = w_bin.detach() - w.detach() + w                        # STE (identity)
        return F.conv2d(x, w_bin, self.bias, self.stride, self.padding)
```

> `torch.sign(0)`은 0을 돌려주므로, 입력이 정확히 0인 원소는 ±1이 아니라 0이 됩니다. ReActNet 공식 구현도 `torch.sign`을 쓰는데, 실수 입력이 정확히 0이 될 확률은 매우 낮아 실제로 큰 문제가 되지는 않습니다. 엄밀하게 ±1만 원한다면 `torch.where(x >= 0, 1., -1.)`을 씁니다.
{: .prompt-info }

`detach()`는 연산 기록 없이 값만 가져와 **상수로 취급**합니다. `a.detach() - b.detach() + b`는 값은 `a`, 기울기는 `b`의 것이 되는 흔한 STE 구현 방식입니다. 이렇게 하면 `backward`를 직접 쓰지 않고 forward만으로 STE를 만들 수 있습니다. 반대로 XNOR-Net 원본 구현처럼 `torch.autograd.Function`으로 backward를 직접 지정하기도 합니다.

---

## 3. XNOR-Net: 이진 conv를 실수 conv에 가깝게

### Binary Weight Networks (BWN)

먼저 **가중치만** 이진화합니다. 가중치 필터 $$W$$를 이진 필터 $$B$$와 실수 scaling factor $$\alpha$$로 근사합니다.

$$
W \approx \alpha B,\qquad B \in \{-1, +1\}^n
$$

$$\lVert W - \alpha B\rVert^2$$를 최소화하는 해는 간단합니다.

$$
B^* = \text{sign}(W),\qquad \alpha^* = \frac{1}{n}\lVert W\rVert_{\ell 1} = \text{mean}(\lvert W\rvert)
$$

입력과의 conv는 곱셈 없이 덧셈·뺄셈으로 할 수 있어, **메모리 32배, 속도 약 2배** 이득입니다. AlexNet에서 정확도는 실수 모델과 거의 같습니다(56.8%).

### XNOR-Net

**입력도** 이진화해서 $$X \approx \beta H$$로 근사합니다. 그러면 conv가

$$
I * W \approx \big(\text{sign}(I) \circledast \text{sign}(W)\big) \odot K\alpha
$$

가 됩니다. $$\circledast$$는 **XNOR + bitcount**로 계산한 conv, $$\odot$$는 같은 크기 행렬의 원소끼리 곱하는 element-wise product입니다. $$K$$는 입력의 각 위치마다 구한 scaling factor $$\beta$$를 모은 행렬로, 입력의 절댓값을 채널 방향으로 평균 낸 뒤 평균 필터로 conv해서 구합니다.

- CPU 기준 약 **58배 빠름**, 메모리 **32배** 절약
- 단점: 새 입력이 들어올 때마다 $$K$$를 다시 계산해야 해서 연산이 늘어납니다. 실제 코드에서는 이 부분을 생략하는 경우가 많습니다.
- AlexNet ImageNet top-1 44.2% (BinaryNet 27.9%보다 크게 개선, 실수 56.6%와는 여전히 차이)

### 블록 순서를 바꾸기

일반 CNN의 **Conv → BN → Activation → Pooling** 순서를 그대로 쓰면 정보 손실이 큽니다. XNOR-Net은

$$
\text{BN} \to \text{BinActiv} \to \text{BinConv} \to \text{Pool}
$$

순서를 씁니다.

- **이진화 전에 BN**: 평균을 0으로 맞춰야 +1과 -1이 고르게 나와 **양자화 오차가 줄어듭니다.**
- **Pooling을 conv 뒤에**: 이진 입력에 바로 max pooling을 하면 대부분 +1이 되어 정보가 사라집니다.

## 4. XNOR-Net++: Scaling factor를 학습하자

XNOR-Net은 가중치를 직접 이진화하고 scaling factor를 해석적으로 계산하기 때문에 양자화 오차가 큽니다. XNOR-Net++(Bulat & Tzimiropoulos, 2019)는 가중치와 활성값의 scaling factor를 하나로 합친 $$\Gamma$$를 **역전파로 학습**합니다.

$$
I * W \approx \big(\text{sign}(I) \circledast \text{sign}(W)\big) \odot \Gamma
$$

- 학습이 쉽고, 추론 때 입력마다 $$K$$를 계산할 필요가 없어 **실제 연산 수가 줄어듭니다.**

$$\Gamma$$의 모양을 네 가지로 실험했습니다. 출력 채널 $$o = 100$$, 출력 크기 $$8 \times 8$$이라면

| Case | $$\Gamma$$의 형태 | 학습할 값의 수 |
|---|---|---|
| 1 | $$\alpha \in \mathbb{R}^{o}$$ (채널별) | 100 |
| 2 | $$\Gamma \in \mathbb{R}^{o \times h \times w}$$ (전체) | 6,400 |
| 3 | $$\alpha \otimes \beta \otimes \gamma$$, $$\alpha \in \mathbb{R}^o, \beta \in \mathbb{R}^h, \gamma \in \mathbb{R}^w$$ | 116 |
| 4 | $$\alpha \otimes \beta$$, $$\alpha \in \mathbb{R}^o, \beta \in \mathbb{R}^{h \times w}$$ | 164 |

**가장 많은 값을 가질 수 있는 Case 2가 오히려 정확도가 가장 낮았습니다.** 값이 너무 많으면 과적합되거나 학습이 어려워지므로, 적절히 분해된(factorized) 형태가 더 좋다는 것이 핵심입니다. 같은 조건에서 XNOR-Net보다 학습이 빠르고 정확도가 높으며, ResNet-18로 ImageNet top-1 약 57%를 달성했습니다.

---

## 5. Bi-Real Net: 실수 정보를 살려 두자

### 문제: 표현력이 너무 작다

이진 활성값은 +1, -1 두 가지뿐입니다. 3×3×32 이진 커널로 1-bit conv를 하면 출력은 -288부터 288까지 2 간격, 즉 **289가지** 값만 가질 수 있습니다. 실수 CNN이 표현하는 다양성(diversity)을 따라가기 어렵고, 층이 깊을수록 실수 활성값과 이진 활성값의 차이가 두드러집니다.

### 해법: 1-bit conv마다 실수 shortcut

Bi-Real Net(Liu et al., ECCV 2018)은 **BN을 지난 실수 활성값을, sign을 거치기 전에 shortcut으로 다음 블록 출력에 더합니다.**

- ResNet은 conv **2개**마다 shortcut을 두지만, Bi-Real은 **1-bit conv 하나**마다 둡니다.
- Shortcut으로 들어오는 값은 이전 층의 실수 출력 그대로이므로 **추가 연산·파라미터가 거의 없습니다.**
- 이진 conv 출력(289가지)에 실수 shortcut이 더해지면, 표현 가능한 값의 경우의 수가 **곱셈적으로** 늘어납니다.

### 학습 기법 세 가지

**(1) ApproxSign: sign의 더 좋은 근사**

Hard-tanh(clip) 대신 sign에 더 가까운 **2차 piecewise 함수**의 미분을 씁니다.

$$
\text{ApproxSign}(x) = \begin{cases} -1 & x < -1 \\ 2x + x^2 & -1 \le x < 0 \\ 2x - x^2 & 0 \le x < 1 \\ 1 & x \ge 1 \end{cases},\qquad
\frac{\partial}{\partial x} = \begin{cases} 2 + 2x & -1 \le x < 0 \\ 2 - 2x & 0 \le x < 1 \\ 0 & \text{otherwise} \end{cases}
$$

sign과의 차이(오차 면적)가 clip은 1이지만 ApproxSign은 **2/3**로 더 작습니다. 위 코드의 `BinaryActivation`이 바로 이 함수입니다.

**(2) Magnitude-aware gradient**

기존 방식은 이진 가중치의 기울기가 실수 가중치의 크기를 반영하지 못합니다. Bi-Real은 forward에서 $$\text{mean}(\lvert W\rvert)\cdot\text{sign}(W)$$를 쓰고 그 크기까지 미분에 반영해 실수 가중치를 업데이트합니다. Scaling factor를 전체 가중치 평균이 아니라 **층(출력 채널)마다** 구해 오차를 줄입니다.

**(3) 초기화**

기존 1-bit CNN은 ReLU를 쓴 실수 모델로 사전학습해 초기화했지만, **ReLU는 음수를 없애서** 이진 활성값(±1)과 분포가 맞지 않습니다. Bi-Real은 이진 활성과 모양이 비슷한 **clip 함수를 쓴 실수 모델**로 초기화합니다.

### 결과

세 기법 각각이 의미 있는 정확도 향상을 보였고, 최종적으로 Bi-Real Net-18은 ImageNet top-1 **56.4%**, Bi-Real Net-34는 **62.2%**로 XNOR-Net보다 크게 높아졌습니다.

---

## 6. Real-to-Binary Net: 학습 방법으로 격차 줄이기

Real-to-Binary(Martinez et al., ICLR 2020)는 구조보다 **학습 방법**에 집중합니다. 아이디어는 "Binary conv의 출력이 Real-valued conv의 출력과 같아지도록 학습하면 성능이 오르지 않을까"입니다.

### (1) Strong Baseline

블록 순서 재배치, **실수 downsample layer**, scaling factor, 2단계 학습(먼저 활성값만 이진화 → 가중치도 이진화) 등 기존 기법들을 조합해 기준이 되는 강한 baseline을 만듭니다.

### (2) Progressive Teacher–Student + Attention Matching

**Knowledge Distillation**(Hinton et al., 2015) 아이디어를 씁니다. 잘 학습된 큰 모델(teacher)의 지식을 작은 모델(student)에 전달합니다.

- **Hard target**: 원래 정답 label
- **Soft target**: teacher의 softmax 확률값. "고양이 0.7, 호랑이 0.2"처럼 클래스 간 유사도 정보가 담겨 있음

실수 모델에서 이진 모델로 **한 번에** 지식을 옮기면 차이가 너무 커서 잘 안 됩니다. 그래서 실수 → 활성값만 이진 → 완전 이진 순서로 **단계적으로** teacher–student를 이어 갑니다. 이때 출력뿐 아니라 중간 층의 **attention map**(feature map의 공간적 분포)도 teacher와 비슷해지도록 맞춥니다.

### (3) Data-driven Channel Re-scaling (Gating)

XNOR-Net의 scaling factor에서 영감을 받았습니다. 이진화하면 정보 손실이 크기 때문에, **이진화하기 전의 실수 입력**을 [SENet]({% post_url 2026-06-02-cnn-architectures %})과 같은 구조에 통과시켜 채널별 중요도를 구하고, 이를 binary conv의 출력에 곱해 **re-scaling**합니다. 입력에 따라 scaling이 달라지는 gating function입니다.

### 결과

ResNet-18로 ImageNet top-1 약 **65.4%**를 달성해, 실수 모델(약 69.3%)과의 격차를 약 4%p까지 줄였습니다.

---

## 7. ReActNet: 활성값 분포를 움직이자

ReActNet(Liu et al., ECCV 2020)의 관찰은 이렇습니다.

> 실수 네트워크에서 활성 함수를 조금 옮기거나 모양을 바꾸는 건 큰 의미가 없다. 하지만 **이진 네트워크는 출력이 ±1뿐이라, threshold를 조금만 옮겨도 출력이 크게 바뀐다.**

Threshold가 너무 낮으면 대부분 +1이 되고(too much features), 너무 높으면 대부분 -1이 됩니다(too few features). 그래서 **분포를 이동(shift)하고 모양을 바꾸는(reshape)** 학습 가능한 파라미터를 추가합니다.

### RSign: 채널별 threshold 학습

$$
x_i^b = \begin{cases} +1 & x_i > \alpha_c \\ -1 & x_i \le \alpha_c \end{cases}
$$

$$\alpha_c$$는 채널 $$c$$마다 학습되는 threshold로, 입력 feature map이 **최적으로 이진화되도록** 학습됩니다.

### RPReLU: 분포를 옮기고 모양을 바꾸기

$$
f(x_i) = \begin{cases} x_i - \gamma_c + \zeta_c & x_i > \gamma_c \\ \beta_c(x_i - \gamma_c) + \zeta_c & x_i \le \gamma_c \end{cases}
$$

1. 입력 분포를 $$-\gamma$$만큼 이동
2. 음수 영역에 $$\beta$$를 곱해 모양 변경 (PReLU처럼)
3. 출력 분포를 $$\zeta$$만큼 이동

모든 파라미터는 채널마다 학습되고, 다른 파라미터와 함께 역전파로 최적화됩니다. 학습 결과 RPReLU의 $$\beta$$는 이진 baseline에 PReLU를 쓴 경우보다 **실수 네트워크의 PReLU 값에 더 가까워졌습니다.** 실수 네트워크의 분포를 흉내 내게 됐다는 뜻입니다.

```python
class RSign(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.binact = BinaryActivation()
    def forward(self, x):
        return self.binact(x - self.alpha)

class RPReLU(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.zeta = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.prelu = nn.PReLU(ch)                      # β
    def forward(self, x):
        return self.prelu(x - self.gamma) + self.zeta
```

### Distributional Loss

일반적인 teacher–student처럼 정답 label을 맞추는 대신, 이진 student가 실수 teacher의 **출력 분포**를 모방하도록 KL divergence를 씁니다.

$$
\mathcal{L}_{distr} = -\frac{1}{n}\sum_{c}\sum_{i=1}^{n} p_c^{R}(X_i)\log\frac{p_c^{B}(X_i)}{p_c^{R}(X_i)}
$$

($$n$$: 배치 크기, $$c$$: 클래스, $$R$$: 실수 teacher, $$B$$: 이진 student)

### Baseline 구조

MobileNetV1을 이진화에 맞게 수정합니다.

- Depthwise conv를 일반 3×3 conv로 바꾸고, 모든 1-bit conv에 **파라미터 없는 shortcut**
- 입력·출력 채널이 같으면 **Normal block**
- 채널이 2배가 되는 곳은 **Reduction block**: 블록을 **복제(duplicate)해서 이어 붙여(concat)** 출력 채널을 맞춤으로써, 여기에도 shortcut을 둘 수 있게 함

### 결과

- Baseline만으로 ImageNet top-1 **61.1%** (87M OPs)
- RSign, RPReLU, distributional loss를 모두 쓴 **ReActNet-A: 69.4%**. 같은 구조의 실수(full-precision) 모델(72.4%)과 약 3%p 차이이고, 실수 ResNet-18(약 69.3%)과는 비슷한 정확도를 훨씬 적은 연산량으로 달성
- RSign과 RPReLU를 **둘 다** 쓸 때 가장 robust한 결과
- 변형: ReActNet-B/C는 downsampling의 1-bit 1×1 conv를 각각 group 실수 conv, 실수 conv로 바꿔 정확도를 더 올림

---

## 8. SA-BNN: +1과 -1의 기울기를 다르게

### Weight Flip 문제

학습률 때문에 실수 가중치가 0 근처에서 왔다 갔다 하면, 아주 작은 변화로도 이진 가중치가 -1과 +1 사이를 오갑니다. 이를 **weight flip**이라고 하며, 학습을 불안정하게 만듭니다.

기존 STE는 **+1 상태와 -1 상태를 구분하지 않고 같은 기울기**를 줍니다. 노이즈가 있으면 weight flip이 더 자주 일어납니다.

### State-Aware Gradient

SA-BNN(Liu et al., AAAI 2021)은 가중치는 학습 가능한 파라미터이므로 그대로 두고, **활성값**에 초점을 맞춥니다. 입력의 부호(상태)에 따라 **서로 다른 학습 가능한 계수** $$\tau^+, \tau^-$$를 기울기에 곱합니다.

- 초기값 실험에서 두 계수를 **서로 다른 값**(예: 0.4와 1)으로 두고 독립적으로 학습할 때 가장 좋았습니다(ResNet-18에서 59.2% vs 53.4%).
- 학습이 진행될수록 weight flip 비율이 줄어 **학습이 안정화**됩니다.
- 삼각형 모양의 도함수를 쓸 때 기울기가 잘려 나가는(clip) 비율이 크게 줄었습니다.
- Bi-Real Net baseline에 SA, 이진화 전 BN, 추가 shortcut을 더할수록 정확도가 계속 올랐고, ResNet-18/34/50 모두에서 당시 최고 성능을 기록했습니다.

### 관련 연구: Bop (Latent Weights Do Not Exist)

Helwegen et al.(2019)은 "BNN에 실수 latent weight는 사실 필요 없다"고 주장하며, weight flip만 직접 결정하는 **Binary Optimizer(Bop)**를 제안했습니다.

$$
m_t = (1 - \gamma)m_{t-1} + \gamma g_t
$$

$$
\text{flip } w \text{ if } \lvert m_t\rvert > \tau \text{ and } \text{sign}(m_t) = \text{sign}(w)
$$

- $$m_t$$: 기울기의 지수 이동 평균, $$\gamma$$: 적응률(클수록 변화에 빨리 반응), $$\tau$$: 임계값(클수록 flip이 덜 일어남)
- 과거 기울기의 **강도(strength)**가 충분히 일관되게 클 때만 뒤집습니다.
- Hyper-parameter가 $$\gamma, \tau$$ 두 개뿐이라 메모리를 적게 씁니다. 대신 적절한 값을 찾아야 합니다.

SA-BNN은 이와 달리 상태를 나눠서 기울기를 조절한다는 점이 다릅니다.

---

## 9. RB-Net: 연산량을 더 줄이면서 정확도도

RB-Net(Liu et al., 2022)은 두 개의 plug-and-play 모듈을 제안합니다.

### RPC (Reshaped Point-wise Convolution)

1. 입력 feature map을 공간적으로 **$$k$$개 조각으로 나누고**, 이를 **채널 방향으로 재배열**(공간 크기는 $$1/k$$, 채널은 $$k$$배)
2. **1×1 point-wise conv** 수행
3. 출력을 다시 재배열해 원래 크기로 복원

3×3 conv 대신 1×1 conv를 쓰므로 연산량이 크게 줄어듭니다. 원래 입력 기준으로 보면 커널 크기가 $$s \times s$$에서 $$k$$로 바뀐 셈이라, **$$k$$가 너무 작으면 receptive field가 좁아져** 특징 표현이 떨어질 수 있습니다. 그래서 모든 conv를 바꾸지 않고, ResNet의 **첫 7×7 conv**와 **basic block의 첫 3×3 conv**만 RPC로 대체합니다.

다른 방법과의 비교:

- **짝수 크기 커널**($$\sqrt{k} \times \sqrt{k}$$와 비슷): 비대칭 receptive field 때문에 픽셀이 한쪽으로 밀리는(pixel shift) 문제가 생기고, 층이 쌓이면 공간 정보가 왜곡됩니다. RPC는 padding 없이 비슷한 효과를 냅니다.
- **Dilated 커널**: 격자 효과(gridding)로 공간 연속성이 깨질 수 있습니다. RPC는 **모든 픽셀에 대해** 연산하므로 공간 정보가 더 잘 보존됩니다.

### BA (Balanced Activation)

ReActNet("활성값 분포가 BNN 표현력에 민감하다")과 Real-to-Binary("1-bit conv 후 SE로 채널별 scale")의 아이디어를 함께 씁니다. SE를 통과한 값을 **ReLU + BN**으로 이루어진 Balanced Activation에 넣어 분포를 고르게 맞춥니다.

### 결과

- **RPC**: baseline 대비 ResNet-18에서 top-1/top-5가 약 1.8%p 감소했지만, **메모리 11.65배 절약, 연산량 34.81배 감소**. ResNet-34에서는 정확도 0.6%p 감소로 **메모리 18.59배, 연산량 51.55배** 절감.
- **BA**: Real-to-Binary 대비 ResNet-18에서 top-1 +1.8%p, top-5 +1.4%p.
- 층별 binarization error(실수와 이진 가중치의 MSE)가 **깊은 층으로 갈수록 크게 작아졌습니다.** 분류에서는 깊은 층이 출력에 큰 영향을 주므로 성능이 보장됩니다.
- Learnable activation scale이 Real-to-Binary는 0.4 근처에 몰렸지만 RB-Net은 고르게 분포했고, 이진화 전후 중간 활성값 시각화도 더 선명했습니다(정보 손실이 적음).
- Bi-Real Net 대비 **약 1/3의 FLOPs로 약 10%p 높은 정확도**.
- 추가 실험: scaling factor를 양자화해도 성능 손실이 크지 않았지만, **skip connection을 양자화하면 성능이 크게 떨어졌습니다.** 완전히 양자화된 네트워크 설계는 앞으로의 과제입니다.

---

## 10. 추론을 넘어 학습까지: BNN Training on the Edge

지금까지는 **추론**을 가볍게 하는 연구였습니다. Wang et al.(2021)은 BNN의 **학습** 자체를 edge 기기에서 가능하게 만드는 방법을 제안합니다. 학습 메모리의 대부분은 역전파를 위해 저장해 둔 활성값이기 때문에, 여기를 줄입니다.

### BinaryNet의 학습 기법 (기반)

- **Shift-based BN**: BN은 학습 중 곱셈이 많습니다. 분산을 가장 가까운 2의 거듭제곱(AP2)으로 근사하면 곱셈을 **비트 shift**로 바꿀 수 있습니다. 예: $$AP2(4) = 2^2$$이므로 4를 곱하는 대신 왼쪽으로 2bit shift.
- **Shift-based AdaMax**: Optimizer의 곱셈도 같은 방식으로 shift로 바꿉니다.
- **첫 번째 층**: 입력이 RGB 3채널뿐이라 실수로 둬도 연산 부담이 크지 않습니다.

### 메모리를 줄이는 근사들

1. **L1-norm BN**: BN의 분산 계산에서 제곱과 제곱근 대신 **L1 norm**을 써서 계산 비용을 줄입니다. BNN은 BN 직후 바로 이진화하므로 이 근사의 영향이 작습니다.
2. **저장할 활성값을 이진화**: 역전파용으로 저장하는 활성값 $$X$$를 이진값 $$\hat{X}$$로 저장해 **메모리 약 32배 절약**
3. **이진 가중치 기울기 양자화**: 저장 공간 약 32배 감소
4. **Bop optimizer 사용**: 기존 optimizer는 latent weight, momentum 등 많은 상태를 저장해야 하지만 Bop은 threshold와 적응률만 필요해 메모리가 적습니다.

### 구현과 결과

- 정확도와 메모리 모두에서 유리했고, **배치 크기가 커져도 메모리 증가가 작았습니다.**
- Naive C++ 라이브러리와 **CBLAS 가속** 버전을 구현했습니다. CBLAS(선형대수 저수준 루틴 모음)를 쓰면 일반 구현보다 **27.66~58.34배 빠르고 학습 시간이 약 10배 감소**하며, 메모리 트래픽이 줄어 전력 소비도 줄었습니다. 대신 1.59~2.08배의 추가 메모리가 필요해, 배치 크기에 따라서는 구현이 불가능한 경우도 있습니다.

---

## 11. 정리

| 연구 | 문제 | 해결 |
|---|---|---|
| BinaryNet | sign 미분 불가 | STE (Hard-tanh) |
| XNOR-Net | 이진 conv 근사 오차 | Scaling factor $$\alpha, K$$, BN→BinAct→BinConv→Pool |
| XNOR-Net++ | scaling 계산 비용·오차 | $$\Gamma$$를 학습, 분해된 형태가 최적 |
| Bi-Real | 표현력 부족 | 1-bit conv마다 실수 shortcut, ApproxSign |
| Real-to-Binary | 학습 방법 | 단계적 teacher–student, attention matching, SE gating |
| ReActNet | 활성값 분포 민감 | RSign, RPReLU, distributional loss |
| SA-BNN | weight flip | 상태별 학습 가능한 기울기 계수 |
| RB-Net | 연산량 | RPC, Balanced Activation |
| Edge Training | 학습 메모리 | 활성값 이진 저장, L1 BN, Bop |

ImageNet top-1 기준으로 BinaryNet(AlexNet) 27.9%에서 ReActNet 69.4%까지, 몇 년 만에 같은 구조의 실수 모델과의 차이가 3%p 수준으로 줄었습니다. (단, 네트워크 구조가 AlexNet → ResNet → 수정된 MobileNet으로 바뀌었으므로 숫자의 향상이 모두 이진화 기법 덕분은 아닙니다.)

다음 글에서는 이런 경량 모델을 실제로 올릴 **하드웨어**(SRAM, PIM, RISC-V)와 **온디바이스 학습**을 다룹니다.

## References

- Courbariaux et al., "BinaryConnect," NeurIPS 2015. / Hubara et al., "Binarized Neural Networks," NeurIPS 2016. [arXiv:1602.02830](https://arxiv.org/abs/1602.02830)
- Rastegari et al., "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks," ECCV 2016. [arXiv:1603.05279](https://arxiv.org/abs/1603.05279)
- Bulat & Tzimiropoulos, "XNOR-Net++: Improved Binary Neural Networks," BMVC 2019.
- Liu et al., "Bi-Real Net: Enhancing the Performance of 1-bit CNNs," ECCV 2018.
- Martinez et al., "Training Binary Neural Networks with Real-to-Binary Convolutions," ICLR 2020.
- Liu et al., "ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions," ECCV 2020. [arXiv:2003.03488](https://arxiv.org/abs/2003.03488)
- Liu et al., "SA-BNN: State-Aware Binary Neural Network," AAAI 2021.
- Helwegen et al., "Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization," NeurIPS 2019.
- Liu et al., "RB-Net: Training Highly Accurate and Efficient Binary Neural Networks With Reshaped Point-Wise Convolution and Balanced Activation," IEEE TCSVT 2022.
- Wang et al., "Enabling Binary Neural Network Training on the Edge," 2021.
- Hinton et al., "Distilling the Knowledge in a Neural Network," NeurIPS Workshop 2014.
- <https://github.com/liuzechun/ReActNet>, <https://github.com/itayhubara/BinaryNet.pytorch>
- [Larq Compute Engine](https://github.com/larq/compute-engine) (BNN 추론 엔진)
