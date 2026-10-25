---
title: "[AI 기초 3] 활성화 함수 총정리: Sigmoid부터 ReLU 가족, STE까지"
description: 활성화 함수가 왜 비선형이어야 하는지, Sigmoid·tanh·ReLU 계열·GELU의 장단점과 쓰임새, 미분할 수 없는 sign 함수를 STE로 학습하는 방법을 PyTorch 실험과 함께 정리합니다.
date: 2026-05-13 20:44:00 +0900
categories: [Neural Network, Basics]
tags: [activation-function, sigmoid, tanh, relu, leaky-relu, prelu, elu, selu, softmax, ste]
math: true
---

> [이전 글]({% post_url 2026-05-08-neural-network-backpropagation %})에서 역전파를 할 때 **활성화 함수의 미분값이 계속 곱해진다**는 것을 확인했습니다. 그렇다면 어떤 활성화 함수를 쓰느냐가 학습이 잘되느냐를 좌우하게 됩니다. 이번 글에서는 대표적인 활성화 함수들을 장단점과 쓰임새 중심으로 정리합니다.

## 1. 활성화 함수란

뉴런이 받은 가중합 $$z = \sum w_ix_i + b$$를 **출력 신호로 바꾸는 함수**입니다.

$$
a = \phi(z)
$$

### 왜 비선형이어야 하나

활성화 함수가 선형 $$h(x) = cx$$라면, 3층을 쌓아도

$$
y(x) = h(h(h(x))) = c^3x
$$

로 결국 **선형 함수 하나**와 같습니다. 층을 아무리 쌓아도 표현력이 늘지 않으니 은닉층에는 반드시 비선형 함수를 씁니다.

### 역전파에서의 역할

[2편]({% post_url 2026-05-08-neural-network-backpropagation %})에서 본 것처럼, 은닉 뉴런 $$h$$의 오차 신호는

$$
\delta_h = \frac{\partial E}{\partial h}\cdot\phi'(z_h)
$$

로 **활성화 함수의 미분 $$\phi'$$이 곱해진 채** 앞쪽 층으로 전달됩니다. 층이 $$L$$개면 $$\phi'$$이 $$L$$번 곱해집니다. (정확히는 각 층의 가중치 행렬 $$W$$도 함께 곱해지므로, 기울기 크기는 "$$W$$와 $$\phi'$$의 곱"이 1보다 큰지 작은지에 달려 있습니다. 그래서 활성화 함수와 함께 **가중치 초기화**도 중요합니다.)

- $$\lvert\phi'\rvert < 1$$이 계속 곱해지면 → **Gradient Vanishing** (기울기 소실)
- $$\lvert\phi'\rvert > 1$$이 계속 곱해지면 → **Gradient Exploding** (기울기 폭주)

그래서 활성화 함수를 고를 때 **미분값의 크기**가 가장 중요한 기준이 됩니다.

## 2. Step Function과 Linear Function

### Step Function

$$
\phi(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}
$$

최초의 퍼셉트론에 쓰인 함수입니다. 출력이 0 또는 1이라 **정보가 너무 많이 희석**되고, **미분값이 거의 모든 곳에서 0**이라 역전파로 학습할 수 없습니다.

### Linear Function

$$
\phi(z) = z
$$

입력을 그대로 출력합니다. 앞서 본 것처럼 층을 쌓는 의미가 없어지고, 이상치가 그대로 전달됩니다. **회귀 문제의 출력층**에서만 씁니다.

## 3. Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}},\qquad \sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)
$$

**장점**
- 출력이 0~1 사이의 연속값이라 **확률처럼 해석** 가능
- 이상치가 들어와도 0 또는 1에 수렴할 뿐이라 출력이 폭주하지 않음
- 미분을 자기 자신으로 쉽게 계산 가능

**단점**
1. **Gradient Vanishing**: $$\sigma'$$의 최댓값이 $$z=0$$에서 **0.25**입니다. 입력이 조금만 커지거나 작아져도 미분값이 0에 가까워지고, 층을 하나 지날 때마다 기울기에 **많아야 0.25가 곱해집니다.** 10층이면 $$0.25^{10} \approx 10^{-6}$$입니다.
2. **출력이 항상 양수 (not zero-centered)**: 출력 평균이 0.5 근처라 다음 층의 입력이 모두 양수입니다. 그러면 한 뉴런의 가중치 기울기들이 **모두 같은 부호**가 되어, 가중치가 지그재그로 움직이며 학습이 느려집니다.
3. `exp` 연산이 상대적으로 비쌉니다.

**쓰임새**: 은닉층에는 거의 쓰지 않고, **이진 분류의 출력층**이나 LSTM의 **게이트**처럼 0~1 값이 필요한 곳에 씁니다.

## 4. Tanh

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1,\qquad \tanh'(z) = 1 - \tanh^2(z)
$$

Sigmoid를 -1~1 범위로 늘린 모양입니다.

**Sigmoid보다 나은 점**
- **출력 중심이 0 (zero-centered)** → 지그재그 문제가 없음
- 미분 최댓값이 **1** (Sigmoid의 4배) → 기울기 소실이 덜함

**한계**: 입력 절댓값이 크면 여전히 미분이 0에 가까워져 **Gradient Vanishing이 남아 있습니다.**

**쓰임새**: 은닉층에서 Sigmoid 같은 역할이 필요할 때, 그리고 **RNN의 기본 활성화 함수**입니다. RNN은 같은 가중치를 시간 방향으로 계속 곱하기 때문에, 출력이 -1~1로 제한되고 기울기가 비교적 오래 유지되는 tanh가 적합합니다([10편 RNN]({% post_url 2026-06-08-rnn-lstm-gru %}) 참고).

## 5. Softmax

$$
\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

세 개 이상의 클래스를 분류하는 **출력층** 함수입니다. "k번째 점수 / 전체 점수"의 형태로, 모든 출력의 합이 1인 **확률 분포**를 만듭니다.

- 지수 함수가 점수 차이를 확대하므로, 가장 큰 점수에 확률이 몰리는 경향이 있습니다. (이름처럼 max를 "부드럽게" 근사한 함수입니다.)
- Cross Entropy와 함께 쓰면 손실을 **logit $$z$$에 대해** 미분한 값이 $$\hat{y} - y$$로 아주 간단해집니다.

### 오버플로우 주의

$$z$$가 크면 $$e^{z}$$가 표현 범위를 넘어갑니다. 그래서 실제 구현은 **최댓값을 빼고** 계산합니다. 결과는 수학적으로 같습니다.

```python
import numpy as np

def softmax(z):
    z = z - np.max(z)            # 오버플로우 방지
    e = np.exp(z)
    return e / e.sum()

print(softmax(np.array([1000., 1001., 1002.])))   # [0.090 0.245 0.665]
```

## 6. ReLU

$$
\text{ReLU}(z) = \max(0, z),\qquad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}
$$

2010년대 딥러닝 부흥을 이끈 함수 중 하나로, 은닉층의 사실상 기본값입니다.

**장점**
- 양수 영역에서 미분이 **항상 1** → 활성화 함수 때문에 기울기가 줄어드는 일은 없음 (Gradient Vanishing을 크게 **완화**)
- 출력 범위가 위로 열려 있어 특정 값에 포화(saturate)되지 않음
- `max` 연산 하나라 **매우 빠름**
- 음수를 0으로 만들어 **희소한(sparse) 표현**을 만듦

**단점**
1. **Dying ReLU**: 입력이 음수면 출력도 기울기도 0입니다. 어떤 뉴런이 모든 입력에 대해 음수를 내게 되면 **영원히 업데이트되지 않습니다.** 학습률이 크거나 bias가 크게 음수로 가면 자주 생깁니다.
2. 0에서 미분 불가능 (실제로는 0 또는 1로 정의해서 사용)
3. 출력이 0 이상이라 zero-centered가 아님

### 직접 확인: 20층 네트워크의 첫 층 기울기

"ReLU면 기울기 소실이 없다"는 말이 어디까지 맞는지 실험해 봅니다. 폭 64, 깊이 20인 MLP의 첫 번째 층에 도달하는 기울기 크기를 비교합니다.

```python
import torch
import torch.nn as nn

def first_layer_grad(act, init=None, depth=20, width=64):
    torch.manual_seed(0)
    layers = []
    for _ in range(depth):
        linear = nn.Linear(width, width)
        if init == "he":
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")  # ReLU용 He 초기화
        layers += [linear, act()]
    model = nn.Sequential(*layers, nn.Linear(width, 1))
    x = torch.randn(256, width)
    model(x).pow(2).mean().backward()           # 아무 손실이나 만들어 역전파
    return model[0].weight.grad.norm().item()   # 첫 번째 층 가중치의 기울기 크기

print(f"Sigmoid           : {first_layer_grad(nn.Sigmoid):.2e}")
print(f"Tanh              : {first_layer_grad(nn.Tanh):.2e}")
print(f"ReLU (기본 초기화): {first_layer_grad(nn.ReLU):.2e}")
print(f"ReLU (He 초기화)  : {first_layer_grad(nn.ReLU, init='he'):.2e}")
```

```text
Sigmoid           : 7.07e-19
Tanh              : 2.85e-07
ReLU (기본 초기화): 3.35e-10
ReLU (He 초기화)  : 5.93e-02
```

(PyTorch 2.13 CPU에서 실행)

Sigmoid는 기울기가 사실상 0입니다. 그런데 ReLU도 **PyTorch의 기본 초기화로는 기울기가 거의 사라집니다.** `nn.Linear`의 기본 초기화는 ReLU를 가정한 값보다 가중치를 작게 잡기 때문에, 층을 지날 때마다 신호가 조금씩 줄어듭니다. ReLU에 맞춘 **He 초기화**(분산 $$2/n_{in}$$)를 쓰면 기울기가 수억 배 커집니다. 결국 깊은 네트워크를 가능하게 한 것은 ReLU 하나가 아니라 **ReLU + 알맞은 초기화 + (이후의) 정규화·잔차 연결**의 조합입니다.

### CNN에서 ReLU를 쓰는 이유

Convolution을 거친 feature map은 "이 위치에 필터와 비슷한 패턴이 얼마나 있는가"를 나타내는 값입니다. ReLU는 이를 **"패턴이 있다(양수) / 없다(0)"**로 정리해 주는 역할을 합니다. 깊은 CNN에서 Sigmoid를 쓰면 기울기 소실로 학습이 잘 안 되기 때문에 ReLU가 표준이 됐습니다.

## 7. ReLU 가족

Dying ReLU를 해결하기 위해 음수 영역에 작은 기울기를 주는 변형들이 나왔습니다.

### Leaky ReLU

$$
\phi(z) = \begin{cases} z & z > 0 \\ 0.01z & z \le 0 \end{cases}
$$

음수 영역에 아주 작은 기울기(0.01)를 줘서 뉴런이 죽지 않게 합니다.

### PReLU (Parametric ReLU)

$$
\phi(z) = \begin{cases} z & z > 0 \\ az & z \le 0 \end{cases}
$$

음수 기울기 $$a$$를 **학습 가능한 파라미터**로 둡니다. 층마다(또는 채널마다) 적절한 $$a$$가 학습됩니다. 표현력이 늘어나는 대신 작은 데이터셋에서는 과적합 위험이 있습니다.

### ELU (Exponential Linear Unit)

$$
\phi(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \le 0 \end{cases}
$$

음수 영역이 부드러운 곡선이라 출력 평균이 0에 가까워지고, 기본값 $$\alpha = 1$$이면 **0에서도 미분이 끊기지 않습니다**(왼쪽 미분 $$\alpha e^0 = \alpha$$, 오른쪽 미분 1). 입력이 크게 음수이면 출력이 $$-\alpha$$로 포화되므로 음수 쪽 노이즈에도 덜 흔들립니다. 대신 음수 영역에서 `exp`를 계산해야 해서 ReLU보다 느립니다.

### SELU (Scaled ELU)

$$
\phi(z) = \lambda\begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \le 0 \end{cases},\qquad \lambda \approx 1.0507,\ \alpha \approx 1.6733
$$

ELU에 정해진 배율 $$\lambda$$를 곱한 함수입니다. 두 상수는 "입력의 평균이 0, 분산이 1이면 출력도 평균 0, 분산 1이 되도록" 유도된 값이라, 층을 거듭 통과해도 활성값 분포가 저절로 평균 0, 분산 1 근처로 유지됩니다(**self-normalizing**). BatchNorm 없이도 깊은 MLP를 학습할 수 있다는 것이 SELU 논문의 주장입니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def stats_after(act, depth=20, width=256):
    torch.manual_seed(0)
    x = torch.randn(1024, width)
    for _ in range(depth):
        linear = nn.Linear(width, width, bias=False)
        nn.init.normal_(linear.weight, std=width ** -0.5)     # LeCun normal 초기화
        x = act(linear(x))
    return x.mean().item(), x.std().item()

for name, act in [("ReLU", F.relu), ("ELU", F.elu), ("SELU", F.selu)]:
    m, s = stats_after(act)
    print(f"{name:4s}  평균 {m:+.3f}  표준편차 {s:.3f}")
```

```text
ReLU  평균 +0.001  표준편차 0.001
ELU   평균 +0.008  표준편차 0.124
SELU  평균 -0.007  표준편차 0.979
```

(PyTorch 2.13 CPU에서 실행)

20층을 지난 뒤 ReLU와 ELU는 활성값이 0 근처로 쪼그라들지만, SELU는 표준편차가 1 가까이 유지됩니다. 단, 이 성질은 **LeCun normal 초기화**와 **fully connected 층**을 전제로 하고, Dropout도 분포를 유지하는 AlphaDropout을 써야 합니다. CNN이나 residual 구조에서는 이 보장이 그대로 성립하지 않아 널리 쓰이지는 않습니다.

## 8. GELU와 STE

### GELU (Gaussian Error Linear Unit)

$$
\text{GELU}(z) = z\,\Phi(z) \approx 0.5\,z\Big(1 + \tanh\Big[\sqrt{2/\pi}\,\big(z + 0.044715\,z^3\big)\Big]\Big)
$$

$$\Phi$$는 표준정규분포의 누적분포함수입니다. ReLU가 입력을 "0보다 크면 통과, 작으면 차단"으로 딱 자른다면, GELU는 **입력이 클수록 통과시킬 확률이 높아지는** 방식으로 부드럽게 거릅니다.

```python
z = torch.tensor([-3.0, -0.75, 0.0, 1.0, 3.0])
print(F.gelu(z))                        # tensor([-0.0040, -0.1700,  0.0000,  0.8413,  2.9960])
print(F.gelu(z, approximate="tanh"))    # tensor([-0.0036, -0.1700,  0.0000,  0.8412,  2.9964])
```

- 모든 곳에서 미분 가능하고, 큰 양수에서는 ReLU처럼 거의 $$z$$ 그대로입니다.
- 음수 영역에서 0으로 바로 잘리지 않고 작은 음수 값(최솟값 약 -0.17, $$z \approx -0.75$$)을 냅니다.
- BERT, GPT, ViT 등 **Transformer의 사실상 표준** 활성화 함수입니다. tanh 근사는 원래 식과 거의 같으면서 계산이 빠릅니다.

### 미분할 수 없는 sign 함수와 STE

[이진 신경망(BNN)]({% post_url 2026-06-17-binary-neural-networks %})은 가중치와 활성값을 ±1로 만들기 위해 **sign 함수**를 활성화 함수처럼 씁니다. 그런데 sign의 미분은 0을 제외한 모든 곳에서 0이라, 그대로 역전파하면 기울기가 전부 사라집니다(2절의 Step Function과 같은 문제).

**STE(Straight-Through Estimator)**는 forward에서는 sign을 그대로 쓰고, backward에서는 **sign이 없었던 것처럼 기울기를 통과**시킵니다. 보통은 $$\lvert x\rvert \le 1$$에서만 통과시키는 Hard-tanh의 미분을 씁니다.

$$
\text{forward: } y = \text{sign}(x),\qquad \text{backward: } \frac{\partial y}{\partial x} \approx \mathbf{1}_{\lvert x\rvert \le 1}
$$

```python
class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1.0, -1.0)          # 0도 +1로 보내 ±1만 출력

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out * (x.abs() <= 1).float()       # Hard-tanh의 미분: |x| ≤ 1에서만 통과

x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
y = SignSTE.apply(x)
y.sum().backward()
print(y.detach(), x.grad)   # tensor([-1., -1.,  1.,  1.,  1.]) tensor([0., 1., 1., 1., 0.])
```

$$\lvert x\rvert$$가 1보다 큰 값은 이미 확실히 +1/-1이므로 기울기를 막아 더 밀지 않습니다.

> `torch.sign`을 그대로 쓰면 입력이 정확히 0인 원소가 0으로 남아 "±1만 나온다"는 BNN의 전제가 깨집니다. 실제 BNN 구현에서 자주 나오는 실수입니다.
{: .prompt-warning }

이 아이디어는 BNN 글에서 Bi-Real Net의 ApproxSign, ReActNet의 RSign으로 발전합니다.

## 9. 어디에 무엇을 쓰나

| 위치 | 함수 | 이유 |
|---|---|---|
| 은닉층 (CNN, MLP) | **ReLU** | 빠르고 기울기 소실이 적음 |
| 은닉층 (Dying ReLU가 문제일 때) | Leaky ReLU, PReLU, ELU | 음수 영역에도 기울기 |
| 은닉층 (Transformer) | GELU | 사실상 표준 |
| 은닉층 (RNN) | tanh | 출력 범위 제한, 기울기 유지 |
| 게이트 (LSTM, GRU, SE block) | Sigmoid | 0~1 비율 필요 |
| 출력층 (이진 분류) | Sigmoid | 확률 하나 |
| 출력층 (다중 분류) | Softmax | 확률 분포 |
| 출력층 (회귀) | Linear | 값 그대로 |
| BNN | sign + STE (Hard-tanh 등) | ±1 출력, 역전파 우회 |

## 10. 정리

- 활성화 함수는 신경망에 **비선형성**을 주고, 역전파에서 **미분값이 곱해지는 통로**입니다.
- Sigmoid·tanh는 포화 구간에서 미분이 0에 가까워져 **기울기 소실**을 일으킵니다.
- ReLU는 양수 영역 미분이 1이라 깊은 네트워크를 가능하게 했지만 **Dying ReLU** 문제가 있고, 이를 Leaky/PReLU/ELU 등이 보완합니다.
- 미분할 수 없는 함수(sign)는 **STE**로 기울기를 우회해서 흘립니다.

다음 글에서는 이렇게 구한 기울기로 **실제로 가중치를 어떻게 업데이트할지** 결정하는 Optimizer와, 학습률을 시간에 따라 조절하는 Scheduler를 다룹니다.

## References

- Nair & Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines," ICML 2010.
- He et al., "Delving Deep into Rectifiers (PReLU)," ICCV 2015.
- Clevert et al., "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)," ICLR 2016.
- Klambauer et al., "Self-Normalizing Neural Networks," NeurIPS 2017.
- Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons," 2013. (STE) [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
- Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)," 2016. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
- [PyTorch – Non-linear Activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- Charu C. Aggarwal, 『신경망과 심층학습』, 제이펍, 2019.
