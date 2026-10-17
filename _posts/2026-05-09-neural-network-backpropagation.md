---
title: "[AI 기초 2] 신경망과 역전파: 숫자로 직접 따라가는 Backpropagation"
date: 2026-05-09 17:15:00 +0900
categories: [Neural Network, Basics]
tags: [neural-network, perceptron, mlp, backpropagation, chain-rule, gradient-descent]
description: 퍼셉트론과 다층 신경망의 구조, chain rule로 기울기를 구하는 역전파를 2-2-1 신경망 예제로 손으로 계산하고 NumPy, PyTorch autograd, 수치 미분으로 같은 값이 나오는지 검증합니다.
math: true
---

> [이전 글]({% post_url 2026-05-07-ml-overview %})에서 선형 모델을 경사 하강법으로 학습시켰습니다. 이번에는 선형 모델을 여러 층으로 쌓은 **신경망**을 만들고, 층이 많아도 모든 가중치의 기울기를 효율적으로 구하는 **역전파(Backpropagation)**를 작은 예제로 직접 계산해 봅니다.

## 1. 퍼셉트론: 신경망의 최소 단위

인공 신경망(ANN)은 사람의 신경 세포(Neuron)를 흉내 낸 모델입니다. 뉴런 하나는 입력에 가중치를 곱해 더하고, **활성화 함수**를 통과시켜 출력합니다.

$$
z = \sum_i w_i x_i + b,\qquad a = \phi(z)
$$

- $$w_i$$: 가중치 (입력의 중요도)
- $$b$$: 편향 (기준선 조정)
- $$\phi$$: 활성화 함수 (Sigmoid, ReLU 등)

### 단층 퍼셉트론의 한계: XOR

단층 퍼셉트론은 직선 하나로만 데이터를 나눌 수 있습니다. AND, OR은 가능하지만 **XOR**은 직선 하나로 나눌 수 없습니다. 이 문제를 풀려면 층을 쌓아야 합니다.

## 2. 다층 퍼셉트론 (MLP)

**입력층 → 은닉층 → 출력층**으로 이루어진 구조입니다. 은닉층이 입력 공간을 비선형적으로 변환해 주기 때문에, XOR처럼 복잡한 경계도 표현할 수 있습니다.

> 단, 층 사이에 **비선형 활성화 함수**가 꼭 있어야 합니다. 선형 함수만 쌓으면 $$W_3(W_2(W_1x)) = (W_3W_2W_1)x$$처럼 결국 선형 함수 하나와 같아집니다. 자세한 내용은 [3편 활성화 함수]({% post_url 2026-05-13-activation-functions %})에서 다룹니다.

### MLP는 어떻게 학습하나: 사격 표적지 비유

1. 처음 가중치는 **무작위로** 정해서 쏜다 (Initial weight).
2. 탄착점(출력)과 과녁 중심(정답)이 얼마나 떨어졌는지 잰다 (Cost).
3. 중심에 가까워지도록 조준(가중치)을 수정한다 (Gradient descent).
4. 반복한다.

3번에서 "각 가중치를 어느 방향으로 얼마나 고쳐야 하는지", 즉 $$\partial E / \partial w$$를 구하는 방법이 **역전파**입니다.

## 3. 학습의 두 단계

### Forward (순전파)

입력과 현재 가중치로 출력을 계산하고, 정답과 비교해 오차를 구합니다. 이번 예제에서는 비용 함수로 MSE(편의상 1/2을 곱함)를 씁니다.

$$
E = \frac{1}{2}(t - y)^2
$$

### Backward (역전파)

출력층부터 거꾸로 거슬러 가면서 **chain rule(연쇄 법칙)**로 각 가중치의 기울기를 구합니다.

$$
\frac{\partial E}{\partial w} = \frac{\partial E}{\partial y}\cdot\frac{\partial y}{\partial z}\cdot\frac{\partial z}{\partial w}
$$

핵심은 **뒤쪽 층에서 계산한 값을 앞쪽 층에서 재사용**한다는 점입니다. 덕분에 가중치가 수백만 개여도 forward 한 번, backward 한 번이면 모든 기울기를 구할 수 있습니다.

## 4. 숫자로 직접 계산해 보기

### 네트워크 설정

입력 2개, 은닉 뉴런 2개, 출력 1개인 네트워크입니다. 모든 뉴런은 Sigmoid를 씁니다.

| 항목 | 값 |
|---|---|
| 입력 | $$x_1 = 0.5,\ x_2 = 0.1$$ |
| 정답 | $$t = 0.9$$ |
| 입력 → $$h_1$$ | $$w_1 = 0.2,\ w_2 = 0.4$$ |
| 입력 → $$h_2$$ | $$w_3 = 0.3,\ w_4 = 0.1$$ |
| 은닉 → 출력 | $$w_5 = 0.6,\ w_6 = 0.5$$ |
| 편향 | $$b_1 = 0.1$$ (은닉), $$b_2 = 0.1$$ (출력) |
| 학습률 | $$\eta = 0.5$$ (예제라서 크게 잡음) |

### Step 1. Forward

$$
z_{h_1} = w_1x_1 + w_2x_2 + b_1 = 0.2(0.5) + 0.4(0.1) + 0.1 = 0.24,\quad h_1 = \sigma(0.24) = 0.5597
$$

$$
z_{h_2} = w_3x_1 + w_4x_2 + b_1 = 0.3(0.5) + 0.1(0.1) + 0.1 = 0.26,\quad h_2 = \sigma(0.26) = 0.5646
$$

$$
z_o = w_5h_1 + w_6h_2 + b_2 = 0.6(0.5597) + 0.5(0.5646) + 0.1 = 0.7181,\quad y = \sigma(0.7181) = 0.6722
$$

$$
E = \frac{1}{2}(0.9 - 0.6722)^2 = 0.02595
$$

### Step 2. 출력층 가중치 $$w_5$$의 기울기

세 조각으로 나눠 계산합니다.

**(1) 오차가 출력에 얼마나 민감한가**

$$
\frac{\partial E}{\partial y} = -(t - y) = y - t = -0.2278
$$

**(2) 출력이 Sigmoid 입력에 얼마나 민감한가**

Sigmoid의 미분은 자기 자신으로 표현됩니다. 그래서 계산이 간단합니다.

$$
\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big) \quad\Rightarrow\quad \frac{\partial y}{\partial z_o} = 0.6722 \times 0.3278 = 0.2203
$$

**(3) Sigmoid 입력이 $$w_5$$에 얼마나 민감한가**

$$z_o = w_5h_1 + w_6h_2 + b_2$$이므로

$$
\frac{\partial z_o}{\partial w_5} = h_1 = 0.5597
$$

**합치기**

앞의 두 조각을 묶어 출력층의 오차 신호 $$\delta_o$$라고 부릅니다.

$$
\delta_o = \frac{\partial E}{\partial y}\cdot\frac{\partial y}{\partial z_o} = -0.2278 \times 0.2203 = -0.05020
$$

$$
\frac{\partial E}{\partial w_5} = \delta_o \cdot h_1 = -0.05020 \times 0.5597 = -0.02810
$$

같은 방법으로 $$\partial E / \partial w_6 = \delta_o \cdot h_2 = -0.02834$$입니다.

기울기가 **음수**라는 건 $$w_5$$를 키우면 오차가 줄어든다는 뜻입니다. 실제로 출력 0.672가 정답 0.9보다 작으니 키워야 맞습니다.

### Step 3. 은닉층 가중치 $$w_1$$의 기울기

은닉층은 출력에 **간접적으로** 영향을 줍니다. 여기서 앞서 구한 $$\delta_o$$를 재사용합니다.

$$
\frac{\partial E}{\partial h_1} = \delta_o \cdot w_5 = -0.05020 \times 0.6 = -0.03012
$$

$$
\delta_{h_1} = \frac{\partial E}{\partial h_1}\cdot h_1(1 - h_1) = -0.03012 \times 0.2464 = -0.00742
$$

$$
\frac{\partial E}{\partial w_1} = \delta_{h_1}\cdot x_1 = -0.00742 \times 0.5 = -0.00371
$$

출력이 여러 개라면 은닉 뉴런 하나가 **모든 출력에 영향을 주므로**, 각 출력에서 오는 기울기를 **모두 더해야** 합니다. 이번 예제는 출력이 하나라서 한 경로만 있습니다.

나머지도 같은 방식으로 구하면 다음과 같습니다.

| 가중치 | 기울기 | 갱신 후 ($$w - \eta\,\partial E/\partial w$$) |
|---|---|---|
| $$w_1$$ | -0.00371 | 0.20186 |
| $$w_2$$ | -0.00074 | 0.40037 |
| $$w_3$$ | -0.00308 | 0.30154 |
| $$w_4$$ | -0.00062 | 0.10031 |
| $$w_5$$ | -0.02810 | 0.61405 |
| $$w_6$$ | -0.02834 | 0.51417 |

(편의상 bias는 고정했습니다.)

### Step 4. 결과 확인

갱신된 가중치로 다시 forward를 하면 $$y = 0.6757$$, $$E = 0.02515$$로 **오차가 줄었습니다.** 이 과정을 수천 번 반복하면 출력이 정답에 가까워집니다.

또 하나 눈여겨볼 점은 **은닉층 기울기가 출력층보다 훨씬 작다**는 것입니다(약 1/10). 이 예제에서는 Sigmoid 미분(여기서는 약 0.25)과 가중치 $$w_5 = 0.6$$, 입력 크기 $$x_1 = 0.5$$가 함께 곱해진 결과입니다. Sigmoid 미분의 최댓값이 0.25라서, 층을 지날 때마다 기울기에 0.25 이하의 값이 계속 곱해집니다. 층이 깊어지면 앞쪽 층은 거의 학습되지 않는데, 이것이 **Gradient Vanishing**입니다.

## 5. 코드로 검증하기

### NumPy로 직접 구현

```python
import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))

x = np.array([0.5, 0.1]); t = 0.9
W1 = np.array([[0.2, 0.4],     # h1 ← x1, x2
               [0.3, 0.1]])    # h2 ← x1, x2
W2 = np.array([0.6, 0.5]); b1, b2, lr = 0.1, 0.1, 0.5

for step in range(3):
    # forward
    h = sigmoid(W1 @ x + b1)
    y = sigmoid(W2 @ h + b2)
    E = 0.5 * (t - y) ** 2

    # backward
    delta_o = (y - t) * y * (1 - y)
    grad_W2 = delta_o * h
    delta_h = delta_o * W2 * h * (1 - h)
    grad_W1 = np.outer(delta_h, x)

    W2 -= lr * grad_W2
    W1 -= lr * grad_W1
    print(f"step {step}: y={y:.4f}, E={E:.5f}")
```

```text
step 0: y=0.6722, E=0.02595
step 1: y=0.6757, E=0.02515
step 2: y=0.6792, E=0.02438
```

step 0은 Step 1의 forward 결과, step 1은 Step 4에서 확인한 갱신 후 결과와 같습니다.

### PyTorch autograd로 같은 값 확인

PyTorch는 forward 연산을 기록해 두었다가 `backward()` 한 번으로 위 과정을 자동으로 해 줍니다.

```python
import torch

x = torch.tensor([0.5, 0.1]); t = torch.tensor(0.9)
W1 = torch.tensor([[0.2, 0.4], [0.3, 0.1]], requires_grad=True)
W2 = torch.tensor([0.6, 0.5], requires_grad=True)

h = torch.sigmoid(W1 @ x + 0.1)
y = torch.sigmoid(W2 @ h + 0.1)
E = 0.5 * (t - y) ** 2
E.backward()

print(W2.grad)   # tensor([-0.0281, -0.0283])
print(W1.grad)   # tensor([[-0.0037, -0.0007], [-0.0031, -0.0006]])
```

손으로 계산한 값과 정확히 같습니다.

### 수치 미분으로 한 번 더 검증하기 (Gradient Check)

역전파를 직접 구현했다면, 정의대로 계산한 **수치 미분**과 비교해 검증할 수 있습니다. 가중치를 아주 조금($$\epsilon$$) 올렸을 때와 내렸을 때의 오차 차이를 보는 것입니다.

$$
\frac{\partial E}{\partial w} \approx \frac{E(w + \epsilon) - E(w - \epsilon)}{2\epsilon}
$$

```python
import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))
x = np.array([0.5, 0.1]); t = 0.9
W1 = np.array([[0.2, 0.4], [0.3, 0.1]]); W2 = np.array([0.6, 0.5])

def loss(W1, W2):
    h = sigmoid(W1 @ x + 0.1)
    y = sigmoid(W2 @ h + 0.1)
    return 0.5 * (t - y) ** 2

eps = 1e-5
W1p, W1m = W1.copy(), W1.copy()
W1p[0, 0] += eps; W1m[0, 0] -= eps                 # w1만 살짝 올리고 내림
numeric = (loss(W1p, W2) - loss(W1m, W2)) / (2 * eps)
print(f"수치 미분  ∂E/∂w1 = {numeric:.8f}")         # -0.00371097
```

역전파로 구한 $$-0.00371097$$과 소수점 8자리까지 같습니다. 그런데 수치 미분은 가중치 **하나마다 forward를 두 번** 해야 해서, 가중치가 백만 개면 forward를 2백만 번 해야 합니다. 역전파는 forward 한 번 + backward 한 번(대략 forward의 2~3배 비용)으로 **모든** 기울기를 구합니다. 그래서 수치 미분은 학습이 아니라 구현을 **검증**할 때만 씁니다. PyTorch에는 이를 자동으로 해 주는 `torch.autograd.gradcheck`가 있습니다.

## 6. 선형 회귀로 보는 역전파

신경망이 아니어도 원리는 같습니다. $$\hat{y} = wx + b$$, $$E = (\hat{y} - y)^2$$이라면

$$
\frac{\partial E}{\partial w} = 2(\hat{y} - y)\cdot x,\qquad \frac{\partial E}{\partial b} = 2(\hat{y} - y)
$$

입력이 3개인 $$\hat{y} = w_1x_1 + w_2x_2 + w_3x_3 + b$$여도 각 $$w_i$$의 기울기는 $$2(\hat{y}-y)\,x_i$$로, **자기에게 곱해진 입력만큼** 책임을 나눠 갖습니다. 역전파는 이 단순한 규칙을 층마다 연쇄적으로 적용하는 것일 뿐입니다.

## 7. 정리

| 개념 | 핵심 |
|---|---|
| 퍼셉트론 | 가중합 + 활성화 함수 |
| MLP | 은닉층으로 비선형 경계 표현 (XOR 해결) |
| Forward | 출력과 오차 계산 |
| Backward | chain rule로 출력층부터 기울기 계산, 뒤쪽 결과를 앞쪽에서 재사용 |
| $$\delta$$ (오차 신호) | $$\partial E / \partial z$$, 층마다 전달되는 값 |
| Gradient Vanishing | Sigmoid 미분(최대 0.25)이 곱해지며 앞쪽 층 기울기가 사라짐 |

다음 글에서는 Gradient Vanishing을 크게 완화한 **ReLU**를 포함해, 여러 활성화 함수의 특징과 쓰임새를 정리합니다.

## References

- Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors," Nature 1986.
- Matt Mazur, "A Step by Step Backpropagation Example" — <https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/>
- [PyTorch – A Gentle Introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
