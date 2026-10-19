---
title: "[AI 기초 4] Optimizer와 Learning Rate Scheduler: SGD부터 Adam, 1cycle까지"
date: 2026-05-15 20:05:00 +0900
categories: [Neural Network, Basics]
tags: [optimizer, sgd, momentum, nag, adagrad, rmsprop, adam, learning-rate, scheduler]
description: SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW가 '방향'과 '보폭'을 각각 어떻게 개선했는지와, Step·Plateau·Cosine·Warm-up·1cycle 같은 학습률 스케줄러를 PyTorch 실험과 함께 정리합니다.
math: true
---

> 역전파로 기울기를 구했다면, 이제 그 기울기로 **가중치를 어느 방향으로 얼마나 움직일지** 정해야 합니다. 이 역할을 하는 것이 **Optimizer**이고, 학습이 진행되면서 보폭을 바꿔주는 것이 **Learning Rate Scheduler**입니다.

## 1. Optimizer란

Optimizer는 비용 함수를 최소로 만드는 파라미터를 찾아가는 **길 찾기 알고리즘**입니다. 영통역에서 단국대학교까지 가는 길이 여러 개이듯, 최솟값까지 가는 방법도 여러 가지입니다.

모든 Optimizer는 경사 하강법에서 출발하고, 개선 방향은 크게 두 갈래입니다.

$$
\theta_{t+1} = \theta_t - \underbrace{\eta}_{\text{보폭}}\cdot\underbrace{\nabla_\theta L(\theta_t)}_{\text{방향}}
$$

| 개선 방향 | 아이디어 | 알고리즘 |
|---|---|---|
| **방향(gradient)** | 과거 기울기의 관성을 이용 | Momentum, NAG |
| **보폭(step size)** | 파라미터마다 학습률을 다르게 | AdaGrad, RMSProp |
| **둘 다** | 관성 + 적응형 학습률 | Adam, Nadam, RAdam |

먼저 용어를 정리합니다.

- **Epoch**: 전체 데이터셋을 한 번 다 학습
- **Iteration (step)**: 파라미터를 한 번 업데이트
- **Mini-batch**: 데이터셋을 batch size만큼 쪼갠 묶음

데이터 60,000개, batch size 64라면 1 epoch = 약 938 iteration입니다.

## 2. 얼마나 많은 데이터로 기울기를 구할까

### Batch Gradient Descent

iteration마다 **전체 데이터**로 기울기를 구합니다. 방향은 정확하지만 한 번 업데이트하는 데 너무 오래 걸리고, 큰 데이터셋은 메모리에 올릴 수도 없습니다.

### Stochastic Gradient Descent (SGD)

**데이터 한 개**로 기울기를 구해 바로 업데이트합니다.

| 장점 | 단점 |
|---|---|
| 업데이트가 매우 빠름 | 진폭이 크고 불안정하게 움직임 |
| 메모리 사용량이 매우 적음 | 최적점 근처에서 계속 흔들려 정확히 수렴하기 어려움 |
| 큰 데이터셋도 학습 가능 | 한 번에 하나씩이라 GPU 병렬 연산을 활용 못 함 |
| 노이즈 덕분에 **local minimum을 빠져나올 가능성** | |

### Mini-batch SGD

실제로는 둘의 중간인 **mini-batch**(보통 32~512개)로 기울기를 구합니다. GPU 병렬성을 살리면서 노이즈도 적당히 유지합니다. 딥러닝에서 "SGD"라고 하면 보통 이 방식을 말합니다.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

## 3. 방향 개선: Momentum과 NAG

### Momentum

경사를 굴러가는 볼링공을 생각해 봅시다. 처음엔 느리지만 같은 방향으로 계속 내려가면 점점 빨라집니다. Momentum은 이 **관성**을 업데이트에 추가합니다.

$$
v_t = \gamma v_{t-1} + \eta\nabla_\theta L(\theta_t),\qquad \theta_{t+1} = \theta_t - v_t
$$

- $$\gamma v_{t-1}$$: **과거 방향** 정보 (보통 $$\gamma = 0.9$$)
- $$\eta\nabla L$$: **현재 방향** 정보

기울기 방향이 계속 같으면 속도가 붙고, 방향이 계속 바뀌는 축(진동)은 서로 상쇄되어 줄어듭니다. 좁은 골짜기에서 지그재그로 움직이던 SGD가 골짜기를 따라 곧게 내려가게 됩니다.

**단점**: 관성 때문에 최솟값을 **지나칠 수 있고**, 최솟값 주변에서 오래 맴돌 수 있습니다.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### NAG (Nesterov Accelerated Gradient)

Momentum이 지나치는 문제를 줄이기 위해, **관성으로 먼저 이동할 위치를 미리 가 보고** 거기서 기울기를 계산합니다.

$$
v_t = \gamma v_{t-1} + \eta\nabla_\theta L(\theta_t - \gamma v_{t-1}),\qquad \theta_{t+1} = \theta_t - v_t
$$

"이대로 가면 어디쯤 도착할지" 보고 미리 브레이크를 밟는 셈입니다.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

## 4. 보폭 개선: AdaGrad와 RMSProp

모든 파라미터에 같은 학습률을 쓰는 게 최선일까요? 자주 크게 업데이트된 파라미터는 이미 많이 배웠으니 조심스럽게, 거의 업데이트되지 않은 파라미터는 과감하게 움직이는 게 나을 수 있습니다.

### AdaGrad (Adaptive Gradient)

과거 기울기 **제곱의 누적합**으로 학습률을 나눕니다.

$$
G_t = G_{t-1} + (\nabla_\theta L)^2,\qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}}\nabla_\theta L
$$

- 기울기가 컸던 파라미터 → $$G$$가 커짐 → **보폭 감소**
- 기울기가 작았던 파라미터 → $$G$$가 작음 → **보폭 증가**

희소한 feature가 많은 문제(자연어 등)에서 효과적입니다.

**단점**: $$G$$가 계속 **누적만** 되기 때문에 학습이 길어지면 학습률이 0에 가까워져 **최솟값에 도착하기 전에 학습이 사실상 멈춥니다.**

### RMSProp

AdaGrad의 누적합을 **지수 이동 평균**으로 바꿉니다. 오래된 기울기는 점점 잊습니다.

$$
G_t = \rho G_{t-1} + (1 - \rho)(\nabla_\theta L)^2,\qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}}\nabla_\theta L
$$

$$\rho = 0.9$$ 정도면 최근 기울기 위주로 학습률을 조절하므로, 학습률이 0으로 줄어드는 문제가 해결됩니다.

```python
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)   # alpha가 ρ
```

## 5. 둘 다: Adam

**Adam(Adaptive Moment Estimation)**은 Momentum과 RMSProp을 합친 알고리즘입니다.

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla_\theta L \qquad \text{(1차 모멘트: 기울기의 지수 평균, Momentum)}
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla_\theta L)^2 \qquad \text{(2차 모멘트: 기울기 제곱의 지수 평균, RMSProp)}
$$

### Bias Correction

$$m_0 = v_0 = 0$$에서 시작하기 때문에 학습 초반에는 두 값이 0 쪽으로 치우칩니다. 이를 보정합니다.

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t},\qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t},\qquad
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t
$$

기본값 $$\beta_1 = 0.9$$, $$\beta_2 = 0.999$$, $$\epsilon = 10^{-8}$$로 대부분의 문제에서 잘 동작해서, 지금도 가장 많이 쓰이는 Optimizer입니다.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
```

### Adam의 변형들

- **Nadam**: Adam에 Nesterov 기법을 더한 것. 종종 Adam보다 조금 빨리 수렴합니다.
- **RAdam (Rectified Adam)**: 학습 초기에는 $$v_t$$를 추정할 샘플이 적어 **적응형 학습률의 분산이 매우 큽니다.** 이 때문에 초반에 엉뚱한 지역 최솟값으로 빠르게 빠지는 문제가 있는데, RAdam은 분산이 클 때 적응형 학습률을 꺼서 이를 보정합니다. (Warm-up이 필요한 이유와도 같습니다.)
- **AdamW**: Weight decay를 기울기에 섞지 않고 파라미터에 직접 적용합니다. Adam에서 L2 정규화가 제대로 작동하지 않는 문제를 해결해 Transformer 학습의 표준이 됐습니다.

```python
optimizer = torch.optim.NAdam(model.parameters(), lr=2e-3)
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### 직접 비교: 좁은 골짜기에서의 움직임

$$y$$ 방향만 50배 가파른 함수 $$f(x, y) = \frac{1}{2}(x^2 + 50y^2)$$에서 100 step 뒤의 loss를 비교합니다. 최솟값은 $$(0, 0)$$, loss 0입니다.

```python
import torch

def run(make_opt, steps=100):
    p = torch.tensor([-5.0, 1.0], requires_grad=True)        # 시작점 (x, y)
    opt = make_opt([p])
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * (p[0] ** 2 + 50 * p[1] ** 2)           # y 방향만 50배 가파른 골짜기
        loss.backward()
        opt.step()
    return loss.item()

cands = {
    "SGD (lr=0.02)":        lambda ps: torch.optim.SGD(ps, lr=0.02),
    "SGD (lr=0.041)":       lambda ps: torch.optim.SGD(ps, lr=0.041),
    "Momentum 0.9":         lambda ps: torch.optim.SGD(ps, lr=0.02, momentum=0.9),
    "Nesterov 0.9":         lambda ps: torch.optim.SGD(ps, lr=0.02, momentum=0.9, nesterov=True),
    "Adam (lr=0.1)":        lambda ps: torch.optim.Adam(ps, lr=0.1),
}
for name, make in cands.items():
    print(f"{name:16s} 100 step 후 loss = {run(make):.2e}")
```

```text
SGD (lr=0.02)    100 step 후 loss = 2.29e-01
SGD (lr=0.041)   100 step 후 loss = 3.92e+05
Momentum 0.9     100 step 후 loss = 1.19e-03
Nesterov 0.9     100 step 후 loss = 5.01e-05
Adam (lr=0.1)    100 step 후 loss = 8.51e-04
```

(PyTorch 2.13에서 실행)

- **SGD**: 가파른 $$y$$ 방향 때문에 학습률을 키울 수 없습니다. 이 함수에서는 $$\eta > 2/50 = 0.04$$이면 $$y$$ 방향으로 튕겨 나가 발산합니다(lr=0.041). 그런데 그 학습률로는 완만한 $$x$$ 방향이 너무 느리게 내려갑니다.
- **Momentum / Nesterov**: 같은 학습률에서도 완만한 방향으로 속도가 붙어 훨씬 빨리 내려갑니다.
- **Adam**: 파라미터마다 기울기 크기로 나눠 주므로, 두 방향의 가파르기 차이를 스스로 보정합니다.

단, 이 결과는 학습률에 따라 순위가 쉽게 바뀝니다. 예를 들어 Nesterov는 lr=0.03에서 오히려 발산합니다. "어떤 optimizer가 최고"가 아니라 **학습률과 함께 튜닝해야 한다**는 점이 이 실험의 진짜 교훈입니다.

## 6. Optimizer 한눈에 비교

| Optimizer | 방향 | 보폭 | 특징 |
|---|---|---|---|
| SGD | 현재 기울기 | 고정 | 단순, 잘 튜닝하면 일반화 성능이 좋음 |
| Momentum | + 과거 관성 | 고정 | 진동 감소, 빠른 수렴 |
| NAG | + 미리 본 관성 | 고정 | 지나침 감소 |
| AdaGrad | 현재 기울기 | 누적 제곱합으로 감소 | 학습률이 0으로 수렴 |
| RMSProp | 현재 기울기 | 지수 평균으로 조절 | AdaGrad 문제 해결 |
| Adam | 1차 모멘트 | 2차 모멘트 | 기본값으로 잘 동작 |
| AdamW | 1차 모멘트 | 2차 모멘트 | 올바른 weight decay |

> 실무 팁: 빠르게 결과를 보고 싶으면 **Adam/AdamW**, 이미지 분류에서 최고 성능을 끝까지 짜내고 싶으면 **SGD + Momentum + 스케줄러** 조합이 자주 쓰입니다. 같은 모델이라도 SGD 계열이 일반화 성능이 약간 더 좋다는 보고(Wilson et al., 2017)가 있지만, 이후 연구에서는 하이퍼파라미터를 충분히 튜닝하면 차이가 크지 않다는 결과도 나와 정답이 정해진 문제는 아닙니다.

## 7. Learning Rate Scheduler

학습률은 가장 중요한 hyper-parameter입니다. 처음에는 크게 움직여 빠르게 내려가고, 최솟값 근처에서는 작게 움직여 정밀하게 안착하는 것이 이상적입니다. Scheduler는 **학습 도중 학습률을 바꿔줍니다.**

### Power Scheduling

$$
\eta(t) = \frac{\eta_0}{(1 + t/s)^c}
$$

처음엔 빠르게, 점점 천천히 감소합니다.

### Exponential Scheduling

$$
\eta(t) = \eta_0 \cdot 0.1^{t/s}
$$

$$s$$ step마다 학습률이 1/10이 됩니다. PyTorch의 `ExponentialLR`은 epoch마다 `gamma`를 곱합니다.

### Piecewise Constant (Step) Scheduling

정해진 epoch마다 학습률을 뚝 떨어뜨립니다. ResNet 학습의 고전적인 방식입니다(예: 30, 60, 90 epoch에서 1/10).

### Performance Scheduling (ReduceLROnPlateau)

**검증 손실이 일정 기간(patience) 개선되지 않으면** 학습률을 줄입니다. 언제 줄일지 미리 정하지 않아도 됩니다.

### Cosine Annealing

$$
\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t\pi}{T}\right)
$$

코사인 곡선을 따라 부드럽게 줄입니다. 최근 가장 많이 쓰입니다.

### Warm-up

학습 초반 몇 epoch 동안은 학습률을 0에서 **선형으로 키웁니다.** 초기 가중치가 무작위라 기울기가 불안정할 때 큰 학습률로 망가지는 것을 막습니다. Transformer 학습에서는 거의 필수입니다.

### 1cycle Scheduling

학습률을 **처음 절반은 올리고, 나머지 절반은 내립니다.** 마지막 몇 epoch는 더 작게 줄입니다. Momentum은 반대로 움직입니다. 적은 epoch으로 빠르게 좋은 성능을 내는 것으로 알려져 있습니다. (Smith의 원래 제안은 이렇지만, PyTorch `OneCycleLR`의 기본값은 전체의 30%(`pct_start=0.3`) 동안 올리고 나머지를 cosine으로 내립니다.)

```python
from torch.optim import lr_scheduler as S

sched = S.StepLR(optimizer, step_size=30, gamma=0.1)
sched = S.ExponentialLR(optimizer, gamma=0.95)
sched = S.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
sched = S.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
sched = S.OneCycleLR(optimizer, max_lr=0.1, total_steps=len(loader) * epochs)

# Warm-up + Cosine 조합
warmup = S.LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = S.CosineAnnealingLR(optimizer, T_max=95)
sched = S.SequentialLR(optimizer, [warmup, cosine], milestones=[5])
```

### 학습 루프에서 호출하는 위치

```python
for epoch in range(epochs):
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        # OneCycleLR은 매 iteration마다 sched.step()
    val_loss = evaluate(model)
    sched.step()              # 대부분: epoch마다
    # ReduceLROnPlateau는 sched.step(val_loss)
```

`optimizer.step()`보다 **먼저** `sched.step()`을 부르면 첫 학습률을 건너뛰게 되니 순서에 주의합니다. (PyTorch 1.1부터 이 순서가 바뀌었고, 순서가 틀리면 경고가 출력됩니다.)

<details markdown="1">
<summary>더 알아보기: PyTorch의 Momentum 식은 위와 조금 다르다</summary>

위 식은 $$v_t = \gamma v_{t-1} + \eta g_t$$처럼 학습률을 속도 안에 넣지만, PyTorch는 $$v_t = \gamma v_{t-1} + g_t$$, $$\theta_{t+1} = \theta_t - \eta v_t$$로 학습률을 **밖에서** 곱합니다. 학습률이 고정이면 두 식은 같지만, **스케줄러로 학습률을 바꾸는 순간 동작이 달라집니다.** PyTorch 방식은 학습률을 줄이면 이미 쌓인 속도까지 즉시 작아집니다. 논문 구현을 옮길 때 성능이 다르게 나오는 원인이 되기도 합니다. (PyTorch `torch.optim.SGD` 문서의 Note 참고)
</details>

## 8. 정리

- Optimizer는 **방향(관성)**과 **보폭(적응형 학습률)** 두 축으로 발전했고, Adam이 둘을 합쳤습니다.
- AdaGrad는 학습률이 0으로 사라지고, RMSProp이 지수 평균으로 이를 고쳤습니다.
- Adam의 초기 불안정은 bias correction, RAdam, warm-up으로 다룹니다.
- Scheduler는 "처음엔 크게, 나중엔 작게"를 구현하며, Step·Plateau·Cosine·1cycle이 대표적입니다.

다음 글에서는 학습은 잘되는데 **새로운 데이터에서는 성능이 떨어지는 과적합** 문제와 그 해결책을 다룹니다.

## References

- Ruder, "An overview of gradient descent optimization algorithms," 2016.
- Kingma & Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Wilson et al., "The Marginal Value of Adaptive Gradient Methods in Machine Learning," NeurIPS 2017.
- [PyTorch – torch.optim](https://pytorch.org/docs/stable/optim.html)
- Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond (RAdam)," ICLR 2020.
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization (AdamW)," ICLR 2019.
- Smith, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates (1cycle)," 2018.
- Aurélien Géron, 『핸즈온 머신러닝』 11장
