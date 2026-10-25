---
title: "[RL] Noisy Networks: 가중치에 노이즈를 넣어 탐험하기"
date: 2026-08-10 21:56:00 +0900
categories: [Reinforcement Learning, Model-Free]
tags: [reinforcement-learning, model-free, dqn, noisy-net, exploration, a3c, dueling-dqn]
description: ε-greedy 대신 네트워크 가중치에 학습 가능한 노이즈를 넣어 상태에 따라 일관되게 탐험하는 NoisyNet의 noisy linear layer, factorized Gaussian, DQN·A3C 적용 방법을 PyTorch 코드와 함께 정리합니다.
math: true
---

> [Double DQN]({% post_url 2026-08-04-double-dqn %})이 **가치 추정**을 개선했다면, **NoisyNet**(Fortunato et al., ICLR 2018)은 **탐험(exploration)** 방식을 바꿉니다. ε-greedy처럼 행동에 무작위성을 더하는 대신, **네트워크 가중치 자체에 학습 가능한 노이즈**를 넣습니다.

## 1. 기존 탐험 방식의 한계

DQN은 **ε-greedy**, A3C는 **entropy bonus**로 탐험합니다.

- **ε-greedy**: step마다 독립적으로 확률 $$\varepsilon$$만큼 무작위 행동. 상태와 무관하게 무작위성이 들어가고, 연속적이고 일관된 탐험(예: 한 방향으로 꾸준히 가 보기)을 하기 어렵습니다.
- **Entropy bonus**: 정책이 너무 빨리 확정되지 않게 막지만, 역시 행동 수준의 국소적인 무작위성입니다.

두 방법 모두 **얼마나 탐험할지($$\varepsilon$$, 엔트로피 계수)를 사람이 정해야** 하고, 상태마다 다르게 조절하지 못합니다.

## 2. 핵심 아이디어

> 네트워크의 **가중치와 bias에 파라미터화된 노이즈**를 더하고, **노이즈의 크기도 학습**하자.

### Contribution

- 가중치에 perturbation을 줘서 탐험하게 만드는 **NoisyNet** 제안
- 노이즈 파라미터는 네트워크의 다른 파라미터와 **함께 경사하강법으로 학습**
- 구현이 간단하고 **연산 오버헤드가 거의 없음**
- DQN, Dueling DQN, A3C 등 **어떤 Deep RL 알고리즘에도** 적용 가능

## 3. Noisy Linear Layer

일반 선형층은

$$
y = wx + b
$$

NoisyNet의 선형층은 가중치와 bias를 **평균 + 학습되는 표준편차 × 노이즈**로 바꿉니다.

$$
y = (\mu^w + \sigma^w \odot \varepsilon^w)\,x + \mu^b + \sigma^b \odot \varepsilon^b
$$

- $$\mu^w, \sigma^w, \mu^b, \sigma^b$$: **학습 가능한 파라미터**
- $$\varepsilon^w, \varepsilon^b$$: 평균 0인 무작위 노이즈 (학습 X)
- $$\odot$$: 원소별 곱

네트워크가 **$$\sigma$$를 키우면 탐험을 많이, 줄이면 적게** 하게 됩니다. 탐험 정도를 데이터로부터 스스로 정하는 것입니다.

### 노이즈를 만드는 두 가지 방법

입력이 $$p$$차원, 출력이 $$q$$차원인 층이라면

| | Independent Gaussian | Factorized Gaussian |
|---|---|---|
| 방식 | 가중치·bias **원소마다** 독립 노이즈 | **입력별**·**출력별** 노이즈를 곱해서 만듦 |
| 필요한 난수 개수 | $$pq + q$$ | $$p + q$$ |
| 사용처 | A3C | DQN, Dueling DQN (난수 생성 비용 절감) |

Factorized 방식은 다음과 같습니다.

$$
\varepsilon^w_{i,j} = f(\varepsilon_i)\,f(\varepsilon_j),\qquad \varepsilon^b_j = f(\varepsilon_j),\qquad f(x) = \text{sgn}(x)\sqrt{\lvert x\rvert}
$$

### 기울기 계산

NoisyNet의 loss는 노이즈에 대한 기댓값 $$\bar{L}(\zeta) = \mathbb{E}_\varepsilon[L(\theta)]$$이므로, 기울기를 쉽게 구할 수 있습니다.

$$
\nabla\bar{L}(\zeta) = \mathbb{E}_\varepsilon\big[\nabla_{\mu, \Sigma}L(\mu + \Sigma \odot \varepsilon)\big]
$$

실제로는 최적화 step마다 **노이즈 샘플 하나**로 Monte Carlo 근사합니다.

### 초기화

| | $$\mu$$ | $$\sigma$$ |
|---|---|---|
| Independent | $$\mathcal{U}\left[-\sqrt{3/p},\ \sqrt{3/p}\right]$$ | $$0.017$$ |
| Factorized | $$\mathcal{U}\left[-1/\sqrt{p},\ 1/\sqrt{p}\right]$$ | $$\sigma_0/\sqrt{p}$$, $$\sigma_0 = 0.5$$ |

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer"""
    def __init__(self, in_f, out_f, sigma0=0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.mu_w = nn.Parameter(torch.empty(out_f, in_f).uniform_(-1 / math.sqrt(in_f), 1 / math.sqrt(in_f)))
        self.sigma_w = nn.Parameter(torch.full((out_f, in_f), sigma0 / math.sqrt(in_f)))
        self.mu_b = nn.Parameter(torch.empty(out_f).uniform_(-1 / math.sqrt(in_f), 1 / math.sqrt(in_f)))
        self.sigma_b = nn.Parameter(torch.full((out_f,), sigma0 / math.sqrt(in_f)))
        self.register_buffer("eps_in", torch.zeros(in_f))
        self.register_buffer("eps_out", torch.zeros(out_f))
        self.reset_noise()

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):                      # p + q개의 난수만 생성
        self.eps_in.copy_(self._f(torch.randn(self.in_f)))
        self.eps_out.copy_(self._f(torch.randn(self.out_f)))

    def forward(self, x):
        if self.training:
            w = self.mu_w + self.sigma_w * torch.outer(self.eps_out, self.eps_in)
            b = self.mu_b + self.sigma_b * self.eps_out
        else:                                   # 평가 때는 평균 가중치만
            w, b = self.mu_w, self.mu_b
        return F.linear(x, w, b)

torch.manual_seed(0)
layer = NoisyLinear(4, 2)
x = torch.ones(1, 4)
for _ in range(3):                              # 노이즈를 바꿀 때마다 출력이 달라짐
    layer.reset_noise()
    print(layer(x).detach().numpy().round(3))
layer.eval()
print("eval:", layer(x).detach().numpy().round(3))
```

```text
[[-0.745  0.576]]
[[-0.837  0.066]]
[[-0.151  1.065]]
eval: [[-0.559  0.46 ]]
```

같은 입력이라도 노이즈를 다시 뽑을 때마다 Q 값(출력)이 달라지고, 그에 따라 greedy 행동도 바뀔 수 있습니다. 이것이 NoisyNet의 탐험입니다. 평가 모드에서는 평균 가중치 $$\mu$$만 쓰므로 출력이 고정됩니다. (평가 때 노이즈를 끌지는 구현마다 다르며, 논문 공식 설정과 다를 수 있습니다.)

## 4. 알고리즘에 적용하기

NoisyNet 에이전트는 **최적화 step마다 새 노이즈를 샘플링**하고, 그 사이에는 고정된 가중치로 행동합니다. (DQN은 보통 환경 4 step마다 한 번 학습하므로, 같은 노이즈로 4 step 정도 행동합니다.) 같은 노이즈로 여러 step 행동하기 때문에, ε-greedy보다 **상태에 의존하는 일관된 탐험**이 됩니다.

> 왜 "상태에 의존"할까요? ε-greedy의 무작위성은 상태와 상관없이 동전을 던지는 것이지만, NoisyNet은 **가중치**가 흔들리므로 같은 노이즈라도 상태에 따라 Q 값이 흔들리는 정도가 달라집니다. 어떤 상태에서는 노이즈가 있어도 최선의 행동이 분명해 거의 바뀌지 않고, 애매한 상태에서는 쉽게 다른 행동이 선택됩니다.
{: .prompt-info }

### NoisyNet-DQN / Dueling DQN

- **ε-greedy를 쓰지 않습니다.** 노이즈가 섞인 Q 값에 대해 그냥 greedy하게 행동합니다.
- Value network의 **FC layer**를 noisy layer로 바꿉니다. (Factorized Gaussian)
- Online network와 target network에 **서로 독립적인 노이즈** $$\varepsilon, \varepsilon'$$를 써서, 두 네트워크 노이즈의 상관관계로 생기는 편향을 피합니다.

$$
\bar{L}(\zeta) = \mathbb{E}\Big[\mathbb{E}_{(x, a, r, y) \sim D}\big[r + \gamma\max_{b}Q(y, b, \varepsilon'; \zeta^-) - Q(x, a, \varepsilon; \zeta)\big]^2\Big]
$$

Dueling DQN에서는 행동 선택에 또 다른 독립 노이즈 $$\varepsilon''$$를 쓰고, target 계산에 Double DQN 방식을 함께 씁니다.

```python
# 매 학습 step마다 노이즈 재샘플링
for module in q_net.modules():
    if isinstance(module, NoisyLinear):
        module.reset_noise()
action = q_net(state).argmax(dim=1)        # ε-greedy 없이 greedy
```

### NoisyNet-A3C

- Policy loss의 **entropy bonus를 제거**합니다.
- Policy network의 FC layer를 noisy layer로 바꿉니다. (Independent Gaussian)

## 5. 결과

Atari 57개 게임에서 **DQN, Dueling DQN, A3C 세 baseline 모두**에 NoisyNet을 적용했을 때

- 대부분의 게임에서 점수가 올랐고
- 사람 대비 정규화 점수의 **평균과 중앙값이 세 알고리즘 모두에서 향상**
- 학습 곡선에서도 더 높은 점수에 도달

특히 탐험이 중요한 일부 게임에서는 사람 수준을 넘는 큰 향상을 보였습니다.

### 노이즈는 결국 사라질까?

$$\sigma$$는 학습되는 값이므로, 최적해가 결정적(deterministic)이라면 네트워크가 $$\sigma$$를 0으로 줄여 노이즈를 없앨 것이라고 예상할 수 있습니다. 이를 확인하기 위해 noisy layer의 평균 노이즈 크기를 측정했습니다.

$$
\bar{\Sigma} = \frac{1}{N_{weights}}\sum_i \lvert\sigma_i^w\rvert
$$

- **마지막 층**에서는 모든 게임에서 $$\bar{\Sigma}$$가 줄었지만
- **그 앞 층(penultimate layer)**에서는 일부 게임에서만 줄었습니다.
- 게임마다 변화 양상도 크게 달랐습니다.

즉 에이전트가 무조건 결정적인 해로 수렴하는 것이 아니라, **문제마다 다른 탐험 전략**을 스스로 만들어 낸다는 뜻입니다.

## 6. 정리

| | ε-greedy / Entropy | NoisyNet |
|---|---|---|
| 무작위성 위치 | 행동 | **가중치** |
| 탐험 정도 | 사람이 정함 | **$$\sigma$$로 학습** |
| 탐험의 성격 | step마다 독립 | 상태 의존적, 여러 step 일관 |
| 추가 비용 | 없음 | 파라미터 약 2배, 연산은 거의 동일 |

NoisyNet은 [Double DQN]({% post_url 2026-08-04-double-dqn %}), [C51]({% post_url 2026-08-12-c51-distributional-rl %}) 등과 함께 **Rainbow**의 구성 요소가 됐습니다.

## References

- Fortunato et al., "Noisy Networks for Exploration," ICLR 2018. [arXiv:1706.10295](https://arxiv.org/abs/1706.10295)
- Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning (A3C)," ICML 2016.
- Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning," ICML 2016.
- Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning," AAAI 2018.
