---
title: "[RL] ICM: 호기심(Curiosity)을 보상으로 쓰는 탐험"
date: 2026-08-29 14:42:00 +0900
categories: [Reinforcement Learning, Exploration]
tags: [reinforcement-learning, exploration, intrinsic-motivation, curiosity, icm, inverse-dynamics, forward-dynamics, noisy-tv]
description: 행동의 결과를 예측하지 못하는 정도를 호기심 보상으로 쓰는 ICM의 inverse·forward dynamics 구조와, 행동과 무관한 노이즈를 특징 공간에서 걸러내는 원리, 실험 결과와 한계를 코드와 함께 정리합니다.
math: true
---

> [Count-Based Exploration]({% post_url 2026-08-26-count-based-exploration %})이 "얼마나 자주 봤는가"로 새로움을 쟀다면, **ICM(Intrinsic Curiosity Module)**(Pathak et al., ICML 2017)은 "**내 행동의 결과를 얼마나 잘 예측하지 못하는가**"를 호기심으로 삼습니다.

## 1. Intrinsic Reward를 설계하는 두 방향

현실 세계의 보상은 대부분 sparse해서 무작위 탐험으로는 학습이 어렵습니다. 그래서 extrinsic reward를 보완하는 intrinsic reward가 필요한데, 설계 방법은 크게 둘로 나뉩니다.

1. **새로운 상태를 탐험하도록 격려** (Novelty): 상태 분포의 통계 모델이 필요 → count-based 방법
2. **자기 행동의 결과를 예측하는 능력의 오차를 줄이도록 격려** (Prediction error): 상태 $$s_t$$와 행동 $$a_t$$로 다음 상태를 예측하는 **환경 동역학 모델**이 필요

2번에는 두 가지 어려움이 있습니다.

- 이미지 같은 **고차원 raw sensory input**의 다음 프레임을 픽셀 단위로 예측하기는 매우 어렵습니다.
- 환경이 **확률적**이면 원래 예측할 수 없는 상태에 계속 끌립니다. 대표적인 예가 **Noisy TV 문제**입니다. 화면에 무작위 노이즈가 나오는 TV가 있으면, 예측 오차가 항상 커서 에이전트는 TV 앞에서 영원히 떠나지 않습니다.

따라서 "**학습 가능한 것만** 예측하게" 만드는 것이 중요합니다.

## 2. 무엇을 예측해야 하나

관측에 영향을 주는 요소를 세 가지로 나눌 수 있습니다.

1. 에이전트가 **제어할 수 있는** 것 (자기 몸의 움직임)
2. 제어할 수는 없지만 **에이전트에게 영향을 주는** 것 (Freeway의 자동차, Montezuma의 해골)
3. 제어할 수도 없고 **영향도 주지 않는** 것 (바람에 흔들리는 나뭇잎, Noisy TV)

좋은 호기심용 특징 공간은 **1과 2를 모델링하고 3에는 영향을 받지 않아야** 합니다. ICM은 픽셀이 아니라 **에이전트의 행동과 관련된 정보만 담은 특징 공간**에서 예측하고, 이 특징 공간을 **자기지도 학습**으로 배웁니다.

### Contribution

- Intrinsic motivation을 이미지 같은 **고차원 연속 상태 공간**으로 확장
- 픽셀을 예측하는 어려운 문제를 **피함**
- 에이전트에 영향을 주지 않는 예측 불가능한 환경 요소에 **흔들리지 않는** intrinsic reward

## 3. ICM의 구조

에이전트는 두 부분으로 구성됩니다.

- **Reward generator (ICM)**: intrinsic reward $$r^i_t$$를 만듦
- **Policy**: 기존 RL처럼 보상의 합을 최대화 (논문은 A3C 사용)

$$
r_t = r^e_t + r^i_t
$$

### Feature Encoder $$\phi$$

상태 $$s_t$$를 특징 벡터 $$\phi(s_t)$$로 바꾸는 CNN입니다. 아래 두 모델이 이 encoder를 공유합니다.

### Inverse Dynamics Model

연속된 두 상태의 특징 $$\phi(s_t), \phi(s_{t+1})$$로 **그 사이에 한 행동** $$\hat{a}_t$$를 예측합니다.

$$
\hat{a}_t = g\big(\phi(s_t), \phi(s_{t+1});\ \theta_I\big),\qquad \mathcal{L}_I = \text{CE}(\hat{a}_t, a_t)
$$

행동을 맞히는 데 필요한 정보만 특징에 담기게 되므로, 행동과 무관한 나뭇잎이나 TV 노이즈는 특징 공간에서 자연스럽게 **걸러집니다.** 이것이 ICM의 핵심입니다.

### Forward Dynamics Model

현재 특징과 행동으로 **다음 특징**을 예측합니다.

$$
\hat{\phi}(s_{t+1}) = f\big(\phi(s_t), a_t;\ \theta_F\big),\qquad \mathcal{L}_F = \frac{1}{2}\big\lVert\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\big\rVert_2^2
$$

### Intrinsic Reward

Forward model의 **예측 오차가 호기심 보상**입니다.

$$
r^i_t = \frac{\eta}{2}\big\lVert\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\big\rVert_2^2
$$

($$\eta$$: scaling factor)

처음 보는 상황에서는 예측이 틀려서 보상이 크고, 익숙해지면 예측이 맞아 보상이 줄어듭니다. 에이전트는 계속 **예측하기 어려운(새로운) 곳**으로 이끌립니다.

### 전체 목적 함수

$$
\min_{\theta_P, \theta_I, \theta_F}\Big[-\lambda\,\mathbb{E}_{\pi(s_t; \theta_P)}\Big[\sum_t r_t\Big] + (1 - \beta)\,\mathcal{L}_I + \beta\,\mathcal{L}_F\Big]
$$

- $$\lambda$$: 정책 학습과 intrinsic reward 학습 사이의 중요도 (0.1)
- $$\beta$$: inverse model과 forward model loss의 비중 (0.2)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):
    def __init__(self, encoder, feat_dim, n_actions, eta=0.01, beta=0.2):
        super().__init__()
        self.enc, self.n_actions, self.eta, self.beta = encoder, n_actions, eta, beta
        self.inverse = nn.Sequential(nn.Linear(2 * feat_dim, 256), nn.ReLU(), nn.Linear(256, n_actions))
        self.forward_model = nn.Sequential(nn.Linear(feat_dim + n_actions, 256), nn.ReLU(), nn.Linear(256, feat_dim))

    def forward(self, s, s_next, a):
        phi, phi_next = self.enc(s), self.enc(s_next)
        a_logits = self.inverse(torch.cat([phi, phi_next], dim=1))
        a_onehot = F.one_hot(a, self.n_actions).float()
        phi_pred = self.forward_model(torch.cat([phi.detach(), a_onehot], dim=1))

        loss_inv = F.cross_entropy(a_logits, a)
        loss_fwd = 0.5 * (phi_pred - phi_next.detach()).pow(2).sum(1).mean()
        r_int = self.eta * 0.5 * (phi_pred - phi_next).pow(2).sum(1).detach()
        return r_int, (1 - self.beta) * loss_inv + self.beta * loss_fwd

# Atari처럼 4프레임을 쌓은 42x42 흑백 관측이라고 가정한 예
torch.manual_seed(0)
encoder = nn.Sequential(nn.Flatten(), nn.Linear(4 * 42 * 42, 288), nn.ELU())   # 실제로는 CNN
icm = ICM(encoder, feat_dim=288, n_actions=4)
s, s_next = torch.randn(8, 4, 42, 42), torch.randn(8, 4, 42, 42)
a = torch.randint(0, 4, (8,))
r_int, loss = icm(s, s_next, a)
loss.backward()
print(r_int.shape, r_int.requires_grad)   # torch.Size([8]) False → 보상은 상수로 정책 학습에 전달
```

`r_int`를 `detach()`하는 이유는, 정책이 보상을 키우려고 **ICM 자체를 바꾸는** 일을 막기 위해서입니다. 보상은 정책 학습에서 상수로 쓰이고, ICM은 자기 loss로만 학습됩니다. 또 위 구현에서는 forward model 입력의 `phi`와 정답 `phi_next`를 detach해서 **특징 공간은 inverse loss로만** 학습되게 했습니다. forward loss까지 encoder로 흘리면, encoder가 모든 상태를 비슷한 특징으로 뭉개 예측 오차를 줄이는 쪽으로 끌려갈 수 있기 때문입니다.

참고로 저자들의 공식 TensorFlow 구현(pathak22/noreward-rl)은 detach 없이 **forward loss도 encoder까지 흘려보냅니다.** 대신 $$\beta = 0.2$$로 forward loss의 비중을 작게 두고 inverse loss(비중 0.8)가 "행동을 맞히려면 정보를 담아야 한다"고 특징 공간을 붙잡아 주므로 붕괴가 일어나지 않습니다. 어느 쪽을 택해도 되지만, 논문 결과를 그대로 재현하려면 공식 구현처럼 두는 편이 안전합니다.

## 4. 실험

**VizDoom**(3D 미로에서 조끼 찾기)과 **Super Mario Bros**에서 실험했습니다.

### Sparse Extrinsic Reward

VizDoom에서 시작 위치에 따라 난이도를 나눴습니다.

- **Dense**: 17개 위치 중 무작위로 시작 (목표 근처에서 시작할 때도 있음)
- **Sparse / Very sparse**: 목표에서 멀리 떨어진 특정 방(13번, 17번)에서 시작

비교 대상은 baseline **A3C**(ε-greedy 탐험), 픽셀 공간에서 forward model만 쓰는 **ICM(pixel)**, 현재 상태만으로 autoencoder 특징을 쓰는 **ICM(aenc)**입니다.

- 모든 설정에서 ICM이 A3C를 압도했고, very sparse에서 A3C는 전혀 학습하지 못했습니다.
- Dense에서 ICM(pixel)이 ICM보다 낮았던 이유를 저자들은 **시작할 때마다 바뀌는 벽 텍스처**로 추측합니다. 픽셀 예측은 행동과 무관한 텍스처 변화에도 오차가 커지기 때문입니다.

### 보상이 전혀 없을 때

Extrinsic reward 없이 **호기심만으로** 학습해 봤습니다.

- VizDoom에서 2,100 step 에피소드 중 약 250 step 만에 **가장 먼 방**까지 탐험했습니다. 호기심이 없으면 탐험 범위가 훨씬 좁았습니다.
- Mario에서는 보상 없이 **1단계의 30% 이상**을 진행했습니다. 적을 피하고 구멍을 뛰어넘는 행동을 스스로 익힌 것입니다(죽으면 처음으로 돌아가 다시 익숙한 장면을 보게 되므로, 호기심 관점에서 죽음은 손해입니다).

### 새로운 환경으로의 일반화

Mario 1단계에서 호기심만으로 학습한 정책을 다른 단계에 옮겨 봤습니다.

1. **그대로 적용**: 시각적으로 비슷한 3단계에서는 점수가 높았지만 2단계에서는 낮았습니다.
2. **호기심만으로 fine-tuning**: 2단계는 처음부터 학습한 것보다 좋아졌지만, 3단계는 오히려 나빠졌습니다. 저자들은 2단계에서는 1단계의 **지식을 활용**했고, 탈출이 어려운 3단계에서는 같은 상태가 반복되며 **호기심이 막혀(curiosity blockade)** 정책이 퇴화했다고 설명합니다.
3. **Extrinsic reward로 fine-tuning**: VizDoom에서 호기심으로 사전학습한 모델을 very sparse 목표로 fine-tuning하자, 처음부터 학습한 모델보다 좋았습니다.

호기심만으로도 환경에 대한 **이해**를 얻을 수 있고, 이것이 새 환경에 일반화된다는 것을 보여 줬습니다.

### 한계 (세미나 정리)

- 일반화 실험들이 서로 같은 조건으로 설계되지 않음
- Sparse와 very sparse를 나누는 기준이 다소 주관적
- Baseline 반복 실험이나, 보상이 전혀 없는 설정의 정량적 결과가 더 있었으면 좋았을 것

## 5. 정리

| 구성 요소 | 역할 |
|---|---|
| Encoder $$\phi$$ | 상태를 특징으로 |
| Inverse model | 행동 예측 → **행동과 관련된 정보만** 특징에 남김 |
| Forward model | 다음 특징 예측 |
| Intrinsic reward | Forward model의 **예측 오차** |

ICM은 픽셀 대신 학습된 특징 공간에서 예측해 Noisy TV 문제를 상당 부분 줄였습니다. 하지만 **예측 대상 자체가 확률적인** 경우(동전 던지기 결과처럼 행동과 관련 있지만 본질적으로 무작위인 것)에는 여전히 약합니다. 다음 글 [RND]({% post_url 2026-09-02-random-network-distillation %})는 이를 **결정적인 예측 문제**로 바꿔 해결합니다.

## References

- Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction," ICML 2017. [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)
- Burda et al., "Large-Scale Study of Curiosity-Driven Learning," ICLR 2019.
- Schmidhuber, "A possibility for implementing curiosity and boredom in model-building neural controllers," 1991.
