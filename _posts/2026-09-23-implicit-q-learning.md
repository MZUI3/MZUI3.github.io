---
title: "[RL] IQL: 데이터 밖의 행동을 한 번도 묻지 않는 Offline RL"
date: 2026-09-23 23:08:00 +0900
categories: [Reinforcement Learning, Offline RL]
tags: [reinforcement-learning, offline-rl, iql, implicit-q-learning, expectile-regression, awr, extrapolation-error, d4rl]
description: 데이터셋에 없는 행동의 가치를 한 번도 계산하지 않고 expectile regression으로 '데이터 안의 max'를 근사하는 IQL의 가치 학습과 AWR 정책 추출을 실행 가능한 예제와 함께 정리합니다.
math: true
---

> Offline RL 카테고리의 첫 글입니다. **IQL(Implicit Q-Learning)**(Kostrikov, Nair & Levine, ICLR 2022)은 offline RL의 핵심 문제인 **분포 밖 행동의 과대평가**를, 아예 **데이터셋에 없는 행동의 가치를 한 번도 계산하지 않는** 방식으로 피합니다.

## 1. Offline RL이란

- **Online RL**: 환경과 상호작용하며 데이터를 모으고 정책을 업데이트
- **Offline RL**: 환경과 상호작용 **없이**, 이미 모아 둔 **고정된 데이터셋**만으로 정책을 학습

의료, 자율주행처럼 탐험 자체가 위험하거나 비싼 분야에서 중요합니다.

### Trade-off

Offline RL은 두 목표 사이의 균형을 맞춰야 합니다.

- **데이터를 모은 정책(behavior policy)보다 개선**: 그러려면 데이터 분포를 어느 정도 **벗어나야** 함
- **분포에서 너무 벗어나지 않게 규제**: 데이터에 없는 행동은 가치를 잘못 추정하기 쉬우므로 억제해야 함

목표는 **데이터셋에 있는 행동들 안에서** 가장 좋은 정책을 학습하는 것입니다.

### Extrapolation Error

Q-learning의 target에는 $$\max_{a'}Q(s', a')$$가 있습니다. 이 max가 고르는 $$a'$$는 **데이터셋에 없는 행동**일 수 있고, 그런 행동의 Q 값은 학습된 적이 없어 엉뚱하게 크게 추정됩니다(extrapolation). Online이라면 그 행동을 실제로 해 보고 바로잡겠지만, offline에서는 바로잡을 방법이 없어 오차가 누적됩니다.

기존 방법들은 behavior policy와의 거리를 규제하거나(BCQ, BRAC), 데이터에 없는 행동에 낮은 가치를 주는(CQL) 등으로 대응했지만, 여전히 **학습 중에 분포 밖 행동의 Q 값을 계산**해야 해서 extrapolation을 완전히 피하지 못했습니다.

## 2. 두 극단: Single-step vs Multi-step DP

- **Single-step dynamic programming** (예: Onestep RL): behavior policy의 Q 함수 $$Q^{\beta}$$를 SARSA로 학습한 뒤 **한 번만** 정책을 개선. 분포 밖 행동을 묻지 않아 안전하지만, 개선 폭이 제한적
- **Multi-step dynamic programming** (예: Q-learning 계열): 개선을 반복해 behavior policy보다 크게 나은 정책을 얻을 수 있지만, 분포 밖 행동을 묻게 됨

IQL은 **분포 밖 행동을 묻지 않으면서도 multi-step 개선**의 효과를 냅니다.

## 3. 핵심 아이디어: Upper Expectile

$$\max_{a'}Q(s', a')$$를 직접 계산하지 않고, **데이터셋에 있는 행동들의 Q 값 분포의 상위 기댓값(upper expectile)**으로 근사합니다. SARSA 형태의 TD 학습을 조금 수정해서, 평균이 아니라 **상위 쪽**을 추정하게 만듭니다.

### Expectile Regression

$$\tau \in (0, 1)$$에 대한 비대칭 제곱 손실입니다.

$$
L_2^\tau(u) = \big\lvert\tau - \mathbf{1}(u < 0)\big\rvert\,u^2
$$

- $$\tau = 0.5$$: 일반 MSE → **평균**
- $$\tau > 0.5$$: 양의 오차(예측이 작을 때)에 더 큰 가중치 → 분포의 **위쪽**을 추정
- $$\tau \to 1$$: 데이터에 있는 값들의 **최댓값**에 가까워짐

```python
import torch

def expectile_loss(diff, tau=0.7):
    """diff = target - prediction"""
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff ** 2).mean()

# 한 상태에서 데이터셋 행동 5개의 Q 값이 아래와 같을 때, τ에 따라 V가 어디로 가는지
q_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0])
for tau in [0.5, 0.7, 0.9, 0.99]:
    v = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.05)
    for _ in range(3000):
        loss = expectile_loss(q_values - v, tau)
        opt.zero_grad(); loss.backward(); opt.step()
    print(tau, round(v.item(), 3))
# 0.5 4.0 / 0.7 5.263 / 0.9 7.692 / 0.99 9.709
```

$$\tau = 0.5$$에서는 평균(4.0)이고, $$\tau$$를 키울수록 데이터 속 최댓값(10)에 다가갑니다. 중요한 점은 $$V$$가 **10을 넘지 않는다**는 것입니다. 데이터에 없는 행동이 아무리 커 보여도 $$V$$는 데이터에 있는 행동의 Q 값들 범위 안에서만 움직이므로, extrapolation으로 부풀려진 값이 target에 끼어들 수 없습니다. 반대로 $$\tau$$를 1에 너무 가깝게 두면 운 좋게 크게 추정된 소수의 Q 값에 끌려가므로, 과제마다 적절한 값(0.7, 0.9 등)을 고릅니다.

## 4. 방법

순서는 **① 가치 함수 학습 → ② 정책 추출**입니다. 가치 함수 학습 단계에는 정책이 전혀 등장하지 않습니다.

### ① Value Function: Expectile Regression

상태 가치 $$V(s)$$가, 데이터셋 행동들의 Q 값 분포에서 upper expectile이 되도록 학습합니다.

$$
\mathcal{L}_V(\psi) = \mathbb{E}_{(s, a) \sim \mathcal{D}}\Big[L_2^\tau\big(Q_{\hat{\theta}}(s, a) - V_\psi(s)\big)\Big]
$$

$$(s, a)$$는 **데이터셋에서만** 뽑습니다. 그래서 $$V(s)$$는 "이 상태에서 데이터에 있는 행동 중 좋은 쪽의 가치"가 됩니다.

### Q Function

Q 함수는 다음 상태의 **$$V$$로 bootstrap**합니다. max가 없습니다.

$$
\mathcal{L}_Q(\theta) = \mathbb{E}_{(s, a, s') \sim \mathcal{D}}\Big[\big(r(s, a) + \gamma V_\psi(s') - Q_\theta(s, a)\big)^2\Big]
$$

$$V$$가 이미 "데이터 안에서의 max"를 근사하고 있으므로, 이 반복은 **데이터에 있는 행동만으로 multi-step 개선**을 하는 것과 같습니다. 그리고 계산 어디에서도 **데이터에 없는 행동의 Q 값을 묻지 않습니다.** 이것이 "implicit"의 의미입니다.

$$V$$를 따로 두는 이유는, $$Q(s', a')$$에 expectile을 바로 적용하면 다음 상태 전이의 **확률성(운 좋은 전이)**까지 상위 쪽으로 평가하게 되기 때문입니다. $$V$$는 행동에 대해서만 expectile을 취합니다. Q는 **double Q-learning**처럼 두 개를 두고 작은 값을 씁니다.

### ② Policy Extraction: Advantage Weighted Regression

가치 함수가 학습되면, 정책은 **데이터셋 행동을 advantage로 가중해서 모방**합니다(AWR).

$$
\mathcal{L}_\pi(\phi) = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\Big[\exp\big(\beta\,(Q_{\hat{\theta}}(s, a) - V_\psi(s))\big)\log\pi_\phi(a \mid s)\Big]
$$

- Advantage $$Q - V$$가 큰(좋은) 행동일수록 강하게 모방
- 여기서도 **데이터셋의 행동만** 사용
- $$\beta$$가 0이면 behavior cloning, 크면 가장 좋은 행동만 모방

```python
def iql_update(batch, q1, q2, q_target1, q_target2, v, policy, tau=0.7, beta=3.0, gamma=0.99):
    s, a, r, s_next, done = batch
    with torch.no_grad():
        q_t = torch.min(q_target1(s, a), q_target2(s, a))
    v_loss = expectile_loss(q_t - v(s), tau)                       # ① V: upper expectile

    with torch.no_grad():
        target = r + gamma * (1 - done) * v(s_next)                # max 없이 V로 bootstrap
    q_loss = ((q1(s, a) - target) ** 2).mean() + ((q2(s, a) - target) ** 2).mean()

    with torch.no_grad():
        w = torch.exp(beta * (q_t - v(s))).clamp(max=100.0)        # ② AWR 가중치
    pi_loss = -(w * policy.log_prob(s, a)).mean()
    return v_loss, q_loss, pi_loss
```

## 5. 실험

### 간단한 미로

미로 예제에서, 같은 데이터로 학습해도 IQL은 SARSA보다 데이터 속 좋은 경로를 찾아내고, 다른 offline 방법보다 **noise가 적은** 가치 추정을 보였습니다.

### D4RL

- **MuJoCo locomotion**: 데이터 품질에 영향을 많이 받는 single-step DP 방식들도 성능이 좋은 영역인데, IQL도 SOTA와 비슷하거나 더 좋았습니다.
- **AntMaze**: 개미 로봇이 미로를 통과하는 과제로, 데이터에 **완전한 성공 궤적이 거의 또는 전혀 없습니다.** 여러 궤적의 조각을 이어 붙여야(stitching) 하므로 **multi-step DP가 필수**입니다. 여기서 IQL은 기존 방법들을 크게 앞섰습니다.
- **Kitchen, Adroit**에서도 좋은 성능을 보였습니다.
- Expectile $$\tau$$는 locomotion에서 0.7, AntMaze에서 0.9처럼 과제에 따라 조정했습니다.

### Online Fine-tuning

Offline으로 사전학습한 뒤 **온라인으로 fine-tuning**할 때도 IQL은 안정적으로 성능이 올랐습니다. CQL 같은 보수적인 방법은 fine-tuning 초반에 성능이 떨어지거나 느리게 오르는 경향이 있었습니다.

## 6. 정리

| 단계 | 사용하는 행동 | 핵심 |
|---|---|---|
| V 학습 | 데이터셋 행동만 | Expectile로 "데이터 안의 max" 근사 |
| Q 학습 | 데이터셋 행동만 | $$r + \gamma V(s')$$, max 없음 |
| 정책 추출 | 데이터셋 행동만 | Advantage 가중 회귀 (AWR) |

IQL은 단순하고 빠르며 하이퍼파라미터에 덜 민감해, 이후 offline RL의 강력한 baseline이 됐습니다. 다음 글에서는 탐험 보너스를 **뒤집어서** offline RL에 쓰는 [Anti-Exploration]({% post_url 2026-09-27-anti-exploration %})을 다룹니다.

## References

- Kostrikov, Nair & Levine, "Offline Reinforcement Learning with Implicit Q-Learning," ICLR 2022. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)
- Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning (CQL)," NeurIPS 2020.
- Peng et al., "Advantage-Weighted Regression," 2019.
- Brandfonbrener et al., "Offline RL Without Off-Policy Evaluation (Onestep RL)," NeurIPS 2021.
- Fu et al., "D4RL," 2020.
