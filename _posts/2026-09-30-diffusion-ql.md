---
title: "[RL] Diffusion-QL: Diffusion 모델을 정책으로 쓰는 Offline RL"
date: 2026-09-30 22:08:00 +0900
categories: [Reinforcement Learning, Offline RL]
tags: [reinforcement-learning, offline-rl, diffusion, diffusion-policy, ddpm, policy-regularization, multi-modal, d4rl]
description: 가우시안 정책이 여러 갈래로 나뉜 행동 분포를 표현하지 못하는 문제와, 정책을 조건부 diffusion 모델로 바꿔 diffusion loss를 규제로, Q 기울기를 개선 신호로 쓰는 Diffusion-QL을 정리합니다.
math: true
---

> Offline RL 정책은 보통 **가우시안**으로 표현합니다. 하지만 데이터 속 행동이 여러 갈래(multi-modal)로 나뉘어 있으면, 가우시안은 그 사이의 어중간한 행동을 고르게 됩니다. **Diffusion-QL**(*Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning*, Wang, Hunt & Zhou, ICLR 2023)은 정책을 **조건부 Diffusion 모델**로 바꿔 이 문제를 풉니다.

## 1. 기존 방법의 한계

### Policy regularization의 한계

[Anti-Exploration]({% post_url 2026-09-27-anti-exploration %})에서 본 것처럼 offline RL은 정책이 데이터 분포 밖으로 나가지 않도록 규제합니다. 그런데 기존 규제 방법은 성능이 좋지 않은 경우가 많았습니다. 예를 들어 behavior policy를 **가우시안으로 근사**한 뒤 그 근처로 제한하면, 실제 데이터 분포와 다른 영역까지 허용하거나 좋은 영역을 막아 **탐색 공간을 잘못 제한**하게 됩니다.

### 가우시안 정책의 표현력

기존 방법들은 대부분 정책을 가우시안으로 표현합니다. 가우시안은 봉우리가 하나뿐이라 **multi-modal한 행동 분포**를 표현하지 못합니다. 행동의 여러 성분 사이의 **의존 관계**도 잘 표현하지 못합니다(대각 공분산을 쓰는 경우).

예를 들어 장애물을 피할 때 데이터에 "왼쪽으로 피하기"와 "오른쪽으로 피하기"가 반반 섞여 있으면, 가우시안은 그 **평균인 "정면으로 돌진"**을 가장 그럴듯한 행동으로 학습할 수 있습니다.

```python
import numpy as np

rng = np.random.default_rng(0)
actions = np.concatenate([rng.normal(-1, 0.1, 500),      # 왼쪽(-1)으로 피한 시연
                          rng.normal(+1, 0.1, 500)])     # 오른쪽(+1)으로 피한 시연
mu, sigma = actions.mean(), actions.std()                # 가우시안 정책의 최대우도 추정
print(f"평균 {mu:.3f}, 표준편차 {sigma:.3f}")             # 평균 -0.005, 표준편차 1.003
print((np.abs(actions) < 0.3).mean())                    # 0.0 → 데이터에는 a≈0인 행동이 하나도 없음
```

가우시안의 확률이 가장 높은 곳(평균 ≈ 0)은 **데이터에 한 번도 나오지 않은 행동**입니다. 정책이 평균 근처를 고르면 장애물에 그대로 부딪치고, 표준편차도 1로 커져 엉뚱한 행동을 자주 샘플링합니다.

## 2. Contribution

1. **DDPM 기반 정책**으로 policy regularization과 **Q-learning guidance**를 함께 수행
2. 간단한 **bandit 실험**으로 diffusion 정책의 표현력 확인
3. **D4RL**의 다양한 환경에서 SOTA 달성

## 3. Diffusion 정책

정책 $$\pi_\theta(a \mid s)$$를 **상태를 조건으로 행동을 생성하는 diffusion 모델**로 정의합니다. 노이즈 $$a^N \sim \mathcal{N}(0, I)$$에서 시작해, 상태 $$s$$를 조건으로 $$N$$번 denoising해서 행동 $$a^0$$을 얻습니다.

$$
\pi_\theta(a \mid s) = p_\theta(a^{0:N} \mid s) = \mathcal{N}(a^N; 0, I)\prod_{i=1}^{N}p_\theta(a^{i-1} \mid a^i, s)
$$

> 두 종류의 시간이 있습니다. 위첨자 $$i$$는 **denoising step**, 아래첨자 $$t$$는 **궤적의 시점**입니다.

Denoising step 수 $$N$$은 이미지 생성(수백~수천)과 달리 **5 정도로 작게** 써도 충분했습니다. 행동은 이미지보다 차원이 훨씬 낮기 때문입니다.

## 4. 학습 목적

$$
\mathcal{L}(\theta) = \underbrace{\mathcal{L}_d(\theta)}_{\text{behavior cloning}} + \underbrace{\mathcal{L}_q(\theta)}_{\text{policy improvement}}
$$

### Behavior Cloning 항 (규제 역할)

데이터셋의 $$(s, a)$$로 일반적인 diffusion 노이즈 예측 loss를 씁니다.

$$
\mathcal{L}_d(\theta) = \mathbb{E}_{i, \epsilon, (s, a) \sim \mathcal{D}}\Big[\big\lVert\epsilon - \epsilon_\theta\big(\sqrt{\bar{\alpha}_i}\,a + \sqrt{1 - \bar{\alpha}_i}\,\epsilon,\ s,\ i\big)\big\rVert^2\Big]
$$

정책이 **학습 데이터의 행동 분포를 따라가게** 합니다. 별도의 behavior policy 근사나 거리 계산 없이, 이 항 자체가 **policy regularization** 역할을 합니다. Diffusion 모델은 multi-modal 분포도 정확히 학습하므로, 규제가 실제 데이터 분포에 맞게 걸립니다.

### Policy Improvement 항 (Q-learning guidance)

정책이 생성한 행동의 Q 값이 크도록 유도합니다.

$$
\mathcal{L}_q(\theta) = -\alpha\cdot\mathbb{E}_{s \sim \mathcal{D},\ a^0 \sim \pi_\theta}\big[Q_\phi(s, a^0)\big]
$$

$$a^0$$은 diffusion 정책의 **reverse chain 전체**를 통과해 샘플링되며, Q의 기울기가 이 chain을 따라 정책 파라미터까지 **역전파**됩니다. 그래서 정책이 **가치가 높은 행동 쪽**으로 샘플을 만들게 됩니다.

Q 값의 스케일은 환경마다 다르므로, 계수를 $$\alpha = \eta / \mathbb{E}_{(s, a) \sim \mathcal{D}}\big[\lvert Q_\phi(s, a)\rvert\big]$$로 정규화해 $$\eta$$ 하나로 두 항의 균형을 조절합니다.

### Critic 학습

Q 함수는 **double Q-learning**으로 학습하고, target의 다음 행동도 diffusion 정책에서 샘플링합니다.

$$
\mathcal{L}(\phi) = \mathbb{E}\Big[\Big(r + \gamma\min_{k=1,2}Q_{\phi'_k}(s', a'^0) - Q_{\phi_k}(s, a)\Big)^2\Big],\qquad a'^0 \sim \pi_{\theta'}(\cdot \mid s')
$$

```python
import torch

def diffusion_ql_losses(batch, policy, q1, q2, q1_t, q2_t, policy_t, eta=1.0, gamma=0.99):
    s, a, r, s_next, done = batch

    # Critic: 다음 행동은 target diffusion 정책에서 샘플링
    with torch.no_grad():
        a_next = policy_t.sample(s_next)                       # N번 denoising
        target = r + gamma * (1 - done) * torch.min(q1_t(s_next, a_next), q2_t(s_next, a_next))
    critic_loss = ((q1(s, a) - target) ** 2).mean() + ((q2(s, a) - target) ** 2).mean()

    # Actor: BC(diffusion loss) + Q guidance (reverse chain 통과해 역전파)
    bc_loss = policy.denoising_loss(a, s)                      # ε 예측 MSE
    a_new = policy.sample(s)                                   # gradient가 흐르는 샘플링
    q_new = torch.min(q1(s, a_new), q2(s, a_new))
    alpha = eta / q2(s, a).abs().mean().detach()
    actor_loss = bc_loss - alpha * q_new.mean()
    return critic_loss, actor_loss
```

## 5. 실험

### 2D Bandit: Multi-modal 표현력

행동이 2차원이고, 좋은 행동이 **여러 군데 떨어져 모여 있는** bandit 문제로 여러 정책 클래스를 비교했습니다.

- **BC (가우시안)**, **BCQ/BEAR (CVAE 등)**: 여러 봉우리를 제대로 잡지 못하고 그 사이나 바깥에 샘플을 흩뿌렸습니다.
- **Diffusion 정책**: 데이터의 여러 봉우리를 정확히 재현했고, Q guidance를 더하면 그중 **가치가 높은 봉우리**에 집중했습니다.

### 계산 비용과 표현력의 trade-off

Denoising step $$N$$을 늘리면 표현력은 좋아지지만 학습·추론 비용이 늘어납니다. 특히 $$\mathcal{L}_q$$는 $$N$$단계 reverse chain 전체를 거쳐 역전파하므로, 학습 시 메모리와 시간이 $$N$$에 비례해 늘어납니다. $$N$$이 너무 작으면 표현력이 부족하고, 어느 정도 이상에서는 성능이 포화했습니다. 논문은 $$N = 5$$를 기본으로 썼습니다.

### D4RL

네 종류의 환경에서 평가했습니다. 데이터셋은 near-optimal 정책이 모은 것부터 사람이 모은 것까지 다양합니다.

| 환경 | 특징 |
|---|---|
| Gym (MuJoCo locomotion) | 표준 연속 제어 |
| AntMaze | 보상이 드물고 궤적 조각을 이어야 함 |
| Adroit | 사람 시연 기반, 고차원 손 조작 |
| Kitchen | 여러 하위 과제를 순서대로 |

Diffusion-QL은 **네 영역 모두에서** 기존 방법(BC, BCQ, CQL, IQL 등)보다 좋거나 비슷한 성능을 기록했고, 특히 AntMaze와 Adroit처럼 행동 분포가 복잡한 과제에서 차이가 컸습니다.

## 6. 정리

| | 가우시안 정책 + 규제 | Diffusion-QL |
|---|---|---|
| 정책 표현 | 단봉 분포 | **다봉 분포 가능** |
| 규제 | behavior policy 근사 + 거리 | **Diffusion loss 자체가 BC 규제** |
| 개선 | Q 최대화 | Reverse chain을 통한 **Q 기울기 역전파** |
| 비용 | 낮음 | Denoising step 수만큼 증가 |

## References

- Wang, Hunt & Zhou, "Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning," ICLR 2023. [arXiv:2208.06193](https://arxiv.org/abs/2208.06193)
- Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020.
- Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement Learning (TD3+BC)," NeurIPS 2021.
- Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning," ICLR 2022.
