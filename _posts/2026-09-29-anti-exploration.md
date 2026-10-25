---
title: "[RL] Anti-Exploration: 탐험 보너스를 뒤집어 Offline RL 하기"
date: 2026-09-29 19:30:00 +0900
categories: [Reinforcement Learning, Offline RL]
tags: [reinforcement-learning, offline-rl, anti-exploration, bonus, cvae, td3, ood-action, kl-regularization]
description: Online RL의 탐험 보너스를 뒤집어 데이터에 없는 행동에 벌점을 주는 Anti-Exploration의 아이디어와 policy regularization과의 관계, CVAE 재구성 오차를 보너스로 쓰는 TD3-CVAE를 실험 코드와 함께 정리합니다.
math: true
---

> Online RL에서 탐험 보너스는 **불확실한 곳으로 가도록** 보상을 더해 줍니다. *Offline Reinforcement Learning as Anti-Exploration*(Rezaeifar et al., AAAI 2022)은 이를 거꾸로 써서, offline RL에서는 **불확실한(데이터에 없는) 행동을 피하도록** 보너스를 **빼 줍니다.** [Exploration 카테고리]({% post_url 2026-08-26-count-based-exploration %})에서 본 보너스 설계를 offline RL로 옮겨 온 연구입니다.

## 1. Offline RL과 Online RL의 차이

- **Online RL**: step이나 episode 단위로 환경과 상호작용해 얻은 샘플로 정책을 업데이트
- **Offline RL**: 상호작용 없이, 미리 모은 샘플로 정책이 수렴할 때까지 학습

### Online의 탐험 보너스를 Offline에서도 쓸 수 있을까?

Offline RL은 정책 최적화에 **고정된 데이터셋**을 씁니다. 불확실한 행동을 해 봐도 결과를 확인할 수 없으니, online처럼 불확실성을 **좇는** 보너스는 의미가 없습니다. 오히려 **불확실성이 높은 행동을 피하도록** 보너스를 정의해야 합니다.

논문은 이를 **Anti-exploration(반탐색)**이라고 정의하고, 이 보너스를 계산하는 새로운 구조를 도입합니다.

## 2. Offline RL이 어려운 이유

- 환경과 상호작용 없이 고정된 데이터로 학습하므로, 테스트 시 상태–행동 분포가 학습 데이터와 크게 다르면 **데이터에 없는 상태–행동 쌍에 비현실적인 가치를 추정**하는 **extrapolation error**가 생깁니다.
- **OOD(Out-Of-Distribution) 행동**이 주된 원인입니다.
- Online이라면 직접 해 보고 Q를 보정할 수 있지만, offline에서는 **누적된 extrapolation error가 가치 함수를 발산**시키는데, 이를 보정할 수 없습니다.

그래서 최근 연구들은 학습된 정책이 **데이터셋 안쪽에 머물도록** 제한합니다. ([IQL]({% post_url 2026-09-26-implicit-q-learning %})은 아예 OOD 행동을 묻지 않는 방법이었습니다.)

## 3. Anti-Exploration

### 보너스 기반 탐험 방법을 Offline으로

Count-based든 prediction-based([RND]({% post_url 2026-09-02-random-network-distillation %}) 등)든, 보너스 $$b(s, a)$$는 **데이터에서 자주 본 상태–행동일수록 작고, 처음 보는 것일수록 큽니다.** Online에서는 이를 보상에 더하지만, offline의 성격상 더하는 것은 타당하지 않습니다.

데이터셋에 있는 상태–행동 쌍이라면 보너스가 항상 0에 가까울 것이므로, **모르는 상태–행동 쌍에 대해서만** 가치에서 보너스를 **빼면** 됩니다.

### 두 가지 Anti-Exploration 식

**(1) Critic의 bootstrap 값에서 빼기**

$$
Q(s, a) \leftarrow r + \gamma\,\mathbb{E}_{a' \sim \pi(\cdot \mid s')}\big[Q(s', a') - b(s', a')\big]
$$

**(2) Actor의 목적에서 빼기**

$$
\pi \leftarrow \arg\max_\pi\ \mathbb{E}_{s \sim \mathcal{D},\ a \sim \pi}\big[Q(s, a) - b(s, a)\big]
$$

두 방식 모두 결국 다음 기댓값을 최대화하도록 수렴합니다.

$$
\mathbb{E}\Big[\sum_t \gamma^t\big(r(s_t, a_t) - b(s_t, a_t)\big)\Big]
$$

데이터에서 벗어난 행동일수록 **벌점**을 받으니, 정책은 데이터 근처에서 가장 좋은 행동을 고르게 됩니다.

## 4. Policy Regularization과의 관계

최근의 많은 offline RL 연구는 **behavior policy를 기준으로 정책을 규제**해서 데이터 분포에서 크게 벗어나지 않게 합니다(behavior-regularized offline RL). Behavior policy와 학습된 정책의 차이에 페널티를 주는 방식입니다.

논문은 Anti-exploration도 **policy regularization과 연결**된다는 것을 보입니다. 보너스로 상태별 행동 분포를 모델링할 수 있으므로, 보너스가 정의하는 정책 $$\pi_b(a \mid s) \propto \exp(-b(s, a))$$와의 **KL divergence**로 규제하는 방식에서 온도(temperature)를 0으로 보낸 극한에 해당합니다.

$$
\max_\pi\ \mathbb{E}\big[Q(s, a)\big] - \lambda\,D_{KL}\big(\pi(\cdot \mid s)\ \|\ \pi_b(\cdot \mid s)\big)
$$

즉 anti-exploration은 **KL-regularized value iteration**의 극한으로 정리되며, 이를 통해 policy regularization 계열 방법들과 연결됩니다.

## 5. 실용적인 방법: TD3 + CVAE

보너스로 무엇을 쓸까요? 논문은 **CVAE(Conditional Variational Autoencoder)**의 **재구성 오차**를 씁니다.

- 데이터셋의 $$(s, a)$$를 encoder로 잠재 표현 $$z$$로 바꾸고
- $$(s, z)$$를 decoder로 다시 행동 $$\hat{a}$$로 복원
- 데이터에 있는 행동은 잘 복원되고, 데이터에 없는 행동은 복원 오차가 큼

$$
b(s, a) = \big\lVert a - D\big(s, E(s, a)\big)\big\rVert^2
$$

이 보너스를 **TD3**의 critic과 actor에 위의 방식으로 넣은 것이 **TD3-CVAE**입니다.

```python
import torch
import torch.nn as nn

class CVAEBonus(nn.Module):
    def __init__(self, s_dim, a_dim, z_dim=8, h=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(s_dim + a_dim, h), nn.ReLU(), nn.Linear(h, 2 * z_dim))
        self.dec = nn.Sequential(nn.Linear(s_dim + z_dim, h), nn.ReLU(), nn.Linear(h, a_dim), nn.Tanh())

    def forward(self, s, a):
        mu, logvar = self.enc(torch.cat([s, a], -1)).chunk(2, -1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        a_hat = self.dec(torch.cat([s, z], -1))
        return a_hat, mu, logvar

    def bonus(self, s, a):
        with torch.no_grad():
            mu, _ = self.enc(torch.cat([s, a], -1)).chunk(2, -1)
            a_hat = self.dec(torch.cat([s, mu], -1))
        return ((a - a_hat) ** 2).sum(-1)                      # 재구성 오차 = anti-exploration bonus

# TD3 critic target:  y = r + γ * (min Q_target(s', a') - α * bonus(s', a'))
# TD3 actor loss:     -(Q(s, π(s)) - α * bonus(s, π(s))).mean()
```

보너스가 정말로 "데이터에 없는 행동"에서 커지는지 장난감 데이터로 확인해 볼 수 있습니다. 상태가 행동을 결정하는 behavior policy로 데이터를 만들고 CVAE를 학습한 뒤, 세 종류의 행동에 대한 보너스를 비교합니다.

```python
torch.manual_seed(0)
S = torch.rand(5000, 2) * 2 - 1                       # 상태: 2차원
A = torch.tanh(2 * S[:, :1] - S[:, 1:])               # 행동: 상태로 정해지는 1차원 행동 (behavior policy)

model = CVAEBonus(s_dim=2, a_dim=1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(3000):
    idx = torch.randint(0, 5000, (256,))
    a_hat, mu, logvar = model(S[idx], A[idx])
    recon = ((a_hat - A[idx]) ** 2).sum(-1).mean()
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1).mean()
    loss = recon + 0.5 * kl                           # CVAE loss = 재구성 오차 + KL
    opt.zero_grad(); loss.backward(); opt.step()

test = torch.arange(1000)
print(model.bonus(S[test], A[test]).mean().item())                          # 데이터 행동
print(model.bonus(S[test], A[test][torch.randperm(1000)]).mean().item())    # shuffled
print(model.bonus(S[test], torch.rand(1000, 1) * 2 - 1).mean().item())      # random
```

```text
0.00016...   ← 데이터 행동
1.029...     ← shuffled (행동 값 자체는 데이터에 있지만 상태와 짝이 안 맞음)
0.839...     ← random
```

데이터 행동의 보너스는 거의 0이고, OOD 행동은 수천 배 큽니다. 특히 **shuffled**는 행동 값 하나하나는 데이터에 실제로 있는 값인데도 보너스가 큽니다. CVAE가 행동을 **상태에 조건부로** 모델링하기 때문이고, 아래 실험에서 RND가 shuffled를 잘 못 잡은 이유와 같은 맥락입니다.

## 6. 실험

### 실험 환경 (D4RL)

**Locomotion**: Hopper, Walker2D, Ant, HalfCheetah에서 이동 거리를 최대화

| 데이터셋 | 수집 정책 |
|---|---|
| Random | 무작위 정책 |
| Medium-replay | SAC 학습 중 replay buffer |
| Medium | 중간 성능의 sub-optimal 정책 |
| Medium-expert | Medium + near-optimal 정책 혼합 |

**Hand manipulation (Adroit)**: 못 박기(Hammer), 문 열기(Door), 공 옮기기(Relocate), 펜 돌리기(Pen)

| 데이터셋 | 수집 방법 |
|---|---|
| Human | 사람 조작 |
| Cloned | Behavior cloning 정책 |
| Expert | Fine-tuning된 전문가 정책 |

### 1. 어떤 보너스가 OOD를 잘 잡나

Walker2d-medium에서 데이터셋의 $$(s, a)$$로부터 세 종류의 OOD 행동을 만들어, RND와 CVAE 보너스가 이를 얼마나 잘 구분하는지 비교했습니다.

- **Shuffled**: 상태는 그대로 두고 행동 순서를 무작위로 섞음 (상태와 맞지 않는 행동)
- **Random**: 무작위 행동
- **Dataset + noise**: 데이터의 행동에 가우시안 노이즈를 더함

**CVAE**는 세 경우 모두에서 데이터 행동과 OOD 행동을 잘 구분했지만, **RND**는 데이터 행동과 OOD 행동의 보너스 차이가 크지 않아 OOD 행동을 잘 잡지 못했습니다. 상태에 **조건부인** 행동 분포를 모델링하는 것이 중요하다는 뜻입니다.

### 2. D4RL 성능

Locomotion과 hand manipulation의 각 데이터셋에서 model-free offline RL 알고리즘들(BC, BCQ, BEAR, BRAC, AWR, CQL)과 비교했습니다. TD3-CVAE는 단순한 구조로 locomotion에서 평균적으로 **CQL과 비슷하고** 나머지 방법보다 좋았으며, hand manipulation에서는 **평균 성능이 가장 높았습니다**.

## 7. 정리

| | Online 탐험 보너스 | Anti-exploration 보너스 |
|---|---|---|
| 목적 | 불확실한 곳으로 **가게** | 불확실한 행동을 **피하게** |
| 사용 | 보상에 **더함** | 가치(critic/actor)에서 **뺌** |
| 해석 | 탐험 장려 | **KL 기반 policy regularization** |
| 보너스 | count, RND 등 | **CVAE 재구성 오차** (상태 조건부) |

다음 글에서는 정책 자체를 **Diffusion 모델**로 표현해, 가우시안 정책이 표현하지 못하는 여러 봉우리의 행동 분포를 다루는 [Diffusion-QL]({% post_url 2026-10-04-diffusion-ql %})을 다룹니다.

## References

- Rezaeifar et al., "Offline Reinforcement Learning as Anti-Exploration," AAAI 2022. [arXiv:2106.06431](https://arxiv.org/abs/2106.06431)
- Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods (TD3)," ICML 2018.
- Fujimoto et al., "Off-Policy Deep Reinforcement Learning without Exploration (BCQ)," ICML 2019.
- Wu et al., "Behavior Regularized Offline Reinforcement Learning (BRAC)," 2019.
- Sohn et al., "Learning Structured Output Representation using Deep Conditional Generative Models (CVAE)," NeurIPS 2015.
