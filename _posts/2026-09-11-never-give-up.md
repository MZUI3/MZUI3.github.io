---
title: "[RL] Never Give Up: 에피소드 내 새로움과 평생 새로움을 함께 쓰는 탐험"
date: 2026-09-11 21:31:00 +0900
categories: [Reinforcement Learning, Exploration]
tags: [reinforcement-learning, exploration, intrinsic-motivation, ngu, episodic-memory, rnd, r2d2, uvfa]
description: 한 에피소드 안의 새로움(k-NN episodic memory)과 학습 전체의 새로움(RND)을 곱해 탐험을 포기하지 않게 만든 NGU의 intrinsic reward와, 탐험 정도가 다른 정책들을 함께 학습하는 구조를 코드와 함께 정리합니다.
math: true
---

> 지금까지의 intrinsic reward는 한 번 새로움이 사라지면 더 이상 탐험할 이유를 주지 못했습니다. **NGU(Never Give Up)**(Badia et al., ICLR 2020)는 이름 그대로 **탐험을 포기하지 않도록**, 한 에피소드 안의 새로움(episodic)과 학습 전체에 걸친 새로움(life-long)을 결합하고, 탐험 정도가 다른 **여러 정책을 동시에** 학습합니다.

## 1. 기존 탐험 방법의 한계

### 새로움이 사라지면 다시 오지 않는다

[Pseudo-count]({% post_url 2026-08-28-count-based-exploration %})처럼 상태의 새로움을 수치화하는 방법은, 방문 횟수를 정확히 세는 대신 이미 방문한 상태와의 차이로 새로움을 계산합니다. 하지만 어떤 상태의 **새로움이 한 번 사라지고 나면**, 그 상태가 이후 학습(downstream)에 중요하더라도 에이전트는 **다시 방문할 이유가 없습니다.**

### 예측 오차 기반의 문제

[ICM]({% post_url 2026-08-29-icm-curiosity %})처럼 forward dynamics 모델의 예측 오차를 쓰는 방법은

- 관측으로 동역학 모델을 만드는 비용이 크고
- Noisy TV 같은 노이즈의 영향을 받으며
- 임의의 환경으로 일반화하기 어렵습니다.

또 새로움 신호가 없어지면 **방향성 없는 탐험**으로 돌아가 지역 최적해에 머무릅니다. 즉, **포기(give up)**해 버립니다.

## 2. NGU의 핵심 아이디어

1. **탐험 정책과 활용 정책을 동시에 학습**
   - Exploitative policy: extrinsic reward 최대화에 집중
   - Exploratory policy: 방향성 없는 탐험으로 축소되지 않고 **계속 탐험** (never give up)
2. **Episodic novelty와 Life-long novelty를 결합**
   - Episodic: 한 에피소드 안에서는 **같은 상태를 다시 방문하지 않도록** 빠르게 억제. 하지만 에피소드가 바뀌면 다시 리셋되므로, 익숙하지만 완전히 탐험하지는 않은 상태를 **주기적으로 다시 찾아가게** 함
   - Life-long: 여러 에피소드에 걸쳐 많이 방문한 상태의 보상을 **천천히 줄임**

### NGU Intrinsic Reward의 세 가지 성질

- 같은 에피소드 안에서 같은 상태를 재방문하는 것을 **빠르게** 억제
- 여러 에피소드에 걸쳐 많이 방문한 상태를 **천천히** 억제
- 상태의 개념이 **에이전트의 행동이 영향을 주지 않는 요소를 무시**

## 3. Intrinsic Reward의 구성

$$
r^i_t = r^{episodic}_t \cdot \min\big\{\max\{\alpha_t, 1\},\ L\big\}
$$

세 부분으로 나눠 보겠습니다.

### ① Embedding Network: 제어 가능한 특징만

먼저 상태를 비교할 공간을 만듭니다. **Siamese 구조**(두 입력이 가중치를 공유하는 CNN을 통과)의 embedding network $$f$$를 두고, 연속된 두 관측 $$x_t, x_{t+1}$$의 embedding으로 **그 사이의 행동** $$a_t$$를 예측하도록 학습합니다(ICM의 inverse dynamics와 같은 방식).

$$
p(a_t \mid x_t, x_{t+1}) = h\big(f(x_t), f(x_{t+1})\big)
$$

그러면 Noisy TV나 화면 속 많은 사람의 움직임처럼 **에이전트 행동과 무관한 요소(meaningless state)**는 embedding에서 무시되고, **제어 가능한 상태**만 남습니다.

### ② Episodic Novelty Module

**Episodic memory** $$M$$에 이번 에피소드에서 본 embedding $$\{f(x_0), \dots, f(x_{t-1})\}$$을 저장합니다. 현재 embedding과 가장 가까운 **$$k$$개의 이웃** $$N_k$$를 찾아, 이웃이 가깝고 많을수록 "이미 비슷한 곳에 와 봤다"고 판단합니다.

$$
r^{episodic}_t = \frac{1}{\sqrt{\sum_{f_i \in N_k}K\big(f(x_t), f_i\big)} + c}
$$

$$
K(x, y) = \frac{\epsilon}{\dfrac{d^2(x, y)}{d_m^2} + \epsilon}
$$

- $$K$$: 가까울수록 1에 가까운 커널 (Dirac delta를 부드럽게 근사한 것). 분모의 합이 **pseudo-count** 역할
- $$d_m$$: $$k$$-최근접 거리의 running average (스케일 정규화)
- $$c$$: 작은 상수 (pseudo-count constant)
- 메모리는 **에피소드마다 초기화**

현재 상태와 이전에 방문한 상태들의 **유사도**로 보상이 정해집니다.

### ③ Life-long Novelty Module

여러 에피소드에 걸친 새로움은 [RND]({% post_url 2026-08-30-random-network-distillation %})로 측정합니다. RND의 예측 오차 $$\text{err}(x_t)$$를 정규화해 **life-long curiosity factor** $$\alpha_t$$를 만듭니다.

$$
\alpha_t = 1 + \frac{\text{err}(x_t) - \mu_e}{\sigma_e}
$$

($$\mu_e, \sigma_e$$: 오차의 running mean과 표준편차)

### 결합

$$\alpha_t$$를 $$[1, L]$$로 잘라서($$L = 5$$) episodic reward에 **곱합니다.**

- 여러 에피소드에서 익숙해진 상태는 $$\alpha_t \to 1$$이 되어 **배율 효과가 사라지고**
- 새로운 상태는 최대 $$L$$배까지 보상이 커집니다.
- 하지만 $$\alpha_t$$가 1 아래로 내려가지는 않으므로, life-long 새로움이 사라져도 **episodic 보상은 남습니다.** 이것이 "포기하지 않는" 이유입니다.

최종 보상은 extrinsic과 intrinsic의 가중합입니다.

$$
r_t = r^e_t + \beta\,r^i_t
$$

```python
import torch

def episodic_reward(emb, memory, k=10, eps=1e-3, c=1e-3, dm2_running=None):
    """emb: (D,), memory: (M, D) — 이번 에피소드의 embedding들"""
    if len(memory) == 0:
        return 1.0 / c
    d2 = ((memory - emb) ** 2).sum(-1)                     # 제곱 거리
    d2_k, _ = d2.topk(min(k, len(d2)), largest=False)      # k-최근접
    dm2 = dm2_running if dm2_running is not None else d2_k.mean()
    kernel = eps / (d2_k / (dm2 + 1e-8) + eps)
    return 1.0 / (kernel.sum().sqrt() + c)

def ngu_reward(r_episodic, rnd_err, mu, sigma, L=5.0):
    alpha = 1 + (rnd_err - mu) / (sigma + 1e-8)
    return r_episodic * min(max(alpha, 1.0), L)

torch.manual_seed(0)
memory = torch.randn(50, 8)                       # 이번 에피소드에서 본 50개 상태의 embedding
d2_all = ((memory[:, None] - memory[None]) ** 2).sum(-1)
dm2 = d2_all.topk(11, largest=False).values[:, 1:].mean()   # k-NN 거리의 평균 (running average 대용)
revisit = memory[0] + 0.01 * torch.randn(8)       # 거의 같은 곳을 다시 방문
new = torch.randn(8) * 3                          # 멀리 떨어진 새 상태
print(float(episodic_reward(revisit, memory, dm2_running=dm2)))   # 약 1.04
print(float(episodic_reward(new, memory, dm2_running=dm2)))       # 약 15.4
print(ngu_reward(0.5, rnd_err=3.0, mu=1.0, sigma=0.5))   # α=5 → 0.5 × 5 = 2.5
print(ngu_reward(0.5, rnd_err=0.2, mu=1.0, sigma=0.5))   # α<1 → 1로 잘림 → 0.5 그대로
```

같은 에피소드에서 이미 와 본 곳 근처는 보상이 약 1, 처음 가는 곳은 약 15로 열 배 넘게 차이 납니다. 그리고 마지막 줄처럼 life-long 새로움이 다 사라져도(RND 오차가 평균보다 작아도) 배율이 1에서 잘릴 뿐 **episodic 보상 0.5는 그대로 남습니다.** "포기하지 않는" 성질이 이 `max(α, 1)` 한 줄에서 나옵니다.

## 4. NGU 에이전트

- **여러 정책을 하나의 네트워크로**: intrinsic reward 계수 $$\beta_j$$와 discount $$\gamma_j$$가 다른 $$N$$개의 쌍 $$\{(\beta_j, \gamma_j)\}_{j=0}^{N-1}$$을 정하고, 네트워크가 $$j$$를 입력으로 받아 $$Q(x, a, j)$$를 출력합니다([UVFA]({% post_url 2026-08-20-hindsight-experience-replay %}) 방식).
  - $$\beta_0 = 0$$: 순수하게 extrinsic만 보는 **활용 정책**
  - $$\beta$$가 클수록 **탐험 정책**, $$\gamma$$가 작을수록 짧은 시야
- **R2D2 기반**: LSTM을 쓰는 분산 강화학습 에이전트 R2D2를 backbone으로, 여러 actor가 각자 다른 $$j$$로 별도 환경에서 데이터를 모읍니다.
- 탐험 정책들이 모은 데이터를 활용 정책도 함께 학습하므로, 활용 정책은 **탐험 덕분에 발견한 보상**을 이용하게 됩니다.

## 5. 실험

### Hard exploration과 Dense reward 게임

R2D2를 baseline(당시 Atari 전체 SOTA)으로, 각 게임의 SOTA 알고리즘과 비교했습니다.

- **Pitfall!**에서 시연 데이터나 사람이 만든 특징 없이 **처음으로 양수 점수**를 얻었고, Montezuma's Revenge 등 어려운 탐험 게임에서 큰 향상을 보였습니다.
- Pong, Breakout 같은 **dense reward 게임에서도 성능을 거의 잃지 않았습니다.** 탐험 보너스가 쉬운 게임을 방해하지 않았다는 뜻입니다.

### Atari 57개 게임

- 57개 중 **51개 게임에서 사람 점수를 넘었습니다.**
- 평균 점수로는 당시 benchmark였던 R2D2를 넘지 못했지만, 일부 게임에서 SOTA와 사람 점수를 크게 넘는 의미 있는 결과를 보였습니다.

## 6. 정리

| 구성 요소 | 역할 | 사라지는 속도 |
|---|---|---|
| Embedding network | 제어 가능한 특징만 추출 (inverse dynamics) | - |
| Episodic novelty | $$k$$-NN pseudo-count, 에피소드마다 리셋 | **빠름** (한 에피소드 안) |
| Life-long novelty | RND 오차로 배율 $$\alpha_t \in [1, L]$$ | **느림** (학습 전체) |
| 정책 가족 | $$(\beta_j, \gamma_j)$$별 탐험·활용 정책 동시 학습 | - |

NGU의 남은 문제는 "모든 정책을 똑같이 학습한다"는 점이었습니다. 이를 **meta-controller**로 해결해 Atari 57개 게임 **전부**에서 사람을 넘은 것이 [Agent57]({% post_url 2026-09-16-agent57 %})입니다.

## References

- Badia et al., "Never Give Up: Learning Directed Exploration Strategies," ICLR 2020. [arXiv:2002.06038](https://arxiv.org/abs/2002.06038)
- Kapturowski et al., "Recurrent Experience Replay in Distributed Reinforcement Learning (R2D2)," ICLR 2019.
- Burda et al., "Exploration by Random Network Distillation," ICLR 2019.
- Schaul et al., "Universal Value Function Approximators," ICML 2015.
