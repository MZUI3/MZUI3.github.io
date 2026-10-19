---
title: "[RL] TRPO에서 PPO까지: 안정적인 Policy Gradient의 계보"
date: 2026-07-05 17:43:00 +0900
categories: [Reinforcement Learning, Model-Free]
tags: [reinforcement-learning, policy-gradient, trpo, ppo, gae, actor-critic, model-free]
description: Policy gradient의 step size 문제에서 출발해 TRPO의 surrogate objective와 KL 제약, 이를 clipping으로 단순화한 PPO, 그리고 GAE까지 수식과 실행 가능한 코드로 정리합니다.
math: true
---

> 강화학습 시리즈의 첫 글입니다. PPO는 지금도 가장 많이 쓰이는 on-policy 알고리즘이고, LLM의 RLHF에서도 핵심으로 쓰입니다. PPO가 왜 그런 모양이 됐는지 이해하려면 TRPO부터 보는 게 가장 빠르기 때문에 두 논문을 한 글로 묶었습니다.

## 1. 출발점: Policy Gradient의 불안정성

Policy gradient는 정책 $$\pi_\theta(a\mid s)$$의 기대 return을 직접 최대화합니다.

$$
\eta(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right],\qquad
\nabla_\theta \eta \approx \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,\hat{A}_t\right]
$$

여기서 **Advantage** $$A(s,a) = Q(s,a) - V(s)$$는 "이 상태에서 이 행동이 평균보다 얼마나 좋은가"를 나타냅니다.

문제는 **step size**입니다.

- 너무 작으면 학습이 느리고
- 너무 크면 정책이 크게 바뀌어 성능이 무너집니다.

지도학습과 달리 강화학습은 **현재 정책이 다음에 쓸 데이터를 만듭니다.** 한 번 나쁜 정책으로 크게 업데이트되면 나쁜 데이터를 모으고, 그걸로 또 학습하면서 회복하기 어렵게 됩니다. 또 데이터 샘플 하나당 gradient 업데이트를 한 번만 할 수 있어서 **샘플 효율도 낮습니다.**

---

## Part 1. TRPO (Schulman et al., 2015)

## 2. 새 정책의 성능을 옛 정책으로 표현하기

TRPO는 다음 항등식에서 출발합니다. 새 정책 $$\tilde\pi$$의 성능은 **옛 정책 $$\pi$$의 성능 + 새 정책이 고른 행동들의 (옛 정책 기준) advantage 합**과 같습니다.

$$
\eta(\tilde\pi) = \eta(\pi) + \mathbb{E}_{\tau\sim\tilde\pi}\left[\sum_t \gamma^t A_\pi(s_t, a_t)\right]
$$

이를 상태 방문 빈도 $$\rho_{\tilde\pi}(s) = \sum_t \gamma^t P(s_t = s)$$로 다시 쓰면

$$
\eta(\tilde\pi) = \eta(\pi) + \sum_s \rho_{\tilde\pi}(s)\sum_a \tilde\pi(a\mid s)\,A_\pi(s,a)
$$

모든 상태에서 뒤쪽 합이 0 이상이면 **성능 향상이 보장**됩니다. 하지만 문제가 있습니다. $$\rho_{\tilde\pi}$$는 **아직 만들지도 않은 새 정책**의 방문 빈도라서 계산할 수 없습니다.

## 3. Surrogate Objective와 Lower Bound

그래서 $$\rho_{\tilde\pi}$$를 옛 정책의 $$\rho_\pi$$로 바꿔치기한 근사식을 씁니다.

$$
L_\pi(\tilde\pi) = \eta(\pi) + \sum_s \rho_\pi(s)\sum_a \tilde\pi(a\mid s)\,A_\pi(s,a)
$$

$$L_\pi$$와 $$\eta$$는 현재 정책에서 값과 gradient가 같습니다. 즉 **충분히 작게만 움직이면** $$L$$을 올리는 것이 곧 $$\eta$$를 올리는 것입니다. 그런데 "얼마나 작게?"에 대한 답이 없습니다.

TRPO는 두 정책의 차이(KL divergence)로 **성능 하한(lower bound)**을 유도합니다.

$$
\eta(\tilde\pi) \ge L_\pi(\tilde\pi) - C \cdot D_{KL}^{\max}(\pi, \tilde\pi),\qquad C = \frac{4\epsilon\gamma}{(1-\gamma)^2}
$$

($$\epsilon = \max_{s,a}\lvert A_\pi(s,a)\rvert$$, 즉 advantage 절댓값의 최댓값입니다. PPO의 clipping 범위 $$\epsilon$$과는 다른 기호입니다.)

우변을 최대화하면 $$\eta$$가 **단조 증가(monotonic improvement)**하는 것이 이론적으로 보장됩니다.

## 4. 실용적인 알고리즘으로 바꾸기

이론식 그대로는 쓰기 어려워서 몇 가지 근사를 합니다.

1. **Penalty → Constraint**: $$C$$가 너무 커서 step이 지나치게 작아지므로, KL을 penalty 대신 **제약 조건**으로 바꿉니다. 이것이 **Trust Region**입니다.
2. **max KL → 평균 KL**: 모든 상태에 대한 max는 계산할 수 없으니 기댓값으로 대체합니다. (논문에서도 이론적 근거가 아닌 heuristic 근사라고 밝힙니다.)
3. **Importance Sampling**: 옛 정책으로 모은 샘플로 새 정책을 평가합니다.

최종 문제는 다음과 같습니다.

$$
\max_\theta\ \mathbb{E}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}\hat{A}_t\right]
\quad \text{s.t.}\quad \mathbb{E}_t\left[D_{KL}\big(\pi_{\theta_{old}}(\cdot\mid s_t)\ \|\ \pi_\theta(\cdot\mid s_t)\big)\right] \le \delta
$$

이 제약 최적화는 **Conjugate Gradient + Line Search**로 풉니다. 샘플 수집 방식은 하나의 trajectory를 따라가는 **single path**와, 일부 상태에서 여러 행동을 다시 rollout해 분산을 줄이는 **vine** 두 가지를 제안합니다(vine은 분산이 작지만 시뮬레이터를 더 많이 호출해야 합니다).

### TRPO의 한계

- 2차 미분(Fisher 정보 행렬) 근사와 conjugate gradient가 들어가 **구현이 복잡**하다.
- 정책과 가치 함수가 파라미터를 공유하거나, dropout 같은 noise가 있는 구조와 잘 맞지 않는다.

---

## Part 2. PPO (Schulman et al., 2017)

## 5. 목표: TRPO의 안정성은 유지하고 구현은 단순하게

PPO는 "**1차 미분(SGD)만으로** TRPO처럼 정책을 조금씩만 바꿀 수 없을까?"에서 출발합니다. 그리고 같은 데이터로 **여러 epoch 동안 minibatch 업데이트**를 할 수 있게 만들어 샘플 효율도 높입니다.

### 왜 KL penalty가 아니었나

TRPO의 constraint를 penalty로 되돌린 형태도 생각할 수 있습니다.

$$
\max_\theta\ \mathbb{E}_t\left[r_t(\theta)\hat{A}_t - \beta\, D_{KL}(\pi_{\theta_{old}} \| \pi_\theta)\right]
$$

하지만 문제마다, 심지어 한 문제의 학습 도중에도 잘 맞는 $$\beta$$ 값 하나를 고르기가 어렵습니다. 그래서 TRPO는 constraint를 썼던 것입니다.

## 6. Clipped Surrogate Objective

확률 비율을 $$r_t(\theta) = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$$로 두면 TRPO의 목적 함수는 $$L^{CPI} = \mathbb{E}_t[r_t(\theta)\hat{A}_t]$$입니다. 제약이 없으면 $$r_t$$가 한없이 커질 수 있으므로, PPO는 **$$r_t$$가 1에서 멀어지는 것 자체를 잘라냅니다.**

$$
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\Big(r_t(\theta)\hat{A}_t,\ \text{clip}\big(r_t(\theta), 1-\epsilon, 1+\epsilon\big)\hat{A}_t\Big)\right]
$$

($$\epsilon = 0.2$$가 기본값)

두 경우로 나눠 보면 직관적입니다.

**$$\hat{A}_t > 0$$ (좋은 행동)**
- 이 행동의 확률을 올리고 싶다 → $$r_t$$가 커지는 방향
- 하지만 $$1+\epsilon$$을 넘으면 objective가 더 이상 늘지 않음 → **과하게 올리지 않음**

**$$\hat{A}_t < 0$$ (나쁜 행동)**
- 이 행동의 확률을 내리고 싶다 → $$r_t$$가 작아지는 방향
- 하지만 $$1-\epsilon$$ 아래로는 objective가 더 이상 늘지 않음 → **과하게 내리지 않음**

숫자로 확인해 보면 더 분명합니다. 옛 정책이 어떤 행동을 0.5 확률로 골랐고 advantage가 +1일 때, 새 정책의 확률에 따라 이 행동의 log 확률에 흐르는 기울기는 다음과 같습니다($$\epsilon = 0.2$$).

| 새 확률 | $$r_t$$ | loss의 기울기 |
|---|---|---|
| 0.5 | 1.0 | -0.5 (확률을 올리라는 신호) |
| 0.6 | 1.2 | -0.6 (아직 범위 경계) |
| 0.8 | 1.6 | **0** (이미 충분히 올렸으므로 신호 없음) |

(아래 9절의 `ppo_loss`와 같은 식으로 계산해 PyTorch 2.13에서 확인한 값입니다.) 범위를 벗어나면 그 샘플은 **기울기가 0이 되어 더 이상 정책을 밀지 않습니다.**

`min`을 씌운 이유는 **비관적인 하한(pessimistic bound)**을 만들기 위해서입니다. clip이 성능을 좋게 보이게 만드는 방향으로는 작동하지 않고, 정책이 너무 멀리 가서 objective가 좋아 보이는 경우만 잘라냅니다. 결과적으로 $$L^{CLIP}$$은 $$L^{CPI}$$의 하한이 되고, 정책 업데이트가 클수록 페널티를 받습니다.

## 7. 전체 Loss

정책과 가치 함수가 네트워크를 공유하면 둘을 합친 loss를 씁니다.

$$
L_t(\theta) = \mathbb{E}_t\left[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2\, S[\pi_\theta](s_t)\right]
$$

- $$L^{VF} = (V_\theta(s_t) - V_t^{target})^2$$: 가치 함수 오차
- $$S$$: **entropy bonus**, 정책이 너무 빨리 확정되지 않도록 탐험을 유지

## 8. GAE: Advantage를 잘 추정하기

Advantage를 몇 step 앞까지 보고 추정할지가 bias–variance trade-off를 결정합니다.

- **1-step** ($$r_t + \gamma V(s_{t+1}) - V(s_t)$$): bias 높음, variance 낮음
- **∞-step** (Monte Carlo return): bias 낮음, variance 높음

**GAE**(Generalized Advantage Estimation)는 모든 n-step 추정치를 지수 가중 평균합니다.

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),\qquad
\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}
$$

$$\lambda=0$$이면 1-step, $$\lambda=1$$이면 Monte Carlo가 되고, 보통 $$\lambda = 0.95$$ 근처를 씁니다. PPO는 길이 $$T$$로 자른(truncated) 버전을 사용합니다.

## 9. 알고리즘과 코드

1. $$N$$개의 actor가 병렬로 $$T$$ step씩 데이터를 수집
2. GAE로 $$\hat{A}_1, \dots, \hat{A}_T$$ 계산
3. 모은 $$NT$$개 데이터로 $$K$$ epoch 동안 minibatch SGD(Adam)로 $$L$$ 최적화
4. $$\theta_{old} \leftarrow \theta$$, 반복

```python
import torch

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    # values는 마지막 상태의 bootstrap 값을 포함해 len(rewards) + 1
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + values[:-1]
    return adv, returns

def ppo_loss(new_logp, old_logp, adv, value, ret, entropy,
             clip_eps=0.2, c1=0.5, c2=0.01):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)          # advantage 정규화
    ratio = torch.exp(new_logp - old_logp)                  # r_t(θ)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()           # L^CLIP (최대화 → 부호 반전)
    value_loss = (value - ret).pow(2).mean()                # L^VF
    return policy_loss + c1 * value_loss - c2 * entropy.mean()

# 3 step 에피소드 (마지막 step에서 종료), 보상 1씩, 가치 추정 0.5
rewards = torch.tensor([1.0, 1.0, 1.0])
values  = torch.tensor([0.5, 0.5, 0.5, 0.0])   # 마지막 0.0은 bootstrap 값 (종료라서 쓰이지 않음)
dones   = torch.tensor([0.0, 0.0, 1.0])
for lam in [0.0, 0.95, 1.0]:
    adv, _ = compute_gae(rewards, values, dones, gamma=0.99, lam=lam)
    print(lam, adv.numpy().round(4))
```

```text
0.0 [0.995  0.995  0.5   ]
0.95 [2.3731 1.4652 0.5   ]
1.0 [2.4701 1.49   0.5   ]
```

$$\lambda = 0$$이면 바로 다음 step의 가치 추정만 믿는 1-step TD($$1 + 0.99 \times 0.5 - 0.5$$), $$\lambda = 1$$이면 실제 보상을 끝까지 더한 Monte Carlo($$1 + 0.99 + 0.99^2 - 0.5$$)와 같아지는 것을 확인할 수 있습니다.

> 실무에서 PPO의 성능은 위 핵심 식만큼이나 **구현 디테일**에 크게 좌우됩니다. advantage 정규화, value loss clipping, 학습률 감소, observation 정규화, 직교 초기화 같은 "코드 수준 최적화"를 빼면 성능이 크게 떨어진다는 분석이 있습니다(Engstrom et al., ICLR 2020). 또 clipping은 기울기를 0으로 만들 뿐이라, 여러 epoch를 돌면 실제 비율 $$r_t$$가 $$[1-\epsilon, 1+\epsilon]$$을 벗어나는 일도 흔합니다.
{: .prompt-warning }

## 10. 실험 결과

- **Surrogate objective 비교** (MuJoCo 7개 환경): clip/penalty가 없는 경우는 무작위 정책보다도 점수가 낮았고, 고정 β와 adaptive β의 KL penalty보다 **clipping($$\epsilon=0.2$$)이 가장 좋았습니다.**
- **연속 제어**: TRPO, CEM, A2C 등과 비교해 대부분의 환경에서 PPO가 가장 좋은 성능을 보였고, 3D Humanoid 달리기·방향 전환·일어나기 같은 고차원 과제도 학습했습니다.
- **Atari**: 학습 전체의 평균 reward(학습 속도)는 PPO가 가장 좋았고, 마지막 100 에피소드 기준(최종 성능)은 ACER가 더 좋았습니다.

## 11. 정리

| | TRPO | PPO |
|---|---|---|
| 정책 변화 제한 | KL divergence **제약** | 확률 비율 **clipping** |
| 최적화 | Conjugate Gradient + Line Search | 일반 SGD / Adam |
| 데이터 재사용 | 샘플당 1회 업데이트 | 여러 epoch minibatch |
| 구현 난이도 | 높음 | 낮음 |
| 이론적 보장 | 단조 향상 lower bound (근사 전) | 경험적 |

PPO는 "이론적으로 엄밀한 TRPO를 **실용적으로 단순화**한 것"이라고 정리할 수 있습니다. 이 단순함 덕분에 로보틱스, 게임, 그리고 **LLM의 RLHF**까지 폭넓게 쓰이게 됐습니다.

다음 글에서는 off-policy 쪽으로 넘어가서, replay buffer를 쓰면서도 actor-critic을 안정적으로 학습시키는 [ACER]({% post_url 2026-08-17-acer %})를 다룹니다.

## References

- Schulman et al., "Trust Region Policy Optimization," ICML 2015.
- Schulman et al., "Proximal Policy Optimization Algorithms," 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Engstrom et al., "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO," ICLR 2020. [arXiv:2005.12729](https://arxiv.org/abs/2005.12729)
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation," ICLR 2016.
- Kakade & Langford, "Approximately Optimal Approximate Reinforcement Learning," ICML 2002.
