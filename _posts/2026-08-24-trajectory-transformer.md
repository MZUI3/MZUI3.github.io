---
title: "[RL] Trajectory Transformer: 강화학습을 하나의 큰 시퀀스 모델링 문제로 보기"
date: 2026-08-24 20:05:00 +0900
categories: [Reinforcement Learning, Model-Based]
tags: [reinforcement-learning, model-based, offline-rl, transformer, sequence-modeling, beam-search, trajectory-transformer]
description: 정책·모델·가치 함수를 따로 두지 않고 궤적 전체를 이산 토큰 시퀀스로 보고 GPT처럼 학습한 뒤 beam search로 계획하는 Trajectory Transformer의 아이디어, 이산화 방법, 실험 결과를 정리합니다.
math: true
---

> [EfficientZero]({% post_url 2026-08-21-efficientzero %})가 환경 모델을 **잘 학습하는 법**에 집중했다면, **Trajectory Transformer**(Janner, Li & Levine, NeurIPS 2021)는 질문을 바꿉니다. "정책, 모델, 가치 함수를 따로 만들 필요 없이, **궤적(trajectory) 전체를 하나의 문장처럼** 생성하면 되지 않을까?" [GPT]({% post_url 2026-07-01-transformer-decoder-to-gpt %})가 다음 단어를 예측하듯, 다음 상태·행동·보상을 예측하는 강화학습입니다.

## 1. 기존 강화학습의 두 가지 기둥

기존 RL은 대부분 두 가정 위에 서 있습니다.

### Factorized Single-Step

현재 상태와 행동에서 **바로 다음 상태와 보상만** 예측합니다.

$$
P(s_{t+1}, r_t \mid s_t, a_t)
$$

이렇게 한 step씩 쪼개서 모델링하고, 이를 이어 붙여 먼 미래를 예측합니다.

### Markov Property

미래는 **현재 상태에만** 의존하고 과거 이력과는 무관하다고 가정합니다.

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1} \mid s_t, a_t)
$$

이 덕분에 문제가 단순해지지만, 대가가 있습니다.

### Long-Horizon 문제

체스나 바둑처럼 **당장의 한 수가 아니라 먼 미래의 결과**를 고려해야 하는 문제입니다. 기존 방법은 이를 작은 하위 문제(sub-problem)로 쪼개 가지치기하듯 풉니다. 그런데 single-step 모델을 여러 번 이어 붙이면 **한 step의 작은 오차가 누적(compounding error)**되어, 몇십 step만 지나도 예측이 크게 틀어집니다.

## 2. 아이디어: 궤적을 하나의 문장으로

강화학습의 목표는 결국 "**높은 보상을 얻는 행동 시퀀스를 생성하는 것**"입니다. 이는 언어 모델이 "그럴듯한 문장을 생성하는 것"과 구조가 같습니다.

그렇다면 궤적을 **구조가 없는 하나의 긴 시퀀스**로 보고, LLM에서 검증된 **Transformer**로 그 분포를 모델링하면 됩니다.

| 언어 모델 | Trajectory Transformer |
|---|---|
| 단어 토큰 | 상태·행동·보상의 **이산화된 토큰** |
| 다음 단어 예측 | 다음 상태/행동/보상 토큰 예측 |
| 문장 생성 (beam search) | **계획(planning)** (beam search) |

### 기존의 "Transformer를 쓴 RL"과 다른 점

AlphaStar나 일부 멀티에이전트 알고리즘 등에서도 Transformer를 쓰지만, 이들은 **기존 RL 학습 알고리즘(Q-learning, policy gradient)을 그대로 두고** 신경망 구조만 Transformer로 바꾼 것입니다. Trajectory Transformer는 반대로 **RL 학습 알고리즘을 빼고**, 궤적의 분포를 모델링하는 시퀀스 모델 하나만 남깁니다.

## 3. 궤적을 토큰으로: 이산화

Transformer에 넣으려면 연속적인 상태·행동·보상을 **이산 토큰**으로 바꿔야 합니다. 상태가 $$N$$차원, 행동이 $$M$$차원이라면, 각 시점을 **차원 하나당 토큰 하나**로 펼쳐 이어 붙입니다.

$$
\tau = \big(\dots,\ s_t^1, s_t^2, \dots, s_t^N,\ a_t^1, a_t^2, \dots, a_t^M,\ r_t,\ \dots\big)
$$

한 시점이 $$N + M + 1$$개의 토큰이 되고, 궤적 길이가 $$T$$면 시퀀스 길이는 $$T(N + M + 1)$$입니다. (Offline RL 실험에서는 계획에 쓰기 위해 reward-to-go도 토큰으로 함께 넣습니다.)

### 차원별로 따로 이산화

상태 벡터 전체를 한 번에 이산화하면 가능한 토큰 수가 차원 수에 대해 지수적으로 늘어납니다. 그래서 **차원마다 독립적으로** $$V$$개(논문은 100개)의 구간으로 나눕니다. 차원 사이의 관계는 Transformer가 앞 차원의 토큰을 보고 다음 차원의 토큰을 예측하면서 학습합니다.

구간을 나누는 방법은 두 가지입니다.

- **Uniform**: 각 차원의 최솟값~최댓값을 **같은 폭**으로 나눔
- **Quantile**: 각 구간에 **같은 개수의 데이터**가 들어가도록 나눔

```python
import numpy as np

def discretize(x, n_bins=100, mode="uniform"):
    """한 차원의 연속값 x를 0 ~ n_bins-1 토큰으로 바꿈"""
    if mode == "uniform":                                    # 같은 폭으로 나누기
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
    else:                                                    # quantile: 같은 개수로 나누기
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)

rng = np.random.default_rng(0)
x = rng.exponential(1.0, 10_000)                             # 꼬리가 긴 차원 (예: 속도)
for mode in ["uniform", "quantile"]:
    counts = np.bincount(discretize(x, mode=mode), minlength=100)
    print(f"{mode:8s} 빈 토큰 {(counts == 0).sum():2d}개, 토큰당 최대 {counts.max()}개, 최소 {counts.min()}개")
```

```text
uniform  빈 토큰 15개, 토큰당 최대 792개, 최소 0개
quantile 빈 토큰  0개, 토큰당 최대 100개, 최소 100개
```

(NumPy 2.5에서 실행)

지수 분포처럼 꼬리가 긴 차원을 uniform으로 나누면, 드문 큰 값 때문에 범위가 넓어져 **100개 중 15개 토큰에는 데이터가 하나도 없고**, 값이 몰린 앞쪽 구간은 토큰 하나에 792개가 들어가 해상도가 떨어집니다. Quantile은 모든 토큰이 정확히 같은 양의 데이터를 갖습니다.

### 학습

GPT와 같은 **decoder-only Transformer**를 쓰고, 이전 토큰들이 주어졌을 때 다음 토큰의 log-likelihood를 최대화합니다(teacher forcing).

$$
\mathcal{L}(\tau) = \sum_{t=1}^{T}\Big[\sum_{i=1}^{N}\log P_\theta(s_t^i \mid s_t^{<i}, \tau_{<t}) + \sum_{j=1}^{M}\log P_\theta(a_t^j \mid a_t^{<j}, s_t, \tau_{<t}) + \log P_\theta(r_t \mid a_t, s_t, \tau_{<t})\Big]
$$

강화학습 문제가 **순수한 지도학습** 문제로 바뀐 것입니다.

## 4. Planning = Beam Search

학습된 모델로 행동을 고르는 방법은 LLM의 **디코딩**과 같습니다. 논문은 한 step씩 가치를 계산하는 대신, 시퀀스 모델 위에서 **Beam Search**로 여러 step 앞을 한꺼번에 탐색합니다.

Beam search는 매 단계 가능한 확장 후보를 만들고, **점수가 가장 높은 $$B$$개(beam width)**만 남기며 진행합니다.

1. 빔 $$Y_0$$를 빈 시퀀스로 초기화
2. $$t = 1, \dots, T$$ 동안
   - 후보 집합 $$C_t$$ = 이전 빔 $$Y_{t-1}$$의 각 시퀀스에 가능한 모든 토큰 하나를 붙인 것
   - $$Y_t$$ = $$C_t$$ 중 점수가 높은 $$B$$개
3. 마지막 빔에서 점수가 가장 높은 시퀀스 반환

```python
def beam_search(model, prefix, horizon, beam_width, score_fn):
    beams = [(prefix, 0.0)]
    for _ in range(horizon):
        candidates = []
        for seq, score in beams:
            probs = model.next_token_probs(seq)                # (V,)
            for tok, p in enumerate(probs):
                candidates.append((seq + [tok], score + score_fn(seq, tok, p)))
        beams = sorted(candidates, key=lambda c: c[1], reverse=True)[:beam_width]
    return beams[0][0]
```

점수 함수만 바꾸면 여러 문제를 같은 모델로 풉니다.

| 목적 | 점수 | 의미 |
|---|---|---|
| **Imitation learning** | 토큰의 log-likelihood | 데이터에서 가장 그럴듯한 행동 (행동 복제) |
| **Goal-reaching** | log-likelihood, 단 **목표 상태를 시퀀스 맨 앞에** 넣음 | 목표에 도달하는 궤적 |
| **Offline RL** | 예측한 **보상 + reward-to-go** | 보상을 최대화하는 궤적 (Q-guided planning) |

Offline RL에서는 log-likelihood 대신 예측 보상과 가치로 점수를 매겨, 가능한 행동 시퀀스 중 **Q값이 가장 높은 것**을 고릅니다.

## 5. 실험

### 장기 예측 정확도

Humanoid 환경에서 100 step 길이의 궤적을 생성해 실제 움직임(reference)과 비교했습니다.

- 비교 대상 **PETS**(Probabilistic Ensembles with Trajectory Sampling)는 single-step만 예측하는 Markovian 앙상블 모델로, 몇 step만 지나도 **오차가 매우 커졌습니다.**
- Trajectory Transformer는 **100 step 넘게** reference와 거의 구분되지 않는 궤적을 안정적으로 생성했습니다.

누적 오차를 정량적으로 보기 위해 각 모델에서 궤적 1000개를 샘플링해 시점별 상태 분포에서 reference 상태의 likelihood를 비교했을 때도 Trajectory Transformer가 크게 앞섰습니다.

흥미로운 점은 **직전 1 step만 보도록 제한한 Markovian Transformer**도 비슷하게 좋았다는 것입니다. 즉 긴 문맥보다 **Transformer 구조와 차원별 autoregressive 이산화**에서 오는 표현력이 장기 정확도에 큰 역할을 했습니다. 다만 상태 차원을 무작위로 절반 가린 **부분 관측** 환경에서는 원래의 긴 문맥 버전이 더 좋아, 긴 조건부(long-horizon conditioning)의 역할도 확인됐습니다.

### Attention 패턴

학습된 attention에서 두 가지 패턴이 보였습니다.

1. **직전 transition에 집중**: Markovian 전략을 스스로 학습
2. **상태는 같은 차원에, 행동은 과거 행동에 집중**: 행동이 과거 상태에만 의존하는 behavior cloning과 반대이고, 일부 궤적 최적화 알고리즘의 **action filtering**과 닮은 패턴

### Offline RL (D4RL)

D4RL 벤치마크의 MuJoCo 3개 환경, 3가지 데이터셋에서 비교했습니다.

| 데이터셋 | 구성 |
|---|---|
| medium | 중간 성능 정책이 만든 데이터 |
| med-expert | medium + 최고 성능 정책 데이터 혼합 |
| mixed (medium-replay) | medium 정책을 학습할 때 쌓인 replay buffer |

| 비교 대상 | 분류 |
|---|---|
| CQL (Conservative Q-Learning), BRAC | Model-free SOTA |
| MBOP | Single-step dynamics 기반 planning |
| Decision Transformer | Return-conditioning 시퀀스 모델 |
| BC | Behavior cloning |

대부분의 데이터셋에서 기존 방법과 **비슷하거나 더 좋은 성능**을 냈습니다. 다만 HalfCheetah med-expert에서는 expert 데이터의 속도 범위가 매우 넓어 **uniform 이산화가 세밀하지 못해** 성능이 낮았고, quantile 이산화는 그 두 배가 넘는 return을 냈습니다.

### Goal-Reaching

Four Rooms 환경에서 목표 상태를 조건으로 준 likelihood-maximizing planning만으로, **보상 설계(reward shaping)나 보상 정보 없이** 목표까지 계획하는 데 성공했습니다.

## 6. 의의와 한계

**의의**

- 정책, 모델, 가치 함수를 **하나의 시퀀스 모델로 통합**하고 beam search만으로 SOTA급 성능
- 연속 데이터를 이산화해 쓰는 것이 기존 연속 dynamics 모델보다 효과적
- 적절한 모델을 고르면 **RL을 지도학습으로 재구성**할 수 있음을 보여 줌

**한계**

- Single-step 모델보다 **느리고 자원 소모가 큼**. 차원마다 토큰을 생성하고 beam search까지 하므로 실시간 제어에는 부담
- 이산화 해상도에 성능이 좌우됨

### Decision Transformer와의 비교

같은 시기에 나온 **Decision Transformer**(Chen et al., 2021)도 RL을 시퀀스 모델링으로 풀지만, 방향이 다릅니다.

| | Trajectory Transformer | Decision Transformer |
|---|---|---|
| 토큰 단위 | 상태·행동의 **차원별** 이산 토큰 | 상태·행동·return을 **벡터 그대로** 임베딩 |
| 예측 대상 | 상태, 행동, 보상 모두 (환경 모델 역할) | **행동만** |
| 행동 선택 | **Beam search로 계획** | 원하는 return을 조건으로 주고 바로 출력 |
| 성격 | Model-based planning에 가까움 | Return-conditioned 정책에 가까움 |

이 "시퀀스 모델로 계획하기" 아이디어는 이후 **Diffusion 모델로 궤적 전체를 생성**하는 [Diffuser]({% post_url 2026-09-19-diffuser %}) 계열 연구로 이어집니다.

## References

- Janner, Li & Levine, "Offline Reinforcement Learning as One Big Sequence Modeling Problem," NeurIPS 2021. [arXiv:2106.02039](https://arxiv.org/abs/2106.02039)
- Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling," NeurIPS 2021.
- Chua et al., "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)," NeurIPS 2018.
- Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning," NeurIPS 2020.
- Yu et al., "MOPO: Model-based Offline Policy Optimization," NeurIPS 2020.
- Fu et al., "D4RL: Datasets for Deep Data-Driven Reinforcement Learning," 2020.
