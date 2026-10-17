---
title: "[RL] Hindsight Experience Replay: 실패한 경험에서도 배우는 법"
date: 2026-08-20 19:17:00 +0900
categories: [Reinforcement Learning, Model-Free]
tags: [reinforcement-learning, model-free, her, goal-conditioned, sparse-reward, ddpg, uvfa, off-policy, robotics]
description: 목표에 도달했을 때만 보상을 주는 sparse reward 환경에서, 실패한 궤적을 '실제로 도착한 곳이 목표였다'고 다시 저장해 학습하는 Hindsight Experience Replay(HER)를 코드와 함께 정리합니다.
math: true
---

> 로봇 팔이 물체를 목표 위치로 옮기는 과제에서, **목표에 도달했을 때만 보상을 주면** 처음에는 거의 한 번도 성공하지 못해 아무것도 배우지 못합니다. **Hindsight Experience Replay(HER)**(Andrychowicz et al., NeurIPS 2017)는 "원래 목표는 실패했지만, **실제로 도착한 곳을 목표로 했다면 성공**이다"라고 경험을 다시 해석해서 이 문제를 풉니다.

## 1. Contribution

- **Sparse reward 환경**에서 에이전트가 샘플 효율적으로 학습하는 방법
- **어떤 off-policy 알고리즘**(DQN, DDPG 등)에도 붙여 쓸 수 있음
- 복잡한 **reward engineering(보상 설계)**이 필요 없음

## 2. 배경

### DDPG (Deep Deterministic Policy Gradient)

DQN은 이산 행동에서만 잘 동작합니다($$\max_a Q$$를 구해야 하므로). **DDPG**는 이를 **연속 행동 공간**으로 확장한 알고리즘입니다.

- **Actor** $$\pi(s)$$: 상태에서 연속 행동을 바로 출력 (결정적 정책)
- **Critic** $$Q(s, a)$$: 행동의 가치를 평가하고, actor는 critic의 Q 값이 커지는 방향으로 학습
- DQN의 **replay buffer**와 **target network** 기법을 그대로 사용 (off-policy)

로봇 팔의 관절 토크처럼 연속적인 행동을 다루므로, HER 논문의 실험은 DDPG를 기반으로 합니다.

### UVFA (Universal Value Function Approximator)

목표가 **여러 개**인 문제를 위해, 가치 함수를 목표 $$g$$에 대해서도 일반화한 것입니다.

- 목표마다 보상 함수 $$r_g: S \times A \to \mathbb{R}$$이 있음
- 에피소드 시작 때 초기 상태와 목표 $$(s_0, g)$$를 샘플링하고, 에피소드 동안 목표는 고정
- 정책과 Q 함수가 상태뿐 아니라 **목표도 입력**으로 받음

$$
\pi: S \times G \to A,\qquad Q^\pi(s_t, a_t, g) = \mathbb{E}\Big[\sum_{i \ge t}\gamma^{i-t}r_g(s_i, a_i)\Big]
$$

구현은 간단히 **상태와 목표를 이어 붙여(concat)** 네트워크에 넣으면 됩니다.

## 3. 동기: Bit-Flipping 문제

HER의 필요성을 보여 주는 장난감 문제입니다.

- 상태: $$n$$개의 비트 배열 $$s \in \{0, 1\}^n$$
- 행동: $$i$$번째 비트를 뒤집기 ($$n$$가지)
- 에피소드마다 초기 상태 $$s_0$$와 목표 상태 $$g$$를 무작위로 정함
- 보상: 목표 상태가 아니면 **-1**, 목표에 도달하면 0

$$
r_g(s, a) = -\big[s \neq g\big]
$$

상태 공간이 $$2^n$$개라, $$n$$이 조금만 커져도 무작위 탐험으로 목표에 도달할 확률은 거의 0입니다. 에이전트는 **항상 -1만** 받고, 어떤 행동이 좋은지 알려 주는 신호가 전혀 없습니다. 실제로 일반 DQN은 **$$n > 13$$이면 실패**했고, 같은 DQN에 HER을 붙이면 $$n = 50$$까지 풀었습니다.

목표까지의 거리로 보상을 주는 **reward shaping**($$-\lVert s - g\rVert^2$$ 등)으로 해결할 수도 있지만, 복잡한 실제 문제에서는 좋은 shaping을 설계하기 어렵고 잘못 설계하면 엉뚱한 행동을 학습합니다. 그래서 논문은 shaping을 쓰지 않습니다.

## 4. HER의 아이디어

에피소드 $$s_1, s_2, \dots, s_T$$가 목표 $$g$$에 도달하지 못했다면, 모든 transition의 보상은 -1뿐입니다. 하지만 이 궤적은 **"$$s_T$$로 가는 방법"**으로서는 완벽한 시연입니다.

그래서 replay buffer에 **두 번** 저장합니다.

1. 원래 목표 $$g$$로: $$(s_t, a_t, r_g, s_{t+1}, g)$$ → 모두 실패(-1)
2. **실제로 도착한 상태 $$s_T$$를 목표로 바꿔서**: $$(s_t, a_t, r_{s_T}, s_{t+1}, s_T)$$ → 마지막에 **성공**

두 번째 궤적 덕분에 에이전트는 "이렇게 하면 $$s_T$$에 갈 수 있다"를 배웁니다. 이렇게 다양한 상태에 도달하는 법을 익히다 보면, 결국 원래 목표에 도달하는 법도 일반화됩니다. 사람이 축구공을 골대 옆으로 찼을 때 "저 방향으로 차려면 이렇게 하면 되는구나"를 배우는 것과 같습니다.

## 5. 알고리즘

1. Off-policy 알고리즘 $$\mathbb{A}$$(DQN, DDPG 등)와 replay buffer $$R$$ 초기화
2. 매 에피소드
   1. 목표 $$g$$와 초기 상태 $$s_0$$ 샘플링
   2. $$t = 0, \dots, T-1$$ 동안 **상태와 목표를 concat**해 행동 $$a_t = \pi(s_t \Vert g)$$ 선택, 실행
   3. 각 transition을 원래 목표로 $$R$$에 저장
   4. 추가 목표 집합 $$G' = \mathbb{S}(\text{현재 에피소드})$$를 샘플링하고, 각 $$g' \in G'$$로 **보상을 다시 계산해** $$R$$에 저장
3. $$N$$번 반복: $$R$$에서 미니배치를 샘플링해 $$\mathbb{A}$$로 학습

```python
import numpy as np

def her_relabel(episode, reward_fn, k=4, strategy="future"):
    """episode: [(s, a, s_next, g, achieved_next), ...]
    achieved_next: s_next에서 실제로 달성한 목표 (예: 물체의 위치)"""
    transitions = []
    T = len(episode)
    for t, (s, a, s_next, g, ach) in enumerate(episode):
        transitions.append((s, a, reward_fn(ach, g), s_next, g))            # 원래 목표

        if strategy == "final":
            new_goals = [episode[-1][4]]
        elif strategy == "future":                                          # 이후 시점에서 k개
            idx = np.random.randint(t, T, size=k)
            new_goals = [episode[i][4] for i in idx]
        else:                                                               # "episode"
            idx = np.random.randint(0, T, size=k)
            new_goals = [episode[i][4] for i in idx]

        for g_new in new_goals:                                             # 목표 바꿔서 다시 저장
            transitions.append((s, a, reward_fn(ach, g_new), s_next, g_new))
    return transitions

# 예: 목표와 거리 ε 이내면 0, 아니면 -1
def reward_fn(achieved, goal, eps=0.05):
    return 0.0 if np.linalg.norm(achieved - goal) <= eps else -1.0

# 1차원 예시: 목표는 위치 1.0인데 실제로는 0.3까지만 이동한 실패 에피소드
np.random.seed(0)
g = np.array([1.0])
pos = [np.array([x]) for x in [0.0, 0.1, 0.2, 0.3]]
episode = [(pos[t], +1, pos[t + 1], g, pos[t + 1]) for t in range(3)]

out = her_relabel(episode, reward_fn, k=2)
print("저장된 transition 수:", len(out))
print("원래 목표 보상:", [r for (_, _, r, _, gg) in out if gg is g])
print("바꾼 목표 보상:", [r for (_, _, r, _, gg) in out if gg is not g])
```

```text
저장된 transition 수: 9
원래 목표 보상: [-1.0, -1.0, -1.0]
바꾼 목표 보상: [0.0, -1.0, 0.0, -1.0, 0.0, 0.0]
```

원래 목표로는 모두 -1이지만, 목표를 바꿔 저장한 transition 중 절반 이상이 보상 0(성공)을 받습니다. 학습 신호가 없던 에피소드에서 신호가 생긴 것입니다.

보상 함수 $$r_g$$를 알고 있어야(목표가 바뀌었을 때 다시 계산할 수 있어야) 한다는 점이 HER의 전제입니다.

### 왜 off-policy 알고리즘에만 쓸 수 있나

목표를 $$g'$$로 바꾼 transition은 **에이전트가 $$g'$$를 노리고 한 행동이 아닙니다.** 실제로는 $$\pi(\cdot \mid s, g)$$로 행동했는데, 데이터에는 $$\pi(\cdot \mid s, g')$$의 경험처럼 저장됩니다. 즉 relabel된 데이터는 *지금 학습하려는 정책과 다른 정책이 만든 데이터*이므로, 이런 데이터로도 학습할 수 있는 off-policy 알고리즘(DQN, DDPG, SAC 등)이 필요합니다. PPO 같은 on-policy 알고리즘에 그대로 붙이면 데이터 분포가 맞지 않습니다.

> 위 코드는 이해를 위해 relabel한 transition을 미리 저장하지만, OpenAI Baselines 같은 실제 구현은 에피소드를 그대로 저장해 두고 **미니배치를 뽑을 때** 일정 확률(예: $$k = 4$$면 80%)로 목표를 바꿉니다. 메모리를 $$k+1$$배 쓰지 않아도 되고, 매번 다른 목표로 relabel되는 효과도 있습니다.
{: .prompt-info }

## 6. 실험

### 로봇 조작 과제

7자유도 로봇 팔(Fetch)로 세 가지 과제를 실험했습니다. 목표는 물체의 목표 위치이고, 물체가 목표에서 일정 거리 안에 있을 때만 성공입니다.

| 과제 | 내용 |
|---|---|
| **Pushing** | 테이블 위 상자를 밀어서 목표 위치로 |
| **Sliding** | 퍽(puck)을 쳐서 미끄러뜨려 멀리 있는 목표로 (팔이 닿지 않는 곳) |
| **Pick-and-place** | 상자를 집어 공중의 목표 위치로 |

- 일반 **DDPG는 어떤 과제도 풀지 못했습니다.**
- **Count-based exploration**을 더한 DDPG는 sliding에서만 약간 진전이 있었습니다.
- **DDPG + HER는 세 과제 모두 높은 성공률**을 달성했습니다. Sparse reward 환경에서 학습을 가능하게 한 것입니다.
- 시뮬레이션에서 학습한 정책을 **실제 로봇**에 옮겨서도 성공했습니다.

### 목표가 하나뿐일 때도 도움이 되나

매 에피소드 목표가 **항상 같은** 환경에서도 HER을 쓰면 쓰지 않을 때보다 성능이 좋았습니다. 다만 **여러 목표로 학습할 때 더 빨리 수렴**해서, 실제로 원하는 목표가 하나여도 다양한 목표로 학습하는 편이 낫다는 결론입니다.

### 어떤 목표로 다시 저장할까

추가 목표를 고르는 전략 네 가지를 비교했습니다.

| 전략 | 추가 목표 |
|---|---|
| **final** | 에피소드의 **마지막 상태** |
| **future** | 현재 transition **이후**에 도달한 상태 중 무작위 $$k$$개 |
| **episode** | 같은 에피소드 안의 상태 중 무작위 $$k$$개 (위치 무관) |
| **random** | 학습 전체에서 도달한 상태 중 무작위 $$k$$개 |

**future 전략, $$k = 4 \sim 8$$**일 때 가장 좋았습니다. 너무 크면 원래 목표의 비중이 줄어 오히려 성능이 떨어졌습니다. 직관적으로, 지금 행동이 **실제로 영향을 줄 수 있는 미래 상태**를 목표로 삼는 것이 가장 유용한 학습 신호가 됩니다.

## 7. 정리

| 개념 | 핵심 |
|---|---|
| Sparse reward | 성공할 때만 보상 → 초기에는 학습 신호가 없음 |
| UVFA | 목표를 입력으로 받는 가치 함수 $$Q(s, a, g)$$ |
| HER | 실패한 궤적을 **실제로 도달한 상태를 목표로** 다시 저장 |
| 조건 | Off-policy 알고리즘, 목표별 보상을 다시 계산할 수 있어야 함 |
| 최적 전략 | future, $$k = 4 \sim 8$$ |

HER은 보상 설계 없이도 어려운 로봇 조작을 학습할 수 있다는 것을 보여 줬고, 이후 goal-conditioned RL과 로봇 학습의 기본 기법이 됐습니다.

## References

- Andrychowicz et al., "Hindsight Experience Replay," NeurIPS 2017. [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)
- Lillicrap et al., "Continuous control with deep reinforcement learning (DDPG)," ICLR 2016.
- Schaul et al., "Universal Value Function Approximators," ICML 2015.
- Plappert et al., "Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research," 2018.
