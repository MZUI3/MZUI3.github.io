---
title: "[RL] AdaptDiffuser: 스스로 만든 데이터로 진화하는 Diffusion Planner"
date: 2026-09-20 17:52:00 +0900
categories: [Reinforcement Learning, Model-Based]
tags: [reinforcement-learning, model-based, diffusion, planning, self-evolving, synthetic-data, inverse-dynamics, maze2d]
description: Diffuser가 학습 데이터에서 벗어난 과제에서 약한 이유와, 보상으로 유도해 스스로 만든 궤적 중 물리적으로 일관된 것만 골라 다시 학습하는 AdaptDiffuser의 자기 진화 방식을 정리합니다.
math: true
---

> [Diffuser]({% post_url 2026-09-19-diffuser %})는 guide 함수만 바꾸면 새 과제도 풀 수 있다고 했지만, 실제로는 **학습 데이터에서 많이 벗어난 과제**에서 품질이 크게 떨어졌습니다. **AdaptDiffuser**(Liang et al., ICML 2023)는 보상으로 유도해 **스스로 전문가 궤적을 생성**하고, 그중 좋은 것만 골라 다시 학습하는 방식으로 planner를 **자기 진화(self-evolving)**시킵니다.

## 1. Contribution

- **적응 능력**: 학습 때 보지 못한 과제에도 적응할 수 있는 diffusion planner
- **고품질 합성 데이터 생성**: 보상 유도로 합성 전문가 데이터를 만들어 모델을 fine-tuning
- **효율적인 학습**: 사전학습된 Diffuser를 기반으로 새 데이터에서 fine-tuning만 해서 학습을 빠르게
- **자기 진화**: 기존 과제와 처음 보는 과제 모두에서 성능 향상

## 2. 배경 복습

### Trajectory Optimization

이산 시간 시스템 $$s_{t+1} = f(s_t, a_t)$$에서 시점별 보상 $$r(s_t, a_t)$$의 합인 목적 함수 $$\mathcal{J}$$를 최대화하는 행동 시퀀스 $$a^*_{0:T}$$를 찾는 문제입니다.

### 시퀀스 데이터를 생성하는 Diffusion

생성 과정은 반복적인 denoising $$p_\theta(\tau^{i-1} \mid \tau^i)$$이고, 궤적 분포는 각 step 조건부 확률의 곱으로 표현됩니다.

$$
p_\theta(\tau^0) = \int p(\tau^N)\prod_{i=1}^{N}p_\theta(\tau^{i-1} \mid \tau^i)\,d\tau^{1:N}
$$

$$p(\tau^N)$$은 표준 정규분포, $$\tau^0$$은 원래 궤적입니다. 파라미터는 $$p_\theta(\tau^0)$$의 음의 로그 우도에 대한 **ELBO**를 최적화해 학습하며, 실제로는 노이즈 예측 loss로 단순화됩니다.

$$
\mathcal{L}_{denoise} = \mathbb{E}_{i, \tau^0, \epsilon}\big[\lVert\epsilon - \epsilon_\theta(\tau^i, i)\rVert^2\big]
$$

$$\epsilon_\theta$$에서 reverse 과정의 평균 $$\mu_\theta$$로 가는 변환이 닫힌 형태라, 이는 평균을 예측하는 것과 같습니다.

### 과제 지향 생성 (Diffuser 방식)

궤적에 대한 정보 $$y(\tau)$$(reward-to-go $$\mathcal{J}(\tau^0)$$, 만족해야 할 제약 조건 등)를 조건으로 생성합니다.

- **보상 최대화 과제**: 시점별 최적성 변수 $$\mathcal{O}_t$$(최적이면 1)를 두면, 특정 Lipschitz 조건 아래에서 조건부 reverse transition이 **평균을 보상 기울기 방향으로 이동한 가우시안**으로 근사됩니다.

  $$
  p_\theta(\tau^{i-1} \mid \tau^i, \mathcal{O}) \approx \mathcal{N}\big(\mu + \Sigma g,\ \Sigma\big),\qquad g = \nabla_\tau \log p(\mathcal{O} \mid \tau)\big|_{\tau = \mu}
  $$

- **특정 지점 조건 과제**(목표 상태 등): 모든 diffusion step에서 해당 값을 조건값으로 **덮어써서** 단순화합니다(inpainting).

## 3. 문제: Guidance만으로는 부족하다

생성된 궤적의 품질은 guide 기울기 $$g$$뿐 아니라, **무조건부 diffusion 모델이 학습한 평균 $$\mu_\theta$$와 공분산 $$\Sigma$$**에 크게 의존합니다.

- 학습된 $$\mu_\theta$$가 최적 궤적에서 멀리 떨어져 있으면, guidance를 아무리 세게 줘도 결과가 **편향되고 품질이 낮습니다.**
- $$\mu_\theta$$의 품질은 학습 데이터에 달려 있는데, 처음 보는 과제에 맞는 데이터는 학습 세트에 거의 없습니다.

즉, 기존 diffusion planner는 데이터가 부족한 과제에서 성능이 제한되고 **적응 능력이 낮았습니다.**

## 4. Self-Evolved Planning

해결책은 간단합니다. **모델이 부족한 데이터를 스스로 만들게** 합니다.

1. 사전학습된 Diffuser로, 목표 과제의 보상으로 유도해 **다양한 궤적을 생성**
2. **Discriminator**로 품질이 좋고 물리적으로 일관된 궤적만 골라냄
3. 골라낸 궤적을 **합성 전문가 데이터**로 데이터셋에 추가
4. 이 데이터로 diffusion 모델을 **fine-tuning**
5. 1~4를 반복 → 모델이 점점 목표 과제에 맞게 진화

### Dynamics Consistency 문제와 Discriminator

보상이 최대가 되도록 생성된 행동은 상태 변화와 **맞지 않을** 수 있습니다(dynamics consistency 문제). 생성된 상태 시퀀스는 그럴듯해도, 함께 생성된 행동으로는 실제로 그 상태들에 갈 수 없는 경우입니다.

그래서 **inverse dynamics model** $$I$$로 생성된 연속 상태 사이의 **실제 행동**을 다시 계산합니다.

$$
\tilde{a}_t = I(s_t, s_{t+1})
$$

그리고 이 행동으로 시뮬레이션(또는 학습된 동역학)했을 때의 상태가 생성된 상태와 **크게 다르지 않은** 궤적만 남깁니다. 동시에 보상이 기준을 넘는 궤적만 통과시킵니다.

```python
def select_expert_trajectories(trajs, inverse_dyn, step_fn, reward_fn, max_dev=0.1, min_reward=None):
    """trajs: 생성된 상태 시퀀스 목록"""
    selected = []
    for states in trajs:
        actions = [inverse_dyn(states[t], states[t + 1]) for t in range(len(states) - 1)]
        sim, ok = states[0], True
        for t, a in enumerate(actions):
            sim = step_fn(sim, a)                                  # 실제(또는 모델) 동역학
            if ((sim - states[t + 1]) ** 2).sum() > max_dev:       # 생성 상태와 너무 다르면 탈락
                ok = False
                break
        if ok and (min_reward is None or reward_fn(states, actions) >= min_reward):
            selected.append((states, actions))
    return selected

import numpy as np

step_fn = lambda s, a: s + np.clip(a, -0.1, 0.1)          # 한 step에 최대 0.1만 움직일 수 있는 점
inverse_dyn = lambda s, s_next: s_next - s
reward_fn = lambda states, actions: -np.linalg.norm(states[-1] - np.array([1.0, 1.0]))  # 목표 (1, 1)

good = [np.array([0.1 * t, 0.1 * t]) for t in range(11)]                # 0.1씩 이동: 가능한 궤적
bad = [np.array([0.0, 0.0]), np.array([0.5, 0.5]), np.array([1.0, 1.0])]  # 한 번에 0.5씩: 불가능
selected = select_expert_trajectories([good, bad], inverse_dyn, step_fn, reward_fn,
                                      max_dev=0.01, min_reward=-0.1)
print(len(selected))     # 1 → 두 궤적 모두 목표에 도달하지만, 물리적으로 가능한 good만 남음
```

`bad` 궤적도 목표 (1, 1)에 도달하므로 보상만 보면 좋은 궤적입니다. 하지만 이 점은 한 step에 0.1까지만 움직일 수 있어서, inverse dynamics로 구한 행동을 실제로 실행하면 생성된 상태를 따라가지 못하고 탈락합니다.

## 5. 보상 유형별 합성 데이터 생성

### Continuous Reward

MuJoCo처럼 **연속적인 보상 함수**가 있는 과제는, 보상을 최적성 확률로 바꿔 위의 guided sampling을 그대로 씁니다.

### Sparse Reward

목표 상태 $$s_g$$에 도달할 때만 보상이 1인 **계단 함수** 형태의 보상은, 미분하면 **Dirac delta**라서 기울기를 쓸 수 없습니다. 그래서 생성된 궤적의 **마지막 상태를 목표 상태로 고정**하는 inpainting으로 처리합니다.

### 복합 보상

현실적인 과제에는 여러 보상이 섞여 있습니다. 예를 들어 "미로를 탈출(주 과제)하면서 **금화를 줍기**(부 과제)". Sparse 보상으로는 부 과제 지점을 **몇 번째 시점에 지나야 하는지** 정하기 어렵습니다. 그래서 목표 지점과 금화 위치를 조건으로 넣어 궤적을 생성하고, 과제 보상을 만족한 궤적만 합성 전문가 데이터로 **데이터 기록에 보관**합니다.

## 6. 실험

| 환경 | 내용 |
|---|---|
| **Maze2D** | 미로에서 목표까지 이동 (기존 과제) |
| **Maze2D + Gold Coin** | 금화를 주워 가며 목표까지 (학습 때 보지 못한 과제) |
| **MuJoCo** | Hopper, Walker, HalfCheetah 등 locomotion (D4RL) |
| **KUKA Robot** | 블록 쌓기에서 **집어서 옮기기(pick-and-place)**로 과제 변경 |

- **기존 과제**: 자기 진화 과정을 거치며 Maze2D와 MuJoCo에서 Diffuser보다 성능이 올랐습니다. 합성 데이터가 원래 데이터의 빈 곳을 메워 준 효과입니다.
- **처음 보는 과제**: 금화 줍기 미로, KUKA pick-and-place처럼 학습 데이터에 없던 과제에서 Diffuser보다 크게 좋았습니다. Diffuser는 guidance만으로는 금화 쪽으로 돌아가는 경로를 잘 만들지 못했습니다.
- 논문은 부록 G에서 **ChatGPT로 더 다양한 미로 레이아웃을 만드는 방법**도 제시하고, 향후 과제로 프롬프트만으로 미로 설정을 생성해 합성 데이터 생성을 돕는 방향을 언급합니다.

<details markdown="1">
<summary>더 알아보기: 자기 데이터로 다시 학습할 때의 위험</summary>

모델이 만든 데이터로 그 모델을 다시 학습하면, 모델이 원래 가진 편향이 데이터를 통해 **스스로 강화**될 수 있습니다. 생성 모델 분야에서는 이렇게 반복할수록 다양성이 줄어드는 현상을 model collapse라고 부릅니다. AdaptDiffuser에서는 discriminator가 이 위험을 줄이는 핵심 장치입니다. 보상 기준과 동역학 일관성이라는 **모델 바깥의 기준**으로 데이터를 걸러서, 틀린 궤적이 학습 데이터로 다시 들어가는 것을 막습니다. 그 대신 생성 → 선별 → fine-tuning을 여러 번 반복해야 하므로 학습 비용이 Diffuser보다 큽니다.
</details>

## 7. 정리

| | Diffuser | AdaptDiffuser |
|---|---|---|
| 새 과제 대응 | Guide 함수만 교체 | Guide로 생성 → **선별 → 재학습** 반복 |
| 품질 한계 | 학습 데이터의 $$\mu_\theta$$에 묶임 | 합성 데이터로 $$\mu_\theta$$ 자체를 이동 |
| 일관성 보장 | - | Inverse dynamics 기반 discriminator |
| Sparse reward | Inpainting | Inpainting + 복합 보상 처리 |

## References

- Liang et al., "AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners," ICML 2023. [arXiv:2302.01877](https://arxiv.org/abs/2302.01877)
- Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis," ICML 2022.
- Ajay et al., "Is Conditional Generative Modeling all you need for Decision-Making? (Decision Diffuser)," ICLR 2023.
- Fu et al., "D4RL: Datasets for Deep Data-Driven Reinforcement Learning," 2020.
