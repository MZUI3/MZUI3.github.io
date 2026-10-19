---
title: "[RL] C51: 기댓값 대신 보상 분포를 학습하는 Distributional RL"
date: 2026-08-12 20:52:00 +0900
categories: [Reinforcement Learning, Model-Free]
tags: [reinforcement-learning, model-free, dqn, distributional-rl, c51, categorical-dqn, bellman-equation]
description: Q 값(기댓값) 대신 return의 분포를 학습하는 C51의 분포 Bellman 방정식, 51개 atom으로 분포를 표현하는 방법, projection 과정을 실행 가능한 예제로 정리합니다.
math: true
---

> DQN 계열의 Q 값은 **미래 보상의 평균(기댓값)** 하나입니다. **A Distributional Perspective on Reinforcement Learning**(Bellemare, Dabney & Munos, ICML 2017)은 평균만이 아니라 **보상의 분포 전체**를 학습하자고 제안합니다. 이 논문의 알고리즘이 **C51(Categorical DQN)**이고, 이후 QR-DQN, IQN 등 Distributional RL 연구의 출발점이 됐습니다.

## 1. 기존 DQN: 기댓값 하나만 본다

DQN은 Bellman 방정식을 따라 **기대 return** $$Q(x, a)$$를 학습하고, 이 값이 가장 큰 행동을 고릅니다.

$$
Q(x, a) = \mathbb{E}\big[R(x, a)\big] + \gamma\,\mathbb{E}\big[Q(X', A')\big]
$$

($$x$$: 상태, $$X'$$: 다음 상태)

하지만 같은 평균이라도 분포는 전혀 다를 수 있습니다.

### 예시: 출근 경로 고르기

평균 소요 시간이 같은 30분인 두 경로가 있다고 합시다.

- **경로 A**: 거의 항상 29~31분
- **경로 B**: 보통 20분인데, 가끔 사고가 나면 60분

기댓값만 보면 두 경로는 구분되지 않습니다(B는 25% 확률로 사고가 나면 평균 30분). 하지만 9시까지 반드시 도착해야 하는 날이라면 A가 낫고, 운에 맡겨도 되는 날이라면 B가 나을 수 있습니다. **분포를 알면 이런 차이를 반영할 수 있습니다.** 실제 환경의 return도 이처럼 **여러 봉우리(multi-modal)**를 갖는 경우가 많습니다.

> 이 예시는 "분포가 담는 정보"를 보여 주기 위한 것입니다. 아래에서 보듯 **C51도 행동은 평균으로 고르기 때문에**, A와 B 중 위험이 적은 쪽을 스스로 고르지는 않습니다. 분포를 이용해 위험을 피하는 행동(risk-sensitive control)은 이후 IQN 같은 연구에서 다룹니다.
{: .prompt-info }

## 2. Distributional Bellman Equation

Return을 기댓값이 아니라 **확률변수** $$Z(x, a)$$로 둡니다. $$Q$$는 $$Z$$의 평균입니다.

$$
Q(x, a) = \mathbb{E}\big[Z(x, a)\big]
$$

그러면 Bellman 방정식도 **분포 사이의 등식**이 됩니다.

$$
Z(x, a) \overset{D}{=} R(x, a) + \gamma\,Z(X', A')
$$

$$\overset{D}{=}$$는 "양변의 **확률 분포가 같다**"는 뜻입니다. 지금 받은 보상에, 다음 상태의 return 분포를 $$\gamma$$배로 줄여서 옮긴 것이 현재 return 분포라는 의미입니다.

논문은 이 분포 Bellman 연산자가 정책 평가(policy evaluation)에서는 Wasserstein 거리 기준으로 **수축(contraction)**한다는 것을 보였습니다. 반면 최적 제어(control)에서는 일반적으로 수축이 보장되지 않아, 실제 알고리즘은 근사적인 방법을 씁니다.

## 3. Categorical DQN (C51)

### 분포를 어떻게 표현하나: Support와 Atom

연속적인 분포를 그대로 다루기는 어려우므로, return의 범위를 **고정된 $$N$$개의 점(atom)**으로 나누고 각 점에 확률을 할당하는 **이산(categorical) 분포**로 근사합니다.

$$
z_i = V_{\min} + i\,\Delta z,\qquad \Delta z = \frac{V_{\max} - V_{\min}}{N - 1},\qquad i = 0, \dots, N-1
$$

- **Support**: atom $$\{z_i\}$$의 집합 (분포의 가로축, return 값)
- $$V_{\min}, V_{\max}$$: support의 최솟값·최댓값 (Atari에서 -10, 10)
- $$N$$: atom 개수

네트워크는 각 행동마다 **$$N$$개 atom의 확률**을 softmax로 출력합니다.

$$
p_i(x, a) = \frac{e^{\theta_i(x, a)}}{\sum_j e^{\theta_j(x, a)}}
$$

논문은 atom 수를 바꿔 가며 실험했고, **51개**일 때 가장 효율적이었습니다. 그래서 이 알고리즘을 **C51**이라고 부릅니다.

### 행동 선택

분포에서 기댓값을 계산해 Q 값을 얻고, 기존 DQN처럼 Q가 가장 큰 행동을 고릅니다.

$$
Q(x, a) = \sum_{i} z_i\,p_i(x, a),\qquad a^* = \arg\max_a Q(x, a)
$$

행동은 여전히 평균으로 고르지만, **학습을 분포로** 한다는 점이 다릅니다.

### 학습: Projected Bellman Update

목표 분포는 다음 상태의 분포를 $$r + \gamma z_j$$로 옮긴 것입니다. 그런데 옮긴 위치는 **원래 atom 위치와 맞지 않습니다.** 예를 들어 $$r = 0.3$$, $$\gamma = 0.99$$면 $$z_j$$가 모두 조금씩 밀려납니다. 예측 분포와 support가 달라 바로 비교할 수 없으므로, 옮긴 확률을 **가장 가까운 두 atom에 거리 비율로 나눠 줍니다(projection).**

1. 다음 상태에서 최적 행동 $$a^* = \arg\max_a \sum_i z_i\,p_i(x', a)$$
2. 각 atom $$j$$에 대해
   - $$\hat{T}z_j = \text{clip}(r + \gamma z_j,\ V_{\min},\ V_{\max})$$
   - $$b_j = (\hat{T}z_j - V_{\min}) / \Delta z$$, $$l = \lfloor b_j\rfloor$$, $$u = \lceil b_j\rceil$$
   - $$m_l \mathrel{+}= p_j(x', a^*)\,(u - b_j)$$, $$m_u \mathrel{+}= p_j(x', a^*)\,(b_j - l)$$
3. 투영된 목표 분포 $$m$$과 예측 분포 $$p(x, a)$$의 **Cross Entropy**를 최소화

숫자로 보면 이해가 쉽습니다. Support가 $$[-10, 10]$$, atom 51개면 $$\Delta z = 0.4$$입니다. 다음 상태의 분포가 $$z = 0$$에 확률 1로 몰려 있고 보상 $$r = 0.3$$을 받았다면, 목표 위치 $$0.3$$은 atom $$0$$과 $$0.4$$ 사이에 떨어집니다. 거리에 반비례하게 나누면 $$0$$에 0.25, $$0.4$$에 0.75가 갑니다. 이렇게 나누면 **확률의 합(1)과 기댓값(0.3)이 그대로 보존**됩니다.

```text
[(0.0, 0.25), (0.4, 0.75)]
합: 1.0  기댓값: 0.29999...
```

(아래 코드의 projection 부분만 떼어 실행한 결과입니다.)

$$
\mathcal{L} = -\sum_i m_i\,\log p_i(x, a)
$$

DQN이 **스칼라 Q 값을 MSE**로 맞췄다면, C51은 **분포를 Cross Entropy(KL divergence)**로 맞춥니다.

```python
import torch

N, V_MIN, V_MAX = 51, -10.0, 10.0
z = torch.linspace(V_MIN, V_MAX, N)                 # support
dz = (V_MAX - V_MIN) / (N - 1)

def c51_loss(online, target, s, a, r, s_next, done, gamma=0.99):
    # online(s), target(s): (B, n_actions, N) 확률 분포 (softmax 출력)
    with torch.no_grad():
        p_next = target(s_next)                                   # (B, A, N)
        a_star = (p_next * z).sum(-1).argmax(1)                    # 기댓값으로 행동 선택
        p_next = p_next[torch.arange(len(a_star)), a_star]         # (B, N)

        Tz = (r[:, None] + gamma * (1 - done[:, None]) * z[None]).clamp(V_MIN, V_MAX)
        b = (Tz - V_MIN) / dz
        l, u = b.floor().long(), b.ceil().long()
        l[(u > 0) & (l == u)] -= 1                                 # b가 정수일 때 질량이 사라지지 않게
        u[(l < N - 1) & (l == u)] += 1

        m = torch.zeros_like(p_next)
        m.scatter_add_(1, l, p_next * (u.float() - b))             # 아래 atom에 분배
        m.scatter_add_(1, u, p_next * (b - l.float()))             # 위 atom에 분배

    log_p = torch.log(online(s)[torch.arange(len(a)), a] + 1e-8)  # (B, N)
    return -(m * log_p).sum(1).mean()                             # cross entropy
```

`l == u`를 보정하는 두 줄은, $$b_j$$가 정확히 정수(atom 위)일 때 위 식의 두 가중치 $$(u - b_j)$$, $$(b_j - l)$$이 모두 0이 되어 확률이 사라지는 것을 막기 위한 것입니다. 또 실무에서는 네트워크가 softmax 확률 대신 logit을 내게 하고 `torch.log_softmax`로 로그 확률을 바로 구하는 편이 수치적으로 안정적입니다.

## 4. 결과

Atari 57개 게임에서 DQN, Double DQN, Dueling, Prioritized replay 등 다른 value-based 방법들과 비교했습니다.

- C51은 사람 대비 정규화 점수의 **평균과 중앙값 모두 당시 최고**를 기록했습니다. DQN 대비 큰 폭의 향상입니다.
- 학습된 분포를 시각화하면 실제로 **여러 봉우리를 가진 분포**가 나타났습니다. 예를 들어 위험한 상황에서는 "살아남는 경우"와 "죽는 경우" 두 갈래의 return이 분리되어 보입니다.
- Atom 수가 적으면(5개 등) 성능이 낮았고, 51개 정도에서 충분히 좋았습니다.

### 왜 분포를 학습하면 좋아질까

행동 선택에는 결국 평균만 쓰는데도 성능이 오르는 이유로 다음이 제시됩니다.

- **더 풍부한 학습 신호**: 스칼라 하나 대신 분포 전체를 맞추므로, 같은 경험에서 더 많은 정보를 얻음 (auxiliary task 효과)
- **학습 안정성**: 평균만 맞출 때 생기는 불안정한 target 변화가 완화됨
- **다봉 분포 보존**: 평균으로 뭉개지는 정보를 유지해 표현 학습에 도움

<details markdown="1">
<summary>더 알아보기: 분포 학습의 이득은 어디서 오나</summary>

Lyle et al.(2019)은 테이블(tabular) 설정이나 선형 함수 근사에서는 분포를 학습해도 기댓값 학습과 **같은 결과**가 나온다는 것을 보였습니다. 즉 이득은 "분포 Bellman 방정식 자체"보다, **딥러닝 함수 근사와 결합했을 때** 표현 학습이 좋아지는 데서 나온다는 해석입니다. 위의 "auxiliary task 효과"가 이와 같은 맥락입니다.
</details>

## 5. 정리

| | DQN | C51 |
|---|---|---|
| 예측 대상 | 기대 return $$Q(x, a)$$ (스칼라) | return 분포 $$Z(x, a)$$ (51개 atom 확률) |
| Bellman 방정식 | 기댓값 등식 | **분포 등식** |
| Loss | MSE | **Cross Entropy** (projection 후) |
| 행동 선택 | $$\arg\max Q$$ | $$\arg\max \sum z_i p_i$$ |

C51은 [Double DQN]({% post_url 2026-08-04-double-dqn %}), [NoisyNet]({% post_url 2026-08-10-noisy-network %}) 등과 함께 **Rainbow**에 통합됐습니다. 이후 atom 위치를 고정하지 않고 분위수(quantile)를 학습하는 **QR-DQN**, 임의의 분위수를 샘플링하는 **IQN**으로 발전했습니다.

## References

- Bellemare, Dabney & Munos, "A Distributional Perspective on Reinforcement Learning," ICML 2017. [arXiv:1707.06887](https://arxiv.org/abs/1707.06887)
- Lyle, Castro & Bellemare, "A Comparative Analysis of Expected and Distributional Reinforcement Learning," AAAI 2019. [arXiv:1901.11084](https://arxiv.org/abs/1901.11084)
- Dabney et al., "Distributional Reinforcement Learning with Quantile Regression (QR-DQN)," AAAI 2018.
- Dabney et al., "Implicit Quantile Networks for Distributional Reinforcement Learning (IQN)," ICML 2018.
- Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning," AAAI 2018.
- Sudharsan Ravichandiran, 『Deep Reinforcement Learning with Python』 2nd ed.
