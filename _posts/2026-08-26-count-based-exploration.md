---
title: "[RL] Count-Based Exploration과 Pseudo-Count: 처음 보는 상태에 보너스 주기"
date: 2026-08-26 20:09:00 +0900
categories: [Reinforcement Learning, Exploration]
tags: [reinforcement-learning, exploration, intrinsic-motivation, count-based, pseudo-count, density-model, montezuma]
description: 보상이 드문 환경에서 처음 보는 상태에 보너스를 주는 count-based exploration과, 거대한 상태 공간에서 밀도 모델로 방문 횟수를 추정하는 pseudo-count의 유도와 성질을 코드와 함께 정리합니다.
math: true
---

> Exploration 카테고리의 첫 글입니다. 보상이 거의 없는(sparse reward) 환경에서 에이전트가 **스스로 탐험할 이유**를 만들어 주는 **intrinsic motivation** 연구의 출발점, *Unifying Count-Based Exploration and Intrinsic Motivation*(Bellemare et al., NeurIPS 2016)을 정리합니다.

## 1. 왜 탐험이 어려운가

가장 기본적인 탐험 방법은 **ε-greedy**입니다. 대부분은 가치가 높은 행동을 고르고, 일정 확률로 무작위 행동을 합니다. 하지만 **Montezuma's Revenge** 같은 게임에서는 거의 쓸모가 없습니다.

- 열쇠를 얻으려면 사다리를 내려가고, 밧줄을 타고, 해골을 피해 점프하는 등 **수십~수백 개의 정확한 행동**을 연속으로 해야 합니다.
- 그 전까지는 **보상이 0**입니다.
- 무작위 행동으로 이 순서를 우연히 맞힐 확률은 사실상 0입니다.

그래서 "**아직 가 보지 않은 상태**를 방문하면 스스로 보상을 주자"는 아이디어, **intrinsic motivation**이 등장했습니다. 환경이 주는 보상을 **extrinsic reward**, 에이전트가 스스로 만드는 보상을 **intrinsic reward(exploration bonus)**라고 부릅니다.

## 2. Count-Based Exploration

가장 직관적인 방법은 **방문 횟수를 세는 것**입니다. MBIE-EB(Strehl & Littman, 2008)는 상태–행동 방문 횟수 $$N(s, a)$$로 보너스를 만들어 Bellman 방정식에 더했습니다.

$$
r^+(s, a) = \frac{\beta}{\sqrt{N(s, a)}}
$$

적게 방문한 곳일수록 보너스가 크고, 여러 번 방문하면 보너스가 줄어듭니다.

### 문제: 같은 상태는 다시 오지 않는다

Atari 화면처럼 **상태 공간이 거대**하면 정확히 같은 화면을 두 번 볼 일이 거의 없습니다. 캐릭터가 같은 자리에 있어도 적이나 시계가 조금만 움직이면 다른 상태가 됩니다. 그러면 모든 상태의 count가 1이라 보너스가 아무 정보도 주지 못하고, ε-greedy보다도 실용적이지 않습니다.

논문은 이를 **Pseudo-count**로 해결합니다. "정확히 같은 상태를 몇 번 봤는가" 대신 "**비슷한 상태를 얼마나 봤는가**"를 일반화해서 추정합니다.

## 3. 밀도 모델에서 Count로

### Density Model

상태 시퀀스 $$x_{1:n}$$을 본 뒤, 다음에 상태 $$x$$가 나올 확률을 주는 **밀도 모델** $$\rho_n(x) = \rho(x \mid x_{1:n})$$을 학습합니다. 논문은 **CTS(Context Tree Switching)**를 썼습니다.

- 원본 화면을 축소·흑백화(42×42, 8단계 밝기)로 전처리
- 각 픽셀의 확률을 **위·왼쪽 이웃 픽셀**을 문맥으로 예측하는 위치별 모델로 계산
- 이미지 전체 확률은 픽셀 확률의 곱

### Empirical Count와의 관계

방문 횟수 $$N_n(x)$$를 실제로 셀 수 있다면, 경험적 분포는

$$
\mu_n(x) = \frac{N_n(x)}{n}
$$

입니다. 여기서 $$x$$를 **한 번 더** 본 뒤의 분포는

$$
\mu'_n(x) = \frac{N_n(x) + 1}{n + 1}
$$

이 됩니다.

### Recoding Probability와 Pseudo-Count

밀도 모델에도 같은 일을 합니다. 지금 확률 $$\rho_n(x)$$와, $$x$$를 한 번 더 학습한 직후의 확률(**recoding probability**) $$\rho'_n(x)$$를 구합니다. 그리고 두 식을 실제 count처럼 맞춰 주는 가상의 count $$\hat{N}_n(x)$$와 총 개수 $$\hat{n}$$을 풉니다.

$$
\rho_n(x) = \frac{\hat{N}_n(x)}{\hat{n}},\qquad \rho'_n(x) = \frac{\hat{N}_n(x) + 1}{\hat{n} + 1}
$$

연립하면 **pseudo-count**가 나옵니다.

$$
\hat{N}_n(x) = \frac{\rho_n(x)\big(1 - \rho'_n(x)\big)}{\rho'_n(x) - \rho_n(x)}
$$

직관적으로, **한 번 더 봤을 때 확률이 크게 오르는 상태**(분모가 큼)는 아직 익숙하지 않은 상태이므로 pseudo-count가 작습니다.

```python
def pseudo_count(rho, rho_prime, eps=1e-8):
    """rho: 지금 x의 확률, rho_prime: x를 한 번 더 학습한 뒤의 확률"""
    gain = max(rho_prime - rho, eps)        # 한 번 더 봤을 때 확률이 오른 정도
    return rho * (1 - rho_prime) / gain

def exploration_bonus(n_hat, beta=0.05):
    return beta / (n_hat + 0.01) ** 0.5

# 검증: 밀도 모델이 정확한 경험적 분포(N/n)라면 pseudo-count는 실제 count와 같아야 한다
n = 100
for N in [0, 1, 10, 50]:
    rho, rho_prime = N / n, (N + 1) / (n + 1)
    n_hat = pseudo_count(rho, rho_prime)
    print(N, round(n_hat, 4), round(exploration_bonus(n_hat), 4))
# 0 0.0 0.5 / 1 1.0 0.0498 / 10 10.0 0.0158 / 50 50.0 0.0071
```

밀도 모델이 단순히 "센 횟수 / 전체"라면 pseudo-count는 실제 방문 횟수를 그대로 돌려줍니다. 식이 count의 **일반화**라는 것을 확인할 수 있습니다. 한 번도 안 본 상태의 보너스(0.5)는 50번 본 상태(0.007)보다 약 70배 큽니다.

> 논문에서 말하는 **prediction gain**은 확률의 차이가 아니라 **로그 확률의 차이** $$\text{PG}_n(x) = \log \rho'_n(x) - \log \rho_n(x)$$입니다. 논문은 pseudo-count가 $$\hat{N}_n(x) \le \big(e^{\text{PG}_n(x)} - 1\big)^{-1}$$로 prediction gain과 연결된다는 것을 보이고, 이를 통해 count 기반 탐험과 정보 이득 기반 intrinsic motivation을 잇습니다.
{: .prompt-info }

### Pseudo-Count의 성질

논문은 pseudo-count가 다음과 같이 행동한다고 정리합니다.

- **처음 보는 사건**에 대해 거의 0
- 믿을 만한 크기를 가짐
- 상태 빈도의 **순서**를 지킴
- 평균적으로 실제 count처럼 **선형으로 증가**
- 비정상(non-stationary) 데이터에도 강건함

Atari **Freeway**(닭이 도로를 건너는 게임)에서 확인한 결과는 다음과 같습니다.

- **시작 위치**: 차가 계속 움직여 매 frame 화면은 다르지만, pseudo-count는 **선형으로 꾸준히 증가**했습니다. 사실상 같은 상태라는 것을 일반화해서 알아본 것입니다.
- **도로를 건너는 순간**(salient event): 처음엔 count가 낮다가 탐험할수록 올라가고, 에이전트를 멈춰 둔 구간에서는 거의 증가하지 않았습니다.

## 4. Exploration Bonus

Pseudo-count로 intrinsic reward를 만듭니다.

$$
R^+_n(x, a) = \beta\,\big(\hat{N}_n(x) + 0.01\big)^{-1/2}
$$

$$\beta$$는 보너스의 비중을 정하는 hyper-parameter로, 실험적으로 **0.05**를 썼습니다. 에이전트는 extrinsic reward와 이 보너스의 합을 최대화합니다.

논문은 또 정보 이론 관점에서 pseudo-count가 **information gain(정보 이득)**과 연결된다는 것을 보여, count-based exploration과 intrinsic motivation을 하나의 틀로 **통합(unifying)**했습니다. 이것이 제목의 의미입니다.

## 5. 실험

### DQN + Exploration Bonus

가장 어려운 Atari 게임 5개에서, 보너스 유무와 optimistic initialization(초기 Q를 높게 두는 고전적 탐험 기법)을 비교했습니다.

- **Montezuma's Revenge**: 보너스 없는 DQN은 첫 방 주변만 맴돌았지만, 보너스를 준 DQN은 **24개 방 중 15개**를 탐험했습니다. Deep RL로 이 게임을 의미 있게 학습한 최초의 사례 중 하나입니다.
- **Venture** 같은 sparse reward 게임에서도 큰 향상을 보였습니다.

### A3C+

A3C는 replay memory 없이 여러 병렬 에이전트의 샘플로 global network를 업데이트합니다. 저자들은 이 구조가 종종 **탐험 부족**으로 이어진다고 보고, pseudo-count 보너스를 결합한 **A3C+**를 정의했습니다. Atari 60개 게임에서 A3C+의 학습 점수 중앙값이 A3C보다 조금 더 높았고, 특히 탐험이 어려운 게임에서 차이가 났습니다.

## 6. 정리

| 개념 | 핵심 |
|---|---|
| Intrinsic motivation | 새로운 상태 방문에 스스로 보상 |
| Count-based bonus | $$\beta / \sqrt{N}$$, 적게 본 곳일수록 큼 |
| 문제 | 큰 상태 공간에서는 같은 상태가 반복되지 않음 |
| Pseudo-count | 밀도 모델의 확률 변화로 "비슷한 상태를 본 횟수" 추정 |
| 결과 | Montezuma's Revenge에서 15개 방 탐험 |

이후 연구들은 밀도 모델 대신 **예측 오차**로 새로움을 측정하는 방향으로 갔습니다. 다음 글에서는 그 대표인 [ICM(Curiosity)]({% post_url 2026-08-29-icm-curiosity %})을 다룹니다.

## References

- Bellemare et al., "Unifying Count-Based Exploration and Intrinsic Motivation," NeurIPS 2016. [arXiv:1606.01868](https://arxiv.org/abs/1606.01868)
- Strehl & Littman, "An analysis of model-based Interval Estimation for Markov Decision Processes (MBIE-EB)," JCSS 2008.
- Bellemare et al., "Skip Context Tree Switching," ICML 2014.
- Ostrovski et al., "Count-Based Exploration with Neural Density Models," ICML 2017.
