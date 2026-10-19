---
title: "[RL] RND: 무작위 네트워크를 따라 하며 새로움 측정하기"
date: 2026-09-02 20:00:00 +0900
categories: [Reinforcement Learning, Exploration]
tags: [reinforcement-learning, exploration, intrinsic-motivation, rnd, random-network-distillation, ppo, montezuma, noisy-tv]
description: 무작위로 고정한 target 네트워크의 출력을 따라 하는 predictor의 오차를 새로움으로 쓰는 RND의 아이디어와, intrinsic·extrinsic return을 나눠 다루는 방법, 정규화 디테일을 코드와 함께 정리합니다.
math: true
---

> [ICM]({% post_url 2026-08-29-icm-curiosity %})은 다음 상태를 예측하는 오차를 호기심으로 썼지만, 결과 자체가 무작위인 상황에서는 오차가 영원히 줄지 않는 문제가 있었습니다. **RND(Random Network Distillation)**(Burda et al., ICLR 2019)는 예측 대상을 **무작위로 초기화한 고정 네트워크의 출력**으로 바꿔 이 문제를 깔끔하게 피합니다. 그리고 Montezuma's Revenge에서 처음으로 **사람 평균 점수를 넘었습니다.**

## 1. 기존 방법의 문제

### 확률적인 transition에 끌린다

예측 오차를 최대화하는 에이전트는, 예측해야 할 정답이 **입력의 확률적 함수**인 transition에 쉽게 끌립니다. 대표적인 것이 Noisy TV입니다. 다음 관측을 예측할 때 TV 화면은 무엇을 해도 맞힐 수 없으니, 에이전트는 TV 앞을 떠나지 않습니다.

### 확장성

Information gain이나 count, pseudo-count 기반 방법은 **수많은 병렬 환경**으로 확장하기 어렵습니다(밀도 모델이나 동역학 모델을 함께 학습해야 하므로 비효율적).

### RND의 목표

- 구현이 **간단**하고
- **고차원 관측**에서 잘 동작하며
- **어떤 정책 최적화 알고리즘**과도 쓸 수 있고
- **forward pass 한 번**으로 계산되는 효율적인 exploration bonus

## 2. 아이디어: 정답이 결정적인 예측 문제

신경망은 **학습한 데이터와 비슷한 입력에서는 예측 오차가 작고**, 처음 보는 입력에서는 오차가 큽니다. 이 성질 자체를 새로움의 척도로 씁니다. 다만 예측할 정답이 **입력의 결정적 함수**여야 확률성 문제가 생기지 않습니다.

두 개의 네트워크를 둡니다.

- **Target network** $$f$$: 무작위로 초기화한 뒤 **영원히 고정**. 관측 $$o$$를 특징 벡터로 바꿈
- **Predictor network** $$\hat{f}$$: 에이전트가 모은 데이터로 학습하며 target의 출력을 따라 함

$$
\mathcal{L}(\theta) = \big\lVert\hat{f}(o; \theta) - f(o)\big\rVert_2^2,\qquad r^i_t = \big\lVert\hat{f}(o_{t+1}; \theta) - f(o_{t+1})\big\rVert_2^2
$$

자주 본 관측은 predictor가 잘 따라 해서 보상이 작고, 처음 보는 관측은 따라 하지 못해 보상이 큽니다. Target이 결정적 함수이므로 **Noisy TV 같은 확률성이 원천적으로 없습니다.**

```python
import torch
import torch.nn as nn

def make_net(out_dim=512):
    return nn.Sequential(nn.Conv2d(1, 32, 8, 4), nn.LeakyReLU(), nn.Conv2d(32, 64, 4, 2), nn.LeakyReLU(),
                         nn.Conv2d(64, 64, 3, 1), nn.LeakyReLU(), nn.Flatten(), nn.LazyLinear(out_dim))

class RND(nn.Module):
    def __init__(self):
        super().__init__()
        self.target, self.predictor = make_net(), make_net()
        for p in self.target.parameters():
            p.requires_grad = False                  # target은 고정

    def forward(self, obs):                          # obs: 정규화된 다음 관측 (B, 1, 84, 84)
        err = (self.predictor(obs) - self.target(obs).detach()).pow(2).sum(1)
        return err.detach(), err.mean()              # intrinsic reward, predictor loss

torch.manual_seed(0)
rnd = RND()
opt = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-4)
seen = torch.randn(64, 1, 84, 84)                 # 에이전트가 반복해서 본 관측
new = torch.randn(64, 1, 84, 84)                  # 처음 보는 관측
for _ in range(200):
    _, loss = rnd(seen)
    opt.zero_grad(); loss.backward(); opt.step()
print(rnd(seen)[0].mean().item())                 # 약 5e-07 → 익숙한 관측은 보상이 거의 0
print(rnd(new)[0].mean().item())                  # 약 0.54  → 처음 보는 관측은 보상이 큼
```

예측 대상이 고정된 함수라서, predictor가 본 관측에서는 오차가 거의 0까지 내려가고 처음 보는 관측에서는 크게 남습니다. 무작위 노이즈 관측이라도 **같은 관측에 대한 정답은 항상 같으므로**, Noisy TV처럼 "아무리 학습해도 줄지 않는 오차"가 생기지 않습니다. (논문 구현은 predictor를 target보다 층을 더 깊게 만들어, 따라 하기 어렵지 않게 합니다.)

### Predictor가 target을 완벽히 외우면?

Predictor가 학습하다가 target을 **모든 입력에서 완벽히 모방**해 버리면 보상이 의미 없어집니다. MNIST로 확인한 결과, 숫자 0과 특정 숫자 하나로 학습했을 때 **학습하지 않은 숫자에 대해서는 계속 오차가 컸습니다.** 지나친 일반화(overgeneralize)는 일어나지 않았습니다.

## 3. 예측 오차의 원인 분석

예측 오차는 일반적으로 네 가지 요인에서 생깁니다.

1. **학습 데이터의 양**: predictor가 비슷한 예시를 적게 봤을수록 오차가 큼 → **탐험에 필요한 바로 그 신호**
2. **확률성(stochasticity)**: 정답 자체가 무작위 (forward dynamics의 Noisy TV)
3. **모델 명세 오류(model misspecification)**: 모델이 정답 함수를 표현하기에 너무 제한적
4. **학습 동역학(learning dynamics)**: 최적화가 정답 근사를 찾지 못함

1번이 우리가 원하는 신호이고, 2·3번은 Noisy TV 같은 문제를 일으킵니다. RND는 target을 결정적 함수로 두어 **2번을 없애고**, predictor를 target을 충분히 표현할 수 있는 구조(같은 구조이거나 그보다 큰 구조)로 만들어 3번을 피합니다(target을 표현할 수 있다면 이론적으로 완벽히 따라 할 수 있음).

### 불확실성 추정과의 관계

Bayesian 관점에서 보면, 사전분포(prior)에서 뽑은 무작위 함수를 데이터에 맞춰 보정하는 것은 **posterior의 불확실성을 추정**하는 방법(randomized prior functions)과 연결됩니다. Bayesian 신경망이 파라미터를 하나의 값이 아니라 분포로 다뤄 출력의 불확실성을 주듯, RND의 예측 오차는 "**이 관측에 대해 얼마나 모르는가**"를 근사합니다.

## 4. Intrinsic Return과 Extrinsic Return 합치기

### Episodic vs Non-episodic

새로운 상태는 **에피소드를 넘어서도** 생길 수 있습니다. 실험해 보니 intrinsic return은 **non-episodic**(에피소드가 끝나도 이어서 계산)으로 다루는 편이 탐험이 더 잘됐습니다. 사람이 게임을 하다 죽어도 "다음엔 저 방에 가 봐야지"라는 호기심은 이어지는 것과 같습니다.

반면 extrinsic return까지 non-episodic으로 하면, 에이전트가 시작 지점의 쉬운 보상을 다시 얻으려고 **일부러 게임을 끝내는** 행동을 반복할 수 있습니다.

### 두 개의 Value Head

그래서 **intrinsic은 non-episodic, extrinsic은 episodic**으로 각각 return을 계산하고, PPO의 가치 함수에 **두 개의 head** $$V_E, V_I$$를 둡니다. 두 return은 서로 다른 discount도 씁니다($$\gamma_E = 0.999$$, $$\gamma_I = 0.99$$). Advantage는 두 head의 advantage를 선형 결합합니다.

$$
A = A_E + c\,A_I
$$

(논문 실험에서는 extrinsic 쪽에 2, intrinsic 쪽에 1을 곱해 $$A = 2A_E + A_I$$로 결합했습니다.)

### 정규화

- **Reward normalization**: 환경과 시간에 따라 intrinsic reward의 크기가 크게 달라지므로, intrinsic return의 **표준편차로 나눠** 일관된 크기를 유지
- **Observation normalization**: 고정된 무작위 target 네트워크는 입력 스케일에 민감해서, 정규화하지 않으면 임베딩의 분산이 극도로 작아집니다. 관측의 각 차원을 running mean/std로 정규화한 뒤 **[-5, 5]로 clipping**하고, predictor와 target에 같은 정규화를 씁니다.

```python
# 관측 정규화 (target, predictor 공통)
obs_n = ((obs - obs_rms.mean) / (obs_rms.var.sqrt() + 1e-8)).clamp(-5, 5)
r_int, rnd_loss = rnd(obs_n)
r_int = r_int / (int_return_rms.var.sqrt() + 1e-8)        # intrinsic reward 정규화
adv = 2.0 * adv_ext + 1.0 * adv_int                         # 두 value head의 advantage 결합 (논문 계수)
```

## 5. 실험

비교 baseline은 두 가지입니다.

- 탐험 보너스 없는 **PPO**
- **Forward dynamics 오차**를 보너스로 쓰는 PPO (ICM 계열)

### RNN vs CNN 정책

Montezuma's Revenge처럼 **부분 관측** 환경에서는 과거 기억이 중요하니 RNN 정책이 더 좋을 것으로 예상했지만, 실제로는 **CNN 정책이 비슷하거나 더 좋았습니다.**

### 결과

- 6개의 어려운 탐험 게임에서 RND가 대부분 가장 좋았고, **두 게임에서 당시 SOTA**를 달성했습니다.
- **Montezuma's Revenge**에서 사람의 시연(demonstration)이나 게임 내부 상태 정보 없이 **처음으로 사람 평균 점수를 넘었고**, 24개 방 중 대부분을 탐험했습니다.
- 반면 Pitfall!처럼 보상이 극도로 드물고 함정이 많은 게임에서는 여전히 어려웠습니다.

## 6. 정리

| | ICM (Forward dynamics) | RND |
|---|---|---|
| 예측 대상 | 다음 상태의 특징 | 고정 무작위 네트워크의 출력 |
| 정답의 성질 | 환경에 따라 **확률적**일 수 있음 | 항상 **결정적** |
| Noisy TV | 영향 있음 | 원천적으로 없음 |
| 계산 | forward + inverse model 학습 | predictor 하나 학습, forward pass 1회 |

RND는 단순함 덕분에 이후 탐험 연구의 표준 baseline이 됐고, [NGU]({% post_url 2026-09-12-never-give-up %})와 [Agent57]({% post_url 2026-09-17-agent57 %})에서도 **life-long novelty** 모듈로 쓰입니다.

## References

- Burda et al., "Exploration by Random Network Distillation," ICLR 2019. [arXiv:1810.12894](https://arxiv.org/abs/1810.12894)
- Osband et al., "Randomized Prior Functions for Deep Reinforcement Learning," NeurIPS 2018.
- Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction," ICML 2017.
- OpenDILab, DI-engine RND documentation
