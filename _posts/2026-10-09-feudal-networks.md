---
title: "[RL] FeUdal Networks: Manager는 방향을, Worker는 행동을"
date: 2026-10-09 19:19:00 +0900
categories: [Reinforcement Learning, Hierarchical RL]
tags: [reinforcement-learning, hierarchical-rl, feudal-networks, fun, manager-worker, transition-policy-gradient, dilated-lstm]
description: 상위 Manager가 학습된 상태 공간에서 '방향'을 목표로 정하고 하위 Worker가 그 방향을 따라 행동하는 FeUdal Networks의 구조, transition policy gradient, dilated LSTM을 코드와 함께 정리합니다.
math: true
---

> [h-DQN]({% post_url 2026-10-08-h-dqn %})은 사람이 정한 목표(열쇠, 문 등) 중에서 고르는 계층적 구조였습니다. **FeUdal Networks(FuN)**(Vezhnevets et al., ICML 2017)는 목표를 사람이 정의하지 않고, **학습된 잠재 공간 안의 "방향"**으로 만듭니다. 상위 Manager는 느린 시간 단위로 방향을 정하고, 하위 Worker는 빠른 시간 단위로 그 방향을 따라 행동합니다.

## 1. 배경: Feudal Reinforcement Learning

보상이 드물고, 기억이 필요한(**non-Markovian**) 환경은 복잡도가 매우 높습니다. 이를 다루기 위해 **봉건제(feudal)**처럼 계층을 두는 **FRL**(Dayan & Hinton, 1993)이 제안됐습니다.

- 목표를 **위에서 아래로** 내려보냄
- **목표 설정과 목표 달성을 분리**: 상위는 목표를 정할 뿐, **어떻게 도달할지는 정하지 않음**
- 하위 Worker는 **높은 시간 해상도**(매 step)에서 동작

| | Manager | Worker |
|---|---|---|
| 하는 일 | 목표(방향) 설정 | 실제 행동 |
| 보상 | **Extrinsic** (환경 보상) | **Intrinsic** (목표를 따른 정도) |
| 시간 단위 | 느림 | 빠름 |

## 2. Contribution

- FRL에서 영감을 받은, **end-to-end로 미분 가능한** 일반적인 모델
- 목표의 **의미(semantic meaning)**를 이용해 Manager를 학습하는 **approximate transition policy gradient**
- 절대적인 목표 지점이 아니라 **방향성을 갖는 목표**. 예: "적을 피해라", "특정 위치에서 점프해라" 같은 상태 변화의 방향
- Manager가 긴 시간 의존성을 기억하고 먼 과거까지 기울기를 전달하도록 돕는 **Dilated LSTM**

## 3. 구조

Worker와 Manager 두 모듈로 구성되고, **지각 모듈(perceptual module)**을 공유합니다.

### 공유 지각 모듈

환경 관측 $$x_t$$에서 패턴을 뽑아 중간 표현 $$z_t$$를 만듭니다(CNN).

$$
z_t = f^{percept}(x_t)
$$

### Manager

Manager 자신의 상태 공간에서 상태 $$s_t$$를 정의하고, 순환 신경망으로 **목표 벡터** $$g_t$$를 출력해 정규화합니다.

$$
s_t = f^{Mspace}(z_t),\qquad h^M_t, \hat{g}_t = f^{Mrnn}(s_t, h^M_{t-1}),\qquad g_t = \frac{\hat{g}_t}{\lVert\hat{g}_t\rVert}
$$

- $$h^M_t$$: Manager의 내부 상태
- $$g_t$$: 크기가 1인 **방향 벡터** (상태 공간에서 어느 쪽으로 가라)

### Worker

최근 $$c$$ step 동안의 목표를 합쳐 **선형 변환** $$\phi$$로 작은 임베딩 $$w_t$$를 만듭니다.

$$
w_t = \phi\Big(\sum_{i=t-c}^{t}g_i\Big)
$$

Worker는 자신의 RNN으로 행동별 임베딩 행렬 $$U_t$$를 출력하고, 목표 임베딩과 곱해 정책을 만듭니다.

$$
h^W_t, U_t = f^{Wrnn}(z_t, h^W_{t-1}),\qquad \pi_t = \text{SoftMax}(U_t\,w_t)
$$

목표 $$g$$가 정책에 **곱셈으로** 들어가므로, 같은 Worker가 목표에 따라 다른 행동을 하게 됩니다. $$\phi$$는 bias가 없어서 목표가 0이면 정책에 영향을 주지 않습니다.

환경은 행동을 받아 다음 관측 $$x_{t+1}$$과 보상 $$r_t$$를 주고, 종료 상태까지 반복합니다.

## 4. 학습

### Manager: Transition Policy Gradient

Manager의 목표를 Worker의 행동을 통해 역전파로 학습시키면, 목표가 의미를 잃고 Worker의 내부 변수처럼 되어 버립니다. 그래서 Manager는 **따로** 학습합니다. 목표 $$g_t$$의 방향이 **$$c$$ step 뒤 실제로 일어난 상태 변화**와 같아지도록, 그 변화가 좋은 결과(advantage)를 냈다면 그 방향을 강화합니다.

$$
\nabla g_t = A^M_t\,\nabla_\theta\,d_{\cos}\big(s_{t+c} - s_t,\ g_t(\theta)\big),\qquad A^M_t = R_t - V^M_t(x_t, \theta)
$$

- $$d_{\cos}$$: 코사인 유사도
- $$R_t$$: 누적 보상, $$A^M_t$$: Manager의 advantage

Worker가 결국 어떤 상태 변화를 만들어 낼지를 하나의 "전이 정책"으로 보고 그 정책 기울기를 근사한 것이라, **transition policy gradient**라고 부릅니다.

### Worker: Intrinsic Reward

Worker는 **최근 목표 방향대로 상태를 움직였는지**로 intrinsic reward를 받습니다.

$$
r^I_t = \frac{1}{c}\sum_{i=1}^{c}d_{\cos}\big(s_t - s_{t-i},\ g_{t-i}\big)
$$

그리고 extrinsic과 intrinsic 보상을 섞어 일반적인 advantage actor-critic으로 학습합니다.

$$
\nabla\pi_t = A^D_t\,\nabla_\theta\log\pi(a_t \mid x_t;\theta),\qquad A^D_t = R_t + \alpha R^I_t - V^D_t(x_t;\theta)
$$

### Dilated LSTM

Manager는 긴 시간 단위로 움직여야 합니다. **Dilated LSTM**은 내부 상태를 $$r$$개(예: 10)의 그룹으로 나눠, 매 step 그중 **하나만** 업데이트합니다. 각 그룹은 $$r$$ step마다 한 번씩 갱신되므로 긴 시간 의존성을 기억하기 쉽고, 출력은 모든 그룹의 합이라 매 step 목표를 낼 수 있습니다.

```python
import torch
import torch.nn as nn

class DilatedLSTM(nn.Module):
    """내부 상태를 r개 묶음으로 나눠, 매 step 그중 하나만 갱신하는 LSTM"""
    def __init__(self, in_dim, hid, r=10):
        super().__init__()
        self.cell, self.r, self.t = nn.LSTMCell(in_dim, hid), r, 0
        self.states = None

    def forward(self, x):
        if self.states is None:
            zeros = torch.zeros(x.size(0), self.cell.hidden_size)
            self.states = [(zeros, zeros) for _ in range(self.r)]
        k = self.t % self.r                                   # 이번 step에 갱신할 묶음
        self.states[k] = self.cell(x, self.states[k])
        self.t += 1
        return torch.stack([h for h, _ in self.states]).sum(0)   # 출력은 모든 묶음의 합

torch.manual_seed(0)
dlstm = DilatedLSTM(in_dim=8, hid=16, r=10)
for t in range(25):
    out = dlstm(torch.randn(2, 8))
print(out.shape, dlstm.t % dlstm.r)    # torch.Size([2, 16]) 5
```

묶음 하나는 10 step에 한 번만 갱신되므로, 같은 LSTM 셀로도 10배 긴 시간 간격의 정보를 담기 쉬워집니다. 역전파 관점에서는 기울기가 10 step을 한 번에 건너뛰는 셈이라 먼 과거까지 잘 전달됩니다. (논문 구현은 묶음 사이에 파라미터를 공유하는 하나의 LSTM 셀을 쓰고, 위 코드도 같은 `self.cell`을 공유합니다.)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FuNHeads(nn.Module):
    def __init__(self, z_dim, d=256, k=16, n_actions=18, c=10):
        super().__init__()
        self.c, self.k, self.n_actions = c, k, n_actions
        self.m_space = nn.Linear(z_dim, d)
        self.m_rnn = nn.LSTMCell(d, d)                      # 실제로는 dilated LSTM
        self.w_rnn = nn.LSTMCell(z_dim, n_actions * k)
        self.phi = nn.Linear(d, k, bias=False)

    def forward(self, z, hm, hw, goal_hist):
        s = F.relu(self.m_space(z))
        hm = self.m_rnn(s, hm)
        g = F.normalize(hm[0], dim=-1)                      # 방향 벡터
        goal_hist = (goal_hist + [g])[-(self.c + 1):]
        w = self.phi(torch.stack(goal_hist).sum(0).detach())   # Worker로 기울기 차단
        hw = self.w_rnn(z, hw)
        U = hw[0].view(-1, self.n_actions, self.k)
        pi = F.softmax(torch.bmm(U, w.unsqueeze(-1)).squeeze(-1), dim=-1)
        return pi, s, g, hm, hw, goal_hist

def worker_intrinsic(s_hist, g_hist, c=10):
    s_t = s_hist[-1]
    return sum(F.cosine_similarity(s_t - s_hist[-1 - i], g_hist[-1 - i], dim=-1)
               for i in range(1, c + 1)) / c

torch.manual_seed(0)
B, z_dim = 2, 64
heads = FuNHeads(z_dim)
hm = (torch.zeros(B, 256), torch.zeros(B, 256))                 # Manager LSTM 상태
hw = (torch.zeros(B, 18 * 16), torch.zeros(B, 18 * 16))         # Worker LSTM 상태
goal_hist, s_hist = [], []
for t in range(12):                                             # 지각 모듈 출력 z 대신 난수 사용
    pi, s, g, hm, hw, goal_hist = heads(torch.randn(B, z_dim), hm, hw, goal_hist)
    s_hist.append(s)
print(pi.shape, pi.sum(-1))                        # torch.Size([2, 18]) tensor([1., 1.])
print(worker_intrinsic(s_hist, goal_hist).shape)   # torch.Size([2]) 배치마다 intrinsic reward 하나
```

`w`를 만들 때 `.detach()`로 목표의 기울기를 끊는 줄이 중요합니다. 이 줄이 없으면 Worker의 loss가 Manager까지 흘러가, 목표 $$g$$가 "상태 공간의 방향"이라는 의미를 잃고 Worker를 돕는 임의의 내부 변수처럼 학습됩니다. Manager는 위의 transition policy gradient로만 학습해야 합니다.

## 5. 실험

Atari와 DeepMind Lab(3D 미로 Labyrinth)에서 LSTM baseline(A3C)과 비교했습니다.

- **Montezuma's Revenge**: 보상이 매우 드문 게임에서 LSTM baseline보다 훨씬 빠르게 점수를 얻었습니다.
- **Ms. Pac-Man, Seaquest, Enduro 등**: 장기 계획이 필요한 게임에서 크게 앞섰고, 일반 게임에서도 성능을 잃지 않았습니다.
- **Labyrinth의 기억 과제**: 긴 기억이 필요한 과제에서 dilated LSTM을 쓴 Manager가 효과적이었습니다.

### 하위 정책의 해석

**Seaquest**에서 Manager가 내는 목표를 몇 개의 군집으로 나눠, 각 목표를 받았을 때 Worker가 어디로 가는지 시각화했습니다.

- LSTM baseline의 정책은 화면 여러 곳을 뒤섞어 돌아다녔습니다.
- FuN의 하위 정책들은 각자 **화면의 특정 영역**에 집중했습니다. 그중 한 하위 정책은 **산소를 보충하러 수면으로 헤엄쳐 올라가는** 행동에 쓰였습니다.

사람이 목표를 정해 주지 않았는데도, 의미 있는 하위 행동이 스스로 생겨난 것입니다.

## 6. 정리

| | h-DQN | FeUdal Networks |
|---|---|---|
| 목표 | 사람이 정한 물체/위치 중 선택 | **학습된 상태 공간의 방향 벡터** |
| Manager 학습 | DQN (extrinsic) | **Transition policy gradient** |
| Worker 보상 | 목표 도달 여부 (0/1) | 상태 변화와 목표 방향의 **코사인 유사도** |
| 장기 기억 | - | **Dilated LSTM** |

## References

- Vezhnevets et al., "FeUdal Networks for Hierarchical Reinforcement Learning," ICML 2017. [arXiv:1703.01161](https://arxiv.org/abs/1703.01161)
- Dayan & Hinton, "Feudal Reinforcement Learning," NeurIPS 1993.
- Kulkarni et al., "Hierarchical Deep Reinforcement Learning (h-DQN)," NeurIPS 2016.
- Bacon et al., "The Option-Critic Architecture," AAAI 2017.
