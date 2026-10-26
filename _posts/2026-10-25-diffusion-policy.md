---
title: "[RL] Diffusion Policy: 로봇의 행동을 조건부 디노이징으로 생성하는 시각운동 정책"
date: 2026-10-25 14:10:00 +0900
categories: [Reinforcement Learning, Imitation Learning]
tags: [reinforcement-learning, imitation-learning, diffusion-policy, ddpm, visuomotor-policy, multimodal, action-chunking, receding-horizon, robotics, behavior-cloning]
description: 시연의 다봉 행동 분포, 고차원 행동 시퀀스, 학습 안정성이라는 behavior cloning의 세 난제를 행동 공간의 조건부 DDPM으로 푼 Diffusion Policy(Chi et al., RSS 2023)의 정식화, receding horizon·시각 조건·시계열 Transformer 설계, 12개 작업에서 평균 46.9% 향상한 결과와 실제 로봇 실험을 정리하고, 다봉 행동을 작은 예제로 확인합니다.
math: true
---

> [Diffuser]({% post_url 2026-09-19-diffuser %})가 diffusion으로 궤적을 **계획**했다면, **Diffusion Policy**(Chi, Feng, Du et al., Columbia·TRI·MIT, RSS 2023)는 diffusion을 **정책** 자체로 씁니다. 관측을 조건으로 노이즈에서 행동 시퀀스를 디노이징하는 것입니다. 시연 데이터의 다봉 행동 분포를 자연스럽게 표현하고, 고차원 행동 시퀀스를 한 번에 내며, 에너지 기반 정책과 달리 학습이 안정적입니다. 4개 벤치마크 12개 작업에서 평균 **46.9%** 향상했고, 실제 로봇의 Push-T에서 사람에 가까운 95% 성공률을 냈습니다. 이후 [π0]({% post_url 2026-10-21-pi0 %})를 비롯한 VLA의 action expert, [ParaDiGMS]({% post_url 2026-07-27-paradigms %})의 가속 대상이 모두 이 정책입니다.

## 1. 문제: 로봇 행동 예측은 보통의 회귀가 아니다

시연에서 정책을 배우는 가장 단순한 꼴은 관측에서 행동으로의 지도 회귀입니다. 그런데 로봇 행동에는 세 가지 특수성이 있습니다.

- **다봉 분포**: 같은 상황에서 왼쪽으로 돌아가도, 오른쪽으로 돌아가도 됩니다. 평균을 내면 가운데로 돌진합니다.
- **시간적 상관**: 행동은 연속된 시퀀스이고, 일관성이 없으면 떨립니다.
- **높은 정밀도**: 접촉이 많은 조작은 작은 오차에 민감합니다.

기존 해법은 행동 표현을 바꾸거나(가우시안 혼합, 양자화된 범주) 정책을 암시적으로 바꿨습니다(IBC: 에너지 함수를 학습하고 최소화). Diffusion Policy는 행동을 **조건부 디노이징 과정**으로 만듭니다. 세 가지 성질을 물려받습니다.

1. score 함수의 기울기를 배우고 Langevin dynamics로 샘플링하므로 **임의의 정규화 가능한 분포**, 즉 다봉 분포를 표현합니다.
2. 이미지 생성에서 보인 것처럼 **고차원 출력**에 잘 확장되어 행동 시퀀스를 통째로 냅니다. 시간 일관성과 비근시안적 계획에 중요합니다.
3. 정규화 상수를 추정하기 위한 negative sampling이 필요 없어 **학습이 안정적**입니다.

## 2. 정식화

### DDPM

$$x^K$$를 가우시안 노이즈에서 뽑고 $$K$$번 디노이징합니다.

$$
x^{k-1} = \alpha\big(x^k - \gamma\, \varepsilon_\theta(x^k, k) + \mathcal{N}(0, \sigma^2 I)\big)
$$

이것은 노이즈 섞인 경사 하강 한 step $$x' = x - \gamma \nabla E(x)$$으로 읽을 수 있고, $$\varepsilon_\theta$$가 기울기장을, $$\gamma$$가 학습률을, 노이즈 스케줄 $$(\alpha, \gamma, \sigma)$$가 학습률 스케줄을 맡습니다. 학습은 데이터 $$x^0$$에 $$k$$번째 노이즈 $$\varepsilon^k$$를 더하고 그 노이즈를 맞추는 MSE입니다.

$$
\mathcal{L} = \text{MSE}\big(\varepsilon^k,\ \varepsilon_\theta(x^0 + \varepsilon^k, k)\big)
$$

### 시각운동 정책으로

두 가지를 바꿉니다. 출력 $$x$$를 **행동**으로, 디노이징을 **관측 $$O_t$$에 조건부**로.

$$
A^{k-1}_t = \alpha\big(A^k_t - \gamma\, \varepsilon_\theta(O_t, A^k_t, k) + \mathcal{N}(0, \sigma^2 I)\big),\qquad \mathcal{L} = \text{MSE}\big(\varepsilon^k,\ \varepsilon_\theta(O_t, A^0_t + \varepsilon^k, k)\big)
$$

- **닫힌 루프 행동 시퀀스**: 시각 $$t$$에 최근 $$T_o$$ step의 관측을 받아 $$T_p$$ step의 행동을 예측하고, 그중 $$T_a$$ step만 실행한 뒤 다시 계획합니다(receding horizon). 시간 일관성과 반응성의 균형입니다.
- **시각 조건**: [Diffuser]({% post_url 2026-09-19-diffuser %})처럼 $$p(A_t, O_t)$$의 결합 분포가 아니라 **조건부 분포 $$p(A_t \mid O_t)$$**를 모델링합니다. 미래 상태를 추론할 필요가 없고, 시각 특징을 디노이징 반복과 무관하게 **한 번만** 뽑으므로 추론이 훨씬 빨라 실시간 제어와 end-to-end 시각 인코더 학습이 가능합니다.

### 왜 IBC보다 안정적인가

암시적 정책은 에너지 기반 모델 $$p_\theta(a \mid o) = e^{-E_\theta(o, a)} / Z(o, \theta)$$를 InfoNCE로 학습하는데, 다루기 어려운 $$Z$$를 negative sample로 추정하는 부정확함이 학습을 불안정하게 만듭니다. Diffusion Policy는 같은 분포의 **score** $$\nabla_a \log p(a \mid o) = -\nabla_a E_\theta(a, o) - \underbrace{\nabla_a \log Z(o, \theta)}_{= 0}$$를 배우므로 학습에도 추론에도 $$Z$$가 나타나지 않습니다. IBC는 학습 손실이 매끄럽게 줄어도 평가 성공률이 요동쳐 checkpoint 선택이 어려웠지만, Diffusion Policy는 하이퍼파라미터가 작업 간에 거의 일정했습니다.

## 3. 설계 결정

### 네트워크 두 가지

- **CNN 기반**: Diffuser의 1D temporal CNN을 바탕으로, 관측 특징과 디노이징 반복 $$k$$를 **FiLM**으로 모든 합성곱 층에 조건으로 넣고, 행동 궤적만 예측하며, receding horizon과 맞지 않는 inpainting 목표 조건은 뺐습니다. 대부분의 작업에서 튜닝 없이 잘 되지만, 시간적 합성곱의 저주파 편향 때문에 속도 명령처럼 **빠르게 변하는 행동**에는 약합니다.
- **시계열 diffusion Transformer**: minGPT를 바탕으로 노이즈 섞인 행동을 입력 토큰으로, $$k$$의 sinusoidal 임베딩을 첫 토큰으로 넣고, 관측 임베딩을 각 디코더 블록의 cross-attention으로 넣습니다. 상태 기반 실험의 최고 성능 대부분이 이쪽이지만 하이퍼파라미터에 더 민감합니다.

권장은 CNN으로 시작하고, 복잡하거나 행동 변화가 빠른 작업에서 Transformer를 시도하는 것입니다.

### 그 밖

- **시각 인코더**: 사전학습 없는 ResNet-18을 카메라마다 따로, timestep마다 독립으로 인코딩. global average pooling을 **spatial softmax**로 바꿔 공간 정보를 유지하고, EMA와 함께 쓸 때 안정적이도록 BatchNorm을 **GroupNorm**으로.
- **노이즈 스케줄**: iDDPM의 square cosine schedule이 가장 좋았습니다. 스케줄이 행동 신호의 고주파·저주파를 얼마나 잡는지를 정합니다.
- **추론 가속**: DDIM으로 학습 100회, 추론 10회 디노이징을 써서 3080 GPU에서 **0.1초** 지연.

## 4. 코드로 보기: 평균을 내는 회귀와 모드를 고르는 diffusion

장애물 위치 $$o$$마다 왼쪽($$o - 0.6$$)과 오른쪽($$o + 0.6$$)으로 비켜 가는 두 행동이 모두 정답인 1차원 시연을 만들고, MSE 회귀 정책과 조건부 DDPM 정책을 같은 데이터로 학습해 비교합니다.

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
# 1) 다봉(multimodal) 시연: 관측 o(장애물의 위치)마다 왼쪽으로 돌아가는 행동과 오른쪽으로 돌아가는 행동이 모두 정답이다
def demos(n):
    o = torch.rand(n, 1) * 2 - 1                                       # 장애물 위치 ∈ [-1, 1]
    side = torch.where(torch.rand(n, 1) < 0.5, -1.0, 1.0)              # 시연자가 무작위로 고른 방향
    a = o + side * 0.6 + 0.05 * torch.randn(n, 1)                      # 행동: 장애물에서 0.6만큼 비켜 간다
    return o, a
O, A = demos(4000)
def valid(o, a): return ((a - o).abs() - 0.6).abs() < 0.15             # 장애물에서 0.45~0.75 떨어지면 성공

# 2) 명시적 정책(MSE 회귀): 두 모드의 평균을 내놓는다
reg = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(reg.parameters(), 1e-3)
for _ in range(2000):
    i = torch.randint(0, len(O), (256,)); loss = F.mse_loss(reg(O[i]), A[i]); opt.zero_grad(); loss.backward(); opt.step()

# 3) Diffusion Policy: 관측을 조건으로 노이즈에서 행동을 K번 디노이징. ε-예측 네트워크 ε_θ(o, a_k, k)
K = 50
betas = torch.linspace(1e-4, 0.05, K); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)
eps_net = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
opt = torch.optim.Adam(eps_net.parameters(), 1e-3)
for _ in range(4000):
    i = torch.randint(0, len(O), (256,)); k = torch.randint(0, K, (256, 1))
    eps = torch.randn(256, 1); a_k = abar[k].sqrt() * A[i] + (1 - abar[k]).sqrt() * eps     # forward process
    loss = F.mse_loss(eps_net(torch.cat([O[i], a_k, k / K], -1)), eps)                       # 식 (5)
    opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def sample(o):                                                          # 식 (4): a_{k-1} = α(a_k − γ ε_θ) + N(0, σ²)
    a = torch.randn(len(o), 1)
    for k in reversed(range(K)):
        eps = eps_net(torch.cat([o, a, torch.full((len(o), 1), k / K)], -1))
        a = (a - betas[k] / (1 - abar[k]).sqrt() * eps) / alphas[k].sqrt()
        if k > 0: a = a + betas[k].sqrt() * torch.randn_like(a)
    return a

# 4) 같은 관측에서 여러 번 행동을 뽑아 비교한다
o_test = torch.full((2000, 1), 0.2)
with torch.no_grad(): a_reg = reg(o_test)
a_dp = sample(o_test)
print(f"장애물 위치 o = 0.2 에서의 정답 행동: 왼쪽 {0.2 - 0.6:.1f} 또는 오른쪽 {0.2 + 0.6:.1f}")
print(f"MSE 회귀 정책   : 행동 {a_reg[0].item():+.2f} (한 값만 냄) | 성공률 {valid(o_test, a_reg).float().mean():.0%}")
print(f"Diffusion Policy: 행동 평균 {a_dp.mean():+.2f}, 왼쪽 모드 {(a_dp < 0.2).float().mean():.0%} / 오른쪽 모드 {(a_dp >= 0.2).float().mean():.0%} | 성공률 {valid(o_test, a_dp).float().mean():.0%}")
o_all, _ = demos(2000)
with torch.no_grad(): print(f"무작위 관측 2000개: 회귀 성공률 {valid(o_all, reg(o_all)).float().mean():.0%} | Diffusion Policy 성공률 {valid(o_all, sample(o_all)).float().mean():.0%}")
```

```text
장애물 위치 o = 0.2 에서의 정답 행동: 왼쪽 -0.4 또는 오른쪽 0.8
MSE 회귀 정책   : 행동 +0.20 (한 값만 냄) | 성공률 0%
Diffusion Policy: 행동 평균 +0.25, 왼쪽 모드 46% / 오른쪽 모드 54% | 성공률 98%
무작위 관측 2000개: 회귀 성공률 0% | Diffusion Policy 성공률 98%
```

MSE 회귀는 두 모드의 평균인 장애물 **한가운데**(+0.20)를 내놓아 늘 부딪힙니다. Diffusion Policy는 같은 관측에서 샘플마다 왼쪽과 오른쪽을 거의 반반으로 고르고 각 샘플은 한 모드에 **확실히 붙어** 98%가 성공합니다. 논문 Figure 4의 Push-T 상황(T 블록의 왼쪽이나 오른쪽으로 돌아가기)에서 LSTM-GMM과 IBC는 한쪽으로 치우치고 BET는 한 모드를 고수하지 못하는 반면 Diffusion Policy는 두 모드를 모두 배우고 rollout마다 하나에 전념한다는 관찰 그대로입니다. 다봉성은 두 군데에서 옵니다. 매 샘플링의 무작위 초깃값이 수렴할 분지를 정하고, 반복 중의 가우시안 섭동이 분지 사이를 오갈 수 있게 합니다.

## 5. 흥미로운 성질들

- **위치 제어가 속도 제어보다 좋습니다.** 대부분의 BC 연구가 속도 제어를 쓰는 것과 반대입니다. 위치 제어에서 다봉성이 더 두드러지는데 Diffusion Policy는 그것을 잘 다루고, 위치 제어는 복합 오차가 적어 행동 시퀀스 예측에 적합하기 때문으로 봅니다. 기준선들(LSTM-GMM, BET)은 위치 제어로 바꾸면 나빠집니다.
- **행동 시퀀스 예측의 이득**: 행동마다 독립적인 다봉 분포를 예측하면 연속 행동이 서로 다른 모드에서 뽑혀 떨리지만, 시퀀스로 예측하면 **시간 일관성**이 생깁니다. 원격조작 중의 멈춤(idle action)에도 강해서, 단일 step 정책이 멈춤 행동에 과적합해 실제 로봇에서 멈춰 버리는 문제를 피합니다.
- **행동 horizon의 trade-off**: $$T_a$$가 1보다 크면 일관성과 멈춤 보정에 좋지만 너무 길면 반응이 느려집니다. 대부분의 작업에서 **8 step**이 최적이었습니다.
- **지연에 강함**: receding horizon 위치 제어 덕에 관측에서 실행까지 4 step의 지연에도 성능을 유지합니다. 속도 제어는 지연에 더 취약합니다.

## 6. 실험

### 시뮬레이션: 12개 작업, 4개 벤치마크

Robomimic 5개 작업(PH: 숙련자 시연, MH: 숙련·비숙련 혼합), Push-T(접촉이 많은 정밀 밀기), Multimodal Block Pushing(두 블록을 아무 순서로), Franka Kitchen(7개 물체, 4개 작업을 아무 순서로). 최고 checkpoint / 마지막 10개 checkpoint 평균, 3 seed × 50 초기 조건입니다.

| 상태 관측 | Lift PH/MH | Can PH/MH | Square PH/MH | Transport PH/MH | ToolHang PH | Push-T |
|---|---|---|---|---|---|---|
| LSTM-GMM | 1.00/0.96, 1.00/0.93 | 1.00/0.91, 1.00/0.81 | 0.95/0.73, 0.86/0.59 | 0.76/0.47, 0.62/0.20 | 0.67/0.31 | 0.67/0.61 |
| IBC | 0.79/0.41, 0.15/0.02 | 0.00/0.00, 0.01/0.01 | 0.00/0.00, 0.00/0.00 | 0.00/0.00, 0.00/0.00 | 0.00/0.00 | 0.90/0.84 |
| BET | 1.00/0.96, 1.00/0.99 | 1.00/0.89, 1.00/0.90 | 0.76/0.52, 0.68/0.43 | 0.38/0.14, 0.21/0.06 | 0.58/0.20 | 0.79/0.70 |
| **DP-CNN** | 1.00/0.98, 1.00/0.97 | 1.00/0.96, 1.00/0.96 | 1.00/0.93, 0.97/0.82 | 0.94/0.82, 0.68/0.46 | 0.50/0.30 | 0.95/0.91 |
| **DP-Transformer** | 1.00/1.00, 1.00/1.00 | 1.00/1.00, 1.00/0.94 | 1.00/0.89, 0.95/0.81 | 1.00/0.84, 0.62/0.35 | 1.00/0.87 | 0.95/0.79 |

이미지 관측에서도 같은 경향이고, 특히 Transport와 ToolHang처럼 복잡한 작업에서 차이가 큽니다. 다단계 작업에서는 Block Push의 두 블록 모두 성공(p2)이 BET 0.71 → DP-T 0.94, Kitchen의 4개 작업 모두(p4)가 BET 0.44 → DP-C 0.99로 **긴 horizon의 다봉성**(하위 목표의 순서가 임의)도 잘 다룹니다.

### 실제 로봇

- **Push-T**(UR5): 블록을 밀어 넣고 가림을 피해 종료 구역으로 가는 2단계 작업. end-to-end Diffusion Policy가 **성공률 95%, IoU 0.80**(사람 0.84)이고, IBC와 LSTM-GMM의 최선은 0%와 20%입니다. 단계 전환 구간의 높은 다봉성과 모호한 결정 경계에서 기준선이 멈추거나(LSTM-GMM 20회 중 8회) 조기에 떠납니다(IBC 6회). 멈춤 행동을 제거하지 않은 데이터로 학습했는데도 그렇습니다. 사전학습 인코더(R3M 80%, ImageNet은 더 낮음)보다 **end-to-end 학습**이 가장 좋았습니다.
- **교란 실험**: 카메라를 3초 가리면 약간 떨리지만 계속 수행하고, 밀던 블록을 옮기면 즉시 반대쪽에서 다시 밀며, 종료 구역으로 가던 중 블록을 옮기면 되돌아와 고친 뒤 다시 갑니다. 시연에 없던 행동입니다.
- **머그 뒤집기**(6DoF, Franka): 잡기 대 밀기, 정방향 대 역방향 잡기 등 매우 다봉인 시연에서 **90%** 성공. LSTM-GMM은 정렬조차 못 했습니다.
- **소스 붓기·바르기**(액체, 주기적 동작)도 수행했습니다.

## 7. 해석

**1. "정책은 생성 모델이다"를 가장 깔끔하게 보인 논문입니다.** 행동의 조건부 분포를 생성 모델로 두면 다봉성, 고차원, 시간 상관이 한꺼번에 풀립니다. 위 예제에서 50줄이 안 되는 코드로 회귀 정책의 0%가 98%가 된 것은 모델 용량의 차이가 아니라 **문제 정의**의 차이입니다. 회귀는 분포의 평균을 묻고, diffusion은 분포에서 샘플을 뽑습니다.

**2. 기여의 절반은 로봇 공학적 설계입니다.** receding horizon, 조건부 모델링으로 시각 특징을 한 번만 뽑기, 위치 제어, spatial softmax와 GroupNorm, square cosine 스케줄, DDIM 10 step. 이 결정들이 없으면 0.1초 지연의 실시간 제어도, 실제 Push-T의 95%도 없습니다. "diffusion을 썼다"가 아니라 "diffusion을 로봇에서 돌아가게 만들었다"가 이 논문입니다.

**3. 행동 시퀀스 예측은 이후 표준이 되었습니다.** $$T_p$$ step을 예측하고 $$T_a$$만 실행하는 설계는 [ACT]({% post_url 2026-10-20-act-aloha %})의 action chunking과 같은 시기의 같은 깨달음이고, [π0]({% post_url 2026-10-21-pi0 %})의 50Hz action chunk로 이어집니다. 멈춤 행동에 강하다는 관찰은 실제 원격조작 데이터를 다루는 사람에게 특히 실용적입니다.

**4. 느리다는 약점은 다른 논문들이 풀었습니다.** 디노이징 반복이 추론 비용이라 논문은 DDIM 10 step으로 타협했습니다. [ParaDiGMS]({% post_url 2026-07-27-paradigms %})는 이 정책의 100 step 샘플링을 병렬화해 0.74초를 0.2초로 줄였고, π0는 flow matching으로 더 적은 step을 씁니다. Diffusion Policy가 "무엇을 생성하나"를 정했다면 그 뒤의 연구가 "얼마나 빨리"를 다룹니다.

**5. 평가를 읽을 때 주의할 점.** 기준선(BET, IBC)은 원 논문 수치를 썼고 LSTM-GMM만 재현했습니다. Robomimic 평가에 코드 버그로 22개 초기 조건만 쓰였다고 각주에 밝혔습니다(모든 방법에 동일). 실제 로봇 실험은 작업당 20회 수준이고, Transformer 변형은 하이퍼파라미터에 민감합니다. 평균 46.9%라는 수치는 작업과 지표에 따라 편차가 큽니다.

## 8. 정리

| 항목 | 내용 |
|---|---|
| 문제 | 시연 행동의 다봉 분포, 행동 시퀀스의 시간 상관, 높은 정밀도. 회귀·GMM·범주화·IBC 모두 한계 |
| 방법 | 관측 조건부 DDPM으로 행동 시퀀스 생성. $$\varepsilon_\theta(O_t, A^k_t, k)$$를 MSE로 학습, Langevin식 반복으로 샘플링 |
| 설계 | receding horizon($$T_o$$, $$T_p$$, $$T_a = 8$$), 시각 특징 한 번만 추출, CNN(FiLM) 또는 시계열 Transformer, spatial softmax + GroupNorm ResNet-18, square cosine 스케줄, DDIM 10 step(0.1초) |
| 결과 | 12개 작업 평균 46.9% 향상. 실제 Push-T 95%(기준선 0~20%), 머그 뒤집기 90% |
| 성질 | 다봉성 표현, 위치 제어와 시너지, 멈춤 행동과 지연에 강함, IBC보다 안정적 학습 |
| 한계 | 디노이징 반복의 추론 비용, Transformer 변형의 튜닝 민감성 |

## References

- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis," ICML 2022. [arXiv:2205.09991](https://arxiv.org/abs/2205.09991)
- Florence et al., "Implicit Behavioral Cloning," CoRL 2021. [arXiv:2109.00137](https://arxiv.org/abs/2109.00137)
- Shafiullah et al., "Behavior Transformers: Cloning k modes with one stone," NeurIPS 2022. [arXiv:2206.11251](https://arxiv.org/abs/2206.11251)
- Mandlekar et al., "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation," CoRL 2021. [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)
