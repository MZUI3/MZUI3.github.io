---
title: "[RL] Phasic Policy Gradient: 정책과 가치 함수의 학습을 단계로 나눠 feature는 공유하되 간섭은 끊기"
date: 2026-09-16 09:32:00 +0900
categories: [Reinforcement Learning, Model-Free]
tags: [reinforcement-learning, model-free, ppo, phasic-policy-gradient, ppg, actor-critic, auxiliary-task, sample-reuse, procgen, representation-learning]
description: actor-critic에서 정책과 가치 함수가 네트워크를 공유하면 feature는 나누지만 목표가 간섭하고, 분리하면 간섭은 없지만 feature를 못 나눈다는 딜레마를, 정책 단계와 보조 단계를 번갈아 두는 것으로 푼 Phasic Policy Gradient(Cobbe et al., ICML 2021)의 알고리즘, Procgen 결과, 샘플 재사용 분석을 정리하고 보조 단계의 동작을 코드로 확인합니다.
math: true
---

> [PPO]({% post_url 2026-07-05-trpo-to-ppo %}) 같은 actor-critic 방법을 구현할 때 늘 마주치는 선택이 있습니다. 정책과 가치 함수가 **네트워크를 공유**할 것인가, **따로** 둘 것인가. 공유하면 한쪽이 배운 feature를 다른 쪽이 쓰지만, 두 목표의 가중치를 정해야 하고 서로 간섭합니다. 따로 두면 간섭은 없지만 feature를 나누지 못합니다. **Phasic Policy Gradient**(PPG, Cobbe, Hilton, Klimov, Schulman, OpenAI, ICML 2021)는 둘의 장점만 취합니다. 학습을 **정책 단계**와 **보조 단계**로 나눠, 정책 단계에서는 분리된 네트워크로 PPO를 돌리고, 보조 단계에서는 가치 함수의 feature를 정책 네트워크로 증류합니다. 덤으로 가치 함수를 정책보다 훨씬 많이 재사용해 학습할 수 있게 되어 Procgen에서 샘플 효율이 크게 올랐습니다.

## 1. 문제: 공유의 두 가지 비용

파라미터 공유의 장점은 분명합니다. 한 목표로 학습한 feature가 다른 목표의 최적화를 돕습니다. 비용도 두 가지입니다.

1. **간섭**: 정책 목표와 가치 목표를 한 네트워크로 함께 최적화하려면 상대 가중치를 정해야 하고, 아무리 잘 정해도 한쪽의 최적화가 다른 쪽을 방해할 위험이 있습니다.
2. **같은 샘플 재사용**: 공유 네트워크는 두 목표를 같은 데이터로, 같은 횟수만큼 학습하도록 사실상 강제합니다. 인위적이고 바람직하지 않은 제약입니다.

PPG로 밝힌 두 관찰이 이 비용이 실재함을 보여 줍니다. 공유 네트워크에서 간섭은 성능을 떨어뜨리고, 가치 함수는 정책보다 **훨씬 높은 샘플 재사용**을 견딥니다.

## 2. 알고리즘

### 정책 단계

정책 네트워크와 가치 네트워크를 **분리**해서 PPO를 돌립니다. 정책은 clipped surrogate에 엔트로피 보너스를 더한 목표로,

$$
L^{\text{clip}} = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta) \hat A_t,\ \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat A_t\big)\Big],\qquad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

가치 네트워크는 GAE로 만든 목표에 대한 제곱 오차로 학습합니다.

$$
L^{\text{value}} = \hat{\mathbb{E}}_t\Big[\tfrac{1}{2}\big(V_{\theta_V}(s_t) - \hat V^{\text{targ}}_t\big)^2\Big]
$$

정책 네트워크에는 **보조 가치 head**가 하나 더 달려 있습니다. 마지막 선형층만 다르고 나머지는 정책과 공유하며, 오직 정책의 표현을 학습시키는 용도입니다.

### 보조 단계

$$N_\pi$$번의 정책 단계마다 한 번, 그동안 모은 모든 상태와 가치 목표를 버퍼 $$B$$에서 꺼내 정책 네트워크를 **보조 목표 + 행동 복제** 손실로 학습합니다.

$$
L^{\text{joint}} = L^{\text{aux}} + \beta_{\text{clone}} \cdot \hat{\mathbb{E}}_t\big[\text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t),\ \pi_\theta(\cdot \mid s_t)]\big],\qquad L^{\text{aux}} = \tfrac{1}{2} \hat{\mathbb{E}}_t\Big[\big(V_{\theta_\pi}(s_t) - \hat V^{\text{targ}}_t\big)^2\Big]
$$

$$\pi_{\theta_{\text{old}}}$$는 보조 단계 직전의 정책입니다. 보조 가치 head로 가치 목표를 맞추되(feature 공유), KL 항으로 **정책 자체는 최대한 보존**합니다. 원칙적으로 $$L^{\text{aux}}$$는 어떤 보조 목표든 될 수 있고, 이 논문에서는 가치 오차만 씁니다. 보조 단계에서는 가치 네트워크도 같은 버퍼로 추가 학습합니다. 두 손실은 파라미터를 공유하지 않으므로 따로 최적화합니다.

```text
phase = 1, 2, ... 반복
    버퍼 B 비움
    iteration = 1..N_π:                       (정책 단계)
        현재 정책으로 rollout, GAE로 V^targ 계산
        E_π epoch: L^clip + β_S·S[π]로 θ_π 갱신
        E_V epoch: L^value로 θ_V 갱신
        (s_t, V^targ_t)를 B에 추가
    B의 모든 상태에 대해 π_old(·|s_t) 저장
    E_aux epoch:                              (보조 단계)
        B 전체로 L^joint로 θ_π 갱신
        B 전체로 L^value로 θ_V 갱신
```

### 하이퍼파라미터의 역할

| 파라미터 | 뜻 | 기본값 |
|---|---|---|
| $$N_\pi$$ | 정책 단계당 정책 갱신 횟수(보조 단계의 빈도) | 32 |
| $$E_\pi$$, $$E_V$$ | 정책 단계에서 정책과 가치의 샘플 재사용 | 1, 1 |
| $$E_{\text{aux}}$$ | 보조 단계에서 버퍼 전체를 도는 epoch 수 | 6 |
| $$\beta_{\text{clone}}$$ | 정책 보존 강도 | 1 |

가치 함수의 샘플 재사용을 늘릴 때는 $$E_V$$가 아니라 보통 $$E_{\text{aux}}$$를 늘립니다. 그 밖에 $$\gamma = 0.999$$, $$\lambda = 0.95$$, rollout 256 step, 클립 0.2, 엔트로피 0.01, 보상 정규화, 학습률 $$5 \times 10^{-4}$$, 환경 64개 × worker 4개, 총 1억 step입니다.

## 3. 코드로 보기: 보조 단계는 feature를 바꾸고 정책은 남긴다

가짜 rollout 데이터로 정책 단계 32번과 보조 단계 1번을 돌리고, 보조 단계 전후로 정책 네트워크의 보조 가치 head 오차와 정책의 변화(KL)를 잽니다. $$\beta_{\text{clone}} = 0$$으로 두면 가치 손실이 정책을 얼마나 끌고 가는지도 봅니다.

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
S, A, N = 16, 4, 2048                                       # 상태 차원, 행동 수, 버퍼 크기
# 1) 정책 네트워크(공유 feature + 정책 head + 보조 가치 head)와 별도의 가치 네트워크
feat = nn.Sequential(nn.Linear(S, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
pi_head, aux_v_head = nn.Linear(64, A), nn.Linear(64, 1)
value_net = nn.Sequential(nn.Linear(S, 64), nn.Tanh(), nn.Linear(64, 1))
opt_pi = torch.optim.Adam(list(feat.parameters()) + list(pi_head.parameters()) + list(aux_v_head.parameters()), 5e-4)
opt_v = torch.optim.Adam(value_net.parameters(), 5e-4)

# 가짜 rollout: 상태, 행동, advantage, 가치 목표 (실제로는 환경에서 GAE로 계산)
states = torch.randn(N, S); actions = torch.randint(A, (N,))
adv = torch.randn(N); v_targ = (states[:, :4].sum(1) * 2 + torch.randn(N) * 0.3)   # 상태의 어떤 함수 + 노이즈

def policy_phase(E_pi=1, E_v=1, eps=0.2):
    """정책 단계: PPO clip 목표로 정책을, MSE로 가치 네트워크를 각각 학습 (서로 다른 네트워크)"""
    with torch.no_grad(): logp_old = F.log_softmax(pi_head(feat(states)), -1).gather(1, actions[:, None]).squeeze(1)
    for _ in range(E_pi):
        logp = F.log_softmax(pi_head(feat(states)), -1).gather(1, actions[:, None]).squeeze(1)
        r = (logp - logp_old).exp()
        L_clip = -torch.min(r * adv, r.clamp(1 - eps, 1 + eps) * adv).mean()
        opt_pi.zero_grad(); L_clip.backward(); opt_pi.step()
    for _ in range(E_v):
        L_v = 0.5 * (value_net(states).squeeze(1) - v_targ).pow(2).mean()
        opt_v.zero_grad(); L_v.backward(); opt_v.step()

def aux_phase(E_aux=6, beta_clone=1.0):
    """보조 단계: 정책 네트워크의 보조 가치 head로 가치 목표를 맞추되(feature 공유), 정책은 KL로 묶어 둔다"""
    with torch.no_grad(): pi_old = F.softmax(pi_head(feat(states)), -1)
    for _ in range(E_aux):
        for i in torch.randperm(N).split(256):
            phi = feat(states[i])
            L_aux = 0.5 * (aux_v_head(phi).squeeze(1) - v_targ[i]).pow(2).mean()
            kl = (pi_old[i] * (pi_old[i].log() - F.log_softmax(pi_head(phi), -1))).sum(1).mean()
            L_joint = L_aux + beta_clone * kl
            opt_pi.zero_grad(); L_joint.backward(); opt_pi.step()
            L_v = 0.5 * (value_net(states[i]).squeeze(1) - v_targ[i]).pow(2).mean()   # 가치 네트워크도 추가 학습
            opt_v.zero_grad(); L_v.backward(); opt_v.step()
    with torch.no_grad():
        kl_after = (pi_old * (pi_old.log() - F.log_softmax(pi_head(feat(states)), -1))).sum(1).mean()
        aux_err = (aux_v_head(feat(states)).squeeze(1) - v_targ).pow(2).mean()
    return kl_after.item(), aux_err.item()

with torch.no_grad(): err0 = (aux_v_head(feat(states)).squeeze(1) - v_targ).pow(2).mean().item()
for _ in range(32): policy_phase()                          # N_π = 32번의 정책 단계
kl, err = aux_phase()
print(f"보조 가치 head 오차: 보조 단계 전 {err0:.3f} → 후 {err:.3f}  (정책 네트워크의 feature가 가치 정보를 담게 됨)")
print(f"보조 단계 동안 정책이 움직인 정도 KL(π_old ‖ π) = {kl:.5f}  (β_clone = 1로 묶어 둠)")
kl_free, _ = aux_phase(beta_clone=0.0)
print(f"β_clone = 0이면 KL = {kl_free:.5f}  (가치 손실이 정책을 끌고 감 = 간섭)")
```

```text
보조 가치 head 오차: 보조 단계 전 16.419 → 후 9.100  (정책 네트워크의 feature가 가치 정보를 담게 됨)
보조 단계 동안 정책이 움직인 정도 KL(π_old ‖ π) = 0.00268  (β_clone = 1로 묶어 둠)
β_clone = 0이면 KL = 0.00793  (가치 손실이 정책을 끌고 감 = 간섭)
```

정책 단계 동안에는 가치 손실이 정책 네트워크에 전혀 닿지 않으므로 보조 가치 head의 오차가 큽니다. 보조 단계가 그 오차를 절반 가까이 줄이면서 공유 feature에 가치 정보를 심고, 그동안 정책의 변화는 KL 0.003 수준에 묶입니다. $$\beta_{\text{clone}}$$을 0으로 두면 같은 가치 손실이 정책을 세 배 더 움직입니다. 이것이 공유 네트워크에서 늘 일어나는 **간섭**이고, PPG는 그것을 보조 단계에 가두고 KL로 제한합니다.

## 4. 실험: Procgen

16개 환경의 Procgen Benchmark에서 하이퍼파라미터를 충분히 조정한 PPO 구현(sample reuse 3, 가치·정책 손실 가중치 최적)과 비교했습니다. 3회 실행의 표준편차를 함께 그렸습니다.

- **PPO 대비**: 거의 모든 환경에서 PPG의 샘플 효율이 유의하게 좋습니다.
- **표현 공유의 중요성은 환경에 따라 다릅니다.** Procgen에서는 정책과 가치의 파라미터 공유가 결정적이지만(부록 B: 분리 네트워크 PPO는 훨씬 나쁨), 저차원 입력 환경에서는 종종 불필요합니다. 입력이 고차원(시각)일수록 PPG가 PPO를 이길 가능성이 크다고 봅니다.

### 정책의 샘플 재사용

PPO에서 epoch 수를 늘리면 정책과 가치가 함께 더 학습되어 효과가 뒤섞입니다. PPG는 둘을 분리하므로 $$E_V$$를 고정하고 $$E_\pi$$만 바꿀 수 있습니다. 결과는 **정책 epoch 1이 거의 항상 최적**이었습니다. PPO가 sample reuse 3에서 가장 좋았던 이유는 정책이 아니라 **가치 함수가 추가 학습을 필요로 했기 때문**이라는 뜻입니다. 학습률을 인위적으로 낮추면 정책 재사용을 늘리는 것이 도움이 되므로, 결론은 "잘 조정된 하이퍼파라미터에서는 정책 epoch 1이 거의 최적"입니다.

### 가치의 샘플 재사용

$$E_{\text{aux}}$$를 1에서 9까지 바꾸면 늘릴수록 대체로 좋아지다가 6 근처에서 포화합니다. 이득의 출처는 두 가지입니다. $$L^{\text{joint}}$$로 더 잘 학습된 feature가 정책에 공유되는 것과, $$L^{\text{value}}$$로 더 정확해진 가치 함수가 다음 정책 단계의 기울기 분산을 줄이는 것입니다. Procgen에서는 전자가 더 중요해 보입니다. 부록 C에서 보조 단계의 $$L^{\text{value}}$$ 최적화를 빼고 대신 정책 단계의 $$E_V$$를 늘려도 거의 같은 성능이 나와, 보조 단계에서 가치 네트워크를 더 학습하는 것은 편리하지만 필수는 아닙니다.

### 보조 단계의 빈도와 목표 함수

- $$N_\pi$$를 2에서 32까지 바꾸면 **보조 단계가 너무 잦을 때 성능이 떨어집니다**. 보조 단계마다 정책 최적화가 방해받고, 자주 하면 그 효과가 누적된다는 추측입니다. 드문 보조 단계가 성공의 열쇠입니다.
- 클리핑 대신 **고정 가중치 KL 페널티**($$\beta_\pi = 1$$)를 써도 PPG에서는 놀랍도록 비슷했습니다. 보상을 정규화해 할인 return의 분산을 1 근처로 맞췄기 때문에 클리핑의 필요가 줄었다고 봅니다.

$$
L^{KL} = \hat{\mathbb{E}}_t\Big[-\hat A_t \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} + \beta_\pi \text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)]\Big]
$$

### 단일 네트워크 PPG

분리 네트워크는 파라미터가 PPO의 약 두 배입니다. 공유 네트워크 하나로도 같은 역학을 흉내 낼 수 있습니다. 정책 단계에서는 가치 기울기를 마지막 공유 층에서 **끊어**(detach) 공유 파라미터에 닿지 않게 하고, 보조 단계에서는 모든 파라미터로 가치 기울기를 흘립니다. 이 변형은 이중 네트워크 PPG에 거의 맞먹었습니다. 정책 단계에서 가치 함수가 잘 학습되지 못할까 걱정했지만 보조 단계의 전체 기울기로 충분했습니다.

## 5. 해석

**1. 딜레마를 시간으로 풀었습니다.** "공유냐 분리냐"는 양자택일처럼 보이지만, PPG는 **언제** 공유하고 **언제** 분리할지로 바꿉니다. 정책 단계에는 분리(간섭 없음), 보조 단계에는 공유(feature 전달). 위 코드에서 본 것처럼 보조 단계의 KL 항이 그 전환의 안전장치입니다.

**2. 가장 유용한 발견은 샘플 재사용의 비대칭입니다.** 정책은 epoch 1이면 충분하고 가치는 6 epoch까지 좋아진다는 결과는 PPO의 "sample reuse 3"이 사실 타협이었음을 밝힙니다. on-policy 방법에서 가치 함수는 사실상 지도 회귀 문제라 데이터를 더 돌려도 되고, 정책은 분포 이동 때문에 조심해야 합니다. 분리하지 않으면 보이지 않던 사실입니다.

**3. 보조 목표의 틀로 일반화됩니다.** $$L^{\text{aux}}$$는 가치 오차 하나였지만, 보조 단계는 어떤 보조 손실이든 정책 표현에 안정적으로 넣는 틀입니다. ITER처럼 새 네트워크로 증류하는 방법과 달리 PPG는 가치 네트워크를 정책 네트워크로 주기적으로 증류하는 셈이고, [Decision Transformer]({% post_url 2026-10-24-decision-transformer %})가 가치 함수 없이 시퀀스 모델링으로 가는 것과는 정반대 방향에서 "가치 함수가 배운 표현"의 가치를 보여 줍니다.

**4. 간섭의 정체.** 보조 단계가 잦으면 나빠진다는 결과는 가치 손실이 정책 표현을 바꾸는 것 자체가 정책 최적화에 비용이라는 뜻입니다. KL로 묶어도 feature가 바뀌면 정책 단계가 다시 적응해야 합니다. 드물게, 그러나 많이(6 epoch) 하는 것이 그 비용을 가장 잘 상각하는 방법이었습니다.

**5. 평가를 읽을 때 주의할 점.** Procgen 한 벤치마크의 결과이고, 저자들도 저차원 입력 환경에서는 표현 공유가 덜 중요하다고 인정합니다. 새 하이퍼파라미터($$N_\pi$$, $$E_{\text{aux}}$$, $$\beta_{\text{clone}}$$)가 늘었고 16개 환경에 공통 값이 통했지만 다른 도메인에서는 재조정이 필요할 수 있습니다. 비교 대상 PPO는 가치 손실 가중치와 sample reuse를 조정한 강한 구현이지만 다른 on-policy 알고리즘과는 비교하지 않았습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 문제 | 정책·가치 네트워크를 공유하면 간섭과 같은 샘플 재사용 강제, 분리하면 feature 공유 불가 |
| 방법 | 정책 단계(분리 네트워크로 PPO, 정책 epoch 1)와 보조 단계(버퍼 전체로 보조 가치 head 학습 + KL로 정책 보존, 6 epoch)를 $$N_\pi = 32$$마다 번갈아 수행 |
| 결과 | Procgen 16개 환경에서 잘 조정된 PPO보다 샘플 효율 향상 |
| 발견 | 정책은 epoch 1이 거의 최적, 가치는 6 epoch까지 이득. 보조 단계는 드물어야 함. KL 페널티도 클리핑만큼 좋음 |
| 변형 | 정책 단계에서 가치 기울기를 끊는 단일 네트워크 PPG가 거의 동등 |
| 한계 | 고차원(시각) 입력에서 특히 유효, 새 하이퍼파라미터 추가 |

## References

- Cobbe, Hilton, Klimov, Schulman, "Phasic Policy Gradient," ICML 2021. [arXiv:2009.04416](https://arxiv.org/abs/2009.04416)
- Schulman et al., "Proximal Policy Optimization Algorithms," 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation," ICLR 2016. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- Cobbe et al., "Leveraging Procedural Generation to Benchmark Reinforcement Learning," ICML 2020. [arXiv:1912.01588](https://arxiv.org/abs/1912.01588)
- Igl et al., "The Impact of Non-stationarity on Generalisation in Deep Reinforcement Learning," 2020. [arXiv:2006.05826](https://arxiv.org/abs/2006.05826)
