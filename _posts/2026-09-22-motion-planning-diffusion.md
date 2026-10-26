---
title: "[Robotics] Motion Planning Diffusion: 디퓨전 사전 분포에서 비용 가이던스로 로봇 궤적을 바로 샘플링하기"
date: 2026-09-22 22:54:00 +0900
categories: [Robotics, Motion Planning]
tags: [motion-planning, diffusion-model, planning-as-inference, guidance, trajectory-optimization, gpmp, panda, imitation-learning]
description: Motion Planning Diffusion(Carvalho et al., IROS 2023)이 전문가 궤적으로 학습한 디퓨전 모델을 사전 분포로 삼고, 계획을 추론으로 보는 관점에서 충돌·평활·관절 한계 비용의 기울기를 디노이징 과정에 가이던스로 넣어 사후 분포에서 궤적을 직접 샘플링하는 방법을 정리합니다. 2D·3D 점질량과 7자유도 Panda 환경에서 CVAE·GPMP·RRTConnect와의 비교, 미지 장애물 일반화를 다루고, 작은 실험으로 가이던스의 효과를 확인합니다.
math: true
---

> [Diffuser]({% post_url 2026-09-19-diffuser %})는 오프라인 RL에서 궤적 전체를 디퓨전으로 생성했고, [Diffusion Policy]({% post_url 2026-10-25-diffusion-policy %})는 시각운동 정책을 디노이징으로 만들었습니다. **Motion Planning Diffusion**(MPD, Carvalho, Le, Baierl, Koert, Peters, TU Darmstadt, IROS 2023)은 디퓨전 모델을 **고전 모션 플래닝의 사전 분포**로 가져옵니다. 최적 플래너가 만든 충돌 없는 궤적으로 디퓨전 모델을 학습해 두고, 새 계획 문제에서는 충돌·평활·관절 한계 같은 표준 비용의 기울기를 디노이징 단계마다 더해 **사후 분포에서 궤적을 바로 뽑습니다**. 사전 분포에서 뽑은 뒤 따로 최적화하는 방식보다 성공률이 높고 다양하며, 학습 때 없던 장애물이 있는 환경에서도 동작합니다.

## 1. 문제: 샘플링 플래너는 비효율적이고, 최적화 플래너는 초기값에 묶인다

샘플링 기반(PRM, RRT)은 완전성이 있지만 표본 효율이 낮고 궤적이 거칠며, 최적화 기반(CHOMP, GPMP, STOMP)은 평활도를 비용으로 넣을 수 있지만 **좋은 초기 궤적**이 없으면 지역 최소(장애물을 뚫고 지나가는 해)에 빠집니다. 그래서 과거의 성공 사례에서 초기 분포를 배우려는 연구가 이어졌습니다. GMM은 고차원·다봉 궤적을 잘 못 담고, 에너지 기반 모델은 고차원 샘플링이 어렵습니다. 디퓨전 모델은 고차원 다봉 데이터에 강하고, 무엇보다 **샘플링 과정에 비용 기울기를 자연스럽게 섞을 수 있습니다**.

## 2. 방법

### 계획을 추론으로

상태 $$s = [q, \dot q]$$, 궤적 $$\tau = (s_0, \ldots, s_{H-1})$$, 과제 $$O$$에 대해 사후 분포는 $$p(\tau \mid O) \propto p(O \mid \tau)\, p(\tau)$$입니다. 우도가 비용의 지수족 $$p_i(o_i \mid \tau) \propto \exp(-c_i(\tau))$$로 분해된다고 두면 MAP는

$$
\tau^* = \arg\min_\tau \sum_i \lambda_i c_i(\tau) - \log p(\tau)
$$

로, 고전 궤적 최적화에 **사전 분포의 로그**가 더해진 형태입니다. GPMP가 가우시안 과정으로 평활 사전을 넣었다면, MPD는 **전문가 데이터로 학습한 디퓨전 사전**을 넣습니다.

### 궤적 디퓨전 모델

DDPM 그대로입니다. 순방향 $$q(\tau_t \mid \tau_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t}\, \tau_{t-1}, \beta_t I)$$, 폐형식 $$\tau_t = \sqrt{\bar\alpha_t}\, \tau_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon$$, 잡음 예측 손실 $$\lVert \varepsilon - \varepsilon_\theta(\tau_t, t) \rVert^2$$. 네트워크는 Diffuser의 시간 U-Net이고, 위치뿐 아니라 **속도까지** 상태에 포함해 학습합니다.

### 가이던스로 사후 샘플링

역방향 각 단계에서 과제 조건부 사후 $$p(\tau_{t-1} \mid \tau_t, O) \propto p(\tau_{t-1} \mid \tau_t)\, p(O \mid \tau_{t-1})$$을 샘플링해야 합니다. 디노이저가 주는 가우시안 $$\mathcal{N}(\mu_t, \Sigma_t)$$는 $$t \to 0$$에서 분산이 작아지므로 로그 우도를 $$\mu_t$$ 주변에서 1차 테일러 전개하면(classifier guidance와 같은 논법), 사후도 가우시안 $$\mathcal{N}(\mu_t + \Sigma_t g, \Sigma_t)$$가 되고

$$
g = \nabla_{\tau} \log p(O \mid \tau)\big|_{\mu_t} = -\sum_i \lambda_i \nabla_\tau c_i(\tau)\big|_{\mu_t}
$$

입니다. 즉 **디노이징 평균을 비용 기울기 방향으로 한 걸음 옮기고** 잡음을 더합니다. 실제로는 $$\Sigma_t$$ 스케일을 빼고 상수 $$\tilde\lambda_i$$를 쓰는데, 이는 $$\lambda_i = \tilde\lambda_i / \sigma_t$$로 둔 것과 같아 디노이징이 끝날수록 우도의 영향이 커집니다. 시작·목표 상태는 매 단계 **하드 셋**(디랙 델타 우도)합니다.

### 비용

- **충돌**: 로봇 몸에 $$K$$개 충돌 구를 두고 PyTorch 미분 가능 순기구학으로 위치를 구한 뒤, 부호 거리 함수에 여유 $$\epsilon$$을 둔 hinge 비용.
- **자기 충돌**: 링크별 구 그룹 간 거리 평균.
- **관절·속도 한계**: 여유를 둔 L2 위반량.
- **엔드이펙터**: SE(3) 거리(위치 L2 + 회전 로그맵 노름).
- **가우시안 과정(평활)**: 상수 속도 모델의 상태 전이 $$\Phi_{t,t+1}$$과 공분산 $$Q_{t,t+1}$$로 $$\frac{1}{2}\sum_t \lVert \Phi s_t - s_{t+1} \rVert^2_{Q^{-1}}$$. GPMP의 사전을 비용으로 재사용한 것입니다.

## 3. 코드로 보기: 사전 샘플링 대 가이던스 샘플링

원점의 원형 장애물을 왼쪽 또는 오른쪽으로 돌아가는 2D 전문가 궤적 2,000개로 작은 디퓨전 모델을 학습하고, 학습 때 없던 **새 장애물**을 왼쪽 경로 위에 놓은 뒤 가이던스 없이 뽑은 궤적과 충돌·평활 비용 기울기를 넣어 뽑은 궤적의 충돌 없는 비율과 모드 분포를 비교합니다.

```python
import torch, torch.nn as nn, numpy as np
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)
H = 16                                            # waypoints per trajectory (2-D point mass)
def expert(n):                                    # go from (-1,0) to (1,0) around a disc at the origin, left or right
    out = []
    for _ in range(n):
        side = np.random.choice([-1, 1]); amp = np.random.uniform(0.5, 0.8)
        s = np.linspace(0, 1, H); x = -1 + 2 * s; y = side * amp * np.sin(np.pi * s) + 0.02 * np.random.randn(H)
        out.append(np.stack([x, y], 1))
    return torch.tensor(np.array(out), dtype=torch.float32).reshape(n, -1)
data = expert(2000)
T = 100; betas = torch.linspace(1e-4, 0.03, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)
net = nn.Sequential(nn.Linear(2 * H + 1, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2 * H))
opt = torch.optim.Adam(net.parameters(), 1e-3)
for step in range(5000):
    x0 = data[torch.randint(0, len(data), (256,))]; t = torch.randint(0, T, (256,)); eps = torch.randn_like(x0)
    xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps
    loss = ((net(torch.cat([xt, t[:, None] / T], 1)) - eps) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
print(f"diffusion prior trained on {len(data)} expert trajectories, final loss {loss.item():.4f}")

obstacles = [([0.0, 0.0], 0.35), ([0.2, 0.55], 0.25)]      # the training disc and a NEW disc sitting on the left-hand mode
def collision_cost(traj):                                   # soft signed-distance cost summed over discs and waypoints
    p = traj.reshape(-1, H, 2); cost = 0
    for c, rad in obstacles: cost = cost + torch.relu(rad + 0.05 - (p - torch.tensor(c)).norm(dim=-1)).sum(-1)
    return cost
def smooth_cost(traj): p = traj.reshape(-1, H, 2); return ((p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) ** 2).sum((1, 2))
def sample(n, lam=0.0):
    x = torch.randn(n, 2 * H)
    for t in reversed(range(T)):
        x[:, :2] = torch.tensor([-1.0, 0.0]); x[:, -2:] = torch.tensor([1.0, 0.0])      # hard-set start and goal
        with torch.no_grad(): eps = net(torch.cat([x, torch.full((n, 1), t / T)], 1))
        mu = (x - (1 - alphas[t]) / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        if lam > 0:                                                                     # guidance: step along the negative cost gradient
            xg = mu.detach().requires_grad_(True); g = -torch.autograd.grad((collision_cost(xg) + 0.1 * smooth_cost(xg)).sum(), xg)[0]
            mu = mu + lam * g
        x = mu + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
    x[:, :2] = torch.tensor([-1.0, 0.0]); x[:, -2:] = torch.tensor([1.0, 0.0]); return x
def report(name, x):
    p = x.reshape(-1, H, 2)
    hit = torch.zeros(len(p), dtype=torch.bool)
    for c, rad in obstacles: hit |= ((p - torch.tensor(c)).norm(dim=-1) < rad).any(1)
    left = (p[:, H // 2, 1] > 0).float().mean()
    print(f"{name:<36} collision-free {100*(1-hit.float().mean()):5.1f}%   left-mode share {left*100:4.1f}%   mean smoothness cost {smooth_cost(x).mean():.3f}")
torch.manual_seed(1); report("prior samples (no guidance)", sample(200))
torch.manual_seed(1); report("MPD: prior + cost-gradient guidance", sample(200, lam=0.3))
```

```text
diffusion prior trained on 2000 expert trajectories, final loss 0.0453
prior samples (no guidance)          collision-free  48.5%   left-mode share 51.0%   mean smoothness cost 0.250
MPD: prior + cost-gradient guidance  collision-free  99.0%   left-mode share 43.5%   mean smoothness cost 0.140
```

- 사전 분포만으로 뽑으면 학습 장애물과 새 장애물을 합쳐 절반 가까이 충돌합니다. 작은 모델의 샘플 품질 한계에 더해, 새 장애물은 사전이 모르기 때문입니다. 비용 기울기를 넣으면 **99%가 충돌 없이** 나오고, 왼쪽 모드의 비율이 51%에서 44%로 줄어듭니다. 가이던스가 막힌 쪽 모드의 궤적을 밀어내되 모드 자체를 없애지는 않는 것입니다.
- 평활 비용도 함께 넣었으므로 2차 차분 비용이 0.25에서 0.14로 줄었습니다. 사전이 주는 "그럴듯한 모양"과 비용이 주는 "제약 만족"이 한 샘플링 안에서 합쳐지는 것이 MPD의 요점입니다.

## 4. 실험

**환경.** PointMass2D Dense(구·상자 무작위), PointMass3D Maze Boxes(좁은 통로), Panda Spheres(7자유도, 무작위 구), Panda Shelf(작업대와 선반, 실제 로봇 검증). 각 환경에 학습 때 없던 장애물을 더한 "Extra Obstacles" 버전으로 일반화를 봅니다. 지표는 100개의 시작·목표 문맥 × 100개 궤적에 대한 계산 시간, 성공률(S), 궤적 내 충돌 비율(I), 경로 길이(PL), 궤적 분산(VAR).

**비교.** GPU 병렬화한 RRTConnect, 정보 없는 직선 초기값의 GPMP, CVAE 사전(같은 U-Net 인코더·디코더), CVAE 사전 후 최적화(CPost), 각 사전을 초기값으로 쓴 GPMP, 디퓨전 사전(DPrior), MPD.

| 환경 | 방법 | 시간(s) | 성공률 | 분산 |
|---|---|---|---|---|
| PointMass2D Dense | CVAE 사전 | 0.02 | 46% | 0.0 |
| | 디퓨전 사전 | 0.3 | 98% | 1.3 |
| | GPMP(직선 초기값) | 26.5 | 52% | 0.03 |
| | MPD | 0.3 | 99% | 1.4 |
| Panda Spheres | RRTConnect | 42.9 | 100% | - |
| | CVAE 사전 | 0.02 | 36% | 0.02 |
| | 디퓨전 사전 | 0.3 | 88% | 17.3 |
| | GPMP | 194.4 | 42% | 0.02 |
| | CVAE 사후(CPost) | 0.8 | 70% | 0.03 |
| | MPD | 1.1 | **100%** | 17.4 |
| Panda Spheres + 추가 장애물 | 디퓨전 사전 | 0.3 | 78% | 18.0 |
| | MPD | 1.1 | **93%** | 18.0 |

- 디퓨전 사전은 CVAE 사전보다 성공률이 훨씬 높고, 분산이 수백 배 커서 **다봉 분포를 실제로 담습니다**. CVAE는 사실상 한 모드로 붕괴합니다.
- MPD는 사전만 쓸 때보다 성공률이 오르고(Panda Spheres 88 → 100%), 추가 장애물에서도 93%를 유지합니다. 사전을 먼저 뽑고 나중에 최적화하는 CPost(70%)보다 사전과 비용을 **같은 과정에서** 섞는 쪽이 낫습니다.
- 디퓨전 사전을 GPMP의 초기값으로 주면 GPMP의 성공률이 단독일 때의 42%에서 100%로 오릅니다. 좋은 사전이 최적화 플래너의 지역 최소 문제를 푼다는 뜻입니다. 다만 GPMP는 수백 초가 걸립니다.
- RRTConnect는 항상 성공하지만 느리고(궤적 100개 순차), 경로가 길고 거칩니다.

**실제 로봇.** Panda Shelf에서 선반 안으로 들어가는 두 모드의 궤적이 같은 시작·목표에서 샘플링되고, 학습 때 없던 상자 장애물이 있는 실제 환경에서 실행됩니다. 엔드이펙터 방향을 유지하는 SE(3) 비용을 추가로 넣었습니다.

## 5. 해석

**1. "사전에서 뽑고 최적화"가 아니라 "최적화하면서 뽑기"입니다.** Diffuser가 보상 기울기를 가이던스로 쓴 것을 모션 플래닝의 언어로 옮기되, 비용이 평활·충돌·관절 한계처럼 **미분 가능하고 검증된 고전 비용**이라는 점이 다릅니다. CVAE 사후 최적화와의 격차(70 대 100%)가 이 설계의 가치를 보여 줍니다.

**2. 디퓨전의 다봉성이 모션 플래닝에서 왜 중요한지 분명합니다.** 선반 왼쪽과 오른쪽, 장애물 위와 아래처럼 위상적으로 다른 해가 있을 때 한 모드로 붕괴하는 CVAE 사전은 반쯤 실패합니다. 위 toy에서도 가이던스가 모드를 지우지 않고 재배분하는 것을 볼 수 있습니다. [AdaptDiffuser]({% post_url 2026-09-21-adaptdiffuser %})가 생성한 궤적을 다시 학습에 쓰는 것과 결합하면 사전 자체를 새 환경에 맞출 수 있습니다.

**3. GP 비용은 두 세계를 잇는 다리입니다.** GPMP의 평활 사전을 비용으로 그대로 가져와, 디퓨전 사전이 놓친 동역학 실현 가능성을 보완합니다. 속도까지 상태에 넣어 학습한 것도 같은 이유이고, 이후 연구들이 가속도·토크 제약을 같은 틀로 넣습니다.

**4. 일반화는 가이던스가 담당합니다.** 사전은 학습 장애물만 알지만 충돌 비용은 현재 장면의 부호 거리 함수에서 계산되므로, 새 장애물은 비용을 통해 들어옵니다. 사전이 "그럴듯한 움직임"을, 비용이 "지금 환경의 제약"을 맡는 분업이고, 추가 장애물 실험(78 → 93%)이 그 효과입니다.

**5. 평가를 읽을 때 주의할 점.** 성공률 표준편차가 ±40~50%로 매우 큰데, 문맥별로 100개 궤적 중 하나라도 성공하면 성공으로 세는 지표라 분포가 양극화되어 있습니다. 학습 데이터는 최적 플래너가 생성한 것이라 사람 시연의 잡음은 없습니다. 환경별로 모델을 따로 학습하며 장면 조건부(점유 지도 입력)가 아니라서, 장면이 크게 바뀌면 재학습이 필요합니다. 위 toy의 2D 점질량은 논문의 7자유도 문제보다 훨씬 쉽습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 문제 | 최적화 플래너의 초기값 의존과 지역 최소, GMM·EBM 사전의 고차원 다봉 한계 |
| 핵심 | 전문가 궤적으로 학습한 디퓨전 사전 + 계획-추론 관점의 사후 샘플링(비용 기울기 가이던스) |
| 샘플링 | 매 단계 $$\mu_t \leftarrow \mu_t - \sum_i \tilde\lambda_i \nabla c_i$$, 시작·목표 하드 셋 |
| 비용 | 충돌(구 + SDF), 자기 충돌, 관절·속도 한계, SE(3) 엔드이펙터, GP 평활 |
| 결과 | Panda Spheres 성공률 MPD 100%(디퓨전 사전 88%, CVAE 사후 70%, GPMP 42%), 추가 장애물 93%, 분산 수백 배 |
| 관계 | Diffuser의 가이던스를 고전 비용으로, GPMP의 사전을 비용으로 재사용 |

## References

- Carvalho, Le, Baierl, Koert, Peters, "Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models," IROS 2023. [arXiv:2308.01557](https://arxiv.org/abs/2308.01557)
- Janner, Du, Tenenbaum, Levine, "Planning with Diffusion for Flexible Behavior Synthesis," ICML 2022. [arXiv:2205.09991](https://arxiv.org/abs/2205.09991)
- Mukadam et al., "Continuous-time Gaussian process motion planning via probabilistic inference," IJRR 2018. [arXiv:1707.07383](https://arxiv.org/abs/1707.07383)
- Dhariwal, Nichol, "Diffusion Models Beat GANs on Image Synthesis," NeurIPS 2021. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
- Ichter, Harrison, Pavone, "Learning Sampling Distributions for Robot Motion Planning," ICRA 2018. [arXiv:1709.05448](https://arxiv.org/abs/1709.05448)
