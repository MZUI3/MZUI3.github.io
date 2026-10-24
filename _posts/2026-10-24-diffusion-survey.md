---
title: "[Generative AI] Diffusion Models: A Comprehensive Survey — DDPM·SGM·Score SDE 세 공식과 효율적 샘플링·우도·특수 구조 데이터의 지도"
date: 2026-10-24 15:00:00 +0900
categories: [Generative AI, Diffusion]
tags: [diffusion-models, survey, ddpm, score-matching, sde, efficient-sampling, likelihood, taxonomy]
description: Yang 등(ACM Computing Surveys 2023)의 디퓨전 모델 서베이를 정리합니다. DDPM, 점수 기반 생성 모델, Score SDE가 같은 원리의 세 공식임을 보이고, 효율적 샘플링(학습 없는 SDE/ODE 솔버와 학습 기반 이산화·절단·증류), 우도 개선(잡음 스케줄·역방향 분산·정확한 우도), 특수 구조 데이터(이산·불변·다양체), 다른 생성 모델과의 연결, 응용, 미래 방향을 지도처럼 그립니다. 작은 실험으로 신호 대 잡음비 스케줄과 점수-잡음 등가, VP/VE의 분산 차이를 확인합니다.
math: true
---

> 이 블로그에서 [잠재 디퓨전]({% post_url 2026-07-03-latent-diffusion-models %}), [잠재 일관성 모델]({% post_url 2026-07-26-latent-consistency-model %}), [Diffuser]({% post_url 2026-09-19-diffuser %}), [Diffusion Policy]({% post_url 2026-10-23-diffusion-policy %}), [Motion Planning Diffusion]({% post_url 2026-10-24-motion-planning-diffusion %})까지 디퓨전 모델을 여러 각도에서 다뤘습니다. 각 논문은 전체 지형의 한 점이고, 그 지형을 한 장으로 그린 것이 **Diffusion Models: A Comprehensive Survey of Methods and Applications**(Yang, Zhang, Song, Hong, Xu, Zhao, Zhang, Cui, Yang, 북경대·OpenAI·UCLA·CMU·UC Merced, ACM Computing Surveys 2023, arXiv v12 2024)입니다. 세 가지 공식(DDPM, SGM, Score SDE)이 같은 원리임을 자족적으로 설명한 뒤, 연구를 **효율적 샘플링, 우도 개선, 특수 구조 데이터**의 세 축으로 분류하고, 다른 생성 모델과의 연결과 여섯 응용 분야를 훑습니다. 이 글은 그 지도를 따라가되, 개별 방법보다 **분류의 논리**에 집중합니다.

## 1. 기초: 세 공식, 하나의 원리

공통 원리는 데이터를 점점 강한 잡음으로 **망가뜨리는 순방향 과정**과, 그것을 **되돌리는 법을 배우는 역방향 과정**입니다. 역방향의 각 단계는 결국 데이터 밀도가 높은 쪽을 가리키는 **점수** $$\nabla_x \log q(x)$$를 추정하는 일입니다.

**DDPM.** 손으로 설계한 순방향 마르코프 체인 $$q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t}\, x_{t-1}, \beta_t I)$$는 닫힌 형태의 주변 분포 $$q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x_0, (1 - \bar\alpha_t) I)$$를 주고($$\bar\alpha_t = \prod_s (1 - \beta_s)$$), 역방향 전이 $$p_\theta(x_{t-1} \mid x_t)$$를 신경망으로 학습합니다. 변분 하한을 단순화하면 **잡음 예측** 손실 $$\mathbb{E}\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2$$이 됩니다.

**SGM.** 점수 매칭으로 $$s_\theta(x, \sigma) \approx \nabla_x \log q_\sigma(x)$$를 배우되, 저밀도 영역의 부정확한 추정을 피하려고 **여러 잡음 수준**으로 섭동한 분포의 점수를 함께 배우고(NCSN), 큰 잡음부터 작은 잡음으로 **담금질 랑주뱅 동역학**으로 샘플링합니다.

**Score SDE.** 잡음 수준을 연속 시간으로 보내면 순방향은 SDE $$dx = f(x,t)dt + g(t)dw$$이고, 역방향도 SDE $$dx = [f - g^2 \nabla_x \log q_t(x)]dt + g\, d\bar w$$이며, 같은 주변 분포를 갖는 **확률 흐름 ODE**도 존재합니다. DDPM은 분산 보존(VP) SDE의, NCSN은 분산 폭발(VE) SDE의 이산화입니다. 이 통일 덕에 샘플링을 수치 미분방정식 풀이로 볼 수 있고, ODE 쪽에서는 정확한 우도 계산도 가능해집니다.

## 2. 코드로 보기: 스케줄, 점수와 잡음, VP와 VE

선형·코사인 $$\beta$$ 스케줄에서 $$\bar\alpha_t$$와 신호 대 잡음비를 계산하고, 한 단계 커널 300번을 합성한 결과가 닫힌 형태 주변 분포와 같은지, 점수가 잡음 예측의 스케일 변환인지, 그리고 VP/VE SDE의 분산이 시간에 따라 어떻게 다른지 봅니다.

```python
import numpy as np
T = 1000
beta_lin = np.linspace(1e-4, 0.02, T)
abar_lin = np.cumprod(1 - beta_lin)
s = 0.008; f = lambda t: np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
abar_cos = f(np.arange(1, T + 1)) / f(0)
print("DDPM forward process q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I): signal-to-noise ratio along the chain")
print("t       linear beta: abar   SNR(dB)     cosine: abar   SNR(dB)")
for t in (1, 50, 200, 500, 800, 1000):
    a, c = abar_lin[t - 1], abar_cos[t - 1]
    print(f"{t:>4}    {a:14.4f}   {10*np.log10(a/(1-a)):7.1f}    {c:12.4f}   {10*np.log10(c/(1-c)):7.1f}")
# the closed-form marginal equals composing the one-step kernels (Monte Carlo check)
rng = np.random.default_rng(0); x = np.ones(20000) * 0.7
for t in range(300): x = np.sqrt(1 - beta_lin[t]) * x + np.sqrt(beta_lin[t]) * rng.standard_normal(len(x))
print(f"\nafter 300 one-step kernels: mean {x.mean():.4f} (closed form {np.sqrt(abar_lin[299])*0.7:.4f}), std {x.std():.4f} (closed form {np.sqrt(1-abar_lin[299]):.4f})")
# noise prediction and score estimation are the same thing up to a scale
abar = abar_lin[299]; eps = rng.standard_normal(5)
xt = np.sqrt(abar) * 0.7 + np.sqrt(1 - abar) * eps
score = -(xt - np.sqrt(abar) * 0.7) / (1 - abar)                   # grad_x log q(x_t | x_0)
print(f"score = -eps / sqrt(1 - abar): max |difference| = {np.abs(score + eps/np.sqrt(1-abar)).max():.1e}")
print("\nSDE view: VP (DDPM) keeps the total variance near 1, VE (NCSN) lets it explode")
for t in (0.25, 0.5, 1.0):
    integ = 0.1 * t + 0.5 * 19.9 * t * t
    vp = np.exp(-integ) + (1 - np.exp(-integ))                       # var of x_t when var(x_0) = 1
    ve = 1 + (0.01 * (50 / 0.01) ** t) ** 2
    print(f"  t={t}: VP var {vp:.3f}   VE var {ve:9.1f}")
```

```text
DDPM forward process q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I): signal-to-noise ratio along the chain
t       linear beta: abar   SNR(dB)     cosine: abar   SNR(dB)
   1            0.9999      40.0          1.0000      43.8
  50            0.9710      15.3          0.9920      20.9
 200            0.6590       2.9          0.8987       9.5
 500            0.0786     -10.7          0.4938      -0.1
 800            0.0015     -28.1          0.0940      -9.8
1000            0.0000     -43.9          0.0000    -324.3

after 300 one-step kernels: mean 0.4385 (closed form 0.4407), std 0.7782 (closed form 0.7769)
score = -eps / sqrt(1 - abar): max |difference| = 2.2e-16

SDE view: VP (DDPM) keeps the total variance near 1, VE (NCSN) lets it explode
  t=0.25: VP var 1.000   VE var       1.0
  t=0.5: VP var 1.000   VE var       1.5
  t=1.0: VP var 1.000   VE var    2501.0
```

- 선형 스케줄은 $$t = 500$$에서 이미 SNR −10.7dB로 신호가 거의 사라지는 반면 코사인은 −0.1dB로 정보를 더 오래 유지합니다. 4.1절 "잡음 스케줄 최적화"가 다루는 문제이고, 코사인 스케줄(Nichol & Dhariwal)이 우도를 높인 이유입니다.
- 한 단계 커널을 300번 합성한 표본의 평균·표준편차가 닫힌 형태와 일치합니다. DDPM 학습이 임의의 $$t$$로 **한 번에 점프**할 수 있는 근거입니다.
- 점수 $$\nabla_x \log q(x_t \mid x_0) = -\epsilon / \sqrt{1 - \bar\alpha_t}$$이 수치적으로 확인됩니다. DDPM의 잡음 예측과 SGM의 점수 추정이 같은 것이라는 2절의 요점입니다.
- VP SDE는 총분산을 1 근처에 유지하고 VE는 $$t = 1$$에서 2,500까지 키웁니다. 같은 역방향 원리라도 샘플러의 수치적 성질이 달라지는 이유입니다.

## 3. 효율적 샘플링

디퓨전의 약점은 수백~수천 번의 네트워크 평가입니다. 서베이는 해법을 **학습 없는** 것과 **학습 기반**으로 나눕니다.

- **SDE 솔버.** 조상 샘플링을 SDE 이산화로 보고 더 나은 이산화를 설계합니다. 예측-교정(predictor-corrector), 적응 스텝, 고차 솔버 등.
- **ODE 솔버.** 확률 흐름 ODE는 잡음이 없어 큰 스텝이 가능합니다. DDIM은 비마르코프 순방향을 통해 결정적 샘플러를 얻고, DPM-Solver와 DEIS는 ODE의 반선형 구조를 이용한 지수 적분기로 10~20스텝을 달성합니다. 학습 없이 사전학습 모델에 바로 적용된다는 것이 장점입니다.
- **최적화된 이산화.** 어떤 시점들을 쓸지 학습합니다.
- **절단된 디퓨전.** 순방향을 끝까지 돌리지 않고 중간에서 멈추어, 그 지점의 분포를 다른 생성 모델(VAE, GAN)로 생성한 뒤 짧은 역방향만 돕니다.
- **지식 증류.** 여러 스텝의 교사를 한 스텝의 학생으로 증류합니다(점진적 증류, 일관성 모델). 이 블로그의 [LCM]({% post_url 2026-07-26-latent-consistency-model %})과 [ADD]({% post_url 2026-10-23-adversarial-diffusion-distillation %})가 여기 속합니다.

## 4. 우도 개선

DDPM의 단순화 손실은 우도를 직접 최대화하지 않습니다. 세 방향이 있습니다. **잡음 스케줄 최적화**(코사인 스케줄, 학습 가능한 스케줄과 SNR 관점의 VDM), **역방향 분산 학습**(iDDPM처럼 $$\Sigma_\theta$$를 학습하거나 해석적으로 최적 분산을 구함), **정확한 우도 계산**(확률 흐름 ODE의 순간 변수 변환으로 정확한 우도를, 또는 점수 매칭 손실과 우도 하한의 관계를 통해 최대 우도 학습). 가중치를 바꾸면 샘플 품질과 우도 사이에 트레이드오프가 생긴다는 점이 공통된 관찰입니다.

## 5. 특수 구조 데이터

가우시안 잡음은 연속 유클리드 데이터를 전제합니다. **이산 데이터**(텍스트, 분자 그래프)는 다항 전이나 마스킹을 쓰는 이산 디퓨전(D3PM 등)으로, **불변 구조**(분자의 회전·순열 불변성)는 등변 네트워크와 불변 사전 분포로, **다양체 데이터**는 알려진 다양체(구면, 리만 다양체)의 SDE나 학습된 잠재 공간([잠재 디퓨전]({% post_url 2026-07-03-latent-diffusion-models %}))에서의 디퓨전으로 다룹니다. 연속 시간 Score SDE가 다양체로 가장 자연스럽게 확장됩니다.

## 6. 다른 생성 모델과의 연결

디퓨전은 **무한히 깊은 계층적 VAE**로 볼 수 있고(인코더가 고정된), GAN과는 판별기를 더하거나 GAN을 디퓨전으로 정규화하는 식으로 결합되며, 정규화 흐름과는 확률 흐름 ODE가 연속 정규화 흐름이라는 점에서 만나고, 자기회귀 모델과는 순서 있는 생성과 디퓨전 생성을 섞는 방향으로, 에너지 기반 모델과는 점수가 에너지의 기울기라는 점에서 연결됩니다. 서베이의 입장은 결합이 "더 강한 결과"로 이어질 수 있다는 것입니다.

## 7. 응용과 조건부 생성

조건부 디퓨전의 기제(분류기 가이던스, 분류기 없는 가이던스, 교차 attention)를 먼저 정리하고, 라벨·텍스트·이미지·의미 맵·그래프 조건을 다룹니다. 응용은 컴퓨터 비전(초해상도·인페인팅·복원·편집, 분할, 비디오, 점군, 이상 탐지), 자연어 생성, 멀티모달(텍스트→이미지·3D·모션·비디오·오디오), 시계열(대체·예측·파형), 강건 학습(적대적 정화), 학제 간(약물·단백질 설계, 재료, 의료 영상 복원)입니다. [영상 복원 서베이]({% post_url 2026-10-24-restoration-survey %})는 이 중 한 칸을 깊게 파고든 것입니다.

## 8. 미래 방향

네 가지입니다. **가정의 재검토**: 순방향이 유한 시간에 정보를 완전히 지운다는 가정은 성립하지 않으며, 슈뢰딩거 다리와 최적 수송이 유한 시간에 지정된 사전 분포로 수렴하는 대안을 줍니다. **이론적 이해**: 왜 디퓨전이 적대 학습 없이 GAN과 겨루는지, 하이퍼파라미터를 어떻게 체계적으로 정할지. **잠재 표현**: 디퓨전의 잠재 공간은 데이터와 차원이 같아 의미적 조작에 약합니다. **AIGC와 디퓨전 파운데이션 모델**: GPT 류의 생성적 사전학습을 디퓨전에 옮기고 창발 능력을 분석하는 일, LLM과의 결합.

## 9. 해석

**1. 세 공식의 통일이 서베이의 가장 유용한 부분입니다.** DDPM 논문만 읽으면 "잡음 예측"으로, NCSN만 읽으면 "점수 매칭"으로 보이는 것이 Score SDE에서 같은 대상의 이산화임을 알면, 샘플러(ODE 솔버), 가이던스(점수에 기울기 더하기), 증류(ODE 궤적 압축)가 모두 하나의 틀 안에 놓입니다. 위 toy의 "점수 = 잡음/스케일" 확인이 그 핵심입니다.

**2. 분류의 축이 "무엇이 부족한가"입니다.** 느리다(샘플링), 우도가 낮다(우도), 가우시안이 안 맞는다(특수 구조)라는 세 결핍으로 연구를 나누는 방식은 실용적입니다. 다만 2023~2024년의 큰 흐름인 **흐름 매칭·정류 흐름**, 디퓨전 Transformer(DiT), 비디오 모델은 서베이 범위에서 얕게 다뤄지거나 빠져 있어, 지도의 가장자리는 스스로 그려야 합니다.

**3. "잠재 표현이 약하다"는 지적이 흥미롭습니다.** VAE·GAN과 달리 디퓨전의 잠재는 잡음과 같은 차원이라 의미 축을 찾기 어렵고, 이것이 편집에 반전(inversion)이나 attention 조작 같은 우회로가 필요한 이유입니다. 잠재 디퓨전이 오토인코더의 잠재 공간을 빌리는 것도 부분적 답입니다.

**4. 응용 목록은 분야가 얼마나 넓어졌는지를 보여 줍니다.** 이미지에서 출발한 방법이 분자·단백질·시계열·로봇 계획까지 갔고, 공통점은 "조건부 분포에서 다양한 고품질 표본을 뽑는 것"이 필요한 모든 곳입니다. [Diffusion Policy]({% post_url 2026-10-23-diffusion-policy %})와 [MPD]({% post_url 2026-10-24-motion-planning-diffusion %})가 로봇 쪽의 예입니다.

**5. 평가를 읽을 때 주의할 점.** 51쪽 서베이는 방법 간 정량 비교를 거의 하지 않고 분류와 요약에 집중하므로, "무엇이 최선인가"는 개별 논문으로 가야 합니다. 저자 중 한 명(Yang Song)이 Score SDE의 저자라 통일 관점이 강조되는 경향이 있습니다. 위 toy는 1차원 스칼라의 산술 확인이고, 네트워크 학습은 없습니다.

## 10. 정리

| 축 | 내용 |
|---|---|
| 기초 | DDPM(이산 마르코프, 잡음 예측) = SGM(다중 잡음 점수 매칭 + 랑주뱅) = Score SDE(연속 시간, VP/VE, 확률 흐름 ODE) |
| 효율적 샘플링 | 학습 없는 SDE/ODE 솔버(DDIM, DPM-Solver), 학습 기반 이산화·절단·증류 |
| 우도 | 잡음 스케줄(코사인, VDM), 역방향 분산 학습, ODE 기반 정확한 우도 |
| 특수 구조 | 이산 디퓨전, 등변 네트워크, 다양체·잠재 공간 디퓨전 |
| 연결 | 계층적 VAE, GAN 결합, 연속 정규화 흐름, 자기회귀, EBM |
| 미래 | 유한 시간 사전 분포(슈뢰딩거 다리), 이론, 잠재 표현, 디퓨전 파운데이션 모델 |

## References

- Yang et al., "Diffusion Models: A Comprehensive Survey of Methods and Applications," ACM Computing Surveys 2023. [arXiv:2209.00796](https://arxiv.org/abs/2209.00796)
- Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Song, Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution," NeurIPS 2019. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations," ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- Nichol, Dhariwal, "Improved Denoising Diffusion Probabilistic Models," ICML 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
- Lu et al., "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps," NeurIPS 2022. [arXiv:2206.00927](https://arxiv.org/abs/2206.00927)
- Kingma, Salimans, Poole, Ho, "Variational Diffusion Models," NeurIPS 2021. [arXiv:2107.00630](https://arxiv.org/abs/2107.00630)
