---
title: "[Generative AI] Latent Consistency Model: 4 step 만에 고해상도 이미지 생성하기"
date: 2026-07-27 22:42:00 +0900
categories: [Generative AI, Diffusion]
tags: [diffusion, consistency-model, lcm, latent-diffusion, distillation, pf-ode, classifier-free-guidance, few-step-sampling]
description: PF-ODE와 Consistency Model의 아이디어에서 출발해, Stable Diffusion을 latent 공간에서 증류해 2~4 step 만에 고해상도 이미지를 만드는 Latent Consistency Model(LCM)의 guided distillation과 skipping-step을 정리합니다.
math: true
---

> [Latent Diffusion Model]({% post_url 2026-07-11-latent-diffusion-models %})은 연산을 latent 공간으로 옮겨 diffusion을 크게 가볍게 만들었지만, 여전히 **이미지 한 장에 수십 step**의 denoising이 필요했습니다. **Latent Consistency Model(LCM)**(Luo et al., 2023)은 Stable Diffusion을 **2~4 step**만으로 고해상도 이미지를 만드는 모델로 증류(distillation)합니다.

## 1. 문제: Diffusion은 느리다

Diffusion 모델은 노이즈에서 시작해 **reverse process를 수십~수백 번 반복**해야 이미지가 나옵니다. 그래서 생성 속도가 느립니다. 이를 줄이는 방법은 크게 두 가지입니다.

1. **더 좋은 ODE solver**: DDIM, DPM-Solver처럼 같은 궤적을 더 큰 보폭으로 따라가기 (그래도 보통 20 step 이상)
2. **Distillation**: 여러 step을 **몇 step 만에** 해내는 모델로 지식을 증류하기

LCM은 2번 방향이며, 그중 **Consistency Model**(Song et al., 2023)을 latent 공간과 텍스트 조건 생성으로 확장한 것입니다.

## 2. 배경: SDE와 PF-ODE

### ODE와 SDE

- **ODE(상미분방정식)**: 함수와 그 도함수 사이의 관계식. 도함수와 초기값을 알면 원래 함수(궤적)를 추정할 수 있습니다.
- **SDE(확률미분방정식)**: ODE에 무작위 노이즈 항이 더해진 것

Song et al.(2021)은 diffusion의 forward 과정(데이터 → 노이즈)을 연속 시간 **SDE**로 표현했습니다.

$$
dx_t = f(t)\,x_t\,dt + g(t)\,dw_t
$$

### PF-ODE (Probability Flow ODE)

같은 SDE에 대해, **시점마다 같은 확률 분포 $$p_t(x)$$를 갖지만 노이즈 항이 없는(결정적인) ODE**가 존재합니다.

$$
\frac{dx_t}{dt} = f(t)\,x_t - \frac{1}{2}g(t)^2\,\nabla_x\log p_t(x_t)
$$

여기서 $$\nabla_x \log p_t$$(score)는 학습된 diffusion 모델의 노이즈 예측으로 근사합니다. 이렇게 하면 노이즈 $$x_T$$에서 데이터 $$x_0$$까지 가는 경로가 **하나의 정해진 궤적**이 됩니다. 기존 diffusion 샘플링은 이 궤적을 **여러 step으로 수치 적분**하며 따라가는 것입니다.

## 3. Consistency Model

### Consistency Function

"궤적을 한 step씩 따라가지 말고, **궤적 위의 어느 점에서든 한 번에 출발점 $$x_0$$으로** 가는 함수를 학습하자"는 것이 아이디어입니다.

$$
f: (x_t, t) \mapsto x_\epsilon
$$

($$\epsilon$$은 0에 아주 가까운 시점)

이 함수는 두 가지 성질을 만족해야 합니다.

- **Self-consistency**: 같은 PF-ODE 궤적 위의 점이라면 **어떤 $$t$$에서 넣어도 출력이 같아야** 합니다.

  $$
  f(x_t, t) = f(x_{t'}, t'),\quad \forall\, t, t' \in [\epsilon, T]
  $$

- **Boundary condition**: 출발점에서는 자기 자신을 내놓아야 합니다.

  $$
  f(x_\epsilon, \epsilon) = x_\epsilon
  $$

경계 조건은 skip connection 형태로 구조적으로 강제합니다.

$$
f_\theta(x, t) = c_{skip}(t)\,x + c_{out}(t)\,F_\theta(x, t),\qquad c_{skip}(\epsilon) = 1,\ c_{out}(\epsilon) = 0
$$

$$t = \epsilon$$이면 네트워크 출력 $$F_\theta$$가 무시되고 입력이 그대로 나옵니다.

### 학습 방법 두 가지

- **Consistency Training (CT)**: 사전학습 모델 없이, 같은 데이터에 다른 노이즈 수준을 준 두 점의 consistency function 출력 차이를 줄이도록 처음부터 학습
- **Consistency Distillation (CD)**: **사전학습된 diffusion 모델(teacher)**을 이용해 학습. LCM이 쓰는 방식입니다.

### Consistency Distillation

1. 데이터 $$x$$에 노이즈를 더해 $$x_{t_{n+1}}$$을 만듭니다.
2. Teacher diffusion 모델과 ODE solver로 **한 step만** 이동해 같은 궤적 위의 이웃 점 $$\hat{x}^\phi_{t_n}$$을 얻습니다.
3. 두 점을 consistency 모델에 넣은 출력이 **같아지도록** 학습합니다.

$$
\mathcal{L}_{CD}(\theta, \theta^-; \phi) = \mathbb{E}\Big[\lambda(t_n)\, d\big(f_\theta(x_{t_{n+1}}, t_{n+1}),\ f_{\theta^-}(\hat{x}^\phi_{t_n}, t_n)\big)\Big]
$$

| 기호 | 의미 |
|---|---|
| $$\theta$$ | 학습 중인 consistency 모델 |
| $$\theta^-$$ | $$\theta$$의 **EMA**(지수 이동 평균), target 네트워크 역할 |
| $$\phi$$ | 사전학습된 diffusion 모델(teacher) |
| $$\hat{x}^\phi_{t_n}$$ | teacher로 한 step 이동한 점 |
| $$\lambda$$ | 시점별 가중치 |
| $$d$$ | 거리 함수 (L2, LPIPS 등) |

이웃한 두 점의 출력을 같게 만드는 것을 모든 구간에서 반복하면, 결국 **궤적 전체의 출력이 $$x_0$$로 일치**하게 됩니다. DQN의 target 네트워크처럼 EMA 모델을 목표로 써서 학습을 안정시킵니다.

## 4. Latent Consistency Model

### 구조 비교

일반 diffusion은 저해상도(64) 픽셀 공간에서 여러 step denoising합니다.

![Pixel-space Diffusion](/assets/img/posts/latent-diffusion/diffusion.png)
_픽셀 공간 Diffusion_

Consistency Model은 같은 픽셀 공간에서 teacher diffusion을 **few-step 모델로 증류**합니다.

![Consistency Model](/assets/img/posts/latent-consistency-model/consistency-diffusion.png)
_Consistency Model: teacher diffusion(위)에서 student(아래)로 증류_

LDM은 512 해상도 이미지를 Encoder로 64 크기 latent로 압축한 뒤 diffusion합니다.

![Latent Diffusion](/assets/img/posts/latent-diffusion/latent-diffusion.png)
_Latent Diffusion Model_

**LCM은 두 아이디어를 합칩니다.** Stable Diffusion(LDM)의 latent 공간에서 consistency distillation을 수행합니다.

![Latent Consistency Model](/assets/img/posts/latent-consistency-model/latent-consistency-model.png)
_Latent Consistency Model: latent 공간에서 teacher LDM을 few-step 모델로 증류_

픽셀 대신 작은 latent에서 증류하므로 학습이 가볍고, 원래 Consistency Model이 다루기 어려웠던 **512~768 고해상도 텍스트 조건 생성**이 가능해집니다.

### Classifier-Free Guidance (CFG) 복습

텍스트 조건 생성에서는 품질을 위해 **CFG**를 씁니다. 조건이 있는 예측과 없는 예측을 섞어, **조건 쪽에 더 큰 가중치**를 줍니다.

$$
\tilde{\epsilon}_\theta(z_t, \omega, c, t) = (1 + \omega)\,\epsilon_\theta(z_t, c, t) - \omega\,\epsilon_\theta(z_t, \varnothing, t)
$$

- $$\epsilon_\theta(z_t, c, t)$$: 텍스트 조건 $$c$$가 있는 노이즈 예측
- $$\epsilon_\theta(z_t, \varnothing, t)$$: 조건 없는 노이즈 예측
- $$\omega$$: guidance scale (클수록 프롬프트를 강하게 따름)

> 이 식은 LCM 논문의 표기입니다. `diffusers` 등에서 흔히 쓰는 $$\epsilon_\varnothing + s(\epsilon_c - \epsilon_\varnothing)$$ 형태로 바꾸면 $$s = 1 + \omega$$입니다. 그래서 라이브러리의 `guidance_scale` 값과 논문의 $$\omega$$는 1만큼 차이 날 수 있습니다.
{: .prompt-info }

문제는 CFG를 쓰면 매 step마다 **모델을 두 번** 돌려야 한다는 것입니다.

### One-Stage Guided Distillation

LCM은 CFG가 적용된 PF-ODE 자체를 증류합니다. Teacher가 한 step 이동할 때 **CFG를 적용한 노이즈 예측**을 쓰고, guidance scale $$\omega$$를 student의 **입력 조건**으로 넣습니다.

$$
f_\theta: (z_t, \omega, c, t) \mapsto z_0
$$

그래서 추론할 때는 CFG를 위해 두 번 돌릴 필요 없이 **한 번의 forward로 guidance가 반영된 결과**가 나옵니다.

### Skipping-Step

이웃한 step($$t_{n+1} \to t_n$$)끼리만 consistency를 맞추면, 1000 step짜리 궤적 전체에 정보가 퍼지는 데 오래 걸립니다. LCM은 **$$k$$ step 떨어진 점**($$t_{n+k} \to t_n$$, 예: $$k = 20$$)끼리 맞춰 증류 속도를 크게 높입니다.

### 효율

이 기법들 덕분에 Stable Diffusion을 LCM으로 증류하는 데 **A100 GPU 약 32시간**이면 충분했다고 보고합니다. 처음부터 diffusion을 학습하는 비용에 비하면 매우 작습니다.

## 5. Multi-step 샘플링

LCM은 1 step으로도 이미지를 만들 수 있지만, 몇 step을 쓰면 품질이 좋아집니다. 방법은 **"한 번에 $$z_0$$ 예측 → 다시 노이즈 추가 → 다시 예측"**을 반복하는 것입니다.

1. $$\hat{z}_0 = f_\theta(z_{T}, \omega, c, T)$$
2. $$\hat{z}_0$$에 더 작은 시점 $$t$$만큼 노이즈를 다시 더해 $$z_t$$
3. $$\hat{z}_0 = f_\theta(z_t, \omega, c, t)$$
4. 정해진 step 수만큼 반복 후 Decoder로 이미지 복원

`diffusers` 라이브러리로 바로 써 볼 수 있습니다. (CUDA GPU가 필요하고, LCM 파이프라인은 diffusers 0.22 이후 버전에 포함되어 있습니다. 처음 실행하면 수 GB의 모델 가중치를 내려받습니다.)

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", torch_dtype=torch.float16).to("cuda")

image = pipe(
    "a photo of a cat wearing an astronaut suit, high detail",
    num_inference_steps=4,       # 일반 SD는 25~50 step
    guidance_scale=8.0,          # ω, 모델 입력으로 들어감 (두 번 돌리지 않음)
).images[0]
image.save("lcm_cat.png")
```

(여기에 직접 실행한 결과 이미지와 사용한 GPU, 생성 시간을 넣으세요. 예: `![LCM 4 step 결과](/assets/img/posts/latent-consistency-model/lcm_cat.png)`, "RTX ○○○○에서 4 step 약 ○초")

> CFG가 모델 안에 증류되어 있어서, 일반 Stable Diffusion처럼 매 step 조건 없는 예측을 따로 계산하지 않습니다. 그 대가로 **negative prompt가 거의 효과가 없고**, guidance 세기를 바꾸는 폭도 학습 때 넣어 준 $$\omega$$ 범위 안으로 제한됩니다. 속도와 맞바꾼 제어 자유도라고 볼 수 있습니다.
{: .prompt-warning }

## 6. 실험 결과

- LAION-5B의 고품질 부분집합(LAION-Aesthetics)으로 512, 768 해상도에서 평가
- **1, 2, 4 step**에서 DDIM, DPM-Solver, DPM-Solver++, Guided-Distill 등과 비교해 FID와 CLIP score 모두 크게 앞섰고, 특히 **1~2 step에서 차이가 큼**
- 4 step 생성 결과가 원래 Stable Diffusion의 수십 step 결과와 시각적으로 비슷한 품질
- **Latent Consistency Fine-tuning (LCF)**: 이미 증류된 LCM을, teacher 없이 특정 스타일 데이터셋(예: 포켓몬, 심슨 스타일)에 맞춰 바로 fine-tuning하는 방법도 제안

이후 LCM 증류를 LoRA 가중치로 만들어 다른 Stable Diffusion 모델에 끼워 쓰는 **LCM-LoRA**로 발전해, 실시간 이미지 생성 도구에 널리 쓰였습니다.

## 7. 정리

| 개념 | 핵심 |
|---|---|
| PF-ODE | 노이즈 → 데이터로 가는 결정적인 궤적 |
| Consistency function | 궤적 위 어느 점에서든 **한 번에 $$x_0$$로** |
| Self-consistency | 같은 궤적의 모든 점이 같은 출력 |
| Consistency Distillation | teacher로 한 step 이동한 점과 출력을 맞추기 (EMA target) |
| LCM | Stable Diffusion의 **latent 공간**에서 CD |
| Guided distillation | CFG를 증류해 $$\omega$$를 입력으로 → 추론 1회 |
| Skipping-step | $$k$$ step 떨어진 점끼리 맞춰 증류 가속 |

다음 Diffusion 글에서는 이미지 속에 **글자를 정확하게 그리는** [TextDiffuser]({% post_url 2026-08-05-textdiffuser %})를 다룹니다.

## References

- Luo et al., "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference," 2023. [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)
- Song et al., "Consistency Models," ICML 2023. [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
- [Hugging Face diffusers – Latent Consistency Model](https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm)
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations," ICLR 2021.
- Ho & Salimans, "Classifier-Free Diffusion Guidance," 2022.
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022.
- Luo et al., "LCM-LoRA: A Universal Stable-Diffusion Acceleration Module," 2023.
