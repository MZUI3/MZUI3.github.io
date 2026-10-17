---
title: "[Generative AI] Latent Diffusion Model(Stable Diffusion) 이해하기"
date: 2026-07-11 11:56:00 +0900
categories: [Generative AI, Diffusion]
tags: [diffusion, stable-diffusion, ldm, latent-space, cross-attention, generative-ai]
description: 픽셀 공간 diffusion이 비싼 이유와, Autoencoder로 이미지를 latent로 압축한 뒤 그 안에서 diffusion을 돌리는 Latent Diffusion Model(Stable Diffusion)의 구조, cross-attention 조건 주입, downsampling factor의 트레이드오프를 정리합니다.
math: true
---

> 생성형 AI 시리즈의 첫 글입니다. Stable Diffusion의 기반이 된 **LDM(Rombach et al., CVPR 2022)**을 정리합니다. 다음 글에서는 LDM을 몇 step 만에 샘플링하도록 만든 **Latent Consistency Model(LCM)**로 이어갑니다.

## 1. Diffusion Model 30초 복습

Diffusion Model은 두 과정으로 이루어집니다.

- **Forward process**: 이미지 $$x_0$$에 가우시안 노이즈를 조금씩 $$T$$번 더해서 완전한 노이즈 $$x_T$$로 만든다. 학습할 것이 없는 고정된 과정입니다.
- **Reverse process**: 노이즈에서 시작해 한 step씩 노이즈를 걷어내며 이미지를 복원한다. 이걸 신경망(보통 **U-Net**)이 학습합니다.

실제로 학습하는 것은 "이 step에서 더해진 노이즈가 무엇인지" 맞히는 것입니다.

$$
L_{DM} = \mathbb{E}_{x,\ \epsilon \sim \mathcal{N}(0,1),\ t}\left[\ \lVert \epsilon - \epsilon_\theta(x_t, t) \rVert_2^2\ \right]
$$

GLIDE, Imagen, DALL·E 2 등이 이 방식으로 뛰어난 이미지를 만들었지만 큰 문제가 하나 있었습니다.

## 2. 문제: 픽셀 공간은 너무 비싸다

Diffusion은 **수백~수천 step 동안 U-Net을 반복 실행**합니다. 그런데 이 연산을 **픽셀 단위**로 하면

- 512×512×3 이미지면 한 step마다 약 78만 개의 값을 처리해야 하고
- 학습에 GPU 수백 일이 걸리며
- 추론도 느립니다

그래서 기존 모델들은 저해상도(예: 64×64)에서 diffusion을 돌리고 별도의 super-resolution 모델로 키우는 방식을 많이 썼습니다.

![Pixel-space Diffusion](/assets/img/posts/latent-diffusion/diffusion.png)
_픽셀 공간 Diffusion: 저해상도(64) 이미지에서 직접 denoising_

더 근본적인 문제도 있습니다. 이미지의 bit 중 상당수는 **사람이 거의 알아채지 못하는 고주파 디테일**입니다. 픽셀 공간 diffusion은 이런 의미 없는 디테일까지 모델링하느라 연산을 낭비합니다.

## 3. 아이디어: 압축은 Autoencoder에, 생성은 Diffusion에

LDM은 학습을 두 단계로 나눕니다.

| 단계 | 담당 | 역할 |
|---|---|---|
| **Perceptual Compression** | Autoencoder | 사람이 못 알아채는 디테일을 버리고 이미지를 작은 latent로 압축 |
| **Semantic Compression** | Diffusion Model | 압축된 latent 공간에서 "무엇을 그릴지"라는 의미 구조를 생성 |

![Latent Diffusion](/assets/img/posts/latent-diffusion/latent-diffusion.png)
_LDM: 512 해상도 이미지를 Encoder로 64 크기 latent로 압축 → latent에서 Diffusion → Decoder로 512 복원_

Autoencoder는 **한 번만 학습해 두면** 여러 diffusion 모델에 재사용할 수 있습니다.

## 4. Stage 1: Perceptual Image Compression

Encoder $$\mathcal{E}$$는 이미지 $$x \in \mathbb{R}^{H\times W\times 3}$$를 latent $$z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$$로 압축하고, Decoder $$\mathcal{D}$$가 다시 복원합니다.

$$
f = \frac{H}{h} \in \{1, 2, 4, 8, 16, 32\}
$$

이 **downsampling factor $$f$$**가 LDM의 핵심 hyper-parameter입니다.

### 흐릿해지지 않게 압축하기

L1/L2 같은 픽셀 단위 loss만 쓰면 복원 이미지가 **흐릿(blurry)**해집니다. 평균적으로 오차가 작은 답이 "흐릿한 평균 이미지"이기 때문입니다. 그래서 LDM의 Autoencoder는 다음을 함께 씁니다.

- **Perceptual loss (LPIPS)**: 픽셀이 아닌 사전학습된 네트워크의 feature 공간에서 비교
- **Patch-based adversarial loss**: 작은 패치 단위로 진짜/가짜를 구분하는 discriminator를 붙여 국소적인 사실감(local realism)을 강화

latent 공간이 너무 제멋대로 퍼지지 않도록 약한 KL 정규화(KL-reg) 또는 vector quantization(VQ-reg)도 함께 사용합니다.

### 2D 구조를 유지한다

이전 연구(예: VQGAN + Transformer)는 latent를 1D 시퀀스로 펼쳐서 썼지만, LDM은 latent를 **2D feature map 그대로** 둡니다. 그래서 diffusion에서 **U-Net의 convolution이 가진 inductive bias**(공간적 국소성)를 그대로 활용할 수 있고, 압축률을 크게 높이지 않아도 품질이 유지됩니다.

## 5. Stage 2: Latent 공간에서의 Diffusion

목적 함수는 픽셀 $$x_t$$ 대신 latent $$z_t$$를 쓴다는 것 말고는 그대로입니다.

$$
L_{LDM} = \mathbb{E}_{\mathcal{E}(x),\ \epsilon \sim \mathcal{N}(0,1),\ t}\left[\ \lVert \epsilon - \epsilon_\theta(z_t, t) \rVert_2^2\ \right]
$$

| 기호 | 의미 |
|---|---|
| $$\mathcal{E}(x)$$ | 입력 이미지의 latent |
| $$\epsilon$$ | 정규분포 노이즈 |
| $$\epsilon_\theta$$ | U-Net이 예측한 노이즈 |
| $$z_t$$ | 시점 $$t$$의 noisy latent |

$$f=8$$이면 512×512 이미지가 64×64 latent가 되므로 U-Net이 처리하는 공간 크기가 **1/64**로 줄어듭니다. Stable Diffusion의 latent는 채널이 4개라, 전체 값의 개수로 보면 $$512 \times 512 \times 3 = 786{,}432$$개에서 $$64 \times 64 \times 4 = 16{,}384$$개로 **약 1/48**입니다. 생성이 끝나면 Decoder를 **딱 한 번** 통과시켜 이미지로 바꿉니다.

<details markdown="1">
<summary>더 알아보기: Stable Diffusion 코드에 나오는 0.18215는 무엇인가</summary>

KL 정규화를 약하게 건 Autoencoder의 latent는 분산이 1이 아닙니다. 그런데 diffusion은 데이터가 대략 표준정규분포 크기라고 가정하고 노이즈 스케줄을 짭니다. 그래서 LDM 논문은 학습 데이터의 latent 표준편차로 나눠 분산을 1 근처로 맞추는데, Stable Diffusion v1에서 이 값이 `1 / 0.18215`입니다. `diffusers`의 `vae.config.scaling_factor`가 바로 이 값이고, Encoder 출력에는 곱하고 Decoder 입력 전에는 나눕니다. 이 상수를 빼먹으면 생성 이미지가 엉망이 되는 것이 흔한 실수입니다.
</details>

## 6. Conditioning: Cross-Attention으로 텍스트를 넣기

"우주복을 입은 고양이"처럼 조건을 주려면 U-Net이 텍스트를 볼 수 있어야 합니다. LDM은 U-Net 중간 층에 **Cross-Attention**을 넣습니다.

$$
Q = W_Q\,\varphi(z_t),\qquad K = W_K\,\tau_\theta(y),\qquad V = W_V\,\tau_\theta(y)
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

- **Query**: U-Net의 중간 feature(이미지 쪽)
- **Key, Value**: 조건 $$y$$를 인코더 $$\tau_\theta$$로 변환한 것(텍스트 쪽)

[Transformer 글]({% post_url 2026-06-14-attention-is-all-you-need %})의 Self-Attention과 수식은 같지만, **Q와 K/V가 서로 다른 출처**라는 점만 다릅니다. 이미지의 각 위치가 "텍스트의 어떤 단어와 관련 있는지" 보고 그 정보를 가져오는 구조입니다.

$$\tau_\theta$$를 바꾸면 텍스트뿐 아니라 **semantic map, layout, 저해상도 이미지** 등 다양한 조건을 같은 방식으로 넣을 수 있다는 것이 큰 장점입니다. (LDM 논문의 text-to-image 모델은 $$\tau_\theta$$로 Transformer를 처음부터 함께 학습했고, 이후 Stable Diffusion v1은 이를 **고정된(frozen) CLIP ViT-L/14 텍스트 인코더**로 바꿨습니다.)

```python
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner, bias=False)     # 이미지 feature
        self.to_k = nn.Linear(context_dim, inner, bias=False)   # 텍스트 임베딩
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, query_dim)

    def forward(self, x, context):
        # x: (B, H*W, query_dim), context: (B, L_text, context_dim)
        B, N, _ = x.shape
        h = self.heads
        q = self.to_q(x).view(B, N, h, -1).transpose(1, 2)
        k = self.to_k(context).view(B, context.size(1), h, -1).transpose(1, 2)
        v = self.to_v(context).view(B, context.size(1), h, -1).transpose(1, 2)
        out = nn.functional.scaled_dot_product_attention(q, k, v)   # PyTorch 2.0+
        return self.to_out(out.transpose(1, 2).reshape(B, N, -1))

import torch

attn = CrossAttention(query_dim=320, context_dim=768)   # SD v1 첫 단계 채널 320, CLIP 임베딩 768
x = torch.randn(2, 64 * 64, 320)                        # 64x64 latent feature를 펼친 것
ctx = torch.randn(2, 77, 768)                           # 토큰 77개짜리 텍스트 임베딩
print(attn(x, ctx).shape)                               # torch.Size([2, 4096, 320])
```

출력 크기는 이미지 쪽(Query)과 같고, 텍스트 길이(77)는 attention 안에서만 쓰입니다. 이미지의 4096개 위치가 각각 77개 토큰 중 어디를 참고할지 고르는 구조입니다.

## 7. 평가 지표

생성 모델은 "정답"이 없어서 평가 지표를 따로 알아둬야 합니다.

| 지표 | 무엇을 보나 | 좋은 방향 |
|---|---|---|
| **IS** (Inception Score) | InceptionV3의 분류 확률로 이미지의 **품질(선명도)**과 **다양성** 평가 | 높을수록 |
| **FID** (Fréchet Inception Distance) | 진짜/가짜 이미지 feature 분포 사이의 거리 | 낮을수록 |
| **LPIPS** | 사전학습 네트워크(VGG 등) feature 기준의 지각적 거리 | 용도에 따라 |
| **PSNR / SSIM** | 픽셀 단위 복원 정확도 (super-resolution, inpainting 등) | 높을수록 |

이 중 **FID**가 가장 널리 쓰입니다. IS는 진짜 데이터와 비교하지 않는다는 한계가 있습니다.

## 8. 실험: Downsampling Factor의 trade-off

논문은 $$f \in \{1, 2, 4, 8, 16, 32\}$$로 바꿔가며 LDM-$$f$$를 비교합니다.

- **$$f$$가 작을 때 (LDM-1, 2)**: 사실상 픽셀 공간과 같아서 **학습이 매우 느림**
- **$$f$$가 너무 클 때 (LDM-32)**: 압축 과정에서 정보가 너무 많이 사라져 **품질이 일찍 정체됨**
- **$$f = 4 \sim 16$$**: 효율과 품질의 균형이 가장 좋음

즉 **학습 속도와 샘플 품질은 $$f$$에 대해 trade-off 관계**이고, 그 균형점을 찾은 것이 LDM의 성과입니다. 이 결과로 LDM은 text-to-image, inpainting, super-resolution, class-conditional 생성에서 픽셀 기반 모델과 비슷하거나 더 좋은 품질을 **훨씬 적은 연산으로** 달성했습니다.

## 9. 정리

| 키워드 | 핵심 |
|---|---|
| 문제 | 픽셀 공간 diffusion은 고주파 디테일까지 모델링해서 비쌈 |
| 해법 | Autoencoder로 perceptual 압축 → latent에서 diffusion |
| Autoencoder | Perceptual loss + Patch GAN으로 blur 없이 압축, 2D 구조 유지 |
| Conditioning | U-Net 안의 Cross-Attention, 조건 인코더 교체로 다양한 조건 처리 |
| Trade-off | Downsampling factor $$f$$: 작으면 느리고, 크면 품질 정체 |

LDM 덕분에 diffusion 연산량은 크게 줄었지만, **sampling step이 수십 번 필요하다**는 문제는 남았습니다. 다음 글에서는 이를 **1~4 step**까지 줄인 **Latent Consistency Model(LCM)**을 다룹니다.

## References

- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium," NeurIPS 2017. (FID)
- Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020.
- Esser et al., "Taming Transformers for High-Resolution Image Synthesis," CVPR 2021.
