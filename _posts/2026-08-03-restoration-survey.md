---
title: "[Computer Vision] Diffusion Models for Image Restoration — 지도 학습과 제로샷, 두 흐름으로 본 디퓨전 기반 영상 복원"
date: 2026-08-03 13:30:00 +0900
categories: [Computer Vision, Image Restoration]
tags: [image-restoration, diffusion-models, survey, super-resolution, deblurring, inpainting, zero-shot, perception-distortion]
description: Li, Ren 등(2023)의 디퓨전 기반 영상 복원 서베이를 정리합니다. SR3로 대표되는 지도 학습 디퓨전과 ILVR로 대표되는 제로샷 디퓨전이라는 두 작업 흐름, 조건 전략(저화질 참조, 전처리 참조, 디퓨전 과정 수정)과 생성 공간(이미지·잔차·잠재), 블라인드/실세계 복원을 위한 왜곡 시뮬레이션·커널 추정·도메인 변환·왜곡 불변 학습, 초해상도·디블러링·인페인팅의 객관 비교, 그리고 샘플링 효율과 모델 압축 등 다섯 과제를 다룹니다. 작은 실험으로 지각-왜곡 트레이드오프를 확인합니다.
math: true
---

> [MWCNN]({% post_url 2026-07-23-mwcnn %})처럼 CNN으로 영상을 복원하면 PSNR은 오르지만 질감은 흐릿합니다. 픽셀 손실(MSE, L1)이 가능한 정답들의 **평균**을 내기 때문이고, SRGAN·ESRGAN은 적대 손실로 질감을 살렸지만 학습이 불안정하고 생성된 질감이 가짜처럼 보이는 문제가 남았습니다. 디퓨전 모델이 생성 품질에서 GAN을 넘자 "디퓨전이 복원도 끌어올릴까"가 자연스러운 질문이 됐고, **Diffusion Models for Image Restoration and Enhancement: A Comprehensive Survey**(Li, Ren, Jin, Lan, Wang, Zeng, Wang, Chen, USTC·NUS·EIT·MSRA, 2023)는 그 답을 처음으로 체계화한 서베이입니다. 핵심 구분은 **지도 학습**(왜곡/정답 쌍으로 디퓨전을 처음부터 학습)과 **제로샷**(사전학습 생성 디퓨전의 사전 지식을 학습 없이 활용)입니다.

## 1. 배경

NCSN(점수 + 담금질 랑주뱅), DDPM(고정 분산 순방향 + 변분 하한), SDE(둘을 연속 시간으로 통일)의 세 기초 모델을 정리하고, 최적화 전략·샘플링 효율·구조·조건 전략 면의 개선을 훑습니다. [디퓨전 서베이]({% post_url 2026-10-17-diffusion-survey %})와 겹치므로 여기서는 복원에 중요한 점만 적습니다. 복원의 목표는 왜곡 이미지 $$y$$가 주어졌을 때 **사후 분포** $$p_\theta(x_{t-1} \mid y, x_t)$$를 배워 $$x_0 \sim q(x \mid y)$$를 뽑는 것이고, 관건은 **$$y$$를 어떻게 조건으로 넣느냐**입니다.

## 2. 두 가지 작업 흐름

### 지도 학습: SR3

SR3는 DDPM 틀에서 저해상도 $$y$$를 bicubic으로 목표 해상도로 키운 뒤 **$$x_t$$와 채널 방향으로 이어 붙여** U-Net에 넣는 가장 직접적인 조건화입니다. 질감 생성은 뛰어나지만 두 한계가 있습니다. 처음부터 학습하려면 **대량의 쌍 데이터**가 필요하고, 실세계에서 왜곡/정답 쌍을 모으기 어렵습니다.

### 제로샷: ILVR

사전학습된 무조건 디퓨전은 ImageNet·FFHQ 같은 대규모 자연 이미지로 학습되어 **구조와 질감의 저장소**로 볼 수 있습니다. ILVR은 샘플링의 매 단계에서 예측된 $$x'_{t-1}$$의 **저주파 성분을 참조 이미지의 것으로 교체**합니다.

$$
x_{t-1} = x'_{t-1} + \Phi_N(y_{t-1}) - \Phi_N(x'_{t-1})
$$

$$\Phi_N$$은 저역 통과 필터, $$y_{t-1}$$은 $$y$$를 $$t-1$$단계까지 확산시킨 것입니다. 구조는 참조에서, 고주파 질감은 생성 사전에서 옵니다. 제로샷 방법의 핵심 과제는 **왜곡 이미지의 데이터 구조를 보존하면서 지각적 사전을 끌어내는 방법**이고, 이후 방법들은 대부분 이 정제 전략을 개선합니다.

## 3. 분류

**지도 학습 디퓨전**은 조건 전략으로 (1) 저화질 참조 이미지(SR3처럼 직접 이어 붙임), (2) 전처리된 참조(다른 네트워크로 먼저 복원하거나 특징을 뽑아 조건으로, IDM·CDPMSR), (3) 디퓨전 과정 수정(저화질 이미지에서 시작하거나 순방향 자체를 왜곡 과정으로 정의)으로 나뉘고, 생성 공간으로 이미지·**잔차**(ResDiff처럼 전처리 결과와 정답의 차이만 생성)·**잠재**(SRDiff처럼 저차원에서 생성)로 나뉩니다.

**제로샷 디퓨전**은 데이터 일관성을 어떻게 강제하느냐로 나뉩니다. 저주파 교체(ILVR), 선형 역문제의 **분해**(DDRM의 SVD, DDNM의 영공간 분해로 측정과 일치하는 성분은 $$y$$에서, 나머지는 생성에서), **사후 기울기**(DPS는 $$\nabla_{x_t} \log p(y \mid x_t)$$를 근사해 매 단계 보정, GDP는 경험적 공식), **플러그 앤 플레이**(DiffPIR은 디퓨전을 디노이저 사전으로 쓰는 HQS 분할) 등입니다.

**블라인드/실세계 복원**은 왜곡이 미지이고 다양해 두 흐름 모두 어렵습니다. 네 방향이 있습니다. **왜곡 시뮬레이션**(Real-ESRGAN류의 고차 왜곡 파이프라인으로 쌍을 합성), **커널 추정**(블러 커널을 디퓨전으로 함께 추정), **도메인 변환**(실세계 왜곡을 합성 왜곡 도메인으로 옮긴 뒤 복원), **왜곡 불변 디퓨전**(왜곡에 불변인 표현에서 생성).

## 4. 코드로 보기: 지각-왜곡 트레이드오프

체커보드 질감의 깨끗한 이미지에 블러와 잡음을 더한 관측에 대해, (a) MSE로 학습한 회귀기처럼 잡음을 평균 내는 평활 추정과 (b) 질감 통계는 정확하지만 패턴이 2픽셀 어긋난 생성 표본의 PSNR·SSIM·선명도를 비교하고, 샘플러별 디노이저 평가 횟수를 셉니다.

```python
import numpy as np
rng = np.random.default_rng(0)
n = 64
yy, xx = np.mgrid[0:n, 0:n]
# a clean image with sharp edges and texture, and a noisy, slightly blurred observation
clean = ((xx // 8 + yy // 8) % 2).astype(float) * 0.6 + 0.2 + 0.05 * np.sin(xx / 2.0)
def blur(img, k=1):
    out = np.zeros_like(img)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): out += np.roll(np.roll(img, dy, 0), dx, 1)
    return out / (2 * k + 1) ** 2
degraded = blur(clean) + 0.15 * rng.standard_normal((n, n))
def psnr(a, b): mse = np.mean((a - b) ** 2); return 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
def ssim(a, b):
    mu_a, mu_b = a.mean(), b.mean(); va, vb = a.var(), b.var(); cov = np.mean((a - mu_a) * (b - mu_b))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (2 * mu_a * mu_b + c1) * (2 * cov + c2) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
def sharpness(a): return np.mean(np.abs(np.diff(a, axis=1))) + np.mean(np.abs(np.diff(a, axis=0)))
# two "restorations": an MSE-trained regressor (averages the noise away, blurs edges) and a generative sample
# (sharp, right texture statistics, but the pattern is placed 2 px off: pixel-wise it is wrong)
mse_estimate = blur(degraded, 1)
sample = np.roll(clean, 2, 1) + 0.03 * rng.standard_normal((n, n))
print("perception-distortion trade-off on a toy image")
print("candidate                 PSNR(dB)   SSIM    sharpness (edge energy)")
for name, img in (("degraded input", degraded), ("MSE-trained estimate", mse_estimate), ("generative sample (2px off)", sample), ("ground truth", clean)):
    print(f"{name:<26} {psnr(img, clean):7.2f}   {ssim(img, clean):.3f}   {sharpness(img):.3f}")
print("-> the blurry estimate wins PSNR/SSIM; the sharp sample keeps real edges but is punished pixel-wise")
print()
for name, steps in (("DDPM sampling", 1000), ("DDIM", 50), ("DPM-Solver", 15), ("one-step distilled", 1)):
    print(f"{name:<20} {steps:>5} denoiser evaluations per restored image")
```

```text
perception-distortion trade-off on a toy image
candidate                 PSNR(dB)   SSIM    sharpness (edge energy)
degraded input               13.89   0.758   0.378
MSE-trained estimate         15.93   0.812   0.141
generative sample (2px off)   10.36   0.501   0.205
ground truth                   inf   1.000   0.147
-> the blurry estimate wins PSNR/SSIM; the sharp sample keeps real edges but is punished pixel-wise

DDPM sampling         1000 denoiser evaluations per restored image
DDIM                    50 denoiser evaluations per restored image
DPM-Solver              15 denoiser evaluations per restored image
one-step distilled       1 denoiser evaluations per restored image
```

- 평활 추정은 PSNR 15.9dB, SSIM 0.81로 가장 좋고, 가장자리 에너지도 0.141로 정답(0.147)과 비슷하게 나옵니다. 인접 픽셀 차이의 평균인 이 지표는 흐려진 경계와 남은 잡음을 구분하지 못해, 이 toy에서는 흐림을 잡아내지 못합니다. 생성 표본은 가장자리 에너지가 0.205로 정답보다 오히려 크고, 2픽셀 어긋났다는 이유로 PSNR이 10.4dB, 즉 **왜곡된 입력보다도 낮습니다**. PSNR·SSIM이 픽셀 정렬을 요구하는 왜곡 지표이고, 지각 품질과 상충한다는 Blau & Michaeli의 트레이드오프가 가장 단순한 형태로 보입니다. 서베이가 FID·LPIPS 같은 지각 지표를 함께 보고하는 이유입니다.
- DDPM 1,000회에서 DDIM 50회, DPM-Solver 15회, 증류 1회로 줄어드는 디노이저 평가 횟수는 6.1절 "샘플링 효율"의 수치입니다. SR3가 224×224 한 장에 약 50초 걸리는 것이 1,000회 평가 때문입니다.

## 5. 실험 비교

데이터셋(DIV2K, Urban100, ImageNet 1K, CelebA 1K 등), 구현 세부, 지표(PSNR, SSIM, FID, LPIPS, 실행 시간, 파라미터, FLOPs)를 정리한 뒤 공개 구현을 직접 돌려 비교합니다.

**4배 초해상도, 지도 학습.** ResDiff가 PSNR·SSIM에서 가장 좋고, PSNR은 다른 디퓨전보다 약 0.5dB 높습니다. 잔차를 생성하고 전처리 이미지를 조건으로 써서 픽셀 수준 일관성이 보장되기 때문입니다. IDM과 CDPMSR은 전처리 특징을 조건으로 써 지각 지표가 좋습니다. SRDiff는 잠재 공간에서 100스텝만 돌려 한 장 43.5초, 84 GFLOPs, 13.2M 파라미터로 IDM·SR3보다 훨씬 가볍습니다.

**4배 초해상도, 제로샷.** DDRM과 DDNM이 여러 지표에서 좋고 ILVR이 뒤따릅니다. 분해 관점의 데이터 일관성이 저주파 일관성보다 강하기 때문입니다. DPS와 GDP는 지각 지표에 집중하며, GDP는 25스텝으로 가장 빠르고 DPS는 매 스텝 보정 때문에 DDIM 가속을 못 써 한 장에 약 141초입니다.

**가우시안 디블러링.** DiffPIR과 DiracDiffusion(왜곡 최적화 버전)이 DDRM·DDNM보다 PSNR 1.0~1.4dB 높고, DiracDiffusion(지각 최적화 버전)과 DiffPIR은 지각 지표도 좋습니다. DiffPIR은 플러그 앤 플레이에 DDIM을 합쳐 충실도와 사실성을 함께 잡고, DiracDiffusion은 점진적 복원과 조기 종료로 지각-왜곡 트레이드오프를 조절합니다. DDRM은 20스텝으로 한 장 10초 미만입니다. 제로샷 방법은 모두 ImageNet 552.8M, CelebA 126M 파라미터의 사전학습 모델을 씁니다.

**인페인팅(좁은 마스크).** 다섯 제로샷 방법을 비교하며, 경향은 비슷합니다.

## 6. 다섯 가지 과제

1. **샘플링 효율.** DDIM, ODE 솔버, 증류, 조건 사전으로 10~20스텝까지 줄었고 DDRM은 8초이지만 실시간에는 멀었습니다. 복원 특유의 해법은 **순수 잡음 대신 저화질 이미지에서 샘플링을 시작**하는 것으로, 저화질 이미지에 이미 풍부한 구조가 있다는 점을 이용합니다.
2. **모델 압축.** DDPM 113.7M, SR3 155.3M 파라미터는 CNN·Transformer 복원 백본보다 훨씬 큽니다. 블록 제거 증류, 스텝별 중요도 기반 가지치기, 양자화가 디퓨전 일반에서 시작됐지만 복원 특화 압축은 거의 없습니다.
3. **왜곡 시뮬레이션과 추정.** 실세계 쌍을 모을 수 없으므로 시뮬레이션의 현실성과 왜곡 추정의 정확성이 관건입니다.
4. **왜곡 불변 학습.** 왜곡 종류와 무관한 표현에서 복원.
5. **프레임워크 설계.** 조건화와 생성 공간의 더 나은 조합.

## 7. 해석

**1. "지도 대 제로샷"은 데이터와 사전의 교환입니다.** 지도 학습은 쌍 데이터로 과제 특화 성능을 얻지만 왜곡이 바뀌면 다시 학습해야 하고, 제로샷은 하나의 생성 사전으로 여러 선형 역문제를 풀지만 **왜곡 모델을 알아야** 하며(DDRM의 SVD는 왜곡 연산자가 필요) 느립니다. 블라인드 복원이 어려운 이유는 두 흐름의 전제가 모두 깨지기 때문입니다.

**2. 평가 지표 문제가 복원에서 가장 첨예합니다.** 위 toy가 보여 주듯 PSNR이 높은 결과와 좋아 보이는 결과가 다르고, 서베이의 표들도 PSNR 순위와 LPIPS·FID 순위가 자주 엇갈립니다. [Perceptual Losses]({% post_url 2026-08-09-perceptual-losses %})가 "PSNR이 낮지만 더 좋다"고 했던 주장이 디퓨전에서는 설계 선택(DiracDiffusion의 PO/DO)으로 명시화됩니다. 어느 지표를 쓸지는 응용(의료 영상이면 왜곡, 사진이면 지각)이 정합니다.

**3. 복원 특화 가속의 방향이 분명합니다.** 생성과 달리 복원은 **출발점이 잡음일 필요가 없습니다.** 저화질 이미지에서 짧은 역방향만 도는 방법, 잔차만 생성하는 방법, 잠재 공간 방법은 모두 "복원은 생성보다 쉬운 문제"라는 점을 이용하며, [잠재 디퓨전]({% post_url 2026-07-03-latent-diffusion-models %})의 교훈이 복원에서 더 강하게 적용됩니다.

**4. 서베이의 실측 비교가 가치 있습니다.** 공개 구현을 같은 데이터셋에서 직접 돌려 PSNR·SSIM·FID·LPIPS·시간·FLOPs를 함께 보고한 것은 논문마다 다른 설정으로 보고된 수치를 그대로 옮긴 서베이보다 신뢰할 만합니다. 제로샷 방법의 비교에서 DPS 141초 대 DDRM 10초 같은 비용 차이가 드러나는 것도 그 덕입니다.

**5. 평가를 읽을 때 주의할 점.** 2023년 8월 기준이라 이후의 StableSR·DiffBIR·SUPIR 같은 잠재 디퓨전 기반 블라인드 복원과 일관성 모델 기반 가속은 얕게 다뤄지거나 빠져 있습니다. 비교는 공개 구현에 한정되고, 디블러링 표의 참조 번호가 본문에서 누락된 곳도 있습니다. 위 toy의 "생성 표본"은 손으로 2픽셀 어긋나게 만든 것으로 실제 디퓨전 표본의 오류 유형과 같지는 않습니다.

## 8. 정리

| 항목 | 내용 |
|---|---|
| 두 흐름 | 지도 학습(SR3: 쌍 데이터로 처음부터, 조건 = 저화질 이미지 연결) / 제로샷(ILVR: 사전학습 생성 사전 + 저주파 교체) |
| 지도 분류 | 조건: 저화질 참조·전처리 참조·과정 수정 / 공간: 이미지·잔차·잠재 |
| 제로샷 분류 | 저주파 교체, 분해(DDRM·DDNM), 사후 기울기(DPS·GDP), 플러그 앤 플레이(DiffPIR) |
| 블라인드 | 왜곡 시뮬레이션, 커널 추정, 도메인 변환, 왜곡 불변 학습 |
| 실측 | SR: ResDiff(PSNR), IDM·CDPMSR(지각), SRDiff(경량); 제로샷 SR: DDRM·DDNM; 디블러: DiffPIR·DiracDiffusion |
| 과제 | 샘플링 효율(SR3 50초/장), 모델 압축(SR3 155M), 왜곡 시뮬레이션·추정, 왜곡 불변, 프레임워크 |

## References

- Li et al., "Diffusion Models for Image Restoration and Enhancement – A Comprehensive Survey," 2023. [arXiv:2308.09388](https://arxiv.org/abs/2308.09388)
- Saharia et al., "Image Super-Resolution via Iterative Refinement" (SR3), TPAMI 2022. [arXiv:2104.07636](https://arxiv.org/abs/2104.07636)
- Choi et al., "ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models," ICCV 2021. [arXiv:2108.02938](https://arxiv.org/abs/2108.02938)
- Kawar, Elad, Ermon, Song, "Denoising Diffusion Restoration Models" (DDRM), NeurIPS 2022. [arXiv:2201.11793](https://arxiv.org/abs/2201.11793)
- Wang, Yu, Zhang, "Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model" (DDNM), ICLR 2023. [arXiv:2212.00490](https://arxiv.org/abs/2212.00490)
- Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems" (DPS), ICLR 2023. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)
- Zhu et al., "Denoising Diffusion Models for Plug-and-Play Image Restoration" (DiffPIR), CVPRW 2023. [arXiv:2305.08995](https://arxiv.org/abs/2305.08995)
- Blau, Michaeli, "The Perception-Distortion Tradeoff," CVPR 2018. [arXiv:1711.06077](https://arxiv.org/abs/1711.06077)
