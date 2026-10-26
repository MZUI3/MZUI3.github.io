---
title: "[Face Swapping] StyleIPSB: 신원을 보존하는 StyleGAN 의미 기저로 모공 수준 세부를 지키는 얼굴 교체"
date: 2026-10-18 14:19:00 +0900
categories: [Generative AI, Face Swapping]
tags: [face-swapping, stylegan, semantic-basis, identity-preserving, 3dmm, hessian, w-plus-subspace, pore-level-detail, cvpr]
description: StyleIPSB(Jiang, Song, Tong, Tang, CVPR 2023)가 StyleGAN 기반 교체의 흐릿함(회귀된 스타일 코드가 W+ 밖으로 나감)과 신원 이동(비매끄러운 신원 임베딩)을 함께 풀기 위해, 포즈·표정·조명 속성 변화를 신원 손실로 나눈 거리 메트릭의 헤시안을 분해해 신원을 보존하는 직교 의미 기저를 만들고, 3DMM 파라미터를 기저 좌표로 매핑하는 네트워크, 잔차 속성 전이 MLP, 마스크 공간 특징 변환으로 배경을 섞는 3단계 교체를 정리합니다. 기저의 표현력·분리·세부 보존, GANSpace·InterFaceGAN과의 비교를 다루고 작은 실험으로 신원 인식 기저와 PCA형 기저의 차이를 확인합니다.
math: true
---

> StyleGAN을 디코더로 쓰면 모공 수준 세부가 가능해야 하는데, [MegaFS]({% post_url 2026-09-06-megafs %})·[FSLSD]({% post_url 2026-09-15-fslsd %}) 같은 방법의 결과는 **흐리고** 어려운 경우 **신원이 밀립니다**(Figure 1). **StyleIPSB**(Jiang, Song, Tong, Tang, 저장대·톈진대, CVPR 2023)의 진단은 둘입니다. 흐림은 인코더가 회귀한 스타일 코드가 **$$\mathcal{W}+$$ 공간 밖**으로 나가기 때문이고, 신원 이동은 신원 임베딩이 **비매끄러운 공간**이라([Smooth-Swap]({% post_url 2026-10-07-smooth-swap %})) 신원을 보존하는 최적화 방향을 찾기 어렵기 때문입니다. 해법은 회귀를 **신원 보존 의미 기저(StyleIPSB)**로 제약하는 것입니다. 기저는 $$\mathcal{W}+$$의 선형 부분 공간이라 그 안의 코드는 선명하게 렌더링되고, 좌표를 바꿔도 신원이 유지되며, 다양한 포즈·표정·조명을 표현합니다. 3DMM이 잡는 큰 의미 변화를 기저 좌표로 번역하는 매핑망이 둘을 잇습니다.

## 1. StyleIPSB 구성

기저가 만족해야 할 세 조건은 **$$\mathcal{W}+$$ 부분 공간**, **신원 보존**, **표현력**입니다. GANSpace·SeFa는 신원을 고려하지 않고, InterFaceGAN은 분류기에 묶여 다양한 표정·조명을 못 다루며, GIF는 $$\mathcal{W}+$$ 밖으로 나가 왜곡됩니다(표 1).

두 스타일 코드 $$w_1, w_2$$에 대해 3DMM 피팅 네트워크 $$M$$(DECA)의 포즈·표정·조명 파라미터 차이를 신원 손실로 **나눈** 거리를 정의합니다.

$$
D_p(w_1, w_2) = \frac{\lVert M(G(w_1))_p - M(G(w_2))_p \rVert^2}{L_{id}(G(w_1), G(w_2))}, \qquad L_{id} = 1 - \cos(ID(I_1), ID(I_2))
$$

표정 $$D_e$$, 조명 $$D_i$$도 같습니다. 속성 차이가 크고 신원 변화가 작을수록 커지는 메트릭이므로, 이 메트릭이 **가장 빨리 변하는 방향**이 원하는 기저입니다. 이미지 다양체의 국소 거리를 2차 근사한 **헤시안** $$D^2(w_0, w_0 + \delta w) \approx \delta w^\top H(w_0)\delta w$$를 분해해 상위 $$m$$개 고유벡터를 취합니다. StyleGAN의 층별 의미에 따라 포즈 기저는 앞 3층, 표정은 4~10층, 조명은 11층 이후에서 만들고, $$w_0$$를 100번 샘플해 평균한 뒤 Gram-Schmidt로 직교화해 $$V = [V_p, V_e, V_i]$$를 얻습니다.

## 2. 3단계 교체

**1단계: 3DMM-StyleGAN 매핑.** 소스·목표의 3DMM 파라미터 $$\alpha_s, \alpha_t$$를 6층 MLP 세 개가 포즈·표정·조명의 기저 좌표 $$\beta_{s \to t}$$로 바꾸고, StyleGAN 인코더가 만든 소스 코드에 더합니다: $$w_{s \to t} = w_s + \beta_{s \to t} V$$. 손실은 신원 $$L_{id}$$와 속성 $$L_{attr}$$(전이 결과의 3D 기하가 목표 표정·포즈와 같은지의 메시 L2 + 같은 형상·알베도로 렌더링한 이미지의 L1). 형상 파라미터는 소스 것으로 고정해 속성 차이만 비교합니다.

**2단계: 세부 속성 전이.** 3DMM이 못 잡는 세부 속성(머리카락 등)을 위해 목표 코드 $$w_t$$와 $$w_{s \to t}$$에서 DAT-MLP가 $$\lambda \in [0,1]^{18 \times 512}$$를 회귀해 $$w'_{s \to t} = \lambda w_{s \to t} + (1 - \lambda) w_t$$로 섞습니다. 두 코드의 속성이 가까우므로 그 사이의 상자는 대부분 $$\mathcal{W}+$$ 부분 공간 안에 있어 세부가 유지된다고 봅니다. 손실은 신원 + 목표와의 지각 + 속성.

**3단계: 블렌딩.** VGGFace 특징을 StyleGAN 특징에 **마스크 공간 특징 변환(MSFT)**으로 섞습니다. 가우시안 블러한 얼굴 마스크 안은 StyleGAN 특징을, 밖은 VGG 특징과 섞은 것을 쓰며, 배경 손실 $$\lVert M \odot (I_r - I_t) \rVert$$과 지각 손실로 학습합니다. 세 단계를 따로 사전학습(학습률 2e-4)한 뒤 함께 학습(2e-5)합니다. FFHQ, 3090 1장.

## 3. 코드로 보기: 신원 인식 기저 대 PCA형 기저

12차원 toy 잠재 공간에 숨겨진 선형 포즈 맵과 신원 맵을 두고(두 축은 신원 중립), 포즈 변화량을 신원 변화량으로 나눈 메트릭이 가장 빨리 커지는 방향을 일반화 고유문제로 구한 기저, 포즈 변화만 최대화하는 PCA형 기저, 무작위 방향을 각각 1단위 움직여 포즈 변화와 신원 손실을 비교하고, 목표 3DMM 포즈를 기저 좌표로 회귀해 전이합니다.

```python
import numpy as np
rng = np.random.default_rng(0)
# a toy latent space w in R^12: a (hidden) linear map gives the 3DMM attribute (pose) and another gives the identity embedding
D = 12
A_pose = rng.standard_normal((2, D)); A_id = rng.standard_normal((4, D))
A_id[:, :2] = 0.0                                              # make the first two latent axes identity-neutral so a clean solution exists
def pose(w): return A_pose @ w
def id_loss(w1, w2): a, b = A_id @ w1, A_id @ w2; return 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
w0 = rng.standard_normal(D)
# StyleIPSB metric D_p = ||pose change||^2 / L_id. Near w0 both numerator and denominator are quadratic forms in dw, so the
# directions along which D_p grows fastest are the generalised eigenvectors of (A_pose^T A_pose) w.r.t. (A_id^T A_id + eps I):
# maximal pose change per unit identity change. (The paper takes the Hessian of the metric; this is its quadratic-form equivalent.)
from scipy.linalg import eigh
Hp = A_pose.T @ A_pose; Hid = A_id.T @ A_id + 1e-2 * np.eye(D)
vals, vecs = eigh(Hp, Hid); basis = vecs[:, ::-1][:, :2]
# compare with plain PCA-style directions (largest pose change regardless of identity) and random directions
U, S, Vt = np.linalg.svd(A_pose); pca_dirs = Vt[:2].T
def evaluate(dirs, name):
    step = 1.0; dp, di = [], []
    for k in range(dirs.shape[1]):
        w1 = w0 + step * dirs[:, k] / np.linalg.norm(dirs[:, k]); dp.append(np.linalg.norm(pose(w1) - pose(w0))); di.append(id_loss(w0, w1))
    print(f"  {name:<34} pose change {np.mean(dp):.2f}   identity loss {np.mean(di):.3f}")
print("moving 1 unit along two basis directions from a random w0:")
evaluate(basis, "StyleIPSB (identity-aware Hessian)"); evaluate(pca_dirs, "largest pose change only (PCA-like)"); evaluate(rng.standard_normal((D, 2)), "random directions")
# the 3DMM-StyleGAN mapping: regress basis coordinates beta from 3DMM parameters, then w_{s->t} = w_s + beta V
target_pose = pose(rng.standard_normal(D)); Bp = basis
beta = np.linalg.lstsq(A_pose @ Bp, target_pose - pose(w0), rcond=None)[0]; w_st = w0 + Bp @ beta
print(f"\nmapping 3DMM pose -> basis coordinates: residual pose error {np.linalg.norm(pose(w_st) - target_pose):.3f}, identity loss of the transferred code {id_loss(w0, w_st):.3f}")
print("-> constrained to the basis, the code stays in a W+ subspace the generator renders sharply, and the pose moves without touching identity")
```

```text
moving 1 unit along two basis directions from a random w0:
  StyleIPSB (identity-aware Hessian) pose change 2.41   identity loss 0.000
  largest pose change only (PCA-like) pose change 2.92   identity loss 0.018
  random directions                  pose change 1.48   identity loss 0.005

mapping 3DMM pose -> basis coordinates: residual pose error 0.000, identity loss of the transferred code 0.000
-> constrained to the basis, the code stays in a W+ subspace the generator renders sharply, and the pose moves without touching identity
```

- PCA형 방향은 포즈를 2.92 바꾸지만 신원 손실 0.018을 내고, 신원 인식 기저는 포즈를 2.41 바꾸면서 신원 손실이 0.000입니다. 메트릭의 분모(신원 손실)가 기저를 신원 중립 부분 공간 쪽으로 돌리는 것이고, 논문 Figure 5에서 StyleIPSB가 GANSpace보다 포즈 편집 시 신원 손실이 작고 지도 학습인 InterFaceGAN과 대등한 결과를 내는 원리입니다. 무작위 방향은 포즈도 신원도 어중간합니다.
- 목표 포즈를 기저 좌표로 회귀하면 포즈 오차 0, 신원 손실 0입니다. 기저에 갇힌 코드는 신원을 건드리지 않고 속성만 옮깁니다. 논문의 표 2(기저 있음: FID 26.1 → 22.2, 표정 3.73 → 3.37, ID 유사도 0.65 → 0.67)가 이 성질의 실제 버전입니다. 이 toy의 "헤시안"을 두 2차 형식의 일반화 고유문제로 바꾼 것은, 메트릭이 비율이라 $$\delta w = 0$$ 근방의 유한차분 헤시안이 불안정했기 때문입니다.

## 4. 실험

**기저의 성질.** StyleIPSB는 pitch·yaw 회전, 색·방향·세기가 다른 조명, 입·눈 개폐, 안구 방향, 눈썹 올리기를 표현합니다(Figure 4). 포즈 편집 시 yaw 각도 대비 신원 손실(Figure 5)은 비지도 GANSpace보다 낮고 지도 InterFaceGAN과 대등하며, 메트릭에 신원 손실을 넣지 않으면 나빠집니다. 표정(미소) 편집에서 다른 방법은 얼굴 형태가 변형되지만 StyleIPSB는 유지합니다(Figure 6). FLAME 3DMM 파라미터로 생성 이미지의 속성을 제어하는 예(Figure 7)로 3DMM과의 호환을 보입니다.

**Ablation(1단계).** 기저 없이 매핑망이 $$w_{s \to t}$$를 직접 회귀하면 모공 수준 세부가 사라지고(Figure 8) FID 26.06 대 22.15, 표정 3.73 대 3.37, ID 0.65 대 0.67로 기저가 모두에서 낫습니다. 포즈는 0.074 대 0.078로 비슷합니다.

**교체 비교.** MegaFS와 FSLSD 대비 모공 수준 세부와 신원 보존이 좋다는 것이 Figure 1의 주장입니다. FaceForensics++ 정량 비교(Table 3)에서 ID 검색은 95.05%로 FaceShifter 90.68%, MegaFS 90.83%, FSLSD 90.05%보다 높고 표정 오차도 2.23으로 가장 낮지만, 포즈 오차는 3.58로 FaceShifter(2.55)·FSLSD(2.46)보다 큽니다. CelebA-HQ(Table 4)에서도 FID 9.37, ID 유사도 0.5378로 MegaFS·RAFS·FSLSD보다 낫고 포즈만 0.078로 가장 나쁩니다. 사용자 연구는 없습니다.

## 5. 해석

**1. "선명함 = 부분 공간 안에 있기"라는 진단이 유용합니다.** 역변환·회귀가 $$\mathcal{W}+$$ 밖으로 나가면 StyleGAN은 흐릿하거나 왜곡된 이미지를 내므로, 편집을 **생성기가 아는 부분 공간**으로 제한하는 것이 세부 보존의 조건입니다. MegaFS·FSLSD가 코드를 자유롭게 조합해 흐려진 것과 대비되고, GAN 역변환 연구의 "분포 안 편집"(e4e)과 같은 문제의식입니다.

**2. 신원 보존을 메트릭의 분모로 넣은 설계.** 속성 변화를 신원 변화로 나눈 비율의 헤시안이라는 발상은, GANSpace의 PCA(변화량만), InterFaceGAN의 분류 경계(라벨 필요)와 달리 **비지도이면서 신원을 명시**합니다. 위 toy의 2.41/0.000 대 2.92/0.018이 그 차이이고, [FSLSD]({% post_url 2026-09-15-fslsd %})의 구조 전이 방향이 신원을 건드릴 수 있다는 문제를 기저 수준에서 해결한 셈입니다.

**3. 3DMM과 StyleGAN의 다리.** StyleRig(생성 이미지만), PIE(느린 최적화), GIF(아티팩트)와 달리 매핑망 하나로 실제 이미지의 포즈·표정·조명을 3DMM 파라미터로 제어하고, 세부 속성은 DAT-MLP가 보충합니다. [FaceInpainter]({% post_url 2026-09-01-faceinpainter %})가 3DMM 코드를 StyleGAN에 직접 넣은 것보다 분리가 명시적입니다.

**4. 한계는 3DMM과 소스 지향 구조입니다.** 포즈·표정·조명이라는 3DMM이 잡는 속성 외의 변화(가림, 머리 모양 변화)는 2단계 혼합에 의존하고, 소스를 목표 속성으로 "재연기"한 뒤 배경을 섞는 구조라 [FSGAN]({% post_url 2026-08-16-fsgan %})류의 얼굴 형태 문제(목표 윤곽과의 불일치)를 블렌딩이 떠안습니다. 기저는 StyleGAN 하나, 층 분할 하나에 묶여 있습니다.

**5. 평가를 읽을 때 주의할 점.** 교체의 정량 비교는 FaceForensics++·CelebA-HQ 두 표뿐이고 사용자 연구는 없으며, 두 표 모두에서 포즈 오차는 대부분의 비교 방법보다 큽니다(FaceForensics++에서는 Deepfakes만 더 큼). 기저 구성의 $$w_0$$ 100개 평균·직교화가 비선형 다양체에서 얼마나 안정적인지는 논의되지 않았습니다. 위 toy는 선형 잠재 공간이라 헤시안 근사의 어려움(비율 메트릭)을 일반화 고유문제로 우회했고, 실제 StyleGAN 다양체의 곡률은 반영하지 않습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 진단 | 흐림 = 회귀 코드가 $$\mathcal{W}+$$ 밖, 신원 이동 = 비매끄러운 신원 공간에서 방향 찾기 실패 |
| StyleIPSB | 속성 변화/신원 손실 메트릭의 헤시안 상위 고유벡터, 포즈(1~3층)·표정(4~10)·조명(11~18) 기저, 100개 $$w_0$$ 평균 + 직교화 |
| 1단계 | 3DMM 파라미터 → MLP → 기저 좌표 $$\beta$$, $$w_{s \to t} = w_s + \beta V$$, 신원 + 메시·렌더 속성 손실 |
| 2단계 | DAT-MLP의 $$\lambda$$로 $$w_{s \to t}$$와 $$w_t$$ 혼합(세부 속성) |
| 3단계 | 마스크 공간 특징 변환으로 VGGFace 배경 특징과 StyleGAN 특징 혼합 |
| 결과 | 기저로 FID 26.1 → 22.2, 포즈 편집 시 신원 손실 GANSpace보다 작음, 모공 수준 세부 유지 |
| 한계 | 3DMM 속성에 한정, 소스 지향 블렌딩의 형태 문제, 포즈 오차가 대부분의 비교 방법보다 큼 |

## References

- Jiang, Song, Tong, Tang, "StyleIPSB: Identity-Preserving Semantic Basis of StyleGAN for High Fidelity Face Swapping," CVPR 2023. [DOI:10.1109/CVPR52729.2023.00042](https://doi.org/10.1109/CVPR52729.2023.00042)
- Härkönen, Hertzmann, Lehtinen, Paris, "GANSpace: Discovering Interpretable GAN Controls," NeurIPS 2020. [arXiv:2004.02546](https://arxiv.org/abs/2004.02546)
- Shen, Gu, Tang, Zhou, "Interpreting the Latent Space of GANs for Semantic Face Editing" (InterFaceGAN), CVPR 2020. [arXiv:1907.10786](https://arxiv.org/abs/1907.10786)
- Tewari et al., "StyleRig: Rigging StyleGAN for 3D Control over Portrait Images," CVPR 2020. [arXiv:2004.00121](https://arxiv.org/abs/2004.00121)
- Feng, Feng, Black, Bolkart, "Learning an Animatable Detailed 3D Face Model from In-The-Wild Images" (DECA), SIGGRAPH 2021. [arXiv:2012.04012](https://arxiv.org/abs/2012.04012)
- Tov et al., "Designing an Encoder for StyleGAN Image Manipulation" (e4e), SIGGRAPH 2021. [arXiv:2102.02766](https://arxiv.org/abs/2102.02766)
