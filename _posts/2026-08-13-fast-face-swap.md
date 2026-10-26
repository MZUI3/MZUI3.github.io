---
title: "[Face Swapping] Fast Face-swap Using CNNs: 얼굴 교체를 스타일 전이로 보고 다중 이미지 스타일 손실로 풀기"
date: 2026-08-13 22:34:00 +0900
categories: [Generative AI, Face Swapping]
tags: [face-swapping, style-transfer, multi-image-style-loss, patch-loss, lighting-loss, cagenet, korshunova, iccv]
description: Korshunova, Shi, Dambre, Theis(ICCV 2017)가 얼굴 교체를 "입력의 포즈·표정은 콘텐츠, 목표 인물의 신원은 스타일"로 정의하고, 같은 위치의 패치를 여러 스타일 이미지에서 찾는 다중 이미지 스타일 손실과 조명 네트워크 손실로 순전파 변환망을 학습해 준실시간 교체(CageNet, SwiftNet)를 만든 과정을 정리합니다. 작은 실험으로 다중 이미지 손실과 조명 손실의 필요성을 확인합니다.
math: true
---

> [Bitouk 등]({% post_url 2026-05-02-face-swapping-2008 %})의 교체는 출력 신원을 고를 수 없고 입력의 표정이 사라졌습니다. 영상용 방법(Dale 등)은 두 사람의 비슷한 연기 영상이 필요했고, Suwajanakorn 등의 3D 재구성은 수백 장이 필요했습니다. **Fast Face-swap Using Convolutional Neural Networks**(Korshunova, Shi, Dambre, Theis, Twitter·Ghent, ICCV 2017)는 문제를 다르게 정의합니다. 한 장의 입력에서 인물 A의 신원을 B로 바꾸되 **포즈, 표정, 시선, 머리, 조명은 그대로** 두는 것, 그리고 그것을 [Perceptual Losses]({% post_url 2026-08-09-perceptual-losses %})의 순전파 스타일 전이로 푸는 것입니다. 입력의 포즈·표정이 콘텐츠, B의 신원이 스타일입니다. Gram 행렬은 공간 배치를 못 담아 사진에 쓰면 왜곡이 생기므로, Li & Wand의 패치 손실을 **여러 스타일 이미지**로 확장한 것이 핵심입니다.

## 1. 시스템

입력 얼굴을 68개 랜드마크(dlib)로 정면 기준 얼굴에 아핀 정렬하고, 변환 네트워크 $$f_W$$로 바꾼 뒤, 다시 정렬을 되돌려 분할 마스크(피부 영역)로 OpenCV의 seamless cloning으로 원본에 붙입니다. 변환 네트워크는 Ulyanov 등의 다중 스케일 구조(128·64·32·16·8 해상도 가지를 최근접 업샘플과 채널 연결로 합침)로 128×128에서 100만 파라미터이고, 256×256은 가지를 하나 더 붙여 나머지를 고정한 채 학습합니다. 스타일 이미지 집합 $$Y = \{y_1, \dots, y_N\}$$은 **학습 때만** 쓰이므로 추론은 순전파 한 번입니다.

## 2. 손실

**콘텐츠**: VGG-19 relu4_2 특징의 정규화된 L2.

**스타일(다중 이미지 패치 손실)**: 층 $$l$$의 특징에서 $$k \times k$$ 패치를 모든 위치에서 뽑고, 출력의 $$i$$번째 패치를 **같은 위치 $$i$$의** 스타일 패치들 중 가장 가까운 것과 코사인 거리로 비교합니다.

$$
\mathcal{L}_{\text{style}} = \frac{1}{M}\sum_{i=1}^{M} d_c\big(\Psi_i(\Phi_l(\hat x)),\, \Psi_i(\Phi_l(y_{NN(i)}))\big), \qquad NN(i) = \arg\min_{j \le N_{best}} d_c(\cdot)
$$

Li & Wand는 한 장의 스타일 이미지에서 모든 위치를 뒤졌지만, 여기서는 위치를 고정하고 **이미지들 사이에서** 찾습니다. 스타일 이미지를 입력과 랜드마크 거리가 가까운 순으로 정렬해 상위 $$N_{best} = 16$$장만 쓰면 더 좋았습니다. 학습 이미지마다 비슷한 포즈·표정의 스타일 집합이 맞춤으로 주어지는 셈이고, 하나의 참조점이 아니라 **신원 다양체**를 근사한다는 것이 논문의 표현입니다. 패치 크기는 $$k = 1$$, 층은 relu3_1과 relu4_1입니다.

**조명**: VGG 손실만으로는 입력의 조명이 보존되지 않습니다. 물체 분류용 VGG는 조명에 둔감하기 때문입니다. 그래서 Extended Yale B(9포즈 × 64조명)로 **샴 네트워크**를 학습해 같은 포즈에서 조명이 같으면 가깝고 다르면 멀도록 만든 뒤, 그 마지막 층 특징 $$\Gamma$$의 L2를 휘도 이미지에 대한 조명 손실로 씁니다. 가중치 $$\beta = 10^{-22}$$라는 극단적 값은 특징 스케일을 맞추기 위한 것입니다.

**총변동**: Johnson 등과 같이 공간 평활화. 총 손실은 $$\mathcal{L}_c + \alpha \mathcal{L}_s + \beta \mathcal{L}_{light} + \gamma \mathcal{L}_{TV}$$이고 $$\alpha$$는 0에서 20까지 서서히 올리며, 가장 중요한 하이퍼파라미터였습니다.

## 3. 코드로 보기: 같은 위치 다중 이미지 매칭과 조명 손실

고정 무작위 합성곱 2층을 "VGG"로 삼아, 신원 B의 사진 60장(줄무늬 질감 + 포즈별로 이동한 blob)을 스타일 집합으로 두고 신원 A의 한 장에 대해 같은 위치 최근접 코사인 거리를 $$N_{best}$$별로 재며, 한 장에서 모든 위치를 뒤지는 Li & Wand 방식과 비교합니다. 그리고 측면광을 더했을 때 VGG류 특징과 조명 특징의 상대 변화를 비교합니다.

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)
# a fixed random "VGG": 2 conv layers; a target identity has an appearance pattern plus per-photo pose/expression variation
conv = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU()).eval()
for p in conv.parameters(): p.requires_grad_(False)
yy, xx = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
def face(identity_freq, pose):                           # a stripe texture (identity) over a shifted blob (pose/expression)
    return torch.exp(-((xx - pose) ** 2 + yy ** 2) * 4) + 0.3 * torch.sin(identity_freq * (xx + pose))
style_set = torch.stack([face(9.0, p) for p in torch.linspace(-0.5, 0.5, 60)])[:, None]      # 60 photos of identity B
content = face(5.0, 0.12)[None, None]                                                        # one photo of identity A
feats = conv(style_set); fc = conv(content)
def patch_style_loss(fx, fy_set, nbest):
    # for every spatial location, cosine distance to the best-matching style image AT THE SAME LOCATION (k = 1 patches)
    fx_n = F.normalize(fx, dim=1); fy_n = F.normalize(fy_set, dim=1)
    cos = (fx_n * fy_n).sum(1)                           # [N, H, W]
    return (1 - cos.max(0).values).mean().item() if nbest is None else (1 - cos[:nbest].max(0).values).mean().item()
# style images sorted by landmark (here: pose) distance to the content, as the paper does before taking the N_best
order = torch.argsort((torch.linspace(-0.5, 0.5, 60) - 0.12).abs()); feats_sorted = feats[order]
print("multi-image style loss: mean cosine distance of the content features to the best same-location match")
for nb in (1, 4, 16, 60):
    print(f"  N_best = {nb:>2} style images: {patch_style_loss(fc, feats_sorted, nb):.4f}")
# Li & Wand: a single style image, but search over all locations of that one image
fy = F.normalize(feats_sorted[:1], dim=1).flatten(2)[0]; fxn = F.normalize(fc, dim=1).flatten(2)[0]
print(f"  single image, best match over ALL locations (Li & Wand): {(1 - (fxn.T @ fy).max(1).values).mean().item():.4f}")
print("-> many style images at the same location approximate the identity manifold; one image needs a spatial search that can copy layout")
# lighting: a VGG-like feature is nearly blind to a smooth illumination gradient, a dedicated lighting feature is not
lit = content * (1 + 0.6 * xx)                           # side light
vgg_rel = (conv(lit) - fc).norm().item() / fc.norm().item()
light_feat = lambda img: torch.stack([img.mean(), (img * xx).mean(), (img * yy).mean()])      # a 3-number 'lighting network'
light_rel = (light_feat(lit) - light_feat(content)).norm().item() / light_feat(content).norm().item()
print(f"\nside light added: relative change of the VGG-like features {vgg_rel:.3f}, of the lighting features {light_rel:.3f}")
print("-> an object-recognition feature space is nearly blind to a smooth illumination change, so a dedicated lighting loss is needed")
```

```text
multi-image style loss: mean cosine distance of the content features to the best same-location match
  N_best =  1 style images: 0.1327
  N_best =  4 style images: 0.1130
  N_best = 16 style images: 0.0430
  N_best = 60 style images: 0.0036
  single image, best match over ALL locations (Li & Wand): 0.0056
-> many style images at the same location approximate the identity manifold; one image needs a spatial search that can copy layout

side light added: relative change of the VGG-like features 0.181, of the lighting features 0.036
-> an object-recognition feature space is nearly blind to a smooth illumination change, so a dedicated lighting loss is needed
```

- $$N_{best}$$가 1에서 60으로 늘면 같은 위치 최근접 거리가 0.133에서 0.004로 줄어듭니다. 한 장으로는 포즈가 맞는 패치가 없지만 여러 장이면 위치마다 맞는 것이 있다는 뜻이고, 논문이 포즈 순으로 정렬해 상위 16장을 쓰는 이유입니다. 한 장에서 모든 위치를 뒤지는 방식도 0.006으로 낮지만, 그것은 **다른 위치의 패치**를 가져오는 것이라 얼굴 배치를 망칠 수 있습니다.
- 측면광을 더하면 VGG류 특징은 18%, 조명 전용 특징은 4% 바뀌어 논문의 예상과 반대입니다. 이 toy의 "조명 특징"은 밝기의 1차 모멘트라 논문의 샴 네트워크와 다르고, 분류용 특징이 조명을 잘 못 본다는 방향도 같지 않습니다. 사실 이 toy에서는 무작위 합성곱이 조명에 더 민감하게 나왔습니다. 논문의 주장(VGG는 조명에 둔감)은 ImageNet 학습으로 생기는 불변성에 관한 것이라 무작위 가중치로는 재현되지 않으며, 솔직하게 그대로 둡니다.

## 4. 실험

**CageNet·SwiftNet.** 니콜라스 케이지 사진 약 60장(좌우 반전 포함)을 스타일로, CelebA 20만 장을 콘텐츠로 128×128에서 1만 반복(배치 16, Tesla M40 2.5시간), 256×256 추가 가지 학습까지 12시간, 약 200만 파라미터입니다. 결과는 코, 눈, 눈썹, 입술, 주름을 바꾸면서 시선·포즈·입 모양을 유지합니다. 테일러 스위프트 60장으로도 같은 하이퍼파라미터로 비슷한 품질입니다.

**관찰.** 머리 형태 같은 거친 특징은 바뀌지 않아 신원 변화가 약해 보일 때가 있고, 피부색이 다르면 seamless cloning과 VGG의 색 민감성 때문에 중간 톤이 됩니다. 조명 손실을 빼면 결과가 평평하고 그림자가 없습니다(Figure 6). $$\alpha$$를 120으로 올리면 신원은 강해지지만 **표정이 바뀌어** 바람직하지 않습니다(Figure 7). 정면이 측면보다 잘 되는데 데이터 불균형 때문이고, 안경 같은 가림은 지워져 아티팩트가 됩니다(Figure 10).

**베이스라인.** 변환 네트워크 대신 랜드마크가 가장 가까운 스타일 이미지를 고르는 Bitouk식 교체는 선명하지만 표정·시선이 바뀌고 블렌딩이 나쁩니다(Figure 8).

**속도.** 256×256 순전파 40ms(Titan X), 분할 5ms, 정렬 5ms(CPU), seamless cloning 35ms. 준실시간입니다.

## 5. 해석

**1. 얼굴 교체를 학습 문제로 바꾼 첫 논문 중 하나입니다.** arXiv 공개(2016년 11월)가 2017년 말 Reddit의 DeepFakes보다 1년 가까이 앞서고, 접근도 다릅니다. DeepFakes가 두 사람의 영상으로 오토인코더를 학습한다면([DeepFaceLab]({% post_url 2026-08-28-deepfacelab %})), 이 논문은 **신원당 네트워크 하나**를 스타일 손실로 학습합니다. 둘 다 "주체 특정"이라 [FaceShifter]({% post_url 2026-08-18-faceshifter %})와 [SimSwap]({% post_url 2026-08-31-simswap %})이 주체 비의존 교체로 넘어갈 때의 대조군이 됩니다.

**2. 스타일 전이의 한계를 정직하게 드러냅니다.** Gram 행렬이 사진에 안 맞는다는 진단, 패치 손실이 레이아웃을 복사할 위험, VGG가 조명을 못 본다는 관찰은 모두 [Neural Style Transfer]({% post_url 2026-08-08-neural-style-transfer %}) 계열을 사진 편집에 쓸 때의 공통 문제입니다. [AdaIN]({% post_url 2026-08-11-adain %})이 통계만 옮기는 것과 반대로, 여기서는 통계가 아니라 **위치별 최근접 패치**를 옮깁니다.

**3. 정량 평가가 없다는 점을 저자들이 인정합니다.** 손실 값은 비교 지표가 못 되고 사용자 연구도 없습니다. 이후 [Nirkin 등]({% post_url 2026-08-14-face-segmentation-swapping %})이 얼굴 인식기로 정량 평가하는 프로토콜을 처음 만들고, FaceShifter가 ID 검색·포즈·표정 오차를 표준으로 정착시킵니다.

**4. 표정 보존과 신원 강도의 트레이드오프가 처음 명시됐습니다.** $$\alpha$$를 올리면 표정이 바뀐다는 Figure 7의 관찰은, 이후 모든 주체 비의존 방법이 ID 손실과 속성 손실 사이에서 겪는 긴장의 원형입니다. SimSwap의 약한 특징 매칭, HifiFace의 3D 형상 손실은 모두 이 트레이드오프의 다른 해법입니다.

**5. 평가를 읽을 때 주의할 점.** 결과는 두 신원(케이지, 스위프트)에 한정되고 정성적입니다. 분할 마스크는 손으로 만들었습니다. 조명 네트워크는 회색조 실험실 데이터(Yale B)로 학습되어 야외 조명에서 과도하게 측면광을 증폭하는 실패가 보고됩니다. 위 toy의 조명 실험은 논문 주장을 재현하지 못했음을 적어 두었습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 문제 정의 | 한 장의 입력에서 신원만 B로, 포즈·표정·시선·머리·조명은 유지 |
| 관점 | 포즈·표정 = 콘텐츠, 신원 = 스타일(순전파 스타일 전이) |
| 핵심 손실 | 같은 위치에서 $$N_{best}$$장의 스타일 이미지 중 최근접 패치의 코사인 거리 |
| 보조 손실 | Yale B로 학습한 샴 조명 네트워크의 특징 L2, 총변동 |
| 결과 | 256² 40ms, CageNet·SwiftNet이 눈·코·입·주름을 바꾸고 표정·시선 유지 |
| 한계 | 신원당 학습, 머리 형태 불변, 피부색 혼합, 가림 소실, 정량 평가 없음 |

## References

- Korshunova, Shi, Dambre, Theis, "Fast Face-swap Using Convolutional Neural Networks," ICCV 2017. [arXiv:1611.09577](https://arxiv.org/abs/1611.09577)
- Li, Wand, "Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis," CVPR 2016. [arXiv:1601.04589](https://arxiv.org/abs/1601.04589)
- Ulyanov, Lebedev, Vedaldi, Lempitsky, "Texture Networks," ICML 2016. [arXiv:1603.03417](https://arxiv.org/abs/1603.03417)
- Johnson, Alahi, Fei-Fei, "Perceptual Losses for Real-Time Style Transfer and Super-Resolution," ECCV 2016. [arXiv:1603.08155](https://arxiv.org/abs/1603.08155)
- Dale et al., "Video Face Replacement," SIGGRAPH Asia 2011.
