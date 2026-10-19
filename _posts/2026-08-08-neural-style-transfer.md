---
title: "[Generative AI] Neural Style Transfer: CNN으로 Content와 Style 분리하기"
date: 2026-08-08 16:35:00 +0900
categories: [Generative AI, Image Synthesis]
tags: [style-transfer, gatys, vgg19, gram-matrix, content-loss, style-loss, image-synthesis]
description: 사전학습된 VGG-19의 특징으로 이미지의 내용과 화풍을 분리하는 Neural Style Transfer의 content loss, Gram matrix 기반 style loss, 픽셀을 직접 최적화하는 과정을 PyTorch 코드와 함께 정리합니다.
math: true
---

> Image Synthesis 카테고리의 첫 글입니다. 사진을 고흐의 「별이 빛나는 밤」 화풍으로 다시 그려 주는 **Neural Style Transfer**의 원조, *Image Style Transfer Using Convolutional Neural Networks*(Gatys, Ecker & Bethge, CVPR 2016)를 정리합니다. 사전학습된 CNN의 특징으로 이미지의 **내용(content)**과 **화풍(style)**을 분리해서 다시 합칩니다.

## 1. Content와 Style

- **Content**: 이미지에 실제로 나타나는 **객체, 장면, 구조**. 예: 건물의 배치와 종류, 사진의 구도
- **Style**: **색상, 질감, 붓 터치** 같은 시각적 특징. 예: 「별이 빛나는 밤」의 색 대비와 소용돌이 질감

두 장의 이미지를 입력으로 받아, **content 이미지의 구조를 style 이미지의 화풍으로** 다시 그린 결과를 만드는 것이 style transfer입니다.

## 2. 기존 방법의 한계

이전의 **texture transfer** 방법은 질감 이미지 조각을 **반복해서 붙인 뒤**, 붙인 조각들 사이의 차이를 최소화하는 방식이었습니다(image quilting 등).

- 이미지 복원이나 질감 전이에서는 좋은 성과
- 하지만 **저수준(low-level) 특징**(픽셀)만 써서, 이미지의 **고수준 의미**를 이해하지 못함
- 학습 파라미터 없이 차이만 계산하므로, 글꼴이나 손글씨 숫자처럼 **단순하고 통제된 환경**에서만 동작

기존 생성 모델은 이미지를 encoding하는 능력이 부족해, content와 style을 **분리하기 어려웠습니다.** 이 논문은 **CNN이 학습한 고수준 특징**을 써서 둘을 분리합니다.

## 3. 특징 추출기: VGG-19

ImageNet으로 사전학습된 **VGG-19**의 특징을 씁니다.

- 당시 이미지 인식 성능이 가장 좋았고, 특징 추출에 효과적이라고 입증된 모델
- 사전학습 덕분에 물체 인식과 위치 파악 능력을 갖춤
- 여러 층에서 **다양한 scale의 feature map**을 얻을 수 있음
- 네트워크 자체는 **학습하지 않고 고정**

### CNN 층에 따른 표현의 변화

CNN은 층이 깊어질수록 **feature map은 작아지고 채널은 많아지며**, 값 하나가 담당하는 영역(receptive field)이 커집니다.

- **Content**: 앞쪽 층의 feature map으로 이미지를 복원하면 **픽셀 세부까지** 거의 그대로 나오지만, 뒤쪽 층에서는 **정확한 픽셀 값은 사라지고 구조와 배치만** 남습니다. 그래서 content는 **깊은 층** 하나로 표현합니다.
- **Style**: 반대 경향입니다. 앞쪽 층의 style 표현은 색 정보 정도만 담지만, **뒤쪽 층까지 여러 층을 합칠수록** style 이미지의 질감을 큰 규모로 잘 재현합니다.

## 4. Content Loss

결과 이미지 $$\vec{x}$$와 content 이미지 $$\vec{p}$$를 같은 VGG에 넣어, 층 $$l$$의 feature map을 비교합니다.

$$
\mathcal{L}_{content}(\vec{p}, \vec{x}, l) = \frac{1}{2}\sum_{i,j}\big(F^l_{ij} - P^l_{ij}\big)^2
$$

- $$F^l, P^l$$: 결과 이미지와 content 이미지의 층 $$l$$ feature map
- $$i$$: 채널, $$j$$: 채널 안의 위치

Feature map을 원소별로 비교해 제곱합을 구합니다. 논문은 VGG-19의 **conv4_2** 층을 씁니다.

## 5. Style Loss와 Gram Matrix

### Gram Matrix

Style은 "**어떤 특징들이 함께 나타나는가**"로 표현합니다. 층 $$l$$에서 채널 $$i$$와 $$j$$의 feature map을 원소별로 곱해 더하면, 두 특징의 **상관관계**가 됩니다.

$$
G^l_{ij} = \sum_k F^l_{ik}\,F^l_{jk}
$$

예를 들어 3채널 2×2 feature map이라면 $$G_{12}$$는 채널 1과 채널 2의 feature map을 원소별로 곱해 더한 값입니다. Gram matrix는 **채널 수 × 채널 수** 크기라, 채널이 많은 뒤쪽 층일수록 **표현력**이 커집니다.

Gram matrix는 **위치 정보를 버리고**(모든 위치를 합산) 특징의 동시 출현만 남기므로, 구도와 무관한 **질감과 화풍**을 나타냅니다.

### Style Loss

결과 이미지의 Gram matrix $$G^l$$과 style 이미지의 Gram matrix $$A^l$$을 층마다 비교합니다.

$$
E_l = \frac{1}{4N_l^2M_l^2}\sum_{i,j}\big(G^l_{ij} - A^l_{ij}\big)^2,\qquad \mathcal{L}_{style}(\vec{a}, \vec{x}) = \sum_{l} w_l\,E_l
$$

- $$N_l$$: 층 $$l$$의 채널 수, $$M_l$$: feature map 크기(높이×너비)
- $$w_l$$: 층별 가중치. 논문은 **conv1_1, conv2_1, conv3_1, conv4_1, conv5_1** 다섯 층을 같은 가중치로 사용

Content와 달리 style은 **여러 층**에서 loss를 구합니다.

## 6. 전체 Loss와 최적화

$$
\mathcal{L}_{total}(\vec{p}, \vec{a}, \vec{x}) = \alpha\,\mathcal{L}_{content}(\vec{p}, \vec{x}) + \beta\,\mathcal{L}_{style}(\vec{a}, \vec{x})
$$

$$\alpha / \beta$$ 비율로 content와 style 중 어느 쪽을 더 살릴지 조절합니다. 비율이 작을수록 화풍이 강해집니다(논문 예시: $$10^{-3}$$, $$10^{-4}$$).

특이한 점은 **네트워크가 아니라 이미지를 최적화**한다는 것입니다. 결과 이미지 $$\vec{x}$$를 흰 노이즈(또는 content 이미지)로 시작해, 고정된 VGG를 통과시킨 loss의 기울기로 **픽셀 값 자체를 업데이트**합니다. 최적화에는 **L-BFGS**를 씁니다. 또 VGG의 max pooling을 **average pooling**으로 바꾸면 결과가 조금 더 부드러웠습니다.

```python
import torch
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
for p in vgg.parameters():
    p.requires_grad_(False)

LAYERS = {"0": "conv1_1", "5": "conv2_1", "10": "conv3_1", "19": "conv4_1", "21": "conv4_2", "28": "conv5_1"}
STYLE = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

def features(x):
    out = {}
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in LAYERS:
            out[LAYERS[name]] = x
    return out

def gram(f):
    b, c, h, w = f.shape
    f = f.view(c, h * w)
    return f @ f.t()

def style_transfer(content, style, alpha=1.0, beta=1e3, steps=300):
    target_c = features(content)["conv4_2"].detach()
    target_s = {l: gram(features(style)[l]).detach() for l in STYLE}
    x = content.clone().requires_grad_(True)                 # 이미지 자체를 최적화
    opt = torch.optim.LBFGS([x])

    def closure():
        opt.zero_grad()
        f = features(x)
        c_loss = 0.5 * ((f["conv4_2"] - target_c) ** 2).sum()
        s_loss = 0.0
        for l in STYLE:
            _, ch, h, w = f[l].shape
            s_loss += ((gram(f[l]) - target_s[l]) ** 2).sum() / (4 * ch ** 2 * (h * w) ** 2) / len(STYLE)
        loss = alpha * c_loss + beta * s_loss
        loss.backward()
        return loss

    for _ in range(steps // 20):
        opt.step(closure)
    return x.detach()
```

실제로 돌릴 때 챙겨야 할 것들입니다.

- `content`, `style`은 `(1, 3, H, W)` 텐서여야 하고, VGG가 학습된 방식대로 **ImageNet 평균·표준편차로 정규화**해서 넣어야 합니다(`transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))`). 결과를 저장할 때는 반대로 되돌립니다.
- 처음 실행하면 VGG-19 사전학습 가중치(약 550MB)를 내려받습니다. CPU로도 돌지만 512px 이미지 기준 GPU를 권장합니다.
- `torch.optim.LBFGS`는 기울기가 아주 작으면(`tolerance_grad`, 기본 1e-7) **바로 멈춥니다.** loss의 크기가 너무 작아지는 $$\alpha, \beta$$ 조합을 쓰면 이미지가 전혀 안 바뀐 채 끝날 수 있으니, 결과가 입력과 같다면 $$\beta$$를 키워 보세요. (이 글을 정리하며 사전학습 가중치 없이 구조만 시험했을 때 실제로 이 현상이 나왔습니다.)

## 7. 결과

- 같은 사진(튀빙겐 강변)을 고흐, 터너, 뭉크, 피카소, 칸딘스키 등의 화풍으로 바꿔, 구조는 유지하면서 화풍이 뚜렷한 결과를 만들었습니다.
- Style을 **얕은 층만으로** 표현하면 국소적인 색과 작은 질감만, **깊은 층까지** 쓰면 더 큰 규모의 화풍 구조가 재현됐습니다.
- $$\alpha / \beta$$를 바꾸면 사진에 가까운 결과부터 원래 그림에 가까운 결과까지 연속적으로 조절됐습니다.

### 한계와 후속 연구

이미지 한 장마다 수백 번의 최적화가 필요해 **느립니다.** 이후 연구는 특정 화풍을 **한 번의 forward로** 입히는 feed-forward 네트워크(Johnson et al., Perceptual Loss), 아무 화풍이나 즉시 입히는 **AdaIN** 등으로 발전했습니다. AdaIN은 [Instance Normalization]({% post_url 2026-05-23-normalization %})의 평균·분산을 style 이미지의 것으로 바꿔 끼우는 방식입니다.

## 8. 정리

| 개념 | 표현 | 사용 층 |
|---|---|---|
| Content | Feature map 자체 | 깊은 층 하나 (conv4_2) |
| Style | **Gram matrix** (채널 간 상관, 위치 무시) | 여러 층 (conv1_1 ~ conv5_1) |
| 최적화 대상 | 네트워크가 아니라 **결과 이미지의 픽셀** | L-BFGS |
| 조절 | $$\alpha / \beta$$ | content ↔ style 비중 |

다음 Image Synthesis 글에서는 픽셀이 아니라 **벡터 그래픽(SVG)**을 생성하는 [DeepSVG]({% post_url 2026-08-15-deepsvg %})를 다룹니다.

## References

- Gatys, Ecker & Bethge, "Image Style Transfer Using Convolutional Neural Networks," CVPR 2016.
- [PyTorch Tutorial – Neural Transfer Using PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- Efros & Freeman, "Image Quilting for Texture Synthesis and Transfer," SIGGRAPH 2001.
- Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR 2015.
- Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution," ECCV 2016.
- Huang & Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization," ICCV 2017.
