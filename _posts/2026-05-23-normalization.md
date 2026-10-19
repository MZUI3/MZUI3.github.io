---
title: "[AI 기초 6] Normalization 총정리: Feature Scaling부터 BN, LN, IN, GN까지"
date: 2026-05-23 15:46:00 +0900
categories: [Neural Network, Basics]
tags: [normalization, batch-normalization, layer-normalization, instance-normalization, group-normalization, weight-normalization, standardization]
description: Normalization·Standardization·Regularization의 차이부터, BN·LN·IN·GN이 어떤 축으로 평균과 분산을 구하는지, 학습·추론 모드 차이와 배치 크기 의존성, Weight Normalization까지 PyTorch로 확인하며 정리합니다.
math: true
---

> "Normalization"이라는 단어는 문맥에 따라 다른 뜻으로 쓰여서 헷갈리기 쉽습니다. 이번 글에서는 먼저 Normalization·Standardization·Regularization의 차이를 정리하고, 딥러닝에서 쓰는 BN, LN, IN, GN, WN이 **어떤 축으로 평균과 분산을 구하는지**를 중심으로 비교합니다.

## 1. 비슷한 세 단어

| 용어 | 한국어 | 의미 |
|---|---|---|
| **Normalization** | 정규화 | 값을 왜곡하지 않고 데이터를 **공통의 스케일**로 맞춤 (Feature Scaling) |
| **Standardization** | 표준화 | 평균 0, 표준편차 1로 변환 (Z-score) |
| **Regularization** | 규제, 일반화 | 모델 복잡도를 줄여 **과적합 방지** ([5편]({% post_url 2026-05-18-overfitting-regularization-augmentation %})) |

## 2. 입력 데이터의 Feature Scaling

키(150~190cm)와 연봉(2000만~1억 원)을 함께 입력으로 쓰면, 숫자가 큰 연봉이 기울기를 지배해 학습이 비효율적입니다. 그래서 feature들의 스케일을 맞춥니다.

### Min-Max Scaling

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

모든 값을 0~1 사이로 맞춥니다. 좁은 의미의 "Normalization"은 보통 이것을 말합니다. **이상치가 있으면** 나머지 값들이 아주 좁은 범위로 압축되는 단점이 있습니다.

### Standardization (Z-score)

$$
x' = \frac{x - \mu}{\sigma}
$$

평균 0, 표준편차 1인 분포로 바꿉니다. 각 값이 **평균에서 얼마나 떨어져 있는지**를 나타내며, 데이터가 정규분포에 가까울 때 특히 유용합니다.

이미지에서는 `transforms.Normalize(mean, std)`가 채널별로 이 표준화를 합니다.

```python
from torchvision import transforms as T
T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))   # ImageNet 통계
```

## 3. 왜 네트워크 내부에도 Normalization이 필요한가

입력만 정규화해도, 층을 지날 때마다 가중치가 업데이트되면서 **중간 활성값의 분포가 계속 바뀝니다.** 앞쪽 층이 조금만 바뀌어도 뒤쪽 층 입장에서는 입력 분포가 매번 달라지는 셈입니다. BN 논문은 이를 **Internal Covariate Shift**라고 불렀습니다. (이후 연구에서 BN의 진짜 효과는 loss surface를 매끄럽게 만드는 것이라는 해석이 더 유력해졌습니다.)

그래서 각 층의 입력을 정규화해 분포를 안정시키는 기법들이 나왔습니다.

## 4. Batch Normalization (BN)

### 동작 방식

미니배치 $$\mathcal{B} = \{x_1, \dots, x_m\}$$에 대해 **채널(feature)마다**

$$
\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m}x_i,\qquad
\sigma_\mathcal{B}^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_\mathcal{B})^2
$$

$$
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}},\qquad
y_i = \gamma\hat{x}_i + \beta
$$

### $$\gamma$$와 $$\beta$$는 왜 필요한가

무조건 평균 0, 분산 1로 만들면 표현력이 제한됩니다. 예를 들어 Sigmoid 앞에서 정규화하면 값이 대부분 선형 구간에 몰려 비선형성을 잃습니다. 그래서 **학습 가능한 scale $$\gamma$$와 shift $$\beta$$**를 둬서, 필요하면 네트워크가 원래 분포로 되돌릴 수도 있게 합니다. 채널마다 2개씩 학습 파라미터가 생깁니다(초기값 $$\gamma = 1$$, $$\beta = 0$$).

### 학습 모드 vs 추론 모드

| | Training mode | Inference mode |
|---|---|---|
| 평균·분산 | **현재 미니배치**의 통계 | 학습 중 모은 **이동 평균(running mean/var)** |
| $$\gamma, \beta$$ | 역전파로 학습 | 학습된 값 고정 |

추론 때는 배치에 샘플이 하나뿐일 수도 있고, 같은 입력에는 항상 같은 출력이 나와야(deterministic) 하므로 저장해 둔 통계를 씁니다. 이동 평균은 지수 이동 평균으로 갱신됩니다.

$$
\hat{\mu} \leftarrow (1 - \text{momentum})\,\hat{\mu} + \text{momentum}\cdot\mu_\mathcal{B}
$$

**평균과 분산은 역전파로 학습되는 값이 아닙니다.** 그래서 `model.eval()`을 빼먹으면 추론 때도 배치 통계를 써서 결과가 이상해집니다.

### PyTorch에서

```python
import torch.nn as nn

nn.BatchNorm1d(num_features=128)   # 입력 (N, C) 또는 (N, C, L)
nn.BatchNorm2d(num_features=64)    # 입력 (N, C, H, W)

# 주요 인자
nn.BatchNorm2d(64,
    eps=1e-5,                  # 분모가 0이 되지 않게 더하는 작은 값
    momentum=0.1,              # running mean/var 갱신 비율
    affine=True,               # γ, β 학습 여부
    track_running_stats=True)  # 이동 평균 추적 여부
```

CNN에서는 보통 **Conv → BN → Activation** 순서로 배치합니다. BN 뒤에 bias $$\beta$$가 있으므로 앞의 Conv는 `bias=False`로 둡니다.

### 장점

- 학습이 **빠르고 안정적**이 됨
- **더 큰 학습률**을 쓸 수 있음
- 초기화에 덜 민감해짐
- 배치마다 통계가 조금씩 달라 **약한 규제 효과**

### 단점

- **미니배치 크기에 의존**: 배치가 작으면 통계가 불안정합니다. 극단적으로 `BatchNorm1d`에 배치 크기 1을 넣으면 채널마다 값이 하나뿐이라 분산을 구할 수 없어, PyTorch는 학습 모드에서 `Expected more than 1 value per channel when training` 에러를 냅니다. `BatchNorm2d`는 배치가 1이어도 H×W개의 위치로 통계를 내므로 동작은 하지만, 한 이미지의 통계라 불안정합니다.
- **RNN에 적용하기 어려움**: 시점마다 통계가 달라서 시점별로 따로 BN을 둬야 하고, 시퀀스 길이가 제각각이면 더 복잡해집니다.
- 학습과 추론의 동작이 달라 버그의 원인이 되기 쉬움

## 5. Layer Normalization (LN)

BN이 **배치 방향**으로 평균을 낸다면, LN은 **샘플 하나 안의 모든 feature**로 평균을 냅니다.

$$
\mu = \frac{1}{H}\sum_{i=1}^{H}x_i,\qquad \sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(x_i - \mu)^2
$$

- **배치 크기와 무관**하게 동작 → 배치 1에서도, 학습과 추론에서 똑같이 동작
- 시퀀스 길이와도 무관 → **RNN과 Transformer의 표준** ([Transformer 글]({% post_url 2026-06-10-attention-is-all-you-need %})의 Add & Norm). Transformer에서는 입력이 (N, 길이, $$d_{model}$$)이고, **토큰 하나의 $$d_{model}$$차원**에 대해 평균·분산을 구합니다(`nn.LayerNorm(d_model)`).

## 6. Instance Normalization (IN)

LN에서 한 단계 더 나아가, **샘플 하나의 채널 하나** 안에서(H×W 공간에 대해서만) 정규화합니다.

이미지의 채널별 평균·분산에는 **스타일(색감, 대비)** 정보가 많이 담겨 있어서, IN은 이를 지워 버립니다. 그래서 **Style Transfer**처럼 원본의 스타일을 없애고 새 스타일을 입혀야 하는 작업에서 효과적입니다. 공간 축(H, W)이 있는 CNN에서만 의미가 있습니다.

## 7. Group Normalization (GN)

채널을 $$G$$개 그룹으로 묶어, **샘플 하나의 그룹 하나** 안에서 정규화합니다. LN과 IN의 중간입니다.

- 모든 채널이 한 그룹($$G=1$$) → LN
- 채널마다 한 그룹($$G=C$$) → IN

배치 크기와 무관하므로, **메모리 문제로 배치를 크게 못 쓰는** Detection, Segmentation, 온디바이스 학습에서 BN 대신 씁니다. 논문 실험에서 BN은 배치 크기가 줄수록 error가 급격히 오르지만, GN은 거의 일정합니다. [TinyTL]({% post_url 2026-06-19-edge-ai-hardware %})도 이 성질을 이용해 배치를 1~8로 줄여 메모리를 아낍니다.

## 8. 네 가지를 한 그림으로

입력 텐서가 (N, C, H, W)일 때 **평균과 분산을 구하는 범위**는 다음과 같습니다.

| 방법 | 정규화 범위 | 통계 개수 | 배치 의존 |
|---|---|---|---|
| **BN** | (N, H, W) — 같은 채널, 배치 전체 | C개 | ○ |
| **LN** | (C, H, W) — 같은 샘플, 모든 채널 | N개 | ✕ |
| **IN** | (H, W) — 같은 샘플, 같은 채널 | N×C개 | ✕ |
| **GN** | (C/G, H, W) — 같은 샘플, 같은 그룹 | N×G개 | ✕ |

```python
import torch, torch.nn as nn

x = torch.randn(8, 32, 16, 16)        # (N, C, H, W)
nn.BatchNorm2d(32)(x)
nn.LayerNorm([32, 16, 16])(x)          # 정규화할 마지막 차원들을 지정
nn.InstanceNorm2d(32)(x)
nn.GroupNorm(num_groups=8, num_channels=32)(x)

# 직접 계산해서 확인 (GN)
g = x.view(8, 8, -1)                   # (N, G, C/G*H*W)
manual = (g - g.mean(-1, keepdim=True)) / torch.sqrt(g.var(-1, unbiased=False, keepdim=True) + 1e-5)
print(torch.allclose(manual.view_as(x), nn.GroupNorm(8, 32, affine=False)(x), atol=1e-5))  # True
```

## 9. Weight Normalization (WN)

앞의 방법들이 **활성값**을 정규화한다면, WN은 **가중치**를 다시 매개변수화합니다. 가중치 벡터를 **크기와 방향으로 분리**합니다.

$$
\mathbf{w} = \frac{g}{\lVert\mathbf{v}\rVert}\mathbf{v}
$$

- $$g$$: 크기 (스칼라, 학습)
- $$\mathbf{v} / \lVert\mathbf{v}\rVert$$: 방향 (단위 벡터)

BN에서 입력을 표준편차로 나누는 것과 비슷한 효과를 내면서, 배치 통계를 쓰지 않습니다. CNN은 보통 **가중치 수가 활성값 수보다 훨씬 적기 때문에** 계산이 더 가볍습니다.

```python
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm   # PyTorch 2.1+

layer = weight_norm(nn.Linear(128, 64))
print(layer.parametrizations.weight.original0.shape)      # g: torch.Size([64, 1])
print(layer.parametrizations.weight.original1.shape)      # v: torch.Size([64, 128])
```

예전 코드에서 보이는 `torch.nn.utils.weight_norm`은 deprecated되었고, 위의 `parametrizations` 버전이 권장됩니다.

## 10. 그 밖의 변형

- **Batch Renormalization**: 미니배치 통계와 이동 평균의 차이를 보정 항으로 넣어, **작은 배치**나 샘플들이 독립적이지 않은 배치에서도 BN을 쓸 수 있게 합니다.
- **Batch-Instance Normalization (BIN)**: IN은 스타일 정보를 완전히 지우는데, 스타일이 중요한 특징인 작업에서는 문제가 됩니다. BIN은 BN과 IN을 **학습 가능한 게이트로 섞어서**, 필요한 만큼만 스타일을 지웁니다. 게이트를 닫으면 IN처럼 동작합니다.
- **Cross mini-Batch Normalization (CmBN)**: 이전 iteration들의 통계를 함께 써서 작은 배치에서 BN을 보완합니다(YOLOv4).

## 11. 정리

**Normalization을 쓰는 이유**

1. 데이터를 **같은 스케일**로 맞춰 학습을 빠르게 만든다.
2. 층마다 활성값 분포를 안정시켜 **큰 학습률**과 **깊은 네트워크**를 가능하게 한다.
3. 재매개변수화($$\gamma$$, $$\beta$$, $$g$$)로 **평균·크기·활성 함수를 독립적으로 조정**할 수 있게 한다.

**무엇을 쓸까**

| 상황 | 추천 |
|---|---|
| 일반적인 CNN, 배치 충분 | BN |
| Transformer, RNN | LN |
| Style Transfer, 이미지 생성 | IN |
| 배치가 작음 (Detection, 온디바이스 학습) | GN |

다음 글부터는 Part 2로 넘어가서, 이미지를 다루는 **CNN의 기본 연산**을 정리합니다.

## References

- Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," ICML 2015.
- Santurkar et al., "How Does Batch Normalization Help Optimization?," NeurIPS 2018. [arXiv:1805.11604](https://arxiv.org/abs/1805.11604)
- Ba et al., "Layer Normalization," 2016.
- Ulyanov et al., "Instance Normalization: The Missing Ingredient for Fast Stylization," 2016.
- Wu & He, "Group Normalization," ECCV 2018. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
- Salimans & Kingma, "Weight Normalization," NeurIPS 2016.
- Ioffe, "Batch Renormalization," NeurIPS 2017. / Nam & Kim, "Batch-Instance Normalization," NeurIPS 2018.
