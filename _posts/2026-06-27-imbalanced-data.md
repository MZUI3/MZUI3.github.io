---
title: "[AI 기초] 불균형 데이터 다루기: Long-tail, Re-sampling, Re-weighting, LDAM"
date: 2026-06-27 16:44:00 +0900
categories: [Neural Network, Basics]
tags: [imbalanced-data, long-tail, re-sampling, re-weighting, smote, focal-loss, ldam, knowledge-distillation]
description: 클래스 개수가 고르지 않은 불균형·long-tail 데이터에서 모델이 head 클래스로 치우치는 문제와, re-sampling(SMOTE), re-weighting, Focal·LDAM loss, 지식 증류 기반 방법을 코드와 함께 정리합니다.
math: true
---

> 실제 데이터는 클래스마다 개수가 고르지 않습니다. 고양이 사진은 수만 장인데 희귀 질환 영상은 수십 장뿐인 식입니다. 이번 글은 [AI 기초 시리즈]({% post_url 2026-05-05-ml-overview %})의 보충편으로, 이런 **불균형(imbalanced) 데이터**와 **Long-tail** 분포를 다루는 대표적인 방법들을 정리합니다.

## 1. Long-tailed Dataset

클래스를 데이터 개수 순으로 늘어놓으면, 소수의 클래스(**head**)에 데이터가 몰리고 대부분의 클래스(**tail**)는 데이터가 매우 적은 **긴 꼬리** 모양이 됩니다. 이 분포는 **멱법칙(power law)**을 따르는 경우가 많습니다.

### 연구용 데이터셋

- **CIFAR-10-LT, CIFAR-100-LT**: 원래 균형 잡힌 CIFAR에서 클래스마다 지수적으로 줄어들게 샘플을 뽑아 만든 long-tail 버전. 가장 많은 클래스와 가장 적은 클래스의 비율(imbalance ratio)을 10, 100 등으로 조절
- **ImageNet-LT**, **iNaturalist**(실제 생물 종 분포)
- **MS-Celeb-1M**: 사람마다 사진 수가 크게 다른 얼굴 데이터

Long-tail 분포로 모델의 **강건성과 일반화 능력**을 시험하고, 데이터가 거의 없는 특수 클래스(예: **희귀 질병 검출**)를 인식하는 방법을 연구합니다.

### 무엇이 문제인가

학습 데이터가 매우 적은 **minor(tail) 클래스**에서 문제가 생깁니다.

- 모델이 head 클래스 쪽으로 **편향**되어, 애매하면 head로 예측
- 전체 정확도는 높아 보여도 tail 클래스 정확도는 매우 낮음
- 평가할 때는 보통 클래스별 정확도를 **균등하게** 평균함 (test set은 균형)

## 2. Re-sampling

데이터의 **개수 자체를 조절**해 균형을 맞춥니다.

- **Over-sampling**: minor 클래스 데이터를 **복제하거나 늘림**
- **Under-sampling**: major 클래스 데이터를 **줄임**

단점도 분명합니다.

- 데이터를 제거하면 정보를 잃고, 단순 복제하면 같은 샘플을 반복해서 **과적합**
- 데이터셋의 분포가 바뀌어 오히려 loss가 커질 수 있음

### SMOTE

단순 복제 대신 **새로운 샘플을 합성**합니다. minor 클래스의 한 샘플과 그 $$k$$-최근접 이웃 사이의 **선분 위 임의의 점**을 새 샘플로 만듭니다.

$$
x_{new} = x_i + \lambda\,(x_{nn} - x_i),\qquad \lambda \sim U(0, 1)
$$

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote(X_minor, n_new, k=5, rng=None):
    rng = rng or np.random.default_rng(0)       # 기본 인자에 객체를 직접 두면 호출 간에 상태가 공유됨
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_minor)
    _, idx = nn.kneighbors(X_minor)
    out = []
    for _ in range(n_new):
        i = rng.integers(len(X_minor))
        j = idx[i, rng.integers(1, k + 1)]          # 자기 자신 제외한 이웃
        out.append(X_minor[i] + rng.random() * (X_minor[j] - X_minor[i]))
    return np.array(out)
```

딥러닝에서는 클래스별로 샘플링 확률을 바꾸는 **class-balanced sampler**를 많이 씁니다.

```python
import torch
from torch.utils.data import WeightedRandomSampler

labels = torch.tensor([0] * 900 + [1] * 90 + [2] * 10)    # 900 / 90 / 10개로 불균형한 레이블

counts = torch.bincount(labels)
weights = (1.0 / counts.float())[labels]             # 적은 클래스일수록 자주 뽑힘
sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

# 확인: 한 epoch 동안 뽑힌 클래스 분포
print(torch.bincount(labels[list(sampler)], minlength=3))   # 예: tensor([325, 355, 320]) 거의 1:1:1
```

클래스 2는 원래 10개뿐인데 한 epoch에 약 320번 뽑힙니다. 같은 10장을 **평균 32번씩 반복해서** 보는 셈이라, augmentation 없이 쓰면 tail 클래스에 과적합되기 쉽다는 위의 단점이 바로 이것입니다.

## 3. Re-weighting

데이터는 그대로 두고 **loss에서 클래스별 가중치**를 다르게 줍니다. Cross Entropy를 바탕으로 tail 클래스에 높은 가중치를 줘서, train과 test의 분포를 맞춘 것처럼 학습합니다.

- **Inverse frequency**: 가중치 $$\propto 1/n_c$$ ($$n_c$$: 클래스 $$c$$의 샘플 수)
- **Effective number**(Class-Balanced Loss): 샘플이 많아질수록 새 정보가 줄어드는 점을 고려해 $$\frac{1-\beta}{1-\beta^{n_c}}$$로 가중
- **Cost-sensitive learning**: 클래스별로 틀렸을 때의 비용을 다르게 설정

단점은 **head 클래스 성능이 떨어지고**, 가중치가 큰 tail 클래스에 **과적합**이 생길 수 있다는 것입니다.

## 4. Loss Function 설계

### Balanced Cross Entropy

클래스별 가중치 $$\alpha_c$$를 곱해 loss 비율을 조절합니다.

$$
\mathcal{L} = -\alpha_c\log p_c
$$

하지만 이미 잘 맞히는 **쉬운 예제(easy example)**도 loss가 조금씩 쌓여, 전체 loss를 쉬운 예제들이 지배할 수 있습니다.

### Focal Loss

[Object Detection]({% post_url 2026-06-06-object-detection %})에서 배경과 물체의 불균형을 풀기 위해 나온 loss입니다. **잘 맞힌 예제의 loss를 줄여** 어려운 예제에 집중하게 합니다.

$$
\mathcal{L}_{FL} = -\alpha_c\,(1 - p_c)^\gamma\log p_c
$$

쉬운 예제는 빠르게 수렴시키고 **어려운 예제(hard example)**에 더 집중해 학습합니다. 단, $$\gamma$$를 잘 설정하지 못하면 CE와 성능 차이가 크지 않을 수 있습니다.

### LDAM Loss (Label-Distribution-Aware Margin)

Cao et al.(NeurIPS 2019)은 **클래스마다 다른 margin**을 줍니다. 데이터가 적은 클래스일수록 결정 경계에서 **더 멀리 떨어지도록**(더 큰 margin) 학습시켜, tail 클래스의 일반화 오차 상한을 줄입니다.

$$
\mathcal{L}_{LDAM} = -\log\frac{e^{z_y - \Delta_y}}{e^{z_y - \Delta_y} + \sum_{j \neq y}e^{z_j}},\qquad \Delta_c = \frac{C}{n_c^{1/4}}
$$

또 학습 **처음부터** re-sampling이나 re-weighting을 쓰면 특징 학습을 방해한다고 보고, **DRW(Deferred Re-Weighting)**를 제안합니다. 먼저 일반 방식으로 학습해 좋은 특징을 얻고, **후반에만** re-weighting을 적용합니다.

```python
import torch
import torch.nn.functional as F

class LDAMLoss(torch.nn.Module):
    def __init__(self, class_counts, max_m=0.5, s=30.0):
        super().__init__()
        m = 1.0 / torch.tensor(class_counts, dtype=torch.float).pow(0.25)
        self.m = m * (max_m / m.max())                  # 가장 적은 클래스의 margin = max_m
        self.s = s

    def forward(self, logits, target, weight=None):     # weight: DRW 단계에서만 클래스 가중치
        margin = self.m.to(logits.device)[target]
        logits_m = logits.clone()
        logits_m[torch.arange(len(target)), target] -= margin
        return F.cross_entropy(self.s * logits_m, target, weight=weight)

criterion = LDAMLoss([900, 90, 10])
print(criterion.m)                     # tensor([0.1623, 0.2887, 0.5000]) 적은 클래스일수록 큰 margin
```

> 위 구현은 logit이 **[-1, 1] 범위의 코사인 유사도**라고 가정합니다. 원 논문 구현도 마지막 층의 특징과 가중치를 각각 L2 정규화한 뒤 내적하고(`NormedLinear`), 그 값에 $$s = 30$$을 곱합니다. 일반 `nn.Linear`의 logit에 그대로 쓰면 margin 0.5는 상대적으로 무의미해지고 $$s$$배 때문에 loss가 폭발합니다. 같은 입력으로 시험해 보면 코사인 logit에서는 loss가 약 14.8, 정규화하지 않은 logit에서는 약 99.5가 나왔습니다(PyTorch 2.13).
{: .prompt-warning }

## 5. 여러 모델과 지식 증류 활용

### LFME (Learning From Multiple Experts)

앙상블 학습의 한 방법입니다. 데이터를 **불균형이 덜한 여러 부분집합**(예: head, medium, tail 그룹)으로 나눠 각각 **expert 모델**을 학습시킨 뒤, 이 expert들의 지식을 하나의 **통합 모델**에 증류합니다. 통합 모델과 expert 사이의 성능 차이가 일정 임계값 안으로 들어올 때까지, 쉬운 것부터 단계적으로(curriculum) 반복 학습합니다.

### BKD (Balanced Knowledge Distillation)

[Knowledge Distillation]({% post_url 2026-06-25-pytorch-training-toolkit %})을 불균형 문제에 맞게 바꿉니다.

- Instance-balanced CE로 일반적인 분류 학습을 하고
- Teacher로부터의 증류 loss에는 **클래스별 가중치**를 줘서 tail 클래스의 지식이 묻히지 않게 함

증류의 temperature 등도 클래스 특성에 맞게 조절해, 중요한 클래스의 정보 손실을 줄이는 방향으로 학습합니다.

## 6. 정리

| 접근 | 방법 | 장점 | 단점 |
|---|---|---|---|
| Re-sampling | Over/Under-sampling, SMOTE, balanced sampler | 간단 | 과적합 또는 정보 손실 |
| Re-weighting | Inverse frequency, effective number | 데이터 그대로 | head 성능 저하, tail 과적합 |
| Loss 설계 | Balanced CE, **Focal**, **LDAM + DRW** | 학습 원리 개선 | 하이퍼파라미터 민감 |
| 모델·증류 | **LFME**, **BKD** | 성능 향상 폭 큼 | 학습 비용 증가 |

실제로는 **좋은 특징을 먼저 학습하고, 분류기 쪽에서 균형을 맞추는**(DRW, decoupling) 순서가 효과적인 경우가 많습니다.

## References

- Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," JAIR 2002.
- Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019.
- Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.
- Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (LDAM)," NeurIPS 2019. [arXiv:1906.07413](https://arxiv.org/abs/1906.07413) / [공식 코드](https://github.com/kaidic/LDAM-DRW)
- Xiang et al., "Learning From Multiple Experts: Self-paced Knowledge Distillation for Long-tailed Classification," ECCV 2020.
- Zhang et al., "Balanced Knowledge Distillation for Long-tailed Learning," 2021.
- Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition," ICLR 2020.
