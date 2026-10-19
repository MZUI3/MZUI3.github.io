---
title: "[AI 기초 5] 과적합과 일반화: Regularization, Dropout, Early Stopping, Data Augmentation"
date: 2026-05-16 16:30:00 +0900
categories: [Neural Network, Basics]
tags: [overfitting, underfitting, bias-variance, regularization, lasso, ridge, dropout, early-stopping, data-augmentation, mixup, cutmix]
description: 과적합과 과소적합을 bias-variance와 loss 곡선으로 진단하는 법, L1·L2 규제와 Dropout·Early Stopping, 그리고 Mixup·CutMix·RandAugment까지 데이터 증강 기법을 코드와 함께 정리합니다.
math: true
---

> 학습 데이터에서는 정확도가 99%인데 새로운 데이터에서는 70%라면? 모델이 데이터를 **이해한 게 아니라 외운** 것입니다. 이번 글에서는 이 **과적합(Overfitting)**을 진단하는 방법과, 모델을 규제하는 방법(Regularization), 데이터를 늘리는 방법(Data Augmentation)을 묶어서 정리합니다.

## 1. Bias와 Variance

모델의 오차는 크게 두 성분으로 나눠 볼 수 있습니다. 과녁에 비유하면 이해가 쉽습니다.

| | Variance 낮음 | Variance 높음 |
|---|---|---|
| **Bias 낮음** | 중심에 모여 있음 (**최적**) | 중심 근처에 흩어져 있음 |
| **Bias 높음** | 엉뚱한 곳에 모여 있음 | 엉뚱한 곳에 흩어져 있음 |

- **Bias (편향)**: 예측의 평균이 정답에서 얼마나 떨어져 있는가 → 모델이 **너무 단순**하면 커짐
- **Variance (분산)**: 데이터가 조금 바뀔 때 예측이 얼마나 흔들리는가 → 모델이 **너무 복잡**하면 커짐

모델을 복잡하게 만들면 bias는 줄고 variance는 늘어납니다. 제곱 오차 기준으로 기대 오차는 $$\text{Bias}^2 + \text{Variance} + \text{(줄일 수 없는 노이즈)}$$로 나뉘고, 앞의 두 항의 합이 가장 작은 **최적 지점**을 찾는 것이 목표입니다.

> 큰 신경망에서는 파라미터 수를 계속 늘리면 test 오차가 한 번 올라갔다가 다시 내려가는 **double descent** 현상도 관찰됩니다(Nakkiran et al., 2019). 고전적인 U자 곡선이 딥러닝에서 항상 그대로 성립하지는 않는다는 점은 알아 두면 좋습니다.
{: .prompt-info }

## 2. Underfitting vs Overfitting

| | Underfitting (과소적합) | Optimal | Overfitting (과대적합) |
|---|---|---|---|
| 상태 | 입력과 출력의 관계를 제대로 잡지 못함 | 패턴을 잘 일반화 | 학습 데이터에만 최적화 |
| Train 성능 | 낮음 | 높음 | **매우 높음** |
| Test 성능 | 낮음 | 높음 | **낮음** |
| 원인 | 모델이 너무 단순, 학습 부족 | | 모델이 너무 복잡, 데이터 부족 |

### Underfitting 해결

- 모델 복잡도 높이기 (hidden layer, 파라미터 수 늘리기)
- 학습 반복 횟수 늘리기
- 더 유용한 feature 추가하기

(데이터를 늘리는 것은 주로 **과적합**에 효과적입니다. 모델이 너무 단순해서 생긴 과소적합은 데이터를 늘려도 잘 해결되지 않습니다.)

Underfitting은 규제(regularization)가 필요한 문제가 아닙니다. 오히려 규제를 풀어야 합니다.

## 3. 데이터셋 나누기: Train / Validation / Test

과적합을 **알아채려면** 학습에 쓰지 않은 데이터로 평가해야 합니다.

| 데이터셋 | 용도 |
|---|---|
| **Training set** | 가중치 학습 |
| **Validation set** | 학습 중 성능 확인, hyper-parameter 선택, early stopping 시점 판단 |
| **Test set** | 마지막에 **딱 한 번** 최종 성능 평가 |

Validation set으로 계속 모델을 고르다 보면 validation set에도 간접적으로 과적합됩니다. 그래서 test set은 끝까지 따로 둬야 합니다.

### Loss 곡선으로 진단하기

- Train loss와 Val loss가 **둘 다 높게 머무름** → Underfitting
- Train loss는 계속 내려가는데 **Val loss가 어느 순간부터 올라감** → Overfitting 시작

### K-fold Cross Validation

데이터가 적으면 validation set을 따로 떼기 아깝습니다. 데이터를 $$K$$개(보통 5)로 나눠서, 하나씩 돌아가며 validation으로 쓰고 $$K$$번 학습한 평균 성능을 봅니다. 결과는 믿을 만하지만 **학습 시간이 $$K$$배** 듭니다.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)                  # 유방암 진단 데이터 569개
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
kf = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X, y, cv=kf)                # fold마다 새로 학습·평가
print(scores.round(3), scores.mean().round(3))
```

```text
[0.965 0.982 0.956 0.956 1.   ] 0.972
```

(scikit-learn 1.9에서 실행) fold마다 정확도가 0.956~1.0으로 꽤 다릅니다. 검증 세트 하나만 썼다면 운에 따라 성능을 과대·과소평가했을 수 있다는 뜻입니다.

> 위 코드에서 표준화(`StandardScaler`)를 `Pipeline` 안에 넣은 것이 중요합니다. 전체 데이터로 평균·표준편차를 먼저 구한 뒤 나누면, 검증 fold의 정보가 학습에 새어 들어가는 **데이터 누수(data leakage)**가 생겨 점수가 실제보다 좋게 나옵니다.
{: .prompt-warning }

## 4. Overfitting을 막는 방법 한눈에 보기

| 방법 | 아이디어 |
|---|---|
| **데이터 늘리기** | 더 모으거나, **Data Augmentation**으로 변형해서 늘림 |
| **모델 복잡도 줄이기** | 층·파라미터 수 줄이기, 다항식 차수 낮추기 |
| **Regularization** | 손실 함수에 가중치 크기에 대한 벌점 추가 (L1, L2) |
| **Dropout** | 학습 중 무작위로 뉴런을 꺼서 공동 적응 방지 |
| **Early Stopping** | Val loss가 오르기 시작하면 학습 중단 |
| **Batch Normalization** | 학습을 안정시키고 약한 규제 효과 ([6편]({% post_url 2026-05-20-normalization %})) |
| **Noise Injection** | 입력이나 가중치에 일부러 노이즈를 넣음 |
| **Ensemble** | 여러 모델의 예측을 평균 |

---

## Part 1. 모델을 규제하기

## 5. Regularization: L1(Lasso)과 L2(Ridge)

과적합된 모델은 데이터 하나하나를 맞추려고 **가중치가 매우 커지는** 경향이 있습니다. 그래서 손실 함수에 **가중치 크기 자체에 대한 벌점**을 더해, 모델을 "펴 줍니다".

### Ridge Regression (L2)

$$
\text{Cost} = \text{MSE} + \lambda\sum_j w_j^2
$$

- $$\lambda$$가 커질수록 가중치가 작아짐 → 각 변수의 영향 감소
- 중요하지 않은 변수의 가중치를 **0에 가깝게** 만듦 (정확히 0은 아님)
- 변수가 많고 가중치들이 비슷한 크기일 때 성능이 좋음
- 딥러닝에서는 **weight decay**라고 부릅니다. 단, 두 개념이 정확히 같은 것은 **SGD를 쓸 때**뿐입니다. Adam처럼 적응형 학습률을 쓰면 L2 벌점의 기울기도 파라미터마다 다르게 나눠져서 효과가 달라지고, 이 문제를 고친 것이 AdamW입니다([4편 Optimizer]({% post_url 2026-05-15-optimizer-and-scheduler %})).

### Lasso Regression (L1)

$$
\text{Cost} = \text{MSE} + \lambda\sum_j \lvert w_j\rvert
$$

- 중요하지 않은 변수의 가중치를 **정확히 0으로** 만듦 → 자동으로 **변수 선택(feature selection)**
- 특성이 많고 그중 일부만 필요할 때 좋음
- 어떤 특성이 중요한지 해석하기 쉬움

### 왜 L1은 0을 만들고 L2는 못 만들까

L2의 벌점 기울기는 $$2\lambda w$$라서 $$w$$가 작아질수록 미는 힘도 작아집니다. 0 근처에서는 거의 밀지 않습니다. 반면 L1의 기울기는 $$\lambda\,\text{sign}(w)$$로 **크기와 상관없이 일정한 힘**으로 0을 향해 밉니다. 그래서 작은 가중치는 결국 0에 도달합니다.

기하학적으로는 L1 제약 영역이 **마름모**(꼭짓점이 축 위에 있음), L2는 **원**이라서, 손실 함수 등고선이 L1 영역과 처음 만나는 점이 축 위(= 어떤 가중치가 0)일 가능성이 높습니다.

### Elastic Net

$$
\text{Cost} = \text{MSE} + \lambda\Big(\alpha\sum\lvert w_j\rvert + (1-\alpha)\sum w_j^2\Big)
$$

L1과 L2를 섞은 것입니다. $$\alpha = 0$$이면 Ridge, $$\alpha = 1$$이면 Lasso입니다. 상관관계가 높은 변수가 여러 개 있을 때 Lasso는 하나만 고르는 경향이 있는데, Elastic Net은 이를 완화합니다.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

Ridge(alpha=1.0).fit(X, y)                    # sklearn의 alpha가 λ
Lasso(alpha=0.1).fit(X, y)
ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)

# PyTorch: L2는 weight_decay로
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
```

## 6. Dropout

학습할 때마다 hidden layer의 뉴런을 **확률 $$p$$로 무작위로 꺼 버립니다** (보통 0.2~0.5).

### 왜 효과가 있나

1. **Co-adaptation 방지**: 특정 뉴런들이 서로에게 의존해 함께 특정 패턴만 외우는 현상을 막습니다. 어떤 뉴런이든 언제 꺼질지 모르니 각자 쓸모 있는 특징을 배워야 합니다.
2. **앙상블 효과**: 매번 다른 부분 네트워크를 학습하는 셈이라, 수많은 작은 네트워크를 평균하는 효과가 납니다.

### 추론할 때는?

추론 시에는 모든 뉴런을 켭니다. 대신 학습 때보다 활성 뉴런이 많아지므로 출력 크기를 맞춰야 합니다. PyTorch는 **학습 때 남은 뉴런을 $$1/(1-p)$$배** 키우는 inverted dropout을 써서, 추론 때는 아무것도 하지 않아도 됩니다. 그래서 `model.eval()`을 꼭 호출해야 합니다.

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(512, 10),
)
model.train()   # dropout 동작
model.eval()    # dropout 꺼짐
```

**주의**: 네트워크와 데이터셋이 충분히 클 때 효과적입니다. 작은 모델에서는 오히려 underfitting을 만들 수 있습니다.

### DropConnect

뉴런(출력)이 아니라 **가중치(연결)**를 무작위로 0으로 만듭니다. Dropout의 일반화된 형태로 볼 수 있습니다. CNN에서 두 방법을 비교한 연구들에서는 데이터와 구조에 따라 우열이 갈립니다.

### DropBlock

CNN의 feature map은 이웃한 픽셀끼리 정보가 비슷해서, 점 하나를 꺼도 주변 픽셀이 정보를 보충합니다. DropBlock은 **연속된 영역을 통째로** 꺼서 이 문제를 해결합니다. YOLOv4 등에서 사용됩니다.

## 7. Early Stopping

epoch마다 validation loss를 측정해서, **더 이상 좋아지지 않으면 학습을 멈추고 가장 좋았던 시점의 가중치를 복원**합니다.

- 별도 연산이 거의 없는 **가장 간단하고 효과적인 규제**
- 불필요한 학습 시간을 줄임

```python
best, patience, wait = float("inf"), 10, 0
for epoch in range(200):
    train_one_epoch(model)
    val = evaluate(model)
    if val < best:
        best, wait = val, 0
        torch.save(model.state_dict(), "best.pt")     # 가장 좋은 가중치 저장
    else:
        wait += 1
        if wait >= patience:
            break
model.load_state_dict(torch.load("best.pt"))
```

---

## Part 2. 데이터를 늘리기: Data Augmentation

## 8. Data Augmentation이란

새로운 데이터를 수집하지 않고 **기존 데이터를 변형해서 데이터 수를 늘리는 방법**입니다. 새 데이터를 모으고 Labeling하는 데는 시간과 비용이 많이 들지만, 이미지 변형은 오래전부터 쓰던 방법이 많아 쉽고 빠릅니다.

### 지켜야 할 원칙

> 변형해도 **Class Label이 바뀌면 안 됩니다.**

숫자 `6`을 180도 회전하면 `9`가 되므로, MNIST에 큰 회전은 쓰면 안 됩니다. 반대로 사진 속 고양이는 좌우로 뒤집어도 고양이입니다.

목표는 노이즈나 변형이 들어와도 잘 인식하는 **Robust한 모델**입니다. 특정 데이터만 잘 설명하는 것(Overfitting)이 아니라 이상치에 덜 민감해지는 것이므로, augmentation 자체가 일종의 정규화입니다.

## 9. 기본 기법

### Geometric Transformation

| 기법 | 설명 | 주의 |
|---|---|---|
| Flip (horizontal/vertical) | 좌우·상하 반전 | 글자, 숫자는 주의 |
| Translation | 위치 이동, 빈 곳은 상수(0~255)로 채움 | |
| Rotation | 회전 | 회전 각도가 크면 Label이 바뀔 수 있음 |
| Cropping | 일부 영역 잘라내기 (이미지 크기가 바뀜) | 물체가 잘려 나가지 않게 |
| Rescale | 크기 조절 | |

### Color Space Transformation

픽셀의 RGB 값을 조절해 **밝기, 채도, 대비, 색상**을 바꿉니다. 조명 조건이 다양한 환경에 강해집니다.

- 단점: 메모리와 학습 시간이 늘고, **중요한 색상 정보를 잃을 수 있습니다.** 예를 들어 어두운 환경을 학습해야 하는데 밝기를 크게 바꾸면 물체가 아예 안 보이게 됩니다.

### Noise Injection

픽셀 단위로 노이즈를 넣습니다. 데이터에는 원래 노이즈가 없는 게 좋지만, 일부러 넣으면 노이즈에 강한 모델이 됩니다.

- **Salt and Pepper**: 무작위 픽셀을 흰색/검은색으로
- **Gaussian**: 정규분포 노이즈를 더함
- **Speckle**: 픽셀 값에 비례하는 곱셈 노이즈

### Kernel Filter

- **Gaussian Filter**: 가우시안 분포 모양의 필터로 이미지를 **흐리게** 만듦 (노이즈 제거, 초점이 맞지 않은 이미지 대비)

  $$
  G(x, y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2 + y^2}{2\sigma^2}}
  $$

- **Edge Filter**: 윤곽선을 강조해 경계를 **선명하게** 만듦

## 10. 고급 기법

### Random Erasing / Cutout

이미지의 일부 영역을 지우고 무작위 값(0~255)이나 0으로 채웁니다. 물체가 일부 가려져도 인식하게 됩니다. 단, 지우는 영역이 너무 크면 **물체 자체를 지워 버릴 수** 있습니다.

### Mixup

두 이미지와 Label을 **비율 $$\lambda$$로 섞습니다.**

$$
\tilde{x} = \lambda x_i + (1-\lambda)x_j,\qquad \tilde{y} = \lambda y_i + (1-\lambda)y_j,\qquad \lambda \sim \text{Beta}(\alpha, \alpha)
$$

"고양이 80% + 양 20%"처럼 모델이 확률적으로 인식하도록 배워, 결정 경계가 부드러워집니다. 하지만 섞인 이미지가 **비현실적**이라는 단점이 있습니다.

### CutMix

Mixup과 Random Erasing을 합친 형태입니다. 한 이미지의 **일부 영역을 잘라 다른 이미지의 해당 영역에 붙이고**, Label은 붙인 면적 비율로 섞습니다. 픽셀을 섞지 않아 자연스럽고, 모델은 "고양이 **또는** 판다"로 인식하게 됩니다.

```python
import numpy as np, torch

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    H, W = x.shape[2:]
    rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - rh // 2, 0, H), np.clip(cy + rh // 2, 0, H)
    x1, x2 = np.clip(cx - rw // 2, 0, W), np.clip(cx + rw // 2, 0, W)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)        # 실제 붙인 면적으로 보정
    return x, y, y[idx], lam

# loss = lam * CE(out, y_a) + (1 - lam) * CE(out, y_b)

np.random.seed(0); torch.manual_seed(0)
x = torch.zeros(4, 3, 32, 32)
y = torch.tensor([0, 1, 2, 3])
x, y_a, y_b, lam = cutmix(x, y)
print(y_a, y_b, round(float(lam), 3))   # tensor([0, 1, 2, 3]) tensor([0, 1, 3, 2]) 0.527
```

각 샘플이 원래 label `y_a`를 약 53%, 붙여 넣은 이미지의 label `y_b`를 약 47% 비율로 갖게 됩니다. (`randperm` 결과에 따라 자기 자신과 섞이는 샘플도 생길 수 있습니다.)

### Mosaic

이미지 **4장을 이어 붙여** 한 장으로 만듭니다. 한 번에 다양한 배경과 작은 물체를 보게 되어 Object Detection(YOLOv4)에서 많이 씁니다.

### GAN 기반 Augmentation

생성 모델(GAN)이 기존 데이터와 비슷한 **새로운 데이터를 만들어** 데이터 수를 늘립니다. 정답이 뻔한 데이터를 많이 넣는 것보다, 모델이 헷갈려 하는 데이터를 넣는 것이 학습 효과가 더 크다는 관점입니다.

## 11. 최적의 Augmentation 찾기

어떤 augmentation이 좋은지는 데이터마다 다릅니다. 이걸 자동으로 찾는 연구도 있습니다.

### AutoAugment

**강화학습**으로 최적의 augmentation 정책(어떤 변형을, 어떤 확률로, 얼마나 세게)을 탐색합니다. 16가지 연산 조합을 탐색하므로 성능은 좋지만 **탐색에 엄청난 시간과 메모리**가 듭니다.

### Population Based Augmentation (PBA)

**PBT(Population Based Training)** 알고리즘을 이용합니다.

1. 같은 모델을 여러 개 동시에 학습
2. 중간중간 성능을 비교
3. 하위 n% 모델의 augmentation 정책을 상위 n%의 정책으로 교체
4. 교체한 정책에 약간의 변화를 주고 반복

AutoAugment와 비슷한 정확도를 **1000배 이상 빠르게** 찾습니다. 이후에는 탐색 공간을 대폭 줄인 **RandAugment**(변형 개수 N과 강도 M만 정함)가 널리 쓰입니다.

## 12. PyTorch에서 사용하기

```python
from torchvision import transforms as T

train_tf = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.RandomErasing(p=0.25),       # ToTensor 이후에 적용
])

test_tf = T.Compose([               # 평가 때는 augmentation 없이
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

Augmentation은 **학습 데이터에만** 적용합니다. 직접 변형 함수를 만드는 방법은 [15편 PyTorch 실전]({% post_url 2026-07-02-pytorch-training-toolkit %})에서 다룹니다.

> 위 코드는 기존 `torchvision.transforms` API 기준입니다(torchvision 0.28에서 실행 확인). 새 코드라면 박스·마스크도 함께 변형할 수 있는 `torchvision.transforms.v2`를 권장하며, v2에서는 `ToTensor()` 대신 `ToImage()`와 `ToDtype(torch.float32, scale=True)`를 씁니다.
{: .prompt-info }

## 13. 정리

| 문제 | 신호 | 해결 |
|---|---|---|
| Underfitting | Train/Val loss 모두 높음 | 모델 키우기, 더 오래 학습 |
| Overfitting | Train loss ↓, Val loss ↑ | 데이터 늘리기(Augmentation), L1/L2, Dropout, Early stopping |

- **L2(Ridge)**는 가중치를 고르게 작게, **L1(Lasso)**은 불필요한 가중치를 0으로 만듭니다.
- **Dropout**은 co-adaptation을 막고 앙상블 효과를 냅니다.
- **Augmentation**은 Label을 보존하는 변형으로 데이터를 늘리며, Mixup·CutMix·Mosaic처럼 이미지를 섞는 방법까지 발전했습니다.

다음 글에서는 입력과 중간 활성값의 분포를 맞춰 학습을 안정시키는 **Normalization**을 다룹니다.

## References

- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," JMLR 2014.
- Nakkiran et al., "Deep Double Descent: Where Bigger Models and More Data Hurt," ICLR 2020. [arXiv:1912.02292](https://arxiv.org/abs/1912.02292)
- Wan et al., "Regularization of Neural Networks using DropConnect," ICML 2013.
- Ghiasi et al., "DropBlock: A regularization method for convolutional networks," NeurIPS 2018.
- Shorten & Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning," J. Big Data 2019.
- Zhang et al., "mixup: Beyond Empirical Risk Minimization," ICLR 2018.
- Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features," ICCV 2019.
- Cubuk et al., "AutoAugment," CVPR 2019. / Ho et al., "Population Based Augmentation," ICML 2019.
- 장윤석, 임현일, "CNN에서의 DropOut과 DropConnect에 대한 성능 비교," 한국정보처리학회 2019.
