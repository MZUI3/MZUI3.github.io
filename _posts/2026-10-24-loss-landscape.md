---
title: "[Optimization] Visualizing the Loss Landscape: 필터 정규화로 손실 지형을 제대로 그리면 무엇이 보이는가"
date: 2026-10-24 04:00:00 +0900
categories: [Neural Network, Optimization]
tags: [loss-landscape, visualization, filter-normalization, sharpness, flat-minima, skip-connection, generalization, hessian, batch-size]
description: Li et al.(NeurIPS 2018)이 제안한 필터 정규화 시각화가 왜 필요한지(스케일 불변성 때문에 기존 1D·2D 플롯이 서로 비교 불가능), 그것으로 본 skip connection·폭·깊이·배치 크기와 손실 지형의 관계, Hessian 고유값으로 확인한 비볼록성을 정리합니다. 작은 실험으로 같은 함수를 계산하는 두 파라미터가 정규화 없이는 전혀 다른 지형으로 보이는 현상을 확인합니다.
math: true
---

> 왜 어떤 네트워크는 잘 학습되고 어떤 네트워크는 안 되는가. 손실 함수의 "지형"을 보면 답이 보일 것 같지만, 수백만 차원의 함수를 2차원 그림으로 그리는 순간 **무엇을 보고 있는지가 불분명**해집니다. **Visualizing the Loss Landscape of Neural Nets**(Li, Xu, Taylor, Studer, Goldstein, University of Maryland·Cornell, NeurIPS 2018)는 기존 시각화의 함정을 짚고, **필터 정규화**라는 간단한 수정으로 서로 다른 네트워크의 지형을 비교 가능하게 만든 뒤, skip connection이 지형을 어떻게 바꾸는지 보여 줍니다. 표지 그림이 된 "ResNet-56 with/without skip connections"가 이 논문에서 나왔습니다.

## 1. 문제: 기존 1D·2D 플롯은 비교할 수 없다

가장 단순한 시각화는 최소점 $$\theta^*$$ 주변에서 무작위 방향 $$\delta$$를 잡고 $$f(\alpha) = L(\theta^* + \alpha \delta)$$를 그리는 것입니다. 두 최소점 사이를 선형 보간해 "SGD가 지나간 길에 장애물이 없더라"는 관찰(Goodfellow et al.)도 같은 종류입니다. 이 방법에는 **스케일 불변성**이라는 함정이 있습니다.

ReLU 네트워크에서 한 층의 가중치를 10배 키우고 다음 층을 10배 줄이면 **함수는 그대로**입니다. Batch Normalization이 있으면 한 층만 임의로 스케일해도 그대로입니다. 그런데 가중치가 큰 네트워크에서는 같은 크기의 섭동 $$\delta$$가 상대적으로 작아서 지형이 **평평하게 보이고**, 가중치가 작으면 뾰족하게 보입니다. 즉 "평평한 최소점이 일반화가 좋다"는 주장도, 두 네트워크의 1D 플롯 비교도, 가중치 스케일을 통제하지 않으면 의미가 없습니다.

## 2. 방법: 필터 정규화

방향 $$\delta$$를 뽑은 뒤, **필터 단위로** 크기를 맞춥니다. 층 $$i$$의 $$j$$번째 필터(FC 층이면 뉴런 하나의 가중치 벡터)에 대해

$$
d_{i,j} \leftarrow \frac{d_{i,j}}{\lVert d_{i,j} \rVert}\, \lVert \theta_{i,j} \rVert
$$

로 섭동의 크기를 해당 필터의 가중치 크기에 비례시킵니다. 이렇게 하면 어떤 필터를 10배 키워도 섭동이 함께 10배 커져서, 플롯이 스케일 변환에 불변이 됩니다. 2D 플롯은 두 방향 $$\delta, \eta$$를 모두 정규화해 $$L(\theta^* + \alpha \delta + \beta \eta)$$를 등고선으로 그립니다. 논문은 필터 정규화된 플롯의 **날카로움이 일반화 오차와 높은 상관관계**를 보인다는 것을 배치 크기와 weight decay를 바꿔 가며 보여 줍니다. 정규화 없이 그리면 큰 배치가 오히려 평평해 보이는 경우가 생깁니다.

## 3. 코드로 보기: 같은 함수, 다른 그림

two-moons 분류를 작은 ReLU MLP로 학습한 뒤, 1층을 10배 키우고 2층을 1/10로 줄인 **함수가 동일한** 파라미터를 하나 더 만듭니다. 같은 무작위 방향을 따라 1D 손실 곡선을 정규화 없이, 그리고 필터 정규화로 그려 비교합니다.

```python
import torch, torch.nn as nn, numpy as np
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)
# two-moons data
n = 400; t = np.random.rand(n) * np.pi
X = np.concatenate([np.stack([np.cos(t), np.sin(t)], 1), np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)])
X = torch.tensor(X + 0.1 * np.random.randn(2 * n, 2), dtype=torch.float32)
Y = torch.tensor([0] * n + [1] * n)
net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
opt = torch.optim.Adam(net.parameters(), 1e-2); lossf = nn.CrossEntropyLoss()
for _ in range(400):
    loss = lossf(net(X), Y); opt.zero_grad(); loss.backward(); opt.step()
theta = [p.detach().clone() for p in net.parameters()]

def loss_at(params):
    with torch.no_grad():
        for p, v in zip(net.parameters(), params): p.copy_(v)
        return lossf(net(X), Y).item()

# a rescaled network: layer1 x10, layer2 /10 -> exactly the same function (ReLU is positively homogeneous)
theta_s = [theta[0] * 10, theta[1] * 10, theta[2] / 10, theta[3], theta[4], theta[5]]
print(f"loss(theta) = {loss_at(theta):.4f}   loss(rescaled theta) = {loss_at(theta_s):.4f}   (same function)")
print(f"||theta|| = {torch.cat([p.flatten() for p in theta]).norm():.1f}   ||rescaled|| = {torch.cat([p.flatten() for p in theta_s]).norm():.1f}")

d = [torch.randn_like(p) for p in theta]
def filter_normalize(d, ref):          # per filter (row of a weight matrix): ||d_i|| <- ||theta_i||
    out = []
    for di, ti in zip(d, ref):
        if di.dim() == 2: out.append(di / di.norm(dim=1, keepdim=True) * ti.norm(dim=1, keepdim=True))
        else: out.append(torch.zeros_like(di))        # biases are left out, as in the paper
    return out
alphas = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
print()
print("1-D loss curve L(theta + alpha * d) along one random direction")
print("alpha           " + "".join(f"{a:>8}" for a in alphas))
for name, base, direc in [("raw, theta        ", theta, d), ("raw, rescaled     ", theta_s, d),
                          ("filter-norm, theta", theta, filter_normalize(d, theta)),
                          ("filter-norm, resc.", theta_s, filter_normalize(d, theta_s))]:
    row = [loss_at([b + a * v for b, v in zip(base, direc)]) for a in alphas]
    print(name + "".join(f"{r:>8.3f}" for r in row))
loss_at(theta)
```

```text
loss(theta) = 0.0001   loss(rescaled theta) = 0.0001   (same function)
||theta|| = 16.2   ||rescaled|| = 69.2

1-D loss curve L(theta + alpha * d) along one random direction
alpha               -1.0    -0.5   -0.25     0.0    0.25     0.5     1.0
raw, theta          29.138   2.099   0.415   0.000   0.203   0.887   6.417
raw, rescaled      175.476  26.389  11.055   0.000  10.608  20.697  23.862
filter-norm, theta   2.220   0.117   0.001   0.000   0.012   0.297   1.525
filter-norm, resc.   2.220   0.117   0.001   0.000   0.012   0.297   1.525
```

- 두 파라미터는 손실이 같은 **동일한 함수**인데, 정규화 없는 곡선은 전혀 다릅니다. 원래 파라미터는 $$\alpha = 0.25$$에서 손실 0.2, 재스케일 파라미터는 10.6입니다. 10배로 키운 1층(이 때문에 전체 파라미터 노름이 16에서 69로 커집니다)은 섭동에 둔감해지고, 1/10로 줄어든 2층은 극도로 민감해지기 때문입니다. 이 그림만 보면 재스케일 쪽이 훨씬 "뾰족한 최소점"처럼 보입니다.
- 필터 정규화를 하면 두 곡선이 **소수점 셋째 자리까지 일치**합니다. 섭동이 각 뉴런의 가중치 크기에 비례하므로 스케일 변환이 상쇄됩니다. 논문이 "날카로움을 비교하려면 먼저 이것부터 해야 한다"고 말하는 이유가 이 표 하나로 보입니다.

## 4. 실험

모두 CIFAR-10, VGG-9와 ResNet 계열입니다.

**배치 크기와 날카로움.** VGG-9를 배치 128과 8192로 학습하면 테스트 오류가 7.37% 대 11.07%입니다. 필터 정규화된 1D·2D 플롯에서 작은 배치의 최소점이 넓고, 큰 배치의 최소점이 좁습니다. weight decay를 끄면 가중치 노름이 커지면서 정규화 없는 플롯은 반대 결론을 내기도 하는데, 정규화된 플롯은 일관되게 일반화 오차와 날카로움이 함께 움직입니다.

**Skip connection.** ResNet-20/56/110에서 shortcut을 제거한 "noshort" 변형은 오류가 8.18% / 13.31% / 16.44%로 깊어질수록 급격히 나빠집니다. 2D 지형을 보면 ResNet-56은 넓고 볼록한 분지인 반면, ResNet-56-noshort는 **중심을 벗어나면 급격히 비볼록한 혼돈 지형**이고, ResNet-110-noshort는 더 심합니다. 얕은 네트워크(20층)는 shortcut 없이도 지형이 온순합니다. 깊이가 비볼록성을 만들고 skip connection이 그것을 막는다는 것이 이 논문의 핵심 그림입니다.

**폭.** Wide-ResNet처럼 필터 수를 2배, 4배, 8배 늘리면 지형이 점점 평평해지고 오류도 내려갑니다. 폭이 비볼록성을 완화한다는 관찰입니다. 121층 DenseNet도 조밀한 연결 덕분에 지형이 매우 온순합니다.

**비볼록성의 정량화.** 2D 플롯이 "볼록해 보인다"고 실제로 볼록한 것은 아닙니다. 저자들은 격자점마다 Hessian의 최소·최대 고유값 비율 $$\lvert \lambda_{min} / \lambda_{max} \rvert$$를 계산해 색으로 겹쳐 그립니다. 볼록해 보이는 영역은 실제로 비율이 작고(거의 볼록), 혼돈 지형에서는 음의 곡률이 큰 영역이 넓게 퍼져 있습니다. 즉 그림은 저차원 투영이지만 비볼록 구조를 **과소 표현할 뿐 날조하지는 않습니다**.

**궤적 시각화.** SGD의 학습 궤적을 무작위 방향에 투영하면 거의 움직이지 않는 것처럼 보이는데, 궤적이 **극히 저차원 부분공간**에 놓여 있기 때문입니다. 궤적의 PCA 상위 두 성분으로 투영하면 진행이 보이고, 대부분의 변동이 두 성분에 담깁니다.

## 5. 해석

**1. 방법론 논문입니다.** 가장 큰 기여는 "ResNet은 평평하다"는 결론이 아니라, 그 결론을 **말할 수 있게 만든 정규화**입니다. 위 실험처럼 정규화 없이는 같은 함수도 다르게 그려지므로, 이 논문 이전의 1D 날카로움 비교는 대부분 다시 봐야 합니다. flat minima 논쟁(Dinh et al.의 "날카로움은 재파라미터화로 조작 가능")에 대한 실용적 답이기도 합니다.

**2. 아키텍처가 최적화 난이도를 결정한다는 직관에 그림을 붙였습니다.** [ResNet]({% post_url 2026-05-28-resnet %})이 왜 깊어질 수 있는지, 폭이 왜 도움이 되는지, DenseNet이 왜 학습이 쉬운지를 하나의 틀(지형의 비볼록성)로 설명합니다. 이후 NAS나 [RevCol]({% post_url 2026-10-24-revcol %}) 같은 구조 설계 논문이 "학습 가능성"을 논할 때 이 틀을 빌립니다.

**3. 저차원 투영의 한계를 스스로 재는 것이 좋습니다.** Hessian 고유값 비율로 "그림이 보여 주는 볼록성이 진짜인가"를 확인한 부분은 시각화 논문으로서 성실한 태도입니다. 그래도 2D 단면은 전체 지형의 극히 일부라는 점을 잊으면 안 됩니다.

**4. 최적화 하이퍼파라미터 해석에도 쓸 수 있습니다.** 배치 크기, weight decay, 학습률이 최소점의 날카로움을 바꾼다는 관찰은 [옵티마이저와 스케줄러]({% post_url 2026-05-15-optimizer-and-scheduler %})에서 다룬 large-batch 학습의 어려움과 직접 연결됩니다. 다만 상관이지 인과를 증명한 것은 아닙니다.

**5. 평가를 읽을 때 주의할 점.** 필터 정규화는 ReLU와 BN의 스케일 불변성은 처리하지만, 모든 재파라미터화 불변성을 처리하지는 않습니다. "날카로움 ↔ 일반화" 상관은 CIFAR-10의 특정 구조에서 관찰된 것이고 Transformer 등으로 일반화하려면 별도 검증이 필요합니다. 위 toy는 BN 없는 은닉층 2개짜리 MLP라 스케일 불변성이 ReLU의 양의 동차성에서만 나오며, 편향은 논문처럼 정규화 방향에서 제외했습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 문제 | 가중치 스케일 불변성 때문에 기존 1D·2D 손실 플롯은 네트워크 간 비교가 불가능 |
| 방법 | 무작위 방향을 필터(뉴런)별로 $$\lVert d_{i,j} \rVert = \lVert \theta_{i,j} \rVert$$가 되도록 정규화 |
| 관찰 1 | 필터 정규화된 날카로움이 일반화 오차와 상관(배치 128 vs 8192: 7.37% vs 11.07%) |
| 관찰 2 | skip connection 제거 시 깊을수록 혼돈 지형(noshort 20/56/110: 8.18 / 13.31 / 16.44%) |
| 관찰 3 | 폭 증가와 DenseNet 연결이 지형을 평평하게 함 |
| 검증 | Hessian 고유값 비율로 투영의 볼록성이 진짜임을 확인, SGD 궤적은 저차원 부분공간 |

## References

- Li, Xu, Taylor, Studer, Goldstein, "Visualizing the Loss Landscape of Neural Nets," NeurIPS 2018. [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)
- Goodfellow, Vinyals, Saxe, "Qualitatively Characterizing Neural Network Optimization Problems," ICLR 2015. [arXiv:1412.6544](https://arxiv.org/abs/1412.6544)
- Keskar et al., "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima," ICLR 2017. [arXiv:1609.04836](https://arxiv.org/abs/1609.04836)
- Dinh, Pascanu, Bengio, Bengio, "Sharp Minima Can Generalize For Deep Nets," ICML 2017. [arXiv:1703.04933](https://arxiv.org/abs/1703.04933)
- He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition," CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
