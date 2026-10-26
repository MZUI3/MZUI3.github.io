---
title: "[Distillation] Be Your Own Teacher: 깊은 층이 얕은 층을 가르치는 자기 증류와 깊이 조절 추론"
date: 2026-07-09 18:26:00 +0900
categories: [Neural Network, Lightweight]
tags: [knowledge-distillation, self-distillation, be-your-own-teacher, deep-supervision, early-exit, adaptive-inference, flat-minima, bottleneck, lightweight]
description: 네트워크를 깊이에 따라 여러 구간으로 나누고 구간마다 분류기를 붙여, 가장 깊은 분류기가 얕은 분류기들을 KL과 feature hint로 가르치는 self distillation(Zhang et al., ICCV 2019)의 세 가지 손실, CIFAR-100·ImageNet 결과, 깊이 조절 추론과 flat minima 분석을 정리하고 작은 예제로 분류기 간 일치율을 확인합니다.
math: true
---

> 전통적인 knowledge distillation은 교사를 먼저 학습하고 학생에게 옮기는 두 단계입니다. 교사 설계에 실험이 많이 들고, 학생이 교사의 지식을 다 흡수하지도 못합니다. **Be Your Own Teacher**(Zhang et al., Tsinghua·IIISCT, ICCV 2019)는 교사와 학생을 **한 네트워크 안**에 둡니다. ResNet을 깊이에 따라 4구간으로 나누고 구간마다 분류기를 붙인 뒤, 가장 깊은 분류기(교사)가 얕은 분류기들(학생)을 가르칩니다. CIFAR-100에서 평균 2.65%p가 오르고, 학습 시간은 전통 증류의 1/4.6이며, 학습이 끝나면 **원하는 깊이에서 추론을 멈출 수** 있습니다. [SRDL]({% post_url 2026-06-30-self-referenced-deep-learning %})이 시간 축의 자기 증류라면 이 논문은 깊이 축입니다.

## 1. 문제: 전통 증류의 두 걸림돌

- **지식 전달 효율**: 학생이 교사의 지식을 거의 다 쓰지 못하고, 교사를 넘는 학생은 드뭅니다.
- **교사 설계와 학습**: 좋은 교사 구조를 찾는 데 많은 실험이 필요하고, 과하게 큰 교사를 학습하는 시간이 깁니다.

ResNet-50·CIFAR-100 기준으로 전통 증류는 26.98시간에 79.33%였는데, self distillation은 5.87시간에 81.04%입니다.

## 2. 방법

### 구조

대상 네트워크를 깊이와 원래 구조에 따라 **구간**으로 나눕니다(ResNet-50은 ResBlock 단위로 4구간). 각 얕은 구간 뒤에 **bottleneck 층 + FC 분류기**를 붙입니다. bottleneck은 구간별 분류기끼리의 간섭을 줄이고, feature hint의 L2 손실을 계산할 자리를 만듭니다. 이 추가 부분은 학습에만 쓰고 추론 때는 떼어 냅니다. 떼지 않으면 깊이 조절 추론의 선택지가 됩니다.

### 세 가지 손실

분류기 $$\theta_{i/C}$$($$C$$개 중 $$i$$번째)의 softmax 출력을 $$q^i$$라 할 때, 가장 깊은 분류기를 뺀 각 분류기는 세 손실로 학습됩니다.

$$
\text{loss}_i = (1 - \alpha)\,\text{CrossEntropy}(q^i, y) + \alpha\,\text{KL}(q^i, q^C) + \lambda \|F_i - F_C\|_2^2
$$

1. **라벨의 cross-entropy**: 데이터셋의 지식을 모든 분류기에 직접 넣습니다.
2. **가장 깊은 분류기와의 KL**: 교사(가장 깊은 분류기)의 softmax를 각 얕은 분류기가 따라 하게 합니다. 교사의 영향이 네트워크 전체에 퍼집니다.
3. **feature hint의 L2**: 가장 깊은 분류기의 feature map과 얕은 분류기의 bottleneck feature 사이의 L2 거리입니다. 크기가 다른 feature를 맞추기 위해 FitNet처럼 합성곱 한 층을 쓰는 대신 bottleneck 구조를 씁니다.

가장 깊은 분류기는 $$\alpha = \lambda = 0$$으로 라벨만 봅니다. 전체 손실은 분류기별 손실의 합입니다.

### Deeply supervised net과의 차이

구간마다 분류기를 붙이는 구조는 DSN(Lee et al., 2015)과 같습니다. 차이는 얕은 분류기를 **라벨이 아니라 증류로** 학습한다는 점이고, bottleneck 층이 얕은 분류기와 깊은 분류기의 충돌을 막습니다. 실험에서 모든 분류기가 DSN보다 좋았고, 얕은 분류기의 이득이 더 컸습니다.

## 3. 코드로 보기: 얕은 분류기가 깊은 분류기와 같은 답을 내게 된다

16차원 합성 분류 문제에서 4구간 residual MLP에 구간별 bottleneck + 분류기를 붙이고, deep supervision(얕은 분류기도 라벨만)과 self distillation(식 5)으로 학습해 분류기별 정확도와 **가장 깊은 분류기와의 예측 일치율**을 비교합니다. 3회 평균입니다.

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
C, D = 10, 16
W1, W2 = torch.randn(D, 64) / D ** 0.5, torch.randn(64, C) / 8   # 정답을 만드는 숨은 비선형 함수
def data(n):
    x = torch.randn(n, D)
    return x, (torch.tanh(x @ W1) @ W2 * 12 + torch.randn(n, C)).argmax(1)   # 라벨 노이즈 포함
Xtr, ytr = data(3000); Xte, yte = data(20000)

class SelfDistillNet(nn.Module):
    """4개 구간(section)으로 나눈 residual MLP. 구간마다 bottleneck + FC 분류기를 붙인다(학습 때만 사용)."""
    def __init__(self, w=48, n_sections=4):
        super().__init__()
        self.stem = nn.Linear(D, w)
        self.sections = nn.ModuleList([nn.Sequential(nn.Linear(w, w), nn.ReLU(), nn.Linear(w, w)) for _ in range(n_sections)])
        self.bottlenecks = nn.ModuleList([nn.Sequential(nn.Linear(w, 24), nn.ReLU()) for _ in range(n_sections)])   # hint용 feature
        self.heads = nn.ModuleList([nn.Linear(24, C) for _ in range(n_sections)])
    def forward(self, x, depth=4):                                       # depth만큼만 실행해 그 분류기로 답한다(적응형 추론)
        x = self.stem(x); feats, logits = [], []
        for sec, bn, head in list(zip(self.sections, self.bottlenecks, self.heads))[:depth]:
            x = F.relu(x + sec(x)); f = bn(x); feats.append(f); logits.append(head(f))
        return feats, logits

def train(mode, epochs=120, alpha=0.3, lam=0.03, T=3):
    net = SelfDistillNet(); opt = torch.optim.SGD(net.parameters(), 0.05, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        for i in torch.randperm(len(Xtr)).split(128):
            feats, logits = net(Xtr[i]); y = ytr[i]
            loss = F.cross_entropy(logits[-1], y)                                 # 가장 깊은 분류기는 라벨만
            for f, z in zip(feats[:-1], logits[:-1]):
                if mode == "deep supervision":                                     # 얕은 분류기도 라벨만
                    loss = loss + F.cross_entropy(z, y)
                elif mode == "self distillation":                                  # 라벨 + 깊은 분류기의 KL + feature L2 (논문 식 5)
                    loss = loss + (1 - alpha) * F.cross_entropy(z, y) \
                        + alpha * T ** 2 * F.kl_div(F.log_softmax(z / T, -1), F.softmax(logits[-1].detach() / T, -1), reduction="batchmean") \
                        + lam * F.mse_loss(f, feats[-1].detach())
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        _, logits = net(Xte)
        accs = [(z.argmax(1) == yte).float().mean().item() * 100 for z in logits]
        agree = [(z.argmax(1) == logits[-1].argmax(1)).float().mean().item() * 100 for z in logits[:-1]]   # 깊은 분류기와의 일치율
    return accs, agree

cost = [sum(p.numel() for p in SelfDistillNet().sections[:k].parameters()) for k in range(1, 5)]
print("분류기별 backbone 연산 비율:", " ".join(f"{k + 1}/4 = {cost[k] / cost[-1]:.2f}" for k in range(4)))
for mode in ["deep supervision", "self distillation"]:
    res = [train(mode) for _ in range(3)]
    accs = [sum(r[0][k] for r in res) / 3 for k in range(4)]; agree = [sum(r[1][k] for r in res) / 3 for k in range(3)]
    print(f"{mode:18s}: 정확도 1/4 {accs[0]:.1f} | 2/4 {accs[1]:.1f} | 3/4 {accs[2]:.1f} | 4/4 {accs[3]:.1f} || 4/4와 예측 일치율 1/4 {agree[0]:.1f}% | 2/4 {agree[1]:.1f}% | 3/4 {agree[2]:.1f}%")
```

```text
분류기별 backbone 연산 비율: 1/4 = 0.25 2/4 = 0.50 3/4 = 0.75 4/4 = 1.00
deep supervision  : 정확도 1/4 75.1 | 2/4 74.6 | 3/4 74.4 | 4/4 74.3 || 4/4와 예측 일치율 1/4 88.9% | 2/4 92.3% | 3/4 95.4%
self distillation : 정확도 1/4 73.7 | 2/4 73.1 | 3/4 73.0 | 4/4 73.0 || 4/4와 예측 일치율 1/4 91.6% | 2/4 96.4% | 3/4 97.9%
```

이 toy는 논문의 정확도 이득을 재현하지 못했습니다. 16차원 입력의 MLP는 깊이가 깊을수록 좋아지는 문제가 아니어서(1/4 분류기가 이미 가장 좋음) "깊은 교사가 얕은 학생을 가르친다"는 전제가 성립하지 않습니다. 대신 다른 성질은 보입니다. 자기 증류로 학습하면 얕은 분류기들이 **가장 깊은 분류기와 같은 답을 내는 비율**이 뚜렷이 오릅니다(2/4 기준 92.3% → 96.4%). 깊이 조절 추론에서 얕은 출구를 쓸 때 전체 모델과 일관된 답을 낸다는 뜻이고, 이는 증류 손실이 얕은 분류기를 깊은 분류기 쪽으로 끌어당긴 직접적인 결과입니다. 정확도 이득은 CNN과 이미지처럼 깊이가 실제로 표현력을 더하는 설정에서 나타납니다.

## 4. 실험

### CIFAR-100: 구조 10가지

| 네트워크 | Baseline | 1/4 | 2/4 | 3/4 | 4/4 | 앙상블 |
|---|---|---|---|---|---|---|
| VGG19(BN) | 64.47 | 63.59 | 67.04 | 68.03 | 67.73 | 68.54 |
| ResNet18 | 77.09 | 67.85 | 74.57 | 78.23 | 78.64 | 79.67 |
| ResNet50 | 77.68 | 68.23 | 74.21 | 75.23 | 80.56 | 81.04 |
| ResNet101 | 77.98 | 69.45 | 77.29 | 81.17 | 81.23 | 82.03 |
| ResNet152 | 79.21 | 68.84 | 78.72 | 81.43 | 81.61 | 82.29 |
| ResNeXt29-8 | 81.29 | 71.15 | 79.00 | 81.48 | 81.51 | 81.90 |
| WideResNet20-8 | 79.76 | 68.85 | 78.15 | 80.98 | 80.92 | 81.38 |
| WideResNet44-8 | 79.93 | 72.54 | 81.15 | 81.96 | 82.09 | 82.61 |
| WideResNet28-12 | 80.07 | 71.21 | 80.86 | 81.58 | 81.59 | 82.09 |
| PyramidNet101-240 | 81.12 | 69.23 | 78.15 | 80.98 | 82.30 | 83.51 |

- 모든 네트워크의 4/4 분류기가 baseline보다 좋고, 앙상블 기준으로 평균 **2.65%p**(ResNeXt 0.61 ~ VGG19 4.07) 향상입니다.
- **깊을수록 이득이 큽니다**(앙상블 기준): ResNet101 +4.05, ResNet18 +2.58.
- softmax 출력을 더한 단순 앙상블은 더 좋습니다.
- 3/4 분류기가 baseline을 넘는 경우가 많아(10개 중 8개), 깊이의 3/4만 돌려도 됩니다.

### ImageNet

| 네트워크 | Baseline | 1/4 | 2/4 | 3/4 | 4/4 | 앙상블 |
|---|---|---|---|---|---|---|
| VGG19(BN) | 70.35 | 42.53 | 55.85 | 71.07 | 72.45 | 73.03 |
| ResNet18 | 68.12 | 41.26 | 51.94 | 62.29 | 69.84 | 68.93 |
| ResNet50 | 73.56 | 43.95 | 58.47 | 72.84 | 75.24 | 74.73 |

평균 2.02%p 향상입니다. 다만 얕은 분류기의 하락이 CIFAR보다 커서 앙상블이 오히려 해로울 때가 있고, 분류기 깊이가 더 중요합니다. 복잡한 작업일수록 네트워크에 **중복이 적다**는 뜻으로 읽습니다.

### 전통 증류·DSN과 비교

같은 연산량의 학생을 기준으로 CIFAR-100에서 비교합니다.

| 교사 → 학생 | Baseline | KD | FitNet | AT | DML | **Self distillation** |
|---|---|---|---|---|---|---|
| ResNet152 → ResNet18 | 77.09 | 77.79 | 78.21 | 78.54 | 77.54 | **78.64** |
| ResNet152 → ResNet50 | 77.68 | 79.33 | 80.13 | 79.35 | 78.31 | **80.56** |
| WRN44-8 → WRN20-8 | 79.76 | 79.80 | 80.48 | 80.65 | 79.91 | **80.92** |
| WRN44-8 → WRN28-12 | 80.07 | 80.95 | 80.53 | 81.46 | 80.43 | **81.58** |

교사가 따로 없는데도 대부분의 전통 증류보다 좋습니다. DSN과 비교하면 ResNet18~152 모두에서 모든 분류기가 더 좋고 얕은 분류기의 차이가 큽니다(ResNet152 2/4: 77.04 → 78.72).

### 깊이 조절 추론

| ResNet | 1/4 | 2/4 | 3/4 | 4/4 | 앙상블 |
|---|---|---|---|---|---|
| ResNet18 정확도 / 가속 | 67.23 / 3.11× | 73.80 / 1.87× | 77.75 / 1.30× | 78.38 / 1.00× | 79.27 / 0.93× |
| ResNet50 | 67.87 / 4.64× | 73.80 / 2.20× | 74.54 / 1.23× | 80.27 / 1.00× | 80.67 / 0.93× |
| ResNet101 | 68.17 / 9.00× | 75.43 / 4.27× | 80.98 / 1.11× | 81.01 / 1.00× | 81.72 / 0.96× |
| ResNet152 | 68.84 / 13.36× | 78.22 / 4.29× | 81.43 / 1.07× | 81.61 / 1.00× | 82.29 / 0.98× |

한 backbone을 공유하므로 응답 시간이 중요할 때는 얕은 분류기를, 정확도가 중요할 때는 깊은 분류기를 씁니다. 3/4 분류기는 평균 1.2배 가속에 baseline 이상이고, 2/4는 3.16배 가속에 3.3%p 손실입니다. 깊은 세 분류기의 앙상블은 0.05%의 추가 연산으로 평균 0.67%p를 더합니다.

## 5. 왜 되는가

논문은 세 가지 설명을 제시하고 각각 실험으로 뒷받침합니다.

- **Flat minima**: 파라미터에 가우시안 노이즈를 더하며 학습 정확도와 손실을 재면, 자기 증류로 학습한 ResNet-18은 노이즈가 커져도 정확도를 유지하는 반면 일반 학습 모델은 급격히 떨어집니다. 더 평평한 최솟값에 수렴해 일반화가 좋다는 해석으로, [SRDL]({% post_url 2026-06-30-self-referenced-deep-learning %})의 최솟값 폭 분석과 같은 결론입니다.
- **Vanishing gradient 완화**: 여러 깊이에 감독이 들어가므로 DSN처럼 기울기 소실을 줄입니다. 층별 기울기 크기를 재면 특히 첫 두 ResBlock에서 자기 증류 모델의 기울기가 큽니다.
- **더 판별적인 feature**: WRN28-12의 분류기별 feature를 PCA로 보면 깊을수록 클러스터가 조밀하고, SSE/SSB(클러스터 내 분산 ÷ 클러스터 간 분산)가 19.21 → 7.54 → 6.08 → 5.73으로 줄어듭니다. 좋은 얕은 분류기가 더 판별적인 feature를 만들고, 그것이 다시 깊은 분류기를 돕는 순환입니다.

## 6. 해석

**1. 교사를 없앤 것이 아니라 교사를 안으로 들였습니다.** 가장 깊은 분류기는 라벨만으로 학습하는 보통 모델이고, 얕은 분류기들이 그것을 따라 합니다. 교사와 학생이 backbone을 공유하므로 교사의 feature는 학생에게 "hint"로 바로 쓸 수 있고(세 번째 손실), 교사를 따로 설계하고 학습하는 비용(26.98시간 → 5.87시간)이 사라집니다.

**2. 이득은 얕은 쪽에서 오지만 깊은 쪽도 좋아집니다.** DSN과의 비교에서 얕은 분류기의 향상이 크지만 4/4 분류기도 baseline보다 1.5~3.3%p 좋습니다(ResNet18~152). 논문의 설명은 "좋은 얕은 분류기가 좋은 feature를 만들어 깊은 분류기를 돕는다"입니다. 위 toy에서 그 효과가 없었던 이유는 깊이가 표현력을 더하지 않는 문제였기 때문이고, 정확도 이득이 깊이에 비례한다는 CIFAR 결과(ResNet101 > ResNet18)와 일관됩니다.

**3. 깊이 조절 추론은 덤이지만 실용적입니다.** 하나의 가중치로 네 가지 속도·정확도 조합을 고를 수 있습니다. 위 toy에서 본 것처럼 자기 증류는 얕은 출구의 답을 깊은 출구와 일치시키므로, early exit을 써도 모델의 "의견"이 바뀌지 않는다는 점이 이 구조의 숨은 장점입니다. 이는 [T-Stitch]({% post_url 2026-10-10-t-stitch %})가 diffusion의 시간 축에서 모델 크기를 바꾼 것과 짝을 이루는, 깊이 축의 적응형 추론입니다.

**4. 논문이 남긴 열린 문제.** $$\alpha$$와 $$\lambda$$가 성능에 영향을 주지만 자원 한계로 탐색하지 못했고, 수렴 뒤 가장 깊은 분류기만 보통 방식으로 더 학습하면 0.3~0.7%p가 더 오른다고 합니다. 얕은 분류기가 flat minima를 찾게 돕지만 마지막에는 깊은 분류기의 수렴을 방해한다는 뜻입니다. 방법을 번갈아 쓰는 것을 향후 과제로 남겼습니다.

**5. 평가를 읽을 때 주의할 점.** 분산이나 반복 실행 수를 보고하지 않습니다. 전통 증류와의 비교는 저자 구현이고 교사가 ResNet152로 고정되어 있습니다. ImageNet은 세 모델뿐이며, 학습 시간 비교(4.6배)는 ResNet-50·CIFAR-100 한 설정입니다. 논문 스스로 이 방법이 압축이나 가속보다는 **성능을 올리는 학습 기법**이라고 선을 긋습니다.

## 7. 정리

| 항목 | 내용 |
|---|---|
| 목표 | 별도 교사 없이 증류의 이득을 얻고, 한 네트워크로 깊이 조절 추론까지 |
| 구조 | 네트워크를 깊이별 구간으로 나누고 구간마다 bottleneck + FC 분류기(학습 전용, 추론 시 제거 가능) |
| 손실 | 각 얕은 분류기에 $$(1-\alpha)$$CE + $$\alpha$$KL(깊은 분류기) + $$\lambda$$L2(feature hint). 깊은 분류기는 CE만 |
| 결과 | CIFAR-100 평균 +2.65%p(10개 구조), ImageNet 평균 +2.02%p, 전통 증류(KD·FitNet·AT·DML)와 DSN보다 좋음 |
| 추론 | 3/4 분류기로 평균 1.2배 가속에 손실 없음, 2/4로 3.16배에 -3.3%p, 깊은 분류기 앙상블 +0.67%p |
| 설명 | flat minima, 기울기 소실 완화, 깊을수록 판별적인 feature |

## References

- Zhang et al., "Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation," ICCV 2019. [arXiv:1905.08094](https://arxiv.org/abs/1905.08094)
- Lee et al., "Deeply-Supervised Nets," AISTATS 2015. [arXiv:1409.5185](https://arxiv.org/abs/1409.5185)
- Romero et al., "FitNets: Hints for Thin Deep Nets," ICLR 2015. [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)
- Zhang et al., "Deep Mutual Learning," CVPR 2018. [arXiv:1706.00384](https://arxiv.org/abs/1706.00384)
- Keskar et al., "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima," ICLR 2017. [arXiv:1609.04836](https://arxiv.org/abs/1609.04836)
