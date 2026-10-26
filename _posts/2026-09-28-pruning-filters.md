---
title: "[Pruning & Quantization] Pruning Filters for Efficient ConvNets: 가중치가 아니라 필터를 통째로 잘라야 빨라진다"
date: 2026-09-28 13:05:00 +0900
categories: [Neural Network, Lightweight]
tags: [pruning, filter-pruning, structured-pruning, l1-norm, channel-pruning, resnet, vgg, flops, lightweight]
description: Li et al.(ICLR 2017)이 ℓ1 노름이 작은 필터를 통째로 제거하고 다음 층의 대응 커널까지 함께 잘라, 희소 라이브러리 없이 VGG-16 FLOPs 34%, ResNet-110 38%를 줄인 구조적 가지치기를 정리합니다. 층별 민감도 분석, ResNet에서의 제약, 한 번에 자르고 재학습하는 전략을 다루고, 작은 CNN으로 ℓ1 기준과 무작위·최대 기준을 비교합니다.
math: true
---

> [Deep Compression]({% post_url 2026-09-20-deep-compression %})은 가중치를 개별적으로 잘라 저장 용량을 크게 줄였지만, 잘린 자리가 불규칙해서 **희소 연산 라이브러리 없이는 빨라지지 않습니다**. 게다가 잘려 나간 파라미터의 대부분은 FC 층에 있고, 연산량은 conv 층이 지배합니다(VGG-16의 FC는 파라미터의 90%, FLOP의 1% 미만). **Pruning Filters for Efficient ConvNets**(Li, Kadav, Durdanovic, Samet, Graf, University of Maryland·NEC Labs, ICLR 2017)는 conv 층의 **필터를 통째로** 제거합니다. 결과는 그냥 더 작은 밀집 네트워크라서 기존 BLAS로 바로 빨라지고, VGG-16은 FLOP 34%, ResNet-110은 38%를 정확도 손실 거의 없이 줄입니다.

## 1. 문제: 비정형 희소성은 가속이 아니다

크기 기반 가중치 가지치기는 세 가지 한계가 있습니다.

- 희소 커널은 전용 라이브러리나 하드웨어가 필요하고, 희소도가 낮으면 오히려 느립니다. 희소 인덱스 저장은 저정밀 가중치에서는 무시할 수 없는 오버헤드입니다.
- 임계값을 아무리 조절해도 **몇 개의 필터가 실제로 사라질지** 예측하기 어렵습니다.
- 층별로 자르고 재학습하는 반복은 깊은 네트워크에서 너무 오래 걸립니다.

필터 가지치기는 제거 개수가 곧 가속 비율이라 목표 속도에 맞춰 조절하기 쉽고, 희소성이 생기지 않습니다.

## 2. 방법

### 필터와 feature map을 함께 제거

$$i$$번째 conv 층이 $$n_i$$개 입력 채널을 $$n_{i+1}$$개 출력 채널로 바꾸고 커널이 $$k \times k$$일 때, 연산량은 $$n_{i+1} n_i k^2 h_{i+1} w_{i+1}$$입니다. 필터 $$F_{i,j}$$를 하나 제거하면 그 출력 feature map $$x_{i+1,j}$$가 사라지고($$n_i k^2 h_{i+1} w_{i+1}$$ 절감), **다음 층에서 그 feature map에 적용되던 커널들도** 사라집니다($$n_{i+2} k^2 h_{i+2} w_{i+2}$$ 추가 절감). 층 $$i$$에서 $$m$$개를 자르면 층 $$i$$와 $$i+1$$ 모두 $$m / n_{i+1}$$만큼 비용이 줄어듭니다.

### 어떤 필터를 자를 것인가: ℓ1 노름

필터의 중요도를 **절댓값 합** $$s_j = \sum \lvert F_{i,j} \rvert$$로 잽니다. 같은 층의 필터는 입력 채널 수가 같으므로 이것은 커널 가중치의 평균 크기이고, 출력 feature map의 크기에 대한 기대치입니다. 절차는 간단합니다. 필터별 $$s_j$$를 계산해 정렬하고, 가장 작은 $$m$$개와 그 feature map, 다음 층의 대응 커널을 제거한 뒤, 새 커널 행렬을 만들어 남은 가중치를 복사합니다. 그룹 희소 정규화(ℓ2,1)와 비교하면, 중요한 필터는 ℓ1과 ℓ2 모두 크기 때문에 선택 결과에 큰 차이가 없었습니다.

### 층별 민감도

층마다 따로 잘라 보고 재학습 없이 정확도를 재면 **민감도**가 나옵니다. 필터 노름 분포의 기울기가 큰 층(노름이 넓게 퍼진 층)은 잘라도 괜찮고, 평평한 층은 민감합니다. 같은 단계(같은 feature map 크기)의 층은 민감도가 비슷해서 **단계별로 같은 비율**을 쓰고, 민감한 층은 덜 자르거나 건너뜁니다.

### 여러 층을 한 번에

- **독립 가지치기**는 각 층을 다른 층과 무관하게 평가하고, **탐욕 가지치기**는 앞 층에서 이미 제거된 feature map에 대응하는 커널은 노름 계산에서 빼서 평가합니다. 많이 자를수록 탐욕 쪽이 정확합니다.
- **ResNet**은 제약이 있습니다. residual block의 첫 층은 자유롭게 자를 수 있지만, 둘째 층의 출력은 identity와 더해지므로 **shortcut의 projection 층이 선택한 인덱스를 그대로** 따라야 합니다. identity 쪽 feature map이 residual보다 중요하므로 shortcut 기준으로 정합니다. CIFAR용 ResNet은 projection이 없어 첫 층만 자릅니다.
- **재학습 전략**은 "한 번에 자르고 재학습"입니다. 민감하지 않은 층은 상당량을 한 번에 잘라도 짧은 재학습(CIFAR 40 epoch, ImageNet 20 epoch, 원래 학습의 1/4)으로 복구됩니다. 민감한 층을 많이 자르면 반복 가지치기가 낫지만 비용이 큽니다.

## 3. 코드로 보기: ℓ1 기준 대 무작위·최대 기준

합성 4클래스 이미지로 작은 CNN을 학습한 뒤, conv1 16개 필터 중 8개를 ℓ1이 가장 작은 순, 가장 큰 순, 무작위로 **물리적으로 제거**(conv2의 대응 입력 커널 포함)하고, 재학습 전후 정확도와 FLOP을 비교합니다.

```python
import torch, torch.nn as nn, torch.nn.functional as F, copy
torch.manual_seed(0); torch.set_num_threads(1)
# synthetic 4-class 16x16 images: a class template, randomly shifted, plus noise
templates = F.avg_pool2d(torch.randn(4, 1, 20, 20), 3, stride=1, padding=1)[:, :, :20, :20]
def make(n):
    y = torch.randint(0, 4, (n,)); dx, dy = torch.randint(0, 4, (2, n))
    x = torch.stack([templates[c, :, i:i + 16, j:j + 16] for c, i, j in zip(y, dx, dy)])
    return x + 0.6 * torch.randn_like(x), y
Xtr, Ytr = make(3000); Xte, Yte = make(1000)

def build(c1=16, c2=32):
    return nn.Sequential(nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c2, 4))
def train(net, steps, lr=3e-3):
    opt = torch.optim.Adam(net.parameters(), lr)
    for _ in range(steps):
        i = torch.randint(0, 3000, (128,)); loss = F.cross_entropy(net(Xtr[i]), Ytr[i]); opt.zero_grad(); loss.backward(); opt.step()
def acc(net):
    with torch.no_grad(): return (net(Xte).argmax(1) == Yte).float().mean().item() * 100
def flops(net):
    tot = []
    def hook(m, i, o): tot.append(o.numel() * m.in_channels * m.kernel_size[0] * m.kernel_size[1])
    hs = [m.register_forward_hook(hook) for m in net if isinstance(m, nn.Conv2d)]
    with torch.no_grad(): net(Xte[:1])
    for h in hs: h.remove()
    return sum(tot)

net = build(); train(net, 600)
base_acc, base_flops = acc(net), flops(net)
print(f"baseline: acc {base_acc:.1f}%, conv FLOPs {base_flops/1e6:.2f}M")

def prune_conv1(net, keep):                       # physically remove filters of conv1 and the matching input kernels of conv2
    small = build(len(keep), 32)
    small[0].weight.data = net[0].weight.data[keep].clone(); small[0].bias.data = net[0].bias.data[keep].clone()
    small[3].weight.data = net[3].weight.data[:, keep].clone(); small[3].bias.data = net[3].bias.data.clone()
    small[7].load_state_dict(net[7].state_dict()); return small

l1 = net[0].weight.detach().abs().sum((1, 2, 3))
order = l1.argsort()
m = 8                                              # prune 8 of 16 filters
print(f"conv1 filter L1 norms (sorted): " + " ".join(f"{v:.2f}" for v in l1[order]))
print()
print("prune 8/16 conv1 filters   acc (no retrain)   acc (retrain 150 steps)   conv FLOPs")
for name, keep in [("smallest L1 removed", order[m:]), ("largest L1 removed", order[:m]), ("random removed", torch.randperm(16)[:m])]:
    s = prune_conv1(net, keep.sort().values); a0 = acc(s); train(s, 150, 1e-3)
    print(f"{name:<26} {a0:>12.1f}%   {acc(s):>18.1f}%   {flops(s)/1e6:>8.2f}M ({flops(s)/base_flops:.2f}x)")
print()
print("FLOP accounting: conv1 loses m/n = 8/16 of its cost and conv2 loses the same 8/16 share of its input kernels")
```

```text
baseline: acc 95.5%, conv FLOPs 0.33M
conv1 filter L1 norms (sorted): 2.88 3.20 3.20 3.30 3.58 3.59 3.71 3.83 3.86 3.95 3.98 4.24 4.40 5.01 5.14 5.24

prune 8/16 conv1 filters   acc (no retrain)   acc (retrain 150 steps)   conv FLOPs
smallest L1 removed                55.9%                 95.0%       0.17M (0.50x)
largest L1 removed                 44.7%                 93.5%       0.17M (0.50x)
random removed                     40.0%                 94.7%       0.17M (0.50x)

FLOP accounting: conv1 loses m/n = 8/16 of its cost and conv2 loses the same 8/16 share of its input kernels
```

- 재학습 없이 절반을 자르면 **ℓ1이 작은 필터를 제거한 쪽(55.9%)이 큰 필터(44.7%)나 무작위(40.0%)보다** 훨씬 덜 망가집니다. 논문 Figure 8의 관찰과 같습니다.
- 짧은 재학습으로 세 경우 모두 93~95%까지 돌아오지만, ℓ1 기준이 baseline(95.5%)에 가장 가깝습니다. 자르는 기준은 재학습 뒤에도 차이를 남깁니다.
- FLOP은 정확히 절반입니다. conv1은 출력 채널이 절반이 되어 $$m / n_{i+1} = 1/2$$, conv2는 입력 커널이 절반이 되어 역시 $$1/2$$이기 때문입니다. 네트워크는 그냥 더 작은 밀집 CNN이라 희소 연산이 필요 없습니다.

## 4. 실험

| 모델 | 오류 | FLOP | FLOP 절감 | 파라미터 절감 |
|---|---|---|---|---|
| VGG-16 (CIFAR-10) | 6.75% | 3.13×10⁸ | | |
| VGG-16-pruned-A | 6.60% | 2.06×10⁸ | 34.2% | 64.0% |
| ResNet-56 | 6.96% | 1.25×10⁸ | | |
| ResNet-56-pruned-B | 6.94% | 9.09×10⁷ | 27.6% | 13.7% |
| ResNet-110 | 6.47% | 2.53×10⁸ | | |
| ResNet-110-pruned-B | 6.70% | 1.55×10⁸ | 38.6% | 32.4% |
| ResNet-34 (ImageNet) | 26.77% | 3.64×10⁹ | | |
| ResNet-34-pruned-B | 27.83% | 2.76×10⁹ | 24.2% | 10.8% |

**VGG-16(CIFAR-10).** 512채널 층(conv 8~13)은 4×4, 2×2 feature map에서 동작해 **재학습 없이도 60%, 재학습하면 90%의 필터를 제거**할 수 있습니다. 첫 층은 CIFAR에서는 둔감하지만(80%를 잘라도 남은 12개가 입력 채널 3보다 많음) 둘째 층은 민감합니다. conv1과 conv 8~13을 50%씩 자르면 FLOP 34% 절감에 오류가 오히려 내려갑니다.

**ResNet-56/110(CIFAR-10).** 대부분의 층이 둔감하고, ResNet-110은 일부 층을 자르면 재학습 없이도 나아집니다. 민감한 층은 feature map 수가 바뀌는 지점 근처(각 단계의 첫·마지막 블록)이고, 깊은 단계일수록 민감합니다. 그래서 단계별 비율을 다르게 잡습니다(ResNet-56-B: 60% / 30% / 10%, ResNet-110-B: 50% / 40% / 30%). 블록이 많은 ResNet-110은 가운데 블록이 중복이라 더 자르기 쉽습니다.

**ResNet-34(ImageNet).** 첫 층만 자르면 FLOP 24%에 약 1% 손실, 둘째 층과 shortcut까지 자르면(pruned-C) 7.5% 절감에 0.75% 손실로 효율이 나쁩니다. residual block의 첫 층을 자르는 것이 더 효과적이고, 이는 깊은 ResNet의 bottleneck 설계(먼저 줄이고 나중에 늘림)와 통합니다.

**기준 비교.** 가장 큰 ℓ1 필터를 제거하면 정확도가 급락하고, 무작위보다 ℓ1 최소 제거가 거의 모든 층과 비율에서 낫습니다. 활성값 통계 기반 기준(평균, 표준편차, ℓ1, ℓ2, 분산)과 비교하면 mean-std가 60%까지는 비슷하지만 그 이상에서 급락하고, ℓ1은 **데이터 없이** 계산되면서 전반적으로 좋습니다.

**처음부터 학습 vs 가지치기 후 재학습.** 잘린 구조를 처음부터 학습하면(VGG 6.88%, ResNet-56-B 8.69%) 재학습한 모델보다 나쁩니다. 작은 용량의 네트워크를 직접 학습하는 것이 어렵다는 뜻입니다.

**실제 시간.** 밀집 모델이므로 추론 시간 절감이 FLOP 절감과 거의 일치합니다.

## 5. 해석

**1. "무엇을 줄이는가"를 명확히 했습니다.** 파라미터 수, 저장 용량, FLOP, 실제 시간은 다른 지표입니다. 이 논문은 FLOP과 시간을 목표로 삼고, 그에 맞는 단위(필터)를 골랐습니다. 이후 채널 가지치기(ThiNet, Network Slimming, He et al.의 LASSO 기반)가 모두 이 틀 안에 있습니다.

**2. 민감도 분석은 구조 이해의 도구입니다.** ResNet에서 단계 경계의 블록이 민감하고 가운데 블록이 중복이라는 발견은 가지치기를 넘어 "ResNet은 얕은 네트워크의 앙상블처럼 동작한다"는 해석과 맞닿습니다. [Loss Landscape]({% post_url 2026-07-12-loss-landscape %})의 skip connection 분석과 함께 읽으면 좋습니다.

**3. 데이터 없는 기준의 가치.** ℓ1 노름은 보정 데이터 없이 가중치만 보고 계산됩니다. 활성값 기반 기준이 더 정교해 보이지만 결과는 비슷하거나 나빴고, 이 단순함이 널리 쓰이게 된 이유입니다. [양자화 입문]({% post_url 2026-06-15-pruning-and-quantization %})의 "크기 기반 기준이 의외로 강하다"는 메시지와 같습니다.

**4. 구조적 가지치기의 한계도 보입니다.** 절감률(30% 안팎)은 비정형 가지치기의 90%보다 훨씬 작고, ResNet처럼 연결 제약이 있으면 자를 수 있는 층이 제한됩니다. [주파수 영역 가지치기]({% post_url 2026-07-15-fdnp %})처럼 다른 축에서 중복을 찾거나, 하드웨어가 비정형 희소성을 지원하는 쪽(2:4 희소성)으로 발전한 배경입니다.

**5. 평가를 읽을 때 주의할 점.** 보고된 값은 재학습 과정의 **최고** 테스트 정확도라 낙관적입니다. 단계별 비율과 건너뛸 층은 민감도 분석으로 수동 결정했고, 자동화된 것이 아닙니다. ImageNet 결과는 ResNet-34 하나뿐이고 손실이 1% 안팎이라 "손실 없음"이라고 하기 어렵습니다. 위 toy의 과제는 쉬워서 정확도 격차가 실제보다 작을 수 있습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 문제 | 가중치 가지치기는 FC에 집중되고 비정형 희소성이라 conv 가속이 안 됨 |
| 단위 | 필터 + 그 feature map + 다음 층의 대응 커널, 층 $$i$$와 $$i+1$$ 모두 $$m / n_{i+1}$$ 절감 |
| 기준 | 필터의 ℓ1 노름(데이터 불필요), 작은 것부터 제거 |
| 전략 | 층별 민감도 → 단계별 동일 비율, 민감한 층 건너뜀, 한 번에 자르고 재학습(원래의 1/4) |
| ResNet | 블록 첫 층은 자유, 둘째 층은 shortcut projection의 선택을 따름 |
| 결과 | VGG-16 FLOP -34%(오류 6.75 → 6.60%), ResNet-110 -38.6%(6.47 → 6.70%), ResNet-34 -24%(약 1% 손실) |

## References

- Li, Kadav, Durdanovic, Samet, Graf, "Pruning Filters for Efficient ConvNets," ICLR 2017. [arXiv:1608.08710](https://arxiv.org/abs/1608.08710)
- Han, Pool, Tran, Dally, "Learning both Weights and Connections for Efficient Neural Networks," NeurIPS 2015. [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)
- Wen et al., "Learning Structured Sparsity in Deep Neural Networks," NeurIPS 2016. [arXiv:1608.03665](https://arxiv.org/abs/1608.03665)
- Liu et al., "Learning Efficient Convolutional Networks through Network Slimming," ICCV 2017. [arXiv:1708.06519](https://arxiv.org/abs/1708.06519)
- He, Zhang, Sun, "Channel Pruning for Accelerating Very Deep Neural Networks," ICCV 2017. [arXiv:1707.06168](https://arxiv.org/abs/1707.06168)
