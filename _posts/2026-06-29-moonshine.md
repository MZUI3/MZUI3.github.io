---
title: "[Distillation] Moonshine: 네트워크를 줄이지 말고 합성곱을 싸게 바꿔서 증류하라"
date: 2026-06-29 15:44:00 +0900
categories: [Neural Network, Lightweight]
tags: [knowledge-distillation, moonshine, attention-transfer, grouped-convolution, bottleneck, depthwise-separable, wide-resnet, model-compression, lightweight]
description: 학생 네트워크를 얇거나 얕게 만드는 대신 교사와 같은 구조를 유지하고 합성곱 블록만 grouped·bottleneck 블록으로 바꾼 뒤 attention transfer로 증류하면, 같은 파라미터 예산에서 더 정확하다는 Moonshine(Crowley et al., NeurIPS 2018)의 블록 설계, CIFAR·ImageNet·Cityscapes 결과를 정리하고 블록별 파라미터 공식을 코드로 확인합니다.
math: true
---

> 메모리가 제한된 기기에 큰 네트워크를 올리려면 작은 학생을 증류해야 합니다. 보통 학생은 교사보다 **얇거나 얕은** 구조입니다. **Moonshine**(Distilling with Cheap Convolutions, Crowley, Gray, Storkey, University of Edinburgh, NeurIPS 2018)은 다른 길을 제안합니다. 교사의 구조를 그대로 두고 **표준 합성곱 블록을 더 싼 블록으로 치환**하는 것입니다. 설계를 다시 할 필요가 없고 같은 하이퍼파라미터로 학습할 수 있으며, 같은 파라미터 예산에서 구조를 줄인 학생보다 정확했습니다. [MobileNet]({% post_url 2026-05-30-mobilenet %})의 depthwise separable 합성곱과 [ResNet]({% post_url 2026-05-28-resnet %})의 bottleneck이 재료입니다.

## 1. 문제: 학생을 어떻게 작게 만들 것인가

큰 교사의 출력으로 작은 학생을 학습하면 학생 혼자 학습할 때보다 좋아집니다. 문제는 학생의 **모양**입니다. 기존 학생은 층이 적거나(얕음) 채널이 적은(얇음) 구조였는데, 이는 학생을 새로 설계하고 하이퍼파라미터를 다시 찾아야 한다는 뜻입니다. 현대 네트워크의 파라미터는 거의 전부 합성곱에 있으므로, 구조는 두고 **합성곱만 싸게** 만들면 어떨까가 이 논문의 질문입니다. 주장은 세 가지입니다.

1. 구조를 줄이는 것보다 **블록을 치환하는 것**이 증류로 더 많이 압축할 수 있습니다.
2. bottleneck 유무에 관계없이 **grouped 합성곱 블록**이 효과적인 대체 블록입니다.
3. 치환은 설계 시간도 학습 복잡도도 싸고, **원래의 optimizer와 하이퍼파라미터를 그대로** 씁니다.

## 2. 증류 방법 두 가지

**Knowledge distillation**(Hinton et al.): 교사 logit $$t$$와 학생 logit $$s$$에 대해

$$
\mathcal{L}_{KD} = (1 - \alpha)\, \mathcal{L}_{CE}(y, \sigma(s)) + 2\alpha T^2 \mathcal{L}_{CE}\big(\sigma(t/T), \sigma(s/T)\big)
$$

**Attention transfer**(Zagoruyko & Komodakis): 선택한 층 $$i$$마다 활성화의 채널별 제곱 평균으로 공간 attention map $$f(A_i)$$를 만들고, 정규화한 map의 차이를 벌합니다.

$$
\mathcal{L}_{AT} = \mathcal{L}_{CE}(y, \sigma(s)) + \beta \sum_{i=1}^{N_L} \left\| \frac{f(A^t_i)}{\|f(A^t_i)\|_2} - \frac{f(A^s_i)}{\|f(A^s_i)\|_2} \right\|_2,\qquad f(A_i) = \frac{1}{N_{A_i}} \sum_j a_{ij}^2
$$

두 네트워크가 같은 층에서 **같은 곳을 보도록** 만드는 손실입니다. 실험에서는 attention transfer가 거의 모든 경우 KD보다 좋았습니다([SAD]({% post_url 2026-07-20-self-attention-distillation %})는 이 attention map을 교사 없이 자기 안에서 씁니다).

## 3. 싼 합성곱 블록

$$N_{in} \times k \times k$$ 필터 $$N_{out}$$개짜리 합성곱의 파라미터는 $$N_{in} N_{out} k^2$$이고, $$N = \max(N_{in}, N_{out})$$으로 두면 $$N^2 k^2$$에 묶입니다. residual 블록 하나에 이런 합성곱이 둘 있습니다(표준 블록 S).

| 블록 | 구조 | 합성곱 파라미터 |
|---|---|---|
| **S** | Conv → Conv | $$2N^2 k^2$$ |
| **G(g)** | GConv(g) → Conv1×1 → GConv(g) → Conv1×1 | $$2N^2 (k^2/g + 1)$$ |
| **B(b)** | Conv1×1($$N \to N/b$$) → Conv → Conv1×1($$N/b \to N$$) | $$N^2 (k^2/b^2 + 2/b)$$ |
| **BG(b, g)** | Conv1×1($$N \to N/b$$) → GConv(g) → Conv1×1($$N/b \to N$$) | $$N^2 \big(k^2/(g b^2) + 2/b\big)$$ |

- **Grouped + Pointwise G(g)**: 채널을 $$g$$개 그룹으로 나눠 그룹 안에서만 섞으면 파라미터가 $$g$$분의 1이 되고, 뒤에 1×1 합성곱을 붙여 그룹 사이를 섞습니다. $$g = N$$이면 depthwise separable입니다.
- **Bottleneck B(b)**: 1×1로 채널을 $$b$$분의 1로 줄인 뒤 합성곱하고 다시 늘립니다.
- **BG(b, g)**: 둘을 합칩니다.

모든 합성곱 앞에 BN + ReLU가 오는 pre-activation이고, $$b$$와 $$g$$를 바꾸면 파라미터 수가 두 자릿수 범위에서 달라집니다. 희소화나 근사 층 같은 방법과 달리 **희소 구조를 저장할 필요가 없고** 어떤 프레임워크에서나 쉽게 구현됩니다.

## 4. 코드로 보기: 블록별 파라미터 공식 검증

WRN-40-2의 conv4 그룹처럼 채널 $$N = 128$$, $$k = 3$$인 블록을 실제 `nn.Conv2d`로 만들어 파라미터를 세고 논문 공식과 비교합니다.

```python
import torch.nn as nn

N, k = 128, 3                                              # 블록의 채널 수, 커널 크기 (WRN-40-2의 conv4 그룹)
def params(m): return sum(p.numel() for p in m.parameters() if p.dim() > 1)   # BN 제외, 합성곱 가중치만

def S():                                                   # 표준 블록: 3x3 합성곱 2개
    return nn.Sequential(nn.Conv2d(N, N, k, padding=1, bias=False), nn.Conv2d(N, N, k, padding=1, bias=False))
def G(g):                                                  # Grouped + Pointwise: (grouped 3x3 → 1x1) × 2
    return nn.Sequential(*[l for _ in range(2) for l in (nn.Conv2d(N, N, k, padding=1, groups=g, bias=False), nn.Conv2d(N, N, 1, bias=False))])
def B(b):                                                  # Bottleneck: 1x1로 N/b로 줄이고 3x3, 다시 1x1로 N
    return nn.Sequential(nn.Conv2d(N, N // b, 1, bias=False), nn.Conv2d(N // b, N // b, k, padding=1, bias=False), nn.Conv2d(N // b, N, 1, bias=False))
def BG(b, g):                                              # Bottleneck + Grouped
    return nn.Sequential(nn.Conv2d(N, N // b, 1, bias=False), nn.Conv2d(N // b, N // b, k, padding=1, groups=g, bias=False), nn.Conv2d(N // b, N, 1, bias=False))

rows = [("S", S(), 2 * N ** 2 * k ** 2), ("G(2)", G(2), 2 * N ** 2 * (k ** 2 / 2 + 1)), ("G(N/4)", G(N // 4), 2 * N ** 2 * (k ** 2 / (N / 4) + 1)),
        ("G(N)", G(N), 2 * N ** 2 * (k ** 2 / N + 1)), ("B(2)", B(2), N ** 2 * (k ** 2 / 4 + 1)), ("BG(2,M/4)", BG(2, N // 8), N ** 2 * (k ** 2 / ((N // 8) * 4) + 1)),
        ("BG(2,M)", BG(2, N // 2), N ** 2 * (k ** 2 / ((N // 2) * 4) + 1))]
print(f"{'블록':12s}{'실측 파라미터':>12s}{'논문 공식':>12s}{'S 대비':>8s}")
for name, m, formula in rows:
    print(f"{name:12s}{params(m):12,d}{int(formula):12,d}{params(m) / params(S()):8.3f}")
```

```text
블록               실측 파라미터       논문 공식    S 대비
S                294,912     294,912   1.000
G(2)             180,224     180,224   0.611
G(N/4)            41,984      41,984   0.142
G(N)              35,072      35,072   0.119
B(2)              53,248      53,248   0.181
BG(2,M/4)         18,688      18,688   0.063
BG(2,M)           16,960      16,960   0.058
```

공식과 실측이 정확히 맞습니다. 눈여겨볼 점은 G(N/4)와 G(N)의 차이입니다. 그룹을 4배 줄여도 파라미터는 0.142와 0.119로 큰 차이가 없는데, 1×1 합성곱의 $$N^2$$ 항이 지배적이기 때문입니다. 뒤의 실험에서 G(N/4)가 G(N)보다 오차가 1.26%p 낮은 것을 생각하면, **완전한 depthwise(g = N)가 늘 최선은 아닙니다**. 비용은 조금 더 들지만 그룹을 덜 쪼개는 쪽이 성능 대비 효율이 좋습니다.

## 5. CIFAR 실험

교사는 WRN-40-2(S 블록, 2.24M 파라미터), 학생은 두 부류입니다. 구조를 줄인 WRN-16-2, 40-1, 16-1(S 블록)과, WRN-40-2 구조에 S-2×2(dilated 2×2 커널), G(g), B(b), BG(2, g)를 넣은 것입니다. 200 epoch, SGD, KD는 $$\alpha = 0.9$$, $$T = 4$$, AT는 $$\beta = 1000$$입니다.

### 같은 예산이면 싼 블록이 이긴다

파라미터 대 오차 그래프(Figure 2)에서 **같은 구조에 싼 블록을 넣은 네트워크가 거의 모두 구조를 줄인 네트워크보다 아래**에 있습니다.

| 네트워크 | 파라미터(K) | MAdds(M) | CIFAR-10 Scr / KD / AT | CIFAR-100 Scr / KD / AT |
|---|---|---|---|---|
| 교사 40-2 S | 2243.5 | 328.3 | 4.79 | 23.85 |
| 16-2 S | 691.7 | 101.4 | 6.53 / 6.03 / 5.66 | 27.63 / 27.97 / 27.24 |
| 40-1 S | 563.9 | 83.6 | 6.48 / 6.39 / 5.50 | 29.64 / 30.21 / 28.24 |
| 16-1 S | 175.1 | 26.8 | 8.81 / 8.75 / 7.72 | 34.00 / 37.28 / 33.74 |
| 40-2 S-2×2 | 1007.1 | 147.4 | 5.89 / 6.03 / 5.09 | 27.20 / 26.98 / 26.09 |
| 40-2 G(2) | 1359.0 | 198.1 | 5.30 / 5.37 / **4.87** | 25.94 / 24.92 / **24.45** |
| 40-2 G(N/8) | 455.8 | 86.4 | 6.07 / 5.61 / 5.06 | 27.85 / 27.05 / 26.15 |
| 40-2 G(N/4) | 363.1 | 62.6 | 6.93 / 6.45 / 5.31 | 28.91 / 27.93 / 26.85 |
| 40-2 G(N) | 293.5 | 44.8 | 8.51 / 8.01 / 6.57 | 31.84 / 29.99 / 30.06 |
| 40-2 B(2) | 431.8 | 64.5 | 6.36 / 6.28 / 5.37 | 28.27 / 28.08 / 26.68 |
| 40-2 BG(2,2) | 286.7 | 43.3 | 6.12 / 6.25 / 5.57 | 28.51 / 28.82 / 28.28 |
| 40-2 BG(2,M/8) | 189.9 | 34.4 | 6.75 / 6.49 / 5.94 | 29.09 / 29.13 / 28.16 |
| 40-2 BG(4,M) | 81.4 | 13.0 | 9.04 / 8.61 / 7.87 | 33.64 / 37.34 / 32.89 |

- **BG(2,2)**는 16-2보다 파라미터가 훨씬 적은데(287K 대 692K) 오차는 더 낮습니다(5.57 대 5.66).
- **G(N/8)**은 교사의 1/5 파라미터로 오차 5.06%, 교사(4.79%)에 근접합니다. **BG(2,M/8)**은 교사의 1/10 미만으로 오차 1.15%p 증가에 그칩니다.
- **G(2)**는 파라미터 60%로 교사와 0.08%p 차이입니다.
- 3×3 커널을 dilated 2×2로 바꾸기만 해도(S-2×2) 절반의 파라미터로 비슷한 성능이 납니다.
- bottleneck이 크면(BG(4,M)) 오차가 크게 늘지만, 그래도 16-1과 비슷한 성능에 파라미터는 절반입니다.

### 증류는 필요하다

모든 경우 AT로 학습한 학생이 처음부터 학습한(Scr) 학생보다 좋고, KD보다도 한 경우(40-2 G(N)의 CIFAR-100, 30.06 대 29.99)를 빼면 좋습니다. 싼 블록의 높은 성능은 **증류에서 나오는 것**이지 데이터로 직접 학습해서 얻을 수 있는 것이 아닙니다. MAdds 기준으로도 싼 블록 네트워크는 표준 블록보다 연산이 적어 대체로 더 빠르지만, 실제 속도는 메모리 페이징과 라이브러리에 따라 다르다고 단서를 답니다.

## 6. ImageNet과 Cityscapes

**ImageNet**: 사전학습 ResNet-34(21.8M)를 교사로 AT 증류. 네 그룹의 출력에 AT 손실, $$\beta = 750$$.

| 모델 | 파라미터 | MAdds | Scratch Top-1 / Top-5 | AT Top-1 / Top-5 |
|---|---|---|---|---|
| Res34 (교사) | 21.8M | 3.669G | 26.73 / 8.57 | - |
| Res18 | 11.7M | 1.818G | 30.36 / 11.02 | 29.18 / 10.05 |
| **Res34-G(4)** | 8.1M | 1.395G | 26.61 / 8.62 | **26.58 / 8.43** |
| Res18-0.5 | 3.2M | 909M | 36.96 / 15.01 | 37.20 / 15.02 |
| **Res34-G(N)** | 3.1M | 559M | 32.98 / 12.26 | **30.16 / 10.66** |

- 약 3M 예산에서 Res34-G(N)의 Top-5 오차가 10.66%로 Res18-0.5(15.02%)보다 훨씬 낮습니다. Res18-0.5는 증류로 오히려 조금 나빠져 **표현력의 한계**에 닿은 것으로 보입니다.
- Res34-G(4)는 처음부터 학습해도 교사와 같은 수준이고(1,300만 개 적은 파라미터로), 증류하면 Top-5에서 **교사를 넘습니다**(8.43 대 8.57). grouped 합성곱의 일반화 능력(Ioannou et al., 2017)이 다시 확인됩니다.

**Cityscapes 분할**: ERFNet(2.06M, 3×1과 1×3 합성곱 쌍)의 블록을 G(N)으로 바꾼 ERFNet-G(N)(0.49M)은 처음부터 학습하면 IoU 65.29, 원래 ERFNet(70.59)을 교사로 AT 증류하면 68.11입니다. 같은 optimizer와 스케줄에 AT 항만 더했습니다.

## 7. 해석

**1. 압축의 자유도를 "구조"에서 "블록"으로 옮겼습니다.** 얇거나 얕은 학생은 설계 공간이 거칠고(너비 배수, 깊이), 새 하이퍼파라미터가 필요합니다. 블록 치환은 $$g$$와 $$b$$라는 두 손잡이로 두 자릿수 범위의 파라미터를 연속적으로 탐색하면서 깊이와 너비, 학습 설정을 그대로 둡니다. 깊이가 보존되므로 attention transfer의 층 대응도 그대로입니다.

**2. 왜 같은 예산에서 더 좋은가.** 논문은 명시적으로 설명하지 않지만, 깊이와 너비가 교사와 같으면 **표현의 "모양"이 교사와 같아** 증류 신호를 받기 쉽다는 해석이 자연스럽습니다. Res18-0.5가 증류로 좋아지지 않는 반면 Res34-G(N)은 2.8%p 좋아진 것이 그 증거입니다. [Auto-KD]({% post_url 2026-06-28-auto-kd %})가 "이종 구조에는 attention 변환"을 권한 것과도 통합니다.

**3. depthwise가 끝이 아닙니다.** grouped 합성곱은 보통 $$g = N$$의 depthwise separable 형태로 쓰이지만, $$g = N/4$$나 $$N/8$$이 적은 비용으로 훨씬 좋았습니다. 위 코드에서 본 것처럼 1×1 합성곱이 파라미터를 지배하므로 그룹을 덜 쪼개는 비용은 작습니다. 그룹 수는 성능과 크기를 맞바꾸는 손쉬운 손잡이입니다.

**4. 증류 방법의 선택도 결과의 일부입니다.** 같은 학생에서 AT가 KD를 거의 언제나 이겼고, KD는 CIFAR-100의 작은 학생에서 처음부터 학습한 것보다 못한 경우도 있었습니다(16-1: 37.28 대 34.00). [DIST]({% post_url 2026-06-26-dist-kd %})가 지적한 "KL의 정확한 일치 요구"가 작은 학생에게 부담이 되는 현상과 맞닿아 있습니다.

**5. 평가를 읽을 때 주의할 점.** CIFAR 결과는 단일 실행이고 분산을 보고하지 않습니다. 비교 대상인 "구조를 줄인 학생"은 WRN의 깊이·너비 조합 세 가지뿐이라 더 잘 설계된 작은 구조와의 비교는 아닙니다. MAdds와 실제 지연 시간의 관계는 측정하지 않았고, grouped 합성곱은 하드웨어에 따라 효율이 크게 다릅니다.

## 8. 정리

| 항목 | 내용 |
|---|---|
| 목표 | 메모리가 제한된 기기용으로 정확도 손실을 최소화하며 파라미터 줄이기 |
| 방법 | 교사 구조를 유지하고 표준 블록을 G(g), B(b), BG(b, g)로 치환 → attention transfer로 증류 |
| 블록 비용 | S: $$2N^2k^2$$, G(g): $$2N^2(k^2/g + 1)$$, B(b): $$N^2(k^2/b^2 + 2/b)$$, BG(b, g): $$N^2(k^2/(gb^2) + 2/b)$$ |
| CIFAR-10 | G(N/8): 교사의 1/5 파라미터로 오차 5.06%(교사 4.79%). BG(2,2)가 16-2보다 적은 파라미터로 더 정확 |
| ImageNet | Res34-G(4) 8.1M: Top-5 8.43%로 교사(8.57%)를 넘음. Res34-G(N) 3.1M: Res18-0.5보다 Top-5 4.4%p 낮음 |
| 교훈 | 증류가 필요하고, AT > KD이며, 그룹을 $$N/4$$~$$N/8$$로 두는 것이 depthwise보다 효율적 |

## References

- Crowley, Gray, Storkey, "Moonshine: Distilling with Cheap Convolutions," NeurIPS 2018. [arXiv:1711.02613](https://arxiv.org/abs/1711.02613)
- Zagoruyko & Komodakis, "Paying More Attention to Attention," ICLR 2017. [arXiv:1612.03928](https://arxiv.org/abs/1612.03928)
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network," 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Ioannou et al., "Deep Roots: Improving CNN Efficiency with Hierarchical Filter Groups," CVPR 2017. [arXiv:1605.06489](https://arxiv.org/abs/1605.06489)
- Zagoruyko & Komodakis, "Wide Residual Networks," BMVC 2016. [arXiv:1605.07146](https://arxiv.org/abs/1605.07146)
- Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," 2017. [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)
