---
title: "[Lightweight] MobileNetV3: NAS와 NetAdapt로 찾고, h-swish와 SE로 다듬은 모바일 백본"
date: 2026-08-07 18:44:00 +0900
categories: [Neural Network, Lightweight]
tags: [lightweight, mobilenetv3, nas, netadapt, h-swish, squeeze-excitation, lr-aspp, ssdlite, on-device, latency]
description: MobileNetV3(Howard et al., ICCV 2019)가 플랫폼 인식 NAS와 NetAdapt로 블록 구조와 층별 채널 수를 찾고, 마지막 단계 재설계·h-swish·SE 블록으로 손으로 다듬어 MobileNetV2 대비 정확도 +3.2%, 지연 -20%를 얻은 과정을 정리합니다. 작은 계산으로 h-swish 근사 오차, SE 파라미터, 마지막 단계의 MAdds 절감을 확인합니다.
math: true
---

> [MobileNetV2]({% post_url 2026-07-22-mobilenetv2 %})까지는 사람이 블록을 설계했습니다. **MobileNetV3**(Howard, Sandler, Chu, Chen, Chen, Tan, Wang, Zhu, Pang, Vasudevan, Le, Adam, Google, ICCV 2019)는 **자동 탐색과 수작업 설계를 섞습니다**. 플랫폼 인식 NAS로 전체 구조를, NetAdapt로 층별 필터 수를 찾은 뒤, 탐색이 놓친 비싼 층을 손으로 고치고 새 비선형성 h-swish와 SE 블록을 넣습니다. 결과는 두 모델입니다. V3-Large는 ImageNet 75.2%를 Pixel 1에서 51ms에, V3-Small은 67.4%를 15.8ms에 냅니다.

## 1. 문제: FLOPs가 아니라 지연 시간

모바일 모델의 목표는 **실제 기기에서의 지연 시간 대비 정확도**입니다. MAdds가 같아도 연산 종류와 메모리 접근 패턴에 따라 속도가 다르고, 이것은 아키텍처 설계만으로는 알기 어렵습니다. 그래서 이 논문은 처음부터 Pixel 폰의 측정 지연을 목적 함수에 넣습니다. 또한 V2에서 비싼 부분이 어디인지(첫 층과 마지막 층), 어떤 비선형성이 양자화와 잘 맞는지 같은 실무적 질문을 함께 다룹니다.

## 2. 방법

### 플랫폼 인식 NAS + NetAdapt

**블록 단위 탐색**은 MnasNet과 같은 RNN 컨트롤러와 분해된 계층적 탐색 공간을 씁니다. 보상은 정확도와 지연의 곱 $$ACC(m) \times [LAT(m)/TAR]^w$$입니다. Large 모델은 $$w = -0.07$$, Small 모델은 $$w = -0.15$$를 씁니다. 작은 모델은 지연이 조금만 변해도 정확도가 크게 움직이므로 지연에 더 민감한 가중치가 필요합니다.

**층 단위 미세 조정**은 NetAdapt를 보완해 씁니다. 매 단계에서 확장 층이나 bottleneck 채널 수를 줄이는 제안들을 만들고(모든 층은 지연을 최소 $$\delta$$만큼 줄여야 하며 $$\delta = 0.01 \lvert L \rvert$$), 각 제안을 잠깐 fine-tuning한 뒤 **$$\Delta Acc / \Delta Latency$$가 가장 큰 제안**을 고릅니다. 원래 NetAdapt는 정확도 변화만 보지만, 지연당 정확도 변화로 바꾸면 모든 제안이 같은 지연 절감을 주는 것이 아니라는 점을 반영할 수 있습니다. 목표 지연에 도달할 때까지 반복합니다.

### 비싼 층 재설계

탐색 결과를 보면 마지막 몇 층과 첫 층이 유독 비쌉니다.

- **마지막 단계.** V2 구조에서는 7×7 해상도에서 1×1 conv로 960 → 320, 다시 1280으로 확장한 뒤 pooling합니다. V3는 **average pooling을 앞으로 옮겨** 1×1 conv로 960채널을 만든 직후 1×1 해상도로 줄이고, 그 뒤에 1280채널 층을 둡니다. 특징 추출이 7×7이 아니라 1×1에서 이뤄지므로 앞의 bottleneck 블록의 투영과 depthwise 층도 필요 없어집니다. 이것만으로 지연 7ms(11%), 30M MAdds가 줄고 정확도는 거의 그대로입니다.
- **첫 층.** 32개 필터 대신 **16개 필터 + h-swish**를 쓰면 정확도를 유지하며 2ms, 10M MAdds를 아낍니다. 32개 필터는 서로 거울상인 경우가 많아 중복이라는 관찰입니다.

### h-swish

swish $$x \cdot \sigma(x)$$는 정확도를 올리지만 시그모이드가 모바일에서 비쌉니다. 그래서 시그모이드를 조각 선형 함수로 바꿉니다.

$$
\text{h-swish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}
$$

장점은 셋입니다. 정확도 차이가 거의 없고, **조각 선형이라 양자화 때 수치 정밀도 손실이 없으며**, ReLU6 하나로 구현되어 메모리 접근을 줄입니다. 비선형성의 비용은 텐서 크기에 비례하므로, 해상도가 큰 앞쪽 층에서는 ReLU를 쓰고 **뒤쪽 절반에서만 h-swish**를 씁니다.

### Squeeze-and-Excite

[SENet]({% post_url 2026-06-02-cnn-architectures %})의 SE 블록을 MnasNet처럼 bottleneck 안에 넣되, SE의 bottleneck 채널을 확장 층의 **1/4**로 고정하고 시그모이드 대신 h-sigmoid를 씁니다. 파라미터는 조금 늘지만 지연 증가 없이 정확도가 오릅니다.

### 분할용 헤드: LR-ASPP

분할에서는 Lite R-ASPP를 제안합니다. SE처럼 큰 stride의 average pooling으로 전역 정보를 뽑아 곱하고, 저해상도 특징과 고해상도 특징을 1×1 conv로 합칩니다. Cityscapes에서 V2 기반 R-ASPP와 비슷한 mIOU로 34% 빠릅니다.

## 3. 코드로 보기: h-swish, SE, 마지막 단계

수식만 보면 작아 보이는 선택들이 실제로 어느 정도 차이인지 숫자로 확인합니다.

```python
import numpy as np
x = np.linspace(-8, 8, 16001)
sigmoid = 1 / (1 + np.exp(-x)); swish = x * sigmoid
h_sigmoid = np.clip(x + 3, 0, 6) / 6; h_swish = x * h_sigmoid
print("hard approximations on [-8, 8]")
print(f"  max |h-sigmoid - sigmoid| = {np.abs(h_sigmoid - sigmoid).max():.4f}")
print(f"  max |h-swish  - swish|    = {np.abs(h_swish - swish).max():.4f}   (swish min = {swish.min():.4f}, h-swish min = {h_swish.min():.4f})")
print("  h-swish is piecewise linear: exact in 8-bit fixed point, no exp/division")

# squeeze-and-excite with 1/4 reduction
def se_params(C, r=4): return C * (C // r) + C // r + (C // r) * C + C
print()
print("SE block params (reduction 1/4):", ", ".join(f"C={c}: {se_params(c):,}" for c in (72, 120, 480, 672)))

# efficient last stage (MobileNetV3-Large, 7x7 input, 160 channels)
S, C = 7, 160
orig = S*S*C*960 + S*S*960*9 + S*S*960*320 + S*S*320*1280 + 1280*1000
eff = S*S*C*960 + 960*1280 + 1280*1000
print()
print("last stage MAdds")
print(f"  original  (1x1 960 -> dw3x3 -> 1x1 320 -> 1x1 1280 at 7x7 -> pool -> fc): {orig/1e6:.1f}M")
print(f"  efficient (1x1 960 at 7x7 -> pool -> 1x1 1280 -> fc):                     {eff/1e6:.1f}M")
print(f"  saved: {(orig-eff)/1e6:.1f}M MAdds   (paper: ~30M MAdds, 7 ms = 11% of latency)")

# where h-swish pays off: cost is per-element, so it is cheap only where the tensor is small (deep layers)
print()
print("activation elements per layer (batch 1): h-swish is per element, so its cost tracks this column")
for res, c in [(112, 16), (56, 24), (28, 40), (14, 80), (14, 112), (7, 160)]:
    print(f"  {res:>3}x{res:<3} x {c:>3} ch -> {res*res*c:>9,} elements")
```

```text
hard approximations on [-8, 8]
  max |h-sigmoid - sigmoid| = 0.0692
  max |h-swish  - swish|    = 0.1423   (swish min = -0.2785, h-swish min = -0.3750)
  h-swish is piecewise linear: exact in 8-bit fixed point, no exp/division

SE block params (reduction 1/4): C=72: 2,682, C=120: 7,350, C=480: 115,800, C=672: 226,632

last stage MAdds
  original  (1x1 960 -> dw3x3 -> 1x1 320 -> 1x1 1280 at 7x7 -> pool -> fc): 44.4M
  efficient (1x1 960 at 7x7 -> pool -> 1x1 1280 -> fc):                     10.0M
  saved: 34.3M MAdds   (paper: ~30M MAdds, 7 ms = 11% of latency)

activation elements per layer (batch 1): h-swish is per element, so its cost tracks this column
  112x112 x  16 ch ->   200,704 elements
   56x56  x  24 ch ->    75,264 elements
   28x28  x  40 ch ->    31,360 elements
   14x14  x  80 ch ->    15,680 elements
   14x14  x 112 ch ->    21,952 elements
    7x7   x 160 ch ->     7,840 elements
```

- h-sigmoid는 시그모이드와 최대 0.07, h-swish는 swish와 최대 0.14 차이 납니다. 모양은 거의 같고 모두 조각 선형이라 8비트 고정소수점에서 정확히 표현됩니다.
- 마지막 단계 재설계로 7×7 해상도에서 수행되던 15M + 20M MAdds가 사라지고 34M가 절감됩니다. 논문이 측정한 30M MAdds, 7ms와 같은 차수입니다.
- 비선형성 비용은 원소 수에 비례합니다. 112×112×16 층은 7×7×160 층보다 원소가 25배 많으므로, h-swish를 앞쪽에 넣으면 정확도 이득보다 지연 비용이 큽니다. 뒤쪽 절반에만 쓰는 이유입니다.

## 4. 실험

**ImageNet.** Pixel 1 단일 코어, TFLite 벤치마크 기준입니다.

| 모델 | Top-1 | MAdds | Params | P-1 지연 |
|---|---|---|---|---|
| V3-Large 1.0 | 75.2% | 219M | 5.4M | 51ms |
| MnasNet-A1 | 75.2% | 315M | 3.9M | 71ms |
| V2 1.0 | 72.0% | 300M | 3.4M | 64ms |
| V3-Small 1.0 | 67.4% | 56M | 2.5M | 15.8ms |
| V2 0.35 | 60.8% | 59.2M | 1.6M | 16.6ms |

V3-Large는 V2보다 **3.2% 정확하면서 20% 빠르고**, V3-Small은 비슷한 지연의 V2 0.35보다 6.6% 정확합니다. 양자화해도 Large 73.8%/44ms 대 V2 70.9%/52ms로 우위가 유지됩니다. 해상도 조절이 width multiplier보다 좋은 트레이드오프를 주지만, 해상도는 과제가 정하는 경우가 많아 항상 쓸 수 있는 손잡이는 아닙니다.

**Ablation.** 비선형성 위치와 최적화된 구현의 효과입니다.

- ReLU만: 74.5%(-0.7%), 지연 -1%
- h-swish를 16채널 층부터: 75.4%(+0.2%), 최적화 구현 +4% 지연, 미최적화 구현은 68.9ms로 크게 느려짐
- h-swish를 112채널 층부터(기본보다 늦게): 75.0%(-0.3%)
- 최적화된 h-swish는 ReLU 대비 1ms만 추가하고, 미최적화 구현 대비 6ms(10% 이상)를 아낌

Figure 9의 개발 진행을 보면 MnasNet 75.2% → +SE/4 75.5% → +h-swish 76.1% → +NetAdapt와 compact 재설계로 정확도를 75.2%로 유지하면서 지연이 크게 줄어듭니다.

**검출과 분할.** SSDLite COCO에서 V3-Large는 V2와 같은 22.0 mAP를 137ms에, C4~C5 사이 채널을 절반으로 줄인 V3-Large†는 119ms에 냅니다(V2 162ms, 27% 빠름). 분할에서 V3-Large + LR-ASPP는 Cityscapes에서 V2 R-ASPP와 비슷한 mIOU로 34% 빠릅니다.

## 5. 해석

**1. 탐색과 수작업은 경쟁이 아니라 분업입니다.** NAS가 찾은 구조에서 사람이 비싼 층을 보고 고친 두 수정(마지막 단계, 첫 층)이 합쳐서 9ms를 아꼈습니다. 탐색 공간 안에 없는 변경은 탐색이 찾을 수 없고, 사람은 탐색 결과를 읽고 공간 밖의 수정을 떠올릴 수 있습니다.

**2. 측정 가능한 목적 함수가 설계를 바꿉니다.** $$w$$를 모델 크기에 따라 달리 둔 것, NetAdapt의 기준을 지연당 정확도로 바꾼 것, 비선형성을 텐서 크기에 따라 선택한 것 모두 "실제 지연"을 기준으로 삼았기 때문에 가능한 결정입니다. FLOPs만 봤다면 h-swish를 전 층에 넣었을 것입니다.

**3. h-swish는 양자화 친화 설계의 좋은 예입니다.** [양자화]({% post_url 2026-06-15-pruning-and-quantization %})에서 비선형 함수는 룩업 테이블이나 정밀도 손실의 원인입니다. 조각 선형으로 바꾸면 정수 연산만으로 정확하게 계산됩니다. 저자들이 TFLite에 h-swish 연산자를 기여했다는 점이 보여 주듯, 모델 설계와 런타임 최적화는 함께 가야 합니다.

**4. SE 1/4 고정은 작은 선택이지만 반복 등장합니다.** 확장 채널의 1/4라는 비율은 이후 EfficientNet, [EdgeViTs]({% post_url 2026-08-25-edgevits %}) 같은 경량 모델에서도 기본값처럼 쓰입니다. 파라미터는 늘지만 7×7 등 작은 해상도에서는 지연 비용이 작기 때문입니다.

**5. 평가를 읽을 때 주의할 점.** 모든 지연은 Pixel 폰 단일 큰 코어 기준이고, 멀티코어나 GPU·NPU에서는 depthwise conv와 SE의 상대 비용이 다릅니다. NAS 비용(TPU Pod 학습)은 논문에 상세히 나오지 않아 재현 비용을 가늠하기 어렵습니다. 위 toy의 마지막 단계 계산은 논문 Figure 5의 구조를 단순화한 추정이라 30M과 정확히 일치하지는 않습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 탐색 | 플랫폼 인식 NAS(블록 구조, $$w = -0.07/-0.15$$) + NetAdapt(층별 채널, $$\Delta Acc / \Delta Latency$$) |
| 수작업 | 마지막 단계 pooling 선행(-7ms, -30M MAdds), 첫 층 16필터(-2ms) |
| 비선형성 | h-swish $$= x \cdot \text{ReLU6}(x+3)/6$$, 뒤쪽 절반에만 적용 |
| 블록 | V2 inverted residual + SE(1/4, h-sigmoid) |
| 결과 | Large 75.2% / 51ms(V2 대비 +3.2%, -20%), Small 67.4% / 15.8ms |
| 다운스트림 | SSDLite 27% 빠름, LR-ASPP 34% 빠름(동급 정확도) |

## References

- Howard et al., "Searching for MobileNetV3," ICCV 2019. [arXiv:1905.02244](https://arxiv.org/abs/1905.02244)
- Tan et al., "MnasNet: Platform-Aware Neural Architecture Search for Mobile," CVPR 2019. [arXiv:1807.11626](https://arxiv.org/abs/1807.11626)
- Yang et al., "NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications," ECCV 2018. [arXiv:1804.03230](https://arxiv.org/abs/1804.03230)
- Ramachandran, Zoph, Le, "Searching for Activation Functions," 2017. [arXiv:1710.05941](https://arxiv.org/abs/1710.05941)
- Hu, Shen, Sun, "Squeeze-and-Excitation Networks," CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
