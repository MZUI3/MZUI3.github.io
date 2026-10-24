---
title: "[CNN] Big-Little Net: 저해상도에는 큰 가지, 고해상도에는 작은 가지를 두는 다중 스케일 설계"
date: 2026-10-24 02:00:00 +0900
categories: [Neural Network, CNN]
tags: [cnn, big-little-net, multi-scale, resnet, resnext, efficient-cnn, speech-recognition, lightweight]
description: Big-Little Net(Chen et al., ICLR 2019)이 ResNet 블록을 저해상도·고용량의 Big-Branch와 고해상도·저용량의 Little-Branch로 나누고 더해서, ResNet/ResNeXt/SEResNeXt에서 연산량을 1.4~2.3배 줄이며 정확도를 올린 설계를 정리합니다. 작은 계산으로 α, β에 따른 가지별 비용 비율을 확인합니다.
math: true
---

> 이미지 인식에서 다중 스케일 특징은 중요하지만, 고해상도 feature map에 깊은 네트워크를 그대로 적용하면 비쌉니다. **Big-Little Net**(Chen, Fan, Xu, Yan, Kalantidis, Rohrbach, Yan, Feris, IBM Research·MIT-IBM Watson AI Lab, ICLR 2019)의 아이디어는 단순합니다. **해상도가 낮은 입력에는 크고 깊은 가지(Big-Branch)를, 해상도가 높은 입력에는 얇고 얕은 가지(Little-Branch)를** 두고, 둘을 더해서 다음 단계로 보냅니다. [ResNet]({% post_url 2026-05-28-resnet %}), ResNeXt, SEResNeXt에 그대로 끼워 넣어 FLOPs를 최대 2배 이상 줄이면서 top-1 오류를 낮췄습니다.

## 1. 문제: 모든 스케일에 같은 용량을 쓸 필요가 있는가

기존 다중 스케일 설계(FPN, HRNet 계열 등)는 여러 해상도의 특징을 만들되 각 해상도에 비슷한 용량의 네트워크를 둡니다. 그런데 고해상도 특징은 **세부 정보**를, 저해상도 특징은 **의미 정보**를 담습니다. 의미 정보를 뽑는 데는 깊은 네트워크가 필요하지만, 세부 정보를 보존하는 데는 얕은 네트워크로 충분할 수 있습니다. 그리고 비용은 해상도의 제곱에 비례하므로, 비싼 연산을 저해상도로 옮기면 전체 비용이 크게 줄어듭니다.

## 2. 방법

### bL-module

ResNet의 한 단계(같은 해상도의 블록 묶음)를 두 가지로 나눕니다.

- **Big-Branch.** 입력을 1/2 해상도로 줄인 뒤 원래 단계와 같은 폭, 같은 깊이(또는 더 깊게)로 처리합니다. 해상도가 1/4이므로 비용도 1/4입니다.
- **Little-Branch.** 원래 해상도에서 채널을 $$\alpha$$배 줄이고 층 수를 $$\beta$$배 줄여 처리합니다. 입력·출력 채널이 모두 $$1/\alpha$$이므로 conv 비용은 $$1/\alpha^2$$, 층 수까지 고려하면 $$1/(\alpha^2 \beta)$$입니다.

Big-Branch 출력은 업샘플링하고, Little-Branch 출력은 1×1 conv로 채널을 맞춘 뒤 **원소별 덧셈**으로 합칩니다. 논문은 결합 방식(덧셈, concatenation 등)과 가지 수 $$K > 2$$도 실험했지만 두 가지와 덧셈이 가장 효율적이었습니다. 논문 표현으로 $$\alpha = 2, \beta = 4$$에서 Little-Branch 비용은 전체의 1/6 이하이고, 네트워크 전체 비용은 baseline보다 30% 가까이 줄어듭니다. 첫 번째 stem과 마지막 단계는 나누지 않습니다.

### 왜 되는가

Little-Branch가 가볍다는 것은 고해상도 경로에 용량이 거의 없다는 뜻입니다. 그런데도 성능이 유지되거나 올라가는 이유는, Big-Branch가 저해상도에서 의미 특징을 충분히 뽑고 Little-Branch는 **세부 정보를 보존해 더해 주는 역할만** 하면 되기 때문입니다. 논문은 $$\alpha, \beta$$를 바꿔 가며 Little-Branch 비용이 전체의 15% 미만이어도 저해상도 표현을 잘 보완한다고 보고합니다.

## 3. 코드로 보기: 가지별 비용

28×28, 256채널, 6개 conv 층의 단계를 bL-module로 바꿨을 때 Big/Little 가지의 MAC 수가 baseline 대비 얼마인지 셉니다.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def count_macs(model, x):
    macs = []
    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            k = mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels // mod.groups
            macs.append(out.numel() // out.shape[0] * k)
    hs = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad(): model(x)
    for h in hs: h.remove()
    return sum(macs)

def stack(C_in, C, K):
    layers = [nn.Conv2d(C_in, C, 3, padding=1)] + [nn.Conv2d(C, C, 3, padding=1) for _ in range(K - 1)]
    return nn.Sequential(*[m for l in layers for m in (l, nn.ReLU())])

class BigLittle(nn.Module):
    def __init__(self, C, K, alpha, beta):
        super().__init__()
        self.big = stack(C, C, K)                             # low resolution, full width, deep
        self.little = stack(C, C // alpha, max(1, K // beta)) # high resolution, narrow, shallow
        self.proj = nn.Conv2d(C // alpha, C, 1)               # match channels before merging
    def forward(self, x):
        b = self.big(F.avg_pool2d(x, 2))
        b = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        return F.relu(b + self.proj(self.little(x)))          # merge by addition

C, K = 256, 6
x = torch.randn(1, C, 28, 28)
base = count_macs(stack(C, C, K), x)
print(f"baseline stage: {K} convs x {C}ch @28x28 -> {base/1e9:.3f} GMACs")
print("alpha  beta   big     little  total   ratio")
for alpha, beta in [(2, 4), (2, 2), (4, 4), (4, 2)]:
    m = BigLittle(C, K, alpha, beta)
    big = count_macs(m.big, F.avg_pool2d(x, 2)); lit = count_macs(nn.Sequential(m.little, m.proj), x)
    tot = count_macs(m, x)
    print(f"{alpha:>5}  {beta:>4}   {big/base:.3f}   {lit/base:.3f}   {tot/base:.3f}   {tot/base:.2f}x")
print()
print("paper, bL-ResNet-50 (alpha=2, beta=4): 2.85 GFLOPs vs 4.09 GFLOPs = 0.70x (stem and last stage are not split)")
```

```text
baseline stage: 6 convs x 256ch @28x28 -> 2.775 GMACs
alpha  beta   big     little  total   ratio
    2     4   0.250   0.093   0.343   0.34x
    2     2   0.250   0.176   0.426   0.43x
    4     4   0.250   0.046   0.296   0.30x
    4     2   0.250   0.067   0.317   0.32x

paper, bL-ResNet-50 (alpha=2, beta=4): 2.85 GFLOPs vs 4.09 GFLOPs = 0.70x (stem and last stage are not split)
```

- Big-Branch는 해상도 1/2이므로 정확히 baseline의 1/4입니다. Little-Branch는 $$\alpha = 2, \beta = 4$$에서 9%로, 합쳐서 **34%**입니다. 논문이 말한 "Little-Branch는 전체의 1/6 이하"가 여기서 0.093/0.343 ≈ 27%로 나오는데, 실제 네트워크에서는 1×1 투영과 병합 비용이 상대적으로 더 작기 때문입니다.
- $$\alpha$$를 4로 올리면 Little-Branch 비용이 더 줄지만 정확도 손실이 커집니다. 논문은 $$\alpha = 2, \beta = 4$$를 기본으로 잡았습니다.
- 전체 네트워크의 비율(bL-ResNet-50에서 0.70배)이 단계 하나의 비율(0.34배)보다 큰 것은 stem, 마지막 단계, 분류기는 나누지 않기 때문입니다.

## 4. 실험

**ImageNet, ResNet-50 백본.** $$\alpha, \beta$$ ablation에서 모든 bL-ResNet-50 변형이 baseline(top-1 오류 23.66%, 4.09 GFLOPs)보다 정확하면서 가볍습니다. 기본 설정 $$\alpha = 2, \beta = 4$$는 22.69%, 2.85 GFLOPs입니다. 변형 간 차이가 작아서 Little-Branch가 가벼워도 보완이 잘 된다는 것을 보여 줍니다.

**더 깊고 넓은 백본.** Tesla K80, 배치 16 기준 속도입니다.

| 모델 | Top-1 오류 | GFLOPs | 속도(ms/batch) |
|---|---|---|---|
| ResNet-101 | 21.95% | 7.80 | 186 |
| bL-ResNet-101 | 21.80% | 3.89 (2.01×) | 140 (1.33×) |
| ResNet-152 | 21.51% | 11.51 | 266 |
| bL-ResNet-152 | 21.16% | 5.04 (2.28×) | 178 (1.49×) |
| ResNeXt-101 (64×4d) | 20.73% | 15.46 | 485 |
| bL-ResNeXt-101 (64×4d) | 20.48% | 7.14 (2.17×) | 263 (1.98×) |
| SEResNeXt-101 (32×4d) | 21.00% | 7.97 | 376 |
| bL-SEResNeXt-101 (32×4d) | 20.87% | 4.08 (1.95×) | 235 (1.60×) |

깊은 모델일수록 절감이 큽니다. bL-ResNet-101은 FLOPs 2배 절감에 오류도 낮고, 절감된 FLOPs를 **입력 해상도 256**에 쓰면(bL-ResNet-152@256 20.34%, 6.58 GFLOPs) baseline보다 여전히 가벼우면서 1% 넘게 정확해집니다. 실제 속도 향상(1.3~2.0배)은 FLOPs 비율보다 작은데, 두 가지가 병렬 분기이고 병합 비용이 있기 때문입니다.

**음성 인식.** 같은 아이디어를 음성 인식용 CNN에 적용해, 연산량을 줄이면서 단어 오류율도 개선했다고 보고합니다.

## 5. 해석

**1. 해상도와 용량의 짝을 바꾼 것이 전부입니다.** 아이디어는 한 문장이지만, "비용은 해상도의 제곱, 의미 특징은 깊이"라는 두 사실을 정확히 겨냥했습니다. 위 계산처럼 Big-Branch 하나만으로 비용이 1/4로 떨어지고, 나머지는 Little-Branch로 세부를 보충하는 구조입니다.

**2. 어떤 백본에도 끼워 넣을 수 있는 모듈 설계입니다.** ResNet, ResNeXt, SE 블록을 바꾸지 않고 단계 수준에서만 나눴기 때문에, [CNN 아키텍처 계보]({% post_url 2026-06-02-cnn-architectures %})의 다른 개선(SE, group conv)과 그대로 결합됩니다. 표에서 SEResNeXt와 결합해도 절감 비율이 유지됩니다.

**3. 절감한 비용을 해상도에 재투자한다는 발상이 유용합니다.** @256 실험은 같은 FLOPs 예산에서 "더 큰 입력 + bL"이 "원래 입력 + 원래 네트워크"보다 낫다는 뜻입니다. EfficientNet의 복합 스케일링과 [MobileNetV3]({% post_url 2026-10-24-mobilenetv3 %})의 "해상도가 multiplier보다 좋은 트레이드오프"라는 관찰과 통합니다.

**4. 이후 설계와의 연결.** 고해상도 저용량 + 저해상도 고용량이라는 배치는 Octave Convolution(저주파/고주파 채널 분리), HRNet의 다중 해상도 병렬 스트림, [RevCol]({% post_url 2026-10-24-revcol %})의 다단계 열 구조와 비교해 볼 수 있습니다. 주파수 관점에서는 [MWCNN]({% post_url 2026-10-24-mwcnn %})처럼 저주파 성분에 연산을 집중하는 것과 같은 직관입니다.

**5. 평가를 읽을 때 주의할 점.** baseline을 모두 저자들이 재학습했기 때문에 원 논문 값과 조금 다릅니다. 속도는 K80, 배치 16 기준이라 최신 GPU나 배치 1의 모바일 환경에서는 두 가지의 병렬성과 병합 비용이 다르게 작용합니다. 위 toy는 단계 하나의 conv MAC만 센 것이고, 업샘플링·정규화·메모리 비용은 포함하지 않았습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 문제 | 모든 해상도에 같은 용량을 쓰면 고해상도 경로가 비용을 지배 |
| Big-Branch | 1/2 해상도, 원래 폭·깊이 → 비용 1/4 |
| Little-Branch | 원래 해상도, 채널 $$1/\alpha$$, 깊이 $$1/\beta$$ → 비용 $$1/(\alpha^2 \beta)$$ |
| 병합 | 업샘플 + 1×1 투영 후 원소별 덧셈, 기본 $$\alpha = 2, \beta = 4$$ |
| 결과 | bL-ResNet-50 22.69% / 2.85G(baseline 23.66% / 4.09G), bL-ResNet-152 FLOPs 2.28배 절감 |
| 활용 | 절감한 FLOPs를 해상도 256에 재투자하면 1% 이상 추가 향상 |

## References

- Chen et al., "Big-Little Net: An Efficient Multi-Scale Feature Representation for Visual and Speech Recognition," ICLR 2019. [arXiv:1807.03848](https://arxiv.org/abs/1807.03848)
- He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition," CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Xie et al., "Aggregated Residual Transformations for Deep Neural Networks," CVPR 2017. [arXiv:1611.05431](https://arxiv.org/abs/1611.05431)
- Chen et al., "Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution," ICCV 2019. [arXiv:1904.05049](https://arxiv.org/abs/1904.05049)
