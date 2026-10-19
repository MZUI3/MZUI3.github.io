---
title: "[Pruning & Quantization] 모델 압축 입문: Pruning과 Quantization의 기본 원리"
date: 2026-06-15 21:21:00 +0900
categories: [Neural Network, Lightweight]
tags: [pruning, quantization, model-compression, lottery-ticket, int8, ste, lightweight]
description: 모델 압축의 두 축인 Pruning(magnitude pruning, CSR, Lottery Ticket)과 Quantization(affine 양자화, PTQ·QAT, STE)의 원리를 PyTorch로 직접 확인하고, 실제 속도 이득이 나는 조건까지 정리합니다.
math: true
---

> 모델 압축 시리즈의 첫 글입니다. 이 글에서 Pruning과 Quantization의 기본 개념을 잡고, Quantization의 극단인 **이진 신경망(BNN)**은 [BNN 총정리 글]({% post_url 2026-06-17-binary-neural-networks %})에서 XNOR-Net부터 ReActNet까지 이어갑니다.

## 1. 모델을 가볍게 만드는 방법들

모델 경량화는 크게 두 갈래로 나뉩니다.

**구조 자체를 바꾸기**
- 효율적인 블록 설계: ResNet bottleneck, SqueezeNet
- 효율적인 합성곱: MobileNet, ShuffleNet ([이전 글]({% post_url 2026-05-30-mobilenet %}) 참고)
- 자동 탐색(NAS): NetAdapt, MnasNet

**이미 있는 모델의 파라미터 줄이기** ← 이 글의 주제
- **Pruning(가지치기)**: 중요하지 않은 가중치를 0으로 만들어 제거
- **Quantization(양자화)**: 32bit 부동소수점을 8bit, 4bit 등으로 줄임
- **Binarization(이진화)**: 가중치와 활성값을 {-1, +1}로만 표현

두 방법은 서로 배타적이지 않아서, Pruning으로 개수를 줄이고 Quantization으로 각각의 크기를 줄이는 식으로 함께 쓸 수 있습니다.

---

## Part 1. Pruning

## 2. Pruning이란

**영향력이 작은 가중치를 삭제해서, 적은 파라미터로 성능 저하가 적은 모델을 만드는 방법**입니다. 목표는 가중치 행렬을 **sparse(희소)**하게 만드는 것입니다.

가장 널리 알려진 방식은 Han et al.(2015)의 **magnitude-based pruning**입니다. "절댓값이 작은 가중치는 출력에 미치는 영향도 작다"는 가정에서 출발합니다.

### 3단계 파이프라인

1. **Train**: 일반적인 방식으로 네트워크를 학습해 어떤 연결이 중요한지 알아낸다.
2. **Prune**: threshold보다 절댓값이 작은 가중치를 0으로 만든다.
3. **Retrain(Fine-tune)**: 남은 가중치를 다시 학습해 제거된 연결의 몫을 보상한다.

2–3단계를 반복할수록 더 높은 비율까지 가지치기할 수 있습니다.

```python
import torch

def magnitude_prune_(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """절댓값 하위 sparsity 비율의 가중치를 0으로 만들고 mask를 반환"""
    k = int(weight.numel() * sparsity)
    if k == 0:
        return torch.ones_like(weight, dtype=torch.bool)
    threshold = weight.abs().flatten().kthvalue(k).values
    mask = weight.abs() > threshold
    weight.data.mul_(mask)
    return mask

# 재학습 중에도 잘린 가중치가 되살아나지 않도록, 매 step 이후 mask를 다시 곱해준다.
# weight.data.mul_(mask)

torch.manual_seed(0)
w = torch.randn(256, 256)
mask = magnitude_prune_(w, 0.9)
print((w == 0).float().mean().item())    # 0.8999... → 90%가 0
```

PyTorch에는 `torch.nn.utils.prune` 모듈로 같은 기능이 이미 들어 있습니다.

```python
import torch.nn as nn
import torch.nn.utils.prune as prune

lin = nn.Linear(256, 256)
prune.l1_unstructured(lin, name="weight", amount=0.9)    # 절댓값 하위 90% 제거
print([n for n, _ in lin.named_parameters()])           # ['bias', 'weight_orig']
print([n for n, _ in lin.named_buffers()])              # ['weight_mask']
print(lin.weight.shape)                                 # torch.Size([256, 256]) 크기는 그대로
```

결과를 보면 `prune`은 가중치를 실제로 지우지 않고, 원본(`weight_orig`)과 0/1 mask(`weight_mask`)를 따로 들고 있다가 forward 때 곱합니다. 즉 **메모리는 오히려 늘고 속도도 그대로**입니다. 학습이 끝나면 `prune.remove(lin, "weight")`로 mask를 가중치에 합치고, 실제 이득은 아래의 희소 저장 형식이나 structured pruning으로 얻어야 합니다.

## 3. 결과

| 모델 | 원본 파라미터 | Pruning 후 | 압축률 | 정확도 변화 |
|---|---|---|---|---|
| AlexNet | 61M | 6.7M | 9× | 없음 |
| VGG-16 | 138M | 10.3M | 13× | 없음 |

특히 파라미터 대부분을 차지하는 **FC layer**가 많이 잘려 나갑니다.

## 4. 잘라낸 0을 어떻게 저장하나: Sparse Matrix와 CSR

가중치를 0으로 바꾸기만 하면 행렬 크기는 그대로이고, 0과의 곱셈도 여전히 수행됩니다. 실제 이득을 보려면 **0이 아닌 값만 저장**해야 합니다.

대표적인 형식이 **CSR(Compressed Sparse Row)**입니다. 행렬을 세 개의 배열로 나눕니다.

- `data`: 0이 아닌 값들
- `col`: 각 값의 열 위치
- `row_ptr`: 각 행이 `data`의 어디서 시작하는지

0이 아닌 값이 $$a$$개, 행이 $$n$$개라면 저장 공간은 $$2a + (n+1)$$이 됩니다. 원래 크기보다 충분히 희소할 때 이득입니다.

> 주의할 점: 이렇게 개별 가중치를 자르는 **unstructured pruning**은 희소 연산을 지원하는 하드웨어나 라이브러리가 없으면 실제 속도가 빨라지지 않을 수 있습니다. 필터나 채널 단위로 통째로 자르는 **structured pruning**은 압축률은 낮지만 일반 GPU에서도 바로 빨라집니다.

## 5. Pruning 성능에 영향을 주는 요소

### Regularization: L1 vs L2

- **L1**: 작은 가중치를 0으로 보내는 성질이 있어 **재학습 없이 Pruning한 직후**의 정확도가 좋습니다.
- **L2**: 가중치를 고르게 작게 유지해 **재학습까지 마친 뒤**의 정확도가 더 좋습니다.

### Dropout 비율 조정

Pruning으로 연결이 줄었으면 모델 용량도 줄어든 것이므로, 재학습할 때 Dropout 비율도 낮춰야 합니다. Han et al.은 다음 식을 제안합니다.

$$
D_r = D_o \sqrt{\frac{C_{ir}}{C_{io}}}
$$

($$D_o$$: 원래 dropout 비율, $$C_{io}, C_{ir}$$: Pruning 전후의 연결 수)

### 가중치를 새로 초기화하지 말 것

재학습할 때 살아남은 가중치를 무작위로 다시 초기화하면 성능이 떨어집니다. 층끼리 서로 맞춰 학습된 상태(**co-adaptation**)를 깨뜨리기 때문입니다. **Pruning 전의 가중치를 그대로 이어서 학습**해야 합니다.

### Iterative Pruning

한 번에 90%를 자르는 **one-shot** 방식보다, 일정 비율씩 자르고 재학습하는 과정을 반복하는 **iterative** 방식이 훨씬 높은 비율까지 정확도를 유지합니다.

## 6. Lottery Ticket Hypothesis

Frankle & Carbin(2019)은 흥미로운 질문을 던집니다. "Pruning으로 찾은 작은 네트워크를 **처음부터** 학습시켜도 똑같이 좋을까?"

결론은 이렇습니다.

> 무작위로 초기화된 큰 네트워크 안에는, **원래 초기값 그대로** 따로 학습시켜도 원본만큼(또는 더 빠르게) 학습되는 작은 부분 네트워크, 즉 **"당첨 복권(winning ticket)"**이 있다.

핵심은 **구조만이 아니라 초기값도 함께** 중요하다는 점입니다. 같은 구조라도 무작위로 다시 초기화하면 성능이 떨어집니다. 이 연구는 "큰 모델이 학습이 잘되는 이유는 좋은 복권을 뽑을 기회가 많기 때문"이라는 해석으로 이어졌습니다.

> 다만 이 결과는 MNIST·CIFAR-10 규모의 비교적 작은 네트워크에서 가장 깔끔하게 성립했습니다. ResNet-50 같은 큰 모델에서는 "맨 처음 초기값"이 아니라 **학습 초기 몇 iteration 뒤의 가중치로 되돌려야(rewinding)** winning ticket이 나왔습니다(Frankle et al., 2020). 또 Liu et al.(2019)은 structured pruning으로 얻은 구조는 **무작위로 다시 초기화해 처음부터 학습해도** 성능이 비슷하다고 보고해, "pruning의 가치는 가중치보다 구조를 찾는 데 있다"는 반대 관점을 제시했습니다.
{: .prompt-info }

---

## Part 2. Quantization

## 7. Quantization이란

연속적인 값(또는 매우 큰 값의 집합)을 **유한한 개수의 이산값**으로 제한하는 과정입니다. 아날로그 신호를 디지털로 바꾸는 ADC와 같은 개념입니다.

딥러닝에서는 보통 **FP32 가중치·활성값을 INT8 같은 저비트 정수로** 바꾸는 것을 말합니다.

### 왜 하나

- **메모리 감소**: FP32 → INT8이면 모델 크기가 1/4
- **연산 속도 향상**: 정수 연산은 부동소수점보다 빠르고 전력도 적게 듭니다
- **메모리 대역폭 절약**: 실제 추론에서는 이게 가장 큰 병목인 경우가 많습니다

### 대가

비트를 줄인 만큼 **정보 손실**이 생깁니다. 비트를 많이 쓸수록 손실은 줄지만 이득도 줄어듭니다. 또 모든 layer가 양자화에 똑같이 강하지는 않아서, 민감한 층(첫 층, 마지막 층 등)은 높은 정밀도로 남겨두기도 합니다.

## 8. 어떻게 양자화하나: Affine Quantization

가장 기본적인 방식은 실수 범위 $$[\alpha, \beta]$$를 정수 범위로 **선형 변환(+ bias)**하는 것입니다.

$$
q = \text{clip}\left(\text{round}\left(\frac{x}{S}\right) + Z,\ q_{\min},\ q_{\max}\right),\qquad
\hat{x} = S\,(q - Z)
$$

- $$S$$ (scale): $$\dfrac{\beta - \alpha}{2^b - 1}$$
- $$Z$$ (zero-point): 실수 0이 대응되는 정수값

종류는 두 가지입니다.

- **Asymmetric**: $$Z \neq 0$$. ReLU 출력처럼 한쪽으로 치우친 분포에 유리
- **Symmetric**: $$Z = 0$$, 범위를 $$[-\max\lvert x\rvert, \max\lvert x\rvert]$$로 둡니다. 절댓값이 가장 큰 값을 127로 매핑하는 방식으로, 계산이 단순해 가중치 양자화에 흔히 씁니다.

```python
import torch

def fake_quantize(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Symmetric per-tensor quantize → dequantize"""
    qmax = 2 ** (num_bits - 1) - 1                 # 127 for int8
    scale = x.abs().max().clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale                               # 양자화 오차가 반영된 실수값

torch.manual_seed(0)
w = torch.randn(256, 256)
print((w - fake_quantize(w, 8)).abs().mean())      # tensor(0.0090)  8bit 오차
print((w - fake_quantize(w, 4)).abs().mean())      # tensor(0.1625)  4bit 오차 (약 18배)
```

비트를 절반으로 줄였을 뿐인데 오차는 약 18배가 됩니다. 정수 레벨 수가 $$2^8 = 256$$개에서 $$2^4 = 16$$개로 16배 줄기 때문입니다. 4bit 이하에서 PTQ만으로는 정확도가 크게 떨어지고 QAT가 필요해지는 이유입니다.

### Clipping 범위 정하기

$$\max\lvert x\rvert$$를 그대로 쓰면 이상치(outlier) 하나 때문에 scale이 커져서 대부분의 값이 몇 개의 정수에 몰립니다. 그래서 상위 0.01%를 잘라내거나(percentile), KL divergence가 최소가 되는 범위를 찾는(TensorRT의 calibration) 방법을 씁니다.

## 9. 언제 양자화하나: PTQ vs QAT

### Post-Training Quantization (PTQ)

FP32로 학습을 끝낸 뒤 가중치(와 활성값)를 양자화합니다.

- 재학습이 필요 없어 빠르고 간단
- 파라미터가 많은 큰 모델은 정확도 하락이 작음
- 작은 모델이나 4bit 이하에서는 정확도가 크게 떨어질 수 있음

PTQ는 다시 둘로 나뉩니다.

- **Dynamic**: 가중치만 미리 양자화하고, 활성값은 추론할 때 그때그때 범위를 계산해 양자화
- **Static**: 보정용 데이터(calibration set)로 활성값 범위를 미리 구해 두고 둘 다 양자화

### Quantization-Aware Training (QAT)

학습 중에 **양자화 → 역양자화(fake quantization)**를 forward에 끼워 넣어, 모델이 양자화 오차를 미리 겪으면서 적응하게 만듭니다. 정확도 하락이 가장 작지만 학습 비용이 듭니다.

## 10. 미분 불가능 문제와 STE

`round()`는 계단 함수라서 거의 모든 점에서 미분값이 0이고, 경계에서는 미분이 정의되지 않습니다. 그대로면 QAT에서 gradient가 흐르지 않습니다.

해결책은 **Straight-Through Estimator(STE)**입니다. forward에서는 반올림을 하고, backward에서는 **round가 없었던 것처럼 gradient를 1로 그대로 통과**시킵니다.

$$
\frac{\partial \hat{x}}{\partial x} \approx 1 \quad (\text{clipping 범위 안에서})
$$

```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output          # gradient를 그대로 통과
```

실무에서는 `x + (torch.round(x) - x).detach()` 한 줄로도 같은 효과를 냅니다. 이 STE는 다음 글에서 다룰 **BNN**에서도 핵심 도구로 쓰입니다.

## 11. 그 밖의 기법들

- **Non-uniform quantization**: 가중치 분포가 0 근처에 몰려 있으므로 log scale 등으로 0 근처에 더 촘촘하게 레벨을 배치
- **Mixed precision**: 민감한 층은 8bit, 둔감한 층은 4bit처럼 층마다 비트 수를 다르게 설정
- **Learnable quantization**: clipping 범위나 scale 자체를 학습 가능한 파라미터로 둠 (PACT, LSQ 등)
- **Binary quantization**: 1bit까지 줄이는 극단적인 경우 → BinaryConnect, XNOR-Net, ReActNet

## 12. 정리

| | Pruning | Quantization |
|---|---|---|
| 줄이는 것 | 파라미터 **개수** | 파라미터 하나의 **비트 수** |
| 핵심 기준 | 가중치 크기(magnitude) | 값의 범위(scale, zero-point) |
| 재학습 | Fine-tuning 필수에 가까움 | PTQ는 불필요, QAT는 필요 |
| 실제 가속 조건 | 희소 연산 지원 또는 structured pruning | INT 연산 지원 하드웨어 |
| 대표 연구 | Han et al. 2015, Lottery Ticket | Google INT8, PTQ/QAT, STE |

다음 글에서는 Quantization을 극단까지 밀어붙인 **XNOR-Net**을 다룹니다. 가중치와 활성값을 모두 ±1로 만들면 곱셈이 **XNOR + popcount** 비트 연산으로 바뀌는데, 이 과정에서 생기는 정확도 손실을 어떻게 줄였는지 살펴보겠습니다.

## References

- Han et al., "Learning both Weights and Connections for Efficient Neural Networks," NeurIPS 2015. [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)
- Frankle et al., "Linear Mode Connectivity and the Lottery Ticket Hypothesis," ICML 2020. [arXiv:1912.05671](https://arxiv.org/abs/1912.05671)
- [PyTorch – Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- Liu et al., "Rethinking the Value of Network Pruning," ICLR 2019.
- Frankle & Carbin, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks," ICLR 2019.
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018.
- Gholami et al., "A Survey of Quantization Methods for Efficient Neural Network Inference," 2021.
- Rastegari et al., "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks," ECCV 2016.
