---
title: "[Generative AI] Speed Is All You Need: 모바일 GPU에서 Stable Diffusion을 12초 안에 돌리는 커널 최적화"
date: 2026-10-23 13:20:00 +0900
categories: [Generative AI, Diffusion]
tags: [diffusion, stable-diffusion, on-device, mobile-gpu, fused-softmax, flashattention, winograd, group-norm, gelu, lightweight, edge-ai]
description: 모델을 바꾸지 않고 GPU 커널만 손봐 Samsung S23 Ultra에서 Stable Diffusion 1.4의 512×512 이미지 생성(20 step)을 12초 아래로 끌어내린 Speed Is All You Need(Chen et al., Google, CVPR 2023 Workshop)의 네 가지 최적화와 그 효과를 정리하고, Winograd와 partially fused softmax를 코드로 확인합니다.
math: true
---

> [Latent Diffusion Model]({% post_url 2026-07-03-latent-diffusion-models %})은 10억 개가 넘는 파라미터를 가진 모델입니다. 이를 서버가 아니라 **휴대폰에서** 돌리면 서버 비용이 들지 않고, 오프라인에서 동작하며, 데이터가 기기를 떠나지 않습니다. 문제는 연산과 메모리입니다. **Speed Is All You Need**(Chen, Sarokin et al., Google, CVPR 2023 Workshop)는 모델 구조나 양자화 없이 **GPU 커널 구현만으로** Stable Diffusion 1.4의 UNet 한 iteration 지연을 Samsung S23 Ultra에서 52%, iPhone 14 Pro Max에서 33% 줄이고, 512×512 이미지 20 step 생성을 **12초 아래**로 끌어내렸습니다. 5쪽짜리 짧은 논문이지만 [Edge AI]({% post_url 2026-06-19-edge-ai-hardware %})에서 다룬 "하드웨어를 아는 구현"이 무엇인지 보여 주는 좋은 사례입니다.

## 1. Stable Diffusion의 네 부품

텍스트에서 이미지를 만드는 추론은 네 부품을 거칩니다.

| 부품 | 역할 |
|---|---|
| Text embedder | CLIP으로 프롬프트 $$y$$를 임베딩 $$\tau_\theta(y)$$로 |
| Noise generation | latent 공간의 시작 노이즈 $$z$$ |
| **Denoising network** | UNet $$\epsilon_\theta(z_t, t, \tau_\theta(y))$$. cross-attention으로 텍스트 조건을 받으며 iteration마다 실행 |
| Image decoder | latent $$\hat z$$를 RGB $$\hat x = \mathcal{D}(\hat z)$$로 |

UNet이 iteration 수만큼 반복되므로 가장 비쌉니다. 논문의 최적화는 Stable Diffusion에 맞춰 설명하지만 다른 큰 diffusion 모델에도 그대로 적용됩니다.

## 2. 네 가지 GPU 최적화

### 전용 커널: Group Norm과 GELU

UNet 전체에 쓰이는 **Group Normalization**은 채널을 그룹으로 나눠 그룹별 평균과 분산으로 정규화합니다.

$$
\hat x_i = \frac{1}{\sigma_g}(x_i - \mu_g)
$$

보통 구현은 reshape → mean → variance → normalize를 **순서대로 별개의 연산**으로 실행하며 중간 텐서를 만듭니다. 이를 GPU shader 하나로 합쳐 **GPU 명령 한 번에** 중간 텐서 없이 끝냅니다.

**GELU**도 마찬가지입니다.

$$
\text{GELU}(x) = \frac{x}{2}\Big[1 + \text{erf}\big(x/\sqrt{2}\big)\Big]
$$

곱셈, 덧셈, 오차 함수에 더해 앞뒤의 split과 곱셈까지 전용 shader 하나의 draw call로 묶었습니다.

### Partially fused softmax

UNet 중간 층의 attention은 $$\text{softmax}(QK^\top/\sqrt d)\, V$$인데, $$Q \in \mathbb{R}^{N \times d}$$, $$K, V \in \mathbb{R}^{M \times d}$$에서 $$N, M \gg d$$라 $$A = QK^\top/\sqrt d$$가 거대합니다. softmax를 두 단계로 나눕니다.

1. **축소 연산**: 행별 최댓값 $$L$$과, 최댓값을 뺀 지수의 합 $$S$$

$$
L = \Big[\max_j a_{ij}\Big],\qquad S = \Big[\sum_j \exp(a_{ij} - \max_k a_{ik})\Big] \in \mathbb{R}^N
$$

2. **원소별 연산**: $$L$$과 $$S$$로 $$A$$를 정규화

큰 행렬 $$A$$ 전체에 softmax를 돌리는 대신, 축소 연산만 shader로 계산해 크기 **$$N \times 2$$**의 텐서를 만들고, 원소별 정규화는 **뒤따르는 $$V$$와의 행렬곱에 융합**합니다. 중간 텐서의 메모리와 전체 지연이 크게 줄어듭니다.

$$A$$에서 $$L, S$$로 가는 축소는 출력이 입력보다 훨씬 작아 병렬성이 제한됩니다. 그래서 $$A$$의 원소를 블록으로 묶어 블록마다 계산한 뒤 최종 결과로 다시 축소하는 **다단계** 구조로 병렬성을 높였고, 스레딩과 메모리 캐시를 세심하게 관리해 이 다단계도 GPU 명령 한 번에 끝냅니다.

### FlashAttention은 골라서

근사 attention(Reformer, Performer 등)은 품질을 내주지만 FlashAttention은 **정확한** attention입니다. 타일링으로 GPU의 HBM과 on-chip SRAM 사이의 읽기·쓰기를 줄이는 IO-aware 알고리즘입니다. 다만 커널이 **레지스터를 많이 씁니다**. 그래서 Adreno와 Apple GPU에서 attention 차원 **$$d = 40$$인 경우에만** FlashAttention을 쓰고, 나머지는 partially fused softmax를 씁니다.

### Winograd convolution

Winograd는 합성곱을 행렬곱의 연속으로 바꾸고, 변환 행렬을 잘 고르면 필요한 곱셈 수를 줄입니다. 대신 메모리와 수치 오차가 늘고, 타일이 클수록 심합니다. Stable Diffusion은 3×3 합성곱에 크게 의존하고, 특히 **image decoder는 90% 이상이 3×3 합성곱**입니다.

| 타일 크기 | FLOPs 절감 | 중간 텐서 메모리 | 가중치 메모리 |
|---|---|---|---|
| 2×2 | 2.25배 | 4배 | 1.77배 |
| **4×4** | **4배** | **2.25배** | **4배** |
| 6×6 | 5.06배 | 1.8배 | 7.12배 |
| 8×8 | 5.76배 | 1.56배 | 11.12배 |

연산 효율과 메모리의 균형점으로 **4×4 타일**을 골랐고, 이득이 나는 곳에만 휴리스틱으로 적용했습니다.

## 3. 코드로 보기: 곱셈 36번이 16번으로, 중간 텐서 100만 개가 2천 개로

Winograd $$F(2 \times 2, 3 \times 3)$$와 partially fused softmax가 원래 연산과 같은 결과를 내면서 무엇을 아끼는지 확인합니다.

```python
import torch, torch.nn.functional as F

torch.manual_seed(0)
# 1) Winograd F(2x2, 3x3): 2x2 출력 타일 하나를 3x3 커널로 계산할 때 곱셈 36번 대신 16번
Bt = torch.tensor([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=torch.float)
G = torch.tensor([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype=torch.float)
At = torch.tensor([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=torch.float)

def winograd_tile(d, g):                                     # d: 4x4 입력 타일, g: 3x3 커널
    U = G @ g @ G.T                                          # 커널 변환(가중치마다 한 번만 미리 계산)
    V = Bt @ d @ Bt.T                                        # 입력 변환(덧셈·뺄셈만)
    M = U * V                                                # 원소별 곱셈 16번 ← 유일한 곱셈
    return At @ M @ At.T                                     # 출력 변환(덧셈·뺄셈만)

img = torch.randn(1, 1, 8, 8); ker = torch.randn(1, 1, 3, 3)
direct = F.conv2d(img, ker)                                  # 6x6 출력
out = torch.zeros(6, 6)
for i in range(0, 6, 2):
    for j in range(0, 6, 2):
        out[i:i + 2, j:j + 2] = winograd_tile(img[0, 0, i:i + 4, j:j + 4], ker[0, 0])
print(f"Winograd F(2x2,3x3)와 직접 합성곱의 최대 차이: {(out - direct[0, 0]).abs().max():.1e}")
print(f"출력 타일당 곱셈: 직접 {2 * 2 * 9}번 → Winograd 16번 ({2 * 2 * 9 / 16:.2f}배 절감), 변환된 커널 크기: 9 → 16 ({16 / 9:.2f}배)")

# 2) Partially fused softmax: softmax(QKᵀ/√d)·V를 "행별 (최댓값 L, 지수합 S)"만 중간 텐서로 두고 V와의 곱에 정규화를 녹인다
N, M, d = 1024, 1024, 40
Q, K, V = torch.randn(N, d), torch.randn(M, d), torch.randn(M, d)
A = Q @ K.T / d ** 0.5
ref = torch.softmax(A, -1) @ V                               # 원래 구현: N×M 크기의 softmax 결과를 통째로 만든다
L = A.max(-1, keepdim=True).values                           # 축소 연산 1: 행 최댓값
S = torch.exp(A - L).sum(-1, keepdim=True)                   # 축소 연산 2: 지수합 → 중간 텐서는 N×2
fused = (torch.exp(A - L) @ V) / S                           # 정규화를 V 곱 뒤로 미룬다(실제 shader에서는 exp와 matmul을 한 커널에서)
print(f"부분 융합 softmax와 원래 구현의 최대 차이: {(fused - ref).abs().max():.1e}")
print(f"softmax 중간 텐서: 원래 N×M = {N * M:,}개 → 융합 후 N×2 = {N * 2:,}개")
```

```text
Winograd F(2x2,3x3)와 직접 합성곱의 최대 차이: 1.9e-06
출력 타일당 곱셈: 직접 36번 → Winograd 16번 (2.25배 절감), 변환된 커널 크기: 9 → 16 (1.78배)
부분 융합 softmax와 원래 구현의 최대 차이: 2.2e-07
softmax 중간 텐서: 원래 N×M = 1,048,576개 → 융합 후 N×2 = 2,048개
```

- Winograd의 곱셈은 $$U \odot V$$ 한 줄뿐이고 나머지 변환은 덧셈·뺄셈입니다. 2×2 타일에서 곱셈이 2.25배 줄고 변환된 커널이 1.78배 커지는데, 논문 Table 2의 2×2 행과 정확히 같은 숫자입니다. 4×4 타일이면 곱셈 4배 절감에 가중치 4배 증가입니다. 결과는 직접 합성곱과 $$10^{-6}$$ 이내로 같고, 타일이 커질수록 이 오차가 커지는 것이 큰 타일을 피하는 이유입니다.
- softmax의 정규화를 $$V$$ 곱 뒤로 미루면 결과가 같으면서 중간에 들고 있어야 할 것이 $$N \times M$$ 행렬에서 $$N \times 2$$ 벡터 둘로 줄어듭니다. 실제 shader는 $$\exp(A - L)$$조차 만들지 않고 행렬곱 커널 안에서 계산합니다.

## 4. 결과

UNet 한 iteration, 512×512 해상도, OpenCL 커널 기준입니다. 각 행은 이전 행의 최적화를 모두 포함합니다.

| 단계 | S23 Ultra 지연(ms) | 중간 텐서(MB) | 가중치(MB) | iPhone 14 Pro Max 지연(ms) |
|---|---|---|---|---|
| Baseline (공개 구현) | 1098 | 105 | 1640 | 1554 |
| + Opt. Softmax | 908 | 105 | 1640 | 1475 |
| + S-GN/GELU | 755 | 85 | 1640 | 1382 |
| + FlashAttention | 660 | 80 | 1639 | 1309 |
| + Winograd (All) | **525** | 84 | 2093 | **1043** |

- S23 Ultra(Adreno 740)에서 **52.2%**, iPhone 14 Pro Max(A16)에서 **32.9%** 지연 감소입니다.
- 가장 큰 단일 효과는 Winograd(S23에서 660→525ms)이지만, 가중치 메모리가 1,640MB에서 2,093MB로 늘었습니다. 변환된 커널이 커지기 때문입니다.
- 전용 커널(GN/GELU)은 중간 텐서를 105MB에서 85MB로 줄이며 속도도 올립니다. 중간 텐서를 만들지 않는 효과입니다.
- 끝에서 끝까지(텍스트 입력부터 디코딩된 이미지까지) S23 Ultra에서 20 step, 512×512 생성이 **12초 미만**입니다.

## 5. 해석

**1. 모델은 한 글자도 바꾸지 않았습니다.** 양자화도, 증류도, step 줄이기도 없습니다. 출력은 원래 Stable Diffusion과 같고(Winograd의 미세한 수치 오차만 제외), 얻은 것은 순수하게 **구현**에서 나왔습니다. 모바일 배포에서 "모델을 바꾸기 전에 구현부터 보라"는 교훈입니다. [Speculative Decoding]({% post_url 2026-10-23-speculative-decoding %})이나 [ParaDiGMS]({% post_url 2026-10-23-paradigms %})처럼 출력을 보존하는 가속이지만, 저쪽이 알고리즘이라면 이쪽은 커널입니다.

**2. 네 최적화의 공통 주제는 "메모리 왕복 줄이기"입니다.** GN/GELU 융합은 중간 텐서를 없애고, fused softmax는 $$N \times M$$ 행렬을 저장하지 않으며, FlashAttention은 HBM 접근을 줄입니다. Winograd만 연산 자체를 줄이는데, 그 대가로 메모리를 더 씁니다. 모바일 GPU는 대역폭이 데스크톱보다 훨씬 좁아서 이 주제가 더 중요합니다.

**3. 하드웨어마다 답이 다릅니다.** FlashAttention을 $$d = 40$$에서만 쓴 것은 레지스터 압박 때문이고, Winograd 타일을 4×4로 고른 것은 메모리 때문이며, 같은 최적화가 S23에서는 52%, iPhone에서는 33%의 효과를 냈습니다. 범용 해법 대신 프로파일링으로 기기별로 고르는 과정이 이 논문의 실제 내용입니다. [AdderNet 글]({% post_url 2026-07-24-addernet %})이나 [ASIP 글]({% post_url 2026-07-28-asip-posit %})에서 본 것처럼 알고리즘의 이득은 하드웨어가 받쳐 줄 때만 실현됩니다.

**4. 메모리 trade-off를 숨기지 않았습니다.** Winograd로 가중치가 450MB 늘어난 것을 표에 그대로 적었습니다. 모바일에서 메모리는 지연만큼 중요한 제약이라, 이 표가 없었다면 "52% 빨라졌다"는 결과를 반만 전달한 셈이 됩니다.

**5. 평가를 읽을 때 주의할 점.** 기기 두 대, 해상도 하나, UNet 한 iteration이 주 측정이고 end-to-end 시간은 S23에서만 보고했습니다. baseline은 공개 구현을 저자들의 내부 OpenCL 커널로 옮긴 것이라, 다른 프레임워크의 baseline과 직접 비교하기 어렵습니다. 생성 품질에 대한 측정은 없고(원리상 같아야 하지만 Winograd의 수치 오차는 검증하지 않음), 전력 소모도 다루지 않았습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 목표 | 모델 변경 없이 모바일 GPU에서 Stable Diffusion 1.4의 추론 지연 단축 |
| 전용 커널 | Group Norm과 GELU를 각각 shader 하나, GPU 명령 한 번으로 융합 |
| Fused softmax | 행별 최댓값·지수합($$N \times 2$$)만 계산하고 정규화는 $$V$$ 행렬곱에 융합. 축소는 블록별 다단계로 병렬화 |
| FlashAttention | 레지스터 부담 때문에 $$d = 40$$인 attention에만 선택 적용 |
| Winograd | 3×3 합성곱에 4×4 타일(FLOPs 4배 절감, 가중치 4배 증가), 이득이 나는 곳에만 |
| 결과 | UNet iteration 지연 S23 Ultra 1098→525ms(-52.2%), iPhone 14 Pro Max 1554→1043ms(-32.9%). 20 step 512×512 생성 12초 미만 |
| 비용 | Winograd로 가중치 메모리 1640→2093MB |

## References

- Chen, Sarokin et al., "Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations," CVPR 2023 Workshop. [arXiv:2304.11267](https://arxiv.org/abs/2304.11267)
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks," CVPR 2016. [arXiv:1509.09308](https://arxiv.org/abs/1509.09308)
- Wu & He, "Group Normalization," ECCV 2018. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
