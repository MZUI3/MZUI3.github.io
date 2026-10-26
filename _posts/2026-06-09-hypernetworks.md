---
title: "[Lightweight] HyperNetworks: 작은 네트워크가 큰 네트워크의 가중치를 만든다"
date: 2026-06-09 15:21:00 +0900
categories: [Neural Network, Architecture]
tags: [hypernetwork, weight-generation, weight-sharing, hyperlstm, lstm, parameter-efficiency, layer-embedding, language-modeling]
description: HyperNetworks(Ha, Dai, Le, ICLR 2017)가 층 임베딩으로부터 CNN 커널을 생성하는 정적 하이퍼네트워크와, 매 시점 LSTM의 가중치를 바꾸는 동적 HyperLSTM을 제안해 "완화된 가중치 공유"라는 관점을 제시한 과정을 정리합니다. 작은 실험으로 하이퍼네트워크의 파라미터 수 계산과 end-to-end 학습을 확인합니다.
math: true
---

> CNN은 층마다 다른 가중치를 가져 유연하지만 깊어질수록 중복이 많고, [RNN]({% post_url 2026-06-08-rnn-lstm-gru %})은 모든 시점이 같은 가중치를 공유해 경직됩니다. **HyperNetworks**(Ha, Dai, Le, Google Brain, ICLR 2017)는 둘 사이에 **"완화된 가중치 공유"**를 놓습니다. 작은 네트워크(하이퍼네트워크)가 큰 네트워크(메인 네트워크)의 가중치를 생성하되, 층 또는 시점마다 다른 **임베딩**을 입력으로 받아 가중치를 조금씩 다르게 만드는 것입니다. CNN에서는 파라미터를 크게 줄이면서 정확도를 거의 유지했고, LSTM에서는 가중치를 시점마다 바꾸는 HyperLSTM이 문자 단위 언어 모델링, 필기 생성, 기계 번역에서 당시 최고 수준의 성능을 냈습니다.

## 1. 동기: 유전형과 표현형

하이퍼네트워크는 진화 연산의 HyperNEAT에서 영감을 받았습니다. 거대한 가중치 공간을 직접 탐색하는 대신, 작은 네트워크가 가중치의 **구조**를 만들게 하면 탐색 공간이 줄어듭니다. 생물학 비유로는 하이퍼네트워크가 유전형, 메인 네트워크가 표현형입니다. HyperNEAT는 진화로 학습해 느렸고, DCT 계수로 가중치를 압축하는 Compressed Weight Search는 사전 분포가 너무 단순했습니다. 이 논문의 하이퍼네트워크는 **역전파로 메인 네트워크와 함께 end-to-end 학습**되고, 입력이 좌표가 아니라 학습 가능한 임베딩 벡터라는 점이 다릅니다.

## 2. 방법

### 정적 하이퍼네트워크: CNN 커널 생성

층 $$j$$의 커널 $$K^j \in \mathbb{R}^{N_{in} f \times N_{out} f}$$를 임베딩 $$z^j \in \mathbb{R}^{N_z}$$로부터 만듭니다. 하이퍼네트워크는 **두 층 선형** 구조입니다.

$$
a_i^j = W_i z^j + B_i \;(i = 1..N_{in}), \qquad K_i^j = \langle W_{out}, a_i^j \rangle + B_{out}, \qquad K^j = [K_1^j; K_2^j; \ldots; K_{N_{in}}^j]
$$

첫 층은 $$N_{in}$$개의 서로 다른 행렬 $$W_i \in \mathbb{R}^{d \times N_z}$$로 임베딩을 입력 채널별 은닉 벡터로 보내고, 둘째 층은 **공유된** 텐서 $$W_{out} \in \mathbb{R}^{f \times N_{out} f \times d}$$로 각 은닉 벡터를 커널 조각으로 바꿉니다. 두 층 모두 선형이라 한 층으로 합칠 수 있지만, $$W_{out}$$을 공유하기 때문에 파라미터가 훨씬 적습니다. 학습 파라미터는 $$N_z D + d(N_z + 1) N_{in} + f N_{out} f (d + 1)$$개로, 메인 네트워크 커널의 $$D N_{in} f N_{out} f$$개보다 훨씬 적습니다.

커널 크기가 층마다 다른 ResNet 계열에는 **기본 크기(16×16) 커널을 생성해 타일처럼 이어 붙입니다**. 32×64 커널이 필요하면 8개의 기본 커널, 즉 8개의 임베딩을 씁니다.

### 동적 하이퍼네트워크: HyperRNN과 HyperLSTM

RNN에서는 하이퍼네트워크 자체도 작은 RNN(HyperRNN cell)이어서, 매 시점 $$t$$에 입력 $$x_t$$와 메인 RNN의 은닉 상태 $$h_{t-1}$$을 보고 임베딩 $$z_h, z_x, z_b$$를 내놓습니다. 임베딩에서 전체 가중치 행렬을 선형 결합으로 만들면 메모리가 $$N_z$$배 늘어 비실용적이므로, **가중치 스케일링 벡터** $$d(z) \in \mathbb{R}^{N_h}$$로 행렬의 각 행을 스케일하는 방식으로 바꿉니다.

$$
h_t = \phi\!\left( d_h(z_h) \odot W_h h_{t-1} + d_x(z_x) \odot W_x x_t + b(z_b) \right)
$$

이 형태는 Layer Normalization이나 Recurrent Batch Normalization의 "활성 전 입력을 선형 스케일"하는 연산과 닮았습니다. 차이는 스케일이 통계량이 아니라 **시점과 샘플마다 학습된 정책**으로 정해진다는 점입니다. LSTM 버전이 HyperLSTM이고, 보조 cell은 128유닛, 임베딩 크기 4~16으로 작습니다.

## 3. 코드로 보기: 파라미터 계산과 end-to-end 학습

WRN-40-1의 36개 conv 커널을 하이퍼네트워크로 생성할 때의 파라미터 수를 논문 공식으로 계산하고, 작은 MLP의 은닉 가중치를 생성하는 하이퍼네트워크를 실제로 학습시켜 봅니다.

```python
import torch, torch.nn as nn
torch.manual_seed(0); torch.set_num_threads(1)

# --- (1) parameter count: WRN-40-1 kernels vs. a static hypernetwork (Nz = d = 64, basic kernel 16x16x3x3)
layers = [(16, 16)] * 12 + [(16, 32)] + [(32, 32)] * 11 + [(32, 64)] + [(64, 64)] * 11
main = sum(ci * co * 9 for ci, co in layers)
Nz = d = 64; Nin = Nout = 16; f = 3
n_basic = sum((ci // 16) * (co // 16) for ci, co in layers)         # tiles of the 16x16 basic kernel
hyper = n_basic * Nz + Nin * (d * Nz + d) + f * Nout * f * (d + 1)
one_layer = n_basic * Nz + Nz * Nin * f * Nout * f
print(f"WRN-40-1 conv kernels (36 layers):     {main:>9,} params")
print(f"hypernetwork, 2-layer (shared W_out):  {hyper:>9,} params  ({n_basic} embeddings of size {Nz})")
print(f"hypernetwork, 1-layer (no sharing):    {one_layer:>9,} params")
print("paper: 0.563M -> 0.097M (WRN-40-1), test error 6.73% -> 8.02%")

# --- (2) a hypernetwork generating the hidden weight matrices of a small MLP, trained end to end
class HyperMLP(nn.Module):
    def __init__(s, width=32, n_hidden=2, Nz=4):
        super().__init__()
        s.inp, s.out = nn.Linear(1, width), nn.Linear(width, 1)
        s.z = nn.Parameter(torch.randn(n_hidden, Nz) * 0.5)                    # one embedding per generated layer
        s.W1 = nn.Parameter(torch.randn(width, Nz, Nz) * 0.3); s.B1 = nn.Parameter(torch.zeros(width, Nz))
        s.Wout = nn.Parameter(torch.randn(width, Nz) / Nz ** 0.5); s.Bout = nn.Parameter(torch.zeros(width))
        s.bias = nn.Parameter(torch.zeros(n_hidden, width))
    def kernels(s):
        a = torch.einsum("inz,lz->lin", s.W1, s.z) + s.B1                   # a_i = W_i z + B_i  (l, Nin, d)
        return torch.einsum("od,lid->lio", s.Wout, a) + s.Bout                 # K_i = <W_out, a_i> + B_out
    def forward(s, x):
        h = torch.relu(s.inp(x))
        for K, b in zip(s.kernels(), s.bias): h = torch.relu(h @ K + b)
        return s.out(h)
    def n_generated(s): return sum(p.numel() for n, p in s.named_parameters() if n in ("z", "W1", "B1", "Wout", "Bout"))

class PlainMLP(nn.Module):
    def __init__(s, width=32, n_hidden=2):
        super().__init__()
        s.inp, s.out = nn.Linear(1, width), nn.Linear(width, 1)
        s.hid = nn.ModuleList([nn.Linear(width, width) for _ in range(n_hidden)])
    def forward(s, x):
        h = torch.relu(s.inp(x))
        for l in s.hid: h = torch.relu(l(h))
        return s.out(h)

x = torch.linspace(-3, 3, 512).unsqueeze(1); y = torch.sin(2 * x) + 0.3 * x
def train(model, steps=3000):
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    for _ in range(steps):
        loss = ((model(x) - y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
plain, hyper_m = PlainMLP(), HyperMLP()
print()
print(f"plain MLP:  hidden-matrix params = {2*32*32:,}  final MSE = {train(plain):.2e}")
print(f"hyper MLP:  hypernet params      = {hyper_m.n_generated():,}  final MSE = {train(hyper_m):.2e}")
```

```text
WRN-40-1 conv kernels (36 layers):       557,568 params
hypernetwork, 2-layer (shared W_out):     91,408 params  (242 embeddings of size 64)
hypernetwork, 1-layer (no sharing):      162,944 params
paper: 0.563M -> 0.097M (WRN-40-1), test error 6.73% -> 8.02%

plain MLP:  hidden-matrix params = 2,048  final MSE = 2.15e-05
hyper MLP:  hypernet params      = 808  final MSE = 6.21e-04
```

- WRN-40-1의 커널 558K개가 하이퍼네트워크 91K개로 줄어듭니다. 논문의 0.563M → 0.097M과 맞습니다. $$W_{out}$$을 공유하지 않는 한 층 하이퍼네트워크는 163K로, 두 층 구조가 왜 더 작은지도 보입니다.
- 작은 회귀 문제에서는 2048개의 은닉 가중치를 808개 파라미터의 하이퍼네트워크로 생성해도 학습이 되지만, 최종 MSE는 30배 높습니다. 같은 하이퍼네트워크에서 나온 두 층의 가중치가 서로 비슷해질 수밖에 없는 **완화된 공유의 대가**이고, 논문이 CIFAR-10에서 1.25~1.5% 정확도 손실을 보고한 것과 같은 방향입니다.

## 4. 실험

**MNIST.** 두 층 CNN의 둘째 층 커널(7×7×16×16, 12,544개 가중치)을 **크기 4의 임베딩**과 4,240개 파라미터의 하이퍼네트워크로 생성해 99.24%(일반 학습 99.28%)를 냈습니다. 생성된 커널은 일반 학습된 커널과 시각적으로 비슷합니다.

**CIFAR-10, Wide ResNet.** conv2~conv4의 36개 커널을 하나의 하이퍼네트워크($$N_z = 64$$)로 생성합니다.

| 모델 | 오류 | 파라미터 |
|---|---|---|
| WRN 40-1 | 6.73% | 0.563M |
| Hyper WRN 40-1 | 8.02% | 0.097M |
| WRN 40-2 | 5.66% | 2.236M |
| Hyper WRN 40-2 | 7.23% | 0.148M |

파라미터를 6~15배 줄이는 대가로 1.3~1.6%를 잃습니다. 층마다 다른 종류의 필터가 필요한데 모든 층이 같은 하이퍼네트워크를 거치므로 64차원의 자유도 안에서만 달라질 수 있기 때문입니다.

**문자 단위 언어 모델링.** Penn Treebank에서 1000유닛 LSTM 1.312 BPC 대비 HyperLSTM 1.265, Layer Norm LSTM 1.267, Layer Norm HyperLSTM 1.250, 2층 Layer Norm HyperLSTM 1.219입니다. 1250유닛 LSTM(1.306)이나 2층 LSTM(1.281)보다 좋으므로 단순히 파라미터가 늘어서가 아닙니다. enwik8에서는 2048유닛 Layer Norm HyperLSTM이 1.340 BPC로 당시 최고 수준에 근접합니다.

**필기 생성(IAM).** 900유닛 HyperLSTM이 로그 손실(log-loss, 낮을수록 좋음) -1,162로 2층 650유닛 LSTM(-1,135), Layer Norm LSTM(-1,096)을 모두 앞섭니다. 이 과제에서는 Layer Norm과 결합하면 오히려 나빠졌습니다(-1,067).

**기계 번역.** GNMT의 LSTM을 HyperLSTM으로 바꾸면 WMT'14 En→Fr에서 단일 모델 BLEU 38.95 → 40.03으로, 8개 앙상블(40.35)에 근접합니다.

**가중치 변화 시각화.** 생성된 텍스트와 필기 아래에 가중치 변화량을 그리면, 단어 안에서는 가중치가 거의 고정되고 **단어 사이나 괄호에서 급격히 바뀝니다**. 가중치가 서서히 변하는 것이 아니라 "체제 전환"처럼 움직입니다.

## 5. 해석

**1. 가중치 공유의 연속체를 제시했습니다.** CNN(공유 없음)과 RNN(완전 공유) 사이에서 임베딩 차원 $$N_z$$가 공유 정도를 조절합니다. 위 실험의 파라미터-성능 트레이드오프가 이 축을 따라 움직입니다. 이후 [ALBERT]({% post_url 2026-07-04-albert %})의 층 간 완전 공유, 조건부 가중치 생성(CondConv, dynamic filter) 등이 이 연속체 위에 놓입니다.

**2. 동적 가중치는 정규화의 학습된 버전으로 읽힙니다.** 행 스케일링 형태의 HyperLSTM은 [Layer Normalization]({% post_url 2026-05-23-normalization %})과 수식이 닮았지만, 은닉 상태 히스토그램을 보면 Layer Norm은 포화를 줄이는 반면 HyperLSTM은 대부분 포화 상태에서 동작합니다. 같은 성능에 도달하는 전혀 다른 정책이라는 점이 흥미롭고, 문자 단위 언어 모델링에서 둘을 합치면 더 좋아지는 것도 설명됩니다(필기 생성에서는 반대로 나빠졌습니다).

**3. 압축보다 조건부 계산의 씨앗으로 더 중요합니다.** 정적 하이퍼네트워크의 압축률은 가지치기나 양자화만큼 실용적이지 않습니다. 반면 "입력에 따라 가중치를 바꾼다"는 동적 하이퍼네트워크의 발상은 이후 메타러닝, 신경망 구조 탐색의 가중치 생성(SMASH, one-shot NAS), 조건부 신경망, 최근의 LoRA 생성 연구까지 이어집니다.

**4. 메모리 제약이 설계를 정했습니다.** 전체 행렬을 생성하는 논문의 식 (5)를 포기하고 행 스케일링 식 (8)로 바꾼 것은 성능이 아니라 메모리 때문입니다. 표현력을 일부 희생해 실용성을 얻은 선택이 결과적으로 정규화와의 연결이라는 해석까지 낳았습니다.

**5. 평가를 읽을 때 주의할 점.** LSTM baseline은 저자들의 구현이고 recurrent dropout 설정에 따라 꽤 달라집니다. HyperLSTM은 파라미터가 baseline보다 15% 정도 많아서, 1250유닛 LSTM과의 비교가 그 효과를 통제하는 역할을 합니다. CNN 실험은 CIFAR-10에 한정되고 ImageNet 규모 결과는 없습니다. 위 toy의 회귀 문제는 아주 작아서 손실 비율을 그대로 일반화할 수 없습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 아이디어 | 하이퍼네트워크가 층(또는 시점) 임베딩 $$z$$로부터 메인 네트워크의 가중치를 생성, end-to-end 역전파 |
| 정적(CNN) | 두 층 선형 하이퍼네트워크, $$W_{out}$$ 공유, 16×16 기본 커널 타일링 |
| 동적(RNN) | HyperRNN cell이 매 시점 $$z$$를 내고 가중치 행을 스케일 $$d(z) \odot W$$ |
| 결과(CNN) | WRN 40-1 0.563M → 0.097M, 오류 6.73 → 8.02% |
| 결과(RNN) | PTB 1.265 BPC(LSTM 1.312), enwik8 1.340, IAM -1,162, GNMT BLEU 40.03 |
| 관점 | 가중치 공유의 연속체, 학습된 동적 정규화 |

## References

- Ha, Dai, Le, "HyperNetworks," ICLR 2017. [arXiv:1609.09106](https://arxiv.org/abs/1609.09106)
- Stanley, D'Ambrosio, Gauci, "A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks," Artificial Life 2009.
- Schmidhuber, "Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks," Neural Computation 1992.
- Ba, Kiros, Hinton, "Layer Normalization," 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Brock, Lim, Ritchie, Weston, "SMASH: One-Shot Model Architecture Search through HyperNetworks," ICLR 2018. [arXiv:1708.05344](https://arxiv.org/abs/1708.05344)
