---
title: "[AI 기초 10] RNN, LSTM, GRU: 시퀀스 데이터를 기억하는 신경망"
date: 2026-06-08 19:42:00 +0900
categories: [Neural Network, Basics]
tags: [rnn, lstm, gru, bptt, vanishing-gradient, sequence-model]
description: 시퀀스 데이터를 다루는 RNN의 구조와 BPTT에서 기울기가 사라지는 이유, 이를 게이트와 cell state로 해결한 LSTM·GRU의 수식과 차이를 PyTorch 코드와 함께 정리합니다.
math: true
---

> CNN이 **공간적인 패턴**을 잘 잡는다면, RNN은 **시간 순서가 있는 데이터**(문장, 음성, 센서 신호)를 다루기 위해 만들어졌습니다. 이 글에서는 RNN의 구조와 한계, 그리고 그 한계를 게이트로 해결한 LSTM과 GRU를 정리합니다. 이 흐름은 이후 [Transformer]({% post_url 2026-06-10-attention-is-all-you-need %})로 이어집니다.

## 1. Feed-Forward vs Recurrent

| | Feed-Forward NN (MLP, CNN) | Recurrent NN |
|---|---|---|
| 신호 방향 | 입력 → 출력, **한 방향** | 은닉층 출력이 **다시 은닉층 입력으로** |
| 입력 길이 | 고정 | **임의의 길이** 시퀀스 |
| 기억 | 없음 (입력마다 독립) | 이전 시점의 정보를 **hidden state**로 전달 |

> 이름이 비슷한 **Recursive Neural Network (RvNN)**은 다른 모델입니다. 하나의 가중치 묶음을 **트리 구조**(예: 문장의 구문 트리)를 따라 재귀적으로 적용합니다. RNN은 이 중 트리가 한 줄(시간 축)인 특수한 경우로 볼 수 있습니다.

## 2. RNN의 구조

### RNN Cell과 Hidden State

각 시점 $$t$$의 RNN cell은 현재 입력 $$x_t$$와 **이전 시점의 hidden state** $$h_{t-1}$$을 함께 받아 새 hidden state를 만듭니다.

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h),\qquad y_t = W_{hy}h_t + b_y
$$

중요한 점은 **모든 시점이 같은 가중치** $$W_{xh}, W_{hh}, W_{hy}$$를 공유한다는 것입니다. 그래서 시퀀스 길이가 달라도 같은 모델을 쓸 수 있고, 파라미터 수도 길이와 무관합니다.

시간 축으로 펼쳐 보면(unroll) 같은 cell을 여러 번 이어 붙인 깊은 네트워크와 같습니다.

```
x1 → [cell] → x2 → [cell] → x3 → [cell] → ...
        h1 ──────→   h2  ──────→  h3
```

### 입출력 형태

| 형태 | 예시 |
|---|---|
| One-to-many | 이미지 → 캡션 생성 |
| Many-to-one | 문장 → 감정 분류, 음성 → 화자 인식 |
| Many-to-many (동기) | 프레임별 동작 인식, 품사 태깅 |
| Many-to-many (seq2seq) | 번역 (Encoder–Decoder) |

### 왜 tanh를 쓰나

- 출력이 **-1~1로 제한**되어, 같은 가중치가 시간마다 반복 곱해져도 값이 폭주하지 않음
- 미분 최댓값이 1로 Sigmoid(0.25)보다 커서 **기울기가 더 오래 유지**됨
- ReLU를 쓰면 출력이 위로 무한정 커질 수 있어, 반복 곱셈 과정에서 값과 기울기가 **발산**하기 쉬움

자세한 비교는 [3편 활성화 함수]({% post_url 2026-05-13-activation-functions %})에 있습니다.

## 3. BPTT와 Vanishing Gradient

RNN은 펼친 네트워크에 역전파를 적용하는 **BPTT(Backpropagation Through Time)**로 학습합니다. 손실을 앞쪽 시점으로 전달하려면 시점마다

$$
\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}\big(1 - h_t^2\big)\,W_{hh}
$$

를 곱해야 합니다. $$k$$ 시점 전까지 가려면 이것이 $$k$$번 곱해집니다.

- $$W_{hh}$$의 크기(가장 큰 특이값)가 **1보다 작으면** → tanh의 미분도 1 이하이므로 기울기가 지수적으로 사라짐 (**Vanishing**)
- **1보다 크면** → 기울기가 지수적으로 커질 수 있음 (**Exploding**). tanh 미분이 1보다 작은 값을 곱해 주므로 항상 폭주하는 것은 아닙니다(Pascanu et al., 2013).

결과적으로 기본 RNN(Vanilla RNN)은 **긴 시퀀스의 앞부분 정보를 학습하지 못합니다.** "나는 프랑스에서 자랐다. ... (긴 문장) ... 그래서 나는 ___를 유창하게 한다"에서 빈칸을 채우려면 멀리 떨어진 "프랑스"를 기억해야 하는데, 이것이 어렵습니다. 이를 **Long-Term Dependency 문제**라고 합니다.

Exploding은 기울기의 크기를 잘라 내는 **gradient clipping**으로 비교적 쉽게 막을 수 있지만, Vanishing은 구조를 바꿔야 해결됩니다.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 4. LSTM (Long Short-Term Memory)

LSTM(Hochreiter & Schmidhuber, 1997)은 **게이트(Gate)**로 정보를 얼마나 기억하고 잊을지 직접 조절합니다. (1997년 원래 LSTM에는 forget gate가 없었고, 지금 쓰는 forget gate는 Gers et al.(2000)이 추가한 것입니다.)

### 게이트란

정보를 **얼마나 통과시킬지 0~1 사이의 비율로 결정**하는 장치입니다. Sigmoid 출력(0~1)을 다른 값에 곱하는 방식이라, 0이면 완전히 차단하고 1이면 그대로 통과시킵니다. 곱셈, 덧셈, 활성화 함수로만 이루어집니다.

### Cell State

LSTM은 hidden state $$h_t$$ 외에 **cell state** $$c_t$$라는 별도의 기억 통로를 둡니다. 이 통로에서는 정보가 **곱셈이 아니라 덧셈**으로 전달되기 때문에 기울기가 잘 사라지지 않습니다. ResNet의 shortcut과 비슷한 원리입니다.

### 세 개의 게이트

$$
\begin{aligned}
f_t &= \sigma(W_f[h_{t-1}, x_t] + b_f) && \text{Forget gate: 이전 기억을 얼마나 지울지} \\
i_t &= \sigma(W_i[h_{t-1}, x_t] + b_i) && \text{Input gate: 새 정보를 얼마나 넣을지} \\
\tilde{c}_t &= \tanh(W_c[h_{t-1}, x_t] + b_c) && \text{새로 넣을 후보 정보} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t && \text{기억 갱신} \\
o_t &= \sigma(W_o[h_{t-1}, x_t] + b_o) && \text{Output gate: 기억 중 무엇을 내보낼지} \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

- **Forget gate**: 문장 주어가 바뀌면 이전 주어 정보를 지움
- **Input gate**: 새 주어 정보를 기억에 추가
- **Output gate**: 지금 시점에 필요한 정보만 출력

**주의할 점**: 게이트에는 반드시 **Sigmoid**를 써서 출력을 0~1 사이로 유지해야 합니다. 후보 정보와 출력에는 tanh를 씁니다.

$$c_t$$의 기울기에서 게이트를 거치는 간접 경로를 빼고 cell state끼리의 직접 경로만 보면 $$\partial c_t / \partial c_{t-1} = f_t$$입니다. Forget gate가 1에 가까우면 기울기가 거의 그대로 먼 과거까지 전달됩니다.

> 실무 팁: 그래서 forget gate의 bias를 **1 정도로 초기화**해 학습 초반에 "기본적으로 기억을 유지"하도록 두면 긴 의존성을 더 잘 배운다는 보고가 있습니다(Jozefowicz et al., 2015). PyTorch `nn.LSTM`은 bias를 0 근처로 초기화하므로 필요하면 직접 바꿔야 합니다.
{: .prompt-tip }

## 5. GRU (Gated Recurrent Unit)

GRU(Cho et al., 2014)는 LSTM을 **단순화**한 구조입니다. Cell state와 hidden state를 하나로 합치고, 게이트를 두 개로 줄였습니다.

$$
\begin{aligned}
z_t &= \sigma(W_z[h_{t-1}, x_t]) && \text{Update gate: 이전 상태를 얼마나 유지할지} \\
r_t &= \sigma(W_r[h_{t-1}, x_t]) && \text{Reset gate: 이전 상태를 얼마나 무시할지} \\
\tilde{h}_t &= \tanh(W[r_t \odot h_{t-1}, x_t]) \\
h_t &= (1 - z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

- **Update gate**가 LSTM의 forget gate와 input gate 역할을 **하나로** 합니다. 기억을 $$1-z$$만큼 남기고 $$z$$만큼 새로 채웁니다.
- 주의: $$z$$와 $$1-z$$의 역할은 문헌마다 반대로 쓰기도 합니다. PyTorch `nn.GRU` 문서의 식은 $$h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$$로, $$z$$가 **이전 상태를 유지하는 비율**입니다. 직접 구현한 GRU와 PyTorch 결과를 비교할 때 헷갈리기 쉬운 부분입니다.
- 성능은 LSTM과 비슷한 경우가 많고, 파라미터와 연산량은 약 3/4 수준입니다.

## 6. 비교

| | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| 상태 | $$h$$ | $$h$$, $$c$$ | $$h$$ |
| 게이트 | 없음 | Forget, Input, Output (3개) | Update, Reset (2개) |
| 파라미터 (같은 hidden 크기) | 1× | 4× | 3× |
| Long-term dependency | 약함 | 강함 | 강함 |
| 특징 | 단순, 짧은 시퀀스용 | 표현력이 가장 큼 | 가볍고 빠름 |

## 7. PyTorch로 써 보기

```python
import torch
import torch.nn as nn

x = torch.randn(4, 20, 16)                  # (batch, seq_len, feature) with batch_first=True

rnn = nn.RNN(16, 32, batch_first=True)
lstm = nn.LSTM(16, 32, num_layers=2, batch_first=True, dropout=0.2)
gru = nn.GRU(16, 32, batch_first=True, bidirectional=True)

out, h = rnn(x)            # out: (4, 20, 32), h: (1, 4, 32)
out, (h, c) = lstm(x)      # LSTM은 hidden과 cell을 함께 반환
out, h = gru(x)            # bidirectional → out: (4, 20, 64), h: (2, 4, 32) 방향별 마지막 hidden

print(sum(p.numel() for p in rnn.parameters()),     # 1,600
      sum(p.numel() for p in nn.LSTM(16, 32).parameters()),   # 6,400  (4배)
      sum(p.numel() for p in nn.GRU(16, 32).parameters()))    # 4,800  (3배)

# Many-to-one 분류기
class SeqClassifier(nn.Module):
    def __init__(self, in_dim, hidden, n_cls):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_cls)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])               # 마지막 시점의 hidden state로 분류
```

## 8. 왜 RNN을 쓰나

- **시퀀스 데이터**: 음성 인식, 번역, 자연어 처리, 시계열 예측
- **임의 길이 입력**: 고정 길이가 아닌 시퀀스를 그대로 처리
- **CNN과 함께 사용**: CNN으로 공간적 특징을, RNN으로 시간적 특징을 추출

마지막 조합은 특히 **음성**에서 많이 씁니다. 음성을 **spectrogram**(시간 × 주파수 이미지)으로 바꾼 뒤 CNN으로 주파수 패턴을, RNN으로 시간 흐름을 학습하는 **CRNN** 구조입니다([14편 음성 분류]({% post_url 2026-06-22-audio-classification %})).

## 9. RNN의 한계와 그 이후

LSTM과 GRU도 근본적인 한계가 남아 있습니다.

1. **순차 연산**: $$h_t$$를 계산하려면 $$h_{t-1}$$이 끝나야 해서 GPU로 병렬화하기 어렵습니다.
2. **고정 길이 병목**: seq2seq에서 Encoder의 모든 정보를 마지막 hidden state 하나에 담아야 합니다.

이를 해결하려고 **Attention**이 나왔고, 결국 RNN을 완전히 빼고 Attention만 남긴 **Transformer**가 등장했습니다. 자세한 내용은 [Transformer 글]({% post_url 2026-06-10-attention-is-all-you-need %})에서 이어집니다.

## 10. 정리

| 개념 | 핵심 |
|---|---|
| RNN | hidden state를 다음 시점에 전달, 가중치를 시간 축으로 공유 |
| BPTT | 시간 축으로 펼쳐서 역전파 |
| Vanishing gradient | 같은 행렬이 반복 곱해져 먼 과거의 기울기가 사라짐 |
| LSTM | cell state(덧셈 경로) + 3개 게이트로 장기 기억 |
| GRU | 2개 게이트로 단순화, 비슷한 성능 |
| 게이트 | Sigmoid(0~1)로 정보 통과 비율 조절 |

## References

- Elman, "Finding Structure in Time," Cognitive Science 1990.
- Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Computation 1997.
- Gers, Schmidhuber & Cummins, "Learning to Forget: Continual Prediction with LSTM," Neural Computation 2000.
- Jozefowicz, Zaremba & Sutskever, "An Empirical Exploration of Recurrent Network Architectures," ICML 2015.
- [PyTorch – nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation," EMNLP 2014.
- Pascanu et al., "On the difficulty of training recurrent neural networks," ICML 2013.
- Christopher Olah, "Understanding LSTM Networks," 2015.
- 딥 러닝을 이용한 자연어 처리 입문 — <https://wikidocs.net/22886>
