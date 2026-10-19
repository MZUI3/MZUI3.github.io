---
title: "[Transformer] Attention Is All You Need 정리: Self-Attention부터 Positional Encoding까지"
date: 2026-06-10 19:48:00 +0900
categories: [Transformer, Architecture]
tags: [transformer, attention, self-attention, positional-encoding, nlp]
description: seq2seq의 병목과 RNN의 순차 연산 문제에서 출발해 Self-Attention, √d_k 스케일링, Multi-Head, Positional Encoding, 연산 복잡도까지 Transformer의 설계 이유를 코드와 함께 정리합니다.
math: true
---

> Transformer 시리즈의 첫 글입니다. 이후 글에서 다룰 ViT, PoolFormer, Mobile-Former, GPT는 모두 이 구조에서 출발하기 때문에, 여기서는 **"왜 Attention만으로 충분했는가"**에 집중해서 정리합니다.

## 1. 등장 배경: seq2seq와 RNN의 한계

Transformer 이전의 번역 모델은 대부분 **seq2seq(Encoder–Decoder) + RNN** 구조였습니다.

- **Encoder**: "I am a student" 같은 입력 문장을 하나의 고정 길이 **Context Vector**로 압축
- **Decoder**: 이 Context Vector를 받아 "나는 학생이다"를 한 단어씩 생성

이 구조에는 두 가지 문제가 있습니다.

1. **병목(Bottleneck)**: 문장이 아무리 길어도 마지막 hidden state 하나에 모든 정보를 욱여넣어야 합니다. 입력이 길어질수록 앞쪽 정보가 소실됩니다.
2. **순차 연산**: RNN은 $$t$$번째 hidden state를 계산하려면 $$t-1$$번째가 끝나야 합니다. 한 번에 한 단어씩만 처리하므로 GPU 병렬화가 어렵고, 시퀀스가 길어질수록 먼 과거로 가는 기울기가 소실(Vanishing Gradient)되기 쉽습니다.

## 2. Attention: 중요한 단어에 집중하기

첫 번째 문제를 해결한 것이 **Attention**입니다. Decoder가 단어를 생성할 때마다 마지막 hidden state 하나만 보는 대신, **Encoder의 모든 time step hidden state를 다시 들여다보고** 지금 필요한 것에 가중치를 줍니다.

동작은 세 단계입니다.

1. 현재 Decoder 상태와 각 Encoder hidden state를 비교해 **Attention Score**를 구한다. (Bahdanau et al., 2015는 작은 신경망으로 점수를 매기는 additive 방식, Luong et al., 2015는 내적 방식을 썼습니다. 여기서는 이해하기 쉬운 내적으로 설명합니다.)
2. Score에 **Softmax**를 적용해 합이 1인 **Attention Weight**로 바꾼다.
3. 각 hidden state에 weight를 곱해 더한 **Weighted Vector(Context)**를 Decoder 입력에 사용한다.

간단한 예시를 보면 감이 옵니다. Score가 `[13, 9, 9]`라면

$$
\text{softmax}([13, 9, 9]) \approx [0.964,\ 0.018,\ 0.018]
$$

이므로 Context는 거의 $$0.964 \cdot H_1$$이 됩니다. Score 차이가 4밖에 안 나도 Softmax가 지수 함수이기 때문에 한 단어에 강하게 집중하게 됩니다.

하지만 이 시점의 Attention은 여전히 **RNN 위에 얹힌 보조 장치**였고, 순차 연산 문제는 그대로였습니다.

## 3. Transformer의 아이디어: RNN을 빼고 Attention만 남기자

2017년 Vaswani et al.의 *Attention Is All You Need*는 과감하게 RNN을 제거합니다. 모든 단어를 **한 번에 병렬로** 넣고, 단어 사이의 관계는 전부 Attention으로 계산합니다.

이렇게 하면 두 가지가 새로 필요해집니다.

- 단어끼리의 관계를 계산하는 방법 → **Self-Attention**
- 병렬로 넣으면서 사라진 **순서 정보** → **Positional Encoding**

## 4. Self-Attention과 Q, K, V

다음 문장을 봅시다.

> "The animal didn't cross the street because **it** was too tired."

사람은 `it`이 `animal`을 가리킨다는 걸 바로 알지만, 모델은 문장 안의 다른 단어들과의 관계를 계산해야 알 수 있습니다. **Self-Attention**은 같은 문장 안의 단어들끼리 Attention을 수행해 이 관계를 학습합니다.

각 토큰 임베딩 $$X$$에 학습 가능한 행렬 세 개를 곱해 세 가지 역할의 벡터를 만듭니다.

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

| 벡터 | 역할 | 비유 |
|---|---|---|
| **Query** | 지금 분석하려는 단어가 "무엇을 찾는지" | 검색어 |
| **Key** | 각 단어가 "어떤 정보를 갖고 있는지" 보여주는 꼬리표 | 문서 제목 |
| **Value** | 실제로 가져갈 내용 | 문서 본문 |

> 검색 비유는 "Query와 Key를 비교해 Value를 가져온다"는 흐름까지만 맞습니다. 실제 검색은 가장 잘 맞는 문서 하나를 고르지만, Attention은 **모든 Value를 가중 평균**합니다. 또 Q, K, V는 모두 같은 입력 $$X$$에서 학습된 행렬로 만들어지므로, 무엇이 "검색어"가 될지도 학습으로 정해집니다.
{: .prompt-info }

Query와 모든 Key를 내적해 관련도를 구하고, 그 비율대로 Value를 섞습니다. 이것이 **Scaled Dot-Product Attention**입니다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### 왜 $$\sqrt{d_k}$$로 나누나?

$$Q, K$$의 각 성분이 평균 0, 분산 1이라면 내적 $$q \cdot k$$의 분산은 $$d_k$$가 됩니다. 차원이 커질수록 값이 커지고, Softmax가 한쪽으로 극단적으로 쏠려 **gradient가 거의 0**이 됩니다. $$\sqrt{d_k}$$로 나눠 분산을 1로 되돌리는 것입니다.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: (batch, heads, seq_len, d_k)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k ** 0.5   # (B, H, L, L)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # 가릴 위치는 softmax 후 0이 되도록
    attn = F.softmax(scores, dim=-1)
    return attn @ v, attn

torch.manual_seed(0)
B, H, L, d_k = 1, 2, 4, 8
q, k, v = (torch.randn(B, H, L, d_k) for _ in range(3))

causal = torch.tril(torch.ones(L, L))   # Decoder용: 자기 자신과 과거만 보도록 하삼각 행렬
out, attn = scaled_dot_product_attention(q, k, v, mask=causal)
print(out.shape)                         # torch.Size([1, 2, 4, 8])
print(attn[0, 0].round(decimals=2))      # 첫 번째 head의 attention weight

# PyTorch 2.0+ 내장 함수와 결과 비교
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(torch.allclose(out, ref, atol=1e-6))  # True
```

```text
torch.Size([1, 2, 4, 8])
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.5900, 0.4100, 0.0000, 0.0000],
        [0.1000, 0.5800, 0.3200, 0.0000],
        [0.3500, 0.1800, 0.4200, 0.0600]])
True
```

각 행이 "이 토큰이 어떤 토큰을 얼마나 참고했는가"이고, 행의 합은 1입니다. 마스크 때문에 대각선 위쪽(미래 토큰)은 모두 0입니다. 실무에서는 직접 구현하기보다 PyTorch 2.0부터 제공되는 `F.scaled_dot_product_attention`을 씁니다. 이 함수는 조건이 맞으면 FlashAttention 같은 메모리 효율적인 커널로 자동 전환됩니다.

## 5. Multi-Head Attention: 여러 시각으로 보기

Attention을 한 번만 하면 하나의 관점(예: 대명사–명사 관계)만 학습하기 쉽습니다. 그래서 $$d_{model}=512$$를 $$h=8$$개의 head로 나눠($$d_k = 64$$) **각자 다른 $$W_Q, W_K, W_V$$로 Attention을 수행**한 뒤 이어 붙입니다.

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O
$$

학습된 모델을 분석해 보면 어떤 head는 문법적 관계를, 어떤 head는 가까운 위치를 주로 보는 식으로 역할이 나뉘는 경향이 관찰됩니다. 다만 역할이 항상 깔끔하게 나뉘는 것은 아니고, 상당수 head는 제거해도 성능이 거의 떨어지지 않는다는 연구(Michel et al., 2019)도 있습니다. head를 나눠도 전체 연산량은 single-head와 거의 같다는 점도 장점입니다.

## 6. Word Embedding

단어는 처음에 **one-hot vector**로 표현됩니다. `고양이 = [1,0,0,0]`, `강아지 = [0,1,0,0]`처럼 구분은 명확하지만, 단어 수만큼 차원이 늘어나고 **모든 단어 사이의 거리가 같아서** "고양이와 강아지가 비슷하다"는 정보를 담지 못합니다.

임베딩은 이를 더 작은 $$K$$차원의 dense vector로 바꿉니다.

| | 동물 | 무리생활 | 꼬리 |
|---|---|---|---|
| 고양이 | 0.7 | 0.1 | 0.7 |
| 강아지 | 0.7 | 0.6 | 0.7 |
| 사람 | 0.7 | 0.8 | 0.2 |

(실제 임베딩 차원에는 이렇게 사람이 읽을 수 있는 의미가 붙지 않지만, 직관적으로는 이런 느낌입니다.)

## 7. Positional Encoding: 순서를 다시 넣어주기

RNN은 순서대로 읽으니 위치 정보가 자연스럽게 들어가지만, Self-Attention은 입력 토큰의 순서를 바꾸면 출력도 똑같이 순서만 바뀌는 성질(**permutation equivariance**)이 있습니다. 즉 위치 정보가 없으면 **"I love you"와 "you love I"를 구분하지 못합니다.** 그래서 임베딩에 위치 벡터를 더해 줍니다.

$$
PE_{(pos,\,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\qquad
PE_{(pos,\,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

sin과 cos을 함께 쓰는 이유는 다음과 같습니다.

- **위치마다 고유한 값**: 차원마다 주기가 다른 파형을 섞기 때문에 서로 다른 위치가 같은 벡터를 갖지 않습니다.
- **상대 위치를 선형 변환으로 표현**: 삼각함수 덧셈 정리 덕분에 $$PE_{pos+k}$$를 $$PE_{pos}$$의 선형 함수로 쓸 수 있습니다. 모델이 "k칸 떨어진 단어"라는 관계를 배우기 쉬워집니다.
- **값의 범위가 [-1, 1]**: 임베딩 값을 압도하지 않습니다.

논문은 학습 가능한 위치 임베딩도 실험했는데 성능은 거의 같았고, 학습 때보다 긴 문장에도 적용할 수 있을 것이라는 이유로 sinusoidal 방식을 택했습니다. 이후 BERT·GPT-2는 학습 가능한 위치 임베딩을, 최근 LLM은 RoPE(회전 위치 임베딩)를 주로 씁니다.

```python
import math
import torch

def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (max_len, d_model)
```

## 8. 전체 구조

Encoder와 Decoder는 각각 같은 블록을 $$N=6$$번 쌓은 구조입니다.

**Encoder 블록**
1. Multi-Head Self-Attention
2. Add & LayerNorm (Residual connection)
3. Position-wise Feed-Forward Network ($$512 \to 2048 \to 512$$)
4. Add & LayerNorm

**Decoder 블록**
1. **Masked** Multi-Head Self-Attention: 미래 단어를 보지 못하도록 마스킹
2. **Cross-Attention**: Query는 Decoder, Key/Value는 Encoder 출력
3. Feed-Forward Network
4. 각 단계마다 Add & LayerNorm

> 원 논문은 `x + Sublayer(x)`를 더한 **뒤에** LayerNorm을 하는 **Post-LN** 구조입니다. 이후 층이 깊어지면 Post-LN이 학습 초기에 불안정하다는 점이 알려져, GPT-2 이후 대부분의 모델은 Sublayer **앞에** LayerNorm을 두는 **Pre-LN**(`x + Sublayer(LN(x))`)을 씁니다.
{: .prompt-info }

### Batch Norm 대신 Layer Norm을 쓰는 이유

- **Batch Norm**: 배치 안의 샘플들끼리 같은 feature를 정규화합니다. 문장 길이가 제각각이고 배치가 작으면 통계가 불안정합니다.
- **Layer Norm**: 샘플 하나 안의 feature들끼리 정규화합니다. 배치 크기와 문장 길이에 영향을 받지 않습니다.

### 연산 복잡도: Self-Attention의 대가

시퀀스 길이를 $$n$$, 표현 차원을 $$d$$라고 할 때 논문은 층 하나의 비용을 이렇게 비교합니다.

| 층 종류 | 층당 연산량 | 순차 연산 수 | 두 토큰 사이 최대 경로 길이 |
|---|---|---|---|
| Self-Attention | $$O(n^2 \cdot d)$$ | $$O(1)$$ | $$O(1)$$ |
| Recurrent | $$O(n \cdot d^2)$$ | $$O(n)$$ | $$O(n)$$ |
| Convolution (커널 $$k$$) | $$O(k \cdot n \cdot d^2)$$ | $$O(1)$$ | $$O(\log_k n)$$ |

- **순차 연산 $$O(1)$$**: 모든 위치를 한 번에 계산하므로 GPU 병렬화가 쉽습니다.
- **최대 경로 길이 $$O(1)$$**: 첫 단어와 마지막 단어가 한 번의 Attention으로 바로 연결되므로 먼 거리의 관계를 배우기 쉽습니다.
- **대가는 $$n^2$$**: 모든 토큰 쌍의 점수를 계산하므로 문장이 2배 길어지면 Attention 연산과 메모리가 4배가 됩니다. 문장 길이 $$n$$이 차원 $$d$$보다 짧은 번역 문장에서는 RNN보다 싸지만, 긴 문서나 고해상도 이미지에서는 이것이 병목이 됩니다. 이후 Sparse Attention, Linformer, FlashAttention 같은 연구가 이 문제를 다룹니다.

## 9. 정리

| 문제 | Transformer의 해법 |
|---|---|
| 고정 길이 Context Vector 병목 | 모든 토큰을 직접 참조하는 Attention |
| RNN의 순차 연산 | 전체 시퀀스를 한 번에 처리하는 Self-Attention |
| 병렬화로 사라진 순서 정보 | Sinusoidal Positional Encoding |
| 하나의 관점만 학습 | Multi-Head Attention |
| 깊은 네트워크 학습 | Residual Connection + LayerNorm |

Transformer 시리즈의 다음 글 [경량 Vision Transformer 동향]({% post_url 2026-06-13-lightweight-vision-transformer %})에서는 이 구조를 이미지에 적용한 **Vision Transformer(ViT)**를 다룹니다. 이미지를 16×16 패치로 잘라 "단어"처럼 다루는 방식과, ImageNet 규모에서는 ResNet보다 성능이 낮았던 이유(**inductive bias** 부족)를 살펴보겠습니다.

## References

- Vaswani et al., "Attention Is All You Need," NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate," ICLR 2015. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Michel et al., "Are Sixteen Heads Really Better than One?," NeurIPS 2019. [arXiv:1905.10650](https://arxiv.org/abs/1905.10650)
- Xiong et al., "On Layer Normalization in the Transformer Architecture," ICML 2020. (Pre-LN vs Post-LN) [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Jay Alammar, "The Illustrated Transformer" — <https://jalammar.github.io/>
- <https://yngie-c.github.io/nlp/2020/07/01/nlp_transformer/>
