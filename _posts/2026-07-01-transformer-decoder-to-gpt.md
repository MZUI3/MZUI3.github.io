---
title: "[LLM] Transformer 디코더에서 GPT로: Decoder-only 언어 모델의 탄생"
date: 2026-07-01 19:03:00 +0900
categories: [Transformer, LLM]
tags: [llm, gpt, decoder-only, causal-attention, language-model, in-context-learning]
description: Transformer에서 Decoder만 남긴 GPT의 구조와 causal mask, 실행 가능한 미니 GPT 구현과 텍스트 생성, GPT-1~3의 변화와 scaling law, RLHF로 이어지는 alignment 문제까지 정리합니다.
math: true
---

> LLM 시리즈의 첫 글입니다. [Transformer 글]({% post_url 2026-06-10-attention-is-all-you-need %})에서 다룬 Encoder–Decoder 구조에서 **Decoder만 떼어내** 만든 GPT가, 어떻게 "다음 단어 맞히기" 하나로 지금의 LLM까지 이어졌는지 정리합니다.

## 1. 언어 모델이란

언어 모델(Language Model)은 문장의 확률을 계산하는 모델입니다. 문장 $$u_1, u_2, \dots, u_n$$의 확률을 chain rule로 쪼개면

$$
P(u_1, \dots, u_n) = \prod_{i=1}^{n} P(u_i \mid u_1, \dots, u_{i-1})
$$

즉 **"앞의 토큰들이 주어졌을 때 다음 토큰의 확률"**만 잘 예측하면 됩니다. 이런 모델을 **autoregressive(자기회귀) 언어 모델**이라고 부르고, 학습 목표는 다음 토큰의 log-likelihood를 최대화하는 것입니다.

$$
\mathcal{L}(\theta) = \sum_i \log P_\theta(u_i \mid u_{i-k}, \dots, u_{i-1})
$$

이 목표의 가장 큰 장점은 **레이블이 필요 없다**는 것입니다. 인터넷의 모든 텍스트가 그대로 학습 데이터가 됩니다.

## 2. 왜 Decoder만 남겼나

원래 Transformer는 번역을 위한 구조였습니다.

| 구성 요소 | 역할 | GPT에 필요한가 |
|---|---|---|
| Encoder | 입력 문장 전체를 양방향으로 읽음 | ✕ (번역할 원문이 따로 없음) |
| Decoder의 Cross-Attention | Encoder 출력 참조 | ✕ (참조할 Encoder가 없음) |
| Decoder의 **Masked Self-Attention** | 앞쪽 토큰만 보고 다음 토큰 생성 | **○** |
| Feed-Forward, Residual, LayerNorm | 블록 공통 | ○ |

언어 모델은 "앞만 보고 다음을 맞히는" 일이므로 Decoder의 Masked Self-Attention이 정확히 그 역할을 합니다. 그래서 GPT는 **Cross-Attention을 뺀 Decoder 블록만 여러 층 쌓은** 구조, 즉 **Decoder-only Transformer**입니다.

> 반대로 Encoder만 쌓은 것이 BERT입니다. BERT는 양방향으로 문맥을 보는 대신 생성에는 약하고, GPT는 한 방향만 보는 대신 자연스럽게 텍스트를 생성할 수 있습니다.

## 3. Causal Mask: 미래를 보지 않기

Self-Attention은 원래 모든 토큰이 모든 토큰을 봅니다. 그대로 학습하면 모델은 다음 토큰을 "예측"하지 않고 그냥 **베껴 씁니다.** 그래서 attention score 행렬에서 **자기보다 뒤에 있는 위치를 $$-\infty$$로 가립니다.**

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,\qquad
M_{ij} =
\begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

Softmax를 거치면 $$-\infty$$ 위치의 가중치는 0이 됩니다. 4개 토큰이라면 마스크는 이런 하삼각 모양입니다.

```
       The  cat  sat  on
The  [  ✓    ·    ·    · ]
cat  [  ✓    ✓    ·    · ]
sat  [  ✓    ✓    ✓    · ]
on   [  ✓    ✓    ✓    ✓ ]
```

이 마스크 덕분에 **길이 $$n$$인 문장 하나로 $$n$$개의 "다음 토큰 예측" 문제를 한 번에 병렬로 학습**할 수 있습니다. RNN 언어 모델처럼 한 토큰씩 순서대로 돌릴 필요가 없어서 학습이 빠릅니다.

## 4. 코드로 보는 GPT 블록

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, -1).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # causal mask 적용
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # Pre-LN (GPT-2 이후)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=768, n_heads=12, n_layers=12):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)    # 학습되는 위치 임베딩
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):                               # GPT-2식 초기화: 평균 0, 표준편차 0.02
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:   # targets = idx를 한 칸 왼쪽으로 민 것
            # reshape: idx[:, 1:]처럼 슬라이싱한 텐서는 메모리가 연속적이지 않아 view(-1)이 실패함
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
```

작은 설정으로 돌려 보면 다음과 같습니다.

```python
torch.manual_seed(0)
model = GPT(vocab_size=100, max_len=32, d_model=64, n_heads=4, n_layers=2)
idx = torch.randint(0, 100, (2, 16))
logits, loss = model(idx[:, :-1], idx[:, 1:])     # 입력: 앞 15개, 정답: 한 칸 뒤 15개
print(logits.shape, round(loss.item(), 3))         # torch.Size([2, 15, 100]) 약 4.6
print(sum(p.numel() for p in GPT(50257, 1024).parameters()))   # 124439808 ≈ GPT-2 small(124M)
```

학습 전 loss가 약 4.6인 것은 정상입니다. 아무것도 모르는 모델이 100개 토큰에 고르게 확률을 주면 cross entropy는 $$\ln 100 \approx 4.6$$이기 때문입니다. 반대로 초기화 코드를 빼고 돌려 보면 loss가 40 이상에서 시작하는데, 이렇게 **"초기 loss가 $$\ln(\text{vocab})$$ 근처인가"**는 언어 모델 구현을 점검하는 간단하고 강력한 방법입니다. 기본 설정(768차원, 12층, 어휘 50,257개, 길이 1024)의 파라미터 수가 GPT-2 small과 같은 약 124M으로 나오는 것도 확인할 수 있습니다.

원래 Transformer와 달라진 점을 정리하면 다음과 같습니다.

- **Cross-Attention 제거**
- **학습되는 Positional Embedding**: sinusoidal 대신 위치마다 임베딩을 학습 (GPT-1~3)
- **Pre-LN**: LayerNorm을 sub-layer **앞**에 둠 (GPT-2부터). 깊게 쌓아도 학습이 안정적
- **GELU 활성 함수**, 입력 임베딩과 출력층의 **weight tying**

## 5. 텍스트 생성: 한 토큰씩

학습은 병렬이지만 생성은 **한 토큰씩** 합니다. 지금까지의 토큰을 넣고 → 다음 토큰의 확률 분포를 얻고 → 하나를 뽑아 뒤에 붙이고 → 반복합니다.

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :] / temperature            # 마지막 위치만 사용
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
```

(실제 구현에서는 이전 토큰의 K, V를 다시 계산하지 않도록 **KV cache**를 씁니다.)

### 디코딩 전략

| 방법 | 동작 | 특징 |
|---|---|---|
| Greedy | 항상 확률 최대 토큰 | 결정적이지만 반복적이고 단조로움 |
| Temperature | logit을 $$T$$로 나눔 | $$T<1$$이면 보수적, $$T>1$$이면 다양함 |
| Top-k | 상위 $$k$$개 중에서만 샘플링 | 엉뚱한 토큰 방지 |
| Top-p (nucleus) | 누적 확률 $$p$$까지의 후보에서 샘플링 | 분포 모양에 따라 후보 수가 자동 조절 |

## 6. GPT-1 → GPT-2 → GPT-3: 규모가 바꾼 것

### GPT-1 (2018): Pre-training + Fine-tuning

- 12층, 약 **117M** 파라미터, BooksCorpus로 사전학습
- 사전학습 뒤 분류·추론·유사도 등 **과제마다 fine-tuning**
- 메시지: "레이블 없는 텍스트로 미리 학습하면, 적은 레이블로도 여러 과제를 잘 푼다"

### GPT-2 (2019): Zero-shot

- 최대 **1.5B** 파라미터, 웹 문서로 만든 WebText로 학습
- **fine-tuning 없이** 프롬프트만으로 요약, 번역, 질의응답을 어느 정도 수행 (zero-shot)
- 메시지: "모든 과제는 결국 다음 토큰 예측의 특수한 경우다." 예를 들어 `"TL;DR:"` 뒤를 이어 쓰게 하면 요약이 됩니다.

### GPT-3 (2020): In-Context Learning

- **175B** 파라미터
- 프롬프트 안에 예시 몇 개만 보여주면(**few-shot**) 가중치 업데이트 없이 새 과제를 수행
- 이를 **In-Context Learning**이라고 부르며, 모델 크기가 커질수록 이 능력이 뚜렷해짐

```
영어를 한국어로 번역하세요.
cat → 고양이
dog → 강아지
bird →
```

모델은 "→ 뒤에 한국어 번역을 쓰는 패턴"을 프롬프트 안에서 파악하고 `새`를 이어 씁니다. **학습 없이 문맥만으로 과제를 배우는 셈**입니다.

비슷한 시기에 발표된 **Scaling Law** 연구(Kaplan et al., 2020)는 언어 모델의 loss가 **파라미터 수, 데이터 양, 연산량에 대해 거듭제곱 법칙**을 따라 예측 가능하게 줄어든다는 것을 보였습니다. "모델을 키우면 좋아진다"가 경험이 아니라 **예측 가능한 법칙**이 된 것입니다.

이후 **Chinchilla** 연구(Hoffmann et al., 2022)는 같은 연산량이라면 모델 크기와 학습 토큰 수를 **비슷한 비율로 함께** 늘려야 최적이며(대략 파라미터 1개당 토큰 20개), GPT-3 같은 당시 모델들은 크기에 비해 데이터가 부족했다고 수정했습니다. 70B 파라미터의 Chinchilla가 더 큰 모델들을 이긴 것이 그 근거였습니다.

## 7. 다음 토큰 예측만으로는 부족한 것

GPT-3는 똑똑했지만 **"인터넷 문서를 이어 쓰는 모델"**이었습니다. 질문을 하면 답을 하는 대신 비슷한 질문을 더 늘어놓기도 했고, 사실이 아닌 내용을 그럴듯하게 쓰거나 유해한 내용을 생성하기도 했습니다.

학습 목표("다음 토큰을 잘 맞히기")와 우리가 원하는 것("사용자의 의도대로 유용하고 안전하게 답하기") 사이에 차이가 있었던 것입니다. 이 차이를 줄이는 과정을 **Alignment**라고 하고, 대표적인 방법이 다음 두 단계입니다.

1. **Instruction Tuning (SFT)**: "지시–응답" 쌍으로 지도학습
2. **RLHF**: 사람이 응답들의 선호 순위를 매기고 → 이를 학습한 **보상 모델**을 만든 뒤 → **PPO**로 언어 모델을 최적화

여기서 RL 시리즈의 [PPO 글]({% post_url 2026-07-05-trpo-to-ppo %})과 연결됩니다. RLHF에서 PPO의 clipping과 KL penalty는 "보상을 올리되 원래 언어 모델에서 너무 멀어지지 않게" 하는 역할을 합니다.

## 8. 정리

| 키워드 | 핵심 |
|---|---|
| Autoregressive LM | 다음 토큰 확률의 곱으로 문장 확률을 모델링, 레이블 불필요 |
| Decoder-only | Cross-Attention 없는 Masked Self-Attention 블록만 쌓음 |
| Causal Mask | 미래 토큰을 가려 한 문장으로 $$n$$개 예측을 병렬 학습 |
| 생성 | 한 토큰씩 샘플링, temperature·top-k·top-p로 다양성 조절 |
| 규모 | GPT-1 fine-tuning → GPT-2 zero-shot → GPT-3 in-context learning |
| Alignment | SFT + RLHF(PPO)로 "이어 쓰기"를 "지시 따르기"로 |

GPT-3를 사람의 지시를 따르는 모델로 바꾼 **InstructGPT와 RLHF**의 3단계 학습 과정은 이 시리즈의 다음 주제로 정리할 예정입니다.

## References

- Radford et al., "Improving Language Understanding by Generative Pre-Training," 2018. (GPT-1)
- Radford et al., "Language Models are Unsupervised Multitask Learners," 2019. (GPT-2)
- Brown et al., "Language Models are Few-Shot Learners," NeurIPS 2020. (GPT-3)
- Kaplan et al., "Scaling Laws for Neural Language Models," 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann et al., "Training Compute-Optimal Large Language Models (Chinchilla)," NeurIPS 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Ouyang et al., "Training language models to follow instructions with human feedback," NeurIPS 2022.
- Andrej Karpathy, nanoGPT — <https://github.com/karpathy/nanoGPT>
