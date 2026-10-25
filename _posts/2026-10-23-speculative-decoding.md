---
title: "[LLM] Speculative Decoding: 작은 모델이 추측하고 큰 모델이 검증하는 무손실 디코딩 가속"
date: 2026-10-23 09:00:00 +0900
categories: [Transformer, LLM]
tags: [llm, speculative-decoding, speculative-sampling, inference-acceleration, rejection-sampling, autoregressive, t5, memory-bandwidth]
description: 작은 근사 모델이 토큰 여러 개를 먼저 추측하고 큰 목표 모델이 한 번의 병렬 실행으로 검증해, 출력 분포를 바꾸지 않고 T5-XXL 디코딩을 2~3배 빠르게 만든 Speculative Decoding(Leviathan et al., ICML 2023)의 알고리즘과 이론 분석을 코드로 확인합니다.
math: true
---

> [Transformer 디코더에서 GPT로]({% post_url 2026-07-01-transformer-decoder-to-gpt %}) 글에서 본 것처럼, decoder-only 언어 모델은 토큰을 하나씩 순차적으로 만듭니다. 토큰 $$K$$개를 만들려면 큰 모델을 $$K$$번 직렬로 돌려야 합니다. **Speculative Decoding**(Leviathan, Kalman, Matias, Google Research, ICML 2023)은 작은 모델이 여러 토큰을 먼저 추측하고 큰 모델이 이를 한 번에 검증하는 방법으로, **출력 분포를 전혀 바꾸지 않고** T5-XXL의 디코딩을 2~3배 빠르게 만들었습니다. 모델 구조나 학습을 바꿀 필요가 없습니다.

## 1. 문제: 큰 모델의 디코딩은 연산이 아니라 메모리에 묶여 있다

큰 autoregressive 모델의 추론이 느린 이유는 두 가지입니다. 한 step이 작은 모델보다 느리고, 그 step이 토큰 수만큼 직렬로 쌓입니다. 이를 줄이는 기존 접근은 두 갈래였습니다.

- **모든 입력을 똑같이 싸게**: distillation, sparsification, quantization, 구조 변경
- **쉬운 step에는 덜 쓰기**(adaptive computation): early exit처럼 쉬운 토큰은 얕은 계산으로 끝내기

둘 다 효과가 있지만 보통 구조나 학습 절차를 바꿔야 하고, 무엇보다 **출력이 원래 모델과 달라집니다**. 이 논문은 두 가지 관찰에서 출발합니다.

1. 어려운 언어 모델링 작업에도 **쉬운 부분 작업**이 섞여 있고, 그 부분은 작은 모델로도 잘 근사됩니다.
2. 큰 모델의 추론은 산술 연산이 아니라 **메모리 대역폭과 통신**에 묶여 있어서, 추가 연산 자원이 남아돕니다.

그래서 계산을 줄이는 대신 **동시성을 늘리는** 방향을 택합니다. CPU의 branch prediction처럼, 필요할지 확실하지 않은 작업을 미리 병렬로 해 두고 나중에 검증하는 **speculative execution**을 확률적 상황으로 일반화한 것입니다.

## 2. 알고리즘

목표 모델을 $$M_p$$, 그 분포를 $$p(x_t \mid x_{<t})$$라 하고, 같은 작업을 더 싸게 근사하는 모델을 $$M_q$$, 분포를 $$q(x_t \mid x_{<t})$$라 합니다. 한 번의 반복은 세 단계입니다.

1. $$M_q$$로 토큰 $$\gamma$$개를 순차적으로 추측합니다.
2. $$M_p$$를 **병렬로 한 번** 돌려 prefix, prefix + 추측 1개, ..., prefix + 추측 $$\gamma$$개에 대한 분포 $$p_1, \ldots, p_{\gamma+1}$$을 모두 얻습니다.
3. 추측을 앞에서부터 검증해 받아들일 수 있는 것은 받아들이고, 처음 거절된 자리에서 **조정된 분포**로 한 토큰을 뽑습니다. 전부 받아들여졌으면 $$p_{\gamma+1}$$에서 한 토큰을 더 뽑습니다.

그래서 $$M_p$$를 한 번 돌릴 때마다 **최소 1개, 최대 $$\gamma + 1$$개**의 토큰이 나옵니다. 최악의 경우에도 일반 디코딩보다 $$M_p$$ 호출 횟수가 많아지지 않습니다.

### Speculative sampling: 분포를 지키는 수락 규칙

핵심은 $$q$$에서 뽑은 샘플을 어떻게 받아들여야 결과가 정확히 $$p$$를 따르는가입니다. $$x \sim q(x)$$를 뽑은 뒤 다음과 같이 합니다.

- $$q(x) \le p(x)$$이면 받아들입니다.
- $$q(x) > p(x)$$이면 확률 $$1 - p(x)/q(x)$$로 거절하고, 조정 분포 $$p'(x) = \text{norm}(\max(0, p(x) - q(x)))$$에서 다시 뽑습니다.

즉 수락 확률은 $$\min(1, p(x)/q(x))$$입니다. 결과가 $$p$$를 따른다는 증명은 짧습니다. 토큰 $$x'$$가 나올 확률은 수락되어 나오는 경우와 거절 후 다시 뽑혀 나오는 경우의 합입니다.

$$
P(x = x') = \underbrace{q(x') \min\!\Big(1, \frac{p(x')}{q(x')}\Big)}_{\min(p(x'), q(x'))} + \underbrace{(1 - \beta)\, p'(x')}_{p(x') - \min(p(x'), q(x'))} = p(x')
$$

여기서 $$\beta$$는 수락 확률이고, 조정 분포의 정규화 상수가 정확히 $$1 - \beta$$라는 점이 맞아떨어집니다. 어떤 $$p$$, $$q$$에 대해서도 성립하므로 **근사 모델은 아무거나 써도** 출력 분포가 보존됩니다.

argmax, top-k, nucleus, temperature 같은 샘플링 방식은 모두 "조정된 분포에서의 표준 샘플링"으로 바꿀 수 있습니다(argmax는 최대 원소만 남기고 정규화한 분포). 그래서 $$p$$와 $$q$$를 그렇게 조정한 뒤 같은 절차를 쓰면 됩니다.

### 일반적인 rejection sampling과의 차이

겉보기에 비슷한 rejection sampling은 $$M = \max_x p(x)/q(x)$$로 나눈 $$p(x)/(Mq(x))$$를 수락 확률로 쓰고 거절되면 처음부터 반복합니다. 반복하지 않고 거절 시 $$p$$에서 직접 뽑는 변형도 가능하지만, 기대 수락 확률이 $$\sum_x p(x) \min_{x'} q(x')/p(x')$$로 speculative sampling의 $$\sum_x \min(p(x), q(x))$$보다 (경우에 따라 훨씬) 낮습니다. 조정 분포 $$\max(0, p - q)$$에서 다시 뽑는 설계가 수락률을 최대로 끌어올리는 셈입니다.

## 3. 분석: 얼마나 빨라지나

### 수락률 α와 반복당 토큰 수

prefix가 주어졌을 때 $$q$$의 추측이 수락될 확률을 $$\beta$$라 하고, 그 기댓값을 $$\alpha = \mathbb{E}(\beta)$$라 합니다. $$\alpha$$는 $$M_q$$가 $$M_p$$를 얼마나 잘 근사하는지의 척도입니다. 각 $$\beta$$가 i.i.d.라고 단순화하면 반복 한 번에 나오는 토큰 수는 성공 확률 $$1 - \alpha$$, 상한 $$\gamma + 1$$인 기하 분포를 따르고, 기댓값은 다음과 같습니다.

$$
\mathbb{E}(\text{토큰 수}) = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}
$$

$$\alpha$$를 계산하는 닫힌 식도 있습니다. $$D_{LK}(p, q) = \sum_x |p(x) - M(x)|$$, $$M = (p + q)/2$$로 정의하면 $$D_{LK} = 1 - \sum_x \min(p(x), q(x))$$이고, 수락 확률은 $$\beta = 1 - D_{LK}(p, q)$$입니다. 따라서

$$
\alpha = \mathbb{E}\big[\min(p, q)\big] = 1 - \mathbb{E}\big[D_{LK}(p, q)\big]
$$

$$D_{LK}$$는 0과 1 사이의 대칭 divergence로, 0이면 $$p = q$$이고 1이면 두 분포의 support가 겹치지 않습니다.

### 벽시계 시간과 연산량

$$M_q$$ 한 번의 실행 시간을 $$M_p$$ 한 번의 실행 시간으로 나눈 **비용 계수**를 $$c$$라 합니다. $$\gamma + 1$$개의 병렬 평가가 한 번의 실행과 같은 시간에 끝난다고 가정하면, 한 반복의 비용은 $$T(c\gamma + 1)$$이고 기대 토큰 수로 나누면 다음 개선 배율이 나옵니다.

$$
\text{속도 향상} = \frac{1 - \alpha^{\gamma + 1}}{(1 - \alpha)(\gamma c + 1)}
$$

$$\alpha > c$$이면 개선되는 $$\gamma$$가 존재하고, $$\gamma = 1$$만 놓고 봐도 최소 $$(1 + \alpha)/(1 + c)$$배입니다. 실험에서 $$M_q$$가 $$M_p$$보다 두 자릿수 작을 때 $$c$$는 늘 0.05 미만이었습니다.

대신 **총 연산량은 늘어납니다**. $$M_p$$를 $$\gamma + 1$$개 위치에서 돌리므로 거절된 추측만큼 계산이 버려지고, 증가 배율은 $$(1 - \alpha)(\gamma \hat c + \gamma + 1)/(1 - \alpha^{\gamma + 1})$$입니다($$\hat c$$는 토큰당 연산량 비율). 반면 가중치와 KV cache를 읽는 **메모리 접근 횟수**는 반복당 한 번이므로 기대 토큰 수만큼 줄어듭니다. 메모리 대역폭이 병목인 상황에 꼭 맞는 교환입니다.

| $$\alpha$$ | $$\gamma$$ | 연산량 | 속도 |
|---|---|---|---|
| 0.6 | 2 | 1.53배 | 1.96배 |
| 0.7 | 3 | 1.58배 | 2.53배 |
| 0.8 | 5 | 1.63배 | 3.69배 |
| 0.9 | 10 | 1.60배 | 6.86배 |

($$c = \hat c = 0$$ 가정, 논문 Table 1)

### γ 고르기

$$c$$와 $$\alpha$$가 주어지면 위 식을 최대로 하는 정수 $$\gamma$$를 수치로 찾으면 됩니다. $$\beta$$가 일정하지 않으므로 실행 중 $$\beta$$를 예측해 $$\gamma$$를 바꾸면 더 좋아질 수 있는데, 완벽한 oracle이 있다면 기대 토큰 수가 $$1/(1 - \alpha)$$까지 올라가 최대 약 60% 추가 개선이 가능하다는 상한을 제시하고 향후 과제로 남겼습니다.

## 4. 코드로 보기: 분포는 그대로, 호출은 1/3

bigram 분포표 두 개를 목표 모델 $$p$$와 근사 모델 $$q$$로 두고 Algorithm 1을 그대로 구현했습니다. $$q$$는 $$p$$의 log 확률에 노이즈를 섞어 "비슷하지만 다른" 모델로 만들었습니다.

```python
import torch

torch.manual_seed(0)
V, GAMMA = 50, 5                                           # 어휘 크기, 한 번에 추측할 토큰 수

# 1) 장난감 모델: 직전 토큰만 보는 bigram 분포표. q는 p에 노이즈를 섞어 "비슷하지만 다른" 작은 모델로 만든다
p_table = torch.softmax(torch.randn(V, V) * 2.0, dim=-1)   # 목표 모델 p(x_t | x_{t-1})
q_table = torch.softmax(torch.log(p_table) + 0.8 * torch.randn(V, V), dim=-1)  # 근사 모델 q

def speculative_step(prefix):
    """Algorithm 1: q로 γ개를 추측하고 p로 한 번에 검증해 1 ~ γ+1개의 토큰을 돌려준다."""
    guesses, qs = [], []
    last = prefix[-1]
    for _ in range(GAMMA):                                 # q를 γ번 순차 실행
        q = q_table[last]
        x = torch.multinomial(q, 1).item()
        guesses.append(x); qs.append(q); last = x
    ps = [p_table[t] for t in [prefix[-1]] + guesses]      # p는 γ+1개 위치를 병렬로 평가(여기서는 표 조회)
    n = GAMMA
    for i, x in enumerate(guesses):
        if torch.rand(1).item() > min(1.0, ps[i][x] / qs[i][x]):   # 확률 min(1, p/q)로 수락
            n = i
            break
    if n == GAMMA:
        extra = ps[GAMMA]                                  # 전부 수락: p에서 한 토큰 더
    else:
        extra = torch.clamp(ps[n] - qs[n], min=0)          # 거절: 조정 분포 norm(max(0, p - q))
        extra = extra / extra.sum()
    t = torch.multinomial(extra, 1).item()
    return guesses[:n] + [t]

# 2) 같은 조건에서 p 단독 샘플링과 speculative decoding의 다음 토큰 분포를 비교한다
start = 7
spec_counts = torch.zeros(V)
for _ in range(20000):
    spec_counts[speculative_step([start])[0]] += 1
print(f"p(x | {start})와 speculative 첫 토큰 분포의 최대 차이: {(spec_counts / spec_counts.sum() - p_table[start]).abs().max():.4f}")

# 3) 긴 문장을 생성하며 반복당 토큰 수를 재고, α = E[min(p, q)]로 예측한 값과 비교한다
alpha = (torch.minimum(p_table, q_table).sum(-1)).mean().item()   # 모든 prefix에 대한 평균 수락률
seq, calls = [start], 0
while len(seq) < 20000:
    seq += speculative_step(seq); calls += 1
print(f"α(이론) = {alpha:.3f}")
print(f"목표 모델 호출 1회당 토큰 수: 실측 {len(seq) / calls:.2f}, 이론 (1-α^(γ+1))/(1-α) = {(1 - alpha ** (GAMMA + 1)) / (1 - alpha):.2f}")
for c in [0.0, 0.02, 0.1]:
    best = max(range(1, 16), key=lambda g: (1 - alpha ** (g + 1)) / ((1 - alpha) * (g * c + 1)))
    print(f"c = {c:.2f}: 최적 γ = {best:2d}, 예상 속도 향상 {(1 - alpha ** (best + 1)) / ((1 - alpha) * (best * c + 1)):.2f}배")
```

```text
p(x | 7)와 speculative 첫 토큰 분포의 최대 차이: 0.0007
α(이론) = 0.753
목표 모델 호출 1회당 토큰 수: 실측 3.30, 이론 (1-α^(γ+1))/(1-α) = 3.31
c = 0.00: 최적 γ = 15, 예상 속도 향상 4.00배
c = 0.02: 최적 γ =  9, 예상 속도 향상 3.23배
c = 0.10: 최적 γ =  5, 예상 속도 향상 2.21배
```

2만 번 뽑은 첫 토큰의 경험 분포가 $$p$$와 0.0007 이내로 같습니다. 샘플링 노이즈 수준이므로 $$q$$를 거쳐 왔다는 흔적이 분포에 남지 않습니다. $$\alpha = \mathbb{E}[\min(p, q)]$$로 계산한 0.753을 식에 넣으면 호출당 3.31개의 토큰이 예상되고, 실측 3.30개와 맞습니다. 목표 모델 호출이 약 1/3로 줄어든 것입니다.

마지막 세 줄은 같은 $$\alpha$$에서 $$c$$에 따라 최적 $$\gamma$$가 어떻게 달라지는지 보여 줍니다. 근사 모델이 공짜에 가까우면($$c = 0$$) $$\gamma$$를 키울수록 좋지만 상한은 $$1/(1 - \alpha) \approx 4$$배이고, $$c = 0.1$$이면 $$\gamma = 5$$ 근처에서 멈추는 것이 낫습니다. 논문 Figure 3과 같은 경향입니다.

## 5. 실험

### T5-XXL 벽시계 시간

T5 1.1 인코더-디코더 모델로 영어-독일어 번역(WMT EnDe)과 요약(CNN/DM)을 실험했습니다. $$M_p$$는 T5-XXL(11B), $$M_q$$는 기존 체크포인트인 T5-large(800M), T5-base(250M), T5-small(77M)이고, TPU-v4 한 개에서 batch size 1로 쟀습니다.

| 작업 | $$M_q$$ | 온도 | $$\gamma$$ | $$\alpha$$ | 속도 |
|---|---|---|---|---|---|
| EnDe | T5-small | 0 | 7 | 0.75 | 3.4배 |
| EnDe | T5-base | 0 | 7 | 0.80 | 2.8배 |
| EnDe | T5-large | 0 | 7 | 0.82 | 1.7배 |
| EnDe | T5-small | 1 | 7 | 0.62 | 2.6배 |
| CNN/DM | T5-small | 0 | 5 | 0.65 | 3.1배 |
| CNN/DM | T5-small | 1 | 5 | 0.53 | 2.3배 |

- $$\alpha$$는 근사 모델이 클수록 높지만, $$c$$도 커져서 **가장 작은 T5-small이 가장 빠릅니다**. 두 자릿수 작은 모델이 균형점이었습니다.
- argmax(온도 0)에서 $$\alpha$$가 더 높습니다. 분포가 날카로울수록 두 모델이 같은 답을 내기 쉽기 때문입니다.
- 이론 예측(Theorem 3.8)과 실측이 대체로 맞고, 차이는 구현 최적화 차이와 $$\beta$$의 i.i.d. 가정 때문입니다.

### 여러 모델의 α

GPT-like 97M 모델(근사 6M), T5-XXL, LaMDA 137B(근사 100M, 2B, 8B)에서 $$\alpha$$를 쟀습니다. 두 자릿수 작은 모델은 0.5~0.9의 $$\alpha$$를 냈고, LaMDA 137B에 LaMDA 8B를 쓰면 0.75였습니다. 흥미롭게도 unigram, bigram 같은 **표 조회 수준의 근사 모델**도 0이 아닌 $$\alpha$$를 냅니다. EnDe 번역에서 bigram은 $$\alpha \approx 0.2$$였고, $$c = 0$$이므로 $$\gamma = 3$$에서 1.25배 빨라집니다.

### 근사 모델의 선택지

- **같은 구조의 작은 모델**: 실험에서 주로 쓴 방식. 이미 학습된 체크포인트를 그대로 씁니다.
- **비용이 0에 가까운 모델**: n-gram 표, 또는 문맥에서 일치하는 prefix가 있으면 그 뒤를 복사하는 휴리스틱. 요약이나 코드 수정처럼 긴 구간이 반복되는 작업에서 $$\alpha$$가 높을 수 있고 배포도 간단합니다.
- **non-autoregressive 모델**: $$\gamma$$번 순차 호출 대신 한 번에 추측합니다.
- **무작위 토큰**: 이론적으로만 흥미로운 극단으로, 어떤 $$M_p$$에서도 아주 작은 개선을 보장합니다.

### Lenience: 분포를 조금 양보하면

출력 분포를 반드시 지키지 않아도 된다면 $$q(x)$$에 $$l \in [0, 1]$$을 곱해 비교하는 **lenience**를 쓸 수 있습니다. 어떤 토큰도 원래 확률의 $$1/l$$배보다 자주 뽑히지 않는다는 보장은 남습니다. T5-XXL에 T5-small을 쓸 때 $$l = 1, 0.5, 0.3, 0.1$$에서 2.5배, 3.1배, 3.6배, 5배로 빨라졌습니다. 본문의 다른 결과는 모두 lenience 없이 가장 엄격한 버전입니다.

## 6. 한계와 향후 과제

- 지연 시간을 줄이는 대가로 **총 연산량이 늘어납니다**. 추가 연산 자원이 없는 환경(이미 compute-bound인 상황)에서는 도움이 되지 않습니다.
- beam search에는 성능 손실을 감수하고 적용할 수 있습니다. $$M_q$$로 더 넓은 beam($$u \ge w$$)을 $$\gamma$$ step 돌린 뒤 $$M_p$$로 후보를 병렬 검증하고, $$\text{top}_w(M_p) \subseteq \text{top}_u(M_q)$$인 동안 받아들이면 결과가 같습니다.
- 근사 모델을 $$M_p$$의 soft target으로 증류하거나 $$\alpha$$를 직접 최적화하는 학습, 근사 모델을 더 빠른 모델로 다시 가속하는 계층 구조, 실행 중 $$\gamma$$를 바꾸는 방법은 모두 남겨 둔 과제입니다. 텍스트 외의 modality도 시험하지 않았습니다.

## 7. 해석

**1. "무손실"이 이 논문의 핵심 가치입니다.** 추론을 빠르게 하는 방법은 많지만 대부분 출력이 달라집니다. speculative sampling은 어떤 근사 모델을 써도 출력 분포가 수학적으로 같다는 보장을 주기 때문에, 품질 평가를 다시 할 필요 없이 서빙 시스템에 끼워 넣을 수 있습니다. 위 예제에서 2만 번 샘플링한 분포가 $$p$$와 구별되지 않은 것이 그 보장입니다.

**2. 병목이 어디인지 정확히 짚었습니다.** 큰 모델의 디코딩은 토큰 하나를 만들 때마다 가중치 전체를 메모리에서 읽어야 해서 연산 장치가 놀고 있습니다. 그 남는 연산으로 $$\gamma + 1$$개 위치를 한꺼번에 평가하면, 메모리 읽기 한 번으로 여러 토큰을 얻습니다. 연산량이 1.5배 늘어도 벽시계 시간이 2~3배 줄어드는 이유입니다. 같은 관찰을 거의 동시에 독립적으로 한 [DeepMind의 Speculative Sampling]({% post_url 2026-10-23-speculative-sampling %})은 분산 서빙 환경에서 Chinchilla 70B로 이를 확인했습니다.

**3. 수락률 α가 모든 것을 결정합니다.** 속도 향상 식의 변수는 $$\alpha$$, $$\gamma$$, $$c$$ 셋뿐이고, 그중 $$\gamma$$는 나머지 둘로 결정되며 $$c$$는 하드웨어 문제입니다. 결국 "근사 모델이 목표 모델을 얼마나 닮았나"가 전부입니다. 그래서 후속 연구는 $$\alpha$$를 올리는 쪽으로 흘러갔습니다. 서빙 중 실제 질의로 근사 모델을 계속 증류하는 [Online Speculative Decoding]({% post_url 2026-10-23-online-speculative-decoding %})이 그 예입니다.

**4. 작은 모델일수록 좋다는 역설.** T5-large가 $$\alpha$$는 가장 높은데 속도는 가장 느렸습니다. 수락률이 조금 오르는 것보다 추측 비용 $$c$$가 커지는 손해가 큽니다. 속도 향상의 상한이 $$1/(1 - \alpha)$$라는 점을 생각하면, $$\alpha$$ 0.75와 0.82의 차이는 4배와 5.6배의 차이지만 $$c$$가 0.02에서 0.11로 늘면 $$\gamma$$를 키울 수 없어 그 상한에 접근하지 못합니다.

**5. 평가를 읽을 때 주의할 점.** 벽시계 시간은 T5-XXL, batch size 1, TPU-v4 한 개에서만 쟀습니다. batch가 커지면 연산 장치가 바빠져 남는 자원이 줄고, 분산 환경에서는 통신이 끼어듭니다. LaMDA와 GPT-like 모델은 $$\alpha$$만 쟀고 실제 속도는 측정하지 않았습니다.

## 8. 정리

| 항목 | 내용 |
|---|---|
| 목표 | 큰 autoregressive 모델의 디코딩을 출력 분포 변화 없이 가속 |
| 방법 | 작은 모델 $$M_q$$로 $$\gamma$$개 추측 → $$M_p$$로 병렬 검증 → speculative sampling으로 수락·거절 |
| 수락 규칙 | $$x \sim q$$를 확률 $$\min(1, p/q)$$로 수락, 거절 시 $$\text{norm}(\max(0, p - q))$$에서 재샘플링 |
| 이론 | $$\alpha = \mathbb{E}[\min(p, q)]$$, 호출당 토큰 $$(1 - \alpha^{\gamma+1})/(1 - \alpha)$$, 속도 $$\frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}$$ |
| 결과 | T5-XXL에 T5-small을 쓸 때 2.6~3.4배(번역), 2.3~3.1배(요약). 출력 동일 |
| 비용 | 총 연산량 증가(거절된 추측), 추가 연산 자원 필요 |

## References

- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding," ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling," 2023. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Stern, Shazeer, Uszkoreit, "Blockwise Parallel Decoding for Deep Autoregressive Models," NeurIPS 2018. [arXiv:1811.03115](https://arxiv.org/abs/1811.03115)
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
