---
title: "[LLM] Stealing Part of a Production Language Model: API 질의만으로 마지막 층과 숨은 차원을 훔치기"
date: 2026-10-23 11:40:00 +0900
categories: [Transformer, LLM]
tags: [llm, model-stealing, model-extraction, security, svd, logit-bias, logprobs, embedding-projection, openai-api]
description: 블랙박스 API의 logprob과 logit bias만으로 Transformer의 마지막 embedding projection 층과 숨은 차원을 복원한 Stealing Part of a Production Language Model(Carlini et al., 2024)의 SVD 기반 공격, 제한된 API를 우회하는 기법, 비용과 방어책을 정리하고 작은 모델로 공격 원리를 재현합니다.
math: true
---

> GPT-4나 PaLM-2의 기술 보고서는 모델 크기조차 공개하지 않습니다. 그런데 모델은 API로 노출되어 있습니다. **Stealing Part of a Production Language Model**(Carlini et al., Google DeepMind·ETH Zurich 등, 2024)은 API 질의만으로 **마지막 층(embedding projection)을 대칭성을 제외하고 통째로** 복원하는 첫 공격을 제시했습니다. 20달러 미만으로 OpenAI의 ada와 babbage의 projection 행렬 전체를 뽑아 숨은 차원이 각각 1024와 2048임을 확인했고, gpt-3.5-turbo의 숨은 차원도 알아냈습니다. 책임 있는 공개 절차를 거쳐 OpenAI와 Google은 API를 수정했습니다.

## 1. 문제 설정

모델 훔치기(model stealing)는 질의 접근만으로 모델의 가중치를 복원하려는 분야입니다. 정확도를 맞추는 공격과 모든 입력에서 함수적으로 동일한 모델을 얻으려는 **고충실도(high-fidelity)** 공격이 있는데, 이 논문은 후자입니다. 기존 고충실도 공격은 ReLU 신경망의 성질을 이용했고, 토큰 입력, 다른 활성화 함수, attention과 LayerNorm, 거대한 규모, 제한된 출력 정밀도 때문에 프로덕션 LLM에는 전혀 통하지 않았습니다.

이 논문의 접근은 **위에서 아래로**입니다. 입력층부터 재구성하는 대신 마지막 층을 바로 뽑습니다. 모델을 다음과 같이 봅니다.

$$
f_\theta(p) = \text{softmax}\big(W \cdot g_\theta(p)\big)
$$

$$g_\theta: \mathcal{X}^N \to \mathbb{R}^h$$는 숨은 상태를 만드는 본체, $$W$$는 $$l \times h$$의 embedding projection 행렬입니다. 핵심은 **숨은 차원 $$h$$가 어휘 크기 $$l$$보다 훨씬 작다**는 점입니다. LLaMA는 $$h \in \{4096, \ldots, 8192\}$$에 $$l = 32{,}000$$이고, GPT-4의 어휘는 약 10만입니다. 그래서 마지막 층은 low-rank이고, 이 성질이 모든 공격의 출발점입니다.

위협 모델은 완벽한 oracle입니다. 공격자는 모델 파라미터를 전혀 모르고, API는 입력에 대한 출력만 돌려주며 타이밍 같은 부채널은 없습니다. API의 종류에 따라 공격을 세 단계로 전개합니다.

| API | 동기 |
|---|---|
| 모든 logit (2절) | 교육용이자 다음 공격의 기초 |
| top-K logprob + logit bias (3절) | 당시의 실제 LLM API (OpenAI 등) |
| logprob 없이 logit bias만 (4절) | 앞으로 나올 수 있는 더 제한된 API |

## 2. logit 벡터 API에 대한 공격

### 숨은 차원 알아내기

무작위 prefix $$n$$개로 모델을 질의해 logit 벡터를 행으로 쌓은 행렬 $$Q$$를 만듭니다. 각 logit은 $$l$$차원이지만 모두 $$W$$가 $$h$$차원에서 올려 보낸 것이라 **$$h$$차원 부분공간** 안에 있습니다. $$n > h$$가 되면 새 질의는 이전 질의들의 선형 결합이 되고, $$Q$$의 rank가 $$h$$에서 멈춥니다.

$$
Q = W \cdot H \quad \Rightarrow \quad \text{rank}(Q) \le h
$$

$$H$$는 $$g_\theta(p_i)$$를 열로 쌓은 $$h \times n$$ 행렬이고, $$H$$와 $$W$$가 모두 rank $$h$$이면 등호가 성립합니다. 실수가 아니라 부동소수점(16비트, 8비트까지도)으로 계산하므로 rank를 그대로 세지 않고, 특잇값을 정렬해 **연속한 특잇값 사이의 곱셈 낙차 $$\lambda_i / \lambda_{i+1}$$가 가장 큰 자리**를 찾습니다. 진짜 특잇값에서 수치 오차에 의한 특잇값으로 넘어가는 지점입니다.

| 모델 | 숨은 차원 | 복원값 | $$W$$ RMS |
|---|---|---|---|
| GPT-2 Small (fp32) | 768 | 757 ± 1 | $$4 \cdot 10^{-4}$$ |
| GPT-2 XL (fp32) | 1600 | 1599 ± 1 | $$6 \cdot 10^{-4}$$ |
| Pythia-1.4B (fp16) | 2048 | 2047 ± 1 | $$3 \cdot 10^{-5}$$ |
| Pythia-6.9B (fp16) | 4096 | 4096 ± 1 | $$4 \cdot 10^{-5}$$ |
| LLaMA 7B (fp16) | 4096 | 4096 ± 2 | $$8 \cdot 10^{-5}$$ |
| LLaMA 65B (fp16) | 8192 | 8192 ± 2 | $$5 \cdot 10^{-5}$$ |

GPT-2 Small만 757로 11 차이가 나는데, 부록에서 이 모델의 마지막 숨은 활성화가 실제로 757개 방향만 쓴다는 것을 보입니다. 공격이 틀린 것이 아니라 모델의 **유효 차원**을 찾은 것입니다. float64로 돌리면 모든 차원이 쓰이지만 가장 작은 열두 개쯤의 특잇값이 나머지보다 훨씬 작습니다.

### 마지막 층 복원

$$Q = U \Sigma V^\top$$로 분해하면 $$U \Sigma$$가 **마지막 층의 회전된 버전**입니다. 어떤 $$h \times h$$ 행렬 $$G$$에 대해 $$\tilde W = U \Sigma = W \cdot G$$입니다. residual connection이 있는 Transformer에서는 $$W$$를 정확히 복원하는 것이 **불가능**하다는 것도 증명합니다. 숨은 상태의 기저를 바꿔도 같은 함수가 되기 때문입니다.

평가는 $$\tilde W G \approx W$$를 최소제곱으로 풀어 $$G$$를 맞춘 뒤 RMS를 잽니다. 위 표의 오차는 무작위 초기화 행렬의 $$2 \cdot 10^{-2}$$보다 100~500배 작습니다. 부록에서는 직교 변환까지만 모호한(놓치는 파라미터가 $$h^2$$이 아니라 약 $$h^2/2$$) 복원이 가능하고 그것이 특정 가정 아래 tight하다는 것도 보이지만, 효율적인 알고리즘은 affine 변환까지만 있습니다.

## 3. 실제 API: top-5 logprob과 logit bias

어떤 프로덕션 모델도 logit 전체를 주지 않습니다. 대신 **상위 K개 토큰의 logprob**과, 지정한 토큰의 logit에 더할 실수 **logit bias** $$b$$를 받습니다.

$$
O(p, b) \leftarrow \text{TopK}\big(\text{logsoftmax}(W g_\theta(p) + b)\big)
$$

OpenAI는 토큰 최대 300개에 $$[-100, 100]$$ 범위의 bias를 허용했습니다. 과제는 이 API로 prefix마다 **logit 벡터 전체**를 복원하는 것이고, 그다음은 2절 공격을 그대로 적용하면 됩니다.

### 기준 토큰으로 4개씩

top-5가 logit이라면 토큰 5개씩 큰 bias $$B$$를 더해 top-5로 밀어 올리고 $$B$$를 빼면 됩니다. 그러나 돌아오는 것은 logprob이라 bias에 의존하는 정규화 항이 붙습니다.

$$
y^B_i = z_i + B - \log\Big(\sum_{j \ne i} e^{z_j} + e^{z_i + B}\Big)
$$

해결은 **기준 토큰** $$R$$(원래 top-1)과의 차이를 보는 것입니다. 토큰 $$i$$에 $$B$$를 더해 top-5에 넣고 $$R$$도 함께 관찰하면 정규화 항이 소거됩니다.

$$
y^B_R - y^B_i + B = z_R - z_i
$$

한 질의에 4개 토큰을 밀어 올려 $$R$$과 비교하므로 **logit 하나에 질의 1/4**입니다. softmax는 덧셈 이동에 불변이므로 $$z_R = 0$$으로 두면 됩니다. 각 logprob을 원래 logit에 대한 선형 제약으로 보고 5개를 한꺼번에 푸는 변형(1/5)도 있지만, 조건이 나쁜 행렬을 다뤄야 해서 실제로는 질의가 더 많이 들고 정밀도도 낮았습니다.

토큰 비용은 질의당 최소 2토큰(입력 1, 출력 1)인데, gpt-3.5-turbo처럼 질의마다 오버헤드 토큰 $$\Delta$$(약 7개)가 붙으면 logit당 $$(2 + \Delta)/4$$가 됩니다. 이를 줄이는 개선안으로, 토큰 $$x$$에 $$B$$를, 다른 네 토큰에 $$B' < B$$를 주어 모델이 $$[x\ x\ \ldots\ x]$$를 계속 뱉게 만들면 생성된 토큰마다 logprob을 얻어 질의당 $$m$$개 prefix를 처리할 수 있습니다.

### top-1에 bias가 {-1, 0}뿐이어도

K = 1이고 bias가 $$\{-1, 0\}$$ 두 값만 허용되는 훨씬 빡빡한 API도 뚫립니다. bias 없이 한 번, 토큰 $$t$$에 $$-1$$을 주고 한 번 질의하면 top 토큰의 logprob이 **아주 조금** 올라가는데, 그 정도가 $$t$$의 logprob을 결정합니다.

$$
\text{logprob}_t = (1/e - 1)^{-1}\big(\exp(y_{\text{top}} - y'_{\text{top}}) - 1\big)
$$

logprob 하나에 질의 1회지만 수치적으로 훨씬 불안정합니다.

## 4. logprob이 전혀 없는 API

공개 후 OpenAI는 logit bias가 top logprob에 영향을 주지 못하게 바꿨습니다. 그래서 logit bias만 있고 logprob이 없는 API, 즉 온도 0에서 **argmax 토큰만** 돌려주는 API에 대한 공격도 만듭니다.

- **이진 탐색**: 토큰 $$i$$가 top-1이 되는 최소 bias를 이진 탐색으로 찾으면 $$z_i - z_0$$을 $$\varepsilon$$ 정밀도로 $$\log(B/\varepsilon)$$번의 질의에 얻습니다.
- **Hyperrectangle**: 여러 토큰에 동시에 bias를 걸고, 토큰 $$k$$가 나왔다는 사실에서 모든 $$j$$에 대해 $$z_k + b_k \ge z_j + b_j$$라는 선형 제약을 모읍니다. 매 라운드 각 $$z_i - z_0$$의 최소·최대를 선형 계획으로 좁히는데, 이 LP는 가중 그래프의 **all-pairs shortest path**로 바꿀 수 있어 Bellman-Ford로 $$O(N^3)$$에 풉니다.
- **균형 잡힌 질의**: 구간 중앙으로 bias를 주면 구간이 넓은 토큰만 자꾸 뽑혀 좁은 토큰에 대한 제약이 모이지 않습니다. 균일 사전분포 아래 각 토큰이 $$1/(N+1)$$ 확률로 나오도록 $$b_i = -(1 - c)\alpha_i - c\beta_i$$, $$c = \exp(-\log(N+1)/N)$$로 bias를 걸면 크게 좋아집니다.

| 공격 | logprob | 정밀도(bit) | logit당 질의 |
|---|---|---|---|
| logprob-4 (3절) | top-5 | 23.0 | 0.25 |
| logprob-5 (부록) | top-5 | 11.5 | 0.64 |
| logprob-1 (top-1, bias ∈ {-1, 0}) | top-1 | 6.1 | 1.0 |
| 이진 탐색 | 없음 | 7.2 | 10.0 |
| hyperrectangle | 없음 | 15.7 | 5.4 |
| one-of-n (균형 질의) | 없음 | 18.0 | 3.7 |

logprob 없이도 logit당 3.7회 질의로 18비트 정밀도를 얻고, 부록의 분석으로는 최적의 2배 이내입니다.

## 5. 코드로 보기: 특잇값의 낙차, W의 복원, top-5로부터의 logit 복원

어휘 2,000개, 숨은 차원 128인 장난감 모델로 세 공격을 재현합니다. 모델 본체 $$g_\theta$$는 프롬프트별 난수 벡터로 흉내 냈습니다.

```python
import torch

torch.manual_seed(0)
L, H = 2000, 128                                           # 어휘 크기 l, 숨은 차원 h (h ≪ l)
W = torch.randn(L, H) / H ** 0.5                           # 마지막 층(embedding projection) W: l × h

def g(prompt_id):                                          # 모델 본체 g_θ(p): 프롬프트 → 숨은 벡터 (여기서는 난수로 흉내)
    return torch.randn(H, generator=torch.Generator().manual_seed(prompt_id))

def api_logits(prompt_id):                                 # 4절의 "모든 logit" API
    return W @ g(prompt_id)

def api_top5(prompt_id, bias):                             # 5절의 실제에 가까운 API: logit bias를 더한 뒤 top-5 logprob만 돌려준다
    z = api_logits(prompt_id) + bias
    lp = torch.log_softmax(z, -1)
    top = torch.topk(lp, 5)
    return dict(zip(top.indices.tolist(), top.values.tolist()))

# 1) 숨은 차원 알아내기: 질의 n개의 logit 벡터를 쌓아 특잇값의 가장 큰 낙차를 찾는다
for n in [64, 128, 192, 256]:
    Q = torch.stack([api_logits(i) for i in range(n)]).double()
    s = torch.linalg.svdvals(Q)
    gap = torch.log(s[:-1]) - torch.log(s[1:])
    print(f"질의 {n:3d}개: 특잇값 낙차가 가장 큰 자리 = {gap.argmax().item() + 1:3d}, 낙차 = {gap.max().item():5.1f}")

# 2) 마지막 층 복원: Q = U Σ Vᵀ에서 U Σ가 W·G (G는 h × h 가역 행렬)와 같다
Q = torch.stack([api_logits(i) for i in range(300)]).double().T     # l × n
U, S, _ = torch.linalg.svd(Q, full_matrices=False)
W_tilde = U[:, :H] * S[:H]
G = torch.linalg.lstsq(W_tilde, W.double()).solution              # 정렬용 h × h 변환(평가에만 사용)
print(f"복원한 W̃·G와 실제 W의 RMS 오차: {(W_tilde @ G - W.double()).pow(2).mean().sqrt():.2e} (무작위 행렬이면 {(torch.randn(L, H) / H ** 0.5 - W).pow(2).mean().sqrt():.2e})")

# 3) top-5 logprob + logit bias만으로 logit 벡터 전체 복원: 기준 토큰 R과 비교하며 4개씩 알아낸다
pid, B = 0, 60.0
R = max(api_top5(pid, torch.zeros(L)), key=lambda k: api_top5(pid, torch.zeros(L))[k])   # 기준: 원래 top-1 토큰
rec, queries = torch.zeros(L), 0
for start in range(0, L, 4):
    idx = [i for i in range(start, min(start + 4, L)) if i != R]
    bias = torch.zeros(L); bias[idx] = B                   # 4개 토큰을 top-5로 밀어 올린다
    out = api_top5(pid, bias); queries += 1
    for i in idx:
        rec[i] = out[i] - B - out[R]                       # y_i - B - y_R = z_i - z_R
true = api_logits(pid); true = true - true[R]
print(f"logit 벡터 복원: 질의 {queries}회, 최대 오차 {(rec - true).abs().max():.1e}")
```

```text
질의  64개: 특잇값 낙차가 가장 큰 자리 =  63, 낙차 =   0.1
질의 128개: 특잇값 낙차가 가장 큰 자리 = 127, 낙차 =   0.8
질의 192개: 특잇값 낙차가 가장 큰 자리 = 128, 낙차 =  14.3
질의 256개: 특잇값 낙차가 가장 큰 자리 = 128, 낙차 =  14.9
복원한 W̃·G와 실제 W의 RMS 오차: 8.52e-09 (무작위 행렬이면 1.25e-01)
logit 벡터 복원: 질의 500회, 최대 오차 5.7e-06
```

- 질의가 $$h = 128$$개 이하일 때는 $$Q$$가 full rank라 낙차가 없다가, 192개가 되자 128번째 자리에서 낙차가 14(약 $$e^{14}$$배)로 튑니다. 논문 Figure 1과 2에서 Pythia-1.4B의 2048번째 특잇값에 스파이크가 생기는 것과 같은 현상입니다.
- $$U \Sigma$$를 affine 정렬하면 $$W$$와의 오차가 $$10^{-8}$$ 수준입니다. 무작위 행렬은 $$10^{-1}$$입니다.
- top-5 logprob만 주는 API에서 기준 토큰 트릭으로 2,000개 logit을 500회 질의에 $$10^{-6}$$ 오차로 복원했습니다. 논문의 "logit당 1/4 질의"입니다.

## 6. 프로덕션 모델 공격 결과

2024년 1월 1일 기준 OpenAI 모델 5개에 사전 허가를 받고 가장 효율적이고 정밀한 logprob-4 공격을 돌렸습니다.

| 모델 | 크기 | 차원 복원 질의 | 비용 | 가중치 복원 질의 | 비용 |
|---|---|---|---|---|---|
| ada | 1024 ✓ | < $$2 \cdot 10^6$$ | $1 | < $$2 \cdot 10^7$$ | $4 |
| babbage | 2048 ✓ | < $$4 \cdot 10^6$$ | $2 | < $$4 \cdot 10^7$$ | $12 |
| babbage-002 | 1536 ✓ | < $$4 \cdot 10^6$$ | $2 | (미실행) | $12 |
| gpt-3.5-turbo-instruct | 비공개 ✓ | < $$4 \cdot 10^7$$ | $200 | (미실행, 추정) | $2,000 |
| gpt-3.5-turbo-1106 | 비공개 ✓ | < $$4 \cdot 10^7$$ | $800 | (미실행, 추정) | $8,000 |

- 복원한 크기는 모두 OpenAI가 확인한 실제 크기와 **정확히 일치**했습니다. 앞의 세 모델은 이미 폐기되어 공개했고, gpt-3.5-turbo의 크기는 OpenAI의 요청으로 비공개입니다.
- 가중치 복원 오차는 $$h \times h$$ 변환을 제외하고 $$7 \cdot 10^{-4}$$ 미만입니다. gpt-3.5는 가중치 보호를 위해 전체 복원은 하지 않고 비용만 추정했습니다.
- 모든 데이터는 OpenAI와 효과를 확인한 뒤 삭제했습니다.

## 7. 방어

**예방**

- **logit bias 제거**: 가장 단순하지만 제약 생성, proxy tuning 같은 정당한 용도가 있습니다.
- **bias 대신 block-list**: 금지 토큰 목록은 일부 용도를 지원하면서 공격은 막습니다.
- **구조 변경**: 공격은 $$h < l$$이라서 됩니다. 마지막 층을 $$h \to t \to l$$($$t > l$$)로 나누고 사이에 비선형을 두면 막히지만, 어휘 크기의 제곱이라 비효율적입니다.
- **사후 구조 변경**: $$W$$에 원래 행렬과 직교하는 가짜 열을 붙이고 forward 때 숨은 벡터에 가우시안 노이즈를 이어 붙이면, 특잇값을 적당히 맞춰 공격자가 모델이 실제보다 넓다고 믿게 만들 수 있습니다. GPT-2 small을 1024차원처럼 보이게 한 예를 보입니다.

**완화**

- **logit bias XOR logprob**: 둘을 함께 쓰면 공격이 10배 싸집니다. 한 질의에 둘 중 하나만 허용하면 됩니다. OpenAI가 2023년에 echo와 logprob의 조합을 막은 것과 같은 유형입니다.
- **노이즈 추가**: 출력 logit에 충분한 노이즈를 더하면 막히지만 모델이 덜 유용해집니다.
- **logit bias 호출 제한**: prefix당 $$\tilde h / 5$$번만 허용하면 $$\tilde h$$보다 작은 모델은 보호되지만, 임계값이 $$h$$와 독립이어야 하고 모든 사용자의 질의 상태를 유지해야 하며 Sybil 공격을 막으려면 전역 풀이 필요해 개인정보 문제가 생깁니다.
- **악의적 질의 탐지**: 기존 anti-abuse 도구처럼 패턴을 탐지합니다.

## 8. 해석

**1. "한 층만"이지만 "프로덕션 모델에서 처음"입니다.** 논문 스스로 이 층을 훔쳐서 당장 할 수 있는 일은 많지 않다고 말합니다. 중요한 것은 배포된 Transformer의 **정확한 정보가 처음으로 훔쳐졌다**는 존재 증명이고, 숨은 차원은 전체 파라미터 수와 상관이 있어 모델 간 상대적 크기를 드러냅니다. 기존 공격이 100뉴런 ReLU 망에서 전체를 훔치는 "all-or-nothing"이었다면, 이 논문은 **부분 정보를 훔치는 것이 훨씬 실용적**임을 보였습니다.

**2. 공격의 뿌리는 선형대수 한 줄입니다.** $$h \ll l$$이면 logit은 $$h$$차원 부분공간에 갇히고, 그 부분공간은 SVD로 읽힙니다. 위 예제에서 192번의 질의로 128이라는 숫자가 튀어나온 것이 전부입니다. 나머지는 제한된 API에서 logit 벡터를 어떻게든 복원하는 공학이고, 그 공학은 "softmax는 덧셈 이동에 불변"이라는 성질을 반복해서 씁니다.

**3. 사소해 보이는 API 설계가 보안을 결정합니다.** logit bias와 logprob은 각각 정당한 기능이지만 둘을 함께 주면 모델의 일부가 새어 나갑니다. OpenAI와 Google이 공개 후 API를 바꾼 것은 그 교훈을 받아들인 결과입니다. 비슷한 긴장이 [워터마크]({% post_url 2026-10-23-watermark-reliability %})에도 있습니다. 모델 소유자가 출력을 통제할 수 있다는 점이 검출을 가능하게 하지만, 그 통제의 세부가 노출되면 공격의 발판이 됩니다.

**4. 열려 있는 확장.** 양자화된 가중치는 무한한 $$W G$$ 중 하나만 정수 격자에 맞으므로 대칭성을 깨고 비트 단위 복원이 가능할 수 있지만, 그 정수 제약 문제는 일반적으로 NP-hard입니다. 한 층 너머로 확장하는 방법은 비선형성 때문에 보이지 않고, logit bias 없이 temperature나 top-k 파라미터로 logit을 새게 하는 길도 남아 있습니다.

**5. 평가를 읽을 때 주의할 점.** 완벽한 oracle 가정 아래의 결과이고, 복원은 affine 변환까지입니다. 프로덕션 실험은 OpenAI의 허가를 받은 5개 모델에 한정되고, gpt-3.5의 가중치 복원 비용은 실측이 아니라 추정입니다. 방어 실험도 예비 수준입니다.

## 9. 정리

| 항목 | 내용 |
|---|---|
| 목표 | 블랙박스 API 질의만으로 프로덕션 LLM의 마지막 층과 숨은 차원 복원 |
| 원리 | $$h \ll l$$이라 logit이 $$h$$차원 부분공간에 있음. 질의 $$n > h$$개의 logit을 쌓아 SVD → 특잇값 낙차로 $$h$$, $$U\Sigma = WG$$로 층 복원 |
| API 우회 | top-5 logprob + logit bias: 기준 토큰과 비교해 질의당 4 logit. logprob 없음: argmax와 bias로 제약을 모아 LP(최단 경로) |
| 결과 | ada 1024, babbage 2048, babbage-002 1536 확인(20달러 미만). gpt-3.5-turbo 차원 확인(200~800달러), RMS $$< 7 \cdot 10^{-4}$$ |
| 한계 | affine 변환까지만 복원, 한 층만, 완벽한 oracle 가정 |
| 방어 | logit bias와 logprob 동시 사용 금지, block-list, 가짜 차원 추가, 노이즈 |

## References

- Carlini et al., "Stealing Part of a Production Language Model," ICML 2024. [arXiv:2403.06634](https://arxiv.org/abs/2403.06634)
- Tramèr et al., "Stealing Machine Learning Models via Prediction APIs," USENIX Security 2016. [arXiv:1609.02943](https://arxiv.org/abs/1609.02943)
- Jagielski et al., "High Accuracy and High Fidelity Extraction of Neural Networks," USENIX Security 2020. [arXiv:1909.01838](https://arxiv.org/abs/1909.01838)
- Carlini, Jagielski, Mironov, "Cryptanalytic Extraction of Neural Network Models," CRYPTO 2020. [arXiv:2003.04884](https://arxiv.org/abs/2003.04884)
- Morris et al., "Language Model Inversion," 2023. [arXiv:2311.13647](https://arxiv.org/abs/2311.13647)
