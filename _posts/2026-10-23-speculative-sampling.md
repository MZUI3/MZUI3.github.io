---
title: "[LLM] Speculative Sampling: Chinchilla 70B를 분산 환경에서 2~2.5배 빠르게"
date: 2026-10-23 09:40:00 +0900
categories: [Transformer, LLM]
tags: [llm, speculative-sampling, speculative-decoding, inference-acceleration, chinchilla, draft-model, memory-bandwidth, distributed-serving]
description: DeepMind가 Google과 독립적으로 제안한 Speculative Sampling(Chen et al., 2023)의 수정 rejection sampling, 분산 서빙에서 draft 모델을 고르는 기준, 그리고 draft 길이 K의 trade-off를 Chinchilla 70B 실험과 간단한 시뮬레이션으로 정리합니다.
math: true
---

> [Speculative Decoding]({% post_url 2026-10-23-speculative-decoding %})과 거의 같은 시기에 DeepMind가 독립적으로 같은 아이디어에 도달했습니다. **Speculative Sampling**(Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper, DeepMind, 2023)은 핵심 원리는 같지만 **수백억 파라미터 모델의 분산 서빙**에 초점을 맞춰, Chinchilla 70B에서 2~2.5배의 디코딩 가속을 보였습니다. 이 글은 겹치는 부분은 짧게 넘기고 이 논문만의 관점에 집중합니다.

## 1. 문제: 메모리 대역폭이 정한 속도 상한

Transformer 샘플링은 보통 **메모리 대역폭에 묶여** 있습니다. 토큰 하나를 만들 때마다 모든 파라미터가 적어도 하나의 가속기 칩을 거쳐야 하므로, 모델 크기를 전체 메모리 대역폭으로 나눈 값이 autoregressive 샘플링(ArS) 속도의 **엄격한 상한**이 됩니다. 모델이 커지면 여러 가속기에 나눠 올려야 해서 장치 간 통신 지연까지 더해집니다.

이 논문은 두 가지 관찰로 이 상한을 넘어섭니다.

1. 짧은 연속 토큰 $$K$$개의 logit을 **병렬로 채점**하는 시간은 토큰 하나를 샘플링하는 시간과 거의 같습니다.
2. 더 작고 빠른 **draft 모델**이 추측한 토큰을 큰 **target 모델**이 채점하고, 수정된 rejection sampling으로 받아들이면 target의 분포가 그대로 보존됩니다.

Figure 1의 결과를 미리 말하면, speculative sampling(SpS)의 초당 토큰 수는 메모리 대역폭이 정한 이상적 상한을 **넘어섭니다**. 한 번의 메모리 읽기로 토큰을 여러 개 얻기 때문입니다.

## 2. 알고리즘

Algorithm 2는 다음을 반복합니다. 이 논문의 표기는 target이 $$q$$, draft가 $$p$$로 Google 논문과 반대입니다.

1. draft 모델 $$p$$를 $$K$$번 돌려 $$\tilde x_1, \ldots, \tilde x_K$$를 뽑습니다.
2. target 모델 $$q$$를 한 번 돌려 $$K + 1$$개의 logit 집합 $$q(\cdot \mid x_{1:n}), q(\cdot \mid x_{1:n}, \tilde x_1), \ldots, q(\cdot \mid x_{1:n}, \tilde x_{1:K})$$를 병렬로 계산합니다.
3. $$t = 1, \ldots, K$$에 대해 $$r \sim U[0, 1]$$을 뽑아 $$r < \min\big(1, q(\tilde x_t)/p(\tilde x_t)\big)$$이면 받아들이고, 아니면 $$(q - p)_+$$에서 다시 뽑고 루프를 빠져나옵니다.
4. 전부 받아들여졌으면 마지막 logit에서 토큰 하나를 더 뽑습니다.

$$(f(x))_+ = \max(0, f(x)) / \sum_x \max(0, f(x))$$입니다. 루프 한 번에 최소 1개, 최대 $$K + 1$$개의 토큰이 나옵니다. 마지막 draft 토큰에 대한 logit이 다음 토큰의 logit이기도 해서, 전부 수락되면 $$K$$개가 아니라 $$K + 1$$개를 얻는다는 점을 논문이 따로 짚습니다.

### 왜 채점이 샘플링만큼 싼가

Megatron 방식으로 샤딩된 큰 Transformer에서 샘플링 시간은 세 요소가 차지합니다.

| 요소 | 샘플링(토큰 1개) | 채점($$K$$개 병렬) |
|---|---|---|
| Linear 층 | batch가 작아 행렬곱이 메모리에 묶임 | 작은 $$K$$에서는 여전히 메모리에 묶여 시간이 비슷 |
| Attention | KV cache 읽기가 대역폭 대부분을 차지 | KV cache 크기는 $$K$$와 무관하므로 차이가 거의 없음 |
| All-reduce | 토큰 몇 개의 activation만 전송해 처리량이 아니라 지연에 묶임 | 마찬가지로 지연에 묶여 시간이 비슷 |

positional encoding, nucleus sampling의 정렬 같은 구현 세부 사항에 따라 약간의 차이는 생길 수 있지만, 위 세 요소가 지배적이면 작은 $$K$$에서 채점은 샘플링보다 크게 느리지 않습니다.

### 수정 rejection sampling의 정당성

Theorem 1의 증명은 Google 논문과 같은 구조입니다. 최종 샘플 $$X$$가 $$x$$일 확률은 draft가 $$x$$를 뽑고 수락되는 경우 $$\min(p(x), q(x))$$와, 거절된 뒤 $$(q - p)_+$$에서 $$x$$가 뽑히는 경우의 합입니다. 거절 확률이 $$\sum_{x'} \max(0, q(x') - p(x'))$$로 $$(q - p)_+$$의 분모와 정확히 같아서, 두 번째 항은 $$\max(0, q(x) - p(x))$$가 되고 합은 $$q(x)$$입니다. 하드웨어 수치 오차 범위 안에서 target 분포가 복원됩니다.

nucleus, top-k, temperature는 rejection sampling 전에 확률을 조정해 두면 되고, 전체 수락률은 그 파라미터에 크게 민감하지 않았다고 합니다. Transformer 본체를 건드리지 않으므로 quantization이나 multi-query attention 같은 다른 최적화와 함께 쓸 수 있습니다.

## 3. 분산 서빙에서 draft 모델 고르기

수락 기준이 분포를 보장하므로 draft는 logit만 내놓으면 무엇이든 됩니다. 논문이 꼽은 선택지는 다음과 같습니다.

- target 모델에 여러 head를 붙여 처음부터 함께 학습(Stern et al., 2018)
- sequence-level distillation으로 $$K$$개 토큰을 병렬 예측하는 모델을 따로 학습(Ge et al., 2022)
- target의 activation 일부를 draft 입력으로 쓰도록 학습
- **그냥 작은 버전의 target 모델**

앞의 세 가지는 강력한 draft를 만들겠지만 target 모델의 출력 데이터가 대량으로 필요하거나 target을 바꿔야 해서 큰 규모에서는 실용적이지 않습니다. 작은 버전의 언어 모델은 이미 학습 도구가 갖춰져 있어 엔지니어링 면에서도 편합니다.

### 깊이보다 너비: 통신 비용을 줄이는 설계

분산 환경에서는 **작은 모델을 아무거나 고르면 안 됩니다**. 모델마다 최적의 추론 배치가 다르기 때문입니다. Chinchilla 70B는 TPU v4 16개에서 14.1ms/token이지만, Chinchilla-optimal 7B는 TPU v4 **4개**에서 5ms/token으로 가장 빠르고, 16개에 올리면 오히려 느려집니다. 작은 모델은 메모리 대역폭과 FLOPs의 이득이 장치 간 통신 비용을 상쇄하지 못하기 때문입니다. 7B를 최적 배치로 쓰면 가속이 작고, 하드웨어도 놀게 됩니다.

해결책은 **층은 적고 너비는 넓은 draft**입니다. 언어 모델의 성능은 깊이와 너비의 비율에 비교적 둔감하다는 관찰(Levine et al., 2020)을 이용해, Chinchilla와 같은 tokenizer와 데이터로 **8층짜리 4B 모델**을 학습했습니다. 통신은 층마다 일어나므로 층이 적으면 통신 비용이 줄고, target과 같은 16개 TPU에서 **1.8ms/token**으로 돌아갑니다. target의 1/8 시간입니다.

| 모델 | $$d_{model}$$ | Heads | 층 | 파라미터 |
|---|---|---|---|---|
| Target (Chinchilla) | 8192 | 64 | 80 | 70B |
| Draft | 6144 | 48 | 8 | 4B |

## 4. 코드로 보기: K를 늘리면 어디까지 빨라지나

논문 Figure 1의 trade-off를 간단한 시뮬레이션으로 재현해 보겠습니다. draft와 target의 토큰당 시간은 논문 수치(1.8ms, 14.1ms)를 쓰고, 토큰 하나가 수락될 확률을 0.8로 두었습니다.

```python
import torch

torch.manual_seed(0)
T_TARGET, T_DRAFT = 14.1, 1.8          # Chinchilla 70B와 4B draft의 토큰당 시간(ms), 논문 수치
alpha = 0.8                            # 토큰 하나가 수락될 확률(단순화: 위치마다 독립)

def simulate(K, n_tokens=128, trials=2000):
    """draft K개를 뽑고 target이 검증하는 루프를 n_tokens를 채울 때까지 반복해 걸린 시간을 잰다."""
    times = []
    for _ in range(trials):
        made, t = 0, 0.0
        while made < n_tokens:
            accepted = 0
            while accepted < K and torch.rand(1).item() < alpha:
                accepted += 1
            made += accepted + 1                           # 거절돼도 1개, 전부 수락되면 K+1개
            t += K * T_DRAFT + T_TARGET                    # draft K번 + target 1번(K+1개 병렬 채점)
        times.append(t)
    times = torch.tensor(times)
    return times.mean().item(), times.std().item()

base = 128 * T_TARGET
print(f"자기회귀 샘플링: 128토큰에 {base:.0f}ms")
for K in range(1, 9):
    m, s = simulate(K)
    print(f"K = {K}: 평균 {m:6.0f}ms (±{s:4.0f}) | 속도 향상 {base / m:.2f}배 | 루프당 {K * T_DRAFT + T_TARGET:5.1f}ms")
```

```text
자기회귀 샘플링: 128토큰에 1805ms
K = 1: 평균   1134ms (±  30) | 속도 향상 1.59배 | 루프당  15.9ms
K = 2: 평균    937ms (±  43) | 속도 향상 1.93배 | 루프당  17.7ms
K = 3: 평균    854ms (±  53) | 속도 향상 2.11배 | 루프당  19.5ms
K = 4: 평균    822ms (±  63) | 속도 향상 2.20배 | 루프당  21.3ms
K = 5: 평균    812ms (±  73) | 속도 향상 2.22배 | 루프당  23.1ms
K = 6: 평균    816ms (±  82) | 속도 향상 2.21배 | 루프당  24.9ms
K = 7: 평균    835ms (±  92) | 속도 향상 2.16배 | 루프당  26.7ms
K = 8: 평균    859ms (± 104) | 속도 향상 2.10배 | 루프당  28.5ms
```

세 가지가 논문의 Figure 1과 같습니다.

- **루프당 시간은 $$K$$에 선형**으로 늘어납니다. draft 호출이 $$K$$번이기 때문입니다.
- **속도 향상은 $$K = 5$$ 근처에서 정점을 찍고 다시 내려갑니다**. 뒤쪽 draft 토큰은 앞 토큰이 모두 수락되어야만 쓸모가 있어서, $$K$$를 키울수록 수락되는 비율(수락 토큰 수 ÷ $$K + 1$$)이 떨어집니다. XSum에서는 $$K = 3$$이 최적이었습니다.
- **분산이 $$K$$와 함께 커집니다**. 평균이 조금 더 좋아지더라도 P90, P99 지연이 중요한 서비스에서는 큰 $$K$$가 손해일 수 있습니다.

## 5. 실험: Chinchilla 70B

batch size 1, $$K = 4$$로 두 벤치마크를 돌렸습니다. XSum은 1-shot 요약(nucleus $$p = 0.8$$, 11,305개 시퀀스, 최대 128토큰), HumanEval은 100-shot 코드 생성($$p = 0.95$$, 온도 0.8, 16,400개 샘플, 최대 512토큰)입니다.

| 샘플링 | 벤치마크 | 점수 | 토큰당 시간 | 속도 |
|---|---|---|---|---|
| ArS (nucleus) | XSum (ROUGE-2) | 0.112 | 14.1ms | 1배 |
| SpS (nucleus) | XSum (ROUGE-2) | 0.114 | 7.52ms | 1.92배 |
| ArS (greedy) | XSum (ROUGE-2) | 0.157 | 14.1ms | 1배 |
| SpS (greedy) | XSum (ROUGE-2) | 0.156 | 7.00ms | 2.01배 |
| ArS (nucleus) | HumanEval (100-shot) | 45.1% | 14.1ms | 1배 |
| SpS (nucleus) | HumanEval (100-shot) | 47.0% | 5.73ms | 2.46배 |

- greedy에서도 수치 오차로 토큰 하나가 달라지면 두 시퀀스가 크게 갈라질 수 있고, 난수 시드 처리와 계산 그래프가 다르므로 **출력이 글자 그대로 같을 수는 없습니다**. 대신 같은 분포에서 나왔는지를 벤치마크 점수로 확인했고, 점수가 같은 수준입니다.
- HumanEval과 greedy XSum의 가속은 하드웨어의 메모리 대역폭 상한(모델 크기 ÷ 전체 대역폭)을 **넘어섰습니다**.
- 수락률은 도메인과 디코딩 방식에 따라 다릅니다. HumanEval이 특히 빠른 이유로 코드에 `for i in range(len(arr)):` 같은 흔한 부분 시퀀스가 많고, 짧은 토큰으로 쪼개지며, 온도가 두 모델의 logit을 모두 날카롭게 만든다는 점을 꼽습니다.

## 6. 해석

**1. 같은 아이디어, 다른 질문.** [Google 논문]({% post_url 2026-10-23-speculative-decoding %})이 "왜 분포가 보존되고 이론적으로 얼마나 빨라지나"에 답했다면, 이 논문은 "**실제 분산 서빙에서 어떻게 돌리나**"에 답합니다. 채점이 샘플링만큼 싼 이유를 Megatron 샤딩의 세 요소로 분해하고, draft 모델을 target과 같은 하드웨어에 올리기 위해 깊이 대신 너비를 택한 설계가 이 논문의 기여입니다.

**2. draft 모델은 "작은 모델"이 아니라 "빠른 모델"이어야 합니다.** 7B 모델을 16개 TPU에 올리면 4개에 올릴 때보다 느리다는 관찰은 중요합니다. 작은 모델의 지연은 파라미터 수가 아니라 통신 횟수, 즉 층 수가 정합니다. 그래서 4B 파라미터인데 8층인 모델이 7B보다 좋은 draft입니다. 단일 가속기 실험에 머문 Google 논문에서는 보이지 않던 설계 변수입니다.

**3. 가속의 상한을 "메모리 대역폭"으로 명시했습니다.** autoregressive 샘플링은 토큰마다 가중치를 전부 읽어야 해서 모델 크기 ÷ 대역폭보다 빨라질 수 없습니다. SpS가 이 상한을 넘었다는 것은 한 번의 가중치 읽기로 평균 2개 이상의 토큰을 얻었다는 뜻이고, 이것이 이 방법이 quantization이나 multi-query attention과 **직교하는** 이유입니다. 그 방법들은 상한 자체를 올리고, SpS는 상한 위로 올라갑니다.

**4. K는 작게.** 시뮬레이션에서 본 것처럼 $$K$$를 키우는 이득은 빠르게 포화하고 분산은 계속 커집니다. 논문도 $$K = 3{\sim}4$$를 썼습니다. 뒤쪽 토큰이 수락될 확률이 $$\alpha^K$$로 줄어드는 구조상, 수락률을 올리지 않고 $$K$$만 키우는 것은 의미가 없습니다. 수락률을 올리는 방향은 [Online Speculative Decoding]({% post_url 2026-10-23-online-speculative-decoding %})에서 이어집니다.

**5. 평가를 읽을 때 주의할 점.** 모든 실험이 batch size 1입니다. batch가 커지면 linear 층이 compute-bound로 바뀌어 "채점이 샘플링만큼 싸다"는 전제가 약해집니다. 4B draft 모델은 Chinchilla와 같은 데이터로 새로 학습한 것이라, 기존 체크포인트만 쓴 Google 논문보다 준비 비용이 큽니다. 수락률 수치는 그래프로만 제시되어 정확한 값을 알 수 없습니다.

## 7. 정리

| 항목 | 내용 |
|---|---|
| 목표 | 수백억 파라미터 모델의 분산 서빙에서 디코딩 지연 단축 |
| 방법 | draft가 $$K$$개 추측 → target이 $$K + 1$$개 logit을 병렬 채점 → 수정 rejection sampling |
| 수락 규칙 | $$r < \min(1, q(\tilde x)/p(\tilde x))$$이면 수락, 아니면 $$(q - p)_+$$에서 재샘플링 |
| draft 설계 | Chinchilla와 같은 데이터로 학습한 8층 4B 모델. 층을 줄여 통신 비용을 낮춰 같은 16 TPU에서 1.8ms/token |
| 결과 | Chinchilla 70B에서 XSum 1.92~2.01배, HumanEval 2.46배. 벤치마크 점수 동일 |
| 주의 | $$K$$가 크면 가속이 포화하고 분산이 커짐. 수치 오차로 글자 단위 출력은 다를 수 있음 |

## References

- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling," DeepMind, 2023. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding," ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Hoffmann et al., "Training Compute-Optimal Large Language Models," 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Pope et al., "Efficiently Scaling Transformer Inference," 2022. [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)
- Levine et al., "The Depth-to-Width Interplay in Self-Attention," 2020. [arXiv:2006.12467](https://arxiv.org/abs/2006.12467)
