---
title: "[LLM] LLM 워터마크는 바꿔 써도 남는가: 사람·GPT의 paraphrase와 Copy-Paste 공격에 대한 신뢰성"
date: 2026-10-23 11:00:00 +0900
categories: [Transformer, LLM]
tags: [llm, watermark, ai-text-detection, green-list, z-score, paraphrase-attack, winmax, detectgpt, retrieval]
description: 초록·빨강 목록 워터마크(Kirchenbauer et al., 2023)가 사람과 GPT-3.5의 바꿔 쓰기, 긴 문서에 일부만 끼워 넣는 Copy-Paste 공격 뒤에도 검출되는지 살핀 On the Reliability of Watermarks for LLMs(Kirchenbauer et al., 2023)의 해싱 설계, WinMax 검출, 사람 실험, 다른 검출기와의 비교를 정리하고 작은 워터마크 구현으로 핵심 성질을 확인합니다.
math: true
---

> LLM이 만든 글이 스팸, 가짜 리뷰, 다음 세대 학습 데이터의 오염으로 이어질 수 있어, 기계가 쓴 글을 가려내는 일이 중요해졌습니다. **워터마크**는 생성 때 눈에 띄지 않는 통계적 패턴을 심어 두고 나중에 확인하는 방법입니다. 이상적인 조건에서는 잘 되지만, 실제로는 사람이 고쳐 쓰고, 다른 모델로 바꿔 쓰고, 긴 문서에 일부만 끼워 넣습니다. **On the Reliability of Watermarks for Large Language Models**(Kirchenbauer, Geiping et al., University of Maryland, 2023)는 그런 "야생"에서 워터마크가 얼마나 살아남는지를 체계적으로 쟀습니다. 결론은 **토큰이 충분히 많으면 사람의 바꿔 쓰기도 결국 검출된다**는 것입니다.

## 1. 기계 생성 텍스트 검출의 세 갈래

| 갈래 | 방식 | 장단점 |
|---|---|---|
| **사후(post-hoc) 분류기** | 사람 글과 기계 글로 이진 분류기를 학습 | 모델 정보가 필요 없지만 도메인 밖에서 약하고 adversarial 공격에 취약 |
| **사후 통계 검출** | perplexity, 엔트로피, DetectGPT의 곡률 기준 | 학습 데이터가 필요 없지만 모델 통계에 접근해야 하고 비용이 큼 |
| **사전(proactive) 검출** | 생성 때 행동을 취함: 모든 출력을 DB에 저장(retrieval) 또는 워터마크 | 모델 소유자의 협조가 필요하지만 훨씬 견고 |

retrieval은 강한 paraphrase에도 견디지만 DB 크기에 따라 비용과 false positive가 늘고, 모든 사용자 출력을 저장하는 것은 개인정보 면에서 문제입니다. 워터마크는 출력 분포를 아주 조금 바꾸는 것으로 수학적 기반 위에 놓을 수 있습니다.

## 2. 초록·빨강 목록 워터마크

이 논문이 다루는 워터마크(Kirchenbauer et al., 2023)는 생성 step마다 어휘를 **초록**과 **빨강**으로 의사 무작위로 나눕니다. 직전 $$h$$개 토큰을 pseudo-random function $$f: \mathbb{N}^h \to \mathbb{N}$$에 넣어 시드를 만들고, 그 시드로 어휘의 $$\gamma |V|$$개를 초록 목록 $$G_t$$로 뽑은 뒤 logit을 조정합니다.

$$
l_{tk} = \begin{cases} l_{tk} + \delta & k \in G_t \\ l_{tk} & \text{그 밖} \end{cases}
$$

이 logit으로 어떤 샘플링을 해도 초록 토큰이 조금 더 자주 나옵니다. 검출은 LLM 없이 가능합니다. 해시 함수만 알면 각 위치의 초록 목록을 다시 만들어 초록에 든 토큰 수 $$|s|$$를 세고 z-score를 구합니다.

$$
z = \frac{|s| - \gamma T}{\sqrt{\gamma(1 - \gamma) T}}
$$

사람이 쓴 글은 초록 비율이 $$\gamma$$ 근처라 $$z \approx 0$$이고, 워터마크 글은 $$z$$가 큽니다. 워터마크는 해시 $$f$$, 문맥 폭 $$h$$, 초록 비율 $$\gamma$$, logit 가산 $$\delta$$ 네 파라미터로 정해집니다.

## 3. 신뢰성을 높이는 설계

### 해싱: 비밀 유지와 견고함의 교환

원 논문은 $$h = 1$$, 즉 직전 토큰 하나로 초록 목록을 정했습니다(**LeftHash**). 그러면 제삼자가 특정 토큰 뒤에 유난히 자주 오는 단어를 찾아 초록 목록을 역추적할 수 있습니다. 비밀을 지키려면 $$h$$를 키워야 하는데, 그러면 바꿔 쓰기로 문맥 하나만 바뀌어도 해시가 깨집니다. 논문은 세 가지 해시 함수를 비교합니다($$s$$는 비밀 salt, $$P$$는 정수 PRF).

| 해시 | 정의 | 성질 |
|---|---|---|
| **Additive** | $$P(s \sum_{i=1}^h x_i)$$ | 문맥 순서에는 둔감하지만 토큰 하나만 빠지거나 바뀌어도 깨짐 |
| **Skip** | $$P(s x_h)$$ | 가장 왼쪽 토큰만 사용. 나머지 변화에는 견디지만 삽입·삭제에 취약 |
| **Min** | $$\min_i P(s x_i)$$ | 순서에 둔감하고 삽입·삭제에 부분적으로 견딤. $$h = 4$$에서 2개가 사라져도 50% 확률로 같은 해시 |

여기에 생성할 토큰 자신도 해시 입력에 넣는 **SelfHash**(원 논문의 Algorithm 3)를 결합하면 문맥 폭이 사실상 1 늘어 무차별 대입이 더 어려워집니다.

Figure 2의 결과는 분명합니다. 작은 $$h$$가 기계 paraphrase에 가장 견고하고, $$h$$가 커지면 Additive는 무너지지만 Skip과 Min은 버팁니다. 대신 Min은 텍스트의 다양성(log diversity)이 낮습니다. 절충점으로 **$$h = 4$$의 Min-SelfHash**는 $$h = 1$$의 Additive-LeftHash와 같은 다양성을 내면서 더 견고합니다. 이후 실험에서 이 둘을 각각 "SelfHash"와 "LeftHash"로 부릅니다.

### 검출: WinMax

원래의 z-test는 문서 전체에 대해 계산하므로, 긴 사람 글 안에 워터마크 한 문단만 들어 있으면 주변 텍스트가 초록 비율을 희석합니다. **WinMax**는 초록 여부의 부분합 $$p_k = \sum_{i \le k} s_i$$를 만들고 **가장 z가 큰 연속 구간**을 찾습니다.

$$
z_{\text{win-max}} = \max_{i < j} \frac{(p_j - p_i) - \gamma (j - i)}{\sqrt{\gamma (1 - \gamma)(j - i)}}
$$

여러 가설을 동시에 검정하는 셈이라, 워터마크 없는 글과 비교해 고정 false positive rate에 맞게 임계값을 보정합니다. 생성 방식은 바꿀 필요가 없습니다. run-length 기반의 더 복잡한 검출기도 시험했지만 이득이 없었습니다.

## 4. 야생에서의 워터마크

위협 모델은 다음과 같습니다. 사용자는 글에 워터마크가 있다는 것은 알지만 해시, $$\gamma$$, $$h$$는 모르고, 검출을 피하려고 일부 또는 전부를 바꿔 씁니다. 이때 두 관찰이 신뢰성의 뼈대입니다.

1. 해시를 모르면 **원문의 n-gram을 하나도 남기지 않아야** 워터마크를 지울 수 있습니다. 긴 단어(여러 토큰)나 구절 하나라도 재활용하면 약해질지언정 워터마크는 남습니다.
2. 바꿔 쓴 글이 워터마크 쪽으로 **아주 조금만** 기울어도, 토큰이 충분하면 검출됩니다. 토큰마다 초록일 확률이 기준보다 $$\varepsilon$$만큼 높다면, 즉 $$|s| = \gamma T (1 + \varepsilon)$$이면 임계값 $$z$$에 대해 $$T = (z^2 - \gamma z^2)/(\varepsilon^2 \gamma)$$개의 토큰 뒤에 검출됩니다.

그래서 논문은 paraphrase가 워터마크를 **지우는** 것이 아니라 **검출에 필요한 토큰 수를 늘리는** 것이라고 봅니다. 실험은 LLaMA-7B(사람 실험은 Vicuna)로 C4 프롬프트에서 온도 0.7로 생성하고, 기본은 LeftHash, $$(\gamma, \delta) = (0.25, 2.0)$$입니다. 텍스트 길이 $$T$$가 결과를 좌우하므로 길이를 ±25토큰 범위로 엄격히 통제하고, 각 측정은 500개 이상의 양성·음성 샘플로 뒷받침했습니다.

### 기계 paraphrase

gpt-3.5-turbo(ChatGPT)와 전용 paraphrase 모델 Dipper 11B로 전문을 다시 쓰게 했습니다. GPT는 긴 입력을 받으면 사실상 요약해 길이를 절반 넘게 줄이기도 합니다.

- 공격 전후 $$T \approx 200$$에서 ROC-AUC가 0.05~0.15 떨어집니다.
- 공격 전 $$T = 600$$이면 Dipper와 GPT가 길이를 500, 300으로 줄여도 손실이 0.1 미만입니다.
- 공격 없는 설정에서는 AUC가 항상 0.999를 넘습니다.

**AUC @ T** 그래프(접두사 길이별 AUC)를 보면 200~300토큰에서는 공격이 성공한 것처럼 보여도 600토큰이면 AUC가 0.9 위로 회복합니다.

### Copy-Paste

워터마크 텍스트 조각을 사람이 쓴 문서 안에 끼워 넣는 합성 공격입니다. "CP-3-10%"는 전체의 10%만 워터마크이고 그것이 3군데로 나뉘어 있다는 뜻입니다. 25%가 남은 경우 기계 paraphrase보다 타격이 커서 200토큰에서 AUC 0.7 미만, 600토큰에서 0.85 미만입니다. 그래도 토큰이 늘면 검출력은 꾸준히 오릅니다.

### 사람의 paraphrase

대학원생 14명에게 Vicuna가 만든 LFQA(Reddit ELI5) 답변을 주고 **워터마크를 지우는 것을 목표로** 길이와 뜻을 유지하며 바꿔 쓰게 했습니다. 상위 3명에게는 100달러 상품권을 걸었습니다. P-SP 점수가 전문가 paraphrase 기준 0.7을 훌쩍 넘어 바꿔 쓰기 자체는 충실했습니다.

- 사람은 Dipper와 GPT보다 **강한 공격자**입니다.
- 그래도 평균적으로 **약 800토큰(500단어)** 뒤에는 명확히 검출됩니다. z = 4는 p값 $$3.2 \times 10^{-5}$$에 해당합니다.
- 예외는 참가자 13으로, 바꿔 쓰기도 강하고 제출한 글도 적었습니다. 반대로 참가자 22와 15는 약 250토큰(200단어)에 검출될 만큼 워터마크에 거의 영향을 주지 못했습니다.

## 5. 코드로 보기: 바꿔 쓸수록 토큰이 더 필요하고, 섞어 넣으면 WinMax가 잡는다

무작위 bigram 언어 모델에 LeftHash 워터마크를 심고, 세 가지를 확인합니다. (1) z-score가 워터마크 글과 일반 글을 가르는지, (2) 토큰의 일부를 무관한 토큰으로 바꾸는 단순한 "바꿔 쓰기"의 강도에 따라 $$z \ge 4$$에 필요한 토큰 수가 어떻게 늘어나는지, (3) 사람 글 600토큰 안에 워터마크 150토큰을 끼워 넣었을 때 전체 z와 WinMax의 차이입니다.

```python
import math, torch

torch.manual_seed(0)
V, GAMMA, DELTA = 1000, 0.25, 2.0                         # 어휘 크기, 초록 비율 γ, logit 가산 δ
lm = torch.randn(V, V) * 1.5                              # 장난감 언어 모델: 직전 토큰 → 다음 토큰 logit
KEY = 15485863

def greenlist(prev):                                       # LeftHash(h = 1): 직전 토큰으로 시드를 만들어 초록 목록 생성
    g = torch.Generator().manual_seed(KEY * (prev + 1))
    return torch.randperm(V, generator=g)[: int(GAMMA * V)]

def generate(T, watermark=True):
    seq = [torch.randint(V, (1,)).item()]
    for _ in range(T):
        logits = lm[seq[-1]].clone()
        if watermark:
            logits[greenlist(seq[-1])] += DELTA            # 초록 토큰의 logit에 δ를 더한다
        seq.append(torch.multinomial(torch.softmax(logits, -1), 1).item())
    return seq

def hits(seq):                                             # 토큰마다 초록 목록에 들었는지 0/1
    return [1 if t in set(greenlist(p).tolist()) else 0 for p, t in zip(seq[:-1], seq[1:])]

def z_score(s):
    T = len(s)
    return (sum(s) - GAMMA * T) / math.sqrt(GAMMA * (1 - GAMMA) * T)

def winmax(s):                                             # 가장 z가 큰 연속 구간을 찾는다
    best, T = -1e9, len(s)
    for i in range(T):
        run = 0
        for j in range(i + 1, T + 1):
            run += s[j - 1]
            best = max(best, (run - GAMMA * (j - i)) / math.sqrt(GAMMA * (1 - GAMMA) * (j - i)))
    return best

def paraphrase(seq, frac):                                 # 토큰의 frac 비율을 무관한 토큰으로 바꾼다
    out = list(seq)
    for i in torch.randperm(len(out))[: int(frac * len(out))].tolist():
        out[i] = torch.randint(V, (1,)).item()
    return out

# 1) 워터마크 유무에 따른 초록 비율과 z-score
wm, plain = generate(600), generate(600, watermark=False)
print(f"초록 토큰 비율: 워터마크 {sum(hits(wm)) / 600:.2f} | 일반 텍스트 {sum(hits(plain)) / 600:.2f} (기대 γ = {GAMMA})")
print(f"z-score (T = 600): 워터마크 {z_score(hits(wm)):5.1f} | 일반 텍스트 {z_score(hits(plain)):5.1f}")

# 2) 바꿔 쓰기 강도별로 z = 4(p ≈ 3e-5)에 도달하는 데 필요한 토큰 수
for frac in [0.0, 0.3, 0.5, 0.7]:
    need = []
    for _ in range(20):
        s = hits(paraphrase(generate(3000), frac))
        T = next((t for t in range(50, 3001, 25) if z_score(s[:t]) >= 4), None)
        need.append(T if T else 3000)
    print(f"토큰의 {frac:.0%}를 바꿔 씀: z ≥ 4에 평균 {sum(need) / len(need):6.0f} 토큰 필요")

# 3) Copy-Paste: 사람이 쓴 600토큰 안에 워터마크 150토큰을 끼워 넣었을 때, 전체 z와 WinMax
human = generate(600, watermark=False)
mixed = human[:300] + generate(150)[1:] + human[300:450]
s = hits(mixed)
print(f"Copy-Paste(25%): 전체 z-score {z_score(s):.1f} | WinMax {winmax(s):.1f} | 일반 텍스트 WinMax {winmax(hits(human)):.1f}")
```

```text
초록 토큰 비율: 워터마크 0.71 | 일반 텍스트 0.25 (기대 γ = 0.25)
z-score (T = 600): 워터마크  26.1 | 일반 텍스트   0.2
토큰의 0%를 바꿔 씀: z ≥ 4에 평균     50 토큰 필요
토큰의 30%를 바꿔 씀: z ≥ 4에 평균     89 토큰 필요
토큰의 50%를 바꿔 씀: z ≥ 4에 평균    288 토큰 필요
토큰의 70%를 바꿔 씀: z ≥ 4에 평균   1616 토큰 필요
Copy-Paste(25%): 전체 z-score 5.9 | WinMax 12.2 | 일반 텍스트 WinMax 3.5
```

- $$\delta = 2$$만으로 초록 비율이 0.25에서 0.71로 오르고, 600토큰에서 z는 26입니다. 일반 글은 0.2입니다.
- 토큰의 절반을 바꿔도 288토큰이면 z가 4를 넘습니다. 70%를 바꾸면 1,616토큰이 필요하지만 **여전히 검출됩니다**. 바꿔 쓰기는 워터마크를 지우지 않고 필요한 토큰 수를 늘릴 뿐이라는 논문의 주장 그대로입니다. 이 예제는 토큰을 무작위로 바꾸는 가장 단순한 모델이라, 실제 paraphrase는 문맥 토큰까지 바꿔 해시를 깨므로 더 강합니다.
- 25%만 워터마크인 문서에서 전체 z는 5.9로 희석되지만 WinMax는 12.2로 그 구간을 찾아냅니다. 다만 일반 글에서도 WinMax는 3.5로 전체 z보다 높게 나옵니다. 여러 구간을 뒤지는 다중 검정이라 **임계값을 따로 보정해야** 하는 이유입니다.

## 6. 다른 검출기와의 비교

공격 전 길이 $$T = 1000$$에서 z-score, WinMax, retrieval(BM25, Krishna et al. 2023), DetectGPT(T5-3B로 100개 perturbation)를 같은 공격으로 비교했습니다.

| 검출기 | GPT·Dipper paraphrase | Copy-Paste | 토큰 수에 따른 변화 |
|---|---|---|---|
| **DetectGPT** | 모든 공격에서 사실상 실패 | 실패 | 공격 아래서는 늘어도 좋아지지 않음 |
| **Retrieval** | 워터마크와 같거나 더 좋음 | 워터마크에 뒤짐 | 상승하지만 단조롭지 않고, 25%·10% CP에서는 T가 늘수록 오히려 떨어짐 |
| **z-score** | 좋음 | 좋음 | 꾸준히 상승 |
| **WinMax** | 좋음 | 가장 심한 CP-3-10%에서 압도적 | 꾸준히 상승 |

retrieval이 Copy-Paste에서 밀리는 이유는 분명합니다. $$T$$가 늘수록 원문(DB에 저장된 생성물)이 차지하는 비율이 줄어 유사도가 떨어집니다. 워터마크만 **모든 공격에서 토큰이 늘수록 꾸준히 강해졌습니다**.

### White-box라면

비밀 키가 유출되거나 워터마크가 공개된 경우, 공격자는 Dipper 같은 강한 paraphrase 모델의 생성 때 $$\delta$$를 **빼는** anti-watermark를 걸고 초록 비율이 정확히 $$\gamma T$$가 되도록 조절해 신호를 완전히 지울 수 있습니다. 이를 막으려면 $$h$$를 충분히 키워 키를 추측하기 어렵게 하거나 여러 키를 동시에 써야 합니다. 논문은 모든 사용 사례가 적대적이지는 않으며, 생성물의 확산을 추적하거나 다음 학습 데이터에서 걸러 내는 선의의 용도에서는 불완전한 워터마크도 충분히 가치 있다고 봅니다.

## 7. 검출 불가능성 이론과의 관계

Sadasivan et al.(2023)은 LLM이 사람의 언어 분포를 완벽히 흉내 내면 검출이 불가능하다고 주장했습니다. 논문의 답은 이렇습니다.

- 표준 방법으로 학습한 LLM은 무작위로 뽑은 사람을 흉내 내지 않습니다. 분류 손실은 사람 글보다 엔트로피가 낮은 분포를 선호하고, 서비스 제공자도 사람 같은 글보다 정돈된 글을 원합니다.
- 설령 사람 분포를 흉내 내더라도, 사람 언어의 분포는 매우 넓어서(같은 정리의 증명도 수학자마다 다름) 워터마크가 출력을 유효한 완성의 **부분집합**에 모으면서도 품질을 지킬 여지가 큽니다.
- 사후 검출기가 실패하는 이유는 두 분포가 비슷해서가 아니라 **차이를 수학적으로 기술할 수 없어서**입니다. 고정 시드의 의사 난수 샘플러를 쓰는 LLM은 프롬프트마다 출력이 하나로 집중되어 사람과 전혀 다른 분포인데도 검출하기 어렵습니다. 워터마크가 작동하는 이유는 분포를 크게 바꿔서가 아니라 **그 변화가 단순한 규칙으로 기술되기 때문**입니다.

## 8. 해석

**1. 검출 성능을 "정확도"가 아니라 "토큰 수의 함수"로 본 것이 이 논문의 핵심 관점입니다.** 200토큰에서 AUC 0.8은 실패처럼 보이지만 600토큰에서 0.9를 넘는 검출기와, 200토큰에서 0.9였다가 길어질수록 떨어지는 검출기는 전혀 다릅니다. 위 예제의 "필요한 토큰 수" 표가 그 관점을 그대로 담고 있습니다.

**2. 워터마크의 힘은 귀무가설이 사람에게 늘 성립한다는 데서 나옵니다.** 사람은 해시를 모르므로 어떤 글을 써도 초록 비율이 $$\gamma$$ 근처입니다. 그래서 z-score는 길이와 무관하게 해석 가능한 p값을 주고, false positive rate를 사용자가 통제할 수 있습니다. 사후 검출기에는 이런 귀무가설이 없습니다.

**3. 설계는 모두 trade-off입니다.** 문맥 폭 $$h$$는 비밀 유지와 견고함을, 해시 함수는 견고함과 다양성을, WinMax는 민감도와 false positive를 맞바꿉니다. Min-SelfHash $$h = 4$$라는 선택은 그 trade-off에서 찾은 하나의 절충점이지 유일한 답은 아닙니다.

**4. 사람 실험이 설득력을 줍니다.** 상금을 걸고 워터마크 제거를 목표로 바꿔 쓰게 했는데도 500단어 안에 대부분 검출되었다는 결과는, 기계 paraphrase 결과보다 훨씬 강한 증거입니다. 동시에 참가자 13처럼 짧은 글에서는 피할 수 있다는 한계도 정직하게 보여 줍니다.

**5. 평가를 읽을 때 주의할 점.** 주 실험은 LLaMA-7B와 C4라는 한 조합이고, 사람 실험은 참가자 14명입니다. retrieval과 DetectGPT는 저자들이 재현한 설정이라 원 저자의 최적 설정과 다를 수 있습니다. Copy-Paste는 합성 공격이고, 실제 사용자가 섞는 방식은 다를 수 있습니다. white-box 공격은 이론적으로 가능하다고만 밝혔고 실험하지 않았습니다.

## 9. 정리

| 항목 | 내용 |
|---|---|
| 목표 | 초록·빨강 목록 워터마크가 바꿔 쓰기와 섞어 넣기 뒤에도 검출되는지 측정 |
| 생성 | 직전 $$h$$개 토큰으로 초록 목록을 만들고 logit에 $$\delta$$ 가산. Min-SelfHash $$h = 4$$가 다양성과 견고함의 절충점 |
| 검출 | $$z = (\lvert s\rvert - \gamma T)/\sqrt{\gamma(1-\gamma)T}$$, 긴 문서 안의 조각은 WinMax로 최대 z 구간 탐색 |
| 결과 | GPT·Dipper paraphrase 후 600토큰에서 AUC 0.9 이상, 사람 paraphrase는 약 800토큰에 검출, CP 25%는 AUC 0.95 이상 |
| 비교 | DetectGPT는 공격 아래 실패, retrieval은 paraphrase에 강하지만 Copy-Paste와 긴 문서에서 약함. 워터마크만 토큰이 늘수록 꾸준히 강해짐 |
| 한계 | 키가 알려지면 anti-watermark로 제거 가능, 짧은 글은 강한 사람 공격자가 피할 수 있음 |

## References

- Kirchenbauer, Geiping et al., "On the Reliability of Watermarks for Large Language Models," 2023. [arXiv:2306.04634](https://arxiv.org/abs/2306.04634)
- Kirchenbauer et al., "A Watermark for Large Language Models," ICML 2023. [arXiv:2301.10226](https://arxiv.org/abs/2301.10226)
- Krishna et al., "Paraphrasing Evades Detectors of AI-generated Text, but Retrieval is an Effective Defense," 2023. [arXiv:2303.13408](https://arxiv.org/abs/2303.13408)
- Mitchell et al., "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature," ICML 2023. [arXiv:2301.11305](https://arxiv.org/abs/2301.11305)
- Sadasivan et al., "Can AI-Generated Text be Reliably Detected?," 2023. [arXiv:2303.11156](https://arxiv.org/abs/2303.11156)
- Chakraborty et al., "On the Possibilities of AI-Generated Text Detection," 2023. [arXiv:2304.04736](https://arxiv.org/abs/2304.04736)
