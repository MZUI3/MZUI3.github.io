---
title: "[RL] Decision Transformer: 원하는 return을 프롬프트로 주고 행동을 생성하는 시퀀스 모델링"
date: 2026-10-24 20:48:00 +0900
categories: [Reinforcement Learning, Model-Based]
tags: [reinforcement-learning, offline-rl, decision-transformer, sequence-modeling, transformer, gpt, return-to-go, credit-assignment, d4rl, atari]
description: 가치 함수도 policy gradient도 없이, (return-to-go, 상태, 행동) 시퀀스를 GPT로 학습하고 원하는 return을 조건으로 행동을 생성하는 Decision Transformer(Chen, Lu et al., NeurIPS 2021)의 표현, 학습·평가 절차, Atari·D4RL·Key-to-Door 결과와 %BC 비교를 정리하고, 논문의 최단 경로 예제를 코드로 재현합니다.
math: true
---

> [Trajectory Transformer]({% post_url 2026-08-24-trajectory-transformer %})가 강화학습을 하나의 시퀀스 모델링 문제로 보고 beam search로 계획했다면, 같은 시기의 **Decision Transformer**(DT, Chen, Lu et al., UC Berkeley·FAIR·Google Brain, NeurIPS 2021)는 더 단순한 길을 택합니다. 궤적을 (return-to-go, 상태, 행동)의 토큰 열로 쓰고 GPT로 **행동만** 예측하게 학습한 뒤, 평가 때 **원하는 return**을 첫 토큰으로 주면 그 return을 달성하는 행동이 나옵니다. 부트스트래핑도, 할인도, 가치 함수도 없습니다. 그런데도 Atari, D4RL, Key-to-Door에서 CQL 같은 offline RL 최고 수준과 맞먹거나 넘습니다. [π*0.6]({% post_url 2026-10-23-pi-star-0-6 %})의 advantage conditioning이나 [Gato]({% post_url 2026-10-20-gato %})가 모두 이 계보 위에 있습니다.

## 1. 발상: TD 대신 시퀀스 모델링

강화학습의 지배적 패러다임인 TD 학습은 부트스트래핑으로 return을 전파합니다. 이것이 deadly triad의 한 축이고, 할인은 근시안적 행동을 유도할 수 있습니다. 시퀀스 모델링으로 바꾸면 얻는 것이 많습니다.

- 부트스트래핑이 없어 **안정적**이고, 할인이 필요 없습니다.
- 언어·비전에서 검증된 **Transformer 학습 인프라**를 그대로 씁니다. 전용 RL 프레임워크가 필요 없습니다.
- self-attention이 상태와 return을 직접 연결하므로 Bellman backup이 천천히 전파하는 것과 달리 **credit assignment가 직접** 이루어지고, 희소하거나 방해적인 보상에 강합니다.
- 넓은 행동 분포를 모델링할 수 있어 일반화와 전이에 유리합니다.

return을 조건으로 지도학습하는 upside-down RL과 발상은 비슷하지만, DT는 **긴 시퀀스의 모델링**에서 이득을 얻으려는 시퀀스 모델링 관점입니다.

### 최단 경로 예제

방향 그래프에서 목표 노드에 도달하면 보상 0, 그 밖에는 -1입니다. **무작위 보행** 데이터만으로 GPT를 학습하고, 평가 때 각 노드에서 가능한 가장 높은 return을 조건으로 주면 **최적 경로**가 생성됩니다. 전문가 시연 없이, dynamic programming 없이, hindsight return 정보와 시퀀스 모델링만으로 정책 개선이 일어나는 것입니다.

## 2. 방법

### 궤적 표현

미래의 원하는 return을 조건으로 행동을 내야 하므로 보상 대신 **return-to-go** $$\hat R_t = \sum_{t'=t}^T r_{t'}$$를 씁니다.

$$
\tau = \big(\hat R_1, s_1, a_1,\ \hat R_2, s_2, a_2,\ \ldots,\ \hat R_T, s_T, a_T\big)
$$

### 구조

마지막 $$K$$ timestep을 넣으므로 토큰은 $$3K$$개입니다. modality마다 선형 임베딩(이미지는 합성곱 인코더)과 LayerNorm을 거치고, **timestep 임베딩**을 더합니다. 보통의 위치 임베딩과 달리 한 timestep의 세 토큰이 같은 임베딩을 공유합니다. 그다음은 causal mask를 쓴 GPT입니다.

### 학습과 평가

데이터셋에서 길이 $$K$$의 부분 시퀀스를 뽑아, **상태 토큰 자리의 출력**이 그 timestep의 행동을 예측하도록 학습합니다(이산 행동은 cross-entropy, 연속 행동은 MSE). 상태나 return을 예측하게 할 필요는 없었습니다.

평가 때는 목표 return(예: 최대 가능 return)과 시작 상태로 생성을 시작합니다. 행동을 실행해 보상 $$r$$을 받으면 목표 return에서 $$r$$을 빼고 다음 상태를 붙여 반복합니다. 문맥은 마지막 $$K$$개만 유지합니다.

```text
target_return = 1                       # 예: 전문가 수준
R, s, a, t = [target_return], [env.reset()], [], [1]
while not done:
    action = DecisionTransformer(R, s, a, t)[-1]
    new_s, r, done = env.step(action)
    R = R + [R[-1] - r]                 # return-to-go를 받은 보상만큼 줄인다
    s, a, t = s + [new_s], a + [action], t + [len(R)]
    R, s, a, t = 마지막 K개만 유지
```

## 3. 코드로 보기: 무작위 보행에서 최단 경로 꺼내기

논문 Figure 2의 예제를 그대로 재현합니다. 노드 12개의 방향 그래프에서 무작위 보행 3,000개(절반 정도만 목표 도달)를 모으고, 작은 2층 Transformer로 (return-to-go, 상태, 행동) 시퀀스를 학습합니다. 평가 때 목표 return을 **최단 거리의 음수**로 주면 최단 경로가, 낮은 return을 주면 긴 경로가 나오는지 봅니다.

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
# 1) 논문 Figure 2의 예: 방향 그래프에서 최단 경로 찾기. 목표 노드에서 보상 0, 그 밖에서는 -1
N_NODES, GOAL, T = 12, 0, 8
edges = {i: sorted({(i + 1) % N_NODES, (i * 3 + 1) % N_NODES, (i + 5) % N_NODES}) for i in range(N_NODES)}   # 노드마다 나가는 간선 3개
A_MAX = 3
def step(s, a): return edges[s][a % len(edges[s])]
# 최단 거리(평가용 정답)
dist = {GOAL: 0}; frontier = [GOAL]
while frontier:
    nxt = []
    for v in range(N_NODES):
        if v not in dist and any(step(v, a) in frontier for a in range(A_MAX)): dist[v] = dist[frontier[0]] + 1; nxt.append(v)
    frontier = nxt

# 2) 학습 데이터: 무작위 보행 궤적과 노드별 return-to-go(남은 보상의 합). 목표에 못 가면 -T로 둔다
def random_walk():
    s = torch.randint(1, N_NODES, (1,)).item(); S, A_ = [], []
    for _ in range(T):
        a = torch.randint(A_MAX, (1,)).item(); S.append(s); A_.append(a); s = step(s, a)
        if s == GOAL: break
    rewards = [-1] * (len(S) - 1) + ([0] if s == GOAL else [-1])
    rtg = [sum(rewards[i:]) if s == GOAL else -T for i in range(len(S))]
    return S, A_, rtg
data = [random_walk() for _ in range(3000)]
print(f"무작위 보행 {len(data)}개 중 목표 도달 {sum(1 for S, A_, r in data if r[0] > -T)}개, 평균 길이 {sum(len(S) for S, A_, r in data) / len(data):.1f}")

# 3) Decision Transformer: (R̂, s, a) 토큰을 번갈아 넣고 causal attention으로 행동 예측
class DT(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.emb_R, self.emb_s, self.emb_a, self.emb_t = nn.Linear(1, d), nn.Embedding(N_NODES, d), nn.Embedding(A_MAX + 1, d), nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.1, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, 2); self.head = nn.Linear(d, A_MAX)
    def forward(self, R, s, a, t):                                       # 모두 (B, K). a의 마지막은 예측 대상이라 패딩 토큰
        pos = self.emb_t(t)
        tok = torch.stack([self.emb_R(R[..., None]) + pos, self.emb_s(s) + pos, self.emb_a(a) + pos], 2).flatten(1, 2)   # (B, 3K, d)
        mask = torch.triu(torch.ones(tok.shape[1], tok.shape[1], dtype=torch.bool), 1)
        h = self.tf(tok, mask=mask)
        return self.head(h[:, 1::3])                                       # 상태 토큰 자리에서 행동을 예측

def batch(items):
    B = len(items); R = torch.zeros(B, T); s = torch.zeros(B, T, dtype=torch.long); a = torch.full((B, T), A_MAX); y = torch.full((B, T), -100)
    for i, (S, A_, rtg) in enumerate(items):
        L = len(S); R[i, :L] = torch.tensor(rtg, dtype=torch.float); s[i, :L] = torch.tensor(S); y[i, :L] = torch.tensor(A_); a[i, :L - 1] = torch.tensor(A_[:-1])
    return R / T, s, a, torch.arange(T)[None].expand(B, T), y

model = DT(); opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
for ep in range(60):
    for i in torch.randperm(len(data)).split(128):
        R, s, a, t, y = batch([data[j] for j in i])
        loss = F.cross_entropy(model(R, s, a, t).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
model.eval()

# 4) 평가: 시작 노드에서 "가장 높은 return"을 조건으로 주고 행동을 생성 (R̂ = -(최단 거리))
@torch.no_grad()
def rollout(start, target_rtg):
    S, A_, R = [start], [], [target_rtg]
    for k in range(T):
        Rt = torch.tensor(R, dtype=torch.float)[None] / T; st = torch.tensor(S)[None]; at = torch.tensor(A_ + [A_MAX])[None]
        logits = model(Rt, st, at, torch.arange(len(S))[None])[0, -1]
        a = logits.argmax().item(); A_.append(a); s = step(S[-1], a)
        if s == GOAL: return len(A_)
        S.append(s); R.append(R[-1] + 1)                                   # 보상 -1을 받았으므로 남은 return은 +1
    return None

hits, total_len, opt_len, tries = 0, 0, 0, 0
for start in range(1, N_NODES):
    L = rollout(start, -dist[start]); tries += 1
    if L is not None: hits += 1; total_len += L; opt_len += dist[start]
print(f"최단 거리를 목표 return으로 준 DT: 시작 노드 {tries}개 중 {hits}개 도달, 평균 경로 길이 {total_len / max(hits, 1):.2f} (최단 {opt_len / max(hits, 1):.2f})")
hits2, len2 = 0, 0
for start in range(1, N_NODES):
    L = rollout(start, -T); hits2 += L is not None; len2 += L or 0
print(f"낮은 return(-{T})을 목표로 준 DT: {hits2}개 도달, 평균 경로 길이 {len2 / max(hits2, 1):.2f}  (같은 모델, 다른 프롬프트)")
```

```text
무작위 보행 3000개 중 목표 도달 1245개, 평균 길이 6.4
최단 거리를 목표 return으로 준 DT: 시작 노드 11개 중 11개 도달, 평균 경로 길이 2.55 (최단 2.55)
낮은 return(-8)을 목표로 준 DT: 8개 도달, 평균 경로 길이 7.62  (같은 모델, 다른 프롬프트)
```

학습 데이터는 평균 6.4 step을 헤매는 무작위 보행뿐인데, 최단 거리를 return으로 주면 **11개 시작 노드 모두에서 정확히 최단 경로**를 생성합니다. 같은 모델에 낮은 return을 주면 일부러 돌아갑니다. 모델이 배운 것은 "이 노드에서 이 행동을 하면 남은 return이 얼마인가"라는 **hindsight 정보와 행동의 연관**이고, return 토큰은 어떤 기술을 꺼낼지 고르는 **프롬프트**입니다. 가치 함수도 Bellman 갱신도 없습니다.

## 4. 실험

비교 대상은 두 부류입니다. TD 학습(CQL이 주 비교 대상, BEAR, BRAC도)과 모방학습(behavior cloning, DT와 같은 구조에 return 조건만 뺀 것).

### Atari (DQN-replay의 1%)

| 게임 | **DT** | CQL | QR-DQN | REM | BC |
|---|---|---|---|---|---|
| Breakout | **267.5 ± 97.5** | 211.1 | 21.1 | 32.1 | 138.9 ± 61.7 |
| Qbert | 15.1 ± 11.4 | **104.2** | 1.7 | 1.4 | 17.3 ± 14.7 |
| Pong | 106.1 ± 8.1 | **111.9** | 20.0 | 39.1 | 85.2 ± 20.0 |
| Seaquest | **2.4 ± 0.7** | 1.7 | 1.4 | 1.0 | 2.1 ± 0.3 |

문맥 $$K = 30$$(Pong은 50). 4개 중 3개에서 CQL과 대등하고 나머지 기준선은 모두 넘습니다. 문맥 길이 1의 MLP는 Transformer보다 못했습니다.

### D4RL (OpenAI Gym)

| 데이터셋 | 환경 | **DT** | CQL | BEAR | BRAC-v | AWR | BC |
|---|---|---|---|---|---|---|---|
| Medium-Expert | HalfCheetah | **86.8** | 62.4 | 53.4 | 41.9 | 52.7 | 59.9 |
| Medium-Expert | Hopper | 107.6 | **111.0** | 96.3 | 0.8 | 27.1 | 79.6 |
| Medium-Expert | Walker | **108.1** | 98.7 | 40.1 | 81.6 | 53.8 | 36.6 |
| Medium | HalfCheetah | 42.6 | **44.4** | 41.7 | 46.3 | 37.4 | 43.1 |
| Medium | Hopper | **67.6** | 58.0 | 52.1 | 31.1 | 35.9 | 63.9 |
| Medium | Walker | 74.0 | **79.2** | 59.1 | 81.1 | 17.4 | 77.3 |
| Medium-Replay | HalfCheetah | 36.6 | **46.2** | 38.6 | 47.7 | 40.3 | 4.3 |
| Medium-Replay | Hopper | **82.7** | 48.6 | 33.7 | 0.6 | 28.4 | 27.6 |
| Medium-Replay | Walker | **66.6** | 26.7 | 19.2 | 0.9 | 15.5 | 36.9 |
| 평균 (Reacher 제외) | | **74.7** | 63.9 | 48.2 | 36.9 | 34.3 | 46.4 |

대부분의 작업에서 최고이고 나머지에서도 경쟁력이 있습니다. 특히 품질이 섞인 Medium-Replay에서 BC와의 차이가 큽니다.

### DT는 상위 데이터의 BC인가

return 상위 X%의 timestep만으로 BC를 하는 **%BC**와 비교했습니다. 데이터가 풍부한 D4RL에서는 %BC가 다른 offline RL과 맞먹고, DT는 최선의 %BC와 비슷합니다. 즉 DT는 전체 분포로 학습한 뒤 특정 부분집합에 **집중**할 수 있습니다. 반면 데이터가 적은 Atari(1%)에서는 %BC가 약하고 DT가 모든 %BC를 크게 넘습니다(Breakout 267.5 대 10%BC 28.5, 100%BC 138.9). 데이터가 적을 때는 목표 return과 다른 궤적까지 **모두 써서 일반화**하는 것이 낫습니다. 어느 부분집합이 최적인지 고르는 것보다 원하는 return을 지정하는 것이 사람에게 자연스럽다는 점도 장점입니다.

### Return을 이해하는가

목표 return을 넓은 범위로 바꾸면 실제 달성 return이 **높은 상관**을 보입니다. Pong, HalfCheetah, Walker에서는 oracle 선과 거의 겹치고, Seaquest에서는 데이터셋의 최대 return보다 높은 목표를 줘도 따라가 **외삽**이 일어나기도 합니다.

### 긴 credit assignment: Key-to-Door

첫 방에서 열쇠를 집고, 빈 방을 지나, 세 번째 방에서 문에 도달해야 보상을 받는 환경입니다. 보상이 에피소드 끝에 있고 중간 행동을 건너뛰어 처음과 연결해야 합니다. 무작위 행동 궤적으로 학습했을 때 성공률입니다.

| 데이터 | **DT** | CQL | BC | %BC | 무작위 |
|---|---|---|---|---|---|
| 무작위 궤적 1K | **71.8%** | 13.1% | 1.4% | 69.9% | 3.1% |
| 무작위 궤적 10K | 94.6% | 13.3% | 1.6% | **95.1%** | 3.1% |

hindsight return을 쓰는 DT와 %BC(성공 에피소드만)는 무작위 보행에서 거의 최적의 경로를 배우고, TD 학습(CQL)은 긴 horizon에 Q값을 전파하지 못합니다. 같은 환경에서 return 토큰까지 예측하게 하면 열쇠를 집거나 문에 도달하는 **결정적 사건에 attention**이 집중되어 정확한 가치 예측(critic)도 가능했습니다.

### 희소 보상

D4RL 보상을 에피소드 끝에 몰아주는 지연 보상 설정에서 TD 학습은 무너지지만 DT와 %BC는 거의 영향을 받지 않습니다. 보상의 밀도에 대한 가정이 최소이기 때문입니다.

## 5. 해석

**1. RL의 핵심 질문을 바꿨습니다.** "이 상태의 가치는 얼마인가"에서 "이 return을 내려면 무엇을 해야 하나"로. 전자는 Bellman 방정식과 부트스트래핑이 필요하고, 후자는 지도학습입니다. 위 예제에서 11줄짜리 rollout 함수가 가치 함수 없이 최단 경로를 찾은 것이 그 전환의 가장 작은 증거입니다.

**2. Transformer가 한 일은 credit assignment입니다.** Key-to-Door에서 CQL이 13%에 머물고 DT가 95%에 이른 차이는 구조에서 옵니다. self-attention은 수백 step 떨어진 열쇠와 보상을 한 번에 연결하고, TD는 그 거리를 step마다 한 칸씩 전파해야 합니다. 문맥 길이 1의 MLP가 Transformer보다 못했다는 각주도 같은 말입니다.

**3. 한계도 분명합니다.** DT는 데이터에 있는 return과 행동의 관계를 배우므로, 환경이 확률적이면 "운 좋게 높은 return을 낸 행동"을 따라 할 수 있습니다. 외삽이 되는 경우도 있지만 보장은 없고, Qbert처럼 CQL에 크게 뒤지는 게임도 있습니다. [Trajectory Transformer]({% post_url 2026-08-24-trajectory-transformer %})가 상태와 보상까지 모델링해 계획한 것, 후속 연구들이 가치 함수를 다시 들여온 것은 이 한계에 대한 응답입니다.

**4. 이 계보가 어디로 갔는지.** return 조건은 곧 프롬프트입니다. [Gato]({% post_url 2026-10-20-gato %})는 여러 작업의 궤적을 한 Transformer에 넣었고, [π*0.6]({% post_url 2026-10-23-pi-star-0-6 %})은 return 대신 advantage의 이진 지표를 조건으로 넣어 시연보다 나은 정책을 뽑았습니다. "조건부 시퀀스 모델링으로 RL을 한다"는 DT의 틀이 로봇 파운데이션 모델까지 이어진 셈입니다.

**5. 평가를 읽을 때 주의할 점.** CQL과 D4RL 기준선 수치는 원 논문에서 가져왔고 BC만 저자들이 돌렸습니다. Atari는 4개 게임, 3 seed이고 분산이 큽니다(Breakout ±97.5). 목표 return은 사람이 정하는 하이퍼파라미터이며, 논문은 "최대 가능 return"을 썼다고만 밝힙니다. Key-to-Door의 %BC는 성공 에피소드만 골라낸 것이라 성공 여부를 미리 안다는 가정이 들어갑니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 발상 | offline RL을 return-to-go 조건부 시퀀스 모델링으로. 가치 함수·policy gradient·부트스트래핑·할인 없음 |
| 표현 | $$(\hat R_t, s_t, a_t)$$ 토큰, modality별 임베딩 + timestep 임베딩(세 토큰 공유), causal GPT |
| 학습·평가 | 상태 토큰 자리에서 행동 예측(CE 또는 MSE). 평가는 목표 return을 주고 받은 보상만큼 줄이며 생성 |
| 결과 | Atari 4개 중 3개 CQL과 대등, D4RL 평균 74.7(CQL 63.9), Key-to-Door 무작위 데이터로 94.6%(CQL 13.3%) |
| 분석 | 데이터가 많으면 최선의 %BC 수준, 적으면 %BC를 크게 넘음. 목표 return과 달성 return의 상관 높음, 지연 보상에 강함 |
| 한계 | 확률적 환경의 운 좋은 궤적, 외삽 보장 없음, 목표 return 선택 필요 |

## References

- Chen, Lu et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling," NeurIPS 2021. [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)
- Janner, Li, Levine, "Offline Reinforcement Learning as One Big Sequence Modeling Problem," NeurIPS 2021. [arXiv:2106.02039](https://arxiv.org/abs/2106.02039)
- Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning," NeurIPS 2020. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)
- Schmidhuber, "Reinforcement Learning Upside Down," 2019. [arXiv:1912.02875](https://arxiv.org/abs/1912.02875)
- Kumar, Peng, Levine, "Reward-Conditioned Policies," 2019. [arXiv:1912.13465](https://arxiv.org/abs/1912.13465)
- Fu et al., "D4RL: Datasets for Deep Data-Driven Reinforcement Learning," 2020. [arXiv:2004.07219](https://arxiv.org/abs/2004.07219)
