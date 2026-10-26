---
title: "[Distillation] Self-Referenced Deep Learning: 절반 학습한 자기 자신을 교사로 쓰는 공짜 증류"
date: 2026-06-30 22:23:00 +0900
categories: [Neural Network, Lightweight]
tags: [knowledge-distillation, self-distillation, srdl, soft-label, learning-rate-schedule, random-restart, flat-minima, fast-optimization, lightweight]
description: 큰 교사를 따로 학습하지 않고, 전체 epoch의 절반을 학습한 모델이 낸 소프트 라벨을 나머지 절반의 학습에 되먹이는 Self-Referenced Deep Learning(Lan, Zhu, Gong, ACCV 2018)의 2단계 절차, stage-complete 학습률 스케줄, 무작위 재시작의 역할과 CIFAR·Tiny ImageNet·Market-1501 결과를 정리하고 작은 실험으로 재현합니다.
math: true
---

> Knowledge distillation은 학생을 좋게 만들지만 **큰 교사를 먼저 학습**해야 합니다. ResNet-32 학생에 WRN-28-10 교사를 쓰면 학습 비용이 157배 늘어납니다. **Self-Referenced Deep Learning**(SRDL, Lan, Zhu, Gong, Queen Mary University of London, ACCV 2018)은 교사를 없앱니다. 같은 모델을 전체 epoch의 절반만 학습한 뒤 그 모델의 예측을 **자기 발견 지식**으로 뽑고, 나머지 절반 동안 라벨과 그 지식을 함께 맞춥니다. 추가 비용은 거의 0이고, 작은 모델일수록 이득이 큽니다. [Be Your Own Teacher]({% post_url 2026-07-09-be-your-own-teacher %})가 깊이 방향으로 자기 증류를 했다면, SRDL은 **시간 방향**으로 합니다.

## 1. 문제: 증류와 빠른 최적화를 함께

논문은 두 문제를 한 번에 풀려 합니다.

- **증류의 비용**: 교사를 학습하는 시간이 학생 학습을 압도합니다. ImageNet 규모에서는 몇 주 단위입니다.
- **빠른 최적화**: 초기화, 학습률 설계 등으로 수렴을 앞당기는 연구가 있지만, 보통 일반화 성능 향상과는 별개로 다뤄졌습니다.

vanilla 학습은 라벨만 보고 $$M$$ epoch를 돕니다. 그 과정에서 **모델 스스로 발견한 지식**(클래스 간 상관)은 버려집니다. SRDL은 그것을 주워 씁니다. 동시 연구인 Born-Again Networks(Furlanello et al., 2018)도 이전 세대의 지식으로 다음 세대를 학습하지만, 세대마다 전체 예산을 쓰므로 총비용이 몇 배가 됩니다. SRDL은 **같은 epoch 수** 안에서 끝냅니다.

## 2. 방법

### 두 단계

1. **1단계**($$M/2$$ epoch): 보통의 cross-entropy로 학습하되, 아래의 특별한 학습률 스케줄을 씁니다. 끝나면 "절반 학습된" 모델 $$\theta^*$$에서 학습 샘플마다 소프트 라벨을 뽑습니다.

$$
\tilde p(c \mid x, \theta^*) = \frac{\exp(z_c / T)}{\sum_j \exp(z_j / T)},\qquad T = 3
$$

2. **2단계**($$M/2$$ epoch): 라벨의 cross-entropy와 소프트 라벨에 대한 KL을 함께 최소화합니다.

$$
\mathcal{L} = \mathcal{L}_{ce} + T^2 R_{kl},\qquad R_{kl} = \sum_j \tilde p(j \mid x, \theta^*) \log \frac{\tilde p(j \mid x, \theta^*)}{\tilde p(j \mid x, \theta)}
$$

$$T^2$$은 Hinton의 KD와 같은 이유로, 소프트 타깃의 기울기가 $$1/T^2$$로 줄어드는 것을 보정합니다. 소프트 라벨 추출 비용은 전체 학습의 0.67% 미만입니다.

### Stage-complete 학습률 스케줄

핵심 설계입니다. 보통의 step decay는 전체 $$M$$ epoch에 맞춰 50%, 75% 지점에서 학습률을 낮춥니다. 이를 그대로 쓰면 1단계가 끝나는 $$M/2$$ 시점에 모델은 **아직 불안정한 지역 최솟값**에 있어 뽑아낸 지식이 설익습니다. 그래서 각 단계가 **자기만의 완전한 decay 스케줄**을 갖게 합니다.

$$
\epsilon_t = \epsilon_0 \times f_{\text{step}}(t, 0.5M)
$$

즉 $$M/2$$ epoch 안에 50%, 75% 지점의 decay를 모두 거쳐 안정된 상태에 도달하게 합니다. 이 하나가 결과를 가릅니다. ResNet-32·CIFAR-100에서 stage-incomplete 스케줄은 **58.11%**, stage-complete는 **71.63%**입니다. 설익은 지식은 도움이 아니라 방해가 됩니다.

### 무작위 재시작

2단계를 $$\theta^*$$에서 이어서 학습하면, 모방 손실과 현재 파라미터가 겹쳐 더 나은 최솟값을 탐색할 여지가 줄어듭니다. 그래서 2단계는 **무작위로 다시 초기화한 모델**에서 시작합니다. 근거는 세 가지입니다. 1단계의 지식 대부분은 소프트 라벨로 추출되었고, 같은 데이터를 다시 쓰며, 새 초기화가 더 나은 최솟값으로 갈 기회를 줍니다. 재시작이 1.90%p를 더합니다(69.73% → 71.63%).

```text
1단계   무작위 초기화 θ
        t = 1..M/2: 학습률 ε_0·f_step(t, M/2), cross-entropy로 갱신
        소프트 라벨 p̃(c | x, θ*) 추출 (T = 3)
2단계   무작위 재시작 θ
        t = 1..M/2: 학습률 ε_0·f_step(t, M/2), L_ce + T²·R_kl로 갱신
```

## 3. 코드로 보기: 같은 epoch, 조금 더 나은 모델

16차원 합성 분류 문제(라벨 노이즈 포함, 학습 샘플 1,500개)에서 작은 MLP를 vanilla로 100 epoch 학습한 것과 SRDL(50 + 50)을 비교합니다. 재시작과 stage-complete 스케줄을 각각 뺀 변형도 함께 봅니다. 5회 평균입니다.

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
C, D, M = 10, 16, 100                                      # 클래스 수, 입력 차원, 전체 epoch 수
W1, W2 = torch.randn(D, 64) / D ** 0.5, torch.randn(64, C) / 8   # 정답을 만드는 숨은 비선형 함수
def data(n):
    x = torch.randn(n, D)
    return x, (torch.tanh(x @ W1) @ W2 * 12 + torch.randn(n, C)).argmax(1)   # 라벨 노이즈 포함
Xtr, ytr = data(1500); Xte, yte = data(20000)

def net(): return nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, C))
def acc(m): return (m(Xte).argmax(1) == yte).float().mean().item() * 100

def train(m, epochs, soft=None, T=3, lr0=0.05, decay_at=(0.5, 0.75)):
    """step-decay(50%, 75% 지점에서 ×0.1)로 epochs만큼 학습. soft가 있으면 CE + T²·KL(soft ‖ 학생)"""
    opt = torch.optim.SGD(m.parameters(), lr0, momentum=0.9, nesterov=True, weight_decay=2e-4)
    for ep in range(epochs):
        for g in opt.param_groups: g["lr"] = lr0 * (0.1 ** sum(ep >= epochs * d for d in decay_at))
        for i in torch.randperm(len(Xtr)).split(128):
            z = m(Xtr[i]); loss = F.cross_entropy(z, ytr[i])
            if soft is not None:
                loss = loss + T ** 2 * F.kl_div(F.log_softmax(z / T, -1), soft[i], reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
    return m

def srdl(stage_complete=True, restart=True, T=3):
    m = net()
    if stage_complete: train(m, M // 2)                                        # 1단계: M/2 epoch 안에 완전한 decay 스케줄
    else: train(m, M // 2, decay_at=(1.0, 1.5))                                # 비교용: 전체 M 기준 스케줄의 앞 절반(= decay 없음)
    with torch.no_grad(): soft = F.softmax(m(Xtr) / T, -1)                     # 자기 발견 지식: 소프트 라벨
    if restart: m = net()                                                       # 2단계는 무작위 재시작
    return acc(train(m, M // 2, soft=soft, T=T))

runs = 5
print(f"vanilla (M = {M} epoch)      : {sum(acc(train(net(), M)) for _ in range(runs)) / runs:.2f}%")
print(f"SRDL (M/2 + M/2, 재시작)     : {sum(srdl() for _ in range(runs)) / runs:.2f}%")
print(f"SRDL, 재시작 없이 이어서 학습: {sum(srdl(restart=False) for _ in range(runs)) / runs:.2f}%")
print(f"SRDL, 1단계 decay 불완전     : {sum(srdl(stage_complete=False) for _ in range(runs)) / runs:.2f}%")
```

```text
vanilla (M = 100 epoch)      : 75.14%
SRDL (M/2 + M/2, 재시작)     : 75.93%
SRDL, 재시작 없이 이어서 학습: 75.65%
SRDL, 1단계 decay 불완전     : 75.43%
```

같은 100 epoch에서 SRDL이 0.8%p 높고, 재시작을 빼면 이득의 일부가, 1단계 스케줄을 불완전하게 두면 대부분이 사라집니다. 작은 합성 문제라 차이는 논문보다 작지만, 세 구성 요소의 순서는 논문의 ablation(Table 4, 5)과 같습니다. 소프트 라벨이 하는 일은 라벨 노이즈가 섞인 경계 근처 샘플에 "확신하지 말라"는 정보를 주는 것으로, 소프트 라벨을 만드는 모델이 충분히 수렴해 있어야 그 정보가 믿을 만합니다.

## 4. 실험

세 분류 데이터셋(CIFAR-10/100, Tiny ImageNet)과 person re-identification(Market-1501)에서 5회 평균입니다. 학습 비용은 forward FLOPs × epoch × 데이터 수로 잽니다.

### Vanilla와 비교

| 모델 | 파라미터 | CIFAR-10 | CIFAR-100 | Tiny ImageNet | 학습 비용 |
|---|---|---|---|---|---|
| ResNet-32 vanilla / SRDL | 0.5M | 92.53 / **93.12** (+0.59) | 69.02 / **71.63** (+2.61) | 53.33 / **55.53** (+2.20) | 동일 |
| WRN-28-10 vanilla / SRDL | 36.5M | 94.98 / **95.41** (+0.43) | 78.32 / **79.38** (+1.06) | 58.38 / **60.80** (+2.42) | 동일 |
| DenseNet-BC vanilla / SRDL | 25.6M | 96.68 / **96.87** (+0.19) | 82.83 / **83.59** (+0.76) | 62.88 / **64.19** (+1.31) | 동일 |

- 세 구조 모두 좋아지고, **작은 ResNet-32가 가장 많이** 좋아집니다. 자원이 제한된 환경에 매력적인 이유입니다.
- ImageNet의 DenseNet-201도 Top-1 77.20 → 77.72, Top-5 94.57 → 94.89입니다.

### KD와 비교

ResNet-32 학생에 WRN-28-10과 ResNet-110을 교사로 쓴 KD와 비교합니다.

| 방법 | 교사 | CIFAR-10 | CIFAR-100 | Tiny ImageNet | 학습 비용(CIFAR) |
|---|---|---|---|---|---|
| Vanilla | - | 92.53 | 69.02 | 53.33 | 0.08 |
| KD | WRN-28-10 (36.5M) | 92.83 | **72.58** | **56.80** | 12.70 (추가 157배) |
| KD | ResNet-110 (1.7M) | 92.75 | 71.17 | 55.06 | 0.30 |
| **SRDL** | 없음 | **93.12** | 71.63 | 55.53 | **0.08** |

- 강한 교사의 KD는 효과가 있지만 추가 비용이 157배입니다(논문의 계산은 12.70/0.08 − 1). SRDL은 CIFAR-100과 Tiny ImageNet에서 그에 근접하고 CIFAR-10에서는 넘습니다.
- ResNet-110 교사의 KD는 **모든 면에서 SRDL에 집니다**. 교사가 과적합되어(마지막 epoch의 cross-entropy 0.0087 대 ResNet-32의 0.1637) 추가 정보가 적다는 가설을 제시합니다. 교사가 있다고 늘 낫지는 않습니다.

### Person re-identification

Market-1501에서 ResNet-50과 DenseNet-121을 SRDL로 학습하면 single-query Rank-1이 각각 87.5 → 89.3, 90.1 → 91.7, mAP가 69.9 → 73.5, 74.0 → 76.8로 오릅니다. re-id 전용 구조나 학습 기법 없이 범용 네트워크로 당시 최고 수준(MLFN 90.0/74.3)을 넘었습니다.

### 그 밖의 분석

- **앙상블**: 절반 학습 모델과 최종 모델을 앙상블하면 72.33%로 0.70%p 더 오릅니다. vanilla로 $$M/2$$씩 두 번 학습한 앙상블(72.02%)보다 좋아, 두 모델이 상보적임을 시사합니다.
- **최솟값의 폭**: 파라미터에 무작위 방향의 섭동을 더했을 때 SRDL 해가 vanilla 해보다 손실이 천천히 오릅니다. 더 **넓은 지역 최솟값**이라 일반화가 좋다는 해석입니다.

## 5. 해석

**1. 교사는 "다른 모델"일 필요가 없습니다.** KD의 효용이 교사의 소프트 라벨에 담긴 클래스 간 상관에서 온다면, 그 상관은 같은 모델도 어느 정도 학습하면 발견합니다. SRDL은 그 발견을 버리지 않고 되먹입니다. 비용은 0에 가깝고, 위 예제에서 본 것처럼 소프트 라벨이 라벨 노이즈를 완화하는 정규화로 작동합니다.

**2. 이 논문의 진짜 기여는 스케줄입니다.** "자기 예측을 타깃으로 쓴다"는 아이디어 자체는 단순하지만, 1단계를 **완전히 수렴시키지 않으면 오히려 해롭다**(58% 대 72%)는 발견이 이 방법을 작동하게 합니다. 지식의 질이 전달 방식보다 중요하다는 교훈이고, [SAD]({% post_url 2026-07-20-self-attention-distillation %})가 "깊은 층이 어느 정도 학습된 뒤에 SAD를 켜야 효과가 빨리 난다"고 한 것과 같은 맥락입니다.

**3. 재시작은 반직관적이지만 합리적입니다.** 이어서 학습하면 될 것 같지만, 모방 손실이 현재 파라미터 근처에 모델을 묶어 둡니다. 지식은 소프트 라벨에 담겨 있으니 파라미터는 버려도 됩니다. snapshot ensemble이 학습률을 올려 다른 최솟값을 찾는 것과 비슷한 발상입니다.

**4. 작은 모델에 더 좋다는 점이 실용적입니다.** 큰 교사를 쓰는 KD는 큰 모델을 학습할 자원이 있는 사람의 도구입니다. SRDL은 학생을 학습할 자원만 있으면 되고, 그 학생이 작을수록 이득이 큽니다. 다만 CIFAR-100처럼 클래스가 많을 때는 작은 모델이 스스로 발견하는 지식이 강한 교사에 못 미칩니다.

**5. 평가를 읽을 때 주의할 점.** 2018년 논문이라 비교 대상이 vanilla KD뿐이고 Born-Again Networks 같은 동시 연구와의 직접 비교는 없습니다. ImageNet 결과는 DenseNet-201 하나이고 1회 실행입니다. "넓은 최솟값" 분석은 한 방향 섭동의 정성적 그래프입니다. 2단계 손실의 가중치($$T^2$$ 고정)와 온도($$T = 3$$)는 탐색하지 않았습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 목표 | 큰 교사 없이, 추가 비용 없이 KD의 일반화 이득 얻기 |
| 방법 | $$M/2$$ epoch 학습 → 소프트 라벨($$T = 3$$) 추출 → 무작위 재시작 후 $$M/2$$ epoch를 CE + $$T^2$$ KL로 학습 |
| 핵심 설계 | 각 단계가 완전한 step-decay 스케줄을 갖는 stage-complete 스케줄(없으면 58% 대 72%), 2단계 무작위 재시작(+1.9%p) |
| 결과 | ResNet-32 CIFAR-100 +2.61%p, Tiny ImageNet +2.20%p. ResNet-110 교사 KD보다 좋고 WRN-28-10 교사 KD에 근접(비용 1/157) |
| 추가 | 절반 모델과의 앙상블 +0.70%p, re-id Market-1501에서 범용 구조로 SOTA 수준 |
| 한계 | 클래스가 많을 때 강한 교사 KD에 못 미침, 비교 범위와 분석이 제한적 |

## References

- Lan, Zhu, Gong, "Self-Referenced Deep Learning," ACCV 2018. [arXiv:1811.07598](https://arxiv.org/abs/1811.07598)
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network," 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Furlanello et al., "Born Again Neural Networks," ICML 2018. [arXiv:1805.04770](https://arxiv.org/abs/1805.04770)
- Huang et al., "Snapshot Ensembles: Train 1, Get M for Free," ICLR 2017. [arXiv:1704.00109](https://arxiv.org/abs/1704.00109)
- Keskar et al., "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima," ICLR 2017. [arXiv:1609.04836](https://arxiv.org/abs/1609.04836)
