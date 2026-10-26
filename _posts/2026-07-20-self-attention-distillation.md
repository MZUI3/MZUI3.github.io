---
title: "[Distillation] Self Attention Distillation: 자기 attention map을 교사로 삼아 가벼운 차선 검출 CNN 학습하기"
date: 2026-07-20 18:20:00 +0900
categories: [Neural Network, Lightweight]
tags: [knowledge-distillation, self-distillation, self-attention-distillation, sad, lane-detection, attention-map, enet, culane, tusimple, lightweight]
description: 얇고 희소한 차선 라벨만으로는 학습이 어렵다는 문제에서 출발해, 어느 정도 학습된 네트워크의 깊은 층 attention map을 얕은 층이 top-down으로 모방하게 하는 Self Attention Distillation(Hou et al., ICCV 2019)의 attention 생성, 모방 경로, 도입 시점, ENet-SAD의 TuSimple·CULane·BDD100K 결과를 정리하고 attention map 계산을 코드로 확인합니다.
math: true
---

> 차선은 길고 얇아서 라벨 픽셀이 배경보다 훨씬 적습니다. 이 **희소한 감독**만으로는 가려짐, 모호한 차선, 어두운 조명에서 모델이 자주 실패합니다. 기존 해법인 multi-task learning은 추가 라벨이 필요하고, SCNN의 message passing은 추론 시간의 35%를 차지합니다. **Self Attention Distillation**(SAD, Hou, Ma, Liu, Loy, CUHK·SenseTime·NTU, ICCV 2019)은 둘 다 피합니다. 어느 정도 학습된 네트워크의 **깊은 층 attention map**이 장면의 맥락을 담고 있다는 관찰에서, 그것을 **얕은 층이 모방**하게 하는 '공짜' 감독을 만듭니다. 라벨도 교사도 추론 비용 증가도 없습니다. 0.98M 파라미터의 ENet-SAD가 20배 큰 SCNN과 맞먹거나 넘었습니다.

## 1. 관찰: 절반 학습된 모델의 attention은 이미 맥락을 안다

ENet을 40K episode까지 학습한 뒤 네 블록의 attention map을 보면, 차선의 위치와 장면의 대략적인 윤곽이 이미 드러납니다. 여기서 SAD를 켜서 block 3이 block 4를, block 2가 block 3을 모방하게 하면 두 가지가 일어납니다.

1. 얕은 층의 attention map이 더 풍부한 맥락을 담도록 **다듬어집니다**.
2. 얕은 층의 표현이 좋아지자 **깊은 층도 좋아집니다**. block 4는 아무것도 모방하지 않는데도 차선 위치의 attention이 훨씬 또렷해집니다.

SAD 없이 60K까지 학습해도 attention map은 거의 좋아지지 않습니다.

## 2. 방법

### Attention map 만들기

$$m$$번째 층의 활성화 $$A_m \in \mathbb{R}^{C_m \times H_m \times W_m}$$에서 공간 map $$\mathbb{R}^{H_m \times W_m}$$를 만드는 함수 $$\mathcal{G}$$를 채널 방향 통계로 정의합니다(Zagoruyko & Komodakis의 attention transfer).

$$
\mathcal{G}_{\text{sum}}(A_m) = \sum_{i=1}^{C_m} |A_{mi}|,\qquad \mathcal{G}^p_{\text{sum}}(A_m) = \sum_{i=1}^{C_m} |A_{mi}|^p,\qquad \mathcal{G}^p_{\max}(A_m) = \max_i |A_{mi}|^p
$$

$$p > 1$$이면 활성화가 큰 영역에 더 큰 가중치를 두고, $$\max$$보다 $$\text{sum}$$이 여러 뉴런의 가중치를 합쳐 덜 편향됩니다. 실험에서 $$\mathcal{G}^2_{\text{sum}}$$이 가장 좋았습니다. 활성화 기반과 기울기 기반 attention 중 **활성화 기반만 효과**가 있었습니다.

### Attention generator와 손실

AT-GEN은 $$\Psi(\cdot) = \Phi(\mathcal{B}(\mathcal{G}^2_{\text{sum}}(\cdot)))$$입니다. 크기가 다르면 bilinear 업샘플링 $$\mathcal{B}$$를 하고, 공간 softmax $$\Phi$$를 적용합니다. 손실은 **연속한 층 사이**의 L2 거리를 더한 것입니다.

$$
\mathcal{L}_{\text{distill}} = \sum_{m=1}^{M-1} \mathcal{L}_d\big(\Psi(A_m), \Psi(A_{m+1})\big)
$$

$$\Psi(A_{m+1})$$이 목표(교사)이고, 경로마다 다른 가중치를 두지 않아도 잘 되었습니다. 전체 손실은 네 항입니다.

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{seg}}(s, \hat s) + \alpha \mathcal{L}_{\text{IoU}}(s, \hat s)}_{\text{분할 손실}} + \underbrace{\beta \mathcal{L}_{\text{exist}}(b, \hat b)}_{\text{차선 존재}} + \underbrace{\gamma \mathcal{L}_{\text{distill}}(A_m, A_{m+1})}_{\text{증류}}
$$

$$\mathcal{L}_{\text{IoU}} = 1 - N_p / (N_p + N_g - N_o)$$는 예측·정답 차선 픽셀의 IoU를 올리는 손실이고, $$\mathcal{L}_{\text{exist}}$$는 SCNN을 따라 붙인 차선 존재 예측 branch(P1)의 이진 cross-entropy입니다. $$\alpha = \beta = \gamma = 0.1$$입니다. 경로는 인접 층 외에 $$M(M-1)/2$$개까지 확장할 수 있습니다.

### 구조 손질

ENet의 인코더 E1~E4와 디코더 D1, D2에 차선 존재 branch P1을 붙이고(인코더 공유), P1에는 파라미터를 늘리지 않고 수용 영역을 키우는 dilated convolution을 썼습니다. CULane에서 E4의 feature map이 36×100밖에 안 되어 정보 손실이 커서 E3와 E4의 출력을 이어 붙였습니다.

## 3. 코드로 보기: attention map과 SAD 손실의 생김새

장난감 활성화에서 세 가지 attention 함수가 "차선" 모양의 활성화를 얼마나 강조하는지, AT-GEN의 업샘플링과 softmax, 그리고 인접 블록 사이의 SAD 손실을 계산합니다.

```python
import torch, torch.nn.functional as F

torch.manual_seed(0)
# 1) 어떤 층의 활성화 A ∈ R^{C×H×W}에서 attention map을 만드는 세 가지 방법(Zagoruyko & Komodakis)
A = torch.relu(torch.randn(16, 8, 8))                      # 채널 16, 8×8 (예: ENet block 3의 출력)
A[:, 2:5, 3] += 3.0                                        # 세로로 긴 "차선" 모양의 강한 활성화
G_sum = A.abs().sum(0)                                     # Σ_i |A_i|
G2_sum = (A ** 2).sum(0)                                   # Σ_i |A_i|²  ← 논문이 고른 방식
G2_max = (A ** 2).max(0).values                            # max_i |A_i|²
for name, g in [("G_sum", G_sum), ("G²_sum", G2_sum), ("G²_max", G2_max)]:
    print(f"{name:7s}: 차선 위치의 평균 / 전체 평균 = {g[2:5, 3].mean() / g.mean():.2f}배")

# 2) AT-GEN Ψ(·) = Φ(B(G²_sum(·))): 크기가 다르면 bilinear 업샘플링(B) 뒤 공간 softmax(Φ)
def at_gen(A, size):
    g = (A ** 2).sum(0, keepdim=True)[None]                # 1×1×h×w
    g = F.interpolate(g, size=size, mode="bilinear", align_corners=False)
    return F.softmax(g.flatten(1), -1).view(size)          # 합이 1인 공간 분포

A2, A3, A4 = torch.relu(torch.randn(8, 16, 16)), torch.relu(torch.randn(16, 8, 8)), torch.relu(torch.randn(32, 8, 8))
for blk in (A2, A3, A4): blk[:, blk.shape[1] // 4: blk.shape[1] // 2, blk.shape[2] // 2] += 2.0   # 같은 자리에 차선
psi2, psi3, psi4 = at_gen(A2, (16, 16)), at_gen(A3, (16, 16)), at_gen(A4, (16, 16))

# 3) SAD 손실: 앞 블록의 attention map이 바로 다음 블록의 것을 모방 (P23 + P34), 목표는 stop-gradient
L_sad = F.mse_loss(psi2, psi3.detach()) + F.mse_loss(psi3, psi4.detach())
print(f"L_distill(P23 + P34) = {L_sad.item():.2e}  (attention map은 합이 1이라 값이 작으므로 γ로 스케일을 맞춘다)")
M = 4
print(f"블록 {M}개에서 가능한 모방 경로 수: M(M-1)/2 = {M * (M - 1) // 2}개 (논문은 인접 경로 P23 + P34가 최선)")
```

```text
G_sum  : 차선 위치의 평균 / 전체 평균 = 6.02배
G²_sum : 차선 위치의 평균 / 전체 평균 = 10.62배
G²_max : 차선 위치의 평균 / 전체 평균 = 4.28배
L_distill(P23 + P34) = 1.07e-02  (attention map은 합이 1이라 값이 작으므로 γ로 스케일을 맞춘다)
블록 4개에서 가능한 모방 경로 수: M(M-1)/2 = 6개 (논문은 인접 경로 P23 + P34가 최선)
```

제곱합($$\mathcal{G}^2_{\text{sum}}$$)이 강한 활성화 영역을 가장 크게 강조합니다(10.6배). 제곱은 큰 값을 키우고, 합은 여러 채널의 증거를 모으기 때문입니다. $$\max$$는 채널 하나의 값만 보므로 노이즈 채널 하나에 흔들립니다. 손실 자체는 모양이 단순합니다. 깊은 블록의 map을 detach해서 목표로 두고 얕은 블록의 map과 L2를 재는 것이 전부이며, 추론 때는 AT-GEN을 쓰지 않으므로 비용이 0입니다.

## 4. 실험

### 데이터셋과 지표

| 데이터셋 | 프레임 | 해상도 | 도로 | 지표 |
|---|---|---|---|---|
| TuSimple | 6,408 | 1280×720 | 고속도로 | 정확도(맞춘 차선 점 / 정답 점), FP, FN |
| CULane | 133,235 | 1640×590 | 도시·시골·고속도로, 9가지 어려운 상황 | 30픽셀 폭 선의 IoU > 0.5를 TP로 한 F1 |
| BDD100K | 80,000 | 1280×720 | 도시·시골·고속도로, 프레임당 8개 이상 차선 | 픽셀 정확도, IoU |

BDD100K는 두 선으로 라벨된 차선의 중심선을 새 목표로 만들고 학습용은 8픽셀로 팽창했습니다. 모든 모델에 같은 분할 손실과 augmentation(회전, crop, flip)을 적용해 공정하게 비교했습니다. SAD는 CULane·BDD100K에서 60K episode 중 40K 시점에 켭니다.

### 결과

**TuSimple**

| 방법 | 정확도 | FP | FN |
|---|---|---|---|
| ENet | 93.02% | 0.0886 | 0.0734 |
| LaneNet | 96.38% | 0.0780 | 0.0244 |
| EL-GAN | 96.39% | 0.0412 | 0.0336 |
| SCNN | 96.53% | 0.0617 | 0.0180 |
| R-18-SAD | 96.02% | 0.0786 | 0.0451 |
| R-34-SAD | 96.24% | 0.0712 | 0.0344 |
| **ENet-SAD** | **96.64%** | 0.0602 | 0.0205 |

**CULane**(F1, 교차로는 FP만)

| 범주 | 비율 | ENet-SAD | R-34-SAD | R-101-SAD | ResNet-101 | SCNN |
|---|---|---|---|---|---|---|
| Normal | 27.7% | 90.1 | 89.9 | 90.7 | 90.2 | 90.6 |
| Crowded | 23.4% | 68.8 | 68.5 | 70.0 | 68.2 | 69.7 |
| Night | 20.3% | 66.0 | 64.6 | 66.3 | 65.9 | 66.1 |
| No line | 11.7% | 41.6 | 42.2 | 43.5 | 41.7 | 43.4 |
| Shadow | 2.7% | 65.9 | 67.7 | 67.0 | 64.6 | 66.9 |
| Dazzle light | 1.4% | 60.2 | 59.9 | 59.9 | 59.8 | 58.5 |
| Curve | 1.2% | 65.7 | 66.0 | 65.7 | 65.5 | 64.4 |
| Crossroad (FP) | 9.0% | 1998 | 1960 | 2052 | 2183 | 1990 |
| **Total** | | **70.8** | 70.7 | **71.8** | 70.8 | 71.6 |
| 실행 시간(ms) | | **13.4** | 50.5 | 171.2 | 171.2 | 133.5 |
| 파라미터(M) | | **0.98** | 22.72 | 52.53 | 52.53 | 20.72 |

**BDD100K**: ENet 34.12% / 14.64 → ENet-SAD **36.56% / 16.02**로 SCNN(35.79% / 15.84)을 넘어 최고입니다. ResNet-101도 SAD로 34.45% → 35.56%.

- ENet-SAD는 SCNN보다 **20배 적은 파라미터와 10배 빠른 속도**로 TuSimple과 BDD100K에서 앞서고 CULane에서 거의 같습니다.
- ResNet-18/34-SAD가 ENet-SAD보다 조금 못한 것은 디코더가 단순한 업샘플링뿐이기 때문입니다.
- 깊은 ResNet-101에도 SAD가 CULane 70.8 → 71.8, BDD100K 34.45 → 35.56으로 도움이 됩니다.
- P1 branch의 기울기를 끊어도 ENet-SAD는 96.61%, 70.8, 36.54%로 거의 같아서, **이득은 SAD 자체**에서 옵니다.

### Ablation

**모방 경로**(TuSimple, ENet)

| 경로 | 정확도 | 경로 | 정확도 | 경로 | 정확도 |
|---|---|---|---|---|---|
| P12 | 91.22% | P23 | 94.72% | P23 + P24 | 95.38% |
| P13 | 91.36% | P24 | 94.63% | **P23 + P34** | **96.64%** |
| P14 | 91.47% | P34 | 95.29% | P24 + P34 | 96.52% |

- SAD는 **중간·높은 층에서** 잘 되고, 낮은 층(block 1)에 걸면 오히려 나빠집니다. 낮은 층은 국소 세부를 검출하도록 되어 있는데, 전역 정보를 담은 높은 층을 따라 하게 하면 그 능력이 망가집니다.
- **인접 층**을 연속으로 모방하는 것이 비인접 층보다 좋습니다(P23 + P34 > P24 + P34). 이웃한 층의 attention이 의미적으로 더 가깝기 때문으로 봅니다.
- **거꾸로**(높은 층이 낮은 층을 모방) 하면 93.02% → 91.26%로 나빠집니다. 낮은 층의 map은 세부가 많고 노이즈가 많아 높은 층의 전역 정보를 흐립니다.

**Deep supervision과 비교**: 중간 층에 1×1 합성곱과 업샘플링으로 예측을 만들어 라벨로 직접 감독하는 deep supervision보다 세 데이터셋 모두에서 SAD가 좋습니다(ENet: TuSimple 94.69 → 96.64, CULane 69.6 → 70.8, BDD100K 15.38 → 16.02 IoU). 라벨은 희소하고 경직되어 있지만 attention 목표는 **부드럽고 장면 구조의 맥락**을 담으며, SAD 경로가 깊은 층에서 얕은 층으로의 피드백 연결 역할을 한다는 설명입니다.

**도입 시점**: 언제 켜든 최종 성능은 거의 같지만 **수렴 속도**가 다릅니다. 너무 이르면 깊은 층이 아직 좋은 목표를 못 만들어 느려집니다.

## 5. 해석

**1. "공짜 감독"의 출처가 분명합니다.** [Be Your Own Teacher]({% post_url 2026-07-09-be-your-own-teacher %})가 깊은 분류기의 출력을, [SRDL]({% post_url 2026-06-30-self-referenced-deep-learning %})이 과거 자신의 출력을 교사로 썼다면, SAD는 깊은 층의 **attention map**을 교사로 씁니다. 분할처럼 공간 구조가 중요한 작업에서 attention map은 출력보다 풍부한 신호이고, 희소한 라벨이 주지 못하는 "어디를 봐야 하는가"를 가르칩니다.

**2. 어느 층이 누구를 가르치느냐가 결과를 가릅니다.** ablation의 결론은 명확합니다. 위에서 아래로, 인접한 층끼리, 낮은 층은 건드리지 말 것. 이것은 CNN의 층별 역할(낮은 층은 세부, 높은 층은 전역)을 존중한 설계이고, 거꾸로 하면 해롭다는 실험이 그 역할 분담을 역으로 증명합니다.

**3. 타이밍이 수렴 속도를 정합니다.** 최종 성능은 비슷해도, 깊은 층이 어느 정도 학습된 뒤(논문은 절반 학습)에 켜야 효과가 빨리 난다는 관찰은 SRDL의 stage-complete 스케줄과 같은 교훈입니다. 자기 증류는 교사가 곧 자신이므로, 자신이 믿을 만해진 뒤에만 가르칠 자격이 생깁니다.

**4. 경량 모델의 가치를 정확히 겨냥했습니다.** 차선 검출은 차량에서 실시간으로 돌아야 합니다. 0.98M 파라미터, 13.4ms의 ENet-SAD가 20.72M, 133.5ms의 SCNN과 맞먹는다는 것은 학술적 성과를 넘어 배포 가능성의 차이입니다. 추론 비용이 0이라는 점이 message passing 대비 결정적 장점입니다.

**5. 평가를 읽을 때 주의할 점.** ENet에는 SAD 외에도 dilated convolution, E3·E4 결합 같은 구조 손질이 들어갔고, ENet baseline(93.02%)에 이 손질이 포함되었는지는 명확하지 않습니다. LaneNet과 EL-GAN은 코드가 없어 논문 수치를 가져왔습니다. BDD100K의 지표(픽셀 정확도, IoU)와 라벨 변환(중심선, 팽창)은 저자들이 정한 것이라 다른 연구와 직접 비교하기 어렵습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 문제 | 길고 얇은 차선의 희소한 라벨로는 맥락을 배우기 어렵고, MTL은 추가 라벨, MP는 추론 비용이 필요 |
| 관찰 | 어느 정도 학습된 모델의 깊은 층 attention map은 이미 차선과 장면 구조를 담는다 |
| 방법 | $$\mathcal{G}^2_{\text{sum}}$$ → 업샘플링 → 공간 softmax로 attention map을 만들고, 얕은 블록이 인접한 깊은 블록의 map을 L2로 모방(top-down, 절반 학습 뒤 도입) |
| 결과 | ENet-SAD(0.98M, 13.4ms): TuSimple 96.64%, CULane F1 70.8, BDD100K 36.56%/16.02로 SCNN(20.72M, 133.5ms)과 대등 이상 |
| Ablation | 중간·높은 층의 인접 경로(P23 + P34)가 최선, 낮은 층·역방향은 해로움, deep supervision보다 좋음 |
| 비용 | 학습 때만 attention generator 사용, 추론 비용 증가 없음 |

## References

- Hou, Ma, Liu, Loy, "Learning Lightweight Lane Detection CNNs by Self Attention Distillation," ICCV 2019. [arXiv:1908.00821](https://arxiv.org/abs/1908.00821)
- Zagoruyko & Komodakis, "Paying More Attention to Attention," ICLR 2017. [arXiv:1612.03928](https://arxiv.org/abs/1612.03928)
- Pan et al., "Spatial As Deep: Spatial CNN for Traffic Scene Understanding," AAAI 2018. [arXiv:1712.06080](https://arxiv.org/abs/1712.06080)
- Paszke et al., "ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation," 2016. [arXiv:1606.02147](https://arxiv.org/abs/1606.02147)
- Lee et al., "Deeply-Supervised Nets," AISTATS 2015. [arXiv:1409.5185](https://arxiv.org/abs/1409.5185)
