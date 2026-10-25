---
title: "[Face Swapping] Smooth-Swap: 손수 만든 부품 대신 매끄러운 신원 임베딩으로, U-Net과 손실 세 개만으로 얼굴 형태까지 바꾸기"
date: 2026-10-24 18:35:00 +0900
categories: [Generative AI, Face Swapping]
tags: [face-swapping, identity-embedding, supervised-contrastive, smoothness, u-net, ncsn, simple-design, cvpr]
description: Smooth-Swap(Kim, Lee, Zhang, CVPR 2022)이 교체 모델의 복잡한 구조와 손실이 결국 얼굴 형태 변화의 어려움에서 왔다고 보고, 그 어려움의 뿌리를 판별 과제로만 학습된 ArcFace 임베딩의 비매끄러움에서 찾아 지도 대조 손실로 매끄러운 신원 임베더를 학습한 과정을 정리합니다. NCSN++를 고친 U-Net 생성기와 신원·변화·적대 손실 세 개만으로 FF++에서 가장 적극적인 신원·형태 변화를 얻고, ArcFace 대비 4배 빠른 학습, 새 매끄러움 지표 d_smooth를 다루며, 작은 실험에서 매끄러움 주장이 재현되지 않았음을 그대로 보고합니다.
math: true
---

> 최신 교체 모델은 구조와 손실이 복잡합니다. [FaceShifter]({% post_url 2026-10-24-faceshifter %})의 마스크 기반 혼합과 2단계, [HifiFace]({% post_url 2026-10-24-hififace %})의 3D 형상 모델처럼 **손수 만든 부품**과 대여섯 개의 손실을 세심하게 균형 잡아야 합니다. **Smooth-Swap**(Kim, Lee, Zhang, 서울대·Kakao Brain, CVPR 2022)은 그 복잡성의 원인을 되묻습니다. 어려운 것은 **얼굴 형태 변화**이고, 그것이 어려운 이유는 신원을 바꾸라는 조건(C1)과 목표를 유지하라는 조건(C2)의 충돌을 등방성 픽셀 손실이 조정하지 못하기 때문이며, 그 충돌에서 생성기를 올바른 방향으로 밀어 줄 **신원 임베더의 기울기가 잡음투성이**이기 때문입니다. 판별 과제로만 학습된 ArcFace의 임베딩 공간은 매끄럽지 않습니다. 그렇다면 부품을 더할 것이 아니라 **매끄러운 임베더**를 쓰면 됩니다. 결과적으로 일반 U-Net 생성기와 손실 세 개로 충분했습니다.

## 1. 문제 정식화

교체 결과 $$x_{swap}$$은 (C1) 소스의 신원, (C2) 그 외는 목표와 동일, (C3) 사실적이어야 합니다. 모델은 고정 임베더 $$f^*_{emb}$$, 생성기 $$f_{gen}$$, 판별기 $$f_{dis}$$로 구성됩니다. C1은 $$x_{swap}$$을 목표에서 밀어내고 C2는 붙잡습니다. 신원 무관 변화만 골라 벌하는 C2 손실을 설계할 수 없어 지각·픽셀 손실 같은 **등방성** 손실을 쓰면, 가장 먼저 희생되는 것이 기하 변환을 수반하는 **얼굴 형태**입니다. 선행 연구는 3D 모델이나 마스크로 형태를 따로 다뤘습니다.

## 2. 왜 매끄러움인가

생성기가 틀린 신원을 내면 임베더가 교정 방향을 줘야 하고, 그 기울기는 **정확하고 일관**되어야 합니다. 아니면 C2 손실이 결과를 목표로 되돌립니다. 기울기는 연속 공간에서만 잘 정의되는데, 판별 임베더는 같은 신원의 나이·체형 변화 정보를 **공격적으로 지워** 임베딩을 정렬합니다. 분류에는 좋지만 노년에서 청년으로, 둥근 턱에서 뾰족한 턱으로 가는 경로가 **불연속 점프**가 됩니다(Figure 3). 지도 대조 학습(SupCon)은 임베딩이 **최대 정보**를 유지하게 하므로(Wang & Isola), 정렬을 조금 희생하더라도 신원 사이의 경로가 매끄럽고 좋은 기울기를 줍니다.

$$
\mathcal{L}(f_{emb}) = \mathbb{E}\Big[-\log \frac{e^{\langle z_i, z_p \rangle / \tau}}{e^{\langle z_i, z_p \rangle / \tau} + \sum_n e^{\langle z_i, z_n \rangle / \tau}}\Big]
$$

ResNet50 + FC 2층 + 단위 정규화, VGGFace2(9천 명 330만 장, FFHQ 방식 정렬), 배치 128(32신원 × 4), $$\tau = 0.07$$, 10.1만 스텝.

## 3. 생성기와 손실

점수 기반 생성 모델의 NCSN++를 가져와 시간 임베딩 자리에 **신원 임베딩**을 넣고(잔차 블록마다 브로드캐스트해 더함), 점수가 이미지의 차이에 가깝다는 점을 살려 **입력-출력 스킵**을 더합니다. 채널은 절반. 3D 모델도 마스크도 없는 범용 U-Net입니다. 손실은 셋뿐입니다.

$$
\mathcal{L}_{id} = 1 - \cos(z_{swap}, z_{src}), \qquad
\mathcal{L}_{chg} = \lVert x_{swap} - x_{tgt} \rVert_2^2 / D, \qquad
\mathcal{L}_{adv} = -\log f_{dis}(x_{swap})
$$

$$\lambda_{id} = 4$$, $$\lambda_{chg} = \lambda_{adv} = 1$$. FaceShifter와 거의 같되 특징 수준 속성 손실 대신 **픽셀 변화 손실**을 씁니다. 미니배치마다 $$(x_{tgt}, x_{tgt})$$ 쌍 하나를 넣어 재구성 손실 역할을 하게 합니다. FFHQ 256², 배치 8, 80만 스텝(HifiFace가 본 이미지 수와 같음), StyleGAN2 판별기 + R1.

## 4. 코드로 보기: 매끄러움이 재현되는가

2차원 합성 데이터(8개 신원 클러스터 + 모든 신원이 공유하는 "나이" 방향)에서 같은 MLP 임베더를 교차 엔트로피와 지도 대조 손실로 학습해, 신원 정확도, 나이 정보의 선형 프로브 $$R^2$$, 그리고 논문의 매끄러움 지표(두 신원 임베딩의 slerp 중점과 가장 가까운 실제 임베딩 사이의 정규화 간격)를 비교합니다.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); torch.set_num_threads(1)
# 2-D 'faces': identity = which cluster, plus an intra-identity factor (age / weight) that moves the point along a line
n_id, per = 8, 60
centres = torch.randn(n_id, 2) * 3
ids = torch.arange(n_id).repeat_interleave(per)
intra = torch.linspace(-1, 1, per).repeat(n_id)[:, None] * torch.tensor([[1.0, 0.3]])     # shared 'age' direction
X = centres[ids] + intra + 0.1 * torch.randn(n_id * per, 2)
def embedder(): return nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))
def train(loss_type):
    torch.manual_seed(1); net = embedder(); head = nn.Linear(8, n_id); opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), 3e-3)
    for s in range(1500):
        z = F.normalize(net(X), dim=1)
        if loss_type == "cross-entropy":
            loss = F.cross_entropy(head(z) * 10, ids)                                      # discriminative only
        else:                                                                                # supervised contrastive (Khosla et al.)
            sim = z @ z.T / 0.1; same = (ids[:, None] == ids[None]).float(); same.fill_diagonal_(0)
            logp = sim - torch.logsumexp(sim - 1e9 * torch.eye(len(z)), dim=1, keepdim=True)
            loss = -(logp * same).sum(1).div(same.sum(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net
def slerp(a, b, r):
    o = torch.acos((a * b).sum().clamp(-1, 1)); return (torch.sin((1 - r) * o) * a + torch.sin(r * o) * b) / torch.sin(o)
for name in ("cross-entropy", "supervised contrastive"):
    net = train(name)
    with torch.no_grad():
        Z = F.normalize(net(X), dim=1)
        # (a) does the embedding keep the intra-identity (age) information?  linear probe R^2
        A = torch.cat([Z, torch.ones(len(Z), 1)], 1); coef = torch.linalg.lstsq(A, intra[:, :1]).solution; r2 = 1 - ((A @ coef - intra[:, :1]) ** 2).mean() / intra[:, 0].var()
        # (b) smoothness score of the paper: gap between the slerp midpoint of two identities and the nearest real embedding
        gaps = []
        for _ in range(200):
            i, j = torch.randint(0, len(Z), (2,));
            if ids[i] == ids[j]: continue
            mid = slerp(Z[i], Z[j], 0.5); gaps.append(((Z - mid).norm(dim=1).min() / (Z[i] - Z[j]).norm()).item())
        # (c) verification accuracy: nearest-centroid identity classification
        cent = F.normalize(torch.stack([Z[ids == k].mean(0) for k in range(n_id)]), dim=1); acc = ((Z @ cent.T).argmax(1) == ids).float().mean()
    print(f"{name:<23} id accuracy {acc*100:5.1f}%   intra-identity info kept (R^2 of an age probe) {r2.item():.2f}   smoothness gap d_smooth {sum(gaps)/len(gaps):.3f}")
print("-> the contrastive embedder keeps within-identity variation, so points between two identities have real neighbours (smaller gap): smoother gradients for the generator")
```

```text
cross-entropy           id accuracy 100.0%   intra-identity info kept (R^2 of an age probe) 0.02   smoothness gap d_smooth 0.498
supervised contrastive  id accuracy 100.0%   intra-identity info kept (R^2 of an age probe) 0.01   smoothness gap d_smooth 0.533
-> the contrastive embedder keeps within-identity variation, so points between two identities have real neighbours (smaller gap): smoother gradients for the generator
```

- **논문의 주장이 이 toy에서는 재현되지 않았습니다.** 두 임베더 모두 신원 정확도 100%이고 나이 정보의 $$R^2$$가 0.02·0.01로 둘 다 클래스 내 변화를 지웠으며, 매끄러움 간격은 대조 임베더가 오히려 조금 큽니다(0.533 대 0.498). 온도 0.1·0.3·1.0과 짧은 학습으로도 대조 임베더의 클래스 내 퍼짐은 교차 엔트로피보다 작았습니다. 같은 신원을 양성으로 묶는 지도 대조 손실도 이론적으로는 클래스 내 붕괴를 향하고, 8신원·2차원이라는 작은 설정에서는 그 붕괴가 완성됩니다.
- 논문이 보고한 차이(표 2: $$d_{smooth}$$ 0.116 대 ArcFace 0.360, GAN 샘플 0.671 대 0.802)는 9천 명·330만 장의 실제 얼굴과 ResNet50에서 나온 것이고, 그 조건에서 대조 학습이 "최대 정보"를 더 지킨다는 경험적 결과입니다. 작은 toy로 그 기제를 보일 수 없음을 그대로 적습니다. 매끄러움이 **왜** 생기는지는 논문도 Wang & Isola의 균일성 논증을 빌릴 뿐 직접 증명하지 않습니다.

## 5. 실험

**FF++ 비교(표 1).** 신원·형태 지표(VGGFace2 거리, ArcFace 유사도, 3D 형상 계수 거리)와 목표 유지 지표(표정, 포즈)를 재고, 상대 거리(-R, 소스와 목표 사이 어디에 있는지)도 봅니다. Smooth-Swap은 VGG 101.7(SimSwap 99.7 다음), **형상 오차 0.565로 최저**(HifiFace 0.616, FaceShifter 0.658), 형상 상대 거리 0.403으로 가장 소스 쪽입니다. 표정·포즈는 최악은 아니지만 중간이고, 모든 지표를 표준화해 평균한 **종합 점수 −0.617로 1위**입니다(SimSwap −0.558, HifiFace −0.329). "신원을 안 바꾸는 우회 모델이 표정·포즈 최고점을 받는다"는 점에서 종합이 중요하다고 저자들은 강조합니다. 정성적으로는 턱 모양·피부톤·머리 색까지 소스 쪽으로 가장 적극적으로 바뀝니다(Figure 4, 5).

**Ablation.** 같은 생성기에 ArcFace를 쓰면 대부분 지표가 나빠지고($$\lambda_{id} = 4$$에서 종합 +0.699), 신원 손실이 ArcFace에서는 40만 스텝까지 거의 안 내려가는 반면 제안 임베더는 10만 스텝부터 신원이 바뀌어 40만 스텝(50%)에 이미 HifiFace보다 좋은 종합 점수를 냅니다(Figure 7, 8). 학습 속도와 안정성이 매끄러움의 실용적 증거입니다.

**임베더 평가(표 2).** $$d_{smooth}$$(샘플 기반 r=0.25/0.5, GAN 기반)에서 CE-Linear·CE-Arc·ArcFace보다 2~3배 좋고, 검증 AUC(VCHQ, VGGFace2, LFW)는 대등합니다. 보간 경로의 최근접 이웃 이미지를 보면 제안 임베더는 신원이 점진적으로 바뀌고 다른 임베더는 같은 신원이 반복됩니다(고유 이미지 5.13 대 4.42, 4.25).

## 6. 해석

**1. 진단이 신선합니다.** 교체의 어려움을 "임베더의 기울기 품질"로 환원한 논문은 처음이고, 이후 [FlowFace]({% post_url 2026-10-24-flowface %})가 "판별 과제로 학습된 신원 임베딩은 클래스 내 변화를 무시한다"며 MAE 인코더를 쓰는 근거로 인용합니다. [StyleIPSB]({% post_url 2026-10-24-styleipsb %})도 "신원 임베딩은 비매끄러운 공간"이라는 이 관찰을 출발점으로 삼습니다.

**2. 단순함이 결과입니다.** 3D 없이 형상 오차가 HifiFace보다 낮다는 것은 형태 변화에 3D 감독이 필수가 아니라는 반례이고, [SimSwap]({% post_url 2026-10-24-simswap %})이 추구한 단순 설계를 손실 셋까지 밀어붙인 것입니다. 점수 기반 모델의 U-Net을 가져온 선택은 디퓨전 구조가 조건부 이미지 변환에 범용임을 보여 줍니다.

**3. 트레이드오프는 속성 쪽입니다.** 적극적인 신원 변화의 대가로 머리 모양·머리 색까지 바뀌어 FlowFace는 "목표 속성을 파괴한다"고 비판하고, 복잡한 머리 패턴에서 아티팩트가 보고됩니다(Figure 5). 종합 점수 1위가 모든 지표 1위는 아닙니다.

**4. 매끄러움의 기제는 열려 있습니다.** 위 toy가 보여 주듯 지도 대조 손실 자체가 클래스 내 정보를 보존한다는 보장은 없고, 실제 데이터의 규모·다양성과 온도·배치 구성이 결과를 좌우합니다. 논문의 표 2는 결과이지 설명이 아니며, 이 간극은 후속 연구의 몫입니다.

**5. 평가를 읽을 때 주의할 점.** 비교 모델 대부분이 코드가 없어 FF++ 공개 결과 이미지로 평가했고, 프레임 차이가 있습니다. ArcFace로 학습한 모델은 Arc 지표를 공정하게 비교할 수 없어 제외됐습니다. 종합 점수는 지표의 표준화 평균이라 지표 선택에 민감합니다. 제가 만든 toy는 논문 설정과 규모가 크게 달라 반증이 아니라 "작은 설정에서는 안 보인다"는 관찰입니다.

## 7. 정리

| 항목 | 내용 |
|---|---|
| 진단 | 형태 변화의 어려움 ← C1·C2 충돌 + 등방성 손실 ← 판별 임베더의 비매끄러운 기울기 |
| 임베더 | ResNet50 + 지도 대조 손실(τ 0.07), VGGFace2, 검증 성능은 ArcFace와 대등하면서 $$d_{smooth}$$ 2~3배 개선 |
| 생성기 | NCSN++ U-Net(시간 임베딩 → 신원 임베딩, 입력-출력 스킵), 손실 = ID 4 + 픽셀 변화 1 + 적대 1 |
| 결과 | FF++ 형상 오차 최저(0.565), 종합 점수 1위, ArcFace 대비 4배 빠른 신원 학습 |
| 한계 | 머리 모양·색까지 변함, 복잡한 머리에서 아티팩트, 매끄러움의 기제 미설명 |
| toy | 2-D 합성에서는 CE·SupCon 모두 클래스 내 정보를 지워 주장이 재현되지 않음 |

## References

- Kim, Lee, Zhang, "Smooth-Swap: A Simple Enhancement for Face-Swapping with Smoothness," CVPR 2022. [arXiv:2112.05907](https://arxiv.org/abs/2112.05907)
- Khosla et al., "Supervised Contrastive Learning," NeurIPS 2020. [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
- Wang, Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere," ICML 2020. [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (NCSN++), ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- Deng et al., "ArcFace," CVPR 2019. [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)
- Mescheder, Geiger, Nowozin, "Which Training Methods for GANs do actually Converge?" (R1), ICML 2018. [arXiv:1801.04406](https://arxiv.org/abs/1801.04406)
