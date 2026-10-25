---
title: "[Generative AI] LIVE: 레이어를 한 겹씩 쌓아 래스터 이미지를 편집 가능한 SVG로 바꾸기"
date: 2026-10-24 08:00:00 +0900
categories: [Generative AI, Vector Graphics]
tags: [vectorization, svg, bezier, differentiable-rendering, live, layer-wise, udf-loss, xing-loss, diffvg]
description: LIVE(Ma et al., CVPR 2022)가 학습 데이터 없이 미분 가능한 렌더러만으로 래스터 이미지를 레이어 구조를 가진 SVG로 변환하는 과정을 정리합니다. 성분 단위 경로 초기화, 윤곽에 집중하는 UDF 손실, 자기 교차를 막는 Xing 손실의 역할과 Emoji·Pics 벤치마크 결과를 다루고, 작은 실험으로 색 평균 편향과 두 손실의 동작을 확인합니다.
math: true
---

> [DeepSVG]({% post_url 2026-08-15-deepsvg %})처럼 벡터 그래픽을 생성하는 딥 모델은 폰트나 아이콘 같은 좁은 도메인에서 학습되어 새 이미지에 잘 일반화되지 않고, 만들어 내는 SVG도 중복된 도형이 많아 편집하기 어렵습니다. **LIVE**(Layer-wise Image Vectorization, Ma, Zhou, Xu, Sun, Filev, Orlov, Fu, Shi, Northeastern·UIUC·Adobe·Picsart, CVPR 2022)는 모델도 라벨도 없이, **미분 가능한 렌더러 위에서 경로를 하나씩 추가하며 최적화**해 래스터 이미지를 사람이 편집할 수 있는 레이어 구조의 SVG로 바꿉니다. 이모지는 4개 경로로 완벽히 복원하고, 같은 경로 수에서 DiffVG보다 MSE가 훨씬 낮습니다.

## 1. 문제: 벡터화에서 빠져 있던 "위상"

래스터화는 성숙한 기술이지만 그 역인 벡터화는 여전히 어렵습니다. 기존 접근은 두 갈래였습니다.

- **분할 전처리 기반**(Photo2ClipArt 등): 이미지를 먼저 영역으로 나눈 뒤 각 영역을 벡터화합니다. 분할 정확도에 의존하고, 고대비 텍스처가 작은 조각으로 쪼개져 중복이 생깁니다.
- **딥 생성 모델 기반**(Im2Vec, DeepSVG 등): 폰트나 이모지 데이터셋으로 학습하므로 도메인 밖에서 무너지고, 생성된 도형의 순서가 불안정합니다.

둘 다 놓친 것은 이미지에 숨어 있는 **레이어 위상**입니다. 사람이 그리는 SVG는 배경 위에 큰 도형, 그 위에 작은 도형이 순서대로 쌓입니다. LIVE는 이 순서를 최적화 과정 자체에 넣습니다.

## 2. 방법

### 점진적 레이어 추가

기본 도형은 닫힌 3차 베지어 경로 하나이고, 경로당 세그먼트 수 $$s$$(기본 4)로 복잡도를 조절합니다. 알고리즘은 단순합니다. 스케줄 $$N$$에 따라 매 단계 $$n$$개의 경로를 새로 초기화해 추가하고, **지금까지의 모든 경로**의 제어점 $$P$$와 색 $$C$$를 미분 가능한 렌더러(DiffVG)로 500회 최적화합니다. 단계 $$i$$에 추가하는 경로 수는 $$\min(2^{i-1}, 32)$$입니다. 먼저 추가된 경로가 아래 레이어, 나중 경로가 위 레이어가 되므로 결과 SVG는 자연스럽게 coarse-to-fine 레이어 구조를 갖습니다.

### 성분 단위 초기화

초기 위치가 나쁘면 중복 도형이 생깁니다. LIVE는 **크고 아직 못 그린 영역**을 다음 목표로 삼습니다. 현재 렌더링과 목표의 픽셀 색 차이를 계산하고, 임계값 $$c_\alpha = 0.1$$ 이하는 이미 맞은 것으로 간주해 버리며, 남은 차이를 200개 구간으로 양자화한 뒤 **가장 큰 연결 성분의 무게중심**에 새 경로를 놓습니다. 경로 자체는 반지름 5픽셀의 원 위에 제어점을 균등 배치하는 원 초기화를 씁니다(무작위 초기화보다 볼록한 도형으로 수렴).

### UDF 손실

MSE $$\lVert I - \hat I \rVert_2^2$$는 모든 픽셀을 똑같이 보기 때문에 최적화 중인 경로와 무관한 픽셀까지 끌어들여 **이미지 전체의 평균 색으로 치우칩니다**. LIVE는 각 픽셀의 경로 윤곽까지의 부호 없는 거리 $$\lvert d_i \rvert$$로 가중치를 만듭니다.

$$
d'_i = \frac{\text{ReLU}(\tau - \lvert d_i \rvert)}{\sum_j \text{ReLU}(\tau - \lvert d_j \rvert)}, \qquad
\mathcal{L}_{\text{UDF}} = \frac{1}{3} \sum_i d'_i \sum_{c=1}^{3} (I_{i,c} - \hat I_{i,c})^2
$$

$$\tau = 10$$이면 윤곽에서 10픽셀 안쪽의 픽셀만 손실에 기여합니다. 경로 내부 깊숙한 곳과 멀리 떨어진 배경은 무시되므로, 색은 윤곽 근처에서 결정되고 모양은 윤곽을 맞추는 쪽으로 움직입니다. 경로가 여럿이면 $$d'_i$$를 경로들에 대해 평균합니다.

### Xing 손실

최적화 중 베지어 경로가 스스로 꼬이는 자기 교차가 자주 생깁니다. 저자들은 3차 베지어에서 **제어점 다각형이 교차하면 곡선도 교차하고, 그 역도 성립**한다는 관찰에서 제어점에 제약을 겁니다. 제어점 $$A, B, C, D$$에 대해 $$\angle ABC$$의 예각·둔각 여부를 $$D_1 = \mathbb{1}(\vec{AB} \times \vec{BC} > 0)$$, $$\vec{AB}$$와 $$\vec{CD}$$ 사이 각의 사인을 $$D_2$$로 두고

$$
\mathcal{L}_{\text{Xing}} = D_1 \,\text{ReLU}(-D_2) + (1 - D_1)\,\text{ReLU}(D_2)
$$

로 $$\theta \ge 180^\circ$$가 되도록 밀어 줍니다. 전체 손실은 $$\mathcal{L}_{\text{UDF}} + 0.01\,\mathcal{L}_{\text{Xing}}$$입니다.

### 데이터

평가용으로 NotoEmoji에서 모은 **Emoji**(134장, 240×240)와 폰트·아이콘·클립아트 **Pics**(153장)를 소개합니다. 모델이 없으므로 둘 다 평가에만 씁니다.

## 3. 코드로 보기: 색 평균 편향, 성분 초기화, Xing 손실

파란 배경 위 빨간 원을 목표로, 아직 윤곽이 안 맞는 경로의 색을 MSE와 UDF 가중 최소제곱으로 각각 풀어 보고, 배경을 그린 뒤의 오차 지도에서 다음 경로의 시작점을 찾고, 꼬인 제어점 다각형의 Xing 손실을 계산합니다.

```python
import numpy as np
H = W = 64
yy, xx = np.mgrid[0:H, 0:W]
# target: a red disk on a blue background
target = np.zeros((H, W, 3)); target[:] = (0.1, 0.2, 0.9)
disk = (xx - 32) ** 2 + (yy - 32) ** 2 <= 14 ** 2
target[disk] = (0.9, 0.2, 0.1)

# (1) a path whose shape does not yet match the disk (radius 18, off-centre): what colour does each loss pick?
path = (xx - 30) ** 2 + (yy - 34) ** 2 <= 18 ** 2
dist = np.sqrt((xx - 30) ** 2 + (yy - 34) ** 2) - 18        # signed distance to the path contour
def best_color(weights):                                    # weighted least squares = weighted mean colour
    w = weights[path]; return (w[:, None] * target[path]).sum(0) / w.sum()
tau = 10
udf_w = np.maximum(0, tau - np.abs(dist))
print("target disk colour       :", np.round(target[32, 32], 2))
print("MSE-optimal path colour  :", np.round(best_color(np.ones((H, W))), 2), " <- the area-weighted mean of what the path covers")
print("UDF-optimal path colour  :", np.round(best_color(udf_w), 2), " <- weighted toward the contour, which here lies outside the disk")
print(f"share of UDF weight within {tau} px of the contour: {udf_w[np.abs(dist) < tau].sum()/udf_w.sum():.2f}; MSE spreads it over all {path.sum()} path pixels")
print("-> both losses are area/contour averages; UDF only helps once the optimiser moves the contour onto the disk's edge,")
print("   so the paper's colour benefit comes from the optimisation trajectory, not from a static fit")

# (2) component-wise initialisation: after the blue background has been fitted, where should the next path start?
canvas = np.zeros_like(target); canvas[:] = (0.1, 0.2, 0.9)
err = np.abs(target - canvas).mean(-1)
mask = err > 0.1
labels = np.zeros_like(mask, dtype=int); n = 0
for i in range(H):
    for j in range(W):
        if mask[i, j] and labels[i, j] == 0:
            n += 1; stack = [(i, j)]; labels[i, j] = n
            while stack:
                a, b = stack.pop()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p, q = a + da, b + db
                    if 0 <= p < H and 0 <= q < W and mask[p, q] and labels[p, q] == 0:
                        labels[p, q] = n; stack.append((p, q))
sizes = [(labels == k).sum() for k in range(1, n + 1)]
k = int(np.argmax(sizes)) + 1
print(f"\nerror map after the background path: {n} component(s), largest has {max(sizes)} px, centre of mass = ({xx[labels == k].mean():.1f}, {yy[labels == k].mean():.1f}) -> the next path is initialised there")

# (3) Xing loss on a cubic Bezier: self-intersecting control polygon vs. a convex one
def xing_loss(A, B, C, D):
    AB, BC, CD = B - A, C - B, D - C
    cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
    D1 = 1.0 if cross(AB, BC) > 0 else 0.0
    D2 = cross(AB, CD) / (np.linalg.norm(AB) * np.linalg.norm(CD))
    return D1 * max(0, -D2) + (1 - D1) * max(0, D2)
convex = [np.array(p, float) for p in [(0, 0), (1, 2), (3, 2), (4, 0)]]
twisted = [np.array(p, float) for p in [(0, 0), (3, 2), (1, 2), (4, 0)]]
print(f"\nXing loss  convex control polygon: {xing_loss(*convex):.3f}   twisted (self-intersecting) polygon: {xing_loss(*twisted):.3f}")
```

```text
target disk colour       : [0.9 0.2 0.1]
MSE-optimal path colour  : [0.59 0.2  0.41]  <- the area-weighted mean of what the path covers
UDF-optimal path colour  : [0.37 0.2  0.63]  <- weighted toward the contour, which here lies outside the disk
share of UDF weight within 10 px of the contour: 1.00; MSE spreads it over all 1009 path pixels
-> both losses are area/contour averages; UDF only helps once the optimiser moves the contour onto the disk's edge,
   so the paper's colour benefit comes from the optimisation trajectory, not from a static fit

error map after the background path: 1 component(s), largest has 613 px, centre of mass = (32.0, 32.0) -> the next path is initialised there

Xing loss  convex control polygon: 0.000   twisted (self-intersecting) polygon: 0.923
```

- 경로가 원보다 크고 어긋나 있으면 MSE가 고른 색은 경로가 덮은 영역의 **면적 가중 평균**(보라색)입니다. 논문이 말하는 색 평균 편향입니다. 이 정적인 비교에서는 UDF도 윤곽 근처(여기서는 원 바깥의 파란 영역)의 평균을 고르므로 더 나은 색을 주지 않습니다. UDF의 이점은 최적화가 진행되며 **윤곽이 원의 경계로 이동할 때** 내부와 먼 배경이 손실에서 빠지는 데서 나옵니다. 즉 논문의 Figure 2는 최적화 궤적의 차이이지 한 시점의 차이가 아닙니다.
- 배경 경로를 그린 뒤 오차 지도의 유일한 연결 성분은 빨간 원이고, 무게중심 (32, 32)에 다음 경로가 놓입니다. 성분 단위 초기화가 "크고 아직 못 그린 것"을 고르는 방식입니다.
- 볼록한 제어점 다각형의 Xing 손실은 0, 꼬인 다각형은 0.92입니다. 손실은 꼬임이 있을 때만 켜지는 hinge 형태입니다.

## 4. 실험

**재구성 품질.** 경로 4개(이모지의 성분 수)와 20개(Im2Vec 기본값)로 비교하면 LIVE는 4개만으로 각 성분을 정확히 복원하고, DiffVG와 Im2Vec은 20개에서도 아티팩트가 남습니다. 정량적으로 Emoji(경로 8~64)와 Pics(32~256) 모두에서 LIVE의 MSE가 DiffVG보다 낮고, **경로 수가 적을수록 격차가 큽니다**. 경로가 충분해지면 성능이 포화합니다.

**레이어 표현.** 이모지와 단순 클립아트에서는 성분 하나가 경로 하나로 끊김 없이 학습됩니다. 사진 같은 복잡한 이미지에서는 완벽한 분리는 안 되지만 coarse-to-fine으로 전체 맥락을 먼저, 지역 세부를 나중에 채우며, 같은 경로 수에서 DiffVG·Neural Painting보다 지역 정보를 잘 복원합니다.

**보간.** VAE가 아닌데도 두 SVG의 대응 경로 제어점을 선형 보간하면 그럴듯한 중간 도형이 나옵니다. 경로 순서가 안정적이기 때문입니다. MNIST VAE의 잠재 보간 결과를 LIVE로 순차 벡터화해(이전 결과를 다음 초기값으로) 연속 애니메이션도 만들 수 있습니다.

**Ablation.** 원 초기화는 무작위 초기화보다 아티팩트가 훨씬 적고 볼록한 결과로 수렴합니다. Xing 손실을 빼면 같은 조건에서 자기 교차가 눈에 띄게 늘어납니다.

**한계.** 레이어별 최적화라 단일 패스 추론보다 느리고, 그라데이션 색이나 세그먼트 수 자동 선택은 없으며, 사진은 amodal 분할과의 결합이 남은 과제입니다.

## 5. 해석

**1. 데이터 없는 최적화가 학습 기반을 이긴 사례입니다.** 미분 가능한 렌더러(DiffVG)가 있으면 벡터화는 "학습"이 아니라 "이미지별 최적화"로 풀 수 있고, 도메인 일반화 문제가 사라집니다. 비용은 이미지당 최적화 시간입니다. [DeepSVG]({% post_url 2026-08-15-deepsvg %})가 잠재 공간의 보간·생성에 강하다면, LIVE는 충실한 재구성과 편집 가능성에 강합니다.

**2. 핵심은 손실보다 "순서"입니다.** 성분 단위 초기화와 점진적 추가가 레이어 위상을 만들고, UDF와 Xing은 그 과정이 어긋나지 않게 돕는 보조 장치입니다. 위 실험처럼 UDF를 정적으로 보면 특별할 게 없지만, "큰 것부터, 윤곽 중심으로"라는 절차 안에서 의미를 가집니다.

**3. 평가 기준이 MSE만은 아닙니다.** 같은 MSE라도 경로 수가 적고 성분과 1:1 대응하는 SVG가 편집에는 훨씬 유용합니다. 논문이 "simplicity and layer-wise topology"를 강조하고 사용자 연구를 붙인 이유입니다. 이후 텍스트-투-SVG(VectorFusion 등)가 LIVE의 경로 최적화를 기본 블록으로 쓰게 됩니다.

**4. 디퓨전 시대의 벡터 그래픽과 이어집니다.** [TextDiffuser]({% post_url 2026-08-01-textdiffuser %}), [GlyphDraw]({% post_url 2026-10-24-glyphdraw %})처럼 래스터로 글자를 그리는 모델의 출력을 LIVE로 벡터화하면 편집 가능한 로고·아이콘이 됩니다. 래스터 생성과 벡터 최적화의 분업입니다.

**5. 평가를 읽을 때 주의할 점.** 벤치마크 두 개는 저자들이 만든 것이고 비교 대상도 세 가지(DiffVG, Im2Vec, Neural Painting)뿐입니다. 사진에 대한 결과는 정성적이며, 처리 시간은 논문에 명시되지 않았습니다. 위 toy는 원판 모양의 목표와 경로만 고정해 다뤄 베지어 최적화의 동역학은 재현하지 않습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 문제 | 학습 기반 벡터화의 도메인 의존성과 중복 도형, 레이어 위상 부재 |
| 핵심 | 닫힌 베지어 경로를 점진적으로 추가하며 미분 가능 렌더러로 전체 최적화(모델·라벨 없음) |
| 초기화 | 오차 지도의 가장 큰 연결 성분 무게중심 + 원 초기화 |
| 손실 | UDF(윤곽 $$\tau = 10$$ 안쪽 픽셀만) + $$0.01 \times$$ Xing(제어점 각도 $$\ge 180^\circ$$) |
| 결과 | 이모지 4경로 완벽 복원, 적은 경로 수에서 DiffVG 대비 MSE 크게 감소, 경로 보간 가능 |
| 한계 | 이미지별 최적화 시간, 단색 채우기, 사진의 완전한 분리는 미해결 |

## References

- Ma et al., "Towards Layer-wise Image Vectorization," CVPR 2022. [arXiv:2206.04655](https://arxiv.org/abs/2206.04655)
- Li, Lukáč, Gharbi, Ragan-Kelley, "Differentiable Vector Graphics Rasterization for Editing and Learning," SIGGRAPH Asia 2020.
- Reddy, Gharbi, Lukáč, Mitra, "Im2Vec: Synthesizing Vector Graphics without Vector Supervision," CVPR 2021. [arXiv:2102.02798](https://arxiv.org/abs/2102.02798)
- Carlier, Danelljan, Alahi, Timofte, "DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation," NeurIPS 2020. [arXiv:2007.11301](https://arxiv.org/abs/2007.11301)
- Favreau, Lafarge, Bousseau, "Photo2ClipArt: Image Abstraction and Vectorization Using Layered Linear Gradients," SIGGRAPH Asia 2017.
