---
title: "[Segmentation] PatchCore: 정상 데이터만으로 결함 위치를 찾는 이상 탐지"
date: 2026-07-30 22:58:00 +0900
categories: [Neural Network, Segmentation]
tags: [anomaly-detection, patchcore, memory-bank, coreset, nearest-neighbor, mvtec-ad, industrial-inspection, segmentation]
description: 정상 이미지의 패치 특징을 메모리 뱅크에 저장하고 최근접 거리로 결함을 찾는 PatchCore의 특징 추출, greedy coreset subsampling, 이미지·픽셀 단위 이상 점수 계산을 코드와 함께 정리합니다.
math: true
---

> Segmentation 카테고리의 첫 글입니다. 공장의 외관 검사에서는 결함 사진은 거의 없고 **정상 제품 사진만** 많습니다. **PatchCore**(*Towards Total Recall in Industrial Anomaly Detection*, Roth et al., CVPR 2022)는 정상 이미지의 **패치 특징을 메모리에 저장**해 두고, 새 이미지의 각 패치가 그 메모리와 얼마나 다른지로 **이미지 단위 판정과 픽셀 단위 결함 위치(segmentation)**를 동시에 얻습니다.

## 1. Anomaly Detection이란

**주어진 데이터에서 예상하지 못한 패턴을 찾는 과정**입니다. 비전 검사 분야의 산업용 이상 탐지는 **제품 외관에 생긴 결함**(긁힘, 찍힘, 오염, 누락 등)을 찾는 것이 목적입니다.

### 왜 일반 분류로 풀기 어렵나

- 결함 사진은 **매우 드물고**, 결함 종류도 예측할 수 없음
- 학습 때 보지 못한 새로운 결함도 잡아야 함

그래서 **정상 데이터만으로 학습**하고, 정상과 다른 것을 이상으로 판단하는 **semi-supervised(또는 one-class)** 방식으로 풉니다. PatchCore도 학습에 정상 이미지만 씁니다.

## 2. 핵심 아이디어: 특징점 기반 이상 감지

1. 사전학습된 CNN으로 **이미지의 각 위치(패치)에서 지역 특징**을 뽑는다.
2. 정상 이미지들의 패치 특징을 모아 **메모리 뱅크**를 만든다.
3. 테스트 이미지의 각 패치를 메모리 뱅크에서 **가장 가까운 정상 패치**와 비교한다.
4. 가까운 정상 패치가 없을수록(거리가 멀수록) **이상 점수가 높다.**
5. 패치별 점수를 모으면 **결함 위치 지도(anomaly map)**가 되고, 그중 최댓값이 **이미지 전체의 이상 점수**가 된다.

학습이라고 할 것이 거의 없습니다. 특징 추출기는 ImageNet으로 사전학습된 것을 **그대로** 쓰고, 정상 특징을 저장하기만 합니다.

## 3. Locally Aware Patch Features

### 어느 층의 특징을 쓸까

ImageNet으로 사전학습된 **WideResNet-50**에서

- **너무 얕은 층**: 일반적이지만 문맥이 부족함
- **너무 깊은 층**: ImageNet 분류에 특화되어 산업 결함과 거리가 있고 해상도가 낮음

그래서 **중간 층(layer 2, layer 3)**의 특징을 씁니다. Layer 3 특징은 layer 2 해상도로 upsampling해서 이어 붙입니다.

### 지역 문맥 집계

각 위치의 특징을 그대로 쓰지 않고, **주변 3×3 이웃의 특징을 평균 풀링으로 모아** 패치 특징으로 씁니다. 해상도는 유지하면서 receptive field를 넓혀, 작은 위치 변화에 강건해집니다.

$$
\phi_{i,j} = \text{AvgPool}_{3\times3}\Big(\big[f^{(2)},\ \text{Up}(f^{(3)})\big]\Big)_{i,j}
$$

## 4. Memory Bank와 Coreset Subsampling

모든 정상 이미지의 모든 패치를 저장하면 메모리 뱅크가 너무 커서 최근접 탐색이 느립니다. 이미지 200장, 이미지당 28×28 패치라면 약 15만 개입니다.

PatchCore는 **Greedy Coreset Subsampling**으로 전체 분포를 잘 대표하는 **부분집합(coreset)**만 남깁니다. 아이디어는 "이미 고른 점들로부터 **가장 먼 점**을 하나씩 추가"하는 것입니다.

$$
m^* = \arg\max_{m \in \mathcal{M} \setminus \mathcal{M}_C}\ \min_{n \in \mathcal{M}_C}\big\lVert\psi(m) - \psi(n)\big\rVert_2
$$

($$\psi$$: 계산을 빠르게 하기 위한 무작위 선형 투영)

이렇게 고르면 전체 특징 공간을 **고르게 덮는** 점들이 남습니다. 메모리 뱅크를 **1~10%**로 줄여도 성능이 거의 유지되고, 추론은 훨씬 빨라집니다. 단순 무작위 샘플링은 드문 정상 패턴을 놓치기 쉽습니다.

## 5. 이상 점수 계산

### 패치 단위

테스트 이미지의 각 패치 특징 $$m^{test}$$에 대해 메모리 뱅크에서 **최근접 이웃(Nearest Neighbour Search)**까지의 거리를 구합니다.

$$
s(m^{test}) = \min_{m \in \mathcal{M}_C}\big\lVert m^{test} - m\big\rVert_2
$$

메모리 뱅크에 저장된 정상 패치와의 **최소 거리**가 멀수록, 그 패치는 정상 범위를 벗어난 것입니다.

### 이미지 단위

모든 패치 점수 중 **최댓값**을 이미지 점수로 씁니다. 결함은 작은 영역에만 있을 수 있으므로 평균보다 최댓값이 맞습니다. 논문은 최근접 정상 패치 주변이 얼마나 촘촘한지에 따라 점수를 약간 보정(re-weighting)합니다. 주변에 다른 정상 패치가 드물면, 즉 그 정상 패치 자체가 이미 드문 정상 패턴이면 점수를 높입니다.

### 픽셀 단위 (Segmentation)

패치 점수 지도를 원래 이미지 크기로 **upsampling**하고 **가우시안 smoothing**하면, 결함이 있는 위치를 보여 주는 **anomaly map**이 됩니다. 임계값을 넘는 픽셀이 결함 영역입니다.

```python
import torch
import torch.nn.functional as F

def greedy_coreset(features, ratio=0.1, proj_dim=128):
    """features: (N, D) 정상 패치 특징 → coreset 인덱스"""
    N = features.shape[0]
    proj = features @ torch.randn(features.shape[1], proj_dim) / proj_dim ** 0.5   # 무작위 투영
    selected = [torch.randint(N, (1,)).item()]
    min_dist = torch.cdist(proj, proj[selected]).squeeze(1)
    for _ in range(int(N * ratio) - 1):
        idx = torch.argmax(min_dist).item()                     # 지금까지 고른 점에서 가장 먼 점
        selected.append(idx)
        min_dist = torch.minimum(min_dist, torch.cdist(proj, proj[idx:idx + 1]).squeeze(1))
    return torch.tensor(selected)

def anomaly_map(test_feats, memory, H, W, out_size):
    """test_feats: (H*W, D) 한 이미지의 패치 특징"""
    d = torch.cdist(test_feats, memory).min(dim=1).values      # 패치별 최근접 정상 패치까지 거리
    amap = d.view(1, 1, H, W)
    amap = F.interpolate(amap, size=out_size, mode="bilinear", align_corners=False)
    return amap.squeeze(), d.max()                               # 픽셀 지도, 이미지 점수

# 장난감 예: 정상 이미지 20장 × 28x28 패치(64차원 특징)로 메모리를 만들고,
# 테스트 이미지의 (10, 5) 위치 패치 하나에만 이상을 심어 본다
torch.manual_seed(0)
normal = torch.randn(20 * 28 * 28, 64)
memory = normal[greedy_coreset(normal, ratio=0.01)]           # 15,680개 → 156개만 저장
test = torch.randn(28 * 28, 64)
test[10 * 28 + 5] += 6.0
amap, score = anomaly_map(test, memory, 28, 28, out_size=(224, 224))
d = torch.cdist(test, memory).min(dim=1).values
print(memory.shape, divmod(int(d.argmax()), 28))              # torch.Size([156, 64]) (10, 5)
print(round(float(score), 2), round(float(d.median()), 2))    # 48.59 9.93 → 이상 패치만 크게 튐
```

메모리 뱅크를 1%(156개)로 줄였는데도, 이상을 심은 위치가 정확히 가장 큰 점수를 받습니다. 실제 사용에서는 위 `normal`, `test` 자리에 WideResNet-50 중간 층에서 뽑은 패치 특징이 들어갑니다. 메모리 뱅크가 커지면 `torch.cdist` 대신 FAISS 같은 근사 최근접 탐색 라이브러리를 쓰는 것이 일반적입니다.

## 6. 실험: MVTec AD

산업 이상 탐지의 표준 벤치마크인 **MVTec AD**(병, 케이블, 나사, 카펫 등 15개 카테고리, 학습은 정상 이미지만)에서 평가했습니다.

- **이미지 단위 AUROC 약 99.6%**로, 당시 최고 성능을 기록했습니다(제목의 "total recall"에 가까운 결과).
- **픽셀 단위 segmentation**에서도 결함 위치를 정확히 찾아, 기존 방법보다 좋거나 비슷했습니다.
- Coreset 비율을 1%까지 줄여도 성능이 크게 떨어지지 않아, **메모리와 속도 효율**이 좋았습니다.
- 정상 학습 이미지가 적은 **few-shot** 설정에서도 강했습니다.

## 7. 정리

| 단계 | 방법 | 목적 |
|---|---|---|
| 특징 추출 | 사전학습 WideResNet-50의 중간 층 + 3×3 이웃 집계 | 일반적이면서 지역 문맥을 담은 패치 특징 |
| 메모리 뱅크 | 정상 이미지 패치 특징 저장 | 정상의 기준 |
| Coreset | Greedy coreset subsampling (1~10%) | 메모리·속도 절감, 분포 보존 |
| 패치 점수 | 최근접 정상 패치까지 거리 | 국소적인 이상 정도 |
| 이미지 판정 | 패치 점수의 최댓값 | 결함 유무 |
| 위치 지도 | Upsampling + Gaussian smoothing | **픽셀 단위 결함 영역** |

PatchCore는 별도의 학습 없이 강한 성능을 내 산업 이상 탐지의 표준 baseline이 됐습니다.

## References

- Roth et al., "Towards Total Recall in Industrial Anomaly Detection (PatchCore)," CVPR 2022. [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)
- [anomalib](https://github.com/openvinotoolkit/anomalib) (PatchCore 등 이상 탐지 구현 모음)
- Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019.
- Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization," ICPR 2021.
- Sener & Savarese, "Active Learning for Convolutional Neural Networks: A Core-Set Approach," ICLR 2018.
