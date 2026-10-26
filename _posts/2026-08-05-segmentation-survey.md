---
title: "[Computer Vision] Image Segmentation Using Deep Learning: A Survey — FCN에서 DeepLab·Mask R-CNN·attention까지 10개 계보"
date: 2026-08-05 13:48:00 +0900
categories: [Computer Vision, Segmentation]
tags: [semantic-segmentation, instance-segmentation, survey, fcn, encoder-decoder, deeplab, mask-rcnn, miou]
description: Minaee 등(TPAMI 2021)의 딥러닝 영상 분할 서베이를 정리합니다. 2019년까지의 100여 개 방법을 FCN, 그래프 모델 결합, 인코더-디코더, 다중 스케일·피라미드, R-CNN 기반, 팽창 합성곱·DeepLab, 순환, attention, 생성·적대, 능동 윤곽의 10개 계보로 묶고, 20여 개 데이터셋, PA·mIoU·Dice 등 지표, PASCAL VOC·Cityscapes·COCO·ADE20k 성능표, 그리고 미래 과제를 다룹니다. 작은 실험으로 지표들의 관계와 수용 영역 계산을 확인합니다.
math: true
---

> [CNN 기초]({% post_url 2026-05-25-cnn-basics %})와 [객체 검출]({% post_url 2026-06-06-object-detection %})을 거쳐 픽셀 단위로 내려가면 분할입니다. 의미 분할은 모든 픽셀에 클래스를, 인스턴스 분할은 거기에 개체 구분까지 붙입니다. **Image Segmentation Using Deep Learning: A Survey**(Minaee, Boykov, Porikli, Plaza, Kehtarnavaz, Terzopoulos, TPAMI 2021)는 FCN(2015) 이후 2019년까지 쏟아진 100개 이상의 방법을 **기술적 기여**를 기준으로 10개 계보로 묶은 지도입니다. 이 글은 각 계보가 **어떤 문제를 풀려고 나왔는지**를 축으로 정리하고, 데이터셋·지표·성능표의 요점을 옮깁니다. 서베이의 시점이 Transformer 이전이라는 점은 마지막에 짚습니다.

## 1. 열 개의 계보

**1. 완전 합성곱 네트워크(FCN).** Long 등은 분류 CNN의 FC 층을 합성곱으로 바꿔 임의 크기 입력에 대해 분할 맵을 내게 했습니다. 깊은 층의 거친 맵을 얕은 층과 **스킵 연결**로 합쳐 세부를 복원합니다(FCN-8s). 문제는 느리고, 전역 문맥을 못 보고, 3D로 바로 안 간다는 점입니다. ParseNet은 전역 평균 특징을 각 위치에 붙여 문맥을 더했습니다.

**2. 그래프 모델 결합.** FCN의 출력은 경계가 흐릿하고 지역 일관성이 약합니다. DeepLab v1은 CRF를 후처리로, CRF-RNN은 CRF 평균장 추론을 **RNN으로 펼쳐** 종단간 학습하고, Piecewise·DPN은 쌍별 포텐셜을 네트워크로 모델링합니다.

**3. 인코더-디코더.** 다운샘플로 압축하고 업샘플로 복원합니다. Deconvolution Network와 SegNet(풀링 인덱스를 저장해 비선형 업샘플), 고해상도 표현을 끝까지 유지하는 HRNet이 일반 분할 쪽이고, 의료 영상에서는 **U-Net**(대칭 스킵 연결, 적은 데이터로 학습), V-Net(3D, Dice 손실), 그리고 다수의 변형이 있습니다.

**4. 다중 스케일·피라미드.** 물체 크기가 다양하므로 여러 해상도의 특징을 합칩니다. FPN(검출에서 왔지만 분할에도), **PSPNet**(피라미드 풀링 모듈로 전역·부분 문맥을 네 스케일로 풀링해 결합), DM-Net, APC-Net, MSCI 등.

**5. R-CNN 기반(인스턴스 분할).** Faster R-CNN의 제안 영역마다 마스크를 예측하는 **Mask R-CNN**(RoIAlign으로 정렬 오차 제거)이 중심이고, PANet(경로 집계), MaskLab, TensorMask(밀집 슬라이딩 윈도우), 실시간 YOLACT 등이 뒤따릅니다.

**6. 팽창 합성곱과 DeepLab.** 풀링 없이 수용 영역을 키우는 **팽창(atrous) 합성곱**이 핵심입니다. DeepLab v2는 여러 팽창률의 병렬 가지 **ASPP**로 다중 스케일을, v3는 배치 정규화와 이미지 수준 특징을 더하고, **v3+**는 인코더-디코더 구조와 Xception 백본, 깊이별 분리 팽창 합성곱을 합쳐 PASCAL VOC 89.0%에 이릅니다. DenseASPP, DUC-HDC도 이 계보입니다.

**7. 순환 네트워크.** 픽셀 간 장거리 의존을 RNN·LSTM으로 모델링합니다(ReSeg, 2D LSTM, Graph LSTM, DA-RNN). 자연스럽지만 느리고 이후 attention으로 대체됩니다.

**8. Attention.** 어디를 볼지 학습합니다. 다중 스케일 입력에 소프트 attention(Chen 등), 역 attention(RAN), 쌍별 attention(PSANet), 위치·채널 이중 attention(**DANet**), 교차 attention으로 비용을 줄인 CCNet, EMANet, 객체 문맥 표현 **OCR**. 2019~2020년 Cityscapes 상위권이 모두 여기 속합니다.

**9. 생성·적대 학습.** 판별기가 분할 맵의 사실성을 판단해 공간적 일관성을 강제하고(Luc 등), 반지도 학습으로 라벨 없는 이미지를 활용합니다.

**10. 능동 윤곽 결합.** 고전적 능동 윤곽 모델(ACM)의 에너지를 손실로 쓰거나(ACWE 손실), 네트워크가 윤곽의 변형을 예측하게 합니다(Deep Active Contours, DARNet).

그 밖에 EncNet(문맥 인코딩), RefineNet(다중 경로 정제), Gated-SCNN(모양 흐름), 포인트 클라우드 분할(PointNet 계열), 그리고 **파놉틱 분할**(의미+인스턴스, Panoptic FPN, UPSNet, Panoptic-DeepLab)이 "기타"에 묶입니다.

## 2. 데이터셋

2D에 PASCAL VOC(21클래스), PASCAL Context(400+), COCO(80 물체 + stuff), **Cityscapes**(50개 도시 주행 장면, 5,000 정밀 + 20,000 거친 라벨), ADE20k(150클래스), SiftFlow, Stanford background, BSD, YouTube-Objects, KITTI, 2.5D(RGB-D)에 NYU-D v2, SUN RGB-D, SUN-3D, ScanNet, 3D에 Stanford 2D-3D-S, ShapeNet Core, Sydney Urban Objects가 정리됩니다.

## 3. 지표

$$p_{ij}$$를 클래스 $$i$$ 픽셀이 $$j$$로 예측된 수라 할 때,

$$
\text{PA} = \frac{\sum_i p_{ii}}{\sum_{i,j} p_{ij}}, \quad
\text{MPA} = \frac{1}{K+1}\sum_i \frac{p_{ii}}{\sum_j p_{ij}}, \quad
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}, \quad
\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}
$$

mIoU는 클래스별 IoU의 평균이고 현대 분할의 표준 지표입니다. Dice는 의료 영상에서 흔하고, 이진 분할에서 **F1과 같습니다**($$2TP/(2TP+FP+FN)$$). 정확도뿐 아니라 속도와 메모리도 평가해야 하지만 대부분의 연구가 정확도만 보고한다는 점을 서베이가 지적합니다.

## 4. 코드로 보기: 지표들의 관계와 수용 영역

배경·큰 물체·작은 물체 3클래스의 합성 정답에 무작위 오류와 작은 클래스에 대한 체계적 오류를 섞은 예측을 만들어 PA, MPA, mIoU, Dice를 계산하고, 팽창·풀링에 따른 수용 영역을 셉니다.

```python
import numpy as np
rng = np.random.default_rng(0)
H = W = 32; K = 3
gt = np.zeros((H, W), int); gt[8:24, 8:24] = 1; gt[20:30, 2:12] = 2
pred = gt.copy(); flip = rng.random((H, W)) < 0.12; pred[flip] = rng.integers(0, K, flip.sum())     # a noisy prediction
pred[22:28, 4:10] = 1                                                                                 # plus a systematic error on the small class
def metrics(gt, pred):
    pa = (gt == pred).mean()
    ious, dices, accs = [], [], []
    for k in range(K):
        g, p = gt == k, pred == k
        inter, union = (g & p).sum(), (g | p).sum()
        ious.append(inter / union); dices.append(2 * inter / (g.sum() + p.sum())); accs.append(inter / g.sum())
    return pa, np.array(ious), np.array(dices), np.array(accs)
pa, ious, dices, accs = metrics(gt, pred)
print(f"pixel accuracy       {pa:.3f}")
print(f"mean pixel accuracy  {accs.mean():.3f}   per class {np.round(accs, 3)}")
print(f"mean IoU             {ious.mean():.3f}   per class {np.round(ious, 3)}")
print(f"mean Dice (= F1)     {dices.mean():.3f}   per class {np.round(dices, 3)}")
print(f"Dice = 2 IoU / (1 + IoU) holds per class: {np.allclose(dices, 2 * ious / (1 + ious))}")
print(f"\nclass pixel shares: {np.round([(gt == k).mean() for k in range(K)], 3)} -> pixel accuracy is dominated by the background, mIoU is not")
def rf(layers):
    r, j = 1, 1
    for k, s, d in layers: r += d * (k - 1) * j; j *= s
    return r
print()
print(f"receptive field: 8 plain 3x3 convs {rf([(3,1,1)]*8)} | 4 convs with dilation 1,2,4,8 {rf([(3,1,1),(3,1,2),(3,1,4),(3,1,8)])} | 4 convs with 3 stride-2 pools between {rf([(3,1,1),(2,2,1),(3,1,1),(2,2,1),(3,1,1),(2,2,1),(3,1,1)])}")
```

```text
pixel accuracy       0.899
mean pixel accuracy  0.829   per class [0.925 0.942 0.62 ]
mean IoU             0.712   per class [0.915 0.734 0.488]
mean Dice (= F1)     0.819   per class [0.955 0.846 0.656]
Dice = 2 IoU / (1 + IoU) holds per class: True

class pixel shares: [0.668 0.234 0.098] -> pixel accuracy is dominated by the background, mIoU is not

receptive field: 8 plain 3x3 convs 17 | 4 convs with dilation 1,2,4,8 31 | 4 convs with 3 stride-2 pools between 38
```

- PA는 0.90으로 높지만 mIoU는 0.71입니다. 픽셀의 67%가 배경이라 배경만 잘 맞춰도 PA가 오르는 반면, mIoU는 클래스를 동등하게 평균해 작은 클래스(IoU 0.49)의 실패를 드러냅니다. 분할 벤치마크가 PA 대신 mIoU를 쓰는 이유입니다. Dice = 2·IoU/(1+IoU)의 관계가 클래스마다 정확히 성립하고, Dice가 IoU보다 항상 큽니다.
- 3×3 합성곱 8층의 수용 영역은 17픽셀인데, 팽창률 1·2·4·8의 4층은 31픽셀, stride-2 풀링 3번을 끼운 4층은 38픽셀입니다. 풀링은 해상도를 잃고 수용 영역을 키우며, 팽창은 해상도를 유지하면서 키웁니다. DeepLab 계보가 팽창 합성곱을 택한 산술입니다.

## 5. 성능

**PASCAL VOC 테스트 mIoU.** FCN 62.2 → DeepLab-CRF 79.7 → PSPNet 85.4 → DeepLabv3 85.7 → DeepLabv3+ 87.8 → DeepLabv3+(사전학습) 89.0 → EfficientNet+NAS-FPN 90.5. 5년 사이 28%p 올랐습니다.

**Cityscapes 테스트 mIoU.** FCN-8s 65.3 → DeepLabv2 70.4 → PSPNet 78.4(서베이 표에는 85.4로 적혀 있으나 원 논문은 78.4) → DANet 81.5 → OCR 82.4 → GS-CNN 82.8 → HRNetV2+OCR 83.7 → Hierarchical MSA 85.1. 서베이는 FCN 대비 약 23% 상대 개선이라고 적었지만, FCN-8s 65.3을 기준으로 계산하면 약 30%(19.8%p)입니다.

**COCO stuff**(mIoU 최고 약 40, OCR/HRNetV2-W48 40.5)와 **ADE20k**(AC-Net 45.9)는 훨씬 어렵습니다. **인스턴스 분할**(COCO test-dev AP)은 Mask R-CNN 37.9, CenterMask 38.3이고 YOLACT-550은 29.8 AP에 33.5 FPS로 실시간입니다. **파놉틱**(COCO val PQ)은 UPSNet 42.5, OCFusion(ResNeXt-101) 45.7입니다. RGB-D(NYUD-v2, SUN-RGBD)도 표로 정리됩니다.

## 6. 과제와 방향

**더 어려운 데이터셋**(더 많은 물체·겹침·3D·시퀀스), **해석 가능한 모델**(왜 그 픽셀을 그렇게 분류했나), **약지도·비지도 학습**(픽셀 라벨은 비싸므로 바운딩 박스·이미지 라벨·자기지도 활용), **실시간 모델**(자율주행·로봇에는 25 FPS 이상이 필요한데 대부분 못 미침, 팽창 합성곱이 한 해법), **메모리 효율**(모바일 배포를 위한 압축·증류), **3D 포인트 클라우드**(불규칙·비정렬 데이터의 표현), 그리고 응용 시나리오(의료, 주행, 원격 탐사)의 특수 요구입니다.

## 7. 해석

**1. 계보의 축은 "문맥을 어떻게 얻는가"입니다.** FCN의 약점(지역적, 흐린 경계)을 CRF(2), 피라미드(4), 팽창(6), RNN(7), attention(8)이 각각 다른 방식으로 메웁니다. 수용 영역을 키우되 해상도를 잃지 않는 것이 공통 목표이고, 위 toy의 산술이 그 트레이드오프의 뼈대입니다. 이 관점에서 보면 이후의 SegFormer·Mask2Former는 attention 계보가 백본까지 올라간 것입니다.

**2. 인코더-디코더와 U-Net의 영향이 분할을 넘어섭니다.** 대칭 스킵 연결은 [잠재 디퓨전]({% post_url 2026-07-03-latent-diffusion-models %})의 디노이저 백본으로, [MWCNN]({% post_url 2026-07-23-mwcnn %})처럼 웨이블릿을 풀링 대신 쓰는 변형으로, 의료 영상의 거의 모든 구조로 퍼졌습니다. 서베이가 의료 인코더-디코더를 별도 절로 둔 이유입니다.

**3. 지표 선택이 결과를 바꿉니다.** PA 대 mIoU의 차이는 클래스 불균형의 문제이고, Dice와 F1의 동일성은 의료와 일반 비전이 같은 것을 다른 이름으로 부른다는 뜻입니다. 속도와 메모리를 보고하지 않는 관행에 대한 서베이의 지적은 [EdgeViTs]({% post_url 2026-08-25-edgevits %})류의 실측 지연 시간 보고가 왜 필요한지와 같은 맥락입니다.

**4. 성능표의 수치는 조건을 봐야 합니다.** 사전학습 데이터(COCO, JFT-300M), 백본 크기, 테스트 시 증강(다중 스케일·뒤집기)이 mIoU를 수 %p 바꿉니다. 서베이도 "표준 데이터셋과 지표를 쓰지 않는 논문이 있어 비교가 어렵다"고 적습니다.

**5. 평가를 읽을 때 주의할 점.** 2019년까지의 방법이라 Transformer 기반 분할(SETR, Segmenter, SegFormer, Mask2Former, SAM)은 없습니다. 성능표는 원 논문 수치를 옮긴 것이라 조건이 통일되지 않았고, 위에 적은 PSPNet Cityscapes 수치처럼 오기도 있습니다. 서베이는 방법 간 우열보다 **분류와 맥락**에 가치가 있습니다. 위 toy의 지표 계산은 3클래스 합성 맵이라 실제 벤치마크의 규모와 다릅니다.

## 8. 정리

| 항목 | 내용 |
|---|---|
| 범위 | 2015~2019 딥러닝 분할 100여 편, 10개 계보, 20여 데이터셋 |
| 계보 | FCN / 그래프 모델 / 인코더-디코더(U-Net) / 피라미드(PSPNet) / R-CNN(Mask R-CNN) / 팽창·DeepLab / RNN / attention(DANet, OCR) / 생성·적대 / 능동 윤곽 |
| 지표 | PA, MPA, IoU·mIoU(표준), Dice(= 이진 F1), 정밀도·재현율 |
| 성능 | VOC 62.2→90.5 mIoU, Cityscapes 65.3→85.1, COCO stuff ~40, ADE20k ~46, 인스턴스 AP ~38, 파놉틱 PQ ~46 |
| 과제 | 어려운 데이터셋, 해석성, 약지도, 실시간, 메모리, 3D |
| 빠진 것 | Transformer 기반 분할(2020 이후) |

## References

- Minaee et al., "Image Segmentation Using Deep Learning: A Survey," TPAMI 2021. [arXiv:2001.05566](https://arxiv.org/abs/2001.05566)
- Long, Shelhamer, Darrell, "Fully Convolutional Networks for Semantic Segmentation," CVPR 2015. [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)
- Ronneberger, Fischer, Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (DeepLabv3+), ECCV 2018. [arXiv:1802.02611](https://arxiv.org/abs/1802.02611)
- Zhao et al., "Pyramid Scene Parsing Network," CVPR 2017. [arXiv:1612.01105](https://arxiv.org/abs/1612.01105)
- He, Gkioxari, Dollár, Girshick, "Mask R-CNN," ICCV 2017. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- Fu et al., "Dual Attention Network for Scene Segmentation," CVPR 2019. [arXiv:1809.02983](https://arxiv.org/abs/1809.02983)
