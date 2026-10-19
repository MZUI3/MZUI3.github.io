---
title: "[AI 기초 9] Object Detection 기초: IoU, NMS부터 R-CNN 계열, YOLO v1~v7까지"
date: 2026-06-07 21:07:00 +0900
categories: [Neural Network, Object Detection]
tags: [object-detection, iou, nms, rcnn, faster-rcnn, yolo, ssd, retinanet, focal-loss, fpn, map]
description: Object Detection의 기본 개념(IoU, NMS, mAP)과 2-stage(R-CNN 계열), 1-stage(YOLO, SSD, RetinaNet) 모델이 어떤 문제를 풀며 발전했는지, YOLO v1~v7의 변화까지 코드와 함께 정리합니다.
math: true
---

> 분류(Classification)가 "이 사진에 고양이가 있다"를 맞히는 문제라면, **Object Detection**은 "고양이가 **어디에** 있고, 개는 또 어디에 있다"까지 찾는 문제입니다. 이 글에서는 Detection의 기본 개념(IoU, NMS, mAP)을 먼저 정리하고, 2-stage(R-CNN 계열)와 1-stage(YOLO, SSD, RetinaNet) 모델의 발전 과정을 따라갑니다.

## 1. Object Detection이란

$$
\text{Object Detection} = \text{Localization} + \text{Classification}
$$

- **Localization**: 물체의 위치를 **Bounding Box** $$(x, y, w, h)$$로 찾기
- **Classification**: 찾은 물체의 클래스 분류하기

### 2-stage vs 1-stage

| | 2-stage | 1-stage |
|---|---|---|
| 방식 | ① 물체가 있을 만한 영역 제안(Region Proposal) → ② 영역별 분류 | 위치와 클래스를 **한 번에** 예측 |
| 장점 | 정확도 높음 | **속도 빠름** (실시간 가능) |
| 단점 | 느림, 초기 모델은 end-to-end 학습 불가 | 작은 물체·밀집 물체에 약함 |
| 대표 모델 | R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN, Cascade R-CNN | YOLO, SSD, RetinaNet |

## 2. 먼저 알아야 할 개념

### IoU (Intersection over Union)

예측 박스 $$B_p$$와 정답 박스 $$B_{gt}$$가 얼마나 겹치는지 나타냅니다(Jaccard 계수).

$$
\text{IoU} = \frac{\lvert B_p \cap B_{gt}\rvert}{\lvert B_p \cup B_{gt}\rvert}
$$

0이면 전혀 안 겹치고 1이면 완전히 일치합니다. 보통 IoU 0.5 이상이면 맞게 찾았다고 봅니다.

```python
def iou(a, b):
    """a, b: (x1, y1, x2, y2)"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)

print(iou((0, 0, 2, 2), (1, 1, 3, 3)))   # 1 / 7 ≈ 0.143
```

### IoU의 변형: GIoU, DIoU, CIoU

IoU를 손실 함수로 쓰면 문제가 있습니다. 두 박스가 **겹치지 않으면 IoU가 항상 0**이라, 얼마나 멀리 떨어져 있는지 알 수 없고 기울기도 0입니다.

| 변형 | 추가로 고려하는 것 | 효과 |
|---|---|---|
| **GIoU** | 두 박스를 감싸는 가장 작은 박스 $$C$$ | 겹치지 않아도 값과 기울기가 있음 |
| **DIoU** | 두 박스 **중심점 사이 거리** | 중심점 penalty로 더 빨리 수렴 |
| **CIoU** | 중심점 거리 + **종횡비(aspect ratio)** | 모양까지 맞춤 |

$$
\text{GIoU} = \text{IoU} - \frac{\lvert C \setminus (B_p \cup B_{gt})\rvert}{\lvert C\rvert},\qquad
\text{DIoU} = \text{IoU} - \frac{\rho^2(b_p, b_{gt})}{c^2}
$$

($$\rho$$: 두 중심점 사이의 유클리드 거리, $$c$$: 감싸는 박스의 대각선 길이)

### NMS (Non-Maximum Suppression)

모델은 같은 물체 주변에 박스를 여러 개 예측합니다. NMS는 **같은 물체를 가리키는 중복 박스를 지웁니다.**

1. Confidence score가 threshold보다 낮은 박스를 제거
2. 남은 박스를 confidence 내림차순으로 정렬
3. 가장 높은 박스를 선택하고, 이 박스와 **IoU가 threshold(예: 0.5) 이상인 박스를 모두 제거** (같은 물체로 판단)
4. 남은 박스로 3을 반복

```python
import torch
from torchvision.ops import nms

boxes = torch.tensor([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]], dtype=torch.float)
scores = torch.tensor([0.9, 0.8, 0.7])
print(nms(boxes, scores, iou_threshold=0.5))   # tensor([0, 2]): 두 번째 박스는 첫 번째와 겹쳐서 제거
```

첫 번째와 두 번째 박스의 IoU는 약 0.82(`torchvision.ops.box_iou`로 확인)라 threshold 0.5를 넘어 제거됩니다. (torchvision 0.28에서 실행)

> NMS의 IoU threshold는 트레이드오프입니다. 너무 낮으면 **서로 겹쳐 서 있는 다른 물체**(군중 속 사람들)까지 지워지고, 너무 높으면 중복 박스가 남습니다. 이를 완화하려고 겹친 박스를 지우는 대신 점수를 깎는 **Soft-NMS**(Bodla et al., 2017)도 쓰입니다.
{: .prompt-info }

### 평가 지표: Precision, Recall, mAP

- **Precision**: 모델이 "물체다"라고 한 것 중 실제로 맞은 비율 → **얼마나 정확한가**
- **Recall**: 실제 물체 중 모델이 찾아낸 비율 → **얼마나 빠짐없이 찾는가**

예를 들어 사진에 사람이 4명 있는데 모델이 박스 6개를 그렸고 그중 3개가 사람이라면

$$
\text{Recall} = \frac{3}{4} = 75\%,\qquad \text{Precision} = \frac{3}{6} = 50\%
$$

- **AP (Average Precision)**: confidence threshold를 바꿔 가며 그린 **Precision–Recall 곡선 아래 면적**
- **mAP**: 모든 클래스 AP의 평균. COCO에서는 IoU threshold를 0.5~0.95까지 바꿔 가며 평균한 mAP를 씁니다.

---

## 3. 2-stage: R-CNN 계열

### R-CNN (2014)

CNN을 Detection에 처음 적용한 모델입니다.

1. **Selective Search**로 물체가 있을 만한 영역(RoI) 약 2000개 추출
2. 각 영역을 잘라 같은 크기로 변형(warp)
3. **영역마다 CNN**으로 특징 추출
4. SVM으로 분류, 회귀로 박스 보정

영역 2000개마다 CNN을 돌리니 **이미지 한 장에 수십 초**가 걸렸습니다.

### Fast R-CNN (2015)

**CNN을 이미지 전체에 한 번만** 돌려 feature map을 얻고, Selective Search로 찾은 영역을 feature map 위에 투영(RoI Projection)해서 씁니다.

투영된 영역은 크기가 제각각인데, FC layer에는 고정된 크기가 들어가야 합니다. 그래서 **RoI Pooling**을 씁니다.

- 투영된 영역을 고정된 격자(예: 7×7)로 나누고
- 격자 칸마다 **Max Pooling**
- → 입력 크기와 상관없이 항상 같은 크기 출력

이는 SPPNet의 Spatial Pyramid Pooling에서 피라미드 레벨이 1개인 특수한 경우입니다. 분류와 박스 회귀도 하나의 네트워크에서 함께 학습합니다.

<details markdown="1">
<summary>더 알아보기: RoI Pooling의 반올림 문제와 RoIAlign</summary>

RoI Pooling은 영역 좌표를 feature map 격자에 맞추려고 **두 번 반올림**합니다(영역 경계를 정수 칸으로, 다시 7×7 칸 경계를 정수로). stride 16인 feature map에서 반 칸 어긋나면 원본 이미지에서는 8픽셀이 어긋납니다. 분류에는 큰 문제가 없지만 픽셀 단위 마스크를 그려야 하는 segmentation에는 치명적입니다. 그래서 Mask R-CNN(He et al., 2017)은 반올림 없이 **bilinear interpolation**으로 정확한 위치의 값을 뽑는 **RoIAlign**을 제안했습니다. torchvision에서는 `torchvision.ops.roi_align`으로 쓸 수 있습니다.
</details>

### Faster R-CNN (2015)

남은 병목은 CPU에서 도는 **Selective Search**였습니다. Faster R-CNN은 이를 신경망인 **RPN(Region Proposal Network)**으로 대체해 전체를 GPU에서 end-to-end로 학습합니다.

**Anchor**: feature map의 각 위치마다 크기와 비율이 다른 **기준 박스** $$k$$개(예: 3 scale × 3 ratio = 9개)를 미리 깔아 둡니다. RPN은 각 anchor에 대해

- **objectness**: 이 anchor에 물체가 있는가 (정답 박스와 IoU > 0.7이면 positive, < 0.3이면 negative)
- **박스 offset**: anchor를 정답 박스에 맞추려면 얼마나 이동·확대해야 하는가 (Smooth L1 loss)

를 예측합니다. RPN 결과에 NMS를 적용해 영역을 고른 뒤, 나머지는 Fast R-CNN과 같습니다.

---

## 4. 1-stage: YOLO v1

**YOLO (You Only Look Once, 2016)**는 Detection을 **하나의 회귀 문제**로 바꿨습니다. 이미지를 한 번만 보고 모든 박스와 클래스를 동시에 예측합니다.

### Unified Detection

1. 입력 이미지(448×448)를 **S×S 격자**로 나눔 (S = 7)
2. 각 격자 칸(grid cell)은 **B개의 bounding box**(B = 2)를 예측. 각 박스는 $$(x, y, w, h, \text{confidence})$$
3. 각 격자 칸은 **C개 클래스의 조건부 확률**(C = 20, PASCAL VOC)을 예측

따라서 출력 텐서는

$$
S \times S \times (B \times 5 + C) = 7 \times 7 \times 30
$$

이고, 박스는 총 $$7 \times 7 \times 2 = 98$$개입니다.

### Confidence Score와 클래스 점수

- **Box confidence**: 물체가 있을 확률 × 예측 박스와 정답 박스의 IoU

  $$
  \text{confidence} = P(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}
  $$

- **Conditional class probability**: 물체가 있을 때 그 물체가 $$i$$번째 클래스일 확률 $$P(\text{Class}_i \mid \text{Object})$$

추론 때는 둘을 곱해 **클래스별 점수**를 얻습니다. 이 점수가 일정 값 미만인 박스를 걸러내고, 남은 박스에 NMS를 적용합니다. Threshold가 없으면 98개 박스가 전부 그려집니다. 논문은 구체적인 값을 밝히지 않고 "모델의 confidence로 걸러낸다"고만 적었으며, 해설 자료들은 흔히 **0.2**를 예로 듭니다. 논문에 따르면 NMS는 mAP를 2~3%p 올려 줍니다.

### 네트워크 구조

GoogLeNet에서 영감을 받은 24개 conv layer + 2개 FC layer입니다. ImageNet으로 사전학습한 앞쪽 20개 conv에 conv 4개와 FC 2개를 추가했습니다. Conv는 특징을 추출하고, FC는 확률과 좌표를 예측합니다.

### Loss Function

편의상 모든 항을 **SSE(제곱 오차 합)**로 계산합니다.

$$
\begin{aligned}
\mathcal{L} =\ & \lambda_{coord}\sum_{i}\sum_{j}\mathbf{1}_{ij}^{obj}\Big[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\Big] \\
&+ \lambda_{coord}\sum_{i}\sum_{j}\mathbf{1}_{ij}^{obj}\Big[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2\Big] \\
&+ \sum_{i}\sum_{j}\mathbf{1}_{ij}^{obj}(C_i - \hat{C}_i)^2
+ \lambda_{noobj}\sum_{i}\sum_{j}\mathbf{1}_{ij}^{noobj}(C_i - \hat{C}_i)^2 \\
&+ \sum_{i}\mathbf{1}_{i}^{obj}\sum_{c}\big(p_i(c) - \hat{p}_i(c)\big)^2
\end{aligned}
$$

그냥 SSE를 쓰면 두 가지 문제가 있어 보정합니다.

1. **위치 오차와 분류 오차를 동등하게 취급** → 좌표 항에 $$\lambda_{coord} = 5$$
2. **물체가 없는 격자 칸이 압도적으로 많아** confidence가 0으로 끌려감 → 물체 없는 칸에 $$\lambda_{noobj} = 0.5$$

또 큰 박스의 1픽셀 오차와 작은 박스의 1픽셀 오차는 의미가 다르므로, $$w, h$$에는 **제곱근**을 씌웁니다. $$\mathbf{1}_{ij}^{obj}$$는 $$i$$번째 칸의 $$j$$번째 박스 중 **정답과 IoU가 가장 큰 박스**만 1입니다.

### 장점과 한계

**장점**
- 초당 45 frame(Fast YOLO는 155 frame)으로 **실시간** 처리
- 이미지 전체를 한 번에 보므로 맥락을 이해해 **배경을 물체로 착각하는 오류(false positive)가 적음**
- 일반화가 잘됨: 자연 이미지로 학습하고 **예술 작품**에서 테스트해도 다른 모델보다 성능이 덜 떨어짐

**한계**
- R-CNN 계열이 후보를 수천 개 제안하는 데 비해 YOLO는 **98개**뿐
- 격자 칸 하나가 **물체를 최대 2개(사실상 클래스 1개)**까지만 예측 → 새 떼처럼 **작은 물체가 모여 있으면** 잘 못 찾음
- 격자 칸에 비해 작은 물체에 약함

---

## 5. 1-stage의 발전

### SSD (Single Shot MultiBox Detector, 2016)

YOLO v1의 작은 물체 문제를 **여러 해상도의 feature map**으로 해결합니다.

- **큰 feature map**(얕은 층)에서는 **작은 물체**를, **작은 feature map**(깊은 층)에서는 **큰 물체**를 찾음
- 각 feature map 위치마다 크기·비율이 다른 **default box**(anchor)를 둠
- **Hard Negative Mining**: 배경(negative) 박스가 압도적으로 많으므로, confidence loss가 가장 큰 negative만 골라 **Negative : Positive = 3 : 1**로 맞춤
- 강한 data augmentation: IoU 조건을 둔 random patch sampling, flip, zoom in/out, 색상 왜곡

### RetinaNet과 Focal Loss (2017)

1-stage가 2-stage보다 정확도가 낮은 주된 원인은 **클래스 불균형**입니다. 이미지 대부분이 배경이라 쉬운 negative 샘플이 loss를 지배합니다.

**Focal Loss**는 이미 잘 맞히는(쉬운) 샘플의 loss를 줄이고, **잘 못 맞히는 샘플에 가중치**를 줍니다.

$$
\text{CE}(p_t) = -\log(p_t),\qquad \text{FL}(p_t) = -\alpha_t(1 - p_t)^\gamma\log(p_t)
$$

$$p_t$$가 1에 가까운(잘 맞힌) 샘플은 $$(1-p_t)^\gamma$$가 0에 가까워져 loss가 거의 사라집니다. $$\gamma = 2$$가 기본값입니다. RetinaNet은 Focal loss와 **FPN**으로 1-stage이면서 2-stage 정확도를 넘었습니다. Focal loss는 의료 데이터처럼 불균형한 분류 문제에도 널리 쓰입니다([14편]({% post_url 2026-06-29-audio-classification %})의 심음 분류).

### FPN (Feature Pyramid Network)

다양한 크기의 물체를 찾기 위한 구조입니다.

- **Bottom-up path**: 일반 backbone. 깊어질수록 해상도는 낮고 의미 정보는 풍부
- **Top-down path**: 깊은 층의 feature를 **upsampling**해서 얕은 층과 합침
- 결과적으로 **모든 해상도에서 의미 정보가 풍부한** feature map을 얻음

### YOLO v2 (YOLO9000, 2017)

- **Batch Normalization** 도입 (Dropout 제거)
- Classifier를 **448×448로 fine-tuning**해 고해상도 입력에 적응
- 가벼운 backbone **DarkNet-19**
- **Anchor box** 도입. 단, 사람이 정하지 않고 학습 데이터의 박스를 **k-means**로 군집화해서 찾음. 거리로 유클리드 거리 대신 $$1 - \text{IoU}$$를 사용
- 박스 좌표를 직접 예측하지 않고 격자 칸 기준 **offset 비율**로 예측해 학습 안정화
- FC layer 제거 → 다양한 해상도 입력 가능, 학습 중 320~608 크기를 무작위로 바꿈(multi-scale training)
- **Passthrough layer**: 26×26×512를 13×13×2048로 재배열해 이어 붙여 작은 물체 정보 보충
- **9000개 클래스**: WordNet 기반 **계층적 트리**로 label을 구성하고 조건부 확률로 분류. Detection 데이터와 Classification 데이터를 함께 학습

### YOLO v3 (2018)

- Softmax 대신 클래스마다 **독립적인 Binary Cross Entropy** → 한 박스에 여러 label 가능(예: 사람, 여성)
- **FPN과 비슷한 구조**로 3개 해상도에서 예측
- backbone DarkNet-53 (shortcut 추가)
- 논문 대신 Tech report로 발표, Focal loss는 효과가 적었다고 보고

### YOLO v4 (2020)

수많은 기법을 실험해 최적 조합을 찾은 논문입니다. 모델을 세 부분으로 나눠 생각합니다.

| 부분 | 역할 | 예시 |
|---|---|---|
| **Backbone** | 이미지를 feature map으로 변환 | VGG16, ResNet-50, EfficientNet, **CSPDarknet53** |
| **Neck** | Backbone과 Head를 연결, feature 정제 | SPP, ASPP, FPN, **PAN**, BiFPN |
| **Head** | 원하는 출력(박스, 클래스)으로 변환 | YOLO, SSD, RetinaNet, Faster R-CNN |

기법들은 두 종류로 분류했습니다.

- **Bag of Freebies (BoF)**: **추론 속도에 영향 없이** 학습 방법만 바꿔 성능 향상
  - Data augmentation: CutMix, Mixup, **Mosaic**(4장 이어 붙이기), Random erase
  - 규제: **DropBlock**, Label smoothing
  - 정규화: Cross mini-Batch Normalization (작은 배치에 효과적)
  - 불균형: Hard negative mining, Focal loss
  - 박스 loss: MSE 대신 **GIoU / DIoU / CIoU loss**
- **Bag of Specials (BoS)**: 추론 비용이 약간 늘지만 정확도가 크게 오르는 모듈
  - **Mish** activation, SPP, SAM, PAN
  - **CSPNet**: 입력 feature를 반으로 나눠 한쪽만 dense layer에 통과시키고 나머지와 concat. 중복되는 gradient 정보를 줄여 연산량 감소
  - **PAN**: FPN의 최하단 정보가 최상단까지 전달되기 어려우므로 bottom-up 경로를 하나 더 추가
  - **BiFPN** (EfficientDet): 입력마다 학습 가능한 가중치를 줘서 합침

최종 구성: Backbone **CSPDarknet53**, Neck **SPP + PAN**, Head **YOLOv3**, Loss **CIoU**

### YOLO v5, v6, v7

- **v5 (Ultralytics, 2020)**: 논문 없이 코드로 공개, **PyTorch** 기반. YOLO 원 저자(Redmon)의 작업은 아닙니다(v4부터 다른 연구자들이 이어 감). 이 저장소의 **v6.0 릴리스**(2021)에서 더 작은 **Nano** 모델(YOLOv5n)이 추가됐습니다.
- **YOLOv6 (Meituan, 2022)**: 이름은 이어지지만 v5와는 **별개의 프로젝트**로, 논문(arXiv:2209.02976)과 함께 공개됐습니다. RepVGG식 재매개변수화 backbone과 anchor-free head를 씁니다.
- **v7 (2022)**:
  - **E-ELAN**: gradient 경로를 조절해 깊어도 학습이 잘되는 구조
  - **Model re-parameterization**: [RepVGG]({% post_url 2026-06-02-cnn-architectures %})처럼 학습 때 여러 갈래를 쓰고 추론 때 합침. 단 identity 연결이 있는 곳에서는 효과가 없어 identity 없는 **RepConvN** 제안
  - **Compound scaling**: depth와 width를 비율을 맞춰 함께 조절
  - **Auxiliary head**와 coarse-to-fine label assignment

### YOLO 버전별 요약

| 버전 | 주요 변화 |
|---|---|
| v1 | 7×7 grid, 1-stage의 시작 |
| v2 | BN, anchor(k-means), DarkNet-19, multi-scale, 9000 클래스 |
| v3 | FPN 구조, multi-label(BCE), DarkNet-53 |
| v4 | CSPDarknet53, SPP+PAN, Mosaic, CIoU, DropBlock 등 BoF/BoS |
| v5 | Ultralytics PyTorch 구현, Nano 등 소형 모델 (v6.0 릴리스) |
| v6 | Meituan, 재매개변수화 backbone, anchor-free |
| v7 | E-ELAN, RepConv, compound scaling, auxiliary head |

> 이후 버전들(YOLOX, YOLOv8 이후 등)은 anchor 없이 물체 중심과 크기를 바로 예측하는 **anchor-free** 방식, 학습 단계에서 중복 예측을 막아 추론 때 **NMS를 없애는** 방식 쪽으로 발전했습니다. 버전 번호가 같은 저자·같은 계보를 뜻하지 않으므로, 쓸 때는 어느 저장소의 어떤 버전인지 밝히는 것이 좋습니다.
{: .prompt-info }

## 6. 정리

- Detection = **Localization + Classification**, 평가는 **IoU**와 **mAP**
- 2-stage는 **Selective Search → RPN**으로 발전하며 빨라졌고, RoI Pooling으로 크기가 다른 영역을 처리합니다.
- 1-stage는 **YOLO v1**에서 시작해 multi-scale feature(SSD, FPN), anchor, **Focal loss**로 정확도를 따라잡았습니다.
- 최신 Detector는 **Backbone–Neck–Head** 구조로 생각하며, 이 틀은 [경량 Vision Transformer]({% post_url 2026-06-20-lightweight-vision-transformer %})에서도 그대로 쓰입니다.

다음 글에서는 이미지가 아닌 **시퀀스 데이터**를 다루는 RNN, LSTM, GRU를 정리합니다.

## References

- Girshick et al., "Rich feature hierarchies for accurate object detection and semantic segmentation (R-CNN)," CVPR 2014.
- He et al., "Spatial Pyramid Pooling in Deep Convolutional Networks," ECCV 2014.
- Girshick, "Fast R-CNN," ICCV 2015. / Ren et al., "Faster R-CNN," NeurIPS 2015.
- Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," CVPR 2016.
- Liu et al., "SSD: Single Shot MultiBox Detector," ECCV 2016.
- Lin et al., "Feature Pyramid Networks for Object Detection," CVPR 2017.
- Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.
- Redmon & Farhadi, "YOLO9000," CVPR 2017. / "YOLOv3: An Incremental Improvement," 2018.
- Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection," 2020.
- Wang et al., "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors," 2022.
- Rezatofighi et al., "Generalized Intersection over Union," CVPR 2019. / Zheng et al., "Distance-IoU Loss," AAAI 2020.
- He et al., "Mask R-CNN," ICCV 2017. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- Li et al., "YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications," 2022. [arXiv:2209.02976](https://arxiv.org/abs/2209.02976)
- Bodla et al., "Soft-NMS — Improving Object Detection With One Line of Code," ICCV 2017.
