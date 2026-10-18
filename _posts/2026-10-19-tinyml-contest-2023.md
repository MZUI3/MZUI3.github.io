---
title: "[Project] TinyML Design Contest 2023: RAM 16KB MCU에서 심실 부정맥 탐지하기"
date: 2026-10-19 00:00:00 +0900
categories: [Neural Network, Lightweight]
tags: [tinyml, edge-ai, stm32, 1d-cnn, iegm, knowledge-distillation, project]
description: ACM/IEEE ICCAD 2023 TinyML Design Contest에서 STM32F303K8(RAM 16KB, Flash 64KB)용 심실 부정맥 탐지 모델을 설계한 과정과 결과(세계 11위)를 정리합니다.
---

> 2023년 ACM/IEEE TinyML Design Contest @ ICCAD에 EmPasLab2 팀(4명)으로 참가해 세계 11위를 했습니다. 이 글은 대회 문제, 모델 구조, 결과와 남은 과제를 정리한 기록입니다. 코드는 [GitHub](https://github.com/MZUI3/iccad2023-tinyml-ecg-anomaly-detection)에 있습니다.

## 1. 대회 문제

이식형 심장 기기에서 얻는 IEGM(심내 심전도) 신호로 생명을 위협하는 **심실 부정맥(VA)** 을 탐지하는 모델을, 마이크로컨트롤러 위에서 돌려야 하는 대회입니다.

| 항목 | 내용 |
|---|---|
| 보드 | STM32F303K8T6 (Cortex-M4, 72MHz) |
| 메모리 | Flash 64KB, SRAM 16KB |
| 입력 | IEGM 5초 구간, 1,250 samples |
| 출력 | VA / 정상 이진 분류 |
| 평가 | F-beta, Generalization, Latency, Flash를 함께 반영 |

F-beta는 recall에 가중치를 더 준 지표입니다. 부정맥을 놓치는 쪽이 오경보보다 훨씬 위험하기 때문입니다. 정확도만 높아서는 안 되고, 보드에서의 추론 시간과 메모리까지 점수에 들어갑니다.

## 2. 신호가 대부분 비어 있다

IEGM은 심박이 일어나는 순간에만 큰 값이 있고, 나머지 구간은 대부분 0 근처입니다. 이미지에서 쓰던 3×1 같은 작은 kernel로 이 신호를 훑으면, 빈 구간에서 의미 없는 곱셈만 반복하게 됩니다. 연산은 늘어나는데 정확도는 오히려 떨어졌습니다.

그래서 첫 층에서 신호를 크게 줄이는 쪽으로 설계를 바꿨습니다.

- 첫 Conv layer의 kernel 길이를 70(약 280ms)으로 늘렸습니다.
- stride를 20(약 80ms)으로 두어 1,250 samples를 60으로 줄였습니다.

## 3. 모델 구조

```python
class model(nn.Module):
    def __init__(self, opts):
        super(model, self).__init__()
        self.layer1 = nn.Sequential(          # 1,250 → 60
            nn.Conv2d(1, 3, kernel_size=(70, 1), stride=(20, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(3),
        )
        self.layer2 = nn.Sequential(          # 60 → 53
            nn.Conv2d(3, 4, kernel_size=(10, 1), padding=(1, 0)),
            nn.ReLU(True),
            nn.BatchNorm2d(4),
        )
        self.layer3 = nn.Sequential(nn.Dropout(0.5), nn.Linear(212, 80), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Dropout(0.15), nn.Linear(80, 40), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Dropout(0.15), nn.Linear(40, 2))

    def forward(self, x):                     # x: (N, 1, 1250, 1)
        x = self.layer2(self.layer1(x))       # (N, 4, 53, 1)
        x = x.view(-1, 212)
        return self.layer5(self.layer4(self.layer3(x)))
```

층별 크기와 연산량을 계산하면 다음과 같습니다.

| 층 | 출력 | 파라미터 | MACs |
|---|---|---:|---:|
| Conv1 (1→3, k=70, s=20) | 3 × 60 | 213 | 12,600 |
| BN1 | 3 × 60 | 6 | – |
| Conv2 (3→4, k=10, p=1) | 4 × 53 | 124 | 6,360 |
| BN2 | 4 × 53 | 8 | – |
| FC 212→80 | 80 | 17,040 | 16,960 |
| FC 80→40 | 40 | 3,240 | 3,200 |
| FC 40→2 | 2 | 82 | 80 |
| **합계** | | **20,713** | **39,200** |

1회 추론에 약 39K MACs입니다. Conv 두 층은 큰 stride 덕분에 연산이 작지만, 파라미터의 82%가 flatten 뒤 첫 FC 층(212→80)에 몰려 있습니다.

작은 모델의 정확도를 보완하려고 Knowledge Distillation을 적용했고, 완성한 모델은 STM32 보드에 C로 포팅했습니다.

## 4. 결과

대회 공식 순위표 기준입니다.

| 순위 | 팀 | F-beta | Latency (ms) | Flash (KB) |
|:---:|---|---:|---:|---:|
| 1 | CDL-EML TU Wien | 0.974 | 1.52 | 12.83 |
| 2 | HugeRabbit | 0.945 | 1.33 | 12.19 |
| 3 | TinyNet | 0.922 | 0.89 | 10.25 |
| **11** | **EmPasLab2 (우리 팀)** | **0.947** | **12.76** | **26.63** |

Generalization은 0.842(환자 19명 중 16명)였습니다.

F-beta는 2위와 3위보다 높았지만, Latency는 상위 3팀보다 8~14배 길었고 Flash도 두 배 이상 컸습니다. 정확도를 지키는 설계는 어느 정도 해냈고, 보드에서의 실행 시간과 메모리는 다음 과제로 남았습니다.

## 5. 다음에 줄일 곳

- **FC 층**: 파라미터 대부분이 첫 FC 층에 있으니, Global Average Pooling처럼 FC 입력을 줄이는 구조가 먼저 시도할 후보입니다.
- **Latency**: 같은 정확도라면 실행 시간이 순위를 갈랐습니다. 모델을 설계하는 단계에서부터 보드 위 Latency를 함께 재면서 구조를 골라야 합니다.

## 참고

- [TinyML Design Contest 2023 Winners](https://tinymlcontest.github.io/TinyML-Design-Contest-2023/Winners.html)
- [GitHub: iccad2023-tinyml-ecg-anomaly-detection](https://github.com/MZUI3/iccad2023-tinyml-ecg-anomaly-detection)
