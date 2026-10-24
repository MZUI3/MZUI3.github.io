---
title: "[Deep Learning] Lottery Ticket Hypothesis: 가지치기로 찾은 희소 부분망은 초기값만 되돌리면 처음부터 학습된다"
date: 2026-10-24 12:20:00 +0900
categories: [Deep Learning, Model Compression]
tags: [lottery-ticket, pruning, sparse-training, initialization, iterative-magnitude-pruning, winning-ticket, frankle-carbin]
description: Frankle과 Carbin(ICLR 2019 best paper)이 제기한 복권 가설, 즉 무작위 초기화된 밀집 네트워크 안에는 원래 초기값으로 되돌려 단독 학습하면 원래 네트워크와 같은 정확도에 같은 속도로 도달하는 부분망(당첨 복권)이 있다는 주장과, 반복적 크기 가지치기로 그것을 찾는 실험을 정리합니다. 무작위 재초기화 대조, 학습률 민감성과 워밍업, 그리고 Liu 등과의 논쟁을 다루고 작은 실험 결과를 솔직하게 보고합니다.
math: true
---

(bisect stub)
