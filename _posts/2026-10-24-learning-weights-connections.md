---
title: "[Deep Learning] Learning both Weights and Connections: 학습-가지치기-재학습으로 AlexNet을 9배, VGG-16을 13배 줄이다"
date: 2026-10-24 14:00:00 +0900
categories: [Deep Learning, Model Compression]
tags: [pruning, magnitude-pruning, sparsity, retraining, iterative-pruning, energy-efficiency, han-dally]
description: Han, Pool, Tran, Dally(NeurIPS 2015)가 학습 → 작은 가중치 가지치기 → 재학습의 3단계로 AlexNet 61M→6.7M, VGG-16 138M→10.3M 파라미터를 정확도 손실 없이 줄인 과정을 정리합니다. 메모리 접근 에너지라는 동기, L1/L2 정규화의 역할, 드롭아웃 비율 조정, 재초기화 대신 가중치 유지, 반복 가지치기, 층별 민감도를 다루고 작은 실험으로 정규화와 반복의 효과를 확인합니다.
math: true
---

(bisect stub)
