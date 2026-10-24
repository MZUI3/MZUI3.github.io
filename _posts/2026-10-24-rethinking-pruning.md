---
title: "[Deep Learning] Rethinking the Value of Network Pruning: 구조적 가지치기의 가치는 가중치가 아니라 구조에 있다"
date: 2026-10-24 12:40:00 +0900
categories: [Deep Learning, Model Compression]
tags: [pruning, structured-pruning, architecture-search, scratch-training, baseline, network-slimming, lottery-ticket]
description: Liu, Sun 등(ICLR 2019)이 6개 구조적 가지치기 방법과 1개 비구조적 방법을 재평가해, 가지치기된 모델을 미세조정하는 것이 같은 구조를 무작위 초기화에서 학습하는 것보다 낫지 않다는 것을 보이고, 자동 가지치기의 가치를 암묵적 구조 탐색으로 재해석한 과정을 정리합니다. Scratch-E/B 예산 설계, 비구조적 가지치기와 ImageNet의 예외, 복권 가설과의 비교를 다루고 작은 실험으로 핵심 현상을 확인합니다.
math: true
---

(bisect stub)
