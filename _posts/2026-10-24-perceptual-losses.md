---
title: "[Generative AI] Perceptual Losses: 픽셀 대신 특징 공간에서 손실을 재면 실시간 스타일 전이가 된다"
date: 2026-10-24 11:00:00 +0900
categories: [Generative AI, Style Transfer]
tags: [style-transfer, perceptual-loss, feature-reconstruction, gram-matrix, super-resolution, vgg, feed-forward, johnson]
description: Johnson, Alahi, Fei-Fei(ECCV 2016)가 사전학습 VGG의 특징 공간에서 정의한 지각 손실로 순전파 변환 네트워크를 학습해, Gatys의 최적화 기반 스타일 전이와 같은 목적 함수를 1,000배 빠르게 풀고 초해상도에서 픽셀 손실보다 또렷한 결과를 얻은 과정을 정리합니다. 작은 실험으로 픽셀 손실과 특징 손실의 차이, Gram 행렬의 공간 불변성을 확인합니다.
math: true
---

(bisect stub)
