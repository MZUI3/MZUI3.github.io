---
title: "[Generative AI] Splice: DINO-ViT 특징을 쪼개어 구조와 외형을 잇는 의미 기반 외형 전이"
date: 2026-10-24 12:00:00 +0900
categories: [Generative AI, Style Transfer]
tags: [appearance-transfer, dino, vit, self-similarity, cls-token, single-image, feature-inversion, splice]
description: Splice(Tumanyan, Bar-Tal, Bagon, Dekel, CVPR 2022)가 자기지도 학습된 DINO-ViT를 외부 의미 사전으로 삼아, 마지막 층 키의 자기유사도를 구조로, [CLS] 토큰을 외형으로 정의하고 이미지 한 쌍만으로 생성기를 학습해 의미적으로 대응하는 영역끼리 외형을 옮긴 과정을 정리합니다. 특징 반전으로 얻은 관찰, 손실 설계, 사용자 연구를 다루고 작은 실험으로 두 표현의 성질을 확인합니다.
math: true
---

(bisect stub)
