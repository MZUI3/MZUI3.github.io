---
title: "[Computer Vision] Diffusion Models for Image Restoration — 지도 학습과 제로샷, 두 흐름으로 본 디퓨전 기반 영상 복원"
date: 2026-10-24 15:20:00 +0900
categories: [Computer Vision, Image Restoration]
tags: [image-restoration, diffusion-models, survey, super-resolution, deblurring, inpainting, zero-shot, perception-distortion]
description: Li, Ren 등(2023)의 디퓨전 기반 영상 복원 서베이를 정리합니다. SR3로 대표되는 지도 학습 디퓨전과 ILVR로 대표되는 제로샷 디퓨전이라는 두 작업 흐름, 조건 전략(저화질 참조, 전처리 참조, 디퓨전 과정 수정)과 생성 공간(이미지·잔차·잠재), 블라인드/실세계 복원을 위한 왜곡 시뮬레이션·커널 추정·도메인 변환·왜곡 불변 학습, 초해상도·디블러링·인페인팅의 객관 비교, 그리고 샘플링 효율과 모델 압축 등 다섯 과제를 다룹니다. 작은 실험으로 지각-왜곡 트레이드오프를 확인합니다.
math: true
---

(bisect stub)
