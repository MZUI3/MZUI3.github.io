---
title: "[Semiconductor] Adaptive Alternate Analog Test: 커널 밀도 결함 필터로 아날로그 테스트를 적응시키기"
date: 2026-10-24 09:20:00 +0900
categories: [Semiconductor, Test]
tags: [analog-test, alternate-test, adaptive-test, kernel-density-estimation, defect-filter, rf, evm, regression, machine-learning-test]
description: Stratigopoulos와 Mir(IEEE Design & Test 2012)가 사양 테스트를 저가의 대체 측정으로 바꾸는 alternate test에, 학습 분포 밖의 칩을 골라내는 두 개의 커널 밀도 결함 필터(엄격·관대)를 붙여 테스트 흐름을 칩마다 적응시키는 방법을 정리합니다. RF 수신기 EVM 사례에서 필터 파라미터에 따른 재검사율·직접 불량 판정률·예측 상관의 트레이드오프를 다루고, 작은 실험으로 적응형 KDE 필터의 동작을 확인합니다.
math: true
---

(bisect stub)
