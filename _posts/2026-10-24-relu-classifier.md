---
title: "[Deep Learning] Deep Learning using ReLU: 소프트맥스 대신 ReLU를 분류층에 쓰면 어떻게 되는가"
date: 2026-10-24 14:40:00 +0900
categories: [Deep Learning, Fundamentals]
tags: [relu, softmax, classification-layer, cross-entropy, dying-relu, mnist, fashion-mnist, agarap]
description: Agarap(2018)이 신경망의 마지막 분류층을 소프트맥스 대신 ReLU로 바꾸고 MNIST, Fashion-MNIST, WDBC에서 비교한 짧은 실험 보고서를 정리합니다. 제안의 내용, 결과(FFNN에서는 대등, CNN과 WDBC에서는 느린 수렴과 낮은 정확도), 저자가 든 죽은 ReLU 설명을 다루고, 작은 실험으로 ReLU 분류층의 손실이 왜 학습을 방해하는지 확인합니다. 결론은 이 아이디어를 쓰지 말아야 할 이유에 가깝습니다.
math: true
---

(bisect stub)
