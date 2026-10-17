---
title: 블로그를 시작하며
description: AI 모델링, 강화학습, 웹 개발을 공부하며 정리하는 MZUI.log의 소개와 카테고리 구성입니다.
date: 2026-05-01 10:12:00 +0900
categories: [Blog, Setup]
tags: [jekyll, chirpy, github-pages]
pin: false
---

## 블로그를 만든 이유

프로젝트 과정과 공부한 내용을 깊이 있게 기록하기 위해 시작했습니다.

주로 연구실 세미나에서 발표하고 공부한 AI 논문(강화학습, 모델 경량화, 생성 모델, Transformer)과, 개발하면서 익힌 기초를 정리합니다.

글을 쓸 때는 다음 원칙을 지키려고 합니다.

- **출처를 남긴다**: 논문 리뷰에는 원 논문(arXiv 등) 링크를 함께 적습니다.
- **돌려 본 코드는 결과까지**: 코드는 복사해서 바로 실행할 수 있게 쓰고, 실행한 환경(라이브러리 버전)과 출력 결과를 함께 남깁니다.
- **처음 보는 사람도 따라올 수 있게**: 전문 용어는 처음 나올 때 한 문장으로 풀고, 깊은 내용은 "더 알아보기"로 접어 둡니다.

## 카테고리 구성

공부한 내용은 다음 네 개의 큰 분야로 나눠 정리합니다.

```text
Neural Network
 ├─ Basics              학습 원리, RNN, PyTorch 실습
 ├─ CNN                 CNN 기초와 대표 아키텍처
 ├─ Object Detection
 ├─ Segmentation
 ├─ Lightweight         Pruning, Quantization, BNN, Edge AI 하드웨어
 └─ Audio               음성·소리 분류

Transformer
 ├─ Architecture
 ├─ Vision Transformer
 └─ LLM

Generative AI
 ├─ Diffusion
 └─ Image Synthesis

Reinforcement Learning
 ├─ Model-Free          (태그: Q-Learning, DQN, PPO)
 ├─ Model-Based
 ├─ Exploration
 ├─ Offline RL
 ├─ Imitation Learning
 ├─ Hierarchical RL
 └─ Multi-Agent
```

블로그 테마가 카테고리를 두 단계까지만 지원하기 때문에, 그보다 세부적인 분류(예: Model-Free 안의 Q-Learning, DQN, PPO)는 **태그**로 붙입니다. 태그 페이지에서 알고리즘별로 모아 볼 수 있습니다.

AI 분야 외에 웹·데이터베이스·Python 같은 개발 기초 글은 **개발** 카테고리에, 블로그 자체에 대한 글은 **Blog** 카테고리에 따로 둡니다.
