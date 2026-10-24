---
title: "[Speech] 계층적 화자 표현으로 하는 원샷 노래 음성 변환: 임베딩 하나로는 가수를 담을 수 없다"
date: 2026-10-24 08:40:00 +0900
categories: [Speech, Voice Conversion]
tags: [singing-voice-conversion, one-shot, speaker-representation, instance-normalization, adain, film, u-net, gan, interspeech]
description: Li, Liu, Shan(Interspeech 2022)이 노래 음성 변환에서 화자 룩업 테이블이나 화자 인식 임베딩 하나 대신, 다운샘플링 스트림의 블록마다 인스턴스 정규화 통계를 뽑아 서로 다른 시간 해상도의 화자 표현을 만들고 업샘플링 블록에 AdaIN으로 주입한 방법을 정리합니다. 작은 실험으로 블록별 통계의 화자 구분력과 평균만 옮기는 AdaIN의 의미를 확인합니다.
math: true
---

(bisect stub)
