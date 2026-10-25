---
title: "[Speech] 계층적 화자 표현으로 하는 원샷 노래 음성 변환: 임베딩 하나로는 가수를 담을 수 없다"
date: 2026-10-24 08:40:00 +0900
categories: [Speech, Voice Conversion]
tags: [singing-voice-conversion, one-shot, speaker-representation, instance-normalization, adain, film, u-net, gan, interspeech]
description: Li, Liu, Shan(Interspeech 2022)이 노래 음성 변환에서 화자 룩업 테이블이나 화자 인식 임베딩 하나 대신, 다운샘플링 스트림의 블록마다 인스턴스 정규화 통계를 뽑아 서로 다른 시간 해상도의 화자 표현을 만들고 업샘플링 블록에 AdaIN으로 주입한 방법을 정리합니다. 작은 실험으로 블록별 통계의 화자 구분력과 평균만 옮기는 AdaIN의 의미를 확인합니다.
math: true
---

> 노래 음성 변환(SVC)은 노래의 가사·멜로디는 두고 목소리만 다른 가수로 바꾸는 과제입니다. 가수를 나타내는 방법은 보통 둘입니다. 학습 가수마다 벡터 하나를 두는 **룩업 테이블(LUT)**, 또는 화자 인식 네트워크(SRN)가 참조 오디오에서 뽑는 **임베딩 하나**입니다. 그런데 노래는 대화보다 훨씬 표현적이어서 **몇 초 안에서도 화자 특성이 크게 요동**칩니다. 이 논문(Li, Liu, Shan, Tencent ARC Lab, Interspeech 2022)은 벡터 하나로 평균적인 특성만 담는 대신, **여러 시간 해상도의 통계를 계층적으로** 뽑아 화자를 표현합니다. LUT와 ECAPA-TDNN 임베딩 기반 시스템보다 자연성·유사도 MOS가 높고, 참조 오디오 몇 초만으로 처음 보는 가수에게도 변환(원샷)이 됩니다.

## 1. 문제: 평균으로는 노래를 설명할 수 없다

SVC는 노래 신호에 섞인 내용, 음높이, 세기, 화자 정체성을 분리한 뒤 화자만 바꿔 재합성합니다. 내용은 사전학습 ASR의 언어 특징, 음높이는 log-F0나 사인 여기 신호, 세기는 A-가중 파워로 뽑는 방법이 정착되어 있습니다. 화자 모델링이 남은 문제입니다.

- LUT는 학습 가수에게는 안정적이지만 **처음 보는 가수에게는 쓸 수 없습니다**.
- SRN 임베딩은 원샷이 되지만, 참조 오디오마다 값이 흔들리고 **한 벡터가 시간에 따라 변하는 특성을 뭉갭니다**.

대화 음성이라면 한 벡터로 압축이 어느 정도 가능하지만, 비브라토·가성·발성 변화가 큰 노래에서는 의심스럽다는 것이 출발점입니다.

## 2. 방법

### 구조: 업샘플링 스트림 하나 + 다운샘플링 스트림 셋

FastSVC(Liu et al., 2021)의 end-to-end 구조를 기반으로 합니다. **업샘플링 스트림**은 언어 특징(홉 640샘플)을 2, 4, 4, 4, 5배씩 올려 오디오 샘플을 직접 생성합니다(채널 288 → 192 → 96 → 48 → 24, 팽창률 1, 3, 9, 27). 세 개의 **다운샘플링 스트림** 중 둘은 음높이와 세기 특징을 각 업샘플링 블록의 시간 해상도에 맞게 내려 FiLM으로 주입하고, 나머지 하나가 이 논문의 핵심입니다.

### 계층적 화자 표현

세 번째 다운샘플링 스트림은 오디오 샘플을 다운샘플링 블록과 **인스턴스 정규화(IN)**를 번갈아 거쳐 언어 특징까지 되돌립니다. 각 블록 뒤의 IN은 시간 축에 대해 샘플별 정규화를 하는데, 이때 계산되는 **시간 통계**가 그 해상도에서의 화자 특성입니다(Chou et al.의 원샷 VC와 같은 직관). 업샘플링 블록 5개에 맞춰 IN도 5개이므로 통계도 5벌이 나오고, 이것이 계층적 화자 표현입니다. 각 통계는 대응하는 업샘플링 블록의 **AdaIN**에 들어가 은닉 특징의 화자 특성을 제거하고 새 화자 특성으로 바꿉니다.

변환 때는 다운샘플링 스트림에 **목표 가수의 참조 오디오**를 넣어 통계를 뽑고 업샘플링 스트림에 섞습니다. 참조가 몇 초면 되므로 원샷 변환입니다.

### 평균만 쓴다

IN은 보통 평균과 분산을 모두 정규화하지만, 이 논문은 **평균만** 씁니다. 분산을 옮기면 음질이 떨어졌는데, 분산 통계가 언어 내용의 영향을 크게 받아 **내용이 다른 참조 오디오에서 가져온 분산은 오히려 해가 된다**는 해석입니다.

### 업샘플링 블록

FiLM이 음높이(8개 하모닉 사인 여기)와 세기 특징에서 스케일 $$\gamma$$와 이동 $$\xi$$를 만들어 $$\tilde U = (\gamma_F + \gamma_L) \odot U + \xi_F + \xi_L$$로 은닉 특징을 변조하고, AdaIN이 시간 통계를 받습니다. 동시 연구인 Unet-TTS와 달리 음높이·세기를 명시적으로 넣어 통계에서 화자 정보만 분리하도록 유도하고, 멜 스펙트로그램과 별도 보코더 대신 샘플을 직접 생성해 다단계 오차 누적을 피합니다.

### 손실

최소제곱 GAN 손실(3개 스케일 판별기) + 다중 스케일 STFT 손실(FFT 2048~64, 75% 겹침) + 다운샘플링 스트림이 복원한 언어 특징의 MSE. 마지막 항이 다운샘플링 스트림의 출력을 **화자 불변**으로 만들어, 화자 정보가 통계 쪽으로 가도록 밀어 줍니다. $$\alpha = \beta = 2.5$$, 판별기는 100k 스텝 후 합류합니다.

## 3. 코드로 보기: 블록별 통계와 평균만 옮기기

고정된 무작위 "다운샘플링 블록"(합성곱 + ReLU + 4배 다운샘플링)을 네 개 쌓고, 두 화자의 합성 발화에서 블록별 시간 평균을 뽑아 화자 내·화자 간 거리를 비교합니다. 그리고 내용이 다른 참조 오디오로 AdaIN을 할 때 평균만 옮기는 경우와 평균·분산을 모두 옮기는 경우의 오차를 봅니다.

```python
import numpy as np
rng = np.random.default_rng(0)
T, C = 1024, 8
# fixed random "down-sampling blocks": conv (3 taps) + ReLU + 4x down-sampling, as in the paper's down-sampling stream
Wb = [rng.normal(0, 1 / np.sqrt(3 * C), (C, C, 3)) for _ in range(4)]
def block(x, W):
    y = sum(W[:, :, k] @ np.roll(x, k - 1, axis=1) for k in range(3)); y = np.maximum(y, 0)
    return y[:, : (y.shape[1] // 4) * 4].reshape(C, -1, 4).mean(2)
def speaker(seed): r = np.random.default_rng(seed); return r.normal(0, 1, C), r.normal(0, 0.3, C)
def utterance(spk, content_seed):            # content = random smooth envelopes; the speaker shifts and tilts each channel
    r = np.random.default_rng(content_seed)
    base = np.cumsum(r.normal(0, 1, (C, T)), 1); base = (base - base.mean(1, keepdims=True)) / base.std(1, keepdims=True)
    off, tilt = spk
    return base * (1 + tilt[:, None]) + off[:, None] + 0.3 * r.normal(0, 1, (C, T))
def hier_stats(x):                           # instance-norm means after each block = speaker statistics at that granularity
    out = []
    for W in Wb:
        x = block(x, W); out.append(x.mean(1))
    return out
A, B = speaker(1), speaker(2)
sA = [hier_stats(utterance(A, i)) for i in range(10)]; sB = [hier_stats(utterance(B, 100 + i)) for i in range(10)]
print("per-block temporal-mean statistics: within-speaker vs. between-speaker distance (10 utterances each)")
print("block (time resolution)   within   between   ratio")
for k in range(4):
    within = np.mean([np.linalg.norm(sA[i][k] - sA[j][k]) for i in range(10) for j in range(i + 1, 10)])
    between = np.mean([np.linalg.norm(sA[i][k] - sB[j][k]) for i in range(10) for j in range(10)])
    print(f"   {k+1} (1/{4**(k+1):<4})            {within:5.2f}    {between:5.2f}    {between/within:5.1f}")
print("-> every block carries speaker information, at a different granularity; a single embedding keeps only one of these")

# AdaIN with mean only vs. mean+variance when the reference has different content (the paper keeps mean only)
def adain(x, ref, use_var):
    xn = x - x.mean(1, keepdims=True)
    if use_var: xn = xn / x.std(1, keepdims=True) * ref.std(1, keepdims=True)
    return xn + ref.mean(1, keepdims=True)
target = utterance(B, 0)                  # the content of utterance A-0 spoken by B (what we want)
src = utterance(A, 0); ref = utterance(B, 103)   # reference from B with different content
for use_var in (False, True):
    conv = adain(src, ref, use_var)
    print(f"{'mean+var' if use_var else 'mean only':<9} AdaIN: relative error to the true B rendition = {np.linalg.norm(conv - target)/np.linalg.norm(target):.3f}")
```

```text
per-block temporal-mean statistics: within-speaker vs. between-speaker distance (10 utterances each)
block (time resolution)   within   between   ratio
   1 (1/4   )             0.44     2.16      4.9
   2 (1/16  )             0.24     0.78      3.2
   3 (1/64  )             0.12     0.25      2.1
   4 (1/256 )             0.09     0.14      1.5
-> every block carries speaker information, at a different granularity; a single embedding keeps only one of these
mean only AdaIN: relative error to the true B rendition = 0.138
mean+var  AdaIN: relative error to the true B rendition = 0.040
```

- 네 블록 모두 화자 간 거리가 화자 내 거리보다 크지만 비율은 4.9배에서 1.5배로 **해상도마다 다릅니다**. 얕은 블록은 세밀한 음색을, 깊은 블록은 거친 전역 특성을 담는 식으로 정보가 나뉘어 있어서, 벡터 하나를 쓰면 이 중 한 해상도만 남습니다. 계층적 표현은 네 벌을 모두 쓰는 것입니다.
- 이 합성 데이터에서는 분산까지 옮기는 쪽이 오차가 작았습니다(0.040 대 0.138). 여기서 화자 특성은 채널별 기울기(분산)로도 정의했기 때문입니다. 논문의 "평균만" 선택은 **실제 노래에서 분산 통계가 언어 내용에 오염된다**는 경험적 관찰이고, 이 toy는 그 조건을 재현하지 않습니다. 분산을 쓸지는 데이터가 결정하는 문제라는 뜻입니다.

## 4. 실험

**설정.** NUS-48E(12명, 가수당 약 15분)로 학습하고 90/5/5로 나눠 in-set 평가, NHSS에서 남녀 2명씩 뽑아 out-set(미지 가수) 평가. 16kHz. 언어 특징은 LibriSpeech로 학습한 Conformer CTC-attention ASR 인코더의 512차원 출력, 음높이는 Parselmouth F0 → 8개 하모닉 사인 여기, 세기는 A-가중 파워. 비교 시스템은 화자 모델링만 LUT 또는 ECAPA-TDNN 임베딩으로 바꾼 동일 구조입니다.

**MOS(5점, 15명 평가).**

| 조건 | 시스템 | 자연성 | 유사도 |
|---|---|---|---|
| in-set | LUT | 3.43 ± 0.09 | 3.22 ± 0.14 |
| in-set | ECAPA-TDNN | 3.36 ± 0.10 | 3.17 ± 0.15 |
| in-set | 제안 | **3.53 ± 0.09** | **3.36 ± 0.12** |
| in-set | 녹음 | 4.69 | 4.39 |
| out-set | ECAPA-TDNN | 3.24 ± 0.10 | 3.20 ± 0.16 |
| out-set | 제안 | **3.35 ± 0.11** | **3.29 ± 0.17** |
| out-set | 녹음 | 4.77 | 4.64 |

in-set에서는 LUT가 임베딩보다 낫습니다. LUT는 같은 가수에게 항상 같은 벡터를 주지만 ECAPA 임베딩은 참조마다 흔들리기 때문입니다. 제안 방법은 둘 다 넘고, LUT가 아예 못 하는 out-set에서도 임베딩 기반보다 좋습니다. 유사도 점수의 신뢰구간이 자연성보다 넓은데, 사람의 유사도 판단이 음높이에 크게 좌우되기 때문으로 봅니다.

**스펙트로그램.** LUT와 임베딩 시스템의 변환 결과는 일부 구간이 흐릿한 반면, 제안 시스템은 하모닉 질감이 또렷합니다. 화자 정보가 부족하면 재합성이 뭉개진다는 해석입니다.

## 5. 해석

**1. "화자 = 벡터 하나"라는 가정을 노래가 깨뜨립니다.** 대화 음성 VC에서 통하던 전역 임베딩이 노래에서는 평균에 불과하다는 문제 제기가 이 논문의 가치입니다. 위 실험처럼 해상도마다 화자 정보의 성격이 다르다면, 어느 한 층의 통계만 쓰는 것은 정보를 버리는 일입니다.

**2. IN 통계를 화자 표현으로 쓰는 계보에 있습니다.** [AdaIN]({% post_url 2026-10-24-adain %})이 이미지의 스타일을 채널 통계로 정의했고, Chou et al.(2019)이 음성의 화자를 IN 통계로 정의했습니다. 이 논문은 그 통계를 **여러 깊이에서** 뽑고, [U-Net]({% post_url 2026-10-24-mwcnn %})식 대칭 구조로 같은 해상도의 블록끼리 연결합니다. Unet-TTS와의 차이(음높이·세기를 명시적으로 분리, 샘플 직접 생성)가 노래라는 도메인에 맞춘 조정입니다.

**3. 평균만 쓰는 결정은 분리(disentanglement)의 실용적 타협입니다.** 분산에 내용이 섞인다는 관찰은 통계가 "순수한 화자"가 아님을 인정하는 것입니다. 언어 특징 복원 MSE로 다운샘플링 스트림을 화자 불변으로 밀어 주는 것도 같은 맥락이며, 완전한 분리는 보장되지 않습니다.

**4. 평가가 전부 주관 평가입니다.** 화자 유사도의 객관 지표(화자 인식 모델의 코사인 유사도)나 음높이 정확도가 없어, 개선이 어느 요소에서 왔는지 분리하기 어렵습니다. 이후 디퓨전 기반 SVC(DiffSVC 등)와 비교하려면 이런 지표가 필요합니다.

**5. 평가를 읽을 때 주의할 점.** 학습 데이터가 12명 3시간 남짓으로 매우 작고, 평가자 15명이 모두 중국어 화자입니다. out-set 가수는 4명뿐이고 MOS 차이(0.1 안팎)는 신뢰구간과 겹칩니다. 위 toy의 "블록"은 학습되지 않은 무작위 합성곱이라 실제 네트워크가 통계에 무엇을 담는지는 보여 주지 못합니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 문제 | LUT는 미지 가수 불가, 임베딩 하나는 시간에 따라 변하는 노래의 화자 특성을 평균으로 뭉갬 |
| 핵심 | 다운샘플링 스트림의 블록마다 IN 시간 평균을 뽑아 5벌의 계층적 화자 표현 → 대응 업샘플링 블록에 AdaIN |
| 조건 주입 | 음높이(8하모닉 사인 여기)·세기(A-가중)는 FiLM, 화자 통계는 AdaIN, 평균만 사용 |
| 손실 | LSGAN + 다중 스케일 STFT + 언어 특징 복원 MSE($$\alpha = \beta = 2.5$$) |
| 결과 | in-set 자연성 3.53 / 유사도 3.36(LUT 3.43 / 3.22), out-set 3.35 / 3.29(ECAPA 3.24 / 3.20), 몇 초 참조로 원샷 |
| 주의 | 소규모 데이터, 주관 평가만, 분산 미사용은 경험적 선택 |

## References

- Li, Liu, Shan, "A Hierarchical Speaker Representation Framework for One-shot Singing Voice Conversion," Interspeech 2022. [arXiv:2206.13762](https://arxiv.org/abs/2206.13762)
- Liu, Cao, Hu, Su, Meng, "FastSVC: Fast Cross-Domain Singing Voice Conversion with Feature-wise Linear Modulation," ICME 2021. [arXiv:2011.05731](https://arxiv.org/abs/2011.05731)
- Chou, Yeh, Lee, "One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization," Interspeech 2019. [arXiv:1904.05742](https://arxiv.org/abs/1904.05742)
- Li, Pu, Huang, Huang, "Unet-TTS: Improving Unseen Speaker and Style Transfer in One-shot Voice Cloning," ICASSP 2022. [arXiv:2109.11115](https://arxiv.org/abs/2109.11115)
- Desplanques, Thienpondt, Demuynck, "ECAPA-TDNN," Interspeech 2020. [arXiv:2005.07143](https://arxiv.org/abs/2005.07143)
