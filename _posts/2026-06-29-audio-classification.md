---
title: "[AI 기초 14] 음성·소리 분류 기초: MFCC, Mel-Spectrogram부터 CRNN, 심음 분류까지"
date: 2026-06-29 20:05:00 +0900
categories: [Neural Network, Audio]
tags: [audio, speech, mfcc, mel-spectrogram, stft, wav2vec, sincnet, pase, crnn, focal-loss, heart-sound]
description: 소리를 신경망에 넣기 위한 STFT, Mel-Spectrogram, MFCC 추출 과정과 각 단계의 이유, 신경망 기반 음성 특징(wav2vec, SincNet, PASE), 심음 분류 논문 두 편을 정리합니다.
math: true
---

> 음성 인식, 화자 인식, 의료용 심음(心音) 분류처럼 **소리**를 다루는 문제도 결국 CNN과 RNN으로 풉니다. 다만 파형을 그대로 넣기보다는, 먼저 소리를 **특징 벡터(feature)**로 바꾸는 과정이 중요합니다. 이 글에서는 대표적인 음성 특징인 MFCC와 Mel-Spectrogram, 신경망 기반 특징 추출, 그리고 실제 심음 분류 논문 두 편을 묶어서 정리합니다.

## 1. 왜 특징 추출이 중요한가

- 아무리 좋은 인식 알고리즘을 써도 **음성 구간을 제대로 검출하지 못하면** 좋은 인식률을 기대하기 어렵습니다.
- 잡음이 많은 환경에서는 검출된 구간에 **잡음이 섞이는** 경우가 많아, 인식 전에 정확한 음성 검출과 특징 추출이 필요합니다.
- 머신러닝에서 데이터를 **벡터로 만든다는 것은 곧 학습이 가능해진다는 뜻**입니다.

16kHz로 녹음한 1초짜리 소리는 16,000개의 숫자입니다. 이 숫자들 자체보다는 **어떤 주파수가 언제 얼마나 강한지**가 소리의 정체를 알려 줍니다.

### 대표적인 음성 특징 추출 기술

- **LPC (Linear Predictive Coefficient)**: 현재 샘플을 과거 샘플의 선형 결합으로 예측하는 계수
- **Cepstrum**: 스펙트럼에 로그를 취하고 다시 역변환한 것. 성대 진동(source)과 성도 모양(filter)을 분리
- **MFCC (Mel-Frequency Cepstral Coefficient)**: 사람 귀의 특성을 반영한 cepstrum (가장 널리 사용)
- **Filter Bank Energy / Mel-Spectrogram**: MFCC 직전 단계의 값
- **Neural Acoustic Feature Extraction**: 신경망이 직접 특징을 학습

## 2. STFT와 Spectrogram

파형 전체를 한 번에 푸리에 변환하면 "어떤 주파수가 있는지"는 알 수 있지만 "**언제**"인지는 사라집니다. 그래서 짧은 구간(frame)으로 잘라 구간마다 푸리에 변환을 하는 **STFT(Short-Time Fourier Transform)**를 씁니다.

### Framing과 Overlap

- 소리를 20~40ms 정도의 짧은 frame으로 자릅니다. 이 정도 구간에서는 소리의 성질이 거의 변하지 않는다고 가정합니다.
- 경계에서 정보가 끊기지 않도록 frame을 **겹치게** 만듭니다. 예를 들어 16kHz 음성을 한 frame에 512개 샘플(32ms)씩 담고 50%씩 겹치면 다음과 같습니다. 이렇게 frame을 옮기는 간격(여기서는 256샘플)을 **hop length**라고 합니다.

  ```text
  frame 1: 샘플    0 ~  511
  frame 2: 샘플  256 ~  767
  frame 3: 샘플  512 ~ 1023
  ```

- 각 frame에 **Hamming/Hanning window**를 곱해 양 끝을 부드럽게 0으로 만듭니다. 잘린 경계에서 생기는 가짜 고주파 성분(spectral leakage)을 줄이기 위해서입니다.

STFT 결과를 **x축 시간, y축 주파수, 색은 세기(power)**로 그린 것이 **Spectrogram**입니다. 소리를 이미지로 바꾼 것이라 CNN을 그대로 쓸 수 있습니다.

### Window 길이의 trade-off

Window가 길면 **주파수 해상도**가 좋아지고, 짧으면 **시간 해상도**가 좋아집니다. 숫자로 보면, 샘플링 주파수 $$f_s$$에서 FFT 길이가 $$N$$이면 주파수 칸 하나의 폭은 $$f_s / N$$입니다.

| 16kHz, FFT 길이 | 시간 폭 | 주파수 칸 폭 |
|---|---|---|
| 256 | 16ms | 62.5Hz |
| 512 | 32ms | 31.25Hz |
| 2048 | 128ms | 7.8Hz |

둘을 동시에 높일 수는 없으므로(불확정성 원리와 같은 구조), 문제에 맞는 길이를 실험으로 찾습니다(아래 심음 분류 논문 참고).

## 3. MFCC 추출 과정

### Mel Scale

사람의 귀는 **저주파의 차이에는 민감하고 고주파의 차이에는 둔감합니다.** 100Hz와 200Hz는 쉽게 구분하지만, 10,000Hz와 10,100Hz는 거의 구분하지 못합니다. Mel scale은 이 특성을 반영한 주파수 단위입니다.

$$
m = 2595\log_{10}\left(1 + \frac{f}{700}\right)
$$

> 이 식은 HTK 방식의 Mel 공식입니다. **librosa의 기본값(`htk=False`)은 1kHz 아래는 선형, 위는 로그로 나누는 Slaney 방식**이라 같은 주파수라도 Mel 값이 다릅니다. 다른 라이브러리(torchaudio 기본값은 HTK)로 만든 특징과 섞어 쓰거나, 학습과 추론을 다른 라이브러리로 할 때 결과가 어긋나는 흔한 원인입니다.
{: .prompt-warning }

### 단계별 과정

1. **Pre-emphasis**: 고주파 성분을 강조합니다. 음성은 고주파로 갈수록 에너지가 약해지기 때문에, 미리 고주파를 키워 스펙트럼의 높낮이를 고르게 맞추려는 것입니다. 다만 고주파 잡음도 함께 커질 수 있고, 최근 딥러닝 파이프라인에서는 뒤의 정규화가 비슷한 역할을 해서 생략하는 경우도 많습니다.

   $$
   y[n] = x[n] - 0.97\,x[n-1]
   $$

2. **Framing + Windowing**: 위에서 본 대로 겹치는 frame으로 자르고 window를 곱합니다.
3. **FFT (DFT)**: 각 frame의 스펙트럼을 구합니다.
4. **Power Spectrum**: 스펙트럼 계수를 제곱합니다.

   $$
   P(k) = \frac{1}{N}\lvert X(k)\rvert^2
   $$

5. **Mel Filter Bank**: Mel scale에서 일정한 간격으로 놓인 **삼각형 필터 묶음**(보통 26~128개)을 적용합니다. 저주파에는 좁은 필터가 촘촘히, 고주파에는 넓은 필터가 듬성듬성 배치됩니다. 결과가 **Mel-Spectrogram**입니다.
6. **Log**: 사람은 소리 크기를 로그 스케일로 느끼므로 로그를 취합니다.
7. **DCT (Discrete Cosine Transform)**: 로그 Mel 에너지를 DCT해서 **서로 상관관계가 적은 계수**로 바꾸고, 앞쪽 12~13개만 남깁니다. 이것이 **MFCC**입니다. 앞쪽 계수는 스펙트럼의 큰 윤곽(성도의 모양, 즉 어떤 음소인지)을, 뒤쪽 계수는 빠르게 출렁이는 세부(성대 진동의 음높이 등)를 담기 때문에 앞쪽만 남기는 것입니다.

아래 코드는 librosa 0.10 기준입니다. `librosa.ex()`는 처음 실행할 때 예제 오디오를 인터넷에서 내려받습니다.

```python
import librosa
import numpy as np

y, sr = librosa.load(librosa.ex("trumpet"), sr=16000)
y = np.append(y[0], y[1:] - 0.97 * y[:-1])                 # pre-emphasis

mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=256, n_mels=64)
log_mel = librosa.power_to_db(mel)                          # (64, frames)  ← CNN 입력용
mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=13)           # (13, frames)

# 동적 특징: 시간에 따른 변화량
delta = librosa.feature.delta(mfcc)                         # 1차 (속도)
delta2 = librosa.feature.delta(mfcc, order=2)               # 2차 (가속도)
features = np.vstack([mfcc, delta, delta2])                 # (39, frames)
```

PyTorch에서는 `torchaudio.transforms.MelSpectrogram`, `torchaudio.transforms.MFCC`로 GPU에서 바로 계산할 수 있습니다.

### MFCC vs Mel-Spectrogram

| | MFCC | Mel-Spectrogram |
|---|---|---|
| 마지막 단계 | DCT 적용 | DCT 이전 |
| 주파수 간 관계 | **De-correlate** (상관관계 제거) | **Correlate** (이웃 주파수끼리 상관) |
| 연산량·크기 | 적음 (13개 정도) | 많음 (64~128개) |
| 적합한 모델 | GMM, HMM, MLP 등 전통적 모델, 일반 도메인 | **CNN** (2D 패턴을 학습), 특정 도메인 |

CNN은 이웃한 값들 사이의 지역적 패턴을 찾는 모델이라, 이웃 주파수의 상관관계가 살아 있는 **Mel-Spectrogram이 더 잘 맞습니다.** 반대로 DCT로 상관관계를 줄인 MFCC는 전통적인 모델에 유리합니다. 예전 음성 인식의 표준이던 GMM-HMM은 계산을 줄이려고 공분산 행렬의 대각 성분만 쓰는(차원끼리 독립이라고 가정하는) 경우가 많았는데, MFCC는 이 가정에 잘 맞았습니다.

### MFCC의 약점

MFCC는 특정 주파수 대역의 **잡음에 취약**합니다. 그래서 CNN 필터로 잡음을 걸러내거나, Attention으로 중요한 특징을 강조하는 방식을 함께 씁니다([13편]({% post_url 2026-06-28-edge-ai-hardware %})의 온디바이스 음성 인식 구상 참고).

## 4. 신경망 기반 특징 추출

사람이 설계한 MFCC 대신, **신경망이 원시 음성(raw waveform)에서 직접 특징을 학습**하는 방법들입니다.

### Wav2Vec / VQ-Wav2Vec

- **Label 없는 대량의 음성**으로 표현(representation)을 먼저 학습합니다. "이 frame 다음에 올 진짜 frame은 무엇인가"를 가짜 후보들 사이에서 맞히는 **contrastive** 방식입니다.
- 그다음 **소량의 labeled 데이터로 fine-tuning**합니다.
- VQ-Wav2Vec은 연속적인 표현을 **이산적인 코드(vector quantization)**로 바꿔, NLP의 BERT 같은 모델을 그대로 쓸 수 있게 합니다. 이후 wav2vec 2.0으로 발전했습니다.

### SincNet

CNN의 첫 번째 층을 **sinc 함수 기반의 band-pass 필터**로 제한합니다. 일반 conv 필터는 모든 값을 학습하지만, SincNet은 필터마다 **낮은 차단 주파수와 높은 차단 주파수 두 개만** 학습합니다.

- 파라미터가 매우 적고 해석 가능
- 첫 층이 의미 있는 주파수 필터가 되도록 강제
- **화자 인식(speaker recognition)**에 특화

### PASE (Problem-Agnostic Speech Encoder)

하나의 **encoder**가 원시 음성을 frame 벡터로 바꾸고, 여러 개의 작은 네트워크(**worker**)가 이 벡터로 서로 다른 자기지도 과제(파형 복원, MFCC 예측, 화자 정보 구분 등)를 풉니다. 여러 과제를 동시에 풀어야 하므로 encoder는 **특정 문제에 치우치지 않은(problem-agnostic)** 일반적인 음성 표현을 배우게 됩니다.

---

## 5. 사례 1: 경량 End-to-End 심음 분류

*Lightweight End-to-End Neural Network Model for Automatic Heart Sound Classification* (Li et al., Information 2021)

**목표**: 파라미터가 매우 적고 정확도가 비교적 높은 모델을 만들어, **휴대용 의료 장비**에서 정상/비정상 심음을 분류합니다.

### 전체 흐름

$$
\text{Heart sound} \rightarrow \text{STFT} \rightarrow \text{CNN} \rightarrow \text{Softmax} \rightarrow \text{정상 / 비정상}
$$

### 특징 추출

- 심장은 0.6~1초에 한 번 뛰므로 **3초씩** 잘라(segmentation) 한 구간에 최소 한 번의 박동이 들어가게 함
- 구간을 **50% 겹치게** 자르고 STFT로 spectrum 추출
- STFT의 window 길이(= FFT 길이)를 바꿔 가며 실험한 결과 **150샘플**(`n_fft = win_length = 150`, `hop_length = 75`)일 때 정확도가 가장 높음. 사용한 PhysioNet/CinC Challenge 2016 데이터는 2,000Hz로 제공되므로, 약 75ms 길이의 window를 37.5ms씩 옮긴 셈입니다

### 모델

- **3개의 conv 층**, 각 층은 **Conv → BN → ReLU**
- 마지막에 **Max pooling**으로 FC 층의 파라미터 수를 크게 줄이고 과적합 방지
- BN으로 수렴 속도 향상
- Softmax로 확률 출력

### Focal Loss로 불균형 해결

임상 데이터는 정상과 비정상 샘플 수가 크게 차이 납니다. 균형을 맞추지 않으면 한쪽 클래스만 잘 맞히고 다른 쪽은 거의 못 맞힙니다. 그래서 [Object Detection 글]({% post_url 2026-06-07-object-detection %})에서 본 **Focal loss**를 씁니다.

$$
\text{FL}(p_t) = -\alpha_t(1 - p_t)^\gamma\log(p_t)
$$

$$\alpha$$로 두 클래스의 비중을, $$(1-p_t)^\gamma$$로 이미 잘 맞히는 쉬운 샘플의 비중을 낮춥니다. 이 논문의 식에서는 **정상(y=0) 샘플에 $$\alpha$$, 비정상(y=1) 샘플에 $$1-\alpha$$**를 곱합니다. 그래서 아래 결과의 $$\alpha = 0.2$$는 비정상 샘플의 loss에 0.8, 정상 샘플에 0.2의 가중치를 준다는 뜻이고, $$\alpha = 0.5$$는 두 클래스를 똑같이 다루는(가중치 없는) 경우입니다.

### 녹음 단위 판정

하나의 녹음은 여러 구간으로 나뉘므로, 구간별 결과를 모아 최종 판정합니다.

$$
\frac{\text{비정상(1)으로 분류된 구간 수}}{\text{전체 구간 수}} \ge \text{threshold} \;\Rightarrow\; \text{비정상}
$$

### 평가 지표

| 지표 | 의미 |
|---|---|
| Sensitivity (Sn) | 실제 비정상 중 비정상으로 맞힌 비율 (= Recall) |
| Specificity (Sp) | 실제 정상 중 정상으로 맞힌 비율 |
| Accuracy (Acc) | 전체 정답 비율 |
| **MAcc** | $$(\text{Sn} + \text{Sp}) / 2$$, 두 클래스의 균형 |

논문은 불균형 데이터에서 Accuracy가 착시를 줄 수 있어 **MAcc를 중요하게** 봤습니다.

### 결과

- $$\alpha = 0.2$$, threshold $$= 0.45$$일 때 Acc와 MAcc가 높고 Sn과 Sp의 차이가 작아 **일반화 능력이 가장 좋음**
- $$\alpha$$를 쓰지 않으면(0.5) MAcc가 약 2%p 낮고, 특히 **Sn이 크게 떨어짐** (비정상을 잘 못 찾음)
- 정확도는 다른 방법과 비슷하지만 **파라미터 수가 훨씬 적고**, 특징 추출이 단순해 **하드웨어 구현에 유리**

## 6. 사례 2: 개선된 MFCC + CRNN 심음 분류

*Heart sound classification based on improved MFCC features and convolutional recurrent neural networks*

### 전체 흐름

전처리된 신호 → **동적 MFCC** 추출 → **CRNN** 학습 → 분류

전처리(저주파 잡음, baseline wandering, 고주파 간섭 제거)는 이미 잘 정립되어 있어 논문에서는 다루지 않고, 특징 추출과 패턴 인식에 집중합니다.

### 개선된 MFCC: Dynamic MFCC

일반 MFCC는 각 frame의 **정적 특성**만 반영합니다. 심음은 시간에 따른 변화가 중요하므로 1차, 2차 차분을 추가해 **동적 특성**을 담습니다.

$$
\Delta c_t = \frac{\sum_{n=1}^{N} n\,(c_{t+n} - c_{t-n})}{2\sum_{n=1}^{N} n^2}
$$

같은 식을 $$\Delta c_t$$에 한 번 더 적용하면 2차 차분 $$\Delta^2 c_t$$를 얻습니다(위 코드의 `delta`, `delta2`).

### CRNN

$$
[\text{Conv} \to \text{MaxPool} \to \text{BN}] \times 3 \rightarrow \text{LSTM} \rightarrow \text{FC} \rightarrow \text{Dropout} \rightarrow \text{Softmax}
$$

- CNN이 **공간적(주파수) 특징**을, LSTM이 **시간적 특징**을 학습
- Conv 블록에는 residual 연결, BN과 dropout을 함께 사용

### PRCNN (Parallel)

CRNN은 CNN을 거친 뒤 LSTM을 쓰기 때문에, **원 신호의 시간적 특징이 CNN을 지나며 일부 손실**될 수 있습니다. PRCNN은 MFCC를 **CNN과 LSTM에 병렬로** 넣고, 두 출력을 하나의 특징 벡터로 합쳐 분류합니다.

### 결과

- LSTM을 쓴 버전(-a)이 GRU를 쓴 버전(-b)보다 좋았고, **CRNN-a와 PRCNN-a**가 정확도와 F1에서 가장 좋음
- 결론적으로 **3개의 conv 층 + 1개의 LSTM 층** 구성이 최상의 결과
- 다른 SOTA 방법, 다른 MFCC 기반 분류기보다 높은 성능

```python
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, n_feat=39, n_cls=2):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d((2, 1)), nn.BatchNorm2d(o))
        self.cnn = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.lstm = nn.LSTM(64 * (n_feat // 8), 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.Dropout(0.3), nn.Linear(32, n_cls))

    def forward(self, x):                      # x: (B, 1, n_feat, T)
        f = self.cnn(x)                        # (B, 64, n_feat/8, T)  주파수 축만 줄임
        f = f.flatten(1, 2).transpose(1, 2)    # (B, T, 64 * n_feat/8)
        _, (h, _) = self.lstm(f)
        return self.head(h[-1])
```

Pooling을 **주파수 축으로만** 해서 시간 해상도를 유지한 뒤 LSTM에 넣는 것이 포인트입니다. 위 코드는 논문 구조를 단순화한 예시입니다(residual 연결 생략). 입력 크기를 넣어 확인해 보면 다음과 같습니다.

```python
import torch

model = CRNN()
x = torch.randn(8, 1, 39, 188)  # 배치 8, 39차원 동적 MFCC, 188 frame (3초, hop 256 @16kHz)
print(model.cnn(x).shape)       # torch.Size([8, 64, 4, 188])  주파수 39 → 4, 시간 188 유지
print(model(x).shape)           # torch.Size([8, 2])
print(sum(p.numel() for p in model.parameters()))  # 108098
```

## 7. 정리

| 단계 | 방법 | 핵심 |
|---|---|---|
| 시간–주파수 변환 | STFT, Spectrogram | 겹치는 frame + window, 길이에 따른 해상도 trade-off |
| 사람 귀 반영 | Mel scale, Mel filter bank | 저주파는 촘촘하게 |
| 압축된 특징 | MFCC (log + DCT) | 상관관계 제거, 13차원 내외 |
| 시간 변화 | Δ, ΔΔ (dynamic MFCC) | 정적 특징의 한계 보완 |
| 학습된 특징 | Wav2Vec, SincNet, PASE | 원시 파형에서 직접 학습 |
| 모델 | CNN, CRNN, PRCNN | 주파수 패턴(CNN) + 시간 패턴(RNN) |
| 불균형 | Focal loss, MAcc | 소수 클래스 보호 |

다음 마지막 글에서는 이런 모델들을 실제로 학습시킬 때 필요한 **PyTorch 실전 도구들**을 정리합니다.

## References

- Davis & Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," 1980. (MFCC)
- Schneider et al., "wav2vec: Unsupervised Pre-training for Speech Recognition," 2019. / Baevski et al., "vq-wav2vec," ICLR 2020.
- Ravanelli & Bengio, "Speaker Recognition from Raw Waveform with SincNet," SLT 2018.
- Pascual et al., "Learning Problem-agnostic Speech Representations from Multiple Self-supervised Tasks (PASE)," Interspeech 2019.
- Li, Yin, Ma, Zhang & Liu, "Lightweight End-to-End Neural Network Model for Automatic Heart Sound Classification," Information 12(2), 54, 2021. [doi:10.3390/info12020054](https://doi.org/10.3390/info12020054)
- Deng et al., "Heart sound classification based on improved MFCC features and convolutional recurrent neural networks," Neural Networks 2020.
- Haytham Fayek, "Speech Processing for Machine Learning: Filter banks, MFCCs and What's In-Between" — <https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html>
- [librosa – `mel_frequencies` (htk 옵션)](https://librosa.org/doc/latest/generated/librosa.mel_frequencies.html)
- <https://ratsgo.github.io/speechbook/docs/neuralfe>
