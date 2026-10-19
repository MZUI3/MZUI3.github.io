---
title: "[AI 기초 1] 머신러닝의 큰 그림: 지도학습, 비지도학습, 강화학습"
date: 2026-05-05 21:29:00 +0900
categories: [Neural Network, Basics]
tags: [machine-learning, supervised-learning, unsupervised-learning, reinforcement-learning, regression, classification, clustering, q-learning, dqn]
description: 머신러닝의 세 가지 학습 방식(지도·비지도·강화학습)과 가설, 비용 함수, 경사 하강법, 로지스틱 회귀와 Cross Entropy, K-Means, Q-Learning까지 딥러닝을 공부하기 전에 필요한 기본 개념을 코드와 함께 정리합니다.
math: true
---

> **AI 기초 시리즈**는 2022년 연구실 세미나에서 다룬 내용을 주제별로 묶어, 딥러닝 논문을 읽기 전에 알아야 할 기본 개념을 정리합니다. 첫 글에서는 머신러닝이 데이터를 학습하는 세 가지 방식과, 이후 모든 글에서 반복해서 나올 **가설(Hypothesis), 비용 함수(Cost), 경사 하강법(Gradient Descent)**을 정리합니다.

## 시리즈 목차

**Part 1. 학습의 원리**
1. **머신러닝의 큰 그림** ← 이번 글
2. [신경망과 역전파]({% post_url 2026-05-08-neural-network-backpropagation %})
3. [활성화 함수 총정리]({% post_url 2026-05-13-activation-functions %})
4. [Optimizer와 Learning Rate Scheduler]({% post_url 2026-05-15-optimizer-and-scheduler %})
5. [과적합과 일반화: Regularization, Dropout, Data Augmentation]({% post_url 2026-05-18-overfitting-regularization-augmentation %})
6. [Normalization 총정리]({% post_url 2026-05-23-normalization %})

**Part 2. 모델 구조**
7. [CNN 기초: Convolution부터 Depthwise Separable까지]({% post_url 2026-05-25-cnn-basics %})
8. [CNN 아키텍처 계보: DenseNet, SENet, ShuffleNet, RepVGG]({% post_url 2026-06-02-cnn-architectures %})
9. [Object Detection 기초: R-CNN부터 YOLO까지]({% post_url 2026-06-06-object-detection %})
10. [RNN, LSTM, GRU]({% post_url 2026-06-08-rnn-lstm-gru %})

**Part 3. 경량화와 하드웨어**
11. [경량 Vision Transformer 동향]({% post_url 2026-06-13-lightweight-vision-transformer %})
12. [이진 신경망(BNN) 총정리]({% post_url 2026-06-17-binary-neural-networks %})
13. [Edge AI를 위한 하드웨어와 온디바이스 학습]({% post_url 2026-06-19-edge-ai-hardware %})

**Part 4. 응용과 실습**
14. [음성·소리 분류 기초: MFCC부터 CRNN까지]({% post_url 2026-06-22-audio-classification %})
15. [PyTorch 실전 도구 모음]({% post_url 2026-06-25-pytorch-training-toolkit %})

> 이 시리즈와 별도로 [ResNet]({% post_url 2026-05-28-resnet %}), [MobileNet]({% post_url 2026-05-30-mobilenet %}), [Pruning & Quantization]({% post_url 2026-06-15-pruning-and-quantization %}), [Transformer]({% post_url 2026-06-10-attention-is-all-you-need %})는 각각 독립된 글로 정리했습니다.

---

## 1. 세 가지 학습 방식

| | 지도학습 | 비지도학습 | 강화학습 |
|---|---|---|---|
| 데이터 | 입력 + **정답(Label)** | 입력만 | 환경과의 상호작용 |
| 목표 | 입력 → 정답 매핑 학습 | 데이터 안의 구조·패턴 발견 | 누적 보상 최대화 |
| 대표 과제 | 회귀, 분류 | 군집화, 연관 규칙, 차원 축소 | 게임, 로봇 제어 |
| 예시 | 공부 시간 → 시험 점수 | 구매 성향별 고객 그룹 | 알파고, 자율주행 |

용어 두 가지를 먼저 짚고 갑니다.

- **Label**: 데이터의 정답 (예: 이 사진은 "고양이")
- **Labeling**: 데이터에 Label을 붙이는 작업. 지도학습의 성능은 **양질의 Labeling**에 크게 좌우됩니다.

---

## 2. 지도학습 (Supervised Learning)

정답이 붙은 데이터(Training set)로 학습합니다. 출력이 무엇이냐에 따라 두 갈래로 나뉩니다.

- **회귀(Regression)**: **연속적인 값**을 예측 (설탕 섭취량 → 혈압 수치)
- **분류(Classification)**: **정해진 카테고리** 중 하나를 예측 (사진 → 개 / 고양이)

### 2.1 Linear Regression

데이터가 대략 직선 형태라면, 데이터를 가장 잘 설명하는 직선을 찾으면 됩니다.

$$
H(x) = Wx + b
$$

- $$x$$: feature (독립 변수, 영향을 주는 값)
- $$y$$: label (종속 변수, 알고 싶은 값)
- $$W$$: weight (가중치), $$b$$: bias (편향)

"가장 잘 설명한다"를 수치로 바꾼 것이 **비용 함수(Cost / Loss function)**입니다. 실제 데이터와 직선 사이의 거리를 재서, 이 값이 작을수록 좋은 모델입니다.

$$
\text{MSE}(W, b) = \frac{1}{m}\sum_{i=1}^{m}\big(H(x^{(i)}) - y^{(i)}\big)^2
$$

### L1 Loss vs L2 Loss

| | L1 (MAE) | L2 (MSE) |
|---|---|---|
| 식 | $$\sum \lvert y - \hat{y} \rvert$$ | $$\sum (y - \hat{y})^2$$ |
| 이상치(outlier)에 대한 반응 | 둔감 | 민감 (오차가 제곱으로 커짐) |
| 언제 쓰나 | 이상치의 영향을 줄이고 싶을 때 | 큰 오차를 강하게 벌주고 싶을 때 |
| 미분 | 0에서 미분 불가 | 어디서나 매끄럽게 미분 가능 |

### 2.2 Gradient Descent: 비용이 가장 낮은 곳 찾기

MSE를 $$W$$에 대한 그래프로 그리면 **아래로 볼록한 2차 함수**가 됩니다. 가장 낮은 지점을 찾는 방법은 간단합니다. **현재 위치의 기울기를 보고, 내려가는 방향으로 조금씩 이동**하면 됩니다.

$$
W := W - \alpha \frac{\partial}{\partial W}\text{Cost}(W)
$$

- $$\alpha$$: **학습률(learning rate)**, 한 번에 얼마나 움직일지 정하는 보폭
- $$:=$$: "할당한다"는 뜻

학습률이 너무 크면 최저점을 지나쳐 발산하고, 너무 작으면 학습이 끝나지 않습니다. 이 보폭을 똑똑하게 조절하는 방법은 [4편 Optimizer]({% post_url 2026-05-15-optimizer-and-scheduler %})에서 자세히 다룹니다.

```python
import numpy as np

# y = 2x + 1 근처의 데이터
rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 100)
y = 2 * x + 1 + rng.normal(0, 1, 100)

W, b, lr = 0.0, 0.0, 0.01
for step in range(2000):
    pred = W * x + b
    dW = 2 * np.mean((pred - y) * x)   # ∂MSE/∂W
    db = 2 * np.mean(pred - y)         # ∂MSE/∂b
    W -= lr * dW
    b -= lr * db

print(f"W={W:.3f}, b={b:.3f}")          # W=1.987, b=0.992
```

노이즈가 섞인 데이터라 정확히 2, 1이 나오지는 않지만 실제 값에 가깝게 찾아갑니다. 학습률을 0.1로 올려 보면 값이 발산하는 것도 직접 확인할 수 있습니다.

### 2.3 Multi-variable Regression

입력이 여러 개라면 $$H(x_1, x_2, x_3) = w_1x_1 + w_2x_2 + w_3x_3 + b$$처럼 늘어납니다. 변수가 많아지면 식이 길어지므로 **행렬**로 씁니다.

$$
H(X) = XW
$$

$$X$$의 각 행은 데이터 하나(instance), 각 열은 변수 하나(variable)입니다. bias $$b$$는 $$X$$에 값이 항상 1인 열을 하나 추가하고 $$W$$에 한 행을 더해 행렬 곱 안에 합쳐 씁니다. 딥러닝에서 데이터를 배치로 묶어 행렬 곱 한 번에 처리하는 것도 이 표현 덕분입니다.

### 2.4 Logistic Regression: 0과 1로 판단하기

"스팸인가 아닌가"처럼 결과가 0 또는 1이어야 할 때, $$H(x) = Wx+b$$는 0보다 작거나 1보다 큰 값이 나올 수 있어 곤란합니다. 그래서 출력을 0~1로 눌러주는 **Sigmoid(Logistic) 함수**를 씌웁니다.

$$
H(x) = \sigma(Wx + b) = \frac{1}{1 + e^{-(Wx+b)}}
$$

출력이 0.5 이상이면 1, 미만이면 0으로 판단합니다.

#### 왜 MSE를 쓰지 않을까

Sigmoid를 거친 가설에 MSE를 쓰면 비용 함수가 **울퉁불퉁한 비볼록(non-convex)** 모양이 되어, 시작점에 따라 다른 지역 최솟값에 빠질 수 있습니다. 그래서 **최대우도법(Maximum Likelihood Estimation)**에서 유도한 비용 함수를 씁니다.

$$
\text{Cost}(W) = -\frac{1}{m}\sum_{i=1}^{m}\Big[y\log H(x) + (1-y)\log\big(1 - H(x)\big)\Big]
$$

직관적으로 보면

- 정답이 1인데 $$H(x) \to 1$$이면 비용 0, $$H(x) \to 0$$이면 비용 → ∞
- 정답이 0인데 $$H(x) \to 0$$이면 비용 0, $$H(x) \to 1$$이면 비용 → ∞

틀릴수록 비용이 폭발적으로 커지고, 이 함수는 볼록해서 경사 하강법으로 안정적으로 최적화됩니다. 이것이 **Binary Cross Entropy**입니다.

### 2.5 분류: Binary vs Multi-class

- **Binary Classification**: 두 그룹으로 나눔. 로지스틱 회귀로 **결정 경계(decision boundary)**를 긋습니다.
- **Multi-class Classification**: 세 개 이상으로 나눔. 클래스마다 점수를 계산한 뒤 **Softmax**로 확률로 바꿉니다.

$$
\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}
$$

모든 확률의 합이 1이 되고, 가장 큰 확률의 클래스를 예측값으로 고릅니다(argmax). 정답 label은 해당 클래스만 1이고 나머지는 0인 벡터로 표현하는데, 이를 **one-hot encoding**이라고 합니다(예: 3개 클래스 중 2번 → `[0, 1, 0]`). 비용 함수는 Binary CE를 일반화한 **Cross Entropy**입니다.

$$
\text{CE} = -\sum_{k} y_k \log \hat{y}_k
$$

Cross Entropy는 원래 **두 확률 분포의 차이**를 재는 값입니다. 딥러닝에서는 "실제 정답 분포"와 "모델이 예측한 분포"의 차이를 줄이는 데 씁니다.

---

## 3. 비지도학습 (Unsupervised Learning)

Label 없이 데이터끼리의 **패턴과 유사성**을 학습합니다. Labeling 비용이 들지 않고, 비슷한 것끼리 묶거나 숨은 구조를 찾을 때 씁니다.

> **분류 vs 군집화**: 분류는 정답 그룹이 정해져 있고 "어디에 속하는가"를 맞히는 것, 군집화는 정답 없이 "비슷한 것끼리 그룹을 만드는" 것입니다.

### 3.1 Clustering: K-Means

1. 군집 수 $$k$$를 정하고, 임의의 중심점(centroid) $$k$$개를 놓는다.
2. 각 데이터를 **유클리드 거리**가 가장 가까운 중심점에 배정한다.

   $$
   d(p, q) = \sqrt{\sum_i (p_i - q_i)^2}
   $$

3. 각 군집의 평균 위치로 중심점을 옮긴다.
4. 중심점이 더 이상 움직이지 않을 때까지 2–3을 반복한다.

```python
from sklearn.cluster import KMeans
import numpy as np

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(size=(50, 2)) + c for c in ([0, 0], [5, 5], [0, 5])])  # 중심이 다른 세 무리
km = KMeans(n_clusters=3, n_init="auto", random_state=0).fit(X)
print(km.cluster_centers_.round(2))
print(np.bincount(km.labels_))        # 군집별 데이터 수
```

```text
[[ 4.83  5.05]
 [-0.41  5.06]
 [-0.01  0.18]]
[51 49 50]
```

(scikit-learn 1.2 이상, 1.9에서 실행) 실제 중심 (5, 5), (0, 5), (0, 0)을 잘 찾았습니다. 군집 번호(0, 1, 2)의 순서는 의미가 없고 실행 환경에 따라 바뀔 수 있습니다.

**한계**: $$k$$를 미리 정해야 하고, 이상치에 민감하며, 처음 중심점을 어디에 놓느냐에 따라 결과가 달라집니다(그래서 scikit-learn은 기본으로 **k-means++** 초기화를 쓰고, `n_init`만큼 여러 번 돌려 가장 좋은 결과를 고릅니다). 원형이 아닌 군집도 잘 찾지 못합니다.

### 3.2 Association: 연관 규칙

"A를 산 사람은 B도 사더라" 같은 규칙을 찾습니다. 장바구니 분석이 대표적입니다(Apriori, Eclat 등).

| 지표 | 식 | 의미 |
|---|---|---|
| 지지도 (Support) | $$P(A \cap B)$$ | 두 항목이 함께 나타나는 빈도 |
| 신뢰도 (Confidence) | $$P(B \mid A) = \dfrac{P(A \cap B)}{P(A)}$$ | A를 샀을 때 B도 살 확률 |
| 향상도 (Lift) | $$\dfrac{P(B \mid A)}{P(B)}$$ | 1이면 우연, 1보다 크면 양의 연관, 작으면 음의 연관 |

신뢰도만 보면 원래 잘 팔리는 상품(B)이 늘 높게 나오기 때문에, **향상도로 "우연인지"를 함께 확인**해야 합니다.

---

## 4. 강화학습 (Reinforcement Learning)

정답 대신 **보상(reward)**으로 배웁니다. 잘하면 보상을, 못하면 벌을 받으면서 **에이전트(Agent)**가 **환경(Environment)**에서 누적 보상이 가장 큰 **행동(Action)**을 찾아갑니다.

| 용어 | 의미 |
|---|---|
| State $$s$$ | 현재 상황 |
| Action $$a$$ | 에이전트가 할 수 있는 행동 |
| Reward $$r$$ | 행동에 대한 피드백 |
| Policy $$\pi$$ | 상태에서 행동을 고르는 규칙 |
| Discount $$\gamma$$ | 미래 보상을 얼마나 할인할지 ($$0 \le \gamma < 1$$) |

### Q-Learning

각 상태–행동 쌍이 얼마나 좋은지를 **Q-value**로 기록한 표(Q-table)를 학습합니다.

$$
Q(s, a) \leftarrow Q(s, a) + \alpha\Big[r + \gamma\max_{a'}Q(s', a') - Q(s, a)\Big]
$$

"지금 받은 보상 + 다음 상태에서 기대할 수 있는 최선의 가치"를 목표로 Q 값을 조금씩 고칩니다. $$\gamma$$가 있어서 당장의 보상보다 **먼 미래에 더 좋은 결과를 가져오는 행동**도 고를 수 있습니다.

### 에이전트를 이루는 세 요소

| 요소 | 역할 | 예시 |
|---|---|---|
| **Policy** $$\pi(a \mid s)$$ | 상태에서 어떤 행동을 할지 결정 | 결정적(deterministic) 또는 확률적(stochastic) |
| **Value function** | 상태(또는 상태–행동)의 미래 가치 예측 | $$V(s)$$, $$Q(s,a)$$ |
| **Model** | 환경이 다음에 어떻게 변할지에 대한 에이전트의 예측 | 전이 확률 $$P(s' \mid s, a)$$ |

**Discounted future reward**는 보상이 늦게 오는 문제를 다룹니다.

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

같은 보상이라도 **더 빨리 얻는 경로**의 가치가 높아지므로, 에이전트는 돌아가지 않는 길을 선택하게 됩니다.

### Exploitation vs Exploration

항상 Q 값이 가장 큰 행동(greedy action)만 고르면(**exploitation**) 처음 발견한 그럭저럭 좋은 길에 갇힙니다. 가끔은 새로운 행동도 시도해야(**exploration**) 더 좋은 길을 찾을 수 있습니다.

- **ε-greedy**: 확률 $$\varepsilon$$으로 무작위 행동, $$1-\varepsilon$$으로 greedy 행동 (예: $$\varepsilon = 0.1$$이면 10번 중 1번 탐험)
- **Decaying ε-greedy**: 학습 초반에는 많이 탐험하고, 점점 $$\varepsilon$$을 줄여 나감
- **Random noise**: Q 값에 노이즈를 더해 argmax를 흔듦

```python
import numpy as np

def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))     # explore
    return int(np.argmax(q_values))                 # exploit

# 에피소드가 진행될수록 탐험 비율 감소
for episode in range(1000):
    epsilon = max(0.01, 1.0 / (episode / 100 + 1))
```

### 알고리즘을 나누는 두 가지 기준

- **Model-free vs Model-based**: 환경 모델(다음 상태와 보상의 예측)을 쓰는가. Q-learning, PPO는 model-free, AlphaZero·MuZero는 model-based입니다.
- **Value-based vs Policy-based**: 가치 함수를 학습해 행동을 고르는가(DQN), 정책을 직접 학습하는가(REINFORCE, PPO). 둘을 함께 쓰면 **Actor-Critic**입니다.
- 비슷하게 자주 나오는 구분으로 **On-policy vs Off-policy**가 있습니다. 지금 학습 중인 정책이 직접 모은 데이터로만 배우면 on-policy(PPO), 다른 정책이 모은 데이터(replay buffer)로도 배우면 off-policy(Q-learning, DQN)입니다.

### DQN

상태가 게임 화면처럼 엄청나게 많으면 Q-table을 만들 수 없습니다. **DQN**(DeepMind, 2015)은 Q-table을 **심층 신경망**으로 대체해 Atari 게임을 화면만 보고 학습했습니다. 이후 강화학습 시리즈의 [TRPO·PPO 글]({% post_url 2026-07-05-trpo-to-ppo %})로 이어집니다.

---

## 5. 정리

| 개념 | 한 줄 요약 |
|---|---|
| 가설 $$H(x)$$ | 입력을 출력으로 바꾸는 모델의 식 |
| 비용 함수 | 예측이 정답과 얼마나 다른지 재는 값 (MSE, CE) |
| 경사 하강법 | 비용의 기울기 반대 방향으로 파라미터를 조금씩 이동 |
| 회귀 / 분류 | 연속값 예측 / 카테고리 예측 |
| Sigmoid / Softmax | 점수를 확률로 변환 (2개 / 여러 개 클래스) |
| 군집화 / 연관 | Label 없이 그룹 만들기 / 함께 일어나는 패턴 찾기 |
| 강화학습 | 보상을 최대화하는 행동 정책 학습 |

다음 글에서는 선형 모델을 여러 층으로 쌓은 **신경망**과, 그 신경망을 학습시키는 **역전파(Backpropagation)**를 숫자 예제로 직접 계산해 봅니다.

## References

- 김성훈, 모두를 위한 딥러닝 강좌
- Mnih et al., "Human-level control through deep reinforcement learning," Nature 2015.
- scikit-learn documentation — KMeans
