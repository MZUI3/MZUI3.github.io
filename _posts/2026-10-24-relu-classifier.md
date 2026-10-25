---
title: "[Deep Learning] Deep Learning using ReLU: 소프트맥스 대신 ReLU를 분류층에 쓰면 어떻게 되는가"
date: 2026-10-24 14:40:00 +0900
categories: [Deep Learning, Fundamentals]
tags: [relu, softmax, classification-layer, cross-entropy, dying-relu, mnist, fashion-mnist, agarap]
description: Agarap(2018)이 신경망의 마지막 분류층을 소프트맥스 대신 ReLU로 바꾸고 MNIST, Fashion-MNIST, WDBC에서 비교한 짧은 실험 보고서를 정리합니다. 제안의 내용, 결과(FFNN에서는 대등, CNN과 WDBC에서는 느린 수렴과 낮은 정확도), 저자가 든 죽은 ReLU 설명을 다루고, 작은 실험으로 ReLU 분류층의 손실이 왜 학습을 방해하는지 확인합니다. 결론은 이 아이디어를 쓰지 말아야 할 이유에 가깝습니다.
math: true
---

> [활성 함수]({% post_url 2026-05-13-activation-functions %}) 글에서 ReLU는 은닉층의 표준이고, 분류는 소프트맥스 + 교차 엔트로피가 표준이라고 정리했습니다. **Deep Learning using Rectified Linear Units**(Agarap, 2018, arXiv)는 "ReLU를 분류층에도 쓰면 어떨까"를 실험한 짧은 보고서입니다. SVM을 분류층에 쓴 Tang(2013)이나 GRU-SVM 같은 선행 연구의 연장선에 있습니다. 결과는 썩 좋지 않습니다. 완전 연결망에서는 소프트맥스와 비슷하지만 CNN과 WDBC에서는 수렴이 느리고 정확도가 낮으며, 저자 스스로 죽은 뉴런 문제를 원인으로 듭니다. 이 글은 그 내용을 정리하되, **왜** 이 설계가 원리적으로 불리한지를 짧은 실험으로 보충합니다. 널리 인용되는 논문은 아니지만, "분류층은 왜 소프트맥스인가"를 되짚기에 좋은 반면교사입니다.

## 1. 제안

끝에서 두 번째 층의 활성 $$h$$에 가중치 $$\theta$$를 곱해 원점수 $$o = \theta h$$를 얻는 것까지는 같습니다. 소프트맥스는 $$p_k = \exp(o_k)/\sum_j \exp(o_j)$$로 확률을 만들고 $$\hat y = \arg\max_k p_k$$를 예측합니다. 제안은 원점수를 0에서 자르는 것입니다.

$$
\hat y = \arg\max_{i} \max(0, o_i)
$$

손실은 교차 엔트로피를 유지하되 확률 대신 ReLU 출력을 넣습니다(논문 식 6).

$$
\ell(\theta) = -\sum y \cdot \log \max(0, \theta x + b)
$$

역전파는 보통과 같고 Adam으로 학습합니다. 구현은 Keras/TensorFlow이고, MNIST와 Fashion-MNIST는 표준화 후 PCA로 784→256차원으로 줄여 씁니다(이 선택이 비교의 공정성에 문제가 될 수 있다고 저자도 적어 둡니다).

## 2. 실험

두 구조입니다. **FFNN**은 512-512(드롭아웃) 두 은닉층, **CNN**은 Keras의 VGG형(conv 32×2, pool, conv 64×2, pool, dense 256)이고, 마지막 dense 층만 소프트맥스 또는 ReLU입니다. WDBC(유방암 진단, 30특징 569개)는 64-32 FFNN입니다. 지표는 10겹 교차검증 정확도, 테스트 정확도, 정밀도/재현율/F1, 혼동 행렬입니다.

| 데이터 / 모델 | 소프트맥스 CV / 테스트 | ReLU CV / 테스트 | F1 |
|---|---|---|---|
| MNIST FFNN | 99.29% / 97.98% | 98.22% / 97.77% | 0.98 / 0.98 |
| MNIST CNN | 97.23% / 95.36% | 73.53% / 91.74% | 0.95 / 0.92 |
| Fashion-MNIST FFNN | 98.87% / 89.35% | 92.23% / 89.06% | 0.89 / 0.89 |
| Fashion-MNIST CNN | 91.96% / 86.08% | 83.24% / 85.84% | 0.86 / 0.86 |
| WDBC FFNN | 91.21% / 92.40% | 87.96% / 90.64% | 0.92 / 0.90 |

**FFNN**에서는 테스트 정확도가 0.2~0.3%p 낮지만 F1이 같아 저자는 "대등"이라고 봅니다. 혼동 행렬도 클래스별로 엎치락뒤치락합니다. **CNN**에서는 ReLU 분류층이 **수렴이 훨씬 느립니다.** MNIST CNN의 10겹 학습 정확도는 1겹 33%에서 10겹 92%로 서서히 오르고(소프트맥스는 처음부터 높음), 교차검증 평균이 73.5%에 그칩니다. 테스트 정확도는 91.7%로 소프트맥스보다 약 4%p 낮습니다. Fashion-MNIST CNN도 같은 패턴이고 격차는 작습니다. **WDBC**에서도 ReLU가 느리게 수렴하고 1.8%p 낮으며, 소프트맥스는 거짓 음성이, ReLU는 거짓 양성이 더 많습니다.

저자의 결론은 "불리한 결과는 아마 **죽은 ReLU** 때문"이라는 것입니다. 기울기가 뉴런을 통해 흐르지 않아 뉴런이 멈추고, 학습이 방해받는다는 설명이고, PELU 같은 ReLU 변형으로 해결될 수 있다고 봅니다. 그러면서도 "DL-ReLU 모델은 여전히 소프트맥스 기반과 비교할 만하거나 더 낫다"고 적는데, 표의 수치가 이를 뒷받침하지는 않습니다.

## 3. 코드로 보기: ReLU 분류층의 손실은 무엇을 하는가

같은 MLP를 (a) 소프트맥스 + 교차 엔트로피, (b) ReLU 분류층으로 학습해 테스트 정확도와, 마지막 배치에서 모든 점수가 0으로 잘린 표본의 비율을 봅니다. 식 (6)을 문자 그대로 쓰면 점수를 키우기만 해도 손실이 음수로 발산하므로, toy에서는 0으로 자른 점수를 합이 1이 되게 정규화한 뒤 음의 로그 가능도를 취하는 가장 관대한 변형을 씁니다.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); torch.set_num_threads(1)
Wtrue = torch.randn(32, 10)
def data(n, seed):
    g = torch.Generator().manual_seed(seed); x = torch.randn(n, 32, generator=g)
    return x, (x @ Wtrue + 1.0 * torch.randn(n, 10, generator=g)).argmax(1)
Xtr, Ytr = data(4000, 1); Xte, Yte = data(2000, 2)
def run(head, steps=600):
    torch.manual_seed(0); net = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 10)); opt = torch.optim.Adam(net.parameters(), 2e-3)
    g = torch.Generator().manual_seed(0); dead = 0
    for s in range(steps):
        i = torch.randint(0, 4000, (128,), generator=g); o = net(Xtr[i])
        if head == "relu":                       # the paper: threshold the raw scores at 0, then a cross-entropy on them
            o = F.relu(o); dead = (o.sum(1) == 0).float().mean().item()
            loss = F.nll_loss(torch.log(o / (o.sum(1, keepdim=True) + 1e-8) + 1e-8), Ytr[i])
        else: loss = F.cross_entropy(o, Ytr[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        o = net(Xte); o = F.relu(o) if head == "relu" else o
        return (o.argmax(1) == Yte).float().mean().item() * 100, dead
for head in ("softmax", "relu"):
    a, dead = run(head)
    print(f"{head:<8} classification head: test accuracy {a:5.1f}%" + (f"   (samples with ALL scores clipped to 0 in the last batch: {dead*100:.1f}%)" if head == "relu" else ""))
print("\nargmax over ReLU(o) equals argmax over o whenever the top score is positive; the head only changes the loss and its gradient")
```

```text
softmax  classification head: test accuracy  81.2%
relu     classification head: test accuracy  63.1%   (samples with ALL scores clipped to 0 in the last batch: 2.3%)

argmax over ReLU(o) equals argmax over o whenever the top score is positive; the head only changes the loss and its gradient
```

- 소프트맥스 81.2% 대 ReLU 63.1%로, 논문의 CNN 결과보다 격차가 큽니다. 마지막 배치에서 2.3%의 표본은 **모든 클래스 점수가 0**으로 잘려 예측이 임의가 되고 기울기도 0입니다. 논문이 말한 죽은 뉴런 현상입니다.
- 정규화를 넣어 준 변형에서도 격차가 큰 이유는 0으로 자르는 연산 자체에 있습니다. 어떤 클래스의 점수가 음수가 되는 순간 그 클래스로는 기울기가 **전혀** 흐르지 않고(소프트맥스는 지수 함수라 항상 흐름), 정답 점수가 잘리면 로그가 발산하거나 기울기가 사라집니다. 논문의 식 (6)은 여기에 더해 분모가 없어 **오답 점수를 낮출 압력이 없으므로** 상황이 더 나쁩니다. 그리고 출력에 관한 한 $$\arg\max \max(0, o) = \arg\max o$$(최고 점수가 양수일 때)이므로 ReLU는 **예측을 바꾸지 않고 손실과 기울기만 나쁘게** 만듭니다.

## 4. 해석

**1. 분류층에 소프트맥스를 쓰는 이유를 역으로 보여 줍니다.** 소프트맥스 교차 엔트로피는 정답의 로그 확률을 최대화하는 동시에 분모를 통해 오답 점수를 누릅니다. 식 (6)은 후자를 버렸고, 그 결과가 느린 수렴과 죽은 출력입니다. "ReLU는 활성 함수이지 확률 모델이 아니다"가 요점이며, 분류층을 바꾸려면 손실도 함께 설계해야 합니다. SVM 분류층(Tang 2013)이 힌지 손실로 **마진**을 넣었던 것과 대비됩니다.

**2. 실험 설계의 교훈도 있습니다.** PCA로 784→256차원으로 줄인 입력을 CNN에 16×16으로 넣은 것, 교차검증과 테스트의 큰 격차(CNN-ReLU CV 73.5% 대 테스트 91.7%), 랩탑 1대에서의 단일 실행 등은 결론의 신뢰도를 낮춥니다. 저자가 "추가 최적화로 대등해질 수 있다"고 적은 부분은 검증되지 않은 추측입니다.

**3. 그래도 읽을 가치는 있습니다.** 짧고 재현 가능하며(코드 공개), 결과를 숨기지 않고 보고합니다. "왜 안 되는가"를 생각하게 만드는 논문이고, [정규화]({% post_url 2026-05-23-normalization %})나 [활성 함수]({% post_url 2026-05-13-activation-functions %})의 역할을 처음 배우는 사람에게 분류층의 설계 원리를 되짚게 하는 데 쓸 수 있습니다.

**4. 평가를 읽을 때 주의할 점.** arXiv 보고서이며 동료 심사를 거치지 않았습니다. 교차검증 수치는 **학습** 정확도이고, 하이퍼파라미터 탐색은 없으며, 시드별 분산도 없습니다. 위 toy는 식 (6)이 아니라 정규화를 더한 관대한 변형을 썼고(그래도 소프트맥스에 크게 뒤짐), 논문의 Keras 구현 세부와 같지는 않습니다.

## 5. 정리

| 항목 | 내용 |
|---|---|
| 제안 | 마지막 층을 $$\arg\max \max(0, \theta h)$$로, 손실은 $$-\sum y \log \max(0, \theta h + b)$$ |
| FFNN | MNIST 97.77 vs 97.98%, Fashion 89.06 vs 89.35%, F1 동일 |
| CNN | MNIST 91.74 vs 95.36%, CV 73.5 vs 97.2%로 수렴이 느림 |
| WDBC | 90.64 vs 92.40% |
| 저자의 설명 | 죽은 ReLU로 기울기가 흐르지 않음 |
| 이 글의 보충 | 손실이 오답 점수를 누르지 않아 원리적으로 불리, 예측은 바꾸지 않고 기울기만 악화 |

## References

- Agarap, "Deep Learning using Rectified Linear Units (ReLU)," 2018. [arXiv:1803.08375](https://arxiv.org/abs/1803.08375)
- Tang, "Deep Learning using Linear Support Vector Machines," ICML 2013 Workshop. [arXiv:1306.0239](https://arxiv.org/abs/1306.0239)
- Nair, Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines," ICML 2010.
- Glorot, Bordes, Bengio, "Deep Sparse Rectifier Neural Networks," AISTATS 2011.
- Trottier, Giguère, Chaib-draa, "Parametric Exponential Linear Unit for Deep Convolutional Neural Networks," ICMLA 2017. [arXiv:1605.09332](https://arxiv.org/abs/1605.09332)
