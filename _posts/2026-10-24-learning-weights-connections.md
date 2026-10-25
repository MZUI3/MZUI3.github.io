---
title: "[Deep Learning] Learning both Weights and Connections: 학습-가지치기-재학습으로 AlexNet을 9배, VGG-16을 13배 줄이다"
date: 2026-10-24 14:00:00 +0900
categories: [Deep Learning, Model Compression]
tags: [pruning, magnitude-pruning, sparsity, retraining, iterative-pruning, energy-efficiency, han-dally]
description: Han, Pool, Tran, Dally(NeurIPS 2015)가 학습 → 작은 가중치 가지치기 → 재학습의 3단계로 AlexNet 61M→6.7M, VGG-16 138M→10.3M 파라미터를 정확도 손실 없이 줄인 과정을 정리합니다. 메모리 접근 에너지라는 동기, L1/L2 정규화의 역할, 드롭아웃 비율 조정, 재초기화 대신 가중치 유지, 반복 가지치기, 층별 민감도를 다루고 작은 실험으로 정규화와 반복의 효과를 확인합니다.
math: true
---

> 2012년 AlexNet의 6,000만 파라미터는 모바일 기기에 올리기에 너무 컸습니다. 문제는 연산보다 **메모리**입니다. 45nm 공정에서 32비트 곱셈은 3.7pJ이지만 DRAM 접근은 640pJ로 약 170배이고, 단순 산술(32비트 정수 덧셈 0.1pJ)보다는 3자리 비쌉니다. 10억 연결 네트워크를 20Hz로 돌리면 DRAM 접근만 12.8W로 모바일 전력 예산을 넘습니다. **Learning both Weights and Connections**(Han, Pool, Tran, Dally, Stanford·NVIDIA, NeurIPS 2015)의 목표는 가중치를 **온칩 SRAM에 들어가도록** 줄이는 것입니다. 방법은 단순합니다. 학습해서 어떤 연결이 중요한지 알아내고, 중요하지 않은 연결을 자르고, 남은 연결을 재학습합니다. 네트워크가 가중치뿐 아니라 **연결 구조도 학습**하는 셈이고, 저자들은 이를 유아기의 시냅스 가지치기에 비유합니다. 결과는 AlexNet 9배, VGG-16 13배 압축에 정확도 손실 없음입니다.

## 1. 방법: 3단계 파이프라인

1. **연결 학습.** 보통처럼 학습하되 목적은 최종 가중치가 아니라 **어떤 연결이 중요한지** 아는 것입니다.
2. **가지치기.** 가중치의 절댓값이 임계값(층 가중치 표준편차 × 품질 계수) 아래인 연결을 모두 제거합니다. 밀집 층이 희소 층이 됩니다.
3. **재학습.** 남은 희소 연결의 가중치를 다시 학습합니다. 이 단계가 **결정적**이어서, 재학습 없이 쓰면 정확도가 크게 떨어집니다.

2~3단계를 반복하면 더 줄일 수 있습니다. 네트워크가 "무엇이 중요한지"를 데이터로 배운다는 점에서 구조가 학습 전에 고정되는 관행과 다릅니다. 입력 또는 출력 연결이 모두 사라진 뉴런은 안전하게 제거되며, 재학습 중 정규화가 죽은 뉴런의 남은 연결도 0으로 밀어 자동으로 정리합니다.

### 설계 세부

- **정규화.** L1은 파라미터를 0 근처로 더 많이 밀어 **가지치기 직후**(재학습 전) 정확도가 좋지만, 남은 연결의 질이 L2만 못해 **재학습 후**에는 L2가 낫습니다. L1로 자르고 L2로 재학습하는 조합도 L2만 쓰는 것을 이기지 못했습니다(한 모드의 파라미터가 다른 모드에 잘 적응하지 않음). 전체적으로 L2가 최선입니다.
- **드롭아웃 비율 조정.** 드롭아웃은 뉴런을 확률적으로 끄지만 추론 때 돌아오고, 가지치기는 영구히 제거합니다. 용량이 줄었으므로 재학습 때 드롭아웃을 줄여야 합니다. 연결 수 $$C_i = N_i N_{i-1}$$이 뉴런 수의 제곱이므로 $$D_r = D_o \sqrt{C_{ir}/C_{io}}$$로 조정합니다.
- **가중치 유지.** 재학습 때 살아남은 연결을 **재초기화하지 않고** 학습된 값에서 시작합니다. CNN에는 깨지기 쉬운 공적응 특징이 있어, 일부 층을 재초기화하고 다시 학습하면 경사 하강이 좋은 해를 찾지 못합니다. 깊은 네트워크의 기울기 소실 때문에 가지치기 오차 회복이 어려우므로 FC를 자를 때는 CONV를 고정하고 FC만 재학습하고, 그 반대도 마찬가지입니다.
- **반복 가지치기.** 한 번에 공격적으로 자르면 AlexNet에서 5배가 한계지만, 가지치기와 재학습을 반복하면 정확도 손실 없이 **9배**까지 갑니다. 각 반복은 탐욕적 탐색입니다. 절댓값에 비례한 확률로 자르는 방식은 더 나빴습니다.

## 2. 실험

| 네트워크 | Top-1 오류 | Top-5 오류 | 파라미터 | 압축률 |
|---|---|---|---|---|
| LeNet-300-100 | 1.64% → 1.59% | - | 267K → 22K | 12× |
| LeNet-5 | 0.80% → 0.77% | - | 431K → 36K | 12× |
| AlexNet | 42.78% → 42.77% | 19.73% → 19.67% | 61M → 6.7M | 9× |
| VGG-16 | 31.50% → 31.34% | 11.32% → 10.88% | 138M → 10.3M | 13× |

**LeNet (MNIST).** 재학습은 원래 학습률의 1/10로 합니다. LeNet-300-100의 첫 FC 층(784×300)의 희소 패턴은 28픽셀 너비의 띠 28개로 나타나고, 비영 가중치는 이미지 **중앙**에 몰립니다. 숫자가 중앙에 쓰이므로 가지치기가 **시각적 주의 영역을 찾아낸** 셈입니다. 가장자리로 가는 연결은 더 많이 잘립니다.

**AlexNet (ImageNet).** 원래 학습은 75시간, 재학습은 원래 학습률의 1/100로 173시간입니다. 가지치기는 배포 직전에 한 번 하는 것이므로 재학습 시간은 큰 문제가 아니라는 입장입니다. 파라미터는 9배, 연산은 3배 줄어듭니다. 층별로 보면 CONV는 35~84%가 남고 FC는 9%(fc1, fc2), 25%(fc3)만 남습니다. 활성의 비영 비율(Act%)도 함께 보고하는데, 희소 활성과 희소 가중치를 곱하면 실제 필요한 FLOP은 더 줄어듭니다(총 30%).

**VGG-16 (ImageNet).** 5회 반복으로 전체의 7.5%(13배)만 남고, 가장 큰 두 FC 층(fc6 103M, fc7 17M)은 각각 **4% 이하**로 줄어듭니다. 실시간 이미지 처리에서 FC 층은 이미지 간 재사용이 없어 이 감소가 특히 중요합니다.

**트레이드오프(Figure 5).** 재학습 없이는 연결의 1/3에서 정확도가 떨어지기 시작하지만, 재학습하면 1/10까지 갑니다. 재학습 없이도 2배는 "공짜"입니다. 반복 가지치기 곡선은 9배에서 손실 0이고 10배부터 급락합니다. 두 지점은 원래보다 약간 **더 정확**한데, 가지치기가 적정 용량을 찾아 과적합을 줄인 것으로 해석합니다.

**층별 민감도(Figure 6).** CONV 층이 FC 층보다 가지치기에 민감하고, 입력 채널이 3개뿐이라 중복이 적은 첫 CONV 층이 가장 민감합니다. 이 민감도로 층별 임계값을 정합니다. 희소 행렬 저장의 오버헤드는 15.6%이고, 상대 인덱스를 쓰면 FC 인덱스는 5비트, CONV는 8비트면 됩니다. 가지치기 후 가중치 분포는 0 근처의 큰 봉우리가 사라지고 재학습으로 더 넓게 퍼진 **이봉** 분포가 됩니다.

**다른 방법과 비교(AlexNet).** 데이터 없는 가지치기 1.5배(정확도 손실 큼), Fastfood 2~3.7배, Collins & Kohli 4배(손실), 층 크기를 단순히 줄이는 Naive Cut 4.4배(4% 손실), SVD 5배(손실)에 비해 9배에 손실 없음입니다.

## 3. 코드로 보기: 정규화, 재학습, 반복의 효과

합성 10클래스 문제에서 256유닛 MLP를 L1 또는 L2 정규화로 학습한 뒤, 가중치 80%와 95%를 크기 기준으로 자르고 재학습 전후 정확도를 비교합니다. 95%는 4라운드 반복 가지치기와도 비교하고, 드롭아웃 조정 규칙을 계산합니다.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); torch.set_num_threads(1)
Wtrue = torch.randn(64, 10) * (torch.rand(64, 10) < 0.2)
def data(n, seed):
    g = torch.Generator().manual_seed(seed); x = torch.randn(n, 64, generator=g)
    return x, (x @ Wtrue + 0.7 * torch.randn(n, 10, generator=g)).argmax(1)
Xtr, Ytr = data(4000, 1); Xte, Yte = data(2000, 2)
def make(): return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 10))
def train(net, steps, reg, lam, masks=None, lr=2e-3, seed=0):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.Adam(net.parameters(), lr)
    for _ in range(steps):
        i = torch.randint(0, 4000, (128,), generator=g); loss = F.cross_entropy(net(Xtr[i]), Ytr[i])
        pen = sum((p.abs().sum() if reg == "L1" else (p ** 2).sum()) for p in net.parameters() if p.dim() == 2)
        opt.zero_grad(); (loss + lam * pen).backward(); opt.step()
        if masks:
            with torch.no_grad():
                for p, m in zip(net.parameters(), masks):
                    if m is not None: p.mul_(m)
def acc(net):
    with torch.no_grad(): return (net(Xte).argmax(1) == Yte).float().mean().item() * 100
def prune(net, frac):
    masks = []
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() == 2:
                thr = p.abs().flatten().kthvalue(int(frac * p.numel())).values; m = (p.abs() > thr).float(); p.mul_(m); masks.append(m)
            else: masks.append(None)
    return masks
print("regularisation during the first training vs. accuracy right after pruning and after retraining")
print("reg   dense    pruned 80%  retrained   pruned 95%  retrained")
for reg, lam in (("L1", 2e-5), ("L2", 2e-4)):
    row = []
    for frac in (0.8, 0.95):
        torch.manual_seed(0); net = make(); train(net, 800, reg, lam); dense = acc(net)
        masks = prune(net, frac); a0 = acc(net); train(net, 800, "L2", 0.0, masks, lr=1e-3); row += [a0, acc(net)]
    print(f"{reg}   {dense:5.1f}    {row[0]:9.1f}   {row[1]:8.1f}    {row[2]:9.1f}   {row[3]:8.1f}")
torch.manual_seed(0); net = make(); train(net, 800, "L2", 2e-4)
for frac in (0.5, 0.75, 0.875, 0.95):                      # iterative prune + retrain to 95%
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() == 2:
                thr = p.abs().flatten().kthvalue(int(frac * p.numel())).values; p.mul_((p.abs() > thr).float())
    masks = [(p != 0).float() if p.dim() == 2 else None for p in net.parameters()]
    train(net, 400, "L2", 0.0, masks, lr=1e-3)
print(f"\niterative prune+retrain to 95% (4 rounds): {acc(net):.1f}%   vs. one-shot 95% + retrain above")
print(f"dropout rule D_r = D_o * sqrt(C_r / C_o): 0.5 * sqrt(0.05) = {0.5 * 0.05 ** 0.5:.3f} when 95% of the connections are gone")
```

```text
regularisation during the first training vs. accuracy right after pruning and after retraining
reg   dense    pruned 80%  retrained   pruned 95%  retrained
L1    79.5         63.5       77.7         30.6       54.9
L2    79.6         62.0       78.8         31.0       54.3

iterative prune+retrain to 95% (4 rounds): 66.9%   vs. one-shot 95% + retrain above
dropout rule D_r = D_o * sqrt(C_r / C_o): 0.5 * sqrt(0.05) = 0.112 when 95% of the connections are gone
```

- 재학습 전에는 L1(63.5%)이 L2(62.0%)보다 조금 낫고 재학습 후에는 L2(78.8%)가 L1(77.7%)보다 낫습니다. 논문의 Figure 5와 같은 교차입니다. L1은 0 근처에 많이 밀어 자르기 좋지만 남은 가중치의 질은 L2가 낫다는 설명이 작은 문제에서도 성립합니다.
- 80%를 자르고 재학습하면 밀집(79.6%) 수준으로 거의 복원되지만, 95%를 한 번에 자르면 재학습해도 54%입니다. 4라운드로 나눠 자르면 66.9%로 12%p 높습니다. 한 번에 자르면 9배가 안 되고 반복하면 된다는 AlexNet 결과의 축소판입니다.
- 95%를 자르면 드롭아웃 0.5는 0.112로 줄여야 한다는 식 (2)의 계산입니다. 용량이 1/20이 된 네트워크에 원래 드롭아웃을 쓰면 과소적합이 됩니다.

## 4. 해석

**1. 현대 가지치기 연구의 출발점입니다.** 크기 기준 가지치기 자체는 1989~1993년의 biased weight decay, OBD, OBS로 거슬러 올라가지만, 이 논문이 ImageNet 규모 CNN에서 10배 압축을 보이고 "학습-가지치기-재학습" 파이프라인을 표준으로 만들었습니다. [Deep Compression]({% post_url 2026-10-24-deep-compression %})은 여기에 양자화와 허프만 부호화를 얹어 35~49배로 늘립니다.

**2. 동기가 에너지라는 점이 중요합니다.** FLOPs가 아니라 DRAM 접근 에너지에서 출발했기 때문에 목표가 "온칩에 들어가는 크기"이고, 결과물인 비구조적 희소성은 범용 GPU에서 속도 이득이 없습니다. 저자들도 희소 DNN 전용 하드웨어를 겨냥한다고 밝히며, 이것이 EIE 가속기와 [엣지 AI 하드웨어]({% post_url 2026-06-19-edge-ai-hardware %})의 희소 지원으로 이어집니다. 범용 하드웨어의 속도를 원한 연구는 [필터 가지치기]({% post_url 2026-10-24-pruning-filters %})처럼 구조적 방향으로 갔습니다.

**3. "가중치를 유지해야 한다"는 주장이 이후 논쟁의 씨앗입니다.** 3.3절의 "재초기화하면 좋은 해를 못 찾는다"는 관찰은 [Lottery Ticket]({% post_url 2026-10-24-lottery-ticket %})의 "초기값으로 되돌리면 된다"와 [Rethinking]({% post_url 2026-10-24-rethinking-pruning %})의 "구조적 가지치기는 처음부터 학습해도 된다"로 각각 수정됩니다. 다만 Rethinking도 비구조적 가지치기의 ImageNet 95% 수준에서는 미세조정이 낫다고 보고해, 이 논문의 설정에서는 원래 주장이 유지됩니다.

**4. 가지치기가 "무엇이 중요한지"를 드러낸다는 관찰이 흥미롭습니다.** LeNet의 중앙 집중 패턴과 층별 민감도(첫 CONV가 가장 민감)는 가지치기를 해석 도구로도 쓸 수 있음을 보여 주고, Rethinking의 "가지치기는 구조 탐색"이라는 재해석과 통합니다.

**5. 평가를 읽을 때 주의할 점.** 압축률은 파라미터 수 기준이고 인덱스 오버헤드(15.6%)를 포함하면 조금 줄어듭니다. 연산 감소(AlexNet 3배)는 희소 연산을 효율적으로 실행할 수 있다는 가정 위의 수치입니다. 재학습 173시간은 원래 학습의 2.3배입니다. 위 toy는 MLP라 CONV 층의 민감도나 FC/CONV 교대 재학습은 다루지 않으며, 미세조정 설정이 짧아 절대 정확도는 논문보다 낮습니다.

## 5. 정리

| 항목 | 내용 |
|---|---|
| 동기 | DRAM 접근(640pJ)이 곱셈(3.7pJ)보다 약 170배, 단순 산술보다 3자리 비쌈 → 가중치를 온칩에 |
| 방법 | 학습 → 임계값 이하 가중치 제거 → 남은 연결 재학습, 반복 가능 |
| 세부 | L2 정규화, 드롭아웃 $$D_r = D_o\sqrt{C_r/C_o}$$, 가중치 유지(재초기화 금지), FC/CONV 교대 재학습, 층별 임계값 |
| 결과 | LeNet 12×, AlexNet 9×(61M→6.7M), VGG-16 13×(138M→10.3M), 정확도 손실 없음 |
| 관찰 | 재학습 없이 2× 공짜, 반복으로 5×→9×, CONV가 FC보다 민감, 첫 CONV가 가장 민감 |
| 한계 | 비구조적 희소성(전용 하드웨어 필요), 긴 재학습, 이후 LTH·Rethinking이 "가중치 유지" 주장을 재검토 |

## References

- Han, Pool, Tran, Dally, "Learning both Weights and Connections for Efficient Neural Networks," NeurIPS 2015. [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)
- LeCun, Denker, Solla, "Optimal Brain Damage," NeurIPS 1990.
- Hassibi, Stork, "Second order derivatives for network pruning: Optimal Brain Surgeon," NeurIPS 1993.
- Han, Mao, Dally, "Deep Compression," ICLR 2016. [arXiv:1510.00149](https://arxiv.org/abs/1510.00149)
- Han et al., "EIE: Efficient Inference Engine on Compressed Deep Neural Network," ISCA 2016. [arXiv:1602.01528](https://arxiv.org/abs/1602.01528)
- Horowitz, "Computing's energy problem (and what we can do about it)," ISSCC 2014.
