---
title: "[RL] Revisiting Intrinsic Reward: 절차적 생성 환경에서는 Episodic 보상만으로 충분하다"
date: 2026-09-14 21:39:00 +0900
categories: [Reinforcement Learning, Exploration]
tags: [reinforcement-learning, exploration, intrinsic-motivation, episodic-reward, lifelong-reward, procedurally-generated, minigrid]
description: 탐험용 intrinsic reward를 life-long과 episodic으로 나눠 각각의 기여를 측정하고, 맵이 매번 바뀌는 절차적 생성 환경에서는 episodic 보상만으로 충분하다는 결론을 낸 연구를 정리합니다.
math: true
---

> [RIDE]({% post_url 2026-09-07-ride %}), [NGU]({% post_url 2026-09-12-never-give-up %}) 같은 최신 탐험 방법은 **life-long 보상**(학습 전체에 걸친 새로움)과 **episodic 보상**(한 에피소드 안의 새로움)을 함께 씁니다. *Revisiting Intrinsic Reward for Exploration in Procedurally Generated Environments*(Wang et al., ICLR 2023)는 둘을 분리해서 각각의 기여를 측정하고, 놀라운 결론을 냅니다. **절차적 생성 환경에서는 episodic 보상이 거의 전부다.**

## 1. 배경

Sparse reward 환경의 탐험은 여전히 Deep RL의 핵심 과제이고, 최근에는 맵이 매번 새로 만들어지는 **procedurally-generated 환경**의 탐험이 주목받고 있습니다. 그중 intrinsic reward를 쓰는 방법들이 뛰어난 성능을 보였습니다.

Intrinsic reward는 두 가지로 나눌 수 있습니다.

| 종류 | 의미 | 예시 |
|---|---|---|
| **Life-long** | 학습 전체에서 이 상태가 얼마나 새로운가 | ICM, RND, RIDE의 impact 항, BeBold |
| **Episodic** | 이번 에피소드에서 이 상태가 얼마나 새로운가 | episodic count, 첫 방문 여부 |

기존 연구들은 **life-long 보상을 설계하는 데 집중**하고, episodic 보상은 이를 **보조하는 수단** 정도로 다뤘습니다. 두 보상이 **각각 얼마나 기여하는지**는 분석된 적이 없었습니다.

### Contribution

- 절차적 생성 환경에서 효율적인 탐험에 **어떤 intrinsic reward가 중요한지** 분석
- Life-long intrinsic reward가 **크게 기여하지 못하는 이유** 분석
- **Episodic intrinsic reward만으로도** SOTA에 준하는 성능을 달성함을 실험적으로 확인

## 2. 실험 설정

MiniGrid의 어려운 탐험 과제 세 가지를 씁니다.

| 과제 | 내용 |
|---|---|
| **KeyCorridor** | 잠긴 문 뒤에 숨겨진 물체를 집기. 열쇠를 찾기 위해 여러 방을 탐색해야 함 |
| **MultiRoom** | 문으로 연결된 방들을 지나 마지막 방의 목표까지. 문을 열어야 다음 방으로 갈 수 있음 |
| **ObstructedMaze** | 잠긴 문 뒤의 물체를 집기. 열쇠는 상자 안에 숨겨져 있고, 문 앞의 장애물도 치워야 함 |

과제 이름의 접미사는 구성을 나타냅니다. 예를 들어 **N7S8**은 크기가 8칸 이하인 방 7개입니다.

### 비교한 보상

**Life-long**: ICM, RND, RIDE, BeBold

**Episodic** 두 가지:

$$
r^{count}_t = \frac{1}{\sqrt{N_{ep}(s_{t+1})}},\qquad r^{visit}_t = \mathbf{1}\big[N_{ep}(s_{t+1}) = 1\big]
$$

- **Episodic count**: 이번 에피소드 방문 횟수의 제곱근 역수
- **Episodic visit**: 이번 에피소드에서 **처음 방문했으면 1**, 아니면 0

```python
from collections import defaultdict

class EpisodicBonus:
    def __init__(self, mode="count"):
        self.mode, self.n = mode, defaultdict(int)

    def reset(self):                       # 에피소드 시작마다
        self.n.clear()

    def __call__(self, state_key):
        self.n[state_key] += 1
        if self.mode == "count":
            return self.n[state_key] ** -0.5
        return float(self.n[state_key] == 1)   # "visit": 첫 방문에만 보상

# 한 에피소드에서 (1,1) → (1,2) → (1,1) → (1,1) → (2,2) 순서로 이동했다면
for mode in ["count", "visit"]:
    bonus = EpisodicBonus(mode)
    print(mode, [round(bonus(s), 3) for s in [(1, 1), (1, 2), (1, 1), (1, 1), (2, 2)]])
# count [1.0, 1.0, 0.707, 0.577, 1.0]
# visit [1.0, 1.0, 0.0, 0.0, 1.0]
```

Count는 재방문할수록 보상이 서서히 줄지만, visit은 **두 번째부터 바로 0**입니다. 그래서 visit을 쓰면 같은 곳을 맴돌 유인이 전혀 없어, 아래 실험처럼 더 빨리 새 방으로 나아갑니다.

> 이렇게 상태를 그대로 세는 것은 MiniGrid처럼 상태가 **작고 이산적인** 환경이라 가능합니다. Atari나 3D 환경처럼 관측이 픽셀이면 같은 상태가 정확히 반복되지 않으므로, [NGU]({% post_url 2026-09-12-never-give-up %})처럼 embedding 공간의 $$k$$-최근접 이웃으로 episodic count를 근사해야 합니다.
{: .prompt-info }

## 3. 각 보상의 개별 기여

### Extrinsic reward가 있는 경우

- **Episodic 보상만** 쓴 경우가, life-long과 episodic을 **결합한** 경우와 비슷하거나 더 좋았습니다.
- **Life-long 보상만** 쓴 경우는 성능 향상이 거의 없었습니다.
- 같은 episodic 보상에 서로 다른 life-long 보상을 붙여 비교하면, 최신 방법인 **RIDE와 BeBold가 ICM, RND보다 특별히 낫지 않았습니다.** 즉 최신 방법의 성능 향상은 대부분 **함께 쓴 episodic 부분 덕분**이었다는 뜻입니다.

### 순수 탐험 (Extrinsic reward 없음)

보상 없이 탐험만 시켰을 때 몇 번째 방까지 가는지 봤습니다.

- **Life-long만** 쓰면 처음 두 개의 방을 넘어가지 못했습니다.
- Episodic count와 episodic visit을 비교하면 **visit이 더 빠른 탐험**을 유도했습니다.

### SOTA 방법과 비교

Episodic count나 episodic visit **하나만** 쓴 방법이, RIDE·BeBold 등 SOTA 방법과 **같거나 더 좋은** 성능을 보였습니다. 훨씬 단순한 방법으로 같은 결과를 얻은 것입니다.

## 4. 왜 Life-long 보상은 도움이 안 될까

### 관찰: 두 상태 사이에서 진동

Life-long 보상만으로 학습한 에이전트를 보면, 종종 **첫 번째 방에 갇혀 두 상태 사이를 오갑니다.** 예를 들어 문을 열었다 닫았다를 반복합니다.

저자들은 그 이유를 **life-long 보상이 상태의 새로움을 정확하게 반영하지 못하기 때문**으로 추측합니다. 맵이 매번 바뀌는 환경에서는 학습 전체에 걸친 방문 빈도가 "지금 이 맵에서 새로운 곳인가"와 잘 맞지 않습니다.

### 검증: 순서를 섞어도 성능이 그대로

이를 확인하기 위해, 미니배치 안에서 life-long 보상의 **순서를 무작위로 섞어** 각 transition에 엉뚱한 보상이 붙도록 만들었습니다. 만약 life-long 보상이 중요한 정보를 담고 있다면 성능이 크게 떨어져야 합니다.

결과적으로 **성능이 크게 떨어지지 않았습니다.** Life-long 보상이 담은 "어느 상태가 새로운가"라는 정보가 실제로는 성능에 중요하지 않았다는 증거입니다. (보상의 전체적인 크기만으로 탐험을 약간 부추기는 정도의 역할)

## 5. 정리

| 질문 | 답 |
|---|---|
| 절차적 생성 환경에서 무엇이 중요한가? | **Episodic 보상** |
| Life-long 보상의 기여는? | 단독으로는 거의 없음, 순서를 섞어도 성능 유지 |
| 최신 life-long 방법(RIDE, BeBold)이 나은가? | 같은 episodic 보상 위에서는 ICM, RND와 비슷 |
| Count vs Visit | Visit(첫 방문 보상)이 더 빠른 탐험 |

이 논문의 메시지는 "새로운 탐험 방법을 평가할 때 **각 구성 요소를 분리해서** 기여를 측정해야 한다"는 것입니다. 단, 이 결론은 **절차적 생성 환경**에 대한 것이고, Atari처럼 같은 맵이 반복되는 환경에서는 life-long 새로움이 여전히 중요합니다. 다음 글 [Agent57]({% post_url 2026-09-17-agent57 %})이 그 예입니다.

## References

- Wang et al., "Revisiting Intrinsic Reward for Exploration in Procedurally Generated Environments," ICLR 2023. [OpenReview](https://openreview.net/forum?id=j3GK3_xZydY)
- Raileanu & Rocktäschel, "RIDE," ICLR 2020.
- Zhang et al., "BeBold: Exploration Beyond the Boundary of Explored Regions," 2020.
- Badia et al., "Never Give Up," ICLR 2020.
