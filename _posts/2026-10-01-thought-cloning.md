---
title: "[RL] Thought Cloning: 행동뿐 아니라 생각까지 모방하기"
date: 2026-10-01 22:03:00 +0900
categories: [Reinforcement Learning, Imitation Learning]
tags: [reinforcement-learning, imitation-learning, thought-cloning, behavior-cloning, language, interpretability, ai-safety, babyai]
description: 전문가의 행동뿐 아니라 행동하며 떠올린 생각(자연어)까지 모방하는 Thought Cloning의 구조와 실험을 정리합니다. 이렇게 학습한 에이전트는 더 빨리 배우고, 더 잘 일반화하며, 위험한 행동을 실행 전에 막을 수 있습니다.
math: true
---

> [DAgger]({% post_url 2026-09-23-dagger %}) 같은 모방학습은 전문가의 **행동**을 따라 합니다. **Thought Cloning(TC)**(Hu & Clune, NeurIPS 2023)은 한 걸음 더 나아가, 사람이 행동하면서 **머릿속으로 하는 생각(언어)**까지 함께 모방하면 더 빨리 배우고, 더 잘 일반화하며, 더 **안전하고 해석 가능한** 에이전트가 된다고 주장합니다.

## 1. 동기: 사람은 언어로 생각한다

사람은 문제를 풀 때 "먼저 빨간 열쇠를 찾아야겠다", "문이 잠겼으니 열쇠가 필요하다"처럼 **언어로 생각**하며 계획하고, 실수하면 생각을 고칩니다. 사람 수준의 성능을 내려면 에이전트도 이런 **사고 능력**이 필요하다는 것이 출발점입니다.

### Thought Cloning > Behavior Cloning

- **학습 속도가 빠름**: 행동의 "이유"를 함께 배우므로 같은 데이터로 더 빨리 학습
- **데이터 규모**: 사람이 말하면서 행동하는 영상(게임 실황, 튜토리얼 등)은 인터넷 규모로 존재
- **AI 안전성과 해석 가능성**: 에이전트가 무슨 생각으로 행동하는지 **읽을 수 있음**

### 생각이 보이면 생기는 일

1. **무엇이 잘못됐는지 파악**할 수 있음 (생각과 행동 중 어디서 틀렸는지)
2. 잘못된 생각을 **수정**할 수 있음
3. 위험하거나 잘못된 행동을 **실행 전에 막을** 수 있음

또 생각이 탐험을 이끌면 시행착오 범위가 줄어들어 **샘플 효율**도 좋아집니다.

## 2. 방법: 두 단계로 된 에이전트

### 데이터

시점마다 **관측 $$o_t$$, 생각 $$th_t$$(자연어), 행동 $$a_t$$**가 기록된 궤적과, 전체 **임무(mission)** $$m$$이 주어집니다.

$$
\mathcal{D} = \big\{(m,\ \{(o_t, th_t, a_t)\}_{t=1}^{T})\big\}
$$

### 구조

- **Thought Generator (상위)**: 임무, 지금까지의 관측 이력, **이전 생각들**을 받아 현재 생각 $$th_t$$를 **자연어 문장**으로 생성
- **Action Generator (하위)**: 임무, 관측 이력과 **현재 생각** $$th_t$$를 조건으로 행동 $$a_t$$를 선택

상위가 "무엇을 할지" 언어로 계획하고, 하위가 그 계획에 맞춰 행동하는 계층적 구조입니다.

### 학습 목적

$$
\mathcal{L} = \underbrace{-\sum_t\log\pi_{action}\big(a_t \mid m,\ o_{\le t},\ th_t\big)}_{\text{Behavior Cloning}} + \alpha\cdot\underbrace{\Big(-\sum_t\log\pi_{thought}\big(th_t \mid m,\ o_{\le t},\ th_{<t}\big)\Big)}_{\text{Thought Cloning}}
$$

- **BC loss**: 전문가의 행동을 따라 하도록 학습합니다.
- **TC loss**: 전문가의 생각 문장을 단어 단위 cross entropy로 따라 하도록 학습합니다.
- 학습 초반에는 action generator에 **정답 생각**을 넣어 주고(teacher forcing), 점점 **스스로 생성한 생각**을 넣도록 바꿔 갑니다.

```python
def thought_cloning_loss(thought_gen, action_gen, batch, alpha=2.0, teacher_forcing=True):
    m, obs, thoughts, actions = batch                       # thoughts: 시점별 정답 문장(토큰)
    th_logits, th_pred = thought_gen(m, obs, thoughts)      # 다음 생각 토큰 예측
    tc_loss = token_cross_entropy(th_logits, thoughts)

    th_input = thoughts if teacher_forcing else th_pred.detach()
    a_logits = action_gen(obs, th_input)                    # 생각을 조건으로 행동
    bc_loss = cross_entropy(a_logits, actions)
    return bc_loss + alpha * tc_loss
```

위 코드는 loss의 구성을 보여 주는 의사 코드입니다. `token_cross_entropy`는 생성된 문장의 토큰마다 cross entropy를 구하는 함수, `cross_entropy`는 행동 분류 loss라고 가정했습니다.

## 3. 실험

### 환경과 데이터

**BabyAI**(격자 세계에서 "파란 공을 집어서 빨간 문 옆에 놓아라" 같은 언어 임무를 수행)를 사용했습니다.

사람이 생각을 말한 데이터는 구하기 어려우므로, 논문은 **합성 생각 데이터**를 만들었습니다.

1. 환경 내부 정보를 아는 **Oracle Solver**로 임무의 해답(행동 순서와 하위 목표)을 얻고
2. 그 하위 목표를 **언어 문장**으로 변환해 생각으로 사용
3. 이렇게 **약 100만 개의 궤적** 생성

### 결과

- **학습 속도와 최종 성능**: 같은 데이터에서 TC가 BC보다 **빠르게 학습**했고 최종 성공률도 높았습니다. 생각 없이 행동만 학습하는 BC는 긴 임무에서 특히 약했습니다.
- **일반화**: 학습 때 보지 못한 환경과 임무(out-of-distribution)에서 TC가 BC보다 훨씬 잘 일반화했습니다. Fine-tuning했을 때도 더 빨리 적응했습니다.
- **해석 가능성**: 에이전트가 생성한 생각을 보면 그 행동의 의도를 이해할 수 있었고, 실패 사례에서는 **잘못된 생각이 실패의 원인**인 경우를 찾아낼 수 있었습니다.

### Precrime Intervention (안전성)

생각이 언어로 보이므로, 에이전트가 **위험한 생각을 하는 순간 행동을 멈추게** 할 수 있습니다. 예를 들어 테스트 환경에서 특정 물체를 "건드리면 안 된다"고 정해 두고, 생성된 생각에 그 물체를 집겠다는 내용이 나오면 실행 전에 중단합니다.

실험에서 이 방식으로 위험 행동의 대부분을 **실행 전에** 막을 수 있었습니다. 행동만 학습한 에이전트는 무엇을 하려는지 미리 알 수 없어 이런 개입이 불가능합니다.

## 4. 정리

| | Behavior Cloning | Thought Cloning |
|---|---|---|
| 모방 대상 | 행동 | **생각(언어) + 행동** |
| 구조 | 단일 정책 | 상위 Thought generator + 하위 Action generator |
| 학습 속도 | 기준 | 더 빠름 |
| 일반화 | 약함 | OOD에서 우수 |
| 해석·안전 | 블랙박스 | 생각을 읽고 **실행 전 개입** 가능 |

한계는 아직 **합성 생각 데이터**로만 검증됐다는 점입니다. 사람이 실제로 말한 생각 데이터(예: 게임 실황 영상)로 확장하는 것이 남은 과제입니다.

<details markdown="1">
<summary>더 알아보기: 생각이 보이면 정말 안전한가</summary>

이 구조는 LLM의 **chain-of-thought**(답하기 전에 추론 과정을 문장으로 쓰게 하는 방식)와 닮았습니다. 그리고 같은 질문이 따라옵니다. 모델이 쓴 생각이 **실제로 행동을 결정한 이유**인가(faithfulness)? Thought Cloning에서는 action generator가 생각을 입력으로 받으므로 생각과 행동이 구조적으로 연결되어 있지만, 행동이 생각을 무시하고 관측만으로 결정될 가능성도 있습니다. 논문의 precrime intervention이 잘 동작한 것은 합성 데이터에서 생각과 행동이 강하게 대응하도록 만들어졌기 때문이기도 합니다. 실제 사람 데이터에서는 말과 행동이 어긋나는 경우가 많아, 안전장치로 쓰려면 생각과 행동의 일치도를 따로 검증해야 합니다.
</details>

## References

- Hu & Clune, "Thought Cloning: Learning to Think while Acting by Imitating Human Thinking," NeurIPS 2023. [arXiv:2306.00323](https://arxiv.org/abs/2306.00323)
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," NeurIPS 2022.
- Chevalier-Boisvert et al., "BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning," ICLR 2019.
- Ross et al., "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)," AISTATS 2011.
- 발표 영상 — <https://www.youtube.com/watch?v=aX7aAauO37U>
