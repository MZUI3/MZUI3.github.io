# 블로그 교정 보고서: older-2-g3

대상 5개 파일(_posts/). 교정 원칙: 맞춤법·띄어쓰기·문장부호·비문·어색한 표현·글 안의 표기 통일만 고치고, 내용·수치·순서·소제목·말투·코드 블록·수식·링크·Liquid 태그·front matter·기술 용어·고유명사는 그대로 둡니다. 애매하면 고치지 않고, 내용이 의심스러운 부분은 "내용 확인 필요"에만 적습니다. 모든 `{% post_url %}` 대상 파일(transformer-decoder-to-gpt, diffuser, trpo-to-ppo, efficientzero, trajectory-transformer, latent-diffusion-models, neural-style-transfer, lightweight-vision-transformer)이 `_posts/`에 있는지 확인했고, 코드 블록은 실행해 출력 블록·주석과 일치하는지 확인했습니다(Trajectory Transformer 이산화, ACER importance weight, HER relabel, DeepSVG 임베딩 shape 모두 일치, NumPy 2.5.1).

## 2026-08-24-trajectory-transformer.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 3절 "차원마다 독립적으로 $$V$$개(논문은 100개)의 구간": 논문의 vocabulary size(구간 수) 확인 필요.
- 5절 "각 모델에서 궤적 1000개를 샘플링해 시점별 상태 분포에서 reference 상태의 likelihood를 비교": 샘플 수(1000개)와 비교 방식 확인 필요.
- 5절 "상태 차원을 무작위로 절반 가린 부분 관측 환경": 논문의 부분 관측 설정(가린 비율) 확인 필요.
- 5절 Offline RL 비교 대상 표(CQL, MOPO, MBOP, BC)와 "HalfCheetah med-expert에서 ... 이산화가 세밀하지 못해 성능이 낮았던 것으로 추정": 논문 표의 baseline 목록과 해당 설명 확인 필요.

## 2026-08-21-efficientzero.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 오래된 데이터의 목표값을 | 오래된 데이터의 목푯값을 | 맞춤법 |

### 내용 확인 필요

- 5절 결과 표 "EfficientZero 190.4% / 116.0%", "SimPLe 44.3% / 14.4%", "SPR 70.4% / 41.5%": 논문 초록·표의 평균/중앙값 HNS와 일치하는지 확인 필요(EfficientZero는 194.3% / 109.0%로 기억됨).
- 5절 "Ablation에서 세 가지 중 self-supervised consistency가 가장 큰 효과": 논문 ablation 결과 확인 필요.

## 2026-08-19-hindsight-experience-replay.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 3절 "일반 DQN은 $$n > 13$$이면 실패, HER을 붙이면 $$n = 50$$까지": 논문 수치 확인 필요.
- 6절 "future 전략, $$k = 4 \sim 8$$일 때 가장 좋았습니다. 너무 크면 ... 성능이 떨어졌습니다": 논문 결과 확인 필요.

## 2026-08-17-acer.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 5절 "논문 실험에서는 $$c = 10$$": 논문의 truncation 상수 값 확인 필요.
- 8절 Atari 결과 "Replay 비율 0, 1, 4, 8", "Replay 비율 4인 ACER는 ... Prioritized replay DQN에 필적", "계산 시간 기준으로는 A3C와 비슷", "Trust region ... 성능을 확실히 끌어올렸습니다": 논문 서술 확인 필요.
- 8절 MuJoCo 과제 목록과 차원(Cartpole 1D, Reacher 3D, Fish 5D, Cheetah 6D, Walker 6D, Humanoid 21D): 논문 확인 필요.

## 2026-08-15-deepsvg.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 두 값 사이의 함수값을 구하는 | 두 값 사이의 함숫값을 구하는 | 맞춤법 |

### 내용 확인 필요

- 5절 "SVG-Icons8: 56개 카테고리, 약 10만 개의 아이콘": 논문 수치 확인 필요.
- 4절 "예측한 path와 정답 path를 대응시킬 때는 path 순서를 정렬해 맞춥니다": 논문의 ordered/Hungarian assignment 서술 확인 필요.
- 5절 "Path 개수 상한 $$N$$(4, 8 등)을 바꾼 비교에서도, 계층 구조가 path 수가 많은 복잡한 아이콘을 더 잘 다뤘습니다": 논문에 이런 비교가 있는지 확인 필요.

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| trajectory-transformer | 차원별 구간 수 $$V$$ = 100 | 문제 없음 | 부록 하이퍼파라미터 표 "Vocabulary size: number of bins used for autoregressive discretization: 100". https://arxiv.org/abs/2106.02039 | 없음 |
| trajectory-transformer | 궤적 1000개 샘플링 후 reference 상태 likelihood 비교 | 문제 없음 | "we sample 1000 trajectories from a fixed starting point to estimate the per-timestep state marginal", 이후 reference policy 상태의 likelihood 보고. https://arxiv.org/abs/2106.02039 | 없음 |
| trajectory-transformer | 부분 관측: 상태 차원을 무작위로 절반 가림 | 문제 없음 | 각 상태의 각 차원을 50% 확률로 mask한 partially-observed Humanoid. https://arxiv.org/abs/2106.02039 | 없음 |
| trajectory-transformer | D4RL 비교 대상 표(CQL, MOPO, MBOP, BC) | 수정함 | arXiv v2(2021-07)는 BC, MOPO, MBOP, CQL과 비교했지만, 글이 인용한 NeurIPS 2021 최종판(v4)의 Table 1은 BC, MBOP, BRAC, CQL, DT와 비교하며 "BRAC와 CQL이 model-free offline RL의 SOTA"라고 씀(MOPO 없음). https://arxiv.org/abs/2106.02039 | MOPO 행을 빼고 BRAC를 CQL 행에 추가, Decision Transformer 행 추가 |
| trajectory-transformer | HalfCheetah med-expert 성능 저하 설명 | 수정함 | 최종판: uniform과 quantile 이산화는 HalfCheetah-Medium-Expert를 빼면 비슷하고, 여기서는 속도 범위가 넓어 uniform 이산화가 expert의 정밀한 동작을 재현하지 못하며 quantile이 두 배 넘는 return을 냄(40.8 vs 95.0). "추정"은 v2의 "possibly due to" 서술. https://arxiv.org/abs/2106.02039 | 문장을 uniform 이산화의 한계로 한정하고 quantile 결과 추가 |
| efficientzero | 결과 표 EfficientZero 190.4% / 116.0% | 수정함 | v1 초록은 190.4% / 116.0%, NeurIPS 최종판(v2, 2021-12) 초록과 Table 2는 194.3% / 109.0%. SimPLe 0.443/0.144, SPR 0.704/0.415는 일치. https://arxiv.org/abs/2111.00210 | 190.4% → 194.3%, 116.0% → 109.0% |
| efficientzero | Ablation에서 consistency가 가장 큰 효과 | 문제 없음 | Table 3: "Each component matters and the consistency one is the most significant" (consistency 제거 시 평균 1.943 → 0.881). https://arxiv.org/abs/2111.00210 | 없음 |
| hindsight-experience-replay | DQN은 $$n > 13$$에서 실패, HER은 $$n = 50$$까지 | 문제 없음 | "DQN without HER can only solve the task for n ≤ 13 while DQN with HER easily solves the task for n up to 50". https://arxiv.org/abs/1707.01495 | 없음 |
| hindsight-experience-replay | future, $$k = 4 \sim 8$$ 최적, 너무 크면 성능 저하 | 문제 없음 | "future with k equal 4 or 8 performs best", k를 8보다 키우면 일반 replay 데이터 비율이 줄어 성능 저하. 논문은 범위가 아니라 4 또는 8이라고 씀. https://arxiv.org/abs/1707.01495 | 없음 |
| acer | 논문 실험에서 $$c = 10$$ | 수정함 | Atari: "importance weight truncation with c = 10", MuJoCo: "The truncation threshold c is set to 5 for ACER". https://arxiv.org/abs/1611.01224 | "논문 실험에서는" → "논문의 Atari 실험에서는" |
| acer | Atari 결과(replay 비율 0/1/4/8, Prioritized DQN에 필적, 계산 시간은 A3C와 비슷) | 문제 없음 | replay ratio 0, 1, 4, 8(0은 A3C), "with enough replay, ACER can match the performance of the best DQN agents", "ACER agents perform similarly to A3C when measured by wall clock time", "ACER with a replay ratio of 4 is an appealing alternative to either the prioritized DQN agent or A3C". https://arxiv.org/abs/1611.01224 | 없음 |
| acer | Atari에서 TR이 성능을 확실히 끌어올림 | 확인 불가 | Figure 1에 TR 유무를 실선/점선으로 그렸으나, 본문은 Atari에서 TR 효과를 직접 서술하지 않고 연속 제어에서 TR 이득이 "much higher"라고만 함. 그림 수치는 텍스트로 확인하지 못함. https://arxiv.org/abs/1611.01224 | 없음 |
| acer | MuJoCo 과제와 행동 차원 | 문제 없음 | "cartpole (1D), reacher (3D), cheetah (6D), fish (5D), walker (6D) and humanoid (21D)", 고차원 과제(humanoid, cheetah, walker, fish)에서 차이가 뚜렷. https://arxiv.org/abs/1611.01224 | 없음 |
| deepsvg | SVG-Icons8 56개 카테고리, 약 10만 개 | 문제 없음 | "our dataset consists of 100,000 high-quality icons in 56 different categories". https://arxiv.org/abs/2007.11301 | 없음 |
| deepsvg | path 순서를 정렬해 정답과 대응 | 문제 없음 | Ordered assignment(정답 path를 정렬, 시작 위치 기준)와 Hungarian assignment를 비교했고 Ordered가 더 좋았음. https://arxiv.org/abs/2007.11301 | 없음 |
| deepsvg | Path 개수 상한 $$N$$(4, 8 등)을 바꾼 비교 | 확인 불가 | 논문 본문·부록의 ablation(Table 2)은 one-stage autoregressive / one-stage feed-forward / Hungarian / Ordered 비교뿐이고, $$N_P$$를 바꾼 비교는 찾지 못함. https://arxiv.org/abs/2007.11301 | 없음 (원 출처 확인 필요) |
| deepsvg | (교정 중 추가 발견) 명령어 벡터에 "커서의 시작 좌표"를 넣음 | 수정함 | Table 1: "The start-position (x1, y1) is implicitly defined as the end-position of the preceding command", 인자는 $$(q_{x1}, q_{y1}, q_{x2}, q_{y2}, x_2, y_2)$$ 6개. https://arxiv.org/abs/2007.11301 | "**커서의 시작 좌표**와 제어점·끝점 좌표 등 인자(argument)" → "제어점·끝점 좌표 등 인자(argument) (커서의 시작 좌표는 직전 명령어의 끝점이라 따로 넣지 않음)". 코드 블록의 `n_args=11`(공식 구현의 arc 인자 포함 개수)은 코드라 그대로 둠 |
