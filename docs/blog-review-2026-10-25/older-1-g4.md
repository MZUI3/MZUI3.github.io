# 블로그 교정·사실 확인 — older-1-g4

대상: `_posts/` 아래 5개 글

## 2026-10-01-thought-cloning.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 또 생각이 탐험을 이끌어 시행착오 범위가 줄어들어 | 또 생각이 탐험을 이끌면 시행착오 범위가 줄어들어 | 어색한 표현 |
| 각 시점마다 **관측 | 시점마다 **관측 | 어색한 표현 |
| 안전 장치로 쓰려면 | 안전장치로 쓰려면 | 띄어쓰기 |

### 내용 확인 필요

- Action Generator의 입력이 "관측 이력과 현재 생각"으로만 적혀 있고, 학습 목적 식의 $$\pi_{action}$$ 조건에도 임무 $$m$$이 빠져 있음. 논문 원문과 대조 필요.
- 코드의 `alpha=2.0`, teacher forcing 스케줄, "약 100만 개의 궤적", precrime intervention 결과("대부분을 막음") 수치 확인 필요.

## 2026-09-29-anti-exploration.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 가치 함수를 발산**시키고 보정할 수 없습니다. | 가치 함수를 발산**시키는데, 이를 보정할 수 없습니다. | 비문 |

### 내용 확인 필요

- 4절: 보너스 정책 $$\pi_b \propto \exp(-b)$$와의 KL 규제와 "같습니다", "특정 보너스를 쓰면 기존의 behavior regularization 방법들을 특수한 경우로 포함" — 논문의 실제 주장(극한 관계인지, 포함 관계인지) 확인 필요.
- 6절 실험 1: "RND는 특히 shuffled처럼 … 잘 잡지 못했습니다" — 논문이 shuffled를 특정했는지 확인 필요.
- 6절 실험 2: "대부분의 데이터셋에서 최고 성능과 비슷한 수준(1σ 이내)" — 논문 Table 1로 재계산 필요. 비교 기준 알고리즘 목록도 확인 필요.
- 5절 장난감 실험의 ```text 출력(0.00016… / 1.029… / 0.839…)과 본문의 "수천 배" 서술 일치 여부.
- AAAI 2022 게재 여부.

## 2026-09-26-implicit-q-learning.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- expectile 예제 코드의 주석 출력(0.5 4.0 / 0.7 5.263 / 0.9 7.692 / 0.99 9.709)이 맞는지 재계산 필요.
- 5절 미로 예제: "IQL은 SARSA보다 …, 다른 offline 방법보다 noise가 적은 가치 추정" — 논문에서 비교한 대상 확인 필요.
- τ 값(locomotion 0.7, AntMaze 0.9), clipped double Q 사용 여부 확인 필요.
- Online fine-tuning: "CQL 같은 보수적인 방법은 fine-tuning 초반에 성능이 떨어지거나 느리게 오르는 경향" — 논문 근거 확인 필요.

## 2026-09-23-dagger.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 4절 알고리즘 1단계: "처음에는 전문가 정책으로 궤적을 모아 D를 만들고, 첫 정책 π̂_1을 학습"한 뒤 다시 i=1에서 β_1=1로 전문가 정책을 따른다고 되어 있어, 논문의 Algorithm 1(D 초기화 방식, π̂_1 초기화) 및 글 속 코드(`D = []`에서 바로 반복 시작)와 어긋날 수 있음.
- 6절 실험 세부 수치(5Hz, 바퀴당 약 1,000개, 15번 반복 후 낙하 없음, 4개의 선형 SVM, p = 0.5 최고)와 Mario 평가 조건("무작위 난이도 스테이지") 확인 필요.
- 1절: "강화학습에 모방학습을 결합한 연구는 DAgger에서 출발해, DQfD, 그리고 … IRL 방향으로 발전" — IRL(Ng & Russell, 2000)은 DAgger보다 앞선 연구라 시간 순서 서술 확인 필요.

## 2026-09-21-adaptdiffuser.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 6절: "Diffuser는 guidance만으로는 금화 쪽으로 돌아가는 경로를 잘 만들지 못했습니다" — 논문의 Gold Coin 결과와 대조 필요.
- "부록 G에서 ChatGPT로 미로 레이아웃 생성" — 부록 번호 확인 필요.
- 5절 sparse reward(단위 계단 함수, Dirac delta) 서술과 discriminator 구성(inverse dynamics + 동역학 일관성 + 보상 기준) 확인 필요.
- 예제 코드 주석 출력 `1` 재계산 필요.

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| 2026-10-01-thought-cloning.md | Action Generator 입력에 임무 $$m$$ 누락 | 수정함 | https://arxiv.org/html/2306.00323 (Section 2: Action Generator는 현재 생각, 관측 이력, 임무를 조건으로 함; Eq. 1의 행동 항도 $$m$$ 포함) | 본문 "관측 이력과 **현재 생각**" → "임무, 관측 이력과 **현재 생각**"; 식의 $$\pi_{action}(a_t \mid o_{\le t}, th_t)$$ → $$\pi_{action}(a_t \mid m, o_{\le t}, th_t)$$ (의사 코드는 손대지 않음) |
| 2026-10-01-thought-cloning.md | α=2, teacher forcing, 100만 궤적, precrime 결과 | 문제 없음 | https://arxiv.org/html/2306.00323 (α=2; teacher-forcing 비율 100%→0% 선형 감소; 1 million trajectories; "almost entirely eliminates all unsafe behaviors"; BossLevel) | - |
| 2026-09-29-anti-exploration.md | KL 규제와의 관계("같습니다", 기존 방법을 특수한 경우로 포함) | 수정함 | https://arxiv.org/abs/2106.06431 (Sec. 4: anti-exploration VI는 온도→0일 때 보너스 기반 정책 $$\pi_b = \mathrm{softmax}(-\beta b/\tau)$$로의 KL-regularized VI의 limiting case; 기존 방법을 특수한 경우로 포함한다는 주장은 없음) | "KL divergence로 규제하는 것과 같습니다" → "…규제하는 방식에서 온도(temperature)를 0으로 보낸 극한에 해당합니다"; "이는 KL-regularized value iteration 틀로 정리되며, 특정 보너스를 쓰면 기존의 behavior regularization 방법들을 특수한 경우로 포함합니다" → "즉 anti-exploration은 KL-regularized value iteration의 극한으로 정리되며, 이를 통해 policy regularization 계열 방법들과 연결됩니다" |
| 2026-09-29-anti-exploration.md | RND가 "특히 shuffled"를 못 잡음 | 수정함 | https://arxiv.org/abs/2106.06431 (Fig. 2 설명: RND는 데이터 행동과 다른 행동의 보너스 차이가 거의 없어 OOD 행동 식별 전반에 약함, shuffled를 특정하지 않음) | "RND는 특히 shuffled처럼 상태와 행동의 조합이 어긋난 경우를 잘 잡지 못했습니다" → "RND는 데이터 행동과 OOD 행동의 보너스 차이가 크지 않아 OOD 행동을 잘 잡지 못했습니다" (5절의 "RND가 shuffled를 잘 못 잡은 이유와 같은 맥락"은 shuffled도 실패 범위에 포함되므로 유지) |
| 2026-09-29-anti-exploration.md | "대부분의 데이터셋에서 최고 성능과 1σ 이내" | 수정함 | https://arxiv.org/abs/2106.06431 (Table 1 재계산: 1σ 이내는 locomotion 6/12, Adroit 5/12로 24개 중 11개뿐; 본문은 "locomotion에서 CQL과 competitive, 나머지보다 우수; hand manipulation에서 모든 방법보다 우수"(평균 기준); 비교 기준은 BC, BEAR, BRAC, AWR, BCQ, CQL) | "(BC, BCQ, BEAR, CQL 등)" → "(BC, BCQ, BEAR, BRAC, AWR, CQL)"; 성능 문장을 "locomotion에서 평균적으로 CQL과 비슷하고 나머지 방법보다 좋았으며, hand manipulation에서는 평균 성능이 가장 높았습니다"로 수정 |
| 2026-09-29-anti-exploration.md | 장난감 실험 출력과 "수천 배" | 문제 없음 | 글의 코드 로컬 재실행: 0.000163… / 1.0291… / 0.8394… (출력과 일치); 1.029/0.00016≈6,400배, 0.839/0.00016≈5,200배 | - |
| 2026-09-29-anti-exploration.md | AAAI 2022 | 문제 없음 | https://ojs.aaai.org/index.php/AAAI/article/view/20783 | - |
| 2026-09-26-implicit-q-learning.md | expectile 예제 출력 | 문제 없음 | 해석해(τ=0.7: 10/1.9=5.263, τ=0.9: 10/1.3=7.692, τ=0.99: 10/1.03=9.709)와 로컬 재실행 결과 모두 주석과 일치 | - |
| 2026-09-26-implicit-q-learning.md | 미로 예제 비교 대상("SARSA보다", "다른 offline 방법보다") | 수정함 | https://arxiv.org/abs/2110.06169 (Sec. 5.1, Fig. 2: one-step 방법(Onestep RL)과 IQL(τ=0.95)만 비교, IQL 가치가 "no longer dominated by noise") | "IQL은 SARSA보다 …, 다른 offline 방법보다 noise가 적은" → "IQL은 single-step 방법(Onestep RL)보다 …, noise가 적은" |
| 2026-09-26-implicit-q-learning.md | τ 값, clipped double Q | 문제 없음 | https://arxiv.org/abs/2110.06169 (부록: AntMaze τ=0.9, β=10; locomotion τ=0.7, β=3; Kitchen/Adroit τ=0.7; Sec. 4: 두 Q의 최솟값 사용) | - |
| 2026-09-26-implicit-q-learning.md | CQL이 "fine-tuning 초반에" 성능이 떨어지거나 느리게 오름 | 확인 불가 | https://arxiv.org/abs/2110.06169 (Table 2는 offline→online 최종 점수만 보고; CQL은 일부 AntMaze 과제에서 점수가 0으로 떨어졌지만 "초반" 추이는 논문에 없음) | - |
| 2026-09-23-dagger.md | 알고리즘 1단계(D·π̂_1 초기화) | 수정함 | https://arxiv.org/abs/1011.0686 (Algorithm 1: "Initialize D←∅. Initialize π̂_1 to any policy in Π."; β_1=1이면 첫 반복은 전문가 정책으로 데이터를 모음; 글의 코드도 `D = []`에서 바로 반복 시작) | "1. 처음에는 전문가 정책으로 궤적을 모아 데이터셋 D를 만들고, 첫 정책 π̂_1을 학습" → "1. 데이터셋 D를 빈 집합으로 두고, 첫 정책 π̂_1을 임의의 정책으로 초기화" |
| 2026-09-23-dagger.md | Mario 평가 "무작위 난이도 스테이지" | 수정함 | https://arxiv.org/abs/1011.0686 (Sec. 5.2: "randomly generated stages of difficulty 1") | "무작위 난이도 스테이지에서" → "무작위로 생성한 난이도 1 스테이지에서" |
| 2026-09-23-dagger.md | 실험 세부 수치(5Hz, ~1,000개/바퀴, 15번 반복, 4개 선형 SVM, p=0.5) | 문제 없음 | https://arxiv.org/abs/1011.0686 (Sec. 5.1–5.2) | - |
| 2026-09-23-dagger.md | DAgger → DQfD → IRL 발전 서술 | 문제 없음 | https://arxiv.org/abs/1011.0686 (참고: 고전 IRL은 2000년부터 있었지만, 문장은 이후 강화학습과 결합한 IRL 계열(GAIL 등)로 읽을 수 있어 그대로 둠) | - |
| 2026-09-21-adaptdiffuser.md | Gold Coin 과제에서 Diffuser가 금화 경로를 못 만듦 | 수정함 | https://arxiv.org/abs/2302.01877 (Sec. 5.3.1, Fig. 4: 두 planner 모두 보상 유도로 금화를 지나는 가운데 경로로 바꿨지만 Diffuser의 경로는 벽과 충돌, AdaptDiffuser는 충돌 없는 경로) | "Diffuser는 guidance만으로는 금화 쪽으로 돌아가는 경로를 잘 만들지 못했습니다" → "Diffuser도 guidance로 금화를 지나는 경로를 만들었지만 벽과 충돌했고, AdaptDiffuser는 충돌 없는 경로를 만들었습니다" |
| 2026-09-21-adaptdiffuser.md | 부록 G(ChatGPT 미로 생성) | 문제 없음 | https://arxiv.org/abs/2302.01877 (Appendix G "Generate Diverse Maze Layouts with ChatGPT") | - |
| 2026-09-21-adaptdiffuser.md | sparse reward·discriminator 서술 | 문제 없음 | https://arxiv.org/abs/2302.01877 (Sec. 4.2–4.3: 단위 계단 함수, 기울기는 Dirac delta; inverse dynamics로 행동을 구해 실행한 상태와 생성 상태의 MSE로 거르고 보상으로 선별) | - |
| 2026-09-21-adaptdiffuser.md | 예제 코드 출력 `1` | 문제 없음 | 손 계산: good은 편차 ≈0·보상 0, bad는 첫 step 편차 0.32 > 0.01로 탈락 | - |
