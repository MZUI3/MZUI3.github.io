# 블로그 교정 보고서: older-1-g3 (2026-10-25)

대상: `_posts/` 6편 (tinyml-contest-2023, cbs-task-assignment, feudal-networks, h-dqn, primal, diffusion-ql)

본문의 ```python 코드 블록은 모두 다시 실행해 보았습니다. CBS 예제, Diffusion-QL numpy 예제, FeUdal 코드의 주석·출력이 실제 실행 결과와 일치합니다. `{% post_url %}` 대상(primal, cbs-task-assignment, h-dqn, feudal-networks, anti-exploration)도 모두 `_posts/`에 있습니다.

## 2026-10-16-tinyml-contest-2023.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 보드(STM32F303K8T6, Cortex-M4 72MHz, Flash 64KB, SRAM 16KB), 입력(5초, 1,250 samples), 평가 항목(F-beta, Generalization, Latency, Flash)이 공식 문제 페이지와 같은지 확인
- 순위표 수치(1~3위, 11위 EmPasLab2)와 Generalization 0.842가 공식 Winners 페이지와 같은지 확인
- 내부 계산 확인: 층별 파라미터와 MACs 합계(20,713 / 39,200), 82% 비율, "8~14배", "두 배 이상"

## 2026-10-13-cbs-task-assignment.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 이 글은 강화학습이 아니라 **탐색(search) 기반 계획 알고리즘**입니다. | 이 글에서 다루는 것은 강화학습이 아니라 **탐색(search) 기반 계획 알고리즘**입니다. | 비문 |

### 내용 확인 필요

- 4절 실험 서술이 논문과 맞는지 확인: 밀집 환경·적은 에이전트에서 이득, makespan 최적화 방법(CBM)과의 비교, ECBS-TA의 확장성
- 예제 출력(```text)과 본문 "A가 한 칸 기다림, 총 비용 +1"이 맞는지 확인

## 2026-10-10-feudal-networks.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- "$$\phi$$는 bias가 없어서 목표가 0이면 정책에 영향을 주지 않습니다": 논문의 설명(bias가 없어서 Manager의 입력을 무시할 수 없음)과 반대로 읽힐 수 있음
- 5절 "Ms. Pac-Man, Seaquest, Enduro 등"에서 크게 앞섰고 "일반 게임에서도 성능을 잃지 않았습니다"가 논문 결과와 맞는지 확인
- Seaquest 하위 정책 시각화에서 "목표를 몇 개의 군집으로 나눠"라는 절차 설명이 논문과 맞는지 확인
- Dilated LSTM(r=10, 출력 합산, 파라미터 공유) 설명 확인
- Dayan & Hinton "NeurIPS 1993" 연도 확인

## 2026-10-08-h-dqn.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 보상이 바로 오는 게임에서는 사람을 크게 넘었지만 | 보상이 바로 오는 게임에서는 사람을 크게 앞섰지만 | 어색한 표현(같은 말 반복) |

### 내용 확인 필요

- "약 230만 step 학습": 논문의 학습 단계(사전 학습/공동 학습)와 맞는지 확인
- 결과 표: DQN 0, Gorila DQN 약 4, h-DQN 약 400점
- "사다리 → 열쇠 → 다시 사다리 → 문" 목표 순서와 "열쇠 목표를 고르는 빈도 증가" 서술 확인
- 4절 "학습 순서"(meta-controller ε=1로 controller 사전 학습, 목표별 성공률로 탐험률 조절) 확인

## 2026-10-06-primal.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 구성 요소는 네 가지입니다. 관측, 행동 공간, 보상 구조, 정책. | 구성 요소는 관측, 행동 공간, 보상 구조, 정책의 네 가지입니다. | 비문 |
| 큰 규모에서 경쟁력 있었습니다 | 큰 규모에서 경쟁력이 있었습니다 | 비문(조사 누락) |

### 내용 확인 필요

- ODrM\*의 inflation factor를 "충돌 회피 계수"로 설명한 부분 확인
- Environment Sampling: "밀도를 균일하게 뽑으면 … 다양한 밀도를 섞어 학습"이 논문 설명과 맞는지 확인
- "실제 로봇과 드론 실험" 확인
- 10×10 FOV, 256 step 업데이트, σ_H = 0.01, blocking −2·10 step 기준, 보상표(Table I), 유효하지 않은 행동 처리 확인

## 2026-10-04-diffusion-ql.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- D4RL 결과 "좋거나 비슷한 성능 … 특히 AntMaze와 Adroit처럼 행동 분포가 복잡한 과제에서 차이가 컸습니다"가 논문 Table 1과 맞는지 확인
- 2D bandit 비교 대상(BC, BCQ/BEAR)과 N=5 기본값·포화 서술 확인
- 수식(정책 정의, L_d, L_q, α 정규화, critic loss) 확인

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| tinyml-contest-2023 | 보드·메모리 | 문제 없음 | https://tinymlcontest.github.io/TinyML-Design-Contest-2023/Problems.html | 페이지에서 STM32L031K6T6에 취소선이 그어져 있고, 빨간 글씨로 STM32F303K8T6(M4 72MHz, 64KB Flash, 16KB SRAM)로 바뀌어 있음 |
| tinyml-contest-2023 | 입력·평가 항목·F-beta | 문제 없음 | https://tinymlcontest.github.io/TinyML-Design-Contest-2023/Problems.html | 5초, 250Hz(1,250 samples), 가중치 2로 recall에 더 큰 가중을 둔다고 명시. 대회 이름은 "ACM/IEEE TinyML Design Contest at ICCAD" |
| tinyml-contest-2023 | 순위표·Generalization | 문제 없음 | https://tinymlcontest.github.io/TinyML-Design-Contest-2023/Winners.html | 1위 0.9736/1.52/12.83, 2위 0.9445/1.33/12.19, 3위 0.9223/0.89/10.25, 11위 EmPasLab2 0.9470/12.76/26.63, Generalization 0.842105 |
| tinyml-contest-2023 | 내부 계산 | 문제 없음 | 본문 재계산 | 파라미터 20,713, MACs 39,200, 17,040/20,713=82%, Latency 8.4~14.3배, Flash 2.07배, 16/19=0.842 |
| cbs-task-assignment | 4절 실험 서술 | 문제 없음 | https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2018/pdfs/p757.pdf | 결론에 "dense environments with few agents"에서 이득이 있다고 쓰여 있음. CBM(makespan)보다 sum of costs가 항상 낮고, ECBS-TA(MinRoot)는 32×32·100대까지 실험 |
| cbs-task-assignment | 예제 출력과 본문 | 문제 없음 | 코드 재실행 | 출력이 ```text와 같음. 비용 6→7(+1) |
| feudal-networks | φ에 bias가 없는 이유 | 수정함 | https://arxiv.org/abs/1703.01161 | 논문 근거: "Since φ has no biases it can never produce a constant non-zero vector"(Manager의 목표가 항상 정책에 영향). 수정 후: "$$\phi$$는 bias가 없어서 목표와 무관한 상수 벡터를 만들 수 없으므로, Manager의 목표는 항상 정책에 영향을 줍니다." |
| feudal-networks | Atari 결과 게임 목록 | 수정함 | https://arxiv.org/abs/1703.01161 | 논문은 Ms. Pacman, Amidar, Gravitar에서 크게 앞서고 Enduro에서는 LSTM이 완전히 실패했으며, Seaquest와 Breakout에서는 LSTM baseline이 더 좋았다고 보고함. 게임 목록을 "Ms. Pac-Man, Amidar, Gravitar, Enduro 등"으로 바꾸고 "일반 게임에서도 성능을 잃지 않았습니다"를 "장기 계획이 덜 중요한 Seaquest, Breakout에서는 LSTM baseline이 더 좋았습니다."로 수정 |
| feudal-networks | Seaquest 시각화 절차 | 수정함 | https://arxiv.org/abs/1703.01161 | 논문은 군집화하지 않고 기록된 목표 하나를 골라 Worker에 고정 입력으로 줌. "목표를 몇 개의 군집으로 나눠, 각 목표를 받았을 때"를 "목표 하나를 골라 Worker에 고정 입력으로 주고, 그때"로 수정. sub-policy 3이 산소를 보충하러 올라간다는 서술은 논문과 같음 |
| feudal-networks | Dilated LSTM | 문제 없음 | https://arxiv.org/abs/1703.01161 | r=10(=c), 같은 θ^LSTM 공유, 출력은 이전 c개 출력을 pooling |
| feudal-networks | Dayan & Hinton 연도 | 문제 없음 | https://arxiv.org/abs/1703.01161 | FuN 참고문헌도 "In NIPS … 1993"(NIPS 5 논문집)으로 표기 |
| h-dqn | "약 230만 step 학습" | 수정함 | https://arxiv.org/abs/1604.06057 | 논문 Fig. 6 설명은 controller 사전 학습 약 230만(2.3 million) step 뒤 공동 학습 약 200만 step. 수정 후: "controller를 약 230만 step 사전 학습한 뒤, 약 200만 step 더 함께 학습" |
| h-dqn | 결과 표(0 / 4.16 / 약 400) | 문제 없음 | https://arxiv.org/abs/1604.06057 | "DQN achieves a score of 0", "Gorila DQN … 4.16", "reward of around +400 per episode" |
| h-dqn | 목표 순서·열쇠 빈도 | 문제 없음 | https://arxiv.org/abs/1604.06057 | Fig. 6(b)에서 열쇠 선택 빈도가 증가함. Fig. 7은 아래 오른쪽 사다리 → 열쇠 → 오른쪽 위 문 순서이고, 본문의 "다시 사다리"는 예시 수준이라 그대로 둠 |
| h-dqn | 학습 순서 | 문제 없음 | https://arxiv.org/abs/1604.06057 | 1단계에서 ε2=1로 controller를 사전 학습하고, ε1,g를 성공률에 따라 annealing함 |
| primal | inflation factor 설명 | 수정함 | https://arxiv.org/abs/1809.03531 | 논문은 ODrM\*를 inflation factor ϵ=1.5, 10의 준최적(suboptimal) 계획기로 사용. "충돌 회피 계수"를 "준최적 허용 계수"로 수정 |
| primal | Environment Sampling 근거 | 수정함 | https://arxiv.org/abs/1809.03531 | 논문 근거: 크기와 밀도를 균일하게 뽑으면 상호작용이 부족해서 "favors smaller and denser environments"로 뽑음. "밀도를 균일하게"를 "크기와 밀도를 균일하게"로, "다양한 밀도를 섞어 학습합니다"를 "작고 장애물이 많은 환경이 더 자주 나오도록 뽑아 학습합니다"로 수정 |
| primal | 실제 로봇·드론 실험 | 수정함 | https://arxiv.org/abs/1809.03531 | 논문의 검증 실험은 공장 모형에서 실제 지상 로봇(AGV) 2대와 시뮬레이션 로봇이 함께 움직이는 하이브리드 실험이고, 드론은 없음. 수정 후: "실제 로봇과 시뮬레이션 로봇이 함께 움직이는 공장 모형 실험으로" |
| primal | FOV·256 step·σ_H·blocking·보상표·유효하지 않은 행동 | 문제 없음 | https://arxiv.org/abs/1809.03531 | 10×10 FOV, "every n = 256 steps", σH=0.01, 벌점 −2와 10 step 기준, Table I 값, 유효한 행동만 샘플링(음의 보상보다 안정적), 직전 위치로 복귀 금지 모두 논문과 같음. 최대 1024대 실험 |
| diffusion-ql | D4RL 결과 서술 | 수정함 | https://arxiv.org/abs/2208.06193 | Table 1 영역별 평균(Diffusion-QL / 최고 baseline): Gym 88.0/77.6, AntMaze 69.6/63.0, Adroit 65.1/56.5, Kitchen 69.0/53.3. 논문은 "outperforms all prior methods by a clear margin on all domain"이라고 씀. AntMaze·Adroit의 차이가 오히려 작아서 "좋거나 비슷한 성능 … 특히 AntMaze와 Adroit처럼 행동 분포가 복잡한 과제에서 차이가 컸습니다"를 "높은 평균 점수를 기록했고, 특히 여러 하위 과제를 순서대로 풀어야 하는 Kitchen에서 차이가 가장 컸습니다"로 수정 |
| diffusion-ql | bandit 비교·N=5 | 문제 없음 | https://arxiv.org/abs/2208.06193 | 4개 Gaussian 모드의 2D bandit에서 BC-MLE/CVAE/MMD와 BCQ, BEAR-MMD, TD3+BC를 비교. "N = 5 is good enough", 수식과 α=η/E\|Q\|도 논문과 같음 |
| diffusion-ql | 코드 주석 출력 | 문제 없음 | 코드 재실행 | 평균 -0.005, 표준편차 1.003, 0.0 |
