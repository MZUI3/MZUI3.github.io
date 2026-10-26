# 블로그 교정 보고서: older-2-g2

대상 5개 파일(_posts/). 교정 원칙: 맞춤법·띄어쓰기·문장부호·비문·어색한 표현·한 글 안의 표기 통일만 고치고, 내용·수치·순서·소제목·말투·코드 블록·수식·링크·Liquid 태그·front matter·기술 용어·고유명사는 건드리지 않았습니다. 애매하면 고치지 않았고, 내용이 의심스러운 부분은 "내용 확인 필요"에만 적었습니다. `{% post_url %}` 대상(icm-curiosity, random-network-distillation, ride, never-give-up, revisiting-intrinsic-reward, agent57, count-based-exploration)은 모두 `_posts/`에 있고 설명과 맞습니다.

## 2026-09-07-ride.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- "같은 ObstructedMaze를 singleton 버전과 procedurally-generated 버전으로 비교" — 논문에서 singleton/절차적 생성 비교에 쓴 환경이 ObstructedMaze가 맞는지.
- "MultiRoomN7S4에서 100개 에피소드의 intrinsic reward 평균을 heatmap으로", "MultiRoomN10S6에서 ... 5천만 frame 동안 탐험 ... RIDE는 모든 방에 도달" — 환경 이름, 에피소드 수, frame 수가 논문과 맞는지.
- "RND, ICM 같은 curiosity 방법은 학습 중 intrinsic reward가 0으로 수렴했지만, RIDE는 의미 있는 크기를 유지" — 논문 서술과 맞는지.
- "Singleton 환경 ... ICM은 extrinsic과 intrinsic reward를 함께 쓰면 성능이 떨어지는 경우가 있었지만, RIDE는 그렇지 않았습니다" — 논문(VizDoom/Mario 실험) 서술과 맞는지.

## 2026-09-05-exploration-via-disagreement.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 이런 곳에서는 예측 오차가 아무리 학습해도 줄지 않아 | 이런 곳에서는 아무리 학습해도 예측 오차가 줄지 않아 | 비문 |
| 오히려 noisy TV가 없는 환경보다도 | 오히려 Noisy TV가 없는 환경보다도 | 통일 |

### 내용 확인 필요

- "Disagreement는 ... 오히려 Noisy TV가 없는 환경보다도 빠르게 학습한 경우가 있었습니다" — 논문 결과와 맞는지.
- Baseline 표(Pathak et al., Burda et al., Pred-Error Variance, Bayesian Disagreement)가 논문의 비교 대상과 맞는지.
- "REINFORCE로 최적화한 Disagreement보다, 미분 가능한 Disagreement가 매우 짧은 시간 안에 높은 상호작용 비율" 및 실제 로봇 결과 서술이 논문과 맞는지.

## 2026-09-02-random-network-distillation.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- "(논문 구현은 predictor를 target보다 층을 더 깊게 만들어 ...)" — 논문 부록의 네트워크 구조와 맞는지.
- MNIST 실험 서술("숫자 0과 특정 숫자 하나로 학습했을 때 학습하지 않은 숫자에 대해서는 계속 오차가 컸습니다") — 논문 실험 설정·결론과 맞는지.
- "6개의 어려운 탐험 게임에서 RND가 대부분 가장 좋았고, 두 게임에서 당시 SOTA", "24개 방 중 대부분을 탐험" — 논문 결과와 맞는지.

## 2026-08-29-icm-curiosity.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- VizDoom 설정: "Dense: 17개 위치 중 무작위", "Sparse / Very sparse: 특정 방(13번, 17번)" — 논문과 맞는지.
- 비교 대상 "ICM(aenc)"(autoencoder 특징) — 논문에 실제로 있는 baseline인지.
- "Dense에서 ICM(pixel)이 ICM보다 낮았던 이유를 ... 시작할 때마다 바뀌는 벽 텍스처로 추측" — 논문의 설명(어느 설정, 어떤 이유)과 맞는지.
- "VizDoom에서 2,100 step 에피소드 중 약 250 step 만에 가장 먼 방까지 탐험" — 수치가 논문과 맞는지.

## 2026-08-26-count-based-exploration.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- "가장 어려운 Atari 게임 5개에서, 보너스 유무와 optimistic initialization을 비교" — 게임 수와 비교 대상이 논문과 맞는지.
- "Atari 60개 게임에서 A3C+의 평균 학습 점수가 A3C보다 조금 더 높았고" — 논문 서술과 맞는지.
- CTS 전처리 "42×42, 8단계 밝기", "위·왼쪽 이웃 픽셀을 문맥으로" — 논문 부록과 맞는지.

## 내용 확인 결과 (2026-10-25)

원문 PDF는 arXiv에서 받아 `pdftotext`로 텍스트를 뽑아 대조했습니다. RND 항목 4(RNN vs CNN)는 Phase 2에서 논문을 읽다가 새로 발견해 추가한 항목입니다.

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| 2026-09-07-ride.md | singleton/절차적 생성 비교 환경이 ObstructedMaze인지 | 문제 없음 | 6.1.2절: ObstructedMaze2Dlh의 singleton 버전에서는 Count, RND, IMPALA도 풀었음 → 절차적 생성 버전이 훨씬 어려움. https://arxiv.org/abs/2002.12292 | - |
| 2026-09-07-ride.md | MultiRoomN7S4 heatmap과 "100개 에피소드 평균" | 수정함 | Figure 4는 샘플한 환경 하나의 intrinsic reward heatmap이고, "averaged over 100 episodes"는 행동 종류별 평균 보상인 Table 1의 설명. 둘을 섞어 서술했음. https://arxiv.org/abs/2002.12292 | "MultiRoomN7S4에서 100개 에피소드의 intrinsic reward 평균을 지도 위에 heatmap으로 그렸습니다." → "MultiRoomN7S4에서 intrinsic reward를 지도 위에 heatmap으로 그리고, 행동 종류별 intrinsic reward를 100개 에피소드에 걸쳐 평균 냈습니다." |
| 2026-09-07-ride.md | MultiRoomN10S6, 5천만 frame, RIDE만 모든 방 도달 | 문제 없음 | 6.1.3절·Figure 7: 외부 보상 없이 50M frame 학습. 절차적 생성 맵에서 Count/RND/ICM은 각각 4·3·2번째 방까지만 갔고, RIDE는 모든 방을 탐험. https://arxiv.org/abs/2002.12292 | - |
| 2026-09-07-ride.md | RND·ICM 보상이 0으로 수렴, RIDE는 유지 | 문제 없음 | Figure 5(MultiRoomN12S10): ICM, RND, Count 보상은 학습 초반에 매우 낮은 값으로 수렴하고, RIDE 보너스는 100M frame 뒤에도 더 높은 값을 유지. https://arxiv.org/abs/2002.12292 | - |
| 2026-09-07-ride.md | ICM은 extrinsic+intrinsic을 함께 쓰면 성능이 떨어짐 | 문제 없음 | 6.2절(Mario): intrinsic과 extrinsic 보상을 함께 쓰면 ICM이 학습을 방해해, 다른 방법보다 늦게, 낮은 값으로 수렴. VizDoom에서는 RIDE가 ICM과 비슷. https://arxiv.org/abs/2002.12292 | - |
| 2026-09-05-exploration-via-disagreement.md | Noisy TV가 있는 환경에서 없는 환경보다 빠르게 학습한 경우 | 확인 불가 | 본문(4.2절 B, Figure 4)은 TV가 없으면 baseline과 비슷하고, TV가 있으면 baseline보다 낫다고만 함. TV 유무 사이의 비교는 그래프로만 확인할 수 있는데, 그래프는 텍스트로 읽을 수 없었음. https://arxiv.org/abs/1906.04161 | - |
| 2026-09-05-exploration-via-disagreement.md | Baseline 표(Pathak, Burda, Pred-Error Variance, Bayesian Disagreement) | 문제 없음 | 3절 Baseline Comparisons: Pathak et al. [ICML 2017], Burda et al. [ICLR 2019](random feature 예측 오차), Pred-Error Variance(예측 오차의 분산), Bayesian Disagreement(Dropout NN). https://arxiv.org/abs/1906.04161 | - |
| 2026-09-05-exploration-via-disagreement.md | REINFORCE 대비 미분 가능 버전, 실제 로봇 결과 | 문제 없음 | 4.3절 B1: 물체 접촉 빈도로 측정했고, REINFORCE보다 "order of magnitude faster". B2: 실제 로봇은 1000개 미만 샘플로 학습했고, 처음 보는 물체와 상호작용한 비율이 약 67%(random 17%, REINFORCE 기반 curiosity 1%). https://arxiv.org/abs/1906.04161 | - |
| 2026-09-02-random-network-distillation.md | predictor가 target보다 층이 더 깊은지 | 문제 없음 | 논문 A.5는 "Exact details are given in the code"라고만 함. 공식 코드에서 target은 conv 3층+fc 1층, predictor는 conv 3층+fc 3층. https://github.com/openai/random-network-distillation/blob/master/policies/cnn_gru_policy_dynamics.py | - |
| 2026-09-02-random-network-distillation.md | MNIST 실험 서술 | 수정함 | 2.2절·Figure 2: 0과 target class의 비율을 바꿔 학습하고, 처음 보는 target class 테스트 이미지의 MSE를 쟀더니, 학습 데이터 속 target class 수가 많을수록 오차가 줄었음. "학습하지 않은 숫자"를 따로 테스트한 실험이 아님. https://arxiv.org/abs/1810.12894 | "숫자 0과 특정 숫자 하나로 학습했을 때 **학습하지 않은 숫자에 대해서는 계속 오차가 컸습니다.**" → "숫자 0과 특정 숫자 하나를 섞어 학습했을 때 **학습 데이터에 그 숫자가 적을수록, 처음 보는 그 숫자 이미지에 대한 오차가 컸습니다.**" |
| 2026-09-02-random-network-distillation.md | "두 게임에서 당시 SOTA" | 수정함 | Figure 7 캡션: "RND achieves state of the art performance on Gravitar, Montezuma's Revenge, and Venture"(세 게임). https://arxiv.org/abs/1810.12894 | "**두 게임에서 당시 SOTA**" → "**세 게임(Gravitar, Montezuma's Revenge, Venture)에서 당시 SOTA**" |
| 2026-09-02-random-network-distillation.md | (추가) RNN vs CNN 정책: "CNN 정책이 비슷하거나 더 좋았습니다" | 수정함 | 3.5절: γ_E = 0.99일 때는 예상과 달리 recurrent 정책이 더 나빴지만, γ_E = 0.999에서는 RNN이 모든 규모에서 CNN보다 나았고, 여러 게임에서도 RNN이 앞서는 경우가 더 많았음. https://arxiv.org/abs/1810.12894 | "실제로는 **CNN 정책이 비슷하거나 더 좋았습니다.**" → "실제로는 $$\gamma_E = 0.99$$ 설정에서 **CNN 정책이 더 좋았습니다.** (다만 $$\gamma_E = 0.999$$에서는 RNN 정책이 더 좋았습니다.)" |
| 2026-08-29-icm-curiosity.md | Dense 17개 위치, sparse/very sparse 13·17번 방 | 문제 없음 | 4.1절: dense는 17개 spawn 위치 중 무작위. sparse와 very sparse는 각각 Room-13, Room-17에서 시작(목표까지 270·350 step). https://arxiv.org/abs/1705.05363 | - |
| 2026-08-29-icm-curiosity.md | baseline "ICM(aenc)" | 수정함 | 3절 Baseline Methods: 비교 대상은 ε-greedy A3C, ICM-pixels + A3C, TRPO로 학습한 VIME. autoencoder baseline은 arXiv 판과 ICML(PMLR) 판 어디에도 없음. https://arxiv.org/abs/1705.05363 , https://proceedings.mlr.press/v70/pathak17a.html | "현재 상태만으로 autoencoder 특징을 쓰는 **ICM(aenc)**입니다." → "TRPO로 학습한 **VIME**입니다." |
| 2026-08-29-icm-curiosity.md | Dense에서 ICM(pixel)이 낮았던 이유(텍스처) | 문제 없음 | 4.1절: dense에서는 에피소드마다 텍스처가 다른 17개 방 중 한 곳에서 시작해, 텍스처가 많아질수록 픽셀 예측 모델을 학습하기 어렵다고 설명. https://arxiv.org/abs/1705.05363 | - |
| 2026-08-29-icm-curiosity.md | "2,100 step 에피소드 중 약 250 step 만에 가장 먼 방" | 수정함 | 4.2절: 에피소드는 2100 step에서 끝나고, 가장 먼 방은 최적으로 움직여도 250 step 넘게 떨어져 있음. 많은 경우 맵 전체를 지나 가장 먼 방까지 갔음. 250 step은 도달하는 데 걸린 시간이 아니라 최적 거리. https://arxiv.org/abs/1705.05363 | "VizDoom에서 2,100 step 에피소드 중 약 250 step 만에 **가장 먼 방**까지 탐험했습니다." → "VizDoom에서 2,100 step 에피소드 안에, 최적 경로로도 250 step 넘게 떨어진 **가장 먼 방**까지 탐험하는 경우가 많았습니다." |
| 2026-08-26-count-based-exploration.md | 어려운 게임 5개, optimistic initialization 비교 | 문제 없음 | 6.1절: ALE 60개 게임 중 "hard" 게임 5개를 골랐고, Machado et al.(2015)의 optimistic initialization과도 비교(Figure 2). https://arxiv.org/abs/1606.01868 | - |
| 2026-08-26-count-based-exploration.md | A3C+ "평균 학습 점수가 조금 더 높았고" | 수정함 | 6.2절·Figure 4: 60개 게임에서 학습 중 점수의 median이 A3C보다 조금 높았고("slightly superior median performance"), 적어도 1/4의 게임에서는 A3C보다 크게 나았음. 평균이 아니라 중앙값. https://arxiv.org/abs/1606.01868 | "A3C+의 평균 학습 점수가 A3C보다" → "A3C+의 학습 점수 중앙값이 A3C보다" |
| 2026-08-26-count-based-exploration.md | CTS 전처리 42×42·8단계, 위·왼쪽 이웃 문맥 | 문제 없음 | 부록: 흑백화한 뒤 42×42로 축소하고 3-bit(8단계) greyscale을 씀. 각 픽셀의 부모는 "upper-left neighbours"인 (i−1,j), (i,j−1), (i−1,j−1), (i+1,j−1). https://arxiv.org/abs/1606.01868 | - |
