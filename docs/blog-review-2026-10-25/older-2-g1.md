# 블로그 교정 보고서: older-2-g1

대상 5개 파일(_posts/). 교정 원칙: 맞춤법·띄어쓰기·문장부호·비문·어색한 표현·글 안의 표기 통일만 고치고, 내용·수치·순서·소제목·말투·코드 블록·수식·링크·Liquid 태그·front matter·기술 용어·고유명사는 그대로 둡니다. 애매하면 고치지 않고, 내용이 의심스러운 부분은 "내용 확인 필요"에만 적습니다. 모든 `{% post_url %}` 대상 파일이 `_posts/`에 있는지 확인했고, 코드 블록은 실행해 주석에 적힌 출력과 일치하는지 확인했습니다(Agent57 bandit, Revisiting EpisodicBonus, NGU episodic_reward, COMA advantage 모두 일치).

## 2026-09-19-diffuser.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 8절 "여러 목표를 순서대로 방문하는 **multi-task** 설정": 논문의 Multi2D가 "목표를 순서대로 여러 개 방문"하는 설정인지, 에피소드마다 목표를 다시 뽑는 설정인지 확인 필요.
- "더 알아보기" 박스 "denoising을 $$N$$번(논문 설정에서 수십 번) 반복": 논문/공식 코드의 diffusion step 수 확인 필요.

## 2026-09-17-agent57.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| (조건 하에) | (조건하에) | 띄어쓰기 |

### 내용 확인 필요

- 1절 표의 "사람 점수를 넘은 게임 수"(MuZero 51, NGU 51, R2D2 52, R2D2 + bandit 54, Agent57 57)와 도입부 "NGU는 … 51개에서 사람을 넘었지만, 평균 점수는 R2D2에 못 미쳤습니다": 논문 Table 1과 대조 필요.
- 4절 "학습 초반 … 큰 β, 작은 γ를 고르고, 학습이 진행될수록 반대 쌍(작은 β, 큰 γ)": 논문 근거 확인 필요.
- "Discount γ를 최대 0.9997까지", "역전파 길이(unroll)를 늘려": 논문 수치 확인 필요.
- 5절 "50억 frame 시점에 51개 게임에서 사람을 넘었고 … 6개가 남았습니다", "Skiing을 약 780억 frame에서": 논문 확인 필요.
- 5절 "MuZero는 … capped 점수는 가장 낮았습니다. Venture 같은 게임에서 무작위 정책과 같은 점수": 확인 필요.
- Ablation "Random Coin(15×15 grid, … 최대 200 step)", "R2D2에 붙였을 때 성능이 약 20% 올랐고": 논문 수치 확인 필요.
- Ablation "Skiing에서는 초반에 높은 β(탐험)를 선호하다가, 후반에는 낮은 β와 높은 γ를 선호": 논문 Fig. 8 설명과 대조 필요(2단계에서 추가로 발견한 항목).

## 2026-09-14-revisiting-intrinsic-reward.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 결과는 **성능이 크게 떨어지지 않았습니다.** | 결과적으로 **성능이 크게 떨어지지 않았습니다.** | 비문 |

### 내용 확인 필요

- 저자·학회 "Wang et al., ICLR 2023"과 OpenReview 링크(id j3GK3_xZydY): 확인 필요.
- 3절 "Life-long만 쓰면 처음 두 개의 방을 넘어가지 못했습니다": 논문 결과 확인 필요.
- 4절 "미니배치 안에서 life-long 보상의 순서를 무작위로 섞어": 논문의 shuffle 실험 방식 확인 필요.

## 2026-09-12-never-give-up.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 5절 "57개 중 51개 게임에서 사람 점수를 넘었습니다", "평균 점수로는 … R2D2를 넘지 못했지만": 확인 필요(Agent57 글과 같은 수치).
- "$$L = 5$$", Pitfall! "처음으로 양수 점수": 논문 확인 필요.
- 5절 "Pong, Breakout 같은 dense reward 게임에서도 성능을 거의 잃지 않았습니다. 탐험 보너스가 쉬운 게임을 방해하지 않았다는 뜻입니다": 논문 Table 2와 대조 필요(2단계에서 추가로 발견한 항목).
- 4절 UVFA 링크가 HER 글(`2026-08-19-hindsight-experience-replay`)을 가리킴: 대상 파일은 존재하고 UVFA를 다루므로 링크 자체는 정상으로 보임.

## 2026-09-09-coma.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 5절 맵 목록(3m, 5m, 5w, 2d_3z)과 행동 공간(이동 4방향, 공격, 정지, no-op): 논문 확인 필요.
- 5절 "COMA는 대부분의 맵에서 승률이 가장 높았고, 특히 유닛 종류가 섞인 맵에서 차이가 컸습니다", "일부 맵에서는 centralised 접근을 쓰는 다른 방법들과 비슷하거나 더 좋은 성능": 논문 결과 확인 필요.
- 마지막 박스 "SMAC에서는 COMA가 QMIX 같은 가치 분해 방법보다 성능이 낮게 나왔습니다(Samvelyan et al., 2019)": 확인 필요.

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| 2026-09-19-diffuser.md | Multi2D를 "여러 목표를 순서대로 방문하는" 설정으로 설명 | 수정함 | 논문 부록: Multi2D는 "randomize the goal location at the beginning of each episode", 즉 에피소드마다 목표 하나를 새로 뽑음. https://arxiv.org/html/2205.09991v2 | "여러 목표를 순서대로 방문하는 **multi-task** 설정" → "목표 위치를 에피소드마다 무작위로 바꾸는 **multi-task** 설정" |
| 2026-09-19-diffuser.md | denoising 횟수 "(논문 설정에서 수십 번)" | 수정함 | 논문: locomotion N=20, block stacking N=100. 공식 코드 config/maze2d.py: n_diffusion_steps 256. https://arxiv.org/html/2205.09991v2 , https://github.com/jannerm/diffuser/blob/maze2d/config/maze2d.py | "(논문 설정에서 수십 번)" → "(논문·공식 코드 설정에서 수십~수백 번)" |
| 2026-09-17-agent57.md | 표의 게임 수(MuZero 51, NGU 51, R2D2 52, R2D2+bandit 54, Agent57 57), NGU 평균 < R2D2 | 문제 없음 | Table 1 "Number of games > human": 57/54/51/52(Retrace)/52/51(MuZero). Mean: NGU 3421.80, R2D2 4622.09. https://arxiv.org/abs/2003.13350 | - |
| 2026-09-17-agent57.md | 초반 큰 β·작은 γ, 이후 반대 쌍 | 문제 없음 | 3.1절: "natural to expect policies with higher βj and lower γj to make more progress early in training, while the opposite…as training progresses". https://arxiv.org/abs/2003.13350 | - |
| 2026-09-17-agent57.md | 최대 discount 0.9997 | 수정함 | 4절·부록 G.1: discount 범위 "ranging from 0.99 to 0.9999", γ0 = 0.9999. https://arxiv.org/abs/2003.13350 | "최대 0.9997까지" → "최대 0.9999까지" |
| 2026-09-17-agent57.md | 역전파 길이(unroll) 증가 | 문제 없음 | 4.3절: backprop through time window 80(R2D2 기본) → 160. https://arxiv.org/abs/2003.13350 | - |
| 2026-09-17-agent57.md | 50억 frame에 51개, 남은 6개, Skiing 780억 frame | 수정함 | 4.1절: 첫 5B frames에 51개 게임 돌파, 이어 Montezuma's Revenge·Pitfall!·Private Eye 같은 탐험 게임, 마지막 Skiing은 78B frames. 남은 6개에는 Skiing도 들어가므로 6개 전부를 "어려운 탐험 게임"이라 부르는 것은 부정확. https://arxiv.org/abs/2003.13350 | "Private Eye 등 어려운 탐험 게임 6개가 남았습니다." → "Private Eye 같은 어려운 탐험 게임을 포함해 6개 게임이 남았습니다." |
| 2026-09-17-agent57.md | MuZero capped 최저, Venture 무작위 수준 | 문제 없음 | 4.1절: MuZero는 uncapped mean/median 최고지만 "the lowest capped scores", Venture에서 "on par with a random policy". https://arxiv.org/abs/2003.13350 | - |
| 2026-09-17-agent57.md | Random Coin(15×15, 최대 200 step, 보상 1) | 문제 없음 | 4.2절: "empty room of size 15 × 15", "at most 200 steps", 동전을 밟으면 보상 1을 받고 종료. https://arxiv.org/abs/2003.13350 | - |
| 2026-09-17-agent57.md | meta-controller로 R2D2 약 20% 향상 | 문제 없음 | 4.4절: "enhancing the final performance of R2D2 by close to 20% CHNS". https://arxiv.org/abs/2003.13350 | - |
| 2026-09-17-agent57.md | (2단계 발견) 게임별 선택 패턴에서 Skiing을 "초반 높은 β → 후반 낮은 β·높은 γ" 예로 든 부분 | 수정함 | 4.4절: Skiing은 "the high discount combination is quickly picked up"(처음부터 한 모드 선호). 초반 탐험적·낮은 discount에서 후반 높은 discount·활용 정책으로 옮겨 간 게임은 Gravitar, Crazy Climber, Beam Rider, Jamesbond. https://arxiv.org/abs/2003.13350 | "예를 들어 Skiing에서는 초반에" → "예를 들어 Gravitar에서는 초반에" |
| 2026-09-14-revisiting-intrinsic-reward.md | Wang et al., ICLR 2023, OpenReview id j3GK3_xZydY | 문제 없음 | Kaixin Wang 외, ICLR 2023. OpenReview forum id j3GK3_xZydY. https://mlanthology.org/iclr/2023/wang2023iclr-revisiting/ | - |
| 2026-09-14-revisiting-intrinsic-reward.md | 순수 탐험에서 life-long만 쓰면 처음 두 방을 넘지 못함 | 문제 없음 | ICLR 포스터의 Pure exploration 그림(MultiRoom-N7S8/N10S10): "lifelong only" 곡선이 탐험한 방 수 약 1~1.3에 머묾. https://iclr.cc/media/PosterPDFs/ICLR%202023/10945.png | - |
| 2026-09-14-revisiting-intrinsic-reward.md | 미니배치 안에서 life-long 보상 순서를 섞는 실험 | 확인 불가 | 본문 PDF(OpenReview)는 브라우저 확인 페이지로 막혀 읽지 못했고, 초록·포스터에는 shuffle 실험이 없음. 초록에는 "lifelong intrinsic reward does not accurately reflect the novelty of states"까지만 있음. https://iclr.cc/virtual/2023/poster/10945 | - |
| 2026-09-12-never-give-up.md | 57개 중 51개 사람 초과, 평균으로는 R2D2 미달 | 문제 없음 | NGU 4.2절: "performing above human level on 51 out of the 57 games", median 1354.4%로 R2D2(1920.6%)보다 낮음. Agent57 Table 1에서도 mean NGU 3421.80 < R2D2 4622.09. https://arxiv.org/abs/2002.06038 | - |
| 2026-09-12-never-give-up.md | $$L = 5$$, Pitfall! 첫 양수 점수 | 문제 없음 | "we fix L = 5 for all our experiments". 초록: Pitfall!에서 "first algorithm to achieve non-zero rewards (with a mean score of 8,400)", 시연·수작업 특징 없이. https://arxiv.org/abs/2002.06038 | - |
| 2026-09-12-never-give-up.md | UVFA 링크가 HER 글을 가리킴 | 문제 없음 | `_posts/2026-08-19-hindsight-experience-replay.md`가 있고 UVFA를 4번 언급함(내부 확인). | - |
| 2026-09-12-never-give-up.md | (2단계 발견) "Pong, Breakout … 성능을 거의 잃지 않았습니다. 탐험 보너스가 쉬운 게임을 방해하지 않았다" | 수정함 | Table 2: Breakout NGU(N=32) 532.8 vs R2D2 837.7, Beam Rider 68.7k vs 188.2k. 본문에 "they do not match R2D2 on Breakout and Beam Rider… intrinsic signal still slightly interfere". Pong 19.6 vs 21.0, QBert 465.8k vs 408.8k. https://arxiv.org/abs/2002.06038 | "- Pong, Breakout 같은 **dense reward 게임에서도 성능을 거의 잃지 않았습니다.** 탐험 보너스가 쉬운 게임을 방해하지 않았다는 뜻입니다." → "- Pong, QBert 같은 **dense reward 게임에서도 성능을 거의 잃지 않았습니다.** 다만 Breakout, Beam Rider에서는 R2D2에 못 미쳐, 탐험 신호가 활용 정책의 학습을 약간 방해하는 것으로 보입니다." |
| 2026-09-09-coma.md | 맵(3m, 5m, 5w, 2d_3z)과 행동(move/attack/stop/noop) | 문제 없음 | 5절: "3 marines (3m), 5 marines (5m), 5 wraiths (5w), or 2 dragoons with 3 zealots (2d 3z)", "move[direction], attack[enemy id], stop, and noop". https://arxiv.org/abs/1705.08926 | - |
| 2026-09-09-coma.md | "대부분의 맵에서 승률이 가장 높았고, 특히 유닛 종류가 섞인 맵에서 차이가 컸습니다" | 수정함 | Table 1 평균 승률: COMA가 네 맵 모두 최고(87/81/82/47). 차순위와의 차이는 3m 4, 5m 10, 5w 6, 2d_3z 8 포인트로 혼합 맵(2d_3z)에서 특히 크지 않음. https://arxiv.org/abs/1705.08926 | "- COMA는 대부분의 맵에서 **승률이 가장 높았고**, 특히 유닛 종류가 섞인 맵에서 차이가 컸습니다." → "- COMA는 네 맵 모두에서 **평균 승률이 가장 높았습니다.**" |
| 2026-09-09-coma.md | centralised 방법들과 비슷하거나 더 좋음 | 문제 없음 | 6절: 최고 성능의 COMA 에이전트가 "in most settings…comparable to the best published win rates"(GMEZO, centralised DQN). https://arxiv.org/abs/1705.08926 | - |
| 2026-09-09-coma.md | SMAC에서 COMA < QMIX | 문제 없음 | SMAC 논문: "IQL, VDN, and QMIX all significantly outperform COMA". https://arxiv.org/abs/1902.04043 | - |
