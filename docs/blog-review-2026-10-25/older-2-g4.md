# 블로그 교정 보고서: older-2-g4

대상 5개 파일(_posts/). 교정 원칙: 맞춤법·띄어쓰기·문장부호·비문·어색한 표현·글 안의 표기 통일만 고치고, 내용·수치·순서·소제목·말투·코드 블록·수식·링크·Liquid 태그·front matter·기술 용어·고유명사는 건드리지 않았습니다. 애매하면 고치지 않았고, 내용이 의심스러운 부분은 "내용 확인 필요"에만 적었습니다. 코드 블록 출력(Noisy Network, Double DQN)은 로컬에서 다시 실행해 본문 출력과 같은 것을 확인했습니다.

## 2026-08-12-c51-distributional-rl.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- "C51은 사람 대비 정규화 점수의 평균과 중앙값 모두 당시 최고를 기록" — 논문 Table 1에 UNREAL도 함께 실려 있어, 비교 범위(value-based 방법 한정)를 확인할 필요가 있음.
- "51개일 때 가장 효율적", "Atom 수가 적으면(5개 등) 성능이 낮았고" — 논문의 atom 수 실험(5/11/21/51)과 일치하는지 확인.
- projection 예시 수치(Δz = 0.4, 0에 0.25, 0.4에 0.75)와 출력 블록의 일치 여부(내부 계산).

## 2026-08-10-noisy-network.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 연속된 일관된 탐험(예: 한 방향으로 꾸준히 가 보기) | 연속적이고 일관된 탐험(예: 한 방향으로 꾸준히 가 보기) | 어색한 표현 |

### 내용 확인 필요

- 표의 "사용처: A3C = Independent Gaussian, DQN·Dueling = Factorized Gaussian" — 논문 설정과 일치하는지 확인.
- "사람 대비 정규화 점수의 평균과 중앙값이 세 알고리즘 모두에서 향상" — 논문 표 수치로 확인.
- "마지막 층에서는 모든 게임에서 Σ̄가 줄었지만, 그 앞 층에서는 일부 게임에서만 줄었습니다" — 논문 Figure 설명과 일치하는지 확인.

## 2026-08-08-neural-style-transfer.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- "논문 예시: α/β = 10^-3, 10^-4" — CVPR 2016 판(arXiv 1508.06576 판과 수치가 다를 수 있음)의 실제 비율 확인.
- "학습 파라미터 없이 차이만 계산하므로, 글꼴이나 손글씨 숫자처럼 단순하고 통제된 환경에서만 동작" — 논문에서 이 한계가 texture transfer가 아니라 이전 content/style 분리 연구에 대한 지적인지 확인.
- "Image Synthesis 카테고리의 첫 글" — 같은 카테고리의 다른 글과 날짜 비교(내부).
- "VGG-19 사전학습 가중치(약 550MB)", "`tolerance_grad`, 기본 1e-7" — PyTorch 공식 문서로 확인.

## 2026-08-04-double-dqn.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 표의 게임별 점수(Road Runner 233%→617%, Asterix 70%→180%, Zaxxon 54%→111%, Double Dunk 17%→397%) — 논문 수치 확인.
- "전체 57개 게임에서도 평균·중앙값 점수가 DQN보다 크게 올랐고" — 논문의 주 평가가 49개 게임이었던 것으로 기억되어 게임 수 확인 필요.
- "max로 고른 추정값은 평균 +0.6~+2.0" — 출력 블록(+0.567 ~ +2.044)과의 일치 여부(내부).

## 2026-08-01-textdiffuser.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 글자(text rendering)에서는 효율이 낮습니다. | 글자(text rendering)에서는 성능이 떨어집니다. | 어색한 표현 |

### 내용 확인 필요

- "키워드 길이(글자 수) 정보를 함께 넣어 박스 너비가 글자 수와 맞게 합니다" — 논문은 글자 수가 아니라 렌더링된 키워드의 너비(width embedding)를 넣는 것으로 기억되어 확인 필요.
- "λ_char를 0, 0.001, 0.01, 0.1, 1로 바꿔 실험했고, 0.01일 때 가장 좋았습니다(λ_char = 0 대비 9.8% 향상)" — 논문 ablation 수치 확인.
- "Stable Diffusion, ControlNet, DeepFloyd IF 등과 비교해 OCR 지표가 크게 높았고, FID·CLIP score는 비슷한 수준을 유지" — 논문 Table의 FID·CLIP score 수치 확인.
- References의 TextDiffuser-2 "2023" — 학회 게재(ECCV 2024) 여부 확인.

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| 2026-08-12-c51-distributional-rl.md | 평균·중앙값 "당시 최고" | 문제 없음 | 논문 Figure 6 표: C51 mean 701%, median 178%로 DQN·DDQN·Duel·Prior·Pr.Duel보다 높음. UNREAL(880%/250%)은 "not altogether comparable"(비동기·게임별 튜닝)로 각주 처리. 본문도 value-based 방법과의 비교로 한정함. https://arxiv.org/abs/1707.06887 | - |
| 2026-08-12-c51-distributional-rl.md | atom 수 실험(51개 최적, 5개 저조) | 문제 없음 | 5.1절·Figure 3: 5/11/21/51 atoms 비교, "using too few atoms can lead to poor behaviour, and that more always increases performance". 실험한 값 중 51이 가장 좋음. https://arxiv.org/abs/1707.06887 | - |
| 2026-08-12-c51-distributional-rl.md | projection 예시 수치와 출력 블록 | 문제 없음 | 내부 계산: b = (0.3+10)/0.4 = 25.75 → l=25(z=0)에 0.25, u=26(z=0.4)에 0.75, 기댓값 0.3. 출력 블록과 일치. | - |
| 2026-08-10-noisy-network.md | A3C=Independent, DQN·Dueling=Factorized | 문제 없음 | 3절: "we use factorised noise for DQN and Duelling and independent noise for the distributed A3C". 초기화 값(√(3/p), 0.017, σ0=0.5)도 일치. https://arxiv.org/abs/1706.10295 | - |
| 2026-08-10-noisy-network.md | 평균·중앙값 세 알고리즘 모두 향상 | 문제 없음 | Table 1: DQN 319/83→379/123, Dueling 524/132→633/172, A3C 293/80→347/94 (mean/median). https://arxiv.org/abs/1706.10295 | - |
| 2026-08-10-noisy-network.md | Σ̄: 마지막 층은 모두 감소, 앞 층은 일부만 | 문제 없음 | "Σ̄ of the last layer ... decreases ... in all cases, whereas in the case of the penultimate layer this only happens for 2 games out of 5" (측정한 5개 게임 기준). https://arxiv.org/abs/1706.10295 | - |
| 2026-08-08-neural-style-transfer.md | α/β 예시 10^-3, 10^-4 | 문제 없음 | CVPR판: Fig 3은 1×10^-3, 8×10^-4, 5×10^-3, 5×10^-4, Fig 4 trade-off에서 α/β = 1×10^-4 사용. 두 값 모두 논문에 나옴. https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf | - |
| 2026-08-08-neural-style-transfer.md | 글꼴·손글씨 한계의 귀속 | 수정함 | 논문은 texture transfer의 한계를 "use only low-level image features"로 들고, 글꼴·손글씨 숫자·얼굴 같은 통제된 이미지에 한정된 것은 이전의 content/style 분리 표현(factorised representations)이라고 씀. "학습 파라미터 없이 차이만 계산하므로"라는 인과도 논문에 없음. (같은 URL) | "학습 파라미터 없이 차이만 계산하므로, 글꼴이나 손글씨 숫자처럼 **단순하고 통제된 환경**에서만 동작" → "content와 style을 분리하는 표현도 얼굴, 글꼴, 손글씨 숫자처럼 **단순하고 통제된 환경**에서만 얻을 수 있었음" |
| 2026-08-08-neural-style-transfer.md | "Image Synthesis 카테고리의 첫 글" | 문제 없음 | 내부 확인: 해당 카테고리 글은 이 글(08-08)과 DeepSVG(08-15) 두 개뿐. | - |
| 2026-08-08-neural-style-transfer.md | 가중치 약 550MB, tolerance_grad 기본 1e-7 | 문제 없음 | VGG19_Weights.IMAGENET1K_V1(=DEFAULT) 548.1MB; LBFGS 시그니처 tolerance_grad=1e-07, max_iter=20. https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html , https://docs.pytorch.org/docs/2.14/generated/torch.optim.LBFGS.html | - |
| 2026-08-04-double-dqn.md | 게임별 점수 표 | 문제 없음 | "Road Runner (from 233% to 617%), Asterix (from 70% to 180%), Zaxxon (from 54% to 111%), and Double Dunk (from 17% to 397%)". https://arxiv.org/abs/1509.06461 | - |
| 2026-08-04-double-dqn.md | 평균·중앙값 집계 게임 수 "57개" | 수정함 | Table 1·2의 요약 통계는 "49 games"(Mnih et al. 2015와 같은 목록). 57개는 Figure 4의 게임별 점수(추가 8개 포함)에만 해당. https://arxiv.org/abs/1509.06461 | "전체 57개 게임에서도 평균·중앙값" → "전체 49개 게임에서도 평균·중앙값" |
| 2026-08-04-double-dqn.md | "+0.6~+2.0"과 출력 블록 | 문제 없음 | 내부: 출력 +0.567~+2.044를 반올림한 값. 코드도 로컬에서 재실행해 출력이 동일함을 확인. | - |
| 2026-08-01-textdiffuser.md | width embedding = 글자 수? | 수정함 | 3.1절 "we encode the width of keywords with an embedding layer", 5.1절 "Arial.ttf ... font size to 24 to obtain the width embedding" → 글자 수가 아니라 렌더링된 폭. https://arxiv.org/html/2305.10855v5 | "키워드 길이(글자 수) 정보를 함께 넣어 박스 너비가 글자 수와 맞게 합니다." → "키워드 너비(Arial 폰트로 렌더링했을 때의 폭) 정보를 함께 넣어 박스 너비가 키워드 길이와 맞게 합니다." |
| 2026-08-01-textdiffuser.md | λ_char 0/0.001/0.01/0.1/1, 0.01 최적, 9.8% | 문제 없음 | Table 2: 0.01에서 Acc 0.494 (0일 때 0.396), 본문 "the score is increased by 9.8%". 엄밀히는 9.8%p이지만 논문 표현과 같음. https://arxiv.org/html/2305.10855v5 | - |
| 2026-08-01-textdiffuser.md | FID·CLIP score "비슷한 수준" | 수정함 | Table 4: FID SD 51.295 / ControlNet 51.485 / DeepFloyd 34.902 / TextDiffuser 38.758, CLIPScore 0.3015 / 0.3424 / 0.3267 / 0.3436(최고). FID는 비슷하거나 더 좋고 CLIP score는 가장 높음. 비교 대상(SD, ControlNet, DeepFloyd)은 일치. https://arxiv.org/html/2305.10855v5 | "FID·CLIP score는 비슷한 수준을 유지했습니다." → "FID·CLIP score는 비슷하거나 더 나은 수준이었습니다." |
| 2026-08-01-textdiffuser.md | TextDiffuser-2 연도 | 수정함 | ECCV 2024 oral로 게재(arXiv 2311.16465는 2023년 11월). https://eccv2024.ecva.net/virtual/2024/oral/528 | References `for Text Rendering," 2023.` → `for Text Rendering," ECCV 2024.` |
