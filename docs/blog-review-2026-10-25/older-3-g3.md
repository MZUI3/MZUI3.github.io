# 블로그 교정 보고서: older-3-g3

대상 5개 파일(_posts/). 교정 원칙: 맞춤법·띄어쓰기·문장부호·비문·어색한 표현·문체/용어 통일만 고치고, 내용·수치·순서·소제목·말투·코드 블록·인라인 코드·수식·링크·Liquid·front matter·기술 용어·고유명사는 건드리지 않았습니다. 애매하면 고치지 않았고, 내용이 의심스러운 부분은 "내용 확인 필요"에 적은 뒤 2단계에서 원문(논문·공식 문서)과 대조했습니다. 코드 예시는 로컬 PyTorch 2.13.0(CPU)으로 다시 실행해 본문 수치와 비교했습니다.

## 2026-07-05-trpo-to-ppo.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| conjugate gradient가 들어가 **구현이 복잡**하다. | conjugate gradient가 들어가 **구현이 복잡**합니다. | 통일 |
| dropout 같은 noise가 있는 구조와 잘 맞지 않는다. | dropout 같은 noise가 있는 구조와 잘 맞지 않습니다. | 통일 |

### 내용 확인 필요

- 6절 표 "loss의 기울기" 값: 0.5 → -0.5, 0.6 → -0.6. 단일 샘플에서 $$\partial(-r_t\hat A_t)/\partial\log\pi = -r_t\hat A_t$$이므로 -1.0, -1.2가 되어야 할 것으로 보임.
- 3절 lower bound 상수 $$C = 4\epsilon\gamma/(1-\gamma)^2$$, $$\epsilon = \max\lvert A_\pi\rvert$$ 정의.
- 10절 실험 결과(7개 MuJoCo 환경, clip/penalty 없음이 무작위 정책보다 낮음, 비교 알고리즘, Atari 두 지표의 PPO/ACER 우위), TRPO 한계(dropout·파라미터 공유와 맞지 않음).
- 9절 GAE 출력 블록과 본문 설명(λ=0은 1 + 0.99×0.5 − 0.5, λ=1은 Monte Carlo) 일치 여부.
- `{% post_url 2026-08-17-acer %}` 대상 파일 존재 여부.

## 2026-07-03-latent-diffusion-models.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 0.18215 설명: "LDM 논문은 학습 데이터의 latent 표준편차로 나눠 분산을 1 근처로 맞춘다".
- 8절 downsampling factor 결과(LDM-1,2 학습 느림, LDM-32 품질 정체, LDM-4~16 균형), "GPU 수백 일", LDM text-to-image의 $$\tau_\theta$$(Transformer, 처음부터 학습).
- 5절 수치 786,432 → 16,384 (약 1/48), 공간 1/64.
- 6절 CrossAttention 코드 주석의 출력 크기 `torch.Size([2, 4096, 320])`.
- `{% post_url 2026-06-10-attention-is-all-you-need %}` 대상 파일 존재 여부.

## 2026-07-01-transformer-decoder-to-gpt.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 코드 주석의 출력값(`torch.Size([2, 15, 100])`, loss 약 4.6, 파라미터 수 124439808)과 본문 "초기화 코드를 빼면 loss가 40 이상에서 시작".
- Chinchilla(70B, 모델 크기와 토큰 수를 같은 비율로 확장, 파라미터당 약 20 토큰, 더 큰 모델들을 이김).
- `{% post_url 2026-06-10-attention-is-all-you-need %}`, `{% post_url 2026-07-05-trpo-to-ppo %}` 대상 파일 존재 여부.

## 2026-06-27-imbalanced-data.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 5절 BKD: "증류의 temperature 등도 클래스 특성에 맞게 조절해" — 논문에서 temperature를 클래스별로 조절하는지 확인 필요.
- 5절 LFME: "통합 모델과 expert 사이의 성능 차이가 일정 임계값 안으로 들어올 때까지 ... 반복 학습" — 논문의 self-paced expert selection 설명과 맞는지 확인 필요.
- 4절 LDAM margin 출력 `tensor([0.1623, 0.2887, 0.5000])`, 원 논문 구현의 `NormedLinear`와 $$s=30$$.
- 2절 WeightedRandomSampler 예시 출력과 본문 "약 320번, 평균 32번씩"(기댓값은 약 333).
- `{% post_url 2026-05-05-ml-overview %}`, `{% post_url 2026-06-06-object-detection %}`, `{% post_url 2026-06-25-pytorch-training-toolkit %}` 대상 파일 존재 여부.

## 2026-06-25-pytorch-training-toolkit.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- PyTorch 2.6부터 `torch.load`의 `weights_only` 기본값이 `True`.
- `torch.utils.tensorboard`: 1.1에서 실험 기능, 1.2(2019년 8월)에서 정식 지원. `SummaryWriter` 기본 경로 `./runs/날짜_호스트명`, `add_scalar(tag, scalar_value, global_step)`.
- `ToTensor()`가 float64 배열을 255로 나누지 않고 dtype을 그대로 둔다는 설명, `T.ConvertImageDtype` 존재 여부.
- `DataParallel` state_dict key와 `strict=False` 반환값(`_IncompatibleKeys(...)`) 주석.
- `{% post_url 2026-05-18-overfitting-regularization-augmentation %}`, `{% post_url 2026-06-17-binary-neural-networks %}`, `{% post_url 2026-05-05-ml-overview %}` 대상 파일 존재 여부.

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| 2026-07-05-trpo-to-ppo.md | 6절 표의 loss 기울기 -0.5, -0.6 | 수정함 | 로컬 재계산(PyTorch 2.13.0): 옛 확률 0.5, $$\hat A=+1$$, 단일 샘플, `new_logp`에 대한 기울기가 r=1.0에서 -1.0, r=1.2에서 -1.2, r=1.6에서 0. 해석적으로도 $$\partial r/\partial\log\pi = r$$이라 기울기는 $$-r\hat A$$ | 표의 -0.5 → -1.0, -0.6 → -1.2 (0 행과 설명 문구는 그대로) |
| 2026-07-05-trpo-to-ppo.md | $$C = 4\epsilon\gamma/(1-\gamma)^2$$, $$\epsilon=\max\lvert A_\pi\rvert$$ | 문제 없음 | https://arxiv.org/abs/1502.05477 (Theorem 1, 식 (8)–(9)) | - |
| 2026-07-05-trpo-to-ppo.md | 평균 KL은 heuristic 근사, single path/vine 설명 | 문제 없음 | https://arxiv.org/abs/1502.05477 (4–5절) | - |
| 2026-07-05-trpo-to-ppo.md | 10절 실험 결과, TRPO 한계(dropout·파라미터 공유) | 문제 없음 | https://arxiv.org/abs/1707.06347 (1절, 6.1절 Table 1: no clipping or penalty -0.39, 6.2–6.4절) | - |
| 2026-07-05-trpo-to-ppo.md | GAE 출력 블록과 본문 설명 | 문제 없음 | 로컬 재실행 결과가 출력 블록과 동일(0.995/2.3731/2.4701 등), 1+0.99×0.5−0.5=0.995, 1+0.99+0.99²−0.5=2.4701 | - |
| 2026-07-05-trpo-to-ppo.md | `post_url 2026-08-17-acer` | 문제 없음 | `_posts/2026-08-17-acer.md` 존재 | - |
| 2026-07-03-latent-diffusion-models.md | 0.18215(latent 표준편차로 나눠 단위 분산) | 문제 없음 | https://arxiv.org/html/2112.10752 (부록 G: 첫 배치에서 component-wise 분산 추정 후 단위 표준편차로 rescale, 부록 D.1) | - |
| 2026-07-03-latent-diffusion-models.md | downsampling factor 결과, GPU 수백 일, $$\tau_\theta$$ Transformer | 문제 없음 | https://arxiv.org/html/2112.10752 (1절, 4.1절, 4.3.1절) | - |
| 2026-07-03-latent-diffusion-models.md | 786,432 → 16,384 (1/48), 공간 1/64 | 문제 없음 | 본문 재계산: 786432/16384 = 48, (512/64)² = 64 | - |
| 2026-07-03-latent-diffusion-models.md | CrossAttention 출력 크기 | 문제 없음 | 로컬 재실행: `torch.Size([2, 4096, 320])` | - |
| 2026-07-03-latent-diffusion-models.md | `post_url 2026-06-10-attention-is-all-you-need` | 문제 없음 | `_posts/2026-06-10-attention-is-all-you-need.md` 존재 | - |
| 2026-07-01-transformer-decoder-to-gpt.md | 코드 출력값과 "초기화 없이 loss 40 이상" | 문제 없음 | 로컬 재실행(seed 0): `torch.Size([2, 15, 100]) 4.599`, 파라미터 124439808, 초기화 제거 시 loss 41.96 (seed 1, 2: 39.94, 42.68) | - |
| 2026-07-01-transformer-decoder-to-gpt.md | Chinchilla 설명 | 문제 없음 | https://arxiv.org/abs/2203.15556 (70B, "scaled equally", Gopher·GPT-3·Jurassic-1·MT-NLG보다 우수) | - |
| 2026-07-01-transformer-decoder-to-gpt.md | post_url 2개 | 문제 없음 | `_posts/2026-06-10-attention-is-all-you-need.md`, `_posts/2026-07-05-trpo-to-ppo.md` 존재 | - |
| 2026-06-27-imbalanced-data.md | BKD temperature를 클래스별로 조절한다는 설명 | 수정함 | https://arxiv.org/abs/2104.10510 (식 (7): $$\mathcal{L}=\mathcal{L}_{CE}+T^2\sum_i\omega_i\hat p_i\log(\hat p_i/p_i)$$, "we fix T = 2 in all our experiments") | "증류의 temperature 등도 클래스 특성에 맞게 조절해," → "증류의 temperature는 모든 클래스에 같은 값($$T = 2$$)을 쓰고, 클래스별 가중치로" |
| 2026-06-27-imbalanced-data.md | LFME "성능 차이가 임계값 안으로 들어올 때까지 반복 학습" | 수정함 | https://arxiv.org/abs/2001.01536 (4.2절 Self-paced Expert Selection: student 정확도가 expert 정확도의 α배를 넘으면 그 expert의 증류 가중치 $$w_l$$을 줄임. 4.3절 Curriculum Instance Selection: 샘플을 쉬운 것부터 어려운 것 순으로 선택) | "통합 모델과 expert 사이의 성능 차이가 일정 임계값 안으로 들어올 때까지, 쉬운 것부터 단계적으로(curriculum) 반복 학습합니다." → "통합 모델의 성능이 expert에 일정 임계값 이상 가까워지면 그 expert의 증류 비중을 점점 줄이고, 학습 샘플은 쉬운 것부터 단계적으로(curriculum) 늘려 갑니다." |
| 2026-06-27-imbalanced-data.md | LDAM margin 출력, NormedLinear, s=30 | 문제 없음 | 로컬 재실행: `tensor([0.1623, 0.2887, 0.5000])`; https://github.com/kaidic/LDAM-DRW (NormedLinear, LDAMLoss s=30) | - |
| 2026-06-27-imbalanced-data.md | sampler 예시 "약 320번, 평균 32번" | 문제 없음 | 로컬 재실행(seed 0–2): 클래스 2가 325, 348, 318회. 본문은 "예:" 출력과 일치하는 예시 수치(기댓값 약 333) | - |
| 2026-06-27-imbalanced-data.md | post_url 3개 | 문제 없음 | `_posts/2026-05-05-ml-overview.md`, `_posts/2026-06-06-object-detection.md`, `_posts/2026-06-25-pytorch-training-toolkit.md` 존재 | - |
| 2026-06-25-pytorch-training-toolkit.md | `weights_only` 기본값 True (2.6부터) | 문제 없음 | https://docs.pytorch.org/docs/2.6/_sources/notes/serialization.rst.txt, https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573 | - |
| 2026-06-25-pytorch-training-toolkit.md | TensorBoard 1.1 실험 기능, 1.2 정식 지원 | 문제 없음 | https://github.com/pytorch/pytorch/releases/tag/v1.2.0 | - |
| 2026-06-25-pytorch-training-toolkit.md | SummaryWriter 기본 경로, add_scalar 인자 | 문제 없음 | https://docs.pytorch.org/docs/2.14/tensorboard.html ("runs/CURRENT_DATETIME_HOSTNAME", `add_scalar(tag, scalar_value, global_step=None, ...)`) | - |
| 2026-06-25-pytorch-training-toolkit.md | ToTensor의 float64 처리, ConvertImageDtype | 문제 없음 | 로컬 재실행(torchvision): float64 배열 → `torch.float64`, `T.ConvertImageDtype` 존재 | - |
| 2026-06-25-pytorch-training-toolkit.md | DataParallel key, `_IncompatibleKeys` 주석 | 문제 없음 | 로컬 재실행: `['module.weight', 'module.bias']`, `_IncompatibleKeys(missing_keys=['weight', 'bias'], unexpected_keys=['module.weight', 'module.bias'])` | - |
| 2026-06-25-pytorch-training-toolkit.md | post_url 3개 | 문제 없음 | `_posts/2026-05-18-overfitting-regularization-augmentation.md`, `_posts/2026-06-17-binary-neural-networks.md`, `_posts/2026-05-05-ml-overview.md` 존재 | - |
