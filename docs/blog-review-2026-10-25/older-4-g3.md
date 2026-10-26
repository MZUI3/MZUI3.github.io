# 블로그 교정·검증 보고서 (older-4-g3, 2026-10-25)

## 2026-05-23-normalization.md

### 수정 목록
수정 없음.

### 내용 확인 필요
- `from torch.nn.utils.parametrizations import weight_norm   # PyTorch 2.1+`: 추가된 버전이 2.1이 맞는지.
- `original0` / `original1` shape 주석([64, 1], [64, 128])이 `nn.Linear(128, 64)` 기준으로 맞는지.
- "TinyTL도 이 성질을 이용해 배치를 1~8로 줄여": TinyTL 논문의 배치 크기와 맞는지.
- `BatchNorm1d` 배치 1 에러 메시지(`Expected more than 1 value per channel when training`)가 실제 PyTorch 메시지와 같은지.
- 교차 링크 `2026-05-18-overfitting-regularization-augmentation`, `2026-06-10-attention-is-all-you-need`, `2026-06-19-edge-ai-hardware`: 대상 파일이 모두 `_posts/`에 있음(확인 완료).

## 2026-05-21-sql-index-intro.md

### 수정 목록
수정 없음.

### 내용 확인 필요
- "위 실험에서도 user_id 종류가 1만 개라 SQLite는 풀 스캔을 골랐습니다": SQLite skip scan 조건은 `ANALYZE` 통계 유무와 선두 컬럼의 중복 수입니다. 이 실험은 50만 행 / 사용자 1만 명이라 값마다 평균 50개가 중복되고, `ANALYZE`도 실행하지 않았습니다. 설명이 틀렸을 수 있습니다.
- "Oracle, MySQL 8.0 이상, SQLite 등은 ... skip scan": MySQL skip scan은 8.0.13부터 지원.

## 2026-05-18-overfitting-regularization-augmentation.md

### 수정 목록
| 원문 | 수정 | 유형 |
|---|---|---|
| augmentation 자체가 일종의 정규화입니다. | augmentation 자체가 일종의 규제입니다. | 통일 (글 전체에서 regularization = "규제"로 씀) |

### 내용 확인 필요
- "PyTorch는 ... inverted dropout을 써서, 추론 때는 아무것도 하지 않아도 됩니다. 그래서 `model.eval()`을 꼭 호출해야 합니다.": "그래서"로 이어지는 인과가 맞지 않습니다(`eval()`은 dropout을 끄기 위해 필요함).
- PBA: "AutoAugment와 비슷한 정확도를 1000배 이상 빠르게": 논문 수치 확인 필요.
- AutoAugment: "16가지 연산", "강화학습" 확인 필요.
- 교차 검증 출력 평균: (0.965+0.982+0.956+0.956+1.0)/5 = 0.9718 → 0.972로 출력과 일치(확인 완료). CutMix 출력 0.527 → 본문의 "약 53% / 47%"와 일치.
- References의 "장윤석, 임현일, 'CNN에서의 DropOut과 DropConnect에 대한 성능 비교,' 한국정보처리학회 2019": 서지 정보 확인 필요.
- 교차 링크 `2026-05-23-normalization`, `2026-05-15-optimizer-and-scheduler`, `2026-06-25-pytorch-training-toolkit`([AI 기초 15]): 모두 있음(확인 완료).

## 2026-05-15-optimizer-and-scheduler.md

### 수정 목록
수정 없음.

### 내용 확인 필요
- 좁은 골짜기 실험의 출력값 5개, 그리고 "Nesterov는 lr=0.03에서 오히려 발산"이라는 서술.
- "PyTorch `OneCycleLR`의 기본값은 전체의 30%(`pct_start=0.3`) 동안 올리고 나머지를 cosine으로 내립니다".
- "PyTorch 1.1부터 이 순서가 바뀌었고, 순서가 틀리면 경고가 출력됩니다".
- PyTorch Momentum 식 차이 설명($$v_t = \gamma v_{t-1} + g_t$$, $$\theta_{t+1} = \theta_t - \eta v_t$$).

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| 2026-05-23-normalization.md | `parametrizations.weight_norm` PyTorch 2.1+ | 문제 없음 | `torch/nn/utils/parametrizations.py`에 `def weight_norm`이 v2.0.0에는 없고 v2.1.0에는 있음. https://raw.githubusercontent.com/pytorch/pytorch/v2.1.0/torch/nn/utils/parametrizations.py / 옛 함수 문서: "This function is deprecated. Use torch.nn.utils.parametrizations.weight_norm()" https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.weight_norm.html | - |
| 2026-05-23-normalization.md | original0/original1 shape | 문제 없음 | 문서 예시에서 Linear(20, 40)일 때 original0 = [40, 1], original1 = [40, 20](dim=0 기본값)이므로 Linear(128, 64)는 [64, 1], [64, 128]. https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.parametrizations.weight_norm.html | - |
| 2026-05-23-normalization.md | TinyTL 배치 1~8 | 문제 없음 | 논문: "fine-tuned for 50 epochs ... with batch size 8"이고 Figure 5에 "Results of TinyTL when trained with batch size 1"이 있음. https://arxiv.org/abs/2007.11622 | - |
| 2026-05-23-normalization.md | BatchNorm 배치 1 에러 메시지 | 문제 없음 | PyTorch 소스의 메시지: "Expected more than 1 value per channel when training, got input size {size}". https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py | - |
| 2026-05-21-sql-index-intro.md | 실험에서 skip scan을 안 쓴 이유 | 수정함 | SQLite 문서: skip scan은 "about 18 or more" 중복일 때만 이득이고, "a skip-scan is never used on a database that has not been analyzed". 실험은 값마다 평균 50개가 중복이고 ANALYZE를 실행하지 않았으므로, 이유는 "종류가 1만 개"가 아니라 ANALYZE를 하지 않았기 때문. https://www.sqlite.org/optoverview.html | "위 실험에서도 user_id 종류가 1만 개라 SQLite는 풀 스캔을 골랐습니다." → "위 실험에서는 `ANALYZE`를 실행하지 않아 SQLite가 선두 컬럼의 중복 정도를 알 수 없으므로 풀 스캔을 골랐습니다(SQLite는 `ANALYZE` 통계가 없으면 skip scan을 쓰지 않습니다)." |
| 2026-05-21-sql-index-intro.md | MySQL 8.0 이상 skip scan | 문제 없음 | "as of MySQL 8.0.13, the optimizer can perform multiple range scans ... called Skip Scan". "8.0 이상"은 넓게 맞는 표현. https://dev.mysql.com/doc/refman/8.0/en/range-optimization.html | - |
| 2026-05-18-overfitting-regularization-augmentation.md | inverted dropout과 `model.eval()`의 인과 | 수정함 | PyTorch Dropout 문서: 학습 때 출력을 1/(1-p)로 키우므로 평가 때는 모듈이 항등 함수가 됨. 이때 `eval()`은 dropout을 끄려고 호출하는 것이라 "그래서"는 인과가 맞지 않음. https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html | "그래서 `model.eval()`을 꼭 호출해야 합니다." → "단, 추론 때 dropout이 꺼지도록 `model.eval()`을 꼭 호출해야 합니다." |
| 2026-05-18-overfitting-regularization-augmentation.md | PBA "1000배 이상 빠르게" | 수정함 | 논문: "three orders of magnitude less overall compute", Figure 1: "requiring 1,000x less GPU hours"(AutoAugment 5000 GPU시간 vs PBA 5시간). "이상"은 과장. https://arxiv.org/abs/1905.05393 | "**1000배 이상 빠르게**" → "**약 1000배 빠르게**" |
| 2026-05-18-overfitting-regularization-augmentation.md | AutoAugment 16가지 연산, 강화학습 | 문제 없음 | "In total, we have 16 operations in our search space."; "we use Reinforcement Learning as the search algorithm". https://arxiv.org/abs/1805.09501 | - |
| 2026-05-18-overfitting-regularization-augmentation.md | 장윤석·임현일(한국정보처리학회 2019) 참고문헌 | 확인 불가 | 웹 검색으로 해당 논문의 서지 정보를 찾지 못함(KIPS/DBpia 원문은 접근 불가). | - |
| 2026-05-15-optimizer-and-scheduler.md | 골짜기 실험 출력 5개, Nesterov lr=0.03 발산 | 문제 없음 | PyTorch SGD(momentum/nesterov)와 Adam 갱신식을 순수 Python으로 재구현해 다시 계산한 결과 2.29e-01, 3.92e+05, 1.19e-03, 5.01e-05, 8.51e-04로 출력과 일치. Nesterov lr=0.03은 1.02e+24로 발산. 갱신식 출처: https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html | - |
| 2026-05-15-optimizer-and-scheduler.md | OneCycleLR 기본값 | 문제 없음 | 문서 기본값: pct_start=0.3, anneal_strategy='cos', three_phase=False. https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.OneCycleLR.html | - |
| 2026-05-15-optimizer-and-scheduler.md | PyTorch 1.1 scheduler 호출 순서 | 문제 없음 | "Prior to PyTorch 1.1.0, the learning rate scheduler was expected to be called before the optimizer's update; 1.1.0 changed this behavior in a BC-breaking way." https://docs.pytorch.org/docs/2.14/optim.html | - |
| 2026-05-15-optimizer-and-scheduler.md | PyTorch Momentum 식 | 문제 없음 | 문서 Note: v_{t+1} = μ v_t + g_{t+1}, p_{t+1} = p_t − lr·v_{t+1}이고, Sutskever et al.은 v_{t+1} = μ v_t + lr·g_{t+1}. https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html | - |
