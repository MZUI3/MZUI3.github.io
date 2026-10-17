# 블로그 글 첨삭 리포트 (2026-10-17)

- 대상: `_posts/` 68편 전체
- 수정본: 각 글 파일에 직접 반영했습니다. 크게 바뀐 곳에는 `<!-- 수정: 이유 -->` 주석을 달았습니다. 이 주석은 `_config.yml`의 `compress_html: comments: all` 설정 때문에 배포된 HTML에서는 빠집니다.
- 이 파일은 `docs/` 폴더에 있어서 `_config.yml`의 `exclude` 설정에 따라 사이트에 빌드되지 않습니다.
- 점수는 **원문 기준**입니다.
- 글에 남긴 `[확인 필요: …]`와 `(여기에 … 적으세요)` 칸은 배포 전에 채우거나 지워야 합니다. 목록은 맨 아래 "남은 할 일"에 모았습니다.
- 공통으로 적용한 것: 모든 글에 `description` front matter 추가, 처음 나오는 용어 정의, 깊은 내용은 `<details markdown="1">`로 분리, 비유를 쓴 곳에는 비유가 어디까지 맞는지 명시, 1차 출처(논문·공식 문서) 링크 추가.

---

## 개발 / Blog

### 2026-05-01-hello-blog.md — 블로그를 시작하며

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 3 | "네 개의 큰 분야"라고 했지만 실제로는 `개발`, `Blog` 카테고리도 있음 |
| 입문자 이해도 | 5 | 카테고리 트리가 명확함 |
| 전문성·깊이 | 2 | 소개 글이라 깊이는 필요 없지만 필자가 누구인지 드러나지 않음 |
| 구성·가독성 | 4 | 짧고 깔끔함 |
| 코드 품질 | – | 코드 없음 |

**가장 시급한 3가지**
1. 본문 카테고리 설명과 실제 카테고리 불일치
2. 필자 소개·블로그 원칙 부재 (채용 담당자가 가장 먼저 여는 글일 수 있음)
3. front matter `date`가 `2026-01-01`인데 파일명은 `2026-10-14` — 정렬은 front matter 기준이라 동작엔 문제없지만 일관성을 위해 하나로 맞추길 권장 (수정하지 않음)

**주요 변경**
- "다음 네 개의 큰 분야로 나눠 정리합니다" → AI 4분야 트리는 유지하고, `개발`·`Blog` 카테고리를 설명하는 문단 추가 → 실제 구조와 일치
- 자기소개 빈칸 추가 → 지어내지 않고 본인이 채우도록 안내

**추가 제안**
- 제목 후보: 「MZUI.log를 시작하며: 무엇을, 어떻게 기록하나」 / 「블로그 소개와 카테고리 안내」 / 「AI·강화학습·개발 공부 기록을 시작합니다」
- tags: `[jekyll, chirpy, github-pages]` 유지
- `_config.yml`의 `social.email`이 `mzui2033@email.com`으로 되어 있습니다. `gmail.com`의 오타인지 확인해 보세요. (글이 아니라서 수정하지 않음)
- 다음 주제: 「Chirpy 테마로 블로그 만들기 — 설정한 것들」, 「블로그 카테고리 구조를 이렇게 바꾼 이유」

### 2026-05-11-rest-api-basics.md — REST API를 설계할 때 먼저 정할 것들

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 틀린 곳은 없지만 401/403 설명이 이름 그대로라 오해 소지 |
| 입문자 이해도 | 3 | REST, API 계약, OpenAPI 정의 없음 |
| 전문성·깊이 | 2 | 왜 그 메서드를 쓰는지(안전성·멱등성)가 빠짐 |
| 구성·가독성 | 4 | 표가 좋음 |
| 코드 품질 | 3 | 응답 예시만 있고 오류 응답 예시 없음 |

**가장 시급한 3가지**
1. 메서드 선택 기준(안전·멱등) 부재 → 면접 단골 질문인데 답이 글에 없음
2. 401 = "인증 안 됨"이라는 함정 설명 부족
3. 근거 링크(RFC) 없음

**주요 변경**
- (정확성) "인증되지 않은 요청에는 401, 권한이 없는 요청에는 403" → 상태 코드 표 + "401은 이름과 달리 인증(authentication) 실패, `WWW-Authenticate` 헤더 동반" → 이름 때문에 생기는 대표적 오용을 막음
- (정확성) 오류 응답 형식 → RFC 9457 Problem Details(7807 대체) 예시와 `application/problem+json` 추가
- 메서드별 안전·멱등 표, PATCH가 멱등이 아닐 수 있는 이유, `Idempotency-Key`(details)
- REST 정의와 "실무 REST ≠ Fielding의 REST" 구분 박스
- 201 + `Location`, 204 추가 / offset vs cursor 페이지네이션(details) / 참고 자료

**추가 제안**
- 제목 후보: 「REST API 설계 기초: URL, 메서드, 상태 코드 정하기」 / 「REST API 메서드와 상태 코드, 기준부터 정리하기」 / 「API 계약을 먼저 정하자: REST 설계 체크리스트」
- tags: `[REST, API, HTTP, OpenAPI]`
- 다음 주제: 「JWT vs 세션 인증」, 「OpenAPI로 API 문서 자동화하기」
- 꼬리질문
  1. PUT과 PATCH 차이는? → 전체 교체 vs 부분 수정, PUT은 멱등·PATCH는 보장 안 됨 (글에 반영)
  2. POST 요청이 타임아웃 나면 재시도해도 되나? → 멱등이 아니므로 중복 생성 위험, Idempotency-Key로 해결 (반영)
  3. 401과 403을 어떻게 구분하나? → 인증 vs 인가 (반영)

### 2026-05-17-sql-index-intro.md — 데이터베이스 인덱스는 언제 도움이 될까

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 틀린 곳 없음, 다만 "왜 빨라지는가"가 없음 |
| 입문자 이해도 | 3 | 실행 계획, 선택도, 복합 인덱스 정의 없음 |
| 전문성·깊이 | 2 | B-Tree, 선두 컬럼 규칙, 정렬 생략 같은 핵심 근거 부재 |
| 구성·가독성 | 4 | 흐름은 좋음 |
| 코드 품질 | 2 | 실행 계획을 보라고만 하고 실제로 보여 주지 않음 |

**가장 시급한 3가지**
1. 인덱스가 빠른 이유(B-Tree, O(log N)) 설명 부재
2. "컬럼 순서가 중요하다"의 이유(선두 컬럼 규칙)를 설명하지 않음
3. 실행 결과가 없는 예시 코드

**주요 변경**
- 책 색인 비유 → 비유의 한계(쓰기 비용) 명시
- B-Tree 내부 동작 + Mermaid 그림 추가
- **Python 표준 `sqlite3`로 바로 실행되는 실험 코드와 실제 실행 결과 추가** (이 PC에서 Python 3.14 / SQLite 3.50.4로 실행: 풀 스캔 평균 13.97ms → 인덱스 0.03ms, 실행 계획 변화, 선두 컬럼 없는 조건은 다시 SCAN). 숫자는 실제 측정값이지만 환경 의존이라고 명시
- 인덱스가 `ORDER BY`의 정렬을 생략시키는 이유 추가
- 선두 컬럼 규칙 표, skip scan·커버링 인덱스(details), 선택도 정의, DB별 EXPLAIN 표, `EXPLAIN ANALYZE`는 실제 실행된다는 주의

**추가 제안**
- 제목 후보: 「DB 인덱스는 왜 빠를까: B-Tree와 복합 인덱스 순서」 / 「SQL 인덱스, 언제 걸고 언제 걸지 말까」 / 「실행 계획으로 확인하는 인덱스의 효과 (SQLite 실습)」
- tags: `[SQL, 데이터베이스, 성능, 인덱스, B-Tree]`
- 다음 주제: 「트랜잭션 격리 수준」, 「N+1 쿼리 문제와 해결」
- 꼬리질문
  1. 해시 인덱스가 아니라 B-Tree를 쓰는 이유? → 범위 검색·정렬 지원
  2. 인덱스를 걸었는데 안 탄다면? → 선두 컬럼 누락, 컬럼에 함수 적용, 낮은 선택도, 통계 부정확
  3. 커버링 인덱스란? → details에 반영

### 2026-05-06-python-virtual-environment.md — Python 가상환경으로 프로젝트 의존성 분리하기

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 틀린 곳 없음 |
| 입문자 이해도 | 3 | 가상환경 정의와 "없으면 생기는 문제"가 없음 |
| 전문성·깊이 | 2 | 활성화가 실제로 무엇을 하는지 없음 |
| 구성·가독성 | 4 | 간결함 |
| 코드 품질 | 4 | 명령은 정확, 결과 출력 없음 |

**가장 시급한 3가지**
1. 활성화의 실체(PATH 변경, pyvenv.cfg) 설명 부재
2. Windows 입문자가 가장 많이 막히는 실행 정책 오류 미언급
3. `pip freeze`의 한계 미언급

**주요 변경**
- 정의 + numpy 버전 충돌 예시 추가
- 활성화 = PATH 앞에 Scripts/bin 추가, `sys.prefix != sys.base_prefix` 확인 코드(실제 실행 결과), "활성화 없이 경로로 실행 가능"
- PowerShell 실행 정책 오류 해결(details, `-Scope Process` 우선 권장)
- `python -m pip`를 쓰는 이유, **실제 실행한** `pip freeze` 출력과 전이 의존성 설명
- pip-tools / Poetry / uv 비교(details)
- (최신 정보) Python 3.13+는 venv 안에 `.gitignore`를 자동 생성 — 직접 실행해 확인함

**추가 제안**
- 제목 후보: 「Python venv 가상환경: 만드는 법부터 동작 원리까지」 / 「Python 가상환경은 왜 쓰고, 활성화는 무엇을 할까」 / 「venv와 requirements.txt로 프로젝트 의존성 관리하기」
- tags: `[Python, venv, pip, 가상환경]`
- 다음 주제: 「uv로 Python 프로젝트 관리하기」, 「conda와 venv의 차이」
- 꼬리질문
  1. 가상환경 폴더를 Git에 올리면 안 되는 이유? → 절대 경로·OS 의존 (반영)
  2. requirements.txt만으로 재현성이 보장되나? → 전이 의존성·Python 버전·OS 차이 → lock 파일 (반영)
  3. conda와 venv 차이? → conda는 Python 자체와 비Python 바이너리까지 관리

### 2026-06-03-github-actions-pages.md — GitHub Actions로 정적 블로그 자동 배포하기

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 실제 워크플로와 일치 |
| 입문자 이해도 | 2 | 정적 사이트, CI/CD, 워크플로 정의 없음 |
| 전문성·깊이 | 2 | 권한, OIDC, job 분리 이유가 없음 |
| 구성·가독성 | 4 | 흐름 목록은 좋음 |
| 코드 품질 | 1 | 워크플로 파일을 보여 주지 않음 |

**가장 시급한 3가지**
1. 실제 YAML이 없어 따라 할 수 없음
2. `permissions`/`id-token`/`fetch-depth: 0`의 이유가 없음
3. 트러블슈팅이 일반론에 그침

**주요 변경**
- 실제 `pages-deploy.yml`을 주석과 함께 수록. `${{ }}`가 Liquid로 해석되지 않도록 `{% raw %}`로 감쌈 (안 감싸면 빌드 깨짐)
- `fetch-depth: 0`이 이 블로그의 `posts-lastmod-hook.rb` 때문에 필요하다는 점, OIDC 설명, build/deploy 분리 이유, htmlproofer 역할
- Pages Source 설정 경고, 실패 원인 표 (미래 날짜 글 미빌드, Liquid 충돌 등)
- Mermaid 흐름도, 경험 칸

**추가 제안**
- 제목 후보: 「Jekyll 블로그 GitHub Actions 자동 배포 워크플로 해부」 / 「GitHub Pages 배포 워크플로, 한 줄씩 읽어 보기」 / 「GitHub Actions + Jekyll: 푸시하면 배포되는 블로그 만들기」
- tags: `[GitHub Actions, GitHub Pages, CI/CD, Jekyll]`
- 다음 주제: 「GitHub Actions 캐시와 빌드 시간 줄이기」, 「Dependabot으로 액션 버전 관리하기」
- 꼬리질문
  1. `GITHUB_TOKEN`과 PAT 차이? → 실행마다 발급·자동 만료, 권한 범위 지정 가능
  2. OIDC 방식이 비밀 키 저장보다 나은 이유? → 장기 비밀 없음, 탈취돼도 짧은 수명
  3. `concurrency`가 없으면? → 연속 푸시 시 오래된 빌드가 나중에 끝나 덮어쓸 수 있음

---

## AI 글

### 2026-08-18-acer.md — ACER

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 3 | Retrace truncation 상수(1)와 gradient truncation 상수(c=10)를 같은 c로 써서 혼동, V를 네트워크가 출력한다고 서술 |
| 입문자 이해도 | 3 | on/off-policy, replay buffer, actor-critic 정의 없음 |
| 전문성·깊이 | 4 | 수식 전개와 각 항의 역할 설명이 좋음 |
| 구성·가독성 | 4 | 문제→해결 표가 좋음 |
| 코드 품질 | – | 코드 없음 |

**가장 시급한 3가지**
1. Retrace의 $$\bar\rho=\min\{1,\rho\}$$와 5절의 $$\min\{c,\rho\}$$(논문 c=10) 기호 분리
2. 이산 행동에서 $$V=\sum_a\pi Q$$로 계산된다는 점 정정
3. "분산이 최대 c배로 제한"은 부정확 → "weight가 c 이하로 묶여 분산이 유계"

**주요 변경**
- (정확성) `min{c, ρ}`(c는 보통 1) → `min{1, ρ}` + "5절의 c와 별개" → 논문 식 (5)와 일치
- (정확성) "Q와 V는 하나의 네트워크가 출력" → "π와 Q를 출력, V는 Σπ·Q로 계산"
- (정확성) Retrace 수렴 보장에 "(테이블 형태의 정책 평가 설정에서)" 조건 추가
- 용어 정리 박스, **importance weight 곱의 분산 폭발 실험 코드 + 실제 실행 결과(NumPy 2.5)** 추가
- "ACER가 PPO만큼 안 쓰인 이유" details (트레이드오프), arXiv 링크

**추가 제안**
- 제목 후보: 「ACER 논문 리뷰: Off-policy Actor-Critic을 안정적으로 만드는 세 가지 기법」 / 「ACER 정리: Truncated IS, Retrace, Efficient TRPO」 / 「Experience Replay를 쓰는 Actor-Critic, ACER」
- tags 유지(충분히 구체적)
- 다음 주제: 「IMPALA와 V-trace」, 「SAC: 최대 엔트로피 off-policy actor-critic」
- 꼬리질문
  1. Importance weight를 자르면 왜 편향이 생기나? → 기댓값 식이 달라짐, ACER는 보정 항으로 복원
  2. Retrace가 Tree-backup이나 Q(λ)보다 나은 점? → off-policy에서도 안전(수렴) + on-policy에 가까우면 효율적(trace를 덜 자름)
  3. 통계량 공간 trust region이 왜 싼가? → 파라미터 공간 Fisher 대신 출력 분포 차원(행동 수)에서 닫힌 해

### 2026-05-13-activation-functions.md — [AI 기초 3] 활성화 함수 총정리

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 3 | `torch.sign`이 0에서 0을 반환하는 버그, ELU 0점 미분 조건 누락, "ReLU가 기울기 소실 해결" 과장 |
| 입문자 이해도 | 5 | 함수별 장단점·쓰임새 구성이 매우 좋음 |
| 전문성·깊이 | 4 | STE까지 연결한 점이 좋음 |
| 구성·가독성 | 5 | 마지막 "어디에 무엇을" 표가 훌륭함 |
| 코드 품질 | 3 | STE 코드에 import 누락·실제 버그, 실행 결과 없음 |

**가장 시급한 3가지**
1. `SignSTE`가 0 입력에 0을 출력 → ±1 이진화가 깨짐
2. "ReLU = Gradient Vanishing 해결" → 초기화에 따라 ReLU도 소실된다는 점이 빠짐
3. ELU가 0에서 매끄럽다는 설명은 α=1일 때만 참

**주요 변경**
- (정확성·코드) `torch.sign(x)` → `torch.where(x >= 0, 1, -1)` + 실행 결과와 경고 박스 (실행 확인)
- (정확성) "최소 1/4씩 줄어든다" → "많아야 0.25배, 10층이면 약 10⁻⁶"
- (정확성) "Gradient Vanishing 해결" → "크게 완화" + **20층 MLP 첫 층 기울기 실험**: Sigmoid 7e-19, ReLU 기본 초기화 3e-10, ReLU+He 초기화 6e-2 (실제 실행값) → "ReLU + 초기화 + 정규화의 조합"이라는 결론
- (정확성) ELU 0점 미분 조건, softmax 미분은 logit 기준임을 명시, 역전파에서 W도 곱해진다는 보충
- GELU의 Φ 정의, SwiGLU 언급, 참고 링크

**추가 제안**
- 제목 후보: 「활성화 함수 총정리: Sigmoid·ReLU·GELU, 무엇을 언제 쓰나」 / 「활성화 함수와 기울기 소실: 실험으로 보는 ReLU의 진짜 역할」 / 「Sigmoid부터 STE까지, 활성화 함수 완벽 정리」
- tags에 `gelu`, `swish` 추가 권장
- 다음 주제: 「가중치 초기화: Xavier와 He는 왜 그 분산인가」, 「GLU 계열(SwiGLU)과 LLM의 FFN」
- 꼬리질문
  1. ReLU를 쓰면 기울기 소실이 완전히 사라지나? → 아님, 초기화·dead neuron (실험 반영)
  2. He 초기화의 분산이 2/n인 이유? → ReLU가 절반을 0으로 만들어 분산이 반으로 줄기 때문
  3. Transformer가 GELU를 쓰는 이유? → 매끄러운 곡선, 실험적으로 더 좋았음(결정적 이론 근거는 약함)

### 2026-06-14-attention-is-all-you-need.md — Attention Is All You Need

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 최초 Attention이 내적 방식인 것처럼 서술, head 역할 분담을 단정 |
| 입문자 이해도 | 5 | 배경→문제→해결 흐름, Q/K/V 표가 좋음 |
| 전문성·깊이 | 3 | 연산 복잡도(n²), Post-LN/Pre-LN 같은 핵심 논점 부재 |
| 구성·가독성 | 5 | 매우 좋음 |
| 코드 품질 | 3 | 함수만 있고 사용 예·결과 없음 |

**가장 시급한 3가지**
1. Self-Attention의 O(n²) 비용과 트레이드오프 부재 (면접 단골)
2. 코드 실행 예와 causal mask 예시 부재
3. Bahdanau attention = additive라는 사실 누락

**주요 변경**
- (정확성) "내적해 Attention Score" → Bahdanau(additive) / Luong(dot) 구분
- (정확성) head 역할 분담 → "관찰되는 경향, 다수 head는 제거해도 성능 유지(Michel 2019)"
- (정확성) "깊어질수록 Vanishing Gradient" → "시퀀스가 길어질수록"
- 검색 비유의 한계 박스, permutation equivariance 용어, 학습형 PE·RoPE 언급, Post-LN vs Pre-LN 박스
- **논문 Table 1 연산 복잡도 비교 섹션** 추가
- causal mask 예제 + 실행 결과 + `F.scaled_dot_product_attention`과 결과 일치 확인

**추가 제안**
- 제목 후보: 「Transformer 논문 정리: Self-Attention은 왜 RNN을 대체했나」 / 「Attention Is All You Need 리뷰: Q·K·V부터 Positional Encoding까지」 / 「Transformer 구조 완전 정리 (코드 포함)」
- tags에 `multi-head-attention` 추가 권장
- 다음 주제: 「KV Cache: GPT 추론이 빨라지는 원리」, 「RoPE: 회전 위치 임베딩」
- 꼬리질문
  1. √d_k로 나누지 않으면? → softmax 포화, 기울기 소실 (글에 있음)
  2. Self-Attention의 시간·메모리 복잡도와 긴 문맥 처리 방법? → O(n²), FlashAttention 등 (반영)
  3. Decoder의 마스킹은 학습과 추론에서 각각 왜 필요? → 학습 시 병렬 teacher forcing에서 미래 누출 방지

### 2026-06-29-audio-classification.md — [AI 기초 14] 음성·소리 분류 기초

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 3 | frame 예시(2048샘플=128ms)가 본문 20~40ms와 모순, pre-emphasis가 잡음을 억제한다는 설명 의문 |
| 입문자 이해도 | 4 | MFCC 단계별 설명이 친절함 |
| 전문성·깊이 | 3 | MFCC가 GMM에 맞는 진짜 이유, 라이브러리 간 Mel 정의 차이 같은 실무 포인트 부재 |
| 구성·가독성 | 4 | 분량이 많지만 흐름은 좋음 |
| 코드 품질 | 3 | librosa 버전·인터넷 필요 미표기, CRNN 사용 예 없음 |

**가장 시급한 3가지**
1. frame 길이 예시 수치 모순
2. 사례 1 논문의 서지 정보와 "window 150"의 단위 불명
3. Focal loss α가 어느 클래스에 걸리는지 불명 → 결과 해석 불가

**주요 변경**
- (정확성) frame 예시를 512샘플(32ms)/hop 256으로 통일, 0~511처럼 끝 인덱스 수정, hop length 정의
- (정확성) pre-emphasis 설명 수정(고주파 잡음도 커질 수 있음, 최근엔 생략도 많음)
- window 길이별 시간·주파수 해상도 표 (계산값)
- **librosa(Slaney) vs torchaudio(HTK) Mel 정의 차이 경고** 추가
- DCT에서 앞쪽 계수만 남기는 이유, 대각 공분산 GMM과 MFCC의 궁합
- CRNN 실행 예(shape, 파라미터 수 108,098 — 실행 확인)
- `[확인 필요]` 3곳: window 150의 단위, α 적용 클래스, 사례 1 논문 서지

**추가 제안**
- 제목 후보: 「MFCC와 Mel-Spectrogram 완벽 정리: 소리를 신경망에 넣는 법」 / 「음성 특징 추출 기초: STFT부터 MFCC, wav2vec까지」 / 「심음 분류로 배우는 오디오 딥러닝 기초」
- tags 유지
- 다음 주제: 「SpecAugment: 음성 데이터 증강」, 「wav2vec 2.0과 자기지도 음성 표현」
- 꼬리질문
  1. CNN에 MFCC 대신 log-Mel을 넣는 이유? → 이웃 주파수 상관 구조 보존 (글에 있음)
  2. 불균형 데이터에서 Accuracy 대신 무엇을 보나? → Sn/Sp, MAcc, F1, AUROC
  3. 녹음 단위 판정 threshold는 어떻게 정하나? → 검증 세트에서 Sn/Sp 트레이드오프(ROC)로 결정, 테스트셋으로 정하면 누출

### 2026-06-25-binary-neural-networks.md — [AI 기초 12] 이진 신경망(BNN) 총정리

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 3 | ReActNet-A(69.4%)를 "실수 ResNet-18과 3%p 차이"로 서술 — 실제 비교 대상은 같은 구조의 실수 모델(72.4%) |
| 입문자 이해도 | 4 | XNOR·popcount 예제가 아주 좋음 |
| 전문성·깊이 | 4 | 연구 흐름이 체계적 |
| 구성·가독성 | 3 | 27KB로 매우 길고 논문 요약 나열 느낌 |
| 코드 품질 | 3 | 학습용 fake-binarization 코드가 실제로 빨라지지 않는다는 점 미언급 |

**가장 시급한 3가지**
1. ReActNet 비교 대상 오류
2. "32배·58배"가 이론값이라는 점, 학습 코드는 float로 돌아간다는 점 누락 → 면접에서 바로 찔리는 부분
3. 분량 — 두 편(기초~Bi-Real / Real-to-Binary~Edge Training)으로 나누는 것을 권장

**주요 변경**
- (정확성) ReActNet-A 비교 대상 수정, 마지막 정리의 "3%p"에도 "같은 구조의 실수 모델" 명시 + "구조 변화도 섞여 있다" 주의
- 이론값 vs 실제(첫/마지막 층 실수, bit-packing 추론 엔진 필요) 경고 박스
- `torch.sign(0)=0` 주의 박스, sign 미분 설명 문장 정리
- 자기 관점 한 문단 추가 안내, arXiv·Larq 링크

**추가 제안**
- 제목 후보: 「이진 신경망(BNN) 정리: XNOR-Net에서 ReActNet까지 정확도를 되찾은 방법」 / 「1비트로 학습하는 신경망, BNN의 원리와 연구 흐름」 / 「BNN 총정리: STE, Bi-Real Net, ReActNet」
- tags 유지
- 다음 주제: 「Larq로 BNN 학습하고 배포하기 (실측)」, 「INT4/INT2 저비트 양자화와 BNN의 차이」
- 꼬리질문
  1. latent weight를 유지하는 이유? → 작은 기울기가 누적돼야 부호가 바뀜 (글에 있음), Bop 반론 (글에 있음)
  2. 첫/마지막 층을 이진화하지 않는 이유? → 입력 채널 적어 이득 작고 정확도 손실 큼
  3. BNN 학습 코드가 GPU에서 빨라지나? → 아니오, float 시뮬레이션 (반영)

### 2026-08-14-c51-distributional-rl.md — C51

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 대체로 정확, 출근 경로 예시가 C51이 위험 회피를 하는 것처럼 읽힘 |
| 입문자 이해도 | 4 | 예시와 단계별 projection 설명이 좋음 |
| 전문성·깊이 | 4 | contraction 언급이 좋음, 이득의 원천에 대한 후속 연구 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | l==u 보정 트릭이 설명 없이 들어 있음 |

**가장 시급한 3가지**
1. 예시의 한계(C51도 평균으로 행동 선택) 명시
2. projection을 숫자로 보여 주는 예시 부재
3. 코드의 `l == u` 보정 이유 설명 부재

**주요 변경**
- 비유 한계 박스, "보상" → "return" 용어 정정
- **projection 수치 예시: r=0.3 → atom 0에 0.25, 0.4에 0.75, 합·기댓값 보존 (실행 확인)**
- `l == u` 보정 이유, `log_softmax` 권장
- Lyle et al.(2019) "이득은 딥러닝 근사에서 온다" details, arXiv 링크

**추가 제안**
- 제목 후보: 「C51 논문 리뷰: Q 값 대신 분포를 학습하는 Distributional RL」 / 「Distributional RL 입문: C51의 분포 Bellman 방정식과 Projection」 / 「C51(Categorical DQN) 정리와 PyTorch 구현」
- 다음 주제: 「QR-DQN과 IQN」, 「Rainbow DQN: 여섯 가지 개선의 결합」
- 꼬리질문
  1. Cross Entropy를 쓰는 이유? Wasserstein이 아니라? → 샘플 기반 Wasserstein 최소화는 편향된 gradient, 그래서 projection + KL (QR-DQN이 이를 해결)
  2. V_min, V_max를 잘못 잡으면? → 범위 밖 return이 잘려 학습 왜곡 → 보상 클리핑과 함께 씀
  3. 분포 Bellman 연산자가 control에서 수축하지 않는다는 의미? → 수렴 보장 없음, 실제로는 동작

### 2026-06-02-cnn-architectures.md — [AI 기초 8] CNN 아키텍처 계보

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 수치 대부분 정확, ShuffleNet unit의 마지막 ReLU 누락, RepVGG identity branch 조건 누락 |
| 입문자 이해도 | 5 | 각 모델의 "왜"가 잘 드러남 |
| 전문성·깊이 | 4 | DenseNet-121 층 수 계산, ShuffleNetV2 원칙 등 좋음 |
| 구성·가독성 | 4 | 길지만 정리 표가 좋음 |
| 코드 품질 | 3 | 블록마다 import가 없어 따로 복사하면 실행 안 됨, RepVGG 검증이 BN 기본값 상태라 너무 쉬움 |

**가장 시급한 3가지**
1. RepVGG 융합 검증을 BN 통계가 바뀐 상태에서 하도록 수정
2. 코드 블록별 import
3. DenseNet 메모리 문제의 실무 해결책 부재

**주요 변경**
- (정확성) ShuffleNet unit "더하기 → ReLU", RepVGG identity branch는 stride 1·채널 동일일 때만, `eval()` 필수 이유
- RepVGG 검증: BN 통계 변경 후 비교, 최대 오차 약 6e-7 (실행 확인)
- 모든 코드 블록 import 보강, SEBlock 사용 예 — 글 전체 코드를 실제로 실행해 확인
- memory-efficient DenseNet(`densenet121(memory_efficient=True)`) 언급, arXiv 링크

**추가 제안**
- 제목 후보: 「ResNet 이후의 CNN: DenseNet, SENet, ShuffleNet, RepVGG 한 번에 정리」 / 「CNN 아키텍처 발전사 — 연결, 채널 attention, 경량화, 재매개변수화」 / 「RepVGG 재매개변수화까지, CNN 구조 계보 정리」
- 다음 주제: 「ConvNeXt: Transformer 시대의 CNN」, 「EfficientNet과 compound scaling」
- 꼬리질문
  1. DenseNet이 파라미터는 적은데 메모리를 많이 쓰는 이유? → concat된 feature map 보관 (반영)
  2. SE block에서 softmax가 아니라 sigmoid를 쓰는 이유? → 여러 채널 동시 강조 (글에 있음)
  3. FLOPs가 적은데 느릴 수 있는 이유? → 메모리 접근, 병렬성 (ShuffleNetV2, 글에 있음)

### 2026-05-22-cnn-basics.md — [AI 기초 7] CNN 기초

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 예제 계산 모두 정확(검증함), "Transformer가 큰 데이터가 필요한 이유"는 ViT로 범위를 좁혀야 함 |
| 입문자 이해도 | 5 | 손 계산 예시, 출력 크기 표가 훌륭함 |
| 전문성·깊이 | 4 | im2col까지 다룸 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | 대부분 결과 포함, SimpleCNN만 사용 예 없음 |

**가장 시급한 3가지**
1. inductive bias 용어 정의 부재
2. equivariance와 invariance 구분
3. checkerboard 해결책이 다소 낙관적

**주요 변경**
- (정확성) "Transformer가 데이터가 많이 필요" → "ViT가", inductive bias 정의
- equivariance vs invariance 설명
- checkerboard: 커널 배수로 맞춰도 생길 수 있음 → resize-convolution 권장 (Odena 2016)
- SimpleCNN 사용 예와 출력(파라미터 94,986 — 실행 확인)
- cuDNN의 implicit GEMM/Winograd/FFT와 `cudnn.benchmark` 설명

**추가 제안**
- 제목 후보: 「CNN 기초 완벽 정리: Convolution, Padding, Stride, Pooling」 / 「합성곱 신경망은 왜 이미지에 강할까: CNN 핵심 개념 정리」 / 「CNN 출력 크기 계산부터 Depthwise Separable Conv까지」
- 다음 주제: 「Receptive field 계산과 유효 수용영역(ERF)」, 「BatchNorm은 CNN에서 무엇을 정규화하나」
- 꼬리질문
  1. 3×3 conv 두 번 vs 5×5 한 번? → 같은 RF, 파라미터 18C² vs 25C², 비선형성 추가 (글에 있음)
  2. 1×1 conv의 역할? → 채널 수 조절, 채널 간 결합 (글에 있음)
  3. CNN은 정말 translation invariant인가? → 아니오, stride/pooling이 equivariance를 깨기도 함 (Zhang 2019, "Making CNNs Shift-Invariant Again")

### 2026-08-06-double-dqn.md — Double DQN

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | Q-learning을 "value iteration 방식"이라 표현, online/target이 독립이라는 가정의 한계 미언급 |
| 입문자 이해도 | 5 | 세 질문 구조가 아주 좋음 |
| 전문성·깊이 | 4 | Theorem 1과 균일 오차 기댓값까지 다룸 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | `dqn_target`에 `no_grad` 누락, 과대평가를 직접 보여 주는 코드 없음 |

**가장 시급한 3가지**
1. "평균 0 노이즈도 max를 취하면 양의 편향"을 직접 확인할 수 있는 예시
2. target network가 online의 지연 복사본이라 완전히 독립이 아님
3. `dqn_target`의 `torch.no_grad()` 누락

**주요 변경**
- (정확성) "value iteration 방식" → "TD 학습 방식"
- **과대평가 시뮬레이션 코드 + 실행 결과**: 행동 2/4/10/30개에서 max 편향 +0.57/+1.03/+1.54/+2.04, double 편향 ≈ 0 (실행 확인)
- online/target 비독립성 박스, `no_grad` 추가, arXiv 링크

**추가 제안**
- 제목 후보: 「Double DQN 논문 리뷰: Q-Learning은 왜 Q 값을 부풀릴까」 / 「Q 값 과대평가와 Double DQN — 시뮬레이션으로 이해하기」 / 「Double DQN 정리: 선택과 평가를 분리하라」
- 다음 주제: 「Dueling DQN」, 「TD3: 연속 행동에서의 과대평가와 clipped double Q」
- 꼬리질문
  1. Double DQN이 과소평가를 일으킬 수는 없나? → 가능(독립 추정의 선택 편향), 실제로 일부 환경에서 관찰됨
  2. 연속 행동에서는 어떻게? → TD3의 clipped double Q (min 사용)
  3. target network를 쓰는 원래 이유는? → 움직이는 target으로 인한 불안정 완화

### 2026-06-28-edge-ai-hardware.md — [AI 기초 13] Edge AI 하드웨어와 온디바이스 학습

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 2 | Posit이 "작은 지수일수록 더 정밀"(반대), CMSIS-NN 레거시 API 미표기, RV64/명령어 길이 혼동, SRAM 읽기 동작 단순화 |
| 입문자 이해도 | 4 | 넓은 주제를 친절하게 풀어 줌 |
| 전문성·깊이 | 3 | 폭은 넓지만 근거 수치(에너지 등)가 없음 |
| 구성·가독성 | 3 | 다섯 주제가 한 글에 섞여 길고 산만함 |
| 코드 품질 | 3 | 예시 C 함수가 deprecated API |

**가장 시급한 3가지**
1. Posit tapered precision 설명 오류
2. CMSIS-NN q7 API가 레거시라는 점
3. 글 분리 권장: (TinyTL·온디바이스 학습) / (SRAM·PIM·RISC-V·Posit 하드웨어)

**주요 변경**
- (정확성) Posit: "underflow 없음 = 작은 수를 더 정밀하게"가 아니라 "0으로 떨어지지 않고 minpos로 반올림, 크기가 1에서 멀수록 정밀도 감소", 정수 비교 가능 장점
- (정확성) SRAM 속도 이유 "(대칭 구조)" → refresh 없음·온칩 배치, 읽기 동작에 precharge·sense amp 반영
- (정확성) "64bit, 128bit 형식은 커스텀 공간이 넓다" → 48/64bit 명령어 인코딩, RV64의 64bit는 레지스터 폭
- (정확성) TinyTL "약 6배" → "최대 약 6.5배"(초록), "채널 6.5배 감소"에 `[확인 필요]`
- Horowitz 에너지 표(덧셈 0.1pJ vs DRAM 640pJ) 추가 → 데이터 이동 병목의 근거
- CMSIS-NN 최신 `_s8` API 경고, 개인 연구 계획 절에 공개 여부 확인 주석

**추가 제안**
- 제목 후보: 「Edge AI 하드웨어 입문: 온디바이스 학습부터 PIM, RISC-V, Posit까지」 / 「TinyTL로 보는 온디바이스 학습의 메모리 문제」 / 「경량 모델을 기기에 올리기 위한 하드웨어 기초」
- 다음 주제: 「MCUNet과 TinyML 배포 실습」, 「int8 양자화 모델을 TFLite Micro로 MCU에 올리기」
- 꼬리질문
  1. 학습 메모리의 대부분은 무엇인가? → 활성값 (글에 있음)
  2. 왜 ReLU는 역전파 메모리가 적나? → 1bit 마스크 (글에 있음)
  3. PIM의 한계는? → 아날로그 연산 정확도·변동성, 프로그래밍 모델, 지원 연산 제한

### 2026-08-22-efficientzero.md — EfficientZero

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 수치·구조 모두 논문과 일치 |
| 입문자 이해도 | 4 | MCTS 정의가 없음 |
| 전문성·깊이 | 4 | 세 개선의 "문제→해결"이 명확, 결과 해석의 주의점 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | consistency loss 코드 적절 |

**가장 시급한 3가지**
1. 샘플 효율과 계산 효율의 구분 (면접에서 "그럼 계산은?"이 바로 나옴)
2. 평균 HNS의 함정과 IQM
3. MCTS 정의

**주요 변경**
- "같은 성능을 500배 적은 데이터" → "DQN 2억 frame 성능에 근접한 결과를 500배 적은 데이터로"
- MCTS 한 문장 정의, "더 알아보기: 숫자를 읽을 때 주의할 점" details, arXiv·Agarwal 2021 링크

**추가 제안**
- 제목 후보: 「EfficientZero 논문 리뷰: 2시간 플레이로 사람을 넘은 Model-Based RL」 / 「MuZero를 샘플 효율적으로: EfficientZero의 세 가지 개선」 / 「Atari 100k를 정복한 EfficientZero 정리」
- 다음 주제: 「MuZero 상세 정리」, 「DreamerV3: world model 기반 RL」
- 꼬리질문
  1. consistency loss에서 stop-gradient가 없으면? → 표현 붕괴(collapse)
  2. value prefix를 LSTM으로 예측하는 이유? → 누적 오차 감소, 시점 오차에 강건
  3. Reanalyze는 왜 off-policy 문제를 줄이나? → 최신 모델·정책으로 target 재계산

### 2026-08-20-hindsight-experience-replay.md — HER

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 5 | bit-flipping 예시와 축구공 비유가 좋음 |
| 전문성·깊이 | 3 | "왜 off-policy만 되나"가 빠짐 (가장 많이 나오는 질문) |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 사용 예·결과 없음, reward_fn이 -0.0을 반환 |

**가장 시급한 3가지**
1. off-policy 전제 이유 설명
2. 코드 사용 예와 실행 결과
3. 실제 구현(샘플링 시점 relabel)과의 차이

**주요 변경**
- HER로 bit-flipping n=50까지 해결 추가
- `reward_fn` 람다 → 함수(−0.0 문제 수정), **1차원 실패 에피소드 relabel 예제와 실행 결과**
- "왜 off-policy 알고리즘에만 쓸 수 있나" 절, Baselines의 샘플링 시점 relabel(future_p=0.8) 박스

**추가 제안**
- 제목 후보: 「Hindsight Experience Replay(HER) 정리: Sparse Reward를 푸는 목표 재라벨링」 / 「실패에서 배우는 강화학습, HER 논문 리뷰」 / 「HER + DDPG로 로봇 팔 학습하기: 개념과 구현」
- 다음 주제: 「Goal-conditioned RL과 UVFA」, 「Fetch 환경에서 HER 직접 돌려 보기 (gymnasium-robotics)」
- 꼬리질문
  1. HER을 PPO에 쓸 수 있나? → 그대로는 불가, 이유는 off-policy (반영)
  2. future 전략이 가장 좋은 이유? → 현재 행동이 영향을 줄 수 있는 상태 (글에 있음)
  3. reward shaping 대비 장점과 한계? → 설계 불필요 / 목표 공간이 정의되고 보상을 재계산할 수 있어야 함


### 2026-07-27-latent-consistency-model.md — Latent Consistency Model

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 수식·수치 정확 |
| 입문자 이해도 | 4 | PF-ODE 부분이 입문자에게 가파름, 그림 활용은 좋음 |
| 전문성·깊이 | 4 | guided distillation·skipping-step까지 다룸, 트레이드오프(negative prompt 등) 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | diffusers 버전·GPU 요구 미표기, 실행 결과 없음 |

**가장 시급한 3가지**
1. 코드 실행 환경(GPU, diffusers 버전)과 결과 이미지
2. 논문 ω와 라이브러리 guidance_scale의 1 차이
3. CFG를 증류한 대가(negative prompt 무력화)

**주요 변경**
- ω ↔ s=1+ω 표기 차이 박스, diffusers 요구 사항, 결과 이미지·측정 시간 빈칸
- negative prompt·guidance 범위 제한 경고, arXiv·diffusers 문서 링크

**추가 제안**
- 제목 후보: 「Latent Consistency Model 논문 리뷰: Stable Diffusion을 4 step으로」 / 「LCM 정리: Consistency Distillation으로 Diffusion 가속하기」 / 「Consistency Model에서 LCM, LCM-LoRA까지」
- 다음 주제: 「DDIM과 DPM-Solver: ODE solver로 보는 diffusion 샘플링」, 「LCM-LoRA로 실시간 이미지 생성 실습」
- 꼬리질문
  1. Consistency model과 progressive distillation 차이? → 단계별 절반 증류 vs 궤적 어디서든 한 번에
  2. 왜 1 step보다 4 step이 낫나? → 노이즈 재주입으로 오차 보정
  3. EMA target이 필요한 이유? → 자기 자신을 목표로 쫓는 불안정 완화 (DQN target과 유사)

### 2026-07-11-latent-diffusion-models.md — Latent Diffusion Model

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | LDM 원 논문과 Stable Diffusion의 텍스트 인코더를 동일시 |
| 입문자 이해도 | 5 | 두 단계 압축 설명과 그림이 좋음 |
| 전문성·깊이 | 4 | f 트레이드오프 좋음, latent 스케일링 같은 실무 디테일 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | CrossAttention 사용 예·shape 없음 |

**가장 시급한 3가지**
1. 텍스트 인코더 서술 정정
2. latent 스케일링 상수(0.18215) 같은 실무 함정
3. 코드 사용 예

**주요 변경**
- (정확성) τθ: LDM 원 논문은 Transformer를 함께 학습, SD v1은 frozen CLIP ViT-L/14
- latent 크기 1/64(공간) vs 1/48(전체 값) 구분
- 0.18215 scaling factor details
- CrossAttention 사용 예와 출력 shape (실행 확인), arXiv·FID 논문 링크

**추가 제안**
- 제목 후보: 「Latent Diffusion Model(Stable Diffusion) 논문 리뷰」 / 「Stable Diffusion은 왜 빠를까: Latent Diffusion 정리」 / 「LDM 구조 완전 정리: Autoencoder, U-Net, Cross-Attention」
- 다음 주제: 「DDPM 수식 유도」, 「Classifier-Free Guidance 정리」
- 꼬리질문
  1. 왜 VAE latent에서 diffusion이 픽셀보다 효율적인가? → 지각적으로 무의미한 고주파 제거 (글에 있음)
  2. f를 크게 하면? → 품질 정체 (글에 있음)
  3. FID의 한계? → Inception feature 의존, 샘플 수에 민감, 텍스트 정합성 반영 못함

### 2026-06-20-lightweight-vision-transformer.md — [AI 기초 11] 경량 Vision Transformer 동향

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 3 | 비교 표에서 Dense(FC)를 "모든 입력에 같은 가중치"로 서술(오류), ViT의 데이터 의존성 과장(DeiT 누락) |
| 입문자 이해도 | 4 | backbone/neck/head 설명 좋음 |
| 전문성·깊이 | 3 | Swin 같은 핵심 연구 누락 |
| 구성·가독성 | 4 | 좋음 |
| 코드 품질 | – | 코드 없음 |

**가장 시급한 3가지**
1. Conv/Dense/Attention 비교 표 오류
2. "ViT는 JFT가 있어야 CNN을 넘는다"는 단순화 → DeiT
3. Swin Transformer 누락

**주요 변경**
- (정확성) 비교 표: 공유 vs 연결마다 다름 vs 입력 의존
- (정확성) ViT: ImageNet-1k < ResNet, 21k ≈, JFT > 로 세분화 + DeiT 추가
- SE와 self-attention 비유의 한계, Swin Transformer 추가, arXiv 링크

**추가 제안**
- 제목 후보: 「Vision Transformer는 왜 무거울까: Inductive Bias와 경량 ViT 정리」 / 「경량 ViT 동향: MobileViT, EfficientFormer, MobileOne」 / 「CNN vs ViT, 그리고 둘을 섞는 경량 모델들」
- 다음 주제: 「Swin Transformer 정리」, 「MAE: Masked Autoencoder」
- 꼬리질문
  1. ViT의 inductive bias가 약하다는 건 구체적으로? → locality·translation equivariance 없음 (글에 있음)
  2. 작은 데이터로 ViT를 학습하려면? → DeiT식 augmentation·distillation, conv stem, 하이브리드 (반영)
  3. FLOPs가 비슷한데 ViT가 모바일에서 느린 이유? → reshape, softmax, LayerNorm 등 메모리 바운드 연산 (글에 있음)

### 2026-05-07-ml-overview.md — [AI 기초 1] 머신러닝의 큰 그림

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | argmax와 one-hot encoding 혼동 |
| 입문자 이해도 | 5 | 표와 예시가 친절함 |
| 전문성·깊이 | 3 | 입문 글이라 적당하나 K-Means 초기화 등 실무 디테일 부재 |
| 구성·가독성 | 5 | 시리즈 목차 포함, 좋음 |
| 코드 품질 | 3 | KMeans 예제가 시드 없음·결과 없음 |

**가장 시급한 3가지**
1. one-hot encoding 설명 오류
2. 코드 결과 부재
3. 2절이 강좌 전개를 그대로 따름 (자기 말로 다시 쓰기)

**주요 변경**
- (정확성) argmax와 one-hot 구분
- L1/L2 표 "이상치를 무시" → "영향을 줄이고 싶을 때"
- GD 결과 실제값(W=1.987, b=0.992), bias를 행렬에 합치는 방법
- KMeans 시드 고정 + 실행 결과(scikit-learn 1.9), k-means++·n_init
- 「모두를 위한 딥러닝」 전개를 따른다는 점을 HTML 주석으로 표시

**추가 제안**
- 제목 후보: 「머신러닝 입문: 지도·비지도·강화학습 한 번에 정리」 / 「가설, 비용 함수, 경사 하강법 — 머신러닝 기본 개념 정리」 / 「AI 기초 1: 머신러닝은 어떻게 배우는가」
- 다음 주제: 「편향-분산 트레이드오프」, 「평가 지표: Precision, Recall, F1, ROC-AUC」
- 꼬리질문
  1. 로지스틱 회귀에 MSE를 안 쓰는 이유? → 비볼록 (글에 있음) + 기울기 소실
  2. Cross Entropy와 KL divergence 관계? → CE = H(p) + KL(p‖q)
  3. K-Means의 k는 어떻게 정하나? → elbow, silhouette

### 2026-07-16-mobile-former.md — Mobile-Former

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 논문과 일치 |
| 입문자 이해도 | 4 | 단계별 설명이 좋음 |
| 전문성·깊이 | 4 | 경량화 포인트를 정확히 짚음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | 동작함, 단순화 버전이라는 점 미표기 |

**가장 시급한 3가지**
1. "훨씬 작다"를 수치로
2. 코드가 단순화 버전임을 명시
3. 원 논문 링크

**주요 변경**
- attention 행렬 크기 비교(18,816 vs 약 980만, 약 520배), 다리 모듈 파라미터 수(실행 확인)
- 단순화 버전 박스, arXiv 링크

**추가 제안**
- 제목 후보: 「Mobile-Former 논문 리뷰: MobileNet과 Transformer를 병렬로 잇다」 / 「6개의 토큰으로 전역 정보를: Mobile-Former 정리」 / 「경량 하이브리드 백본 Mobile-Former 구조 분석」
- 다음 주제: 「Dynamic ReLU와 동적 네트워크」, 「DETR: Transformer 기반 객체 검출」
- 꼬리질문
  1. 왜 이미지 쪽 projection을 없앴나? → HW가 커서 비용 대부분 (글에 있음)
  2. 토큰 수 M을 늘리면? → 정확도↑ 비용↑
  3. 직렬 하이브리드(MobileViT)와 비교한 장단점? → 병렬은 매 블록 양방향 교환, 구현 복잡

### 2026-05-31-mobilenet.md — MobileNet

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 수치 모두 정확, 존재하지 않는 "다음 글(ShuffleNet, MobileOne)" 안내 |
| 입문자 이해도 | 5 | 연산량 유도가 아주 친절함 |
| 전문성·깊이 | 3 | FLOPs ≠ 실제 속도라는 핵심 논점 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 사용 예·비교 없음 |

**가장 시급한 3가지**
1. FLOPs와 실제 속도의 차이
2. 다음 글 링크 정정
3. 비교 실험 코드

**주요 변경**
- 다음 글 안내를 실제 글(CNN 아키텍처 계보, Mobile-Former) 링크로 수정, ReLU6 언급
- **일반 conv vs depthwise separable 비교 실험**: 파라미터 0.115배, CPU 시간 60.1ms → 34.8ms (실측) → arithmetic intensity·메모리 대역폭 설명
- arXiv 링크

**추가 제안**
- 제목 후보: 「MobileNet 논문 리뷰: Depthwise Separable Convolution의 원리」 / 「연산량 1/9의 비밀, MobileNet 정리 (FLOPs와 실제 속도)」 / 「MobileNet V1·V2 핵심 정리」
- 다음 주제: 「MobileNetV3와 NAS」, 「EfficientNet」
- 꼬리질문
  1. depthwise separable의 연산량 비율 유도 → 1/N + 1/D_K² (글에 있음)
  2. 왜 FLOPs만큼 빨라지지 않나? → 메모리 바운드 (반영)
  3. Linear bottleneck에서 ReLU를 빼는 이유? → 저차원에서 정보 손실 (글에 있음)

### 2026-05-09-neural-network-backpropagation.md — [AI 기초 2] 신경망과 역전파

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 손 계산 모두 정확 (코드로 재검증함) |
| 입문자 이해도 | 5 | 숫자 예제로 따라가는 구성이 최고 |
| 전문성·깊이 | 3 | gradient check, 역전파 비용 같은 한 단계 아래 내용 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | NumPy 코드 결과 미표기 |

**가장 시급한 3가지**
1. 역전파 검증 방법(gradient check) 부재
2. 은닉층 기울기 감소 원인 설명의 단순화
3. NumPy 실행 결과

**주요 변경**
- 은닉층 기울기 1/10이 Sigmoid만의 효과가 아님 명시
- NumPy 실행 결과 추가 (손 계산과 일치 확인)
- **Gradient check 절**: 수치 미분 −0.00371097 = 역전파 값, 비용 비교, `torch.autograd.gradcheck`
- "ReLU가 해결" → "크게 완화", 참고 링크

**추가 제안**
- 제목 후보: 「역전파(Backpropagation) 손으로 계산하기: 2-2-1 신경망 예제」 / 「신경망과 역전파 — chain rule부터 PyTorch autograd까지」 / 「숫자로 이해하는 역전파」
- 다음 주제: 「자동 미분(Autograd)은 어떻게 동작하나: 계산 그래프」, 「가중치 초기화」
- 꼬리질문
  1. 역전파가 왜 효율적인가? → 중간 결과 재사용, reverse-mode AD (반영)
  2. 은닉 뉴런이 여러 출력에 연결되면? → 기울기 합 (글에 있음)
  3. 역전파 구현 검증 방법? → gradient check (반영)

### 2026-08-11-noisy-network.md — Noisy Networks

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 초기화 값까지 정확 |
| 입문자 이해도 | 4 | "상태 의존적 탐험"이 왜 그런지 설명 부족 |
| 전문성·깊이 | 4 | σ 수렴 분석까지 다룸 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | 사용 예·결과 없음 |

**가장 시급한 3가지**
1. 상태 의존적 탐험의 이유
2. 코드 사용 예
3. 노이즈 재샘플 주기 구체화

**주요 변경**
- NoisyLinear 사용 예 + 실행 결과(노이즈마다 출력 변화, eval은 고정)
- 재샘플 주기(DQN 4 step), "왜 상태 의존적인가" 박스, arXiv 링크

**추가 제안**
- 제목 후보: 「NoisyNet 논문 리뷰: ε-greedy 없이 탐험하는 법」 / 「Noisy Networks for Exploration 정리와 PyTorch 구현」 / 「가중치에 노이즈를: NoisyNet DQN」
- 다음 주제: 「Rainbow DQN」, 「Parameter space noise(Plappert et al.)와 비교」
- 꼬리질문
  1. factorized noise를 쓰는 이유? → 난수 생성 비용 p+q (글에 있음)
  2. online/target 노이즈를 독립으로 두는 이유? → 상관으로 인한 편향 (글에 있음)
  3. σ가 0으로 수렴하지 않는 이유? → 문제마다 다른 탐험 전략 (글에 있음)

### 2026-05-20-normalization.md — [AI 기초 6] Normalization 총정리

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | "배치 1이면 분산 0"은 BN1d에만 해당 |
| 입문자 이해도 | 5 | 축 비교 표가 훌륭함 |
| 전문성·깊이 | 4 | ICS 재해석까지 언급 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | weight_norm 블록 import 누락 |

**가장 시급한 3가지**
1. 배치 크기 1 설명 정정
2. Transformer LN의 정규화 축 구체화
3. weight_norm API 버전

**주요 변경**
- (정확성) 배치 1: BN1d는 PyTorch 에러, BN2d는 공간 위치로 계산됨 (실행 확인)
- Transformer의 LN은 토큰 하나의 d_model 차원 기준
- weight_norm import, g·v shape 확인, 구 API deprecated 안내, arXiv 링크

**추가 제안**
- 제목 후보: 「BatchNorm vs LayerNorm vs GroupNorm: 정규화 축으로 정리하기」 / 「딥러닝 Normalization 총정리 (BN, LN, IN, GN, WN)」 / 「Normalization은 어떤 축으로 평균을 내는가」
- 다음 주제: 「RMSNorm과 LLM의 정규화」, 「BN을 fine-tuning할 때의 함정 (freeze BN)」
- 꼬리질문
  1. 추론 때 BN이 다르게 동작하는 이유? → running stats (글에 있음)
  2. Transformer가 BN 대신 LN을 쓰는 이유? → 가변 길이·배치 의존성 (글에 있음)
  3. Conv 뒤 BN이면 conv bias가 필요 없는 이유? → β가 흡수 (글에 있음)


### 2026-06-07-object-detection.md — [AI 기초 9] Object Detection 기초

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 3 | YOLOv5 저장소의 v6.0 릴리스와 별개 프로젝트인 YOLOv6(Meituan)을 섞어 서술, YOLO v1 threshold 0.2의 출처 불명 |
| 입문자 이해도 | 5 | IoU·NMS·mAP 예시가 친절함 |
| 전문성·깊이 | 4 | 폭이 넓음, RoIAlign 같은 핵심 디테일 부재 |
| 구성·가독성 | 4 | 매우 길지만 버전별 요약 표가 좋음 |
| 코드 품질 | 4 | 정확함 (실행 확인) |

**가장 시급한 3가지**
1. YOLOv5/v6 서술 정정
2. RoI Pooling의 양자화 문제와 RoIAlign
3. 최신 흐름(anchor-free, NMS-free) 한 줄 언급

**주요 변경**
- (정확성) YOLOv5(Ultralytics) v6.0 릴리스의 Nano 모델 vs YOLOv6(Meituan, 논문 있음) 분리, 요약 표 수정
- (정확성) YOLO v1 threshold 0.2 → "해설 자료에서 흔히" + `[확인 필요]`
- NMS 예시 IoU 0.82 확인, NMS threshold 트레이드오프와 Soft-NMS, RoIAlign details, anchor-free/NMS-free 박스, 참고 문헌

**추가 제안**
- 제목 후보: 「Object Detection 입문: IoU, NMS, mAP부터 R-CNN, YOLO까지」 / 「2-stage vs 1-stage 객체 검출 총정리」 / 「YOLO v1~v7 변천사와 Detection 기본 개념」
- 다음 주제: 「mAP 계산 직접 구현하기」, 「DETR: Transformer 기반 end-to-end 검출」
- 꼬리질문
  1. mAP@0.5와 mAP@[.5:.95] 차이? → 위치 정밀도 요구 수준
  2. 1-stage가 2-stage보다 정확도가 낮았던 이유? → 클래스 불균형, Focal loss (글에 있음)
  3. anchor 방식의 단점? → 하이퍼파라미터 많음, 데이터셋 의존 → anchor-free

### 2026-05-15-optimizer-and-scheduler.md — [AI 기초 4] Optimizer와 Scheduler

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 대부분 정확, "SGD가 일반화 좋다"는 단정, 1cycle 설명이 PyTorch 기본값과 다름 |
| 입문자 이해도 | 5 | 비유와 표가 좋음 |
| 전문성·깊이 | 3 | 직접 비교 실험, PyTorch 구현과 논문 식 차이 같은 깊이 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | API 나열 위주 |

**가장 시급한 3가지**
1. optimizer 차이를 체감할 실험
2. PyTorch momentum 식이 논문 식과 다르다는 점
3. 근거 없는 단정 보완

**주요 변경**
- **좁은 골짜기 함수에서 SGD/Momentum/Nesterov/Adam 비교 실험 + 실행 결과**, SGD 발산 조건(η > 2/50) 설명, "순위는 학습률에 따라 바뀐다"
- SGD 일반화 주장에 Wilson 2017과 반론 추가
- OneCycleLR 기본값(pct_start=0.3), 스케줄러 호출 순서(1.1부터), PyTorch momentum 식 차이 details
- (작업 중 발견) 수식 `\frac`이 제어 문자로 깨졌던 것을 수정하고, 전체 글에서 제어 문자가 없는지 검사함

**추가 제안**
- 제목 후보: 「Optimizer 총정리: SGD, Momentum, Adam, AdamW를 실험으로 비교하기」 / 「딥러닝 Optimizer와 Learning Rate Scheduler 완벽 정리」 / 「Adam은 왜 기본값으로 잘 동작할까」
- 다음 주제: 「Learning rate finder와 warm-up의 이유」, 「Sharpness-Aware Minimization(SAM)」
- 꼬리질문
  1. Adam의 bias correction이 필요한 이유? → 0 초기화로 초반 추정치가 0쪽으로 치우침 (글에 있음)
  2. Adam + L2 정규화가 weight decay와 다른 이유? → 적응형 스케일링, AdamW (글에 있음)
  3. Warm-up이 필요한 이유? → 초기 적응형 학습률 분산, 큰 배치 (글에 있음)

### 2026-05-16-overfitting-regularization-augmentation.md — [AI 기초 5] 과적합과 일반화

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | "과소적합 해결: 데이터 늘리기"는 맞지 않는 처방, L2 = weight decay를 무조건 동일시 |
| 입문자 이해도 | 5 | 표와 예시가 친절함 |
| 전문성·깊이 | 4 | L1이 0을 만드는 이유 설명이 좋음 |
| 구성·가독성 | 4 | 길지만 Part 구분이 좋음 |
| 코드 품질 | 3 | K-fold 코드가 실행 불가(정의되지 않은 함수) |

**가장 시급한 3가지**
1. 과소적합 처방 정정
2. 실행 가능한 K-fold 예제와 데이터 누수 주의
3. weight decay ≡ L2 조건

**주요 변경**
- (정확성) 과소적합 해결에서 "데이터 늘리기" → "유용한 feature 추가" + 설명
- (정확성) L2와 weight decay는 SGD에서만 동일, Adam이면 AdamW
- bias-variance 분해식, double descent 박스
- **K-fold를 실행 가능한 예제로 교체 + 결과(0.956~1.0, 평균 0.972), Pipeline으로 데이터 누수 방지**
- CutMix 사용 예·결과, transforms v2 안내

**추가 제안**
- 제목 후보: 「과적합 완벽 정리: 진단부터 Regularization, Dropout, Data Augmentation까지」 / 「Overfitting은 왜 생기고 어떻게 막을까」 / 「L1·L2 규제와 Mixup·CutMix 정리」
- 다음 주제: 「Label Smoothing과 Knowledge Distillation의 관계」, 「데이터 누수(Data Leakage) 사례 모음」
- 꼬리질문
  1. Dropout을 추론 때 끄는 이유와 inverted dropout? (글에 있음)
  2. L1이 sparse한 해를 만드는 이유? (글에 있음)
  3. validation set에도 과적합될 수 있나? → 반복 선택으로 가능, test set 분리 (글에 있음)

### 2026-06-23-pruning-and-quantization.md — Pruning & Quantization 입문

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 정확, Lottery Ticket을 일반 법칙처럼 서술 |
| 입문자 이해도 | 5 | CSR, affine quantization 설명이 친절함 |
| 전문성·깊이 | 4 | structured/unstructured 구분 좋음, prune 모듈의 실제 동작 미언급 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | 결과 미표기 |

**가장 시급한 3가지**
1. `torch.nn.utils.prune`이 실제로는 메모리를 늘린다는 점 (면접 함정)
2. Lottery Ticket의 적용 범위와 반론
3. 코드 결과

**주요 변경**
- magnitude prune 결과(90% 0), **`prune.l1_unstructured` 실제 동작 확인(weight_orig + weight_mask, 크기 그대로)** → `prune.remove` 안내
- Lottery Ticket: 큰 모델은 rewinding 필요(Frankle 2020), Liu 2019 반론
- fake quantize 결과(8bit 0.0090 vs 4bit 0.1625, 약 18배)와 해석
- 다음 글 링크를 BNN 글로

**추가 제안**
- 제목 후보: 「모델 압축 입문: Pruning과 Quantization 원리와 PyTorch 실습」 / 「INT8 양자화와 가지치기, 실제로 빨라지는 조건은?」 / 「Pruning, Lottery Ticket, PTQ·QAT 한 번에 정리」
- 다음 주제: 「PyTorch로 INT8 PTQ 해 보기 (실측)」, 「LLM 양자화: GPTQ, AWQ」
- 꼬리질문
  1. unstructured pruning이 GPU에서 안 빨라지는 이유? (글에 있음)
  2. per-tensor vs per-channel 양자화? → 채널별 scale로 오차 감소
  3. QAT에서 STE의 역할? (글에 있음)

### 2026-07-02-pytorch-training-toolkit.md — [AI 기초 15] PyTorch 실전 도구 모음

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 3 | TensorBoard 지원 버전(1.4) 의심, skimage→ToTensor float64 함정 미인지 |
| 입문자 이해도 | 5 | 각 도구의 "왜"를 설명함 |
| 전문성·깊이 | 3 | 실무 함정(module. 접두사, weights_only) 부재 |
| 구성·가독성 | 4 | 길지만 정리 표가 좋음 |
| 코드 품질 | 3 | 학습 재개에 strict=False 권장은 위험 |

**가장 시급한 3가지**
1. `strict=False` + DataParallel `module.` 접두사 함정 (실제로 아무것도 안 불러와도 에러 없음)
2. torch.load `weights_only` 기본값 변경과 보안
3. skimage 결과가 float64로 남는 문제

**주요 변경**
- **DataParallel state_dict를 strict=False로 넣으면 missing/unexpected만 나오고 아무것도 안 들어감 — 실행 결과로 시연**
- PyTorch 2.6 `weights_only=True` 기본값과 pickle 보안 경고
- skimage resize → ToTensor → torch.float64 경고와 해결책
- TensorBoard 버전 `[확인 필요]`, 참고 링크

**추가 제안**
- 제목 후보: 「PyTorch 실험 필수 도구: 체크포인트, TensorBoard, Knowledge Distillation」 / 「PyTorch 모델 저장·불러오기의 함정들」 / 「argparse부터 KD까지, PyTorch 학습 스크립트 뼈대」
- 다음 주제: 「DistributedDataParallel로 멀티 GPU 학습」, 「Weights & Biases로 실험 관리」
- 꼬리질문
  1. state_dict 저장을 권장하는 이유? (글에 있음 + 보안)
  2. KD에서 T²를 곱하는 이유? (글에 있음)
  3. `model.eval()`과 `torch.no_grad()` 차이? → 동작 모드 vs 기울기 기록 (글에 있음)

### 2026-07-15-qmix.md — QMIX

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 4 | CTDE·IGM 설명 좋음 |
| 전문성·깊이 | 4 | hypernetwork를 쓰는 이유까지 설명, 한계를 구체 예시로 보여 주지 않음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | 단조성 확인 코드 없음 |

**가장 시급한 3가지**
1. 한계(non-monotonic)를 구체 행렬로
2. 단조성이 구조로 보장됨을 코드로 확인
3. 존재하지 않는 후속 글 예고

**주요 변경**
- **autograd로 ∂Q_tot/∂Q_a ≥ 0 확인 (무작위 초기화에서도 모두 양수)**
- non-monotonic 보상 행렬 예시(8 / −12 / 0), QTRAN·WQMIX·QPLEX 언급으로 예고 문구 교체
- COMA 글 링크, arXiv 링크

**추가 제안**
- 제목 후보: 「QMIX 논문 리뷰: 멀티에이전트 협력을 위한 단조 가치 분해」 / 「VDN에서 QMIX까지, MARL 가치 분해 정리」 / 「QMIX의 hypernetwork는 왜 필요한가」
- 다음 주제: 「QPLEX와 IGM 완전 표현」, 「SMAC 벤치마크로 QMIX 돌려 보기」
- 꼬리질문
  1. IGM이란? (글에 있음)
  2. 왜 abs로 가중치를 양수로 만드나? (글에 있음)
  3. QMIX가 실패하는 게임은? (반영)

### 2026-05-23-resnet.md — ResNet

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | "projection shortcut의 파라미터가 블록 전체의 두 배"로 읽히는 문장, 원 ResNet shortcut이 완전한 identity가 아님을 미언급 |
| 입문자 이해도 | 5 | degradation → residual 흐름이 명확 |
| 전문성·깊이 | 4 | pre-activation, v1.5 같은 디테일 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | 사용 예 없음 |

**주요 변경**
- (정확성) projection shortcut 문장을 실제 수치(본체 70,400 vs projection 66,048)로 수정
- pre-activation ResNet 박스, stride 위치(ResNet v1.5) 설명, 블록 사용 예와 shape
- 다음 글 링크 정정, arXiv 링크

**추가 제안**
- 제목 후보: 「ResNet 논문 리뷰: Residual Connection은 왜 깊은 학습을 가능하게 했나」 / 「Degradation 문제와 ResNet의 해법」 / 「ResNet BasicBlock·Bottleneck PyTorch 구현」
- 다음 주제: 「Pre-activation ResNet과 identity mapping」, 「ResNeXt와 Wide ResNet」
- 꼬리질문
  1. degradation과 overfitting의 차이? (글에 있음)
  2. 왜 F(x)=0이 identity 학습보다 쉬운가? (글에 있음)
  3. bottleneck에서 identity shortcut이 중요한 이유? (반영)

### 2026-06-13-rnn-lstm-gru.md — [AI 기초 10] RNN, LSTM, GRU

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 1997 LSTM에 forget gate가 있었던 것처럼 서술, "고윳값" 표현 |
| 입문자 이해도 | 5 | 좋음 |
| 전문성·깊이 | 4 | 파라미터 수 비교가 좋음, GRU 식 표기 차이 같은 실무 함정 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 5 | 파라미터 수 주석까지 정확 (실행 확인) |

**주요 변경**
- (정확성) forget gate는 Gers et al.(2000)이 추가
- (정확성) 고윳값 → 특이값, 폭주 조건은 필요조건
- forget gate bias 1 초기화 팁, PyTorch GRU 식의 z 의미 차이, bidirectional h shape
- 참고 문헌

**추가 제안**
- 제목 후보: 「RNN, LSTM, GRU 완벽 정리: 기울기 소실과 게이트」 / 「LSTM은 어떻게 장기 기억을 유지할까」 / 「시퀀스 모델 기초: RNN에서 GRU까지」
- 다음 주제: 「Seq2Seq와 Bahdanau Attention」, 「Mamba 같은 상태공간모델(SSM)」
- 꼬리질문
  1. LSTM이 vanishing을 완화하는 원리? → cell state 덧셈 경로 (글에 있음)
  2. GRU와 LSTM 중 무엇을 고르나? → 데이터·자원, 실험적 비슷 (글에 있음)
  3. gradient clipping은 vanishing에도 효과가 있나? → 아니오, exploding용 (글에 있음)

### 2026-08-05-textdiffuser.md — TextDiffuser

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 정확, λ_char 값 불명 |
| 입문자 이해도 | 4 | 구조 설명이 명확 |
| 전문성·깊이 | 3 | "CLIP이 글자 모양을 못 담는다"의 근본 원인(토크나이저) 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | – | 코드 없음 |

**주요 변경**
- subword 토크나이저가 글자 정보를 지운다는 설명(character-blind, Liu et al. 2023)
- latent 공간 loss의 실용적 이점 박스, λ_char `[확인 필요]`, arXiv 링크

**추가 제안**
- 제목 후보: 「TextDiffuser 논문 리뷰: Diffusion으로 글자를 정확하게 그리기」 / 「이미지 생성 모델은 왜 글자를 못 쓸까: TextDiffuser 정리」 / 「Layout + Character-aware loss로 보는 텍스트 렌더링」
- 다음 주제: 「TextDiffuser-2와 LLM 기반 레이아웃」, 「한글 텍스트 렌더링의 어려움」
- 꼬리질문
  1. 왜 2단계로 나눴나? → 레이아웃과 렌더링 분리 (글에 있음)
  2. OCR 지표의 한계? → OCR 모델 자체 오류, 스타일 평가 불가
  3. 한국어로 확장하려면? → 글자 수·조합형 구조, 데이터

### 2026-08-23-trajectory-transformer.md — Trajectory Transformer

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | QMIX를 Transformer 사용 예로 링크(부정확), "Markov property 대신 beam search" 표현 |
| 입문자 이해도 | 4 | 언어 모델 비유가 좋음 |
| 전문성·깊이 | 5 | Markovian Transformer ablation, DT 비교까지 훌륭 |
| 구성·가독성 | 4 | 길지만 흐름이 좋음 |
| 코드 품질 | 3 | 이산화 함수 사용 예 없음 |

**주요 변경**
- (정확성) QMIX 링크 제거, beam search 문장 수정
- **quantile vs uniform 이산화 비교 예제 + 결과(uniform은 빈 토큰 20개, quantile은 모두 100개씩)**
- Diffuser 글 링크, arXiv 링크

**추가 제안**
- 제목 후보: 「Trajectory Transformer 논문 리뷰: 강화학습을 시퀀스 모델링으로」 / 「GPT처럼 계획하는 강화학습, Trajectory Transformer」 / 「Trajectory Transformer vs Decision Transformer」
- 다음 주제: 「Decision Transformer 정리」, 「Offline RL 벤치마크 D4RL 이해하기」
- 꼬리질문
  1. 왜 연속값을 이산화했나? → multi-modal 분포 표현 (글에 있음)
  2. beam search의 비용 문제? → 차원별 토큰 × 빔 폭, 실시간 부담 (글에 있음)
  3. DT와의 차이? (글에 있음)

### 2026-07-07-transformer-decoder-to-gpt.md — Transformer 디코더에서 GPT로

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 내용은 정확, 존재하지 않는 다음 글(InstructGPT) 예고 |
| 입문자 이해도 | 5 | 좋음 |
| 전문성·깊이 | 4 | scaling law까지, Chinchilla 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 2 | **실행하면 RuntimeError**(슬라이싱한 targets에 view), 초기화가 없어 초기 loss가 약 42 |

**가장 시급한 3가지**
1. 코드 버그 2개
2. 초기 loss ≈ ln(vocab) 점검법
3. Chinchilla로 scaling law 갱신

**주요 변경**
- (코드) `view(-1)` → `reshape(-1)`, GPT-2식 std 0.02 초기화 추가 → 초기 loss 41.96 → 4.60 (실행 확인)
- 실행 예와 결과, 기본 설정 파라미터 수 124,439,808 ≈ GPT-2 small 확인
- Chinchilla 추가, 다음 글 예고 문구 수정, arXiv 링크

**추가 제안**
- 제목 후보: 「GPT는 어떻게 만들어졌나: Decoder-only Transformer 완전 정리」 / 「미니 GPT 직접 구현하기 (PyTorch)」 / 「GPT-1에서 GPT-3까지, 그리고 RLHF」
- 다음 주제: 「InstructGPT와 RLHF 3단계」, 「KV Cache와 추론 최적화」
- 꼬리질문
  1. 학습은 병렬인데 생성은 왜 순차인가? (글에 있음)
  2. weight tying의 장점? → 파라미터 절감, 입출력 임베딩 공유
  3. 초기 loss가 ln(V)보다 훨씬 크면 무엇을 의심하나? → 초기화·스케일 문제 (반영)

### 2026-07-12-trpo-to-ppo.md — TRPO에서 PPO까지

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | lower bound의 ε 정의 누락 (PPO의 ε과 혼동 가능) |
| 입문자 이해도 | 4 | clip 두 경우 설명이 좋음 |
| 전문성·깊이 | 4 | TRPO 유도 흐름 좋음, 구현 디테일의 중요성 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | GAE 사용 예 없음 |

**주요 변경**
- (정확성) C 안의 ε = max|A| 정의
- clipping 효과를 기울기 표로(r=1.6이면 기울기 0, 실행 확인)
- GAE 사용 예와 λ별 결과(1-step TD ↔ MC 확인)
- Engstrom 2020 "구현 디테일" 경고, ACER 링크

**추가 제안**
- 제목 후보: 「TRPO에서 PPO까지: 안정적인 Policy Gradient 완벽 정리」 / 「PPO clipping은 왜 동작할까 — 수식과 코드로」 / 「TRPO·PPO·GAE 핵심 정리」
- 다음 주제: 「PPO 구현 디테일 37가지 정리」, 「RLHF에서의 PPO」
- 꼬리질문
  1. PPO가 on-policy인데 여러 epoch 재사용해도 되는 이유? → ratio 보정 + clipping으로 멀리 못 감
  2. GAE의 λ 역할? (반영)
  3. TRPO의 단조 향상 보장이 실제 알고리즘에서도 유지되나? → 근사(평균 KL 등)로 깨짐 (글에 있음)


### 2026-07-22-adabin.md — AdaBin

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 정확, 코드의 std가 unbiased라 식(1/n)과 불일치, "Maxout"이 원래 Maxout과 다른 정의임을 미언급 |
| 입문자 이해도 | 4 | 중심·거리 아이디어가 명확 |
| 전문성·깊이 | 3 | "보정 항"이 무엇인지 전개가 없음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | 사용 예 없음 |

**주요 변경**
- **β가 있을 때 XNOR 내적이 네 항으로 분해되는 식** 추가(어느 항이 미리 계산 가능한지)
- `std(unbiased=False)`, 채널별 두 값 확인 예(실행 확인), Maxout 명칭 주의, arXiv 링크

**추가 제안**
- 제목 후보: 「AdaBin 논문 리뷰: 이진값의 중심을 옮기는 BNN」 / 「{-1,+1}을 넘어서: Adaptive Binary Set 정리」 / 「AdaBin으로 보는 BNN 분포 맞추기」
- 다음 주제: 「ReCU와 dead weight 문제」, 「BNN의 활성 분포 시각화 실습」
- 꼬리질문: ① β를 두면 XNOR 연산이 깨지지 않나? → 4항 분해 (반영) ② 가중치 α·β를 평균·표준편차로 두는 근거? → 가우시안 가정 하 KL 최소화 ③ 추론 시 추가 비용? → 채널별 상수 몇 개

### 2026-09-20-adaptdiffuser.md — AdaptDiffuser

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | ChatGPT 미로 생성 언급의 출처 불명 |
| 입문자 이해도 | 4 | Diffuser 복습이 친절 |
| 전문성·깊이 | 3 | 자기 데이터로 재학습할 때의 위험(편향 강화) 논의 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 선별 함수 사용 예 없음 |

**주요 변경**
- 선별 함수 사용 예: **보상은 높지만 물리적으로 불가능한 궤적이 걸러지는 것** 확인(실행)
- model collapse 위험과 discriminator의 역할 details, ChatGPT 언급 `[확인 필요]`, arXiv 링크

**추가 제안**
- 제목 후보: 「AdaptDiffuser 논문 리뷰: 스스로 데이터를 만들어 진화하는 Diffusion Planner」 / 「Diffuser의 일반화 한계와 AdaptDiffuser」 / 「합성 전문가 데이터로 planner 적응시키기」
- 다음 주제: 「Decision Diffuser」, 「Inverse dynamics model 정리」
- 꼬리질문: ① 왜 guidance만으로 부족한가? → μθ가 데이터에 묶임 (글에 있음) ② discriminator 기준은? → 동역학 일관성 + 보상 (반영) ③ 반복할수록 생기는 위험? → 편향 강화 (반영)

### 2026-07-24-addernet.md — AdderNet + Approximate Adder

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 3 | "BN의 곱셈은 추론 때 앞 연산에 합칠 수 있다"는 L1 연산에서는 성립하지 않음 |
| 입문자 이해도 | 4 | 좋음 |
| 전문성·깊이 | 4 | 하드웨어 도구까지 연결한 점이 좋음 |
| 구성·가독성 | 3 | AdderNet과 근사 덧셈기·도구 이야기가 섞여 초점이 흐림 |
| 코드 품질 | 4 | 정확 (실행 확인) |

**주요 변경**
- (정확성) BN 문장을 논문의 실제 논리(BN 곱셈은 남지만 conv 대비 무시할 만큼 적음)로 교체
- 의도적으로 정의한 근사 기울기라 gradcheck가 실패하는 것이 정상이라는 설명
- SCALE-Sim V2의 GEMM 입력 설명, arXiv·PyTorch extension 링크

**추가 제안**
- 제목 후보: 「AdderNet 논문 리뷰: 곱셈 없는 CNN」 / 「L1 거리로 만드는 합성곱, AdderNet과 근사 덧셈기」 / 「AdderNet부터 Systolic Array까지」
- 다음 주제: 「ShiftAddNet」, 「SCALE-Sim으로 CNN 가속기 시뮬레이션 해 보기 (실습)」
- 꼬리질문: ① 입력 기울기를 HardTanh로 자르는 이유? (글에 있음) ② L1 연산이 행렬 곱 라이브러리를 못 쓰는 이유? (글에 있음) ③ 근사 덧셈기를 하위 비트에만 쓰는 이유? (글에 있음)

### 2026-09-16-agent57.md — Agent57

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 3 | 배경 표의 DQN 23, Rainbow 53이 의심스러움(특히 Rainbow > R2D2) |
| 입문자 이해도 | 4 | 좋음 |
| 전문성·깊이 | 4 | capped HNS 설명이 훌륭 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | UCB 사용 예 없음 |

**주요 변경**
- 표의 DQN·Rainbow 숫자 `[확인 필요]` + HTML 주석
- **sliding-window UCB가 최적 arm 변화를 따라가는 시뮬레이션**(전반 2번 71% → 후반 0번 69%, 실행 확인)
- arXiv 링크

**추가 제안**
- 제목 후보: 「Agent57 논문 리뷰: Atari 57개 게임을 모두 넘다」 / 「NGU에서 Agent57까지: 탐험과 활용의 meta-controller」 / 「Agent57의 Q 분해와 bandit meta-controller」
- 다음 주제: 「R2D2 정리」, 「MEME/BBF 등 이후 Atari 샘플 효율 연구」
- 꼬리질문: ① capped HNS를 쓰는 이유? (글에 있음) ② Q를 분리하면 왜 안정적인가? → 스케일 분리 (글에 있음) ③ sliding window가 필요한 이유? → 비정상 bandit (반영)

### 2026-09-27-anti-exploration.md — Anti-Exploration

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 4 | online/offline 대비가 명확 |
| 전문성·깊이 | 4 | KL 정규화 연결이 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | CVAE 학습 코드·보너스 검증 없음 |

**주요 변경**
- **CVAE 학습 + 보너스 비교 실험**: 데이터 0.00016 / shuffled 1.03 / random 0.84 (실행) → 상태 조건부 모델링의 중요성
- arXiv 링크

**추가 제안**
- 제목 후보: 「Anti-Exploration 논문 리뷰: 탐험 보너스를 뒤집은 Offline RL」 / 「Offline RL에서 OOD 행동을 막는 법: Anti-Exploration」 / 「TD3-CVAE로 보는 behavior regularization」
- 다음 주제: 「TD3+BC」, 「CQL 정리」
- 꼬리질문: ① offline RL에서 OOD 행동이 문제인 이유? (글에 있음) ② RND가 shuffled를 못 잡는 이유? → 상태 조건부가 아님 (반영) ③ 보너스 계수 α를 키우면? → BC에 가까워짐

### 2026-07-28-asip-posit.md — ASIP Designer로 Posit 연산기

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | Posit 정밀도가 "0~1 구간"에 집중된다는 설명(정확히는 크기 1 근처) |
| 입문자 이해도 | 3 | 도구 용어가 많아 입문자에게 어려움 |
| 전문성·깊이 | 4 | 직접 해 본 흐름이 드러남 |
| 구성·가독성 | 4 | 좋음 |
| 코드 품질 | – | 개념 코드만 |

**주요 변경**
- Posit 정밀도 설명 정정(Edge AI 글과 통일)
- **직접 비교한 면적·타이밍 결과를 넣을 자리** 안내 박스 — 이 글의 가장 큰 차별점이 될 부분
- Synopsys 제품 링크 `[확인 필요]`

**추가 제안**
- 제목 후보: 「Synopsys ASIP Designer로 Posit 덧셈기 만들기」 / 「ASIP 설계 흐름 정리: nML, PDG, Chess」 / 「전용 명령어 프로세서(ASIP) 입문」
- 다음 주제: 「Posit 곱셈기 구현과 비교」, 「RISC-V 커스텀 명령어 실습」
- 꼬리질문: ① ASIP와 ASIC 가속기의 트레이드오프? (글에 있음) ② Posit 하드웨어가 IEEE보다 단순한 이유? → 예외 처리 적음 ③ 직접 RTL vs ASIP 생성 RTL 결과 차이? → 실측값 필요

### 2026-10-14-cbs-task-assignment.md — CBS-TA

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 4 | CBS 복습이 좋음 |
| 전문성·깊이 | 4 | search forest와 lazy 생성이 잘 설명됨 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 2 | cbs() 단독으로 실행 불가 |

**주요 변경**
- **3×3 격자 최소 구현(시공간 BFS + 충돌 검사)과 실행 결과** 추가 → 한 에이전트가 1칸 기다리는 최적해
- vertex conflict만 검사한다는 한계와 edge conflict 처리 방법 박스
- 최적성 전제(low-level 최단 경로) 명시

**추가 제안**
- 제목 후보: 「CBS-TA 논문 리뷰: 작업 할당과 경로 계획을 동시에」 / 「Conflict-Based Search 구현으로 이해하기」 / 「다중 로봇 경로 계획 MAPF와 CBS」
- 다음 주제: 「ECBS와 focal search」, 「Lifelong MAPF(창고 로봇)」
- 꼬리질문: ① CBS가 최적인 이유? (글에 있음) ② 할당 수가 n!인데 어떻게 탐색? → Murty 순위 할당 lazy 생성 (글에 있음) ③ 에이전트가 많아지면? → ECBS, 우선순위 계획

### 2026-09-07-coma.md — COMA

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 4 | credit assignment 설명 좋음 |
| 전문성·깊이 | 3 | 후속 벤치마크(SMAC)에서의 평가 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | advantage 숫자 예시 없음 |

**주요 변경**
- counterfactual advantage 숫자 예(Q=[2,5,1], π=[0.2,0.5,0.3] → baseline 3.2, 실행 확인)
- SMAC에서 QMIX보다 낮았던 결과와 해석 박스, arXiv 링크

**추가 제안**
- 제목 후보: 「COMA 논문 리뷰: Counterfactual Baseline으로 기여도 나누기」 / 「멀티에이전트 credit assignment와 COMA」 / 「COMA vs QMIX」
- 다음 주제: 「MAPPO: 멀티에이전트 PPO」, 「SMAC 벤치마크 정리」
- 꼬리질문: ① baseline이 기울기 기댓값을 안 바꾸는 이유? (글에 있음) ② difference reward와의 차이? (글에 있음) ③ COMA가 SMAC에서 약한 이유? (반영)

### 2026-08-28-count-based-exploration.md — Count-Based Exploration과 Pseudo-Count

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 코드 주석의 "prediction gain"은 로그 차이여야 함 |
| 입문자 이해도 | 5 | Montezuma 예시와 유도가 친절 |
| 전문성·깊이 | 4 | 성질 정리가 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 검증 예 없음 |

**주요 변경**
- **pseudo-count가 정확한 경험적 분포에서 실제 count를 복원하는지 검증**(0/1/10/50 그대로, 실행)
- prediction gain 정의(로그 차이)와 부등식 관계 박스, arXiv 링크

**추가 제안**
- 제목 후보: 「Pseudo-Count 논문 리뷰: 처음 보는 상태에 보너스 주기」 / 「Count-based exploration을 거대한 상태 공간으로」 / 「Montezuma's Revenge와 intrinsic motivation의 시작」
- 다음 주제: 「PixelCNN 밀도 모델 기반 pseudo-count(Ostrovski 2017)」, 「#Exploration (해시 기반 count)」
- 꼬리질문: ① 왜 1/√N인가? → 신뢰구간 폭(UCB) ② 밀도 모델이 일반화를 못 하면? → count와 같아짐 (반영) ③ pseudo-count의 단점? → 밀도 모델 품질 의존

### 2026-09-21-dagger.md — DAgger

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 내용 정확, 코드가 옛 Gym API(4-tuple) |
| 입문자 이해도 | 5 | 운전 비유가 좋음 |
| 전문성·깊이 | 4 | no-regret 분석 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | deprecated API |

**주요 변경**
- 코드를 **Gymnasium API**(reset → (obs, info), step → 5-tuple)로 갱신 + terminated/truncated 구분 이유
- 안전 문제와 HG-DAgger, 참고 링크

**추가 제안**
- 제목 후보: 「DAgger 논문 리뷰: 모방학습의 분포 이동 해결하기」 / 「Behavior Cloning은 왜 실패하나 — DAgger 정리」 / 「O(εT²)에서 O(εT)로: DAgger」
- 다음 주제: 「GAIL」, 「Diffusion Policy(로봇 모방학습)」
- 꼬리질문: ① BC의 오차가 T²로 커지는 이유? (글에 있음) ② β를 줄여 가는 이유? (글에 있음) ③ 사람 전문가 비용을 줄이려면? → HG-DAgger, 능동 질의 (반영)

### 2026-08-17-deepsvg.md — DeepSVG

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | SVG 예시 계산까지 정확 |
| 입문자 이해도 | 5 | 래스터/벡터 비교가 좋음 |
| 전문성·깊이 | 3 | 좌표를 분류로 다루는 이유 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | 사용 예 없음 |

**주요 변경**
- 임베딩 사용 예와 shape, **좌표 양자화(회귀 대신 분류)의 이유** 설명, W3C·arXiv 링크

**추가 제안**
- 제목 후보: 「DeepSVG 논문 리뷰: Transformer로 벡터 그래픽 생성하기」 / 「SVG를 생성하는 계층적 Transformer VAE」 / 「아이콘 보간 애니메이션, DeepSVG 정리」
- 다음 주제: 「IconShop / StarVector 등 LLM 기반 SVG 생성」, 「VAE와 posterior collapse」
- 꼬리질문: ① non-autoregressive로 만든 이유? → 보간 매끄러움 (글에 있음) ② path 순서 무관성은 어떻게? (글에 있음) ③ 좌표를 분류로 다루는 이유? (반영)

### 2026-07-13-deformable-convolution.md — Deformable Convolution

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | translation invariance 표현, "29.4 → 33.1"의 지표 불명 |
| 입문자 이해도 | 5 | 좋음 |
| 전문성·깊이 | 3 | 비용(실행 속도) 논의 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | 정확, zero-init 의미 확인 없음 |

**주요 변경**
- translation equivariance로 정정, 수치 `[확인 필요]`
- **offset 0일 때 일반 conv와 결과가 같음을 확인**(실행), 파라미터·실행 속도 트레이드오프 박스, 링크

**추가 제안**
- 제목 후보: 「Deformable Convolution 논문 리뷰: 격자를 벗어나는 합성곱」 / 「DCN과 Deformable RoI Pooling 정리 (torchvision 코드)」 / 「물체 모양에 맞춰 보는 CNN, DCN」
- 다음 주제: 「DCNv2/v3와 InternImage」, 「Deformable DETR」
- 꼬리질문: ① offset이 소수인데 어떻게 미분하나? → bilinear (글에 있음) ② STN과의 차이? → 전역 변환 vs 위치별 offset ③ 왜 일부 층에만 넣나? → 속도·안정성 (반영)

### 2026-09-19-diffuser.md — Diffuser

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 4 | 인과/반인과 설명이 좋음 |
| 전문성·깊이 | 4 | 성질 정리 좋음, 계산 비용 논의 부족 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 가정한 인터페이스 미표기 |

**주요 변경**
- 코드가 가정하는 model/value_fn 인터페이스 박스(가짜 모델로 실행 흐름 확인), 계획 비용 details, 공식 코드 링크

**추가 제안**
- 제목 후보: 「Diffuser 논문 리뷰: Diffusion으로 계획하기」 / 「Planning with Diffusion — guided sampling과 inpainting」 / 「Diffuser로 보는 model-based RL의 새 방향」
- 다음 주제: 「Decision Diffuser」, 「Diffusion Policy」
- 꼬리질문: ① 왜 autoregressive가 아니라 동시에 예측? (글에 있음) ② 목표 조건은 어떻게? → inpainting (글에 있음) ③ 실시간 제어에 쓰기 어려운 이유? (반영)

### 2026-09-30-diffusion-ql.md — Diffusion-QL

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 5 | "정면 돌진" 예시가 좋음 |
| 전문성·깊이 | 4 | reverse chain 역전파 비용 언급 부족 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 개념 코드 |

**주요 변경**
- **가우시안을 이봉 데이터에 맞추면 평균(데이터에 없는 행동)에 확률이 몰리는 것을 숫자로 확인**(실행)
- N에 비례하는 역전파 비용, arXiv 링크

**추가 제안**
- 제목 후보: 「Diffusion-QL 논문 리뷰: Diffusion 정책으로 Offline RL」 / 「가우시안 정책의 한계와 Diffusion Policy」 / 「Diffusion Policy + Q-learning 정리」
- 다음 주제: 「IDQL」, 「Consistency Policy」
- 꼬리질문: ① 왜 N=5로 충분? → 행동 차원 낮음 (글에 있음) ② diffusion loss가 규제가 되는 이유? (글에 있음) ③ 추론 속도 문제의 해결책? → step 축소, 증류

### 2026-09-04-exploration-via-disagreement.md — Exploration via Disagreement

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 5 | ICM과의 대비가 명확 |
| 전문성·깊이 | 4 | epistemic/aleatoric 구분 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 핵심 주장을 확인하는 실험 없음 |

**주요 변경**
- **Noisy TV 실험**: 예측 오차는 1.01로 그대로인데 불일치는 0.000001, 처음 보는 구간은 0.0325 (실행) → 주장을 직접 검증
- arXiv 링크

**추가 제안**
- 제목 후보: 「Exploration via Disagreement 논문 리뷰」 / 「앙상블 불일치로 탐험하기: Noisy TV 문제의 해법」 / 「미분 가능한 탐험, Disagreement」
- 다음 주제: 「Deep Ensembles와 불확실성 추정」, 「Plan2Explore」
- 꼬리질문: ① epistemic vs aleatoric 불확실성? (글·실험) ② 왜 미분 가능한가? → 다음 상태 불필요 (글에 있음) ③ 앙상블 비용을 줄이려면? → 헤드만 여러 개(공유 몸통)

### 2026-10-09-feudal-networks.md — FeUdal Networks

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 4 | Manager/Worker 표가 좋음 |
| 전문성·깊이 | 4 | transition policy gradient 설명 좋음, dilated LSTM은 말로만 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 사용 예 없음 |

**주요 변경**
- **Dilated LSTM 최소 구현** 추가(실행 확인)
- FuNHeads 사용 예와 출력, `.detach()`가 필요한 이유, arXiv 링크

**추가 제안**
- 제목 후보: 「FeUdal Networks 논문 리뷰: Manager는 방향을, Worker는 행동을」 / 「계층적 강화학습 FuN 정리」 / 「Transition Policy Gradient와 Dilated LSTM」
- 다음 주제: 「HIRO」, 「Option-Critic」
- 꼬리질문: ① 목표를 방향으로 둔 이유? → 절대 위치보다 일반화 (글에 있음) ② Manager를 Worker 역전파로 학습하면? → 목표 의미 상실 (반영) ③ dilated LSTM의 장점? (반영)


### 2026-07-23-fq-vit.md — FQ-ViT

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 4 | 양자화 복습이 친절 |
| 전문성·깊이 | 4 | 원인 분석이 좋음, PTF 효과를 숫자로 보여 주지 않음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 함수만 있고 사용 예 없음 |

**주요 변경**
- **layer-wise vs PTF 비교 실험**: 큰 채널 하나 때문에 작은 채널 오차가 0.26 → PTF로 0.008 (약 33배 감소, 실행 확인)
- arXiv 링크

**추가 제안**
- 제목 후보: 「FQ-ViT 논문 리뷰: LayerNorm과 Softmax까지 양자화하기」 / 「ViT 양자화는 왜 어려울까 — PTF와 Log-Int-Softmax」 / 「Vision Transformer 완전 정수 추론 정리」
- 다음 주제: 「SmoothQuant: LLM 활성값 outlier 다루기」, 「I-BERT의 정수 근사 함수들」
- 꼬리질문: ① 채널별 scale을 쓰면 왜 LayerNorm이 정수로 안 되나? (글에 있음) ② attention map에 log2 양자화가 맞는 이유? (글에 있음) ③ PTQ와 QAT 중 무엇을? → 비용·정확도 트레이드오프

### 2026-10-08-h-dqn.md — h-DQN

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 5 | 사람의 문제 풀이 비유가 좋음 |
| 전문성·깊이 | 3 | 계층 RL의 핵심 난제(상위 정책의 non-stationarity) 부재 |
| 구성·가독성 | 4 | 독자에게 의미 없는 세미나 파일명 메모가 본문에 있음 |
| 코드 품질 | 3 | 옛 Gym API |

**주요 변경**
- Gymnasium API로 갱신, 세미나 파일명 메모를 HTML 주석으로 이동
- non-stationarity와 HIRO의 해결 방식 보충, arXiv 링크

**추가 제안**
- 제목 후보: 「h-DQN 논문 리뷰: 목표를 나눠 푸는 계층적 강화학습」 / 「Meta-controller와 Controller로 보는 Hierarchical RL」 / 「Montezuma's Revenge와 h-DQN」
- 다음 주제: 「HIRO」, 「Option-Critic」
- 꼬리질문: ① 두 층의 시간 단위가 다른 이유? (글에 있음) ② 목표를 사람이 정해야 하는 한계? (글에 있음) ③ 동시 학습 시 문제? (반영)

### 2026-08-29-icm-curiosity.md — ICM

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 5 | 세 종류의 관측 요소 설명이 훌륭 |
| 전문성·깊이 | 4 | 직접 정리한 한계 절이 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | detach 위치의 의미 설명 부재 |

**주요 변경**
- 사용 예, `r_int.detach()`와 forward 입력 detach의 이유(특징 붕괴 방지) 설명, 공식 구현 확인 `[확인 필요]`, arXiv 링크

**추가 제안**
- 제목 후보: 「ICM 논문 리뷰: 호기심을 보상으로 쓰는 탐험」 / 「Curiosity-driven Exploration과 Noisy TV 문제」 / 「Inverse dynamics로 배우는 행동 관련 특징」
- 다음 주제: 「Large-Scale Study of Curiosity (Burda 2019)」, 「Plan2Explore」
- 꼬리질문: ① inverse model이 노이즈를 거르는 원리? (글에 있음) ② ICM이 동전 던지기에 약한 이유? (글에 있음) ③ 픽셀 예측이 나쁜 이유? (글에 있음)

### 2026-07-06-imbalanced-data.md — 불균형 데이터 다루기

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 3 | LDAM 코드가 정규화된(cosine) logit 전제라는 점 누락 → 일반 logit에 쓰면 loss 폭발 |
| 입문자 이해도 | 5 | 좋음 |
| 전문성·깊이 | 4 | DRW, decoupling까지 다룸 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | SMOTE 기본 인자에 rng 객체(상태 공유), 사용 예 없음 |

**주요 변경**
- **LDAM 전제(NormedLinear, s=30) 경고 + 코사인 logit 14.8 vs 일반 logit 99.5 비교**(실행)
- SMOTE 기본 인자 수정, sampler 동작 확인(약 1:1:1, tail 샘플을 평균 32번 반복 → 과적합 위험), margin 값 확인, 공식 코드 링크

**추가 제안**
- 제목 후보: 「불균형 데이터 완벽 정리: Re-sampling부터 LDAM까지」 / 「Long-tail 분류에서 tail 클래스를 살리는 방법」 / 「Focal Loss, LDAM-DRW, Decoupling 비교」
- 다음 주제: 「Decoupling(cRT, τ-normalization)」, 「Logit Adjustment」
- 꼬리질문: ① 불균형 데이터에서 accuracy 대신? → balanced accuracy, macro-F1 ② DRW를 늦게 거는 이유? (글에 있음) ③ over-sampling의 위험? (반영)

### 2026-09-23-implicit-q-learning.md — IQL

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 4 | expectile 설명이 좋음 |
| 전문성·깊이 | 5 | V를 따로 두는 이유까지 설명 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | expectile 동작 확인 예 없음 |

**주요 변경**
- **τ에 따라 V가 평균(4.0)에서 최댓값(10)으로 옮겨 가되 10을 넘지 않는 것을 확인**(실행) → "데이터 밖 값이 target에 못 끼어든다"는 핵심 설명
- arXiv 링크

**추가 제안**
- 제목 후보: 「IQL 논문 리뷰: 데이터 밖 행동을 묻지 않는 Offline RL」 / 「Expectile Regression으로 이해하는 Implicit Q-Learning」 / 「IQL vs CQL」
- 다음 주제: 「CQL 정리」, 「IDQL: IQL + Diffusion Policy」
- 꼬리질문: ① τ를 1로 두면? → 운 좋은 추정에 끌림 (반영) ② V를 따로 두는 이유? (글에 있음) ③ AWR의 β 역할? (글에 있음)

### 2026-08-08-neural-style-transfer.md — Neural Style Transfer

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 층 인덱스까지 정확 |
| 입문자 이해도 | 5 | content/style 설명이 좋음 |
| 전문성·깊이 | 4 | 후속 연구 연결이 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 입력 정규화·L-BFGS 조기 종료 같은 실행 함정 미언급, 결과 이미지 없음 |

**주요 변경**
- 실행 조건(ImageNet 정규화, 가중치 다운로드) 안내
- **L-BFGS가 기울기가 작으면 바로 멈추는 함정** — 사전학습 가중치 없이 구조를 시험할 때 실제로 이미지가 안 바뀌는 현상을 재현함, β를 키우면 해결
- 결과 이미지 빈칸, PyTorch 튜토리얼 링크

**추가 제안**
- 제목 후보: 「Neural Style Transfer 논문 리뷰와 PyTorch 구현」 / 「Gram Matrix로 화풍을 표현하는 법」 / 「CNN으로 content와 style 분리하기」
- 다음 주제: 「AdaIN 정리」, 「Perceptual loss와 실시간 style transfer」
- 꼬리질문: ① Gram matrix가 style을 나타내는 이유? → 위치 정보 제거 (글에 있음) ② 왜 이미지를 최적화하나? (글에 있음) ③ 속도 문제의 해결? → feed-forward, AdaIN (글에 있음)

### 2026-09-11-never-give-up.md — NGU

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 4 | 구성 요소별 설명이 좋음 |
| 전문성·깊이 | 4 | 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 사용 예 없음 |

**주요 변경**
- episodic reward 비교(재방문 ≈1.04 vs 새 상태 ≈15.4), α 클리핑 예 → `max(α, 1)`이 "포기하지 않는" 이유임을 코드로 보여 줌
- arXiv 링크

**추가 제안**
- 제목 후보: 「Never Give Up 논문 리뷰: Episodic과 Life-long 새로움의 결합」 / 「NGU의 intrinsic reward 완전 분석」 / 「k-NN episodic memory로 탐험하기」
- 다음 주제: 「R2D2」, 「BYOL-Explore」
- 꼬리질문: ① episodic과 life-long을 곱하는 이유? (반영) ② embedding을 inverse dynamics로 학습하는 이유? (글에 있음) ③ 여러 β 정책을 함께 학습하는 이점? (글에 있음)

### 2026-07-31-patchcore.md — PatchCore

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 5 | 단계별 설명이 명확 |
| 전문성·깊이 | 4 | coreset 설명 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 사용 예 없음 |

**주요 변경**
- **장난감 예**: 1% coreset(156개)으로도 심어 둔 이상 패치 (10, 5)를 정확히 찾음(실행)
- FAISS 언급, anomalib 링크, arXiv 링크

**추가 제안**
- 제목 후보: 「PatchCore 논문 리뷰: 정상 데이터만으로 결함 찾기」 / 「산업 이상 탐지의 표준 baseline, PatchCore」 / 「Memory Bank와 Coreset으로 보는 Anomaly Detection」
- 다음 주제: 「PaDiM」, 「anomalib으로 MVTec AD 돌려 보기 (실측)」
- 꼬리질문: ① 왜 중간 층 특징? (글에 있음) ② coreset이 random보다 나은 이유? (글에 있음) ③ 이미지 점수에 max를 쓰는 이유? (글에 있음)

### 2026-07-17-poolformer.md — PoolFormer

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 비교 대상 "ViT-B 계열 약 80%"가 모호 |
| 입문자 이해도 | 5 | MetaFormer 설명이 명확 |
| 전문성·깊이 | 4 | identity ablation 소개가 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | 사용 예 없음 |

**주요 변경**
- 비교 표를 DeiT-B(81.8%, 86M)로 구체화
- token mixer 파라미터 0개 확인(실행), "attention 불필요"로 확대 해석하지 말라는 박스, arXiv 링크

**추가 제안**
- 제목 후보: 「PoolFormer 논문 리뷰: Attention 대신 Pooling이어도 될까」 / 「MetaFormer: Transformer 성능의 진짜 원인」 / 「Token Mixer 관점에서 보는 ViT, MLP-Mixer, PoolFormer」
- 다음 주제: 「MetaFormer Baselines(후속 연구)」, 「ConvNeXt」
- 꼬리질문: ① AvgPool에서 x를 빼는 이유? (글에 있음) ② identity mixer로도 74%가 나오는 의미? (글에 있음) ③ 언어 모델에도 적용될까? (반영)

### 2026-10-04-primal.md — PRIMAL

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | 보상 구체 값 없음 |
| 입문자 이해도 | 5 | 문제 설명이 친절 |
| 전문성·깊이 | 4 | blocking penalty 설명 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 원래 경로가 없는 경우 예외 |

**주요 변경**
- 보상 값(−0.3/−0.5/−2/+20)과 상대 크기의 설계 이유 (`[확인 필요]`)
- blocking 판정 예외 처리, 학습 시에만 쓰는 비싼 연산이라는 맥락, PRIMAL2 언급, arXiv 링크

**추가 제안**
- 제목 후보: 「PRIMAL 논문 리뷰: 강화학습 + 모방학습으로 수백 대 로봇 경로 찾기」 / 「분산 MAPF와 blocking penalty」 / 「CBS vs PRIMAL: 최적성과 확장성」
- 다음 주제: 「PRIMAL2(lifelong MAPF)」, 「SCRIMP 등 통신 기반 MAPF」
- 꼬리질문: ① 왜 분산 정책인가? (글에 있음) ② 이기적 행동을 막는 방법? (글에 있음) ③ 좁은 통로에서 약한 이유? (글에 있음)

### 2026-08-30-random-network-distillation.md — RND

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 4 | advantage 결합 계수(논문은 2:1) 미기재 |
| 입문자 이해도 | 5 | 오차 원인 네 가지 분석이 훌륭 |
| 전문성·깊이 | 5 | 정규화 디테일까지 다룸 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 동작 확인 예 없음 |

**주요 변경**
- **반복해서 본 관측의 보상 ≈5e-7 vs 처음 보는 관측 ≈0.54**(실행), target 고정 확인
- A = 2A_E + A_I 명시, arXiv 링크

**추가 제안**
- 제목 후보: 「RND 논문 리뷰: 무작위 네트워크로 새로움 측정하기」 / 「Montezuma's Revenge에서 사람을 넘은 RND」 / 「Random Network Distillation과 Noisy TV」
- 다음 주제: 「Randomized Prior Functions」, 「NGU에서 RND의 역할」
- 꼬리질문: ① target을 고정하는 이유? (글에 있음) ② 관측 정규화가 필수인 이유? (글에 있음) ③ intrinsic을 non-episodic으로 두는 이유? (글에 있음)

### 2026-09-15-revisiting-intrinsic-reward.md — Revisiting Intrinsic Reward

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확, 결론의 적용 범위도 명시 |
| 입문자 이해도 | 5 | 좋음 |
| 전문성·깊이 | 4 | shuffle 실험 설명이 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 4 | 사용 예 없음 |

**주요 변경**
- count vs visit 숫자 비교, 픽셀 환경에서는 정확한 count가 불가능하다는 박스, OpenReview 링크 `[확인 필요]`

**추가 제안**
- 제목 후보: 「절차적 생성 환경에서는 episodic 보상이면 충분하다」 / 「Life-long vs Episodic intrinsic reward 비교 연구 정리」 / 「탐험 보상의 구성 요소를 분리해 보기」
- 다음 주제: 「E3B: 연속 상태에서의 episodic bonus」, 「Procgen 벤치마크」
- 꼬리질문: ① shuffle 실험이 의미하는 것? (글에 있음) ② Atari에서도 같은 결론일까? → 아님 (글에 있음) ③ count와 visit의 차이? (반영)

### 2026-09-06-ride.md — RIDE

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 5 | singleton vs 절차적 생성 구분이 좋음 |
| 전문성·깊이 | 4 | 좋음 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 사용 예 없음 |

**주요 변경**
- 걷기(0.1) vs 문 열기(3.0), 반복 시 3.0 → 2.12 → 1.73 감소 예(실행)
- Farama Minigrid 안내, arXiv 링크

**추가 제안**
- 제목 후보: 「RIDE 논문 리뷰: 상태를 크게 바꾸는 행동에 보상 주기」 / 「Impact-driven exploration과 절차적 생성 환경」 / 「MiniGrid에서의 탐험 보상 비교」
- 다음 주제: 「BeBold」, 「AMIGo(teacher-student 목표 생성)」
- 꼬리질문: ① 분모의 episodic count가 없으면? → 왕복 꼼수 (반영) ② ICM과의 차이? → 오차가 아니라 변화량 (글에 있음) ③ 보상이 0으로 수렴하지 않는 이유? (글에 있음)

### 2026-09-29-thought-cloning.md — Thought Cloning

| 항목 | 점수(5점) | 한 줄 코멘트 |
|---|---|---|
| 정확성 | 5 | 정확 |
| 입문자 이해도 | 5 | 동기 설명이 좋음 |
| 전문성·깊이 | 3 | 해석 가능성 주장의 한계(faithfulness) 부재 |
| 구성·가독성 | 5 | 좋음 |
| 코드 품질 | 3 | 의사 코드임을 미표기 |

**주요 변경**
- 의사 코드 표기, chain-of-thought와의 연결과 faithfulness 문제 details, arXiv 링크

**추가 제안**
- 제목 후보: 「Thought Cloning 논문 리뷰: 생각까지 모방하는 에이전트」 / 「언어로 생각하는 강화학습 에이전트」 / 「Behavior Cloning에서 Thought Cloning으로」
- 다음 주제: 「ReAct: LLM 에이전트의 추론과 행동」, 「VPT: 영상으로 배우는 Minecraft 에이전트」
- 꼬리질문: ① 생각 데이터를 어떻게 만들었나? → 합성 (글에 있음) ② precrime intervention의 한계? (반영) ③ BC보다 일반화가 좋은 이유? → 하위 목표 단위의 추상화


---

## 2차 정리 (2026-10-17): 주석 삭제와 확인 항목 처리

- 글 안의 HTML 주석(`<!-- 수정: … -->`, `<!-- 확인 필요 … -->`) 148개를 모두 지웠습니다.
- `[확인 필요]` 표시는 1차 출처로 확인해 모두 사실로 바꾸거나, 확인되지 않은 내용은 지웠습니다.

| 글 | 확인 결과 | 출처 |
|---|---|---|
| audio-classification | window 150은 **샘플 수**(n_fft = win_length = 150, hop 75), 데이터는 PhysioNet/CinC 2016(2,000Hz) / Focal loss에서 정상(y=0)에 α, 비정상(y=1)에 1−α → α=0.2는 비정상에 0.8의 가중치 / 논문은 Li, Yin, Ma, Zhang & Liu, *Information* 12(2), 54, 2021 | MDPI 원문, Crossref |
| edge-ai-hardware | TinyTL의 6.5배는 전체 메모리 절감률(채널 수 아님) → 수치 삭제 / CMSIS-NN **4.0.0**(2022.11)에서 레거시 q7 함수 제거, 계속 쓰려면 CMSIS 5.9.0 | arXiv 초록, CMSIS-NN Releases |
| object-detection | YOLO v1 논문에는 threshold 값이 없음("confidence로 걸러낸다"만 있음), NMS가 mAP를 2~3% 올린다는 내용 추가 | 논문 원문 |
| pytorch-training-toolkit | TensorBoard는 1.1 실험 기능 → 1.2(2019.8) 정식 | 릴리스 보도, PR #21786 |
| textdiffuser | λ_char를 0~1로 실험, **0.01**이 최적(λ=0 대비 9.8% 향상) | 논문 원문 |
| rest-api-basics | Idempotency-Key 초안은 07판(2025.10)을 마지막으로 2026년 4월 만료, RFC 아님 | IETF Datatracker |
| adaptdiffuser | ChatGPT로 미로 레이아웃 생성은 **논문 부록 G**의 내용 | 논문 원문 |
| agent57 | 논문에 DQN·Rainbow 숫자는 없음 → 해당 행 삭제, MuZero 51 / NGU 51 / R2D2 52 / R2D2+bandit 54 / Agent57 57로 교체 | 논문 본문·Table 1 |
| asip-posit | 예전 Synopsys 제품 페이지는 현재 MIPS 사이트로 연결됨 → 링크 교체 | 리다이렉트 확인 |
| deformable-convolution | 29.4 → 33.1은 COCO test-dev, ResNet-101 Faster R-CNN의 mAP@[0.5:0.95] (R-FCN은 30.8 → 34.5) | 논문 Table 5 |
| icm-curiosity | 공식 구현은 forward loss도 encoder까지 흘림(stop_gradient 없음) → 본문에 차이와 이유 설명 추가 | pathak22/noreward-rl `model.py` |
| primal | Table I: 이동 −0.3, 목표에서 정지 0, 목표 밖 정지 −0.5, 충돌 −2, 완료 +20 → 표로 추가 | 논문 원문 |
| revisiting-intrinsic-reward | OpenReview 링크 ID(`j3GK3_xZydY`)가 맞음 | ICLR 2023 포스터 페이지 |

### 새로 채운 빈칸

- **github-actions-pages**: 이 저장소의 실제 Actions 기록(실패 4회)을 확인해 "실제로 겪은 실패 사례" 절을 썼습니다. 네 번 모두 `Test site`(htmlproofer) 단계에서 실패했고, 원인은 ① 존재하지 않는 이미지 경로, ② `http://` 링크(html-proofer의 `enforce_https` 기본값이 true)였으며, 각각 해당 수정 커밋에서 바로 통과했습니다.
- **python-virtual-environment**: 출력 예시를 확인한 환경(Windows 11, Python 3.14)을 적었습니다.
- **hello-blog**: 블로그가 다루는 주제와, 지금 글들이 실제로 따르는 작성 원칙(출처 표기, 실행 결과 첨부, 용어 정의)을 적었습니다. 전공·소속 같은 개인 정보는 넣지 않았으니 원하면 덧붙이세요.

### 아직 남은 빈칸 (제가 채울 수 없는 것)

| 글 | 빈칸 | 이유 |
|---|---|---|
| rest-api-basics | API 계약을 정하며 겪은 경험 | 본인 경험이라 지어낼 수 없음 |
| acer | 세미나에서 받은 질문·구현 경험 | 본인 경험 |
| asip-posit | 직접 설계 vs ASIP 생성 덧셈기의 면적·타이밍 비교 | 본인 실험 결과 |
| latent-consistency-model | 실행 결과 이미지, GPU, 생성 시간 | GPU와 수 GB 모델 다운로드 필요 |
| neural-style-transfer | 결과 이미지 세 장과 설정 | VGG 가중치(약 550MB)와 content·style 이미지 다운로드 필요 |

## 검증 방법과 한계

- **실행한 것**: 각 글의 Python 코드 블록을 이 PC(Python 3.14, PyTorch 2.13 CPU, torchvision 0.28, NumPy 2.5, scikit-learn 1.9, SQLite 3.50.4)에서 실제로 돌렸고, 글에 넣은 실행 결과는 모두 이렇게 얻은 값입니다. 측정 시간(SQL 인덱스, MobileNet)도 이 PC의 실측값이며, 글에 환경 의존이라고 적었습니다.
- **실행하지 못한 것**: GPU나 대용량 가중치 다운로드가 필요한 코드(LCM `diffusers`, Style Transfer 사전학습 VGG, librosa 예제)는 구조만 확인했습니다. Diffuser·h-DQN·DAgger처럼 외부 환경이나 모델 객체를 가정한 코드는 가짜 객체로 실행 흐름만 확인했습니다.
- **정적 검사**: 68편 모두 front matter(YAML) 파싱, `description` 존재, `$$`·코드 펜스 짝, `<details markdown="1">` 짝, Liquid 태그(`{{`, `{%`) 충돌, `post_url` 대상 존재, 이미지 경로, 제어 문자, 표 안 수식의 `|`를 스크립트로 검사해 문제 없음을 확인했습니다.
- **Jekyll 빌드는 로컬에서 돌리지 못했습니다.** 이 PC에 Ruby가 없어서입니다. push하면 GitHub Actions의 빌드와 htmlproofer가 최종 확인합니다. 로컬에서 보려면 Ruby 설치 후 `bundle exec jekyll serve`를 실행하세요.
- 수정은 커밋하지 않았습니다. `git diff`로 글별 변경을 확인할 수 있습니다.
