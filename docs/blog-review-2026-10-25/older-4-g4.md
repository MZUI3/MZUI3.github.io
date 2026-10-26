# 블로그 교정·내용 확인 보고서 (older-4-g4, 2026-10-25)

## 2026-05-13-activation-functions.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 9절 표의 ReLU 행 "빠르고 기울기 소실이 없음"이 본문과 맞지 않음. 본문은 ReLU가 기울기 소실을 "크게 완화"한다고 하고, 6절 실험에서는 PyTorch 기본 초기화로 20층을 쌓으면 기울기가 3.35e-10까지 줄어든다고 보여 줌.
- 6절 "He 초기화를 쓰면 기울기가 수억 배 커집니다": 실제 비율은 5.93e-02 / 3.35e-10 ≈ 1.77×10^8(약 1.8억 배).
- 코드 출력과 주석 값(20층 첫 층 기울기, SELU 통계, GELU 값, STE 출력, softmax 주석)과 "PyTorch 2.13" 실행 환경 표기.
- GELU 최솟값 "약 -0.17, z ≈ -0.75", ELU의 0에서 미분 연속 설명, SELU 상수 λ ≈ 1.0507, α ≈ 1.6733.
- `{% post_url %}` 대상(2026-05-08-neural-network-backpropagation, 2026-06-08-rnn-lstm-gru, 2026-06-17-binary-neural-networks): `_posts/`에 모두 있음.

## 2026-05-11-rest-api-basics.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- Idempotency-Key 초안 상태 "2025년 10월의 07판을 마지막으로 2026년 4월에 만료되었고 RFC로 발행되지는 않았습니다(2026년 10월 확인)": 최신 상태 확인 필요.
- 401에 `WWW-Authenticate` 헤더, 401 대 403 구분, DELETE·PUT 멱등성, 201과 `Location` 헤더.
- RFC 9457이 RFC 7807을 대체한다는 설명, `application/problem+json`.

## 2026-05-08-neural-network-backpropagation.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 손으로 계산한 forward, δ, 기울기, 갱신 후 가중치 표, 갱신 후 y = 0.6757, E = 0.02515를 다시 계산해 확인할 필요가 있음.
- "w1과 w5 기준 약 1/8", "역전파로 구한 -0.00371097과 소수점 8자리까지 같습니다".
- `{% post_url %}` 대상(2026-05-05-ml-overview, 2026-05-13-activation-functions): 모두 있음.

## 2026-05-05-ml-overview.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 경사 하강법 코드 주석 `W=1.987, b=0.992`와 "학습률을 0.1로 올리면 발산" 설명.
- K-Means 출력과 "scikit-learn 1.2 이상, 1.9에서 실행" 표기. `n_init="auto"`는 1.2에서 추가됨.
- "scikit-learn은 기본으로 k-means++ 초기화를 쓰고, `n_init`만큼 여러 번 돌려": 틀린 말은 아님. 다만 기본값 `n_init="auto"`에 k-means++를 쓰면 실제로는 1회만 실행함.
- DQN(DeepMind, 2015, Nature), AlphaZero·MuZero를 model-based로, PPO를 on-policy로 분류한 부분.
- 시리즈 목차와 본문의 `{% post_url %}` 20개: `_posts/`에 모두 있음.

## 2026-05-03-python-virtual-environment.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- `venv`가 Python 3.3에 추가됐다는 설명, 활성화가 PATH 맨 앞에 Scripts/bin을 넣는다는 설명, 활성화는 필수가 아니라는 설명, 3.13부터 `.gitignore`를 자동 생성한다는 설명, 가상환경은 옮길 수 없다는 설명.
- `pip freeze` 출력 버전(certifi 2026.7.22, charset-normalizer 3.5.2, idna 3.20, requests 2.34.2, urllib3 2.8.0)이 실제로 있는지.

## 2026-05-01-hello-blog.md

### 수정 목록

수정 없음.

### 내용 확인 필요

없음. (개인 소개 글이라 교정만 함)

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
| --- | --- | --- | --- | --- |
| 2026-05-13-activation-functions.md | 9절 표 ReLU "기울기 소실이 없음" | 수정함 | 글 안에서 서로 맞지 않음. 본문에는 "Gradient Vanishing을 크게 **완화**"라고 되어 있고, 6절 실험 출력에서는 기본 초기화 ReLU의 첫 층 기울기가 3.35e-10임. 해당 코드를 PyTorch 2.13.0+cpu로 다시 실행해 같은 출력을 확인함. | "빠르고 기울기 소실이 없음" → "빠르고 기울기 소실이 적음" |
| 2026-05-13-activation-functions.md | "수억 배" | 문제 없음 | 5.93e-02 / 3.35e-10 = 1.77×10^8(약 1.8억 배). 반올림하면 억 단위라 표현 범위 안이라고 봄. | - |
| 2026-05-13-activation-functions.md | 코드 출력과 주석, PyTorch 2.13 | 문제 없음 | 이 PC(torch 2.13.0+cpu)에서 다섯 코드 블록을 다시 실행함. 7.07e-19 / 2.85e-07 / 3.35e-10 / 5.93e-02, SELU 표준편차 0.979, GELU·STE 텐서, softmax [0.090 0.245 0.665]가 모두 일치함. | - |
| 2026-05-13-activation-functions.md | GELU 최솟값, SELU 상수 | 문제 없음 | z·Φ(z)를 수치로 구하면 z = -0.7518에서 최솟값 -0.16997임. PyTorch 문서는 SELU 상수를 "α=1.6732632423543772848170429916717 and scale=1.0507009873554804934193349852946"로 적음. https://docs.pytorch.org/docs/2.14/generated/torch.nn.SELU.html | - |
| 2026-05-11-rest-api-basics.md | Idempotency-Key 초안 상태 | 수정함 | Datatracker 기록: -07은 2025-10-15에 올라와 2026-04-18에 "Document has expired"가 됨. 이후 "draft-ietf-httpapi-idempotency-key-header-08"이 2026-10-22에 올라왔고 현재 "Active Internet-Draft"로 2027-04-25에 만료됨. RFC는 아님. "07판을 마지막으로"는 이제 사실이 아님. https://datatracker.ietf.org/doc/draft-ietf-httpapi-idempotency-key-header/history/ | "2025년 10월의 07판을 마지막으로 2026년 4월에 만료되었고 RFC로 발행되지는 않았습니다" → "2025년 10월의 07판이 2026년 4월에 만료된 뒤 2026년 10월 22일에 08판이 새로 올라왔고, 아직 RFC로 발행되지는 않았습니다" |
| 2026-05-11-rest-api-basics.md | 401/403, WWW-Authenticate, 멱등성, 201 Location | 문제 없음 | RFC 9110 §15.5.2: "The server generating a 401 response MUST send a WWW-Authenticate header field." §15.5.4: 403은 "the server understood the request but refuses to fulfill it". PUT·DELETE·안전한 메서드는 §9.2.2에서 멱등으로 정의됨. https://www.rfc-editor.org/rfc/rfc9110.html | - |
| 2026-05-11-rest-api-basics.md | RFC 9457이 RFC 7807 대체, application/problem+json | 문제 없음 | RFC 9457 헤더에 "Obsoletes: 7807"이 있고, 미디어 타입은 application/problem+json임. https://www.rfc-editor.org/rfc/rfc9457 | - |
| 2026-05-08-neural-network-backpropagation.md | 손 계산 수치 전체 | 문제 없음 | NumPy로 다시 계산함. h = (0.55971, 0.56464), y = 0.67220, E = 0.025947, δo = -0.050195, ∂E/∂w5·w6 = -0.028095 / -0.028342, ∂E/∂w1..w4 = -0.00371097 / -0.00074219 / -0.00308479 / -0.00061696, 갱신 후 가중치 0.20186 / 0.40037 / 0.30154 / 0.10031 / 0.61405 / 0.51417, 갱신 후 y = 0.67574, E = 0.025147. 표와 본문이 모두 일치하고 PyTorch autograd 출력도 같음. | - |
| 2026-05-08-neural-network-backpropagation.md | "약 1/8", "소수점 8자리까지 같음" | 문제 없음 | ∂E/∂w1 / ∂E/∂w5 = 0.132(≈ 1/7.6). 역전파 값을 소수점 8자리로 쓰면 -0.00371097이라 수치 미분 주석과 같음. | - |
| 2026-05-05-ml-overview.md | GD 결과, lr 0.1에서 발산 | 문제 없음 | 코드를 다시 실행함. lr = 0.01에서 W = 1.987, b = 0.992이고, lr = 0.1에서는 nan으로 발산함. | - |
| 2026-05-05-ml-overview.md | K-Means 출력, scikit-learn 1.2 이상·1.9 | 문제 없음 | 이 PC(scikit-learn 1.9.0)에서 다시 실행해 중심점 3개와 [51 49 50]이 같게 나옴. `n_init="auto"`는 1.2에서 도입됨. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html | - |
| 2026-05-05-ml-overview.md | "n_init만큼 여러 번 돌려" | 문제 없음 | `n_init` 값만큼 실행하는 것은 맞음. 기본 'auto'에 k-means++를 쓰면 1회만 실행한다는 보충 정도라 수정하지 않음. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html | - |
| 2026-05-05-ml-overview.md | DQN 2015, RL 분류 | 문제 없음 | Mnih et al., "Human-level control through deep reinforcement learning," Nature 518, 2015. PPO는 on-policy, DQN·Q-learning은 off-policy로 보는 것이 표준 분류임. https://www.nature.com/articles/nature14236 | - |
| 2026-05-03-python-virtual-environment.md | venv 동작 설명 | 문제 없음 | Python 문서: "Added in version 3.3."; 활성화는 바이너리 폴더를 "prepend that directory to your PATH"; "You don't specifically *need* to activate"; "Changed in version 3.13: venv now creates a .gitignore file for Git by default."; "environments are inherently non-portable"; `sys.prefix != sys.base_prefix`로 판별. https://docs.python.org/3/library/venv.html | - |
| 2026-05-03-python-virtual-environment.md | pip freeze 출력 버전 | 문제 없음 | PyPI에서 확인함. requests 최신 2.34.2, certifi 2026.7.22(2026-07-22 릴리스), charset-normalizer 3.5.2(2026-09-30), idna 3.20(2026-09-17), urllib3 2.8.0(2026-09-15). 모두 존재함. https://pypi.org/project/requests/ , https://pypi.org/project/certifi/ , https://pypi.org/project/idna/ , https://pypi.org/project/urllib3/ , https://pypi.org/project/charset-normalizer/ | - |
