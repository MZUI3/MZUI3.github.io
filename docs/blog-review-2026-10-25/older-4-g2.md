# 블로그 교정 보고서 (older-4-g2, 2026-10-25)

## 2026-06-04-github-actions-pages.md

### 수정 목록
수정 없음. (맞춤법·띄어쓰기·비문 기준으로 고칠 곳이 없었습니다. 사실 확인 결과에 따른 수정 1건은 아래 "내용 확인 결과"에 적었습니다.)

### 내용 확인 필요
- 도입부 "이 저장소는 ... `main` 브랜치의 변경을 GitHub Pages에 배포합니다": 이 저장소의 기본 브랜치는 `master`입니다(`origin/HEAD -> origin/master`). 워크플로는 `branches: [main, master]`라 둘 다 동작합니다. mermaid 다이어그램의 "main에 push"는 코드 블록이라 손대지 않았습니다.
- `fetch-depth: 0` 설명("기본으로 마지막 커밋 하나만"), `future: false` 기본값, html-proofer `enforce_https` 기본값, `id-token: write`가 필요한 이유.
- 워크플로 YAML 예시가 실제 `.github/workflows/pages-deploy.yml`과 맞는지(액션 버전, 단계 이름). 실제 파일의 `--ignore-urls` 옵션은 "핵심만 옮긴" 예시라 빠져 있습니다.

## 2026-06-02-cnn-architectures.md

### 수정 목록
수정 없음.

### 내용 확인 필요
- DenseNet-121 "약 8M 파라미터로, 약 25M인 ResNet-50과 비슷한 정확도".
- SENet 수치: ILSVRC 2017 top-5 error 2.251%, 약 25% 감소, SE-ResNet-50 3.86 → 3.87 GFLOPs(0.26%), top-1 error 약 1.5%p 감소, r = 16, NoSqueeze, 마지막 stage 포화.
- ShuffleNet 수치: MobileNet의 1×1 conv 비중 약 95%, ShuffleNet 2× vs MobileNet 1.0 약 3%p, AlexNet 대비 약 13배.
- RepVGG-A/B 경쟁 대상(ResNet-18/34/50, RegNet·ResNeXt), CVPR 2021.
- 코드 주석의 출력값: DenseBlock (1, 256, 56, 56), channel shuffle [0, 3, 1, 4, 2, 5], RepVGG 오차 약 6e-07.
- 교차 링크: `2026-05-28-resnet`, `2026-05-30-mobilenet`, `2026-06-13-lightweight-vision-transformer` 대상 파일이 있는지.

## 2026-05-30-mobilenet.md

### 수정 목록
수정 없음. (사실 확인에 따른 수정 1건은 아래에 적었습니다.)

### 내용 확인 필요
- "원 논문은 ReLU 대신 ... ReLU6를 썼습니다": MobileNetV1 논문 본문은 "batchnorm and ReLU"라고만 적은 것으로 기억됩니다. ReLU6는 MobileNetV2 논문과 TF 구현에 나옵니다.
- 표의 수치(VGG-16 71.5%/138M/15,300M, GoogLeNet 69.8%/6.8M/1,550M, MobileNet 70.6%/4.2M/569M), "1/32, 1/27", "약 1%p 하락", VGG-16·ResNet-152 크기.
- 코드 출력 `589824 67840 0.115`와 "약 1/9", "2배도 빨라지지 않았습니다"(60.1/34.8 ≈ 1.73배)가 맞는지.
- 교차 링크: `2026-06-02-cnn-architectures`, `2026-07-13-mobile-former`.

## 2026-05-28-resnet.md

### 수정 목록
수정 없음.

### 내용 확인 필요
- ILSVRC 2015 1위, 앙상블 top-5 error 3.57%, 152층·약 60M 파라미터.
- 본문 "bottleneck 본체 70,400개, projection shortcut 66,048개"와 코드 주석 70400이 맞는지.
- 도입부 `[MobileNetV2]({% post_url 2026-05-30-mobilenet %})`: V1 글로 연결되는데, 지금은 `2026-10-24-mobilenetv2.md`라는 V2 전용 글도 있습니다.

## 2026-05-25-cnn-basics.md

### 수정 목록
수정 없음.

### 내용 확인 필요
- Convolution 예시 출력 [4 9; 6 5], 파라미터 수 73,856, SimpleCNN 94,986, 출력 크기 표, ConvTranspose 16→32.
- 교차 링크: `2026-06-19-edge-ai-hardware`, `2026-06-10-attention-is-all-you-need`, `2026-06-02-cnn-architectures`, `2026-05-30-mobilenet`.
- 224×224×3 ≈ 15만, 가중치 1.5억.

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| 2026-06-04-github-actions-pages.md | 이 저장소의 배포 브랜치 `main` | 수정함 | 로컬 저장소: `git symbolic-ref refs/remotes/origin/HEAD` → `refs/remotes/origin/master`, 워크플로 `branches: [main, master]` (`.github/workflows/pages-deploy.yml`) | "`main` 브랜치의 변경을" → "`main`(또는 `master`) 브랜치의 변경을" (검토 단계에서 "이 저장소는"이 한 문장에 두 번 들어가던 표현을 워크플로 설정 `branches: [main, master]`에 맞춰 다듬음) |
| 2026-06-04-github-actions-pages.md | checkout 기본값은 커밋 하나 | 문제 없음 | "Only a single commit is fetched by default", fetch-depth Default: 1 — https://github.com/actions/checkout | - |
| 2026-06-04-github-actions-pages.md | Jekyll `future: false` 기본값 | 문제 없음 | 기본 설정 목록에 "future : false" — https://jekyllrb.com/docs/configuration/default/ | - |
| 2026-06-04-github-actions-pages.md | html-proofer `enforce_https` 기본 검사 | 문제 없음 | enforce_https 기본값 true, "Fails a link if it's not marked as https" — https://github.com/gjtorikian/html-proofer | - |
| 2026-06-04-github-actions-pages.md | `id-token: write`가 deploy-pages OIDC에 필요 | 문제 없음 | "The id-token permission is necessary to request the OIDC JWT token" — https://github.com/actions/deploy-pages | - |
| 2026-06-04-github-actions-pages.md | YAML 예시와 실제 워크플로 일치 | 문제 없음 | 실제 파일의 액션 버전(checkout@v7, configure-pages@v6, upload-pages-artifact@v5, deploy-pages@v5)과 단계 이름이 같음. `--ignore-urls`는 생략(본문이 "핵심만 옮긴 것"이라고 밝힘) | - |
| 2026-06-02-cnn-architectures.md | DenseNet-121 8M ≈ ResNet-50 25M 정확도 | 문제 없음 | DenseNet 표 3: DenseNet-121 top-1 25.02%. 논문은 "a DenseNet that requires as much computation as a ResNet-50 performs on par with a ResNet-101"처럼 대략적인 비교를 함. 흔히 쓰는 근사 표현이라 그대로 둠 — https://arxiv.org/abs/1608.06993 | - |
| 2026-06-02-cnn-architectures.md | SENet 수치 | 문제 없음 | "reduced the top-5 error to 2.251%, ... relative improvement of ∼25%", "∼3.87 GFLOPs, corresponding to a 0.26% relative increase", 표 2: ResNet-50 24.80 → SE 23.29(1.51), ResNet-101 23.17 — https://arxiv.org/abs/1709.01507 | - |
| 2026-06-02-cnn-architectures.md | ShuffleNet 수치 | 문제 없음 | 표: 1.0 MobileNet-224 569 MFLOPs 29.4% vs ShuffleNet 2× (g=3) 524 MFLOPs 26.3%, Δ3.1. "∼13× actual speedup over AlexNet". 95%는 MobileNet 논문의 "spends 95% of it's computation time in 1 × 1 convolutions" — https://arxiv.org/abs/1707.01083 , https://arxiv.org/abs/1704.04861 | - |
| 2026-06-02-cnn-architectures.md | RepVGG-A/B 경쟁 대상, CVPR 2021 | 문제 없음 | "RepVGG-A to compete against other lightweight and middleweight models including ResNet-18/34/50, and RepVGG-B against the high-performance ones"(비교 표에 ResNeXt, RegNet 포함). arXiv Comments "CVPR 2021" — https://arxiv.org/abs/2101.03697 | - |
| 2026-06-02-cnn-architectures.md | 코드 주석 출력값 | 문제 없음 | 로컬 PyTorch 2.13.0+cpu로 재실행: channel shuffle [0,3,1,4,2,5], RepVGG allclose True / 최대 오차 5.96e-07. DenseBlock 64+6×32=256은 계산으로 확인 | - |
| 2026-06-02-cnn-architectures.md | 교차 링크 대상 | 문제 없음 | `_posts/`에 2026-05-28-resnet.md, 2026-05-30-mobilenet.md, 2026-06-13-lightweight-vision-transformer.md 모두 있음 | - |
| 2026-05-30-mobilenet.md | "원 논문은 ReLU6를 썼다" | 수정함 | V1 논문 본문은 "Depthwise and Pointwise layers followed by batchnorm and ReLU"이고 ReLU6라는 말이 없음 — https://arxiv.org/abs/1704.04861 . ReLU6는 V2 논문 "We use ReLU6 as the non-linearity because of its robustness when used with low-precision computation" — https://arxiv.org/abs/1801.04381 , 공식 TF 구현 `activation_fn=tf.nn.relu6` — https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py | "(원 논문은 ReLU 대신 저정밀도 연산에 유리하도록 ... **ReLU6**를 썼습니다.)" → "(원 논문은 ReLU라고만 적었지만, 공식 TensorFlow 구현과 MobileNetV2는 저정밀도 연산에 유리하도록 ... **ReLU6**를 썼습니다.)" |
| 2026-05-30-mobilenet.md | 결과 표, 1/32·1/27, 약 1%p | 문제 없음 | 표 8: MobileNet 70.6%/569/4.2, GoogleNet 69.8%/1550/6.8, VGG 16 71.5%/15300/138. "32 times smaller and 27 times less compute intensive". 표 4: Conv MobileNet 71.7% vs 70.6%(1.1%p) — https://arxiv.org/abs/1704.04861 | - |
| 2026-05-30-mobilenet.md | 파라미터 출력과 본문 | 문제 없음 | 256·256·9=589,824, 256·9+256·256=67,840, 비율 0.115 ≈ 1/8.7. 60.1/34.8 ≈ 1.73배라 "2배도 빨라지지 않았다"와 맞음 | - |
| 2026-05-30-mobilenet.md | 교차 링크 대상 | 문제 없음 | 2026-06-02-cnn-architectures.md, 2026-07-13-mobile-former.md 있음 | - |
| 2026-05-28-resnet.md | ILSVRC 2015, 3.57%, 152층 | 문제 없음 | "An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task." — https://arxiv.org/abs/1512.03385 | - |
| 2026-05-28-resnet.md | 70,400 / 66,048 | 문제 없음 | 로컬 재계산: bottleneck(conv 3개 + BN 3개) 70,400개, 1×1 conv 256→256 + BN 66,048개 | - |
| 2026-05-28-resnet.md | MobileNetV2 링크가 V1 글을 가리킴 | 문제 없음 | 2026-05-30-mobilenet.md의 7절이 MobileNetV2(inverted residual)를 다루므로 링크 의미는 맞음. 2026-10-24-mobilenetv2.md로 바꿀지는 글쓴이가 판단할 일이라 그대로 둠 | - |
| 2026-05-25-cnn-basics.md | 예시·코드 수치 | 문제 없음 | 로컬 PyTorch로 재실행: conv2d [[4,9],[6,5]], 73,856, SimpleCNN 94,986, ConvTranspose 16→32, 출력 크기 표 30/32/16/112 | - |
| 2026-05-25-cnn-basics.md | 교차 링크 대상 | 문제 없음 | 2026-06-19-edge-ai-hardware.md, 2026-06-10-attention-is-all-you-need.md, 2026-06-02-cnn-architectures.md, 2026-05-30-mobilenet.md 모두 있음 | - |
