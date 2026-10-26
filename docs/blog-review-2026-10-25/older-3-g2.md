# 블로그 교정 보고서: older-3-g2

대상 5개 파일(_posts/). 교정 원칙: 맞춤법·띄어쓰기·문장부호·비문·어색한 표현·글 안의 표기 통일만 고치고, 내용·수치·순서·소제목·말투·코드 블록·인라인 코드·수식·링크·Liquid·front matter·기술 용어·고유명사는 건드리지 않았습니다. 애매한 것은 고치지 않았고, 내용이 의심스러운 부분은 "내용 확인 필요"에 적은 뒤 2단계에서 원 논문으로 확인했습니다. 모든 `{% post_url %}` 대상 파일이 `_posts/`에 있음을 확인했습니다.

## 2026-07-18-adabin.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 4절 "대부분의 층에서 이진값이 거의 모두 양수인 경우가 많았습니다": 원 논문 표현과 맞는지 확인 필요
- 5절 CIFAR-10·ImageNet 수치(93.1%, ReCU +0.3%p, 격차 1.7%p, RBNN +0.4%p, VGG-small 92.3%·SLB +0.3%p, SiBNN +3.4%p·격차 2.7%p, 63.1%·IR-Net +5.0%p·ReCU +2.1%p, ResNet-34 ReCU +1.3%p, ReActNet 설정 66.4%·+0.9%p)
- Maxout 초기값(γ+ = 1, γ− = 0.25), 가중치 α_w·β_w 식(평균·표준편차), 절 이름 "Weight Equalization"

## 2026-07-16-poolformer.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| Vision Transformer가 잘되는 이유는 흔히 **Self-Attention** 덕분이라고 생각합니다. | Vision Transformer가 잘되는 것은 흔히 **Self-Attention** 덕분이라고 생각합니다. | 비문 |

### 내용 확인 필요

- 표의 PoolFormer-M48 82.5%·73M, DeiT-B 81.8%·86M
- Ablation: identity Token Mixer(S12) top-1 약 74%, 앞쪽 pooling + 뒤쪽 attention 혼합 구조
- 코드 주석의 블록 파라미터 수 33344 (내부 재계산)

## 2026-07-13-mobile-former.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 2절 "MobileNetV3 기반 inverted residual block": 논문이 인용한 블록이 MobileNetV2인지 V3인지
- 3절 Dynamic ReLU 계수를 "첫 번째 토큰 또는 토큰 평균"으로 만든다는 설명
- 4절 Head에서 local feature와 concat하는 토큰이 어느 토큰인지
- 2절 "토큰은 처음에 무작위로 초기화", 4절 블록 11개·토큰 6개×192차원, 26M ~ 508M FLOPs
- 3절 (4) "이미 계산된 출력을 재사용하면서 attention을 적용하기 때문에, 추가 메모리 사용량도 최소화됩니다."의 근거
- 5절 77.9%, MobileNetV3 대비 +1.3%p·연산량 17% 절감, RetinaNet·DETR 결과
- 3절 끝 수치(18,816 대 약 980만, 약 520배, 파라미터 9,432개·9,264개)

## 2026-07-10-qmix.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- 6절 팀 보상 구성(피해량 + 처치당 10점 + 전멸 시 200점, 최댓값 20으로 정규화)
- 6절 결과: VDN이 5m·8m에서 강하고 3m에서 약함, QMIX가 모든 맵에서 가장 좋음
- Ablation 표의 VDN-S "일부 맵(3m)에서 QMIX 이상"
- 맵 구성(3m, 5m, 8m, 2s3z, 3s5z, 1c3s5z)과 행동 공간

## 2026-07-07-deformable-convolution.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 그래서 주변 4개 픽셀 값을 **선형 보간(bilinear interpolation)**해서 값을 얻습니다. | 그래서 주변 4개 픽셀을 **선형 보간(bilinear interpolation)**해서 값을 얻습니다. | 어색한 표현(같은 말 반복) |

### 내용 확인 필요

- 5절 "ResNet-101 기준 파라미터는 약 0.75%만 늘었고"
- 5절 COCO test-dev Faster R-CNN 29.4 → 33.1(상대 약 13%), R-FCN 30.8 → 34.5, "논문 Table 5"
- 5절 "작은 물체나 배경에서는 좁게 모이는": 배경 영역의 샘플링 범위
- 3절 offset 층 0 초기화, 4절 γ = 0.1, 인용 박스의 파라미터 수 1,314개·576개 (내부 재계산)

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| adabin | 양수 이진값이 나타나는 층 | 수정함 | https://arxiv.org/abs/2208.08084 (3.3절: "almost all positive in very few layers") | "대부분의 층에서 이진값이 **거의 모두 양수**인 경우가 많았습니다" → "일부 층에서는 이진값이 **거의 모두 양수**였습니다" |
| adabin | CIFAR-10·ImageNet 수치 전부 | 문제 없음 | https://arxiv.org/abs/2208.08084 (4.1·4.2절, Table 2–4) | - |
| adabin | Maxout 초기값 1·0.25, α_w = ‖w−β_w‖₂/√n(표준편차), 3.1절 "Weight Equalization" | 문제 없음 | https://arxiv.org/abs/2208.08084 (식 8, 식 15) | - |
| poolformer | M48 82.5%·73M, DeiT-B 81.8%·86M | 문제 없음 | https://arxiv.org/abs/2111.11418 (Table 2) | - |
| poolformer | identity 74.3%, 혼합 구조(pooling 2단계 + attention 2단계 81.0%) | 문제 없음 | https://arxiv.org/abs/2111.11418 (Ablation, Hybrid stages) | - |
| poolformer | 블록 파라미터 33344 | 문제 없음 | 내부 재계산: GroupNorm 2×128 + 1×1 conv (64·256+256) + (256·64+64) = 33344 | - |
| mobile-former | inverted residual의 출처 | 수정함 | https://arxiv.org/abs/2108.05895 (3.1·3.2절: inverted bottleneck block [26] = Sandler et al., MobileNetV2) | "MobileNetV3 기반" → "MobileNetV2 기반" |
| mobile-former | Dynamic ReLU 계수 생성 | 수정함 | https://arxiv.org/abs/2108.05895 (3.2절: 두 MLP를 Former 출력의 첫 번째 global token에 적용) | "(첫 번째 토큰 또는 토큰 평균)" → "(첫 번째 토큰)" |
| mobile-former | Head에서 concat하는 토큰 | 수정함 | https://arxiv.org/abs/2108.05895 (3.3절: "concatenates with the first global token") | "**global token과 concat**" → "**첫 번째 global token과 concat**" |
| mobile-former | 토큰 무작위 초기화, 블록 11개, 토큰 6×192, 26M~508M | 문제 없음 | https://arxiv.org/abs/2108.05895 (3.1·3.3절) | - |
| mobile-former | 77.9%, +1.3%p, 17% 절감, RetinaNet +8.6 AP, DETR 확장 | 문제 없음 | https://arxiv.org/abs/2108.05895 (초록) | - |
| mobile-former | "이미 계산된 출력을 재사용 … 추가 메모리 사용량 최소화" | 확인 불가 | https://arxiv.org/abs/2108.05895 (본문에 메모리 관련 서술 없음, 연산량만 언급) | - |
| mobile-former | 18,816 / 약 980만 / 약 520배 / 9,432·9,264 | 문제 없음 | 내부 재계산: 3136×6, 3136², 3136/6 ≈ 523; 192·24+24+24·192+192 = 9432, 2×(192·24+24) = 9264 | - |
| qmix | 보상 구성과 정규화 | 문제 없음 | https://arxiv.org/abs/1803.11485 (6.1절) | - |
| qmix | VDN 5m·8m 강세·3m 약세, QMIX 모든 맵 최고 | 문제 없음 | https://arxiv.org/abs/1803.11485 (7.1절) | - |
| qmix | VDN-S가 3m에서 QMIX 이상 | 수정함 | https://arxiv.org/abs/1803.11485 (7.2절: QMIX는 모든 ablation보다 낫거나 비슷, 3m 같은 동종 맵에서는 비선형 분해가 꼭 필요하지 않음) | "일부 맵(3m)에서 QMIX 이상" → "일부 맵(3m)에서 QMIX와 비슷" |
| qmix | 맵 구성·행동 공간 | 문제 없음 | https://arxiv.org/abs/1803.11485 (6.1절) | - |
| deformable-convolution | 파라미터 약 0.75% 증가 | 수정함 | https://arxiv.org/abs/1703.06211 (Table 4: DeepLab 46.0M → 46.1M, Faster R-CNN 58.3M → 59.9M, R-FCN 47.1M → 49.5M; 0.75%라는 값은 없음) | "파라미터는 **약 0.75%**만 늘었고" → "파라미터 증가는 **작았고**(DeepLab 46.0M → 46.1M, 논문 Table 4)" |
| deformable-convolution | COCO test-dev 29.4 → 33.1(약 13%), 30.8 → 34.5, Table 5 | 문제 없음 | https://arxiv.org/abs/1703.06211 (Table 5, 4.3절 "11%, 13%, and 12% relatively higher") | - |
| deformable-convolution | 배경 영역의 샘플링 범위 | 수정함 | https://arxiv.org/abs/1703.06211 (4.2절, Table 2: 배경의 필터 크기는 중간 물체와 큰 물체 사이) | "작은 물체나 배경에서는 좁게 모이는 등" → "작은 물체에서는 좁게 모이며 배경에서는 비교적 넓게 퍼지는 등" |
| deformable-convolution | offset 0 초기화, γ = 0.1, 1,314·576 | 문제 없음 | https://arxiv.org/abs/1703.06211 (2.1·2.2절); 내부 재계산 8·18·9+18 = 1314, 8·8·9 = 576 | - |
