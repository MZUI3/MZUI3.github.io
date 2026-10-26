# 블로그 교정 보고서: older-3-g4

대상 5개 파일(_posts/). 교정 원칙: blog-review 스킬 규칙 2~4에 따라 맞춤법·띄어쓰기·문장부호·비문·어색한 표현·같은 글 안의 문체와 용어 통일만 고쳤습니다. 내용, 수치, 순서, 소제목, 말투, 코드 블록, 인라인 코드, 수식, 링크, Liquid, front matter, 기술 용어, 고유명사는 1단계에서 건드리지 않았습니다. 내용상 의심되는 부분은 "내용 확인 필요"에 적고, 2단계에서 원문(논문, 공식 문서)과 대조해 확인된 오류만 최소한으로 고쳤습니다. 코드 블록의 출력 주석은 PyTorch 2.13로 직접 실행해 확인했습니다(CRNN 출력 크기와 파라미터 수 108098, fake_quantize 오차 0.0090/0.1625, magnitude pruning 0.8999..., `prune` 파라미터·버퍼 이름 모두 일치). 다섯 글은 이전에 정리가 잘 되어 있어 1단계 수정은 적습니다.

## 2026-06-22-audio-classification.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- Li et al.(Information 2021) 사례의 세부 수치: 저자 순서, 3초 분할과 50% 중첩, STFT window 150샘플/hop 75, 2,000Hz, $$\alpha = 0.2$$·threshold 0.45, focal loss에서 $$\alpha$$를 정상(y=0) 쪽에 곱하는 표기, $$\alpha$$ 미사용 시 MAcc 약 2%p 하락.
- Deng et al.(Neural Networks 2020) 사례: 서지 정보, CRNN 블록 순서(Conv → MaxPool → BN), residual 사용, LSTM(-a)과 GRU(-b) 비교, 3 conv + 1 LSTM이 최상이라는 결론.
- Mel 공식 주석: librosa 기본값 `htk=False`(Slaney), torchaudio 기본값 HTK.
- References의 librosa `mel_frequencies` 링크가 아직 유효한지.
- `{% post_url %}` 교차 링크(2026-06-19-edge-ai-hardware, 2026-06-06-object-detection) 대상 파일이 있는지.
- 코드 출력 주석(`torch.Size([8, 64, 4, 188])`, `108098`)과 본문 설명이 맞는지.

## 2026-06-19-edge-ai-hardware.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| BNN을 쓰려면 별도의 라이브러리와 하드웨어 접근이 필요하다. | BNN을 쓰려면 별도의 라이브러리와 하드웨어 접근이 필요합니다. | 통일 |

### 내용 확인 필요

- TinyTL 수치: MobileNetV2 파라미터 4.3배 감소인데 활성값은 1.1배만 감소, bias만 학습할 때 약 12배 절약, "해상도를 320으로 올리면 … 메모리를 최대 약 6.5배 절약(논문 초록 기준)", 배치 크기 1일 때 약 16MB, lite residual(group conv, 1×1 projection 제거, average pooling 다운샘플링), ReLU의 1bit 마스크.
- Horowitz/Han et al. 에너지 표(0.1 / 3.7 / 5 / 640 pJ).
- SRAM 기반 BNN 가속기 표의 연도(2019 / 2020 / 2022)와 제목.
- Cococcioni et al. 사례: Spike, Ariane(CVA6), custom opcode `0001011`, LeNet 적용, IEEE TETC 2021.
- CMSIS-NN 경고 상자: 레거시 `q7` 함수가 4.0.0(2022년 11월)에서 제거되었는지, CMSIS 5.9.0 안내, `q7_t` → `int8_t` 타입 변경.
- Posit 예시 계산(posit\<8,1\> `0 110 1 011` = 11, 최댓값 4096, 최소 양수 1/4096).
- `{% post_url %}` 교차 링크(pruning, normalization, binary-neural-networks, audio-classification) 대상 파일이 있는지.

## 2026-06-17-binary-neural-networks.md

### 수정 목록

| 원문 | 수정 | 유형 |
|---|---|---|
| 0이고(0에서는 임펄스 형태) 역전파하면 기울기가 전부 사라집니다 | 0이고(0에서는 임펄스 형태), 역전파하면 기울기가 전부 사라집니다 | 문장부호 |

### 내용 확인 필요

- AlexNet ImageNet top-1(실수 56.6%, BWN 56.8%, XNOR-Net 44.2%, BinaryNet 27.9%), XNOR-Net 58배 속도·32배 메모리, BWN 약 2배 속도.
- "XNOR-Net 원본 구현처럼 `torch.autograd.Function`으로 backward를 직접 지정": 원본 구현의 프레임워크 확인.
- XNOR-Net++ $$\Gamma$$ 표의 Case 번호와 "Case 2가 오히려 정확도가 가장 낮았습니다"라는 문장.
- Bi-Real Net: 18층 56.4%, 34층 62.2%, 289가지 출력값과 ApproxSign 오차 면적 2/3(재계산), "Scaling factor를 … 층(출력 채널)마다".
- Real-to-Binary 65.4%, 실수 ResNet-18 69.3%.
- ReActNet: baseline 61.1%(87M OPs), ReActNet-A 69.4%, 실수 대응 모델 72.4%, ReActNet-B/C 설명, RPReLU의 β 관찰, reduction block 설명.
- SA-BNN: 계수를 활성값에 적용, 초기값 실험 "(ResNet-18에서 59.2% vs 53.4%)"의 근거.
- RB-Net 수치: RPC 메모리 11.65배·연산량 34.81배(ResNet-18), 18.59배·51.55배(ResNet-34), BA top-1 +1.8%p·top-5 +1.4%p, Bi-Real 대비 약 1/3 FLOPs로 약 10%p.
- Wang et al.(2021) CBLAS 결과: "27.66~58.34배 빠르고", "1.59~2.08배의 추가 메모리", "배치 크기에 따라서는 구현이 불가능".
- `{% post_url %}` 교차 링크(pruning, edge-ai-hardware, activation-functions, cnn-architectures) 대상 파일이 있는지.

## 2026-06-15-pruning-and-quantization.md

### 수정 목록

수정 없음.

### 내용 확인 필요

- Han et al.(2015) 표(AlexNet 61M → 6.7M 9×, VGG-16 138M → 10.3M 13×), dropout 조정식, L1/L2 정규화 설명.
- "비트를 절반으로 줄였을 뿐인데 오차는 약 18배 … 정수 레벨 수가 256개에서 16개로 16배 줄기 때문입니다": 설명에 나온 16배와 출력에서 나온 18배가 다릅니다.
- 코드 출력 주석(0.8999..., `['bias', 'weight_orig']`, `['weight_mask']`, 0.0090, 0.1625)이 맞는지.
- `{% post_url %}` 교차 링크(binary-neural-networks, mobilenet) 대상 파일이 있는지.

## 2026-06-14-supernova-internship-2023.md

### 수정 목록

수정 없음.

### 내용 확인 필요

없음. (작성자의 개인 경험을 다룬 글이라 지시에 따라 교정만 하고 사실 확인은 하지 않았습니다.)

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| audio-classification | Li et al. 서지(저자 Li, Yin, Ma, Zhang, Liu / Information 12(2), 54, 2021) | 문제 없음 | https://api.crossref.org/works/10.3390/INFO12020054 | - |
| audio-classification | Li et al. 세부 수치(3초·50%, window 150/hop 75, α=0.2·threshold 0.45, α를 정상 쪽에 곱하는 표기, MAcc 약 2%p) | 확인 불가 | https://www.mdpi.com/2078-2489/12/2/54 (WebFetch 403, 본문 접근 불가) | - |
| audio-classification | Deng et al. 서지(Neural Networks 130, 2020) | 문제 없음 | https://api.crossref.org/works/10.1016/j.neunet.2020.06.015 | - |
| audio-classification | Deng et al. 구조와 결과(Conv→MaxPool→BN, residual, LSTM/GRU 비교, 3 conv + 1 LSTM) | 확인 불가 | https://doi.org/10.1016/j.neunet.2020.06.015 (Elsevier 본문 접근 불가) | - |
| audio-classification | librosa 기본값 `htk=False`(Slaney, 1kHz 아래 선형·위 로그), torchaudio `mel_scale` 기본값 `'htk'` | 문제 없음 | https://librosa.org/doc/latest/api/generated/librosa.mel_frequencies.html , https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html | - |
| audio-classification | librosa 참고 링크 | 수정함 | 기존 https://librosa.org/doc/latest/generated/librosa.mel_frequencies.html 은 404(librosa 1.0.0 문서 개편). 새 위치 https://librosa.org/doc/latest/api/generated/librosa.mel_frequencies.html | 링크 URL만 새 위치로 교체(외부 링크라 htmlproofer `--disable-external`과는 무관) |
| audio-classification | post_url 교차 링크 | 문제 없음 | `_posts/`에 2026-06-19-edge-ai-hardware.md, 2026-06-06-object-detection.md 있음 | - |
| audio-classification | 코드 출력(`[8, 64, 4, 188]`, `[8, 2]`, 108098) | 문제 없음 | 직접 실행(PyTorch 2.13) | - |
| edge-ai-hardware | TinyTL 4.3배/1.1배(MobileNetV2-1.4 대 ResNet-50), bias만 학습할 때 약 12배(FT-Full 391MB 대 TinyTL-B 32MB), 배치 1에서 약 16MB, lite residual(group conv, 1×1 projection 제거, 2×2 average pooling), ReLU 1bit 마스크 | 문제 없음 | https://arxiv.org/abs/2007.11622 (PDF 본문 1절, 3.2절, 표 2, 그림 5) | - |
| edge-ai-hardware | "해상도를 320으로 올리면 … 최대 약 6.5배 절약(논문 초록 기준)" | 수정함 | https://arxiv.org/abs/2007.11622 : 표 2와 4.2절은 TinyTL-L+B@320이 FT-Full과 같은 수준의 정확도로 6.0배 절약. 6.5배는 해상도별 비교(그림 3, 3.9~6.5배)의 최댓값 | "비슷한 정확도로 메모리를 약 6배 절약 (해상도별로 비교하면 3.9~6.5배, 초록의 "최대 6.5배")"로 수정 |
| edge-ai-hardware | 에너지 표(int ADD 0.1, float MULT 3.7, SRAM 5, DRAM 640 pJ) | 문제 없음 | https://arxiv.org/abs/1506.02626 (Table 1) | - |
| edge-ai-hardware | SRAM BNN 2019(6T SRAM, VLSI Symposium 2019)·2020(XNOR-free, ICCAD 2020) | 문제 없음 | https://pure.skku.edu/en/publications/area-efficient-and-variation-tolerant-in-memory-bnn-computing-usi/ , https://remotecenter.postech.ac.kr/handle/2014.oak/105720 | - |
| edge-ai-hardware | SRAM BNN 2022 "8T2C SRAM Cell" 논문 | 확인 불가 | 검색으로 같은 제목의 논문을 찾지 못함 | - |
| edge-ai-hardware | Cococcioni et al.: Spike, Ariane, custom opcode `0001011`, posit↔posit·float·fixed 변환, TETC(2021년 10월 온라인 게재, 2022년 10권 4호) | 문제 없음 | https://ieeexplore.ieee.org/document/9583876 , https://www.european-processor-initiative.eu/wp-content/uploads/2020/12/phd_workshop_rossi_pres.pdf | - |
| edge-ai-hardware | Cococcioni et al.이 LeNet에 적용했는지 | 확인 불가 | 같은 연구진의 2020 발표 자료에는 10층 CNN만 나오고 TETC 본문은 접근 불가 | - |
| edge-ai-hardware | CMSIS-NN 4.0.0의 레거시 연산자 제거, CMSIS 5.9.0 안내, q7_t → int8_t | 문제 없음 | https://github.com/ARM-software/CMSIS-NN/releases/tag/v4.0.0 (릴리스 노트 날짜 "21 November", 연도 2022) | - |
| edge-ai-hardware | Posit 예시 계산 | 문제 없음 | 재계산: 4^1 × 2^1 × 1.375 = 11, maxpos = 4^6 = 4096 | - |
| edge-ai-hardware | post_url 교차 링크 | 문제 없음 | `_posts/`에 대상 4개 파일 모두 있음 | - |
| binary-neural-networks | AlexNet 56.6/56.8/44.2/27.9%, 58배·32배, BWN 약 2배 | 문제 없음 | https://arxiv.org/abs/1603.05279 | - |
| binary-neural-networks | "XNOR-Net 원본 구현처럼 `torch.autograd.Function`으로" | 수정함 | https://github.com/allenai/XNOR-Net (원본은 Lua Torch 7 구현), https://github.com/jiecaoyu/XNOR-Net-PyTorch (`class BinActive(torch.autograd.Function)`) | "XNOR-Net의 PyTorch 재구현(XNOR-Net-PyTorch)처럼"으로 수정 |
| binary-neural-networks | XNOR-Net++ Case 표와 "Case 2가 가장 낮았다" | 수정함 | https://arxiv.org/abs/1909.13863 (Table 1: Case 1 α 55.5%, Case 2 dense 56.1%, Case 3 α⊗β 56.7%, Case 4 α⊗β⊗γ 57.1%) | 표의 Case 3/4 행을 논문 번호대로 바꿈(3 = α⊗β, 164개 / 4 = α⊗β⊗γ, 116개). 문장을 "가장 많은 값을 가질 수 있는 Case 2(56.1%)보다, 이를 분해한 Case 3(56.7%)과 Case 4(57.1%)가 더 정확했습니다."로 수정 |
| binary-neural-networks | Bi-Real 56.4%/62.2%, 289가지, 오차 면적 2/3 | 문제 없음 | https://arxiv.org/abs/1808.00278 (초록, 식 (3), 그림 3 설명) | - |
| binary-neural-networks | Bi-Real scaling factor "층(출력 채널)마다" | 문제 없음 | https://arxiv.org/abs/1808.00278 식 (6)은 층 l 단위. 같은 저자의 PyTorch 구현(https://github.com/liuzechun/ReActNet/blob/master/resnet/2_step2/birealnet.py)은 dim=1,2,3으로 평균을 내 출력 채널 단위로 계산하므로 표현을 그대로 둠 | - |
| binary-neural-networks | Real-to-Binary 65.4%, ResNet-18 69.3%, ReActNet baseline 61.1%(87M OPs)·A 69.4%·실수 대응 모델 72.4%, B/C 설명, RPReLU β 관찰, reduction block | 문제 없음 | https://arxiv.org/abs/2003.03488 (Table 1~3, 5.3~5.4절) | - |
| binary-neural-networks | SA-BNN "(ResNet-18에서 59.2% vs 53.4%)" | 수정함 | https://ojs.aaai.org/index.php/AAAI/article/view/16306 : 59.2% 대 53.4%는 τ를 0.4와 1로 고정했을 때와 SA가 없을 때의 비교. 두 계수를 0.4와 1로 초기화해 따로 학습했을 때는 60.8% | 괄호를 "(ResNet-18에서 60.8%. 0.4와 1로 고정하기만 해도 SA가 없을 때의 53.4%보다 높은 59.2%)"로 수정 |
| binary-neural-networks | RB-Net 전체 결과(ResNet-18 66.8%, Real-to-Binary 대비 +1.4%, 52M 대 165M) | 문제 없음 | https://pure.bit.edu.cn/en/publications/rb-net-training-highly-accurate-and-efficient-binary-neural-netwo/ (초록. Bi-Real-18 56.4% 대비 약 10%p, 연산량 약 1/3과도 맞음) | - |
| binary-neural-networks | RB-Net 세부 수치(RPC 11.65/34.81/18.59/51.55배, BA +1.8%p/+1.4%p) | 확인 불가 | IEEE TCSVT 본문 접근 불가. 초록의 전체 모델 개선 폭(+1.4%)과 본문의 BA +1.8%p는 비교 조건이 다를 수 있음 | - |
| binary-neural-networks | Wang et al. CBLAS 결과 | 수정함 | https://arxiv.org/abs/2102.04270v5 (6.2.2절: 속도 8.60~29.76배, 메모리 1.59~2.08배 증가, Keras 대비 메모리 27.66~58.34배 감소, 가속한 기존 방식은 배치 40 초과 시 실행 실패) | 27.66~58.34배는 속도가 아니라 Keras 대비 메모리이므로 속도를 8.60~29.76배로 바로잡고, 메모리 비교와 배치 40 한계를 원문대로 정리 |
| binary-neural-networks | post_url 교차 링크 | 문제 없음 | `_posts/`에 대상 4개 파일 모두 있음 | - |
| pruning-and-quantization | Han et al. 표, dropout 식, L1/L2 설명 | 문제 없음 | https://arxiv.org/abs/1506.02626 (Table 1, 3.1~3.2절) | - |
| pruning-and-quantization | 오차 18배를 "레벨 16배 감소"로 설명한 부분 | 수정함 | 재계산: 코드의 대칭 양자화 scale은 max\|x\|/127 대 max\|x\|/7로 127/7 ≈ 18.1배. 실행 출력 0.1625/0.0090 ≈ 18.1과 일치 | "…16개로 줄면서, 위 코드의 대칭 양자화에서는 한 칸의 간격(scale)이 $$127/7 \approx 18$$배로 커지기 때문입니다."로 수정 |
| pruning-and-quantization | 코드 출력 주석 | 문제 없음 | 직접 실행(PyTorch 2.13): 0.899993..., `['bias', 'weight_orig']`, `['weight_mask']`, `torch.Size([256, 256])`, 0.0090, 0.1625 | - |
| pruning-and-quantization | post_url 교차 링크 | 문제 없음 | `_posts/`에 2026-06-17-binary-neural-networks.md, 2026-05-30-mobilenet.md 있음 | - |

## 검토 단계 추가 수정 (2026-10-25)

| 파일 | 원문 | 수정 | 근거 |
|---|---|---|---|
| 2026-06-17-binary-neural-networks.md | 1.59~2.08배의 메모리가 더 필요하고, 같은 가속을 적용한 기존 방식은 | 1.59~2.08배의 메모리가 더 필요합니다. 반면 같은 가속을 적용한 기존 방식은 | 배치 40 초과 시 실행 실패는 제안 방식의 단점이 아니라 가속한 기존(standard) 구현의 한계입니다(arXiv 2102.04270v5 6.2.2절). "대신" 뒤에 이어 쓰면 제안 방식의 단점처럼 읽혀서 문장을 나눴습니다. |
