# 블로그 교정·사실 확인 보고서 (older-4-g1, 2026-10-25)

대상: 2026-06-13-lightweight-vision-transformer.md, 2026-06-10-attention-is-all-you-need.md, 2026-06-08-rnn-lstm-gru.md, 2026-06-06-object-detection.md

## 2026-06-13-lightweight-vision-transformer.md

### 수정 목록
수정 없음.

### 내용 확인 필요
- 파라미터 표: MobileNetV2 약 3.5M, ResNet-50 약 25M, ViT-B/16 86M, ViT-L/16 307M, ViT-H/14 632M.
- ViT 데이터 규모별 결과: ImageNet-1k(약 130만 장)에서는 ResNet보다 낮고, ImageNet-21k(약 1,400만 장)에서 비슷하고, JFT-300M(3억 장)에서 CNN을 넘었다는 서술.
- BEiT: dVAE tokenizer, 패치의 약 40% 마스킹, ICLR 2022.
- MobileViT: 약 5~6M 파라미터, ICLR 2022.
- EfficientFormer: 큰 커널·stride patch embedding을 3×3 conv(stride 2) stem으로 대체, 마지막 stage만 3D 블록, iPhone latency 측정, NeurIPS 2022.
- MobileOne: Apple, depthwise/pointwise 분기 재매개변수화, iPhone에서 1ms 이하, CVPR 2023.
- *Early Convolutions Help Transformers See Better*: 2021, NeurIPS 2021.
- cross-link 대상 파일(2026-06-10-attention-is-all-you-need, 2026-06-06-object-detection, 2026-06-02-cnn-architectures, 2026-06-19-edge-ai-hardware, 2026-06-17-binary-neural-networks) 존재 여부와 "9편 Object Detection", 다음 글 BNN([AI 기초 12]) 번호 일치 여부.

## 2026-06-10-attention-is-all-you-need.md

### 수정 목록
수정 없음.

### 내용 확인 필요
- softmax([13, 9, 9]) ≈ [0.964, 0.018, 0.018] 계산.
- 코드 출력(attention weight 행렬, `True`)이 본문 설명(행 합 1, 대각선 위 0)과 맞는지.
- 학습 가능한 위치 임베딩과 sinusoidal의 성능이 "거의 같았고", 긴 문장으로의 외삽 가능성 때문에 sinusoidal을 택했다는 서술.
- 연산 복잡도 표(Self-Attention O(n²·d)/O(1)/O(1), Recurrent O(n·d²)/O(n)/O(n), Convolution O(k·n·d²)/O(1)/O(log_k n)).
- d_model=512, h=8, d_k=64, FFN 512→2048→512, N=6, Multi-head 연산량이 single-head와 비슷하다는 서술.
- 첫 문단에서 예고한 ViT, PoolFormer, Mobile-Former, GPT 글의 존재 여부와 끝부분 cross-link(2026-06-13-lightweight-vision-transformer).

## 2026-06-08-rnn-lstm-gru.md

### 수정 목록
| 원문 | 수정 | 유형 |
|---|---|---|
| $$c_t$$의 기울기를 보면, 게이트를 거치는 간접 경로를 빼고 cell state끼리의 직접 경로만 보면 | $$c_t$$의 기울기에서 게이트를 거치는 간접 경로를 빼고 cell state끼리의 직접 경로만 보면 | 어색한 표현(같은 말 반복) |

### 내용 확인 필요
- 파라미터 수 주석(RNN 1,600 / LSTM 6,400 / GRU 4,800)과 출력 shape 주석.
- PyTorch `nn.GRU` 식 $$h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$$과 `nn.LSTM` bias 초기화가 "0 근처"라는 서술.
- forget gate는 Gers et al.(2000)이 추가, forget gate bias 1 초기화는 Jozefowicz et al.(2015).
- cross-link 대상(2026-05-13-activation-functions "3편", 2026-06-22-audio-classification "14편", 2026-06-10-attention-is-all-you-need) 존재 여부와 번호.

## 2026-06-06-object-detection.md

### 수정 목록
수정 없음.

### 내용 확인 필요
- NMS 예시의 IoU 약 0.82, 출력 `tensor([0, 2])`, "torchvision 0.28에서 실행" 표기.
- YOLO v1: 448×448, 7×7×30, 24 conv + 2 FC(앞 20개 conv ImageNet 사전학습), λ_coord=5, λ_noobj=0.5, 45/155 fps, NMS가 mAP를 2~3%p 올림, 논문이 threshold 값을 밝히지 않았다는 서술, 예술 작품 일반화 실험.
- YOLO v2: BN으로 dropout 제거, 448 fine-tuning, k-means 거리 1−IoU, multi-scale 320~608, passthrough 26×26×512 → 13×13×2048, DarkNet-19.
- YOLO v3: BCE, 3개 scale, DarkNet-53, Focal loss 효과가 적었다는 보고.
- YOLOv5 v6.0 릴리스(2021)에서 Nano 모델 추가.
- R-CNN "이미지 한 장에 수십 초".
- cross-link 대상(2026-06-02-cnn-architectures, 2026-06-22-audio-classification, 2026-06-13-lightweight-vision-transformer) 존재 여부와 "14편 심음 분류" 내용 일치.

## 내용 확인 결과 (2026-10-25)

| 파일 | 항목 | 판정 | 근거(URL) | 수정 내용 |
|---|---|---|---|---|
| lightweight-vision-transformer | ViT 파라미터 표 | 문제 없음 | ViT 논문 Table 1: "ViT-Base … 86M / ViT-Large … 307M / ViT-Huge … 632M". MobileNetV2는 torchvision 기준 3.5M(논문 3.4M), ResNet-50은 25.6M으로 "약" 표기 범위. https://arxiv.org/abs/2010.11929 | - |
| lightweight-vision-transformer | 데이터 규모별 ViT vs ResNet, 이미지 수 | 문제 없음 | "ImageNet … 1.3M images … ImageNet-21k … 14M images … JFT … 303M"; ImageNet에서는 "a few percentage points below ResNets", Fig. 3에서 21k는 BiT 범위와 겹치고 JFT-300M에서 "they shine". https://arxiv.org/abs/2010.11929 | - |
| lightweight-vision-transformer | BEiT dVAE, 40% 마스킹 | 문제 없음 | "the image tokenizer learned by discrete variational autoencoder (dVAE)", "We randomly mask approximately 40%". https://arxiv.org/html/2106.08254 | - |
| lightweight-vision-transformer | MobileViT 파라미터, ICLR 2022 | 문제 없음 | "top-1 accuracy of 78.4% with about 6 million parameters", Comments: "Accepted at ICLR'22" (MobileViT-S 5.6M). https://arxiv.org/abs/2110.02178 | - |
| lightweight-vision-transformer | EfficientFormer stem, 3D 블록 위치, iPhone, NeurIPS 2022 | 문제 없음 | "Patch embedding with large kernel and stride is a speed bottleneck on mobile devices" → 3×3 stride-2 conv 두 개의 stem; iPhone 12(CoreML) 측정; 공개 모델(L1/L3/L7)은 MB3D를 마지막 stage에만 둠. NeurIPS 2022 proceedings 수록. https://arxiv.org/html/2206.01191 , https://papers.neurips.cc/paper_files/paper/2022/hash/5452ad8ee6ea6e7dc41db1cbd31ba0b8-Abstract-Conference.html | - |
| lightweight-vision-transformer | MobileOne | 문제 없음 | "inference time under 1 ms on an iPhone12", "Accepted at CVPR 2023", 저자 소속 Apple, "factorized into depthwise and pointwise layers". https://arxiv.org/abs/2206.04040 | - |
| lightweight-vision-transformer | Early Convolutions, NeurIPS 2021 | 문제 없음 | Xiao, Singh, Mintun, Darrell, Dollár, Girshick; Comments: "NeurIPS 2021". https://arxiv.org/abs/2106.14881 | - |
| lightweight-vision-transformer | cross-link, 편 번호 | 문제 없음 | 대상 5개 파일 모두 `_posts/`에 존재. object-detection은 [AI 기초 9], binary-neural-networks는 [AI 기초 12]로 본문 서술과 일치. edge-ai-hardware에 TinyTL ReLU(1bit 마스크) 설명 있음. | - |
| attention-is-all-you-need | softmax 예시 | 문제 없음 | 재계산: e⁴/(e⁴+2)=0.9647, 1/(e⁴+2)=0.0177 → [0.964, 0.018, 0.018]. | - |
| attention-is-all-you-need | 코드 출력 vs 본문 | 문제 없음 | 로컬(torch) 재실행 결과 attention 행렬과 `True`가 출력 블록과 동일. 본문 설명(행 합 1, 미래 위치 0)과 일치. | - |
| attention-is-all-you-need | 위치 임베딩 비교 서술 | 문제 없음 | 3.5절: "produced nearly identical results", sinusoidal은 "may allow the model to extrapolate to sequence lengths longer". https://arxiv.org/html/1706.03762v7 | - |
| attention-is-all-you-need | 복잡도 표, 하이퍼파라미터, multi-head 연산량 | 문제 없음 | Table 1과 동일; d_model=512, h=8, d_k=64, d_ff=2048, N=6; "the total computational cost is similar to that of single-head attention". https://arxiv.org/html/1706.03762v7 | - |
| attention-is-all-you-need | 예고한 후속 글, cross-link | 문제 없음 | 2026-06-13-lightweight-vision-transformer, 2026-07-13-mobile-former, 2026-07-16-poolformer, 2026-07-01-transformer-decoder-to-gpt 모두 존재. | - |
| rnn-lstm-gru | 파라미터 수, shape 주석 | 문제 없음 | 재계산: nn.RNN(16,32)=32·16+32·32+32+32=1,600, LSTM 4배 6,400, GRU 3배 4,800(로컬 실행으로도 확인). | - |
| rnn-lstm-gru | PyTorch GRU 식, 초기화 | 문제 없음 | 문서 식 "h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_(t-1)"; 가중치·bias 모두 U(−√k, √k), k=1/hidden_size → bias가 0 근처라는 서술과 맞음. https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html | - |
| rnn-lstm-gru | Gers 2000, Jozefowicz 2015 | 문제 없음 | forget gate는 Gers, Schmidhuber & Cummins(Neural Computation 2000)가 도입했고, Jozefowicz et al.(ICML 2015)은 forget gate bias를 1로 초기화할 것을 권장함(References 서지와 일치, 널리 알려진 사실). | - |
| rnn-lstm-gru | cross-link, 편 번호 | 문제 없음 | activation-functions = [AI 기초 3], audio-classification = [AI 기초 14](CRNN 절 있음), attention 글 존재. | - |
| object-detection | NMS 예시, torchvision 0.28 | 문제 없음 | 로컬 재실행: `nms` → tensor([0, 2]), `box_iou` = 0.8223; 설치된 torchvision 0.28.0. | - |
| object-detection | YOLO v1 수치, threshold 서술 | 문제 없음 | "24 convolutional layers followed by 2 fully connected layers", "first 20 convolutional layers", "λcoord = 5 and λnoobj = .5", "non-maximal suppression adds 2-3% in mAP", Picasso/People-Art; threshold는 "thresholds the resulting detections by the model's confidence"라고만 씀. 45/155 fps는 abstract. https://arxiv.org/abs/1506.02640 | - |
| object-detection | YOLO v2 세부 | 문제 없음 | "remove dropout", "448×448 resolution for 10 epochs", "d(box, centroid) = 1 − IOU", "{320, 352, …, 608}", "26×26×512 feature map into a 13×13×2048", Darknet-19. https://arxiv.org/abs/1612.08242 | - |
| object-detection | YOLO v3 세부 | 문제 없음 | "binary cross-entropy loss", "predicts boxes at 3 different scales", Darknet-53, "We tried using focal loss. It dropped our mAP about 2 points." https://arxiv.org/abs/1804.02767 | - |
| object-detection | YOLOv5 v6.0 Nano | 문제 없음 | v6.0 릴리스(2021-10-12)가 "new P5 and P6 'Nano' models: YOLOv5n and YOLOv5n6" 추가. https://github.com/ultralytics/yolov5/releases/tag/v6.0 | - |
| object-detection | R-CNN "수십 초" | 문제 없음 | R-CNN 논문은 GPU 13s/장, CPU 53s/장, Fast R-CNN 논문의 VGG16 R-CNN은 47s/장으로 "수십 초" 표현 범위. https://arxiv.org/abs/1504.08083 | - |
| object-detection | cross-link | 문제 없음 | cnn-architectures(RepVGG 포함), audio-classification(심음 분류, focal-loss 태그), lightweight-vision-transformer 모두 존재. | - |
