# 내용 확인 필요 (묶음 보고 모음, 고치지 않음)

## g13
- face-swapping-2008 §3, thought-cloning §3: 수식의 `\tilde`, `\times`가 탭 문자로 깨져 있음(HEAD부터)
- latent-diffusion-models §4: 복원 표 굵은 행 VQ f=4(27.43/0.58)보다 KL f=4(27.53/0.27)가 두 지표 모두 좋음
- infoswap ablation: "세 설정 모두 표준편차 이상 하락"이 표와 불일치(w/o IIB만 해당)
- diffusion-ql: "표준오차 8~11"이 표의 AntMaze ±8.6, ±7.6과 불일치
- dqn: Figure 2 캡션 다음 줄이 빈 줄 없이 붙음(렌더링)
- gemini-robotics: ALOHA 2 ICL 평균 65 vs 계산 64.3

## g01
- glyphdraw L56: Figure 3 캡션 다음 줄이 빈 줄 없이 붙음(렌더링). L157 "4.17자" vs "7.19글자" 단위 혼용
- mobile-former L238: 9.8G 대비 65G~84G를 "약 7배"(84G는 8.6배)
- latent-consistency-model L180: SD 스케줄 1,000 step인데 "수천에서 수십으로"
- patchcore L166: "Tesla V4 GPU"(T4/V100?)
- faceinpainter L144: 같은 표를 "(Table 3)"과 "(표 2)"로 지칭
- mobilefaceswap L163: 107M→0.5M을 "100배"(약 200배), L36 0.413+0.083=0.496M인데 0.495M
- vision-language-action-models L90: Gato 행 "[ViT]" 링크가 경량 ViT 서베이 글을 가리킴(ViT 글은 2026-10-26-vit)

## g06
- hierarchical-speaker-svc §5-2: "[U-Net]" 링크가 mwcnn 글을 가리킴
- ste §4: "ResNet-20 항등 STE 54%"(4비트)와 "2비트 ReLU STE 48%"가 한 문장에 섞임
- deeperforensics §2: 100명 × 30쌍인데 "쌍마다 3,210개 응답"
- live-vectorization §2: 제어점 다각형 교차 ⇔ 곡선 교차 "역도 성립"은 일반적으로 엄밀하지 않음
- audio-classification §6: 심음 2,000Hz인데 CRNN 예시 주석은 16kHz(예시용이면 무관)

## g12
- wafer-map-wm811k L57~58: Figure 5 캡션 다음 줄이 빈 줄 없이 붙음(렌더링)
- decision-transformer: D4RL 표 BC 평균 46.4/47.7이 서로 바뀐 듯. Medium HalfCheetah·Walker, Medium-Replay HalfCheetah에서 CQL이 굵은데 BRAC-v가 더 높음
- smooth-swap toy: 마지막 print 문구("smaller gap")가 출력(0.533 vs 0.498)과 반대
- e4s toy: "112 hair pixels kept on top of the source face"는 실제로 두 마스크 머리카락 픽셀 수 차이
- rainbow 해석 1: "(하나만 빼고 그렇다)"가 ablation(dueling·Double 영향 작음)과 잘 안 맞음
- splice-vit 해석 5: "표준편차 8~18%p"인데 표에 7.7 있음

## g07
- layout-transformer: PartNet 표 마지막 행 끝에 "| 실패 사례로…" 문단이 같은 줄에 붙음(렌더링, 표 8번째 셀)
- diffuser: Maze2D "Medium 265"(공식 256?)
- pruning-and-quantization: 도입부는 다음 글이 BNN 총정리, 끝부분은 "다음 글에서 XNOR-Net"
- adversarial-diffusion-distillation: $$N = 4$$와 $$\tau_n$$ 표기 혼용
- hires-face-swapping: 코드 출력 라벨 "variance reduction ~ 1/sqrt(n)"이 결과(떨림 증가)와 반대(본문은 설명함)
- gr00t-n1 해석 1: "실제 88시간이 생성 827시간, 시뮬레이션 6,500시간으로 불어난" — 시뮬레이션 출처 혼동. GR-1 표 평균(10.2, 76.8)이 단순 평균과 다름
- gato: Skill Mastery BC-IMP 평균 74.6 vs 계산 74.7

## g05
- brush-your-text: 향상 폭 description "약 30%p" vs 도입 "평균 25%p"
- speed-is-all-you-need: Winograd 표 메모리 열(타일 클수록 감소)과 본문("타일이 클수록 늘어남") 방향 반대
- deepsvg: `[ViT]` 링크가 경량 ViT 글을 가리킴
- faceshifter: 본문 "(6) 큰 포즈" vs Figure 10 캡션 (6) "반투명 물체". toy "mean inside the face"는 실제로 마스크 전체 평균
- styleipsb: "GTX 3090(논문 표기)"
- diffusion-policy: Push-T 표 바로 다음 줄에 목록이 붙음(렌더링). Kitchen p4 0.44→0.99를 "213% 향상"(약 125%). 20회 시행인데 성공률 79%

## g09
- neural-mmo L119, explainable-mapf-xg-cbs L154: 캡션 다음 줄이 빈 줄 없이 붙음(렌더링)
- explainable-mapf-xg-cbs: 태그 `aaai`인데 ICAPS 2022 논문
- cbs-task-assignment §4: "19명일 때 비용 절감이 7에서 가장 많이" — 7이 무엇인지 불명
- pruning-filters 해석 4: 2:4 희소성을 비정형 희소성 하드웨어 지원으로 설명(반정형)
- cnn-architectures: "[Attention]" 링크가 경량 ViT 글을 가리킴
- kd-survey: MobileNet 링크가 mobilenetv2 글을 가리킴

## g14
- relu-classifier: 폴드별 표 마지막 행 같은 줄에 본문이 붙음(렌더링)
- resnet §3: "위 PyTorch 구현은 옵션 B"인데 코드는 아래 4절
- self-referenced-deep-learning: Figure 3 캡션 다음 줄이 빈 줄 없이 붙음(렌더링)
- albert: 도입 BERT-xlarge RACE 73.9→54.3이 5절 표(73.8, 39.7)와 다름. "RACE 차이 5.2%"는 %p
- fslsd: Figure 2 캡션 `\vec`의 `\v`가 세로 탭(0x0B)으로 깨짐(HEAD부터)
- learning-weights-connections: "재학습 전 L1, 후 L2가 낫다"는 80% 열에만 맞음(95% 열은 반대)
- flowface: FF++ 표 SimSwap ID 평균 94.43 vs (93.63+96.22)/2=94.93. F_swa 98.70 vs ablation 98.71
- stealing-production-lm: 코드 주석 절 번호(4절, 5절)가 본문(2절, 3절)과 불일치. 6절 표 babbage-002 가중치 복원 "< 4·10^6"은 4·10^7 오기로 보임

## g02
- hypernetworks L44, phasic-policy-gradient L183: 캡션 다음 줄이 빈 줄 없이 붙음(렌더링)
- be-your-own-teacher L197~198: Figure 4 캡션 바로 다음에 목록이 붙음(렌더링). L161 "평균 2.02%p 향상", L185 "2/4는 3.3%p 손실"이 표에서 안 나옴
- edgevits: "MobileViT보다 2~2.7배 느림" vs 계산 2.1~2.8배
- implicit-q-learning L190: CQL 전체 182.8 vs 151.5+31.5=183.0
- anti-exploration L231: "β를 과제와 덜 무관하게"는 뜻이 반대(과제에 덜 의존하게). 코드 α vs 본문 β
- pi-star-0-6 L339: 에스프레소 처리량 "2배 넘게" vs 그래프 15→29(1.9배)

## g16
- openvla: 부록 Table 7 Franka-Tabletop 행에서 DP(matched)·Octo·OpenVLA(scratch)가 모두 43.4
- aot: 본문 Figure 1 (a)~(d) 지칭이 캡션(열 기준)과 불일치
- moonshine: "1,300만 개 적은 파라미터" vs 21.8M−8.1M=13.7M
- adaptive-alternate-analog-test: "이 글에 나오지 않는다"의 "이 글"이 모호. HD 행 합 99.89
- rnn-lstm-gru: unroll ASCII 그림이 입력끼리 사슬로 이어진 모양(코드 블록)

## g03
- mobilenet: 0.50 MobileNet-160(60.2%) vs AlexNet(57.2%)을 "4%p 높다"(3.0%p; 논문 "4% better")
- motion-planning-diffusion: 구현 표 `\times`가 탭 문자로 깨짐(HEAD부터). "충돌하는 것을 걸러 낸 25개를 고르지 않고 그대로"가 모순처럼 읽힘
- rafswap: Figure 3 캡션 다음 줄이 빈 줄 없이 붙음(렌더링)
- rt-2: 같은 수치 평균이 일반화 표 62, 크기 표 63(산술 62.3)
- fsgan: FaceShifter를 "같은 해"(arXiv 2019, CVPR 2020)

## g15
- big-little-net: 정리 표 "0.7~1.2%p" vs @256 표 0.91~1.17%p
- dist-kd: COCO 표 뒤 문단 다음 줄에 목록이 붙음(렌더링). "1.7%, 1.0% 올렸다"는 %p
- mappo: Figure 5 캡션과 본문의 강도 차이. 정리 표 권고 4 "ε < 0.2" vs 본문 "0.2 이하"
- perceptual-losses: toy 해석(깊을수록 이동에 둔감)을 출력(17%→21%)이 뒷받침 못 함. 출력 문구 "looks as bad as a different image"가 0.348 vs 1.999와 불일치
- face-segmentation-swapping: toy에서 `ideal`=`aware`라 "0 px"는 자명
- efficientzero: 식의 l ≤ k에서 k 미정의(앞 식은 n). "32 seeds" 확인. "셋 다 빼면"은 하나씩 뺀 ablation. "이 PDF에는 부록이 없다" 맥락 부족
- overfitting-regularization-augmentation: "(Nakkiran et al., 2019)" vs References "ICLR 2020"

## g10
- restoration-survey: Table 5 굵은 표시가 최솟값과 불일치(ImageNet FID DPS 52.61 < DiffPIR 53.21, CelebA FID DiffPIR 58.56 < Dirac-PO 59.45). Table 6 LPIPS 굵은 표시 없음. Figure 6 캡션 φ_N vs 본문 Φ_N. "DDRM은 8초" vs 표 8.8~10.1s
- pi0: flow matching 설명 "행동 쪽으로 가는 방향"과 목표 ε − A_t(노이즈 방향)가 어긋남
- ptq-vit: ViT-L FP 76.53 < ViT-B 77.91(원 논문 수치인지 확인)

## g04
- adaptdiffuser: MuJoCo 표 평균 83.4 vs 계산 83.3
- dbcsr: GPU 단락 "n³ FLOPs … 12.5n GFLOP/s 상한"은 2n³이어야 성립. toy 해설 "평균 7.7×7.7 원소" vs 실제 약 8×8. Figure 2 캡션 바로 다음에 목록이 붙음(렌더링)
- watermark-reliability: "800토큰(500단어)"과 "250토큰(200단어)" 비율 불일치. toy "0% 바꿔 씀: 평균 50 토큰"은 탐색 시작값(하한)
- acer: PPO를 "같은 해"(ACER arXiv 2016.11, PPO 2017.7)

## g17
- auto-kd: Figure 2 캡션 바로 다음에 목록이 붙음(렌더링). 5절 "0.7~1%p" vs 8절·표 "0.5~1%p"
- trpo-to-ppo: "PyTorch 2.13" 버전 실재 여부
- revcol 6절 1항: "분류 동등, 전이 이득" 주장이 "ADE20K 분할은 섞여 있다"와 다소 충돌
- mobilenetv2 1절: 매니폴드 조건 서술이 논문 진술을 꼬아 옮김
- paradigms: [DDPM] 링크가 latent-diffusion-models, [DDIM이나 DPM-Solver] 링크가 latent-consistency-model 글을 가리킴
- fast-face-swap: 조명 toy 출력(0.181 > 0.036)과 코드 결론 문장이 반대, description "조명 손실의 필요성을 확인한다"가 본문("재현하지 못했다")과 불일치. Figure 9가 두 곳에 인용됨
- never-give-up: UVFA 링크가 hindsight-experience-replay 글을 가리킴
- dagger: OCR 결과 문단 SEARN 서술이 헷갈림(α 값 차이)

## g08
- tacotron 5장 3번: "[디퓨전 기반 음성 모델]" 링크가 latent-diffusion-models(이미지) 글을 가리킴
- mwcnn: "Urban100에서 VDSR보다 0.9~1.4dB" vs 표 ×2 +1.54, ×4 +1.09
- adain: Figure 1 캡션 다음 줄이 빈 줄 없이 붙음(렌더링)
- t-stitch: Figure 4 캡션 다음 줄이 빈 줄 없이 붙음(렌더링). "FID 9.21(원래 9.19)" vs 표 DiT-XL 9.20. toy "앞 40~50%면 0.6배" vs 출력 0.61·0.51배

## g11
- imbalanced-data: prompt-warning의 "(PyTorch 2.13)" 버전 표기
- icm-curiosity: Figure 7 캡션 바로 다음에 목록이 붙음(렌더링)
- megafs: 같은 CelebA-HQ 설정인데 Ablation 표(W++: ID 0.5816, 포즈 3.8179, 표정 1.6489)와 Table 2·5(FTM: 0.5014, 3.58, 2.87) 수치가 다름
- agent57 6장: "평균·중앙값이 NGU·R2D2보다 높고"인데 중앙값은 1933.49 < R2D2 1935.86

## 공통 패턴
- 그림 캡션(`_Figure …_`) 줄 바로 다음 줄에 빈 줄 없이 본문·목록이 붙은 곳이 여러 글에 있다(kramdown에서 캡션과 한 문단으로 합쳐질 수 있음). 표 마지막 행과 같은 줄에 본문이 붙은 곳도 있다(layout-transformer, relu-classifier).
- `	`, ``가 제어 문자로 바뀌어 `	ilde`, `	imes`, `ec`이 깨진 수식(face-swapping-2008, thought-cloning, motion-planning-diffusion, fslsd). HEAD부터 있던 문제다.
- `[ViT]`, `[Attention]` 등 일부 링크가 ViT 원논문 글(2026-10-26-vit) 대신 경량 ViT 서베이 글을 가리킨다.
