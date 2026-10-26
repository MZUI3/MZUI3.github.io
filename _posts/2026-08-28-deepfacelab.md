---
title: "[Face Swapping] DeepFaceLab: 추출·학습·변환 파이프라인과 DF/LIAE 구조로 영화급 교체를 만드는 공개 프레임워크"
date: 2026-08-28 21:41:00 +0900
categories: [Generative AI, Face Swapping]
tags: [face-swapping, deepfacelab, autoencoder, shared-encoder, liae, trueface, xseg, subject-specific, open-source]
description: DeepFaceLab(Petrov 등, 2020)이 DeepFakes의 공유 인코더·신원별 디코더 구조를 추출(검출·정렬·분할)·학습(DF/LIAE, DSSIM+MSE, TrueFace)·변환(역정렬, 색 전이, 포아송 블렌딩, 초해상도) 세 단계의 완성된 파이프라인으로 정리한 공개 프레임워크를 다룹니다. 주체 특정 교체가 작동하는 원리와 LIAE가 조명을 더 잘 물려받는 이유, FF++ 정량 비교와 ablation, 그리고 작은 실험으로 공유 인코더가 신원을 흘릴 때 잠재 정렬(TrueFace)이 왜 필요한지 확인합니다.
math: true
---

> 2018년 Reddit의 DeepFakes는 두 사람의 영상으로 오토인코더를 학습해 얼굴을 바꾸는 파이프라인을 대중화했지만, 결과는 거칠었고 작업 흐름은 불투명했습니다. **DeepFaceLab**(Petrov, Gao, Chervoniy, Liu 등, 2020, arXiv; 프리랜서·USTC)은 그 파이프라인을 **추출 → 학습 → 변환**의 느슨하게 결합된 모듈로 다시 설계한 공개 프레임워크입니다. 연구 논문이라기보다 시스템 보고서이지만, 영화 VFX와 유튜브(1억 조회)에서 가장 널리 쓰인 교체 도구이고, DFDC 데이터셋의 일부가 이것으로 만들어졌습니다. 저자들의 입장은 "최선의 방어는 공격"으로, 탐지 연구에 고품질 가짜를 공급하고 대중이 딥페이크의 존재를 알게 하는 것이 탐지만큼 중요하다는 것입니다. 이 글은 왜 공유 인코더가 교체를 가능하게 하는지, DF와 LIAE가 무엇이 다른지에 집중합니다.

## 1. 추출

- **검출**: S3FD 기본, RetinaFace로 교체 가능.
- **정렬**: 랜드마크가 시간 안정성의 열쇠입니다. 표준 포즈는 히트맵 기반 2DFAN, 큰 오일러 각은 3D 사전을 가진 PRNet을 쓰고, 연속 프레임의 랜드마크를 시간 평활화하는 옵션이 있습니다. Umeyama의 점 패턴 정합으로 기준 템플릿에 대한 **유사 변환**을 구하며, 변환이 가역이므로 변환 단계에서 되돌립니다. 추출 범위는 half/full/whole-face.
- **분할**: TernausNet으로 머리카락·손가락·안경을 제외한 얼굴 마스크를 만들고, 특정 영상에서 실패하면 **XSeg**로 사용자가 몇 장만 라벨링해 전용 분할 모델을 few-shot 학습합니다.

## 2. 학습: DF와 LIAE

**DF 구조**는 DeepFakes와 같습니다. src와 dst가 **공유하는 Encoder + Inter**, 그리고 **신원별 Decoder** 둘. 각 디코더는 자기 신원만 재구성하도록 학습되므로 표정이 짝지어질 필요가 없습니다. 추론 때 dst 얼굴을 src 디코더에 넣으면 "왜곡된 src"로 착각해 src 신원에 dst 포즈·표정의 얼굴을 냅니다([SimSwap]({% post_url 2026-08-31-simswap %})이 분석하듯 신원 정보가 디코더 가중치에 들어 있어 일반화는 안 됩니다). 약점은 dst의 **조명** 같은 정보를 충분히 물려받지 못한다는 것입니다.

**LIAE 구조**는 Encoder와 Decoder를 공유하고 **Inter를 두 개**(InterAB, InterB) 둡니다. InterAB는 src·dst 모두의 잠재를, InterB는 dst의 잠재만 만들고, src는 $$F^{AB}_{src} \Vert F^{AB}_{src}$$, dst는 $$F^{AB}_{dst} \Vert F^{B}_{dst}$$를 채널 연결해 디코더에 넣습니다. $$F^B_{dst}$$를 붙이는 것은 잠재의 방향을 필요한 클래스 쪽으로 옮기기 위한 것이고, InterAB가 두 신원의 **정렬된 공통 표현**을 갖게 됩니다. Ablation(Figure 6)에서 LIAE가 dst의 얼굴 형태와 조명을 더 잘 물려받습니다.

**손실과 기법**: DSSIM + MSE 혼합(DSSIM은 얼굴을 빨리 일반화하고 MSE는 선명도), 눈 영역에 가중치를 둔 마스크 SSIM, GAN 옵션(세부·주름·점이 살아남), **TrueFace**(LIAE에서 $$F^{AB}_{src} \approx F^{AB}_{dst}$$, DF에서 $$F_{src} \approx F_{dst}$$가 되도록 강제해 변환 결과가 dst를 더 닮게 함), 합성곱 인식 초기화, 학습률 드롭아웃. 다중 GPU, 반정밀도, 2GB VRAM에서도 동작합니다.

## 3. 변환

생성된 얼굴과 마스크를 Umeyama 역변환으로 원래 위치에 놓고, 피부톤을 맞추는 **색 전이 알고리즘 5종**(RCT, IDT 등) 중 하나를 적용한 뒤 **포아송 블렌딩**으로 경계를 잇고, 마지막으로 사전학습 얼굴 **초해상도** 네트워크로 선명하게 합니다. 저자들은 현존 방법의 결과가 대체로 매끈하고 세부(점, 주름)가 없다는 점에서 이 단계가 무시되어 왔다고 지적합니다.

## 4. 코드로 보기: 공유 인코더 교체와 잠재 정렬

두 "신원"을 1차원 32샘플 얼굴(고유 질감 × 포즈에 따라 이동하는 봉우리 + 입 열림 봉우리)로 만들고, 공유 인코더(6차원 병목)와 신원별 디코더를 DF식으로 학습합니다. dst 얼굴을 src 디코더에 넣은 결과를 "이상적 교체"(src 질감, dst 포즈·입)와 비교하고, TrueFace처럼 두 신원의 잠재 평균·표준편차를 맞추는 항을 더했을 때를 비교합니다.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); torch.set_num_threads(1)
# two 'identities' as 1-D faces of 32 samples: identity = a fixed texture pattern, pose/expression = a shift and a mouth-opening scalar
x = torch.linspace(-1, 1, 32)
tex = {"src": 0.6 + 0.25 * torch.sin(7 * x), "dst": 0.4 + 0.25 * torch.cos(4 * x)}
def face(who, shift, mouth):
    return tex[who] * torch.exp(-((x - shift) ** 2) * 2) + mouth * torch.exp(-((x - 0.5 - shift) ** 2) * 30)
def data(who, n): sh, mo = torch.rand(n) * 0.6 - 0.3, torch.rand(n) * 0.5; return torch.stack([face(who, s, m) for s, m in zip(sh, mo)]), sh, mo
def pose_of(f): return (f * x).sum(1) / f.sum(1)
def texcorr(f, who): return torch.corrcoef(torch.stack([f.mean(0), tex[who]]))[0, 1].item()
def run(trueface):
    torch.manual_seed(0)
    enc = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 6))                     # shared Encoder + Inter (DF structure)
    dec = {w: nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 32)) for w in tex}  # one Decoder per identity
    opt = torch.optim.Adam(list(enc.parameters()) + [p for d in dec.values() for p in d.parameters()], 3e-3)
    for s in range(3000):
        loss, lat = 0, {}
        for w in tex:
            b, _, _ = data(w, 32); z = enc(b + 0.05 * torch.randn_like(b)); lat[w] = z
            loss = loss + F.mse_loss(dec[w](z), b)                                           # each decoder reconstructs ONLY its own identity
        if trueface:   # DFL 'TrueFace': pull the two identities' latent distributions together (here: match mean and std per dim)
            loss = loss + 2.0 * ((lat["src"].mean(0) - lat["dst"].mean(0)) ** 2).sum() + 2.0 * ((lat["src"].std(0) - lat["dst"].std(0)) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        b_dst, sh, mo = data("dst", 300); swapped = dec["src"](enc(b_dst))
        ideal = torch.stack([face("src", s, m) for s, m in zip(sh, mo)])                     # source identity at the destination's pose
        print(f"{'with' if trueface else 'without'} latent alignment (TrueFace):")
        print(f"  own-identity reconstruction MSE: src {F.mse_loss(dec['src'](enc(data('src',300)[0])), data('src',300)[0]).item():.4f}, dst {F.mse_loss(dec['dst'](enc(b_dst)), b_dst).item():.4f}")
        print(f"  swap dst -> src decoder: MSE to the ideal swap {F.mse_loss(swapped, ideal).item():.4f}, MSE to the dst input {F.mse_loss(swapped, b_dst).item():.4f}")
        print(f"  pose kept |centre(swap) - centre(dst)| = {(pose_of(swapped) - pose_of(b_dst)).abs().mean().item():.3f}  (dst vs a random src face: {(pose_of(b_dst) - pose_of(data('src',300)[0])).abs().mean().item():.3f})")
run(False); run(True)
print("-> a shared encoder can still smuggle identity through the latent; forcing the two identities' latents onto one distribution leaves identity to the decoders")
```

```text
without latent alignment (TrueFace):
  own-identity reconstruction MSE: src 0.0259, dst 0.0000
  swap dst -> src decoder: MSE to the ideal swap 0.0061, MSE to the dst input 0.0181
  pose kept |centre(swap) - centre(dst)| = 0.028  (dst vs a random src face: 0.135)
with latent alignment (TrueFace):
  own-identity reconstruction MSE: src 0.0256, dst 0.0003
  swap dst -> src decoder: MSE to the ideal swap 0.0006, MSE to the dst input 0.0293
  pose kept |centre(swap) - centre(dst)| = 0.036  (dst vs a random src face: 0.135)
-> a shared encoder can still smuggle identity through the latent; forcing the two identities' latents onto one distribution leaves identity to the decoders
```

- 정렬 없이도 교체는 어느 정도 됩니다(이상적 교체와의 MSE 0.006 대 dst 입력과의 0.018). 포즈(봉우리 중심)는 0.03 이내로 유지되어 공유 인코더가 포즈를 공통으로 담는다는 것이 보입니다.
- 그러나 6차원 병목에도 인코더는 신원을 **일부 흘립니다.** 잠재 분포를 맞추는 항을 더하면 이상적 교체와의 MSE가 0.0006으로 10배 줄고 dst 입력과는 더 멀어집니다. 두 신원의 잠재가 같은 분포를 가져야만 "신원은 디코더가, 포즈는 잠재가" 담는 분업이 완성되고, 이것이 TrueFace와 LIAE의 InterAB가 하는 일입니다. DeepFakes가 입력에 워핑 노이즈를 넣는 것도 같은 목적입니다.

## 5. 평가

FaceForensics++의 500영상 × 100프레임에서 DeepFakes, Nirkin 등과 비교합니다. 공정성을 위해 96×96 출력의 경량 DF 구조(Quick96, GAN·TrueFace 없음), 1080Ti 1장에서 평균 3시간 학습입니다.

| 방법 | SSIM ↑ | 지각 손실 ↓ | 검증 거리 ↓ | 랜드마크 ↓ | 포즈 ↓ |
|---|---|---|---|---|---|
| DeepFakes | 0.71 | 0.41 | 0.69 | 1.15 | 4.75 |
| Nirkin 등 | 0.65 | 0.50 | 0.66 | 0.35 | 6.01 |
| DFL | 0.73 | 0.39 | 0.61 | 0.73 | 1.12 |

포즈와 신원(dlib 검증 거리)에서 가장 좋고 랜드마크는 Nirkin 등이 좋습니다. Ablation(표 2)에서 DF → DFHD → LIAE → LIAEHD로 갈수록 SSIM이 0.73 → 0.80으로 오르고, GAN은 단독으로 지표를 조금 해치지만 TrueFace와 함께 쓰면 최고(LIAEHD+GAN+TrueFace SSIM 0.80, 포즈 0.83)입니다. 저자들은 초해상도로 살아난 눈·치아 같은 세부가 SSIM류 지표에 반영되지 않는다고 적습니다.

## 6. 해석

**1. 주체 특정 교체의 원리를 가장 명확히 보여 주는 시스템입니다.** 공유 인코더가 "표정·포즈"를, 신원별 디코더가 "얼굴"을 담당한다는 분업은 [Disney의 comb 네트워크]({% post_url 2026-08-30-hires-face-swapping %})가 P명으로 확장하고, SimSwap이 "신원이 디코더에 박혀 있어 일반화가 안 된다"는 진단 끝에 ID 주입 모듈로 대체합니다. 위 toy의 잠재 정렬이 보여 주듯 이 분업은 저절로 완성되지 않습니다.

**2. 파이프라인의 완성도가 품질을 정합니다.** 랜드마크 시간 평활화, 색 전이, 포아송 블렌딩, 초해상도는 어느 것도 새롭지 않지만, 연구 논문들이 무시한 이 단계들이 "영화급"을 만듭니다. [FSGAN]({% post_url 2026-08-16-fsgan %})·[FaceShifter]({% post_url 2026-08-18-faceshifter %})가 종단간 학습으로 이 단계를 흡수하려 한 것과 대비되는 공학적 접근입니다.

**3. 데이터 규모를 전제로 합니다.** 최대 10만 장의 src 데이터셋을 지원하고 "데이터가 많을수록 좋다"고 명시합니다. 인물마다 학습이 필요하다는 한계 때문에 주체 비의존 방법들이 나왔지만, 충분한 데이터와 시간이 있을 때의 품질은 여전히 높아 [MobileFaceSwap]({% post_url 2026-10-05-mobilefaceswap %})은 "고정 신원 경량 모델이 더 좋다"는 관찰을 DFL에서 빌려 옵니다.

**4. "방어로서의 공개"라는 입장.** 탐지기에 데이터를 공급하고 대중의 식별력을 높인다는 주장은 [DeeperForensics]({% post_url 2026-08-22-deeperforensics %})·AOT의 "탐지를 위한 생성"과 같은 논리이지만, 가장 강력한 도구를 누구나 쓸 수 있게 한다는 점에서 논쟁적입니다. 논문 자체는 이 긴장을 길게 다루지 않습니다.

**5. 평가를 읽을 때 주의할 점.** 비교는 96×96 경량 설정이고 기준선은 FF++에 수록된 오래된 결과입니다. 지표는 dlib 검증·SSIM처럼 거칠며, 주체 특정 방법과 비의존 방법을 같은 표에 놓는 것은 공정하지 않습니다(DFL은 두 영상으로 3시간 학습). 저자 다수가 프리랜서이고 동료 심사를 거친 출판이 아닙니다. 위 toy의 "TrueFace"는 평균·분산 매칭이라 실제 구현의 손실과 같지 않습니다.

## 7. 정리

| 항목 | 내용 |
|---|---|
| 성격 | 주체 특정(src↔dst) 교체의 공개 프레임워크, 추출·학습·변환 3단계 |
| 추출 | S3FD 검출, 2DFAN/PRNet 랜드마크 + 시간 평활화, Umeyama 정렬, TernausNet/XSeg 분할 |
| 학습 | DF(공유 Enc+Inter, 디코더 2) / LIAE(공유 Enc+Dec, InterAB+InterB), DSSIM+MSE, 눈 가중 마스크, GAN, TrueFace |
| 변환 | 역정렬, 색 전이 5종, 포아송 블렌딩, 얼굴 초해상도 |
| 결과 | FF++에서 포즈 1.12, 검증 0.61로 DeepFakes·Nirkin보다 우수, LIAEHD+GAN+TrueFace SSIM 0.80 |
| 한계 | 인물마다 학습, 대량 데이터 필요, 거친 지표, 공개의 양면성 |

## References

- Petrov et al., "DeepFaceLab: Integrated, flexible and extensible face-swapping framework," 2020. [arXiv:2005.05535](https://arxiv.org/abs/2005.05535)
- Umeyama, "Least-Squares Estimation of Transformation Parameters Between Two Point Patterns," TPAMI 1991.
- Bulat, Tzimiropoulos, "How far are we from solving the 2D & 3D Face Alignment problem?" (2DFAN), ICCV 2017. [arXiv:1703.07332](https://arxiv.org/abs/1703.07332)
- Iglovikov, Shvets, "TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation," 2018. [arXiv:1801.05746](https://arxiv.org/abs/1801.05746)
- Reinhard, Ashikhmin, Gooch, Shirley, "Color Transfer between Images," IEEE CG&A 2001.
- Dolhansky et al., "The DeepFake Detection Challenge (DFDC) Dataset," 2020. [arXiv:2006.07397](https://arxiv.org/abs/2006.07397)
