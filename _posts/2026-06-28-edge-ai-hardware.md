---
title: "[AI 기초 13] Edge AI를 위한 하드웨어와 온디바이스 학습: TinyTL, SRAM·PIM, RISC-V, Posit"
date: 2026-06-28 17:27:00 +0900
categories: [Neural Network, Lightweight]
tags: [edge-ai, on-device-learning, tinytl, sram, pim, in-memory-computing, risc-v, posit, cmsis-nn, fpga, lightweight]
description: 경량 모델을 실제 기기에서 돌리고 학습하는 데 필요한 TinyTL의 학습 메모리 절감, SRAM과 PIM, RISC-V 커스텀 명령어, Posit 숫자 체계, CMSIS-NN을 알고리즘부터 하드웨어까지 정리합니다.
math: true
---

> 모델을 아무리 가볍게 만들어도, 결국 **어떤 하드웨어에서 어떻게 돌릴지**가 실제 속도와 전력을 결정합니다. 이 글에서는 경량 모델이 실제 기기로 내려가는 과정에서 만나는 주제들을 묶어 정리합니다.
>
> 1. **온디바이스 학습**: 추론이 아니라 학습까지 기기에서 하려면 무엇이 문제인가 (TinyTL)
> 2. **메모리**: SRAM 구조와, 메모리 안에서 연산하는 PIM
> 3. **프로세서**: 오픈소스 ISA인 RISC-V와 커스텀 명령어
> 4. **숫자 표현**: 부동소수점을 대체하려는 Posit
> 5. **소프트웨어**: MCU용 NN 라이브러리 CMSIS-NN

---

## Part 1. 온디바이스 학습

## 1. 왜 기기에서 학습하나

보안(개인 데이터를 서버로 보내지 않음), 통신 비용, 방대한 데이터 처리 문제 때문에 **edge 기기에서 직접 학습**하려는 수요가 늘고 있습니다. 사용자의 목소리나 사진에 맞춰 모델을 미세 조정(fine-tuning)하는 경우가 대표적입니다.

문제는 **메모리**입니다. 추론은 forward만 하면 되지만, 학습은 역전파를 위해 **중간 활성값(activation)을 모두 저장**해야 합니다. MobileNetV2 기준으로 학습에 필요한 메모리가 추론보다 훨씬 큽니다.

## 2. TinyTL: 파라미터가 아니라 메모리를 줄여라

TinyTL(Cai et al., NeurIPS 2020)의 핵심 관찰은 이렇습니다.

> MobileNetV2는 파라미터 수가 ResNet-50의 **4.3분의 1**이지만, 학습 시 **활성값 메모리는 ResNet-50과 거의 같다(차이가 1.1배에 불과).**

경량 모델은 파라미터를 줄이는 데는 성공했지만, 학습 메모리의 진짜 주범인 **활성값**은 줄이지 못했다는 뜻입니다. 그래서 TinyTL의 부제가 *"Reduce Memory, Not Parameters"*입니다.

### Transfer Learning 복습

사전학습된 모델을 목적에 맞게 재사용하는 방법은 데이터 크기와 유사성에 따라 세 가지입니다([Pruning 글]({% post_url 2026-06-23-pruning-and-quantization %})의 Fine-tuning 참고).

1. **전체 재학습**: 큰 데이터셋 필요
2. **Conv의 일부만 고정**하고 나머지 층과 classifier 학습
3. **Conv는 모두 고정**하고 classifier만 학습

3번은 학습 파라미터가 매우 적지만 정확도도 크게 떨어집니다. 여기에 BN 층만 추가로 학습하면 정확도는 꽤 오르지만, **활성값 메모리는 거의 그대로**입니다. 즉 기존 방법들은 학습할 파라미터를 줄였을 뿐, **학습 메모리는 줄이지 못했습니다.**

### 왜 bias만 학습하면 메모리가 줄까

층 $$y = Wa + b$$의 기울기를 보면

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y}\,a^\top,\qquad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}
$$

**가중치의 기울기를 구하려면 입력 활성값 $$a$$를 저장해야 하지만, bias의 기울기에는 필요 없습니다.** 그래서 가중치를 고정하고 bias만 학습하면 활성값을 저장할 필요가 크게 줄어듭니다.

또 활성 함수도 메모리에 영향을 줍니다. Sigmoid나 h-swish는 역전파 때 입력값 전체가 필요하지만, **ReLU는 양수였는지 여부(1bit 마스크)만** 있으면 됩니다. 그래서 TinyTL은 ReLU 계열을 씁니다.

### TinyTL의 세 가지 구성

**(1) Weight는 고정, bias만 fine-tune**

메모리를 약 12배 절약하지만, 모델 용량(capacity)이 줄어 **정확도가 떨어집니다**(underfitting).

**(2) Lite Residual Learning**

줄어든 용량을 보충하기 위해, 고정된 backbone 옆에 **작은 residual 모듈**을 붙여 이것만 학습합니다. 이 모듈의 활성값도 작게 유지해야 하므로 두 가지를 지킵니다.

- **해상도 줄이기**: 활성값 크기는 해상도에 비례하므로, average pooling으로 down-sampling한 뒤 연산하고 up-sampling으로 원래 크기에 맞춤
- **Inverted bottleneck 피하기**: MobileNetV2의 depthwise conv는 적은 용량을 보완하려고 채널을 크게 늘리는데(inverted bottleneck), 이 때문에 활성값이 커집니다. Lite residual은 **group conv**를 쓰고 projection layer를 빼서 **채널 수를 크게 줄입니다.**

**(3) Group Normalization**

메모리를 더 줄이려면 **배치 크기를 줄여야** 합니다. 그런데 BN은 배치가 작으면 통계가 불안정해 error가 커지므로, 배치 크기와 무관한 **GN**을 씁니다([6편 Normalization]({% post_url 2026-05-20-normalization %})).

### 결과

- Lite residual + bias 학습이 정확도와 학습 메모리 모두 가장 좋음
- 해상도를 320으로 올리면 전체 fine-tuning과 **비슷한 정확도로 메모리를 최대 약 6.5배 절약** (논문 초록 기준)
- 활성값을 가지치기하는(dynamic activation pruning) 방법과 달리 **정확도 저하 없이** 메모리를 줄임
- 배치 크기 8 → 1로 줄여도 성능은 비슷하고, 학습 메모리가 **약 16MB**까지 떨어짐 → 느린 DRAM 대신 **빠른 SRAM**만으로 학습 가능

TinyTL 이후 *On-Device Training Under 256KB Memory*(Lin et al., NeurIPS 2022)처럼 마이크로컨트롤러에서 학습하는 연구로 이어졌습니다. BNN의 학습 메모리를 줄이는 방법은 [BNN 글]({% post_url 2026-06-25-binary-neural-networks %})의 마지막 절에 정리했습니다.

## 3. 사례: 온디바이스 음성 인식 프로젝트 구상

세미나에서 다뤘던 프로젝트 구상을 예로 들면, 지금까지 배운 기법들이 어떻게 조합되는지 볼 수 있습니다.

**동기**: 키오스크를 어려워하는 디지털 취약계층은 목소리로 주문하길 원하는 경우가 많습니다. 하지만 대부분의 음성 인식은 서버에서 소프트웨어로 처리합니다. 이를 **하드웨어(FPGA → ASIC)**에서 처리하고, 기기에서 학습까지 하는 것이 목표입니다.

**구성**

| 단계 | 방법 |
|---|---|
| 데이터 | AI Hub의 텍스트–음성 매칭 데이터 |
| 특징 추출 | **MFCC**를 FPGA에서 실시간 추출 ([14편]({% post_url 2026-06-29-audio-classification %})) |
| 노이즈 대응 | MFCC는 특정 주파수 노이즈에 약하므로 **CNN 필터**로 걸러냄 |
| 특징 강조 | **Attention**(SE block, SimAM 등)으로 글자를 구분하는 특징 증폭 |
| 출력 | 실제 사용 빈도가 있는 한글 2,350자를 클래스로 |
| 경량화 | **RepVGG**로 multi-branch를 제거해 메모리 확보, 필요하면 양자화 |
| 온디바이스 학습 | **TinyTL** 기반 |
| 구현 | C 언어 + hex, 필요하면 Verilog |

**검토할 문제들**

- MFCC 같은 시간–주파수 이미지에 **1D conv**를 쓰면 시간 정보 손실이 크지 않을까?
- SimAM 같은 3D attention 가중치는 늘어나는 파라미터 대비 성능 향상이 얼마나 될까?
- 대부분의 경량 라이브러리는 **INT8** 기준인데, BNN을 쓰려면 별도의 라이브러리와 하드웨어 접근이 필요하다.
- 온디바이스 학습으로 오히려 성능이 나빠지지는 않을까?

---

## Part 2. 메모리: SRAM과 PIM

## 4. SRAM 구조

**SRAM(Static RAM)**은 플립플롭 방식의 메모리 cell을 가진 RAM입니다.

| | SRAM | DRAM |
|---|---|---|
| 저장 방식 | 플립플롭 (트랜지스터 4~6개) | 커패시터 1개 + 트랜지스터 1개 |
| Refresh | 필요 없음 (전원이 있는 한 유지) | 주기적으로 필요 |
| 속도 | **빠름** (refresh 없음, 로직과 같은 공정으로 칩 안에 배치) | 느림 |
| 집적도·가격 | 낮음·비쌈 | 높음·쌈 |
| 용도 | **캐시**, 소용량 온칩 메모리 | 주기억장치 |

### 6T SRAM Cell

트랜지스터 6개로 이루어집니다.

- **두 쌍의 인버터**(트랜지스터 4개)가 서로 맞물려 0과 1을 안정적으로 유지
- **접근 트랜지스터 2개**가 읽기·쓰기를 담당
- **Word Line (WL)**: cell을 선택하는 신호선
- **Bit Line (BL, /BL)**: 데이터가 실리는 선

**Write**: BL에 쓸 값(예: 1)을 올림 → WL에 1을 인가해 cell 선택 → 전압 차이로 인버터들이 1 또는 0을 출력 → WL을 0으로 내리면 두 접근 트랜지스터가 꺼지고 값이 cell에 유지됨

**Read**: BL과 /BL을 미리 같은 전압(보통 VDD)으로 충전(precharge) → WL에 1을 인가해 접근 트랜지스터를 켬 → cell에 저장된 값에 따라 BL과 /BL 중 한쪽이 조금 방전됨 → **감지 증폭기(sense amplifier)**가 두 선의 작은 전압 차이를 0 또는 1로 읽음

## 5. 폰 노이만 병목과 PIM

### 폰 노이만 구조

**프로세서와 메모리가 분리**되어 있고, 메모리가 명령어와 데이터를 저장하면 프로세서가 가져와 연산합니다. AI 연산은 프로세서와 메모리 사이를 오가는 데이터 양이 매우 많아서, **데이터 이동이 전체 성능을 제한하는 병목**(bottleneck)이 됩니다. 실제로 연산 자체보다 데이터를 옮기는 데 에너지가 훨씬 많이 듭니다. Horowitz(ISSCC 2014)가 45nm 공정에서 정리한 값(Han et al., 2015의 인용 기준)을 보면 차이가 뚜렷합니다.

| 동작 | 에너지 (대략) |
|---|---|
| 32bit 정수 덧셈 | 0.1 pJ |
| 32bit 부동소수점 곱셈 | 3.7 pJ |
| 32bit SRAM(캐시) 읽기 | 5 pJ |
| 32bit DRAM 읽기 | 640 pJ |

DRAM에서 숫자 하나를 가져오는 비용이 덧셈 수천 번과 맞먹습니다. TinyTL이 학습 메모리를 SRAM 크기로 줄이려 한 이유, PIM이 메모리 안에서 연산하려는 이유가 모두 여기에 있습니다.

### 하버드 구조

폰 노이만 구조는 버스가 하나라 명령어와 데이터를 동시에 가져올 수 없습니다. 하버드 구조는 **프로그램 메모리와 데이터 메모리를 분리**해 동시에 접근할 수 있게 했습니다. 현대 CPU의 L1 캐시가 이 방식입니다. 하지만 근본적으로는 여전히 연산과 메모리가 분리되어 병목이 남습니다.

### PIM (Processing-In-Memory)

**메모리 안에서 일부 연산을 직접** 합니다.

- 프로세서와 메모리가 주고받는 데이터 양 감소
- 전체 성능 향상, 에너지 소모 감소

### SRAM 기반 BNN 가속기

[BNN]({% post_url 2026-06-25-binary-neural-networks %})의 XNOR 연산은 비트 단위라 SRAM 배열 안에서 처리하기 좋습니다. 이 방향의 연구 흐름은 다음과 같습니다.

| 연도 | 연구 | 특징 |
|---|---|---|
| 2019 | Area-Efficient and Variation-Tolerant In-Memory BNN Computing using 6T SRAM Array | 6T SRAM에서 XNOR 연산 (signed input ±1) |
| 2020 | Energy-efficient XNOR-free In-Memory BNN Accelerator with Input Distribution Regularization | unsigned input(1/0)으로 바꿔 XNOR 없이 처리 |
| 2022 | Energy-Efficient In-Memory BNN Accelerator Design Based on 8T2C SRAM Cell | 8T2C cell 제안 |

---

## Part 3. 프로세서: RISC-V

## 6. RISC-V란

**RISC 기반의 오픈소스 ISA(Instruction Set Architecture)**입니다. 대부분의 ISA(x86, ARM)는 라이선스가 필요하지만 RISC-V는 누구나 무료로 쓰고 확장할 수 있습니다. 실생활의 소형·고속·저전력 기기를 고려해 설계되어, 현재 ARM이 주로 차지한 영역을 대체할 후보로 주목받고 있습니다.

### Base ISA와 확장

| Base | 설명 |
|---|---|
| RV32I | 32bit 정수 기본 명령어 |
| RV32E | RV32I의 축소판, 레지스터 16개 (임베디드용) |
| RV64I | 64bit 정수 |
| RV128I | 128bit 정수 (아직 draft) |

필요한 기능만 **모듈처럼** 붙여 씁니다. 예를 들어 RV32IMAC는 다음을 합친 것입니다.

- **I** (Base Integer): 약 40개의 기본 명령어. **Load-Store 구조**로 load/store 명령어로만 메모리에 접근
- **M** (Multiply/Divide): 곱셈과 나눗셈
- **A** (Atomic): Load-Reserved/Store-Conditional 등. 여러 코어가 같은 메모리에 접근할 때 순서를 직렬화해 의도치 않은 결과를 방지
- **C** (Compressed): 자주 쓰는 명령어를 **16bit로 압축**. 즉시값(imm)이나 offset이 작은 경우에 쓰며, 코드 밀도를 높임
- 그 외 F/D(단·배정밀도 부동소수점), V(벡터) 등

### 명령어 형식

32bit 명령어는 6가지 형식이 있습니다.

| 필드 | 의미 |
|---|---|
| opcode | 명령어 종류 |
| rd | 결과를 저장할 레지스터 |
| rs1, rs2 | 연산에 쓸 레지스터 |
| funct3, funct7 | 세부 기능 (합쳐서 10bit) |
| imm | 즉시값 (명령어 안에 직접 들어 있는 상수, 메모리 접근 불필요) |

- **R-type**: 레지스터끼리 연산 (`add rd, rs1, rs2`)
- **I-type**: rs2 + funct7 자리를 12bit 즉시값으로 사용
- **S-type**: store는 결과 레지스터가 필요 없으므로 rd 자리를 즉시값에 추가로 사용
- **B-type**: 분기. 즉시값은 13bit지만 최하위 bit가 항상 0이라 저장하지 않음
- **U/J-type**: 큰 즉시값, 점프

B·J 형식은 즉시값 비트들이 **섞인 위치**에 저장되어 있습니다. 이상해 보이지만, 다른 형식과 **같은 비트 위치를 최대한 공유**하게 해서 디코더의 신호 fan-out과 mux 비용을 줄이기 위한 설계입니다. 부호 비트도 항상 31번 bit에 있습니다.

### Custom Instruction

RISC-V의 가장 큰 장점은 **직접 명령어를 추가할 수 있다**는 것입니다. NN 가속기를 붙여 전용 명령어로 호출할 수 있습니다.

주의할 점이 있습니다.

- 32~64bit 정수 ISA의 대부분은 RISC-V 재단이 **고정**했습니다. 이 형식을 침해하면 기존 RISC-V 소프트웨어가 돌아가지 않습니다.
- 대신 **custom 용도로 예약된 opcode**(예: custom-0 = `0001011`)나 기존 명령어가 쓰지 않는 비트 공간(**brownfield**)을 이용합니다.
- 32bit보다 긴 명령어 인코딩(48bit, 64bit 등)은 더 많은 비트를 제공해 커스텀할 공간이 넓습니다. (RV64/RV128의 64·128bit는 레지스터 폭이고, 명령어 길이와는 다른 이야기입니다.)

---

## Part 4. 숫자 표현: Posit

## 7. 부동소수점의 문제

IEEE 754 부동소수점에는 몇 가지 불편함이 있습니다.

- 매우 작은 수와 큰 수에 같은 정밀도를 배분해 **비트를 비효율적으로 사용**
- 수많은 NaN 패턴과 ±0, ±∞ 등 **예외 처리가 복잡**해 하드웨어가 커짐
- 지수가 표현 가능한 최솟값보다 작아지는 **underflow**
- 반올림 때문에 결합 법칙 같은 **수학 법칙이 깨지는** 경우가 있음

## 8. Posit 형식

Posit(Gustafson, 2017)은 IEEE 754를 대체하기 위해 제안된 숫자 체계입니다. $$n$$bit posit은 네 부분으로 구성됩니다.

```
| sign | regime (가변 길이) | exponent (es bit) | fraction (나머지) |
```

**Regime**: 같은 비트가 연속되다가 **반대 비트가 나오면 끝나는** 가변 길이 영역입니다.

- 1이 $$m$$개 연속 → $$k = m - 1$$
- 0이 $$m$$개 연속 → $$k = -m$$

**Exponent**: es bit의 부호 없는 정수 $$e$$. 부동소수점과 달리 **bias가 없습니다.**

**Fraction**: 나머지 비트. Regime이 길어지면 fraction(과 exponent)이 아예 없을 수도 있습니다.

값은 다음과 같습니다.

$$
x = (-1)^s \times \text{useed}^{\,k} \times 2^{e} \times (1 + f),\qquad \text{useed} = 2^{2^{es}}
$$

### 예시: posit\<8, 1\>

`0 110 1 011`을 해석해 봅시다.

- sign = 0 (양수)
- regime = `110` → 1이 2개 연속 후 0 → $$k = 1$$
- useed $$= 2^{2^1} = 4$$
- exponent = `1` → $$e = 1$$
- fraction = `011` → $$f = 3/8$$

$$
x = 4^1 \times 2^1 \times (1 + 0.375) = 11
$$

posit\<8, 1\>로 표현할 수 있는 가장 큰 수는 $$4^6 = 4096$$, 가장 작은 양수는 $$1/4096$$입니다.

### Posit vs Float

| | Float (IEEE 754) | Posit |
|---|---|---|
| 정밀도 분포 | 범위 전체에 균일 | **1 근처에 집중** (tapered precision) |
| Underflow | 있음 | 0으로 떨어지지 않음 (가장 작은 양수 minpos로 반올림) |
| 예외 값 | 여러 NaN, ±0, ±∞ | **NaR**(Not a Real) 하나, 0 하나 |
| 비교 | 같은 값이 다른 비트일 수 있음 (+0과 -0) | 값과 비트가 일대일이고, 비트를 **정수처럼 비교**해도 대소 관계가 맞음 |
| 하드웨어 | 예외 처리로 복잡 | 상대적으로 단순 |

Posit의 정밀도는 크기가 **1 근처일 때 가장 높고, 아주 작거나 아주 큰 수로 갈수록 regime이 길어져 fraction 비트가 줄어듭니다.** 즉 underflow가 없다는 것이 "작은 수를 더 정밀하게 표현한다"는 뜻은 아닙니다.

딥러닝 가중치와 활성값은 대부분 **크기가 작은 값**에 몰려 있습니다. 적절히 스케일을 맞추면 Posit의 정밀도가 높은 구간을 활용할 수 있어, **같은 비트 수로 float보다 높은 정확도**를 내거나 **더 적은 비트로 같은 정확도**를 낼 수 있다는 결과들이 보고되어 NN 하드웨어에서 주목받습니다.

## 9. RISC-V + Posit

*A Lightweight Posit Processing Unit for RISC-V Processors in Deep Neural Network Applications*(Cococcioni et al., 2021)는 둘을 결합합니다.

1. RISC-V ISA 시뮬레이터(Spike)와 **Ariane(CVA6) 64bit RISC-V FPGA core** 사용
2. Posit 연산기 **PPU**(Posit Processing Unit)를 구현하고, float↔posit, fixed↔posit, posit↔posit 변환 명령어를 **custom opcode(`0001011`)**로 추가
3. CNN의 조상 격인 **LeNet**에 적용해 기존 결과와 비교

ISA를 직접 확장할 수 있는 RISC-V이기에 가능한 접근입니다.

---

## Part 5. 소프트웨어: MCU에서 NN 돌리기

## 10. CMSIS-NN과 Basic Block

**CMSIS-NN**은 ARM Cortex-M 마이크로컨트롤러용으로 최적화된 NN 연산 함수 라이브러리입니다. 아래 예시는 8bit(`q7_t`)·16bit(`q15_t`) 고정소수점을 쓰는 **초기(레거시) API**입니다.

> 최신 CMSIS-NN은 TensorFlow Lite의 int8 양자화 방식에 맞춘 `arm_convolve_s8()` 같은 `_s8`/`_s16` API를 중심으로 바뀌었고, 이 양자화 규격을 따르지 않는 `q7` 계열 레거시 함수는 **CMSIS-NN 4.0.0**(2022년 11월)에서 제거되었습니다. 레거시 함수를 계속 쓰려면 CMSIS 5.9.0을 써야 하고, `q7_t`, `q15_t` 같은 타입 이름도 `int8_t`, `int16_t`로 바뀌었습니다. 새로 시작한다면 최신 API 문서를 기준으로 하세요.
{: .prompt-warning }

```c
arm_convolve_HWC_q7_basic_nonsquare(
    const q7_t *Im_in,                                   // 입력 (HWC 순서)
    const uint16_t dim_im_in_x, const uint16_t dim_im_in_y,
    const uint16_t ch_im_in,
    const q7_t *wt, const uint16_t ch_im_out,            // 가중치
    const uint16_t dim_kernel_x, const uint16_t dim_kernel_y,
    /* padding, stride, bias, shift, output, buffer ... */);
```

**Basic Block**은 원래 컴파일러 용어로, **진입점 외에는 들어오는 분기가 없고 출구 외에는 나가는 분기가 없는 직선 코드 묶음**을 말합니다. MCU에서 NN을 구현할 때는 이런 함수들(conv, pooling, activation, FC)을 **블록처럼 이어 붙여** 네트워크를 만듭니다.

- MobileNet → depthwise conv 함수 + pointwise conv 함수의 반복
- ResNet → conv 함수 + 원소별 덧셈(shortcut)
- DenseNet → conv 함수 + 메모리 상의 concat

PyTorch의 `nn.Module`로 만든 블록이 MCU에서는 이런 **함수 호출의 연속**과 미리 계산된 파라미터 헤더 파일(`*_parameters.h`)로 바뀝니다. TensorFlow Lite Micro 같은 프레임워크가 내부적으로 CMSIS-NN을 호출합니다.

---

## 11. 정리

| 층위 | 주제 | 핵심 |
|---|---|---|
| 알고리즘 | TinyTL | 학습 메모리의 주범은 **활성값**. bias만 학습 + lite residual + GN |
| 메모리 | SRAM, PIM | 빠른 온칩 메모리, 메모리 안에서 연산해 **데이터 이동 병목** 해소 |
| 프로세서 | RISC-V | 오픈 ISA, **custom instruction**으로 NN 가속기 연결 |
| 숫자 표현 | Posit | 1 근처에 정밀도 집중, 예외 단순 |
| 소프트웨어 | CMSIS-NN | MCU용 최적화 NN 함수를 블록처럼 조합 |

경량화는 모델 구조만의 문제가 아니라 **알고리즘–숫자 표현–메모리–프로세서**가 함께 맞물리는 문제입니다. 다음 글에서는 응용 분야로, 이 하드웨어 위에서 돌릴 **음성·소리 분류**의 기초를 다룹니다.

## References

- Cai et al., "TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning," NeurIPS 2020. [arXiv:2007.11622](https://arxiv.org/abs/2007.11622)
- Horowitz, "Computing's Energy Problem (and what we can do about it)," ISSCC 2014.
- Lin et al., "On-Device Training Under 256KB Memory," NeurIPS 2022.
- Ding et al., "RepVGG," CVPR 2021. / Yang et al., "SimAM: A Simple, Parameter-Free Attention Module," ICML 2021.
- "FPGA-based real-time MFCC extraction for automatic audio indexing on FM broadcast data," DASIP 2015.
- Patterson & Waterman, 『The RISC-V Reader』 / RISC-V ISA Specification
- Gustafson & Yonemoto, "Beating Floating Point at its Own Game: Posit Arithmetic," 2017.
- Cococcioni et al., "A Lightweight Posit Processing Unit for RISC-V Processors in Deep Neural Network Applications," IEEE TETC 2021.
- Lai et al., "CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs," 2018. [arXiv:1801.06601](https://arxiv.org/abs/1801.06601)
- [CMSIS-NN GitHub](https://github.com/ARM-software/CMSIS-NN)
- SIGARCH, "Posit: A Potential Replacement for IEEE 754"
