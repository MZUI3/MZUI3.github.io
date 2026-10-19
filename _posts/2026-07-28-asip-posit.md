---
title: "[Lightweight] Synopsys ASIP Designer로 Posit 연산기 직접 만들기"
date: 2026-07-28 20:25:00 +0900
categories: [Neural Network, Lightweight]
tags: [hardware, asip, posit, nml, pdg, processor-design, risc-v, lightweight, edge-ai]
description: 범용 프로세서와 전용 회로의 중간인 ASIP의 개념과 Synopsys ASIP Designer의 설계 흐름(nML, PDG, Primitive·Chess header)을, Posit 덧셈기를 직접 넣어 보는 예제로 정리합니다.
math: true
---

> [Edge AI 하드웨어]({% post_url 2026-06-19-edge-ai-hardware %}) 글에서 부동소수점을 대체하는 **Posit** 숫자 체계와 RISC-V 커스텀 명령어를 정리했습니다. 이번 글은 그 후속으로, NN에 맞는 연산기를 **직접 프로세서에 넣는** 방법인 **ASIP(Application Specific Instruction-set Processor)**와 Synopsys **ASIP Designer**의 설계 흐름을, Posit 덧셈기 구현을 예로 정리합니다.

## 1. 배경: NN에 맞는 하드웨어

좋은 NN 하드웨어를 만들려면 **연산기와 구조를 NN 친화적으로** 구현해야 합니다.

- 신경망마다 빠르고 정확하게 동작하려면 **특정 함수**(활성 함수 등)가 필요
- 숫자 표현도 중요. NN의 가중치·활성값은 대부분 크기가 작은 값에 몰려 있고, **크기 1 근처에 정밀도를 집중**하는(tapered precision) **Posit**이 IEEE 754 부동소수점의 대안으로 제안됨. 스케일을 잘 맞추면 같은 비트 수로 더 높은 정확도를 낼 수 있다는 보고가 있습니다.

### Posit 복습

$$n$$bit posit은 sign, regime, exponent, fraction으로 구성됩니다.

- **Regime**: 같은 비트가 연속되다가 반대 비트에서 끝남. 첫 비트가 0이고 $$M$$개 연속이면 $$k = -M$$, 1이면 $$k = M - 1$$
- **Exponent**: $$es$$bit, bias 없는 2의 거듭제곱. $$\text{useed} = 2^{2^{es}}$$
- **Fraction**: 나머지 비트. Regime이 전체를 채우면 exponent와 fraction은 0

$$
x = (-1)^s \times \text{useed}^{\,k} \times 2^{e} \times (1 + f)
$$

자세한 예시는 [Edge AI 글]({% post_url 2026-06-19-edge-ai-hardware %})에 있습니다.

## 2. ASIP란

**범용 마이크로프로세서(GP core)**와 **고정된 전용 회로(hardwired datapath)**의 중간 형태입니다.

| | 범용 프로세서 | ASIP | 전용 회로 (ASIC 가속기) |
|---|---|---|---|
| 프로그래밍 | 자유로움 | **가능** (전용 명령어 추가) | 불가 |
| 성능·전력 | 낮음 | **높음** | 가장 높음 |
| 유연성 | 높음 | 중간 | 낮음 |

병렬 연산 능력을 가지면서도 **소프트웨어로 프로그래밍**할 수 있습니다.

### Synopsys ASIP Designer의 장점

- **성능 최대화**: 구조 최적화, 명령어·데이터·태스크 수준 병렬화
- **전력 최소화**: 전력에 최적화된 RTL 생성 (예: 코어 power gating)
- **프로그래밍 가능성**: 구성 요소나 알고리즘을 바꿀 때 SoC 시뮬레이션을 처음부터 다시 할 필요 없이, C와 모듈을 연결

## 3. 설계 흐름

```
nML (ISA 정의)  ──┐
PDG (primitive 동작) ─┼──▶ ASIP Designer ──▶ C 컴파일러·SDK·시뮬레이터 자동 생성
Primitive header ─┤                     └──▶ HDL (Verilog / VHDL) 생성
Chess header ─────┘
```

1. **nML**로 명령어 집합(ISA)과 구조를 정의
2. 이를 바탕으로 **컴파일러와 SDK를 자동 생성**
3. 생성된 도구로 응용 프로그램을 컴파일하고 **시뮬레이터**로 실행해, HW와 SW를 함께 최적화
4. 만족하면 **HDL을 생성**해 합성

## 4. 구성 파일들

### Primitive Header File

프로세서의 **기본 데이터 타입과 기본 연산(primitive)**을 선언합니다. C 헤더(`<processor>.h`)와 같은 형식이며, primitive들은 `<processor>_primitive` 네임스페이스에 둡니다.

- **Primitive type**: C++ 클래스처럼 선언. 기존 C 타입이 primitive 타입에 매핑됨
- **Primitive function**: nML에서 레지스터 전송(register transfer)을 구현하는 데 쓰이며, C++ 함수처럼 선언. 기존 C 연산자와 intrinsic 함수가 여기에 매핑됨

예: "2 word 입력 → 1 word 출력"인 posit 덧셈 primitive

### nML: Structural Skeleton

**저장소, 기능 유닛, 연결 요소, 사용자 정의 primitive 타입**을 선언합니다.

- 메모리 데이터 타입, 주소 타입, 데이터 선택 비트 타입
- 메모리, **레지스터 파일**, **파이프라인 레지스터**, transitory(임시 연결)
- 기능 유닛(functional unit)

### nML: Instruction-set Description

명령어 집합을 **두 종류의 규칙**으로 기술합니다.

- **Or-rule**: 서로 **대체 가능한** 명령어들 (이 중 하나)
- **And-rule**: **직교적이고 병렬로** 실행되는 명령어 조합

각 규칙에는 세 가지 속성이 있습니다.

- **Action**: 명령어의 동작을 레지스터 전송 모델로 기술
- **Syntax**: 어셈블리 문법
- **Image**: 명령어의 비트 인코딩

### PDG Description

Primitive 함수가 **실제로 어떻게 동작하는지**를 PDG 언어로 정의합니다.

- 프로세서 제어 유닛, 여러 사이클이 걸리는 기능 유닛, I/O 인터페이스의 내부 로직 구현
- **C 기반**: `+ - * / %` 같은 연산자와 `for`, `switch` 같은 제어문 지원
- C와 달리 **타입의 비트 폭이 자유로움** (예: 13bit 정수)
- C++, VHDL, Verilog로 변환 가능

### Chess Header File

C 컴파일러(**Chess**)를 위한 파일로, ASIP에서 쓸 **C 언어 타입, 연산, intrinsic 함수**를 정의하고 이를 primitive 타입·함수에 매핑합니다.

- 새로운 응용 타입 정의, 기존 타입 매핑
- 타입 표현, 타입 승격(promotion), **연산자 매핑**
- 함수 승격, inline 함수, 서브루틴

예를 들어 클래스 문법으로 16bit 중 15bit를 fraction으로 쓰는 `frac_t` 타입을 만들고, `operator+`를 덧셈 primitive에 매핑하면, C 코드에서 `a + b`라고 쓰는 것만으로 전용 명령어가 생성됩니다.

## 5. 예제: Posit 덧셈기(PAU) 구현

1. **Primitive header**: posit 데이터 타입(예: `posit8_2`, 8bit·es=2)과 posit 연산 primitive 선언
2. **nML**: 메인 nML 스크립트에 posit 연산 유닛(PAU)과 명령어(`opc.n`) 추가. nML view에서 PAU 연산이 올바르게 연결됐는지 확인
3. **PDG**: **posit decoder**(regime·exponent·fraction 분리)와 **posit adder** 로직 작성
4. **Chess header**: `posit8_2` 타입의 연산을 `+`, `&` 등 C 연산자로 승격
5. **빌드**: 컴파일러 생성, HDL(Verilog·VHDL) 생성

```c
// Chess header에서 연산자를 posit 덧셈 primitive에 매핑했다면,
// 응용 C 코드는 일반 연산처럼 쓸 수 있다.
posit8_2 posit_add(posit8_2 a, posit8_2 b) {
    posit8_2 s = a + b;          // → 생성된 PAU 명령어로 컴파일
    return s;
}
```

### 검증과 합성

- ASIP 디버거로 **마이크로코드, 파이프라인, I/O, 통계** 확인
- 생성된 데이터 타입과 PAU VHDL을 **Vivado**로 가져와, 기존에 직접 설계한 posit 덧셈기와 ASIP가 생성한 posit 덧셈기를 같은 **testbench**에서 비교 (결과가 다르면 시뮬레이션 중단)
- **Synopsys Design Compiler**로 두 설계를 합성해 면적·타이밍을 비교

이렇게 하면 직접 RTL을 짜는 것보다 빠르게, **컴파일러까지 딸린** 전용 프로세서를 만들고 검증할 수 있습니다.

## 6. 정리

| 파일 | 역할 |
|---|---|
| Primitive header | 기본 데이터 타입·연산 선언 |
| nML | 구조(저장소·유닛·연결) + 명령어 집합(Or/And rule, action·syntax·image) |
| PDG | Primitive 함수의 내부 동작 (C 기반, 비트 폭 자유) |
| Chess header | C 타입·연산자를 primitive에 매핑 (컴파일러용) |
| 결과물 | 컴파일러·SDK·시뮬레이터 + Verilog/VHDL |

RISC-V 커스텀 명령어가 **기존 ISA에 명령어를 추가**하는 방식이라면, ASIP Designer는 **ISA 자체를 처음부터 설계**하면서 도구까지 함께 생성하는 방식입니다.

## References

- Synopsys, ASIP Designer — nML Manual, PDG Manual, Chess Compiler Manual, Processor Modeling Manual, Checkers Simulator Manual (라이선스 사용자용 문서)
- [ASIP Designer 제품 소개](https://mips.com/processor-solutions/asips-tools/) (예전 Synopsys 제품 페이지 주소는 현재 MIPS 사이트로 연결됩니다. 2026년 10월 확인)
- Gustafson & Yonemoto, "Beating Floating Point at its Own Game: Posit Arithmetic," 2017.
- SIGARCH, "Posit: A Potential Replacement for IEEE 754"
- Towards Data Science, "Using Activation Functions in Neural Nets"
