---
title: "[Robotics] 시간 최적 shortcut으로 매니퓰레이터 궤적 매끄럽게 만들기: 포물선과 직선의 닫힌 해"
date: 2026-10-24 10:40:00 +0900
categories: [Robotics, Motion Planning]
tags: [trajectory-smoothing, shortcutting, time-optimal, bang-bang, manipulator, collision-checking, puma, asimo, icra]
description: Hauser와 Ng-Thow-Hing(ICRA 2010)이 샘플링 플래너의 거친 경로를 속도·가속도 한계 아래 시간 최적인 포물선-직선 구간으로 이어 붙이는 shortcut 휴리스틱으로 매끄럽게 만든 방법을 정리합니다. 단일 관절 시간 최적 보간의 닫힌 해, 최소 가속도 보간으로 다관절 동기화, 포물선 경로의 정확한 충돌 검사, PUMA·ASIMO 실험과 온라인 평활화를 다루고, 작은 실험으로 보간 시간과 shortcut 효과를 확인합니다.
math: true
---

> PRM이나 RRT 같은 샘플링 플래너는 빠르게 충돌 없는 경로를 찾지만, 그 결과는 **정점마다 멈춰야 정확히 따라갈 수 있는 꺾인 선**입니다. 그대로 실행하면 느리고 부자연스럽습니다. 스플라인 피팅은 원래 경로를 벗어나 충돌할 수 있고, 수치 궤적 최적화는 매 반복마다 충돌 검사를 해야 해서 비쌉니다. 이 논문(Hauser, Ng-Thow-Hing, Indiana University·Honda Research Institute, ICRA 2010)은 로봇공학과 애니메이션에서 오래 쓰인 **shortcut 휴리스틱**을 두 가지로 바꿉니다. 구성 공간이 아니라 **구성·속도 상태 공간**에서 동작하고, 직선 대신 **속도·가속도 한계 아래 시간 최적인 포물선-직선 구간**을 shortcut으로 씁니다. 이 구간의 닫힌 해가 주요 기여이고, PUMA와 ASIMO에서 몇 초 만에 실행 시간을 40% 넘게 줄였습니다.

## 1. 문제: 빠르고, 안전하고, 자연스러운 움직임을 한 번에

고자유도 로봇이 사람 환경에서 움직이려면 세 가지를 동시에 만족해야 합니다. 계획이 빨라야 하고(속도), 충돌이 없어야 하며(안전), 보기에 부자연스럽지 않아야 합니다(미감). 비용 함수 최적화(장애물 포텐셜, 시간, 토크, 에너지)는 품질이 좋지만, 수백~수천 지점의 충돌 질의가 반복마다 필요해 실시간에는 너무 느립니다. Figure 1의 경로 하나를 충돌 검사하는 데만 0.15초가 걸립니다. shortcut은 최적성은 없지만 빠르고 구현이 쉬우며 실제로 품질 좋은 경로를 냅니다.

## 2. 방법

### 가정과 알고리즘

궤적 $$u(t)$$는 조각별 상수 가속도(포물선과 직선의 연결)로 표현되고, 실현 가능하려면 (1) 구성이 충돌 없는 집합 $$\mathcal{F}$$에 있고, (2) $$\lvert u'(t) \rvert \le v_{max}$$, (3) $$\lvert u''(t) \rvert \le a_{max}$$를 원소별로 만족해야 합니다. 목표는 실현 가능성을 지키며 **실행 시간을 최대한 줄이는 것**입니다(시간 최적 궤적은 장애물에 바짝 붙으므로, 실제로는 여유 마진이나 가중 목적을 넣을 수 있습니다).

1. 꺾인 선 경로를 각 정점에서 멈추는 시간 최적 궤적으로 바꾼다.
2. $$N$$번 반복: 궤적 위의 두 시각 $$t_a, t_b$$를 무작위로 뽑아 상태 $$(a, c) = (u(t_a), u'(t_a))$$, $$(b, d) = (u(t_b), u'(t_b))$$를 얻고, 두 상태를 잇는 시간 최적 보간 $$s(t)$$를 계산한다. 충돌이 없으면 그 구간을 $$s$$로 바꾼다.

### 단일 관절 시간 최적 보간

위치 $$x_1 \to x_2$$, 속도 $$v_1 \to v_2$$를 $$\lvert v \rvert \le v_{max}$$, $$\lvert a \rvert \le a_{max}$$ 아래 가장 빨리 잇는 궤적은 bang-bang 원리에 따라 네 가지 원시 동작 조합 중 하나입니다. $$P^+P^-$$, $$P^-P^+$$(가속 후 감속 포물선 둘), $$P^+L^+P^-$$, $$P^-L^-P^+$$(최대 속도 직선이 사이에 끼는 경우). 각 조합에 대해 변곡 시각을 대수적으로 풉니다.

- $$P^+P^-$$: 변곡 시각 $$t_P$$는 이차방정식 $$a_{max} t^2 + 2 v_1 t + \frac{v_1^2 - v_2^2}{2 a_{max}} + x_1 - x_2 = 0$$의 해 중 $$0 \le t$$이고 최고 속도 $$v_1 + a_{max} t_P$$가 한계 안인 것, 총 시간 $$T = 2 t_P + (v_1 - v_2)/a_{max}$$.
- $$P^+L^+P^-$$: 직선 구간 길이 $$t_L = \frac{v_2^2 + v_1^2 - 2 v_{max}^2}{2 v_{max} a_{max}} + \frac{x_2 - x_1}{v_{max}}$$, 가속 $$t_{P1} = (v_{max} - v_1)/a_{max}$$, 감속 구간, 셋 다 음수가 아니어야 유효하고 $$T = t_{P1} + t_L + t_{P2}$$.
- 나머지 둘은 $$a_{max}$$(와 $$v_{max}$$)의 부호를 뒤집은 것입니다.

유효한 조합 중 최소 $$T$$를 고릅니다. 이것이 함수 $$f(x_1, x_2, v_1, v_2, v_{max}, a_{max})$$입니다.

### 다관절: 가장 느린 관절에 맞추는 최소 가속도 보간

관절은 독립이라 가정하므로 $$T = \max_k f(\cdot)$$로 가장 느린 관절이 전체 시간을 정하고, 나머지 관절은 **종료 시각 $$T$$가 고정된 채 가속도를 최소화하는 보간** $$g(\cdot, T, t)$$를 씁니다. 같은 네 조합에 대해 가속도 $$a$$를 이차방정식 $$T^2 a^2 + \sigma(2T(v_1 + v_2) + 4(x_1 - x_2)) a - (v_2 - v_1)^2 = 0$$($$\sigma = \pm 1$$) 또는 직선 포함 경우의 닫힌 식으로 풀고, 전환 시각 $$t_S = \frac{1}{2}(T + (v_2 - v_1)/a)$$가 $$[0, T]$$에 있고 최고 속도가 한계 안인지 확인합니다. 모든 관절이 동시에 도착하므로 움직임이 자연스럽습니다.

### 정확한 충돌 검사

이산화해서 점검하면 해상도 $$\epsilon$$보다 작은 침투를 놓치고, shortcut이 거듭되어 장애물에 가까워질수록 위험합니다. 저자들은 Schwarzer 등의 적응형 이분법을 포물선 경로로 확장합니다. 구간 $$[t_1, t_2]$$에서 로봇 기하가 움직이는 최대 작업공간 거리 $$\lambda$$를 구하고(포물선은 경로의 경계 상자 대각선으로 $$\Delta q$$를 대체), 양 끝의 장애물 거리 $$\delta_1, \delta_2$$에 대해 $$\lambda \le \delta_1 + \delta_2$$면 구간 전체가 충돌 없음이 **보장**되며, 아니면 이분해 재귀합니다. 세밀한 이산화보다 2~10배 느리지만 정확합니다.

## 3. 코드로 보기: 보간 시간과 shortcut 효과

논문 Figure 5의 네 경우에 대해 단일 관절 시간 최적 보간 시간을 닫힌 해로 계산하고, 원형 장애물을 피하는 2관절 꺾인 경로에서 정점마다 멈추는 실행 시간이 shortcut으로 어떻게 줄어드는지, 남은 정점을 속도를 유지한 채 지나면 얼마나 더 줄어드는지 봅니다.

```python
import numpy as np, math
rng = np.random.default_rng(0)
def interp_time(x1, x2, v1, v2, vmax, amax):
    """execution time of the time-optimal 1-D trajectory under |v|<=vmax, |a|<=amax (bang-bang classes)."""
    best = math.inf
    for s in (1, -1):                                   # P+P- (s=1) and P-P+ (s=-1)
        a = s * amax
        A, B, C = a, 2 * v1, (v1 ** 2 - v2 ** 2) / (2 * a) + x1 - x2
        disc = B * B - 4 * A * C
        if disc >= 0:
            for tp in ((-B + math.sqrt(disc)) / (2 * A), (-B - math.sqrt(disc)) / (2 * A)):
                T = 2 * tp + (v1 - v2) / a
                if tp >= 0 and T >= tp and abs(v1 + a * tp) <= vmax + 1e-9: best = min(best, T)
        vm = s * vmax                                   # P+L+P- / P-L-P+
        tp1 = (vm - v1) / a; tp2 = (vm - v2) / a
        tl = (v2 ** 2 + v1 ** 2 - 2 * vm ** 2) / (2 * vm * a) + (x2 - x1) / vm
        if tp1 >= 0 and tp2 >= 0 and tl >= 0: best = min(best, tp1 + tl + tp2)
    return best
def shortcut_time(a, b, va, vb, vmax, amax):            # multi-joint: the slowest joint decides
    return max(interp_time(a[k], b[k], va[k], vb[k], vmax[k], amax[k]) for k in range(len(a)))

print("1-D time-optimal interpolants (vmax = amax = 1), the paper's Fig. 5 cases:")
for (x1, x2, v1, v2) in [(0, 1, 0, 0), (0, 3, 0, 0), (0, 0, 1, 0), (0, -0.5, 1, 1)]:
    print(f"  x {x1:>4} -> {x2:>4}, v {v1} -> {v2}:  T = {interp_time(x1, x2, v1, v2, 1, 1):.3f} s")

# a jerky piecewise-linear path (2 joints) around a disc obstacle: stop-at-every-milestone execution time vs. shortcutting
path = np.array([[0, 0], [1, 1.2], [2, 1.5], [3, 1.3], [4, 1.6], [5, 0.5], [6, 0]], float)
obstacle, rad = np.array([3.0, 0.3]), 0.9
vmax, amax = np.ones(2), np.ones(2)
def seg_time(a, b): return shortcut_time(a, b, np.zeros(2), np.zeros(2), vmax, amax)
def collision_free(a, b):
    pts = a[None] + np.linspace(0, 1, 40)[:, None] * (b - a)[None]
    return np.all(np.linalg.norm(pts - obstacle, axis=1) > rad)
nodes = [p.copy() for p in path]
T0 = sum(seg_time(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))
print(f"\ninitial trajectory (stop at each of {len(nodes)} milestones): {T0:.2f} s")
for it in range(60):                                     # shortcutting on the milestone list, timed with the optimal interpolant
    if len(nodes) >= 3:
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        if j - i >= 2 and collision_free(nodes[i], nodes[j]):
            old = sum(seg_time(nodes[k], nodes[k + 1]) for k in range(i, j)); new = seg_time(nodes[i], nodes[j])
            if new < old: nodes = nodes[:i + 1] + nodes[j:]
    if it in (9, 19, 39, 59):
        print(f"  after {it+1:>2} shortcut attempts: {len(nodes)} milestones, {sum(seg_time(nodes[k], nodes[k+1]) for k in range(len(nodes)-1)):.2f} s")
# the same milestones, but passing through interior milestones with a non-zero velocity instead of stopping
through = 0; v_prev = np.zeros(2)
for k in range(len(nodes) - 1):
    d = nodes[k + 1] - nodes[k]; v_next = np.zeros(2) if k == len(nodes) - 2 else 0.5 * d / np.abs(d).max()
    through += shortcut_time(nodes[k], nodes[k + 1], v_prev, v_next, vmax, amax); v_prev = v_next
print(f"  same milestones, non-zero velocity at interior milestones: {through:.2f} s")
```

```text
1-D time-optimal interpolants (vmax = amax = 1), the paper's Fig. 5 cases:
  x    0 ->    1, v 0 -> 0:  T = 2.000 s
  x    0 ->    3, v 0 -> 0:  T = 4.000 s
  x    0 ->    0, v 1 -> 0:  T = 2.414 s
  x    0 -> -0.5, v 1 -> 1:  T = 4.500 s

initial trajectory (stop at each of 7 milestones): 12.30 s
  after 10 shortcut attempts: 3 milestones, 8.00 s
  after 20 shortcut attempts: 3 milestones, 8.00 s
  after 40 shortcut attempts: 3 milestones, 8.00 s
  after 60 shortcut attempts: 3 milestones, 8.00 s
  same milestones, non-zero velocity at interior milestones: 7.25 s
```

- 네 경우의 보간 시간(2.0, 4.0, 2.41, 4.5초)은 논문 Figure 5의 곡선과 일치합니다. 0에서 3까지 정지 상태로 가면 가속 1초, 최대 속도 직선 2초, 감속 1초의 $$P^+L^+P^-$$가 됩니다. 속도 1로 출발해 제자리에 멈추려면 $$P^-P^+$$로 2.41초가 듭니다.
- 7개 정점에서 모두 멈추면 12.3초인 경로가 shortcut 10회 안에 정점 3개, 8.0초로 줄고 이후 수렴합니다. 논문 Figure 8처럼 **처음 몇 번의 shortcut이 대부분의 이득**을 냅니다. 남은 정점을 속도를 유지하며 지나면 7.25초로 더 줄어드는데, 이것이 상태 공간 shortcut(속도까지 잇는)의 몫입니다.

## 4. 실험

**PUMA 760 시뮬레이션.** 약 8,000 폴리곤 로봇, 20,000 폴리곤 환경에서 테이블 아래의 컵을 잡는 작업. SBL 플래너가 만든 초기 경로 10개의 최적 실행 시간은 5.7~10.4초였고, **200회 shortcut 후 평균 40% 감소**해 4.1~6.6초가 됐습니다. 3GHz PC에서 계획 0.6초, 평활화 2.6초, 총 3.2초. 닫힌 해 덕분에 shortcut 자체는 무시할 만한 시간이고, **평활화 시간은 충돌 검사가 지배**합니다. 실행 시간은 초반에 빠르게 줄고 이후 수확 체감합니다(Figure 7, 8).

**온라인 평활화.** 평활화를 끝내고 실행하면 언제 멈출지 정하기 어렵지만, shortcut은 **현재 시각 이후의 경로에서만** 뽑도록 하면 실행과 평활화를 병렬로 돌릴 수 있습니다. 로봇을 멈추지 않고도 고품질 경로가 나옵니다(Figure 9).

**ASIMO.** 테이블 위 물체를 미는 시스템에 통합해, 50회 shortcut이 평균 0.9초 계산으로 실행 시간을 2.4초(46%) 줄였고, 관찰자들이 움직임이 훨씬 자연스럽다고 판단했습니다.

## 5. 해석

**1. 기여는 휴리스틱이 아니라 닫힌 해입니다.** shortcut 자체는 오래된 기법이고, 이 논문의 핵심은 속도·가속도 제약 아래 두 상태를 잇는 시간 최적 보간을 **반복 없이 계산하는 공식**입니다. 그 덕분에 수천 번의 shortcut 시도가 가능하고, 남는 병목이 충돌 검사뿐이라는 점이 실용성의 근거입니다.

**2. 구성 공간에서 상태 공간으로의 전환이 품질을 바꿉니다.** 위 toy에서 같은 정점을 멈추지 않고 지나기만 해도 10% 가까이 줄었습니다. 속도를 잇는 shortcut은 정점의 정지를 없애 "로봇 같은" 끊김을 지우고, 이것이 ASIMO 관찰자들이 느낀 자연스러움의 정체입니다.

**3. 최적성은 포기하고 안전성은 포기하지 않았습니다.** 지역 최적성도 보장하지 않지만, 정확한 충돌 검사를 포물선 경로로 확장해 shortcut이 장애물에 가까워질수록 커지는 위험을 막았습니다. [Motion Planning Diffusion]({% post_url 2026-10-24-motion-planning-diffusion %})처럼 학습된 사전으로 좋은 초기 궤적을 얻는 최근 방법도, 마지막에는 이런 제약 보장 후처리가 필요합니다.

**4. 한계는 입력 경로와 독립 관절 가정입니다.** 후처리이므로 나쁜 위상의 입력 경로(장애물을 멀리 돌아가는)는 고치지 못하고, 관절별 박스 한계는 실제 토크·동역학 결합을 단순화한 것입니다. 저자들도 계획과 평활화의 통합, 비균일 shortcut 전략을 후속 과제로 적었고, 이후 연구가 jerk 한계와 동역학 제약으로 확장했습니다.

**5. 평가를 읽을 때 주의할 점.** 실험은 두 로봇, 각각 10개 안팎의 경로이고, 비교 대상(스플라인, 수치 최적화)과의 정량 비교는 없습니다. "자연스러움"은 관찰자 판단입니다. 위 toy는 정점 간 직선 shortcut만 시도하고 속도를 잇는 상태 공간 shortcut은 마지막 한 줄로만 흉내 냈습니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 문제 | 샘플링 플래너의 꺾인 경로는 정점마다 멈춰야 하고 느리며 부자연스러움 |
| 핵심 | 상태(위치·속도) 공간에서 두 점을 잇는 시간 최적 포물선-직선 구간으로 shortcut |
| 닫힌 해 | 4가지 bang-bang 조합($$P^+P^-$$, $$P^-P^+$$, $$P^+L^+P^-$$, $$P^-L^-P^+$$)의 변곡 시각을 대수적으로 계산 |
| 다관절 | 가장 느린 관절의 $$T$$에 맞춰 나머지는 최소 가속도 보간 |
| 충돌 검사 | 적응형 이분법 + 포물선 경계 상자로 정확한 보장 |
| 결과 | PUMA 200회 shortcut 평균 -40%(3.2초 계산), ASIMO 50회 -46%(0.9초), 온라인 평활화 가능 |

## References

- Hauser, Ng-Thow-Hing, "Fast Smoothing of Manipulator Trajectories using Optimal Bounded-Acceleration Shortcuts," ICRA 2010.
- Geraerts, Overmars, "Creating High-quality Paths for Motion Planning," IJRR 2007.
- Schwarzer, Saha, Latombe, "Exact Collision Checking of Robot Paths," WAFR 2002.
- Sánchez, Latombe, "On Delaying Collision Checking in PRM Planning: Application to Multi-Robot Coordination," IJRR 2002.
- Kunz, Stilman, "Time-Optimal Trajectory Generation for Path Following with Bounded Acceleration and Velocity," RSS 2012.
