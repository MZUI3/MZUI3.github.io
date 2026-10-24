---
title: "[MAPF] XG-CBS: 사람이 그림 몇 장으로 검증할 수 있는 다중 에이전트 경로 계획"
date: 2026-10-24 10:00:00 +0900
categories: [Multi-Agent, Path Finding]
tags: [mapf, cbs, explainable-ai, conflict-based-search, segmentation, a-star, trust, aaai]
description: Kottinger, Almagor, Lahijanian(AAAI 2022)이 다중 에이전트 경로 계획의 결과를 "경로가 서로 겹치지 않는 시간 구간 몇 개"로 나눠 보여 주는 설명 가능 MAPF 문제를 CBS로 푸는 XG-CBS를 정리합니다. 분할 충돌 제약, 설명 지향 저수준 탐색 XG-A*·WXG-A*·SR-A*의 완전성과 속도 트레이드오프, 벤치마크 결과를 다루고, 작은 실험으로 최단 계획과 설명 가능한 계획의 차이를 확인합니다.
math: true
---

> [CBS-TA]({% post_url 2026-10-13-cbs-task-assignment %})에서 본 Conflict-Based Search(CBS)는 다중 에이전트 경로 계획(MAPF)의 대표 알고리즘입니다. 그런데 항공 관제나 위험물 창고처럼 규제가 강한 현장에서는 계획이 옳다는 것을 **사람 감독자가 확인하고 승인**해야 합니다. 수십 개의 경로가 시간에 따라 얽힌 그림을 보고 충돌이 없다고 확신하기는 어렵습니다. 선행 연구(Almagor & Lahijanian, 2020)는 계획을 **시간 구간으로 잘라 각 구간 안에서는 경로들이 서로 겹치지 않게** 그려 주는 설명 방식을 제안했는데, 그림 수(index)가 적은 계획을 찾는 문제는 에이전트 2명만 있어도 NP-hard이고 중앙집중 알고리즘은 확장되지 않았습니다. **XG-CBS**(Explanation-Guided CBS, Kottinger, Almagor, Lahijanian, CU Boulder·Technion, AAAI 2022)는 CBS의 제약 트리에 "분할 충돌"을 추가하고, 그림 수를 줄이도록 유도하는 저수준 탐색을 설계해 이 문제를 분산적으로 풉니다.

## 1. 문제: 설명 가능 MAPF via Segmentation

그래프 $$G$$ 위의 $$n$$개 에이전트에 대한 계획 $$P = \{\pi_1, \ldots, \pi_n\}$$이 정점 충돌과 간선 충돌이 없을 때, **정점 분리 분해**란 시간 경계 $$1 = t_0 < t_1 < \cdots < t_r = K + 1$$로 잘라 각 구간 $$[t_{k-1}, t_k - 1]$$ 안에서 모든 에이전트의 부분 경로가 **정점을 공유하지 않게** 만드는 것입니다. $$r$$을 분해의 index라 하고, 최소 index는 탐욕적으로 구간을 늘려 가며 다항 시간에 계산됩니다. 사람은 선분의 교차를 시각 처리 초기에 알아차리므로, 구간마다 그림 하나를 보면 충돌 없음이 바로 확인됩니다(Figure 1).

**문제 1.** 주어진 $$r$$에 대해 index가 $$r$$ 이하인 계획을 찾거나 없음을 답하라. 결정 문제는 에이전트 2명에서도 NP-완전이고, 어려움이 에이전트 수뿐 아니라 **환경 크기**에 대한 것입니다(고전 MAPF는 에이전트 수가 고정되면 다항).

## 2. 방법

### 고수준: 분할 충돌

CBS는 각 에이전트를 독립적으로 A*로 계획하고(루트), 충돌이 있으면 "에이전트 $$i$$는 시각 $$t$$에 정점 $$v$$에 있을 수 없다"는 제약을 가진 두 자식을 만들어 재계획하는 두 단계 탐색입니다. XG-CBS는 충돌이 없지만 **index가 $$r$$보다 큰 계획**도 충돌로 봅니다. 탐욕 분해에서 구간 $$[t_{\ell-1}, t_\ell]$$을 $$t_\ell$$까지 늘리지 못하게 만든 쌍 $$(i, j)$$와 정점 $$v$$가 있으므로, 에이전트 $$i$$가 $$T_i$$에 $$v$$를 방문하는 것과 $$j$$가 $$T_j$$에 $$v$$를 방문하는 것을 각각 금지하는 두 자식을 만듭니다. 노드 비용은 **index를 1차 기준**(충돌은 구간의 끝으로 간주), 경로 길이 합을 2차 기준으로 두어 설명 가능한 계획을 먼저 탐색합니다. 저수준 경로 길이에는 NP 소속 증명에서 나온 상한을 두어 트리를 유한하게 만들고, 그 덕에 XG-CBS는 **완전**합니다(정리 1).

### 저수준: 어떤 A*를 쓸 것인가

표준 A*는 최단 경로를 찾으므로 분할 충돌을 **아주 국소적으로** 피합니다. 교차 지점을 비켜도 같은 경로를 근처에서 다시 만나 새 분할 충돌이 생기고, 트리가 많이 커집니다. 대신 매우 빠릅니다.

- **XG-A\*.** 노드를 $$(v, t, H, i)$$로 확장합니다. $$H$$는 마지막 분할 이후의 방문 정점 이력, $$i$$는 다른 에이전트의 경로 $$P_{-1}$$과 합친 현재 index입니다. 1차 휴리스틱은 index, 2차는 목표까지의 거리입니다. index 1인 경로를 모두 뒤진 뒤에야 index 2로 넘어가므로 **최소 index 경로로 강하게 유도**되지만, 목표로 가는 휴리스틱이 아니라 그래프 크기에 대해 느립니다(Remark 2). 또 다른 에이전트의 교차가 만든 분할 시점에 맞춰 자신의 교차를 **동기화**해 새 구간을 만들지 않는 행동이 나옵니다(Remark 1). 완전합니다(정리 2).
- **WXG-A\*.** $$f_w = w f_x + (1 - w) f_a$$로 두 비용을 섞습니다. 경로 길이가 index보다 훨씬 크므로 $$w$$가 매우 크지 않으면 길이가 지배합니다. 최적 $$w$$는 인스턴스마다 달라 미리 정하기 어렵고, 이력 $$H$$를 유지하므로 여전히 느립니다.
- **SR-A\*(Segmentation-Respecting).** 이력을 버리는 대신, 다른 에이전트의 경로를 **구간별 시간 제한 장애물**로 봅니다. $$P_{-1}$$의 어떤 경로가 구간 $$[t_i, t_{i+1}]$$ 안에 정점 $$v$$를 지나면, $$v$$는 그 구간 내내 장애물입니다. 이 장애물을 피하면 새 구간이 생기지 않으므로 기존 분할을 "존중"합니다. 탐색 공간이 줄어 **완전성은 잃지만** 속도가 A* 수준입니다.

## 3. 코드로 보기: 최단 계획과 설명 가능한 계획

2차선 도로가 교차하는 8×8 지도에서 네 에이전트(동·서·남·북 방향)가 교차로를 지납니다. 합계 비용 최소 CBS 계획의 index를 계산하고, "같은 그룹 안에서는 경로가 완전히 분리되고 그룹은 순서대로 움직인다"는 제약으로 다시 계획해 index와 길이의 트레이드오프를 봅니다.

```python
import heapq, itertools
# an 8x8 map with a two-lane road crossing: rows 3-4 and columns 3-4 are road, the rest is blocked
free = {(r, c) for r in range(8) for c in range(8) if r in (3, 4) or c in (3, 4)}
agents = [((3, 0), (3, 7)), ((4, 7), (4, 0)), ((0, 3), (7, 3)), ((7, 4), (0, 4))]     # east, west, south, north
names = ["east", "west", "south", "north"]
def nbrs(v): r, c = v; return [u for u in [(r, c), (r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)] if u in free]

def astar(start, goal, blocked, t0=0, maxT=40):
    # blocked: set of (vertex, t) the agent may not occupy; the agent waits at start until t0
    h = lambda v: abs(v[0] - goal[0]) + abs(v[1] - goal[1])
    pq = [(t0 + h(start), t0, start, [start] * (t0 + 1))]; seen = set()
    while pq:
        f, t, v, path = heapq.heappop(pq)
        if v == goal and all((goal, s) not in blocked for s in range(t, maxT)): return path
        if (v, t) in seen or t >= maxT: continue
        seen.add((v, t))
        for u in nbrs(v):
            if (u, t + 1) in blocked: continue
            heapq.heappush(pq, (t + 1 + h(u), t + 1, u, path + [u]))
    return None
def at(path, t): return path[min(t, len(path) - 1)]
def first_conflict(paths):
    T = max(map(len, paths))
    for t in range(T):
        for i, j in itertools.combinations(range(len(paths)), 2):
            if at(paths[i], t) == at(paths[j], t): return (i, j, at(paths[i], t), t)
    return None
def cbs():
    root = {i: set() for i in range(4)}
    paths = [astar(s, g, root[i]) for i, (s, g) in enumerate(agents)]
    pq = [(sum(map(len, paths)), 0, root, paths)]; k = 1
    while pq:
        cost, _, cons, paths = heapq.heappop(pq)
        c = first_conflict(paths)
        if c is None: return paths
        i, j, v, t = c
        for a in (i, j):
            new = {x: set(s) for x, s in cons.items()}; new[a].add((v, t))
            p = astar(agents[a][0], agents[a][1], new[a])
            if p is None: continue
            np_ = list(paths); np_[a] = p; heapq.heappush(pq, (sum(map(len, np_)), k, new, np_)); k += 1
def decompose(paths):                       # greedy vertex-disjoint decomposition -> segments (t0, t1)
    T = max(map(len, paths)); cuts = []; t0 = 0
    while t0 < T:
        t1 = t0
        while t1 < T:
            segs = [set(at(p, t) for t in range(t0, t1 + 1)) for p in paths]
            if any(segs[i] & segs[j] for i, j in itertools.combinations(range(len(paths)), 2)): break
            t1 += 1
        cuts.append((t0, max(t1, t0 + 1))); t0 = max(t1, t0 + 1)
    return cuts
def show(label, paths):
    print(f"{label}: path lengths {[len(p) - 1 for p in paths]}, makespan {max(map(len, paths)) - 1}, index {len(decompose(paths))}, segments {decompose(paths)}")

plan = cbs(); show("CBS (sum of costs)          ", plan)

# explanation-guided planning: order the agents into groups that run one after another; within a group every route
# must be vertex-disjoint for the whole plan, so each group is one picture (one segment)
def grouped_plan(groups):
    paths = {}; t0 = 0
    for g in groups:
        blocked = set()
        for i, p in paths.items():                                                                     # earlier groups: their routes while moving,
            blocked |= {(v, t) for v in set(p) for t in range(0, len(p))}                             # then only their goal cells
            blocked |= {(p[-1], t) for t in range(len(p), 60)}
        for i in range(4):
            if i not in paths and i not in g: blocked |= {(agents[i][0], t) for t in range(0, 60)}    # later groups wait on their start cells
        for i in g:
            p = astar(agents[i][0], agents[i][1], blocked, t0=t0, maxT=60)
            if p is None: return None
            paths[i] = p; blocked |= {(v, t) for v in set(p) for t in range(0, 60)}                    # same group: fully disjoint
        t0 = max(len(paths[i]) for i in g)
    return [paths[i] for i in range(4)]
best = None
for k in (1, 2):                             # try one group (index 1), then two groups
    for perm in itertools.permutations(range(4)):
        for cut in ([4] if k == 1 else [1, 2, 3]):
            groups = [perm[:cut], perm[cut:]] if k == 2 else [perm]
            groups = [g for g in groups if g]
            cand = grouped_plan(groups)
            if cand and first_conflict(cand) is None and (best is None or sum(map(len, cand)) < sum(map(len, best[0]))): best = (cand, groups)
    if best: break
if best:
    cand, groups = best
    show("explanation-guided (XG) plan", cand)
    print("groups:", " then ".join("{" + ", ".join(names[i] for i in g) + "}" for g in groups))
    print("-> the shortest plan interleaves four crossings and needs several pictures; letting the two horizontal agents")
    print("   cross first and the two vertical agents afterwards gives a plan a person can verify from two pictures")
```

```text
CBS (sum of costs)          : path lengths [8, 8, 7, 7], makespan 8, index 3, segments [(0, 4), (4, 5), (5, 9)]
explanation-guided (XG) plan: path lengths [7, 7, 15, 15], makespan 15, index 2, segments [(0, 11), (11, 16)]
groups: {east, west} then {south, north}
-> the shortest plan interleaves four crossings and needs several pictures; letting the two horizontal agents
   cross first and the two vertical agents afterwards gives a plan a person can verify from two pictures
```

- 최단 계획은 네 에이전트가 모두 7~8스텝 안에 교차로를 지나므로 makespan 8이지만, 설명하려면 **그림 3장**이 필요합니다. 교차로 네 칸을 시간차로 공유하기 때문입니다.
- 가로 두 에이전트가 먼저 지나고 세로 두 에이전트가 기다렸다 지나는 계획은 makespan이 15로 두 배 가까이 길지만 **그림 2장**으로 끝납니다. 각 그림 안에서 두 경로는 서로 다른 차선이라 겹치지 않습니다. 가로·세로 경로는 평면 그래프에서 반드시 교차하므로 이 지도에서 index 1은 불가능하고, 2가 최소입니다. 논문의 Figure 4(도로 교차)가 보여 주는 "최단 계획은 index 2, 더 긴 계획은 index 1"과 같은 종류의 트레이드오프입니다.

## 4. 실험

**사례 연구.** 도로 교차(Figure 4)에서 CBS 계획은 시각적으로 검증하기 어렵고 분해하면 2장이 필요하지만, XG-CBS는 더 긴 index-1 계획을 **0.05초**에 찾았습니다. 표준 A*를 저수준으로 쓴 XG-CBS는 15분 안에 끝나지 않았는데, index-1 계획이 드물고 최단 경로에서 많이 벗어나 있기 때문입니다. "겉보기 충돌"(Figure 5)에서도 XG-A*가 0.5초에 더 나은 설명을 찾았습니다.

**벤치마크.** 9×9(4~12명), 16×16(5~20명), 33×33(10~30명) 격자, 조합마다 100개 인스턴스, 알고리즘당 5분 제한. CBS로 먼저 풀어 그 index를 상한으로 XG-CBS를 돌리고(first), 상한을 낮춰 가며 시간 초과 직전까지 간 결과(best)를 봅니다.

- **CBS**: 길이와 시간은 최선이지만 index가 항상 높습니다.
- **A\***: 빠른 덕분에 결국 XG-A*보다 작은 index에 도달하기도 하며 경로가 짧지만, 비용을 낮추다 보면 시간 초과합니다.
- **XG-A\***: 처음부터 낮은 index를 내고 9×9·12명처럼 좁고 붐비는 환경에서는 A*를 이기지만, 이력 의존성 때문에 33×33에서는 거의 항상 시간 초과합니다. WXG-A*도 같은 이유로 큰 환경에서 못 씁니다.
- **SR-A\***: 이론상 불완전한데도 성공률이 CBS와 같고, index를 크게 줄이며(33×33·30명에서 **약 24 → 6 세그먼트**), 계산 시간도 CBS와 비슷하거나 더 빠릅니다. 대가는 경로 길이 증가입니다.

결론: 큰 환경에서 설명 가능성을 원하면 SR-A*가 확실한 선택이고, 작은 환경에서는 A*가 무난하며 붐비면 XG-A*가 낫고, $$w$$를 손으로 맞출 수 있으면 WXG-A*가 가장 작은 index를 줍니다.

## 5. 해석

**1. 설명 가능성을 "최적화 목표"로 번역한 것이 핵심입니다.** XAI의 많은 연구가 사후 설명을 만들지만, 이 논문은 설명하기 쉬운 계획을 **처음부터** 찾습니다. 그림 수라는 지표는 사람의 시각 처리(선분 교차 인지)에 근거를 두고 있어 자의적이지 않습니다.

**2. 완전성과 확장성의 교환이 솔직하게 드러납니다.** 완전한 XG-A*는 그래프 크기에 지수적이라 문제의 본질적 어려움(NP-hard in environment size)을 피할 수 없고, 불완전한 SR-A*가 실제로는 가장 쓸모 있습니다. [PRIMAL]({% post_url 2026-10-06-primal %})이 학습으로 완전성을 포기하고 확장성을 얻은 것과 같은 선택을 탐색 알고리즘 안에서 한 셈입니다.

**3. 위 실험처럼 설명 가능성은 공짜가 아닙니다.** index를 줄이면 makespan과 비용이 늘어납니다. 현장에서는 "그림 2장으로 승인받는 계획"과 "최단이지만 검증에 긴 시간이 드는 계획" 중 무엇이 전체 운영 시간을 줄이는지가 기준이 되어야 하고, 논문은 그 비용 모델까지는 제시하지 않습니다.

**4. 응용은 신뢰를 넘어섭니다.** 저자들이 짚듯 테더(줄)로 연결된 로봇·드론은 경로가 꼬이면 안 되고, 3D 배관 라우팅이나 다층 회로 기판 설계에서는 교차가 적은 계획이 직접 비용 절감입니다. index는 이런 문제에서 **물리적 제약**이 됩니다.

**5. 평가를 읽을 때 주의할 점.** 벤치마크는 무작위 격자이고 ICBS 같은 CBS 개선판과의 비교는 없습니다(저자들은 의도적으로 바닐라 CBS를 썼다고 밝힙니다). WXG-A*는 $$w$$ 튜닝이 인스턴스 의존이라 일괄 실험이 성능을 대변하지 못합니다. 위 toy의 "그룹 순차 계획"은 논문의 SR-A*가 아니라 그 직관을 단순화한 휴리스틱입니다.

## 6. 정리

| 항목 | 내용 |
|---|---|
| 문제 | 계획을 정점 분리 시간 구간 $$r$$개로 설명; index $$\le r$$인 계획 찾기는 2명에서도 NP-완전(환경 크기에 대해) |
| 고수준 | CBS 제약 트리에 분할 충돌 추가, 비용 = index(1차) + 길이(2차), 경로 길이 상한으로 완전성 |
| 저수준 | A*(빠름, 국소 회피) / XG-A*(index 최소화, 완전, 느림) / WXG-A*(가중 혼합) / SR-A*(구간별 시간 장애물, 불완전, 빠름) |
| 결과 | 도로 교차 index-1 계획 0.05초; 33×33·30명에서 SR-A*가 index 24 → 6, 시간은 CBS 수준 |
| 트레이드오프 | index ↓ ⇔ 경로 길이 ↑ |
| 응용 | 감독자 승인, 테더 로봇, 배관·PCB 라우팅 |

## References

- Kottinger, Almagor, Lahijanian, "Conflict-Based Search for Explainable Multi-Agent Path Finding," ICAPS/AAAI 2022. [arXiv:2202.09930](https://arxiv.org/abs/2202.09930)
- Almagor, Lahijanian, "Explainable Multi Agent Path Finding," AAMAS 2020.
- Sharon, Stern, Felner, Sturtevant, "Conflict-based search for optimal multi-agent pathfinding," Artificial Intelligence 2015.
- Boyarski et al., "ICBS: Improved Conflict-Based Search Algorithm for Multi-Agent Pathfinding," IJCAI 2015.
- Stern et al., "Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks," SoCS 2019. [arXiv:1906.08291](https://arxiv.org/abs/1906.08291)
