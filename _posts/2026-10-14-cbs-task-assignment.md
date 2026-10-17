---
title: "[MAPF] CBS-TA: 작업 할당과 다중 에이전트 경로 계획을 한 번에"
date: 2026-10-14 21:50:00 +0900
categories: [Reinforcement Learning, Multi-Agent]
tags: [multi-agent, mapf, cbs, conflict-based-search, task-assignment, tapf, search, planning]
description: 다중 에이전트 경로 계획(MAPF)의 대표 알고리즘 CBS를 실행 가능한 코드로 복습하고, 작업 할당과 경로 계획을 search forest로 동시에 최적화하는 CBS-TA와 ECBS-TA를 정리합니다.
math: true
---

> 이 글은 강화학습이 아니라 **탐색(search) 기반 계획 알고리즘**입니다. 하지만 [PRIMAL]({% post_url 2026-10-04-primal %}) 같은 학습 기반 MAPF가 비교 대상으로 삼는 대표적인 최적 계획기라서 Multi-Agent 카테고리에 함께 정리합니다. *Conflict-Based Search with Optimal Task Assignment*(Hönig et al., AAMAS 2018)는 "**누가 어느 목표로 갈지**"와 "**각자 어떤 경로로 갈지**"를 동시에 최적화합니다.

## 1. 문제 정의

### MAPF

**무방향 그래프** $$G = (V, E)$$ 위에서 여러 에이전트가 이동합니다.

- $$V$$: 위치(격자 칸)
- $$E$$: 단위 가중치의 간선(통로)
- 매 시점 각 에이전트는 이웃 칸으로 이동하거나 제자리에 머묾
- **충돌 금지**: 두 에이전트가 같은 시점에 같은 칸에 있으면 안 되고(vertex conflict), 같은 간선을 서로 반대로 지나가도 안 됨(edge conflict)
- 목표: 모든 에이전트의 **경로 비용 합(sum of costs)** 최소화

### Labeled MAPF vs TAPF

- **Labeled MAPF**: 각 에이전트의 목표가 **정해져** 있음. 기존 연구 대부분이 여기에 해당하며 CBS, M\*, ECBS 등으로 풂
- **TAPF (Target Assignment and Path Finding)**: 목표 집합만 주어지고, **어느 에이전트가 어느 목표로 갈지도 정해야** 함. 창고에서 "선반 5개를 누가 가져올지"가 정해지지 않은 상황

기존에는 TAPF를 **두 단계**로 풀었습니다.

1. **작업 할당**: 거리 기반으로 에이전트–목표 매칭 (헝가리안 알고리즘 등)
2. **경로 계획**: 정해진 할당으로 MAPF

하지만 거리만 보고 할당하면, 경로를 짤 때 **충돌이 많아 실제로는 비싼** 할당이 될 수 있습니다. 논문은 둘을 **동시에** 탐색해 최적해를 구합니다.

## 2. CBS 복습

**CBS(Conflict-Based Search)**(Sharon et al., 2015)는 두 단계 탐색입니다.

### Low-level

각 에이전트의 경로를 **혼자 있다고 가정**하고 A\*로 구합니다. 단, 상위에서 받은 **제약 조건**(예: "에이전트 1은 시점 3에 칸 (2, 4)에 있으면 안 됨")을 지킵니다.

### High-level: Constraint Tree

1. 루트 노드: 제약 없이 각자 최단 경로를 구함
2. 비용이 가장 작은 노드를 꺼내 경로들 사이의 **첫 충돌**을 찾음
3. 충돌이 없으면 → **해답**
4. 충돌 $$(a_i, a_j, v, t)$$가 있으면 노드를 **둘로 분기**
   - 왼쪽: $$a_i$$는 시점 $$t$$에 $$v$$에 있으면 안 됨
   - 오른쪽: $$a_j$$는 시점 $$t$$에 $$v$$에 있으면 안 됨
5. 각 자식에서 제약이 추가된 에이전트만 low-level을 다시 풀고, 2로 돌아감

비용 순서로 노드를 꺼내므로 처음 찾은 해가 **최적**입니다. (low-level 탐색이 각 에이전트의 최단 경로를 정확히 구한다는 전제에서입니다.)

```python
import heapq, itertools

def cbs(agents, starts, goals, low_level, find_first_conflict):
    tie = itertools.count()
    root = {"constraints": {a: set() for a in agents}}
    root["paths"] = {a: low_level(starts[a], goals[a], root["constraints"][a]) for a in agents}
    root["cost"] = sum(len(p) for p in root["paths"].values())
    open_list = [(root["cost"], next(tie), root)]
    while open_list:
        _, _, node = heapq.heappop(open_list)
        conflict = find_first_conflict(node["paths"])
        if conflict is None:
            return node["paths"]                                  # 충돌 없음 = 최적해
        a_i, a_j, v, t = conflict
        for agent in (a_i, a_j):                                  # 두 갈래로 분기
            child = {"constraints": {k: set(c) for k, c in node["constraints"].items()},
                     "paths": dict(node["paths"])}
            child["constraints"][agent].add((v, t))
            child["paths"][agent] = low_level(starts[agent], goals[agent], child["constraints"][agent])
            if child["paths"][agent] is not None:
                child["cost"] = sum(len(p) for p in child["paths"].values())
                heapq.heappush(open_list, (child["cost"], next(tie), child))
    return None
```

위 함수만으로는 실행할 수 없으니, 3×3 격자에서 돌려 볼 수 있는 최소한의 low-level 탐색과 충돌 검사를 붙여 봅니다.

```python
from collections import deque

GRID = ["...",
        "...",
        "..."]                                            # '#'를 넣으면 장애물
H, W = len(GRID), len(GRID[0])

def neighbors(v):
    r, c = v
    for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:   # 제자리 포함
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and GRID[nr][nc] != "#":
            yield (nr, nc)

def low_level(start, goal, constraints, max_t=20):
    """시간까지 포함한 BFS: constraints = {(칸, 시점)} 금지 목록"""
    queue, seen = deque([(start, 0, [start])]), {(start, 0)}
    while queue:
        v, t, path = queue.popleft()
        if v == goal and all((goal, k) not in constraints for k in range(t, max_t)):
            return path                                   # 도착 후 계속 머물러도 제약 위반이 없으면 완료
        if t >= max_t:
            continue
        for u in neighbors(v):
            if (u, t + 1) not in constraints and (u, t + 1) not in seen:
                seen.add((u, t + 1))
                queue.append((u, t + 1, path + [u]))
    return None

def find_first_conflict(paths):
    """같은 시점에 같은 칸에 있는 첫 충돌(vertex conflict)만 찾음"""
    T = max(len(p) for p in paths.values())
    at = lambda p, t: p[min(t, len(p) - 1)]               # 도착한 뒤에는 목표에 머문다고 가정
    agents = list(paths)
    for t in range(T):
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                if at(paths[a], t) == at(paths[b], t):
                    return a, b, at(paths[a], t), t
    return None

starts = {"A": (1, 0), "B": (0, 1)}     # A는 왼쪽→오른쪽, B는 위→아래로 가운데 칸을 지나감
goals = {"A": (1, 2), "B": (2, 1)}
independent = {a: low_level(starts[a], goals[a], set()) for a in starts}
print(find_first_conflict(independent))   # ('A', 'B', (1, 1), 1) → 시점 1에 가운데서 충돌
print(cbs(["A", "B"], starts, goals, low_level, find_first_conflict))
```

```text
('A', 'B', (1, 1), 1)
{'A': [(1, 0), (1, 0), (1, 1), (1, 2)], 'B': [(0, 1), (1, 1), (2, 1)]}
```

혼자일 때의 최단 경로는 시점 1에 가운데 칸 (1, 1)에서 부딪칩니다. CBS는 "A는 시점 1에 (1, 1)에 있으면 안 됨" / "B는 …" 두 갈래로 나눈 뒤, 비용 합이 더 작은 해(A가 한 칸 기다림, 총 비용 +1)를 찾습니다.

> 이 예제의 `find_first_conflict`는 **같은 칸 충돌(vertex conflict)만** 검사합니다. 실제 MAPF에서는 두 에이전트가 같은 간선을 반대 방향으로 지나가는 **맞바꾸기(edge conflict)**도 검사하고, 이때는 "에이전트가 시점 $$t$$에 $$u \to v$$로 이동하면 안 됨" 같은 방향이 있는 제약을 추가해야 합니다.
{: .prompt-warning }

## 3. CBS-TA: Search Forest

### 아이디어

할당이 정해지지 않았으므로, **가능한 할당마다 하나의 constraint tree**를 둡니다. 여러 트리의 숲, 즉 **search forest**입니다.

- 각 트리의 루트 = 특정 **에이전트–목표 할당** + 그 할당에서의 제약 없는 최단 경로
- 모든 트리의 노드를 **하나의 우선순위 큐**에 넣고 비용 순서로 탐색

### 필요할 때만 트리 만들기

할당의 수는 $$n!$$이라 처음부터 모든 트리를 만들 수 없습니다. 그래서 루트를 **필요할 때(lazily)** 만듭니다.

1. 처음에는 **비용이 가장 작은 할당** 하나로 루트를 만듦 (헝가리안 알고리즘)
2. 어떤 트리의 루트를 꺼낼 때마다, **그다음으로 비용이 작은 할당**으로 새 루트를 만들어 큐에 추가
3. 다음 순위의 할당은 **순위 할당(ranked assignment)** 알고리즘(Murty의 방법)으로 차례대로 구함

이렇게 하면 할당을 "경로 충돌까지 고려한 실제 비용" 순서로 탐색하게 되고, 처음 찾은 충돌 없는 해가 할당과 경로를 모두 고려한 **최적해**가 됩니다.

### ECBS-TA

최적해를 구하는 대신 **최적 비용의 $$w$$배 이내**의 해를 빠르게 구하는 bounded-suboptimal 버전입니다. ECBS처럼 **focal search**를 써서, 비용이 상한 이내인 노드들 중 **충돌이 적은 노드**를 먼저 탐색합니다. 에이전트가 많을 때 훨씬 빠릅니다.

## 4. 실험

- **밀집된 환경**(좁은 공간에 장애물이 많음)에서 **에이전트 수가 비교적 적은** 경우를 기준으로 평가했습니다.
- 두 단계 방식(할당 후 경로 계획)과 비교해, CBS-TA는 **비용 합 기준 최적해**를 보장하면서 실용적인 시간 안에 해를 찾았습니다.
- 기존 TAPF 방법 중 **makespan**(가장 늦게 도착하는 에이전트의 시간) 최적화 방법과 비교하면, 목적이 다르므로 비용 합에서는 CBS-TA가 더 좋았습니다.
- ECBS-TA는 품질을 조금 양보하는 대신 더 많은 에이전트로 확장됐습니다.

## 5. 정리

| | 두 단계 방식 | CBS-TA |
|---|---|---|
| 할당 | 거리만 보고 먼저 결정 | 경로 충돌까지 고려해 **함께 탐색** |
| 구조 | 하나의 constraint tree | 할당마다 트리, **search forest** |
| 할당 열거 | 한 번 | 필요할 때 다음 순위 할당 생성 |
| 보장 | 할당이 고정되어 최적 아님 | 비용 합 기준 **최적** (ECBS-TA는 $$w$$-준최적) |

학습 기반 [PRIMAL]({% post_url 2026-10-04-primal %})은 확장성을, CBS 계열은 **최적성**을 장점으로 가집니다. 실제 시스템에서는 둘을 섞어 쓰기도 합니다.

## References

- Hönig et al., "Conflict-Based Search with Optimal Task Assignment," AAMAS 2018.
- Sharon et al., "Conflict-based search for optimal multi-agent pathfinding," AIJ 2015.
- Barer et al., "Suboptimal Variants of the Conflict-Based Search Algorithm for the Multi-Agent Pathfinding Problem (ECBS)," SoCS 2014.
- Ma & Koenig, "Optimal Target Assignment and Path Finding for Teams of Agents," AAMAS 2016.
- Murty, "An Algorithm for Ranking all the Assignments in Order of Increasing Cost," Operations Research 1968.
