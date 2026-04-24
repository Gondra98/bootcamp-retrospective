---
aliases:
  - "Day 55_자료구조 : Deque · Tree · BST · Graph"
---
# Day 55_자료구조 : Deque · Tree · BST · Graph
## 📅 2026-04-23

---
# 📄 stru5_dequeue.py — Deque (데큐)

## 📌 개념 정리

### Deque란?

**Deque (Double-Ended Queue)** 는 **양쪽 끝에서 삽입과 삭제가 모두 가능한 자료구조**다.

- Queue(앞에서만 제거) + Stack(같은 쪽 삽입·삭제)의 특성을 모두 가짐
- Python에서는 `collections.deque`로 구현 (`list`보다 양쪽 삽입/삭제가 O(1)로 빠름)
- `maxlen` 옵션으로 고정 크기 설정 가능 → 슬라이딩 윈도우 구현에 적합

**실생활 예시** : 놀이공원 VIP 우선 탑승, 실시간 로그 분석, 이동 평균 계산

```
appendleft →  [ VIP지수 | 철수 | 영희 | 민수 ]  ← append
   popleft ←  [                              ]  → pop
              front                          back
```

---

### 핵심 용어

|용어|설명|
|---|---|
|`append(x)`|오른쪽(back)에 데이터 추가 — 일반 Queue처럼 동작|
|`appendleft(x)`|왼쪽(front)에 데이터 추가 — VIP처럼 앞에 삽입|
|`pop()`|오른쪽(back)에서 데이터 제거 및 반환 — Stack처럼 동작|
|`popleft()`|왼쪽(front)에서 데이터 제거 및 반환 — Queue처럼 동작|
|`maxlen`|최대 크기 제한 — 초과 시 반대쪽 요소 자동 제거|

---

### Queue · Stack · Deque 비교

|구분|Queue|Stack|**Deque**|
|---|---|---|---|
|구조|FIFO|LIFO|FIFO + LIFO|
|삽입 위치|back(뒤)|top(위)|**양쪽 모두**|
|제거 위치|front(앞)|top(위)|**양쪽 모두**|
|실생활 예시|대기 줄|뒤로 가기|VIP 대기열, 로그|
|Python 구현|`deque`|`list`|`deque`|

> Queue와 Stack은 한쪽 방향으로만 동작하지만,  
> **Deque는 양쪽 어느 방향으로도 삽입·제거가 가능**하다.

---

## 💻 전체 실습 코드

### 1단계 - 일반 대기열 추가 (`append`)

```python
from collections import deque   # 파이썬 내장 모듈

dq = deque()  # 빈 데큐 생성
print('놀이공원 대기 시작')

# 일반인은 뒤쪽으로 들어옴 (Queue처럼 FIFO)
dq.append('철수')
dq.append('영희')
dq.append('민수')
print('일반 대기 : ', list(dq))   # ['철수', '영희', '민수']
print()
```

**출력 결과**

```
놀이공원 대기 시작
일반 대기 :  ['철수', '영희', '민수']
```

---

### 2단계 - VIP 우선 삽입 (`appendleft`)

```python
# VIP 고객(지수)은 앞쪽으로 들어옴 — Deque의 핵심 기능
dq.appendleft('VIP지수')
print('현재 대기 줄 상태: ', list(dq))   # ['VIP지수', '철수', '영희', '민수']
print()
```

**동작 과정**

```
append 3회   : [철수 | 영희 | 민수]
appendleft   : [VIP지수 | 철수 | 영희 | 민수]
                ↑ 앞에 삽입
```

---

### 3단계 - 앞쪽 제거 (`popleft` — Queue 방식)

```python
# 놀이기구에 탑승 — 맨 앞 사람부터 (FIFO)
person = dq.popleft()
print(person, '탑승')                          # VIP지수 탑승
print('현재 대기 줄 상태2: ', list(dq))         # ['철수', '영희', '민수']
print()
```

---

### 4단계 - 뒤쪽 제거 (`pop` — Stack 방식)

```python
# 줄 맨 뒷사람 줄서기 포기 — 맨 뒤 사람 제거 (LIFO)
person = dq.pop()
print(person, ': 줄서기 포기')                  # 민수 : 줄서기 포기
print('현재 대기 줄 상태3: ', list(dq))         # ['철수', '영희']
```

---

### 전체 실행 결과

```
놀이공원 대기 시작
일반 대기 :  ['철수', '영희', '민수']

현재 대기 줄 상태:  ['VIP지수', '철수', '영희', '민수']

VIP지수 탑승
현재 대기 줄 상태2:  ['철수', '영희', '민수']

민수 : 줄서기 포기
현재 대기 줄 상태3:  ['철수', '영희']
```

---

## 📊 동작 과정 전체 시각화

```
① append 3회
   front → [철수 | 영희 | 민수] ← back

② appendleft('VIP지수')
   front → [VIP지수 | 철수 | 영희 | 민수] ← back

③ popleft()  →  VIP지수 탑승
   front → [철수 | 영희 | 민수] ← back

④ pop()  →  민수 줄서기 포기
   front → [철수 | 영희] ← back
```

---

## 심화 — `maxlen` 옵션

```python
# 고정 크기 데큐 : 용량 초과 시 반대쪽 요소가 자동으로 밀려남
# → 최신 N개 데이터만 유지하는 슬라이딩 윈도우 구현에 최적

log = deque(maxlen=3)

log.append('로그1')  # ['로그1']
log.append('로그2')  # ['로그1', '로그2']
log.append('로그3')  # ['로그1', '로그2', '로그3']
log.append('로그4')  # ['로그2', '로그3', '로그4'] ← 로그1 자동 제거

print(list(log))     # ['로그2', '로그3', '로그4']
```

**동작 원리**

```
maxlen=3 설정 시, 새 요소 추가 → 반대쪽 요소 자동 제거

append('로그4') :
  before : [로그1 | 로그2 | 로그3]
  after  :         [로그2 | 로그3 | 로그4]
                    ↑ 로그1 자동 제거
```

> `maxlen` 초과 시 별도 삭제 코드 없이 자동 관리됨  
> 실시간 로그 분석, 이동 평균 계산 등에 활용 가능

---

## 📊 메소드 요약

|메소드|방향|설명|시간복잡도|
|---|---|---|---|
|`append(x)`|→ 오른쪽 삽입|일반 enqueue|O(1)|
|`appendleft(x)`|← 왼쪽 삽입|VIP 앞 삽입|O(1)|
|`popleft()`|← 왼쪽 제거|앞사람 탑승 (Queue)|O(1)|
|`pop()`|오른쪽 제거 →|뒷사람 이탈 (Stack)|O(1)|

> `list.pop(0)`은 O(n)이지만 `deque.popleft()`는 **O(1)**  
> Queue 구현 시 `list` 대신 `deque`를 써야 하는 이유

---

## 🔑 핵심 포인트

> Deque는 **양쪽 모두** 삽입·삭제 가능 — Queue + Stack의 결합  
> `appendleft` / `popleft` 로 앞쪽, `append` / `pop` 으로 뒤쪽 조작  
> `maxlen` 설정 시 용량 초과 요소가 **반대쪽에서 자동 제거** → 슬라이딩 윈도우에 적합  
> 모든 삽입·삭제 연산이 **O(1)** — `list.pop(0)`의 O(n) 단점 없음

---
# 📄 stru6_tree.py — Tree (트리)

## 📌 개념 정리

### Tree란?

**Tree** 는 **노드들이 나무가지처럼 연결된 계층적 비선형 자료구조**다.

- **시작점(루트)이 1개**이고 **순환(cycle)이 없다**
- 부모 → 자식 방향으로만 연결 — 자식이 부모를 가리키는 역방향 없음
- 연결 리스트·스택·큐와 달리 **비선형** 구조 (1:N 관계)

**실생활 예시** : 회사 조직도, 파일 시스템 폴더 구조, HTML DOM 트리

```
               [CEO]              ← 루트 (Root)
              /   |   \
       [개발팀장][기획팀장][영업팀장]  ← 내부 노드
        /   \       |       /   \
  [백엔드][프론트] [서비스기획] [국내][해외]  ← 리프 (Leaf)
```

---

### 핵심 용어

|용어|설명|
|---|---|
|`루트 (Root)`|트리의 시작점 — 부모가 없는 유일한 노드|
|`노드 (Node)`|트리를 구성하는 각각의 데이터 단위|
|`부모 (Parent)`|자식 노드를 가지는 상위 노드|
|`자식 (Child)`|부모 노드에 연결된 하위 노드|
|`리프 (Leaf)`|자식이 없는 노드 — 트리의 끝|
|`레벨 (Level)`|루트로부터의 깊이 (루트 = 0)|
|`재귀 (Recursion)`|함수가 자기 자신을 호출 — 트리 탐색의 핵심|

---

### 선형 vs 비선형 자료구조

|구분|선형|**비선형 (Tree)**|
|---|---|---|
|관계|1:1 (한 줄로 연결)|**1:N (가지 형태)**|
|예시|Linked List, Stack, Queue|**Tree**, Graph|
|탐색 방식|순차 탐색|계층적 탐색 (재귀)|
|순환 여부|없음|**없음** (Graph와의 차이)|

> Tree는 **순환이 없는** 계층 구조 — 순환이 있으면 Graph가 된다

---

## 💻 전체 실습 코드

### 1단계 - 트리 데이터 정의 (딕셔너리)

```python
# 딕셔너리로 트리 표현 : key = 노드, value = 자식 노드 리스트
company = {
    'CEO':      ['개발팀장', '기획팀장', '영업팀장'],  # 루트 노드 — 자식 3개
    '개발팀장': ['백엔드', '프론트엔드'],              # 내부 노드 — 자식 2개
    '기획팀장': ['서비스기획'],                        # 내부 노드 — 자식 1개
    '영업팀장': ['국내영업', '해외영업'],              # 내부 노드 — 자식 2개
    '백엔드':   [],    # 리프 노드 — 자식 없음
    '프론트엔드': [],  # 리프 노드
    '서비스기획': [],  # 리프 노드
    '국내영업': [],    # 리프 노드
    '해외영업': []     # 리프 노드
}
```

> `value = []` 이면 리프 노드 — 자식이 없어 재귀가 더 이상 진행되지 않는다

---

### 2단계 - `showTree` 함수 (재귀 탐색)

```python
# 트리 구조 출력 함수
def showTree(node, level):
    print(' ' * level + ' - ' + node)  # level만큼 들여쓰기 → 계층 표현
    # 현재 노드의 자식을 순서대로 재귀 호출
    for child in company[node]:
        showTree(child, level + 1)      # level + 1 → 한 단계 더 들여씀
```

**재귀 동작 원리**

```
showTree('CEO', 0)
  → print ' - CEO'
  → showTree('개발팀장', 1)
      → print '  - 개발팀장'
      → showTree('백엔드', 2)
          → print '   - 백엔드'
          → company['백엔드'] = [] → 반복 없음 (재귀 종료)
      → showTree('프론트엔드', 2)
          → print '   - 프론트엔드'
          → 재귀 종료
  → showTree('기획팀장', 1) ...
  → showTree('영업팀장', 1) ...
```

> `company[node] = []` 일 때 `for` 루프가 실행되지 않아 재귀가 자동 종료된다  
> 이것이 트리 재귀 탐색의 **기저 조건 (Base Case)**

---

### 3단계 - 실행

```python
print('회사 조직도')
showTree('CEO', 0)
```

**출력 결과**

```
회사 조직도
 - CEO
  - 개발팀장
   - 백엔드
   - 프론트엔드
  - 기획팀장
   - 서비스기획
  - 영업팀장
   - 국내영업
   - 해외영업
```

---

## 📊 동작 과정 시각화

**레벨(Level) 구조**

```
Level 0 :          CEO
                  / | \
Level 1 :  개발팀장 기획팀장 영업팀장
            / \      |      / \
Level 2 : 백엔드 프론트 서비스기획 국내 해외
```

**들여쓰기 = 레벨**

```
' ' * 0 + ' - CEO'        →  - CEO
' ' * 1 + ' - 개발팀장'   →   - 개발팀장
' ' * 2 + ' - 백엔드'     →    - 백엔드
```

---

## 📊 메소드/구조 요약

|항목|내용|
|---|---|
|자료구조|`dict` — key: 노드명, value: 자식 리스트|
|탐색 방식|**DFS (깊이 우선 탐색)** — 재귀로 자식을 끝까지 탐색 후 다음 형제로 이동|
|재귀 종료 조건|`company[node] == []` — 자식이 없는 리프 노드|
|들여쓰기|`' ' * level` — 레벨이 깊을수록 더 들여씀|

---

## 🔑 핵심 포인트

> Tree는 **비선형** 자료구조 — 1:N 관계, 순환 없음  
> 딕셔너리로 `{노드: [자식 리스트]}` 형태로 트리를 표현할 수 있다  
> `showTree`는 **재귀** 로 동작 — 자식이 없는 리프 노드에서 자동 종료  
> 들여쓰기 `' ' * level` 로 계층 구조를 시각적으로 표현  
> 탐색 순서는 **DFS (깊이 우선)** — 한 가지를 끝까지 내려간 후 다음 가지로 이동

---

# 📄 stru7_btree.py — Binary Tree (이진 트리)

## 📌 개념 정리

### Binary Tree란?

**Binary Tree (이진 트리)** 는 **자식이 둘 이하(차수 ≤ 2)인 노드로 구성된 트리**다.

- 각 노드는 **왼쪽(left)** 과 **오른쪽(right)** 자식만 가질 수 있음
- 자식이 없으면 `None` 으로 표현
- 노드 방문 순서에 따라 **전위 · 중위 · 후위** 순회 3가지로 나뉨
- 모든 순회는 **DFS (깊이 우선 탐색)** 기반 + **재귀**로 구현

**실생활 예시** : 수식 파싱, 이진 탐색 트리(BST), 허프만 압축 코드

```
          [A]          ← 루트
         /   \
       [B]   [C]       ← 내부 노드
      /   \
    [D]   [E]          ← 리프 노드
```

---

### 핵심 용어

|용어|설명|
|---|---|
|`차수 (Degree)`|노드가 가진 자식 수 — 이진 트리는 최대 2|
|`left / right`|왼쪽·오른쪽 자식 노드|
|`None`|자식이 없음을 표현 — 재귀 종료 조건|
|`순회 (Traversal)`|트리의 모든 노드를 빠짐없이 방문하는 방법|
|`DFS`|깊이 우선 탐색 — 한 가지를 끝까지 내려간 후 다음 가지로|

---

### 순회 3가지 — 방문 순서 차이

|순회|순서|특징|
|---|---|---|
|**전위 (Pre-order)**|**루트 → 왼쪽 → 오른쪽**|루트를 가장 먼저 방문|
|**중위 (In-order)**|**왼쪽 → 루트 → 오른쪽**|BST에서 오름차순 정렬 결과|
|**후위 (Post-order)**|**왼쪽 → 오른쪽 → 루트**|루트를 가장 마지막에 방문|

> **루트를 언제 방문하느냐**가 세 순회의 유일한 차이다  
> 왼쪽 → 오른쪽 방향은 세 순회 모두 동일

---

## 💻 전체 실습 코드

### 1단계 - 이진 트리 데이터 정의

```python
# 딕셔너리로 이진 트리 표현
# key = 노드명, value = (왼쪽 자식, 오른쪽 자식) 튜플
tree = {
    'A': ('B', 'C'),      # A의 왼쪽=B, 오른쪽=C
    'B': ('D', 'E'),      # B의 왼쪽=D, 오른쪽=E
    'C': (None, None),    # C는 자식 없음 → 리프 노드
    'D': (None, None),    # D는 자식 없음 → 리프 노드
    'E': (None, None)     # E는 자식 없음 → 리프 노드
}
```

**트리 구조**

```
        A
       / \
      B   C
     / \
    D   E
```

---

### 2단계 - 전위 순회 `preOrder` (루트 → 왼쪽 → 오른쪽)

```python
def preOrder(node):
    if node is None:    # 기저 조건 : 자식이 없으면 종료
        return
    print(node, end=' ')        # ① 루트 먼저 출력
    left, right = tree[node]    # 왼쪽·오른쪽 자식 분리
    preOrder(left)              # ② 왼쪽 서브트리 재귀
    preOrder(right)             # ③ 오른쪽 서브트리 재귀
```

**동작 과정**

```
preOrder('A')
  ① print A
  ② preOrder('B')
      ① print B
      ② preOrder('D') → print D → None 종료
      ③ preOrder('E') → print E → None 종료
  ③ preOrder('C') → print C → None 종료

결과 : A B D E C
```

---

### 3단계 - 중위 순회 `inOrder` (왼쪽 → 루트 → 오른쪽)

```python
def inOrder(node):
    if node is None:    # 기저 조건
        return
    left, right = tree[node]
    inOrder(left)               # ① 왼쪽 서브트리 재귀 먼저
    print(node, end=' ')        # ② 루트 출력 (왼쪽 다 끝난 후)
    inOrder(right)              # ③ 오른쪽 서브트리 재귀
```

**동작 과정**

```
inOrder('A')
  ① inOrder('B')
      ① inOrder('D') → None 종료 → print D
      ② print B
      ③ inOrder('E') → None 종료 → print E
  ② print A
  ③ inOrder('C') → None 종료 → print C

결과 : D B E A C
```

> BST(Binary Search Tree)에서 중위 순회를 하면 **오름차순 정렬** 결과가 나온다

---

### 4단계 - 후위 순회 `postOrder` (왼쪽 → 오른쪽 → 루트)

```python
def postOrder(node):
    if node is None:    # 기저 조건
        return
    left, right = tree[node]
    postOrder(left)             # ① 왼쪽 서브트리 재귀 먼저
    postOrder(right)            # ② 오른쪽 서브트리 재귀
    print(node, end=' ')        # ③ 루트는 가장 마지막에 출력
```

**동작 과정**

```
postOrder('A')
  ① postOrder('B')
      ① postOrder('D') → None 종료 → print D
      ② postOrder('E') → None 종료 → print E
      ③ print B
  ② postOrder('C') → None 종료 → print C
  ③ print A

결과 : D E B C A
```

---

### 5단계 - 실행

```python
print('전위 순회 결과 : ')
preOrder('A')       # A B D E C

print('중위 순회 결과 : ')
inOrder('A')        # D B E A C  ← BST라면 오름차순 정렬

print('후위 순회 결과 : ')
postOrder('A')      # D E B C A
```

**출력 결과**

```
전위 순회 결과 : 
A B D E C 
중위 순회 결과 : 
D B E A C 
후위 순회 결과 : 
D E B C A 
```

---

## 📊 동작 과정 전체 시각화

**트리 구조 + 순회 결과 한눈에 보기**

```
        A
       / \
      B   C
     / \
    D   E

전위 (Pre)  : A → B → D → E → C   (루트 먼저)
중위 (In)   : D → B → E → A → C   (루트 중간)
후위 (Post) : D → E → B → C → A   (루트 마지막)
```

**순회별 루트 방문 시점**

```
전위 : [A] B D E C      ← A가 맨 앞
중위 :  D B E [A] C     ← A가 중간
후위 :  D E B C [A]     ← A가 맨 뒤
```

---

## 📊 순회 요약

|순회|함수|방문 순서|결과|주요 활용|
|---|---|---|---|---|
|전위|`preOrder`|루트→왼→오|A B D E C|트리 복사, 직렬화|
|중위|`inOrder`|왼→루트→오|D B E A C|BST 정렬 출력|
|후위|`postOrder`|왼→오→루트|D E B C A|트리 삭제, 수식 계산|

---

## 🔑 핵심 포인트

> Binary Tree는 **자식이 최대 2개** — 왼쪽(left)과 오른쪽(right)  
> `node is None` 이 재귀의 **기저 조건** — 리프 노드를 넘어가면 종료  
> 세 순회의 차이는 **루트를 언제 방문하느냐** 뿐 — 왼→오 방향은 동일  
> 중위 순회는 **BST에서 오름차순 정렬** 결과를 보장  
> 모든 순회는 **DFS 기반 재귀**로 구현

---

# 📄 stru8_bst.py — BST (이진 탐색 트리)

## 📌 개념 정리

### BST란?

**BST (Binary Search Tree, 이진 탐색 트리)** 는 **구조 + 정렬 규칙**을 모두 가진 이진 트리다.

- 각 노드 기준으로 **왼쪽 서브트리 < 현재 노드 < 오른쪽 서브트리**
- 입력 순서에 따라 트리 모양이 달라짐
- **중위 순회(왼쪽 → 현재 → 오른쪽)** 를 하면 오름차순 정렬 결과가 나옴

**실생활 예시** : 사전 검색, 데이터베이스 인덱스, 범위 탐색

```
       [5]          ← 루트
      /   \
    [3]   [7]
   /   \     \
 [2]   [4]   [9]
```

---

### 핵심 규칙

|규칙|설명|
|---|---|
|`left < node`|왼쪽 자식은 현재 노드보다 **작은 값**|
|`node < right`|오른쪽 자식은 현재 노드보다 **큰 값**|
|중위 순회|항상 **오름차순 정렬** 결과 보장|
|입력 순서|삽입 순서에 따라 트리 모양이 달라짐|

---

### 이진 트리 vs BST

|구분|이진 트리|**BST**|
|---|---|---|
|자식 수|최대 2개|최대 2개|
|정렬 규칙|❌ 없음|✅ left < node < right|
|탐색 효율|O(n)|**O(log n)** (균형 잡힌 경우)|
|중위 순회|임의 순서|**오름차순 정렬**|

> 이진 트리는 **구조만** 있고, BST는 **구조 + 정렬 규칙**이 있다

---

## 💻 전체 실습 코드

### 1단계 - Node 클래스 정의

```python
class Node:
    def __init__(self, key):
        self.key = key      # 노드가 저장하는 값
        self.left = None    # 왼쪽 자식 노드 (더 작은 값들이 저장)
        self.right = None   # 오른쪽 자식 노드 (더 큰 값들이 저장)
```

---

### 2단계 - `insert` : BST 삽입 (재귀)

```python
def insert(root, key):
    if root is None:                            # 현재 위치가 비어있으면
        return Node(key)                        # 새 노드 생성 후 반환
    
    if key < root.key:                          # 삽입값이 현재 노드보다 작으면
        root.left = insert(root.left, key)      # 왼쪽 서브트리에 재귀 삽입
    else:                                       # 삽입값이 현재 노드 이상이면
        root.right = insert(root.right, key)    # 오른쪽 서브트리에 재귀 삽입

    return root     # 현재 노드 반환 — 없으면 None 반환 → 트리가 루트만 남고 사라짐
```

**동작 과정 — `[5, 3, 7, 2, 4, 9]` 삽입**

```
insert(5) : root = None → Node(5) 생성
            [5]

insert(3) : 3 < 5 → 왼쪽
            [5]
           /
          [3]

insert(7) : 7 > 5 → 오른쪽
            [5]
           /   \
          [3]  [7]

insert(2) : 2 < 5 → 왼쪽, 2 < 3 → 왼쪽
            [5]
           /   \
          [3]  [7]
         /
        [2]

insert(4) : 4 < 5 → 왼쪽, 4 > 3 → 오른쪽
            [5]
           /   \
          [3]  [7]
         /   \
        [2]  [4]

insert(9) : 9 > 5 → 오른쪽, 9 > 7 → 오른쪽
            [5]
           /   \
          [3]  [7]
         /   \    \
        [2]  [4]  [9]
```

> `return root` 가 반드시 있어야 함  
> 없으면 `insert`가 `None` 반환 → `root = None` 으로 덮어씌워져 트리가 사라짐

---

### 3단계 - `inorder` : 중위 순회 (정렬 결과 생성)

```python
def inorder(root, result):
    if root is None:                    # 기저 조건 — 자식이 없으면 종료
        return
    inorder(root.left, result)          # ① 왼쪽 노드(작은 값들) 먼저 방문
    result.append(root.key)             # ② 현재 노드 값 저장
    inorder(root.right, result)         # ③ 오른쪽 노드(큰 값들) 방문
```

**동작 과정**

```
inorder(5)
  ① inorder(3)
      ① inorder(2) → inorder(None) 종료 → append(2) → inorder(None) 종료
      ② append(3)
      ③ inorder(4) → inorder(None) 종료 → append(4) → inorder(None) 종료
  ② append(5)
  ③ inorder(7)
      ① inorder(None) 종료
      ② append(7)
      ③ inorder(9) → inorder(None) 종료 → append(9) → inorder(None) 종료

result : [2, 3, 4, 5, 7, 9]  ← 오름차순 정렬
```

---

### 4단계 - 실행

```python
values = [5, 3, 7, 2, 4, 9]
root = None
for v in values:
    root = insert(root, v)  # BST에 삽입하고 루트(최상단)를 갱신

sorted_result = []
inorder(root, sorted_result)
print('결과 : ', sorted_result)
```

**출력 결과**

```
결과 :  [2, 3, 4, 5, 7, 9]
```

---

## 📊 동작 과정 전체 시각화

**최종 트리 구조 + 중위 순회 경로**

```
        [5]
       /   \
     [3]   [7]
    /   \     \
  [2]   [4]   [9]

중위 순회 : 2 → 3 → 4 → 5 → 7 → 9  (오름차순)
```

**삽입 규칙 요약**

```
삽입값 < 현재 노드  →  왼쪽으로 이동
삽입값 >= 현재 노드 →  오른쪽으로 이동
현재 위치가 None   →  새 노드 생성
```

---

## 📊 함수 요약

| 함수                      | 역할                    | 반환값                   |
| ----------------------- | --------------------- | --------------------- |
| `insert(root, key)`     | BST 규칙에 따라 노드 삽입 (재귀) | 현재 노드 (`root`)        |
| `inorder(root, result)` | 중위 순회로 정렬 결과 수집       | 없음 (`result` 리스트에 누적) |

---

## 🔑 핵심 포인트

> BST는 **구조 + 정렬 규칙** — `left < node < right` 항상 유지  
> `insert` 는 재귀로 삽입 위치를 찾고 반드시 `return root` 로 마무리  
> `return root` 가 없으면 `None` 반환 → **트리 전체가 사라지는 버그** 발생  
> 중위 순회는 BST에서 항상 **오름차순 정렬** 결과를 보장  
> 입력 순서에 따라 트리 모양이 달라져 탐색 효율도 달라짐

---
# 📄 stru9_graph.py — Graph (그래프)

## 📌 개념 정리

### Graph란?

**Graph** 는 **Node(Vertex, 정점)** 와 **Edge(간선)** 의 집합으로 이루어진 자료구조다.

- 루트가 없고 계층 구조도 없음 — 노드 간 자유롭게 연결 가능
- 사이클(순환)이 있을 수 있음
- 연결되지 않은 노드도 존재 가능

**실생활 예시** : 지하철 노선도, SNS 친구 관계, 네비게이션 경로

```
    A
   / \
  B   C
 / \   \
D   E   F
```

---

### Tree vs Graph

|구분|Tree|**Graph**|
|---|---|---|
|구조|계층 구조|일반 네트워크 구조|
|루트|✅ 있음|❌ 없음|
|사이클(순환)|❌ 없음|✅ 있을 수 있음|
|연결 여부|항상 연결|연결 / 비연결 모두 가능|
|관계|부모 → 자식 (1:N)|노드 간 자유 연결 (N:M)|

> Tree는 Graph의 특수한 형태 — 사이클 없고 항상 연결된 Graph

---

### 탐색 방법 2가지

| 구분  | DFS                 | BFS                |
| --- | ------------------- | ------------------ |
| 이름  | 깊이 우선 탐색            | 너비 우선 탐색           |
| 방식  | 한 경로를 끝까지 탐색 후 백트래킹 | 가까운 노드부터 레벨 단위로 탐색 |
| 구현  | 재귀 또는 스택            | 큐 (deque)          |
| 활용  | 경로 추적, 백트래킹         | **최단 거리 탐색**       |

---

## 💻 전체 실습 코드

### 1단계 - 그래프 데이터 정의

```python
# 딕셔너리로 그래프 표현
# key = 노드, value = 연결된 이웃 노드들의 튜플
graph = {
    'A': ('B', 'C'),  # A는 B, C와 연결
    'B': ('D', 'E'),  # B는 D, E와 연결
    'C': ('F',),      # C는 F와 연결 — 요소가 1개여도 튜플이므로 쉼표 필수
    'D': (),          # D는 연결된 노드 없음 (리프)
    'E': (),          # E는 연결된 노드 없음 (리프)
    'F': ()           # F는 연결된 노드 없음 (리프)
}
```

> `('F')` 는 문자열 `'F'` 로 인식됨 → 반드시 `('F',)` 로 써야 튜플

---

### 2단계 - DFS (깊이 우선 탐색)

```python
# DFS - 깊이 우선 탐색 방식
# 재귀함수 또는 스택으로 구현
# 경로 추적, 백트래킹에 적합
def dfsFunc(graph, start, visited):
    visited.append(start)               # 현재 노드 방문 처리
    for next_node in graph[start]:      # 현재 노드의 이웃 노드 순회
        if next_node not in visited:    # 아직 방문 안 한 노드면
            dfsFunc(graph, next_node, visited)  # 재귀로 더 깊이 탐색

visited_dfs = []    # 방문 순서 저장용
dfsFunc(graph, 'A', visited_dfs)
print('DFS 방문 순서 : ', visited_dfs)
```

**동작 과정**

```
dfsFunc('A') → visited: [A]
  └─ dfsFunc('B') → visited: [A, B]
      └─ dfsFunc('D') → visited: [A, B, D]
          └─ 이웃 없음 → 종료 (백트래킹)
      └─ dfsFunc('E') → visited: [A, B, D, E]
          └─ 이웃 없음 → 종료 (백트래킹)
  └─ dfsFunc('C') → visited: [A, B, D, E, C]
      └─ dfsFunc('F') → visited: [A, B, D, E, C, F]
          └─ 이웃 없음 → 종료
```

**출력 결과**

```
DFS 방문 순서 :  ['A', 'B', 'D', 'E', 'C', 'F']
```

> 방문 즉시 아래로 내려감 — **재귀 (call stack)** 이 자연스럽게 DFS를 구현

---

### 3단계 - BFS (너비 우선 탐색)

```python
# BFS - 너비 우선 탐색 방식
# 큐로 구현 (FIFO) — 최단 거리 탐색에 적합
from collections import deque

def bfsFunc(graph, start):
    visited = [start]       # 시작 노드 방문 처리
    queue = deque([start])  # 시작 노드를 큐에 추가

    while queue:
        node = queue.popleft()              # 가장 먼저 들어온 노드 꺼냄
        for next_node in graph[node]:       # 현재 노드의 이웃 노드 확인
            if next_node not in visited:
                visited.append(next_node)   # 방문 처리
                queue.append(next_node)     # 다음 탐색 대상으로 큐에 추가

    return visited

visited_bfs = bfsFunc(graph, 'A')
print('BFS 방문 순서 : ', visited_bfs)
```

**동작 과정**

```
초기 : visited=[A], queue=[A]

popleft → A 꺼냄 | 이웃: B, C → queue=[B, C], visited=[A, B, C]
popleft → B 꺼냄 | 이웃: D, E → queue=[C, D, E], visited=[A, B, C, D, E]
popleft → C 꺼냄 | 이웃: F   → queue=[D, E, F], visited=[A, B, C, D, E, F]
popleft → D 꺼냄 | 이웃 없음 → queue=[E, F]
popleft → E 꺼냄 | 이웃 없음 → queue=[F]
popleft → F 꺼냄 | 이웃 없음 → queue=[] → 종료
```

**출력 결과**

```
BFS 방문 순서 :  ['A', 'B', 'C', 'D', 'E', 'F']
```

> 방문 즉시 큐에 쌓고 먼저 들어온 것부터 처리 → **거리(레벨) 개념**이 생김

---

## 📊 DFS vs BFS 탐색 순서 비교

```
그래프 구조 :
    A
   / \
  B   C
 / \   \
D   E   F

DFS : A → B → D → E → C → F   (깊이 우선 — 한 가지 끝까지)
BFS : A → B → C → D → E → F   (너비 우선 — 같은 레벨 먼저)
```

**레벨(거리) 시각화 — BFS**

```
Level 0 :        A
Level 1 :      B   C
Level 2 :    D   E   F
```

---

## 📊 함수 요약

|함수|탐색 방식|구현|반환|
|---|---|---|---|
|`dfsFunc(graph, start, visited)`|깊이 우선|재귀|없음 (`visited` 리스트에 누적)|
|`bfsFunc(graph, start)`|너비 우선|큐 (`deque`)|`visited` 리스트|

---

## 🔑 핵심 포인트

> Graph는 **루트 없음, 사이클 가능, 비연결 가능** — Tree보다 자유로운 구조  
> `('F',)` 처럼 요소가 1개인 튜플은 **쉼표 필수** — 없으면 문자열로 인식  
> DFS는 **재귀(call stack)** 로 자연스럽게 구현 — 경로 추적·백트래킹에 적합  
> BFS는 **큐(deque)** 로 구현 — 레벨 단위 탐색으로 **최단 거리** 탐색에 적합  
> `visited` 리스트로 이미 방문한 노드를 체크 — 없으면 무한 루프 발생

---
