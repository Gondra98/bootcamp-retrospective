# Day 54_선형 자료구조 : Linked List · Stack · Queue
## 📅 2026-04-22

---
# 📄 stru2_linked.py — Linked List (연결 리스트)

## 📌 개념 정리

### Linked List란?

**Linked List (연결 리스트)** 는 각 데이터(Node)가 다음 데이터의 **주소(포인터)** 를 가지고 연결된 구조다.

- 배열처럼 연속된 메모리가 아닌, **임의의 메모리 공간**에 흩어져 포인터로 연결
- 중간 삽입·삭제가 빠름 (배열은 O(n), 연결 리스트는 포인터 변경만으로 O(1))
- 인덱스로 직접 접근 불가 → 처음(head)부터 포인터를 따라 탐색해야 함

**실생활 예시** : 놀이공원 대기 줄, 음악 플레이리스트, 브라우저 방문 기록

```
[철수 | •]──→ [영희 | •]──→ [민수 | None]
  head                          tail
```

---

### 핵심 구성 요소

|구성 요소|설명|
|---|---|
|`Node`|데이터(`name`) + 다음 노드 주소(`next`)를 가지는 단위|
|`head`|연결 리스트의 시작점 (첫 번째 노드의 주소)|
|`next`|다음 노드를 가리키는 포인터 (마지막 노드는 `None`)|
|`current`|탐색 시 현재 위치를 나타내는 임시 변수|

---

### 배열(list) vs 연결 리스트 비교

|항목|배열 (list)|연결 리스트|
|---|---|---|
|메모리|연속된 공간|임의의 공간|
|인덱스 접근|✅ O(1) 빠름|❌ O(n) 느림|
|중간 삽입·삭제|❌ O(n) 느림|✅ O(1) 빠름 (노드 찾은 후)|
|크기 변경|비용 있음|유연함|

---

## 💻 전체 실습 코드

### 1단계 - Node 클래스 정의

```python
class Node:
    def __init__(self, name):
        self.name = name    # 노드가 저장하는 데이터 (이름)
        self.next = None    # 다음 노드를 가리키는 포인터 (초기값: None = 연결 없음)
```

> `next = None` : 아직 연결된 다음 노드가 없다는 의미.  
> 노드가 연결되면 `next`에 다음 노드 객체의 주소가 저장된다.

---

### 2단계 - LinkedList 클래스 정의

```python
class LinkedList:
    def __init__(self):
        self.head = None    # 리스트 시작점. 처음엔 아무 노드도 없으므로 None
```

---

### 3단계 - `append` : 노드를 맨 뒤에 추가

```python
    def append(self, name):
        new_node = Node(name)           # 새 노드 생성

        if self.head is None:           # 리스트가 비어있으면
            self.head = new_node        # 새 노드가 곧 head
            return

        # 이미 노드가 있다면 마지막 노드까지 이동
        current = self.head
        while current.next:             # next가 None이면 마지막 노드
            current = current.next

        current.next = new_node         # 마지막 노드의 next에 새 노드 연결
```

**동작 과정**

```
append('철수') : head → [철수|None]
append('영희') : head → [철수|•] → [영희|None]
append('민수') : head → [철수|•] → [영희|•] → [민수|None]
```

---

### 4단계 - `show` : 전체 노드 출력

```python
    def show(self):
        current = self.head             # head부터 시작
        while current:                  # None이 아닌 동안 (노드가 있는 동안)
            print(current.name, end="->")
            current = current.next      # 다음 노드로 이동
        print('끝')
```

**출력 예시**

```
철수->영희->민수->끝
```

---

### 5단계 - `insert_after` : 특정 노드 뒤에 삽입

```python
    # target노드를 찾고 → 새 노드 만들고 → 기존 연결 변경
    def insert_after(self, target_name, new_name):
        current = self.head

        while current:
            if current.name == target_name:         # 대상 노드를 찾으면
                new_node = Node(new_name)            # 새 노드 생성
                new_node.next = current.next         # ① 새 노드의 next = 기존 다음 노드
                current.next = new_node              # ② 대상 노드의 next = 새 노드
                return                               # ①→② 순서 중요! 반대면 연결 끊김
            current = current.next
```

**동작 과정 — 영희 뒤에 지수 삽입**

```
삽입 전 : [철수|•] → [영희|•] → [민수|None]

① new_node.next = current.next  →  [지수|•] → [민수|None]
② current.next  = new_node      →  [영희|•] → [지수|•] → [민수|None]

삽입 후 : [철수|•] → [영희|•] → [지수|•] → [민수|None]
```

> ① → ② 순서가 반드시 지켜져야 한다!  
> 순서가 바뀌면 `current.next`가 먼저 `new_node`로 덮여서 민수와의 연결이 끊긴다.

---

### 6단계 - `remove` : 특정 노드 삭제

```python
    def remove(self, name):
        # 경우 1 : 삭제 대상이 head(맨 앞)인 경우
        if self.head and self.head.name == name:
            self.head = self.head.next  # head를 두 번째 노드로 변경
            return

        # 경우 2 : 중간 또는 끝 노드 삭제
        # 삭제 대상의 이전 노드를 찾아서 연결을 건너뜀
        current = self.head
        while current and current.next:
            if current.next.name == name:
                current.next = current.next.next    # 대상 노드를 건너뛰어 연결
                return
            current = current.next
```

**동작 과정 — 영희 삭제**

```
삭제 전 : [철수|•] → [영희|•] → [지수|•] → [민수|None]

current.next = current.next.next
→ 철수의 next를 지수로 변경 (영희를 건너뜀)

삭제 후 : [철수|•] → [지수|•] → [민수|None]
```

---

### 7단계 - 전체 실행

```python
line = LinkedList()
line.append('철수')
line.append('영희')
line.append('민수')

print('현재 줄 상태 : ')
line.show()                             # 철수->영희->민수->끝
print()

# 영희 뒤에 지수를 삽입
line.insert_after('영희', '지수')
print('지수를 삽입 줄 상태 : ')
line.show()                             # 철수->영희->지수->민수->끝
print()

# 영희가 줄서기를 포기 (삭제)
line.remove('영희')
print('영희 삭제 후 줄 상태 : ')
line.show()                             # 철수->지수->민수->끝
```

**출력 결과**

```
현재 줄 상태 : 
철수->영희->민수->끝

지수를 삽입 줄 상태 : 
철수->영희->지수->민수->끝

영희 삭제 후 줄 상태 : 
철수->지수->민수->끝
```

---

## 📊 동작 과정 전체 요약

|메소드|동작|시간복잡도|
|---|---|---|
|`append(name)`|맨 뒤에 노드 추가|O(n) — 끝까지 탐색 필요|
|`show()`|전체 노드 순서 출력|O(n)|
|`insert_after(target, new)`|특정 노드 뒤에 삽입|O(n) — 탐색 후 O(1) 삽입|
|`remove(name)`|특정 노드 삭제|O(n) — 탐색 후 O(1) 삭제|

---

## 🔑 핵심 포인트

> Linked List는 **포인터(next)** 로 노드를 연결 — 메모리상 연속되지 않아도 됨  
> `insert_after`는 **① new_node.next 먼저, ② current.next 나중** 순서가 반드시 지켜져야 함  
> `remove`는 **삭제 대상의 이전 노드**를 찾아 `next`를 건너뛰는 방식으로 동작  
> head가 삭제 대상일 때는 별도 처리 필요 — `self.head = self.head.next`

---

# 📄 stru3_stack.py — Stack (스택)

## 📌 개념 정리

### Stack이란?

**Stack**은 **LIFO (Last In, First Out)** 구조로,  
마지막에 들어온 데이터가 먼저 나가는 자료구조다.

- 데이터는 **위(top)에서만 추가·제거**
- 중간 데이터에 직접 접근·수정 불가 (인덱스 접근은 Python list 기능이지 Stack 개념 아님)
- Python에서는 `list`의 `append()` / `pop()`으로 구현 가능

**실생활 예시** : 브라우저 뒤로 가기, 실행 취소(Ctrl+Z), 함수 호출 스택

```
push →  [ 회전목마 ]  ← top (가장 나중에 들어온 것)
        [  바이킹  ]
        [ T-express]  ← bottom (가장 먼저 들어온 것)
           pop ↑
```

---

### 핵심 파라미터 / 용어

|용어|설명|
|---|---|
|`push`|top에 데이터 추가|
|`pop`|top에서 데이터 제거 및 반환|
|`top`|스택의 맨 위 요소 (가장 나중에 들어온 데이터)|
|`LIFO`|Last In First Out — 나중에 넣은 것이 먼저 나옴|

---

### Queue vs Stack 핵심 차이

|구분|Queue|Stack|
|---|---|---|
|구조|FIFO (선입선출)|LIFO (후입선출)|
|추가 위치|back(뒤)|top(위)|
|제거 위치|front(앞)|top(위)|
|실생활 예시|대기 줄|뒤로 가기, 실행 취소|
|Python 구현|`deque`|`list`|

> Queue는 **먼저 온 사람이 먼저** 나가고,  
> Stack은 **나중에 온 사람이 먼저** 나간다.

---

## 💻 전체 실습 코드

### 1단계 - Stack 기본 사용 (`list`)

```python
stack = []  # Python의 list를 Stack처럼 사용
print('놀이 공원 입장')

# ── 탑승 기록 추가 (PUSH) ──
stack.append('T-express 탑승')      # top에 추가
print('기록 : ', stack)             # ['T-express 탑승']

stack.append('바이킹 탑승')
print('기록 : ', stack)             # ['T-express 탑승', '바이킹 탑승']

stack.append('회전목마 탑승')
print('기록 : ', stack)             # ['T-express 탑승', '바이킹 탑승', '회전목마 탑승']

# ── 가장 최근 기록 삭제 (POP) ──
# ⚠️ 주의 : pop(0), pop(1) 처럼 인덱스 지정은 Stack 개념 위반
last_action = stack.pop()           # 마지막(top) 요소 제거 → '회전목마 탑승'
print('마지막 기록 취소 후 현재 : ', stack)  # ['T-express 탑승', '바이킹 탑승']

last_action = stack.pop()           # '바이킹 탑승' 제거
print('마지막 기록 취소 후 현재 : ', stack)  # ['T-express 탑승']
```

**출력 결과**

```
놀이 공원 입장
기록 :  ['T-express 탑승']
기록 :  ['T-express 탑승', '바이킹 탑승']
기록 :  ['T-express 탑승', '바이킹 탑승', '회전목마 탑승']
마지막 기록 취소 후 현재 :  ['T-express 탑승', '바이킹 탑승']
마지막 기록 취소 후 현재 :  ['T-express 탑승']
```

---

### 2단계 - MyStack 클래스 구현

```python
class MyStack:
    def __init__(self, iterable=None):
        self._data = []             # 내부 전용 저장소 (_언더스코어 = 외부 직접 접근 비권장)
        if iterable is not None:
            for x in iterable:
                self.push(x)        # 초기값이 있으면 순서대로 push

    def push(self, x):
        """top에 요소 추가 (삽입)"""
        self._data.append(x)
        return x

    def pop(self):
        """top에서 요소 제거 및 반환 — 비어있으면 예외 발생"""
        if not self._data:
            raise IndexError('스택이 비어 있음')
        return self._data.pop()     # list의 pop()은 마지막 요소 제거 → LIFO와 일치

    def is_empty(self):
        """스택이 비어있으면 True 반환"""
        return not self._data

    def __repr__(self):
        """print() 시 top → bottom 순서로 내용 출력"""
        top_to_bottom = list(reversed(self._data))  # 위에서부터 보이도록 뒤집기
        return f'Stack(top -> bottom {top_to_bottom})'
```

#### MyStack 클래스 메소드 요약

|메소드|기능|반환값|
|---|---|---|
|`push(x)`|top에 데이터 추가|추가한 값|
|`pop()`|top에서 데이터 제거 및 반환|제거된 값|
|`is_empty()`|비어있는지 확인|True / False|

---

### 3단계 - LIFO 동작 확인 (`demo1Func`)

```python
def demo1Func():
    s = MyStack()

    # A → B → C → D 순서로 push
    for item in ['A', 'B', 'C', 'D']:
        s.push(item)
        print(f'push {item} -> ', s)

    # LIFO에 따라 D → C → B → A 순서로 pop
    print('LIFO에 따라 하나씩 추출')
    while not s.is_empty():
        print(f'pop -> ', s.pop(), '| 현재는:', s)
```

**출력 결과**

```
push A ->  Stack(top -> bottom ['A'])
push B ->  Stack(top -> bottom ['B', 'A'])
push C ->  Stack(top -> bottom ['C', 'B', 'A'])
push D ->  Stack(top -> bottom ['D', 'C', 'B', 'A'])
LIFO에 따라 하나씩 추출
pop ->  D | 현재는: Stack(top -> bottom ['C', 'B', 'A'])
pop ->  C | 현재는: Stack(top -> bottom ['B', 'A'])
pop ->  B | 현재는: Stack(top -> bottom ['A'])
pop ->  A | 현재는: Stack(top -> bottom [])
```

> push 순서 A→B→C→D, pop 순서 D→C→B→A  
> **나중에 들어온 D가 가장 먼저 나오는** LIFO 구조 확인

---

### 4단계 - Stack으로 문자열 뒤집기 (`demo2Func`)

```python
def demo2Func(text: str) -> str:
    s = MyStack(text)       # 문자열을 한 글자씩 push
                            # 'Python' → P,y,t,h,o,n 순서로 쌓임
    out = []                # 뒤집힌 문자를 담을 리스트
    while not s.is_empty():
        out.append(s.pop()) # LIFO로 꺼내면 역순으로 나옴
    return ''.join(out)     # 리스트를 문자열로 합치기


if __name__ == '__main__':
    demo1Func()
    print(demo2Func('Python is good'))  # doog si nohtyP
    print(demo2Func('파이썬 만세'))      # 세만 썬이파
```

**출력 결과**

```
doog si nohtyP
세만 썬이파
```

**동작 원리**

```
입력 'ABC' → push 순서 : A → B → C

Stack 상태 :
  [C]  ← top
  [B]
  [A]  ← bottom

pop 순서 : C → B → A  →  결과 : 'CBA'
```

---

## 📊 동작 과정 시각화

**push 동작 (A→B→C→D)**

```
push A :  [A]
push B :  [B]  ← top
          [A]
push C :  [C]  ← top
          [B]
          [A]
push D :  [D]  ← top
          [C]
          [B]
          [A]
```

**pop 동작 (LIFO)**

```
초기 상태 :  [D] ← top     pop → D   남은 스택: [C, B, A]
             [C]            pop → C   남은 스택: [B, A]
             [B]            pop → B   남은 스택: [A]
             [A]            pop → A   남은 스택: []
```

---

## 🔑 핵심 포인트

> Stack은 **LIFO** — 나중에 넣은 것이 먼저 나온다  
> `pop(0)`, `pop(1)` 처럼 인덱스 지정은 Python list 기능이지 **Stack 개념 위반**  
> `__repr__`에서 `reversed()`를 써서 top → bottom 순서로 출력하면 시각적으로 직관적  
> Stack에 iterable을 넣으면 문자열·리스트도 한 글자/요소씩 push 가능 → **문자열 뒤집기 응용**

---

# 📄 stru4_queue.py — Queue (큐)

## 📌 개념 정리

### Queue란?

**Queue**는 **FIFO (First In, First Out)** 구조로,  
먼저 들어온 데이터가 먼저 나가는 자료구조다.

- 데이터는 **뒤(back)에서 추가**, **앞(front)에서 제거**
- 중간 데이터에 직접 접근·수정 불가
- Python에서는 `collections.deque`로 구현 (`list`보다 양쪽 삽입/삭제가 O(1)로 빠름)

**실생활 예시** : 놀이공원 대기 줄, 프린터 출력 대기, 네트워크 패킷 처리

```
입력(enqueue) →  [ 민수 | 영희 | 철수 ] → 출력(dequeue)
                   back               front
```

---

### 핵심 용어

|용어|설명|
|---|---|
|`enqueue`|큐의 뒤(back)에 데이터 추가|
|`dequeue`|큐의 앞(front)에서 데이터 제거 및 반환|
|`front`|큐의 맨 앞 요소 (다음에 나올 데이터)|
|`FIFO`|First In First Out — 먼저 넣은 것이 먼저 나옴|

---

### Queue vs Stack 핵심 차이

|구분|Queue|Stack|
|---|---|---|
|구조|FIFO (선입선출)|LIFO (후입선출)|
|추가 위치|back(뒤)|top(위)|
|제거 위치|front(앞)|top(위)|
|실생활 예시|대기 줄, 프린터|뒤로 가기, 실행 취소|
|Python 구현|`deque`|`list`|

> Queue는 **먼저 온 사람이 먼저** 나가고,  
> Stack은 **나중에 온 사람이 먼저** 나간다.

---

## 💻 전체 실습 코드

### 1단계 - Queue 기본 사용 (`deque`)

```python
from collections import deque   # list 대신 deque를 Queue 구현

# deque 주요 메소드
# append(x)     : 우측(back)에 추가      ← enqueue 역할
# appendleft(x) : 좌측(front)에 추가
# pop()         : 우측(back) 제거
# popleft()     : 좌측(front) 제거       ← dequeue 역할

queue = deque()
print('놀이 공원 기구 대기 시작')

# ── 줄서기 (enqueue) ──
queue.append('철수')
print('첫번째 줄서기 : ', list(queue))   # ['철수']
queue.append('영희')
print('두번째 줄서기 : ', list(queue))   # ['철수', '영희']
queue.append('민수')
print('세번째 줄서기 : ', list(queue))   # ['철수', '영희', '민수']

print()

# ── 탑승 (dequeue) — FIFO ──
first_person = queue.popleft()          # 가장 먼저 들어온 '철수'가 나옴
print(first_person, '놀이 기구 탑승')   # 철수 놀이 기구 탑승
print('현재 대기줄 : ', list(queue))    # ['영희', '민수']
print()

# 중간 데이터 처리 불가 — 반드시 front에서만 꺼낼 수 있음
first_person = queue.popleft()          # 그 다음 '영희'가 나옴
print(first_person, '놀이 기구 탑승')   # 영희 놀이 기구 탑승
print('현재 대기줄 : ', list(queue))    # ['민수']
print()

# ── 다음 탑승 예정자 확인 (제거 없이 조회) ──
if queue:
    print('탑승 예정자:', queue[0])     # 민수
else:
    print('대기자 없음')
```

**출력 결과**

```
놀이 공원 기구 대기 시작
첫번째 줄서기 :  ['철수']
두번째 줄서기 :  ['철수', '영희']
세번째 줄서기 :  ['철수', '영희', '민수']

철수 놀이 기구 탑승
현재 대기줄 :  ['영희', '민수']

영희 놀이 기구 탑승
현재 대기줄 :  ['민수']

탑승 예정자: 민수
```

---

### 2단계 - Queue 클래스 구현

```python
from collections import deque

class Queue:
    def __init__(self, iterable=None):
        self._data = deque()            # 내부 전용 저장소 (_언더스코어 = 외부 직접 접근 비권장)
        if iterable is not None:
            for x in iterable:
                self.enqueue(x)         # 초기값이 있으면 순서대로 enqueue

    def enqueue(self, x):
        """데이터를 큐의 뒤(back)에 추가"""
        self._data.append(x)
        return x

    def dequeue(self):
        """큐의 앞(front)에서 데이터를 꺼냄 — 비어있으면 예외 발생"""
        if not self._data:
            raise IndexError('큐 비어 있음')
        return self._data.popleft()

    def front(self):
        """맨 앞 요소를 제거 없이 조회만 함"""
        if not self._data:
            raise IndexError('큐 비어 있음')
        return self._data[0]

    def is_empty(self):
        """큐가 비어있으면 True 반환"""
        return not self._data

    def size(self):
        """현재 큐에 저장된 요소 수 반환"""
        return len(self._data)

    def clear(self):
        """큐를 전체 비움"""
        self._data.clear()

    def __repr__(self):
        """print() 시 front → back 순서로 내용 출력"""
        return f'Queue(front -> back {list(self._data)})'
```

#### Queue 클래스 메소드 요약

|메소드|기능|반환값|
|---|---|---|
|`enqueue(x)`|back에 데이터 추가|추가한 값|
|`dequeue()`|front에서 데이터 제거 및 반환|제거된 값|
|`front()`|front 값 조회 (제거 안 함)|front 값|
|`is_empty()`|비어있는지 확인|True / False|
|`size()`|저장된 요소 수|정수|
|`clear()`|전체 비우기|None|

---

### 3단계 - FIFO 동작 확인 (`demo1Func`)

```python
def demo1Func():
    imsi1 = Queue()
    imsi2 = Queue([10, 20, 30])         # 초기값을 넣어 생성
    print(imsi1)                        # Queue(front -> back [])
    print(imsi2)                        # Queue(front -> back [10, 20, 30])
    print(imsi2.front())                # 10 (맨 앞 조회)
    print(imsi2.size())                 # 3
    imsi2.clear()
    print(imsi2)                        # Queue(front -> back [])
    print('--------------')

    q = Queue()
    # A → B → C → D 순서로 enqueue
    for item in ['A', 'B', 'C', 'D']:
        q.enqueue(item)
        print(f'enqueue {item} -> ', q)

    # FIFO에 따라 A → B → C → D 순서로 dequeue
    print('FIFO에 따라 하나씩 추출')
    while not q.is_empty():
        print(f'dequeue -> ', q.dequeue(), '| 남은:', q)
```

**출력 결과**

```
Queue(front -> back [])
Queue(front -> back [10, 20, 30])
10
3
Queue(front -> back [])
--------------
enqueue A ->  Queue(front -> back ['A'])
enqueue B ->  Queue(front -> back ['A', 'B'])
enqueue C ->  Queue(front -> back ['A', 'B', 'C'])
enqueue D ->  Queue(front -> back ['A', 'B', 'C', 'D'])
FIFO에 따라 하나씩 추출
dequeue ->  A | 남은: Queue(front -> back ['B', 'C', 'D'])
dequeue ->  B | 남은: Queue(front -> back ['C', 'D'])
dequeue ->  C | 남은: Queue(front -> back ['D'])
dequeue ->  D | 남은: Queue(front -> back [])
```

> enqueue 순서 A→B→C→D, dequeue 순서 A→B→C→D  
> **먼저 들어온 A가 가장 먼저 나오는** FIFO 구조 확인

---

### 4단계 - 프린터 출력 시뮬레이션 (`demo2Func`)

```python
def demo2Func(jobs, ppm=15):
    q = Queue(jobs)     # 작업 목록을 큐에 입력 (튜플 리스트)
    t_sec = 0.0         # 누적 시간 (초)
    order = []          # 처리된 문서 순서 기록

    print('프린터로 출력하기')
    while not q.is_empty():
        doc, pages = q.dequeue()                # FIFO — 먼저 들어온 작업부터 처리
        duration = (pages / ppm) * 60.0         # 출력 시간 계산 : 페이지수 / 분당페이지수 * 60
        t_sec += duration
        order.append(doc)
        print(f't={t_sec:6.1f}초 | 출력 : {doc:10s}({pages}페이지)')

    print('처리순서(FIFO) : ', order)


if __name__ == '__main__':
    demo1Func()
    print('문서 프린터로 출력 시뮬레이션 - FIFO')
    jobs = [('abc.pdf', 10), ('nice.doc', 30), ('good.txt', 5)]
    demo2Func(jobs, ppm=20)     # 현재 프린터는 1분에 20장 출력
```

**출력 결과**

```
문서 프린터로 출력 시뮬레이션 - FIFO
프린터로 출력하기
t=  30.0초 | 출력 : abc.pdf   (10페이지)
t= 120.0초 | 출력 : nice.doc  (30페이지)
t= 135.0초 | 출력 : good.txt  (5페이지)
처리순서(FIFO) :  ['abc.pdf', 'nice.doc', 'good.txt']
```

**출력 시간 계산 공식**

```
duration = (pages / ppm) * 60

abc.pdf  : (10 / 20) * 60 =  30.0초
nice.doc : (30 / 20) * 60 =  90.0초  →  누적 120.0초
good.txt : ( 5 / 20) * 60 =  15.0초  →  누적 135.0초
```

---

## 📊 동작 과정 시각화

**enqueue 동작 (A→B→C→D)**

```
enqueue A : front → [A] ← back
enqueue B : front → [A | B] ← back
enqueue C : front → [A | B | C] ← back
enqueue D : front → [A | B | C | D] ← back
```

**dequeue 동작 (FIFO)**

```
초기 상태 : front → [A | B | C | D] ← back

dequeue → A   남은: [B | C | D]
dequeue → B   남은: [C | D]
dequeue → C   남은: [D]
dequeue → D   남은: []
```

---

## 🔑 핵심 포인트

> Queue는 **FIFO** — 먼저 넣은 것이 먼저 나온다  
> `deque`를 쓰는 이유 : `list.pop(0)`은 O(n)이지만 `deque.popleft()`는 **O(1)**  
> `front()`는 꺼내지 않고 조회만 — `dequeue()`와 혼동 주의  
> 프린터 시뮬레이션처럼 **순서가 중요한 작업 처리**에 Queue가 적합

---
