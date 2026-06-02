# Day79_강화학습 Q-Learning · GridWorld · CartPole 이산화

## 📅 2026-06-02

---
# 📄 rl2qlearning.ipynb — Q-Learning · GridWorld · epsilon-greedy · 벨만 방정식

---

## 1. 강화학습 개념

강화학습은 **정답(라벨) 없이** 에이전트가 직접 행동하고 보상을 받아 학습하는 방식이다.

```
현재 상태 → 행동 선택 → 보상 확인 → 다음 행동 개선
```

|개념|설명|
|---|---|
|상태 (State)|현재 위치 (GridWorld에서는 (row, col))|
|행동 (Action)|위 / 아래 / 왼쪽 / 오른쪽|
|보상 (Reward)|행동 결과에 대한 점수|
|Q-table|상태별 행동 점수표 (state × action 행렬)|
|Q-learning|경험을 통해 Q-table을 조금씩 갱신하는 알고리즘|
|epsilon-greedy|탐험(exploration)과 이용(exploitation)을 조절하는 전략|
|정책 (Policy)|학습 후 Q-table에서 가장 큰 값의 행동을 선택하는 것|

> 이 구조는 Gymnasium의 `env.step()` 구조와 동일하다.  
> GridWorld는 직접 환경을 구현한 것이고, Gymnasium은 CartPole 같은 환경을 라이브러리로 제공한다.

---

## 2. GridWorld 환경 설정

```python
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# 3행 4열 격자 세계
ROWS = 3
COLS = 4

START = (0, 0)  # 시작 위치
GOAL  = (2, 3)  # 목표 (도달 시 +10, 에피소드 종료)
TRAP  = (1, 1)  # 함정 (도달 시 -10, 에피소드 종료)

# 행동 정의: 방향별 (행 변화량, 열 변화량)
actions = {
    0: (-1,  0),  # 위
    1: ( 1,  0),  # 아래
    2: ( 0, -1),  # 왼쪽
    3: ( 0,  1),  # 오른쪽
}

action_names = {0: '위', 1: '아래', 2: '왼쪽', 3: '오른쪽'}

num_states  = ROWS * COLS     # 12개 상태
num_actions = len(actions)    # 4개 행동

# Q-table 초기화: 모든 값을 0으로 시작
# shape: (12, 4) → 12개 상태 × 4개 행동
Q = np.zeros((num_states, num_actions))
```

### GridWorld 지도

```
S  →  ↓  ↓
↑  X  →  ↓
↑  →  →  G
```

- `S` = START (0,0)
- `G` = GOAL (2,3)
- `X` = TRAP (1,1)

### 위치 ↔ 상태 번호 변환

```python
# 위치 (row, col) → 상태 번호 (0~11)
# 예: (2, 3) → 2 * 4 + 3 = 11
def pos_to_state(pos):
    row, col = pos
    return row * COLS + col

# 상태 번호 → 위치 (row, col)
# 예: 11 → (11 // 4, 11 % 4) = (2, 3)
def state_to_pos(state):
    row = state // COLS
    col = state % COLS
    return (row, col)
```

---

## 3. 하이퍼파라미터

```python
alpha         = 0.1    # 학습률: Q값을 얼마나 빠르게 갱신할지 (0~1)
gamma         = 0.9    # 할인율: 미래 보상을 현재 가치로 얼마나 반영할지 (0~1)
epsilon       = 1.0    # 탐험율: 1.0 = 완전 탐험, 0.1 = 거의 이용
epsilon_decay = 0.995  # 매 에피소드마다 epsilon에 곱해지는 감소율
epsilon_min   = 0.1    # epsilon의 최솟값 (이 이하로 내려가지 않음)

episodes  = 1000  # 총 학습 에피소드 수
max_steps = 30    # 에피소드 하나에서 최대 이동 횟수
```

> epsilon은 1.0에서 시작해서 0.995씩 곱하며 줄어든다.  
> 약 460 에피소드 후에 0.1(epsilon_min)에 도달하고 이후 고정된다.

---

## 4. 환경 이동 함수 (step)

Gymnasium의 `env.step(action)`과 동일한 구조로, 행동을 실행하고 결과를 반환한다.

```python
def step(pos, action):
    row, col = pos
    dr, dc = actions[action]  # 행동에 따른 행·열 변화량

    next_row = row + dr
    next_col = col + dc

    # 경계 밖으로 나가면 제자리 유지, 패널티 -2
    if next_row < 0 or next_row >= ROWS or next_col < 0 or next_col >= COLS:
        return pos, -2, False

    next_pos = (next_row, next_col)

    if next_pos == GOAL:
        return next_pos, +10, True   # 목표 도달: 보상 +10, 에피소드 종료
    elif next_pos == TRAP:
        return next_pos, -10, True   # 함정 도달: 보상 -10, 에피소드 종료
    else:
        return next_pos, -1, False   # 일반 이동: 보상 -1 (빠른 경로 유도)
```

|상황|보상|done|
|---|---|---|
|경계 밖 시도|-2|False|
|일반 이동|-1|False|
|TRAP 도달|-10|True|
|GOAL 도달|+10|True|

---

## 5. epsilon-greedy 행동 선택

```python
def choose_action(state, epsilon):
    if random.random() < epsilon:
        # 탐험: 무작위 행동 선택 (새로운 경험 수집)
        return random.randint(0, num_actions - 1)
    else:
        # 이용: Q-table에서 가장 높은 값의 행동 선택
        return np.argmax(Q[state])
```

> epsilon이 높을수록 탐험 비율이 높고, 낮을수록 학습된 Q-table을 따른다.  
> 초반에는 많이 탐험하고, 후반에는 이용 위주로 행동한다.

---

## 6. Q-learning 학습 루프

### 벨만 방정식 (Q-learning 핵심)

```
Q(s, a) = Q(s, a) + α × [r + γ × max Q(s', a') - Q(s, a)]
           ↑ 기존값      ↑ 학습률  ↑ target       ↑ TD error
```

- `r` : 이번 행동으로 받은 보상
- `γ × max Q(s', a')` : 다음 상태에서 가장 좋은 행동의 미래 가치
- `TD error` : 목표값과 현재 Q값의 차이 → 이 오차만큼 Q값을 업데이트

```python
for episode in range(episodes):
    pos   = START
    state = pos_to_state(pos)  # 시작 상태: (0,0) → 0

    for step_count in range(max_steps):
        # 1. 행동 선택 (epsilon-greedy)
        action = choose_action(state, epsilon)

        # 2. 행동 실행 → 결과 반환
        next_pos, reward, done = step(pos, action)
        next_state = pos_to_state(next_pos)

        # 3. Q-table 업데이트 (벨만 방정식)
        old_q  = Q[state, action]
        # Q(s, a) = r + γ × max Q(s', a')
        target   = reward + gamma * np.max(Q[next_state])
        td_error = target - old_q                      # 목표값과 현재 Q값의 차이
        Q[state][action] = old_q + alpha * td_error    # Q값 갱신

        # 4. 상태 이동
        pos   = next_pos
        state = next_state

        if done:
            break

    # 에피소드 종료 후 epsilon 감소 (탐험 → 이용으로 전환)
    epsilon = max(epsilon_min, epsilon_decay * epsilon)
```

---

## 7. 학습 결과 및 최적 정책

```python
# 학습된 Q-table 출력
print(np.round(Q, 2))

# 각 상태에서 최선의 행동 출력
for state in range(num_states):
    pos = state_to_pos(state)
    if pos == GOAL:
        print(f'상태 {state} {pos} : 목표지점')
        continue
    if pos == TRAP:
        print(f'상태 {state} {pos} : 함정')  # Q값이 전부 0 (학습 불가)
        continue
    best_action = np.argmax(Q[state])
    print(f'상태 {state} {pos} : {action_names[best_action]}')
```

> TRAP 칸의 Q값이 전부 0인 이유: 함정에 빠지면 바로 에피소드가 끝나서  
> 함정에서 나가는 행동을 학습할 기회 자체가 없기 때문이다.

---

## 8. 최적 경로 시뮬레이션

```python
# 학습된 Q-table로 실제 경로 추적
pos  = START
path = [pos]

for i in range(20):
    state  = pos_to_state(pos)
    action = np.argmax(Q[state])  # 탐험 없이 최선의 행동만 선택

    next_pos, reward, done = step(pos, action)
    path.append(next_pos)
    pos = next_pos

    if done:
        break

print('이동 경로:', path)
# 결과: [(0,0), (0,1), (0,2), (1,2), (1,3), (2,3)]
```

### 최적 경로 화살표 지도

```
S  →  ↓  ↓
↑  X  →  ↓
↑  →  →  G
```

TRAP(X)을 피해 START(S)에서 GOAL(G)까지 5번의 이동으로 도달한다.

---

## 9. GridWorld vs Gymnasium 비교

|항목|GridWorld (직접 구현)|CartPole (Gymnasium)|
|---|---|---|
|환경|3×4 격자|카트 + 막대 물리 시뮬레이션|
|상태|위치 (row, col) → 12가지 이산값|위치·속도·각도·각속도 → 연속값|
|행동|상/하/좌/우 4가지|왼쪽/오른쪽 2가지|
|보상|GOAL +10, TRAP -10, 이동 -1|버틴 매 스텝마다 +1|
|풀이 방법|Q-table (이산 상태)|DQN 등 딥러닝 기반 (연속 상태)|
|step 반환값|`next_pos, reward, done`|`state, reward, done, truncated, info`|

> CartPole은 상태가 연속값이라 Q-table로 풀 수 없다.  
> 이산화(discretize)하거나 DQN처럼 신경망으로 Q값을 근사해야 한다.

---
# 📄 rl3cartpole.ipynb — CartPole · Q-table 이산화 · Gymnasium

---

## 1. CartPole 개요

카트 위에 세워진 막대를 최대한 오래 쓰러뜨리지 않는 게임이다.

```
목표   : 200 스텝 동안 Pole이 넘어지지 않게 유지
종료조건 1 : Pole이 수직 기준 ±12도 이상 기울었을 때
종료조건 2 : 카트가 환경 밖으로 벗어났을 때 (위치 ±2.4 초과)
종료조건 3 : 200 스텝에 도달했을 때
```

### GridWorld와 CartPole 비교

|항목|GridWorld|CartPole|
|---|---|---|
|상태|(row, col) 위치 → 12가지 이산값|카트위치·속도·막대각도·각속도 → 연속값|
|행동|상/하/좌/우 4가지|왼쪽(0) / 오른쪽(1) 2가지|
|보상|GOAL +10, TRAP -10, 이동 -1|버틴 매 스텝마다 +1|
|풀이|Q-table (이산)|Q-table + **이산화** 필요|
|환경|직접 구현|Gymnasium 라이브러리 제공|

---

## 2. 설치 및 환경 생성

```python
# CartPole 환경이 포함된 gymnasium classic-control 설치
!pip install gymnasium[classic-control]
```

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# CartPole-v1 환경 생성
env = gym.make("CartPole-v1")

# 상태 공간 확인: 연속값 4개 (카트위치, 카트속도, 막대기울기, 막대각속도)
print(env.observation_space)
# → Box([-4.8, -inf, -0.41887903, -inf], [4.8, inf, 0.41887903, inf], (4,), float32)

# 행동 공간 확인: 0(왼쪽), 1(오른쪽) 2가지
print(env.action_space.n)  # → 2
```

### CartPole 상태 공간 (observation_space)

CartPole의 상태는 4개의 연속값으로 구성된다.

|인덱스|상태 요소|실제 범위|
|---|---|---|
|0|카트 위치|-4.8 ~ 4.8|
|1|카트 속도|-∞ ~ ∞|
|2|막대 기울기 (각도)|-0.418 ~ 0.418 rad|
|3|막대 각속도|-∞ ~ ∞|

> 실제 범위가 무한대(inf)인 값이 있어서 Q-table로 바로 사용할 수 없다.  
> 의미 있는 범위로 **인위적으로 클리핑(clip)** 후 이산화해야 한다.

---

## 3. 상태 공간 이산화 (Discretization)

CartPole의 상태는 연속값이라 Q-table에 그대로 쓸 수 없다.  
**이산화**: 연속적인 실수 범위를 N개의 구간(bin)으로 나눠서 정수 인덱스로 변환하는 작업이다.

### 실험 범위 설정 (인위적 클리핑)

```python
# 실제 관측 범위는 너무 넓거나 무한대이므로
# 실험적으로 의미 있는 범위로 제한한다
obs_space_low  = np.array([-2.4, -3.0, -0.5, -2.0])
obs_space_high = np.array([ 2.4,  3.0,  0.5,  2.0])
```

|상태 요소|실험 범위|이유|
|---|---|---|
|카트 위치|-2.4 ~ 2.4|이 범위 벗어나면 종료|
|카트 속도|-3.0 ~ 3.0|실험적으로 유효 범위|
|막대 기울기|-0.5 ~ 0.5|종료 조건(±0.418) 포함|
|막대 각속도|-2.0 ~ 2.0|실험적으로 유효 범위|

### 구간 수(bins) 결정

```python
# 각 상태 차원을 몇 개의 구간으로 나눌지 결정
# 구간이 많을수록 정밀하지만 Q-table 크기가 커짐
state_bins = [6, 12, 6, 12]
#              ↑   ↑   ↑   ↑
#           카트 카트 막대 막대
#           위치 속도 각도 각속도
```

|상태 요소|구간 수|범위|
|---|---|---|
|카트 위치|6|-2.4 ~ 2.4|
|카트 속도|12|-3.0 ~ 3.0|
|막대 기울기|6|-0.5 ~ 0.5|
|막대 각속도|12|-2.0 ~ 2.0|

> 속도와 각속도는 구간을 더 세밀하게(12개) 나눈다.  
> 속도 변화가 행동 선택에 더 큰 영향을 미치기 때문이다.

---

## 4. Q-table 초기화

```python
# Q-table shape: (6, 12, 6, 12, 2)
# 앞 4개 차원 = 이산화된 상태 인덱스
# 마지막 차원 = 행동 수 (왼쪽/오른쪽)
q_table = np.zeros(state_bins + [env.action_space.n])

print(q_table.shape)  # → (6, 12, 6, 12, 2)
print(6*12*6*12*2)    # → 10368 (총 상태-행동 쌍의 수)
```

### Q-table 크기 비교

|환경|Q-table shape|총 크기|
|---|---|---|
|GridWorld|(12, 4)|48|
|CartPole (이산화)|(6, 12, 6, 12, 2)|10,368|

> GridWorld는 12개 상태 × 4개 행동 = 48칸이었다.  
> CartPole은 이산화 후에도 10,368칸으로 훨씬 크다.  
> 이처럼 상태가 복잡해질수록 Q-table 방식은 한계가 생기고,  
> 이를 해결하기 위해 **DQN(Deep Q-Network)** 같은 딥러닝 기반 방법이 등장한다.

---

## 5. 핵심 개념 정리

### 이산화란?

```
연속값 예시: 카트 위치 = 0.73 (실수)
                  ↓
구간 분할: [-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4]
                  ↓
이산 인덱스: 0.73은 4번 구간에 해당 → 인덱스 3
                  ↓
Q-table 접근: q_table[3, ?, ?, ?, :]
```

실제로는 `np.digitize(value, bins)` 함수로 연속값을 구간 인덱스로 변환한다.

### Gymnasium env.step() 구조

```python
# CartPole에서 행동 실행
obs, reward, terminated, truncated, info = env.step(action)
#    ↑        ↑           ↑            ↑
# 새 상태   보상(+1)   종료여부    최대스텝초과

# GridWorld에서 행동 실행 (직접 구현)
next_pos, reward, done = step(pos, action)
```

> Gymnasium v0.26 이후 `done` 하나가 `terminated + truncated` 두 개로 분리됐다.  
> `terminated` = 실패로 종료 (막대 쓰러짐, 범위 벗어남)  
> `truncated` = 시간 초과로 종료 (200 스텝 도달)

---

## 6. 전체 흐름 요약

```
1. 환경 생성: gym.make("CartPole-v1")
       ↓
2. 관측 범위 설정 (obs_space_low / high)
       ↓
3. 이산화 구간 수 결정 (state_bins = [6, 12, 6, 12])
       ↓
4. Q-table 초기화: np.zeros([6, 12, 6, 12, 2])
       ↓
5. 학습 루프 (GridWorld와 동일한 Q-learning 구조)
   - 상태 관측 → 이산화 → Q-table 인덱스 변환
   - epsilon-greedy로 행동 선택
   - env.step(action)으로 결과 반환
   - 벨만 방정식으로 Q-table 갱신
       ↓
6. 학습된 Q-table로 CartPole 플레이
```

> GridWorld에서 배운 Q-learning 구조(벨만 방정식, epsilon-greedy, Q-table 갱신)가  
> CartPole에서도 그대로 적용된다.  
> 차이점은 연속 상태를 이산화하는 전처리 단계가 추가된다는 것이다.

