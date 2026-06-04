# Day80_강화학습 Q-Learning CartPole 구현 및 시각화

## 📅 2026-06-04

---

# 📄 rl3cartpole.ipynb — CartPole · Q-Learning · FuncAnimation

---

## 📌 과제 개요

> **목표**: 강화학습(Q-Learning)으로 CartPole 막대를 200 스텝 동안 쓰러뜨리지 않고 유지하기  
> **환경**: `gymnasium` CartPole-v1  
> **종료 조건**: 막대가 수직 기준 ±12도 이상 기울거나, 카트가 범위 밖으로 벗어나거나, 200 스텝 도달

---

## 📌 핵심 개념 정리

### 연속 상태공간 → 이산화 (Discretize)

Q-Table은 정수 인덱스로 접근해야 하기 때문에, 연속적인 실수값 상태를 고정된 구간(bin)으로 나눠 정수 인덱스로 변환해야 한다.

```
연속값 obs = [-0.02, 0.03, -0.01, 0.05]  (카트위치, 속도, 막대각, 막대각속도)
              ↓ discretize_state()
이산 인덱스 = (3, 6, 2, 5)  → Q-Table[3][6][2][5] 에 접근
```

### ε-Greedy (엡실론 그리디)

탐험(Exploration)과 활용(Exploitation)의 균형을 조절하는 전략.

|조건|동작|
|---|---|
|`rand() < ε`|랜덤 행동 선택 (탐험)|
|`rand() >= ε`|Q값이 최대인 행동 선택 (활용)|

- 초반: ε = 1.0 → 랜덤 행동 많이 → 다양한 경험 수집
- 후반: ε → 0.05 → Q-Table을 믿고 최적 행동

### 벨만 방정식 (Q-Learning 갱신)

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

|기호|의미|
|---|---|
|$\alpha$|학습률 (얼마나 빠르게 갱신할지)|
|$\gamma$|감가율 (미래 보상의 현재 가치)|
|$r$|이번 스텝 실제 보상|
|$\max Q(s', a')$|다음 상태에서 가능한 최대 Q값 (미래 가치)|
|$r + \gamma \max Q(s', a')$|TD Target (목표값)|
|$\text{TD Target} - Q(s,a)$|TD Error (현재값과 목표값의 차이)|

---

## 📌 환경 설정 및 Q-Table 초기화

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# 환경 생성
env = gym.make("CartPole-v1")
# observation_space: [카트위치(-4.8~4.8), 카트속도(-inf~inf), 막대각도(-0.418~0.418), 막대각속도(-inf~inf)]
# action_space: 0(왼쪽), 1(오른쪽) 총 2가지

# 실제 obs_space가 너무 넓으므로 인위적으로 실험 범위 제한
obs_space_low  = np.array([-2.4, -3.0, -0.5, -2.0])
obs_space_high = np.array([ 2.4,  3.0,  0.5,  2.0])

# 각 상태 차원을 몇 개의 구간(bin)으로 나눌지 결정
# [카트위치 6구간, 카트속도 12구간, 막대각도 6구간, 막대각속도 12구간]
state_bins = [6, 12, 6, 12]

# Q-Table 초기화: shape = (6, 12, 6, 12, 2) → 총 10368개의 상태-행동 쌍
q_table = np.zeros(state_bins + [env.action_space.n])
```

---

## 📌 상태 이산화 함수

```python
def discretize_state(state):
    # 1단계: 각 값이 [low, high] 범위 안에서 0.0~1.0 비율로 표현
    ratios = (state - obs_space_low) / (obs_space_high - obs_space_low)

    # 2단계: 비율 × bin 개수 → 어느 칸인지 정수 인덱스로 변환
    discrete = (ratios * state_bins).astype(int)

    # 3단계: 범위를 벗어난 값(음수, 초과) 클리핑 → Q-Table 인덱스 안전하게 유지
    return tuple(np.clip(discrete, 0, np.array(state_bins) - 1))

# np.clip 예시
# a = np.array([-2, 0, 3, 7, 10])
# np.clip(a, 0, 5) → [0, 0, 3, 5, 5]
```

---

## 📌 하이퍼파라미터 초기화

```python
alpha         = 0.1     # 학습률: 새로운 정보를 얼마나 반영할지
gamma         = 0.99    # 감가율: 미래 보상을 현재 가치로 얼마나 반영할지
epsilon       = 1.0     # 초기 탐험율: 1.0 = 100% 랜덤 행동
epsilon_decay = 0.999   # 에피소드마다 epsilon에 곱해지는 감소율
epsilon_min   = 0.05    # epsilon 최솟값 (완전히 0이 되지 않도록)
episodes      = 1000    # 총 학습 에피소드 수

reward_list   = []      # 에피소드별 총 보상 기록 (그래프용)
trajectories  = []      # 10ep마다 궤적 저장 (애니메이션용)
best_reward   = -np.inf # 지금까지 달성한 최고 보상 (-inf로 초기화해야 첫 ep부터 갱신됨)
```

---

## 📌 Q-Learning 학습 루프

```python
for ep in range(episodes):
    obs, _ = env.reset()          # 환경 초기화, 초기 관측값 수령
    state = discretize_state(obs) # 연속값 → Q-Table 인덱스
    total_reward = 0
    trajectory = []

    for step in range(200):       # 최대 200 스텝
        # ε-Greedy 행동 선택
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # 탐험: 랜덤 행동
        else:
            action = np.argmax(q_table[state])  # 활용: Q값 최대 행동

        # 환경에 행동 적용 → 결과 수령
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # terminated: 막대 쓰러짐 / truncated: 200스텝 초과

        done = terminated or truncated
        next_state = discretize_state(next_obs)

        # 미래 가치 계산: 다음 상태에서 가능한 최대 Q값
        best_next_q = np.max(q_table[next_state])

        # 벨만 방정식으로 Q-Table 갱신
        # state + (action,) → 예: (3,6,2,5) + (1,) = (3,6,2,5,1) 5차원 인덱스
        q_table[state + (action,)] += alpha * (reward + gamma * best_next_q - q_table[state + (action,)])

        state = next_state         # 상태 갱신
        obs   = next_obs           # 시각화용 obs 갱신

        total_reward += reward
        trajectory.append(obs.copy())  # 애니메이션용 궤적 저장 (.copy() 필수 - 참조 문제 방지)

        if done:
            break                  # 종료 조건 만족 시 루프 탈출

    reward_list.append(total_reward)

    # 최고 보상 갱신 시 출력
    if total_reward > best_reward:
        best_reward = total_reward
        print(f'Episode {ep} : Reward improved to {total_reward}')

    # 10 에피소드마다 궤적 저장
    if ep % 10 == 0:
        trajectories.append(trajectory)

    # epsilon 감소: 에피소드마다 ×0.999 → 점점 랜덤 행동 줄임
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
```

---

## 📌 보상 그래프 시각화

```python
plt.figure(figsize=(12, 5))
plt.plot(reward_list, alpha=0.7, color='royalblue', label='episode reward')
plt.xlabel('episode')
plt.ylabel('total reward')
plt.legend()
plt.tight_layout()
plt.show()
```

<img src="rl3cartpole.png" width="700">

> 초반(0~300ep): 보상이 낮고 불안정 — epsilon이 높아 랜덤 행동 많음  
> 후반(700~1000ep): 높은 보상 스파이크 증가 — Q-Table이 점점 최적화됨

---

## 📌 궤적 평탄화 및 라벨링

```python
flat_states    = []  # 여러 에피소드의 궤적을 한 줄로 펼침
episode_labels = []  # 각 obs가 몇 번째 에피소드 데이터인지 태깅

# 10ep마다 저장된 trajectories를 순서대로 펼치기
episode_numbers = list(range(0, episodes, 10))  # [0, 10, 20, ..., 990]

for i, traj in enumerate(trajectories):
    flat_states.extend(traj)
    # 해당 궤적의 스텝 수만큼 에피소드 번호 반복
    episode_labels.extend([episode_numbers[i]] * len(traj))

frame_count = len(flat_states)  # 총 애니메이션 프레임 수
```

> **주의**: `trajectories`와 `episode_numbers` 길이가 다르면 `IndexError` 발생  
> 초기화 셀 재실행 없이 학습 루프만 여러 번 돌리면 `trajectories`가 누적됨 → 반드시 초기화 후 재실행

---

## 📌 CartPole 애니메이션 (FuncAnimation)

```python
fig, ax = plt.subplots()  # plt.subplot() 아님 주의! → subplots()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.5, 1.5)
ax.set_title('Cartpole simulation')
ax.set_xlabel('Cart position')
ax.set_ylabel('height')

# 카트: 검은 사각형 패치
cart_width  = 0.4
cart_height = 0.2
cart_y      = 0.0
cart_rect = Rectangle((0, 0), cart_width, cart_height, color='black')
ax.add_patch(cart_rect)

# 막대: 빨간 선
pole_len  = 1.0
pole_line = ax.plot([], [], 'r-', lw=4)[0]  # ax.plot()은 리스트 반환 → [0]으로 선 객체 추출

# 에피소드 번호 텍스트
episode_text = ax.text(0.05, 1.4, '', transform=ax.transData, fontsize=12, color='blue')

def updateFunc(frame):
    x      = flat_states[frame][0]   # 카트 위치 (x축 좌표)
    theta  = flat_states[frame][2]   # 막대 각도 (라디안)
    ep_num = episode_labels[frame]   # 현재 프레임의 에피소드 번호

    # 카트 위치 갱신: 중심 기준으로 왼쪽에서 사각형 시작
    cart_rect.set_xy([x - cart_width / 2, cart_y])

    # 막대 끝 좌표 계산 (삼각함수)
    x_start = x
    y_start = cart_y + cart_height          # 카트 윗면에서 시작
    x_end   = x_start + pole_len * np.sin(theta)  # 수평 성분
    y_end   = y_start + pole_len * np.cos(theta)  # 수직 성분
    pole_line.set_data([x_start, x_end], [y_start, y_end])

    # 에피소드 텍스트 갱신
    episode_text.set_text(f'Episode: {ep_num}')

    # 갱신된 객체 반환 (blit 모드에서 어떤 객체를 다시 그릴지 알려줌)
    return cart_rect, pole_line, episode_text

ani = FuncAnimation(fig, updateFunc, frames=frame_count, interval=50, repeat=False)
plt.close(fig)        # 정적 figure 창 닫기
HTML(ani.to_jshtml()) # 애니메이션을 HTML+JS로 변환해 노트북에 표시
```

<img src="rl3cartpole2.png" width="500">

---

### 🔍 디버깅 팁 — `frames=1`로 첫 프레임만 확인

전체 애니메이션을 돌리기 전에 카트/막대가 제대로 그려지는지 먼저 확인할 때 쓰는 테스트 코드.

```python
# 디버깅용: frames=1 → 첫 프레임 하나만 렌더링해서 위치/모양 확인
# 전체 frame_count로 돌리면 시간이 오래 걸리므로, 그리기 로직 검증 시 먼저 사용
ani = FuncAnimation(fig, updateFunc, frames=1, interval=50, repeat=False)
plt.close(fig)
HTML(ani.to_jshtml())
```

첫 프레임만 렌더링하면 카트는 보이지만 막대(pole_line)는 아직 초기 데이터(`[]`)라 보이지 않음 → `updateFunc` 안의 `set_data` 로직이 맞는지 확인하는 용도.

<img src="rl3cartpole3.png" width="500">

재생 컨트롤러(슬라이더, Once/Loop/Reflect)도 `frames=1`일 때 동일하게 생성됨 — 컨트롤러 UI 자체는 `to_jshtml()`이 자동으로 붙여주는 것.

<img src="rl3cartpole4.png" width="500">

---

## 📌 자주 만난 에러 정리

|에러|원인|해결|
|---|---|---|
|`TypeError: '>' not supported between 'float' and 'list'`|`best_reward = []`로 초기화|`best_reward = -np.inf`로 변경|
|`ValueError: x and y must have same first dimension`|`reward_list`가 비어있음 (학습 안 돌림)|`range(episodes)`로 학습 루프 실행|
|`IndexError: list index out of range`|`trajectories`와 `episode_numbers` 길이 불일치|초기화 셀 재실행 후 학습 재실행|
|`WARNING: Animation size exceeded limit`|애니메이션 용량 20MB 초과|동작에는 문제 없음, 무시 가능|

---

## 📌 epsilon 감소 방식 비교

```python
# 방법 1: 곱하기 (권장)
epsilon *= epsilon_decay   # 1.0 → 0.999 → 0.998 → ... → 0.05에서 멈춤

# 방법 2: 음수 decay로 더하기
epsilon_decay = -0.001
epsilon += epsilon_decay

# ❌ 잘못된 방법 (양수를 더하면 epsilon이 계속 커짐)
epsilon += 0.999  # 1.0 → 1.999 → 2.998 → ... (항상 랜덤 행동만 함)
```