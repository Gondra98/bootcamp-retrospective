# Day76_강화학습 : Gymnasium · LiDAR 시뮬레이션 · SimpleLidarEnv

## 📅 2026-05-28

---
# 📄 lidar2.py — Gymnasium · LiDAR · SimpleLidarEnv

> ⚠️ 미완성 — `render()` 내 에이전트 삼각형 시각화 파트 미구현

---

## 📌 핵심 개념 정리

### 🔷 Gymnasium이란?

OpenAI Gym의 후속 프로젝트로, 강화학습 환경을 표준화된 인터페이스로 제공하는 라이브러리.

```
현재 상태(obs) 제공 → 에이전트가 행동 선택 → 환경이 행동 반영 → 보상 + 새 상태 반환
```

|구성 요소|설명|
|---|---|
|`action_space`|에이전트가 선택 가능한 행동 범위 정의|
|`observation_space`|환경이 반환하는 상태 벡터 형태 정의|
|`reset()`|에피소드 초기화|
|`step(action)`|행동 적용 → (obs, reward, terminated, truncated, info) 반환|
|`render()`|현재 상태 시각화|

> `gym.Env`를 상속받아 커스텀 환경을 만들 때 위 두 가지(`action_space`, `observation_space`)는 **반드시 선언**해야 한다.

---

### 🔷 Ray Marching (레이 전진) 방식 LiDAR 시뮬레이션

실제 레이저 물리학이 아닌, **2D 평면에서 가상의 레이를 step 단위로 전진시키며 충돌 여부를 체크**하는 방식으로 LiDAR를 시뮬레이션.

```
에이전트 위치에서 NUM_RAYS개의 레이를 FOV 범위로 균등 분포해 발사
    ↓
각 레이를 STEP_MARCH 단위로 전진
    ↓
경계 이탈 or 장애물 충돌 시 stop → 거리 기록
    ↓
충돌 거리 배열(dists) 반환 → 강화학습 상태(state)로 활용
```

---

### 🔷 보상 설계 (Reward Shaping)

|조건|보상|
|---|---|
|매 스텝|-0.01 (생존 패널티, 빠른 도달 유도)|
|목표에 가까워질수록|+접근량 (Δdist)|
|목표 도달|+1.0|
|충돌|-1.0|

---

## 🗺️ 전체 흐름

```
STEP 1. 환경 상수 설정 (월드 크기, 장애물, LiDAR 파라미터)
        ↓
STEP 2. 유틸 함수 정의
        inside_worldFunc() — 경계 충돌 체크
        hit_circleFunc()   — 원형 장애물 충돌 체크
        cast_lidar()       — 레이 발사 + 충돌 거리 계산
        ↓
STEP 3. SimpleLidarEnv 클래스 정의 (gym.Env 상속)
        ↓
STEP 4. 메인 실행 — reset() → step() 반복 → render()
```

---

## 💻 전체 코드

### STEP 1. 환경 상수 설정

```python
WORLD_W, WORLD_H = 20.0, 15.0                                              # 시뮬레이션 공간 크기
OBSTACLES = [(6.0, 4.0, 0.5), (8.0, 10.0, 1.5), (15.0, 5.0, 1.0)]        # (cx, cy, r)
NUM_RAYS = 20           # LiDAR 광선 수
FOV = np.deg2rad(150)   # 시야각 150도
MAX_RANGE = 8.0         # LiDAR 최대 감지 거리
STEP_MARCH = 0.05       # 레이 전진 단위 거리
```

---

### STEP 2. 유틸 함수

```python
def inside_worldFunc(x, y):
    # 좌표가 시뮬레이션 공간 경계 내에 있는지 여부
    return (0.0 <= x <= WORLD_W) and (0.0 <= y <= WORLD_H)

def hit_circleFunc(px, py, cx, cy, r):
    # 레이 끝점이 원형 장애물과 충돌했는지 여부
    return (px - cx) ** 2 + (py - cy) ** 2 < r ** 2
```

> ⚠️ 주의: `hit_circleFunc`에서 `<` 비교 연산자 누락 시 항상 숫자값을 반환해 `if hit_circleFunc(...)` 가 **항상 True**가 되는 버그 발생

```python
def cast_lidar(x, y, theta, num_rays=NUM_RAYS, fov=FOV,
               max_range=MAX_RANGE, step=STEP_MARCH):
    start = theta - fov / 2  # 시야각 왼쪽 끝 각도 (첫 번째 레이)
    angles = start + np.arange(num_rays) * (fov / max(num_rays - 1, 1))  # 균등 분포 각도 배열
    dists = np.full(num_rays, max_range, dtype=np.float32)  # 거리 배열 초기화 = max_range

    for i, ang in enumerate(angles):
        dist = 0.0
        hit = False
        while dist < max_range:
            px = x + math.cos(ang) * dist   # 레이 끝점 x
            py = y + math.sin(ang) * dist   # 레이 끝점 y
            if not inside_worldFunc(px, py):
                hit = True
                break
            for (cx, cy, r) in OBSTACLES:
                if hit_circleFunc(px, py, cx, cy, r):
                    hit = True
                    break
            if hit:
                break
            dist += step  # 충돌 없으면 레이 전진
        dists[i] = min(dist, max_range)  # 충돌 거리 기록

    return dists, angles
```

---

### STEP 3. SimpleLidarEnv 클래스

```python
class SimpleLidarEnv(gym.Env):
    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode

        # 반드시 선언해야 하는 두 가지
        self.action_space = spaces.Discrete(3)           # 0: 좌회전, 1: 직진, 2: 우회전
        self.observation_space = spaces.Box(
            low=0.0, high=MAX_RANGE,
            shape=(NUM_RAYS,), dtype=np.float32          # LiDAR 거리 벡터 20개
        )

        self.v = 0.25                                    # 전진 속도
        self.steer_delta = np.deg2rad(8)                 # 회전 각도 8도
        self.goal = np.array([18.0, 12.0], dtype=np.float32)  # 목표 지점
        self.goal_radius = 0.6                           # 목표 도달 판정 반경
        self.max_steps = 400                             # 에피소드 최대 스텝

        self.fig, self.ax = None, None
        self._state = None          # [x, y, theta]
        self._prev_goal_dist = None
        self._steps = 0
```

**주요 메서드**

```python
def _get_obs(self):
    # 현재 상태에서 LiDAR 거리 벡터 반환 (모델 입력값)
    x, y, th = self._state
    obs, _ = cast_lidar(x, y, th)
    return obs.astype(np.float32)

def _get_info(self):
    # 목표까지 거리, 현재 스텝 수 반환
    x, y, _ = self._state
    d = np.linalg.norm(np.array([x, y]) - self.goal)
    return {'goal_dist': float(d), 'steps': self._steps}

def _collision(self):
    # 경계 이탈 or 장애물 충돌 여부
    x, y, _ = self._state
    if not inside_worldFunc(x, y):
        return True
    for (cx, cy, r) in OBSTACLES:
        if hit_circleFunc(x, y, cx, cy, r + 0.25):  # 에이전트 크기 여유값 0.25 추가
            return True
    return False

def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self._state = np.array([2.0, 2.0, np.deg2rad(30.0)], dtype=np.float32)
    self._steps = 0
    self._prev_goal_dist = np.linalg.norm(self._state[:2] - self.goal)
    return self._get_obs(), self._get_info()

def step(self, action):
    self._steps += 1
    x, y, th = self._state

    if action == 0: th += self.steer_delta   # 좌회전
    elif action == 2: th -= self.steer_delta  # 우회전

    x += math.cos(th) * self.v
    y += math.sin(th) * self.v
    self._state = np.array([x, y, th], dtype=np.float32)

    goal_dist = np.linalg.norm(self._state[:2] - self.goal)
    process = self._prev_goal_dist - goal_dist   # 목표 접근량
    self._prev_goal_dist = goal_dist

    reward = 1.0 * process - 0.01   # 접근 보상 - 스텝 패널티
    terminated, truncated = False, False

    if goal_dist < self.goal_radius:  # 목표 도달
        reward += 1.0
        terminated = True
    if self._collision():             # 충돌
        reward -= 1.0
        terminated = True
    if self._steps >= self.max_steps: # 스텝 초과
        terminated = True

    return self._get_obs(), reward, terminated, truncated, self._get_info()
```

---

### STEP 4. render() — ⚠️ 미완성

```python
def render(self):
    if self.render_mode == "human":
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, WORLD_W)
        ax.set_ylim(0, WORLD_H)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Simple Lidar Env')
        ax.plot([0, WORLD_W, WORLD_W, 0, 0], [0, 0, WORLD_H, WORLD_H, 0], lw=2)

        for (cx, cy, r) in OBSTACLES:
            circ = plt.Circle((cx, cy), r, edgecolor='tab:red', facecolor='none', lw=2)
            ax.add_patch(circ)

        goal = plt.Circle(tuple(self.goal), self.goal_radius, edgecolor='tab:blue', facecolor='none')
        ax.add_patch(goal)

        x, y, th = self._state
        L = 0.6
        # ↓ 미구현 — 삼각형 에이전트 시각화 추가 필요
```

**미완성 파트 (추가 필요)**

```python
        # __init__에 추가
        self.fig, self.ax = plt.subplots()

        # render()에 추가
        tip   = np.array([x + L * math.cos(th), y + L * math.sin(th)])
        left  = np.array([x + (L/2) * math.cos(th + np.deg2rad(120)),
                          y + (L/2) * math.sin(th + np.deg2rad(120))])
        right = np.array([x + (L/2) * math.cos(th - np.deg2rad(120)),
                          y + (L/2) * math.sin(th - np.deg2rad(120))])

        triangle = plt.Polygon([tip, left, right], edgecolor='black', facecolor='tab:green')
        ax.add_patch(triangle)
        plt.pause(0.05)
```

---

### STEP 5. 메인 실행

```python
if __name__ == '__main__':
    env = SimpleLidarEnv()
    obs, info = env.reset()
    total_reward = 0.0

    for t in range(500):
        action = env.action_space.sample()   # 무작위 행동 선택
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()

        if terminated or truncated:
            print(f'Episode end at step={t}, total_reward={total_reward:.3f}, info={info}')
            obs, info = env.reset()
            total_reward = 0.0

    env.close()
```

---

## 📌 핵심 정리

- `gym.Env` 상속 시 `action_space`, `observation_space` 반드시 선언
- `cast_lidar()` : 레이를 STEP_MARCH 단위로 전진시키며 충돌 거리 계산 → 강화학습 **상태(state)** 로 활용
- `hit_circleFunc()` : `<` 비교 연산자 누락 주의 → 항상 True 버그 발생
- `persist=True` 없이 `step()` 반복 시 에피소드 내 상태 유지 안 됨
- 보상 설계 : 접근 보상 + 스텝 패널티 + 도달 보너스 + 충돌 패널티 조합
- `_collision()` 에서 `r + 0.25` 여유값 → 에이전트 크기 고려한 충돌 판정