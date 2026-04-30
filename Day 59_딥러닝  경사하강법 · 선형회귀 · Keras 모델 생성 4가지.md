# Day 59_딥러닝 : 경사하강법 · 선형회귀 · Keras 모델 생성 4가지

## 📅 2026-04-29

---
# 📄 미분 기초 — 평균변화율과 순간변화율(미분계수)

---

## 📌 개념 정리

### 미분이란?

**미분** 은 **함수의 순간변화율을 구하는 연산**이다.

- 딥러닝 역전파(Backpropagation)의 핵심 원리
- 손실 함수를 각 가중치로 미분 → 가중치 업데이트 방향 결정

---

## 📌 평균변화율 (Average Rate of Change)

### 개념

두 점 사이의 변화량 비율 → **할선(Secant Line)의 기울기**

### 서울 → 부산 예시

```
서울 → 부산
  거리  : 400 km
  시간  : 4시간
  평균속도 : 400 / 4 = 100 km/h  ← 평균변화율
```

### 공식

$$\text{평균변화율} = \frac{\Delta \text{거리}}{\Delta \text{시간}} = \frac{\Delta y}{\Delta x} = \frac{f(t_2) - f(t_1)}{t_2 - t_1} = \text{기울기}$$

---

## 📌 순간변화율 (Instantaneous Rate of Change)

### 개념

$t_2$ 를 $t_1$ 에 **무한히 가까이** 보낼 때의 극한값 → **접선(Tangent Line)의 기울기**  
= **미분계수 (Derivative at a point)**

### 수식

$$\lim_{t \to 2} \frac{f(t) - f(2)}{t - 2} = \lim_{t \to 2^-} \frac{f(t) - f(2)}{t - 2} = \lim_{t \to 2^+} \frac{f(t) - f(2)}{t - 2}$$

$$\lim_{t \to 2} \frac{f(t) - f(2)}{t - 2} = \text{순간변화율}$$

> 좌극한 = 우극한 일 때만 극한이 존재 → 미분 가능

---

## 📌 그래프 해석

<img src="평균변화율.png" width="500">

|선|의미|
|---|---|
|B (파란 직선)|서울 → 부산 전체 **평균변화율** (할선)|
|A, A₁, A₂ (빨간 할선)|t₂가 t₁에 가까워질수록 접선에 수렴|
|대전 점 (파란 점)|t₁ 지점 — 순간변화율을 구하려는 점|
|검은 곡선 위 접선|**순간변화율** = 미분계수|

> `t₂ → t₁` : 할선이 점점 접선으로 수렴하는 과정 = 미분의 기하학적 의미

---

## 📌 평균변화율 vs 순간변화율 비교

|구분|평균변화율|순간변화율|
|---|---|---|
|정의|두 점 사이의 변화량 비율|한 점에서의 극한값|
|기하학적 의미|할선(Secant Line)의 기울기|접선(Tangent Line)의 기울기|
|수식|$\dfrac{f(t_2)-f(t_1)}{t_2-t_1}$|$\lim_{t \to a}\dfrac{f(t)-f(a)}{t-a}$|
|다른 이름|—|미분계수 / 도함수값|

---

## 📌 딥러닝에서의 역할

|개념|딥러닝에서의 역할|
|---|---|
|미분 (순간변화율)|손실 함수를 각 가중치로 미분 → 경사(Gradient) 계산|
|기울기 (Gradient)|손실을 줄이는 방향 결정|
|경사 하강법|기울기 **반대 방향**으로 가중치 업데이트|

```
손실 L을 가중치 W로 미분
  → 기울기 ∂L/∂W 계산
  → W = W - lr × ∂L/∂W  (가중치 업데이트)
  → 반복 → loss 감소
```

---

## 🔑 핵심 포인트

> **평균변화율** = Δy/Δx = 두 점을 잇는 **할선의 기울기**  
> **순간변화율** = 극한(lim) → **접선의 기울기** = 미분계수  
> `t₂ → t₁` : 할선이 접선으로 수렴 — 미분의 기하학적 의미  
> **좌극한 = 우극한** 일 때만 극한 존재 → 미분 가능  
> **딥러닝 역전파** = 손실 함수를 미분해 각 가중치의 책임량(Gradient) 계산

---
# 📄 tf6gd.py — 비용 함수(Cost Function) · 경사하강법 · 최적 가중치

---

## 📌 개념 정리

### 비용 함수(Cost Function)란?

**비용 함수** 는 **머신러닝 모델의 예측값과 실제값 간의 오차를 수치화하는 함수**다.

- 오차가 클수록 비용 함수 값이 커짐
- 학습 목표 = 비용 함수 값을 **최소화**하는 W(가중치)와 B(편향)를 찾는 것
- 인공 신경망은 **델타 규칙(경사하강법)** 으로 W와 B를 반복 갱신

### 평균제곱오차 (MSE, Mean Squared Error)

**경사하강법** 은 최소제곱법 대신 **MSE** 를 정의하고, 그 오차를 최소화하기 위해 파라미터를 반복 갱신한다.

$$\text{Cost(MSE)} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

|기호|의미|
|---|---|
|$y_i$|실제값|
|$\hat{y}_i$|예측값 (= $w \cdot x + b$)|
|$n$|데이터 수|

### 경사하강법(Gradient Descent)이란?

**경사하강법** 은 **비용 함수의 기울기(미분값)를 이용해 W를 최솟값 방향으로 반복 업데이트하는 알고리즘**이다.

$$W := W - \alpha \cdot \frac{\partial \text{Cost}}{\partial W}$$

|기호|의미|
|---|---|
|$\alpha$|학습률 (learning rate) — 한 번에 얼마나 이동할지|
|$\frac{\partial \text{Cost}}{\partial W}$|cost를 W로 미분한 기울기 (Gradient)|

```
현재 w에서 cost 함수의 기울기(미분값) 계산
  → 기울기 반대 방향으로 w를 조금씩 이동
  → cost가 줄어드는 방향으로 반복
  → cost 최솟값에 수렴 → 최적 w 획득
```

---

## 📌 그래프 해석

<img src="tf6gd.png" width="500">

| 구간      | 의미                        |
| ------- | ------------------------- |
| w < 1.0 | 기울기 음수 → w를 오른쪽으로 이동      |
| w > 1.0 | 기울기 양수 → w를 왼쪽으로 이동       |
| w = 1.0 | 기울기 = 0 → **최솟값, 최적 가중치** |

> x = y 데이터이므로 **w = 1.0** 일 때 cost = 0 (완벽한 예측)  
> cost 곡선은 **U자형 포물선(볼록 함수)** — 전역 최솟값이 반드시 존재

---

## 💻 전체 실습 코드

```python
import math
import numpy as np

# ─────────────────────────────────────────
# 비용 함수(MSE) 직접 계산
# ─────────────────────────────────────────
real = np.array([10, 9, 3, 2, 11])   # 실제값 y
# pred = np.array([11, 5, 2, 4, 3])  # 예측값 — 차이가 큰 경우 (cost 크게 나옴)
pred = np.array([10, 8, 3, 4, 10])   # 예측값 — 차이가 작은 경우 (cost 작게 나옴)

cost = 0
for i in range(len(real)):
    # (예측값 - 실제값)² 를 누적 합산 → MSE 분자 계산
    cost += math.pow(pred[i] - real[i], 2)
    print(cost)

# 누적합 / 데이터 수 = MSE
print('cost : ', cost / len(real))
# 실제값과 예측값의 차이가 작을수록 cost → 0에 근사
# wx + b 수식에서 w와 b를 최적의 추세선이 만들어지도록 갱신해야 함

# ─────────────────────────────────────────
# 최적의 W(가중치) 탐색 — w를 바꿔가며 cost 변화 시각화
# ─────────────────────────────────────────
import tensorflow as tf
import matplotlib.pyplot as plt
import koreanize_matplotlib

x = [1, 2, 3, 4, 5]   # 입력값
y = [1, 2, 3, 4, 5]   # 정답값 (x=y 이므로 최적 w=1.0)
b = 0                  # bias는 편의상 0으로 고정

# w 후보값과 그에 대응하는 cost를 저장할 리스트
w_val = []
cost_val = []

# w를 -5.0 ~ 4.9 범위로 0.1 간격으로 변화시키며 cost 계산
for i in range(-30, 50):
    feed_w = i * 0.1                                   # 탐색할 w 후보값

    hypothesis = tf.multiply(feed_w, x) + b            # 예측값 : y_hat = w * x + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))   # MSE = mean((y_hat - y)²)

    cost_val.append(cost)     # cost 기록
    w_val.append(feed_w)      # w 기록
    print(f'{i}, cost:{cost.numpy():.4f}, weight:{feed_w:.1f}')

# w에 따른 cost 변화를 U자형 곡선으로 시각화
# → 곡선의 최저점(w=1.0)이 최적 가중치
plt.plot(w_val, cost_val, marker='o')
plt.xlabel('w(가중치)')
plt.ylabel('cost(손실,비용)')
plt.show()
```

---

## 🔑 핵심 포인트

> **비용 함수(MSE)** = $\frac{1}{n}\sum(\hat{y}-y)^2$ — 예측 오차를 수치화, 작을수록 좋은 모델  
> **경사하강법** = cost를 W로 미분 → 기울기 반대 방향으로 W 갱신 반복  
> **w = 1.0** 일 때 cost = 0 → x = y 데이터의 최적 가중치  
> **U자형 포물선** — 전역 최솟값 존재, 경사하강법으로 반드시 수렴 가능  
> `tf.reduce_mean(tf.square(hypothesis - y))` = TensorFlow로 MSE 계산하는 기본 패턴

---
# 📄 tf7reg.py — 단순 선형회귀 · 상관계수 · 결정계수

---

## 📌 개념 정리

### 단순 선형회귀(Simple Linear Regression)란?

**단순 선형회귀** 는 **입력 변수(x) 하나로 연속형 출력 변수(y)를 예측하는 모델**이다.

- 데이터를 가장 잘 설명하는 **직선(추세선)** 을 찾는 것이 목표
- 수식 : $\hat{y} = w \cdot x + b$
- 딥러닝에서는 `Dense(1, activation='linear')` 출력층으로 구현

|기호|의미|
|---|---|
|$w$|가중치 (weight) — 직선의 기울기|
|$b$|편향 (bias) — 직선의 절편|
|$\hat{y}$|예측값|

---

### 상관계수 (Correlation Coefficient)

**상관계수** 는 **두 변수 간의 선형 관계의 강도와 방향을 -1 ~ 1 사이의 값으로 나타낸 지표**다.

$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \cdot \sum(y_i-\bar{y})^2}}$$

|범위|의미|
|---|---|
|1.0에 가까울수록|강한 양의 선형 관계|
|-1.0에 가까울수록|강한 음의 선형 관계|
|0에 가까울수록|선형 관계 없음|

> 이번 실습 데이터 상관계수 = **0.975** → 매우 강한 양의 선형 관계

---

### 결정계수 (R², Coefficient of Determination)

**결정계수(R²)** 는 **모델이 실제 데이터의 분산을 얼마나 설명하는지 나타내는 지표**다.

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

|범위|의미|
|---|---|
|1.0|완벽한 예측|
|0.9 이상|매우 좋은 모델|
|0.0|모델이 아무런 설명력 없음|
|음수|평균으로 예측하는 것보다 나쁜 모델|

> 이번 실습 R² = **0.914** → 전체 분산의 약 91.4%를 설명

---

### 모델 구조

```
입력층  : 특성 1개 (공부 시간)
  ↓
은닉층  : Dense(5) + relu   ← 비선형 패턴 학습
  ↓
출력층  : Dense(1) + linear ← 연속형 값 그대로 출력 (회귀)
```

### 파라미터 수 계산

|레이어|계산|파라미터 수|
|---|---|---|
|dense (입력1 → 출력5)|(1+1) × 5|**10**|
|dense_1 (입력5 → 출력1)|(5+1) × 1|**6**|
|**합계**||**16**|

---

### 활성화 함수 선택 이유

|레이어|활성화 함수|이유|
|---|---|---|
|은닉층|`relu`|비선형성 추가 → 복잡한 패턴 학습|
|출력층|`linear`|연속형 값을 그대로 출력 — 회귀 기본값|

---

## 📌 그래프 해석

<img src="tf7reg.png" width="500">

- 빨간 점 (`o`) : 실제값
- 파란 점선 (`--`) : 모델 예측값 (추세선)
- 점들이 추세선에 가까울수록 R²가 1에 가까움

---

## 💻 전체 실습 코드

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import numpy as np

# ─────────────────────────────────────────
# 1) 데이터 준비
# ─────────────────────────────────────────
xdata = np.array([1,2,3,4,5], dtype='float32').reshape(-1, 1)  # 입력 : 공부 시간
ydata = np.array([1.2, 2.0, 3.0, 3.5, 5.5])                   # 정답 : 성적

# 상관계수 : -1 ~ 1 사이 값, 1에 가까울수록 강한 양의 선형 관계
# 0.975 → 공부 시간과 성적 사이에 매우 강한 양의 선형 관계
print('상관 계수 : ', np.corrcoef(xdata.ravel(), ydata.ravel()))

# ─────────────────────────────────────────
# 2) 모델 구성
# ─────────────────────────────────────────
model = Sequential()
model.add(Input((1, )))                           # 입력층 : 특성 1개 (공부 시간)
model.add(Dense(units=5, activation='relu'))      # 은닉층 : 뉴런 5개, relu로 비선형성 추가
model.add(Dense(units=1, activation='linear'))    # 출력층 : 뉴런 1개, linear = 연속형 값 그대로 출력
print(model.summary())
# 파라미터 수 : dense=(1+1)×5=10 / dense_1=(5+1)×1=6 → 총 16개

# ─────────────────────────────────────────
# 3) 컴파일
# ─────────────────────────────────────────
# loss='mse' : 회귀 문제의 기본 손실 함수 — 예측값과 실제값의 차이 제곱 평균
# optimizer='sgd' : 확률적 경사하강법으로 가중치 업데이트
model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

# ─────────────────────────────────────────
# 4) 학습
# ─────────────────────────────────────────
# shuffle=True : 매 epoch마다 데이터 순서를 섞어 학습 → 과적합 방지, 일반화 성능 향상
model.fit(x=xdata, y=ydata, epochs=100, batch_size=1, verbose=1, shuffle=True)

# ─────────────────────────────────────────
# 5) 평가
# ─────────────────────────────────────────
loss_eval = model.evaluate(x=xdata, y=ydata)
print('loss_eval : ', loss_eval)  # [loss, mse]

# ─────────────────────────────────────────
# 6) 예측 및 결정계수
# ─────────────────────────────────────────
pred = model.predict(xdata)
print('pred : ', pred.ravel())   # 예측값
print('real : ', ydata.ravel())  # 실제값

# 결정계수(R²) : 모델이 실제 데이터의 분산을 얼마나 설명하는지
# 1.0 = 완벽한 예측 / 0.9 이상 = 매우 좋은 모델
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(ydata, pred))  # 0.914 → 전체 분산의 91.4% 설명

# ─────────────────────────────────────────
# 7) 시각화
# ─────────────────────────────────────────
import matplotlib.pyplot as plt

# 빨간 점(실제값) vs 파란 점선(예측 추세선) 비교
plt.scatter(xdata, ydata, color='r', marker='o', label='real')
plt.plot(xdata, pred, 'b--', label='pred')
plt.legend()
plt.show()

# ─────────────────────────────────────────
# 8) 새로운 값 예측
# ─────────────────────────────────────────
# 학습 범위(1~5) 밖의 값도 예측 가능 — 선형 외삽(extrapolation)
new_x = np.array([1.5, 5.7, -3.0]).reshape(-1, 1)
new_pred = model.predict(new_x)
print('새값 예측 결과 : ', new_pred.ravel())
# [ 1.629  6.270  -1.106 ] ← 학습 범위 밖 값도 직선으로 외삽해서 예측
```

---

## 🔑 핵심 포인트

> **단순 선형회귀** = 입력 1개로 연속형 출력 예측 — 출력층 `linear` / 손실함수 `mse`  
> **상관계수** = -1 ~ 1, 절댓값이 클수록 강한 선형 관계 / 이번 실습 = **0.975**  
> **결정계수(R²)** = 모델의 설명력, 1에 가까울수록 좋음 / 이번 실습 = **0.914**  
> `shuffle=True` — 매 epoch마다 데이터 순서 섞기 → 과적합 방지  
> **외삽(Extrapolation)** — 학습 범위 밖 값도 예측 가능하나 신뢰도는 낮아질 수 있음

---
# 📄 tf8reg.py — Keras 모델 생성 방법 4가지 · 선형회귀 실습

---

## 📌 개념 정리

### Keras 모델 생성 방법 비교

|방법|특징|사용 시기|
|---|---|---|
|**Sequential API**|레이어를 순서대로 쌓는 구조|단순한 선형 구조 모델|
|**Functional API**|다중 입출력, 브랜치 구조 가능|복잡한 모델 (ResNet 등)|
|**Model Subclassing**|`Model` 상속, 완전 커스텀|연구용, 세밀한 제어 필요 시|
|**Custom Layer**|`Layer` 상속, 직접 연산 정의|가중치/연산을 직접 구현할 때|

---

### 실습 데이터

**공부 시간에 따른 성적** — 회귀(Regression) 문제

|공부 시간 (x)|성적 (y)|
|---|---|
|1|15|
|2|32|
|3|39|
|4|55|
|5|60|

> 연속형 값 예측 → 출력층 활성화 함수 : `linear` / 손실 함수 : `mse`

---

### Custom Layer 동작 원리

**`build()`** — 레이어가 처음 호출될 때 가중치를 초기화하는 메서드

**`call()`** — 실제 순전파 연산을 정의하는 메서드

```
model.fit() 호출
  → 첫 번째 데이터 입력 시 build() 자동 호출 → 가중치 초기화
  → 이후 매 forward pass마다 call() 호출 → y = wx + b 연산
```

---

## 💻 전체 실습 코드

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import optimizers
import numpy as np

# ─────────────────────────────────────────
# 데이터 준비
# ─────────────────────────────────────────
# reshape(-1, 1) : 케라스 모델 입력은 2차원 (샘플수, 특성수) 형태 필요
xdata = np.array([1,2,3,4,5], dtype=np.float32).reshape(-1, 1)  # (5, 1)
ydata = np.array([15,32,39,55,60], dtype=np.float32).reshape(-1, 1)  # (5, 1)

# ─────────────────────────────────────────
# 방법1 — Sequential API
# ─────────────────────────────────────────
# 레이어를 순서대로 쌓는 가장 단순한 구조
# 단방향 흐름(입력→출력)만 가능
print('모델 생성 방법1 - Sequential API')
model = Sequential()
model.add(Input((1, )))                        # 입력층 : 특성 1개
model.add(Dense(units=4, activation='relu'))   # 은닉층 : 뉴런 4개, relu로 비선형성 추가
model.add(Dense(units=1, activation='linear')) # 출력층 : 연속형 값 그대로 출력 (회귀)
# Sequential 리스트 방식 (동일한 결과, 참고용)
# model = Sequential([
#     Input((1, )),
#     Dense(units=4, activation='relu'),
#     Dense(units=1, activation='linear')
# ])

print(model.summary())
# 파라미터 수 : dense=(1+1)×4=8 / dense_1=(4+1)×1=5 → 총 13개

# loss='mse' : 회귀 문제의 기본 손실 함수
# optimizer=SGD : 확률적 경사하강법으로 가중치 업데이트
opti = optimizers.SGD(learning_rate=0.001)
model.compile(loss='mse', optimizer=opti, metrics=['mse'])
history = model.fit(x=xdata, y=ydata, batch_size=1, epochs=100, verbose=2)

loss_metrics = model.evaluate(x=xdata, y=ydata)
print('loss_metrics : ', loss_metrics)  # [loss, mse]
ypred = model.predict(xdata, verbose=0)

from sklearn.metrics import r2_score
# r2_score : 결정계수 — 모델이 실제 분산을 얼마나 설명하는지 (1.0에 가까울수록 좋음)
print('설명력 : ', r2_score(ydata, ypred))
print('실제값 : ', ydata.ravel())
print('예측값 : ', ypred.ravel())

import matplotlib.pyplot as plt

# 실제값(빨간 점) vs 예측 추세선(파란 점선) 비교
plt.scatter(xdata, ydata, color='r', marker='o', label='real')
plt.plot(xdata, ypred, 'b--', label='pred')
plt.legend()
plt.show()

# epoch별 mse 변화 시각화 — loss가 수렴하면 학습이 잘 된 것
plt.plot(history.history['mse'], label='mse')
plt.xlabel('epochs')
plt.legend()
plt.show()

# ─────────────────────────────────────────
# 방법2 — Functional API
# ─────────────────────────────────────────
# 레이어를 함수처럼 호출해서 연결하는 방식
# 다중입력, 다중출력, 공유층, 비순차적 흐름 모두 가능
print('\n 모델 생성 방법2 - Functional API')
from tensorflow.keras.models import Model

inputs = Input(shape=(1, ))                               # 입력 텐서 정의
output1 = Dense(units=4, activation='relu')(inputs)       # 입력을 은닉층에 연결
outputs = Dense(units=1, activation='linear')(output1)    # 은닉층을 출력층에 연결

# 입력과 출력 텐서를 연결해 모델 생성
model2 = Model(inputs, outputs)

opti2 = optimizers.SGD(learning_rate=0.001)
model2.compile(loss='mse', optimizer=opti2, metrics=['mse'])
history2 = model2.fit(x=xdata, y=ydata, batch_size=1, epochs=100, verbose=2)
loss_metrics2 = model2.evaluate(x=xdata, y=ydata)
print('loss_metrics2 : ', loss_metrics2)
ypred2 = model2.predict(xdata, verbose=0)

print('설명력 : ', r2_score(ydata, ypred2))
print('실제값 : ', ydata.ravel())
print('예측값 : ', ypred2.ravel())

# ─────────────────────────────────────────
# 방법3 — Model Subclassing
# ─────────────────────────────────────────
# tf.keras.Model을 상속받아 모델을 직접 클래스로 정의
# __init__ : 레이어 정의 / call : 순전파 연산 정의
print('\n모델 생성 방법3 - Sub classing : Model을 상속 받아 직접 모델 생성')
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(units=4, activation='relu')   # 은닉층 정의
        self.d2 = Dense(units=1, activation='linear') # 출력층 정의

    def call(self, x):
        # Input 클래스 대신 call 메서드의 매개변수로 입력 받음
        # 순전파 흐름 : x → d1(은닉층) → d2(출력층)
        x = self.d1(x)
        return self.d2(x)

model3 = MyModel()

opti3 = optimizers.SGD(learning_rate=0.001)
model3.compile(loss='mse', optimizer=opti3, metrics=['mse'])
history3 = model3.fit(x=xdata, y=ydata, batch_size=1, epochs=100, verbose=2)
loss_metrics3 = model3.evaluate(x=xdata, y=ydata)
print('loss_metrics3 : ', loss_metrics3)
ypred3 = model3.predict(xdata, verbose=0)

print('설명력 : ', r2_score(ydata, ypred3))
print('실제값 : ', ydata.ravel())
print('예측값 : ', ypred3.ravel())

# ─────────────────────────────────────────
# 방법3-1 — Custom Layer
# ─────────────────────────────────────────
# tf.keras.layers.Layer를 상속받아 레이어 자체를 직접 구현
# build() : 가중치 초기화 (첫 호출 시 자동 실행)
# call()  : 실제 연산 정의 (순전파마다 실행)
print('\n모델 생성 방법3 - 1 : Custom Layer 층 사용')
from tensorflow.keras.layers import Layer

class MyLayer(Layer):
    def __init__(self, units=1, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.units = units  # 출력 뉴런 수 저장

    def build(self, input_shape):
        # 첫 번째 데이터 입력 시 자동 호출 → 가중치(w)와 편향(b) 초기화
        print(f'build:input_shape={input_shape}')
        # add_weight : 학습 가능한 가중치 생성
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),  # (입력수, 출력수)
            initializer='random_normal',           # 랜덤 정규분포로 초기화
            trainable=True                         # 역전파로 업데이트 가능
        )
        self.b = self.add_weight(
            shape=(self.units, ),   # (출력수,)
            initializer='zeros',    # 0으로 초기화
            trainable=True
        )

    def call(self, inputs):
        # 선형 변환 : y = wx + b (행렬곱 + 편향)
        return tf.matmul(inputs, self.w) + self.b

class MLP(Model):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.linear1 = MyLayer(2)  # 은닉층 : 출력 뉴런 2개
        self.linear2 = MyLayer(1)  # 출력층 : 출력 뉴런 1개

    def call(self, inputs):
        net = self.linear1(inputs)   # 은닉층 통과
        net = tf.nn.relu(net)        # relu 활성화 함수 적용
        return self.linear2(net)     # 출력층 통과

model4 = MLP()

opti4 = optimizers.SGD(learning_rate=0.001)
model4.compile(loss='mse', optimizer=opti4, metrics=['mse'])
history4 = model4.fit(x=xdata, y=ydata, batch_size=1, epochs=100, verbose=0)
loss_metrics4 = model4.evaluate(x=xdata, y=ydata)
print('loss_metrics4 : ', loss_metrics4)
ypred4 = model4.predict(xdata, verbose=0)

print('설명력 : ', r2_score(ydata, ypred4))
print('실제값 : ', ydata.ravel())
print('예측값 : ', ypred4.ravel())
```

---

## 📌 방법별 비교 정리

|항목|Sequential|Functional|Subclassing|Custom Layer|
|---|---|---|---|---|
|구현 난이도|⭐|⭐⭐|⭐⭐⭐|⭐⭐⭐⭐|
|유연성|낮음|중간|높음|매우 높음|
|다중 입출력|❌|✅|✅|✅|
|가중치 직접 제어|❌|❌|❌|✅|
|주요 메서드|`add()`|`Model(inputs, outputs)`|`call()`|`build()` + `call()`|

---

## 📌 Custom Layer 핵심 메서드

|메서드|호출 시점|역할|
|---|---|---|
|`__init__()`|객체 생성 시|하이퍼파라미터 저장|
|`build()`|첫 데이터 입력 시 자동 호출|가중치 초기화 (`add_weight`)|
|`call()`|순전파마다 호출|실제 연산 정의 (`y = wx + b`)|

> `build()` 가 `call()` 보다 먼저 실행됨 — 가중치가 없으면 연산 불가

---

## 🔑 핵심 포인트

> **Sequential** = 단순 스택 구조, 가장 쉬움 — 현업 기본 시작점  
> **Functional API** = 레이어를 함수처럼 연결 — 복잡한 구조에 적합  
> **Subclassing** = `Model` 상속, `call()`에 순전파 정의 — 연구용  
> **Custom Layer** = `Layer` 상속, `build()`로 가중치 직접 생성 — 가장 세밀한 제어  
> `add_weight` — Custom Layer에서 학습 가능한 가중치를 직접 등록하는 함수  
> `tf.matmul(inputs, self.w) + self.b` = 선형 변환 $y = Wx + b$ 의 행렬 연산

---

# 📄 tf9reg_adver.py — 다중 선형회귀 · 정규화 · plot_model · validation_split

---

## 📌 개념 정리

### 다중 선형회귀(Multiple Linear Regression)란?

**다중 선형회귀** 는 **여러 개의 입력 변수(x)로 연속형 출력 변수(y)를 예측하는 모델**이다.

$$\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$

|입력 변수|의미|
|---|---|
|`tv`|TV 광고비|
|`radio`|라디오 광고비|
|`newspaper`|신문 광고비|
|`sales`|매출액 (정답, y)|

---

### 정규화(Normalization)란?

**정규화** 는 **feature 간 단위/스케일 차이를 0~1 범위로 맞추는 전처리 작업**이다.

- feature 간 단위 차이가 클 경우 → 큰 값의 feature가 학습을 지배
- 정규화로 모든 feature를 동등한 스케일로 맞춰 학습 안정화

|방법|수식|특징|
|---|---|---|
|**MinMaxScaler**|$(x - x_{min}) / (x_{max} - x_{min})$|0~1 범위로 변환|
|**StandardScaler**|$(x - \mu) / \sigma$|평균 0, 표준편차 1|

> `minmax_scale(fdata, axis=0)` — axis=0 : 열(feature) 기준으로 정규화

---

### validation_split vs validation_data

|방법|설명|사용 시기|
|---|---|---|
|`validation_split=0.2`|train 데이터의 20%를 자동으로 검증용으로 분리|간편하게 쓸 때|
|`validation_data=(x_test, y_test)`|검증 데이터를 직접 지정|데이터를 직접 관리할 때|

> `validation_split=0.2` 사용 시 — **shuffle 없이 뒤에서 20%** 를 검증용으로 사용

---

### plot_model이란?

**`tf.keras.utils.plot_model()`** 은 **케라스 모델 구조를 이미지 파일로 저장하는 함수**다.

- `graphviz` + `pydot` 설치 필요
- 모델 구조 시각화 → 레이어 연결 관계, shape, activation 한눈에 확인

```bash
pip install pydot
# graphviz 실행파일도 별도 설치 필요 (https://graphviz.gitlab.io/download/)
```

---

### 모델 구조

```
입력층  : 특성 3개 (tv, radio, newspaper)
  ↓
은닉층1 : Dense(16) + relu
  ↓
은닉층2 : Dense(8) + relu
  ↓
출력층  : Dense(1) + linear  ← 연속형 값 출력 (회귀)
```

### 파라미터 수 계산

|레이어|계산|파라미터 수|
|---|---|---|
|dense (입력3 → 출력16)|(3+1) × 16|**64**|
|dense_1 (입력16 → 출력8)|(16+1) × 8|**136**|
|dense_2 (입력8 → 출력1)|(8+1) × 1|**9**|
|**합계**||**209**|

---

## 💻 전체 실습 코드

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# 1) 데이터 로드 및 준비
# ─────────────────────────────────────────
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv")
print(data.head(2))
del data['no']   # 불필요한 인덱스 컬럼 제거
print(data.head(2))

# feature(입력)와 label(정답) 분리
fdata = data[['tv', 'radio', 'newspaper']]  # 입력 변수 3개
ldata = data.iloc[:, [3]]                   # sales (4번째 컬럼)
print(fdata.head(2))
print(ldata[:2])

# ─────────────────────────────────────────
# 2) 정규화 (MinMax Scaling)
# ─────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler

# MinMaxScaler 객체 방식 (주석 처리 — 참고용)
# scaler = MinMaxScaler(feature_range=(0, 1))
# fedata = scaler.fit_transform(fdata)

# minmax_scale : 함수형으로 바로 사용
# axis=0 : 열(feature) 기준으로 정규화 → 각 feature를 0~1 범위로 변환
# copy=True : 원본 데이터 보존
fedata = minmax_scale(fdata, axis=0, copy=True)
print(fedata[:3])

# ─────────────────────────────────────────
# 3) train / test 분리
# ─────────────────────────────────────────
from sklearn.model_selection import train_test_split

# stratify : 분류에서 클래스 비율 유지용 → 회귀에서는 사용 안 함
x_train, x_test, y_train, y_test = train_test_split(
    fedata, ldata, shuffle=True, test_size=0.3, random_state=123
)
print(x_train[:2], x_train.shape)  # (140, 3)
print(x_test[:2], x_test.shape)    # (60, 3)

# ─────────────────────────────────────────
# 4) Sequential API 모델 구성
# ─────────────────────────────────────────
model = Sequential()
model.add(Input(shape=(3, )))                          # 입력층 : feature 3개
model.add(Dense(units=16, activation='relu'))          # 은닉층1 : 뉴런 16개
model.add(Dense(units=8, activation='relu'))           # 은닉층2 : 뉴런 8개
model.add(Dense(units=1, activation='linear'))         # 출력층 : 회귀 → linear
print(model.summary())
# 파라미터 수 : (3+1)×16=64 / (16+1)×8=136 / (8+1)×1=9 → 총 209개

# ─────────────────────────────────────────
# 5) 모델 구조 이미지 저장
# ─────────────────────────────────────────
# graphviz + pydot 설치 필요
# pip install pydot / graphviz 실행파일 설치 및 PATH 등록
tf.keras.utils.plot_model(
    model,
    to_file='aaa.png',
    show_shapes=True,               # 각 layer의 입력/출력 shape 표시
    show_layer_names=True,          # layer 이름 표시
    show_dtype=True,                # 데이터 타입 표시
    show_layer_activations=True,    # activation 함수 표시
    dpi=96                          # 이미지 해상도
)

# ─────────────────────────────────────────
# 6) 컴파일 및 학습
# ─────────────────────────────────────────
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# validation_split=0.2 : x_train의 뒤 20%를 자동으로 검증용으로 분리
# → 매 epoch마다 val_loss, val_mse를 함께 기록
history = model.fit(
    x_train, y_train,
    epochs=100, batch_size=32, verbose=2,
    validation_split=0.2
)

# ─────────────────────────────────────────
# 7) 평가
# ─────────────────────────────────────────
ev_loss = model.evaluate(x_test, y_test, verbose=0)
print('ev_loss : ', ev_loss)  # [loss, mse]

# history.history : epoch별 loss/mse/val_loss/val_mse 딕셔너리
print('history val_loss : ', history.history['val_loss'])  # 검증 loss
print('history val_mse : ', history.history['val_mse'])    # 검증 mse
print('history loss : ', history.history['loss'])          # 학습 loss
print('history mse : ', history.history['mse'])            # 학습 mse

# ─────────────────────────────────────────
# 8) loss 시각화
# ─────────────────────────────────────────
import matplotlib.pyplot as plt

# val_loss가 loss보다 높고 계속 올라가면 과적합 신호
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['loss'], label='loss')
plt.legend()
plt.show()

# ─────────────────────────────────────────
# 9) 예측 및 결정계수
# ─────────────────────────────────────────
from sklearn.metrics import r2_score
print('설명력:', r2_score(y_test, model.predict(x_test)))

pred = model.predict(x_test[:5])
print('예측값 : ', pred.ravel())
print('실제값 : ', y_test[:5].values.ravel())

# ─────────────────────────────────────────
# 10) Functional API
# ─────────────────────────────────────────
print('\n\nFunctional api를 사용한 방법 -------------')
from tensorflow.keras.models import Model

# name 인자 : 텐서보드, plot_model 등에서 레이어 식별에 사용
inputs = Input(shape=(3, ), name='input_layer')
x = Dense(units=16, activation='relu', name='hidden_layer1')(inputs)
x = Dense(units=16, activation='relu', name='hidden_layer2')(x)   # name 중복 주의
outputs = Dense(units=1, activation='linear', name='output_layer')(x)

# 입력 텐서와 출력 텐서를 연결해 모델 생성
func_model = Model(inputs, outputs)

func_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
history = func_model.fit(
    x_train, y_train,
    epochs=100, batch_size=32, verbose=2,
    validation_split=0.2
)

ev_loss = func_model.evaluate(x_test, y_test, verbose=0)
print('ev_loss : ', ev_loss)
```

---

## 📌 history.history 키 정리

|키|의미|
|---|---|
|`loss`|epoch별 train 손실|
|`mse`|epoch별 train MSE|
|`val_loss`|epoch별 validation 손실|
|`val_mse`|epoch별 validation MSE|

> `val_` 접두사가 붙으면 검증 데이터 기준 — `validation_split` 또는 `validation_data` 사용 시 생성됨

---

## 🔑 핵심 포인트

> **다중 선형회귀** = 입력 변수 여러 개 → 연속형 출력 예측, 출력층 `linear` / 손실함수 `mse`  
> **정규화** = feature 간 스케일 차이 제거 → 학습 안정화, `minmax_scale` 로 0~1 변환  
> **validation_split=0.2** = train 데이터 뒤 20% 자동 검증용 분리 — shuffle 없이 뒤에서 자름  
> **plot_model** = 모델 구조를 이미지로 저장 — graphviz + pydot 설치 필요  
> **val_loss 증가** = 과적합 신호 → Early Stopping, Dropout 고려  
> `history.history` = epoch별 loss/mse/val_loss/val_mse 딕셔너리 — 학습 곡선 그릴 때 사용


