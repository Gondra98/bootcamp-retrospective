# Day 58_딥러닝 : Keras 모델링 · OR·XOR 신경망

## 📅 2026-04-28

---
# 📄 tf3.py — 조건 연산 · 관계/논리 연산 · 배열 변환 · 자료형 변환

---
## 📌 개념 정리

### 조건 연산이란?

**조건 연산** 은 **텐서 환경에서 if문을 대체하는 함수**다.

- 파이썬의 일반 `if` 문은 Graph Execution 환경에서 사용 불가
- `tf.cond`, `tf.case` 를 사용해 텐서 조건 분기 처리
- 반드시 **lambda로 감싸서** 전달해야 함 — 조건 평가 전 양쪽 실행 방지

### 배열 변환 함수란?

**배열 변환 함수** 는 **텐서의 형태(shape)나 차원(dimension)을 변환하는 함수**다.

- `reshape` : 원소 수를 유지하며 형태 변환
- `squeeze` : 크기 1인 차원 제거
- `expand_dims` : 크기 1인 차원 추가
- `cast` : 자료형(dtype) 변환

> CV(컴퓨터 비전)에서 이미지 텐서 shape 조정 시 자주 사용

---

## 📌 조건 연산

### tf.cond() — 삼항 연산자

```python
x = tf.constant(7)
y = tf.constant(3)

# tf.cond(조건, 참일 때 실행할 함수, 거짓일 때 실행할 함수)
# 반드시 lambda로 감싸야 함 — 조건 평가 전에 양쪽이 실행되는 걸 방지
# x > y (7 > 3) → True → tf.add(x, y) = 10
result1 = tf.cond(x > y, lambda: tf.add(x, y), lambda: tf.subtract(x, y))
print(result1)  # tf.Tensor(10, shape=(), dtype=int32)
```

### tf.case() — 다중 조건 (if-elif-else)

```python
f1 = lambda: tf.constant(1)                  # 조건 True 시 반환값
f2 = lambda: tf.constant(tf.multiply(2, 3))  # default 반환값 (2×3=6)

# [(조건, 실행함수)] 리스트를 순서대로 평가
# 모두 False면 default 실행
# x < y (7 < 3) → False → default f2 실행 → 6
result2 = tf.case([(tf.less(x, y), f1)], default=f2)
print(result2)  # tf.Tensor(6, shape=(), dtype=int32)
```

|함수|구조|특징|
|---|---|---|
|`tf.cond`|조건 1개 (삼항)|True/False 중 하나 실행|
|`tf.case`|조건 여러 개|if-elif-else 구조, 순서대로 평가|

---

## 📌 관계 연산 — bool Tensor 반환

```python
# 두 값을 비교하여 True/False bool Tensor를 반환
print(tf.equal(1, 2))           # 1 == 2  → False
print(tf.not_equal(1, 2))       # 1 != 2  → True
print(tf.less(1, 2))            # 1 < 2   → True
print(tf.greater(1, 2))         # 1 > 2   → False
print(tf.greater_equal(1, 2))   # 1 >= 2  → False
```

---

## 📌 논리 연산

```python
# bool 값 간의 논리 연산
print(tf.logical_and(True, False))  # AND → 둘 다 True여야 True → False
print(tf.logical_or(True, False))   # OR  → 하나라도 True면 True → True
print(tf.logical_not(True))         # NOT → True/False 반전 → False
```

---

## 📌 tf.unique() — 중복 제거

```python
kbs = tf.constant([1, 2, 2, 3, 2])

# unique() : 고유값(val)과 역인덱스(idx)를 함께 반환
val, idx = tf.unique(kbs)

# val : 중복 제거된 고유값            → [1, 2, 3]
# idx : 원본 각 원소가 val의 몇 번째인지 → [0, 1, 1, 2, 1]
print('val : ', val)  # tf.Tensor([1 2 3], shape=(3,), dtype=int32)
print('idx : ', idx)  # tf.Tensor([0 1 1 2 1], shape=(5,), dtype=int32)

# tf.gather(val, idx) 하면 idx를 이용해 원본 복원 가능
```

---

## 📌 reduce_* 함수 — 집계 + 차원 축소

```python
ar = [[1., 2.],
      [3., 4.]]

# axis 미지정 : 전체 원소를 대상으로 집계
print(tf.reduce_mean(ar).numpy())           # 2.5   (1+2+3+4)/4

# axis=0 : 행을 없애는 방향 (세로로 합침) → 열별 집계
# [1,2] + [3,4] → [(1+3)/2, (2+4)/2]
print(tf.reduce_mean(ar, axis=0).numpy())   # [2. 3.]

# axis=1 : 열을 없애는 방향 (가로로 합침) → 행별 집계
# [1,2] → 1.5 / [3,4] → 3.5
print(tf.reduce_mean(ar, axis=1).numpy())   # [1.5 3.5]

print(tf.reduce_max(ar).numpy())            # 4.0  전체 최대값
```

> `axis=0` : 행을 없애는 방향 (세로로 합침)  
> `axis=1` : 열을 없애는 방향 (가로로 합침)

---

## 📌 tf.reshape() — 형태 변환

```python
t = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]])
# t.shape = (1, 4, 3) → 총 원소 수 = 1×4×3 = 12

print(tf.reshape(t, shape=[12]))    # (12,)  1차원으로 펼치기
print(tf.reshape(t, shape=[2, 6]))  # (2, 6) 명시적 지정
print(tf.reshape(t, shape=[-1, 6])) # (2, 6) 행 자동 계산: 12÷6=2
print(tf.reshape(t, shape=[2, -1])) # (2, 6) 열 자동 계산: 12÷2=6

# -1 : 나머지 차원을 자동 계산 (shape 인자 중 하나에만 사용 가능)
```

---

## 📌 tf.squeeze() — 크기 1인 차원 제거

```python
t  = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]])  # shape (1,4,3)
t2 = np.array([[[0], [3], [6], [9]]])                              # shape (1,4,1)

# 크기가 1인 차원을 자동으로 모두 제거
print(tf.squeeze(t))   # (1,4,3) → (4,3)  axis=0 제거
print(tf.squeeze(t2))  # (1,4,1) → (4,)   크기 1인 차원 모두 제거

# 특정 차원만 제거하고 싶을 경우 axis 명시
# tf.squeeze(t, axis=0)  → axis=0만 제거
```

|입력 shape|squeeze 결과|제거된 차원|
|---|---|---|
|`(1, 4, 3)`|`(4, 3)`|axis=0|
|`(1, 4, 1)`|`(4,)`|axis=0, axis=2|

---

## 📌 tf.expand_dims() — 차원 추가

기준 : `tarr.shape = (2, 3)`

```python
tarr = tf.constant([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)

# axis=0 : 맨 앞에 새 차원 추가 → (2,3) : (1,2,3)
sbs = tf.expand_dims(tarr, 0)
print(sbs.numpy())   # [[[1 2 3] [4 5 6]]]

# axis=1 : 각 행을 별도 묶음으로 → (2,3) : (2,1,3)
sbs = tf.expand_dims(tarr, 1)
print(sbs.numpy())   # [[[1 2 3]] [[4 5 6]]]

# axis=2 : 각 원소를 별도 묶음으로 → (2,3) : (2,3,1)
sbs = tf.expand_dims(tarr, 2)
print(sbs.numpy())   # [[[1][2][3]] [[4][5][6]]]

# axis=-1 : 마지막 위치에 새 차원 추가 → axis=2와 동일한 결과
# (2,3) → (2,3,1)
sbs = tf.expand_dims(tarr, -1)
print(sbs.numpy())   # [[[1][2][3]] [[4][5][6]]]
```

**squeeze ↔ expand_dims 관계**

```
squeeze     : 크기 1인 차원 제거
expand_dims : 크기 1인 차원 추가
→ 서로 역연산 관계
```

**실무 활용 (CV)**

|상황|코드|
|---|---|
|배치 차원 추가 `(H,W)` → `(1,H,W)`|`expand_dims(x, 0)`|
|채널 차원 추가 `(H,W)` → `(H,W,1)`|`expand_dims(x, -1)`|
|모델 출력 `(1,1)` → 스칼라|`squeeze(x)`|

---

## 📌 tf.cast() — 자료형 변환

```python
num  = tf.constant([1, 2, 3])    # dtype: int32
num2 = tf.cast(num, tf.float32)  # int32 → float32로 변환
print(num2, num2.dtype)
# tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32)
```

|dtype|설명|
|---|---|
|`tf.int32`|32비트 정수|
|`tf.float32`|32비트 실수 (딥러닝 기본값)|
|`tf.float64`|64비트 실수|
|`tf.bool`|불리언|

> 딥러닝 연산은 대부분 `float32` 기반이므로 정수 데이터 입력 시 cast 필수

---

## 💻 전체 실습 코드

```python
import tensorflow as tf
import numpy as np

x = tf.constant(7)
y = tf.constant(3)

# ─────────────────────────────────────────
# 조건 연산
# ─────────────────────────────────────────
# tf.cond(조건, 참일때 실행, 거짓일때 실행) — lambda로 감싸야 함
# x > y (7 > 3) → True → tf.add(x, y) = 10
result1 = tf.cond(x > y, lambda: tf.add(x, y), lambda: tf.subtract(x, y))
print(result1)  # tf.Tensor(10, shape=(), dtype=int32)

f1 = lambda: tf.constant(1)                  # 조건 True 시 반환값
f2 = lambda: tf.constant(tf.multiply(2, 3))  # default 반환값 (2×3=6)

# x < y (7 < 3) → False → default f2 실행 → 6
result2 = tf.case([(tf.less(x, y), f1)], default=f2)
print(result2)  # tf.Tensor(6, shape=(), dtype=int32)

# ─────────────────────────────────────────
# 관계 연산 — bool Tensor 반환
# ─────────────────────────────────────────
print(tf.equal(1, 2))           # 1 == 2  → False
print(tf.not_equal(1, 2))       # 1 != 2  → True
print(tf.less(1, 2))            # 1 < 2   → True
print(tf.greater(1, 2))         # 1 > 2   → False
print(tf.greater_equal(1, 2))   # 1 >= 2  → False

# ─────────────────────────────────────────
# 논리 연산
# ─────────────────────────────────────────
print(tf.logical_and(True, False))  # AND → False
print(tf.logical_or(True, False))   # OR  → True
print(tf.logical_not(True))         # NOT → False

# ─────────────────────────────────────────
# unique() — 중복 제거 + 역인덱스
# ─────────────────────────────────────────
kbs = tf.constant([1, 2, 2, 3, 2])
val, idx = tf.unique(kbs)
# val : 고유값 [1, 2, 3]
# idx : 역인덱스 [0, 1, 1, 2, 1] → tf.gather(val, idx)로 원본 복원 가능
print('val : ', val)
print('idx : ', idx)

# ─────────────────────────────────────────
# reduce_* — 집계 + 차원 축소
# ─────────────────────────────────────────
ar = [[1., 2.], [3., 4.]]

print(tf.reduce_mean(ar).numpy())           # 2.5   전체 평균
print(tf.reduce_mean(ar, axis=0).numpy())   # [2. 3.]   열별 평균
print(tf.reduce_mean(ar, axis=1).numpy())   # [1.5 3.5] 행별 평균
print(tf.reduce_max(ar).numpy())            # 4.0   전체 최대값

# ─────────────────────────────────────────
# reshape — 형태 변환
# ─────────────────────────────────────────
t = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]])
# t.shape = (1, 4, 3) → 총 원소 수 = 12

print(tf.reshape(t, shape=[12]))    # (12,)  1차원으로 펼치기
print(tf.reshape(t, shape=[2, 6]))  # (2, 6) 명시적 지정
print(tf.reshape(t, shape=[-1, 6])) # (2, 6) 행 자동 계산: 12÷6=2
print(tf.reshape(t, shape=[2, -1])) # (2, 6) 열 자동 계산: 12÷2=6

# ─────────────────────────────────────────
# squeeze — 크기 1인 차원 제거
# ─────────────────────────────────────────
print(tf.squeeze(t))    # (1,4,3) → (4,3)

t2 = np.array([[[0], [3], [6], [9]]])  # shape (1,4,1)
print(t2.shape)         # (1, 4, 1)
print(tf.squeeze(t2))   # (1,4,1) → (4,) 크기 1인 차원 모두 제거

# ─────────────────────────────────────────
# expand_dims — 차원 추가
# ─────────────────────────────────────────
tarr = tf.constant([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)

# axis=0 : 맨 앞에 차원 추가 → (1,2,3)
print(tf.expand_dims(tarr, 0).numpy())

# axis=1 : 중간에 차원 추가 → (2,1,3)
print(tf.expand_dims(tarr, 1).numpy())

# axis=2 : 끝에 차원 추가 → (2,3,1)
print(tf.expand_dims(tarr, 2).numpy())

# axis=-1 : 마지막 위치 추가 → axis=2와 동일, (2,3,1)
print(tf.expand_dims(tarr, -1).numpy())

# ─────────────────────────────────────────
# cast — 자료형 변환
# ─────────────────────────────────────────
num  = tf.constant([1, 2, 3])    # dtype: int32
num2 = tf.cast(num, tf.float32)  # int32 → float32
print(num2, num2.dtype)
# tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32)
```

---

## 🔑 핵심 포인트

> `tf.cond` — 삼항 연산자, 반드시 **lambda로 감싸야** 함  
> `tf.case` — 다중 조건, `[(조건, 함수)]` 리스트 순서대로 평가  
> `tf.unique` — 고유값(`val`) + 역인덱스(`idx`) 반환, `tf.gather`로 원본 복원  
> `reduce_*` — `axis=0` 열별 집계 / `axis=1` 행별 집계  
> `reshape` — `-1` 하나만 사용 가능, 나머지 차원 자동 계산  
> `squeeze` — 크기 1인 차원 **모두** 제거, 특정 차원만 제거 시 `axis` 명시  
> `expand_dims` — `axis=-1` 은 마지막 위치 추가, CV에서 채널 차원 추가 시 자주 사용  
> `tf.cast` — 딥러닝 연산은 `float32` 기반, 정수 입력 시 cast 필수

---
# Keras 기본 개념 및 모델링 순서

---

# 📄 Keras — 개념 · 모델링 7단계 · 손실함수 · 옵티마이저 · 정규화

---

## 📌 Keras란?

**Keras** 는 **TensorFlow 위에서 동작하는 고수준 딥러닝 API**다.

- 직관적이고 쉬운 API 제공 — `model.fit()` 한 줄로 학습 루프 전체 처리
- backend로 TensorFlow, Theano, CNTK 등을 지원
- 케라스의 가장 핵심적인 데이터 구조는 **"모델"**
- `Sequential` 에 `Dense` 레이어를 쌓는 **스택 구조**가 기본
- 복잡한 모델은 Functional API 또는 Model Subclassing으로 구현

> Keras = TensorFlow를 쉽게 쓰게 해주는 포장지

---

## 📌 신경망 설계 3가지 방법

|방법|특징|사용 시기|
|---|---|---|
|**Sequential API**|레이어를 순서대로 쌓는 구조|단순한 선형 구조 모델|
|**Functional API**|다중 입출력, 브랜치 구조 가능|복잡한 모델 (ResNet 등)|
|**Model Subclassing**|`tf.keras.Model` 상속, 완전 커스텀|연구용, 세밀한 제어 필요 시|

---

## 📌 Dense() 주요 인자

|인자|설명|
|---|---|
|첫번째 인자|출력 뉴런의 수|
|`input_dim`|입력 뉴런의 수 (첫 번째 레이어에만 지정)|
|`activation`|활성화 함수 종류|

### 활성화 함수 선택 기준

|활성화 함수|사용 위치|특징|
|---|---|---|
|`relu`|**은닉층**|음수 → 0, 양수 → 그대로 출력|
|`sigmoid`|**이진분류** 출력층|0~1 확률 출력|
|`softmax`|**다중분류** 출력층|클래스별 확률 출력, 합=1 보장|
|`linear`|**회귀** 출력층|변환 없이 그대로 출력 (기본값)|

---

## 📌 Keras 핵심 레이어 종류

|종류|레이어|
|---|---|
|**Fully Connected**|`Dense`|
|**Convolutional**|`Conv1D`, `Conv2D`, `Conv3D`, `Conv2DTranspose`|
|**Pooling**|`MaxPooling1D/2D/3D`, `AveragePooling1D`|
|**RNN**|`GRU`, `LSTM`, `ConvLSTM2D`|
|**기타**|`BatchNormalization`, `Dropout`, `Embedding`|

---

## 📌 케라스 모델링 7단계

### 전체 흐름

```
데이터 수집/전처리 → 모델 구성 → 컴파일 → 학습 → 학습과정 확인 → 평가 → 예측
```

---

### STEP 1 — 데이터 세트 생성

- 원본 데이터를 불러오거나 생성
- 훈련세트 / 검증세트 / 시험세트로 분리
- 딥러닝 모델이 처리할 수 있도록 **포맷 변환** (정규화, One-hot encoding 등)

|데이터 종류|역할|
|---|---|
|train|모델 학습에 사용|
|validation|학습 중 과적합 모니터링|
|test|학습 완료 후 최종 성능 평가|

> 수능 비유 — train=모의고사, test=작년 수능, real=올해 수능

---

### STEP 2 — 모델 구성

- `Sequential` 모델 생성 후 필요한 레이어를 순서대로 추가
- 복잡한 구조가 필요할 경우 Functional API 사용
- `model.summary()` 로 모델 구조 확인 가능
- `plot_model()` 로 모델 시각화 가능 (`pip install pydot` 필요)

---

### STEP 3 — 모델 컴파일

- `compile()` 함수로 학습 방법을 설정
- 학습 전 반드시 수행해야 함

|인자|설명|
|---|---|
|`optimizer`|가중치 업데이트 방법 (`'adam'`, `'sgd'` 등)|
|`loss`|손실 함수 — 모델 성능을 측정하는 기준|
|`metrics`|학습 모니터링 지표 (`'accuracy'` 등)|

---

### STEP 4 — 모델 학습

- `fit()` 함수로 훈련 데이터를 이용해 모델 학습
- 반환된 `history` 객체에 epoch별 loss / accuracy가 기록됨

|인자|설명|
|---|---|
|첫번째 인자|훈련 데이터 (X)|
|두번째 인자|정답 레이블 (y)|
|`epochs`|전체 데이터 반복 횟수|
|`batch_size`|한 번에 학습할 데이터 수 (기본값 32)|
|`verbose`|출력 방식 (0=무출력, 1=진행바, 2=한줄)|

---

### STEP 5 — 학습 과정 확인

- epoch별 loss / accuracy 추이를 시각화해서 학습 상태 판단
- loss가 수렴 → 학습이 잘 된 것
- validation loss가 다시 올라감 → 과적합 신호

---

### STEP 6 — 모델 평가

- `evaluate()` 함수로 시험세트를 이용해 최종 성능 측정
- 학습에 사용하지 않은 데이터로 평가해야 의미 있음
- loss와 metrics(accuracy 등) 값 반환

---

### STEP 7 — 예측

- `predict()` 함수로 새로운 입력 데이터에 대한 출력 생성
- softmax 출력 → 각 클래스에 속할 확률값
- `np.argmax()` 로 확률이 가장 높은 클래스를 최종 예측값으로 사용

---

## 📌 Epoch · Batch size · Iteration

|용어|의미|
|---|---|
|**Epoch**|전체 데이터를 한 번 다 학습한 것 (순전파 + 역전파 완료)|
|**Batch size**|한 번에 학습하는 데이터 수 — 가중치 업데이트 단위|
|**Iteration**|1 epoch를 끝내기 위해 필요한 배치 수|

### 수치 예시

```
전체 데이터 = 2,000개 / batch_size = 200

Iteration = 2,000 / 200 = 10회/epoch  ← 가중치 업데이트 횟수
Epoch = 50
총 업데이트 = 10 × 50 = 500회
```

### 경사 하강법 종류 비교

|방법|batch_size|특징|
|---|---|---|
|**Batch GD**|전체 데이터|안정적, 메모리 많이 필요, 느림|
|**SGD**|1|빠름, 불안정, 노이즈 많음|
|**Mini-Batch GD**|32~256|속도와 안정성의 균형 **(현업 기본)**|

---

## 📌 손실 함수 (Loss Function)

**손실 함수** 는 **예측값과 실제값의 차이(오차)를 수치화하는 함수**다.

- 오차가 클수록 손실 함수 값이 커짐
- 학습 = 손실 함수 값을 최소화하는 가중치 W와 편향 b를 찾는 과정
- 손실 함수 선택이 학습 방향을 결정하므로 매우 중요

### 손실 함수 선택 기준

|문제 유형|손실 함수|비고|
|---|---|---|
|회귀|`mse` (Mean Squared Error)|오차 제곱 평균|
|이진분류|`binary_crossentropy`|출력 뉴런 1개, sigmoid|
|다중분류|`categorical_crossentropy`|출력 뉴런 N개, softmax|

### MSE vs Cross-Entropy

- **MSE** : 오차를 제곱한 평균 → 연속형 변수 예측에 사용
- **Cross-Entropy** : 낮은 확률로 맞추거나 높은 확률로 틀릴수록 loss가 더 커짐 → 분류 문제에 적합

---

## 📌 옵티마이저 (Optimizer)

**옵티마이저** 는 **손실 함수 값을 줄이는 방향으로 가중치를 업데이트하는 알고리즘**이다.

### 옵티마이저 계보

```
SGD
 └→ Momentum  : SGD + 관성 추가 → 로컬 미니멈 탈출 효과
     └→ Adagrad  : 매개변수마다 다른 학습률 적용
         └→ RMSprop : Adagrad의 학습률 급감 문제 개선
             └→ Adam : Momentum + RMSprop 결합 (방향 + 학습률 모두 처리)
```

### 옵티마이저 비교

|옵티마이저|특징|추천 상황|
|---|---|---|
|`SGD`|단순, 느림|기본 이해용|
|`Momentum`|관성으로 로컬 미니멈 탈출|SGD 개선 필요 시|
|`Adagrad`|파라미터별 개별 학습률|희소 데이터|
|`RMSprop`|Adagrad 학습률 급감 보완|RNN 계열|
|`Adam`|Momentum + RMSprop 결합|**현업 기본값**|

---

## 📌 과적합 & 정규화 (Regularization)

### 개념 정리

|용어|의미|
|---|---|
|**최적화**|훈련 데이터에서 최고 성능을 얻도록 모델 조정|
|**일반화**|새로운 데이터에서도 잘 동작하는 정도|
|**과소적합**|모델이 너무 단순 → 훈련 데이터조차 제대로 학습 못함|
|**과대적합**|훈련 데이터에 너무 특화 → 새 데이터에서 성능 저하|

### 과적합 방지 방법

**1) Early Stopping — 조기 종료**

- 검증 손실(val_loss)이 더 이상 개선되지 않으면 학습을 자동으로 중단
- 불필요한 학습으로 인한 과적합을 방지

**2) Dropout — 드롭아웃**

- 학습 시 지정한 비율의 뉴런을 무작위로 비활성화
- 특정 뉴런에 의존하지 않도록 강제 → 일반화 성능 향상
- 비율은 보통 **0.2 ~ 0.5** 사용
- 테스트 시에는 드롭아웃 적용 안함 (Keras가 자동 처리)

**3) L1 / L2 Regularization — 가중치 규제**

- 큰 가중치에 패널티를 부여해 가중치가 작은 값을 가지도록 강제
- 오캄의 면도날 원칙 : **간단한 모델이 복잡한 모델보다 과적합에 강함**

|규제 방법|공식|특징|
|---|---|---|
|**L1 (Lasso)**|가중치 절댓값 합|일부 가중치를 0으로 만듦 (희소성)|
|**L2 (Ridge / weight decay)**|가중치 제곱합|가중치를 전체적으로 작게 만듦|

---

## 📌 Keras 제공 샘플 데이터셋

|데이터셋|설명|
|---|---|
|`CIFAR10`|10종류 카테고리, 32×32 컬러 이미지 50,000장|
|`CIFAR100`|100종류 카테고리, 32×32 컬러 이미지 50,000장|
|`MNIST`|0~9 숫자, 28×28 흑백 이미지 60,000장|
|`Fashion MNIST`|10종류 의류, 28×28 흑백 이미지 60,000장|
|`IMDB`|영화 리뷰 감성분석 (positive/negative) 25,000개|
|`Boston Housing`|보스턴 주택 가격 회귀 데이터|

---

## 🔑 핵심 포인트

> Keras = TF 고수준 API — `Sequential` + `Dense` 스택 구조가 기본  
> **모델링 7단계** : 데이터 → 모델 구성 → 컴파일 → 학습 → 확인 → 평가 → 예측  
> **활성화 함수** : 은닉층→relu / 이진분류→sigmoid / 다중분류→softmax / 회귀→linear  
> **손실함수** : 회귀→MSE / 이진분류→binary_crossentropy / 다중분류→categorical_crossentropy  
> **Epoch** = 전체 데이터 1회 학습 / **Iteration** = 1 epoch 내 가중치 업데이트 횟수  
> **옵티마이저** : 현업 기본값은 `Adam` — Momentum + RMSprop 결합  
> **과적합 방지** : Early Stopping / Dropout(0.2~0.5) / L1·L2 Regularization

---
# 📄 tf4keras.py — Sequential 모델 · 순전파 · 역전파

---

## 📌 개념 정리

### 다층 신경망(MLP)이란?

**다층 신경망(Multi-Layer Perceptron)** 은 **입력층 · 은닉층 · 출력층으로 구성된 신경망**이다.

- 입력층 : 데이터를 받아들이는 층
- 은닉층 : 데이터의 패턴을 학습하는 층 (0개 이상)
- 출력층 : 최종 예측값을 출력하는 층

> 은닉층이 없으면 선형 분리 가능한 문제만 풀 수 있음  
> 은닉층이 있어야 XOR처럼 복잡한 문제 해결 가능

---

### 순전파 vs 역전파

**순전파 (Forward Propagation)**

- 입력 데이터가 입력층 → 은닉층 → 출력층 방향으로 흘러 예측값을 만드는 과정
- `입력 x → 은닉층 → 출력 y_pred → 손실 loss 계산`

**역전파 (Backpropagation)**

- 예측이 틀렸을 때 오차를 거꾸로 거슬러 올라가며 각 가중치의 책임 정도를 계산하고 수정하는 과정
- `손실 loss → 출력층 → 은닉층 → 입력층 방향`
- epochs=2 이상이면 역전파 알고리즘이 적용됨

```
예측이 틀림
  → 손실값(loss) 계산
  → 어느 가중치가 얼마나 책임이 있는지 계산 (미분)
  → 그 가중치를 조금씩 수정 (옵티마이저)
  → 반복
```

> 한마디로 역전파는 손실값을 기준으로 각 가중치의 책임 정도를 계산하고,  
> 그 결과를 이용해 모델의 가중치를 수정하는 **딥러닝 학습의 핵심 알고리즘**

---

### OR 게이트란?

|x1|x2|OR 출력|
|---|---|---|
|0|0|0|
|0|1|1|
|1|0|1|
|1|1|1|

- 둘 중 하나라도 1이면 1을 출력
- **선형 분리 가능** → 은닉층 없이 뉴런 1개로도 학습 가능

---

### 모델 구조

```
입력층 (2개 특성)
  ↓
Dense(1) + sigmoid   ← 출력층 (이진분류)
  ↓
예측값 (0 or 1)
```

> OR 게이트는 단순한 문제라 은닉층 없이 뉴런 1개 구조로 충분  
> 사실상 로지스틱 회귀와 동일한 구조

---

### sigmoid 출력 해석

- sigmoid는 0~1 사이의 확률값을 출력
- **0.5 기준**으로 잘라서 0/1 분류
- 다중분류에서는 이 역할을 `np.argmax()` 가 대신함

```
proba > 0.5  →  1 (양성)
proba ≤ 0.5  →  0 (음성)
```

---

### 모델 저장 / 로드

- 학습된 모델을 `.keras` 파일로 저장
- 나중에 불러와서 재학습 없이 바로 예측에 사용
- 실제 서비스 배포 시 필수 패턴

---

## 💻 전체 실습 코드

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

# ─────────────────────────────────────────
# 1) 데이터 수집 및 가공
# ─────────────────────────────────────────
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[1]])  # OR 게이트 : 둘 중 하나라도 1이면 1
# y = np.array([0,1,1,1])       # 1차원으로 써도 가능 (sklearn보다 유연한 구조)

# ─────────────────────────────────────────
# 2) 모델 설정
# ─────────────────────────────────────────
# Sequential : 레이어를 순서대로 쌓는 구조
model = Sequential([
    Input(shape=(2,)),       # 입력층 : 특성 2개 (x1, x2)
    Dense(units=1),          # 출력층 : 뉴런 1개 (이진분류)
    Activation('sigmoid')    # 활성화 함수 : 출력을 0~1 확률로 변환
])

# add() 방식으로도 동일하게 구성 가능
# model = Sequential()
# model.add(Input(shape=(2,)))
# model.add(Dense(units=1))
# model.add(Activation('sigmoid'))

# ─────────────────────────────────────────
# 3) 모델 컴파일 (학습 방법 설정)
# ─────────────────────────────────────────
# loss     : 손실함수 — 예측값과 실제값의 차이를 수치화
# optimizer: 손실값을 줄이는 방향으로 가중치를 업데이트하는 알고리즘
# metrics  : 학습 과정을 모니터링할 지표

# 옵티마이저 종류별 비교 (주석 해제해서 교체 가능)
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.1), metrics=['accuracy'])

# OR 게이트는 이진분류 → loss: binary_crossentropy
# Adam : Momentum + RMSprop 결합, 현업 기본값 옵티마이저
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])

# ─────────────────────────────────────────
# 4) 모델 학습 — 순전파 + 역전파 반복
# ─────────────────────────────────────────
# epochs=2 이상이면 역전파 알고리즘 적용
# 순전파 → 손실 계산 → 역전파 → 가중치 업데이트 반복
# batch_size=1 : 데이터 1개씩 보고 가중치 업데이트 (SGD 방식)
model.fit(x=x, y=y, epochs=30, batch_size=1, verbose=2)

# ─────────────────────────────────────────
# 5) 모델 평가
# ─────────────────────────────────────────
# evaluate() : [loss, accuracy] 반환
loss_metrics = model.evaluate(x=x, y=y)
print('loss_metrics : ', loss_metrics)

# ─────────────────────────────────────────
# 6) 예측
# ─────────────────────────────────────────
pred = model.predict(x=x)
print('예측 확률 : ', pred)
# sigmoid 출력 = 각 입력에 대한 1일 확률
# ex) 0.82 → 1로 분류 / 0.34 → 0으로 분류

proba = model.predict(x=x, verbose=0)
pred = (proba > 0.5).astype('int32')  # 0.5 기준으로 0/1 분류
print('예측 값 : ', pred.ravel())     # ravel() : 1차원으로 펼치기
print('실제 값 : ', y.ravel())

# ─────────────────────────────────────────
# 7) 모델 저장
# ─────────────────────────────────────────
# 학습된 가중치와 구조를 .keras 파일로 저장
# 재학습 없이 언제든 불러와서 예측 가능
model.save('tf4model.keras')

# ─────────────────────────────────────────
# 8) 모델 로드 및 예측
# ─────────────────────────────────────────
from tensorflow.keras.models import load_model

mymodel = load_model('tf4model.keras')      # 저장된 모델 불러오기
new_proba = mymodel.predict(x=x, verbose=0)
new_pred = (new_proba > 0.5).astype('int32')
print('예측 값 : ', new_pred.ravel())       # 저장 전과 동일한 결과
```

---

## 📌 순전파 vs 역전파 흐름 요약

```
─────────────────────────────────────────────────────
순전파 (Forward)
  입력 x → 가중치 연산 → 활성화 함수 → 예측값 y_pred → 손실 loss 계산
─────────────────────────────────────────────────────
역전파 (Backward)
  손실 loss → 출력층 ← 은닉층 ← 입력층 방향으로 미분
  → 각 가중치가 손실에 얼마나 기여했는지 계산
  → 옵티마이저가 가중치를 조금씩 수정
─────────────────────────────────────────────────────
이 과정을 epochs 횟수만큼 반복 → 점점 오차가 줄어듦
─────────────────────────────────────────────────────
```

---

## 📌 OR vs XOR 비교

|구분|OR 게이트|XOR 게이트|
|---|---|---|
|선형 분리|✅ 가능|❌ 불가능|
|필요 구조|뉴런 1개 (은닉층 없음)|은닉층 필요|
|모델 복잡도|단순|다층 신경망 필요|

> XOR이 선형 분리 불가능한 이유가 1980년대 **퍼셉트론의 한계**로 알려진 문제  
> 이를 해결하기 위해 **다층 신경망(MLP) + 역전파** 가 등장

---

## 🔑 핵심 포인트

> **순전파** : 입력 → 출력 방향으로 예측값 계산  
> **역전파** : 손실 → 입력 방향으로 가중치 책임 계산 후 수정 — epochs=2 이상 적용  
> `sigmoid` 출력 → **0.5 기준**으로 0/1 분류 / 다중분류는 `np.argmax()`  
> `ravel()` — 다차원 배열을 1차원으로 펼치기  
> `model.save()` / `load_model()` — 학습된 모델 저장 · 로드, 배포 시 필수 패턴  
> OR 게이트 = 선형 분리 가능 → 은닉층 불필요 / XOR = 선형 분리 불가 → 은닉층 필요

---
# 📄 tf5xor.py — 다층 신경망 · 파라미터 수 · 학습 곡선

---

## 📌 개념 정리

### XOR 게이트란?

**XOR(Exclusive OR)** 은 **두 입력이 서로 다를 때만 1을 출력**하는 논리회로다.

|x1|x2|XOR 출력|
|---|---|---|
|0|0|0|
|0|1|1|
|1|0|1|
|1|1|0|

- **선형 분리 불가능** → 직선 하나로 0과 1을 나눌 수 없음
- 단층 퍼셉트론(뉴런 1개)으로는 학습 불가
- **은닉층이 있는 다층 신경망(MLP)** 이 반드시 필요

> 1980년대 퍼셉트론의 한계로 알려진 문제  
> 이를 해결하기 위해 다층 신경망 + 역전파 알고리즘이 등장

---

### OR vs XOR 비교

|구분|OR 게이트|XOR 게이트|
|---|---|---|
|선형 분리|✅ 가능|❌ 불가능|
|모델 구조|뉴런 1개 (은닉층 없음)|은닉층 필요|
|활성화 함수|sigmoid 1개|relu(은닉) + sigmoid(출력)|

---

### 이번 모델 구조

```
입력층  : 특성 2개 (x1, x2)
  ↓
은닉층1 : Dense(5) + relu   ← 비선형 패턴 학습
  ↓
은닉층2 : Dense(5) + relu   ← 더 복잡한 패턴 학습
  ↓
출력층  : Dense(1) + sigmoid ← 0~1 확률 출력
```

> relu를 은닉층에 쓰는 이유 : 비선형성을 추가해서 XOR처럼 복잡한 경계를 학습할 수 있게 함  
> sigmoid를 출력층에 쓰는 이유 : 이진분류 확률값(0~1) 출력

---

### 파라미터 수 계산

**공식 : (입력 수 + 1) × 출력 수**  
`+1` 은 각 뉴런마다 붙는 **bias(편향)** 항

|레이어|계산|파라미터 수|
|---|---|---|
|dense (입력2 → 출력5)|(2+1) × 5|**15**|
|dense_1 (입력5 → 출력5)|(5+1) × 5|**30**|
|dense_2 (입력5 → 출력1)|(5+1) × 1|**6**|
|**합계**||**51**|

---

### history 객체

- `model.fit()` 의 반환값
- epoch별 loss / accuracy 기록이 딕셔너리 형태로 저장됨

```
history.history['loss']      → epoch별 loss 리스트
history.history['accuracy']  → epoch별 accuracy 리스트
```

---

### 학습 곡선 (Learning Curve)

- epoch에 따른 loss / accuracy 변화를 시각화
- 학습이 잘 됐는지 판단하는 기준

|상태|loss 곡선|accuracy 곡선|
|---|---|---|
|정상 학습|점점 감소 후 수렴|점점 증가 후 수렴|
|과적합|train loss 감소, val loss 증가|train만 높아짐|
|학습 부족|loss가 높은 채로 유지|accuracy가 낮은 채로 유지|

---

## 💻 전체 실습 코드

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

# ─────────────────────────────────────────
# 1) 데이터 수집 및 가공
# ─────────────────────────────────────────
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])  # XOR 게이트 : 두 입력이 다를 때만 1

# ─────────────────────────────────────────
# 2) 모델 설정
# ─────────────────────────────────────────
model = Sequential()
model.add(Input(shape=(2,)))                        # 입력층 : 특성 2개

# XOR은 선형 분리 불가 → 은닉층 필요
# relu : 비선형성 추가 → 복잡한 결정 경계 학습 가능
model.add(Dense(units=5, activation='relu'))        # 은닉층1 : 뉴런 5개
model.add(Dense(units=5, activation='relu'))        # 은닉층2 : 뉴런 5개
model.add(Dense(units=1, activation='sigmoid'))     # 출력층 : 이진분류 확률 출력

# 모델 구조 및 파라미터 수 확인
# 파라미터 수 공식 : (입력수 + 1) × 출력수  (+1은 bias)
# dense   : (2+1)×5 = 15
# dense_1 : (5+1)×5 = 30
# dense_2 : (5+1)×1 = 6  → 총 51개
print(model.summary())

# ─────────────────────────────────────────
# 3) 컴파일
# ─────────────────────────────────────────
# XOR도 이진분류 → binary_crossentropy
# OR 게이트보다 복잡한 문제라 epochs를 늘려야 학습이 수렴함
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.01),  # 학습률 0.01
    metrics=['accuracy']
)

# ─────────────────────────────────────────
# 4) 학습
# ─────────────────────────────────────────
# XOR은 OR보다 복잡 → epochs를 200으로 늘림 (OR은 30이었음)
history = model.fit(x=x, y=y, epochs=200, batch_size=1, verbose=1)

# ─────────────────────────────────────────
# 5) 평가
# ─────────────────────────────────────────
loss_metrics = model.evaluate(x=x, y=y)
print('loss_metrics : ', loss_metrics)  # [loss, accuracy]

# history.history : epoch별 loss/accuracy가 딕셔너리로 저장
print(history.history)

# ─────────────────────────────────────────
# 6) 예측
# ─────────────────────────────────────────
proba = model.predict(x=x, verbose=0)
# sigmoid 출력값 → 0.5 기준으로 0/1 분류
pred = (proba > 0.5).astype('int32')
print('pred : ', pred)

# ─────────────────────────────────────────
# 7) 학습 곡선 시각화
# ─────────────────────────────────────────
import matplotlib.pyplot as plt

# loss가 내려가고 accuracy가 올라가면 학습이 잘 된 것
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.ylabel('loss / accuracy')
plt.legend(loc='best')  # 범례를 가장 적절한 위치에 자동 배치
plt.show()
```

---

## 📌 OR → XOR 변경 포인트 비교

|항목|OR (tf4)|XOR (tf5)|
|---|---|---|
|모델 구조|Dense(1) + sigmoid|Dense(5,relu) × 2 + Dense(1,sigmoid)|
|파라미터 수|3개|51개|
|epochs|30|200|
|학습 난이도|쉬움 (선형 분리 가능)|어려움 (비선형 경계 필요)|

---

## 🔑 핵심 포인트

> **XOR** = 선형 분리 불가능 → **은닉층 + relu** 로 비선형 경계 학습  
> **파라미터 수** = (입력수 + 1) × 출력수 — `+1`은 bias  
> **relu** 은닉층 / **sigmoid** 출력층 — 이진분류 기본 조합  
> `history.history` — epoch별 loss/accuracy 딕셔너리, 학습 곡선 그릴 때 사용  
> **학습 곡선** — loss 감소 + accuracy 증가 → 정상 학습 / val_loss 증가 → 과적합 신호

