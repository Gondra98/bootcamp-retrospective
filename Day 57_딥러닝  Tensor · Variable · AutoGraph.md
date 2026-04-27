# Day 57_딥러닝 : Tensor · Variable · AutoGraph
## 📅 2026-04-27

---

# TensorFlow · PyTorch 개념 정리

---

# 📄 딥러닝 프레임워크 개요

---

## 📌 개념 정리

### 딥러닝 프레임워크란?

**딥러닝 프레임워크** 는 **신경망 모델을 설계·학습·배포하기 위한 라이브러리**다.

- 행렬 연산, 자동 미분, GPU 연산 등을 추상화하여 제공
- 대표 프레임워크 : **TensorFlow** (Google), **PyTorch** (Meta)

---

## 📌 TensorFlow

### TensorFlow란?

**TensorFlow** 는 **Google이 개발한 오픈소스 딥러닝 프레임워크**다.

- 2015년 공개, 2019년 TF 2.0에서 Keras 공식 통합
- 배포·프로덕션 환경에 강점 (TF Serving, TF Lite, TF.js)
- 기본 실행 방식 : **Eager Execution** (즉시 실행)
- `@tf.function` 으로 **Graph Execution** 전환 가능

---

### 핵심 구성 요소

|구성 요소|설명|
|---|---|
|`tf.Tensor`|텐서플로의 기본 데이터 자료구조 — 다차원 배열|
|`tf.constant`|값 변경 불가 텐서 — 입력 데이터에 사용|
|`tf.Variable`|값 변경 가능 텐서 — 가중치(weight), bias에 사용|
|`tf.function`|AutoGraph — 파이썬 코드를 그래프 연산으로 자동 변환|
|`Keras`|TF의 고수준 API — 모델 설계를 쉽게 해주는 포장지|

---

### 실행 방식 2가지

```python
# 1) Eager Execution (기본) — 즉시 실행
a = tf.constant(5)      # 바로 계산됨

# 2) Graph Execution — @tf.function 데코레이터 사용
@tf.function
def calcFunc(a, b):     # 최적화된 그래프로 변환 후 실행
    if a < b:
        return tf.add(10, a)
    else:
        return tf.square(a)
```

|구분|Eager Execution|Graph Execution|
|---|---|---|
|실행 시점|즉시|그래프 빌드 후|
|디버깅|쉬움|어려움|
|성능|보통|빠름 (최적화)|
|사용법|기본값|`@tf.function`|

---

### tf.constant vs tf.Variable

```python
# tf.constant — 값 변경 불가 (입력 데이터)
a = tf.constant([1, 2])
# a[0] = 10  → 에러!

# tf.Variable — 값 변경 가능 (가중치, bias)
v = tf.Variable(1.0)
v.assign(123)           # 값 변경
v.assign_add(10)        # 더하기 후 치환
v.assign_sub(5)         # 빼기 후 치환
```

| 구분    | `tf.constant` | `tf.Variable`   |
| ----- | ------------- | --------------- |
| 값 변경  | ❌ 불가          | ✅ 가능 (`assign`) |
| 용도    | 입력 데이터        | 가중치, bias       |
| 자동 미분 | ❌             | ✅               |

---

### Keras — TF 고수준 API

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# 모델 정의
model = Sequential([
    Dense(units=1, input_shape=(2,)),
    Activation('sigmoid')
])

# 컴파일 + 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=10, batch_size=1)

# 평가 + 예측
model.evaluate(x, y)
model.predict(x)

# 저장 / 로드
model.save('model.keras')
model2 = load_model('model.keras')
```

> Keras = TensorFlow를 쉽게 쓰게 해주는 포장지 `model.fit()` 한 줄로 학습 루프 전체를 처리

---

### 형변환

```python
# ndarray → Tensor
tf.convert_to_tensor(np.array([1, 2]))

# Tensor → ndarray
tf.constant([1, 2]).numpy()

# 자동 형변환
tf.add(np.array([1, 2]), 5)     # → Tensor 타입으로 자동 변환
np.add(tf.constant([1, 2]), 2)  # → ndarray 타입으로 자동 변환
```

---

## 📌 PyTorch

### PyTorch란?

**PyTorch** 는 **Meta(Facebook)가 개발한 오픈소스 딥러닝 프레임워크**다.

- 2017년 공개, 연구·실험 환경에 강점
- **동적 연산 그래프(Dynamic Computation Graph)** — 실행하면서 그래프 생성
- Pythonic한 문법 — 디버깅이 직관적
- 최신 논문 대부분이 PyTorch 기반으로 코드 공개

---

### 핵심 구성 요소

|구성 요소|설명|
|---|---|
|`torch.Tensor`|PyTorch의 기본 데이터 자료구조|
|`nn.Module`|모델 정의의 기본 클래스 — 상속해서 사용|
|`nn.Linear`|선형 변환 레이어 (fully connected)|
|`torch.optim`|옵티마이저 모음 (Adam, SGD 등)|
|`torch.no_grad()`|gradient 계산 비활성화 — 평가/추론 시 사용|

---

### 모델 정의 방식

```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)   # 입력 2, 출력 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):               # 순전파 정의
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

model = SimpleModel()
```

---

### 학습 루프 구조

```python
criterion = nn.BCELoss()                        # 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 옵티마이저

for epoch in range(10):
    for xi, yi in zip(x_tensor, y_tensor):
        optimizer.zero_grad()   # ① gradient 초기화
        output = model(xi)      # ② forward (예측)
        loss = criterion(output, yi)
        loss.backward()         # ③ gradient 계산 (역전파)
        optimizer.step()        # ④ 가중치 업데이트
```

> PyTorch는 학습 루프를 **직접 구현** — 세밀한 제어 가능

---

### 평가 및 예측

```python
# 평가 — gradient 계산 끄기 (메모리 절약, 속도 향상)
with torch.no_grad():
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    pred = (outputs > 0.5).int()
    accuracy = (pred == y_tensor.int()).float().mean()

# Tensor → numpy 변환
proba = model(x_tensor).detach().numpy()
```

---

### 모델 저장 / 로드

```python
# 저장 — state_dict : 학습된 파라미터(가중치 + bias) 딕셔너리
torch.save(model.state_dict(), 'model.pt')

# 로드
model2 = SimpleModel()
model2.load_state_dict(torch.load('model.pt'))
model2.eval()   # 평가 모드 전환
```

---

## 📌 TensorFlow vs PyTorch 비교

### 선택 기준

|상황|선택|
|---|---|
|연구, 논문 구현, 빠른 실험|**PyTorch**|
|모바일 / 웹 배포|**TensorFlow** (TF Lite, TF.js)|
|기업 환경, 대규모 학습|**TensorFlow** (TPU 지원)|
|CNN, YOLO, GAN, Transformer|**PyTorch**|
|의료영상 AI (MONAI 기반)|**PyTorch**|

---

### 코드 구조 비교

|구분|TensorFlow (Keras)|PyTorch|
|---|---|---|
|모델 정의|`Sequential([...])`|`class` 상속 (`nn.Module`)|
|학습 루프|`model.fit()` 한 줄|`for` 루프 직접 구현|
|손실 함수|`model.compile(loss=...)`|`nn.BCELoss()` 등 직접 지정|
|평가|`model.evaluate()`|`torch.no_grad()` 블록|
|저장|`.keras` 파일|`.pt` 파일 (state_dict)|
|디버깅|상대적으로 어려움|직관적 (Pythonic)|
|연산 그래프|정적 (Static) → `@tf.function`|동적 (Dynamic) — 기본값|

---

### 손실 함수 비교

|손실 함수|TF/Keras|PyTorch|
|---|---|---|
|이진 분류|`binary_crossentropy`|`nn.BCELoss()`|
|다중 분류|`categorical_crossentropy`|`nn.CrossEntropyLoss()`|
|회귀|`mse`|`nn.MSELoss()`|

---

### 결론

```
연구 / 실험 / 논문 구현   → PyTorch
배포 / 모바일 / 기업 환경 → TensorFlow
둘 다 필요하다면?         → PyTorch로 연구 → TensorFlow로 배포 변환 가능
```

> **의료영상 AI 목표라면 PyTorch + MONAI 조합 추천** RSNA, NIH Chest X-ray 등 Kaggle 의료 대회 코드 대부분이 PyTorch 기반

---

## 🔑 핵심 포인트

> `tf.constant` — 값 변경 불가, 입력 데이터용 `tf.Variable` — 값 변경 가능, 가중치·bias용 (`assign`, `assign_add`, `assign_sub`) Keras = TF 고수준 API — `Sequential`, `model.fit()` 으로 간단하게 학습 PyTorch = `nn.Module` 상속 + 학습 루프 직접 구현 — 세밀한 제어 가능 `torch.no_grad()` — 평가 시 gradient 계산 비활성화 (메모리 절약) `state_dict()` — 모델 파라미터(가중치+bias)를 딕셔너리로 저장/로드 **현재 트렌드 : 연구는 PyTorch, 배포는 TensorFlow** — 의료 AI는 PyTorch 우세

---
# TensorFlow 기초 : Tensor · Variable · 난수

---

# 📄 tf1.py — Tensor · Variable · 난수

---

## 📌 개념 정리

### Tensor란?

**Tensor** 는 **텐서플로에서 데이터를 담는 기본 자료구조**다.

- `ndarray`와 유사하지만 텐서플로 연산에 최적화된 객체
- GPU 연산 가능, 자동 미분 지원 (`tf.Variable`)
- `tf.constant` : 값 변경 불가 / `tf.Variable` : 값 변경 가능

---

### Tensor 차원 (rank)

|표현|차원|이름|
|---|---|---|
|`tf.constant(12)`|0d|스칼라 (Scalar)|
|`tf.constant([12])`|1d|벡터 (Vector)|
|`tf.constant([[12]])`|2d|행렬 (Matrix)|
|`tf.constant([[[12]]])`|3d|텐서 (Tensor)|

```python
print(tf.constant(12))          # tf.Tensor(12, shape=(), dtype=int32)
print(tf.constant([12]))        # tf.Tensor([12], shape=(1,), dtype=int32)
print(tf.constant([[12]]))      # tf.Tensor([[12]], shape=(1, 1), dtype=int32)
print(tf.rank(tf.constant([[12, 1]])))  # tf.Tensor(2, shape=(), dtype=int32)
```

---

### print() vs tf.print()

|구분|`print()`|`tf.print()`|
|---|---|---|
|출력 대상|파이썬 객체 자체|텐서 실제 값|
|출력 예|`tf.Tensor(12, shape=(), dtype=int32)`|`12`|
|용도|정보 중심 출력|값 중심 출력|

---

### ndarray vs tf.constant vs tf.Variable

| 구분     | `np.ndarray` | `tf.constant` | `tf.Variable`   |
| ------ | ------------ | ------------- | --------------- |
| 값 변경   | ✅ 가능         | ❌ 불가          | ✅ 가능 (`assign`) |
| GPU 연산 | ❌            | ✅             | ✅               |
| 자동 미분  | ❌            | ❌             | ✅               |
| 주요 용도  | 일반 수치 연산     | 입력 데이터        | 가중치, bias       |

---

### Broadcast 연산

```python
a = tf.constant([1, 2])
b = tf.constant([3, 4])
c = a + b           # [4, 6] — 같은 shape끼리 요소별 연산

d = tf.constant([3])
e = c + d           # [7, 9] — shape이 달라도 자동으로 맞춰서 연산 (Broadcast)
```

> `[4, 6]` + `[3]` → `[3]`이 `[3, 3]`으로 자동 확장되어 연산

---

### 형변환

```python
# ndarray → Tensor
tf.convert_to_tensor(7)         # tf.Tensor(7, shape=(), dtype=int32)

# Tensor → ndarray (numpy)
tf.constant(7).numpy()          # 7

# 자동 형변환
tf.add(np.array([1, 2]), 5)     # → Tensor 타입으로 자동 변환
np.add(tf.constant([1, 2]), 2)  # → ndarray 타입으로 자동 변환
```

---

### tf.Variable — 변수 선언 및 사용

```python
v1 = tf.Variable(1.0)              # 스칼라 변수
v2 = tf.Variable(tf.ones((2,)))    # [1. 1.] — 1로 채워진 변수
v3 = tf.Variable(tf.zeros((2,)))   # [0. 0.] — 0으로 채워진 변수
```

**값 변경 방법**

```python
v1.assign(123)          # 값 치환
v2.assign([30, 40])     # 값 치환

aa = tf.Variable(tf.zeros((2, 1)))  # [[0][0]]
aa.assign(tf.ones((2, 1)))          # [[1][1]]

aa.assign_add([[2], [3]])   # 더하기 치환 → [[3][4]]
aa.assign_sub([[2], [3]])   # 빼기 치환  → [[1][1]]
aa.assign(aa * [[2], [3]])  # 곱하기 치환 → [[2][3]]  assign_mul() 없음
aa.assign(aa / [[2], [3]])  # 나누기 치환 → [[1][1]]  assign_div() 없음
```

|메서드|동작|
|---|---|
|`.assign(val)`|값 치환|
|`.assign_add(val)`|더하기 후 치환|
|`.assign_sub(val)`|빼기 후 치환|
|`.assign(v * val)`|곱하기 치환 (assign_mul 없음)|
|`.assign(v / val)`|나누기 치환 (assign_div 없음)|

> `v1 = 123` 은 에러 — Variable 자체를 파이썬 변수로 덮어쓰면 안 됨 반드시 `.assign()` 사용

---

### 난수 처리

```python
tf.random.uniform([1], 0, 1)               # 균등분포 — [shape], 최소, 최대
tf.random.normal([3], 0, 1)                # 정규분포 — [shape], 평균, 표준편차
tf.random.normal([3, 2], mean=0, stddev=1) # 정규분포 — 3행 2열
```

|함수|분포|파라미터|
|---|---|---|
|`tf.random.uniform`|균등분포|shape, minval, maxval|
|`tf.random.normal`|정규분포|shape, mean, stddev|

> 딥러닝에서 **가중치 초기화** 할 때 `tf.random.normal` 주로 사용

---

## 💻 전체 실습 코드

```python
import tensorflow as tf
import numpy as np

print(tf.__version__)                               # 2.21.0
print('즉시 실행 모드 : ', tf.executing_eagerly())   # True
print('GPU 사용 정보 확인 : ', tf.config.list_physical_devices('GPU'))  # []

# ─────────────────────────────────────────
# Tensor 차원
# ─────────────────────────────────────────
print(tf.constant(12))          # 0d 스칼라
print(tf.constant([12]))        # 1d 벡터
print(tf.constant([[12]]))      # 2d 행렬
print(tf.constant([[12, 1]]))   # shape=(1, 2)
print(tf.rank(tf.constant([[12, 1]])))  # tf.Tensor(2)

tf.print(tf.constant(12))       # 12 — 실제값 출력

# ─────────────────────────────────────────
# ndarray vs Tensor
# ─────────────────────────────────────────
imsi = np.array([1, 2])     # 값변경 가능
imsi[0] = 10

a = tf.constant([1, 2])     # 값변경 불가
b = tf.constant([3, 4])

c = a + b                   # [4 6] — 요소별 덧셈
d = tf.constant([3])
e = c + d                   # [7 9] — Broadcast

# ─────────────────────────────────────────
# 형변환
# ─────────────────────────────────────────
tf.convert_to_tensor(7)
tf.constant(7).numpy()

arr = np.array([1, 2])
tfarr = tf.add(arr, 5)      # → Tensor
np.add(tfarr, 2)            # → ndarray

# ─────────────────────────────────────────
# tf.Variable
# ─────────────────────────────────────────
v1 = tf.Variable(1.0)
v2 = tf.Variable(tf.ones((2,)))
v3 = tf.Variable(tf.zeros((2,)))

v1.assign(123)
v2.assign([30, 40])

aa = tf.Variable(tf.zeros((2, 1)))
aa.assign(tf.ones((2, 1)))
aa.assign_add([[2], [3]])   # [[3][4]]
aa.assign_sub([[2], [3]])   # [[1][1]]
aa.assign(aa * [[2], [3]])  # [[2][3]]
aa.assign(aa / [[2], [3]])  # [[1][1]]

# ─────────────────────────────────────────
# 난수
# ─────────────────────────────────────────
tf.random.uniform([1], 0, 1)
tf.random.normal([3], 0, 1)
tf.random.normal([3, 2], mean=0, stddev=1)
```

---

## 🔑 핵심 포인트

> `tf.constant` — 값 변경 불가, 입력 데이터용 `tf.Variable` — 값 변경 가능, 가중치·bias용 — 반드시 `.assign()` 으로 변경 `print()` 는 객체 정보 출력, `tf.print()` 는 텐서 실제값 출력 Broadcast — shape이 달라도 자동으로 맞춰서 연산 `tf.add(ndarray)` → Tensor / `np.add(tensor)` → ndarray 자동 형변환 `tf.random.normal` — 가중치 초기화에 주로 사용 (정규분포) `assign_mul()`, `assign_div()` 는 없음 — `assign(v * val)` 형태로 사용

---
# TensorFlow 기초 : tf.cond · AutoGraph · @tf.function

---

# 📄 tf2.py — tf.cond · AutoGraph · @tf.function

---

## 📌 개념 정리

### 실행 방식 2가지

|구분|Eager Execution|Graph Execution|
|---|---|---|
|실행 시점|즉시 실행|그래프 빌드 후 최적화 실행|
|디버깅|쉬움|어려움|
|성능|보통|빠름|
|사용법|기본값|`@tf.function`|

---

### tf.constant vs tf.Variable 출력 형태

```python
# tf.constant → Tensor 형태 출력
node1 = tf.constant(3, dtype=tf.float32)
print(node1)    # tf.Tensor(3.0, shape=(), dtype=float32)

# tf.Variable → Variable 형태 출력 (이름 + numpy 값 포함)
node3 = tf.Variable(3, dtype=tf.float32)
print(node3)    # <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=3.0>
```

> Variable은 이름(`Variable:0`)과 numpy 값이 함께 출력됨

---

### tf.cond — 텐서플로의 if문

```python
a = tf.constant(5)
b = tf.constant(10)

# ✅ 정상 — lambda로 감싸야 함
result = tf.cond(a < b, lambda: tf.add(10, a), lambda: tf.square(a))

# ❌ 에러 — 값을 바로 넣으면 조건 평가 전에 양쪽 모두 실행됨
result = tf.cond(a < b, tf.add(10, a), tf.square(a))
```

|구분|설명|
|---|---|
|형식|`tf.cond(조건, 참일때_함수, 거짓일때_함수)`|
|주의|반드시 **lambda로 감싸야** 함|
|이유|조건 평가 전에 양쪽이 실행되는 걸 방지|

---

### @tf.function — AutoGraph

**AutoGraph** 는 **파이썬 코드를 텐서플로 그래프 연산으로 자동 변환**하는 기능이다.

```python
@tf.function               # AutoGraph 개입 — 그래프 연산으로 자동 변환
def calcFunc1(a, b):
    if a < b:              # 파이썬 if → AutoGraph가 tf.cond로 변환
        return tf.add(10, a)
    else:
        return tf.square(a)
```

> `@tf.function` 안에서 `if, for, while, break, continue, return` 사용 시 AutoGraph가 개입

---

### @tf.function 주의사항

| 상황                       | 결과                               |
| ------------------------ | -------------------------------- |
| `if, for, while` 사용      | AutoGraph가 자동으로 그래프 변환           |
| 함수 내부에서 `tf.Variable` 선언 | ❌ 에러 — 구조가 고정적이어야 함              |
| 외부 변수 사용                 | `global` 키워드 필요                  |
| 문자열 포맷팅 (`format()`)     | ❌ 에러 — SymbolicTensor는 format 불가 |
| `tf.print` 인자 분리         | ✅ 정상                             |

---

### tf.print 포맷팅 주의

```python
@tf.function
def calcFunc4(dan):
    for i in range(1, 10):
        result = tf.multiply(dan, i)

        # ❌ 에러 — SymbolicTensor는 format() 불가
        tf.print('{}*{}={}'.format(dan, i, result))

        # ✅ 정상 — 인자로 분리해서 넘김
        tf.print(dan, '*', i, '=', result)
```

---

### global 변수 사용

```python
imsi = tf.constant(0)
su = tf.Variable(1)

@tf.function
def calcFunc3():
    global imsi             # 외부 변수임을 명시
    # su = tf.Variable(1)   # ❌ 에러 — 함수 내부 Variable 선언 불가
    for _ in range(3):
        imsi = tf.add(imsi, su)   # 파이썬 연산자 대신 tensor 연산자 권장
    return imsi
```

---

## 💻 전체 실습 코드

```python
import tensorflow as tf
import numpy as np

# ─────────────────────────────────────────
# tf.constant vs tf.Variable 출력 비교
# ─────────────────────────────────────────
node1 = tf.constant(3, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1)    # tf.Tensor(3.0, shape=(), dtype=float32)
print(node2)    # tf.Tensor(4.0, shape=(), dtype=float32)

adddata = tf.add(node1, node2)
print('adddata : ', adddata)    # tf.Tensor(7.0, shape=(), dtype=float32)

node3 = tf.Variable(3, dtype=tf.float32)
node4 = tf.Variable(4.0)
print(node3)    # <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=3.0>
print(node4)    # <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>

imsi1 = tf.add(node3, node4)    # 텐서 더하기 연산
print('imsi1 : ', imsi1)        # 7.0
node4.assign_add(node3)         # 변수값에 더하기 후 치환
print(node4)                    # numpy=7.0

# ─────────────────────────────────────────
# tf.cond — 텐서플로의 if문
# ─────────────────────────────────────────
a = tf.constant(5)
b = tf.constant(10)

result = tf.cond(a < b, lambda: tf.add(10, a), lambda: tf.square(a))
print('result : ', result)      # tf.Tensor(15, shape=(), dtype=int32)

# ─────────────────────────────────────────
# @tf.function — AutoGraph
# ─────────────────────────────────────────
@tf.function
def calcFunc1(a, b):    # tf.cond를 AutoGraph로 대체
    if a < b:
        return tf.add(10, a)
    else:
        return tf.square(a)

result2 = calcFunc1(a, b)
print('result2 : ', result2)    # tf.Tensor(15, shape=(), dtype=int32)

# ─────────────────────────────────────────
# 반복문 처리
# ─────────────────────────────────────────
@tf.function
def calcFunc2(n):
    hap = tf.constant(0)
    for i in tf.range(n):
        hap += i
    return hap

print('hap : ', calcFunc2(10))  # tf.Tensor(45, shape=(), dtype=int32)

# ─────────────────────────────────────────
# global 변수 사용
# ─────────────────────────────────────────
imsi = tf.constant(0)
su = tf.Variable(1)

@tf.function
def calcFunc3():
    global imsi
    for _ in range(3):
        imsi = tf.add(imsi, su)     # tensor 연산자 권장
    return imsi

print('imsi : ', calcFunc3())   # tf.Tensor(3, shape=(), dtype=int32)

# ─────────────────────────────────────────
# tf.print 포맷팅
# ─────────────────────────────────────────
@tf.function
def calcFunc4(dan):
    for i in range(1, 10):
        result = tf.multiply(dan, i)
        tf.print(dan, '*', i, '=', result)  # 인자 분리 방식

calcFunc4(3)
```

---

## 📊 tf.cond vs @tf.function 비교

```
# tf.cond 방식
result = tf.cond(a < b, lambda: tf.add(10, a), lambda: tf.square(a))

# @tf.function 방식 (AutoGraph)
@tf.function
def calcFunc(a, b):
    if a < b:           ← AutoGraph가 내부적으로 tf.cond로 변환
        return tf.add(10, a)
    else:
        return tf.square(a)
```

> 두 방식의 동작 결과는 동일 — `@tf.function`이 더 직관적이고 가독성 좋음

---

## 🔑 핵심 포인트

> `tf.cond(조건, 참_함수, 거짓_함수)` — 반드시 **lambda로 감싸야** 함 (조건 평가 전 실행 방지) `@tf.function` — AutoGraph가 파이썬 제어문을 그래프 연산으로 자동 변환 `@tf.function` 내부에서 `tf.Variable` 선언 ❌ — 구조가 고정적이어야 함 외부 변수 사용 시 `global` 키워드 필요 `tf.print` 에서 문자열 `format()` ❌ — 인자 분리 방식으로 출력 파이썬 연산자(`+`) 대신 **tensor 연산자** (`tf.add`) 권장