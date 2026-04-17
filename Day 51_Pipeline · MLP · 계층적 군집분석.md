# Day 51_Pipeline · MLP · 계층적 군집분석
## 📅 2026-04-17

---
# 📄 ex48regressior.py — Regressor 성능 비교 (diabetes dataset)

---

## 🧠 개념 정리

### Pipeline이란

> 전처리 단계와 모델을 하나로 묶어 순서대로 자동 실행하는 도구 `fit()` 한 번으로 전처리 + 학습, `predict()` 한 번으로 전처리 + 예측이 동시에 실행됨

```
Pipeline([
    ("scaler", StandardScaler()),   # 1단계: 전처리
    ("model",  LinearRegression())  # 2단계: 모델
])
→ fit() 호출 시    : scaler.fit_transform() → model.fit() 자동 순서 실행
→ predict() 호출 시 : scaler.transform()    → model.predict() 자동 순서 실행
```

### Pipeline 파라미터 명명 규칙

> GridSearchCV와 함께 쓸 때 파라미터 이름은 반드시 `"단계이름__파라미터명"` (언더바 2개)

```python
"model__n_estimators": [100, 200]
#  ↑ Pipeline step 이름   ↑ 해당 모델의 파라미터
```

### 스케일링 필요 여부

> 모델 종류에 따라 스케일링이 필요한 경우와 불필요한 경우가 다름

|스케일링 필요|스케일링 불필요|
|---|---|
|LinearRegression|RandomForest|
|SVR|XGBoost / LightGBM|
|KNN|DecisionTree|
|Ridge, Lasso|트리 계열 전반|

> 트리 계열은 데이터를 분기 순서로만 보기 때문에 값의 크기가 달라도 결과가 같음 반면 LinearRegression, SVR, KNN은 거리나 크기를 직접 계산하므로 스케일 차이가 결과에 영향을 줌

### 평가 지표

|지표|설명|좋은 값|
|---|---|---|
|R² (결정계수)|모델이 데이터를 얼마나 잘 설명하는지 (0~1)|1에 가까울수록 좋음|
|RMSE|예측값과 실제값의 평균 오차|0에 가까울수록 좋음|

---

## 1단계 : 데이터 로드 및 확인

```python
from sklearn.datasets import load_diabetes

data = load_diabetes()
# feature 10개 (age, sex, bmi, bp 등) - 이미 표준화된 연속형 데이터
# target : 1년 후 당뇨 진행 수치 (연속형) → 회귀 문제

x = data.data    # (442, 10)
y = data.target  # [151.  75. ...] : 연속형 숫자

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)
```

---

## 2단계 : Pipeline + GridSearchCV 모델 정의

> 각 모델을 딕셔너리로 관리 → 루프 한 번으로 모든 모델을 동일한 방식으로 처리 가능

```python
models = {
    "LinearRegression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),         # 거리/크기 기반 → 스케일링 필요
            ("model", LinearRegression())
        ]),
        "params": {
            "model__fit_intercept": [True, False]  # 절편 사용 여부
        }
    },
    "RandomForest": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(random_state=42))
        ]),
        "params": {
            "model__n_estimators": [100, 200],    # 트리 개수
            "model__max_depth": [None, 5, 10],    # 트리 깊이 (None=제한없음)
            "model__min_samples_split": [2, 5]    # 분기 최소 샘플 수
        }
    },
    "XGBoost": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(random_state=42, verbosity=0))
        ]),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.05], # 학습률 (작을수록 천천히 학습)
            "model__max_depth": [3, 5]
        }
    },
    "SVR": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),         # 커널 거리 계산 → 스케일링 필수
            ("model", SVR())
        ]),
        "params": {
            "model__C": [0.1, 1, 10],             # 규제 강도 (클수록 덜 규제)
            "model__gamma": ["scale", "auto"],    # 커널 영향 범위
            "model__kernel": ["rbf"]              # 비선형 커널
        }
    },
    "KNN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),         # 유클리드 거리 사용 → 스케일링 필수
            ("model", KNeighborsRegressor())
        ]),
        "params": {
            "model__n_neighbors": [3, 5, 7],      # 참조할 이웃 수
            "model__weights": ["uniform", "distance"]  # 가중치 방식
        }
    }
}
```

---

## 3단계 : GridSearchCV 실행 (루프)

> 모든 모델을 동일한 루프로 처리 → 코드 중복 없이 5개 모델 한 번에 튜닝

```python
result = []       # 성능 결과 저장용 리스트
best_models = {}  # 모델명 : 최적 모델 객체 저장용 딕셔너리

for name, config in models.items():
    print(f"{name} 튜닝중...")

    grid = GridSearchCV(
        config["pipeline"],  # 전처리 + 모델이 묶인 파이프라인
        config["params"],    # 탐색할 파라미터 조합
        cv=5,                # 5겹 교차검증
        scoring="r2",        # R² 기준으로 최적 파라미터 선택
        n_jobs=-1            # 모든 CPU 코어 사용
    )
    grid.fit(x_train, y_train)  # 전처리 + 파라미터 탐색 + 학습 한 번에 실행

    best_model = grid.best_estimator_   # 최적 파라미터로 학습된 모델
    pred = best_model.predict(x_test)   # 전처리 + 예측 자동 실행

    rmse = np.sqrt(mean_squared_error(y_test, pred))  # 평균 오차
    r2 = r2_score(y_test, pred)                        # 설명력

    result.append([name, rmse, r2])     # 결과 리스트에 추가
    best_models[name] = best_model      # 딕셔너리에 모델 저장

    print("best params : ", grid.best_params_)
    print("R2 : ", r2)
```

### 실행 결과

```
LinearRegression 튜닝중...
best params :  {'model__fit_intercept': True}
R2 :  0.4526027629719195

RandomForest 튜닝중...
best params :  {'model__max_depth': 5, 'model__min_samples_split': 2, 'model__n_estimators': 200}
R2 :  0.45980874589723

XGBoost 튜닝중...
best params :  {'model__learning_rate': 0.01, 'model__max_depth': 3, 'model__n_estimators': 200}
R2 :  0.4639255401595046

SVR 튜닝중...
best params :  {'model__C': 10, 'model__gamma': 'scale', 'model__kernel': 'rbf'}
R2 :  0.4937125101265256

KNN 튜닝중...
best params :  {'model__n_neighbors': 7, 'model__weights': 'distance'}
R2 :  0.4446218777004597
```

### 성능 순위

|순위|모델|R²|
|---|---|---|
|1|SVR|0.4937|
|2|XGBoost|0.4639|
|3|RandomForest|0.4598|
|4|LinearRegression|0.4526|
|5|KNN|0.4446|

> 전체적으로 R²가 0.45~0.49 수준으로 낮음 → diabetes dataset 자체가 예측하기 어려운 데이터 SVR이 비선형 패턴을 잘 잡아서 1위

---

## 4단계 : 결과 저장 및 시각화

```python
# 결과를 DataFrame으로 변환
df_results = pd.DataFrame(result, columns=['modelname', 'rmse', 'r2'])

# 성능 비교 시각화 (R² + RMSE 나란히)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x="modelname", y="r2", data=df_results)
plt.title("튜닝 모델 결정계수")
plt.xticks(rotation=10)

plt.subplot(1, 2, 2)
sns.barplot(x="modelname", y="rmse", data=df_results)
plt.title("튜닝 모델 RMSE")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
```

![[ex48regressior1.png]]

---

## 5단계 : 최고 모델 예측 시각화

> 실제값 vs 예측값을 산점도로 그려 모델 성능을 직관적으로 확인 빨간 점선(y=x)에 가까울수록 예측이 정확함

```python
# R² 기준 1위 모델 선택
best_modelname = df_results.sort_values("r2", ascending=False).iloc[0]["modelname"]
best_model = best_models[best_modelname]
pred = best_model.predict(x_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred)                                              # 실제값 vs 예측값
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],  # 완벽한 예측선 (y=x)
         'r--')
plt.title(f'최고 모델 {best_modelname}')
plt.xlabel("실제값")
plt.ylabel("예측값")
plt.show()
```

![[ex48regressior2.png]]

---

## 📌 핵심 정리

### 전체 흐름

```
데이터 로드 (442, 10) → 연속형 target → 회귀 문제
    ↓
train / test split (8:2)
    ↓
모델별 Pipeline 정의 (전처리 + 모델)
    ↓
GridSearchCV (cv=5, scoring='r2') → 최적 파라미터 탐색
    ↓
결과 DataFrame 저장 → 시각화 (barplot)
    ↓
최고 모델 선택 → 실제값 vs 예측값 산점도
```

### 모델 딕셔너리 구조

```python
models = {
    "모델명": {
        "pipeline": Pipeline([...]),  # 전처리 + 모델
        "params": { "model__파라미터": [...] }  # 탐색할 파라미터
    }
}
# 루프에서 config["pipeline"], config["params"] 로 꺼내 사용
```

### 주요 함수 정리

|함수|설명|
|---|---|
|`Pipeline([("이름", 객체)])`|전처리 + 모델 순서대로 묶기|
|`GridSearchCV(pipeline, params, cv, scoring)`|파이프라인 통째로 파라미터 탐색|
|`grid.best_estimator_`|최적 파라미터로 학습된 파이프라인 반환|
|`grid.best_params_`|최적 파라미터 딕셔너리 반환|
|`r2_score(y_test, pred)`|결정계수 (0~1, 높을수록 좋음)|
|`mean_squared_error(y_test, pred)`|MSE → `np.sqrt()` 로 RMSE 계산|

### 주의사항

| 상황               | 올바른 처리                                                       |
| ---------------- | ------------------------------------------------------------ |
| Pipeline 파라미터 명명 | `"model__파라미터"` (언더바 2개 필수)                                  |
| 결과 저장 변수명        | `result` (리스트) / `best_models` (딕셔너리) 혼용 주의                  |
| 최고 모델 선택         | `iloc[0]` 대신 `sort_values("r2", ascending=False).iloc[0]` 사용 |
| 컬럼명 일관성          | DataFrame 컬럼명과 시각화 코드의 컬럼명 반드시 동일하게                          |
# 📄 인공신경망 (Artificial Neural Network)

> 참고 : [브런치 - 인공신경망](https://brunch.co.kr/@gdhan/6)

---

## 🧠 개념 정리

### 인공신경망(ANN)이란

> 기계학습과 인지과학 분야에서 고안한 학습 알고리즘 신경세포의 신호 전달체계를 모방한 인공뉴런(노드)이 학습을 통해 결합 세기를 변화시켜 문제를 해결하는 모델 전반을 가리킨다

|생물학적 신경세포|인공뉴런(퍼셉트론)|
|---|---|
|가지돌기 (Dendrite)|입력값|
|신경세포체 (Soma)|가중치 합산|
|축삭돌기 (Axon)|활성화 함수 → 출력값|

---

## 핵심 개념

### 퍼셉트론 (Perceptron)

> 신경세포가 임계값(threshold, θ)을 넘으면 신호를 전달하는 원리를 수학적으로 모델링한 것 각 입력 신호(xᵢ)에 가중치(ωᵢ)를 곱해 합산한 후 활성화 함수를 거쳐 결괏값을 출력

- AND, OR, NAND 같은 단순한 논리 게이트 구현 가능
- 가중치의 조합은 무수히 많으며, 원하는 연산을 만족하는 조합을 찾는 것이 목표

![[perceptron.png]]

---

### 단층 퍼셉트론의 한계

> 단층 퍼셉트론은 **선형 결정 경계**만 표현 가능 → XOR 문제 해결 불가

|문제|단층 퍼셉트론|이유|
|---|---|---|
|AND|✅ 가능|선형 분리 가능|
|OR|✅ 가능|선형 분리 가능|
|NAND|✅ 가능|선형 분리 가능|
|XOR|❌ 불가|비선형 → 직선 하나로 분리 불가|

---

### 다층 퍼셉트론 (MLP, Multi-Layer Perceptron)

> 입력층과 출력층 사이에 **은닉층(Hidden Layer)** 을 추가해 비선형 문제 해결 XOR = NAND + OR 의 결괏값을 다시 AND → 다층 구조로 해결 가능

```
입력층 → 은닉층 → 출력층
```

- 현재 모든 인공신경망은 입력층 / 은닉층 / 출력층으로 구성됨
- 은닉층을 여러 개 쌓을수록 더 복잡한 문제 해결 가능

---

### 활성화 함수 (Activation Function)

> 출력값을 **비선형(Non-Linear)** 으로 변환하는 수학적 장치 선형 함수만 쓰면 은닉층을 아무리 쌓아도 하나의 층과 동일한 효과 → 비선형 변환 필수

#### 활성화 함수가 비선형이어야 하는 이유

```
f(x) = cx 일 때, 은닉층이 3개라면
f(f(f(x))) = c³x  →  결국 g(x) = cx³ 인 1개 층과 동일
→ 은닉층을 여러 개 둔 효과가 없음
```

#### 주요 활성화 함수

|함수|특징|단점|
|---|---|---|
|Sigmoid|출력값 0~1 사이로 압축|양 끝 기울기 ≈ 0 → 기울기 소실 문제|
|Tanh|출력값 -1~1 사이|여전히 기울기 소실 발생|
|**ReLU**|0보다 크면 그대로 출력, 0 이하면 0|**기울기 소실 방지 → 현재 가장 많이 사용**|
|Leaky ReLU|0 이하에서도 작은 기울기 유지|ReLU의 Dead Neuron 문제 보완|

> 대부분의 인공신경망은 **ReLU** 를 기본 활성화 함수로 사용

---

### 학습 (Learning)

> 인공신경망이 무수히 많은 가중치(ωᵢ)의 조합을 **스스로** 찾아가는 과정

```
입력값 → 모델 f(W) → 예측 결괏값 ŷ
                          ↓
                       손실함수 L(y, ŷ)
                          ↓
                       가중치 W 변경
                          ↑ (반복)
```

---

### 손실 함수 (Loss Function)

> 예측값(Prediction)과 정답값(Labeling)이 **얼마나 다른지** 측정하는 함수 같은 의미로 **비용함수(Cost Function)** 라고도 부름 두 값이 비슷할수록 작은 값이 나오도록 설계됨

|손실 함수|사용 상황|
|---|---|
|평균제곱오차 (MSE, Mean Squared Error)|회귀 문제|
|교차엔트로피 오차 (Cross Entropy Error)|분류 문제|

---

## 📌 핵심 정리

### 전체 흐름

```
단층 퍼셉트론 (AND, OR, NAND 가능)
    ↓ XOR 해결 못함 (선형 한계)
다층 퍼셉트론 등장 (은닉층 추가)
    ↓ 선형으로만 전파되면 층이 의미없음
활성화 함수 도입 (비선형 변환)
    ↓ 가중치를 어떻게 찾을까?
손실 함수로 오차 측정 → 가중치 변경 반복 = 학습
```

### 용어 정리

| 용어                 | 설명                               |
| ------------------ | -------------------------------- |
| 퍼셉트론 (Perceptron)  | 인공신경망의 기본 단위, 단층 신경망             |
| 은닉층 (Hidden Layer) | 입력층과 출력층 사이의 중간 층                |
| 가중치 (Weight, ω)    | 각 입력 신호의 중요도를 나타내는 값             |
| 임계값 (Threshold, θ) | 신호를 전달할지 말지 결정하는 기준값             |
| 활성화 함수             | 출력을 비선형으로 변환하는 함수                |
| 손실 함수              | 예측값과 정답값의 차이를 측정하는 함수            |
| 기울기 소실             | 층이 깊어질수록 기울기가 0에 수렴해 학습이 안 되는 문제 |
# 📄 ex49perceptron.py — Perceptron (논리회로 · 일반 분류)

---

## 🧠 개념 정리

### Perceptron이란

> sklearn이 제공하는 단층신경망 (뉴런, 노드) 이진 분류(Binary Classification)만 가능한 선형 모델

### Perceptron 학습 방식

> 딥러닝의 경사하강법과 달리 **틀린 것만 골라서 고치는** 알고리즘

```
예측 → 맞았는지 확인 → 틀리면 Weight 갱신, 맞으면 통과 → max_iter 만큼 반복
```

- 선형회귀식 기반 (LogisticRegression을 기반으로 함)
- input에 대한 가중치 합 계산 후 실제값과 예측값 비교 (Loss Function)
- 역전파를 통해 W를 갱신하기를 max_iter 만큼 반복

### 결정 경계 (Decision Boundary)

> 퍼셉트론의 결정 경계 수식 : `w1*x1 + w2*x2 + b = 0` x2 기준으로 풀면 : `x2 = -(w1*x1 + b) / w2`

### 주요 파라미터

|파라미터|설명|
|---|---|
|`max_iter`|학습 반복 횟수 (epoch)|
|`eta0`|학습률 (Learning Rate) — 가중치 갱신 크기|
|`random_state`|재현성을 위한 난수 시드|
|`coef_`|학습된 가중치 W|
|`intercept_`|학습된 바이어스 b|

### XOR 문제와 단층 퍼셉트론의 한계

> 퍼셉트론은 **선형 모델**이므로 직선 하나로 분리할 수 없는 XOR 문제는 해결 불가 → 해결하려면 은닉층을 추가한 MLP(다층 퍼셉트론) 필요

---

## 실습 1 : 논리회로 분류

```python
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

feature = np.array([[0,0], [0,1], [1,0], [1,1]])

# label = np.array([0,0,0,1])  # AND  → 해결 가능
# label = np.array([0,1,1,1])  # OR   → 해결 가능
label = np.array([0,1,1,0])    # XOR  → 해결 못함 (선형 모델 한계)

ml = Perceptron(max_iter=10).fit(feature, label)  # max_iter = epoch(학습 횟수)
pred = ml.predict(feature)
print('pred : ', pred)
print('acc : ', accuracy_score(label, pred))
```

### 결과 해석

|논리회로|단층 퍼셉트론|이유|
|---|---|---|
|AND|✅ 가능|선형 분리 가능|
|OR|✅ 가능|선형 분리 가능|
|XOR|❌ 불가|직선 하나로 0과 1 분리 불가|

---

## 실습 2 : 일반 자료 분류

```python
x = np.array([
    [2, 3],  # label  1
    [3, 3],  # label  1
    [1, 1],  # label  1
    [5, 2],  # label -1
    [6, 1]   # label -1
])
y = np.array([1, 1, 1, -1, -1])

model = Perceptron(max_iter=100000, eta0=0.1, random_state=42)  # eta0 = 학습률
model.fit(x, y)

pred = model.predict(x)
print('예측값 : ', pred)   # [ 1  1  1  1 -1]
print('실제값 : ', y)      # [ 1  1  1 -1 -1]
print('정확도 : ', accuracy_score(y, pred))  # 0.8

print('가중치(W) : ', model.coef_)       # [[-0.4  0.8]]
print('바이어스(B) : ', model.intercept_) # [0.4]
```

### 정확도가 0.8인 이유

```
예측값 : [ 1  1  1  1 -1]
실제값 : [ 1  1  1 -1 -1]
                    ↑ [5, 2]를 -1이어야 하는데 1로 예측 → 5개 중 1개 오답 → 0.8
```

> 데이터가 5개뿐이라 결정 경계가 완벽하게 수렴하기 어려움 `max_iter=100000`을 줘도 이 데이터 구조상 완벽한 분리 불가

---

## 결정 경계 시각화

```python
import matplotlib.pyplot as plt
import koreanize_matplotlib

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr')  # 데이터 산점도 (파랑=1, 빨강=-1)

w = model.coef_[0]        # 가중치 [w1, w2]
b = model.intercept_[0]   # 바이어스

x_vals = np.linspace(0, 7, 100)
y_vals = -(w[0] * x_vals + b) / w[1]  # x2 = -(w1*x1 + b) / w2

plt.plot(x_vals, y_vals)
plt.title('sklearn Perceptron Decision Boundary')
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
```

![[ex49perceptron.png]]

### 결정 경계 수식 유도

```
퍼셉트론 결정 경계 : w1*x1 + w2*x2 + b = 0
x2 기준으로 정리  : x2 = -(w1*x1 + b) / w2
```

---

## 📌 핵심 정리

### 전체 흐름

```
논리회로 (AND/OR/XOR)
    ↓ XOR은 선형 분리 불가 → 단층 퍼셉트론의 한계 확인
일반 자료 분류 (이진 레이블 1 / -1)
    ↓ 학습 후 가중치(coef_), 바이어스(intercept_) 확인
결정 경계 시각화 → w1*x1 + w2*x2 + b = 0 을 직선으로 표현
```

### 주요 함수 정리

| 함수 / 속성                                    | 설명                               |
| ------------------------------------------ | -------------------------------- |
| `Perceptron(max_iter, eta0, random_state)` | 단층 신경망 모델 생성                     |
| `model.fit(x, y)`                          | 학습                               |
| `model.predict(x)`                         | 예측                               |
| `model.coef_`                              | 학습된 가중치 W (shape: [1, feature수]) |
| `model.intercept_`                         | 학습된 바이어스 b                       |
| `accuracy_score(y, pred)`                  | 정확도 계산                           |

### 주의사항

|상황|내용|
|---|---|
|XOR 분류|단층 퍼셉트론으로 해결 불가 → MLP 필요|
|레이블 형식|Perceptron은 1 / -1 또는 0 / 1 모두 사용 가능|
|결정 경계 시각화|`coef_[0]`, `intercept_[0]` 으로 직선 수식 계산 후 plot|

---

# 📄 ex50perceptron_iris.py — Perceptron 분류 (iris dataset)

---

## 🧠 개념 정리

### LogisticRegression과 다중 클래스

> `LogisticRegression`은 다중 클래스(label)를 지원하도록 일반화됨 이를 **Softmax Regression** 또는 **Multinomial Logistic Regression** 이라고 부름

### 표준화 (StandardScaler)

> 데이터 크기를 표준화하여 최적화 과정에서 안정성, 수렴속도 향상, 과적합/과소적합 방지 효과

```python
sc = StandardScaler()
sc.fit(x_train)               # train으로만 fit (기준 생성)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test) # test는 transform만 (fit 하면 데이터 누수)

# 원복
ori_x_train = sc.inverse_transform(x_train)
```

> iris dataset은 feature 간 크기 차이가 거의 없어 표준화 효과 미미 → 생략 가능

### predict_proba (Softmax 출력값)

> 예측 결과는 softmax 확률값 중 가장 큰 인덱스가 출력된 값 `Perceptron`은 `predict_proba` 미지원 → `LogisticRegression`, `MLPClassifier` 등에서만 사용 가능

### 과적합 판단

> `train score`와 `test score`의 차이가 크면 **과적합(Overfitting)** 의심

|상태|의미|
|---|---|
|train score ≈ test score|정상 (일반화 잘 됨)|
|train score >> test score|과적합 (훈련 데이터에만 치우침)|

---

## 1단계 : 데이터 로드 및 확인

```python
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

# 꽃잎 길이(2열), 꽃잎 너비(3열) 2개 feature만 사용 → 시각화(2D) 목적
x = iris.data[:, [2, 3]]
y = iris.target   # 0, 1, 2 (3개 클래스)

# 꽃잎 길이와 너비의 상관계수 → 매우 높음
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])[0, 1])  # 0.9628654314027961

print(x.shape, y.shape)  # (150, 2)  (150,)
```

---

## 2단계 : 데이터 분할

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0
)
print(x_train.shape, x_test.shape)  # (105, 2)  (45, 2)
```

---

## 3단계 : Perceptron 학습 및 평가

```python
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

model = Perceptron(max_iter=1000, random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  2 1 1 2 0 2 0 0]
print('실제값 : ', y_test)
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  1 1 1 2 0 2 0 0]

print(f'총 갯수:{len(y_test)}, 오류수:{(y_test != y_pred).sum()}')
# 총 갯수:45, 오류수:1
```

### 정확도 확인

```python
# 방법 1 : accuracy_score
print(accuracy_score(y_test, y_pred))  # 0.9777...

# 방법 2 : model.score()
print('test score : ', model.score(x_test, y_test))    # 0.9777...
print('train score : ', model.score(x_train, y_train)) # 0.9523...
# test ≈ train → 과적합 없음
```

> 주석의 정확도 0.6은 잘못된 주석 — 오류수 1개면 정확도는 44/45 = **0.9777**

---

## 4단계 : 모델 저장 및 불러오기

```python
import joblib  # pickle보다 빠르고 대용량(numpy 배열 포함) 지원

joblib.dump(model, 'logimodel.pkl')  # 모델 저장
del model                             # 메모리에서 삭제

read_model = joblib.load('logimodel.pkl')  # 모델 불러오기
# 이후 read_model 사용
```

---

## 5단계 : 새로운 데이터로 예측

```python
new_data = np.array([[5.5, 2.2], [0.6, 0.3], [1.1, 0.5]])

new_pred = read_model.predict(new_data)
print('예측 결과 : ', new_pred)  # [2 0 0]
# softmax 확률값 중 가장 큰 인덱스가 출력됨

# predict_proba는 Perceptron에서 지원 안 함
# → LogisticRegression, MLPClassifier 등에서만 사용 가능
```

---

## 6단계 : 결정 경계 시각화

```python
import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib.colors import ListedColormap

def plot_decision_regionFunc(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')
    colors  = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap    = ListedColormap(colors[:len(np.unique(y))])

    # 격자 좌표 생성 (결정 경계 배경 그리기용)
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    # ravel() : 2D → 1D, .T : 전치 → predict() 입력 형태로 변환
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)  # 원래 배열 모양으로 복원

    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)  # 배경 색상으로 경계 표현

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
                    color=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

# train + test 합쳐서 전체 분포 시각화
x_combined = np.vstack((x_train, x_test))
y_combined  = np.hstack((y_train, y_test))
plot_decision_regionFunc(
    X=x_combined, y=y_combined,
    classifier=read_model,
    test_idx=range(105, 150),  # test 데이터 인덱스 (동그라미로 표시)
    title='scikit-learn제공'
)
```

![[ex50perceptron_iris.png]]

---

## 📌 핵심 정리

### 전체 흐름

```
iris 로드 → 꽃잎 길이/너비 2개 feature 선택 (시각화 목적)
    ↓
train/test split (7:3)
    ↓
Perceptron 학습 (max_iter=1000)
    ↓
정확도 확인 (accuracy_score, model.score)
    ↓
joblib으로 모델 저장 → 불러오기
    ↓
새로운 데이터 예측 + 결정 경계 시각화
```

### 주요 함수 정리

|함수 / 속성|설명|
|---|---|
|`np.corrcoef(a, b)[0,1]`|두 변수 간 상관계수|
|`model.score(x, y)`|정확도 한 번에 계산|
|`joblib.dump(model, 'file.pkl')`|모델 저장|
|`joblib.load('file.pkl')`|모델 불러오기|
|`np.vstack()`|배열 수직 결합 (행 추가)|
|`np.hstack()`|배열 수평 결합 (열 추가)|
|`np.meshgrid()`|격자 좌표 생성 (결정 경계 그릴 때 사용)|
|`arr.ravel()`|다차원 배열 → 1차원으로 펼치기|
|`plt.contourf()`|등고선 배경으로 결정 경계 표현|

### 주의사항

|상황|내용|
|---|---|
|스케일링 순서|`fit`은 train에만, test는 `transform`만|
|`predict_proba`|Perceptron은 미지원 → MLPClassifier 등 사용|
|정확도 주석 오류|오류수 1개 → 정확도 0.9777 (0.6은 잘못된 주석)|

---

# 📄 MLP (Multi-Layer Perceptron) 다층신경망

---

## 🧠 개념 정리

### MLP란

> 여러 개의 퍼셉트론 뉴런을 여러 층으로 쌓은 **다층신경망 구조** 입력층과 출력층 사이에 하나 이상의 **은닉층(Hidden Layer)** 을 가짐 인접한 두 층의 뉴런 간에는 **완전 연결(Fully Connected)** 됨

![[mlp.png]]

### 단층 퍼셉트론 vs MLP

|구분|단층 퍼셉트론|MLP|
|---|---|---|
|은닉층|없음|1개 이상|
|결정 경계|직선 (선형)|곡선 (비선형)|
|XOR 해결|❌ 불가|✅ 가능|
|활성화 함수|Step 함수|ReLU, tanh 등 비선형|
|학습 방식|틀린 것만 수정|역전파로 전체 가중치 갱신|
|sklearn|`Perceptron`|`MLPClassifier`|

### 다층 뉴런이 필요한 이유

> 단층 퍼셉트론은 선형 분리만 가능 → XOR 같은 비선형 문제 해결 불가 뉴런을 늘리고 층을 추가하면 **다각형 모양의 복잡한 결정 경계** 생성 가능 일반적으로 MLP는 **2~3개 은닉층**으로 구성

---

## 핵심 개념

### 순전파 / 역전파

```
순방향 전파 (Forward Propagation)
입력층 → 은닉층 → 출력층 → 예측값 계산

역방향 전파 (Back Propagation)
출력층 → 은닉층 → 입력층 방향으로 오차 전달 → 가중치 갱신
```

> 순방향으로 예측하고, 역방향으로 오차를 줄여나가는 과정을 반복하는 것이 MLP 학습

---

### 활성화 함수 (Activation Function)

> 출력값을 **비선형**으로 변환하는 장치 선형 함수만 쓰면 층을 아무리 쌓아도 하나의 층과 동일한 효과

```
f(x) = cx 일 때, 은닉층 3개라면
f(f(f(x))) = c³x  →  결국 1개 층과 동일
→ 비선형 활성화 함수가 반드시 필요
```

|함수|출력 범위|실무 사용|
|---|---|---|
|Step|0 or 1|퍼셉트론에만 사용|
|Sigmoid|0 ~ 1|MLP까지만, Deep Learning 비권장|
|Tanh|-1 ~ 1|Sigmoid보다 낫지만 기울기 소실 있음|
|**ReLU**|0 또는 양수|**실무 가장 많이 사용**|
|Leaky ReLU|음수도 작은 값|ReLU Dead Neuron 보완|
|ELU|음수 구간 부드러움|Leaky ReLU 개선 버전|

> 활성화 함수 선택 순서 : ReLU → 부족하면 Leaky ReLU / ELU → 그래도 부족하면 tanh

---

### 손실 함수 (Loss Function)

> 예측값과 정답값이 **얼마나 다른지** 측정하는 함수 학습이 잘 될수록 loss가 0에 가까워짐

|손실 함수|사용 상황|
|---|---|
|MSE (평균제곱오차)|회귀 문제|
|Cross Entropy Error|분류 문제|

---

### MLP 학습 과정 (역전파)

```
1. 순전파 : 입력 → 신경망 → 예측값 출력
2. 오차 계산 : L = (y - ŷ)  → 예측이 틀릴수록 값이 커짐
3. 미분(기울기 계산) : 오차가 줄어드는 방향 탐색
4. 가중치 W 갱신
5. 1~4 반복  →  역전파 (Back Propagation)
```

> 미분을 쓰는 이유 : 오차를 **어떤 방향으로 줄여야 할지** 알기 위해

---

### 파라미터 수 계산

```
param = (input 뉴런 수 + 1) × output 뉴런 수
# +1 은 바이어스(bias) 항
```

---

## sklearn MLPClassifier 주요 파라미터

|파라미터|설명|
|---|---|
|`hidden_layer_sizes`|은닉층 구조 — `(10,)` : 1층 10개, `(20, 10)` : 2층 각 20/10개|
|`activation`|활성화 함수 — `relu`(기본 권장), `tanh`, `sigmoid`|
|`solver`|가중치 최적화 방식 — `adam` (기본 권장)|
|`learning_rate_init`|초기 학습률|
|`max_iter`|학습 반복 횟수 (epoch) — 권장 500~1000|
|`verbose`|학습 중 로그 출력 여부 (1=출력)|

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(20, 10),  # 은닉층 2개
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)
```

---

## 📌 핵심 정리

### 전체 흐름

```
단층 퍼셉트론 → XOR 해결 불가 (선형 한계)
    ↓
은닉층 추가 → MLP (비선형 경계 가능)
    ↓
활성화 함수 (비선형 변환) → 층을 쌓는 효과 발생
    ↓
손실 함수로 오차 측정
    ↓
역전파 → 가중치 갱신 반복 = 학습
```

### 주의사항

| 상황                      | 내용                                  |
| ----------------------- | ----------------------------------- |
| `max_iter` 부족           | ConvergenceWarning → 500~1000으로 늘리기 |
| `hidden_layer_sizes`    | 문자열 넣으면 오류 → 반드시 정수 또는 튜플           |
| 스케일링                    | MLP는 StandardScaler 표준화 권장          |
| Deep Learning에서 Sigmoid | 기울기 소실 문제 → 사용 비권장                  |

---
# 📄 ex51mlp.py — MLP 다층신경망 (논리회로 · make_moons)

---

## 🧠 개념 정리

### MLP (Multi-Layer Perceptron)이란

> 여러 개의 퍼셉트론 뉴런을 여러 층으로 쌓은 **다층신경망 구조** 입력층과 출력층 사이에 하나 이상의 **은닉층(Hidden Layer)** 을 가짐 인접한 두 층의 뉴런 간에는 **완전 연결(Fully Connected)** 됨

```
입력층 → 은닉층1 → 은닉층2 → ... → 출력층
         (뉴런 n개) (뉴런 m개)
```

### 단층 퍼셉트론 vs MLP

|구분|단층 퍼셉트론|MLP|
|---|---|---|
|구조|입력층 + 출력층|입력층 + 은닉층(1개 이상) + 출력층|
|XOR 해결|❌ 불가|✅ 가능|
|결정 경계|직선 (선형)|비선형 곡선|

### 주요 파라미터

|파라미터|설명|
|---|---|
|`hidden_layer_sizes`|은닉층 구조 — `(10,)` : 1개층 뉴런 10개, `(10, 10)` : 2개층 각 10개|
|`activation`|활성화 함수 — `relu`, `sigmoid`, `tanh`|
|`solver`|가중치 최적화 방식 — `adam` (기본 권장)|
|`learning_rate_init`|초기 학습률|
|`max_iter`|학습 반복 횟수 (epoch) — 권장 500~1000|
|`verbose`|학습 중 로그 출력 여부 (1=출력)|
|`random_state`|재현성을 위한 난수 시드|

### make_moons란

> 초승달 모양 2개가 겹쳐있는 **비선형 분류용 인공 데이터셋** 단순 직선으로 분리 불가 → MLP 같은 비선형 모델에 적합

|파라미터|설명|
|---|---|
|`n_samples`|생성할 데이터 수|
|`noise`|노이즈 크기 (클수록 경계가 불명확)|
|`random_state`|재현성 시드|

---

## 실습 1 : 논리회로 분류 (XOR)

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

feature = np.array([[0,0], [0,1], [1,0], [1,1]])

# label = np.array([0,0,0,1])  # AND
# label = np.array([0,1,1,1])  # OR
label = np.array([0,1,1,0])    # XOR → MLP로 해결 가능

ml = MLPClassifier(
    hidden_layer_sizes=10,       # 은닉층 1개, 뉴런 10개
    solver='adam',               # 가중치 최적화 방식
    learning_rate_init=0.01,     # 초기 학습률
    max_iter=500,                # 권장 횟수 : 500~1000
    verbose=1                    # 학습 로그 출력
).fit(feature, label)

pred = ml.predict(feature)
print('pred : ', pred)                       # [0 1 1 0]
print('acc : ', accuracy_score(label, pred)) # 1.0
```

### 결과 해석

> 단층 퍼셉트론은 XOR 해결 불가 → MLP는 은닉층 덕분에 비선형 경계 학습 가능 → XOR 해결

---

## 실습 2 : make_moons 분류

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 초승달 모양 비선형 데이터 생성
x, y = make_moons(n_samples=300, noise=0.2, random_state=42)
# x : 2개 feature (좌표값)
# y : 0 또는 1 (2개 클래스)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(10, 10),  # 은닉층 2개, 각 뉴런 10개
    activation='relu',            # 활성화 함수
    solver='adam',
    max_iter=1000,
    random_state=42
)
model.fit(x_train, y_train)

pred = model.predict(x_test)
print('acc : ', accuracy_score(y_test, pred))  # 0.9666...
```

### 결과 해석

> `make_moons`는 직선으로 분리 불가한 비선형 데이터 MLP가 은닉층 2개로 비선형 경계를 학습해 **96.7% 정확도** 달성

---

## 📌 핵심 정리

### 전체 흐름

```
실습1) XOR 논리회로
    단층 퍼셉트론 → 해결 불가
    MLP (은닉층 1개, 뉴런 10개) → 해결 가능 (acc 1.0)

실습2) make_moons 비선형 분류
    make_moons으로 인공 데이터 생성
    MLP (은닉층 2개, 각 뉴런 10개) → acc 0.967
```

### hidden_layer_sizes 작성법

```python
hidden_layer_sizes=10         # 은닉층 1개, 뉴런 10개
hidden_layer_sizes=(10,)      # 위와 동일
hidden_layer_sizes=(10, 10)   # 은닉층 2개, 각 뉴런 10개
hidden_layer_sizes=(100, 50, 25)  # 은닉층 3개
```

### 주요 함수 정리

|함수|설명|
|---|---|
|`MLPClassifier(...)`|다층신경망 분류 모델 생성|
|`make_moons(n_samples, noise)`|비선형 인공 데이터 생성|
|`model.fit(x, y)`|학습|
|`model.predict(x)`|예측|
|`accuracy_score(y, pred)`|정확도 계산|

### 주의사항

|상황|내용|
|---|---|
|`max_iter` 설정|너무 작으면 수렴 안 됨 → 권장 500~1000, ConvergenceWarning 뜨면 늘리기|
|`hidden_layer_sizes`|문자열(`'adam'`) 넣으면 오류 → 반드시 정수 또는 튜플로 지정|
|비선형 데이터|단층 퍼셉트론 대신 MLP 사용|

---

# 📄 ex52mlp_wine.py — MLP 다항분류 (wine dataset)

---

## 🧠 개념 정리

### MLP 학습 과정 (역전파)

> MLP는 오차를 미분(기울기)으로 계산해 가중치를 갱신하는 방식으로 학습

```
1. 모델이 예측 (Forward)
2. 오차 계산 : L = (y - ŷ)  → 예측이 틀릴수록 값이 커짐
3. 미분(기울기 계산) : 오차가 줄어드는 방향 탐색
4. 가중치 W 갱신
5. 1~4 반복  →  역전파 (Back Propagation)
```

### 손실 함수 (Loss Function)

> 예측값(ŷ)과 실제값(y)의 차이를 수치로 표현 학습이 잘 될수록 loss가 0에 가까워짐 → **loss curve가 우하향**하면 정상 학습

### 미분을 쓰는 이유

> 오차를 **어떤 방향으로 줄여야 할지** 알기 위해 사용 미분값(기울기)이 가중치를 얼마나, 어떤 방향으로 바꿀지 알려줌

### MLP에서 스케일링이 필요한 이유

> MLP는 가중치 합산 + 활성화 함수 기반 → feature 간 크기 차이가 크면 학습 불안정 **StandardScaler로 표준화 권장**

---

## 1단계 : 데이터 로드 및 확인

```python
from sklearn.datasets import load_wine

data = load_wine()
x = data.data    # (178, 13) : 13개 feature
y = data.target  # [0 1 2] : 3개 클래스 (와인 등급)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y      # 클래스 비율 유지하며 분리
)
```

---

## 2단계 : 스케일링

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # train으로만 fit
x_test_scaled  = scaler.transform(x_test)        # test는 transform만
```

---

## 3단계 : MLP 모델 생성 및 학습

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(20, 10),  # 은닉층 2개 (1층:뉴런 20개, 2층:뉴런 10개)
    activation='relu',            # 활성화 함수
    solver='adam',                # 손실 최소화 방식
    learning_rate_init=0.001,     # 초기 학습률
    max_iter=150,                 # 학습 횟수 (epoch)
    random_state=42,
    verbose=1                     # 학습 중 로그 출력
)

model.fit(x_train_scaled, y_train)
pred = model.predict(x_test_scaled)
```

---

## 4단계 : 성능 평가

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print('accuracy_score : ', accuracy_score(y_test, pred))  # 0.9444...
print(classification_report(y_test, pred))
#               precision  recall  f1-score  support
#            0       1.00    1.00      1.00       12
#            1       0.88    1.00      0.93       14
#            2       1.00    0.80      0.89       10
#     accuracy                         0.94       36
```

### 결과 해석

> 클래스 2(등급 2)에서 recall 0.80 → 실제 2등급 10개 중 2개를 1등급으로 오분류 클래스 0, 1은 완벽하게 분류

---

## 5단계 : 혼동행렬 시각화

```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('confusion_matrix')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
```

![[ex52mlp_wine.png]]

---

## 6단계 : train loss curve 시각화

```python
# model.loss_curve_ : epoch별 loss 값 리스트 (verbose=1 또는 학습 후 자동 저장)
plt.plot(model.loss_curve_)
plt.title('train loss curve')
plt.xlabel('iteration(epoch)')
plt.ylabel('loss')
plt.show()
```

![[ex52mlp_wine2.png]]

> loss가 1.5에서 시작해 0.1 수준으로 수렴 → 정상적으로 학습됨 곡선이 우하향하며 수렴할수록 학습이 잘 된 것

---

## 📌 핵심 정리

### 전체 흐름

```
wine 데이터 로드 (178행, 13 feature, 3클래스)
    ↓
train/test split (stratify=y)
    ↓
StandardScaler 표준화  ← MLP는 스케일링 권장
    ↓
MLPClassifier (은닉층 2개: 20→10)
    ↓
평가 (accuracy, classification_report)
    ↓
혼동행렬 heatmap + loss curve 시각화
```

### MLP 학습 핵심 요약

| 단계      | 내용                      |
| ------- | ----------------------- |
| Forward | 입력 → 신경망 → 예측값 출력       |
| Loss 계산 | 예측값과 실제값의 차이 측정         |
| 미분      | 오차가 줄어드는 방향(기울기) 계산     |
| 가중치 갱신  | 기울기 방향으로 W 업데이트         |
| 역전파     | 위 과정을 출력층 → 입력층 방향으로 반복 |

### 주요 함수 정리

|함수 / 속성|설명|
|---|---|
|`MLPClassifier(hidden_layer_sizes, activation, solver)`|다층신경망 모델 생성|
|`model.loss_curve_`|epoch별 loss 값 리스트 (학습 후 사용 가능)|
|`confusion_matrix(y_test, pred)`|혼동행렬 반환|
|`sns.heatmap(cm, annot=True, fmt='d')`|혼동행렬 시각화|
|`classification_report(y_test, pred)`|precision / recall / f1 한 번에 출력|

### 주의사항

|상황|내용|
|---|---|
|스케일링|MLP는 StandardScaler 표준화 권장|
|`loss_curve_` 사용|`fit()` 이후에만 접근 가능|
|`max_iter` 부족|loss curve가 수렴 안 하면 늘리기|
|`stratify=y`|클래스 불균형 데이터에서 비율 유지 목적|

---
# 📄 ex53mlp_iris.py — MLP 분류 (iris dataset)

---

## 🧠 개념 정리

### LogisticRegression과 다중 클래스

> `LogisticRegression`은 다중 클래스를 지원하도록 일반화됨 이를 **Softmax Regression** 또는 **Multinomial Logistic Regression** 이라고 부름

### 표준화 (StandardScaler)

> 데이터 크기를 표준화하여 최적화 과정에서 안정성, 수렴속도 향상 효과

```python
sc = StandardScaler()
sc.fit(x_train)               # train으로만 fit (기준 생성)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test) # test는 transform만

# 원복
ori_x_train = sc.inverse_transform(x_train)
```

> iris dataset은 feature 간 크기 차이가 거의 없어 표준화 효과 미미 → 생략 가능

### predict_proba (Softmax 출력값)

> 예측 결과는 softmax 확률값 중 가장 큰 인덱스가 출력된 값 `MLPClassifier`는 `predict_proba` 지원 → 각 클래스에 속할 확률 반환 가능

---

## 1단계 : 데이터 로드 및 확인

```python
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

# 꽃잎 길이(2열), 꽃잎 너비(3열) 2개 feature만 사용 → 시각화(2D) 목적
x = iris.data[:, [2, 3]]
y = iris.target   # 0, 1, 2 (3개 클래스)

# 꽃잎 길이와 너비의 상관계수 → 매우 높음
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])[0, 1])  # 0.9628654314027961

print(x.shape, y.shape)  # (150, 2)  (150,)
```

---

## 2단계 : 데이터 분할

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0
)
print(x_train.shape, x_test.shape)  # (105, 2)  (45, 2)

# iris는 feature 간 크기 차이가 거의 없어 표준화 효과 미미 → 생략
```

---

## 3단계 : MLP 모델 학습 및 평가

```python
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

model = MLPClassifier(
    hidden_layer_sizes=(20, 10),  # 은닉층 2개 (1층:뉴런 20개, 2층:뉴런 10개)
    activation='relu',            # 활성화 함수
    solver='adam',                # 손실 최소화 방식
    learning_rate_init=0.001,     # 초기 학습률
    max_iter=1000,                # 학습 횟수 (epoch)
    random_state=42,
    verbose=1                     # 학습 중 로그 출력
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  2 1 1 2 0 2 0 0]
print('실제값 : ', y_test)
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  1 1 1 2 0 2 0 0]

print(f'총 갯수:{len(y_test)}, 오류수:{(y_test != y_pred).sum()}')
# 총 갯수:45, 오류수:1
```

### 정확도 확인

```python
# 방법 1 : accuracy_score
print(accuracy_score(y_test, y_pred))  # 0.9777...

# 방법 2 : model.score()
print('test score : ', model.score(x_test, y_test))    # 0.9777...
print('train score : ', model.score(x_train, y_train)) # 0.9523...
# test ≈ train → 과적합 없음
```

> 주석의 정확도 0.6은 잘못된 주석 — 오류수 1개면 정확도는 44/45 = **0.9777**

---

## 4단계 : 모델 저장 및 불러오기

```python
import joblib  # pickle보다 빠르고 대용량(numpy 배열 포함) 지원

joblib.dump(model, 'logimodel.pkl')  # 모델 저장
del model                             # 메모리에서 삭제

read_model = joblib.load('logimodel.pkl')  # 모델 불러오기
```

---

## 5단계 : 새로운 데이터로 예측

```python
new_data = np.array([[5.5, 2.2], [0.6, 0.3], [1.1, 0.5]])

new_pred = read_model.predict(new_data)
print('예측 결과 : ', new_pred)  # [2 0 0]
# softmax 확률값 중 가장 큰 인덱스가 출력됨

# MLPClassifier는 predict_proba 지원 가능
print(read_model.predict_proba(new_data))
# [[4.50e-06 5.66e-03 9.94e-01]  → 클래스 2일 확률 99.4%
#  [9.99e-01 1.37e-05 2.88e-06]  → 클래스 0일 확률 99.9%
#  [9.99e-01 1.37e-05 2.88e-06]] → 클래스 0일 확률 99.9%
```

---

## 6단계 : 결정 경계 시각화

```python
import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib.colors import ListedColormap

def plot_decision_regionFunc(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')
    colors  = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap    = ListedColormap(colors[:len(np.unique(y))])

    # 격자 좌표 생성 (결정 경계 배경 그리기용)
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    # ravel() : 2D → 1D, .T : 전치 → predict() 입력 형태로 변환
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)  # 원래 배열 모양으로 복원

    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)  # 배경 색상으로 경계 표현

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
                    color=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined = np.vstack((x_train, x_test))
y_combined  = np.hstack((y_train, y_test))
plot_decision_regionFunc(
    X=x_combined, y=y_combined,
    classifier=read_model,
    test_idx=range(105, 150),
    title='scikit-learn제공'
)
```

![[ex53mlp_iris.png]]

---

## 📌 핵심 정리

### 전체 흐름

```
iris 로드 → 꽃잎 길이/너비 2개 feature 선택 (시각화 목적)
    ↓
train/test split (7:3)
    ↓
MLPClassifier (은닉층 2개: 20→10, relu, adam)
    ↓
정확도 확인 (accuracy_score, model.score)
    ↓
joblib으로 모델 저장 → 불러오기
    ↓
새로운 데이터 예측 (predict_proba 확인)
    ↓
결정 경계 시각화 (contourf)
```

### Perceptron vs MLPClassifier 비교

|구분|Perceptron|MLPClassifier|
|---|---|---|
|은닉층|없음|있음 (hidden_layer_sizes)|
|XOR 해결|❌|✅|
|predict_proba|❌ 미지원|✅ 지원|
|활성화 함수|Step|relu, tanh 등|
|학습 방식|틀린 것만 수정|역전파 전체 갱신|

### 주요 함수 정리

|함수 / 속성|설명|
|---|---|
|`MLPClassifier(hidden_layer_sizes, activation, solver)`|다층신경망 모델 생성|
|`model.fit(x, y)`|학습|
|`model.predict(x)`|예측 (클래스 인덱스 반환)|
|`model.predict_proba(x)`|각 클래스 확률 반환 (softmax 출력)|
|`model.score(x, y)`|정확도 한 번에 계산|
|`joblib.dump(model, 'file.pkl')`|모델 저장|
|`joblib.load('file.pkl')`|모델 불러오기|
|`np.meshgrid()`|격자 좌표 생성 (결정 경계 시각화)|
|`plt.contourf()`|등고선 배경으로 결정 경계 표현|

### 주의사항

|상황|내용|
|---|---|
|스케일링|train으로만 fit, test는 transform만|
|정확도 주석 오류|오류수 1개 → 정확도 0.9777 (0.6은 잘못된 주석)|
|predict_proba|Perceptron 미지원, MLPClassifier는 지원|
|표준화된 모델로 예측 시|새 데이터도 반드시 sc.transform() 후 예측|

---

# 📄 군집분석 (Clustering Analysis)

---

## 🧠 개념 정리

### 군집분석이란

> 분류 기준이 없는 상태에서 데이터 속성을 고려해 스스로 전체 데이터를 N개의 소그룹으로 묶어내는 **비지도학습** 분석법 유사성이 높은 데이터를 묶고, 서로 다른 그룹 간의 이질성을 계산하는 과정을 거침

![[군집분석.png]]

> 군집화는 **집단 내 동질성**과 **집단 간 이질성**을 기반으로 이뤄짐

---

## 핵심 개념

### 군집분석 종류

| 종류       | 설명                          | 대표 알고리즘     |
| -------- | --------------------------- | ----------- |
| 계층적 군집   | 유사한 개체를 반복적으로 병합/분리         | 병합적, 분할적 방법 |
| 비계층적 군집  | 군집 수를 미리 정하고 중심에 가까운 개체를 포함 | K-Means     |
| 밀도 기반 군집 | 밀도가 높은 영역을 하나의 군집으로 구분      | DBSCAN      |

---

## 계층적 군집 (Hierarchical Clustering)

> 가장 유사한 개체를 묶어 나가는 과정을 반복하여 원하는 개수의 군집 형성 결과는 **덴드로그램(Dendrogram)** 으로 표현 → 군집 간 구조적 관계 파악 가능

![[계층적_군집.png]]

### 병합적 방법 vs 분할적 방법

|방법|설명|
|---|---|
|병합적 (Agglomerative)|작은 군집에서 출발 → 군집을 병합해 나감|
|분할적 (Divisive)|큰 군집에서 출발 → 군집을 분리해 나감|

### 계층적 군집 연결법 종류

![[계층적_군집_연결법_종류.png]]

|연결법|설명|
|---|---|
|최단 연결법 (Single Linkage)|두 군집 간 가장 짧은 거리로 군집 형성|
|최장 연결법 (Complete Linkage)|두 군집 간 가장 긴 거리로 군집 형성|
|중심 연결법 (Centroid Linkage)|두 군집의 중심 간의 거리로 군집 형성|
|평균 연결법 (Average Linkage)|두 군집 간 거리의 평균으로 군집 형성|
|와드 연결법 (Ward Linkage)|군집 내 오차제곱합(ESS)을 최소화하는 방향으로 군집 형성|

---

## 비계층적 군집 — K-Means Clustering

> 사전에 결정된 군집 수 K에 기초하여 전체 데이터를 K개의 군집으로 구분 가장 대표적인 군집 분석 방법

### K-Means 동작 과정

```
1. 클러스터 개수 K값 결정
2. 데이터 공간에 클러스터 중심 K개 임의 할당
3. 각 데이터를 가장 가까운 중심에 배정
4. 각 클러스터 중심을 해당 클러스터 데이터의 평균으로 업데이트
5. 클러스터 중심이 변하지 않을 때까지 3~4 반복
```

---

## 밀도 기반 군집 — DBSCAN

> **밀도 기반**으로 군집 할당 — 반지름 내에 n개 이상의 포인트를 가진 것을 하나의 군집으로 구분 K-Means와 달리 군집 수 K를 미리 정할 필요 없음 (단, 반지름과 최소 포인트 수는 사전 정의 필요)

![[밀도_기반_군집.png]]

### DBSCAN 포인트 종류

|포인트|설명|
|---|---|
|Core Point (코어 포인트)|반지름 내에 최소 n개 이상의 포인트를 가진 점|
|Border Point (보더 포인트)|코어 포인트의 반지름 내에 이웃하지만 최소 개수 미충족|
|Noise Point (노이즈 포인트)|코어 포인트도 보더 포인트도 아닌 점 → 군집에서 제외|

### DBSCAN 특징

> 노이즈 포인트를 제외하기 때문에 **노이즈에 강함** 군집의 모양과 크기가 다양하게 나올 수 있음 (원형이 아니어도 됨) K-Means는 원형 군집에 적합, DBSCAN은 불규칙한 형태에 적합

### 최적 반지름 찾는 법

> 군집 안 데이터 포인트들과 k번째 인접 이웃까지의 거리를 계산 **거리가 급격하게 늘어나는 지점** = 최적 반지름

---

## 📌 핵심 정리

### K-Means vs DBSCAN 비교

|구분|K-Means|DBSCAN|
|---|---|---|
|군집 수|사전에 K 지정 필요|필요 없음|
|사전 지정값|K (군집 수)|반지름, 최소 포인트 수|
|군집 모양|원형에 적합|불규칙한 형태도 가능|
|노이즈 처리|모든 점을 군집에 포함|노이즈 포인트 제외|
|속도|빠름|데이터 많으면 느림|

### 용어 정리

|용어|설명|
|---|---|
|군집 (Cluster)|유사한 데이터를 묶은 소그룹|
|덴드로그램 (Dendrogram)|계층적 군집 결과를 나타내는 트리 형태 시각화|
|ESS (오차제곱합)|군집 내 데이터 간 거리의 제곱합 → 와드 연결법에서 사용|
|코어 포인트|반지름 내 최소 포인트 수를 충족하는 중심 포인트|
|보더 포인트|코어 포인트 반지름 내 이웃하는 경계 포인트|
|노이즈 포인트|어느 군집에도 속하지 않는 이상치 포인트|

---

# 📄 unsuper1cl.py — 계층적 클러스터링 (Hierarchical Clustering)

---

## 🧠 개념 정리

### 군집분석 (Clustering Analysis)이란

> 데이터 간의 유사도를 정의하고 그 유사도에 가까운 것부터 순서대로 합쳐가는 방법 거리나 상관계수 등을 이용해 비슷한 특성을 가진 개체를 그룹으로 만듦 **label을 제공하지 않는 비지도학습** — 스스로 그룹을 찾아냄

### 계층적 클러스터링 방식

|방식|설명|
|---|---|
|응집형 (Agglomerative)|군집의 크기를 점점 늘리기 — 상향식 (Bottom-Up)|
|분리형 (Divisive)|군집의 크기를 점점 줄여나가기 — 하향식 (Top-Down)|

### 주요 함수

|함수|설명|
|---|---|
|`pdist(df, metric)`|모든 점 쌍 간의 거리를 1D 배열로 계산|
|`squareform(dist_vec)`|pdist 결과를 사각형 행렬(거리 행렬)로 변환|
|`linkage(dist_vec, method)`|응집형 계층적 클러스터링 수행|
|`dendrogram(row_clusters)`|클러스터 계층 구조를 계통도로 시각화|

### 덴드로그램 (Dendrogram)

> 계층적 클러스터링 결과를 **트리 형태**로 시각화한 그래프 y축 = 유클리드 거리 → 높을수록 두 군집이 멀리 떨어져 있음을 의미 색상으로 군집 구분 가능

---

## 1단계 : 데이터 생성 및 시각화

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

np.random.seed(123)          # 재현성을 위한 시드 고정
var = ['x', 'y']
labels = ['점0','점1','점2','점3','점4']

x = np.random.random_sample([5, 2]) * 10  # 5개 점, x/y 좌표 랜덤 생성 (0~10 범위)
df = pd.DataFrame(x, columns=var, index=labels)
print(df)
#            x         y
# 점0  0.696469  0.286139
# 점1  0.226851  0.551315
# 점2  0.719469  0.423106
# 점3  0.980764  0.684830
# 점4  0.480932  0.392118

# 산점도 시각화
plt.scatter(x[:, 0], x[:, 1], c='blue', marker='o', s=50)
for i, txt in enumerate(labels):
    plt.text(x[i, 0], x[i, 1], txt)  # 각 점에 레이블 표시
plt.grid(True)
plt.show()
```

![[unsuper1cl.png]]

---

## 2단계 : 점 간 거리 계산

```python
from scipy.spatial.distance import pdist, squareform

# pdist : 모든 점 쌍 간의 유클리드 거리를 1D 배열로 계산
# (5개 점 → 5C2 = 10개 거리값)
dist_vec = pdist(df, metric='euclidean')
print('dist_vec : ', dist_vec)
# [5.3931329  1.38884785 4.89671004 2.40182631 5.09027885 7.6564396
#  2.99834352 3.69830057 2.40541571 5.79234641]

# squareform : 1D 거리 배열 → 사각형 거리 행렬로 변환 (보기 편하게)
row_dist = pd.DataFrame(squareform(dist_vec), columns=labels, index=labels)
print(row_dist)
#           점0        점1        점2        점3        점4
# 점0  0.000000  5.393133  1.388848  4.896710  2.401826
# 점1  5.393133  0.000000  5.090279  7.656440  2.998344
# 점2  1.388848  5.090279  0.000000  3.698301  2.405416
# 점3  4.896710  7.656440  3.698301  0.000000  5.792346
# 점4  2.401826  2.998344  2.405416  5.792346  0.000000
# 대각선 = 0 (자기 자신과의 거리), 값이 작을수록 두 점이 가까움
```

---

## 3단계 : 계층적 클러스터링 수행

```python
from scipy.cluster.hierarchy import linkage

# linkage : 응집형 계층적 클러스터링 수행
# method='ward' : 군집 내 오차제곱합(ESS)을 최소화하는 방향으로 병합
row_clusters = linkage(dist_vec, method='ward')

df2 = pd.DataFrame(
    row_clusters,
    columns=['클러스터id1', '클러스터id2', '거리', '클러스터멤버']
)
print(df2)
#    클러스터id1  클러스터id2        거리  클러스터멤버
# 0      0.0      2.0  1.388848     2.0   ← 점0 + 점2 먼저 병합 (가장 가까움)
# 1      4.0      5.0  2.657109     3.0   ← 점4 + (점0+점2 군집) 병합
# 2      1.0      6.0  5.454004     4.0   ← 점1 + 위 군집 병합
# 3      3.0      7.0  6.647102     5.0   ← 점3 + 전체 병합 (가장 멀음)
```

### linkage 결과 해석

> `클러스터id1`, `클러스터id2` : 병합되는 두 군집의 인덱스 원래 데이터 인덱스(0~4) 이후의 숫자(5, 6, 7...)는 새로 생성된 군집을 의미 `거리` : 두 군집이 병합될 때의 거리 → 클수록 나중에 합쳐진 것 `클러스터멤버` : 병합 후 군집에 속한 데이터 개수

---

## 4단계 : 덴드로그램 시각화

```python
from scipy.cluster.hierarchy import dendrogram

# dendrogram : 클러스터 계층 구조를 계통도로 시각화
row_dend = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('유클리드 거리')  # y축 = 병합 시 두 군집 간 거리
plt.show()
```

![[unsuper1cl2.png]]

> 같은 색 = 같은 군집 y축 값이 낮을수록 일찍 병합 (유사도 높음), 높을수록 나중에 병합 (유사도 낮음) 덴드로그램을 특정 높이에서 자르면 원하는 군집 수로 나눌 수 있음

---

## 📌 핵심 정리

### 전체 흐름

```
랜덤 데이터 5개 생성 (x, y 좌표)
    ↓
pdist() : 모든 점 쌍 간 유클리드 거리 계산
    ↓
squareform() : 1D 거리 배열 → 사각형 거리 행렬 변환
    ↓
linkage(method='ward') : 응집형 계층적 클러스터링 수행
    ↓
dendrogram() : 결과를 계통도로 시각화
```

### 주요 함수 정리

|함수|파라미터|설명|
|---|---|---|
|`pdist(df, metric='euclidean')`|metric : 거리 측정 방식|점 쌍 간 거리 계산|
|`squareform(dist_vec)`|-|1D → 사각형 행렬 변환|
|`linkage(dist_vec, method)`|method : ward, single, complete 등|계층적 클러스터링|
|`dendrogram(row_clusters, labels)`|labels : 축 레이블|계통도 시각화|

### linkage method 종류

|method|설명|
|---|---|
|`ward`|군집 내 오차제곱합 최소화 → 가장 많이 사용|
|`single`|최단 연결법|
|`complete`|최장 연결법|
|`average`|평균 연결법|
|`centroid`|중심 연결법|

### 주의사항

|상황|내용|
|---|---|
|`pdist` 결과|1D 배열 → `squareform`으로 사각형 행렬로 변환해야 보기 편함|
|linkage id|원본 인덱스(0~N-1) 이후 숫자는 새로 생성된 군집 id|
|군집 수 결정|덴드로그램의 특정 높이에서 잘라 원하는 수로 나눌 수 있음|

---
# 📄 unsuper2cl.py — 계층적 군집분석 (학생 점수 데이터)

---

## 🧠 개념 정리

### 이 실습에서 추가된 핵심 함수

|함수|설명|
|---|---|
|`linkage(scores, method)`|응집형 계층적 클러스터링 수행|
|`dendrogram(linked, labels)`|덴드로그램 시각화|
|`fcluster(linked, t, criterion)`|덴드로그램을 잘라 군집 레이블 반환|
|`plt.axhline(y, color, linestyle)`|덴드로그램에 수평 절단선 추가|

### fcluster 파라미터

|파라미터|설명|
|---|---|
|`t`|군집 수 또는 절단 거리 기준값|
|`criterion='maxclust'`|t를 군집 수로 해석 → 최대 t개의 군집으로 나눔|
|`criterion='distance'`|t를 거리 기준으로 해석 → 거리 t 이하로 묶음|

### 덴드로그램 절단선 (axhline)

> 덴드로그램에 수평선을 그어 군집 수를 시각적으로 표현 `y=25` 위치에서 자르면 → 선 아래 색깔별 군집 수만큼 분리됨

---

## 1단계 : 데이터 준비

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

students = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
scores = np.array([76,95,65,85,60,92,55,88,83,72]).reshape(-1, 1)
# reshape(-1, 1) : 1D 배열 → 2D 열벡터로 변환 (linkage 입력 형태 맞추기)
print('점수 : ', scores)
```

---

## 2단계 : 계층적 클러스터링 + 덴드로그램

```python
# ward 연결법 : 군집 내 오차제곱합(ESS)을 최소화하는 방향으로 병합
linked = linkage(scores, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked, labels=students)
plt.title('Student score')
plt.xlabel('Students')
plt.ylabel('Distance')

# 덴드로그램에 절단선 추가 → y=25에서 자르면 3개 군집으로 나뉨
plt.axhline(y=25, color='red', linestyle='--', label='cut at 25')
plt.legend()
plt.grid(True)
plt.show()
```

![[unsuper2cl.png]]

> 빨간 점선(y=25) 아래에서 색깔별로 3개 군집 확인 가능 y축이 낮을수록 먼저 병합(유사도 높음), 높을수록 나중에 병합(유사도 낮음)

---

## 3단계 : 군집 레이블 추출

```python
# fcluster : 덴드로그램을 잘라 각 데이터에 군집 레이블 부여
# criterion='maxclust' : 최대 3개 군집으로 나누기
clusters = fcluster(linked, t=3, criterion='maxclust')

print('학생자료 군집결과')
for stu, cluster in zip(students, clusters):
    print(f"{stu} : cluster {cluster}")
# s1 : cluster 2   ← 중간 그룹
# s2 : cluster 1   ← 상위 그룹
# s3 : cluster 3   ← 하위 그룹
# s4 : cluster 1
# s5 : cluster 3
# s6 : cluster 1
# s7 : cluster 3
# s8 : cluster 1
# s9 : cluster 1
# s10 : cluster 2
```

---

## 4단계 : 군집별 통계 분석

```python
# 군집별 학생 이름과 점수를 딕셔너리로 정리
cluster_info = {}
for student, cluster, score in zip(students, clusters, scores.flatten()):
    if cluster not in cluster_info:
        cluster_info[cluster] = {"students": [], "scores": []}
    cluster_info[cluster]["students"].append(student)
    cluster_info[cluster]["scores"].append(score)

# 군집별 평균 점수 출력
print('군집별로 평균점수와 이름 출력')
for cluster_id, info in sorted(cluster_info.items()):
    avg_score = np.mean(info["scores"])
    student_list = ", ".join(info["students"])
    print(f"Cluster {cluster_id}: 평균 점수={avg_score:.2f}, 학생들={student_list}")
# Cluster 1: 평균 점수=88.60, 학생들=s2, s4, s6, s8, s9  ← 상위권
# Cluster 2: 평균 점수=74.00, 학생들=s1, s10              ← 중위권
# Cluster 3: 평균 점수=60.00, 학생들=s3, s5, s7           ← 하위권
```

---

## 5단계 : 군집별 Scatter Plot

```python
x_positions = np.arange(len(students))
y_scores = scores.ravel()           # 2D → 1D 변환
colors = {1:'red', 2:'blue', 3:'green'}  # 군집별 색상 지정

plt.figure(figsize=(10, 6))
for i, (x, y, cluster) in enumerate(zip(x_positions, y_scores, clusters)):
    plt.scatter(x, y, color=colors[cluster], s=100)   # 군집 색으로 점 표시
    plt.text(x, y + 1.5, students[i], fontsize=12, ha='center')  # 이름 표시

plt.xticks(x_positions, students)
plt.xlabel('Students')
plt.ylabel('Score')
plt.title('Score cluster')
plt.grid(True)
plt.show()
```

![[unsuper2cl2.png]]

> 빨강(Cluster 1) = 상위권 (평균 88.6), 파랑(Cluster 2) = 중위권 (평균 74), 초록(Cluster 3) = 하위권 (평균 60)

---

## 📌 핵심 정리

### 전체 흐름

```
학생 10명 점수 데이터 준비
    ↓
linkage(method='ward') : 계층적 클러스터링
    ↓
dendrogram + axhline(y=25) : 절단선으로 군집 수 시각화
    ↓
fcluster(t=3, criterion='maxclust') : 3개 군집으로 레이블 추출
    ↓
군집별 평균 점수 / 학생 이름 분석
    ↓
Scatter plot으로 군집 시각화
```

### 군집 결과 요약

|군집|색상|학생|평균 점수|
|---|---|---|---|
|Cluster 1|빨강|s2, s4, s6, s8, s9|88.60|
|Cluster 2|파랑|s1, s10|74.00|
|Cluster 3|초록|s3, s5, s7|60.00|

### 주요 함수 정리

|함수|설명|
|---|---|
|`linkage(scores, method='ward')`|계층적 클러스터링 수행|
|`dendrogram(linked, labels)`|덴드로그램 시각화|
|`fcluster(linked, t=3, criterion='maxclust')`|군집 레이블 배열 반환|
|`plt.axhline(y=25, color, linestyle)`|덴드로그램 절단선 추가|
|`scores.flatten()` / `scores.ravel()`|2D → 1D 배열 변환|

### 활용 분야

> 성적 그룹 분석, 고객 등급 분류, 사용자 행동 패턴 군집화 등에 활용 가능