# Day 44_규제 회귀 & 로지스틱 회귀

## 📅 2026-04-07

---

## 🔑 핵심 개념 - 과적합(Overfitting)이란?

> 모델이 훈련 데이터에 너무 맞춰져서 새로운 데이터에는 예측을 못하는 현상 → 이를 방지하기 위해 **규제(Regularization)** 를 사용

---

## 🔑 규제 회귀 모델 비교

|모델|규제 방식|특징|
|---|---|---|
|**LinearRegression**|없음|기본 선형회귀|
|**Ridge**|L2 (가중치 **제곱합** 최소화)|다중공선성 처리에 효과적. 계수를 0에 가깝게 줄임|
|**Lasso**|L1 (가중치 **절대값의 합** 최소화)|영향력 적은 변수를 **아예 0으로** 만듦 → 변수 선택 기능|
|**ElasticNet**|L1 + L2 동시 적용|Ridge + Lasso 혼합. 두 규제의 장점을 모두 가짐|

> **alpha** : 규제 강도. 값이 클수록 규제가 강해짐 (계수가 더 많이 줄어듦)

---

## 📄 ex14linear_regu.py - Ridge / Lasso / ElasticNet (iris 데이터)

### 💡 개념

- **train/test 분리** : 학습은 train으로, 평가는 test로 → 과적합 방지
- **규제 회귀** : LinearRegression에 제약조건을 추가해 오버피팅을 방지

### 데이터 준비 및 분리

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error

iris = load_iris()
print(iris.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target
iris_df["target_names"] = iris.target_names[iris.target]
print(iris_df[:3])

# train dataset, test dataset으로 나누기
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(iris_df, test_size=0.3, random_state=12)
# test_size=0.3 → 전체 데이터의 30%를 테스트용으로 분리
```

### ① LinearRegression (기본)

```python
from sklearn.linear_model import LinearRegression
# 학습은 train dataset 으로 작업
model_linear = LinearRegression().fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])
print('slope : ', model_linear.coef_)      # 0.42259168
print('bias : ', model_linear.intercept_)  # -0.39917733

# 모델 평가는 test dataset 으로 작업
pred = model_linear.predict(test_set.iloc[:, [2]])
print('예측값 : ', np.round(pred[:5].flatten(), 1))
print('실제값 : ', test_set.iloc[:, [3]][:5].values.flatten())

from sklearn.metrics import r2_score
print('r2_score(결정계수):{}'.format(r2_score(test_set.iloc[:, [3]], pred)))  # 0.93833

plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]], color='red')
plt.plot(np.array(test_set.iloc[:, [2]]), model_linear.predict(test_set.iloc[:, [2]]))
plt.show()
```

### ② Ridge (L2 규제)

> **개념** : 가중치 제곱합을 최소화. 다중공선성 문제 처리에 효과적. 계수를 완전히 0으로 만들지는 않음

```python
from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=10).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])

print(model_ridge.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]]))  # 0.91880
print(model_ridge.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))    # 0.94101
pred_ridge = model_ridge.predict(test_set.iloc[:, [2]])
print('ridge predict : ', pred_ridge[:5])
print('r2_score(결정계수):{}'.format(r2_score(test_set.iloc[:, [3]], pred_ridge)))  # 0.9410

plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]], color='blue')
plt.plot(np.array(test_set.iloc[:, [2]]), model_ridge.predict(test_set.iloc[:, [2]]))
plt.show()
```

### ③ Lasso (L1 규제)

> **개념** : 가중치 절대값의 합을 최소화. 영향력 적은 변수를 **완전히 0으로** 만들어 변수 선택 기능 제공

```python
from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=0.1).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])

print(model_lasso.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]]))  # 0.913863
print(model_lasso.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))    # 0.940663
pred_lasso = model_lasso.predict(test_set.iloc[:, [2]])
print('lasso predict : ', pred_lasso[:5])
print('r2_score(결정계수):{}'.format(r2_score(test_set.iloc[:, [3]], pred_lasso)))

plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]], color='green')
plt.plot(np.array(test_set.iloc[:, [2]]), model_lasso.predict(test_set.iloc[:, [2]]))
plt.show()
```

### ④ ElasticNet (L1 + L2 혼합)

> **개념** : Ridge + Lasso의 형태. 가중치 절대값의 합(L1)과 제곱합(L2)을 동시에 제약 조건으로 가지는 모형

```python
from sklearn.linear_model import ElasticNet
model_elastic = ElasticNet(alpha=0.1).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])

print(model_elastic.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]]))  # 0.913863
print(model_elastic.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))    # 0.940663
pred_elastic = model_elastic.predict(test_set.iloc[:, [2]])
print('ElasticNet predict : ', pred_elastic[:5])
print('r2_score(결정계수):{}'.format(r2_score(test_set.iloc[:, [3]], pred_elastic)))

plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]], color='cyan')
plt.plot(np.array(test_set.iloc[:, [2]]), model_elastic.predict(test_set.iloc[:, [2]]))
plt.show()
```

---

## 📄 ex15nonlinear.py - 다항회귀 / 비선형회귀 (Boston 주택가격 데이터)

### 💡 개념

- **비선형회귀** : 직선이 아닌 곡선으로 데이터를 설명. 선형 가정이 맞지 않을 때 사용
- **PolynomialFeatures** : 입력 변수에 다항식 항을 추가해서 곡선 형태로 만듦
- **degree** : 차수. degree=2 → 2차 곡선, degree=3 → 3차 곡선

```
degree=1 : y = ax + b          (직선)
degree=2 : y = ax² + bx + c   (포물선)
degree=3 : y = ax³ + bx² + cx + d  (3차 곡선)
```

> 차수가 높을수록 R²는 올라가지만 과적합 위험도 증가!

### Boston 데이터 컬럼 설명

|컬럼|설명|
|---|---|
|CRIM|자치시별 1인당 범죄율|
|RM|주택 1가구당 평균 방의 개수|
|LSTAT|모집단의 하위계층 비율(%)|
|MEDV|주택가격 중앙값 (단위: $1,000) ← **종속변수**|

### 데이터 준비 및 다항 특성 변환

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/housing.data", header=None, sep=r'\s+')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

print(df.corr())    # LSTAT(하층비율), MEDV(집값) : -0.737663

x = df[['LSTAT']].values  # 독립변수 : 하위계층 비율
y = df['MEDV'].values      # 종속변수 : 주택가격

# 다항 특성
quad = PolynomialFeatures(degree=2)   # 2차
cubic = PolynomialFeatures(degree=3)  # 3차
x_quad = quad.fit_transform(x)
x_cubic = cubic.fit_transform(x)
print(x_quad[:3])
# [[ 1.      4.98   24.8004]      ← [1, x, x²]
print(x_cubic[:3])
# [[  1.         4.98      24.8004   123.505992]   ← [1, x, x², x³]
```

### 단순/2차/3차 회귀 비교

```python
model = LinearRegression()

# 단순회귀 (1차)
model.fit(x, y)
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]
y_lin_fit = model.predict(x_fit)
model_r2 = r2_score(y, model.predict(x))
print('model_r2 : ', model_r2)  # 0.5441462975864797

# 2차 회귀
model.fit(x_quad, y)
y_quad_fit = model.predict(quad.fit_transform(x_fit))
q_r2 = r2_score(y, model.predict(x_quad))
print('q_r2 : ', q_r2)  # 0.6407168971636612

# 3차 회귀
model.fit(x_cubic, y)
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))
c_r2 = r2_score(y, model.predict(x_cubic))
print('c_r2 : ', c_r2)  # 0.6578476405895719

plt.scatter(x, y, label='초기 데이터')
plt.plot(x_fit, y_lin_fit, linestyle=':', label='linear fit(d=1), $R^2=%.2f$'%model_r2, c='b', lw=3)
plt.plot(x_fit, y_quad_fit, linestyle='-', label='quad fit(d=2), $R^2=%.2f$'%q_r2, c='r', lw=3)
plt.plot(x_fit, y_cubic_fit, linestyle='--', label='cubic fit(d=3), $R^2=%.2f$'%c_r2, c='k', lw=3)
plt.xlabel("하위계층 비율")
plt.ylabel("주택가격")
plt.legend()
plt.show()
```

> R² 비교 : 1차(0.54) < 2차(0.64) < 3차(0.66) → 차수가 높을수록 설명력 증가

---

## 📄 ex16nonlinear.py - 비선형회귀 (간단한 예제)

### 💡 개념

- 상관계수가 낮아도 **비선형 관계**일 수 있음
- `PolynomialFeatures(degree=2, include_bias=False)` 로 x² 항 추가
- 선형모델로 만든 특징 행렬에 LinearRegression 적용 → 곡선 피팅

```python
# 비선형회귀분석(Non-linear regression)
# 직선의 회귀선을 곡선으로 변환해 보다 더 정확하게 데이터 변화를 예측하는 데 목적이 있다.
# 선형가정이 어긋날 때(비정규성) 대처할 수 있는 방법으로 다항식 항을 추가한 다항회귀모델을 사용함
# 입력 데이터 특징 변환으로  선형모델을 개선

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])

print(np.corrcoef(x, y)[0, 1])  # 0.48076197382041164

# 선형회귀모델 적용
x = x[:, np.newaxis]    # 차원 확대. 1차원을 2차원으로 변환
model = LinearRegression().fit(x, y)
ypred = model.predict(x)
print('예측값 y : ', ypred)              # [2.  2.7 3.4 4.1 4.8]
print('결정계수 : ', r2_score(y, ypred)) # 0.23113207547169834  ← 낮음!

# 비선형 모델 작성
# 여러 방법 중 가장 일반적인 방법을 사용(PolynomialFeatures, log변환, curve_fit ...)
poly = PolynomialFeatures(degree=2, include_bias=False)     # degree=열수
x2 = poly.fit_transform(x)  # 특징 행렬을 만듦
print(x2)
# [[ 1.  1.]
#  [ 2.  4.]
#  [ 3.  9.]
#  [ 4. 16.]
#  [ 5. 25.]]  ← x와 x² 열이 추가됨

model2 = LinearRegression().fit(x2, y)  # 특징 행렬값으로 모델 생성
ypred2 = model2.predict(x2)
print('예측값 y : ', ypred2)            # [4.14285714 1.62857143 1.25714286 3.02857143 6.94285714]
print('결정계수 : ', r2_score(y, ypred2))  # 0.9892183288409704  ← 크게 향상!
```

> 선형회귀 R²: **0.23** → 다항회귀(degree=2) R²: **0.99** 로 대폭 향상!

---

## 📝 L1 / L2 규제 수식 정리

> 머신러닝/통계학에서 말하는 L1 규제, L2 규제는 바로 **회귀계수(모델 파라미터)에 대한 제약 조건**을 말한다. 이는 **정규화(Regularization)** 기법의 핵심으로, 오버피팅을 방지하고 모델의 일반화 성능을 높이는 데 사용된다.

#### L1 규제 (Lasso)

$$\min_{\beta} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p}|\beta_j|$$

- 벌점(패널티)이 **회귀계수의 절댓값(|β|)의 합**
- 효과 : 일부 계수를 **0으로 수축** → 변수 선택(Feature selection) 효과
- 불필요한 변수를 자동으로 제거 → **희소(sparse) 모델** 생성
- 단점 : 강한 다중공선성이 있을 경우, 변수 선택이 불안정할 수 있음

#### L2 규제 (Ridge)

$$\min_{\beta} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p}\beta_j^2$$

- 벌점이 **회귀계수 제곱합(β²)**
- 효과 : 큰 계수를 강하게 줄임 → **다중공선성 완화**
- 모든 변수를 조금씩 줄임 (0은 잘 안됨)
- 단점 : 변수 선택은 하지 않음 → 해석력이 낮을 수 있음

#### Elastic Net (혼합)

$$\min_{\beta} \sum(y - \hat{y})^2 + \lambda_1 \sum|\beta| + \lambda_2 \sum\beta^2$$

- L1 + L2 규제를 결합. Lasso의 **변수 선택** + Ridge의 **안정성** 절충
- **고차원 데이터**에서 특히 유용

### 규제 회귀 한눈에 비교

|구분|Ridge|Lasso|Elastic Net|
|---|---|---|---|
|규제 방식|L2|L1|L1 + L2|
|계수 축소|0에 **가깝게**|**완전히 0 가능**|둘 다|
|변수 선택|❌|✅|✅|
|다중공선성|잘 처리|불안정 가능|잘 처리|
|언제 쓰나|변수 간 상관 높을 때|불필요 변수 제거 필요|둘 다 문제일 때|

### PolynomialFeatures 정리

```python
PolynomialFeatures(degree=2)                      # [1, x, x²] 생성
PolynomialFeatures(degree=2, include_bias=False)  # [x, x²] 생성 (1 제외)
```

> 차수가 높을수록 R²↑ 하지만 과적합 위험도↑ → 적절한 degree 선택이 중요!

---

# 로지스틱 회귀 (Logistic Regression)

---

## 🔑 기계학습 방법 전체 분류

|학습 방법|가능 분야|주요 알고리즘|
|---|---|---|
|**지도학습**|예측·추정 (정량적)|Linear Regression, SVM, Neural Network, ARIMA|
|**지도학습**|분류 (정성적)|**Logistic Regression**, Decision Tree, k-NN, Random Forest, Naive Bayes|
|**비지도학습**|패턴/구조 발견|Association Rule, Network Analysis|
|**비지도학습**|그룹화|k-Means, Hierarchical Clustering, SOM|
|**비지도학습**|차원 축소|PCA, Factor Analysis, SVD|

> 전통적인 방법 중 실무(캐글 등)에서 제일 많이 쓰는 건 **Random Forest**

---

## 🔑 지도학습 - 회귀 vs 분류

|구분|출력값|예시|
|---|---|---|
|**회귀 (Regression)**|연속적인 숫자|집값, 연봉, 판매량|
|**분류 (Classification)**|범주 (0 or 1)|스팸 여부, 질병 유무|

> 로지스틱 회귀는 이름에 "회귀"가 붙지만 실제로는 **분류** 알고리즘! `y = Wx + b` 선형 모델을 사용하지만, 시그모이드 함수로 확률(0~1)로 변환해서 분류함

---

## 📄 로지스틱 함수 유도 과정

### 1단계 - Odds (승산비)

$$Odds = \frac{\mu}{1-\mu}$$

- μ가 0~1 사이의 값 → Odds는 0~∞ 범위로 확장됨
- 예) 성공확률 80%, 실패확률 20% → Odds = 0.8/0.2 = **4** (성공이 실패보다 4배 높음)

|확률(p)|0%|10%|50%|90%|100%|
|---|---|---|---|---|---|
|Odds|0|0.11|1|9|∞|

### 2단계 - Logit 함수 (로그 변환)

$$z = logit = \log\left(\frac{\mu}{1-\mu}\right)$$

|원래 비율(p)|0|0.5|1근사|
|---|---|---|---|
|Odds P/(1-P)|0|1|∞|
|로그 오즈비 log(오즈)|-∞|0|∞|

### 3단계 - Logistic 함수 (로짓의 역함수)

$$\sigma(z) = \frac{1}{1+e^{-z}} = \frac{e^z}{e^z+1}$$

- x → -∞ 이면 **0**에 가까워짐 / x → +∞ 이면 **1**에 가까워짐 / x = 0 이면 **0.5**

> **로지스틱 함수는 로짓 함수의 역함수!** 로지스틱 회귀는 선형 모델 출력을 확률로 바꾸기 위해 로지스틱 함수를 사용하고, 그 확률을 다시 log-odds로 해석할 때는 로짓 함수를 쓴다.

---

## 📄 로지스틱 회귀 전체 흐름

### 핵심 공식

$$P(y=1|x) = \sigma(wx+b) = \frac{1}{1+e^{-(wx+b)}}$$

```
wx + b 값은  -∞ ~ +∞  범위 → 시그모이드로 0~1 확률 변환
확률 ≥ 0.5  →  클래스 1 (양성)
확률 < 0.5  →  클래스 0 (음성)
```

---

## 📄 확률과 가능도 (Likelihood)

### 선형회귀 vs 로지스틱 회귀 최적화 방식

| |선형회귀|로지스틱 회귀|
|---|---|---|
|최적화 방법|**최소제곱법** (MSE 최소화)|**최대 가능도 추정 (MLE)**|
|목표|오차 줄이기|맞을 확률 높이기|

### 단계별 학습 흐름

|단계|설명|
|---|---|
|①|`wx + b` → 시그모이드 함수 → **확률 출력**|
|②|확률을 가지고 각 샘플의 **가능도** 계산|
|③|전체 데이터에 대한 **가능도 함수** 정의|
|④|계산 편의를 위해 **로그 가능도** 사용|
|⑤|로그 가능도의 부호를 반전시켜 **손실 함수(Log Loss)** 정의|
|⑥|이 손실을 **최소화**하여 최적의 w, b를 학습|

### 손실 함수 (Binary Cross Entropy)

$$\mathcal{L}(w,b) = -\sum_{i=1}^{n}\left[y_i \cdot \log\left(\sigma(wx_i+b)\right) + (1-y_i) \cdot \log\left(1-\sigma(wx_i+b)\right)\right]$$

### 예제로 이해하기

```
입력값 x=2,  정답 y=1,  가중치 w=1.5,  편향 b=-1

① 선형 조합   : z = wx + b = 1.5×2 + (-1) = 2
② 시그모이드  : σ(2) = 1/(1+e⁻²) ≈ 0.8808  → 확률 88%
③ 가능도      : P(y=1|x) = 0.8808
④ 로그 가능도 : log(0.8808) ≈ -0.1269
⑤ 손실 함수   : Loss = -log(0.8808) ≈ 0.1269

→ 손실값이 작을수록 예측이 정답과 가까워짐!
```

---

## 📄 ex17sigmoid.py - 시그모이드 함수 직접 구현

### 💡 개념

- `wx + b` 자체는 **logit한 값** : `log(p / (1-p)) = wx + b` 로 정의됨
- `z = wx + b` → `sigmoid(z)` → `p (0~1 확률값)`

```python
# sigmoid function 적용 연습
# 로지스틱회귀에서는 wx + b 자체는 logit한 값이다. log(p / (1 - p)) = wx + b 라고 정의됨
# 그러므로 z = wx + b --> sigmoid(z) --> p(0 ~ 1)

import math
def sigmoidFunc(num):
    return 1 / (1 + math.exp(-num))

print(sigmoidFunc(3))    # 0.9525741268224334
print(sigmoidFunc(1))    # 0.7310585786300049
print(sigmoidFunc(-5))   # 0.0066928509242848554
print(sigmoidFunc(-10))  # 4.5397868702434395e-05
```

> z값이 클수록 → 1에 가까운 확률, 작을수록 → 0에 가까운 확률

### NumPy로 시그모이드 + 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

x = np.linspace(-10, 10, 50)    # 입력 자료 (연속형)

# 선형결합(이미 logit값)
w = 1.5
b = -2
z = w * x + b   # logit 값

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

p = sigmoid(z)  # 확률값 얻음

# 일부 값 보기
print("x[:3] : ", np.round(x[:3], 3))
print("z[:3](logit) : ", np.round(z[:3], 3))
print("p[:3](확률):", p[:3])
# x[:3] :  [-10.     -9.592  -9.184]
# z[:3](logit) :  [-17.    -16.388 -15.776]
# p[:3](확률): [4.13993755e-08 7.63639448e-08 1.40858451e-07]

# 시각화
plt.figure(figsize=(8, 5))
plt.plot(x, p, label='sigmoid(z)', color='blue')
plt.axhline(0.5, color='red', linestyle="--")
plt.title("z = wx + b --> sigmoid --> 확률")
plt.xlabel("x(입력값)")
plt.ylabel("p(확률값)")
plt.grid(True)
plt.legend()
plt.show()
```

---

## 📄 ex18mtcar.py - 로지스틱 회귀 분류 (mtcars 데이터)

### 💡 개념

- **logit()** : 로지스틱 회귀 전용 변환 함수
- **glm()** : 일반화된 선형모델 → logit()을 포함한 전체 모델
- **Confusion matrix(혼돈행렬)** : 예측값 vs 실제값 집계표
- **accuracy_score** : 분류 정확도 계산

### 데이터 준비

```python
# LogisticRegresion(로지스틱 회귀분석)
# 선형결합을 로그오즈(logit())로 해석하고, 이를 시그모이드 함수를 통해 확률로 변환!
# 이항분류(다항도 가능), 독립변수:연속형, 종속변수:범주형
# LogisticRegresion을 근거로 뉴럴넷의 뉴련에서 사용함

import statsmodels.api as sm

mtcars = sm.datasets.get_rdataset('mtcars').data

# 연비와 마력수에 따른 변속기 분류(수동=1, 자동=0)
mtcar = mtcars.loc[:, ['mpg','hp','am']]
print(mtcar['am'].unique())  # [1(수동) 0(자동)]
```

### 방법1 - logit()

```python
import numpy as np
import statsmodels.formula.api as smf

formula = 'am ~ hp + mpg'   # '종속변수 ~ 독립변수 + ...'
result = smf.logit(formula=formula, data=mtcar).fit()
print(result.summary())
#                            Logit Regression Results
# ==============================================================================
# Dep. Variable:                     am   No. Observations:                   32
# Pseudo R-squ.:                  0.5551   Log-Likelihood:                -9.6163
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    -33.6052     15.077     -2.229      0.026     -63.156      -4.055
# hp             0.0550      0.027      2.045      0.041       0.002       0.108
# mpg            1.2596      0.567      2.220      0.026       0.147       2.372
# ==============================================================================

pred = result.predict(mtcar[:10])
print('예측값 : ', np.around(pred.values))
# 예측값 :  [0. 0. 1. 0. 0. 0. 0. 1. 1. 0.]
print('실제값 : ', mtcar['am'][:10].values)
# 실제값 :  [1 1 1 0 0 0 0 0 0 0]
```

### ✅ 혼동행렬 (Confusion Matrix)

> **"예측이 얼마나 맞고 틀렸는지 한눈에 보는 표"**

```
              예측값
           0 (음성)  1 (양성)
실제값 0    TN        FP
      1    FN        TP
```

| 이름     | 풀네임            | 의미                 |
| ------ | -------------- | ------------------ |
| **TN** | True Negative  | 실제 0, 예측 0 → 정답 ✅  |
| **TP** | True Positive  | 실제 1, 예측 1 → 정답 ✅  |
| **FP** | False Positive | 실제 0인데 예측 1 → 오탐 ❌ |
| **FN** | False Negative | 실제 1인데 예측 0 → 미탐 ❌ |

> **T/F** = 예측이 맞았냐(True) 틀렸냐(False) **P/N** = 예측값이 1(Positive)이냐 0(Negative)이냐

```python
print('수치에 대한 집계표(Confusion matrix, 혼돈행렬) 확인 ---')
conf_tab = result.pred_table()
print(conf_tab)
# [[16.  3.]
#  [ 3. 10.]]
# TN=16 : 실제 자동(0), 예측 자동(0) → 정답
# TP=10 : 실제 수동(1), 예측 수동(1) → 정답
# FP=3  : 실제 자동(0)인데 수동(1)으로 예측 → 오답
# FN=3  : 실제 수동(1)인데 자동(0)으로 예측 → 오답

print('분류 정확도 : ', (16 + 10) / len(mtcar))     # 0.8125
print('분류 정확도 : ', (conf_tab[0][0] + conf_tab[1][1]) / len(mtcar))  # 0.8125

from sklearn.metrics import accuracy_score
pred2 = result.predict(mtcar)
print('분류 정확도 : ', accuracy_score(mtcar['am'], np.around(pred2)))  # 0.8125
```

### 주요 성능 지표

```
정확도 (Accuracy)  = (TN + TP) / 전체     → 전체 중 맞은 비율
정밀도 (Precision) = TP / (TP + FP)       → 1이라고 예측한 것 중 진짜 1
재현율 (Recall)    = TP / (TP + FN)       → 실제 1 중에서 맞게 잡은 비율
```

|상황|중요한 지표|이유|
|---|---|---|
|암 진단|**재현율**|실제 암 환자를 놓치면 안 됨 (FN 최소화)|
|스팸 필터|**정밀도**|중요한 메일을 스팸으로 잘못 분류하면 안 됨 (FP 최소화)|

### 방법2 - glm() (일반화 선형모델)

```python
print('*' * 10)
# glm() - 일반화된 선형모델
result2 = smf.glm(formula=formula, data=mtcar, family=sm.families.Binomial()).fit()
# Binomial() : 이항분포, Gaucian : 정규분포 - 기본값
# → logit()과 결과 동일 (계수, 정확도 모두 같음)

glm_pred2 = result2.predict(mtcar)
print('glm 모델 분류 정확도:', accuracy_score(mtcar['am'], np.around(glm_pred2)))  # 0.8125

# 새로운 값으로 분류
import pandas as pd
newdf = pd.DataFrame()
newdf['mpg'] = [10, 30, 120, 200]
newdf['hp'] = [100, 110, 80, 130]

new_pred = result2.predict(newdf)
print('예측 결과 : ', np.around(new_pred.values))   # [0. 1. 1. 1.]
print('예측 결과 : ', np.rint(new_pred.values))     # [0. 1. 1. 1.]
# np.around()와 np.rint() 둘 다 반올림으로 0/1 분류
```

---

## 📄 ex18quiz.py - 로지스틱 회귀 실습 (외식 분류)

### 💡 문제

> 소득 수준에 따른 외식 성향 분류. 주말 저녁 외식=1, 외식 안 함=0. 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력

```python
# [로지스틱 분류분석 문제1]
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score

data = [
    ['토',0,57], ['토',0,39], ['토',0,28], ['화',1,60], ['토',0,31],
    ['월',1,42], ['토',1,54], ['토',1,65], ['토',0,45], ['토',1,98],
    ['토',1,60], ['토',0,41], ['토',1,52], ['일',1,75], ['월',1,45],
    ['화',0,46], ['수',0,39], ['목',1,70], ['금',1,44], ['토',1,74],
    ['토',1,65], ['토',0,46], ['토',0,39], ['일',1,60], ['토',1,44],
    ['일',0,30], ['토',0,34]
]

df = pd.DataFrame(data, columns=['요일','외식유무','소득수준'])

# 주말(토,일)만 필터링
weekend_df = df[df['요일'].isin(['토','일'])]

# logit 모델
formula = '외식유무 ~ 소득수준'
result = smf.logit(formula=formula, data=weekend_df).fit()
print(result.summary())

pred = result.predict(weekend_df)
print('예측값 : ', np.around(pred.values))
print('실제값 : ', weekend_df['외식유무'].values)

# Confusion matrix
conf_tab = result.pred_table()
print(conf_tab)
print('분류 정확도 : ', (conf_tab[0][0] + conf_tab[1][1]) / len(weekend_df))

# glm 모델
result2 = smf.glm(formula=formula, data=weekend_df, family=sm.families.Binomial()).fit()
glm_pred = result2.predict(weekend_df)
print('glm 모델 분류 정확도:', accuracy_score(weekend_df['외식유무'], np.around(glm_pred)))

# 사용자 입력으로 예측
num = int(input("소득 수준 입력:"))
new_df = pd.DataFrame({'소득수준': [num]})

new_pred = result.predict(new_df)
print('예측 결과(logit) : ', np.around(new_pred.values))

new_pred2 = result2.predict(new_df)
print('예측 결과(glm)   : ', np.around(new_pred2.values))
```

---

## 📄 신경망과의 관계

> 신경망에서 하나의 뉴런(노드)에 활성화 함수로 **시그모이드를 쓰면 이항 로지스틱 회귀와 같다** → 이항 로지스틱 회귀는 신경망, 나아가 **딥러닝과 깊은 관련**이 있다

|뉴런 구성요소|설명|
|---|---|
|axon (축삭돌기)|몸체에서 뻗어나와 다른 뉴런의 수상돌기와 연결|
|dendrite (수상돌기)|다른 뉴런의 축삭돌기와 연결, 나뭇가지 형태|
|synapse (시냅스)|축삭돌기와 수상돌기의 연결 지점. 신호 전달|

---

## 📝 핵심 요약

### logit() vs glm() 비교

| |logit()|glm()|
|---|---|---|
|역할|로지스틱 회귀 전용|일반화 선형모델 (logit 포함)|
|family 지정|불필요|`Binomial()` 지정 필요|
|결과|동일|동일|
|언제 쓰나|간단하게 로지스틱|다양한 분포 적용 가능|

### 로지스틱 회귀 전체 흐름

```
입력 x
  ↓
wx + b  (선형 결합 = logit값)
  ↓
σ(wx+b)  (시그모이드 → 확률 0~1)
  ↓
확률 ≥ 0.5 → 클래스 1
확률 < 0.5 → 클래스 0
```