# Day 42_선형회귀분석

## 📅 2026-04-03

---

# 📊 선형회귀분석 (Linear Regression)

**날짜** : 2026-04-03 
**태그** : #머신러닝 #지도학습 #회귀분석 #파이썬 #statsmodels #sklearn

---

## 🔷 개요

> 전통적 방법의 선형회귀는 **기계학습 중 지도학습**에 해당한다. 각 데이터에 대한 **잔차제곱합(RSS)이 최소**가 되는 추세선(회귀선)을 찾아, 독립변수가 종속변수에 얼마나 영향을 주는지 **인과관계를 분석**하고 **정량적인 모델**을 생성한다.

|구분|조건|
|---|---|
|독립변수 (X)|연속형|
|종속변수 (Y)|연속형|
|전제 조건|두 변수 간 **상관관계 및 인과관계**가 있어야 함|

---

## 🔷 핵심 개념

### ① 잔차 (Residual)

- **잔차** = 실제값(y) - 예측값(ŷ)
- 회귀선은 모든 잔차의 제곱합(RSS)이 최소가 되도록 결정됨 → **최소제곱법(OLS)**

### ② SST / SSR / SSE

|기호|이름|의미|화살표 색|
|---|---|---|---|
|SST|Total Sum of Squares|전체 변동 (실제값 - 평균)|🔴 빨강|
|SSR|Regression Sum of Squares|회귀선이 설명한 변동 (예측값 - 평균)|🟢 초록|
|SSE|Error Sum of Squares|설명 못한 오차 (실제값 - 예측값)|🟠 주황|

### ③ SST / SSR / SSE 시각적 이해

![[SST_SSR_SSE_diagram.png]]

- **회귀선(Best-fit Line)** : 잔차제곱합(SSE)이 최소가 되도록 그어진 선
- **평균선(Mean Line)** : 빨간 점선, 기준값 ȳ

$$SST = SSR + SSE$$

$$R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$$

> 💡 SSE가 작을수록 → R²가 1에 가까워짐 → 회귀선이 데이터를 잘 설명함

---

### ④ R-squared (결정계수, 설명력)

$$R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$$

- 독립변수(X)가 종속변수(Y)를 **얼마나 설명하는가**를 나타내는 수치
- **0 ~ 1** 사이 값, 1에 가까울수록 좋음
- 단, **1에 너무 근사하면 과적합(Overfitting)** 가능성 ⚠️

### ④ Adj. R-squared (수정된 결정계수)

- 독립변수가 많아질수록 R²는 자동으로 커지는 문제를 **보정**한 지표
- **다중회귀**에서 모델 비교 시 R² 대신 사용해야 함

### ⑤ t-value & p-value

|지표|의미|
|---|---|
|t-value|회귀계수가 0인지 검정하는 통계량. 클수록 유의|
|p-value|해당 회귀계수가 통계적으로 유의미한지 판단|

- **p-value < 0.05** → 통계적으로 유의미한 변수 ✅
- **p-value ≥ 0.05** → 유의하지 않음 ❌ → 모델에서 제외 고려

### ⑥ 과적합 (Overfitting)

- 학습 데이터에는 너무 잘 맞지만, **새로운 데이터에는 맞지 않는** 상태
- R² ≈ 1 이면서 실제 예측 성능이 낮으면 의심
- 해결책 : 정규화, 변수 선택, 교차검증 등

---

## 🔷 OLS Regression Results 해석

> `statsmodels`의 `smf.ols().fit()` 후 `.summary()` 출력 결과 해석

```
                            OLS Regression Results
==============================================================================
Dep. Variable:     만족도   R-squared:        0.588
Model:             OLS   Adj. R-squared:   0.586
Method:  Least Squares   F-statistic:      374.0
                         Prob (F-statistic): 2.24e-52
==============================================================================
                 coef    std err    t      P>|t|   [0.025   0.975]
Intercept      0.7789    0.124   6.273    0.000    0.534    1.023
적절성            0.7393    0.038  19.340    0.000    0.664    0.815
==============================================================================
Omnibus:       11.674   Durbin-Watson:    2.185
Prob(Omnibus):  0.003   Jarque-Bera (JB): 16.003
Skew:          -0.328   Prob(JB):         0.000335
Kurtosis:       4.012   Cond. No.         13.4
==============================================================================
```

### 주요 항목 설명

|항목|의미|해석 기준|
|---|---|---|
|**R-squared**|설명력|1에 가까울수록 좋음 (과적합 주의)|
|**Adj. R-squared**|수정된 설명력|다중회귀에서 R²보다 신뢰도 높음|
|**F-statistic**|모델 전체의 유의성|클수록 좋음|
|**Prob(F-statistic)**|모델 전체 p-value|< 0.05 이면 모델 유의 ✅|
|**coef**|회귀계수 (기울기/절편)|회귀식 구성|
|**P>\|t\|**|계수별 p-value|< 0.05 이면 해당 변수 유의 ✅|

### 잔차 정규성 검정 항목

|항목|의미|해석 기준|
|---|---|---|
|**Omnibus / Prob(Omnibus)**|잔차의 정규성 검정|p > 0.05 → 정규성 만족 ✅|
|**Skew (왜도)**|잔차의 비대칭 정도|0에 가까울수록 좋음. +→오른쪽 꼬리, -→왼쪽 꼬리|
|**Kurtosis (첨도)**|분포의 뾰족함|3 = 정규분포. >3 뾰족+꼬리두꺼움, <3 평평|
|**Jarque-Bera / Prob(JB)**|왜도+첨도 기반 정규성 검정|p > 0.05 → 정규성 만족 ✅|
|**Durbin-Watson**|잔차의 자기상관 검정|2 = 정상. <2 양의 자기상관, >2 음의 자기상관|
|**Cond. No.**|다중공선성 지표|<10 문제없음, 10~30 의심, >30 문제 ⚠️|

---

## 🔷 3가지 구현 방법 비교

|구분|라이브러리|모델 객체 생성|predict()|주요 특징|
|---|---|---|---|---|
|방법1|`scipy.stats.linregress()`|❌|❌ (np.polyval 사용)|빠르고 간단, 기울기/절편/p값만 반환|
|방법2|`sklearn.LinearRegression`|✅|✅|ML 표준 방식, `.coef_` `.intercept_`|
|방법3|`statsmodels smf.ols()`|✅|✅|통계 요약(summary) 제공, 가장 상세|

---

## 🔷 코드 정리

### 공통 전처리 패턴 (결측치 & 이상치)

```python
import pandas as pd
import numpy as np

# 결측치 처리 : 해당 컬럼 평균으로 대체
df['컬럼'] = df['컬럼'].fillna(df['컬럼'].mean())

# 이상치 제거 : 조건으로 행 제거
df = df[df['운동'] <= 10]
```

---

### 방법1 : scipy.stats.linregress()

```python
from scipy import stats
import numpy as np

model = stats.linregress(x, y)

print('기울기 :', model.slope)
print('절편   :', model.intercept)
print('p값    :', model.pvalue)
# 회귀식: y = slope * x + intercept

# 예측 (predict 미지원 → np.polyval 사용)
new_x = 3.0
pred = np.polyval([model.slope, model.intercept], np.array([new_x]))
print('예측값 :', pred)
```

> ⚠️ `linregress`는 `.predict()` 미지원 → `np.polyval([slope, intercept], x)` 사용

---

### 방법2 : sklearn LinearRegression

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# X는 반드시 2D로 reshape
xx = np.array(x).reshape(-1, 1)
yy = np.array(y)

model = LinearRegression()
fit_model = model.fit(xx, yy)   # 최소제곱법으로 기울기, 절편 반환

print('기울기(coef)   :', fit_model.coef_)
print('절편(intercept):', fit_model.intercept_)

# 예측
y_pred = fit_model.predict(np.array([[new_x]]))
print('예측값 :', y_pred)
```

> 💡 `.reshape(-1, 1)` : sklearn은 X를 2D 배열로 받아야 함

---

### 방법3 : statsmodels smf.ols()

```python
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# ols는 1D 배열 + DataFrame 필요
x1 = xx.flatten()   # 2D → 1D (ravel()도 동일)
data2 = pd.DataFrame({'x1': x1, 'y1': yy})

model = smf.ols(formula="y1 ~ x1", data=data2).fit()
print(model.summary())         # 전체 통계 요약
print('기울기:', model.params['x1'])
print('절편  :', model.params['Intercept'])

# 예측
new_df = pd.DataFrame({'x1': [new_x]})
print('예측값:', model.predict(new_df))
```

> 💡 `xx.flatten()` 또는 `xx.ravel()` : 2D → 1D 차원 축소 💡 formula 문법 : `"종속변수 ~ 독립변수1 + 독립변수2"`

---

## 🔷 실습 예제 요약

### ex5linear.py : make_regression으로 기본 맛보기

```python
from sklearn.datasets import make_regression
import numpy as np

np.random.seed(12)
x, y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True)
# coef = 89.47..., bias = 100
# 회귀식 : y = 89.47 * x + 100

# 수동 예측
y_pred = 89.47430739278907 * -0.67794537 + 100
# → 39.34 (실제값과 일치 확인)
```

---

### ex6linear.py : IQ → 시험점수 예측 (실제 데이터)

```python
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

score_iq = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/...")

x = score_iq.iq
y = score_iq.score

print('상관계수 :', np.corrcoef(x, y)[0, 1])   # 0.88222 → 강한 양의 상관

model = stats.linregress(x, y)
print('기울기 :', model.slope)
print('절편   :', model.intercept)

# 예측
newdf = pd.DataFrame({'iq': [55, 66, 77, 88, 150]})
print(np.polyval([model.slope, model.intercept], newdf))

# 시각화
plt.scatter(x, y)
plt.plot(x, model.slope * x + model.intercept, c='r')
plt.show()
```

---

### ex7ols_result.py : 음용수 만족도 분석 + summary 해석

```python
import statsmodels.formula.api as smf

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/.../drinking_water.csv")

# 상관관계 확인
print(df.corr())
# 적절성 ↔ 만족도 : 0.766853 → 강한 양의 상관

model = smf.ols(formula='만족도 ~ 적절성', data=df).fit()
print(model.summary())

# 주요 결과
# R-squared : 0.588   → 적절성이 만족도의 약 59% 설명
# p-value   : 2.24e-52 < 0.05  → 유의한 모델 ✅
# 회귀식 : 만족도 = 0.7393 * 적절성 + 0.7789

print('예측값 :', model.predict()[:5])
print('실제값 :', df.만족도[:5].values)
```

---

### ex8ols_iris.py : 상관관계 약/강 비교 + 다중선형회귀

```python
import seaborn as sns
import statsmodels.formula.api as smf

iris = sns.load_dataset('iris')

# 상관관계 약한 경우 (-0.117570)
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
# R² = 0.0138,  p-value = 0.152 > 0.05  → 유의하지 않은 모델 ❌

# 상관관계 강한 경우 (0.871754)
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit()
# R² = 0.760,  p-value ≈ 1e-47 < 0.05  → 유의한 모델 ✅

# 새로운 값 예측
new_data = pd.DataFrame({'petal_length': [1.1, 0.5, 6.0]})
print(result2.predict(new_data))   # [4.756, 4.511, 6.760]

# 다중선형회귀 (독립변수 여러 개)
result3 = smf.ols(formula='sepal_length ~ petal_length + petal_width', data=iris).fit()
# formula에 + 로 독립변수 추가
```

> 💡 **상관관계가 낮은 변수**로 회귀모델을 만들면 p-value > 0.05 → 신뢰할 수 없는 모델

---

### ex9ols_mtcar.py : mtcars 데이터 단순/다중 선형회귀

```python
import statsmodels.api
import statsmodels.formula.api as smf

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
# 컬럼: mpg(연비), cyl, disp, hp(마력), drat, wt(차체무게), ...

# 상관계수
# hp ↔ mpg : -0.776  (마력 높을수록 연비 낮아짐)
# wt ↔ mpg : -0.868  (무게 무거울수록 연비 낮아짐)

# 단순선형회귀 : hp → mpg
result = smf.ols(formula='mpg ~ hp', data=mtcars).fit()
# 회귀식 : mpg = -0.0682 * hp + 30.0989
print(result.predict(pd.DataFrame({'hp': [110]})))  # 22.59

# 다중선형회귀 : hp + wt → mpg
result2 = smf.ols(formula='mpg ~ hp + wt', data=mtcars).fit()
# 회귀식 : mpg = -0.0318*hp + (-3.8778*wt) + 37.2273
print(result2.predict(pd.DataFrame({'hp': [110], 'wt': [5]})))  # 14.34
```

---

### ex6linear_quiz.py : 회귀분석 문제 (지상파/종편/운동)

```python
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# 데이터 생성
data = {
    '지상파': [0.9,1.2,1.2,1.9,3.3,4.1,5.8,2.8,3.8,4.8,np.nan,0.9,3.0,2.2,2.0],
    '종편':   [0.7,1.0,1.3,2.0,3.9,3.9,4.1,2.1,3.1,3.1,3.5,0.7,2.0,1.5,2.0],
    '운동':   [4.2,3.8,3.5,4.0,2.5,2.0,1.3,2.4,1.3,35.0,4.0,4.2,1.8,3.5,3.5]
}
df = pd.DataFrame(data)

# 1. 결측치 처리 : 각 컬럼 평균으로 대체
df['지상파'] = df['지상파'].fillna(df['지상파'].mean())

# 2. 이상치 제거 : 운동 10시간 초과 제거
df = df[df['운동'] <= 10]   # 35.0 행 제거

# ---- 문제1 : 지상파 → 운동 ----
x, y = df['지상파'], df['운동']

# 방법1: linregress
model1 = stats.linregress(x, y)
# slope ≈ -0.6684, intercept ≈ 4.7097  (지상파 많이 볼수록 운동 감소)

# 방법2: LinearRegression
xx = np.array(x).reshape(-1, 1)
fit = LinearRegression().fit(xx, np.array(y))

# 방법3: ols
df2 = pd.DataFrame({'x1': x.values, 'y1': y.values})
m = smf.ols("y1 ~ x1", data=df2).fit()

# ---- 문제2 : 지상파 → 종편 ----
x2, y2 = df['지상파'], df['종편']
model2 = stats.linregress(x2, y2)
# slope ≈ 0.7726, intercept ≈ 0.2951  (지상파 많이 볼수록 종편도 증가)
```

---

## 🔷 자주 쓰는 패턴 치트시트

```python
# ✅ 상관계수 확인
np.corrcoef(x, y)[0, 1]
df[['x', 'y']].corr()

# ✅ 결측치 평균 대체
df['col'] = df['col'].fillna(df['col'].mean())

# ✅ 이상치 제거 (조건)
df = df[df['col'] <= 상한값]

# ✅ 2D 변환 (sklearn용)
xx = np.array(x).reshape(-1, 1)

# ✅ 1D 변환 (ols용)
x1 = xx.flatten()   # 또는 xx.ravel()

# ✅ 다중회귀 formula 자동 생성
cols = "+".join(df.columns.difference(['제외할컬럼']))
smf.ols(f'y ~ {cols}', data=df).fit()

# ✅ np.polyval로 예측
np.polyval([slope, intercept], np.array([new_x]))
```

---

## 🔷 핵심 포인트 요약

> [!important] 꼭 기억할 것
> 
> 1. **회귀 전 항상 상관계수 확인** → 낮으면 모델 신뢰도 낮음
> 2. **p-value < 0.05** → 유의한 모델/변수
> 3. **R² 높다고 좋은 게 아님** → 과적합 주의
> 4. **sklearn X는 2D** (`reshape(-1,1)`), **ols X는 1D** (`flatten()`)
> 5. `linregress`는 `predict()` 없음 → `np.polyval` 사용
> 6. 다중회귀에서 변수 비교는 **Adj. R²** 기준

---

_출처 : 수업 실습 파일 ex5~ex9linear.py / ex6linear_quiz.py_