# Day 43_선형회귀 적절성 검정

## 📅 2026-04-06

---

## 🔑 선형회귀분석의 기본 충족 조건

|조건|설명|
|---|---|
|**선형성**|독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다|
|**정규성**|잔차항(오차항)이 정규분포를 따라야 한다|
|**독립성**|다중회귀 분석 시 독립변수의 값이 서로 관련되지 않아야 한다|
|**등분산성**|다중회귀 분석 시 그룹 간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다|
|**다중공선성**|다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다|

> **Durbin-Watson** : 잔차의 자기상관(autocorrelation) 검정 지표. 잔차들이 서로 독립적인가? 시간 흐름 데이터에서 중요 (시계열)
> 
> - 값의 범위: 0 ~ 4
> - **2이면 정상** (자기상관 없음)
> - **< 2이면 양의 자기상관**
> - **> 2이면 음의 자기상관**

---

## 🔑 p-value 해석 원칙

- **p-value < 0.05** → 귀무가설 기각 → "이 변수는 의미 있다"
- **p-value > 0.05** → 귀무가설 유지 → "차이가 없다고 볼 수 있다"

> ⚠️ 정규성/등분산성 검정은 반대! p > 0.05 → 조건 만족

---

## 📄 ex10adver.py - 단순선형회귀 (Advertising 데이터)

### 💡 개념

- **단순선형회귀** : 독립변수 1개, 종속변수 1개
- **OLS (Ordinary Least Squares)** : 잔차의 제곱합을 최소화하는 직선을 찾는 방법
- **R²(결정계수)** : 모델이 데이터 분산을 얼마나 설명하는지 (0~1, 높을수록 좋음)
- **잔차(Residual)** : 실제 관측값과 모델이 예측한 값의 차이. 모델이 데이터를 얼마나 잘 설명하는지 보여주는 척도

### 데이터 준비

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import statsmodels.formula.api as smf

advdf = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv", usecols=[1,2,3,4])

print(advdf.head(3), advdf.shape)
#       tv  radio  newspaper  sales
# 0  230.1   37.8       69.2   22.1
# 1   44.5   39.3       45.1   10.4
# 2   17.2   45.9       69.3    9.3 (200, 4)
```

### 상관관계 확인

```python
print(advdf.corr())
#                  tv     radio  newspaper     sales
# tv         1.000000  0.054809   0.056648  0.782224  ← tv와 sales 상관 강함
# radio      0.054809  1.000000   0.354104  0.576223
# newspaper  0.056648  0.354104   1.000000  0.228299
# sales      0.782224  0.576223   0.228299  1.000000
```

### 단순선형회귀 모델 생성 (x: tv, y: sales)

```python
# 단순선형회귀모델 - ols
# x:tv, y:sales
lm = smf.ols(formula='sales ~ tv', data=advdf).fit()
print(f"coeffient:{lm.params}, p-value:{lm.pvalues}, r-squared:{lm.rsquared}")
# coeffient:Intercept    7.032594
# tv           0.047537
# dtype: float64, p-value:Intercept    1.406300e-35
# tv           1.467390e-42
# dtype: float64, r-squared:0.611875050850071
```

### 모델 요약 결과

```python
print(lm.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  sales   R-squared:                       0.612
# Model:                            OLS   Adj. R-squared:                  0.610
# Method:                 Least Squares   F-statistic:                     312.1
# Date:                Mon, 06 Apr 2026   Prob (F-statistic):           1.47e-42
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      7.0326      0.458     15.360      0.000       6.130       7.935
# tv             0.0475      0.003     17.668      0.000       0.042       0.053
# ==============================================================================
# Omnibus:                        0.531   Durbin-Watson:                   1.935
# Prob(Omnibus):                  0.767   Jarque-Bera (JB):                0.669
# Skew:                          -0.089   Prob(JB):                        0.716
# Kurtosis:                       2.779   Cond. No.                         338.
# ==============================================================================
```

> 회귀식 : `sales = 0.0475 × tv + 7.0326` tv 광고비 1 증가 → 판매량 0.0475 증가

### 예측

```python
# 기존 데이터 예측
x_new = pd.DataFrame({'tv': advdf.tv[:3]})
print('실제값 : ', advdf.sales[:3])
print('예측값 : ', lm.predict(x_new).values)
# 실제값 :  0    22.1 / 1    10.4 / 2     9.3
# 예측값 :  [17.97077451  9.14797405  7.85022376]

print('직접 계산 : ', lm.params.tv * 230.1 + lm.params.Intercept)
# 직접 계산 :  17.970774512765537

# 경험하지 않은 tv 광고비에 따른 상품 판매량 예측
my_new = pd.DataFrame({'tv': [100, 350, 780]})
print('예측 상품 판매량 : ', lm.predict(my_new).values)
# 예측 상품 판매량 :  [11.78625759 23.6704177  44.11117309]
```

### ✅ 적절성 조건 체크

#### 잔차 계산

```python
# 잔차(Residual) : 실제 관측값과 모델이 예측한 값의 차이를 의미다.
# 모델이 데이터를 얼마나 잘 설명하는지 보여주는 척도.
fitted = lm.predict(advdf)      # lm.predict(advdf.tv)
residual = advdf['sales'] - fitted
print('실제값:', advdf['sales'][:5].values)
print('예측값:', fitted[:5].values)
print('잔차값 :', residual[:5].values)
print('잔차평균값 :', np.mean(residual[:5]))
# 실제값: [22.1 10.4  9.3 18.5 12.9]
# 예측값: [17.97077451  9.14797405  7.85022376 14.23439457 15.62721814]
# 잔차값 : [ 4.12922549  1.25202595  1.44977624  4.26560543 -2.72721814]
# 잔차평균값 : 1.6738829920227805
```

#### 정규성 검정 (Shapiro-Wilk)

> **개념** : 잔차가 정규분포를 따르는지 확인. p > 0.05이면 정규성 만족

```python
from scipy.stats import shapiro
import statsmodels.api as sm

stat, p = shapiro(residual)
print(f"통계량 : {stat}, p-value:{p}")
# 통계량 : 0.9905306561484953, p-value:0.21332551436720226 > 0.05이므로 정규성 만족
print("정규성 만족" if p > 0.05 else "정규성 위배")

# Q-Q plot으로 시각화
sm.qqplot(residual, line='s')
plt.title("Q-Q plot으로 정규성 만족 확인")
plt.show()
```

#### 선형성 검정 (RESET test)

> **개념** : 독립변수의 변화에 종속변수도 변화하나 특정한 패턴이 있으면 안됨. p > 0.05이면 선형성 만족

```python
from statsmodels.stats.diagnostic import linear_reset   # 선형성 확인 모듈
reset_result = linear_reset(lm, power=2, use_f=True)
print('reset_result 결과 : ', reset_result.pvalue)
print("선형성 만족" if reset_result.pvalue > 0.05 else "선형성 위배")
# reset_result 결과 :  0.055736587109527704
# 선형성 만족

# 선형성 검정 시각화
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color': 'red'})
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='grey')
plt.show()
```

#### 등분산성 검정 (Breusch-Pagan)

> **개념** : 모든 x값에서 오차의 퍼짐이 유사해야 한다. p > 0.05이면 등분산성 만족

```python
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residual, sm.add_constant(advdf['tv']))
bp_stat, bp_pvalue = bp_test[0], bp_test[1]
print(f"breuschpagan test : 통계량:{bp_stat}, p-value:{bp_pvalue}")
print("등분산성 만족" if bp_pvalue > 0.05 else "등분산성 위배")
# breuschpagan test : 통계량:48.037965662293594, p-value:4.180455907755742e-12
# 등분산성 위배 ← 단순회귀에서는 위배될 수 있음
```

#### Cook's Distance (이상치 탐지)

> **개념** : 특정 데이터가 회귀모델에 얼마나 영향을 주는지 확인. 영향력 있는 관측치(이상치)를 탐지하는 진단 방법. 데이터가 적을 때, 이상치가 의심스러울 때, 모델 결과가 이상하게 나올 때 사용

```python
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lm).cooks_distance  # 쿡거리, 인덱스 반환

# 쿡거리가 가장 큰 5개 확인
print(cd.sort_values(ascending=False).head())
# 35     0.060494 / 178    0.056347 / 25     0.038873
# 175    0.037181 / 131    0.033895

# 쿡거리가 가장 큰(영향력이 큰) 관측치 원본 확인
print(advdf.iloc[[35, 178, 25, 175, 131]])
#         tv  radio  newspaper  sales
# 35   290.7    4.1        8.5   12.8  ← tv 높은데 sales 낮음 → 이상치!
# 대부분 tv 광고비는 매우 높으나 sales가 낮음 - 모델이 예측하기 어려운 포인트들

# 시각화
fig = sm.graphics.influence_plot(lm, alpha=0.05, criterion="cooks")
plt.show()
```

### 다중선형회귀 (x: tv, radio, newspaper)

```python
# x:tv, radio, newspaper / y:sales
lm_mul = smf.ols(formula='sales ~ tv + radio + newspaper', data=advdf).fit()
print(lm_mul.summary())
#                  coef    std err          t      P>|t|
# ------------------------------------------------------------------------------
# Intercept      2.9389      0.312      9.422      0.000
# tv             0.0458      0.001     32.809      0.000
# radio          0.1885      0.009     21.893      0.000
# newspaper     -0.0010      0.006     -0.177      0.860  ← p=0.860 > 0.05 → 유의하지 않음!
```

> newspaper는 p-value = 0.860으로 **유의하지 않은 변수** → 제거 고려

---

## 📄 ex11carseats.py - 다중회귀 + 적절성 검정 (Carseats 데이터)

### 💡 개념

- **다중회귀** : 독립변수 여러 개로 종속변수 예측
- **변수 선택** : 상관계수와 p-value를 기준으로 의미 있는 변수만 선택
- **VIF** : 분산 팽창 지수. 다중공선성 진단 지표. **10 초과 시 문제**

### 데이터 준비 및 전처리

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import statsmodels.formula.api as smf

cardf = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv")

# 문자형 컬럼 제거 (ShelveLoc, Urban, US)
cardf = cardf.drop([cardf.columns[6], cardf.columns[9], cardf.columns[10]], axis=1)
```

### 상관관계로 변수 선택

```python
print(cardf.corr())
# → |상관계수| > 0.1 인 변수 선택: Income, Advertising, Price, Age
# → Population(0.05), Education(-0.05) 제외
```

### 모델 생성 및 요약

```python
lm = smf.ols(formula='Sales ~ Income + Advertising + Price + Age', data=cardf).fit()
print(lm.summary())
#                   coef    std err          t      P>|t|
# -------------------------------------------------------------------------------
# Intercept      15.1829      0.777     19.542      0.000
# Income          0.0108      0.004      2.664      0.008
# Advertising     0.1203      0.017      7.078      0.000
# Price          -0.0573      0.005    -11.932      0.000
# Age            -0.0486      0.007     -6.956      0.000
# ← 모든 변수 p-value < 0.05 → 전부 유의!
# Durbin-Watson: 1.931
```

### ✅ 적절성 조건 체크

#### 잔차 계산

```python
cardf_lm = cardf.iloc[:, [0, 2, 3, 5, 6]]  # Sales, Income, Advertising, Price, Age
fitted = lm.predict(cardf_lm)
residual = cardf_lm['Sales'] - fitted
print('잔차 평균 : ', np.mean(residual))    # -1.4122036873231992e-15 ≈ 0 (정상)
```

#### 정규성 검정

```python
from scipy.stats import shapiro
import statsmodels.api as sm

stat, p = shapiro(residual)
print(f"통계량 :{stat}, p-value:{p}")
print("정규성 만족" if p > 0.05 else "정규성 위배")

sm.qqplot(residual, line='s')
plt.title("Q-Q plot으로 정규성 만족 확인")
plt.show()
```

#### 선형성 검정

```python
from statsmodels.stats.diagnostic import linear_reset   # 선형성 확인 모듈
reset_result = linear_reset(lm, power=2, use_f=True)
print('reset_result 결과 : ', f'{reset_result.pvalue:.5f}')
print("선형성 만족" if reset_result.pvalue > 0.05 else "선형성 위배")

sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color': 'red'})
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='grey')
plt.show()
```

#### 등분산성 검정

```python
# 등분산성 검정 : 독립변수의 모든 값에 오차들의 분산은 일정해야 한다
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residual, sm.add_constant(cardf['Sales']))
bp_stat, bp_pvalue = bp_test[0], bp_test[1]
print(f"breuschpagan test : 통계량:{bp_stat}, p-value:{bp_pvalue}")
print("등분산성 만족" if bp_pvalue > 0.05 else "등분산성 위배")
```

#### 독립성 검정 (Durbin-Watson)

```python
# 독립성 검정 : 다중회귀 분석 시 독립변수의 값이 서로 관련되지 않아야 한다.
# 잔차가 자기상관(인접 관측치의 오차가 상관됨)이 있는지 확인
# Durbin-Watson : 잔차의 자기상관(autocorrelation) 검정 지표.
# 잔차들이 서로 독립적인가? 시간 흐름 데이터에서 중요 (시계열)
# 값의 범위는 0 ~ 4 이고   2이면 정상 (자기상관 없음).   < 2이면 양의 자기상관,  > 2이면 음의 자기상관
import statsmodels.api as sm
print('Durbin-Watson : ', sm.stats.stattools.durbin_watson(residual))
# 1.9314981270829592이므로 잔차의 자기상관은 없다.
```

#### 다중공선성 검정 (VIF)

```python
# 다중공선성 검정 : 다중회귀 분석 시 독립변수 간에 강한 상관관계가 있어서는 안된다.
# VIF(variance_inflation_factor, 분산 인플레 요인, 분산 팽창 지수)
# : 값이 10을 넘으면 다중 공선성이 발생하는 변수라고 할 수 있다.
from statsmodels.stats.outliers_influence import variance_inflation_factor
cardf_ind = cardf[['Income', 'Advertising', 'Price', 'Age']]  # 독립변수들
vifdf = pd.DataFrame()
vifdf['변수'] = cardf_ind.columns
vifdf['vif_value'] = [variance_inflation_factor(cardf_ind.values, i) for i in range(cardf_ind.shape[1])]

print(vifdf)    # 10을 초과하지 않았으므로 모두 만족
#             변수  vif_value
# 0       Income   5.971040
# 1  Advertising   1.993726
# 2        Price   9.979281
# 3          Age   8.267760

sns.barplot(x="변수", y="vif_value", data=vifdf)
plt.title("VIF")
plt.show()
```

### 모델 저장 및 재사용

```python
# 유의한 모델이므로 생성된 모델을 파일로 저장하고 이를 재사용
# 방법1
# import pickle
# with open('carseat.pickle', 'wb') as obj:   # 저장
#     pickle.dump(lm, obj)
# with open('carseat.pickle', 'rb') as obj:   # 읽기
#     mymodel = pickle.load(lm, obj)

# 방법2 : pickle은 binary로 I/O해야 하므로 번거롭다
import joblib
joblib.dump(lm, 'carseat.model')
# 이후 부터는 아래 처럼 읽어 사용하면 됨(lm은 없어도 됨)
mymodel = joblib.load('carseat.model')
print("새로운 값으로 Sales 예측")
new_df = pd.DataFrame({"Income": [35, 62], "Advertising": [6, 3], "Price": [105, 88], "Age": [32, 55]})
pred = mymodel.predict(new_df)
print("Sales 예측 결과 : ", pred.values)    # [8.71289759 8.49715914]
```

---

## 📄 ex12linear.py - sklearn LinearRegression + 성능 지표

### 💡 개념

- **sklearn LinearRegression** : `summary()` 미지원. 성능 지표를 별도로 계산해야 함
- **MinMaxScaler** : 정규화. 독립변수 값을 0~1 범위로 변환

|지표|공식|특징|
|---|---|---|
|**R²**|예측분산 / 실제분산|설명력 (0~1)|
|**MSE**|오차² 평균|모델 내부 비교, 단위가 제곱|
|**RMSE**|√MSE|보고/해석용, 원래 단위로 해석 가능|
|**설명분산점수**|-|오차 분산이 작으면 점수 높음|

> ⚠️ r2_score 하나만 보고 모델 판단 X → **r2_score + MSE 또는 r2_score + RMSE** 함께 사용

```python
# LinearRegression 클래스 사용 : 평가 score 정리
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression       # summary() 지원안함
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler  # 정규화 클래스

sample_size = 100
np.random.seed(1)
x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 10, sample_size) + x * 30

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1))

model = LinearRegression().fit(x_scaled, y)
print('회귀계수(slope):', model.coef_)       # [1350.4161554]
print('절편(intercept,bias):', model.intercept_)  # -691.1877661754081
print('결정계수(R^2):', model.score(x_scaled, y)) # 0.9987875127274646
y_pred = model.predict(x_scaled)

# 모델 성능 확인 함수 작성
def myRegScoreFunc(y_true, y_pred):
    # 결정계수 : 실제 관측값의 분산대비 예측값의 분산을 계산하여 데이터 예측의 정확도 성능 측정 지표
    print(f"r2_score(결정계수):{r2_score(y_true, y_pred)}")
    # 모델이 데이터의 분산을 얼마나 잘 설명하는지 나타내는 지표 (오차 분산이 작은면 점수 높음)
    print(f"explained_variance_score(설명분산점수):{explained_variance_score(y_true, y_pred)}")
    # 오차를 제곱해 평균 구함(오차가 커질수록 손실함수 값이 빠르게 증가함. 값이 작으면 모델 성능 우수)
    print(f"mean_squared_error(MSE, 평균제곱 오차):{mean_squared_error(y_true, y_pred)}")
    imsi = mean_squared_error(y_true, y_pred)   # RMSE로 변환해서 확인
    # 예측값이 실제값에서 평균적으로 얼마나 틀리는지 (오차 크기)
    print(f"mean_squared_error(RMSE, 평균제곱 오차의 제곱근):{np.sqrt(imsi)}")
    # 9.281592052012815 (평균적으로 약 + - 9 정도 틀림)

myRegScoreFunc(y, y_pred)
# r2_score(결정계수):0.9987875127274646
# explained_variance_score(설명분산점수):0.9987875127274646
# mean_squared_error(MSE, 평균제곱 오차):86.14795101998747
```

---

## 📄 ex13linear_mtcar.py - mtcars 데이터로 실습

### 💡 개념

- **statsmodels 내장 데이터셋** : `get_rdataset()` 으로 R 데이터셋 불러오기
- **hp → mpg 예측** : 마력(hp)이 연비(mpg)에 미치는 영향 분석

```python
# LinearRegression 클래스 사용 : 평가 score - mtcars dataset 사용
from sklearn.linear_model import LinearRegression
import statsmodels.api
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data

x = mtcars[['hp']].values   # 2차원 반환
y = mtcars['mpg'].values    # 1차원 반환

lmodel = LinearRegression().fit(x, y)
print('slope : ', lmodel.coef_)        # [-0.06822828]
print('intercept : ', lmodel.intercept_)  # 30.098860539622496

pred = lmodel.predict(x)
print('예측값 : ', np.round(pred[:5], 1))   # [22.6 22.6 23.8 22.6 18.2]
print('실제값 : ', y[:5])                   # [21.  21.  22.8 21.4 18.7]

# 모델 성능 지표
# MSE : 모델 내부비교, 계산 편리(단위가 제곱한 값)
# RMSE : 보고/해석용, 해석이 용이(원래 단위)
# 회귀 평가 지표는 고정된 점수 범위가 없다.(데이터 스케일에 따라 다름)
# 그래서 모델끼리 상대적인 비교를 한다.
print('MSE : ', mean_squared_error(y, pred))             # 13.989822298268805
print('RMSE : ', np.sqrt(mean_squared_error(y, pred)))   # 3.7402970868994894
print('r2_score : ', r2_score(y, pred))                  # 0.602437341423934
# r2_score 하나만 보고 모델 판단 X(이상치에 민감, 변수가 많으면 증가...), 설명력만 봄
# 모델 성능은 r2_score와 MSE 또는 r2_score와 RMSE를 사용하도록 한다.

print("새로운 hp로 mpg 예측 ---")
new_hp = [[100], [110], [120], [130]]
new_pred = lmodel.predict(new_hp)
print('예측 결과 : ', np.round(new_pred.flatten(), 2))   # [23.28 22.59 21.91 21.23]
```

---

## 📄 ex13quiz - Flask 웹 연동 (jikwon 연봉 예측)

### 💡 개념

이번 실습의 핵심은 **ML 모델 + Flask + Ajax** 를 하나로 연결하는 것이에요.

|구성요소|역할|
|---|---|
|**make_model()**|서버 시작 시 DB에서 데이터를 가져와 LinearRegression 모델을 1번만 학습|
|**Flask 라우팅**|`/` → 기본 화면, `/predict` → Ajax 예측 요청 처리|
|**axios (Ajax)**|버튼 클릭 시 페이지 새로고침 없이 서버에 POST 요청 후 결과만 화면에 갱신|
|**Jinja2 템플릿**|`{{ }}` 로 Flask에서 전달한 변수를 HTML에 출력|

> **왜 make_model()을 서버 시작 시 1번만 실행하나?** 학습은 시간이 걸리므로, 매 요청마다 다시 학습하면 느려짐. 한 번 학습한 모델을 `global` 변수로 저장해두고 예측에만 재사용!

### 폴더 구조

```
ex13quiz/
├── app.py
└── templates/
    └── main.html
```

### app.py

```python
# 회귀분석 문제 4)
# 원격 DB의 jikwon 테이블에서 근무년수에 대한 연봉을 이용하여 회귀분석 모델을 작성하시오.
# Django 또는 Flask로 작성한 웹에서 근무년수를 입력하면 예상 연봉이 나올 수 있도록 프로그래밍 하시오.
# LinearRegression 사용. Ajax 처리!!!      참고: Ajax 처리가 힘들면 그냥 submit()을 해도 됩니다.

from flask import Flask, render_template, request, jsonify
import pymysql
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime

app = Flask(__name__)

# 전역 변수로 모델과 결과값 저장
model = None
r2 = 0
coef = 0
intercept = 0
jik_avg = []


def get_connection():
    return pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='123',
        database='test',
        port=3306,
        charset='utf8'
    )


def make_model():
    global model, r2, coef, intercept, jik_avg

    # DB 연결
    conn = get_connection()
    cur = conn.cursor()

    # 직원 테이블에서 입사일과 연봉을 조회
    cur.execute("""
        select jikwonibsail, jikwonpay
        from jikwon
        where jikwonibsail is not null and jikwonpay is not null
    """)

    # 조회 결과를 DataFrame으로 변환
    df = pd.DataFrame(cur.fetchall(), columns=['jikwonibsail', 'jikwonpay'])

    # 직급별 평균 연봉 조회
    cur.execute("""
        select ifnull(jikwonjik, '직급없음') as jikwonjik,
        round(avg(jikwonpay), 0) as avg_pay
        from jikwon
        group by jikwonjik
    """)

    # 직급별 평균 연봉 결과를 DataFrame으로 변환
    avg_df = pd.DataFrame(cur.fetchall(), columns=['jikwonjik', 'avg_pay'])

    # DB 연결 종료
    conn.close()

    # 입사일 컬럼을 datetime 형식으로 변환
    df['jikwonibsail'] = pd.to_datetime(df['jikwonibsail'], errors='coerce')

    # NaN, NaT가 들어있는 행 제거
    df = df.dropna()

    # 현재 연도 - 입사 연도 = 근무년수 계산
    df['years'] = datetime.now().year - df['jikwonibsail'].dt.year

    # 근무년수가 0 이상인 데이터만 사용
    # 이상한 데이터가 있으면 제외
    df = df[df['years'] >= 0]

    # 독립변수(x) 설정 - 근무년수(years)를 사용
    x = df[['years']]

    # 종속변수(y) 설정 - 연봉(jikwonpay)을 사용
    y = df['jikwonpay']

    # 선형회귀 모델 객체 생성 후 학습
    model = LinearRegression()
    model.fit(x, y)

    # 회귀계수(기울기) 저장 - coef_는 배열 형태라 첫 번째 값 사용
    coef = round(float(model.coef_[0]), 4)

    # 절편 저장
    intercept = round(float(model.intercept_), 4)

    # 결정계수(R²) 계산 - 실제 y값과 예측값 비교 후 백분율 형태로 저장
    r2 = round(r2_score(y, model.predict(x)) * 100, 2)

    # 직급별 평균 연봉 DataFrame을 딕셔너리 리스트 형태로 변환
    jik_avg = avg_df.to_dict(orient='records')


# 메인 페이지 라우팅
# 브라우저에서 / 주소로 접속하면 실행
@app.route('/')
def index():
    # 기본 화면에 보여줄 예측값 계산 - 근무년수 3년을 넣어 예측
    pred = round(float(model.predict(pd.DataFrame({'years': [3]}))[0]), 2)

    # 예측값이 음수면 0으로 보정 - 연봉은 음수가 될 수 없기 때문
    if pred < 0:
        pred = 0

    # main.html 파일을 화면에 보여주고 예측값, 설명력, 회귀식, 직급별 평균연봉 데이터 전달
    return render_template(
        'main.html',
        pred=pred,
        r2=r2,
        coef=coef,
        intercept=intercept,
        jik_avg=jik_avg
    )


# 예측 처리 라우팅
# axios가 /predict 주소로 POST 요청을 보내면 실행
@app.route('/predict', methods=['POST'])
def predict():
    # axios는 보통 JSON 형태로 데이터를 전송하므로 request.get_json()으로 전달값을 받음
    data = request.get_json()

    # 사용자가 입력한 근무년수를 가져옴 - 문자열일 수 있으므로 float로 변환
    years = float(data['years'])

    # 입력된 근무년수로 연봉 예측 - DataFrame 형태로 컬럼명을 맞춰서 예측
    pred = model.predict(pd.DataFrame({'years': [years]}))

    # 예측 결과를 소수점 둘째 자리까지 반올림
    pred_value = round(float(pred[0]), 2)

    # 예측값이 음수면 0으로 보정 - 실제 연봉은 음수가 될 수 없으므로 처리
    if pred_value < 0:
        pred_value = 0

    # 예측 결과를 JSON 형식으로 반환 - axios의 response.data로 받을 수 있음
    return jsonify({
        'pred': pred_value,
        'r2': r2,
        'coef': coef,
        'intercept': intercept
    })


if __name__ == '__main__':
    make_model()   # 서버 시작 시 모델 학습 1번만 실행
    app.run(debug=True)
```

### templates/main.html

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>연봉 예측</title>

    <!-- axios 라이브러리 불러오기 -->
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
</head>
<body>
    <h2>근무년수 -> 예상 연봉(LinearRegression)</h2>

    <!-- 사용자가 근무년수를 입력하는 영역 -->
    근무년수 : <input type="text" id="years" value="3">
    <button type="button" id="btn">확인</button>

    <!-- 예측 결과 출력 영역 -->
    <div style="margin-top:15px; background:#f3f3f3; width:280px; padding:10px;">
        <div><b>예상연봉액 : <span id="predArea">{{ pred }}</span></b></div>
        <div>설명력(R²) : <b><span id="r2Area">{{ r2 }}</span>%</b></div>
        <div>회귀식 : <b>pay = <span id="coefArea">{{ coef }}</span> * years + <span id="interceptArea">{{ intercept }}</span></b></div>
    </div>

    <!-- 직급별 평균 연봉 출력 영역 -->
    <h3>직급별 연봉평균</h3>
    <table border="1">
        <tr>
            <th>직급</th>
            <th>평균 연봉</th>
        </tr>

        <!-- Flask에서 전달한 직급별 평균 연봉 데이터를 반복 출력 -->
        {% for row in jik_avg %}
        <tr>
            <td>{{ row['jikwonjik'] }}</td>
            <td>{{ "{:,.0f}".format(row['avg_pay']) }}</td>
        </tr>
        {% endfor %}
    </table>

    <script>
        // 확인 버튼 클릭 시 실행
        document.getElementById('btn').addEventListener('click', function () {

            // 입력한 근무년수 값 읽기
            const years = document.getElementById('years').value;

            // axios를 사용해 서버의 /predict 주소로 POST 요청 전송
            // JSON 형태로 years 값을 보냄
            axios.post('/predict', {
                years: years
            })
            .then(function(response) {
                // 서버가 보낸 응답 데이터 확인
                console.log('서버 응답:', response.data);

                // 응답받은 예측 결과를 화면에 출력
                document.getElementById('predArea').textContent = response.data.pred;
                document.getElementById('r2Area').textContent = response.data.r2;
                document.getElementById('coefArea').textContent = response.data.coef;
                document.getElementById('interceptArea').textContent = response.data.intercept;
            })
            .catch(function(error) {
                // 서버 요청 실패 시 실행
                console.error('오류 발생:', error);
                alert('오류 발생');
            });
        });
    </script>
</body>
</html>
```

### Ajax 흐름 정리

```
버튼 클릭
  ↓
axios.post('/predict', { years: 입력값 })   ← 페이지 새로고침 없이 서버로 전송
  ↓
Flask /predict 라우트에서 JSON 수신
  ↓
model.predict()로 연봉 예측
  ↓
jsonify({ pred, r2, coef, intercept }) 반환
  ↓
.then(response) → 화면의 span 태그 내용만 갱신
```

> **fetch() vs axios()** : 둘 다 Ajax 처리 가능. axios는 JSON 변환을 자동으로 해줘서 더 편함

---

## 📝 핵심 요약

### smf.ols vs sklearn LinearRegression 비교

|항목|`smf.ols` (statsmodels)|`LinearRegression` (sklearn)|
|---|---|---|
|summary()|✅ 지원|❌ 미지원|
|p-value|✅ 자동 제공|❌ 직접 계산 필요|
|성능 지표|R², AIC, BIC 등|score(), 별도 import 필요|
|용도|통계 분석, 해석|머신러닝 파이프라인|

### 모델 저장 방법 비교

|방법|코드|특징|
|---|---|---|
|pickle|`pickle.dump()` / `pickle.load()`|binary I/O 필요해서 번거로움|
|joblib|`joblib.dump()` / `joblib.load()`|더 간편, 대용량 배열에 최적화|