# Day 40_통계 검정 (t검정 & ANOVA)

## 📅 2026-04-01

---

# 📊 t검정 (t-test) 완전 정리

> **가설 검정의 흐름**: 귀무가설 설정 → 정규성 검정 → (등분산 검정) → t검정 → p-value 해석

---

## t검정 종류 한눈에 보기

|검정 종류|언제 사용?|핵심 조건|scipy 함수|
|---|---|---|---|
|**독립표본 t검정**|두 집단(서로 다른 대상)의 평균 비교|정규성 + 등분산성|`ttest_ind()`|
|**대응표본 t검정**|동일 집단의 처리 전/후 비교|차이값의 정규성|`ttest_rel()`|
|**Mann-Whitney U**|정규성 불만족 시 비모수 대안|없음|`mannwhitneyu()`|

> [!tip] 검정 순서
> 
> 1. **정규성 검정** `shapiro()` → p > 0.05 이면 정규성 만족
> 2. **등분산 검정** `levene()` → p > 0.05 이면 등분산 만족 (독립표본만)
> 3. **t검정** 실행 → p < 0.05 이면 귀무가설 기각

---

## test11t.py — 독립표본 t검정 (강수 여부와 매출액)

### 📌 개념

**독립표본 t검정 (Independent Samples t-test)**

- 서로 **독립적인 두 집단**의 평균을 비교
- 두 집단이 같은 모집단에서 왔는지 검정
- 이 예제에서는 **강수 있는 날 vs 맑은 날**의 매출액 평균 비교

### 🔬 가설

- **귀무가설 H₀**: 강수 여부에 따라 매출액 평균에 차이가 **없다**
- **대립가설 H₁**: 강수 여부에 따라 매출액 평균에 차이가 **있다**

### 💻 코드

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib

pd.set_option("display.max_columns", None)

# 매출 데이터 읽기
sales_data = pd.read_csv(
    "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv",
    dtype={'YMD': 'object'}  # int -> object 변환해 읽기
)

# 날씨 데이터 읽기
wt_data = pd.read_csv(
    "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv"
)

# 날짜 형식 통일: "2018-06-01" → "20180601"
wt_data.tm = wt_data.tm.map(lambda x: x.replace("-", ""))

# 두 데이터 병합 (YMD ↔ tm 기준)
frame = sales_data.merge(wt_data, how="left", left_on="YMD", right_on='tm')
data = frame.iloc[:, [0, 1, 7, 8]]  # YMD, AMT, maxTa, sumRn

# 강수 여부 컬럼 추가: 강수량 > 0 이면 1, 없으면 0
data['rain_yn'] = (data.loc[:, ('sumRn')] > 0) * 1

# 두 집단 분리
sp = np.array(data.iloc[:, [1, 4]])  # AMT, rain_yn
tg1 = sp[sp[:, 1] == 0, 0]  # 맑은 날 매출액
tg2 = sp[sp[:, 1] == 1, 0]  # 비 온 날 매출액

print('tg1(맑은날) 매출액 평균 :', np.mean(tg1))  # 761040.25
print('tg2(비온날) 매출액 평균 :', np.mean(tg2))  # 757331.52

# 시각화 (box plot)
plt.boxplot([tg1, tg2], meanline=True, showmeans=True, notch=True)
plt.show()

# 1단계: 정규성 검정 (Shapiro-Wilk)
print(stats.shapiro(tg1).pvalue)  # 0.056 > 0.05 → 정규성 만족
print(stats.shapiro(tg2).pvalue)  # 0.882 > 0.05 → 정규성 만족

# 2단계: 등분산 검정 (Levene)
print(stats.levene(tg1, tg2).pvalue)  # 0.712 > 0.05 → 등분산 만족

# 3단계: 독립표본 t검정
print(stats.ttest_ind(tg1, tg2, equal_var=True))
# statistic=0.101, pvalue=0.919, df=326
```

### 📊 결과 해석

|검정|p-value|판정|
|---|---|---|
|정규성 (맑은날)|0.056 > 0.05|✅ 만족|
|정규성 (비온날)|0.882 > 0.05|✅ 만족|
|등분산 (Levene)|0.712 > 0.05|✅ 만족|
|**t검정**|**0.919 > 0.05**|**귀무가설 채택**|

> **결론**: 강수 여부에 따라 매출액 평균에 통계적으로 유의미한 차이가 없다.

---

## test12t.py — 대응표본 t검정 (특강 전후 시험 점수)

### 📌 개념

**대응표본 t검정 (Paired Samples t-test)**

- **동일한 대상**에 대해 처리 전과 후를 측정하여 비교
- 두 측정값이 1:1로 대응됨
- 집단 간 비교가 아니므로 **등분산 가정 불필요**
- 예: 특강 전/후, 약 복용 전/후, 광고 전/후 선호도

### 🔬 가설

- **귀무가설 H₀**: 특강 전후의 시험 점수는 차이가 **없다**
- **대립가설 H₁**: 특강 전후의 시험 점수는 차이가 **있다**

### 💻 코드

```python
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(123)
x1 = np.random.normal(75, 10, 100)  # 특강 전 점수 (평균 75)
x2 = np.random.normal(80, 10, 100)  # 특강 후 점수 (평균 80)

# 정규성 확인 (시각화)
sns.displot(x1, kde=True)
sns.displot(x2, kde=True)
plt.show()

# 정규성 검정
print(stats.shapiro(x1).pvalue)  # 0.274 > 0.05 → 정규성 만족
print(stats.shapiro(x2).pvalue)  # 0.102 > 0.05 → 정규성 만족

# 대응표본 t검정
print(stats.ttest_rel(x1, x2))
# statistic=-3.003, pvalue=0.003, df=99
```

### 📊 결과 해석

- p-value = **0.003 < 0.05** → **귀무가설 기각**
- 특강 전후 시험 점수에 통계적으로 유의미한 차이가 있다.

> [!note] 독립표본 vs 대응표본 차이
> 
> - **독립표본**: 남학생 집단 vs 여학생 집단 (서로 다른 사람)
> - **대응표본**: 동일 학생의 특강 전 점수 vs 특강 후 점수 (같은 사람)

---

## test13t.py — 대응표본 t검정 (수술 전후 몸무게)

### 🔬 가설

- **귀무가설 H₀**: 복부 수술 전후 몸무게 변화가 **없다**
- **대립가설 H₁**: 복부 수술 전후 몸무게 변화가 **있다**

### 💻 코드

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

baseline  = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
follow_up = [62.4, 64.6, 70.4, 62.6, 80.1, 73.2, 58.2, 71.0, 101.0]

print(np.mean(baseline))   # 78.41
print(np.mean(follow_up))  # 71.50
print('평균의 차이 :', np.mean(baseline) - np.mean(follow_up))  # 6.911

# 시각화
plt.bar(np.arange(2), [np.mean(baseline), np.mean(follow_up)])
plt.xlim(0, 1)
plt.xlabel('수술 전후', fontdict={'fontsize': 12, 'fontweight': 'bold'})
plt.show()

# 대응표본 t검정
result = stats.ttest_rel(baseline, follow_up)
print(result)
# statistic=3.668, pvalue=0.006, df=8
```

### 📊 결과 해석

- p-value = **0.006 < 0.05** → **귀무가설 기각**
- 복부 수술 전후 몸무게에 통계적으로 유의미한 차이가 있다.

---

## test13_quiz1.py — 독립표본 t검정 (포장지 색상별 매출액)

### 🔬 가설

- **귀무가설 H₀**: 포장지 색상에 따른 매출액에 차이가 **없다**
- **대립가설 H₁**: 포장지 색상에 따른 매출액에 차이가 **있다**

### 💻 코드

```python
import numpy as np
from scipy import stats

blue = np.array([70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80])
red  = np.array([60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66])

# 1단계: 정규성 검정
shapiro_blue = stats.shapiro(blue)
shapiro_red  = stats.shapiro(red)
print(shapiro_blue)  # p > 0.05 → 정규성 만족
print(shapiro_red)   # p > 0.05 → 정규성 만족

# 2단계: 등분산 검정
levene = stats.levene(blue, red)
print(f"Levene p-value: {levene.pvalue:.4f}")  # 0.4392 > 0.05 → 등분산 만족

# 3단계: 독립표본 t검정
t_stat, p_val = stats.ttest_ind(blue, red, equal_var=True)
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")
# t=2.934, p=0.0083
```

### 📊 결과 해석

- p-value = **0.0083 < 0.05** → **귀무가설 기각**
- 포장지 색상에 따라 매출액에 통계적으로 유의미한 차이가 있다.

---

## test13_quiz2.py — 독립표본 t검정 or Mann-Whitney U (콜레스테롤)

### 📌 개념

**비모수 검정 (Mann-Whitney U test)**

- 정규성 가정이 **만족되지 않을 때** 독립표본 t검정의 대안
- 순위(rank)를 기반으로 두 집단의 분포를 비교
- 정규성이 확인되면 t검정, 아니면 Mann-Whitney U로 자동 전환

### 🔬 가설

- **귀무가설 H₀**: 남녀 간 콜레스테롤 양에 차이가 **없다**
- **대립가설 H₁**: 남녀 간 콜레스테롤 양에 차이가 **있다**

### 💻 코드

```python
import numpy as np
from scipy import stats

men_pop   = np.array([0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2,
                      3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8])
women_pop = np.array([1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5,
                      2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4])

# 각 15명씩 무작위 비복원 추출
men_sample   = np.random.choice(men_pop, 15, replace=False)
women_sample = np.random.choice(women_pop, 15, replace=False)

# 정규성 검정
shapiro_men   = stats.shapiro(men_sample)
shapiro_women = stats.shapiro(women_sample)
print(f"남자 p-value: {shapiro_men.pvalue:.4f}")
print(f"여자 p-value: {shapiro_women.pvalue:.4f}")

if shapiro_men.pvalue > 0.05 and shapiro_women.pvalue > 0.05:
    # 정규성 만족 → 등분산 확인 후 독립표본 t검정
    levene   = stats.levene(men_sample, women_sample)
    is_equal = levene.pvalue > 0.05
    t_stat, p_val = stats.ttest_ind(men_sample, women_sample, equal_var=is_equal)
    test_name = "독립표본 t-검정"
else:
    # 정규성 불만족 → 비모수 검정
    u_stat, p_val = stats.mannwhitneyu(men_sample, women_sample, alternative='two-sided')
    test_name = "Mann-Whitney U 검정"

print(f"분석 방법: {test_name}")
print(f"p-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("결론: 귀무가설 기각 → 남녀 간 콜레스테롤 양에 유의미한 차이 존재")
else:
    print("결론: 귀무가설 채택 → 남녀 간 콜레스테롤 양에 유의미한 차이 없음")
```

> [!warning] 랜덤 추출 주의 `np.random.choice`로 매번 다른 샘플이 추출되므로 결과가 실행마다 달라질 수 있음. 재현성이 필요하면 `np.random.seed(숫자)` 를 앞에 추가.

---

## test13_quiz3.py — Mann-Whitney U 검정 (부서별 연봉)

### 📌 개념

- DB에서 데이터를 불러와 검정 수행
- 정규성 불만족 → **Mann-Whitney U 비모수 검정** 적용
- 결측치는 **해당 부서 평균**으로 대체

### 🔬 가설

- **귀무가설 H₀**: 총무부와 영업부 직원의 연봉 평균에 차이가 **없다**
- **대립가설 H₁**: 총무부와 영업부 직원의 연봉 평균에 차이가 **있다**

### 💻 코드

```python
import pandas as pd
import numpy as np
from scipy import stats
import MySQLdb

try:
    conn = MySQLdb.connect(
        host='127.0.0.1', user='root', passwd='123', db='test', charset='utf8'
    )
    sql = """
        SELECT b.busername, j.jikwonpay
        FROM jikwon j
        JOIN buser b ON j.busernum = b.buserno
        WHERE b.busername IN ('총무부', '영업부')
    """
    df = pd.read_sql(sql, conn)
except Exception as e:
    pass
finally:
    conn.close()

# 결측치 처리: 부서별 평균으로 채우기
df['jikwonpay'] = df['jikwonpay'].fillna(
    df.groupby('busername')['jikwonpay'].transform('mean')
)

# 데이터 분리
total_dept = df[df['busername'] == '총무부']['jikwonpay']
sales_dept = df[df['busername'] == '영업부']['jikwonpay']

# 정규성 검정
shapiro_total = stats.shapiro(total_dept)
shapiro_sales = stats.shapiro(sales_dept)
print(f"총무부 p-value: {shapiro_total.pvalue:.4f}")  # p < 0.05 → 정규성 불만족
print(f"영업부 p-value: {shapiro_sales.pvalue:.4f}")  # p < 0.05 → 정규성 불만족

# 비모수 검정 (Mann-Whitney U)
u_stat, p_val = stats.mannwhitneyu(total_dept, sales_dept, alternative='two-sided')
print(f"p-value: {p_val:.4f}")  # 0.4721 > 0.05 → 귀무가설 채택
```

### 📊 결과 해석

- p-value = **0.4721 > 0.05** → **귀무가설 채택**
- 총무부와 영업부 간 연봉 평균에 통계적으로 유의미한 차이가 없다.

---

## test13_quiz4.py — 대응표본 t검정 (중간/기말 성적)

### 🔬 가설

- **귀무가설 H₀**: 중간고사와 기말고사 성적의 차이가 **없다**
- **대립가설 H₁**: 중간고사와 기말고사 성적의 차이가 **있다**

### 💻 코드

```python
import numpy as np
from scipy import stats

midterm = np.array([80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80])
final   = np.array([90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95])

# 대응표본은 차이값(diff)의 정규성만 확인
diff = final - midterm
shapiro_test = stats.shapiro(diff)
print(f"차이값의 정규성 검정 p-value: {shapiro_test.pvalue:.4f}")
# 0.3011 > 0.05 → 정규성 만족

# 대응표본 t검정
t_stat, p_val = stats.ttest_rel(midterm, final)
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")
# t=-2.776, p=0.018
```

### 📊 결과 해석

- p-value = **0.018 < 0.05** → **귀무가설 기각**
- 중간고사와 기말고사 성적에 유의미한 차이가 있으며, 학업능력이 변화했다고 볼 수 있다.

---

## 검정 통계 핵심 개념 모음

### 📐 유의수준 (α, alpha)

- 일반적으로 **α = 0.05** (5%) 사용
- p-value < α → 귀무가설 **기각** (통계적으로 유의미한 차이 있음)
- p-value ≥ α → 귀무가설 **채택** (차이가 없다고 볼 수 없음)

### 📐 Shapiro-Wilk 정규성 검정

```python
stats.shapiro(data).pvalue
# p > 0.05 → 정규 분포를 따름 (정규성 만족)
# p ≤ 0.05 → 정규 분포를 따르지 않음
```

### 📐 Levene 등분산 검정

```python
stats.levene(group1, group2).pvalue
# p > 0.05 → 두 집단의 분산이 같음 (등분산 만족)
# p ≤ 0.05 → 두 집단의 분산이 다름 → equal_var=False 사용
```

### 📐 검정 함수 요약

```python
# 독립표본 t검정
stats.ttest_ind(group1, group2, equal_var=True)   # 등분산
stats.ttest_ind(group1, group2, equal_var=False)  # 이분산 (Welch's t-test)

# 대응표본 t검정
stats.ttest_rel(before, after)

# 비모수: Mann-Whitney U
stats.mannwhitneyu(group1, group2, alternative='two-sided')
```

### 📐 검정 선택 플로우

```
두 집단의 평균 비교
       │
       ├─ 같은 대상인가? (쌍체 데이터)
       │       │
       │    YES └─→ 대응표본 t검정 (ttest_rel)
       │                └─ 차이값의 정규성만 확인하면 됨
       │
       └─ NO (독립된 두 집단)
               │
         정규성 만족? (shapiro p > 0.05)
               │
            YES └─→ 등분산 만족? (levene p > 0.05)
                         │
                      YES └─→ 독립표본 t검정 equal_var=True
                      NO  └─→ Welch's t검정 equal_var=False
               │
            NO └─→ Mann-Whitney U 검정 (비모수)
```

---

# 📊 분산분석 (ANOVA) 완전 정리

> **t검정은 두 집단 비교, ANOVA는 세 집단 이상 비교** 관련 노트: [[t검정_정리]]

---

## 분산분석이란?

**ANOVA (ANalysis Of Variance)**

세 집단 이상의 평균을 한 번에 비교하는 통계 기법.

### 왜 t검정을 반복하면 안 될까?

집단이 3개일 때 t검정을 쌍으로 반복하면 비교 횟수가 3번(A-B, A-C, B-C)이 된다. 각 검정마다 α=0.05의 오류 가능성이 있고, 이를 반복할수록 **제1종 오류(Type I Error, 실제로는 차이가 없는데 있다고 결론)**가 누적 증가한다. ANOVA는 이 문제를 한 번의 검정으로 해결한다.

### F값의 의미

$$F = \frac{\text{집단 간 분산 (Between-group variance)}}{\text{집단 내 분산 (Within-group variance)}}$$

- F값이 **클수록** → 집단 간 차이가 집단 내 차이보다 크다 → 귀무가설 기각 가능성 ↑
- F값이 **1에 가까울수록** → 집단 간 차이가 없다 → 귀무가설 채택

### 일원분산분석 조건

|조건|검정 방법|기준|
|---|---|---|
|정규성|Shapiro-Wilk|p > 0.05|
|등분산성|Levene or Bartlett|p > 0.05|

> [!warning] 조건 불만족 시 대안
> 
> - 정규성 불만족 → **Kruskal-Wallis 검정** `stats.kruskal()`
> - 등분산성 불만족 → **Welch's ANOVA** `pingouin.welch_anova()`

---

## 검정 선택 플로우

```
세 집단 이상의 평균 비교
         │
   정규성 만족? (shapiro p > 0.05, 각 집단)
         │
      YES └─→ 등분산성 만족? (levene p > 0.05)
                   │
                YES └─→ 일원분산분석 f_oneway() 또는 anova_lm()
                NO  └─→ Welch's ANOVA  pingouin.welch_anova()
         │
      NO └─→ Kruskal-Wallis  stats.kruskal()
         │
         └─ 유의미한 차이 발견 (p < 0.05)?
                   │
                YES └─→ 사후검정 Tukey HSD  pairwise_tukeyhsd()
                          → 어느 집단 간에 차이가 있는지 확인
                NO  └─→ 분석 종료 (귀무가설 채택)
```

---

## test14a.py — 일원분산분석 (교육방법별 시험 점수)

### 📌 개념

- **독립변수 (범주형)**: 교육방법 (방법1, 방법2, 방법3) — 3그룹
- **종속변수 (연속형)**: 실기시험 점수
- `ols()` 회귀분석 결과로 F통계량을 구해 ANOVA를 수행하는 방법

### 🔬 가설

- **귀무가설 H₀**: 세 가지 교육방법에 따른 시험 점수에 차이가 **없다**
- **대립가설 H₁**: 세 가지 교육방법에 따른 시험 점수에 차이가 **있다**

### 💻 코드

```python
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/three_sample.csv"
)

# 이상치 제거 (score > 100 제외)
data = data.query("score <= 100")
print(len(data))  # 78

# 교차표: 교육방법별 건수 확인
data2 = pd.crosstab(index=data['method'], columns='count')
data2.index = ['방법1', '방법2', '방법3']
print(data2)

# 교차표: 교육방법별 만족/불만족
data3 = pd.crosstab(data['method'], data['survey'])
data3.index = ['방법1', '방법2', '방법3']
data3.columns = ['만족', '불만족']
print(data3)

# 일원분산분석 (ols + anova_lm 방식)
linreg = ols("data['score'] ~ data['method']", data=data).fit()
result = sm.stats.anova_lm(linreg, typ=1)
print(result)
#                 df    sum_sq      mean_sq     F        PR(>F)
# data['method']  1.0  27.98     27.98       0.122    0.727
# Residual       76.0  17398.13  228.92       NaN      NaN

# 사후검정 (Tukey HSD)
tukResult = pairwise_tukeyhsd(endog=data["score"], groups=data["method"])
print(tukResult)
#  group1 group2  meandiff  p-adj    lower    upper   reject
#       1      2    0.9725  0.9702  -8.9458  10.8909  False
#       1      3    1.4904  0.9363  -8.8183  11.799   False
#       2      3    0.5179  0.9918  -9.6125  10.6483  False

# 시각화
tukResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
```

### 📊 결과 해석

- p-value = **0.7276 > 0.05** → **귀무가설 채택**
- 세 가지 교육방법에 따른 시험 점수에 유의미한 차이가 없다.
- Tukey HSD에서도 모든 쌍의 `reject = False` → 집단 간 차이 없음

> [!note] anova_lm의 typ 파라미터
> 
> - `typ=1` : 순서형 제곱합 (Sequential SS) — 변수를 투입 순서대로 분해
> - `typ=2` : 계층형 제곱합 (Hierarchical SS) — 일반적으로 권장
> - `typ=3` : 편향되지 않은 제곱합 (Marginal SS) — 불균형 데이터에 사용

---

## test15a.py — 일원분산분석 (편의점 지역별 알바 급여)

### 📌 개념

- **독립변수**: 지역 그룹 (1, 2, 3)
- **종속변수**: 월급(pay)
- `f_oneway()`와 `anova_lm()` **두 가지 방법** 모두 사용
- `C(group)` — 범주형(Categorical) 변수임을 명시

### 🔬 가설

- **귀무가설 H₀**: GS편의점 3개 지역 알바생의 급여 평균에 차이가 **없다**
- **대립가설 H₁**: GS편의점 3개 지역 알바생의 급여 평균에 차이가 **있다**

### 💻 코드

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import urllib.request

uri = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3.txt"

# numpy로 읽기
data = np.genfromtxt(urllib.request.urlopen(uri), delimiter=",")
print(data.shape)  # (22, 2)

# 집단별 데이터 분리
gr1 = data[data[:, 1] == 1, 0]
gr2 = data[data[:, 1] == 2, 0]
gr3 = data[data[:, 1] == 3, 0]
print(np.mean(gr1))  # 316.625
print(np.mean(gr2))  # 256.444
print(np.mean(gr3))  # 278.0

# 정규성 검정
print(stats.shapiro(gr1).pvalue)  # 0.3336 > 0.05 → 정규성 만족
print(stats.shapiro(gr2).pvalue)  # 0.6561 > 0.05 → 정규성 만족
print(stats.shapiro(gr3).pvalue)  # 0.8324 > 0.05 → 정규성 만족

# 등분산성 검정
print(stats.levene(gr1, gr2, gr3).pvalue)   # 0.0458 < 0.05 → 등분산 불만족
print(stats.bartlett(gr1, gr2, gr3).pvalue) # 0.3508 > 0.05 → 등분산 만족

# 시각화
plt.boxplot([gr1, gr2, gr3], showmeans=True)
plt.show()

df = pd.DataFrame(data=data, columns=['pay', 'group'])

# 방법1: anova_lm()
lmodel = ols('pay ~ C(group)', data=df).fit()  # C(group): 범주형 명시
print(anova_lm(lmodel, typ=1))
# p = 0.043589 < 0.05 → 귀무가설 기각

# 방법2: f_oneway()
f_stat, p_val = stats.f_oneway(gr1, gr2, gr3)
print('f_stat :', f_stat)
print('p_val  :', p_val)  # 0.043589

# 사후 검정
tukResult = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
print(tukResult)

# 시각화
tukResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
```

### 📊 결과 해석

- p-value = **0.0436 < 0.05** → **귀무가설 기각**
- GS편의점 3개 지역 알바생의 급여 평균에 통계적으로 유의미한 차이가 있다.
- Tukey HSD로 어느 집단 사이에 차이가 있는지 확인

> [!tip] Levene vs Bartlett
> 
> - **Levene**: 이상치에 덜 민감, 분포 가정 없음 → 일반적으로 더 많이 사용
> - **Bartlett**: 정규 분포를 가정, 정규성이 확보된 경우 더 강력한 검정

> [!note] anova_lm vs f_oneway 비교
> 
> ||`anova_lm()`|`f_oneway()`|
> |---|---|---|
> |입력|OLS 회귀 모델|집단별 배열 직접 전달|
> |출력|상세 ANOVA 테이블 (df, SS, MS, F, p)|F통계량 + p값만|
> |활용|결과 테이블이 필요할 때|빠른 검정이 필요할 때|

---

## test16a.py — 일원분산분석 (온도 그룹별 매출액)

### 📌 개념

- **연속형 변수를 범주형으로 변환**: `pd.cut()`으로 온도를 3구간으로 나눔
- **정규성/등분산성 불만족 시 대안** 검정까지 포함한 완전한 흐름
- Kruskal-Wallis와 Welch's ANOVA 둘 다 실습

### 🔬 가설

- **귀무가설 H₀**: 온도에 따라 매출액 평균에 차이가 **없다**
- **대립가설 H₁**: 온도에 따라 매출액 평균에 차이가 **있다**

### 💻 코드

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
from pingouin import welch_anova
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 데이터 로드 및 병합 (test11t.py와 동일한 전처리)
sales_data = pd.read_csv(
    "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv",
    dtype={'YMD': 'object'}
)
wt_data = pd.read_csv(
    "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv"
)
wt_data.tm = wt_data.tm.map(lambda x: x.replace("-", ""))
frame = sales_data.merge(wt_data, how="left", left_on="YMD", right_on='tm')
data = frame.iloc[:, [0, 1, 7, 8]]  # YMD, AMT, maxTa, sumRn

# 연속형 온도 → 범주형 3그룹
# 0: 추움 (-5~8°C), 1: 보통 (8~24°C), 2: 더움 (24~37°C)
data['ta_gubun'] = pd.cut(data.maxTa, bins=[-5, 8, 24, 37], labels=[0, 1, 2])

# 집단별 데이터 분리
x1 = np.array(data[data.ta_gubun == 0].AMT)  # 추움
x2 = np.array(data[data.ta_gubun == 1].AMT)  # 보통
x3 = np.array(data[data.ta_gubun == 2].AMT)  # 더움

# 집단별 평균
np.set_printoptions(suppress=True)
print(np.mean(x1))  # 1032362.31
print(np.mean(x2))  #  818106.87
print(np.mean(x3))  #  553710.93

# 등분산성 검정
print(stats.levene(x1, x2, x3).pvalue)   # 0.039 < 0.05 → 등분산 불만족
print(stats.bartlett(x1, x2, x3).pvalue) # 0.009 < 0.05 → 등분산 불만족

# 정규성 검정
print(stats.shapiro(x1).pvalue)  # 0.248 > 0.05 → 만족
print(stats.shapiro(x2).pvalue)  # 0.038 < 0.05 → 불만족
print(stats.shapiro(x3).pvalue)  # 0.318 > 0.05 → 만족

# 기본 일원분산분석
print(stats.f_oneway(x1, x2, x3))
# statistic=99.19, pvalue=2.36e-34 → 귀무가설 기각

# 정규성 불만족 시 대안: Kruskal-Wallis (비모수)
print(stats.kruskal(x1, x2, x3))
# statistic=132.70, pvalue=1.53e-29 → 귀무가설 기각

# 등분산성 불만족 시 대안: Welch's ANOVA
print(welch_anova(dv="AMT", between='ta_gubun', data=data))
#      Source  ddof1   ddof2       F         p_unc     np2
# 0  ta_gubun     2  189.65  122.22  7.91e-35  0.379

# 사후 검정 (Tukey HSD)
spp = data.loc[:, ['AMT', 'ta_gubun']]
tukResult = pairwise_tukeyhsd(endog=spp['AMT'], groups=spp['ta_gubun'], alpha=0.05)
print(tukResult)

# 시각화
tukResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
```

### 📊 결과 해석

|검정 방법|통계량|p-value|결론|
|---|---|---|---|
|f_oneway (기본)|99.19|2.36e-34|귀무 기각|
|kruskal (비모수)|132.70|1.53e-29|귀무 기각|
|welch_anova|122.22|7.91e-35|귀무 기각|

모든 방법에서 **귀무가설 기각** → 온도에 따라 매출액 평균에 유의미한 차이가 있다.

---

## 사후검정 Tukey HSD

### 📌 개념

ANOVA로 "어딘가에 차이가 있다"는 것을 확인한 후, **구체적으로 어느 집단 사이에 차이가 있는지** 특정하는 검정.

- **Tukey HSD (Honestly Significant Difference)**
- 모든 집단 쌍을 비교
- `reject = True` → 해당 두 집단 사이에 유의미한 차이 **있음**
- `reject = False` → 해당 두 집단 사이에 유의미한 차이 **없음**

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukResult = pairwise_tukeyhsd(
    endog=df['score'],   # 종속변수 (수치형)
    groups=df['group'],  # 집단 변수 (범주형)
    alpha=0.05           # 유의수준 (기본값)
)
print(tukResult)
# group1 group2 meandiff p-adj   lower   upper  reject
#      1      2   -60.18  0.001  -95.3  -25.1   True   ← 차이 있음
#      1      3    -5.12  0.961  -40.3   30.1   False  ← 차이 없음
#      2      3    55.06  0.003   19.9   90.2   True   ← 차이 있음

# 시각화 (신뢰구간이 겹치면 차이 없음, 안 겹치면 차이 있음)
tukResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
```

> [!warning] Tukey HSD 주의사항 원래 각 집단의 **표본 수가 동일하다는 가정** 하에 고안된 방법. 집단 간 표본 수 차이가 크면 결과의 신뢰도가 떨어질 수 있음.

---

## ANOVA 핵심 개념 모음

### 📐 pd.cut() — 연속형 → 범주형 변환

```python
# bins: 구간 경계값, labels: 각 구간에 부여할 레이블
data['ta_gubun'] = pd.cut(
    data.maxTa,
    bins=[-5, 8, 24, 37],   # 4개 경계 → 3개 구간
    labels=[0, 1, 2]         # 추움, 보통, 더움
)
```

### 📐 주요 함수 요약

```python
# 기본 일원분산분석
stats.f_oneway(gr1, gr2, gr3)

# 정규성 불만족 → 비모수
stats.kruskal(gr1, gr2, gr3)

# 등분산성 불만족 → Welch's ANOVA
from pingouin import welch_anova
welch_anova(dv='종속변수명', between='집단변수명', data=df)

# 회귀 기반 ANOVA 테이블 출력
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
lmodel = ols('y ~ C(group)', data=df).fit()
anova_lm(lmodel, typ=1)

# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
pairwise_tukeyhsd(endog=df['y'], groups=df['group'], alpha=0.05)
```

### 📐 ANOVA 결과 테이블 읽는 법

```
              df      sum_sq     mean_sq       F       PR(>F)
C(group)     2.0    1234.56      617.28    5.23    0.008   ← 이 p값 확인
Residual    19.0    2241.44      117.97     NaN      NaN
```

- `df`: 자유도 (집단 수 - 1 / 전체 n - 집단 수)
- `sum_sq`: 제곱합
- `mean_sq`: 평균 제곱 = sum_sq / df
- `F`: F통계량 = 집단간 mean_sq / 집단내 mean_sq
- `PR(>F)`: p-value → **이 값이 0.05보다 작으면 귀무가설 기각**

### 📐 np2 (편향 제곱 에타, η²) — 효과 크기

Welch's ANOVA 결과에 포함되는 `np2` 값:

- 0.01 미만: 효과 작음
- 0.06 ~ 0.14: 효과 중간
- 0.14 이상: 효과 큼

---

