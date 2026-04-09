---
aliases:
---
# Day 39_통계 검정 정리 (카이제곱 & t-검정)

## 📅 2026-03-31

---
## 🗂️ 전체 흐름 한눈에 보기

```
통계 검정
├── 카이제곱 검정 (범주형 변수끼리)
│   ├── 독립성 검정 — "두 변수가 서로 관련 있나?"
│   └── 동질성 검정 — "두 집단의 분포가 같은가?"
│
└── t-검정 (수치형 변수의 평균 비교)
    ├── 단일 표본 t-검정 — "내 샘플 평균 == 알려진 값?"
    └── 독립 표본 t-검정 — "두 그룹의 평균이 같은가?"
```

---

## 🔑 핵심 개념

### 귀무가설 vs 대립가설

|용어|기호|뜻|
|---|---|---|
|**귀무가설**|H₀|차이 없다 / 관련 없다 (기본 가정)|
|**대립가설**|H₁|차이 있다 / 관련 있다 (주장)|

### 판정 기준

```
p-value < 0.05  →  귀무가설 기각  ✅ (통계적으로 유의미한 차이/관련)
p-value > 0.05  →  귀무가설 채택  ❌ (우연히 발생한 자료)
```

> 💡 **p-value** = 귀무가설이 맞다고 가정했을 때 현재 데이터가 우연히 나올 확률. 이 확률이 5% 미만이면 → "이건 우연이 아니다!" → 귀무가설 기각

---

## 📄 test5X2.py — 독립성 검정

> **독립성 검정** : 동일한 집단 내에서 두 범주형 변수가 서로 관련 있는가?

### 실습 1 — 교육 수준 vs 흡연율

- 귀무 : 교육 수준과 흡연율은 관련 없다
- 대립 : 교육 수준과 흡연율은 관련 있다

```python
import pandas as pd
import scipy.stats as stats

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/smoke.csv")
print(data.head(3))
print(data['education'].unique())   # [1:대학원졸 2:대졸 3:고졸]
print(data['smoking'].unique())     # [1:과흡연 2:보통 3:노담]

# 교차표(빈도표) 만들기
ctab = pd.crosstab(index=data['education'], columns=data['smoking'])
ctab.index   = ['대학원졸', '대졸', '고졸']
ctab.columns = ['과흡연', '보통', '노담']
print(ctab)

# 이원 카이제곱 검정
chi_result = [ctab.loc['대학원졸'], ctab.loc['대졸'], ctab.loc['고졸']]
chi2, p, dof, expected = stats.chi2_contingency(chi_result)

print(f"chi2:{chi2}, p:{p}, dof:{dof}")
# chi2:18.910915, p:0.000818257, dof:4
print("expected:\n", expected)    # 기대도수
```

**결과 해석**

- p = 0.000818 < 0.05 → **귀무가설 기각**
- ✅ 교육 수준과 흡연율 사이에 **관련이 있다**
- chi2(18.91) > 임계치(9.49) → 귀무가설 기각 (판정2)

---

### 실습 2 — 성별 vs 음료 선호도

- 귀무 : 성별과 음료 선호는 서로 관련 없다
- 대립 : 성별과 음료 선호는 서로 관련 있다

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

data = pd.DataFrame({
    '게토레이': [30, 20],
    '포카리':   [20, 30],
    '비타500':  [10, 30]
}, index=['남성', '여성'])

chi2, p, dof, expected = stats.chi2_contingency(data)

print('p-value :', p)        # 0.0033880
print('chi2 value :', chi2)  # 11.375
print('dof :', dof)
print('기대도수:\n', expected)

# 히트맵 시각화
sns.heatmap(data=data, annot=True, fmt='d', cmap="Blues")
plt.title('성별에 따른 음료 선호')
plt.xlabel('음료')
plt.ylabel('성별')
plt.show()
```

**결과 해석**

- p = 0.00339 < 0.05 → **귀무가설 기각**
- ✅ 성별과 음료 선호도는 **관련이 있다**

---

## 📄 test6X2.py — 동질성 검정

> **동질성 검정** : 두 집단 이상의 분포 비율이 서로 같은가?

| 독립성 검정 | 동질성 검정       |              |
| ------ | ------------ | ------------ |
| 집단 수   | 1개           | 2개 이상        |
| 질문     | 두 변수가 관련 있나? | 집단들의 분포가 같나? |

### 실습 1 — 교육 방법별 만족도

- 귀무 : 교육 방법에 따른 만족도 차이가 없다
- 대립 : 교육 방법에 따른 만족도 차이가 있다

```python
import pandas as pd
import scipy.stats as stats

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/survey_method.csv")
print(data['method'].unique())   # [1 2 3]
print(data['survey'].unique())   # [1 2 3 4 5]

ctab = pd.crosstab(index=data['method'], columns=data['survey'])
ctab.index   = ["방법1", "방법2", "방법3"]
ctab.columns = ['매우만족', '만족', '보통', '불만족', '매우불만족']
print(ctab)

chi2, p, dof, expected = stats.chi2_contingency(ctab)
print(f"chi2:{chi2}, p:{p}, dof:{dof}")
# chi2:6.544667, p:0.586457, dof:8
print("expected:\n", expected)
```

**결과 해석**

- p = 0.5864 > 0.05 → **귀무가설 채택**
- ✅ 교육 방법에 따른 만족도 차이가 **없다** (우연히 발생한 자료)

---

### 실습 2 — 연령대별 SNS 이용률

- 귀무 : 연령대별 SNS 서비스 이용률은 동일하다
- 대립 : 연령대별 SNS 서비스 이용률은 동일하지 않다

```python
import pandas as pd
import scipy.stats as stats

data = pd.read_csv("sns_data.csv")
print(data["age"].unique())      # [1 2 3]
print(data["service"].unique())  # ['F' 'T' 'K' 'C' 'E']

ctab2 = pd.crosstab(index=data["age"], columns=data["service"])
chi2, p, dof, expected = stats.chi2_contingency(ctab2)
print(f"chi2:{chi2}, p:{p}, dof:{dof}")
# chi2:102.752, p:1.168e-18, dof:8
# p < 0.05 → 귀무가설 기각 : 연령대별 이용률은 동일하지 않다

# 전체 1439건 중 500건 샘플링 후 재검정
print("전체 건수:", len(data))
samp_data = data.sample(n=500, replace=True, random_state=1)

ctab3 = pd.crosstab(index=samp_data["age"], columns=samp_data["ser"])
chi2, p, dof, expected = stats.chi2_contingency(ctab3)
print(f"chi2:{chi2}, p:{p}, dof:{dof}")
```

---

## 📄 test7t.py / test8t.py — 단일 표본 t-검정 기초

> **단일 표본 t-검정** : 하나의 집단 표본 평균이 알려진 모수(기준값)와 같은가?

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 귀무 : 해당 집단의 평균 키가 177이다
# 대립 : 해당 집단의 평균 키가 177이 아니다
one_sample = [167.0, 182.7, 169.6, 176.8, 185.0]
print(np.array(one_sample).mean())  # 176.22

result = stats.ttest_1samp(one_sample, popmean=177)
print(result)
# statistic=-0.221, pvalue=0.836, df=4
# p=0.836 > 0.05 → 귀무가설 채택 : 평균 키는 177이다

result2 = stats.ttest_1samp(one_sample, popmean=165)
print(result2)
# statistic=3.185, pvalue=0.034
# p=0.034 < 0.05 → 귀무가설 기각 : 평균 키는 165가 아니다

sns.displot(one_sample, bins=10, kde=True)
plt.xlabel('data')
plt.ylabel('value')
plt.show()
```

---

## 📄 test8tquiz.py / test9t.py — 단일 표본 t-검정 실전

### 문제 1 — 새 전구 수명 300시간 검정

- 귀무 : 새 전구의 평균 수명은 300시간이다
- 대립 : 새 전구의 평균 수명은 300시간이 아니다

```python
import numpy as np
import scipy.stats as stats

data = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]

print('표본 평균:', np.mean(data))   # 291.21
print('표본 크기:', len(data))       # 14

# ① 정규성 검정 (t-검정 전 필수!)
result = stats.shapiro(data)
print(result)
# p > 0.05 → 정규성 만족 → t-검정 진행 가능

# ② 단일 표본 t-검정
t_result = stats.ttest_1samp(data, popmean=300)
print(t_result)
```

---

### 문제 2 — 노트북 평균 사용 시간 5.2시간 검정

- 귀무 : A회사 노트북의 평균 사용 시간은 5.2시간이다
- 대립 : A회사 노트북의 평균 사용 시간은 5.2시간이 아니다

```python
import pandas as pd
import scipy.stats as stats

data2 = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv")

# 전처리
data2['time'] = data2['time'].replace("     ", "").str.strip()  # 공백 제거
data2['time'] = pd.to_numeric(data2['time'], errors='coerce')   # 숫자 변환
data2 = data2.dropna(subset=['time'])                            # null 제거

print('표본 평균:', data2['time'].mean())
print('표본 크기:', len(data2))

result2 = stats.shapiro(data2['time'])        # 정규성 검정
t_result2 = stats.ttest_1samp(data2['time'], popmean=5.2)
print(t_result2)
```

---

### 문제 3 — 전국 미용 요금 15,000원 검정

- 귀무 : 전국 평균 미용 요금은 15,000원이다
- 대립 : 전국 평균 미용 요금은 15,000원이 아니다

```python
import numpy as np
import pandas as pd
import scipy.stats as stats

# pip install xlrd  (xls 파일 읽기용)
data3 = pd.read_excel('2026.02_data.xls')

data4 = data3.iloc[0, 2:]                       # 지역 데이터만 추출
data4 = pd.to_numeric(data4, errors='coerce')
data4 = data4.dropna()

print('표본 평균 미용 요금:', data4.mean())
print('표본 크기:', len(data4))

result3 = stats.shapiro(data4)                   # 정규성 검정
t_result3 = stats.ttest_1samp(data4, popmean=15000)
print(t_result3)
```

---

## 📄 test10t.py — 독립 표본 t-검정

> **독립 표본 t-검정** : 서로 다른 두 집단의 평균이 같은가?

- 귀무 : 두 교육 방법의 평균 시험 점수에 차이가 없다
- 대립 : 두 교육 방법의 평균 시험 점수에 차이가 있다

```python
from scipy import stats
from scipy.stats import levene
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/two_sample.csv")

# 집단 분리
ms = data[['method', 'score']]
m1 = ms[ms['method'] == 1]['score']   # 방법 1
m2 = ms[ms['method'] == 2]['score']   # 방법 2

# NaN 처리
print(m1.isnull().sum())   # 0
print(m2.isnull().sum())   # 2
m2 = m2.fillna(m2.mean())  # NaN → 평균으로 대체

# ① 정규성 검정
print('score1:', stats.shapiro(m1))   # p=0.368 > 0.05 → 정규성 만족
print('score2:', stats.shapiro(m2))   # p=0.671 > 0.05 → 정규성 만족

# 시각화
sns.histplot(m1, kde=True)
sns.histplot(m2, kde=True, color='blue')
plt.show()

# ② 등분산성 검정
print('등분산성:', levene(m1, m2).pvalue)   # 0.457 > 0.05 → 등분산 만족

# ③ 독립 표본 t-검정
result = stats.ttest_ind(m1, m2, equal_var=True)   # 등분산 → True
print('result:', result)
# statistic=-0.196, pvalue=0.845, df=48
# p=0.845 > 0.05 → 귀무가설 채택 : 두 방법 점수 차이 없다
```

### 독립 표본 t-검정 순서

```
① 정규성 검정 (shapiro)
   └─ p > 0.05 → 정규분포 만족

② 등분산성 검정 (levene)
   ├─ p > 0.05 → 등분산 → equal_var=True
   └─ p < 0.05 → 이분산 → equal_var=False (Welch t-test)

③ ttest_ind() 수행 → p-value 판정
```

---

## 📄 test11t.py — 독립 표본 t-검정 (날씨 & 매출)

> 강수 여부에 따라 음식점 매출 평균에 차이가 있는가?

- 귀무 : 강수 여부에 따라 매출액 평균에 차이가 없다
- 대립 : 강수 여부에 따라 매출액 평균에 차이가 있다

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
    dtype={'YMD': 'object'}
)

# 날씨 데이터 읽기
wt_data = pd.read_csv(
    "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv"
)

# 날짜 형식 맞추기 (2018-06-01 → 20180601)
wt_data.tm = wt_data.tm.map(lambda x: x.replace('-', ''))

# 두 데이터 병합
frame = sales_data.merge(wt_data, how="left", left_on="YMD", right_on='tm')
data = frame.iloc[:, [0, 1, 7, 8]]   # YMD, AMT, avgTa, sumRn

# 강수 여부 컬럼 추가 (비 오면 1, 맑으면 0)
data['rain_yn'] = (data['sumRn'] > 0) * 1

# 집단 분리
sp = np.array(data.iloc[:, [1, 4]])   # AMT, rain_yn
tg1 = sp[sp[:, 1] == 0, 0]           # 맑은 날 매출
tg2 = sp[sp[:, 1] == 1, 0]           # 비 온 날 매출

print('맑은날 매출 평균:', np.mean(tg1))
print('비온날 매출 평균:', np.mean(tg2))

# 박스플롯 시각화
plt.boxplot([tg1, tg2], meanline=True, showmeans=True, notch=True)
plt.show()
```

---

## 🔄 검정 선택 가이드

```
변수 종류로 검정 방법 선택하기

두 변수 모두 범주형?
  └─ 카이제곱 검정
      ├─ 집단 1개 → 독립성 검정
      └─ 집단 2개 이상 → 동질성 검정

독립변수=범주형, 종속변수=수치형?
  └─ t-검정
      ├─ 집단 1개 (알려진 값과 비교) → 단일 표본 t-검정
      └─ 집단 2개 비교 → 독립 표본 t-검정
```

---

## 📦 자주 쓰는 함수 모음

```python
from scipy import stats
from scipy.stats import levene

# 카이제곱 검정
chi2, p, dof, expected = stats.chi2_contingency(교차표)

# 교차표 만들기
ctab = pd.crosstab(index=df['A'], columns=df['B'])

# 정규성 검정
statistic, p = stats.shapiro(데이터)

# 등분산성 검정
result = levene(집단1, 집단2)

# 단일 표본 t-검정
result = stats.ttest_1samp(데이터, popmean=기준값)

# 독립 표본 t-검정
result = stats.ttest_ind(집단1, 집단2, equal_var=True/False)
```

---

## ✏️ 오늘의 핵심 요약

- **p < 0.05** → 귀무가설 **기각** ✅ (유의미한 차이/관련)
- **p > 0.05** → 귀무가설 **채택** ❌ (우연에 의한 자료)
- 카이제곱 → 범주형 vs 범주형
- t-검정 → 범주형 vs 수치형 (평균 비교)
- t-검정 전 **정규성 검정** → **등분산성 검정** 순서 필수!

---

_📁 파일 목록: test5X2.py · test6X2.py · test7t.py · test8t.py · test8tquiz.py · test9t.py · test10t.py · test11t.py_