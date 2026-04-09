# Day 34_데이터 시각화 & DB 연동

## 📅 2026-03-24

---

# 📈 plot1.py — Matplotlib 기초

> **핵심 주제:** matplotlib을 이용한 기본 선 그래프, subplot, 스타일 지정, 그래프 저장

---

## 🔧 기본 세팅

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='malgun gothic')      # Windows 한글 폰트 (Mac: AppleGothic)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 기호 깨짐 방지
```

> ⚠️ 한글이 포함된 그래프를 그릴 때는 반드시 폰트 설정을 먼저 해야 깨지지 않는다.

---

## 1. 기본 선 그래프

```python
x = ("서울", "인천", "수원")   # tuple 사용 (list도 가능, set은 순서 없어 비권장)
y = [5, 3, 7]

plt.xlim([-1, 3])               # x축 범위 설정
plt.ylim([0, 10])               # y축 범위 설정
plt.yticks(list(range(0, 11, 3)))  # y축 눈금 직접 지정 (0, 3, 6, 9)
plt.plot(x, y)
plt.show()
```

**포인트**

- `xlim`, `ylim` : 축의 표시 범위를 강제 지정
- `yticks` : 눈금(tick) 라벨을 원하는 값으로 직접 설정
- x에 `list`, `tuple`은 순서가 보장되지만 `set`은 순서가 없으므로 그래프에 부적합

---

## 2. numpy 배열로 그래프 + 값 라벨 표시

```python
data = np.arange(1, 11, 2)   # [1, 3, 5, 7, 9]
plt.plot(data)                # x축은 자동으로 [0,1,2,3,4] 설정됨

x = [0, 1, 2, 3, 4]
for a, b in zip(x, data):
    plt.text(a, b, str(b))    # 각 데이터 포인트 위에 값을 텍스트로 표시
plt.show()
```

**포인트**

- `np.arange(start, stop, step)` : 등간격 배열 생성
- x 없이 `plt.plot(data)` 하면 x는 자동으로 인덱스(0, 1, 2...)로 설정됨
- `plt.text(x, y, 문자열)` : 특정 좌표에 텍스트 라벨 추가

---

## 3. 선 스타일 & 마커 지정

```python
x = np.arange(10)
y = np.sin(x)

# plt.plot(x, y)               # 기본 선
# plt.plot(x, y, 'bo')         # 파란 원형 마커
plt.plot(x, y, 'go--', linewidth=2, markersize=12)
plt.show()
```

**포맷 문자열 규칙 `'go--'`**

|문자|의미|
|---|---|
|`g`|색상: green|
|`o`|마커: 원형|
|`--`|선 스타일: 점선|

**자주 쓰는 색상:** `r`(red), `g`(green), `b`(blue), `k`(black), `y`(yellow)  
**자주 쓰는 마커:** `o`(원), `s`(사각), `^`(삼각), `+`(플러스), `*`(별)  
**선 스타일:** `-`(실선), `--`(점선), `-.`(점선혼합), `:`(도트)

---

## 4. 여러 그래프 겹치기 (Hold) + 범례 & 레이블

```python
x = np.arange(0, np.pi * 3, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize=(10, 5))     # 그래프 전체 크기 지정 (width, height 인치)
plt.plot(x, y_sin, 'r')         # 선 그래프
plt.scatter(x, y_cos)           # 산점도
plt.xlabel('x 축')              # x축 레이블
plt.ylabel('y 축')              # y축 레이블
plt.title('sine & cosine')      # 제목
plt.legend(['sine', 'cosine'])  # 범례 (입력 순서대로 매칭)
plt.show()
```

**포인트**

- `plt.figure(figsize=(w, h))` : 그래프 캔버스 크기를 인치 단위로 지정
- `plt.plot()` + `plt.scatter()` 를 연달아 호출하면 같은 Figure에 겹쳐 그려짐 (hold 동작)
- `plt.legend()` : 리스트 순서대로 각 plot에 범례 이름 부여

---

## 5. Subplot — 하나의 Figure에 여러 그래프

```python
plt.subplot(2, 1, 1)    # (행수, 열수, 현재 위치) → 2행 1열 중 첫 번째
plt.plot(x, y_sin)
plt.title('sine')

plt.subplot(2, 1, 2)    # 2행 1열 중 두 번째
plt.plot(x, y_cos)
plt.title('cosine')

plt.show()
```

**포인트**

- `plt.subplot(rows, cols, index)` : 전체를 rows×cols 격자로 나누고 index번째 칸 선택
- index는 1부터 시작, 왼쪽→오른쪽, 위→아래 순서

---

## 6. grid, legend 위치, 다중 선 그래프

```python
irum = ['a', 'b', 'c', 'd', 'e']
kor  = [80, 50, 70, 70, 90]
eng  = [60, 70, 80, 90, 100]

plt.plot(irum, kor, 'ro-')      # 빨간 원형 실선
plt.plot(irum, kor, 'bo--')     # 파란 원형 점선
plt.ylim([50, 100])
plt.title('시험 점수')
plt.legend(['국어', '영어'], loc=4)  # loc=4 : 우측 하단 / loc='best' : 자동
plt.grid(True)                       # 격자 표시
```

**legend loc 값**

|값|위치|
|---|---|
|0 / `'best'`|자동(최적)|
|1|우측 상단|
|2|좌측 상단|
|3|좌측 하단|
|4|우측 하단|

---

## 7. 그래프 저장 & 이미지로 불러오기

```python
fig = plt.gcf()         # 현재 Figure 객체 가져오기 (get current figure)
plt.show()
fig.savefig('plot1.png')  # 파일로 저장 (png, jpg, pdf 등 지원)

# 저장한 이미지를 다시 불러와 출력
from matplotlib.pyplot import imread
img = imread('plot1.png')
plt.imshow(img)
plt.show()
```

**포인트**

- `plt.show()` 이후에도 `fig` 객체를 통해 저장 가능
- `savefig()`는 `show()` 이전에 호출해야 빈 이미지가 저장되지 않음 (순서 주의)
- `imshow()` : 이미지 배열(numpy array)을 그래프 영역에 표시

---

## 📌 이 파일의 핵심 요약

| 함수                              | 역할          |
| ------------------------------- | ----------- |
| `plt.plot()`                    | 선 그래프       |
| `plt.scatter()`                 | 산점도         |
| `plt.text()`                    | 좌표에 텍스트 표시  |
| `plt.xlim()` / `plt.ylim()`     | 축 범위 설정     |
| `plt.yticks()`                  | y축 눈금 직접 지정 |
| `plt.xlabel()` / `plt.ylabel()` | 축 레이블       |
| `plt.title()`                   | 그래프 제목      |
| `plt.legend()`                  | 범례          |
| `plt.grid()`                    | 격자 표시       |
| `plt.subplot()`                 | 여러 그래프 분할   |
| `plt.figure(figsize=)`          | 캔버스 크기 지정   |
| `fig.savefig()`                 | 파일로 저장      |
| `plt.imshow()`                  | 이미지 출력      |

---
# 📊 plot2.py — 다양한 차트 종류 & 인터페이스 스타일

> **핵심 주제:** Matplotlib의 두 가지 인터페이스 방식 비교, 다양한 차트 종류, 시계열 시각화

---

## 🔧 기본 세팅

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## 1. 두 가지 인터페이스 스타일 비교

### 방식 1 — Matplotlib 스타일 (절차적)

```python
x = np.arange(10)

plt.figure()
plt.subplot(2, 1, 1)    # row, column, panel number
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.show()
```

### 방식 2 — 객체 지향(OOP) 스타일 (권장)

```python
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
plt.show()
```

**비교 정리**

|구분|Matplotlib 스타일|객체 지향 스타일|
|---|---|---|
|방식|`plt.subplot()` 호출|`fig, ax = plt.subplots()`|
|접근|전역 상태 기반 (암묵적)|ax 객체 직접 지정 (명시적)|
|권장|간단한 작업|복잡한 다중 그래프, 실무 권장|

> 💡 실무에서는 **객체 지향 스타일**이 더 명시적이고 유지보수에 유리하여 권장된다.

---

## 2. 히스토그램 + 선 그래프 (subplot 활용)

```python
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)   # 1행 2열 중 첫 번째
ax2 = fig.add_subplot(1, 2, 2)   # 1행 2열 중 두 번째

ax1.hist(np.random.randn(1000), bins=100, alpha=0.9)  # 히스토그램
ax2.plot(np.random.rand(1000))                        # 선 그래프
plt.show()
```

**포인트**

- `fig.add_subplot(rows, cols, index)` : figure에 subplot 추가하는 OOP 방식
- `bins` : 히스토그램 막대 개수
- `alpha` : 투명도 (0.0 투명 ~ 1.0 불투명)

---

## 3. 막대 그래프 (bar / barh)

```python
data = [50, 80, 100, 90, 70]

# 세로 막대
plt.bar(range(len(data)), data)
plt.show()

# 가로 막대 + 오차 막대
err = np.random.rand(len(data))
plt.barh(range(len(data)), data, alpha=0.4, xerr=err)
plt.show()
```

**포인트**

- `plt.bar(x, height)` : 세로 막대 그래프
- `plt.barh(y, width)` : 가로 막대 그래프
- `xerr` / `yerr` : 오차 막대(error bar) 추가
- `alpha` : 투명도 조절로 겹침 시 가시성 향상

---

## 4. 원 그래프 (Pie Chart)

```python
data = [50, 80, 100, 90, 70]

plt.pie(
    data,
    colors=['yellow', 'blue', 'red'],   # 각 조각 색상
    explode=(0, 0.2, 0, 0.1, 0)         # 조각을 중심에서 분리하는 거리
)
plt.title('Pie Chart')
plt.show()
```

**포인트**

- `explode` : 튜플로 각 조각별 분리 거리 지정 (0이면 붙어있음)
- `colors` : 데이터 수보다 적으면 반복 적용됨

---

## 5. 박스플롯 (Box Plot)

```python
data = [1, 50, 80, 100, 90, 70, 300]
plt.boxplot(data)
plt.show()
```

**박스플롯 구성 요소**

```
       ─────   ← 최댓값 (이상치 제외)
         │
    ┌────┤
    │    │     ← Q3 (75th percentile)
    │ 박스 │
    │    │     ← 중앙값 (median, Q2)
    │    │
    └────┤     ← Q1 (25th percentile)
         │
       ─────   ← 최솟값 (이상치 제외)
       
       ○       ← 이상치 (outlier)
```

> 💡 박스플롯은 **데이터 전체 분포 파악 + 이상치 확인**에 매우 효과적이다.

---

## 6. 버블 차트 (Scatter + 크기 변수)

```python
n = 30
np.random.seed(0)   # 재현 가능한 난수를 위한 시드 고정
x     = np.random.rand(n)
y     = np.random.rand(n)
color = np.random.rand(n)
scale = np.pi * (np.random.rand(n) * 15) ** 2   # 점 면적 계산

plt.scatter(x, y, c=color, s=scale)
plt.show()
```

**포인트**

- `c` : 색상 (배열로 넘기면 colormap 적용)
- `s` : 마커 크기 (면적 기준, 배열로 넘기면 버블 차트)
- `np.random.seed(0)` : 동일한 난수 결과를 재현하기 위한 시드 설정

---

## 7. 시계열 데이터 시각화

```python
import pandas as pd

fdata = pd.DataFrame(
    np.random.randn(1000, 4),
    index=pd.date_range('1/1/2000', periods=1000),  # 날짜 인덱스
    columns=list('abcd')
)
print(fdata.head(3))
print(fdata.tail(3))

fdata = fdata.cumsum()   # 누적합 → 시계열 추세처럼 만들기
print(fdata.head(3))

plt.plot(fdata)
plt.show()
```

**포인트**

- `pd.date_range(start, periods)` : 일정 간격의 날짜 인덱스 생성
- `cumsum()` : 누적합 (cumulative sum) — 각 행까지의 합을 계산
- DataFrame을 그대로 `plt.plot()`에 넘기면 각 열이 하나의 선으로 표시됨

---

## 8. Pandas 내장 plot 기능

```python
fdata.plot()              # 선 그래프 (기본)
fdata.plot(kind='bar')    # 막대 그래프
plt.xlabel("time")
plt.ylabel("data")
plt.show()
```

**`kind` 옵션 목록**

|값|차트 종류|
|---|---|
|`'line'`|선 그래프 (기본)|
|`'bar'`|세로 막대|
|`'barh'`|가로 막대|
|`'hist'`|히스토그램|
|`'box'`|박스플롯|
|`'scatter'`|산점도|
|`'pie'`|원 그래프|

> 💡 pandas의 `.plot()`은 내부적으로 matplotlib을 사용하므로 `plt.xlabel()` 등 추가 설정이 그대로 적용된다.

---

## 📌 이 파일의 핵심 요약

|차트|함수|주요 용도|
|---|---|---|
|히스토그램|`ax.hist()`|데이터 분포 확인|
|세로 막대|`plt.bar()`|범주별 값 비교|
|가로 막대|`plt.barh()`|범주별 값 비교 (가로)|
|원 그래프|`plt.pie()`|비율 표현|
|박스플롯|`plt.boxplot()`|분포 + 이상치 확인|
|버블 차트|`plt.scatter(s=)`|3변수 관계 표현|
|시계열|`pd.DataFrame.plot()`|시간 흐름에 따른 추세|

---
# 🎨 plot3.py — Seaborn & IQR 이상치 처리

> **핵심 주제:** Seaborn 라이브러리 소개 및 주요 차트, IQR 기반 이상치 탐지 및 제거, 박스플롯으로 전후 비교

---

## 🔧 기본 세팅

```python
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

import seaborn as sns
```

---

## 1. Seaborn이란?

- `matplotlib`의 기능을 **보충**하는 고수준 시각화 라이브러리
- 더 아름다운 **기본 스타일** 제공
- **통계 시각화**에 특화 (분포, 관계, 범주형 등)
- pandas DataFrame과 매우 잘 연동됨

---

## 2. 타이타닉 데이터셋 로드

```python
titanic = sns.load_dataset("titanic")
print(titanic.info(max_cols=None))   # 모든 컬럼 정보 출력
```

**타이타닉 주요 컬럼**

|컬럼|설명|
|---|---|
|`age`|나이|
|`sex`|성별|
|`class`|객실 등급 (First, Second, Third)|
|`survived`|생존 여부 (0: 사망, 1: 생존)|
|`fare`|운임|

---

## 3. 분포도 — displot

```python
sns.displot(titanic['age'])
plt.title("나이 차트")
plt.show()
```

**포인트**

- `displot()` : 데이터의 **분포(distribution)** 를 히스토그램 + KDE 곡선으로 표시
- 단일 변수의 전체적인 분포 파악에 적합

---

## 4. 박스플롯 — boxplot

```python
sns.boxplot(y='age', data=titanic, palette="Paired")
plt.show()
```

**포인트**

- `palette` : 색상 팔레트 지정 (`"Paired"`, `"Set1"`, `"pastel"` 등)
- `y=` : 세로 방향으로 나이 분포 표시
- 이상치(outlier)가 점으로 표시됨

---

## 5. 관계 산점도 — relplot

```python
sns.relplot(x='sex', y='age', data=titanic)
plt.show()
```

**포인트**

- `relplot()` : 두 변수 간의 **관계(relation)** 를 산점도로 표시
- `x`에 범주형 변수를 넣으면 그룹별 분포 확인 가능

---

## 6. 히트맵 — heatmap (pivot_table 연계)

```python
# pivot_table : 교차 집계표 생성
titanic_pivot = titanic.pivot_table(
    index='class',     # 행: 객실 등급
    columns='sex',     # 열: 성별
    aggfunc='size'     # 집계 함수: 데이터 개수
)
print(titanic_pivot)

sns.heatmap(
    titanic_pivot,
    cmap=sns.light_palette("gray"),  # 색상 팔레트
    annot=True,                      # 각 셀에 값 표시
    fmt="d"                          # 정수 형식으로 표시
)
plt.show()
```

**포인트**

- `pivot_table()` : index × columns 조합의 집계 테이블 생성
- `aggfunc='size'` : 해당 조합의 데이터 건수를 집계
- `annot=True` : 셀 안에 숫자 값 표시
- `fmt="d"` : 정수 포맷 (`"f"` 는 소수, `".1f"` 는 소수점 1자리)

---

## 7. IQR 기반 이상치 탐지 & 제거

### 7-1. 데이터 정의

```python
data = [10, 12, 13, 15, 14, 12, 11, 100]   # 100이 이상치
df = pd.DataFrame({'score': data})
```

### 7-2. IQR 계산

```python
Q1 = df['score'].quantile(0.25)   # 1사분위수 (하위 25%)
Q3 = df['score'].quantile(0.75)   # 3사분위수 (상위 75%)
IQR = Q3 - Q1                     # 사분위 범위
```

**IQR(Interquartile Range) 개념**

```
|-------Q1-------중앙값-------Q3-------|
   ↑ 하위 25%                상위 25% ↑

정상 범위: Q1 - 1.5×IQR  ~  Q3 + 1.5×IQR
이 범위를 벗어나는 값 → 이상치(Outlier)
```

### 7-3. 이상치 경계 계산

```python
lower_bound = Q1 - 1.5 * IQR   # 하한선
upper_bound = Q3 + 1.5 * IQR   # 상한선
```

### 7-4. 이상치 & 정상치 분리

```python
# 이상치 추출 (하한선 미만 OR 상한선 초과)
outliers = df[(df['score'] < lower_bound) | (df['score'] > upper_bound)]

# 정상치만 필터링
filtered_df = df[(df['score'] >= lower_bound) & (df['score'] <= upper_bound)]

print("이상치 값:")
print(outliers)
```

---

## 8. 이상치 제거 전후 박스플롯 비교

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 이상치 포함 박스플롯
sns.boxplot(y=df['score'], ax=axes[0], color='salmon')
axes[0].set_title('이상치 포함 데이터')
axes[0].set_ylabel('Score')
axes[0].grid(True)

# 이상치 제거 후 박스플롯
sns.boxplot(y=filtered_df['score'], ax=axes[1], color='lightblue')
axes[1].set_title('이상치 제거 후')
axes[1].set_ylabel('Score')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

**포인트**

- `plt.subplots(1, 2)` : 1행 2열로 두 그래프를 나란히 배치
- `ax=axes[0]` : 특정 subplot 위치에 그래프를 그릴 때 `ax` 파라미터로 지정
- `plt.tight_layout()` : subplot 간 겹침 방지, 자동 여백 조정
- `color=` : seaborn boxplot의 단색 지정

---

## 📌 이 파일의 핵심 요약

|함수|역할|
|---|---|
|`sns.displot()`|단변수 분포도|
|`sns.boxplot()`|박스플롯 (분포 + 이상치)|
|`sns.relplot()`|두 변수 관계 산점도|
|`sns.heatmap()`|히트맵|
|`df.pivot_table()`|교차 집계표 생성|
|`quantile(0.25/0.75)`|사분위수 계산|

> **IQR 이상치 탐지 공식**
> 
> - 하한 = Q1 - 1.5 × IQR
> - 상한 = Q3 + 1.5 × IQR
> - 이 범위를 벗어나는 값 → 이상치

---
# 🌸 plot4iris.py — Iris 데이터셋 시각화

> **핵심 주제:** 실제 데이터셋(Iris)을 활용한 산점도, 색상 분류, scatter_matrix, seaborn pairplot

---

## 🔧 기본 세팅

```python
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline    # jupyter notebook에서 인라인 출력 시 사용
```

---

## 1. Iris 데이터셋 개요

```python
iris_data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/iris.csv")

print(iris_data.info())      # 컬럼 정보, 데이터 타입, null 여부
print(iris_data.head(3))     # 상위 3행
print(iris_data.tail(3))     # 하위 3행
```

**데이터셋 구성**

|항목|내용|
|---|---|
|행 수|150행|
|종류|3가지 (`setosa`, `versicolor`, `virginica`)|
|특성 수|4개|

**컬럼 설명**

|컬럼|설명|
|---|---|
|`Sepal.Length`|꽃받침 길이|
|`Sepal.Width`|꽃받침 너비|
|`Petal.Length`|꽃잎 길이|
|`Petal.Width`|꽃잎 너비|
|`Species`|꽃 종류 (목표 변수)|

---

## 2. 기본 산점도

```python
plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'])
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')    # ⚠️ 코드에서 ylabel이 xlabel로 잘못 쓰여 있음
plt.title('iris data')
plt.show()
```

> ⚠️ 원본 코드에 `plt.xlabel('Petal.Length')` 로 잘못 작성된 부분이 있음. 정확히는 `plt.ylabel('Petal.Length')` 이어야 한다.

---

## 3. 종류별 색상 구분 산점도

### 고유 종류 확인

```python
print(iris_data['Species'].unique())
# 출력: ['setosa' 'versicolor' 'virginica']
```

### 종류별 색상 코드 매핑

```python
cols = []   # 꽃의 종류에 따라 다른 색 번호를 저장
for s in iris_data['Species']:
    choice = 0
    if s == 'setosa':
        choice = 1
    elif s == 'versicolor':
        choice = 2
    elif s == 'virginica':
        choice = 3
    cols.append(choice)
```

**포인트**

- `unique()` : 컬럼의 고유 값 목록 반환
- 각 종류에 숫자(1, 2, 3)를 부여해서 `c` 파라미터에 넘기면 colormap으로 자동 색상 구분

### 색상 구분 산점도 출력

```python
plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'], c=cols)
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.title('iris data')
plt.show()
```

**포인트**

- `c=cols` : 각 데이터 포인트의 색상을 리스트로 지정
- 숫자 배열을 넘기면 matplotlib이 자동으로 colormap을 적용하여 색상을 구분해줌

---

## 4. Scatter Matrix — 모든 변수 쌍 산점도 (pandas)

```python
from pandas.plotting import scatter_matrix

iris_col = iris_data.loc[:, 'Sepal.Length':'Petal.Length']   # Sepal.Length ~ Petal.Length 열 선택
scatter_matrix(iris_col, diagonal='kde')
plt.show()
```

**포인트**

- `scatter_matrix()` : 모든 컬럼 쌍의 산점도를 한 번에 그리는 행렬형 차트
- `diagonal='kde'` : 대각선(같은 변수)에는 KDE(커널 밀도 추정) 곡선 표시
    - `diagonal='hist'` 로 바꾸면 히스토그램으로 표시
- `loc[:, 'A':'B']` : 열 범위로 슬라이싱 (A열부터 B열까지)

---

## 5. Pairplot — seaborn으로 더 예쁜 산점도 행렬

```python
import seaborn as sns

sns.pairplot(iris_data, hue='Species', height=3)
plt.show()
```

**포인트**

- `pairplot()` : scatter_matrix의 seaborn 버전, 더 직관적이고 보기 좋음
- `hue='Species'` : 종류별로 색상을 구분하여 표시 (범례 자동 생성)
- `height` : 각 subplot의 크기 (인치)
- 대각선에는 기본적으로 KDE 곡선이 그려짐

**scatter_matrix vs pairplot 비교**

|구분|scatter_matrix (pandas)|pairplot (seaborn)|
|---|---|---|
|라이브러리|pandas|seaborn|
|hue(색상 분류)|직접 구현 필요|`hue=` 파라미터로 간단 지원|
|디자인|기본 matplotlib 스타일|더 세련된 스타일|
|권장|빠른 확인|발표/보고용|

---

## 📌 이 파일의 핵심 요약

|함수|역할|
|---|---|
|`pd.read_csv(url)`|URL에서 CSV 직접 로드|
|`df['col'].unique()`|고유 값 목록 확인|
|`plt.scatter(c=)`|색상 구분 산점도|
|`scatter_matrix()`|모든 변수 쌍 산점도 행렬 (pandas)|
|`sns.pairplot(hue=)`|색상 구분 산점도 행렬 (seaborn)|

> **Iris 데이터셋은 머신러닝 분류 문제의 대표적인 예제 데이터셋으로, 시각화 실습에 자주 사용된다.**

---
# 🚲 plot5bike.py — 자전거 공유 시스템 EDA

> **핵심 주제:** 실제 kaggle 데이터셋을 이용한 탐색적 데이터 분석(EDA), datetime 파생변수 생성, barplot / boxplot / regplot 시각화

---

## 🔧 기본 세팅

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import koreanize_matplotlib     # 한글 폰트 자동 설정 라이브러리

plt.style.use('ggplot')         # ggplot 스타일 적용 (R의 ggplot2 스타일)
```

> 💡 `koreanize_matplotlib` : 한 줄로 한글 폰트 설정을 자동으로 처리해주는 라이브러리  
> 💡 `plt.style.use('ggplot')` : 그래프 전체에 ggplot 스타일 테마 적용

---

## 1. 데이터셋 개요

**출처:** Kaggle — Bike Sharing in Washington D.C. Dataset (일부 수정)

```python
train = pd.read_csv(
    "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv",
    parse_dates=['datetime']    # datetime 컬럼을 날짜 타입으로 자동 파싱
)
```

**컬럼 설명**

|컬럼|설명|
|---|---|
|`datetime`|일시|
|`season`|계절 (봄=1, 여름=2, 가을=3, 겨울=4)|
|`holiday`|공휴일(1) / 평일(0)|
|`workingday`|근무일(1) / 비근무일(0)|
|`weather`|날씨 (맑음=1, 안개=2, 눈·비=3, 폭우=4)|
|`temp`|섭씨 온도|
|`atemp`|체감 온도|
|`humidity`|습도|
|`windspeed`|풍속|
|`casual`|비회원 대여량|
|`registered`|회원 대여량|
|`count`|총 대여량 (목표 변수)|

---

## 2. EDA — 탐색적 데이터 분석 (기본 확인)

```python
pd.set_option('display.width', None)    # 출력 너비 제한 해제

print(train.info())             # 컬럼 정보, 데이터 타입, null 수
print(train.dtypes)             # 각 컬럼의 데이터 타입
print(train.shape)              # (10886, 12) — 행, 열 수
print(train.columns)            # 컬럼명 목록
print(train.head(3))            # 상위 3행
print(train.temp.describe())    # 온도 컬럼 기술 통계량
print(train.isnull().sum())     # 컬럼별 결측치(NaN) 수
```

**포인트**

- `info()` : 전체 데이터 구조 한눈에 파악
- `describe()` : count, mean, std, min, 25%, 50%, 75%, max 요약 제공
- `isnull().sum()` : 결측치 여부 확인 → 전처리 필요 여부 판단

---

## 3. datetime 파생 변수 생성

```python
train['year']   = train['datetime'].dt.year     # 연도
train['month']  = train['datetime'].dt.month    # 월
train['day']    = train['datetime'].dt.day      # 일
train['hour']   = train['datetime'].dt.hour     # 시
train['minute'] = train['datetime'].dt.minute   # 분
train['second'] = train['datetime'].dt.second   # 초

print(train.head(1))
print(train.columns)
```

**포인트**

- `parse_dates=['datetime']` 로 로드했기 때문에 `.dt` 연산자 사용 가능
- `.dt.year`, `.dt.month` 등으로 날짜의 각 부분을 별도 컬럼으로 추출
- 파생변수를 만들면 **년/월/일/시간별 분석이 가능**해짐

---

## 4. 시간 단위별 대여량 시각화 — barplot

```python
figure, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)
figure.set_size_inches(15, 5)   # 전체 그림 크기 설정

sns.barplot(data=train, x='year',  y='count', ax=ax1)
sns.barplot(data=train, x='month', y='count', ax=ax2)
sns.barplot(data=train, x='day',   y='count', ax=ax3)
sns.barplot(data=train, x='hour',  y='count', ax=ax4)

ax1.set(ylabel='대여수', title='년도별 대여')
ax2.set(ylabel='월',     title='월별 대여')
ax3.set(ylabel='일',     title='일별 대여')
ax4.set(ylabel='시간',   title='시간별 대여')

plt.show()
```

**포인트**

- `sns.barplot()` : 기본적으로 평균값과 신뢰구간(95%)을 함께 표시
- `ax=ax1` : 특정 subplot에 그래프를 그릴 위치 지정 (OOP 스타일)
- `ax.set(ylabel=, title=)` : 축 레이블과 제목을 한 번에 설정

---

## 5. 분포 확인 — boxplot

```python
fig, axes = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(12, 10)

sns.boxplot(data=train, y="count",                  orient="v", ax=axes[0][0])
sns.boxplot(data=train, y="count", x="season",      orient="v", ax=axes[0][1])
sns.boxplot(data=train, y="count", x="hour",        orient="v", ax=axes[1][0])
sns.boxplot(data=train, y="count", x="workingday",  orient="v", ax=axes[1][1])

axes[0][0].set(ylabel='대여수',  title='대여')
axes[0][1].set(xlabel='계절',    ylabel='대여수', title='계절별 대여량')
axes[1][0].set(xlabel='시간',    ylabel='대여수', title='시간별 대여량')
axes[1][1].set(xlabel='근무일',  ylabel='대여수', title='근무일에 따른 대여량')

plt.show()
```

**포인트**

- `x=` 에 범주형 변수를 넣으면 그룹별 박스플롯이 나란히 표시됨
- `orient="v"` : 세로 방향 (`"h"` 는 가로)
- `axes[row][col]` : 2차원 인덱스로 subplot 위치 접근

---

## 6. 연속 변수와 대여량의 관계 — regplot

```python
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
figure.set_size_inches(12, 5)

sns.regplot(x='temp',      y='count', data=train, ax=ax1)
sns.regplot(x='humidity',  y='count', data=train, ax=ax2)
sns.regplot(x='windspeed', y='count', data=train, ax=ax3)

plt.show()
```

**포인트**

- `regplot()` : 산점도 + **회귀선(regression line)** + 신뢰구간을 함께 표시
- 연속 변수와 목표 변수 간의 **선형 관계 여부** 파악에 유용
- 온도↑ → 대여량↑, 습도↑ → 대여량↓ 같은 패턴을 시각적으로 확인 가능

---

## 📌 이 파일의 핵심 요약

|함수|역할|
|---|---|
|`pd.read_csv(parse_dates=)`|CSV 로드 + 날짜 타입 자동 변환|
|`.dt.year` / `.dt.month` 등|datetime 파생변수 추출|
|`sns.barplot()`|범주별 평균 + 신뢰구간 막대 그래프|
|`sns.boxplot(x=그룹)`|그룹별 분포 비교 박스플롯|
|`sns.regplot()`|산점도 + 회귀선|
|`plt.style.use('ggplot')`|전체 그래프 스타일 테마 변경|

> **EDA(탐색적 데이터 분석)의 흐름:**
> 
> 1. 데이터 로드 → 2. 기본 정보 확인 → 3. 결측치 확인 → 4. 파생변수 생성 → 5. 시각화로 패턴 발견

---

# 🗃️ pandasdb1.py — Pandas + SQLite DB 연동 기초

> **핵심 주제:** SQLite 인메모리 DB 생성, 데이터 삽입/조회, DB 결과를 DataFrame으로 변환, DataFrame을 DB에 저장

---

## 🔧 기본 세팅

```python
import sqlite3
```

---

## 1. SQLite 인메모리 DB 생성 & 테이블 만들기

```python
sql = "create table if not exists extab(product varchar(10), maker varchar(10), weight real, price integer)"

conn = sqlite3.connect(':memory:')  # 인메모리 DB 생성 (프로그램 종료 시 사라짐)
conn.execute(sql)
conn.commit()
```

**포인트**

- `sqlite3.connect(':memory:')` : 파일이 아닌 **메모리에만 존재하는 임시 DB** 생성
    - 파일로 저장하려면 `sqlite3.connect('파일명.db')` 사용
- `create table if not exists` : 테이블이 없을 때만 생성 (중복 오류 방지)
- `conn.commit()` : 변경 사항을 DB에 확정 반영 (DML 이후 필수)

**컬럼 타입**

|컬럼|타입|설명|
|---|---|---|
|`product`|`varchar(10)`|문자열 최대 10자|
|`maker`|`varchar(10)`|문자열 최대 10자|
|`weight`|`real`|실수 (소수점 포함)|
|`price`|`integer`|정수|

---

## 2. 데이터 삽입

### executemany — 여러 행 한 번에 삽입

```python
data = [('mouse', 'samsung', 12.5, 5000), ('keyboard', 'lg', 52.5, 35000)]
isql = "insert into extab values(?,?,?,?)"   # ? : 플레이스홀더 (바인딩 파라미터)

conn.executemany(isql, data)   # 리스트의 각 튜플을 순서대로 삽입
conn.commit()
```

### execute — 단일 행 삽입

```python
data1 = ('pen', 'abc', 5.0, 1200)
conn.execute(isql, data1)
conn.commit()
```

**포인트**

- `?` 플레이스홀더 : SQL 인젝션 방지 + 코드 가독성 향상
- `executemany()` : 동일한 SQL에 여러 데이터를 반복 실행할 때 사용
- `execute()` : 단일 실행

---

## 3. 데이터 조회

```python
cursor = conn.execute("select * from extab")
rows = cursor.fetchall()    # 모든 결과 행을 리스트로 반환

for a in rows:
    print(a)
```

**포인트**

- `conn.execute(SQL)` : SQL 실행 후 cursor 반환
- `cursor.fetchall()` : 전체 결과를 `[(행1), (행2), ...]` 형태의 리스트로 반환
- `cursor.fetchone()` : 결과 중 첫 번째 행 하나만 반환

---

## 4. DB 결과 → DataFrame 변환

### 방법 1 : fetchall() 결과를 직접 DataFrame으로

```python
import pandas as pd

df1 = pd.DataFrame(rows, columns=['product', 'maker', 'weight', 'price'])
print(df1)
print(df1.describe())   # 수치형 컬럼의 기술 통계량
```

### 방법 2 : read_sql() 사용 (권장)

```python
df2 = pd.read_sql("select * from extab", conn)
print(df2)

# SQL 집계 함수도 바로 DataFrame으로
print(pd.read_sql("select count(*) as 건수 from extab", conn))
```

**비교**

|방법|특징|
|---|---|
|`pd.DataFrame(rows, columns=)`|fetchall() 결과를 수동으로 변환, 컬럼명 직접 지정 필요|
|`pd.read_sql(sql, conn)`|SQL 실행 + DataFrame 변환을 한 번에 처리, 권장 방식|

---

## 5. DataFrame → DB 저장 (to_sql)

```python
data = {
    'product': ['연필', '볼펜', '지우개'],
    'maker':   ['모나미', '모나미', '모나미'],
    'weight':  [2.3, 3.0, 5.0],
    'price':   (1000, 2000, 500)    # tuple도 가능
}
frame = pd.DataFrame(data)
print(frame)

frame.to_sql("extab", conn, if_exists='append', index=False)
```

**`if_exists` 옵션**

|값|동작|
|---|---|
|`'fail'`|테이블이 이미 존재하면 오류 발생 (기본값)|
|`'replace'`|기존 테이블 삭제 후 새로 생성하여 저장|
|`'append'`|기존 테이블에 데이터 추가 (행 삽입)|

**포인트**

- `index=False` : DataFrame의 인덱스(0, 1, 2...)를 DB에 저장하지 않음
- `to_sql()` 이후 추가된 데이터 확인:

```python
df3 = pd.read_sql("select * from extab", conn)
print(df3)   # 총 6행 (영어 3개 + 한글 3개)
```

---

## 6. 연결 종료

```python
cursor.close()
conn.close()
```

> ⚠️ DB 작업이 끝나면 반드시 `cursor`와 `conn`을 닫아야 리소스가 해제된다.

---

## 📌 이 파일의 핵심 요약

|함수/메서드|역할|
|---|---|
|`sqlite3.connect(':memory:')`|인메모리 DB 연결|
|`conn.execute(sql)`|단일 SQL 실행|
|`conn.executemany(sql, data)`|여러 행 일괄 삽입|
|`conn.commit()`|변경사항 확정|
|`cursor.fetchall()`|전체 조회 결과 반환|
|`pd.DataFrame(rows, columns=)`|조회 결과 → DataFrame|
|`pd.read_sql(sql, conn)`|SQL 실행 + DataFrame 변환|
|`df.to_sql(table, conn, if_exists=)`|DataFrame → DB 저장|
|`cursor.close()` / `conn.close()`|리소스 해제|

---

# 🗃️ pandasdb2.py — Pandas + SQLite DB 연동 심화

> **핵심 주제:** pandasdb1.py와 동일한 흐름으로 SQLite 연동을 복습하고, `to_sql`을 통한 DataFrame → DB 저장을 확실히 익힌다.

---

## 🔧 기본 세팅

```python
import sqlite3
```

---

## 전체 코드 흐름

pandasdb2.py는 pandasdb1.py와 **동일한 코드 구조**로 구성되어 있으며, 핵심 개념을 반복 학습하기 위한 파일이다.

---

## 1. 인메모리 DB 생성 & 테이블 생성

```python
sql = "create table if not exists extab(product varchar(10), maker varchar(10), weight real, price integer)"

conn = sqlite3.connect(':memory:')
conn.execute(sql)
conn.commit()
```

- `':memory:'` : 디스크에 저장되지 않는 임시 DB. 프로그램 종료 시 사라짐
- `if not exists` : 이미 테이블이 있으면 오류 없이 넘어감

---

## 2. 데이터 삽입 — executemany & execute

```python
data = [('mouse', 'samsung', 12.5, 5000), ('keyboard', 'lg', 52.5, 35000)]
isql = "insert into extab values(?,?,?,?)"

conn.executemany(isql, data)   # 리스트의 각 튜플을 순서대로 삽입

data1 = ('pen', 'abc', 5.0, 1200)
conn.execute(isql, data1)      # 단일 행 삽입

conn.commit()
```

- `?` : 바인딩 파라미터. 값을 직접 문자열로 넣지 않고 안전하게 바인딩
- `executemany` : 반복 삽입을 한 번의 호출로 처리

---

## 3. 데이터 조회

```python
cursor = conn.execute("select * from extab")
rows = cursor.fetchall()

for a in rows:
    print(a)
```

- `fetchall()` : 결과 전체를 `[(튜플1), (튜플2), ...]` 로 반환

---

## 4. DB 결과 → DataFrame

```python
import pandas as pd

# 방법 1: fetchall 결과를 수동으로 DataFrame 변환
df1 = pd.DataFrame(rows, columns=['product', 'maker', 'weight', 'price'])
print(df1)
print(df1.describe())   # 수치형 컬럼의 기술 통계 (count, mean, std, min, max 등)

# 방법 2: read_sql로 바로 변환 (권장)
df2 = pd.read_sql("select * from extab", conn)
print(df2)

# 집계 쿼리도 바로 DataFrame으로
print(pd.read_sql("select count(*) as 건수 from extab", conn))
```

**`describe()` 출력 항목 정리**

|항목|설명|
|---|---|
|`count`|데이터 개수 (null 제외)|
|`mean`|평균|
|`std`|표준편차|
|`min`|최솟값|
|`25%`|1사분위수|
|`50%`|중앙값|
|`75%`|3사분위수|
|`max`|최댓값|

---

## 5. DataFrame → DB 저장 (to_sql)

```python
data = {
    'product': ['연필', '볼펜', '지우개'],
    'maker':   ['모나미', '모나미', '모나미'],
    'weight':  [2.3, 3.0, 5.0],
    'price':   (1000, 2000, 500)
}
frame = pd.DataFrame(data)
print(frame)

frame.to_sql("extab", conn, if_exists='append', index=False)
```

**`to_sql()` 파라미터**

|파라미터|설명|
|---|---|
|`"extab"`|저장할 테이블명|
|`conn`|DB 연결 객체|
|`if_exists='append'`|기존 테이블에 이어서 삽입|
|`index=False`|DataFrame 인덱스를 DB에 저장하지 않음|

### 저장 결과 확인

```python
df3 = pd.read_sql("select * from extab", conn)
print(df3)
# 총 6행: mouse, keyboard, pen + 연필, 볼펜, 지우개
```

---

## 6. 연결 종료

```python
cursor.close()
conn.close()
```

---

## 📌 DB ↔ DataFrame 전체 흐름 정리

```
[SQLite DB]
    │
    │  conn.execute() / executemany()
    ▼
[데이터 삽입/조회]
    │
    │  pd.read_sql()          cursor.fetchall() + pd.DataFrame()
    ▼                         ▼
[DataFrame]  ←────────────────────────────
    │
    │  df.to_sql(if_exists='append')
    ▼
[SQLite DB에 저장]
```

---

## 📌 pandasdb1 vs pandasdb2 비교

> 두 파일은 **동일한 코드 내용**으로 구성되어 있다.  
> SQLite + pandas 연동 흐름을 반복 실습하기 위한 목적으로 사용된 파일이다.  
> 핵심 개념은 pandasdb1.md를 함께 참고할 것.

---

## 📌 이 파일의 핵심 요약

|함수/메서드|역할|
|---|---|
|`sqlite3.connect(':memory:')`|인메모리 SQLite DB 생성|
|`conn.executemany(sql, list)`|여러 행 일괄 삽입|
|`conn.execute(sql, tuple)`|단일 행 삽입 or SQL 실행|
|`cursor.fetchall()`|전체 결과 리스트로 반환|
|`pd.DataFrame(rows, columns=)`|조회 결과를 수동으로 DataFrame 변환|
|`pd.read_sql(sql, conn)`|SQL 실행 + 바로 DataFrame으로 반환|
|`df.describe()`|수치형 컬럼 기술 통계량 출력|
|`df.to_sql(name, conn, if_exists=, index=)`|DataFrame → DB 테이블에 저장|
|`cursor.close()` / `conn.close()`|DB 리소스 해제|
