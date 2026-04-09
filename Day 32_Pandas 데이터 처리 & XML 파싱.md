# Day 32_Pandas 데이터 처리 & XML 파싱

## 📅 2026-03-20

---
## 📚 목차

1. [[#1. DataFrame 재구조화 — `stack` / `unstack`|DataFrame 재구조화]]
2. [[#2. 범주화 — `pd.cut` / `pd.qcut`|범주화 — pd.cut / pd.qcut]]
3. [[#3. `groupby` + `agg` — 그룹별 집계 연산|groupby + agg]]
4. [[#4. `merge` — DataFrame 병합 (SQL JOIN)|merge — DataFrame 병합]]
5. [[#5. `concat` — DataFrame 단순 연결|concat — DataFrame 단순 연결]]
6. [[#6. `pivot` / `pivot_table` — 피벗 테이블|pivot / pivot_table]]
7. [[#Pandas File I/O & Chunk 처리|File I/O & Chunk 처리]]
8. [[#Pandas DataFrame 저장 & 출력 (File Output)|DataFrame 저장 & 출력]]
9. [[#Pandas + openpyxl 엑셀 스타일링|openpyxl 엑셀 스타일링]]
10. [[#Pandas 실습 문제 — 타이타닉 & 데이터 분석|실습 문제 — 타이타닉]]
11. [[#Python XML 파싱 — ElementTree & 기상청 데이터|XML 파싱 — ElementTree]]

---

# Pandas DataFrame 재구조화, 범주화, 병합

## 1. DataFrame 재구조화 — `stack` / `unstack`

> **개념**
> 
> - `stack()` : **열(column) → 행(row)** 방향으로 접어 MultiIndex Series 생성
> - `unstack()` : **행(row) → 열(column)** 방향으로 펼쳐 원래 형태로 복원 데이터를 넓은 형태(wide format) ↔ 긴 형태(long format)로 자유롭게 변환할 때 사용

```python
import pandas as pd
import numpy as np

# 2행 3열 DataFrame 생성 (값: 1000~1005)
# reshape(2,3) : 6개 요소를 2행 3열로 변환
df = pd.DataFrame(
    1000 + np.arange(6).reshape(2, 3),
    index=['대전', '서울'],       # 행 이름 (도시)
    columns=['2020', '2021', '2022']  # 열 이름 (연도)
)
print(df)
```

```
    2020  2021  2022
대전  1000  1001  1002
서울  1003  1004  1005
```

```python
# stack() : 열 → 행 (Wide → Long 변환)
# 연도 열이 행 인덱스의 두 번째 레벨로 이동 → (도시, 연도) MultiIndex Series
df_row = df.stack()
print(df_row)
```

```
대전  2020    1000
    2021    1001
    2022    1002
서울  2020    1003
    2021    1004
    2022    1005
dtype: int64
```

```python
# unstack() : 행 → 열 (Long → Wide 변환)
# stack()의 역연산 → 원본 DataFrame으로 복원
df_col = df_row.unstack()
print(df_col)
```

```
    2020  2021  2022
대전  1000  1001  1002
서울  1003  1004  1005
```

> 💡 **핵심 포인트** `unstack()`은 `stack()`의 역연산으로 원본 DataFrame과 동일하게 복원됨 `stack(level=-1)` 처럼 level 인자로 어느 축을 이동할지 지정 가능

---

## 2. 범주화 — `pd.cut` / `pd.qcut`

> **개념** 연속형(수치) 데이터를 **구간(bin)으로 나눠 범주형(Categorical) 데이터**로 변환
> 
> - `pd.cut()` : **직접 구간 경계값**을 지정 (값 기준 분할)
> - `pd.qcut()` : **분위수(quantile) 기준**으로 균등 분할 (빈도 기준 분할)

### 2-1. `pd.cut` — 값 범위 기준 분할

```python
price = [10.3, 5.5, 7.8, 3.6]
cut = [3, 7, 9, 11]   # 구간 경계값 → (3,7], (7,9], (9,11] 3개 구간 생성
                      # 경계값 개수 - 1 = 구간 개수 (4개 경계 → 3개 구간)

# pd.cut : 각 값이 어느 구간에 속하는지 범주형으로 반환
result_cut = pd.cut(price, cut)
print(result_cut)
```

```
[(9, 11], (3, 7], (7, 9], (3, 7]]
Categories (3, interval[int64, right]): [(3, 7] < (7, 9] < (9, 11]]
```

```python
# value_counts() : 각 구간별 데이터 개수 집계
print(result_cut.value_counts())
```

```
(3, 7]     2
(7, 9]     1
(9, 11]    1
Name: count, dtype: int64
```

> 💡 **구간 표기법** : `(` = 해당 값 미포함(열린 구간), `]` = 해당 값 포함(닫힌 구간) 예) `(3, 7]` = 3 초과 7 이하

### 2-2. `pd.qcut` — 분위수(빈도) 기준 분할

```python
datas = pd.Series(np.arange(1, 1001))   # 1~1000 숫자 시리즈

# head(n) : 앞 n개, tail(n) : 뒤 n개 확인 (기본값 5)
print(datas.head(3))
print(datas.tail(2))
```

```
0    1
1    2
2    3
dtype: int64

998     999
999    1000
dtype: int64
```

```python
# qcut(data, q) : data를 q등분 — 각 구간의 데이터 개수가 최대한 균등하도록 경계 자동 결정
# cut과 달리 경계값을 직접 지정하지 않음
result_cut2 = pd.qcut(datas, 3)
print(result_cut2)
```

```
0       (0.999, 334.0]
1       (0.999, 334.0]
...
999    (667.0, 1000.0]
Length: 1000, dtype: category
Categories (3, interval[float64, right]): [(0.999, 334.0] < (334.0, 667.0] < (667.0, 1000.0]]
```

```python
print(result_cut2.value_counts())
```

```
(0.999, 334.0]     334
(334.0, 667.0]     333
(667.0, 1000.0]    333
Name: count, dtype: int64
```

💡 **cut vs qcut 비교**

|구분|`pd.cut`|`pd.qcut`|
|---|---|---|
|분할 기준|값의 범위(구간 경계 직접 지정)|데이터 분위수(빈도 균등)|
|각 구간 데이터 수|불균등 가능|최대한 균등|
|사용 시점|의미 있는 경계값이 있을 때|균등 분할이 필요할 때|

---

## 3. `groupby` + `agg` — 그룹별 집계 연산

> **개념** `groupby(기준)` 으로 데이터를 그룹화한 뒤, `agg()`로 **여러 집계 함수를 한 번에** 적용 SQL의 `GROUP BY` + 집계 함수와 동일한 개념

### 3-1. `agg` 내장 함수 사용

```python
# groupby(기준) : 기준 값이 같은 것끼리 묶음
# result_cut2 의 구간 레이블을 기준으로 datas 그룹화
group_col = datas.groupby(result_cut2)

# agg(['함수명', ...]) : 여러 집계 함수를 한 번에 적용
print(group_col.agg(['count', 'mean', 'std', 'min']))
```

```
                 count   mean        std  min
(0.999, 334.0]     334  167.5  96.561725    1
(334.0, 667.0]     333  501.0  96.273049  335
(667.0, 1000.0]    333  834.0  96.273049  668
```

### 3-2. 사용자 정의 함수 + `apply`

```python
# apply용 함수 : 그룹(gr)을 받아 dict 형태로 결과 반환
# agg보다 자유도가 높아 복잡한 계산에 적합
def summaryFunc(gr):
    return {
        'count': gr.count(),   # 데이터 개수
        'mean':  gr.mean(),    # 평균
        'std':   gr.std(),     # 표준편차
        'min':   gr.min()      # 최솟값
    }

# apply(함수) : 각 그룹에 사용자 함수 적용 → Long 형태로 반환
print(group_col.apply(summaryFunc))
```

```
(0.999, 334.0]   count    334.000000
                 mean     167.500000
                 std       96.561725
                 min        1.000000
...
```

```python
# unstack() : Long → Wide 변환으로 보기 좋게 정리
print(group_col.apply(summaryFunc).unstack())
```

```
                 count   mean        std    min
(0.999, 334.0]   334.0  167.5  96.561725    1.0
(334.0, 667.0]   333.0  501.0  96.273049  335.0
(667.0, 1000.0]  333.0  834.0  96.273049  668.0
```

> 💡 `agg` vs `apply`
> 
> - `agg` : 내장 함수명(문자열) 또는 함수 객체 리스트를 직접 전달, 빠르고 간결
> - `apply` : 그룹 전체를 인자로 받는 사용자 함수, 더 유연하지만 느릴 수 있음

---

## 4. `merge` — DataFrame 병합 (SQL JOIN)

> **개념** 두 DataFrame을 **공통 키(key) 열**을 기준으로 합치는 연산 SQL의 JOIN과 동일한 개념으로, `how` 인자로 조인 방식 결정

```python
df1 = pd.DataFrame({
    'data1': range(7),
    'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b']
})
df2 = pd.DataFrame({
    'key': ['a', 'b', 'd'],
    'data2': range(3)
})
```

### 4-1. INNER JOIN (교집합, 기본값)

```python
# on='key' : 두 df에서 공통으로 사용할 키 열 지정
# how 생략 시 기본값 = 'inner' → 양쪽 모두 존재하는 key만 포함
print(pd.merge(df1, df2, on='key'))
print(pd.merge(df1, df2, on='key', how='inner'))   # 동일 결과
```

```
   data1 key  data2
0      0   b      1
1      1   b      1
2      2   a      0
3      4   a      0
4      5   a      0
5      6   b      1
```

> key = 'c', 'd' 는 양쪽에 모두 없으므로 제외됨

### 4-2. OUTER JOIN (합집합)

```python
# how='outer' : 어느 한쪽에만 있어도 포함, 없는 값은 NaN
print(pd.merge(df1, df2, on='key', how='outer'))
```

```
   data1 key  data2
...
6    3.0   c    NaN   ← df1에만 있음
7    NaN   d    2.0   ← df2에만 있음
```

### 4-3. LEFT / RIGHT JOIN

```python
# how='left' : 왼쪽(df1) 전체 유지, df2에 없으면 NaN
print(pd.merge(df1, df2, on='key', how='left'))

# how='right' : 오른쪽(df2) 전체 유지, df1에 없으면 NaN
print(pd.merge(df1, df2, on='key', how='right'))
```

### 4-4. 공통 열 이름이 없는 경우 (`left_on` / `right_on`)

```python
df3 = pd.DataFrame({'key2': ['a', 'b', 'd'], 'data2': range(3)})

# left_on : 왼쪽 df의 키 열 이름
# right_on : 오른쪽 df의 키 열 이름 (서로 다를 때 각각 지정)
print(pd.merge(df1, df3, left_on='key', right_on='key2'))
```

💡 **JOIN 방식 요약**

|`how`|설명|결과 행|
|---|---|---|
|`inner`|양쪽 모두 key 존재|교집합만|
|`outer`|어느 한쪽에만 있어도 포함|합집합 (없으면 NaN)|
|`left`|왼쪽 df 전체 유지|왼쪽 기준|
|`right`|오른쪽 df 전체 유지|오른쪽 기준|

---

## 5. `concat` — DataFrame 단순 연결

> **개념** `merge`처럼 키를 기준으로 매핑하는 것이 아니라, 단순히 **행 또는 열 방향으로 이어 붙이는** 연산
> 
> - `axis=0` (기본값) : 행 방향으로 세로 연결 (위아래 합치기)
> - `axis=1` : 열 방향으로 가로 연결 (좌우 합치기)

```python
# axis=0 : 행 단위로 아래에 붙이기
# 한쪽에만 있는 열은 NaN으로 자동 채워짐
print(pd.concat([df1, df3], axis=0))
```

```
   data1  key key2  data2
0    0.0    b  NaN    NaN   ← df1에는 key2, data2 없음
...
0    NaN  NaN    a    0.0   ← df3에는 data1, key 없음
```

```python
# axis=1 : 열 단위로 옆에 붙이기
# 행 수가 다르면 짧은 쪽은 NaN으로 채워짐
print(pd.concat([df1, df3], axis=1))
```

> 💡 **merge vs concat**
> 
> - `merge` : 공통 key 기반 정렬·매핑 후 병합 → 관계형 조인
> - `concat` : 단순 이어 붙이기 → 없는 값은 NaN으로 채움

---

## 6. `pivot` / `pivot_table` — 피벗 테이블

> **개념**
> 
> - `pivot()` : 특정 두 열을 행/열 인덱스로 사용해 **데이터 재배치** (중복 불가)
> - `pivot_table()` : `pivot` + `groupby`의 혼합 — **중복 값을 집계 함수로 처리** 가능

```python
data = {
    'city': ['강남', '강북', '강남', '강북'],
    'year': [2000, 2001, 2002, 2002],
    'pop':  [3.3,  2.5,  3.0,  2.0]
}
df = pd.DataFrame(data)
```

### 6-1. `pivot()` — 단순 재배치

```python
# index : 행 인덱스로 사용할 열
# columns : 열 인덱스로 사용할 열
# values : 셀 값으로 사용할 열
# (city, year) 조합이 중복되면 오류 → 중복 없을 때만 사용 가능
print(df.pivot(index='city', columns='year', values='pop'))
```

```python
# set_index(['a','b']).unstack() 으로 pivot과 동일한 결과 구현 가능
print(df.set_index(['city', 'year']).unstack())
```

### 6-2. `describe()` — 기술통계

```python
# count/mean/std/min/25%/50%/75%/max 를 한번에 출력
print(df['pop'].describe())
```

### 6-3. `pivot_table()` — 집계 포함 피벗

```python
# aggfunc 기본값 = 'mean' → city별 수치 열 평균 자동 계산
print(df.pivot_table(index=['city']))

# aggfunc에 리스트 전달 → 여러 집계 함수 동시 적용
print(df.pivot_table(index=['city', 'year'], aggfunc=[len, 'mean']))

# values : 집계할 열 지정, aggfunc=len → 행 개수 세기
print(df.pivot_table(values='pop', index='city', aggfunc=len))
```

### 6-4. `margins` — 소계/합계 행·열 추가

```python
# margins=True : 행/열 끝에 전체 합계(All) 행·열 자동 추가
print(df.pivot_table(values='pop', index=['year'], columns=['city'], margins=True))

# fill_value=0 : NaN을 0으로 대체
print(df.pivot_table(values='pop', index=['year'], columns=['city'],
                     margins=True, fill_value=0))
```

---

## 7. `groupby` — 그룹 기반 집계

```python
# groupby(['열이름']) : 해당 열의 값이 같은 행끼리 묶음
# 결과는 항상 Long 형태 (pivot_table과 다름)
print(df.groupby(['city']).sum())
print(df.groupby(['city']).mean())
```

> 💡 **pivot_table vs groupby**
> 
> - `groupby` : 연산 결과가 항상 **Long 형태**
> - `pivot_table` : 결과를 바로 **Wide(크로스탭) 형태**로 반환, 시각화에 유리

---

## 📌 전체 개념 요약

|기능|메서드|핵심 역할|
|---|---|---|
|형태 변환|`stack()`|열 → 행 (Wide → Long)|
|형태 변환|`unstack()`|행 → 열 (Long → Wide)|
|범주화|`pd.cut()`|구간 경계 직접 지정|
|범주화|`pd.qcut()`|분위수 기준 균등 분할|
|그룹 집계|`groupby().agg()`|내장/사용자 집계 함수 적용|
|병합(키 기반)|`pd.merge()`|SQL JOIN, 키 열 매핑|
|단순 연결|`pd.concat()`|행/열 방향 이어 붙이기|
|재배치|`pivot()`|두 열을 행/열 인덱스로 재구성|
|재배치+집계|`pivot_table()`|중복 처리 + 소계 추가 가능|

---

# Pandas File I/O & Chunk 처리

---

## 📂 파일 읽기 기초

### [1] `read_csv` — 기본 읽기

```python
import pandas as pd
import numpy as np

# read_csv : CSV(쉼표 구분) 파일을 DataFrame으로 읽음
# 첫 행을 자동으로 헤더(열 이름)로 인식
df = pd.read_csv('ex1.csv')
print(df, type(df))
```

```
   a   b   c   d message
0  1   2   3   4   hello
1  5   6   7   8   world
2  9  10  11  12     foo
<class 'pandas.core.frame.DataFrame'>
```

---

### [2] `read_table` — 구분자 직접 지정

```python
# read_table : 기본 구분자가 탭(\t) → CSV 읽으려면 sep=',' 명시 필요
df = pd.read_table('ex1.csv', sep=',')

# skip_blank_lines=True (기본값) : 칼럼명·데이터 앞의 빈 줄 자동 무시
df = pd.read_table('ex1.csv', sep=',', skip_blank_lines=True)
print(df)
```

---

## 🌐 URL에서 읽기 + header / names 옵션

```python
# display.max_columns : 열이 많아 중간이 ... 으로 생략되는 것을 방지
pd.set_option('display.max_columns', None)

url2 = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv'
```

### [3] 기본 읽기

```python
# GitHub 파일은 반드시 raw.githubusercontent.com 주소 사용
df = pd.read_csv(url2)
print(df)
```

### [4] `header=None` — 모든 행을 데이터로 처리

```python
# header=None : 파일에 헤더가 없다고 간주
# ⚠️ 파일에 헤더가 있는데 이 옵션을 쓰면 헤더 행도 데이터로 읽힘
df = pd.read_csv(url2, header=None)
print(df)
```

### [5] `skiprows=1`

```python
# skiprows=N : 앞에서 N개 행을 건너뛰고 읽기 시작
df = pd.read_csv(url2, header=None, skiprows=1)
print(df)
```

### [6] `names` — 열 이름 직접 지정

```python
# header=None + names : 헤더 행도 데이터로 포함 + 새 이름 적용
df = pd.read_csv(url2, header=None, names=['a', 'b', 'c', 'd', 'e'])
print(df)
```

---

## 📄 공백 구분자 파일 읽기

### [7] `read_csv` — 공백 파일을 쉼표로 읽으면?

```python
df = pd.read_csv(url3)
print(df)   # 한 열에 모든 내용이 뭉쳐서 출력됨
```

### [8] `read_table(sep=r'\s+')` — 공백 구분자

```python
# r'\s+' 반드시 raw string 사용 → '\s+' 로 쓰면 SyntaxWarning 발생
df = pd.read_table(url3, sep=r'\s+')
print(df)
print(df.iloc[:, 0])
```

### [9] `skiprows=[1, 3]` — 특정 행 번호 건너뛰기

```python
# 리스트로 전달 시 해당 행 번호만 선택적으로 제외
df = pd.read_table(url3, sep=r'\s+', skiprows=[1, 3])
print(df)
```

---

## 📏 `read_fwf` — 고정폭 파일 읽기

```python
# read_fwf : 구분자 없이 열 너비가 고정된 파일 읽기
# widths=(10, 3, 5) : 각 열의 문자 너비
df = pd.read_fwf(url_fwf, header=None,
                 widths=(10, 3, 5),
                 names=('data', 'name', 'price'),
                 encoding='utf8')
print(df)
print(df.iloc[:, 0])
print(df['data'])
```

---

## 🧱 Chunk 처리 — 대용량 파일 분할 읽기

> 대용량 파일을 한 번에 메모리에 올리지 않고, **일정 크기(chunk)씩 나눠서 처리**하는 방식.

|구분|내용|
|---|---|
|장점|메모리 절약, 대용량 파일 처리 가능, 분산처리 효과|
|단점|여러 번 반복 읽기 → 전체 처리 속도는 느림|

### 샘플 데이터 생성 & 저장

```python
import time

n_rows = 10000
data = {
    'id':     range(1, n_rows + 1),
    'name':   [f'Student_{i}' for i in range(1, n_rows + 1)],
    'score1': np.random.randint(50, 101, size=n_rows),
    'score2': np.random.randint(50, 101, size=n_rows)
}
df = pd.DataFrame(data)
print(df.head())
print(df.tail(3))

csv_fname = 'students.csv'
df.to_csv(csv_fname, index=False)
```

### 전체 한 번에 읽기

```python
start_all = time.time()
df_all = pd.read_csv(csv_fname)
average_all_1 = df_all['score1'].mean()
average_all_2 = df_all['score2'].mean()
time_all = time.time() - start_all
```

### Chunk 단위로 읽기

```python
chunk_size = 1000
total_score1 = 0
total_score2 = 0
total_count  = 0
start_chunk_total = time.time()

for i, chunk in enumerate(pd.read_csv(csv_fname, chunksize=chunk_size)):
    start_chunk = time.time()

    first_student = chunk.iloc[0]
    print(f"Chunk {i+1} 첫번째 학생:ID={first_student['id']}, 이름={first_student['name']}",
          f"score1={first_student['score1']}",
          f"score2={first_student['score2']}")

    total_score1 += chunk['score1'].sum()
    total_score2 += chunk['score2'].sum()
    total_count  += len(chunk)

    end_chunk = time.time()
    elapsed = end_chunk - start_chunk
    print(f"    처리 시간: {elapsed:7f}")

time_chunk_total = time.time() - start_chunk_total
average_chunk1 = total_score1 / total_count
average_chunk2 = total_score2 / total_count
```

### 처리 결과 출력

```python
print('\n처리 결과')
print(f'전체 학생 수 : {total_count}')
print(f"score1 총합 : {total_score1}, 평균 : {average_chunk1:3f}")
print(f"score2 총합 : {total_score2}, 평균 : {average_chunk2:3f}")
print(f"전체 한 번에 처리 시간 : {time_all:7f}초")
print(f"청크로 처리한 총 시간  : {time_chunk_total:7f}초")
```

> 💡 데이터가 작을수록 chunk 방식이 오히려 느리다. chunk가 진가를 발휘하는 건 **수백 MB ~ GB 이상** 파일을 다룰 때다.

### 처리 시간 시각화

```python
import matplotlib.pyplot as plt

# Windows : 'Malgun Gothic', Mac : 'AppleGothic', Linux : 'NanumGothic'
plt.rc('font', family='Malgun Gothic')

labels = ['전체 한번에 처리', '청크로 처리']
times  = [time_all, time_chunk_total]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, times, color=['skyblue', 'red'])

for bar, time_val in zip(bars, times):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x 위치 : 막대 중앙
        bar.get_height(),                    # y 위치 : 막대 꼭대기
        f'{time_val:.3f}초',
        ha='center', va='bottom',
        fontsize=10
    )

plt.ylabel('처리 시간(초)')
plt.grid(linestyle='--')
plt.tight_layout()
plt.show()
```

---

## ✅ 핵심 옵션 정리

|옵션|설명|예시|
|---|---|---|
|`sep`|구분자|`sep=','` / `sep=r'\s+'`|
|`header`|헤더 행 번호 (`None` = 헤더 없음)|`header=None`|
|`names`|열 이름 직접 지정|`names=['a','b','c']`|
|`skiprows`|건너뛸 행 (숫자 or 리스트)|`skiprows=1` / `skiprows=[1,3]`|
|`skip_blank_lines`|빈 줄 무시 (기본 `True`)|`skip_blank_lines=True`|
|`encoding`|파일 인코딩|`encoding='utf-8'`|
|`chunksize`|chunk 단위 행 수|`chunksize=1000`|
|`index`|to_csv 저장 시 인덱스 포함 여부|`index=False`|

|함수|용도|기본 구분자|
|---|---|---|
|`read_csv`|쉼표 구분 파일|`,`|
|`read_table`|탭/공백 구분 파일|`\t`|
|`read_fwf`|고정 너비 파일|너비(`widths`) 직접 지정|

---

# Pandas DataFrame 저장 & 출력 (File Output)

---

## 📋 DataFrame 생성

```python
import pandas as pd

# 중첩 dict → DataFrame : 바깥 key가 열 이름, 안쪽 key가 행 이름
items = {
    'apple':  {'count': 10,   'price': 1500},
    'orange': {'count':  5,   'price':  800}
}

df = pd.DataFrame(items)
print(df)
```

```
       apple  orange
count     10       5
price   1500     800
```

---

## 📤 다양한 형식으로 출력·저장

### `to_clipboard()` — 클립보드 복사

```python
# 클립보드에 복사 → 메모장, 엑셀 등에 바로 붙여넣기 가능
df.to_clipboard()
```

### `to_html()` — HTML 테이블로 변환

```python
# HTML <table> 태그로 변환 → 웹 페이지에 바로 삽입 가능
print(df.to_html())
```

### `to_json()` — JSON 문자열로 변환

```python
# 기본 형식: {열: {행: 값}} 구조
print(df.to_json())
```

```json
{"apple":{"count":10,"price":1500},"orange":{"count":5,"price":800}}
```

---

## 💾 CSV 저장 — `to_csv()`

```python
df.to_csv('result1.csv', sep=',')                              # 인덱스 포함
df.to_csv('result2.csv', sep=',', index=False)                 # 인덱스 제외
df.to_csv('result3.csv', sep=',', index=False, header=False)   # 인덱스·헤더 모두 제외
```

|옵션|효과|
|---|---|
|기본|첫 열에 인덱스(count, price) 포함|
|`index=False`|인덱스 열 제거|
|`header=False`|열 이름 행 제거|

### 전치(T) 후 저장 & 다시 읽기

```python
df2 = df.T   # 행과 열을 뒤바꿈

# encoding='utf-8-sig' : BOM 포함 → 엑셀에서 한글 깨짐 방지
df2.to_csv('result4.csv', sep=',', index=False, encoding='utf-8-sig')

redata = pd.read_csv('result4.csv')
print(redata)
```

|인코딩|특징|사용 시점|
|---|---|---|
|`utf-8`|BOM 없음|일반 텍스트, Python 내부 처리|
|`utf-8-sig`|BOM 포함|엑셀에서 열 때 한글 깨짐 방지|

---

## 📊 엑셀 저장 & 읽기

```python
df3 = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Oscar'],
    'age':  [24, 22, 29],
    'city': ['seoul', 'suwon', 'incheon']
})

# ※ openpyxl 패키지 필요 : pip install openpyxl
df3.to_excel('result.xlsx', index=False, sheet_name='work1')

exdf = pd.ExcelFile('result.xlsx')
print(exdf.sheet_names)   # ['work1']
df4 = exdf.parse('work1')
print(df4)
```

---

## ✅ 저장 형식별 정리

|메서드|형식|주요 옵션|
|---|---|---|
|`to_clipboard()`|클립보드|—|
|`to_html()`|HTML 문자열|—|
|`to_json()`|JSON 문자열|`orient=`|
|`to_csv()`|CSV 파일|`index=`, `header=`, `encoding=`|
|`to_excel()`|xlsx 파일|`sheet_name=`, `index=`|

---

# Pandas + openpyxl 엑셀 스타일링

---

## 1. DataFrame 생성 & 총금액 열 추가

```python
import pandas as pd

df = pd.DataFrame({
    '상품명': ['Mouse', 'Keyboard', 'Monitor'],
    '수량':   [10, 5, 2],
    '가격':   [12000, 25000, 300000]
})

df['총금액'] = df['수량'] * df['가격']
```

---

## 2. ExcelWriter로 엑셀 파일 생성

```python
# with 블록 종료 시 자동 저장
with pd.ExcelWriter('report.xlsx', engine='openpyxl') as writer:
    # startrow=2 : 0-indexed → 실제 3행부터 시작
    df.to_excel(writer, sheet_name='Report', index=False, startrow=2)
    ws = writer.sheets['Report']
```

💡 **`startrow` 값과 실제 행 위치**

|startrow|실제 시작 행|헤더 위치|
|---|---|---|
|0 (기본값)|1행|1행|
|2|3행|3행|

---

## 3. 제목 & 헤더 스타일

```python
from openpyxl.styles import Font, PatternFill, Alignment

ws['A1'] = '상품 판매 보고서'
ws['A1'].font = Font(bold=True, size=14)

header_font = Font(bold=True, color='FFFFFF')
header_fill = PatternFill(start_color='4F81BD', fill_type='solid')

for cell in ws[3]:
    cell.font      = header_font
    cell.fill      = header_fill
    cell.alignment = Alignment(horizontal='center')
```

---

## 4. 열 너비 자동 조정

```python
for col in ws.columns:
    max_length = 0
    col_letter = col[0].column_letter
    for cell in col:
        try:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        except:
            pass
    ws.column_dimensions[col_letter].width = max_length + 2
```

---

## 5. 숫자 포맷 & 테이블 스타일

```python
# 콤마 포맷
for row in ws.iter_rows(min_row=4, min_col=2, max_col=4):
    for cell in row:
        if isinstance(cell.value, (int, float)):
            cell.number_format = '#,##0'
```

💡 **자주 쓰는 number_format**

|포맷|예시|
|---|---|
|`#,##0`|120,000|
|`#,##0.00`|120,000.00|
|`0%`|75%|
|`YYYY-MM-DD`|2024-01-01|

```python
from openpyxl.worksheet.table import Table, TableStyleInfo

tab = Table(displayName="Table1", ref=f"A3:D{len(df)+3}")
style = TableStyleInfo(
    name="TableStyleMedium9",
    showRowStripes=True,
    showFirstColumn=False, showLastColumn=False, showColumnStripes=False
)
tab.tableStyleInfo = style
ws.add_table(tab)
```

---

## 6. 합계 행 & 정렬

```python
total_row = len(df) + 4
ws[f'A{total_row}'] = '합계'
ws[f'D{total_row}'] = f'=SUM(D4:D{len(df)+3})'   # 엑셀 수식 직접 입력

for row in ws.iter_rows(min_row=4, max_row=ws.max_row):
    for cell in row:
        cell.alignment = Alignment(horizontal='center')
```

---

## 📐 최종 파일 구조

```
행  | A              | B    | C       | D
----+----------------+------+---------+-----------
 1  | 상품 판매 보고서  |      |         |   ← 제목 (굵게, 14pt)
 2  |                |      |         |   ← 빈 행
 3  | 상품명           | 수량  | 가격     | 총금액  ← 헤더 (파란 배경)
 4  | Mouse          | 10   | 12,000  | 120,000
 5  | Keyboard       |  5   | 25,000  | 125,000
 6  | Monitor        |  2   | 300,000 | 600,000
 7  | 합계            |      |         | =SUM(D4:D6)
```

---

## ✅ 핵심 정리

|기능|코드|
|---|---|
|엑셀 저장 (스타일 포함)|`pd.ExcelWriter('파일', engine='openpyxl')`|
|특정 행부터 시작|`df.to_excel(writer, startrow=N)`|
|워크시트 접근|`writer.sheets['시트명']`|
|셀 값 입력|`ws['A1'] = '값'`|
|글꼴 스타일|`Font(bold=True, size=14, color='FFFFFF')`|
|배경색|`PatternFill(start_color='HEX', fill_type='solid')`|
|정렬|`Alignment(horizontal='center')`|
|열 너비|`ws.column_dimensions['A'].width = N`|
|숫자 포맷|`cell.number_format = '#,##0'`|
|엑셀 수식 입력|`ws['D7'] = '=SUM(D4:D6)'`|

---

# Pandas 실습 문제 — 타이타닉 & 데이터 분석

---

## 📌 데이터 소스

|파일|URL|
|---|---|
|titanic_data.csv|https://github.com/pykwon/python/blob/master/testdata_utf8/titanic_data.csv|
|human.csv|https://github.com/pykwon/python/tree/master/testdata_utf8|
|tips.csv|https://github.com/pykwon/python/tree/master/testdata_utf8|

**titanic 열 구성**

|열|설명|
|---|---|
|Survived|0 = 사망, 1 = 생존|
|Pclass|1 = 1등석, 2 = 2등석, 3 = 3등석|
|Sex|male = 남성, female = 여성|
|Age|나이|
|SibSp|동승한 자매/배우자 수|
|Parch|동승한 부모/자식 수|
|Fare|승객 요금|
|Embarked|탑승지 (C=셰르부르, Q=퀸즈타운, S=사우샘프턴)|

---

## 문제 5 — 타이타닉 생존 분석

### 문제 5-1. 나이대별 생존자 수

```python
import pandas as pd
import numpy as np

df = pd.read_csv('titanic.csv')

bins   = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]

df['나이대'] = pd.cut(df['Age'], bins=bins, labels=labels)

# observed=True : pandas 2.0 이후 Categorical groupby 경고 방지
result = df.groupby('나이대', observed=True)['Survived'].sum()
result = result.reset_index()
result.columns = ['나이대', '생존자수']
print(result)
```

### 문제 5-2. 성별×선실 생존율 피벗테이블

```python
# 샘플1
pivot1 = df.pivot_table(
    values='Survived', index='Sex', columns='Pclass', aggfunc='mean'
)
print(pivot1)

# 샘플2 — 백분율, 소수 둘째자리
pivot2 = df.pivot_table(
    values='Survived',
    index=['Sex', '나이대'],
    columns='Pclass',
    aggfunc='mean'
)
pivot2 = (pivot2 * 100).round(2)
print(pivot2)
```

---

## 문제 6 — CSV 데이터 전처리

### 문제 6-1. human.csv

```python
df_h = pd.read_csv('human.csv', skipinitialspace=True)
df_h.columns = df_h.columns.str.strip()   # 열 이름 공백 제거

df1 = df_h.dropna(subset=['Group'])       # Group NaN 행 삭제
print(df1[['Career', 'Score']])
print(df1[['Career', 'Score']].mean())
```

> 💡 `skipinitialspace=True` vs `str.strip()`
> 
> - `skipinitialspace` : 구분자 바로 뒤 공백만 제거
> - `str.strip()` : 양쪽 끝 공백 + `\n` 모두 제거

### 문제 6-2. tips.csv

```python
df3 = pd.read_csv('tips.csv')

print(df3.info())               # 구조 확인
print(df3.head(3))              # 앞 3행
print(df3.describe())           # 요약 통계량
print(df3['smoker'].value_counts())   # 흡연자/비흡연자 수
print(df3['day'].unique())      # 요일 고유값
```

---

## ✅ 핵심 정리

|기능|코드|설명|
|---|---|---|
|나이대 범주화|`pd.cut(df['Age'], bins, labels)`|연속값 → 구간 범주|
|그룹별 생존자 수|`groupby('나이대')['Survived'].sum()`|합계 = 생존 인원|
|생존율 피벗|`pivot_table(aggfunc='mean')`|평균 = 생존율|
|백분율 변환|`(pivot * 100).round(2)`|소수 → 백분율|
|NA 행 삭제|`dropna(subset=['열이름'])`|특정 열 기준|
|열 이름 공백 제거|`df.columns.str.strip()`|양끝 공백 제거|
|구조 확인|`df.info()`|타입·결측치 한눈에|
|요약 통계|`df.describe()`|수치형 기술통계|
|빈도 집계|`series.value_counts()`|고유값별 개수|
|고유값 목록|`series.unique()`|중복 없는 값 배열|

---

# Python XML 파싱 — ElementTree & 기상청 데이터

---

## 📌 XML 이란?

> **eXtensible Markup Language** — 데이터를 태그로 구조화하는 형식. HTML과 비슷하지만 데이터 저장·전송 목적.

|용어|설명|예시|
|---|---|---|
|요소(Element)|태그로 감싼 단위|`<item>...</item>`|
|태그(Tag)|요소의 이름|`item`, `name`|
|속성(Attribute)|태그 안의 키=값|`ta="15"`|
|텍스트(Text)|태그 사이의 내용|`홍길동`|
|루트(Root)|최상위 요소 (단 하나)|`<items>`|
|네임스페이스|태그 충돌 방지 접두어|`{current}weather`|

---

## 1. XML 파일 파싱

```python
import xml.etree.ElementTree as etree

# parse() : XML 파일 → ElementTree 객체
xmlfile = etree.parse("my.xml")

# getroot() : 루트 Element 반환
root = xmlfile.getroot()
print(root.tag)        # items
print(root[0].tag)     # item
print(root[0][0].tag)  # name
```

**트리 구조**

```
<items>      ← root
  <item>     ← root[0]
    <name>   ← root[0][0]
    <tel>    ← root[0][1]
```

---

## 2. 값 추출 — `find()` / `.text` / `.get()`

```python
# find('태그') : 첫 번째 일치 자식 반환
# .text : 태그 사이 텍스트
myname = root.find("item").find("name").text
mytel  = root.find("item").find("tel").text
```

---

## 3. 기상청 XML — URL에서 읽기

```python
import requests

url = "https://www.kma.go.kr/XML/weather/sfc_web_map.xml"
res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
res.raise_for_status()   # 4xx/5xx 오류 시 예외 발생

# fromstring() : 문자열 → 루트 Element (파일 없이 바로 파싱)
root = etree.fromstring(res.text)
```

---

## 4. 네임스페이스 제거

```python
# {current}weather → weather
for elem in root.iter():
    if '}' in elem.tag:
        elem.tag = elem.tag.split('}', 1)[1]
```

---

## 5. 속성값 & 반복 순회

```python
weather = root.find('weather')

# .get('속성명') : 태그의 속성값
year  = weather.get('year')
month = weather.get('month')
day   = weather.get('day')
hour  = weather.get('hour')

# findall() : 일치하는 모든 자식 → 리스트
for local in weather.findall("local"):
    name = local.text.strip()   # 앞뒤 공백·\n 제거
    ta   = local.get('ta')
    print(f"{name} 지역 온도는 {ta}")
```

---

## ✅ 핵심 메서드 정리

|메서드|설명|반환|
|---|---|---|
|`etree.parse('파일')`|XML 파일 파싱|`ElementTree`|
|`etree.fromstring('문자열')`|XML 문자열 파싱|`Element` (루트)|
|`tree.getroot()`|루트 요소 반환|`Element`|
|`elem.tag`|태그 이름|`str`|
|`elem.text`|태그 안 텍스트|`str`|
|`elem.get('속성')`|속성값 반환|`str`|
|`elem.find('태그')`|첫 번째 자식 검색|`Element`|
|`elem.findall('태그')`|모든 자식 검색|`list[Element]`|
|`elem.iter()`|모든 하위 요소 순회|`iterator`|

|구분|`etree.parse()`|`etree.fromstring()`|
|---|---|---|
|입력|파일 경로 / 파일 객체|XML 문자열|
|반환|`ElementTree`|`Element` (루트)|
|루트 접근|`.getroot()` 필요|바로 루트 사용|
|사용 시점|로컬 파일 읽을 때|URL 응답 문자열 파싱 시|

---
