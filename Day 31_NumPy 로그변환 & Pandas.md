# Day 31_NumPy 로그변환 & Pandas

## 📅 2026-03-19

---

# 📊 NumPy 로그 변환 (Log Transformation)

## 🤔 로그 변환을 사용하는 이유

머신러닝에서 데이터 분석 시 로그를 사용하면:

1. 📉 **스케일 차이 축소** — 큰 수치 범위를 압축해줌
    - `log(10) = 1`, `log(100) = 2`, `log(1000) ≈ 3`
2. 🔔 **정규분포 근사** — 치우친(편향된) 데이터를 정규분포에 가깝게 변환 가능
3. 📐 **선형화** — 지수 관계를 선형 관계로 바꿔줌 (모델링에 유리)

---

## 🔢 주요 로그 함수 비교

|함수|설명|NumPy|
|---|---|---|
|`log2`|밑이 2인 로그|`np.log2(x)`|
|`log10`|상용로그 (밑이 10)|`np.log10(x)`|
|`log` / `ln`|자연로그 (밑이 e)|`np.log(x)`|

---

## 💻 로그 변환 코드

### `test()` — 기본 로그 변환 및 정규화

```python
import numpy as np
np.set_printoptions(suppress=True, precision=6)  # 지수 표기 억제, 소수점 6자리

values = np.array([345, 34.5, 3.45, 0.345, 0.01, 0.1, 10, 100])

# 세 가지 로그 함수 비교 (값 3.45 기준)
print(np.log2(3.45), ' ', np.log10(3.45), ' ', np.log(3.45))

print("원본 값 : ", values)

# 상용로그 변환 (밑 10)
log_values = np.log10(values)
print('log_values : ', log_values)

# 자연로그 변환 (밑 e ≈ 2.718)
ln_values = np.log(values)
print('ln_values : ', ln_values)

# 정규화 : 로그 변환된 값을 0~1 범위로 스케일 조정
# 공식 : (x - min) / (max - min)
min_log = np.min(log_values)
max_log = np.max(log_values)
normalized = (log_values - min_log) / (max_log - min_log)
print("정규화 결과 : ", normalized)
```

```
log2(3.45) / log10(3.45) / log(3.45) :
1.7865963618908067   0.5378190950732742   1.2383742310432684

원본 값 :  [345.     34.5     3.45    0.345   0.01    0.1    10.    100.   ]
log_values :  [ 2.537819  1.537819  0.537819 -0.462181 -2.       -1.        1.        2.      ]
ln_values :  [ 5.843544  3.540959  1.238374 -1.064211 -4.60517  -2.302585  2.302585   4.60517 ]
정규화 결과 :  [1.       0.77963  0.55926  0.338889 0.       0.22037  0.661111 0.881481]
```

> 📌 **정규화 공식** $$\hat{x} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
> 
> 정규화 결과의 최솟값은 항상 0, 최댓값은 항상 1이 된다.

---

### `LogTrans` 클래스 — 로그 변환 & 역변환

편차가 매우 큰 데이터(예: 0.001 ~ 10000)를 로그 스케일로 압축하고, 필요 시 원본으로 복원하는 클래스.

```python
class LogTrans:
    def __init__(self, offset: float = 1.0):
        # offset : log(0) = -inf 오류를 막기 위해 x에 더해주는 값
        # 0이나 음수 데이터가 있어도 안전하게 처리 가능
        self.offset = offset

    def transform(self, x: np.ndarray) -> np.ndarray:
        # 로그 변환 : y = ln(x + offset)
        return np.log(x + self.offset)

    def inverse_trans(self, x_log: np.ndarray) -> np.ndarray:
        # 역변환 : x = e^y - offset  (transform의 역연산)
        return np.exp(x_log) - self.offset


data = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], dtype=float)

log_trans = LogTrans(offset=1.0)
data_log_scaled = log_trans.transform(data)       # 로그 변환
reversed_data   = log_trans.inverse_trans(data_log_scaled)  # 역변환

print("원본      : ", data)
print("로그변환  : ", data_log_scaled)
print("역변환    : ", reversed_data)
```

```
원본      :  [    0.001     0.01      0.1       1.       10.      100.     1000.    10000.   ]
로그변환  :  [0.001    0.00995  0.09531  0.693147 2.397895 4.615121 6.908755 9.21044 ]
역변환    :  [    0.001     0.01      0.1       1.       10.      100.     1000.    10000.   ]
```

> 💡 역변환 결과가 원본과 동일 → 변환·복원이 정확히 작동함을 확인

---

## ✅ 로그변환 핵심 정리

- `np.log10` → 상용로그, 스케일 직관적 비교에 유리
- `np.log` → 자연로그, 역변환(`np.exp`)이 용이해 ML에서 자주 사용
- `offset` 추가로 0 또는 음수 데이터도 안전하게 처리 가능
- 로그 변환 후 정규화하면 모델 학습에 더 안정적인 입력 제공

---

---

# 🐼 Pandas 기초 — Series & DataFrame

## 🧐 Pandas란?

- 고수준의 자료구조 (`Series`, `DataFrame`)와 빠르고 쉬운 데이터 분석용 함수 제공
- 통합된 시계열 연산, 축약연산, 누락 데이터 처리, SQL, 시각화 등을 제공
- **데이터 랭글링(Data Wrangling)**, **데이터 먼징(Data Munging)** 을 효율적으로 처리 가능

---

## 📌 Series — 1차원 자료구조

> 일련의 객체를 담을 수 있는 **1차원 배열**과 같은 자료구조로 **색인(index)** 을 갖는다.

### 생성

```python
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# 기본 생성 : 자동으로 0부터 정수 인덱스 부여
obj = pd.Series([3, 7, -5, 4])
print(obj)
```

```
0    3
1    7
2   -5
3    4
dtype: int64
```

### 커스텀 인덱스 & 기본 연산

```python
# index 인자로 라벨 인덱스 직접 지정
obj2 = pd.Series([3, 7, -5, 4], index=['a', 'b', 'c', 'd'])
print(obj2)

# 합계 — 세 가지 방법 모두 동일한 결과
print(obj2.sum(), ' ', np.sum(obj2), ' ', sum(obj2))

# 표준편차
print(obj2.std())
```

```
a    3
b    7
c   -5
d    4
dtype: int64

sum: 9   np.sum: 9   sum(): 9
std: 5.123475382979799
```

### 속성

```python
print(obj2.values)  # 값만 → numpy ndarray 반환
print(obj2.index)   # 인덱스 반환
```

```
values: [ 3  7 -5  4]
index: Index(['a', 'b', 'c', 'd'], dtype='str')
```

### 인덱싱

```python
print(obj2['a'])          # 라벨 → 스칼라
print(obj2[['a']])        # 리스트로 감싸면 Series 반환
print(obj2[['a', 'b']])   # 다중 라벨 선택
print(obj2['a':'c'])      # 라벨 슬라이싱 (끝 'c' 포함)

print(obj2.iloc[2])       # 정수 위치 기반 (pandas 3.x 권장)
print(obj2[1:4])          # 정수 슬라이싱 (끝 미포함)
print(obj2.iloc[[2, 1]])  # 정수 다중 선택

print('a' in obj2)        # 인덱스 포함 여부 확인
print('k' in obj2)
```

```
obj2['a']: 3

obj2[['a']]:
 a    3
dtype: int64

obj2[['a','b']]:
 a    3
b    7
dtype: int64

obj2['a':'c']:
 a    3
b    7
c   -5
dtype: int64

obj2.iloc[2]: -5

obj2[1:4]:
 b    7
c   -5
d    4
dtype: int64

obj2.iloc[[2,1]]:
 c   -5
b    7
dtype: int64

'a' in obj2: True
'k' in obj2: False
```

> ⚠️ **슬라이싱 주의**
> 
> - 라벨 슬라이싱 `['a':'c']` → 끝 **포함**
> - 정수 슬라이싱 `[1:4]` → 끝 **미포함**
> 
> ⚠️ **pandas 3.x 변경사항** 문자열 인덱스를 가진 Series에서 `obj2[2]` 같은 정수 접근은 오류 발생. 반드시 `obj2.iloc[2]` 를 사용할 것.

### dict → Series 변환

```python
names = {'mouse': 5000, 'keyboard': 25000, 'monitor': 450000}

obj3 = Series(names)   # dict의 key → 인덱스, value → 값
print(obj3)

obj3.index = ['마우스', '키보드', '모니터']  # 인덱스 교체
obj3.name = "상품가격"                        # Series 이름 설정
print(obj3)
```

```
mouse         5000
keyboard     25000
monitor     450000
dtype: int64

마우스      5000
키보드     25000
모니터    450000
Name: 상품가격, dtype: int64
```

---

## 📋 DataFrame — 2차원 자료구조

> 2차원 표 형태의 자료구조. 각 열은 서로 다른 타입을 가질 수 있다.

### Series → DataFrame 변환

```python
df = pd.DataFrame(obj3)   # Series를 DataFrame으로 변환 (Series가 하나의 열이 됨)
print(df)
```

```
        상품가격
마우스    5000
키보드   25000
모니터  450000
```

### dict → DataFrame 생성

```python
data = {
    'irum': ['홍길동', '한국인', '신기해', '공기밥', '한가해'],
    'juso': ('역삼동', '신당동', '역삼동', '역삼동', '신사동'),  # 튜플도 가능
    'nai':  [23, 25, 33, 231, 35],
}

frame = pd.DataFrame(data)
print(frame)
```

```
  irum juso  nai
0  홍길동  역삼동   23
1  한국인  신당동   25
2  신기해  역삼동   33
3  공기밥  역삼동  231
4  한가해  신사동   35
```

### 열 접근

```python
print(frame['irum'])    # 딕셔너리 방식으로 열 접근 → Series 반환
print(frame.irum)       # 속성 방식 (열 이름이 유효한 식별자일 때만 가능)

# columns 인자로 열 순서 직접 지정
print(DataFrame(data=data, columns=['juso', 'irum', 'nai']))
```

```
frame['irum']:
 0    홍길동
1    한국인
2    신기해
3    공기밥
4    한가해
Name: irum, dtype: str

열 순서 지정:
  juso irum  nai
0  역삼동  홍길동   23
1  신당동  한국인   25
2  역삼동  신기해   33
3  역삼동  공기밥  231
4  신사동  한가해   35
```

### NaN (결측치)

존재하지 않는 열을 `columns`에 지정하면 자동으로 `NaN`으로 채워진다.

```python
frame2 = pd.DataFrame(data, columns=['irum', 'nai', 'juso', 'tel'],
                      index=['a', 'b', 'c', 'd', 'e'])
print(frame2)   # tel 열 → 데이터 없으므로 NaN
```

```
  irum  nai juso  tel
a  홍길동   23  역삼동  NaN
b  한국인   25  신당동  NaN
c  신기해   33  역삼동  NaN
d  공기밥  231  역삼동  NaN
e  한가해   35  신사동  NaN
```

```python
# 단일 값으로 열 전체 채우기
frame2['tel'] = '111-1111'
print(frame2)
```

```
  irum  nai juso       tel
a  홍길동   23  역삼동  111-1111
b  한국인   25  신당동  111-1111
c  신기해   33  역삼동  111-1111
d  공기밥  231  역삼동  111-1111
e  한가해   35  신사동  111-1111
```

```python
# Series로 일부 인덱스만 채우기 (인덱스 기반 매핑, 나머지는 NaN)
val = pd.Series(['222-2222', '333-3333', '444-4444'], index=['b', 'c', 'e'])
frame2['tel'] = val
print(frame2)
```

```
  irum  nai juso       tel
a  홍길동   23  역삼동       NaN
b  한국인   25  신당동  222-2222
c  신기해   33  역삼동  333-3333
d  공기밥  231  역삼동       NaN
e  한가해   35  신사동  444-4444
```

### 전치 & values

```python
print(frame2.T)             # 전치 : 행 ↔ 열 뒤바꾸기
print(frame2.values)        # 2D numpy ndarray 반환
print(frame2.values[0, 1])  # [행, 열] 위치로 값 접근
print(frame2.values[0:2])   # 행 슬라이싱
```

```
전치(T):
         a         b         c    d         e
irum  홍길동       한국인       신기해  공기밥       한가해
nai    23        25        33  231        35
juso  역삼동       신당동       역삼동  역삼동       신사동
tel   NaN  222-2222  333-3333  NaN  444-4444

values[0,1]: 23

values[0:2]:
 [['홍길동' 23 '역삼동' nan]
  ['한국인' 25 '신당동' '222-2222']]
```

### 행/열 삭제 — `drop()`

```python
frame3 = frame2.drop('d')            # 행 삭제 (axis=0 기본값)
print(frame3)

frame4 = frame2.drop('tel', axis=1)  # 열 삭제
print(frame4)
```

```
행 삭제(drop 'd'):
  irum  nai juso       tel
a  홍길동   23  역삼동       NaN
b  한국인   25  신당동  222-2222
c  신기해   33  역삼동  333-3333
e  한가해   35  신사동  444-4444

열 삭제(drop 'tel'):
  irum  nai juso
a  홍길동   23  역삼동
b  한국인   25  신당동
c  신기해   33  역삼동
d  공기밥  231  역삼동
e  한가해   35  신사동
```

> 💡 `drop()`은 원본을 수정하지 않고 **새 DataFrame을 반환**한다. 원본 수정이 필요하면 `inplace=True` 사용.

### 정렬 — `sort_index()`

```python
# axis=0 : 행 인덱스 기준 정렬
print(frame2.sort_index(axis=0, ascending=False))

# axis=1 : 열 이름 기준 정렬
print(frame2.sort_index(axis=1, ascending=True))
```

```
행 인덱스 내림차순 정렬:
  irum  nai juso       tel
e  한가해   35  신사동  444-4444
d  공기밥  231  역삼동       NaN
c  신기해   33  역삼동  333-3333
b  한국인   25  신당동  222-2222
a  홍길동   23  역삼동       NaN

열 인덱스 오름차순 정렬:
  irum juso  nai       tel
a  홍길동  역삼동   23       NaN
b  한국인  신당동   25  222-2222
c  신기해  역삼동   33  333-3333
d  공기밥  역삼동  231       NaN
e  한가해  신사동   35  444-4444
```

### 순위 & 빈도수

```python
print(frame2.rank(axis=0))              # 각 열 내에서 값의 순위 매김
print(frame2['juso'].value_counts())    # juso 열의 값별 빈도 집계
```

```
rank():
    irum  nai  juso  tel
a   5.0  1.0   4.0  NaN
b   4.0  2.0   1.0  1.0
c   2.0  3.0   4.0  2.0
d   1.0  5.0   4.0  NaN
e   3.0  4.0   2.0  3.0

juso value_counts():
 juso
역삼동    3
신당동    1
신사동    1
Name: count, dtype: int64
```

### 문자열 분리 응용

```python
data = {
    'juso':  ['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon': [23, 25, 15]
}
fr = pd.DataFrame(data)

# split()[0] : 공백 기준으로 나눠 앞부분(구) 추출
result1 = Series(x.split()[0] for x in fr.juso)
# split()[1] : 뒷부분(동) 추출
result2 = Series(x.split()[1] for x in fr.juso)

print(result1)
print(result2)
print(result1.value_counts())  # 구별 빈도 집계
```

```
구(앞부분):
 0    강남구
1     중구
2    강남구
dtype: str

동(뒷부분):
 0    역삼동
1    신당동
2    대치동
dtype: str

구별 빈도:
 강남구    2
중구     1
Name: count, dtype: int64
```

---

## ✅ Series & DataFrame 핵심 정리

|기능|코드|비고|
|---|---|---|
|라벨 인덱싱|`s['a']`, `s[['a','b']]`|단일 → 스칼라, 리스트 → Series|
|위치 인덱싱|`s.iloc[0]`, `s.iloc[[0,1]]`|pandas 3.x 권장|
|라벨 슬라이싱|`s['a':'c']`|끝 **포함**|
|정수 슬라이싱|`s[1:4]`|끝 **미포함**|
|결측치 생성|없는 열 지정 시 자동 `NaN`||
|행 삭제|`df.drop('행이름')`|원본 유지, 새 df 반환|
|열 삭제|`df.drop('열이름', axis=1)`||
|빈도 집계|`s.value_counts()`||
|전치|`df.T`||
|정렬|`df.sort_index(axis=0/1, ascending=T/F)`||

---

---

# 🔁 Pandas — 재색인 & 불리언 인덱싱 & loc/iloc

## 🔄 재색인 (reindex)

기존 Series/DataFrame의 인덱스를 새로운 순서로 재정렬하거나, 없는 인덱스를 추가할 때 사용.

### Series 재색인

```python
from pandas import Series
import numpy as np

data = Series([1, 3, 2], index=(1, 4, 2))
print(data)

# 인덱스를 (1, 2, 4) 순서로 재정렬
data2 = data.reindex((1, 2, 4))
print(data2)
```

```
원본:
 1    1
4    3
2    2
dtype: int64

재색인(1,2,4):
 1    1
2    2
4    3
dtype: int64
```

### 없는 인덱스 채우기

```python
# 없는 인덱스(0, 3, 5) → NaN으로 자동 채워짐
print(data2.reindex([0, 1, 2, 3, 4, 5]))

# fill_value : NaN 대신 지정한 값으로 채우기
print(data2.reindex([0, 1, 2, 3, 4, 5], fill_value=777))
```

```
없는 인덱스(NaN):
 0    NaN
1    1.0
2    2.0
3    NaN
4    3.0
5    NaN
dtype: float64

fill_value=777:
 0    777
1      1
2      2
3    777
4      3
5    777
dtype: int64
```

### NaN 채우기 — `method` 옵션

|method|설명|
|---|---|
|`'ffill'` / `'pad'`|⬆️ NaN을 **앞(이전)** 값으로 채움 (Forward Fill)|
|`'bfill'` / `'backfill'`|⬇️ NaN을 **뒤(다음)** 값으로 채움 (Backward Fill)|

```python
# ffill : 앞 값으로 전파 (인덱스 0은 이전 값이 없으므로 NaN 유지)
print(data2.reindex([0, 1, 2, 3, 4, 5], method='ffill'))

# bfill : 뒤 값으로 전파 (인덱스 5는 다음 값이 없으므로 NaN 유지)
print(data2.reindex([0, 1, 2, 3, 4, 5], method='bfill'))
```

```
method=ffill:
 0    NaN  ← 이전 값 없음
1    1.0
2    2.0
3    2.0  ← 앞 값 2.0으로 채워짐
4    3.0
5    3.0  ← 앞 값 3.0으로 채워짐
dtype: float64

method=bfill:
 0    1.0  ← 뒤 값 1.0으로 채워짐
1    1.0
2    2.0
3    3.0  ← 뒤 값 3.0으로 채워짐
4    3.0
5    NaN  ← 다음 값 없음
dtype: float64
```

---

## 🎭 불리언 인덱싱 (Boolean Indexing)

조건식으로 True/False 마스크를 만들어 행이나 값을 필터링한다.

```python
from pandas import DataFrame
import numpy as np

df = DataFrame(np.arange(12).reshape(4, 3),
               index=['1월', '2월', '3월', '4월'],
               columns=['강남', '강북', '서초'])
print(df)
```

```
원본:
     강남  강북  서초
1월   0   1   2
2월   3   4   5
3월   6   7   8
4월   9  10  11
```

```python
# 단일 열 조건 → True/False 마스크 생성
print(df['강남'] > 3)

# 마스크를 DataFrame에 적용 → 조건에 맞는 행만 필터링
print(df[df['강남'] > 3])
```

```
df['강남'] > 3:
 1월    False
2월    False
3월     True
4월     True
Name: 강남, dtype: bool

강남 > 3인 행:
     강남  강북  서초
3월   6   7   8
4월   9  10  11
```

```python
# 전체 DataFrame에 조건 적용 → 해당 값만 일괄 치환
print(df < 3)
df[df < 3] = 0   # 3 미만인 모든 값을 0으로 바꾸기
print(df)
```

```
df < 3:
        강남     강북     서초
1월   True   True   True
2월  False  False  False
3월  False  False  False
4월  False  False  False

3 미만 → 0 치환:
     강남  강북  서초
1월   0   0   0
2월   3   4   5
3월   6   7   8
4월   9  10  11
```

---

## 🗂️ 슬라이싱 메서드 — `loc` & `iloc`

### `loc` — 라벨 기반

```python
print(df.loc['3월', :])          # '3월' 행 전체 선택
print(df.loc[:'2월'])            # 처음 ~ '2월' 행 (끝 포함)
print(df.loc[:'2월', ['서초']])  # 처음 ~ '2월', '서초' 열만
```

```
df.loc['3월', :]:
 강남    6
강북    7
서초    8
Name: 3월, dtype: int64

df.loc[:'2월']:
     강남  강북  서초
1월   0   0   0
2월   3   4   5

df.loc[:'2월', ['서초']]:
     서초
1월   0
2월   5
```

### `iloc` — 정수 위치 기반

```python
print(df.iloc[2])        # 2번째 행 (3월)
print(df.iloc[:3])       # 0 ~ 2번 행 (끝 미포함)
print(df.iloc[:3, 2])    # 0 ~ 2번 행, 2번 열(서초)
print(df.iloc[:3, 1:3])  # 0 ~ 2번 행, 1 ~ 2번 열(강북, 서초)
```

```
df.iloc[2]:
 강남    6
강북    7
서초    8
Name: 3월, dtype: int64

df.iloc[:3]:
     강남  강북  서초
1월   0   0   0
2월   3   4   5
3월   6   7   8

df.iloc[:3, 2]:
 1월    0
2월    5
3월    8
Name: 서초, dtype: int64

df.iloc[:3, 1:3]:
     강북  서초
1월   0   0
2월   4   5
3월   7   8
```

**loc vs iloc 비교**

||`loc`|`iloc`|
|---|---|---|
|기준|라벨(이름)|정수 위치(번호)|
|슬라이싱 끝|✅ **포함**|❌ **미포함**|
|예시|`df.loc[:'2월']`|`df.iloc[:3]`|

---

## ✅ 재색인·불리언·loc/iloc 핵심 정리

- `reindex` : 인덱스 재정렬 + 없는 인덱스 추가 (기본값 NaN)
- `fill_value` : 없는 인덱스를 특정 값으로 채움
- `method='ffill'` : 앞 값으로 전파 / `method='bfill'` : 뒷 값으로 전파
- `df[조건]` : 불리언 마스크로 행 필터링
- `df[df < 3] = 0` : 조건에 맞는 **값**을 일괄 치환
- `loc` : 라벨 기반, 슬라이싱 끝 **포함**
- `iloc` : 정수 기반, 슬라이싱 끝 **미포함**

---

---

# ➕ Pandas — 연산 & NaN 처리 & 통계 메서드

## 🔢 Series / DataFrame 연산

같은 인덱스끼리 연산하며, **인덱스가 일치하지 않으면 NaN** 이 된다.

### Series 연산

```python
from pandas import Series, DataFrame
import numpy as np

s1 = Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = Series([4, 5, 6, 7], index=['a', 'b', 'd', 'c'])
print(s1)
print(s2)

# 같은 인덱스끼리 연산, 불일치 인덱스(d) → NaN
print(s1 + s2)
print(s1.add(s2))   # + 연산자와 동일

print(s1.mul(s2))   # 곱셈 (sub, div 도 동일하게 사용)
```

```
s1:
 a    1
b    2
c    3
dtype: int64

s2:
 a    4
b    5
d    6
c    7
dtype: int64

s1 + s2:
 a     5.0
b     7.0
c    10.0
d     NaN   ← s1에 'd' 없음
dtype: float64

s1.mul(s2):
 a     4.0
b    10.0
c    21.0
d     NaN
dtype: float64
```

### DataFrame 연산

```python
df1 = DataFrame(np.arange(9).reshape(3, 3), columns=list('kbs'),
                index=['서울', '대전', '부산'])
df2 = DataFrame(np.arange(12).reshape(4, 3), columns=list('kbs'),
                index=['서울', '대전', '제주', '광주'])

# 불일치 인덱스(부산, 제주, 광주) → NaN 발생
print(df1 + df2)

# fill_value=0 : NaN을 0으로 대체한 후 연산에 참여
print(df1.add(df2, fill_value=0))
```

```
df1 + df2 (NaN 발생):
       k    b     s
광주  NaN  NaN   NaN
대전  6.0  8.0  10.0
부산  NaN  NaN   NaN
서울  0.0  2.0   4.0
제주  NaN  NaN   NaN

df1.add(df2, fill_value=0):
       k     b     s
광주  9.0  10.0  11.0
대전  6.0   8.0  10.0
부산  6.0   7.0   8.0
서울  0.0   2.0   4.0
제주  6.0   7.0   8.0
```

> 💡 `fill_value` : 인덱스 불일치로 생기는 NaN을 연산 **전에** 지정값으로 대체한다.

### 연산 메서드 정리

|연산자|메서드|
|---|---|
|`+`|`.add(fill_value=)`|
|`-`|`.sub(fill_value=)`|
|`*`|`.mul(fill_value=)`|
|`/`|`.div(fill_value=)`|

---

## 🚫 NaN (결측값) 처리

```python
df = DataFrame([[1.4, np.nan], [7, -4.5], [np.nan, np.nan], [0.5, -1]],
               columns=['one', 'two'])
print(df)
```

```
원본:
    one  two
0  1.4  NaN
1  7.0 -4.5
2  NaN  NaN
3  0.5 -1.0
```

### `dropna()` — NaN 포함 행/열 제거

```python
# how='any' (기본값) : NaN이 하나라도 있는 행 삭제
print(df.dropna())

# how='all' : 모든 값이 NaN인 행만 삭제 (행 2만 해당)
print(df.dropna(how='all'))

# subset : 특정 열에 NaN이 있는 행만 삭제
print(df.dropna(subset=['one']))

# axis='columns' : NaN 포함 열 전체 삭제
print(df.dropna(axis='columns'))
```

```
dropna() (how='any'):
    one  two
1  7.0 -4.5
3  0.5 -1.0

dropna(how='all'):
    one  two
0  1.4  NaN
1  7.0 -4.5
3  0.5 -1.0

dropna(subset=['one']):
    one  two
0  1.4  NaN
1  7.0 -4.5
3  0.5 -1.0

dropna(axis='columns'):
 Empty DataFrame  ← 모든 열에 NaN이 있어 전부 삭제됨
Columns: []
Index: [0, 1, 2, 3]
```

|옵션|설명|
|---|---|
|`how='any'`|NaN이 **하나라도** 있으면 삭제 (기본값)|
|`how='all'`|**모두** NaN일 때만 삭제|
|`subset=['열']`|특정 열 기준으로 판단|
|`axis='rows'`|행 방향으로 삭제 (기본)|
|`axis='columns'`|열 방향으로 삭제|

---

## 📐 통계 메서드

```python
# sum() : NaN은 자동으로 제외하고 합산
print(df.sum())           # 열 방향 합계 (axis=0, 기본값)
print(df.sum(axis=1))     # 행 방향 합계
```

```
sum():
 one    8.9
two   -5.5
dtype: float64

sum(axis=1):
 0    1.4   ← two가 NaN이지만 자동 제외
1    2.5
2    0.0   ← 둘 다 NaN → 0 (skipna=True 기본값)
3   -0.5
dtype: float64
```

> - `skipna=True` (기본값) : NaN을 건너뛰고 계산
> - `skipna=False` : NaN이 있으면 결과도 NaN

### `describe()` — 요약 통계량

```python
print(df.describe())   # 수치형 열의 요약 통계량
```

```
              one       two
count  3.000000  2.000000   ← NaN 제외 개수
mean   2.966667 -2.750000
std    3.521837  2.474874
min    0.500000 -4.500000
25%    0.950000 -3.625000
50%    1.400000 -2.750000
75%    4.200000 -1.875000
max    7.000000 -1.000000
```

```python
# 문자열(object) Series의 describe()
words = Series(['봄', '여름', '가을', '봄'])
print(words.describe())
```

```
문자열 describe():
 count     4      ← 전체 개수
unique    3      ← 고유값 개수
top       봄      ← 최빈값
freq      2      ← 최빈 빈도
dtype: object
```

**타입별 describe() 출력 비교**

|타입|출력 항목|
|---|---|
|🔢 수치형|count, mean, std, min, 25%, 50%, 75%, max|
|🔤 문자열|count, unique, top(최빈값), freq(최빈 빈도)|

---

## ✅ 연산·NaN·통계 핵심 정리

- 인덱스 불일치 연산 → NaN 발생, `fill_value`로 사전 대체 가능
- `dropna(how='any')` : NaN 하나라도 → 삭제 / `how='all'` : 전부 NaN일 때만 삭제
- `drop()` 은 기본적으로 원본을 바꾸지 않음 → `inplace=True` 로 원본 수정
- `sum(skipna=True)` : NaN 제외하고 합산 (기본값)
- `describe()` : 수치형은 기술통계, 문자열은 빈도 통계 출력

---

#numpy #pandas #Series #DataFrame #로그변환 #재색인 #불리언인덱싱 #loc #iloc #NaN #결측값 #데이터분석 #머신러닝
