# Day 35_MariaDB 데이터 분석

## 📅 2026-03-25

---
# 🧩 pandasdb_quiza.py — MariaDB 연동 종합 실습 (Quiz A)

> **핵심 주제:** pymysql로 MariaDB 연결, DataFrame 변환, 파일 저장, pivot_table, crosstab, left join, quantile, groupby 시각화

---

## 🔧 기본 세팅

```python
import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import csv

config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123',
    'database': 'test',
    'port': 3306,
    'charset': 'utf8'
}
```

**포인트**

- `pymysql` : Python에서 MariaDB/MySQL에 접속하기 위한 드라이버
- `koreanize_matplotlib` : 한글 폰트 설정을 자동으로 처리
- `config` dict를 `**config`로 언패킹하여 `pymysql.connect()`에 전달

---

## 1. DB 연결 & 사번/이름/부서명/연봉/직급 DataFrame 작성

```python
conn = pymysql.connect(**config)
cursor = conn.cursor()

sql = """
    select jikwonno, jikwonname, busername, jikwonpay, jikwonjik
    from jikwon inner join buser on jikwon.busernum=buser.buserno
"""
cursor.execute(sql)

df = pd.DataFrame(cursor.fetchall(),
                   columns=['사번', '이름', '부서명', '연봉', '직급'])
print(df.head(3))
```

**포인트**

- `INNER JOIN` : 두 테이블에서 조건이 일치하는 행만 결합
- `cursor.fetchall()` : 전체 결과를 튜플 리스트로 반환
- `pd.DataFrame(rows, columns=[...])` : 컬럼명과 함께 DataFrame 생성

---

## 2. DataFrame → CSV 파일로 저장 & 다시 읽기

```python
with open('jikwoninfo.csv', mode='w', encoding='utf-8') as fobj:
    writer = csv.writer(fobj)
    writer.writerow(df.columns)     # 헤더 행 저장
    writer.writerows(df.values)     # 데이터 행 저장

df2 = pd.read_csv('jikwoninfo.csv')
print(df2.head(3))
```

**포인트**

- `csv.writer` : Python 표준 라이브러리로 CSV 직접 작성
- `df.columns` : 컬럼명 리스트 → 헤더로 사용
- `df.values` : 데이터를 numpy 배열로 반환 → `writerows()`로 일괄 저장
- `pd.read_csv()` : 저장한 CSV를 다시 DataFrame으로 로드

---

## 3. 부서명별 연봉 합계 / 최대 / 최소 — pivot_table

```python
result = pd.pivot_table(df2, index='부서명', values='연봉', aggfunc=['sum', 'max', 'min'])
result.columns = ['연봉합', '최대', '최소']
print(result)
```

**포인트**

- `pivot_table(index=, values=, aggfunc=[...])` : 여러 집계 함수를 한 번에 적용
- `aggfunc` 에 리스트를 넣으면 MultiIndex 컬럼이 생성됨 → `.columns` 에 직접 이름 지정으로 단순화

---

## 4. 부서명 × 직급 교차 테이블 — crosstab

```python
ctab = pd.crosstab(df['부서명'], df['직급'], margins=True)
print('교차표\n', ctab)
```

**포인트**

- `pd.crosstab(행, 열)` : 두 범주형 변수의 빈도(교차)표 생성
- `margins=True` : 행/열 합계(All)를 자동으로 추가

---

## 5. 직원별 담당 고객 출력 — LEFT OUTER JOIN + fillna

```python
sql = """
    select jikwonno, jikwonname, gogekno, gogekname, gogektel
    from jikwon left outer join gogek on jikwon.jikwonno=gogek.gogekdamsano
"""
df3 = pd.read_sql(sql, conn)
df3 = df3.fillna("담당 고객 X")
print(df3)
```

**포인트**

- `LEFT OUTER JOIN` : 왼쪽 테이블(jikwon)의 모든 행을 유지, 매칭되는 gogek 없으면 NULL
- `pd.read_sql(sql, conn)` : SQL 실행 + 바로 DataFrame으로 변환
- `fillna("담당 고객 X")` : NaN(NULL) 값을 지정 문자열로 채움

---

## 6. 연봉 상위 20% 직원 출력 — quantile

```python
threshold = df2['연봉'].quantile(0.8)   # 상위 20% = 하위 80% 기준값
print(df2[df2['연봉'] >= threshold])
```

**포인트**

- `quantile(0.8)` : 전체 데이터의 80번째 백분위수(= 상위 20% 기준선)
- `df[df['연봉'] >= threshold]` : boolean indexing으로 조건에 맞는 행만 필터링

---

## 7. SQL 1차 필터링 후 pandas 분석 — 연봉 상위 50% 직급별 평균

```python
sql = "select jikwonjik as 직급, jikwonpay as 연봉 from jikwon"
df4 = pd.read_sql(sql, conn)

pay_median = df4['연봉'].median()            # 중앙값 = 상위 50% 기준선
df4 = df4[df4['연봉'] >= pay_median]         # 연봉 상위 50%만 필터링

df4_pivot = df4.pivot_table(values='연봉', index='직급', aggfunc='mean')
print(df4_pivot)
```

**분석 흐름**

```
SQL로 전체 직급/연봉 읽기
    ↓
pandas로 중앙값 계산 (median)
    ↓
중앙값 이상인 행만 필터링 (상위 50%)
    ↓
직급별 평균 연봉 pivot_table로 출력
```

**포인트**

- `median()` : 데이터를 정렬했을 때 정중앙 값 (= 50번째 백분위수)
- SQL에서 모두 필터링하는 것보다 **1차 SQL + 2차 pandas 분석** 패턴이 유연성 있음

---

## 8. 부서명별 평균 연봉 — 가로 막대 그래프

```python
buser_ypay = df.groupby(['부서명'])['연봉'].mean()   # 부서명별 연봉 평균
print(buser_ypay)

plt.barh(range(len(buser_ypay)), buser_ypay, alpha=0.4)     # 가로 막대
plt.yticks(range(len(buser_ypay)), buser_ypay.index)         # y축 눈금을 부서명으로
plt.xlabel('평균 연봉')
plt.ylabel('부서별')
plt.show()
```

**포인트**

- `groupby(['부서명'])['연봉'].mean()` : 부서명으로 그룹화 → 연봉 평균 계산 → Series 반환
- `plt.barh(x위치, 값)` : 가로 막대 그래프
- `plt.yticks(위치, 라벨)` : y축 눈금 위치와 라벨을 수동으로 지정 (부서명 표시)

---

## 9. 예외 처리 & 연결 종료

```python
except Exception as e:
    print('처리 오류 : ', e)

finally:
    cursor.close()
    conn.close()
```

**포인트**

- `try ~ except ~ finally` : DB 작업 중 오류가 발생해도 `finally`에서 반드시 연결 종료
- `Exception as e` : 모든 종류의 예외를 포괄적으로 잡음

---

## 📌 이 파일의 핵심 요약

|기능|코드|
|---|---|
|DB 연결|`pymysql.connect(**config)`|
|SQL 실행 + DataFrame|`pd.read_sql(sql, conn)`|
|CSV 저장|`csv.writer` + `writerows(df.values)`|
|부서별 집계|`pivot_table(aggfunc=['sum','max','min'])`|
|교차표|`pd.crosstab(행, 열, margins=True)`|
|NULL 처리|`df.fillna("담당 고객 X")`|
|상위 N% 필터|`df[df['연봉'] >= df['연봉'].quantile(0.8)]`|
|중앙값 기준 필터|`df[df['연봉'] >= df['연봉'].median()]`|
|그룹별 평균|`df.groupby('부서명')['연봉'].mean()`|
|가로 막대|`plt.barh()` + `plt.yticks()`|

---

# 🧩 pandasdb_quizb.py — MariaDB 연동 종합 실습 (Quiz B)

> **핵심 주제:** pymysql + pandas로 성별 연봉 분석, pivot_table, 세로 막대 그래프, crosstab

---

## 🔧 기본 세팅

```python
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123',
    'database': 'test',
    'port': 3306,
    'charset': 'utf8'
}
```

---

## 1. DB 연결 & 데이터 로드

```python
conn   = pymysql.connect(**config)
cursor = conn.cursor()

sql = """
    SELECT j.jikwonno   AS 사번,
           j.jikwonname AS 직원명,
           b.busername  AS 부서명,
           j.jikwongen  AS 성별,
           j.jikwonpay  AS 연봉
    FROM   jikwon j
    INNER JOIN buser b ON j.busernum = b.buserno
"""

df = pd.read_sql(sql, conn)
print(df.head(3))
```

**포인트**

- SQL에서 `AS` 로 컬럼 별칭을 지정하면 DataFrame 컬럼명이 자동으로 한글로 생성됨
- `pd.read_sql(sql, conn)` : SQL 실행과 DataFrame 변환을 한 번에 처리

---

## 2. 성별 연봉 평균 — pivot_table

```python
pt = pd.pivot_table(df, index='성별', values='연봉', aggfunc='mean')
print('성별 연봉 평균 (pivot_table)')
print(pt)
```

**포인트**

- `index='성별'` : 성별(남/여)을 행으로 그룹화
- `aggfunc='mean'` : 연봉의 평균값을 집계
- 결과는 성별별 평균 연봉을 담은 DataFrame

---

## 3. 성별 평균 연봉 — 세로 막대 그래프

```python
pt.plot(kind='bar', legend=False, color=['steelblue', 'coral'], rot=0)
plt.title('성별 평균 연봉')
plt.xlabel('성별')
plt.ylabel('연봉')
plt.tight_layout()
plt.show()
```

**포인트**

- `pt.plot(kind='bar')` : pandas DataFrame의 내장 plot 기능으로 바로 막대 그래프 생성
- `legend=False` : 범례 숨김 (성별이 이미 x축에 표시되므로 불필요)
- `color=['steelblue', 'coral']` : 남/여 막대 색상을 직접 지정
- `rot=0` : x축 라벨(남/여) 회전 각도 0도 (기본값은 90도로 기울어짐)
- `plt.tight_layout()` : 레이블이 잘리지 않도록 여백 자동 조정

---

## 4. 부서명 × 성별 교차 테이블 — crosstab

```python
ct = pd.crosstab(df['부서명'], df['성별'])
print('부서명 성별 교차 테이블')
print(ct)
```

**포인트**

- `pd.crosstab(행, 열)` : 두 범주형 변수의 조합별 빈도수를 표로 정리
- 결과: 각 부서에 남/여가 몇 명인지 한눈에 확인 가능

**출력 예시**

```
성별    남   여
부서명        
개발부   3    2
영업부   4    1
인사부   2    3
```

---

## 5. 예외 처리 & 연결 종료

```python
except pymysql.OperationalError as e:
    print('DB 오류 :', e)
except Exception as e:
    print('처리 오류 :', e)
finally:
    try:
        cursor.close()
        conn.close()
    except:
        pass
```

**포인트**

- `pymysql.OperationalError` : DB 연결 실패, 잘못된 쿼리 등 DB 관련 오류를 구체적으로 잡음
- `finally` 안에 `try~except` : cursor/conn이 생성되지 않았을 경우도 안전하게 처리
- quiza와의 차이: 예외 처리를 더 세분화하고 finally도 방어적으로 작성

---

## 📌 이 파일의 핵심 요약

|기능|코드|
|---|---|
|SQL 별칭으로 한글 컬럼|`SELECT col AS 한글명`|
|성별 평균 연봉|`pivot_table(index='성별', aggfunc='mean')`|
|DataFrame 내장 plot|`pt.plot(kind='bar', color=, rot=)`|
|부서×성별 빈도표|`pd.crosstab(df['부서명'], df['성별'])`|
|안전한 종료|`finally: try: cursor.close() except: pass`|

> **Quiz A vs Quiz B 비교**
> 
> - Quiz A : 다양한 집계(합/최대/최소), left join, quantile, groupby 시각화
> - Quiz B : 성별 분석에 집중, pandas 내장 plot 활용, 방어적 예외 처리 패턴

---

# 🧩 pandasdb_quizc.py — MariaDB 연동 종합 실습 (Quiz C)

> **핵심 주제:** 3테이블 JOIN, 로그인 입력 처리, 성별 연봉 분포 시각화 (boxplot + histplot)

---

## 🔧 기본 세팅

```python
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import os

config = {
    'host': '127.0.0.1',
    'password': '123',
    'user': 'root',
    'database': 'test',
    'port': 3306,
    'charset': 'utf8'
}
```

---

## 1. DB 연결 & 3테이블 JOIN으로 데이터 로드

```python
conn = pymysql.connect(**config)
cursor = conn.cursor()

sql = """
    select 
        jikwonno, jikwonname, busername, jikwonjik, busertel, jikwongen, jikwonpay, 
        gogekno, gogekname, gogektel 
    from jikwon 
    inner join buser on jikwon.busernum = buser.buserno
    left outer join gogek on jikwon.jikwonno = gogek.gogekdamsano
"""
cursor.execute(sql)

df_raw = pd.DataFrame(cursor.fetchall(),
    columns=['사번', '이름', '부서명', '직급', '부서전화', '성별', '연봉',
             '고객번호', '고객명', '고객전화'])

df = df_raw.drop_duplicates(subset=['사번'])
```

**포인트**

- `INNER JOIN buser` : 직원-부서 조인 (부서 없는 직원 제외)
- `LEFT OUTER JOIN gogek` : 담당 고객이 없는 직원도 포함 (NULL로 채워짐)
- 고객을 여러 명 담당하는 직원은 행이 중복될 수 있음
- `drop_duplicates(subset=['사번'])` : 사번 기준으로 중복 행 제거 → 직원 1명 = 1행

---

## 2. 분석용 DataFrame 정리

```python
dfc = df[['사번', '이름', '부서명', '직급', '부서전화', '성별', '연봉']]
dfc = dfc.rename(columns={'이름': '직원명'})
```

**포인트**

- 필요한 컬럼만 선택하여 분석용 DataFrame `dfc` 생성
- `rename(columns={구컬럼: 새컬럼})` : 특정 컬럼명만 변경

---

## 3. 사번 & 직원명 입력받아 로그인 처리

```python
while True:
    jikwonno = input('사번을 입력하세요. 종료:q\t')

    if jikwonno == 'q':
        break

    if not jikwonno.isdigit():
        print('사번은 숫자만 입력하세요\n')
        continue

    jikwonname = input('이름을 입력하세요. 종료:q\t')
    if jikwonname == 'q':
        break

    if any(dfc['사번'] == int(jikwonno)):
        name = dfc[dfc['사번'] == int(jikwonno)]['직원명'].iloc[0]
        if name == jikwonname:
            # 로그인 성공 → 이후 출력/시각화 실행
            ...
        else:
            print("사번과 이름 정보가 일치하지 않습니다.\n")
    else:
        print("존재하지 않는 사번 입니다.\n")
```

**입력 처리 흐름**

```
사번 입력
  ├─ 'q' → 종료
  ├─ 숫자 아님 → 경고 후 재입력
  └─ 숫자 → 이름 입력
              ├─ 'q' → 종료
              ├─ 사번 없음 → "존재하지 않는 사번"
              ├─ 이름 불일치 → "정보 불일치"
              └─ 일치 → 로그인 성공 → 출력 + 시각화
```

**포인트**

- `jikwonno.isdigit()` : 입력값이 숫자로만 이루어져 있는지 확인
- `any(dfc['사번'] == int(jikwonno))` : 사번이 DataFrame에 존재하는지 boolean으로 확인
- `dfc[조건]['직원명'].iloc[0]` : 조건에 맞는 행의 직원명 첫 번째 값 추출
- `iloc[0]` : 인덱스 위치 기반으로 첫 번째 요소 접근

---

## 4. 로그인 성공 시 — 전 직원 출력

```python
print(dfc.drop(columns=['연봉'], axis=1))
print("인원수 : ", dfc['사번'].count(), "명")
```

**포인트**

- `drop(columns=['연봉'])` : 연봉 컬럼을 제외하고 출력 (민감 정보 보호)
- `df['사번'].count()` : null이 아닌 사번의 수 = 전체 직원 수

---

## 5. 성별 연봉 분리

```python
male   = dfc[dfc['성별'] == '남']['연봉']
female = dfc[dfc['성별'] == '여']['연봉']
```

**포인트**

- boolean indexing으로 남/여 각각의 연봉 Series를 분리
- 이후 시각화에서 독립적으로 사용

---

## 6. 2×2 subplot 구성

```python
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(15, 10)
```

**포인트**

- `((ax1,ax2),(ax3,ax4))` : 2행 2열 subplot을 한 번에 변수에 언패킹
- `set_size_inches(15, 10)` : 전체 그림 크기를 15×10인치로 지정

---

## 7. 성별 연봉 분포 + 이상치 확인 — boxplot

```python
sns.boxplot(y=male,   ax=ax1)
sns.boxplot(y=female, ax=ax2)

ax1.set(xlabel='남성', ylabel='연봉[원]', title='남성 연봉 분포')
ax2.set(xlabel='여성', ylabel='연봉[원]', title='여성 연봉 분포')
```

**포인트**

- `sns.boxplot(y=Series)` : 단일 Series의 분포를 박스플롯으로 표시
- 박스플롯으로 중앙값, IQR, 이상치를 한눈에 확인
- `ax.set(...)` : 제목, 축 레이블을 한 번에 설정

---

## 8. 남/여 연봉 분포 비교 — histplot

```python
sns.histplot(data=male,   bins=10, ax=ax3)
sns.histplot(data=female, bins=10, ax=ax4)

ax3.set(xlabel='연봉[원]', ylabel='인원수[명]', title='남성 연봉 분포 비교')
ax4.set(xlabel='연봉[원]', ylabel='인원수[명]', title='여성 연봉 분포 비교')

plt.show()
break
```

**포인트**

- `sns.histplot()` : seaborn의 히스토그램 (기본 KDE 없음, displot보다 ax 지정 편리)
- `bins=10` : 구간(막대) 수를 10개로 지정
- 박스플롯(분포 요약) + 히스토그램(실제 분포 형태) 을 함께 보면 분석이 풍부해짐
- 로그인 성공 시 시각화 후 `break`로 루프 종료

---

## 9. 예외 처리

```python
except pymysql.OperationalError as e:
    print(e)

finally:
    cursor.close()
    conn.close()
```

---

## 📌 이 파일의 핵심 요약

|기능|코드|
|---|---|
|3테이블 JOIN|`INNER JOIN` + `LEFT OUTER JOIN`|
|중복 제거|`df.drop_duplicates(subset=['사번'])`|
|컬럼명 변경|`df.rename(columns={구:신})`|
|숫자 입력 검증|`input().isdigit()`|
|존재 여부 확인|`any(df['사번'] == int(입력값))`|
|값 추출|`df[조건]['컬럼'].iloc[0]`|
|컬럼 제외 출력|`df.drop(columns=['연봉'])`|
|성별 분리|`df[df['성별']=='남']['연봉']`|
|2×2 subplot 언패킹|`((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)`|
|연봉 분포 박스플롯|`sns.boxplot(y=Series, ax=)`|
|연봉 히스토그램|`sns.histplot(data=Series, bins=, ax=)`|

> **boxplot vs histplot**
> 
> - `boxplot` : 중앙값, IQR, 이상치를 요약해서 보여줌 → **분포 요약**
> - `histplot` : 각 구간에 얼마나 많은 데이터가 있는지 보여줌 → **분포 형태 파악**
> - 두 가지를 함께 쓰면 데이터를 더 입체적으로 분석할 수 있다.

---

# 🗄️ pandasdb3.py — SQLAlchemy로 원격 DB 연동

> **핵심 주제:** SQLAlchemy 엔진으로 MariaDB 연결, DataFrame을 원격 DB 테이블에 저장, 읽기

---

## 🔧 기본 세팅

```python
import pandas as pd
from sqlalchemy import create_engine
```

**포인트**

- `sqlalchemy` : Python의 대표적인 ORM/DB 연결 라이브러리
- `pymysql`은 직접 커서를 다루는 저수준 방식, `sqlalchemy`는 더 추상화된 고수준 방식
- `pip install sqlalchemy` 로 설치

---

## 1. DataFrame 생성

```python
data = {
    'code': [10, 11, 12],
    'sang': ['사이다', '맥주', '와인'],
    'su':   [20, 22, 5],
    'dan':  [5000, '3000', '70000']   # ⚠️ 숫자와 문자열이 혼재
}

frame = pd.DataFrame(data)
print(frame)
```

**포인트**

- `dan` 컬럼에 정수(`5000`)와 문자열(`'3000'`, `'70000'`)이 섞여 있음
- pandas는 타입이 혼재하면 전체를 `object(문자열)` 타입으로 처리
- DB 저장 시 타입 불일치 문제가 생길 수 있으므로 실무에서는 타입 통일 권장

---

## 2. SQLAlchemy 엔진 생성

```python
engine = create_engine("mysql+pymysql://root:123@127.0.0.1:3306/test?charset=utf8")
```

**Connection URL 구조**

```
mysql+pymysql :// root : 123 @ 127.0.0.1 : 3306 / test ? charset=utf8
     ↑              ↑     ↑        ↑          ↑      ↑         ↑
  드라이버         유저  비밀번호   호스트     포트   DB명    인코딩
```

- `mysql+pymysql` : MySQL/MariaDB를 pymysql 드라이버로 접속
- `create_engine()` : 실제 연결은 즉시 맺지 않고, 사용 시점에 연결 (Lazy connection)

---

## 3. DataFrame → DB 테이블에 저장

```python
frame.to_sql(name="sangdata", con=engine, if_exists='append', index=False)
```

**포인트**

- `name="sangdata"` : 저장할 테이블 이름
- `con=engine` : pymysql의 `conn`이 아닌 sqlalchemy `engine` 객체 사용
- `if_exists='append'` : 테이블이 이미 있으면 기존 데이터에 이어서 추가
- `index=False` : DataFrame의 인덱스(0,1,2)를 DB 컬럼으로 저장하지 않음

**`if_exists` 옵션 정리**

|값|동작|
|---|---|
|`'fail'`|테이블 존재하면 오류 (기본값)|
|`'replace'`|기존 테이블 삭제 후 새로 생성|
|`'append'`|기존 테이블에 행 추가|

---

## 4. DB에서 데이터 읽기

```python
df = pd.read_sql("select * from sangdata", engine)
print(df)
```

**포인트**

- `pd.read_sql()` 에 `engine` 객체를 직접 넘길 수 있음 (pymysql conn과 동일하게 사용)
- 저장한 데이터가 제대로 들어갔는지 바로 확인

---

## 5. 예외 처리

```python
except Exception as e:
    print("처리 오류:", e)
```

---

## 📌 pymysql vs SQLAlchemy 비교

|구분|pymysql|SQLAlchemy|
|---|---|---|
|방식|저수준 (커서 직접 조작)|고수준 (엔진 기반 추상화)|
|연결|`pymysql.connect(**config)`|`create_engine(URL)`|
|DataFrame 저장|직접 지원 안 함 (수동 INSERT 필요)|`df.to_sql(con=engine)`|
|DataFrame 읽기|`pd.read_sql(sql, conn)`|`pd.read_sql(sql, engine)`|
|권장 상황|간단한 쿼리, 커서 직접 제어|DataFrame ↔ DB 저장/읽기, ORM|

> 💡 `pd.to_sql()`을 사용하려면 SQLAlchemy engine이 필요하다.  
> pymysql의 `conn`을 직접 넘기면 최신 pandas 버전에서 경고 또는 오류가 발생할 수 있다.

---

## 📌 이 파일의 핵심 요약

|기능|코드|
|---|---|
|SQLAlchemy 엔진 생성|`create_engine("mysql+pymysql://user:pw@host:port/db")`|
|DataFrame → DB 저장|`df.to_sql(name=, con=engine, if_exists=, index=False)`|
|DB → DataFrame 읽기|`pd.read_sql("SELECT ...", engine)`|

---
# 📂 프로젝트: fpro18db (Flask & MariaDB 연동 자료 조회)

## 1. 프로젝트 구조

Plaintext

```
fpro18db/
├── app.py             # Flask 서버 및 데이터 분석 로직 (Main)
└── templates/         # HTML 템플릿 파일 저장소
    ├── index.html     # 메인 화면
    └── dbshow.html    # 데이터 출력 화면
```

---

## 2. 소스 코드 및 상세 설명

### 📄 app.py

Flask를 이용해 웹 서버를 구동하며, DB에서 데이터를 가져와 Pandas로 가공하는 핵심 파일입니다.

```Python
from flask import Flask, render_template, request
import pymysql
import pandas as pd
import numpy as np
from markupsafe import escape

app = Flask(__name__)

# [설정] MariaDB 접속 정보
db_config = {
    'host':'127.0.0.1', 
    'user':'root', 
    'password':'123',
    'database':'test', 
    'port':3306, 
    'charset':'utf8mb4'
}

# [함수] DB 연결 객체 생성 함수
def get_connection():
    return pymysql.connect(**db_config)

@app.route("/")
def index():
    """메인 페이지 렌더링"""
    return render_template("index.html")

@app.route("/dbshow", methods=['GET', 'POST'])
def dbshow():
    """DB 데이터를 조회하고 Pandas로 통계를 내어 화면에 전달"""
    
    # 1. 사용자가 입력한 부서명 파라미터 가져오기 (GET 방식)
    dept = request.args.get("dept", "").strip()

    # 2. SQL문 작성: jikwon 테이블과 buser 테이블을 Join하여 부서명 추출
    sql = """
        select j.jikwonno as 직원번호, j.jikwonname as 직원명, b.busername as 부서명,
        b.busertel as 부서전화, j.jikwonpay as 연봉, j.jikwonjik as 직급
        from jikwon j
        inner join buser b on j.busernum=b.buserno
    """
    params = []
    
    # 부서명 검색어가 있을 경우 조건절 추가
    if dept:
        sql += " where b.busername like %s"
        params.append(f"%{dept}%") # Partial match (포함하는 단어 검색)

    sql += " order by j.jikwonno asc"

    # 3. 데이터베이스 실행 및 결과 로드
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() # 실제 데이터(튜플 형태)
            cols = [c[0] for c in cur.description] # 컬럼명 리스트 추출

    # 4. Pandas DataFrame으로 변환 (분석 용이성)
    df = pd.DataFrame(rows, columns=cols)

    # 5. 직원 상세 정보 처리 (HTML 표 형식으로 변환)
    if not df.empty:
        # 필요한 컬럼만 선택하여 HTML Table 태그로 변환
        jikwondata = df[['직원번호', '직원명', '부서명', '부서전화', '연봉']].to_html(index=False)
    else:
        jikwondata = "직원 정보가 없어요"

    # 6. 직급별 연봉 통계 분석 (Pandas의 강력한 집계 기능)
    if not df.empty:
        stats_df = (
            df.groupby("직급")["연봉"] # 직급으로 그룹화하여 연봉 계산
            .agg(
                평균 = "mean",                   # 평균 연봉
                표준편차=lambda x:x.std(ddof=0),  # 모집단 표준편차
                인원수="count"                   # 해당 직급 인원수
            )
            .round(2)                          # 소수점 2자리 반올림
            .reset_index()                     # 인덱스를 컬럼으로 전환
            .sort_values(by='평균', ascending=False) # 평균 기준 내림차순 정렬
        )
        stats_df['표준편차'] = stats_df['표준편차'].fillna(0) # 데이터가 1개일 때 발생하는 결측치 처리
        statsdata = stats_df.to_html(index=False)
    else:
        statsdata = "통계 대상 자료가 없어요"

    # 7. 결과를 템플릿(HTML)으로 전송
    return render_template("dbshow.html",
                           dept=escape(dept),      # 보안(XSS)을 위한 이스케이프 처리
                           jikwondata=jikwondata,  # 직원 목록 표
                           statsdata=statsdata)    # 통계 표

if __name__=='__main__':
    # Flask 앱 실행 (디버그 모드 활성화 시 코드 수정 시 자동 재시작)
    app.run(debug=True)
```

### 📄 index.html

```HTML
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>인사 관리 시스템</title>
</head>
<body>
    <h2>사내 인사 자료 조회</h2>
    <p><a href="/dbshow">전체 직원 및 통계 정보 보러가기</a></p>
</body>
</html>
```

### 📄 dbshow.html

```HTML
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>조회 결과</title>
</head>
<body>
    <h2>사내 인사 자료 조회</h2>
    <a href="/dbshow">전체 데이터 다시보기</a>

    </body>
</html>
```

---

## 3. 핵심 포인트 정리

1. **DB 연동 프로세스**:
    
    - `pymysql`을 통해 DB 연결 → `cursor`를 통해 SQL 실행 → `fetchall()`로 결과 수집.
        
2. **Pandas의 활용**:
    
    - 단순한 데이터 조회를 넘어 `groupby`와 `agg`를 통해 웹 서버 단에서 **데이터 분석(평균, 표준편차)**을 즉석에서 수행.
        
3. **데이터 시각화 (Web)**:
    
    - `df.to_html()`을 사용하여 복잡한 `<table>` 태그 생성을 자동화함.
        
4. **보안 (Security)**:
    
    - `markupsafe.escape`를 사용하여 사용자 입력값(`dept`)을 통한 스크립트 공격(XSS)을 방어함.
        
5. **예외 처리**:
    
    - `if not df.empty`를 통해 조회 결과가 없을 경우 발생할 수 있는 오류를 사전에 방지함.