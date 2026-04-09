# Day 30_데이터 분석 개론 및 Numpy & 벡터 기초 실무
# 📅 2026-03-18

---
# 📊 데이터 분석 개론 - 1강

## 1. 데이터 분석이란?

입력 데이터가 주어졌을 때, 데이터 사이의 관계를 파악하거나
그 관계를 이용해 새로운 출력 데이터를 만들어내는 과정.

> 쉽게 말하면: "데이터를 보고 뭔가를 알아내거나, 만들어내는 것"

---

## 2. 데이터 분석의 유형

### 🔮 예측 (Prediction)
- 가장 많이 쓰이는 유형
- 입력 데이터 → 다른 데이터 출력
- ⚠️ "미래 예측"과는 다름! 시간 개념 없음
- 예시:
  - 꽃잎 크기 → 식물 종 분류
  - 얼굴 사진 → 사람 이름 출력

### 📈 미래예측 (Forecasting)
- 시계열 데이터를 분석해서 진짜 미래를 예측
- 예시: 주가 예측, 날씨 예보

### 🗂️ 클러스터링 (Clustering)
- 비슷한 데이터끼리 자동으로 묶는 것
- 예시: 고객 유형 분류

### 📐 모사 (Approximation)
- 복잡한 현상을 수학적으로 비슷하게 흉내내는 것
- 완벽한 모델 대신 "최대한 비슷하게" 재현

---

## 3. 룰 기반 vs 머신러닝

| 구분     | 룰 기반            | 머신러닝            |
| ------ | --------------- | --------------- |
| 규칙 출처  | 사람이 직접 작성       | 데이터에서 스스로 학습    |
| 유연성    | 낮음 (예외에 약함)     | 높음 (새 패턴 적응 가능) |
| 설명 가능성 | 명확함             | 불투명할 수 있음       |
| 한계     | 규칙 많아지면 유지보수 힘듦 | 데이터 많을수록 성능 향상  |

### 예시로 비교
- **룰 기반 스팸 필터**: "제목에 '무료' 포함 → 스팸 처리" (사람이 규칙 작성)
- **ML 기반 스팸 필터**: 수천 개 이메일 학습 → 스스로 스팸 패턴 파악

> 💡 요즘 AI (ChatGPT, 이미지 인식 등)는 대부분 머신러닝 기반!

---

## 🗒️ 핵심 요약
1. 데이터 분석 = 입력 → 관계 파악 → 출력
2. 예측 ≠ 미래예측 (forecasting은 따로 구분)
3. 룰 기반은 명확하지만 유연성 부족, ML은 복잡한 상황에 강함

---

## 📊 데이터 분석 개론 - 2강: 데이터 과학자 소양

---
## 1. 데이터 과학자의 3가지 필요조건

데이터 과학자는 세 가지 역량이 교차하는 사람이다.

| 역량 | 설명 |
|------|------|
| 🧠 비즈니스 지식 | 기획/영업 경험, 고객 심리 이해 |
| 📐 통계/설계 지식 | 통계 해석 능력, 연구 방법론 |
| 💻 IT/데이터 처리 | Python/R 등 프로그래밍 능력 |

---

## 2. 비즈니스 데이터 분석 흐름
```
현실 vs 이상 파악 → 문제 발견 → 데이터 수집·가공 → 분석 → 액션
```

> ⚠️ 복잡한 모델 ≠ 항상 좋은 결과. **문제에 맞는 기법 선택**이 핵심!

---

## 3. 문제 발견의 3가지 관점

### 1️⃣ 크기 보기 — 요인의 영향도 파악
### 2️⃣ 분해해서 보기 — 상호배제·전체포괄로 원인 요소 찾기
- 예: 매상 = 1인당 매상액 × 구매자 수

### 3️⃣ 비교해서 보기 — 문제 있는 데이터 vs 없는 데이터 비교

---

## 4. 데이터 수집·가공

### 수집 체크리스트
- [ ] 필요한 데이터가 무엇인가?
- [ ] 어딘가에 저장되어 있는가? (File / DB / Hadoop)
- [ ] 없다면 새로 수집하거나 대체 데이터를 쓸 수 있는가?

### 전처리
- **수치화**: 남/여 → 1/0 같이 범주형 데이터를 숫자로 변환
- **결측값 처리**: 빈 데이터를 제거하거나 평균값 등으로 대체

---

## 5. 데이터 분석의 두 종류

| 구분 | 의사결정 지원 | 자동화·최적화 |
|------|-------------|-------------|
| 목적 | 사람의 행동 결정 지원 | 컴퓨터가 스스로 판단 |
| 기법 | 단순집계, 크로스 집계 | 머신러닝, 알고리즘 구축 |

### 분석 방법 선택 기준
- **그룹 비교**: 두 집단 간 차이 분석
- **자료 추이**: 데이터 변화 흐름 파악
- **자료 예측**: 미래 데이터 예측

---

## 6. 액션

분석 완료 후 실행 단계. 어느 쪽이든 **설득(의사소통) 비용**이 중요.

- **의사결정 지원**: 분석 담당자 → 기획/경영진 설득 → 실행
- **자동화·최적화**: 분석 담당자 → 개발/운영팀 설득 → 시스템 구현

---

## 🗒️ 핵심 요약
1. 데이터 과학자 = 비즈니스 + 통계 + IT의 교차점
2. 분석 흐름: 문제 발견 → 수집·가공 → 분석 → 액션
3. 전처리(수치화, 결측값 처리)가 실무의 핵심
4. 분석 목적에 맞는 기법 선택이 가장 중요

---

## 🌐 참고 — 주요 플랫폼·데이터셋

### 🏆 Kaggle
- 데이터 사이언스 경진 대회 플랫폼 (2010년 설립, 구글 인수)
- 기업이 문제를 올리면 전 세계 참가자가 모델로 경쟁
- 브라우저에서 바로 Python 코드 작성·실행 가능 (노트북 환경 제공)
- 무료 데이터셋, 튜토리얼 강의도 제공
- 입문 추천: **타이타닉 생존 예측** (이진 분류 문제)
- 알고리즘 연습 = 백준/프로그래머스, 데이터 분석 연습 = **캐글**
- https://www.kaggle.com

### 🖼️ ImageNet
- 스탠퍼드·프린스턴 대학이 구축한 대규모 이미지 데이터셋
- 1,400만 장 이상, 21,841개 카테고리
- 컴퓨터 비전·딥러닝 연구 발전의 핵심 기반
- **ILSVRC 대회**: 매년 이 데이터로 이미지 분류 AI 경쟁
  - 2012년 AlexNet이 압도적 성능 → 현대 딥러닝 시대의 시작
- 오늘 배운 "얼굴 사진 → 사람 이름 출력" 같은 예측이 바로 이런 데이터로 학습한 모델
- https://www.image-net.org

---

## 📊 데이터 분석 개론 - 3강: Python을 이용한 데이터 분석

---

## 1. 빅데이터 (Big Data)

**정의**: 일반적인 DB 도구로 수집·저장·관리·분석하기 어려운 대규모 데이터 (Gartner)

### 3V 특성
| V | 의미 | 설명 |
|---|------|------|
| **Volume** | 규모 | TB, PB 단위의 대용량 |
| **Variety** | 다양성 | 정형(Structured) + 비정형(Unstructured) |
| **Velocity** | 속도 | 실시간 스트림, 배치 처리 |

### 데이터 분석 및 추론 절차
```
데이터 전처리
    ↕
기초통계분석 (EDA)
    ↕
변수추출 / 데이터 Manipulation
    ↕
분석모형선정
    ↕
데이터 분석
    ↕
분석모형평가
```

---

## 2. Python 데이터 분석 주요 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| **numpy** | 고속 수치 연산, 다차원 배열 |
| **pandas** | 데이터 표현 및 처리 (Series, DataFrame) |
| **matplotlib** | 시각화 (그래프) |
| **seaborn** | matplotlib 기반 고급 시각화 |
| **scipy** | 과학 분석 알고리즘 |
| **scikit-learn** | 머신러닝, 추론 통계 |
| **TensorFlow / PyTorch** | 딥러닝 |

> 참고 사이트: https://pydata.org

---

## 3. 통계학의 분류

### 기술통계 (Descriptive Statistics)
- 수집한 데이터를 요약·정리하는 통계
- 평균, 중위수, 최빈수, 분산, 표준편차 등
- **추론통계의 기초작업**을 수행하기 위한 과정

### 추론통계 (Inferential Statistics)
- 표본(sample)으로 모집단(population) 전체의 특성을 추론
- 상관분석, 회귀분석, 분류, 딥러닝 등

> 💡 기술통계 → 추론통계 순서로 진행

---

## 4. 주요 라이브러리 개요

### 🔢 Numpy
- 다차원 배열 객체 **ndarray** 제공
- Python 리스트보다 **빠르고 메모리 효율적**
- 주요 기능:
  - 배열 연산 (elementwise: +, -, *, /)
  - 슬라이싱, 전치(Transpose)
  - **브로드캐스팅**: 크기가 다른 배열 간 연산 가능
```python
import numpy as np
a = np.array([1, 2, 3])  # 1차원 배열 생성
```

### 🐼 Pandas
- 고수준 자료구조 **Series(1D)**, **DataFrame(2D)** 제공
- SQL과 유사한 데이터 조작 기능
- 주요 기능: 슬라이싱, 정렬, 결합, group by, pivot, 파일 읽기/저장
```python
from pandas import Series, DataFrame

# Series: 1차원 (인덱스 있는 배열)
obj = Series([3, 7, -5, 4], index=['a', 'b', 'c', 'd'])

# DataFrame: 2차원 (표 형태)
data = {
    'irum': ['홍길동', '한국인', '신기해'],
    'nai': [23, 25, 33]
}
df = DataFrame(data)
```

### 📈 Matplotlib / Seaborn
- **matplotlib**: 기본 시각화 (line, bar, scatter, histogram 등)
- **seaborn**: matplotlib 기반 고급 시각화, 색상 테마 강화
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.show()
```

---

## 5. 입력 데이터 vs 출력 데이터

| 구분 | 기호 | 다른 이름 |
|------|------|----------|
| **입력 데이터** | X | 독립변수, 특징(feature), 설명변수 |
| **출력 데이터** | Y | 종속변수, 라벨(label), 클래스(class) |

> 💡 입출력 데이터를 정확히 정의하는 것이 예측 문제 해결의 첫 번째 단계!

---

## 6. 기술통계 상세

### 기술통계량 유형
- **중심 경향값(대표값)**: 평균(mean), 중위수(median), 최빈값(mode)
- **산포도**: 표준편차(stddev), 분산(variance), 범위(range), 사분위범위(IQR)
- **분포도**: 왜도(Skewness), 첨도(Kurtosis)

### 척도(Scale)에 따른 데이터 분류
```
변수
├── 범주형 (정성적 - 수량화 불가)
│   ├── 명목형: 카테고리. 예) 성별(남/녀), 혈액형
│   └── 순서형: 순서 있음. 예) 학점(A,B,C,D,F)
│
└── 수치형 (정량적 - 수량화 가능)
    ├── 이산형: 정수, 셀 수 있음. 예) 2, 5, 10
    └── 연속형: 연속 수치. 예) 키 175.3, 180.2
```

---

## 7. 통계 분석 방법 선택 기준

| 독립변수 | 종속변수 | 분석 방법 |
|---------|---------|---------|
| 범주형 | 범주형 | **카이제곱 검정** |
| 범주형 (2집단) | 연속형 | **T검정** |
| 범주형 (3집단↑) | 연속형 | **ANOVA (분산분석)** |
| 연속형 | 범주형 | **로지스틱 회귀** |
| 연속형 | 연속형 | **선형회귀, 구조방정식** |

---

## 8. 주요 검정 방법

### 📊 카이제곱 검정 (χ² 검정)
- **두 범주형 변수** 사이의 관련성 분석
- 교차표(분할표)를 만들어 분석
- 3가지 목적: **독립성 / 적합도 / 동질성**
- `scipy.stats.chisquare()` 사용
```
예시: 성별 → 선호 커피브랜드 차이가 있는가?
귀무가설: 차이가 없다
대립가설: 차이가 있다
```

### 📊 T검정
- **범주형(2집단) → 연속형** 데이터 분석
- 두 집단의 **평균 차이** 검정
- `scipy.stats.ttest_ind()` 사용
```
예시: 성별 → A사 커피 만족도 차이가 있는가?
     서울 vs 부산 고등학생 수능점수 차이
```

### 📊 ANOVA (분산분석)
- **범주형(3집단↑) → 연속형** 데이터 분석
- T검정보다 더 많은 집단 비교 가능
- F검정 통계량 사용
```
예시: 직업(화이트칼라/블루칼라/주부/학생) → 커피 만족도 차이?
- 집단 간 유의미한 차이 있는지 확인
- 각 집단끼리 어떤 차이인지 확인
```

| F값(절대치) | 유의수준 α | 분야 |
|-----------|---------|------|
| ≥ 2.58 | 0.01 | 의·생명 |
| ≥ 1.96 | 0.05 | 사회과학 |
| ≥ 1.645 | 0.10 | 일반 |

---

## 🗒️ 핵심 요약
1. 빅데이터 = Volume + Variety + Velocity (3V)
2. Python 분석 핵심 라이브러리: numpy, pandas, matplotlib, scikit-learn
3. 입력(X) = 독립변수 / 출력(Y) = 종속변수
4. 기술통계(요약) → 추론통계(예측·검정) 순서
5. 분석 방법 선택은 **변수의 척도(범주형 vs 수치형)**에 따라 결정

---
## 🔢 4강: \[실습] 통계량 산출 및 원리 이해 (abc.py)

---
### 1. 실습 분석: 순수 파이썬 vs Numpy (`abc.py`)

이 실습은 통계량의 **수학적 원리**를 직접 코드로 구현해보고, 이를 **Numpy** 라이브러리로 얼마나 효율적으로 처리할 수 있는지 비교하는 핵심 과정입니다.

#### 💻 실습 코드 및 한 줄 설명

```Python
# pro5anal/abc.py
# 통계량 : 데이터의 특징을 하나의 숫자로 요약한 것.
# 표본 데이터를 추출해 전체(모집단) 데이터를 짐작 가능
# 평균, 분산, 표준편차 ...

grades = [1, 3, -2, 4]      # 변량 (Variable: 분석 대상이 되는 수치들)

# [1] 데이터 출력: 리스트 내부의 값을 하나씩 꺼내서 출력
def show_grades(grades):
    for g in grades:
        print(g, end = " ")

show_grades(grades)
print()

# [2] 합계 계산: tot 변수에 모든 변량을 누적해서 더함
def grades_sum(grades):
    tot = 0
    for g in grades:
        tot += g
    return tot

print('합은 ' ,grades_sum(grades))

# [3] 평균 계산: 산술 평균 (데이터의 총합 / 데이터의 개수)
def grades_ave(grades):
    ave = grades_sum(grades) / len(grades)
    return ave

print('평균은 ',grades_ave(grades))

# [4] 분산(Variance) 계산: 데이터가 평균으로부터 얼마나 퍼져 있는가?
# 원리: '편차(데이터-평균)의 제곱'들의 평균을 구함
def grades_variance(grades):
    ave = grades_ave(grades)
    vari = 0
    for su in grades:
        # (값 - 평균)을 제곱하는 이유: 음수를 양수화하고, 평균에서 멀수록 가중치를 줌
        vari += (su - ave) ** 2 
    return vari / len(grades)    # 분모가 n이면 모분산 (현재 코드)
    # return vari / (len(grades) - 1) # 분모가 n-1이면 표본분산

print('분산은 ',grades_variance(grades))

# [5] 표준 편차(Standard Deviation)
# 원리: 분산에 루트(√)를 씌움. 제곱된 단위를 원래 데이터 단위로 되돌리는 과정.
def grades_std(grades):
    return grades_variance(grades) ** 0.5  # ** 0.5는 루트 연산과 동일함

print('표준편차는 ',grades_std(grades))


# ---------------------------------------------------------
# [6] 넘파이(Numpy) 전원 함수 사용: 위 모든 로직이 한 줄로 압축됨
# ---------------------------------------------------------
print('\n넘파이 전원 함수 사용')
import numpy
print('합 ', numpy.sum(grades))      # np.sum()
print('평균 ', numpy.mean(grades))    # np.mean()
print('분산 ', numpy.var(grades))     # np.var()
print('표준편차 ', numpy.std(grades)) # np.std()
```

---
### 2. 🧠 메모리 구조 비교: Numpy Array vs Python List

![[Numpy Array vs Python List.png]]
#### 1. Python List: "포인터 참조 방식" (느림)
* **구조:** 리스트 객체는 실제 데이터가 아니라, 데이터가 저장된 **메모리 주소(Pointer)**의 목록을 가지고 있습니다.
* **특징:**
    * **Overhead:** 데이터를 하나 읽을 때마다 `주소 확인 -> 주소로 이동 -> 데이터 읽기`라는 단계를 거쳐야 합니다.
    * **비연속적:** 데이터들이 메모리 이곳저곳에 흩어져 있어 한꺼번에 불러오기 비효율적입니다.
    * **유연성:** 한 리스트 안에 숫자, 문자 등 다양한 타입을 섞어서 담을 수 있습니다.

#### 2. Numpy Array: "연속 메모리 방식" (빠름)
* **구조:** 같은 타입의 데이터가 메모리 상에 **빈틈없이 다닥다닥** 붙어 있습니다.
* **특징:**
    * **Contiguous Memory:** 컴퓨터가 데이터를 읽을 때 옆에 있는 데이터를 바로바로 가져올 수 있어 압도적으로 빠릅니다.
    * **No Pointers:** 주소를 찾아가는 번거로움이 없습니다.
    * **고정 타입:** 모든 데이터가 같은 타입이어야만 이 구조가 유지됩니다. (수치 계산 최적화)
    

---

### 🗒️ 핵심 요약

1. **통계량**은 데이터의 특징을 대표값 하나로 요약하여 전체를 파악하게 돕는다.
    
2. **분산**은 편차 제곱의 평균이며, 데이터가 얼마나 '흩어져' 있는지를 보여준다.
    
3. **Numpy**는 연속 메모리 구조를 활용해 파이썬 기본 리스트보다 훨씬 빠른 속도로 통계 연산을 수행한다.

---
## 🔢 5강: \[실습] Numpy ndarray의 특징 및 기초 (`numpy1.py`)

Numpy의 `ndarray`는 단순한 배열을 넘어 다차원 수치 구조를 효율적으로 처리하기 위한 도구입니다.

### 1. 💻 실습 코드 및 상세 설명

```Python
# pro5anal/numpy1.py
import numpy as np

# [1] 데이터 타입의 일관성
ss = ['tom', 'james', 'oscar', 1, True]  # 리스트는 여러 타입 혼합 가능
ss2 = np.array(ss)                       # ndarray는 상위 타입으로 자동 변환 (전부 문자열로 변환됨)

print(ss2, ' ', type(ss2))

# [2] 벡터화 연산 (Vectorized Operation)
li = list(range(1, 10))
print(li * 10)       # 리스트: 요소를 10번 반복함
num_arr = np.array(li)
print(num_arr * 10)  # ndarray: 각 요소에 10을 곱함 (산술 연산 가능)

# [3] 배열 생성 및 특수 행렬
a = np.array([1, 2, 3.5], dtype='float32') # dtype으로 타입 지정 가능
b = np.array([[1, 2, 3], [4, 5, 6]])      # 2차원 배열 (행렬)
print(b.shape)                            # 배열의 구조 확인 (2, 3)

c = np.zeros((2, 2))  # 모든 요소가 0인 행렬
d = np.ones((2, 2))   # 모든 요소가 1인 행렬
e = np.eye(3)         # 단위 행렬 (대각선만 1)

# [4] 난수 생성 (Random)
print(np.random.rand(5))    # 0~1 사이 균등 분포 난수
print(np.random.randn(5))   # 표준 정규 분포 난수
np.random.seed(0)           # 난수 고정 (재현성 확보)

# [5] 인덱싱 및 슬라이싱 (중요!)
a = np.array([1, 2, 3, 4, 5])
print(a[1:5:2])  # 시작:끝:간격 (2, 4 출력)

# [6] 주소 치환 vs 복사 (Copy)
b = a              # 주소만 전달 (a를 바꾸면 b도 바뀜)
c = np.copy(a)     # 실제 값을 복사 (별개의 데이터가 됨)
```

---
### 2. Numpy 핵심 원리 요약

#### 1️⃣ 타입 캐스팅 (Type Casting)

Numpy 배열은 모든 요소가 동일한 타입이어야 합니다. 여러 타입이 섞이면 아래 우선순위에 따라 자동으로 변환됩니다.

> **int $\rightarrow$ float $\rightarrow$ complex $\rightarrow$ str** (문자열이 가장 상위)

#### 2️⃣ 벡터화 연산 (Vectorization)

파이썬 리스트는 반복문(`for`)을 써야 요소별 계산이 가능하지만, Numpy는 배열 전체에 연산자를 적용하면 내부적으로 **모든 요소에 동시 적용**됩니다. 이것이 성능 차이의 핵심입니다.

#### 3️⃣ 얕은 복사(View) vs 깊은 복사(Copy)

- `b = a`: 단순히 이름표만 하나 더 붙이는 것입니다. `b`를 수정하면 원본 `a`도 오염됩니다.
    
- `c = np.copy(a)`: 메모리에 새로운 공간을 확보하여 내용을 복제합니다. 원본 데이터를 보호해야 할 때 필수입니다.

---
# 선형대수학: 벡터 기본 정리

## 1. 수학 객체의 계층 구조

데이터 분석에서 다루는 수치 데이터의 형태와 성질에 따른 분류입니다.

|**구분**|**정의**|**특징**|**Python 구현**|
|---|---|---|---|
|**스칼라(Scalar)**|숫자 1개|크기만 있고 방향 없음 (예: 50kg)|`x = 5`|
|**리스트(List)**|값의 나열|여러 타입 저장 가능, 수학 연산 부적합|`[1, 2, 3]`|
|**넘파이 배열**|수치 연산 배열|동일 타입 저장, 고속 대규모 연산 최적화|`np.array([1, 2])`|
|**벡터(Vector)**|방향 + 크기|1차원 배열에 기하학적 의미 부여|`np.array([3, 4])`|
|**행렬(Matrix)**|벡터의 집합|2차원 배열, 선형 변환/가중치 표현|`np.array([[1,2],[3,4]])`|

---

## 2. 벡터의 기하학적 의미

벡터는 공간에서의 **"위치, 방향, 이동"**을 나타내는 화살표입니다.

### 2.1 벡터의 크기 (노름, Norm)

- **정의**: 원점에서 벡터 끝점까지의 거리(길이)
    
- **공식 ($R^2$):** $\|v\| = \sqrt{x^{2} + y^{2}}$
    
- **예시**: $v = (3, 4)$ 일 때, $\|v\| = \sqrt{3^2 + 4^2} = 5$

### 2.2 단위벡터 (Unit Vector)

- **정의**: 길이가 1인 벡터. 방향 정보만 유지하고 크기를 1로 만든 것.
    
- **공식**: $\hat{a} = \dfrac{a}{\|a\|}$ (정규화, Normalization)


---

## 3. 기본 벡터 연산

![vector addition and scalar multiplication, AI로 생성](https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcTJwlrg-515Vbi1znOKcwffWsFRXvjQ27pIShWdbMJPqHLpjhtu_s2TQIkN881f4DejJAtPLeVb8g0j1QHE3wnphLPC_bUesiA5br_rgUFa7jqe27k)

### 3.1 벡터 덧셈 (Addition)

- **수식**: $a + b = (x_1 + x_2, y_1 + y_2)$
    
- **의미**: 기하학적으로 화살표를 "이어 붙이는" 결과물.
    

### 3.2 스칼라 곱 (Scalar Multiplication)

- **수식**: $ka = (kx, ky)$
    
- **의미**: 방향은 유지한 채 길이를 $k$배 변환.
    

---

## 4. 내적(Dot Product)과 각도

두 벡터가 "**얼마나 같은 방향을 향하는가**" 를 수치화한 값입니다.

### 4.1 계산 공식

- **좌표 기준**: $a \cdot b = a_1 b_1 + a_2 b_2$
    
- **기하적 정의**: $a \cdot b = \|a\| \|b\| \cos \theta$


### 4.2 내적 값 해석

- **$a \cdot b > 0$**: 예각 ($0^\circ \sim 90^\circ$), 비슷한 방향
    
- **$a \cdot b = 0$**: 직각 ($90^\circ$), **서로 수직(직교, Orthogonal)** ★핵심
    
- **$a \cdot b < 0$**: 둔각 ($90^\circ \sim 180^\circ$), 반대 방향에 가까움


### 4.3 각도 구하기

$$\cos \theta = \dfrac{a \cdot b}{\|a\| \|b\|}$$

$$\theta = \cos^{-1}\left(\dfrac{a \cdot b}{\|a\| \|b\|}\right)$$

---

## 5. 벡터 공간과 구성 요소

### 5.1 선형독립 (Linearly Independent)

어떤 벡터도 집합 내 다른 벡터들의 선형결합으로 표현할 수 없는 상태입니다.

- **독립**: $[1, 0], [0, 1]$ (서로 다른 축)
    
- **종속**: $[1, 0], [2, 0]$ (같은 직선 상에 존재)
    

### 5.2 기저(Basis)와 차원(Dimension)

- **기저**: 선형독립이면서 해당 공간의 모든 벡터를 표현할 수 있는 최소 벡터 집합.
    
- **차원**: 기저 벡터의 개수 (예: $\mathbb{R}^2$ 차원은 2).
    
- **Span(생성)**: 벡터들의 선형결합으로 도달할 수 있는 전체 공간의 영역.
    

---

## 6. 직교(Orthogonal)와 직교화

### 6.1 직교의 중요성

내적이 0인 직교 상태는 축들이 서로 독립적임을 뜻하며, 계산을 단순화하고 독립적인 분석을 가능하게 합니다.

### 6.2 그람-슈미트(Gram–Schmidt) 직교화

비직각 벡터들을 서로 수직인 벡터 집합으로 변환하는 알고리즘입니다.

- **원리**: $v_2$에서 $u_1$ 방향 성분(사영)을 제거하여 수직 성분만 남김.
    
- **사영(Projection) 공식**: $\text{proj}_{u}(v) = \dfrac{v \cdot u}{u \cdot u} u$
    

---

## 7. 직교행렬 (Orthogonal Matrix)

### 7.1 정의 및 성질

정사각행렬 $Q$가 $Q^T Q = QQ^T = I$ (즉, $Q^T = Q^{-1}$)를 만족할 때입니다.

- **벡터 길이 보존**: $\|Qx\| = \|x\|$
    
- **내적 보존**: $(Qx) \cdot (Qy) = x \cdot y$
    
- **의미**: 회전, 반사와 같이 각도와 크기를 유지하는 변환.
    

### 7.2 예시 (90도 회전 행렬)

$$Q = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$$

---

## 8. 최종 요약 💡

- **스칼라**: 숫자 1개 (크기만)
    
- **벡터**: 숫자 나열 (방향+크기)
    
- **행렬**: 벡터의 모음 (공간 변환)
    
- **벡터는 "행이 1개 또는 열이 1개인 행렬"** 로 간주할 수 있음.

---

## 🔢 6강:  \[실습] Numpy 배열 연산 총정리 (`numpy2.py`)

파이썬 리스트보다 압도적으로 빠른 Numpy의 핵심 기능(배열 연산, 내적, 브로드캐스팅 등)을 확인하는 실습입니다.

### 💻 전체 실습 코드

```Python
# numpy2.py
# 배열 연산
import numpy as np

x = np.array([[1, 2], [3, 4]], dtype=np.float32)
# x = np.array([[1, 2], [3, 4]])
print(x, ' ',x.dtype)
y = np.arange(5, 9).reshape((2, 2))     # 구조 변경 1차원 -> 2차원
y = y.astype(np.float32)
print(y, ' ',y.dtype)

print()
print(x + y)        # 파이썬 연산자(느림)
print(np.add(x, y)) # 넘파이 함수 (유니버셜 함수)(빠름)

print()
print(x - y)
print(np.subtract(x, y))

print()
print(x * y)
print(np.multiply(x, y))

print()
print(x / y)
print(np.divide(x, y))

print()
print('dot은 numpy 모듈의 함수나 배열 객체의 인스턴스 메소드로 사용이 가능하다')
v = np.array([9, 10])
w = np.array([11, 12])
print(v * w)    # 요소별 곱셈   9 * 11, 10 * 11

# 벡터의 내적(행렬곱)
print(v.dot(w))     # 내적의 결과는 스칼라(크기만 있고 방향은 없음)
print(np.dot(v, w)) # 9 * 11 + 10 * 12
print(np.dot(x, v))

print()
# 배열 계산 함수
print(x)
print(np.mean(x), ' ', np.var(x))
print(np.max(x), ' ', np.min(x))
print(np.argmax(x), ' ', np.argmin(x))  # index 반환
print(np.cumsum(x))     # 누적합
print(np.cumprod(x))    # 누적곱

print()
names1 = np.array(['tom', 'james', 'tom', 'oscar'])
names2 = np.array(['tom', 'page', 'john'])
print(np.unique(names1))
print(np.intersect1d(names1, names2))   # 교집합
print(np.intersect1d(names1, names2, assume_unique=True))       # 교집합(중복 허용)
print(np.union1d(names1, names2))   # 합집합

print('\n전치(Transpose) - 2차원 배열에서 행과 열의 위치를 바꿈')
print(x)
print(x.T)
print(x.transpose())
print(x.swapaxes(0, 1))

print('\nBroadcasting : 크기가 다른 배열 간의 연산 - 작은 배열을 여러 번 반복해 큰 배열과 연산')
x = np.arange(1, 10).reshape(3, 3)
y = np.array([1, 0, 1])
print(x)
print(y)
print(x + y)

np.savetxt("my.txt", x)     # 배열 file i/o     loadtxt()
```

---

### 📝 핵심 원리 요약 (코드 매칭 가이드)

**1️⃣ 유니버셜 함수 (Ufuncs)**

> `print(np.add(x, y))`

- 파이썬 기본 기호(`+`, `-`)를 써도 되지만, **Numpy 전용 함수**(`np.add`, `np.subtract` 등)를 사용하는 것이 대용량 데이터에서 속도가 훨씬 빠릅니다.
    

**2️⃣ 요소별 곱셈(`*`) vs 내적(`np.dot`) ★매우 중요**

> `print(v * w)` vs `print(np.dot(v, w))`

- `*`: 단순히 같은 위치에 있는 숫자끼리 곱해서 배열로 반환합니다.
    
- `np.dot()`: 우리가 선형대수에서 배운 진짜 **내적**! 두 벡터의 크기와 방향을 고려하여 **스칼라(숫자 1개)** 값을 뱉어냅니다.
    

**3️⃣ 통계 및 배열 함수**

> `np.argmax(x)`, `np.unique(names1)`

- `argmax()`, `argmin()`: 가장 큰/작은 값 자체가 아니라, 그 값의 **인덱스(위치 번호)** 를 반환합니다. (분류 모델에서 정답 찾을 때 자주 씀)
    
- `unique()`, `intersect1d()`, `union1d()`: 배열 안의 문자열이나 숫자들을 집합처럼 다뤄 중복을 제거하거나 교/합집합을 구합니다.
    

**4️⃣ 전치 (Transpose)**

> `x.T` 또는 `x.transpose()`

- 2차원 배열의 **행과 열을 뒤집는** 기능입니다. 데이터의 구조를 변환할 때 필수적입니다.
    

**5️⃣ 브로드캐스팅 (Broadcasting)**

> `print(x + y)` (3x3 행렬 + 길이 3짜리 1D 배열)

- 크기가 서로 다른 배열을 연산할 때, 에러를 내지 않고 **Numpy가 알아서 작은 배열을 큰 배열의 크기에 맞게 복사(확장)하여 계산**해 주는 마법 같은 기능입니다.

---
# 🔢 7강: \[실습] Numpy 데이터 조작 및 샘플링 (numpy3.py)

데이터 분석의 전처리 과정에서 가장 많이 쓰이는 **데이터 합치기, 자르기, 조건부 추출, 그리고 무작위 샘플링**을 다룹니다.

## 1. 배열의 확장: 행(Row)과 열(Column) 추가

기존 배열에 새로운 데이터를 이어 붙이는 직관적인 방법입니다.

- **`np.c_` (Column)**: 열을 옆으로 붙임 (가로 확장)
    
- **`np.r_` (Row)**: 행을 아래로 붙임 (세로 확장)
    

---

## 2. 편집 함수: Append, Insert, Delete

배열의 특정 위치에 데이터를 넣거나 빼는 함수입니다.

|**함수**|**기능**|**핵심 파라미터**|
|---|---|---|
|**`np.append`**|배열의 끝에 데이터 추가|`axis=0`(행), `axis=1`(열)|
|**`np.insert`**|원하는 인덱스 위치에 데이터 삽입|`obj`(위치), `values`(값)|
|**`np.delete`**|특정 인덱스의 데이터 삭제|`obj`(삭제할 위치)|

---

## 3. 조건 연산: `np.where`

조건에 따라 값을 선택하거나, 조건에 맞는 데이터의 **위치(Index)**를 찾을 때 사용합니다.

- **값 선택**: `np.where(조건, 참일때_값, 거짓일때_값)`
    
- **위치 찾기**: `np.where(조건)` $\rightarrow$ 인덱스 번호를 반환
    

---

## 4. 배열의 결합과 분할

데이터 덩어리를 하나로 합치거나 여러 개로 쪼개는 기술입니다.

- **`np.concatenate`**: 여러 배열을 하나로 이어 붙임.
    
- **`np.hsplit` (Horizontal)**: 좌우로 분할 (열 기준).
    
- **`np.vsplit` (Vertical)**: 상하로 분할 (행 기준).
    

---

## 5. 표본 추출 (Sampling)

데이터셋에서 무작위로 데이터를 뽑는 과정입니다.

- **복원 추출 (Replace=True)**: 뽑은 것을 다시 넣음 (중복 허용).
    
- **비복원 추출 (Replace=False)**: 한 번 뽑은 것은 다시 뽑지 않음 (중복 불가).
    

---

### 💻 전체 실습 코드 (`numpy3.py`)

```Python
# numpy3.py
# 배열에 행, 열 추가 ...

import numpy as np

aa = np.eye(3)
print(aa)

bb = np.c_[aa, aa[2]]   # 2열과 동일한 열 추가
print(bb)

cc = np.r_[aa, [aa[2]]]     # 2행과 동일한 행 추가
print(cc)

print('--append, insert, delete ---')
a = np.array([1,2,3])
print(a)
b = np.append(a, [4, 5])
b = np.append(a, [4, 5], axis=0)    # 행 기준
print(b)
c = np.insert(a, 0, [6, 7])
print(c)
d = np.delete(a, 1)
print(d)

print()
aa = np.arange(1, 10).reshape(3,3)
print(aa)
print(np.insert(aa, 1, 99))
print(np.insert(aa, 1, 99, axis=0))     # 행기준
print(np.insert(aa, 1, 99, axis=1))     # 열기준

print()
# 조건 연산 where(조건, 참, 거짓)
x = np.array([1,2,3])
y = np.array([4,5,6])
conditionData = np.array([True, False, True])
result = np.where(conditionData, x, y)
print(result)

print()
aa = np.where(x >= 2)
print(aa)   # (array([1, 2]),)  인덱스
print(x[aa]) # 인덱스를 이용한 데이터 추출

print()
# 배열 결합
kbs = np.concatenate([x, y])
print(kbs)
# 배열 분할
mbc, sbs = np.split(kbs, 2)
print(mbc)
print(sbs)

print()
a = np.arange(1, 17).reshape(4, 4)
print(a)
# 배열 좌우로 분할
x1, x2 = np.hsplit(a, 2)
print(x1)
print(x2)
print()
print(np.vsplit(a, 2))

print('\n표본 추출(sampling) - 복원, 비복원')
li = np.array([1,2,3,4,5,6,7])

# 복원
for _ in range(5):
    print(li[np.random.randint(0, len(li) - 1)], end = ' ')

print()
# 비복원
import random
print(random.sample(li.tolist(), 5))  # random.sample()은 대상이 list type

print()
# choice
print(list(np.random.choice(range(1, 46), 6)))
print(list(np.random.choice(range(1, 46), 6, replace=True)))    # 복원
print(list(np.random.choice(range(1, 46), 6, replace=False)))   # 비복원
```

---

> [!tip] **7강 핵심 요약**
> 
> 1. **`np.where`**는 조건에 맞는 값의 **위치(인덱스)**를 찾을 때 가장 강력하다.
>     
> 2. **`hsplit` / `vsplit`**은 거대한 데이터 행렬을 학습용/테스트용으로 쪼갤 때 유용하다.
>     
> 3. 로또 번호 생성기처럼 중복이 없어야 하는 추출은 **`replace=False`** (비복원)를 사용한다.


