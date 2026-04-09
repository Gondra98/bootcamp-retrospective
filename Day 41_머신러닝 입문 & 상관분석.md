# Day 41_머신러닝 입문 & 상관분석

## 📅 2026-04-02

---

# 🤖 머신러닝 첫 수업 — 개요 & 역사

> **태그**: #머신러닝 #딥러닝 #AI역사 #수업노트  
> **학습 순서**: 이 노트 → [[공분산_상관계수_완전정복]] → 회귀 → 분류 → ...

---

## 1. 머신러닝이란?

> **기계 학습(Machine Learning)** 은 분석 및 예측 모델 구축을 자동화하는 AI/데이터 과학의 하위 분야

- 반복적으로 데이터를 학습하는 알고리즘 사용
- **어디에서 볼 것인지 명시적으로 프로그래밍하지 않아도** 숨겨진 통찰력을 찾아냄

### 전통적 프로그래밍 vs 머신러닝

|구분|전통적 프로그래밍|머신러닝 (딥러닝)|
|---|---|---|
|방식|사람이 논리/절차를 직접 작성|대량 데이터로 학습 → 모델이 규칙 발견|
|예시|윈도우 OS (코드 라인 수억 개)|이미지 분류, 손글씨 인식|
|한계|이미지 분류 등 고차원 문제에 사실상 불가능|최적의 W(가중치)를 찾아 일반화된 모델 생성|

---

## 2. 인공신경망 & 딥러닝의 역사

```
1943 ──▶ 1958 ──▶ 1960 ──▶ 암흑기 ──▶ 2006 ──▶ 현재
맥컬럭     로젠블렛   위드로/테드   XOR 미해결   힌튼      딥러닝 전성기
뇌세포     퍼셉트론   아달라인      SVM 강세    DBN 발표
모델링     완성       개발
```

### 연도별 정리

#### 🧠 1943년 — 신경망 모델 등장

- **맥컬럭(McCulloch)과 피츠(Pitts)** 가 처음으로 뇌세포를 수학적으로 모델링
- 인공신경망 이론의 출발점

#### ⚡ 1958년 — 퍼셉트론 (Perceptron)

- **로젠블렛(Rosenblatt)** 이 뇌 모델에 **가중치(weight)** 를 추가
- 퍼셉트론: 순입력함수의 반환값을 **임계값** 기준으로 **1과 -1로 분류**

#### 📐 1960년 — 아달라인 (Adaline)

- **위드로(Widrow)와 테드(Hoff)** 가 신경망 초기 형태인 **아달라인** 개발
- **델타 규칙(Delta Rule)** 사용:
    - 활성화 함수를 통해 실제값과 예측값의 차이가 클 경우 **가중치를 반복 갱신**
    - = **경사 하강법(Gradient Descent)** 학습
    - 싱글 레이어 퍼셉트론에서 연결강도(가중치) 갱신에 사용

#### 🌑 암흑기 — XOR 문제 + 기술적 한계

- XOR 문제 미해결 등 효과적인 학습 모델을 찾지 못함
- **Vanishing Gradient Problem**: 레이어가 깊어질수록 기울기가 0에 가까워져 학습이 안 되는 문제
- **과적합(Overfitting)**: 학습 데이터에는 오차↓, 실제 데이터에는 오차↑
- 높은 시간 복잡도 + 컴퓨터 성능 한계
- → **SVM 등 전통적 모델이 강세**

#### 🌟 2006년 — 딥러닝 부활

- **제프리 힌튼(Geoffrey Hinton, 토론토대)** 이 **심층 신뢰 신경망(Deep Belief Network, DBN)** 논문 발표
- 딥러닝 부활의 신호탄

#### 🚀 현재 — 딥러닝 전성기

|문제|해결책|
|---|---|
|Vanishing Gradient|Sigmoid 대신 **ReLU**, **ELU** 활성화 함수 도입|
|Overfitting|**Dropout Layer** — 학습 중 랜덤하게 뉴런 비활성화|
|성능 한계|**GPU 기반 병렬처리**, 빅데이터|

### 딥러닝 발전의 3대 원동력

1. 🖥️ **GPU 기반 병렬처리** — 컴퓨팅 파워의 급격한 발달
2. 📦 **빅데이터** — 인터넷을 통해 축적된 방대한 데이터
3. 💡 **혁신적 알고리즘** — ReLU, Dropout, DBN 등

---

## 3. 신경망 모델 구조 비교

|항목|FCNN (완전연결 신경망)|LeNet (초기 CNN)|AlexNet (대형 CNN)|
|---|---|---|---|
|입력 처리|이미지를 1D로 펼침 → 공간정보 손실|Conv+Pooling으로 공간정보 유지|깊은 Conv 스택 + 대규모 채널|
|전형 입력|벡터 (예: 784)|32×32×1 (MNIST류)|227×227×3 (ImageNet)|
|대표 구조|FC–FC–softmax|Conv–Pool–Conv–Pool–FC–softmax|Conv×5–MaxPool–FC×3–softmax|
|활성화|tanh/sigmoid → 요즘 ReLU|원형은 tanh + 평균풀링|**ReLU 대중화**|
|파라미터|입력 크기에 비례해 급증|수만 개 (~60K)|수천만 개 (~60M)|
|용도|표형(tabular), 소형 피처|숫자 인식, 소형 비전, 교재|ImageNet급 물체 인식|
|단점|해상도↑ → 파라미터 폭증, 과적합 쉬움|소형 문제에만 적합|계산/메모리 매우 큼|

> 🔗 신경망 구조 시각화: http://alexlenail.me/NN-SVG/index.html  
> 🔗 브라우저에서 직접 실습: https://playground.tensorflow.org

---

## 4. 머신러닝 전체 알고리즘 지도

> 💡 아래 트리에서 **수학적 기초가 먼저 필요한 항목**은 [[공분산_상관계수_완전정복]] 참고

```
MachineLearning
 ├── 📦 데이터 전처리 (Preprocessing)
 │    ├── 결측값 처리 (Missing Data)
 │    ├── 범주형 데이터 (Categorical Data)
 │    ├── 데이터 분할 (Train/Test Split)
 │    └── 피처 스케일링 (Feature Scaling)
 │
 ├── 📈 회귀 (Regression)         ← 연속값 예측
 │    ├── 단순 선형 회귀             ★ OLS 최소제곱법 → 공분산_상관계수_완전정복 §6
 │    ├── 다중 선형 회귀
 │    ├── 다항 회귀
 │    ├── SVR
 │    ├── 결정 트리 회귀
 │    └── 랜덤 포레스트 회귀
 │
 ├── 🔖 분류 (Classification)     ← 카테고리 예측
 │    ├── 로지스틱 회귀              ★ 상관계수로 피처 선택 → 공분산_상관계수_완전정복 §2
 │    ├── K-NN
 │    ├── SVM / Kernel SVM
 │    ├── 나이브 베이즈
 │    ├── 결정 트리
 │    ├── 랜덤 포레스트
 │    └── XGBoost ⭐
 │
 ├── 🔵 군집화 (Clustering)       ← 레이블 없이 그룹화
 │    ├── K-Means
 │    └── 계층적 군집화
 │
 ├── 🛒 연관 규칙 (Association)
 │    └── Apriori
 │
 ├── 🎮 강화학습 (Reinforcement Learning)
 │    ├── UCB (Upper Confidence Bound)
 │    └── Thompson Sampling
 │
 ├── 📝 자연어 처리 (NLP)
 │    ├── 전처리
 │    └── Bag of Words
 │
 ├── 🧠 딥러닝 (Deep Learning)
 │    ├── ANN (인공신경망)
 │    └── CNN (합성곱 신경망)
 │
 └── 📉 차원 축소 (Dimensionality Reduction)
      ├── PCA                       ★ 공분산 행렬 기반 → 공분산_상관계수_완전정복 §1
      ├── LDA
      └── Kernel PCA
```

> **지도학습**: 회귀, 분류 (정답 레이블 있음)  
> **비지도학습**: 군집화, 차원축소 (정답 레이블 없음)  
> **강화학습**: 보상을 최대화하는 행동 학습

---

## 5. 핵심 용어 정리

|용어|설명|
|---|---|
|**퍼셉트론 (Perceptron)**|뇌의 뉴런을 모방한 가장 기본적인 분류 모델|
|**가중치 (Weight, W)**|입력 신호의 중요도를 조절하는 파라미터|
|**델타 규칙 (Delta Rule)**|오차에 비례해 가중치를 갱신하는 규칙 = 경사 하강법|
|**경사 하강법 (Gradient Descent)**|손실 함수를 최소화하는 방향으로 가중치 업데이트|
|**역전파 (Backpropagation)**|출력층에서 입력층 방향으로 오차를 전파하며 가중치 갱신|
|**Vanishing Gradient**|레이어가 깊어질수록 기울기가 0에 가까워져 학습이 안 되는 문제|
|**과적합 (Overfitting)**|학습 데이터에만 지나치게 최적화 → 새로운 데이터에 취약|
|**ReLU**|`f(x) = max(0, x)` — Vanishing Gradient 해결용 활성화 함수|
|**Dropout**|학습 중 랜덤하게 뉴런을 꺼서 과적합 방지|
|**DBN (Deep Belief Network)**|힌튼이 2006년 발표한 딥러닝 핵심 알고리즘|

---

## 6. 수학 기초 → ML 연결 고리

> ML을 제대로 이해하려면 수학 기초가 필수! 각 개념이 어디에 쓰이는지 미리 파악해두자.

|수학 개념|ML에서 쓰이는 곳|공부할 노트|
|---|---|---|
|**공분산**|피처 간 관계 파악, PCA의 공분산 행렬|[[공분산_상관계수_완전정복]] §1|
|**상관계수 r**|피처 선택 (r > 0.3 유의미), EDA 단계|[[공분산_상관계수_완전정복]] §2|
|**최소제곱법 (OLS)**|선형 회귀의 w, b 계산 원리|[[공분산_상관계수_완전정복]] §6|
|**경사 하강법**|신경망 가중치 업데이트 (델타 규칙)|이 노트 §2|
|**선형대수 (행렬)**|데이터 표현, 연산 효율화|-|

### 💡 추천 학습 순서

```
① [[공분산_상관계수_완전정복]]
        │
        ├─ 공분산/상관계수 개념
        ├─ EDA에서 피처 관계 파악
        └─ OLS로 선형회귀 원리 이해
                │
                ▼
② 머신러닝_첫수업_개요와역사 (이 노트)
        │
        ├─ 알고리즘 전체 지도 파악
        └─ 딥러닝 역사 & 용어 정리
                │
                ▼
③ 회귀 → 분류 → 군집화 → 딥러닝 순서로 심화
```

---

## 7. 참고 자료 링크

### 입문 동영상

- 머신러닝 입문: https://www.youtube.com/watch?v=j3za7nv7RfI
- 왜 딥러닝이 화두인가: https://www.youtube.com/watch?v=C2FS9WVm7j4
- 딥러닝 기본개념: https://www.youtube.com/playlist?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
- 역전파 설명: https://www.youtube.com/watch?v=573EZkzfnZ0

### 실습 도구

- TensorFlow Playground (브라우저 실습): https://playground.tensorflow.org
- 신경망 구조 시각화: http://alexlenail.me/NN-SVG/index.html
- scikit-learn 공식: http://scikit-learn.org/stable/index.html

### 읽을거리

- 쉽게 풀어쓴 딥러닝: http://slownews.kr/41461
- ML 알고리즘 정리 (영문): https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/
- 인공신경망 용어 정리: https://brunch.co.kr/@gdhan/6
- 김성훈 교수 딥러닝 자료: https://wikidocs.net/160538

---

# 📊 공분산 & 상관계수 완전정복

> **태그**: #파이썬 #데이터분석 #통계 #머신러닝기초  
> **관련 파일**: `ex1corr.py` `ex2corr.py` `ex3corr.py` `ex4ols.py`  
> **상위 노트**: [[머신러닝_첫수업_개요와역사]] — ML 전체 맥락 확인  
> **이 노트의 위치**: ML 수학 기초 → 회귀/분류/차원축소의 토대

---

## ⚡ 이 노트가 ML에서 쓰이는 곳

|이 노트의 개념|ML 어디에 쓰이나|
|---|---|
|**공분산 행렬**|PCA(주성분 분석) — 차원 축소의 핵심 연산|
|**상관계수 r**|EDA(탐색적 분석)에서 입력 피처 선택 기준 (r > 0.3)|
|**최소제곱법 OLS**|단순/다중 선형 회귀의 가중치 w, 절편 b 계산 원리|
|**산점도·히트맵**|데이터 전처리 단계에서 피처 관계 시각화|

> 👉 ML 전체 알고리즘 지도는 [[머신러닝_첫수업_개요와역사]] §4 참고

---

## 1. 핵심 개념 정리

### 📌 분산 vs 공분산

|구분|설명|변수 수|
|---|---|---|
|**분산 (Variance)**|데이터가 평균에서 얼마나 퍼져 있는지 (거리)|1개|
|**공분산 (Covariance)**|두 변수가 함께 어떤 방향으로 움직이는지 (거리 + 방향)|2개|

### 📌 공분산 행렬 (2차원)

$$\text{Cov}(X, Y) = \begin{bmatrix} \text{Var}(X) & \text{Cov}(X,Y) \ \text{Cov}(Y,X) & \text{Var}(Y) \end{bmatrix}$$

- **대각선**: 각 변수의 분산 (Var)
- **비대각선**: 두 변수 간의 공분산 (Cov)
    - `Cov > 0` → 우상향 (같이 증가)
    - `Cov < 0` → 우하향 (반대 방향)
    - `Cov = 0` → 선형관계 없음

---

## 2. 공분산의 한계 → 상관계수의 등장

> ⚠️ **문제**: 두 데이터가 같은 패턴을 가져도, **단위(스케일)에 따라 공분산 크기가 달라짐**  
> → 공분산만으로는 관계의 절대적 강도를 판단하기 어렵다!

**해결책**: 공분산을 **표준화**하여 `[-1, 1]` 범위로 만든 것이 **상관계수 (r)**

$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

### 📌 ML에서 상관계수 해석 기준 (경험적)

|범위|의미|
|---|---|
|`r > 0.3`|✅ 양적(positive) 상관 관계|
|`r < -0.3`|✅ 음적(negative) 상관 관계|
|`-0.3 ≤ r ≤ 0.3`|❌ 의미 없는 상관 관계|

> ⚠️ **주의**: 상관계수는 **선형 관계**만 측정!  
> 비선형 데이터 (예: y = x²)는 공분산, 상관계수 모두 0이 나올 수 있음 → 의미 없음

---

## 3. ex1corr.py — NumPy로 공분산·상관계수 계산

### 공분산 행렬 `np.cov()`

```python
import numpy as np

# 우상향
print(np.cov(np.arange(1,6), np.arange(2,7)))
```

> **▶ 출력결과**
> 
> ```
> [[2.5  2.5]
>  [2.5  2.5]]
> ```
> 
> 💡 비대각선 2.5 > 0 → 우상향 확인

```python
# 직선 (y 변화 없음)
print(np.cov(np.arange(1,6), np.array([3,3,3,3,3])))
```

> **▶ 출력결과**
> 
> ```
> [[2.5  0. ]
>  [0.   0. ]]
> ```
> 
> 💡 공분산 0 → y가 고정이므로 방향 없음

```python
# 우하향
print(np.cov(np.arange(1,6), np.arange(6,1,-1)))
```

> **▶ 출력결과**
> 
> ```
> [[ 2.5  -2.5]
>  [-2.5   2.5]]
> ```
> 
> 💡 비대각선 -2.5 < 0 → 우하향 확인

### 상관계수 `np.corrcoef()` — 피어슨 상관행렬

```python
x = [8,3,6,6,9,4,3,9,3,4]
y = [6,2,4,6,9,5,1,8,4,5]

print(np.mean(x), np.var(x))   # x 평균, 분산
print(np.mean(y), np.var(y))   # y 평균, 분산
```

> **▶ 출력결과**
> 
> ```
> 5.5  5.45      ← x: 평균 5.5, 분산 5.45
> 5.0  5.4       ← y: 평균 5.0, 분산 5.4
> ```

```python
print(np.corrcoef(x, y))
print(np.corrcoef(x, y)[0,1])
```

> **▶ 출력결과**
> 
> ```
> [[1.         0.86636865]
>  [0.86636865 1.        ]]
> 
> 0.8664     ← r > 0.3 → 강한 양적 상관 ✅
> ```

### 스케일이 달라도 상관계수는 동일하다

```python
x2 = np.array(x) * 10

print('x,y의 공분산 :', np.cov(x,y)[0,1])
print('x2,y2의 공분산 :', np.cov(x2,y)[0,1])

print(np.corrcoef(x,y)[0,1])
print(np.corrcoef(x2,y)[0,1])
```

> **▶ 출력결과**
> 
> ```
> x,y의 공분산  :  5.2222    ← 단위 영향 받음
> x2,y의 공분산 : 52.2222    ← 10배 커짐! (스케일 영향)
> 
> 0.8664    ← x,y 상관계수
> 0.8664    ← x2,y 상관계수 → 완전히 동일! (표준화 덕분)
> ```

### ⚠️ 비선형 데이터 — 상관계수 의미 없음

```python
m = np.array([-3,-2,-1,0,1,2,3])
n = np.array([9,4,1,0,1,4,9])   # y = x^2

print(np.cov(m,n)[0,1])
print(np.corrcoef(m,n)[0,1])
```

> **▶ 출력결과**
> 
> ```
> 0.0    ← 공분산 = 0
> 0.0    ← 상관계수 = 0
> ```
> 
> ⚠️ 수치는 0이지만 실제로는 포물선(y=x²) 관계 존재!  
> → **반드시 산점도로 시각화해서 확인해야 함**

---

## 4. ex2corr.py — Pandas & Seaborn으로 상관관계 분석

### 데이터 불러오기

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from pandas.plotting import scatter_matrix

uri = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv"
data = pd.read_csv(uri)
# 컬럼: 친밀도(1~5), 적절성(1~5), 만족도(1~5) — 264행
```

### 공분산 & 상관계수 행렬

```python
print(data.cov())
```

> **▶ 출력결과 (공분산 행렬)**
> 
> ```
>        친밀도      적절성      만족도
> 친밀도  0.941569  0.416422  0.375663
> 적절성  0.416422  0.739011  0.546333
> 만족도  0.375663  0.546333  0.686816
> ```

```python
print(data.corr())
```

> **▶ 출력결과 (피어슨 상관계수 행렬)**
> 
> ```
>        친밀도    적절성    만족도
> 친밀도  1.000000  0.499209  0.467145
> 적절성  0.499209  1.000000  0.766853
> 만족도  0.467145  0.766853  1.000000
> ```

```python
# 방법 선택
data.corr(method='pearson')   # 연속형 변수, 정규분포 → 모수검정
data.corr(method='spearman')  # 서열척도, 비정규분포 → 비모수검정
data.corr(method='kendall')   # 서열척도, 비정규분포
```

### 특정 컬럼 기준 상관계수 정렬

```python
co_re = data.corr()
print(co_re['만족도'].sort_values(ascending=False))
```

> **▶ 출력결과**
> 
> ```
> 만족도    1.000000
> 적절성    0.766853  ← 만족도와 가장 강한 양적 상관 ✅
> 친밀도    0.467145  ← 만족도와 양적 상관 ✅
> ```
> 
> 💡 **해석**: 적절성이 올라가면 만족도가 가장 크게 올라감 (r=0.767)

### 시각화 3종 세트

#### 1) 산점도 (두 변수 비교)

```python
data.plot(kind='scatter', x='만족도', y='적절성')
plt.show()
```

#### 2) Scatter Matrix (모든 변수 쌍 한 번에)

```python
attr = ['친밀도', '적절성', '만족도']
scatter_matrix(data[attr], figsize=(10,6))
plt.show()
```

#### 3) Heatmap (상관계수를 색으로 표현)

```python
sns.heatmap(data.corr(), annot=True)
plt.show()
```

#### 4) 고급 Heatmap (하삼각형만 + 수치 표시)

```python
corr = data.corr()

# 상삼각형 마스킹
mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

vmax = np.abs(corr.values[~mask]).max()
fig, ax = plt.subplots()

sns.heatmap(corr, mask=mask, vmin=-vmax, vmax=vmax,
            square=True, linecolor="lightgray", linewidths=1, ax=ax)

for i in range(len(corr)):
    ax.text(i+0.5, len(corr)-(i+0.5), corr.columns[i],
            ha="center", va="center", rotation=45)
    for j in range(i+1, len(corr)):
        s = "{:.3f}".format(corr.values[i,j])
        ax.text(j+0.5, len(corr)-(i+0.5), s, ha="center", va="center")
ax.axis("off")
plt.show()
```

---

## 5. ex3corr.py — 실전: 관광지 방문 상관관계 분석

### 목표

> 서울 5대 궁궐의 **외국인 입장객 수**와  
> 중국인 / 일본인 / 미국인 방문자 수 간의 **상관관계** 분석

### 데이터 구조

|파일|컬럼|기간|
|---|---|---|
|`서울특별시_관광지입장정보_2011_2016.json`|yyyymm, resNm(관광지명), ForNum(외국인입장객수)|2011~2016|
|`중국인방문객.json`|yyyymm, visit_cnt|72개월|
|`일본인방문객.json`|yyyymm, visit_cnt|72개월|
|`미국인방문객.json`|yyyymm, visit_cnt|72개월|

### 전체 코드 흐름

```python
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

def setScatterGraph(tour_table, all_table, tourpoint):
    tour = tour_table[tour_table['resNm'] == tourpoint]
    merge_table = pd.merge(tour, all_table, left_index=True, right_index=True)
    # 결과: resNm, ForNum, china, japan, usa 5개 컬럼

    fig = plt.figure()
    fig.suptitle(tourpoint + ' 상관관계분석')

    plt.subplot(1,3,1)
    plt.xlabel('중국인 방문수')
    plt.ylabel('외국인 입장객 수')
    r1 = merge_table['china'].corr(merge_table['ForNum'])
    plt.title('r={:.5f}'.format(r1))
    plt.scatter(merge_table['china'], merge_table['ForNum'], alpha=0.7, c='red')

    plt.subplot(1,3,2)
    plt.xlabel('일본인 방문수')
    r2 = merge_table['japan'].corr(merge_table['ForNum'])
    plt.title('r={:.5f}'.format(r2))
    plt.scatter(merge_table['japan'], merge_table['ForNum'], alpha=0.7, c='green')

    plt.subplot(1,3,3)
    plt.xlabel('미국인 방문수')
    r3 = merge_table['usa'].corr(merge_table['ForNum'])
    plt.title('r={:.5f}'.format(r3))
    plt.scatter(merge_table['usa'], merge_table['ForNum'], alpha=0.7, c='blue')

    plt.tight_layout()
    plt.show()
    plt.close()

    return [tourpoint, r1, r2, r3]


def processFunc():
    jsonTP = json.loads(open("ex3_data/서울특별시_관광지입장정보_2011_2016.json", 'r', encoding='utf-8').read())
    tour_table = pd.DataFrame(jsonTP, columns=('yyyymm','resNm','ForNum'))
    tour_table = tour_table.set_index('yyyymm')
    resNm = tour_table.resNm.unique()
    # ['창덕궁', '운현궁', '경복궁', '창경궁', '종묘']

    def load_visitor(filepath, col_name):
        data = json.loads(open(filepath, 'r', encoding='utf-8').read())
        df = pd.DataFrame(data, columns=('yyyymm','visit_cnt'))
        df = df.rename(columns={'visit_cnt': col_name})
        return df.set_index('yyyymm')

    china_table = load_visitor("ex3_data/중국인방문객.json", 'china')
    japan_table = load_visitor("ex3_data/일본인방문객.json", 'japan')
    usa_table   = load_visitor("ex3_data/미국인방문객.json", 'usa')

    all_table = pd.merge(china_table, japan_table, left_index=True, right_index=True)
    all_table = pd.merge(all_table, usa_table, left_index=True, right_index=True)
    # china, japan, usa 컬럼 / 72행 (201101~201612)

    r_list = []
    for tourpoint in resNm[:5]:
        r_list.append(setScatterGraph(tour_table, all_table, tourpoint))

    r_df = pd.DataFrame(r_list, columns=['고궁명','중국','일본','미국'])
    r_df = r_df.set_index('고궁명')

    r_df.plot(kind='bar', rot=50)
    plt.show()


if __name__ == "__main__":
    processFunc()
```

> **▶ merge_table.info() 출력 (예: 창덕궁)**
> 
> ```
> Index: 69 entries, 201101 to 201609
> Data columns (total 5 columns):
>  #   Column  Non-Null Count  Dtype
>  0   resNm   69 non-null     object
>  1   ForNum  69 non-null     int64
>  2   china   69 non-null     int64
>  3   japan   69 non-null     int64
>  4   usa     69 non-null     int64
> ```
> 
> 💡 관광지 데이터(69개월) ∩ 방문객 데이터(72개월) → 교집합 69개월 병합

### 핵심 포인트

> - `pd.merge(..., left_index=True, right_index=True)` → **인덱스(yyyymm) 기준 병합**
> - `.corr()` → pandas Series 간 상관계수 계산
> - 관광지별 결과를 리스트에 모아 → DataFrame 변환 → 막대그래프 비교

---

## 6. ex4ols.py — 최소제곱법 (OLS) 으로 선형 회귀

### 개념

> 실제 데이터를 **선형대수**로 직선(회귀선)에 가장 잘 맞추는 w(기울기), b(절편) 구하기

$$y = wx + b \quad \Rightarrow \quad \hat{y} = 0.96x - 0.99$$

### 코드

```python
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.5, 2.1])

# 행렬 A 구성 [x, 1] 형태
A = np.vstack([x, np.ones(len(x))]).T
print(A)
```

> **▶ 출력결과**
> 
> ```
> [[0.  1.]
>  [1.  1.]
>  [2.  1.]
>  [3.  1.]]
> ```
> 
> 💡 첫 번째 열 = x값, 두 번째 열 = 1(절편을 위한 상수)

```python
w, b = lin.lstsq(A, y, rcond=None)[0]
print(w, b)
```

> **▶ 출력결과**
> 
> ```
> w = 0.96    ← 기울기 (x 1 증가 시 y 0.96 증가)
> b = -0.99   ← y절편 (x=0일 때 y값)
> → 회귀식: ŷ = 0.96x - 0.99
> ```

```python
for i in range(4):
    print(f'x={i} 실제값: {y[i]}  예측값: {w*i+b:.4f}')
```

> **▶ 출력결과**
> 
> ```
> x=0  실제값: -1.0   예측값: -0.9900   오차: -0.01
> x=1  실제값:  0.2   예측값: -0.0300   오차: +0.23
> x=2  실제값:  0.5   예측값:  0.9300   오차: -0.43
> x=3  실제값:  2.1   예측값:  1.8900   오차: +0.21
> ```
> 
> 💡 오차의 제곱합 = 0.282 → 이게 수학적으로 가능한 최솟값!

```python
print('x=1.23456 예측값:', w * 1.23456 + b)
```

> **▶ 출력결과**
> 
> ```
> x=1.23456 예측값: 0.1952
> ```
> 
> 💡 학습 데이터에 없는 x값도 회귀식으로 예측 가능!

### 왜 최소제곱법인가?

> 모든 데이터 포인트와 직선 사이의 **오차의 제곱합이 최소**가 되는 직선을 찾음  
> = 편미분을 통해 해석적으로 계산 가능 (`np.linalg.lstsq`)

---

## 7. 핵심 함수 모음표

|함수|라이브러리|용도|
|---|---|---|
|`np.cov(x, y)`|numpy|공분산 행렬|
|`np.corrcoef(x, y)`|numpy|피어슨 상관계수 행렬|
|`df.cov()`|pandas|DataFrame 공분산 행렬|
|`df.corr()`|pandas|DataFrame 상관계수 행렬|
|`series.corr(series2)`|pandas|두 Series 간 상관계수|
|`pd.merge(..., left_index=True, right_index=True)`|pandas|인덱스 기준 병합|
|`scatter_matrix(df)`|pandas.plotting|모든 변수 쌍 산점도|
|`sns.heatmap(corr, annot=True)`|seaborn|상관계수 히트맵|
|`lin.lstsq(A, y)`|numpy.linalg|최소제곱법 선형회귀|

---

## 8. 자주 헷갈리는 포인트 ✅

1. **공분산 ≠ 상관계수**: 공분산은 단위 영향을 받지만, 상관계수는 표준화되어 있음
2. **비선형 데이터**: 상관계수가 0이어도 관계가 없는 것이 아닐 수 있음 (반드시 산점도 확인!)
3. **`corr()` 기본값**: pandas의 `.corr()`은 기본적으로 **pearson** 방법
4. **`np.corrcoef()` 반환값**: 행렬 반환 → 실제 계수는 `[0,1]` 또는 `[1,0]`으로 추출
5. **`merge` 인덱스 병합**: `left_index=True, right_index=True` 옵션으로 날짜(yyyymm) 기준 병합

---

## 9. 분석 흐름 요약

```
데이터 로딩 (JSON/CSV)
    ↓
DataFrame 변환 + 인덱스 설정
    ↓
필요한 컬럼 필터링 / merge
    ↓
상관계수 계산 (.corr())
    ↓
시각화 (scatter / heatmap / bar)
    ↓
해석: r > 0.3? r < -0.3?
```


