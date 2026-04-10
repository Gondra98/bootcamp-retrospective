# Day 47_랜덤포레스트_XGBoost_LightGBM

## 📅 2026-04-10

---
# 📄 ex28rfiris.py — 랜덤포레스트 실습 (Iris 다항분류)

#머신러닝 #RandomForest #Iris #다항분류 #실습 #sklearn #Python

---

## 📌 이번 실습 포인트

- Iris 데이터셋으로 **3개 클래스** 분류
- `RandomForestClassifier` 적용
- 혼동행렬 3가지 방법으로 정확도 확인
- `joblib` 으로 모델 저장 & 불러오기
- `predict_proba` 로 softmax 확률값 확인
- 결정경계 시각화 함수 (`plot_decision_regionFunc`)

---

## Step 0 — Import

```python
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler    # 표준화
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
```

> `LogisticRegression` — 다중 클래스(label)를 지원하도록 일반화됨  
> 이를 **Softmax Regression** 또는 **Multinomial Logistic Regression** 이라고 부름

---

## Step 1 — 데이터 로드 & 확인

```python
iris = datasets.load_iris()
print(iris.keys())
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']

print(iris.target)
# [0 0 0 ... 1 1 1 ... 2 2 2]  ← 클래스 0,1,2 각 50개씩

print(iris.data[:3])
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]]

# 꽃잎 길이(col2)와 꽃잎 너비(col3)의 상관계수 확인
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])[0, 1])   # 0.9628654314027961
```

> `np.corrcoef` — 두 변수 간 상관계수 계산 (-1 ~ 1)  
> **0.963** → 꽃잎 길이와 너비는 매우 강한 양의 상관관계

---

## Step 2 — feature / label 선택

```python
# 꽃잎 길이(index 2), 꽃잎 너비(index 3) 두 특성만 사용
x = iris.data[:, [2, 3]]
y = iris.target

print(x.shape, ' ', y.shape)    # (150, 2)   (150,)
print(x[:3], y[:3], set(map(int, y)))
# [[1.4 0.2]
#  [1.4 0.2]
#  [1.3 0.2]] [0 0 0] {0, 1, 2}
```

**Iris 클래스 구성**

|클래스|이름|샘플 수|
|---|---|---|
|0|Setosa|50개|
|1|Versicolor|50개|
|2|Virginica|50개|

> 전체 4개 특성 중 **꽃잎 길이·너비 2개만** 사용  
> 두 특성 간 상관계수 0.963 → 거의 같은 정보를 담고 있음

---

## Step 3 — Train/Test 분할

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (105, 2) (45, 2) (105,) (45,)
```

> 150개 → train 105개 (70%) / test 45개 (30%)

---

## Step 4 — (참고) 표준화 Scaling

```python
"""
# 표준화 : 데이터 크기를 맞춰 최적화 안정성·수렴속도 향상, 과적합 방지 효과
sc = StandardScaler()
sc.fit(x_train)         # train 기준으로 fit (평균·표준편차 계산)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)   # ← test는 transform만 (fit_transform 쓰면 안됨)

# 스케일링 결과 원복
ori_x_train = sc.inverse_transform(x_train)
"""
```

> ⚠️ `fit`은 반드시 **train 데이터에만** 수행  
> test에 `fit_transform` 쓰면 → 데이터 누수(Data Leakage) 발생  
> **Iris 데이터는 특성 간 크기 차이가 거의 없어 표준화 효과 미미** → 주석 처리

**표준화가 필요한 경우 vs 불필요한 경우**

|알고리즘|표준화 필요 여부|
|---|---|
|로지스틱 회귀, SVM, KNN|✅ 필요 (거리·크기 기반)|
|의사결정나무, 랜덤포레스트|❌ 불필요 (분기 기준 기반)|

---

## Step 5 — 모델 생성 & 학습

```python
from sklearn.ensemble import RandomForestClassifier

# n_estimators=500 : 결정트리 500개 생성
model = RandomForestClassifier(criterion='gini', n_estimators=500, random_state=0)
model.fit(x_train, y_train)
```

---

## Step 6 — 예측

```python
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  2 1 1 2 0 2 0 0]
print('실제값 : ', y_test)
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  1 1 1 2 0 2 0 0]

print(f'총 갯수:{len(y_test)}, 오류수:{(y_test != y_pred).sum()}')
# 총 갯수:45, 오류수:1
```

---

## Step 7 — 정확도 확인 3가지 방법

### 방법 1 — accuracy_score

```python
print(f'{accuracy_score(y_test, y_pred)}')  # 0.9555555555555556
```

### 방법 2 — 혼동행렬 (Confusion Matrix)

```python
con_mat = pd.crosstab(y_test, y_pred, rownames=['실제값'], colnames=['예측값'])
print(con_mat)
# 예측값   0   1   2
# 실제값
# 0    16   0   0
# 1     0  17   1  ← class1 → class2 로 1개 오분류
# 2     0   1  10  ← class2 → class1 로 1개 오분류

print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))
# 0.9555555555555556
```

**혼동행렬 구조**

```
              예측 0   예측 1   예측 2
실제 0 (Setosa)     [ 16      0      0  ]  ← 완전 정답
실제 1 (Versicolor) [  0     17      1  ]  ← 1개 오분류
실제 2 (Virginica)  [  0      1     10  ]  ← 1개 오분류
```

> Versicolor ↔ Virginica 경계에서 총 2개 오분류  
> Setosa는 완전 분리 → 꽃잎 크기가 뚜렷하게 다르기 때문

### 방법 3 — model.score

```python
print('test score : ', model.score(x_test, y_test))     # 0.9555555555555556
print('train score : ', model.score(x_train, y_train))  # 0.9904761904761905
```

> test score와 train score 차이가 크면 **과적합(Overfitting) 의심**  
> 여기선 약 3.5% 차이 → 허용 범위

---

## Step 8 — 모델 저장 & 불러오기

```python
import joblib   # pickle보다 빠르고 대용량 지원

# 모델 저장
joblib.dump(model, 'logimodel.pkl')   # .sav, .model 등 확장자 무관

# 기존 모델 삭제 후 불러오기
del model
read_model = joblib.load('logimodel.pkl')
```

> 학습이 오래 걸리는 모델을 매번 다시 학습하지 않아도 됨  
> 이후 코드에서는 `read_model` 사용

---

## Step 9 — 새로운 데이터 예측

```python
new_data = np.array([[5.5, 2.2], [0.6, 0.3], [1.1, 0.5]])

# 주의 : 표준화된 자료로 모델 생성 시
# sc.fit(new_data)
# new_data = sc.transform(new_data)

new_pred = read_model.predict(new_data)
print('예측 결과 : ', new_pred)   # [2 0 0]

# softmax 확률값 확인
print(read_model.predict_proba(new_data))
# [[0. 0. 1.]   → Virginica  100%
#  [1. 0. 0.]   → Setosa     100%
#  [1. 0. 0.]]  → Setosa     100%
```

> `predict` → 확률이 가장 높은 클래스 1개 출력  
> `predict_proba` → 각 클래스별 확률 출력 (합계 = 1.0)  
> 랜덤포레스트에서는 500개 트리의 투표 비율이 확률값이 됨

---

## Step 10 — 결정경계 시각화

```python
import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib.colors import ListedColormap

def plot_decision_regionFunc(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')      # 마커 모양 5개
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정경계 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    # meshgrid : x1, x2 범위의 격자점 생성
    xx, yy = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    # 격자점 전체에 대해 예측 → 결정경계 색칠
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # 클래스별 산점도
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
                    color=cmap(idx), marker=markers[idx], label=cl)

    # test 데이터 표시
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

# train + test 합쳐서 전체 시각화
x_combined_std = np.vstack((x_train, x_test))   # 수직 결합
y_combined = np.hstack((y_train, y_test))         # 수평 결합

plot_decision_regionFunc(
    X=x_combined_std,
    y=y_combined,
    classifier=read_model,
    test_idx=range(105, 150),   # test 데이터 인덱스 (105~149)
    title='scikit-learn제공'
)
```

**결정경계 시각화 결과**

![[ex28rfiris.png]]

**시각화 함수 동작 원리**

```
① meshgrid로 평면 전체 격자점 생성
② 모든 격자점에 predict() 적용
③ 예측 클래스에 따라 배경 색칠 (contourf)
④ 실제 데이터 포인트를 산점도로 오버레이
⑤ test 데이터는 별도 마커(o)로 표시
```

> 의사결정나무·랜덤포레스트 특유의 **수직·수평 계단 경계선** 확인 가능  
> `np.vstack` — 배열을 수직(행 방향)으로 결합  
> `np.hstack` — 배열을 수평(열 방향)으로 결합

---

## 💡 핵심 개념 정리

### predict vs predict_proba

| |predict|predict_proba|
|---|---|---|
|출력|클래스 레이블 1개|각 클래스별 확률 배열|
|예시|`[2]`|`[0. 0. 1.]`|
|활용|최종 예측|확신 정도 확인, ROC 곡선|

### joblib vs pickle

| |joblib|pickle|
|---|---|---|
|속도|빠름|느림|
|대용량|지원|제한적|
|사용법|`joblib.dump / load`|`pickle.dump / load`|

> 머신러닝 모델 저장에는 **joblib 권장**

---
# 📄 ex29rf_adult.py — 랜덤포레스트 실습 (Adult 소득 예측)

#머신러닝 #RandomForest #Pipeline #ColumnTransformer #OneHotEncoding #실습 #sklearn #Python

---

## 📌 이번 실습 포인트

- Adult 데이터셋 — 연봉 50K 초과 여부 **이진 분류**
- `Pipeline` — 전처리 + 모델을 하나로 묶어서 관리
- `ColumnTransformer` — 숫자형 / 범주형 컬럼에 **다른 전처리** 적용
- `SimpleImputer` — 결측치 자동 처리
- `OneHotEncoder` — 범주형 데이터 인코딩
- `GridSearchCV` — 최적 하이퍼파라미터 탐색

---

## Step 0 — Import

```python
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline           # 전처리 + 모델을 하나로 묶어서 실행
from sklearn.compose import ColumnTransformer   # 컬럼별 전처리를 다르게 적용
from sklearn.impute import SimpleImputer        # 결측치 처리
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
```

---

## Step 1 — 데이터 로드 & 확인

```python
# fetch_openml : sklearn에서 제공하는 외부 데이터셋 로더
# as_frame=True : pandas DataFrame 형태로 반환
adult = fetch_openml(name='adult', version=2, as_frame=True)
print(type(adult))  # <class 'sklearn.utils._bunch.Bunch'>

df = adult.frame    # DataFrame 추출
pd.set_option('display.max_columns', None)  # 전체 컬럼 출력
print(df.head(5))
print(df.shape)     # (48842, 15)
print(df.info())
# age, fnlwgt 등 int64 6개
# workclass, education 등 category 9개
# 결측치 : workclass(2799), occupation(2809), native-country(857)
```

**데이터 구성**

|컬럼|타입|설명|
|---|---|---|
|age|int64|나이|
|workclass|category|고용 형태|
|fnlwgt|int64|인구 가중치|
|education|category|학력|
|education-num|int64|학력 수치|
|marital-status|category|혼인 상태|
|occupation|category|직업|
|relationship|category|가족 관계|
|race|category|인종|
|sex|category|성별|
|capital-gain|int64|자본 이득|
|capital-loss|int64|자본 손실|
|hours-per-week|int64|주당 근무 시간|
|native-country|category|출신 국가|
|**class**|category|**연봉 (target)**|

---

## Step 2 — target 인코딩

```python
# class 컬럼 : '>50K' 또는 '<=50K' 문자열 → 1 또는 0으로 변환
# lambda : 익명 함수 — x에 '>50K'가 포함되면 1, 아니면 0
df['class'] = df['class'].apply(lambda x: 1 if '>50K' in x else 0)
print(set(df['class']))  # {0, 1}
```

> `'>50K' in x` — `'>50K.'` 처럼 점이 붙은 경우도 포함되도록 `in` 연산자 사용

---

## Step 3 — feature / label 분리

```python
x = df.drop('class', axis=1)   # feature (독립변수) — class 컬럼 제외
y = df['class']                 # label (종속변수) — 0: ≤50K, 1: >50K
```

---

## Step 4 — 컬럼 타입별 분리

```python
# 숫자형 컬럼 : int64, float64
num_cols = x.select_dtypes(include=['int64', 'float64']).columns
# → ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# 범주형 컬럼 : category, object
cat_cols = x.select_dtypes(include=['category', 'object']).columns
# → ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
```

> 숫자형과 범주형은 전처리 방식이 다르기 때문에 먼저 분리

---

## Step 5 — 전처리 파이프라인 구성

### 숫자형 파이프라인

```python
num_pipeline = Pipeline([
    # 결측치 → 중앙값(median)으로 채우기
    # mean(평균)은 이상치에 민감하므로 median 권장
    ('imputer', SimpleImputer(strategy='median')),

    # 표준화 : 평균 0, 표준편차 1로 변환
    # 랜덤포레스트는 표준화 불필요하지만 파이프라인 일관성을 위해 포함
    ('scaler', StandardScaler())
])
```

### 범주형 파이프라인

```python
cat_pipeline = Pipeline([
    # 결측치 → 최빈값(most_frequent)으로 채우기
    ('imputer', SimpleImputer(strategy='most_frequent')),

    # One-Hot 인코딩 : 범주형 → 0/1 벡터로 변환
    # handle_unknown='ignore' : 학습 시 없던 범주가 test에 나와도 에러 없이 무시
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

**SimpleImputer 전략 비교**

|strategy|설명|사용 시점|
|---|---|---|
|`mean`|평균으로 채움|숫자형, 이상치 없을 때|
|`median`|중앙값으로 채움|숫자형, 이상치 있을 때|
|`most_frequent`|최빈값으로 채움|범주형|
|`constant`|지정값으로 채움|특정 값으로 채울 때|

---

## 💡 One-Hot Encoding 개념

범주형 데이터를 머신러닝 모델이 이해할 수 있도록 **0과 1로만 이루어진 벡터**로 변환하는 방법

**왜 필요한가 — Label Encoding의 문제**

```
Label Encoding : workclass → [0, 1, 2, 3, ...]
→ 모델이 "Private(0) < Self-emp(1) < Gov(2)" 처럼
  크기 순서가 있다고 오해할 수 있음
→ 실제로는 순서 없는 명목형 데이터
```

**One-Hot Encoding 방식**

```
workclass 컬럼 (4개 범주)

         Private  Self-emp  Gov  Never-worked
Private → [  1       0      0       0      ]
Self-emp→ [  0       1      0       0      ]
Gov     → [  0       0      1       0      ]
Never   → [  0       0      0       1      ]
```

> 해당 범주 위치만 1, 나머지는 전부 0 → 크기 관계 없음

**장단점**

| |내용|
|---|---|
|장점|범주 간 순서/크기 오해 없음, 모델이 각 범주를 독립적으로 학습|
|단점|범주 종류가 많으면(High Cardinality) 컬럼 수 폭발 → 메모리·속도 저하|

**Label Encoding vs One-Hot Encoding**

| |Label Encoding|One-Hot Encoding|
|---|---|---|
|방식|0, 1, 2... 숫자로 변환|각 범주를 별도 컬럼으로|
|적합|트리 계열 모델|선형 모델, 거리 기반 모델|
|순서 오해|있음|없음|
|컬럼 수|변화 없음|범주 수만큼 증가|

---

## Step 6 — ColumnTransformer로 전처리 결합

```python
# ColumnTransformer : 컬럼별로 다른 전처리를 한 번에 적용
# ('이름', 파이프라인, 적용할 컬럼) 형식
preprocess = ColumnTransformer([
    ('num', num_pipeline, num_cols),  # 숫자형 컬럼 → 결측치 처리 + 표준화
    ('cat', cat_pipeline, cat_cols)   # 범주형 컬럼 → 결측치 처리 + One-Hot 인코딩
])
```

**ColumnTransformer 동작 구조**

```
원본 데이터 (48842 × 14)
         ↓
┌─────────────────────────────────┐
│        ColumnTransformer        │
│  ┌──────────────────────────┐   │
│  │ num_cols (숫자형 6개)     │   │
│  │  → SimpleImputer(median) │   │
│  │  → StandardScaler        │   │
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │ cat_cols (범주형 8개)     │   │
│  │  → SimpleImputer(mode)   │   │
│  │  → OneHotEncoder         │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
         ↓
   전처리 완료 데이터 (수치형으로 통일)
```

---

## Step 7 — 전체 파이프라인 구성

```python
# 전처리 + 모델을 하나의 파이프라인으로 연결
# fit() 한 번으로 전처리 + 학습이 순서대로 자동 실행됨
pipeline = Pipeline([
    ('prep', preprocess),                               # 1단계: 전처리
    ('model', RandomForestClassifier(random_state=12))  # 2단계: 모델 학습
])
```

**Pipeline을 쓰는 이유**

```python
# ❌ 파이프라인 없이 쓰면
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
model.fit(x_train, y_train)
x_test = sc.transform(x_test)    # ← test는 transform만, 실수하기 쉬움
model.predict(x_test)

# ✅ 파이프라인으로 묶으면
pipeline.fit(train_x, train_y)   # 전처리 + 학습 한번에
pipeline.predict(test_x)         # 전처리 + 예측 한번에 → 실수 방지
```

---

## Step 8 — Train/Test 분할

```python
# stratify=y : 클래스 비율 유지 (불균형 데이터 대응)
# adult 데이터 : ≤50K 약 75%, >50K 약 25% → 불균형
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.3, random_state=12, stratify=y
)
```

---

## Step 9 — 하이퍼파라미터 튜닝 (GridSearchCV)

```python
# Pipeline 내부 파라미터 지정 형식 : '스텝명__파라미터명' (언더바 두 개)
param_grid = {
    'model__n_estimators': [100, 200],          # 트리 개수
    'model__max_depth': [5, 10, None],           # 트리 깊이 (None = 제한 없음)
    'model__class_weight': [None, 'balanced']    # 클래스 불균형 보정
}
# 총 조합 수 : 2 × 3 × 2 = 12가지

# StratifiedKFold : 각 fold마다 클래스 비율 유지
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

grid = GridSearchCV(
    pipeline,           # 전체 파이프라인 사용
    param_grid,         # 탐색할 파라미터 조합
    cv=cv,              # 교차검증 방식
    scoring='roc_auc',  # 평가 기준 (이진분류 → roc_auc 권장)
    n_jobs=-1           # 모든 CPU 코어 병렬 사용 → 속도 향상
)

grid.fit(train_x, train_y)  # 전처리 + 파라미터 탐색 + 학습 한번에 수행

print('최적의 파라미터 : ', grid.best_params_)
# {'model__class_weight': None, 'model__max_depth': 10, 'model__n_estimators': 200}
print('최적의 모델 : ', grid.best_estimator_)
```

**GridSearchCV 동작 구조**

```
param_grid 12가지 조합 × cv=5
→ 총 60번 학습·검증 수행
→ roc_auc 가장 높은 조합 선택
→ refit=True(기본값) → 최적 파라미터로 자동 재학습
```

---

## Step 10 — 예측 & 평가

```python
# grid.predict : 최적 모델로 예측 (별도 fit 불필요)
pred = grid.predict(test_x)

# predict_proba : 각 클래스별 확률값
# [:, 1] → 클래스 1(>50K)에 대한 확률만 추출
proba = grid.predict_proba(test_x)[:, 1]

print('정확도 : ', accuracy_score(test_y, pred))        # 0.8574
print('roc_auc 점수 : ', roc_auc_score(test_y, proba))  # 0.9106
print('classification_report : \n', classification_report(test_y, pred))
#               precision    recall  f1-score   support
#            0       0.87      0.95      0.91     11147
#            1       0.79      0.55      0.65      3506
#     accuracy                           0.86     14653
#    macro avg       0.83      0.75      0.78     14653
# weighted avg       0.85      0.86      0.85     14653
```

**classification_report 해석**

```
클래스 0 (≤50K) : precision 0.87, recall 0.95 → 잘 맞춤
클래스 1 (>50K) : precision 0.79, recall 0.55 → recall 낮음
                  → 실제 고소득자 중 45%를 저소득으로 오분류
```

> recall이 낮은 이유 — 데이터 불균형 (0: 75%, 1: 25%)  
> `class_weight='balanced'` 적용 시 recall 개선 가능하나 precision 하락 trade-off

---

## 💡 핵심 개념 정리

### Pipeline 구조 전체 흐름

```
fit(train_x, train_y) 호출 시
         ↓
[ColumnTransformer]
  num_cols → SimpleImputer(median) → StandardScaler
  cat_cols → SimpleImputer(mode)   → OneHotEncoder
         ↓
[RandomForestClassifier]
  전처리된 데이터로 학습
         ↓
predict(test_x) 호출 시
  동일한 전처리 자동 적용 후 예측
```

### roc_auc vs accuracy

|지표|설명|불균형 데이터|
|---|---|---|
|accuracy|전체 정답 비율|부적합 (다수 클래스에 치우침)|
|roc_auc|클래스 구분 능력|**적합**|

> roc_auc **0.91** → 매우 우수한 분류 성능  
> accuracy **0.86** 보다 roc_auc가 더 신뢰할 수 있는 지표

---
# 📄 ex29quiz.py — 랜덤포레스트 실습 (Red Wine Quality 분류)

#머신러닝 #RandomForest #다중분류 #WineQuality #실습 #sklearn #Python

---

## 📌 이번 실습 포인트

- Kaggle Red Wine Quality 데이터셋 — 품질 등급 **다중 분류 (3~8)**
- `RandomForestClassifier` — 배깅 기반 앙상블
- 교차 검증 (`cross_val_score`)
- 특성 중요도 (`feature_importances_`) 확인 & 시각화
- 새로운 데이터 예측

---

## Step 0 — Import

```python
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler    # 랜덤포레스트는 불필요, 참고용
from sklearn.ensemble import RandomForestClassifier
```

---

## Step 1 — 데이터 로드 & 확인

```python
df = pd.read_csv("winequality-red.csv")
# winequality-red.csv : 세미콜론(;) 구분자 사용
# UCI에서 직접 로드 시 : pd.read_csv(url, sep=';')

print(df.head(5), df.shape)  # (1599, 12)
print(df.info())
# 결측치 없음 → 전처리 최소화 가능
# 랜덤포레스트는 트리 기반 → 스케일링 불필요
```

**데이터 구성**

|컬럼|설명|
|---|---|
|fixed acidity|고정 산도|
|volatile acidity|휘발성 산도|
|citric acid|구연산|
|residual sugar|잔당|
|chlorides|염화물|
|free sulfur dioxide|유리 이산화황|
|total sulfur dioxide|총 이산화황|
|density|밀도|
|pH|산성도|
|sulphates|황산염|
|alcohol|알코올 도수|
|**quality**|**품질 등급 (target, 3~8)**|

---

## Step 2 — feature / label 분리

```python
# quality 컬럼을 제외한 나머지 11개 → feature
x = df.drop('quality', axis=1)

# quality 컬럼 → label (3, 4, 5, 6, 7, 8 — 6개 클래스)
y = df['quality']
```

**클래스 분포**

```
3  :   10개  ░░░░░░░░░░
4  :   53개  ░░░░░░░░░░
5  :  681개  ██████████████████████
6  :  638개  █████████████████████
7  :  199개  ██████░░░░
8  :   18개  ░░░░░░░░░░
```

> 5, 6등급에 데이터가 집중 → **클래스 불균형**  
> 이로 인해 정확도가 낮게 나오는 건 데이터 특성상 어쩔 수 없음

---

## Step 3 — Train/Test 분할

```python
# test_size=0.3 : 전체의 30%를 테스트용으로 분리
# random_state=42 : 분할 결과 고정 (재현 가능)
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.3, random_state=42
)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
# (1117, 11) (479, 11) (1117,) (479,)
```

---

## Step 4 — 모델 생성 & 학습

```python
# RandomForestClassifier : 여러 개의 결정트리를 앙상블(배깅)한 분류 모델
# criterion='gini' : 분할 기준 — 불순도(Gini Index) 사용
# n_estimators=500 : 결정트리 500개 생성
# random_state=42 : 재현을 위한 시드 고정
model = RandomForestClassifier(criterion='gini', n_estimators=500, random_state=42)
model.fit(train_x, train_y)  # train 데이터로 학습
```

**랜덤포레스트 내부 동작**

```
① train 데이터에서 Bootstrap 샘플 500개 생성 (복원 추출)
② 각 샘플로 결정트리 500개 독립 학습
   (분기마다 feature도 랜덤하게 일부만 사용 → sqrt(11) ≈ 3개)
③ test 데이터 → 500개 트리에 모두 통과
④ 500개 예측 결과를 다수결(투표) → 최종 예측 등급
```

---

## Step 5 — 예측 & 성능 확인

```python
pred = model.predict(test_x)         # 예측 클래스 (3~8 중 하나)
proba = model.predict_proba(test_x)  # 각 클래스별 확률 (shape: 479 × 6)

print('예측값 : ', pred[:5])
# [5 6 5 6 6]
print('실제값 : ', np.array(test_y[:5]))
# [5 4 5 5 6]

print('맞춘 갯수 : ', sum(test_y == pred))               # 337
print('전체 대비 맞춘 비율: ', sum(test_y == pred) / len(test_y))
print('분류 정확도 : ', accuracy_score(test_y, pred))    # 0.7035
```

> 정확도 **70.35%** — 6개 클래스 분류치고는 준수한 수치  
> 랜덤 예측(클래스 6개 균등)이면 약 16.7% → 랜덤보다 훨씬 우수

**train vs test 정확도 비교 (과적합 확인)**

```python
# 추가 확인 권장
print('train score : ', model.score(train_x, train_y))  # ~0.99 (과적합 의심)
print('test score : ', model.score(test_x, test_y))     # 0.7035
```

> train(99%) vs test(70%) 차이가 크면 **과적합(Overfitting)**  
> `max_depth`, `min_samples_split` 조정으로 완화 가능

---

## Step 6 — 교차 검증 (K-Fold)

```python
# cross_val_score : K-Fold 교차검증을 한 줄로 수행
# 분류 문제 → 내부적으로 Stratified K-Fold 자동 적용
# cv=5 : 5번 분할해서 각각 학습·검증 반복
cross_vali = cross_val_score(model, x, y, cv=5)
print(cross_vali)
# [0.53125  0.56740  0.60815  0.58934  0.58934]

print('교차 검증 평균 정확도 : ', np.round(np.mean(cross_vali), 5))
# 0.5771
```

**단순 test vs 교차검증 비교**

```
단순 test 정확도  : 0.7035  ← 운이 좋은 분할일 수 있음
교차검증 평균     : 0.5771  ← 더 신뢰할 수 있는 실제 성능
```

> 두 수치의 차이가 크다 → 단순 test 결과가 **과대평가**됐을 가능성  
> 교차검증 평균을 실제 성능으로 보는 것이 더 정확

**K-Fold 동작 구조 (cv=5)**

```
전체 데이터 1599개
┌──────┬──────┬──────┬──────┬──────┐
│ 검증 │ 학습 │ 학습 │ 학습 │ 학습 │  1회 → 정확도 0.531
├──────┼──────┼──────┼──────┼──────┤
│ 학습 │ 검증 │ 학습 │ 학습 │ 학습 │  2회 → 정확도 0.567
├──────┼──────┼──────┼──────┼──────┤
│ 학습 │ 학습 │ 검증 │ 학습 │ 학습 │  3회 → 정확도 0.608
├──────┼──────┼──────┼──────┼──────┤
│ 학습 │ 학습 │ 학습 │ 검증 │ 학습 │  4회 → 정확도 0.589
├──────┼──────┼──────┼──────┼──────┤
│ 학습 │ 학습 │ 학습 │ 학습 │ 검증 │  5회 → 정확도 0.589
└──────┴──────┴──────┴──────┴──────┘
                              평균 → 0.5771
```

---

## Step 7 — 특성 중요도

```python
print('중요 변수 확인하기 ---')

# feature_importances_ : 각 특성이 불순도 감소에 기여한 정도
# - 값의 합 = 1.0
# - 수치가 클수록 예측에 더 많이 기여함
print('특성(변수) 중요도 : ', model.feature_importances_)
# [0.07797  0.10185  0.07041  0.06891  0.08102  0.06623
#  0.10149  0.09389  0.07249  0.10978  0.15595]
```

**특성 중요도 해석**

```
alcohol           : 0.156  ████████████████  ← 1위, 알코올 도수
sulphates         : 0.110  ███████████░░░░░  ← 2위, 황산염
volatile acidity  : 0.102  ██████████░░░░░░  ← 3위, 휘발성 산도
total sulfur dio  : 0.101  ██████████░░░░░░  ← 4위
...
```

> **알코올 도수(alcohol)** 가 와인 품질 예측에 가장 큰 영향  
> 황산염, 휘발성 산도 순으로 중요

---

## Step 8 — 특성 중요도 시각화

```python
import matplotlib.pyplot as plt

n_features = x.shape[1]   # 특성 수 = 11

plt.figure(figsize=(15, 8))

# barh : 수평 막대 그래프
# range(n_features) : y축 위치 [0, 1, ..., 10]
# model.feature_importances_ : 각 특성의 중요도 값
plt.barh(range(n_features), model.feature_importances_, align='center')

plt.xlabel('Feature importance Score')
plt.ylabel('Features')

# y축 눈금을 컬럼명으로 교체
plt.yticks(np.arange(n_features), x.columns)
plt.ylim(-1, n_features)   # y축 여백 설정

plt.show()
plt.close()
```

![[ex29quiz.png]]

> `x.shape[1]` → 열 수(특성 수) = 11  
> 정렬 없이 원래 컬럼 순서대로 출력 — 정렬하려면 `np.argsort` 사용

---

## Step 9 — 새로운 데이터 예측

```python
# 새로운 와인 샘플 : 11개 특성값을 리스트로 입력
# [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
#  free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
new_wine = [[7.5, 0.5, 0.3, 2.0, 0.08, 15.0, 50.0, 0.997, 3.3, 0.6, 10.5]]

new_pred = model.predict(new_wine)
print(f"새로운 와인 샘플의 예측 품질 등급: {new_pred[0]}")  # 예: 6

# 추가 — 클래스별 확률 확인
new_proba = model.predict_proba(new_wine)
print(dict(zip(model.classes_, new_proba[0].round(2))))
# 예: {3: 0.0, 4: 0.02, 5: 0.35, 6: 0.48, 7: 0.13, 8: 0.02}
# → 6등급 확률 48%로 가장 높음
```

---

## 💡 핵심 개념 정리

### 다중 분류 vs 이진 분류

||이진 분류|다중 분류|
|---|---|---|
|클래스 수|2개 (0, 1)|3개 이상|
|예시|생존/사망, 스팸/정상|와인 품질 3~8|
|`predict_proba`|shape: (n, 2)|shape: (n, 클래스수)|
|`roc_auc_score`|그대로 사용|`multi_class='ovr'` 필요|

### 정확도가 낮은 이유 — 클래스 불균형

```
5등급 681개  ████████████████████
6등급 638개  ███████████████████
7등급 199개  ██████
4등급  53개  ██
8등급  18개  █
3등급  10개  ░

→ 모델이 5, 6등급에 치우쳐 학습
→ 3, 4, 8등급은 샘플이 너무 적어 잘 못 맞춤
→ 전체 정확도가 낮게 나오는 근본 원인
```

### 과적합 확인 & 완화 방법

```python
# 과적합 확인
print(model.score(train_x, train_y))  # ~0.99 (너무 높음)
print(model.score(test_x, test_y))    # 0.70  (차이가 큼)

# 완화 방법
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,         # 트리 깊이 제한
    min_samples_split=5,  # 분할 최소 샘플 수 증가
    min_samples_leaf=2,   # 리프 최소 샘플 수 증가
    random_state=42
)
```
# 📄 ex30rf_regressor.py — 랜덤포레스트 회귀 (캘리포니아 주택 가격 예측)

#머신러닝 #RandomForest #회귀 #Regressor #RandomizedSearchCV #실습 #sklearn #Python

---

## 📌 이번 실습 포인트

- 랜덤포레스트로 **회귀(Regression)** 문제 풀기
- 분류(`Classifier`)와 회귀(`Regressor`) 차이 이해
- 평가 지표 — `MSE`, `R2(결정계수)`
- 특성 중요도 **내림차순 정렬** 시각화
- `RandomizedSearchCV` — 빠른 하이퍼파라미터 탐색

---

## Step 0 — Import

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Classifier가 아닌 Regressor
from sklearn.metrics import r2_score, mean_squared_error
```

> 분류 → `RandomForestClassifier`  
> 회귀 → `RandomForestRegressor`  
> 예측값이 카테고리(0,1,2)가 아닌 **연속적인 숫자(집값)** 이기 때문

---

## Step 1 — 데이터 로드 & 확인

```python
# fetch_california_housing : 캘리포니아 주택 가격 데이터
# as_frame=True : pandas DataFrame 형태로 반환
housing = fetch_california_housing(as_frame=True)

print(housing.DESCR)            # 데이터 설명문
print(housing.data[:2])         # feature 값 (상위 2행)
print(housing.target[:2])       # label 값 (주택 가격, 상위 2행)
print(housing.feature_names[:2])# feature 컬럼명

df = housing.frame  # data + target 합친 전체 DataFrame
print(df.head(3))
print(df.info())
```

**데이터 구성**

|컬럼|설명|
|---|---|
|MedInc|중위 소득|
|HouseAge|주택 평균 연령|
|AveRooms|평균 방 수|
|AveBedrms|평균 침실 수|
|Population|인구 수|
|AveOccup|평균 거주자 수|
|Latitude|위도|
|Longitude|경도|
|**MedHouseVal**|**중위 주택 가격 (target)**|

> 총 20,640개 샘플 / 결측치 없음

---

## Step 2 — feature / label 분리

```python
# MedHouseVal(중위 주택 가격) 제외한 8개 컬럼 → feature
x = df.drop('MedHouseVal', axis=1)

# MedHouseVal → label (연속형 숫자 — 회귀 대상)
y = df['MedHouseVal']
```

---

## Step 3 — Train/Test 분할

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
# 20640개 → train 14448개 (70%) / test 6192개 (30%)
# 회귀 문제라 stratify 불필요 (연속값은 비율 유지 개념 없음)
```

---

## Step 4 — 모델 생성 & 학습

```python
# RandomForestRegressor : 회귀용 랜덤포레스트
# n_estimators=200 : 결정트리 200개 생성
rfmodel = RandomForestRegressor(n_estimators=200, random_state=42)
rfmodel.fit(x_train, y_train)
```

**분류 vs 회귀 비교**

| |RandomForestClassifier|RandomForestRegressor|
|---|---|---|
|예측값|클래스 (0, 1, 2...)|연속 숫자 (집값)|
|집계 방식|다수결 (투표)|**평균**|
|평가 지표|accuracy, f1|MSE, R2|

---

## Step 5 — 예측 & 평가

```python
y_pred = rfmodel.predict(x_test)

# MSE (Mean Squared Error) : 오차 제곱의 평균 → 낮을수록 좋음
print(f'MSE : {mean_squared_error(y_test, y_pred):.3f}')    # 0.254

# R2 (결정계수) : 0~1, 높을수록 좋음 (1이면 완벽)
print(f'R2(결정계수) : {r2_score(y_test, y_pred):.3f}')      # 0.807
```

**평가 지표 개념**

```
MSE = 평균((실제값 - 예측값)²)
→ 오차가 클수록 제곱으로 패널티 → 이상치에 민감

R2 = 1 - (모델 오차 합 / 전체 분산)
→ 0.807 = 모델이 주택 가격 변동의 80.7%를 설명
→ 나머지 19.3%는 모델이 설명 못하는 부분
```

|R2 값|해석|
|---|---|
|1.0|완벽한 예측|
|0.8~1.0|우수|
|0.6~0.8|보통|
|0.6 미만|미흡|

> R2 **0.807** → 우수한 성능

---

## Step 6 — 특성 중요도 시각화

```python
importances = rfmodel.feature_importances_

# np.argsort : 오름차순 정렬했을 때의 인덱스 반환
# [::-1]     : 뒤집기 → 내림차순 (중요도 높은 순)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))

# bar : 수직 막대 그래프 (이전 실습의 barh 수평과 다름)
# importances[indices] : 중요도를 내림차순으로 정렬
plt.bar(range(x.shape[1]), importances[indices], align='center')

# x축 눈금을 컬럼명으로 교체 (내림차순 정렬된 순서로)
plt.xticks(range(x.shape[1]), x.columns[indices], rotation=45)
plt.xlabel('feature name')
plt.ylabel('feature importances')
plt.tight_layout()
plt.show()
```

**`np.argsort` 동작 원리**

```python
importances = [0.05, 0.53, 0.14, 0.09, 0.08, 0.04, 0.09, 0.08]
#               0     1     2     3     4     5     6     7    ← 인덱스

np.argsort(importances)      # [5, 0, 7, 4, 6, 3, 2, 1]  ← 작은 것부터 인덱스
np.argsort(importances)[::-1]# [1, 2, 3, 6, 4, 7, 0, 5]  ← 큰 것부터 인덱스
                             #  ↑ MedInc(1)이 가장 중요
```

---

## Step 7 — 중요도 순위 DataFrame 저장

```python
print('중요 변수 순위정보 저장')

# indices로 컬럼명과 중요도를 같은 순서(내림차순)로 정렬해서 저장
ranking = pd.DataFrame({
    'feature': x.columns[indices],
    'importance': importances[indices]
})
print(ranking)
#       feature  importance
# 0      MedInc    0.525400  ← 중위소득이 집값에 가장 큰 영향
# 1    AveOccup    0.138819
# 2   Longitude    0.086695
# 3    Latitude    0.086512
# 4    HouseAge    0.054694
# 5    AveRooms    0.045933
# 6  Population    0.032089
# 7   AveBedrms    0.029859
```

> **MedInc(중위소득)** 이 0.525로 압도적 1위  
> 소득이 높은 지역일수록 집값이 높다는 현실 반영  
> Longitude(경도), Latitude(위도)도 상위권 → 위치가 집값에 중요

---

## Step 8 — RandomizedSearchCV 파라미터 튜닝

```python
from sklearn.model_selection import RandomizedSearchCV

# 탐색할 파라미터 범위 정의
param_dist = {
    'n_estimators': [200, 400, 800],            # 트리 개수
    'max_depth': [None, 10, 20, 30],            # 트리 최대 깊이
    'min_samples_leaf': [1, 2, 4],              # 리프노드 최소 샘플 수
    'min_samples_split': [2, 4, 8],             # 노드 분할 최소 샘플 수
    'max_features': [None, 'sqrt', 'log2', 1.0, 0.8, 0.6]  # 분할 시 고려할 최대 특성 수
}
# 전체 조합 수 : 3 × 4 × 3 × 3 × 6 = 648가지

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),  # 모델
    param_distributions=param_dist,          # 탐색 범위
    n_iter=20,       # 648가지 중 20개만 랜덤 샘플링
    scoring='r2',    # 회귀 → r2 기준으로 평가
    cv=3,            # 3겹 교차검증
    random_state=42,
    verbose=1        # 진행 상황 출력
)

search.fit(x_train, y_train)  # 탐색 + 학습 수행

print('best_params : ', search.best_params_)
best = search.best_estimator_   # 최적 파라미터로 재학습된 모델
print('best_score : ', search.best_score_)
print('final R2 : ', r2_score(y_test, best.predict(x_test)))
```

**GridSearchCV vs RandomizedSearchCV**

```
GridSearchCV    : 648가지 전부 탐색 × cv=3 → 1944번 학습 (느림)
RandomizedSearchCV : 20가지만 랜덤 탐색 × cv=3 →   60번 학습 (빠름)
```

| |GridSearchCV|RandomizedSearchCV|
|---|---|---|
|탐색 방식|전체 조합 탐색|랜덤 일부만 탐색|
|속도|느림|**빠름**|
|최적 보장|보장|보장 안 됨|
|사용 시점|조합 수 적을 때|**조합 수 많을 때**|

> `n_iter` 높일수록 더 많이 탐색 → 더 좋은 파라미터 찾을 가능성 높아지나 속도 느려짐  
> 보통 `n_iter=20~50` 사이를 많이 사용

---

## 💡 핵심 개념 정리

### MSE vs R2

```
실제값 : [3.0, 2.5, 4.0]
예측값 : [2.8, 2.6, 3.9]

MSE = ((3.0-2.8)² + (2.5-2.6)² + (4.0-3.9)²) / 3
    = (0.04 + 0.01 + 0.01) / 3
    = 0.02   ← 낮을수록 좋음

R2  = 1 - (모델 오차 합 / 전체 분산)
    = 1에 가까울수록 좋음
```

### `np.argsort` + `[::-1]` 패턴

```python
# 값을 직접 정렬하는 게 아니라 "정렬했을 때의 인덱스" 를 반환
# 이 인덱스로 다른 배열도 같은 순서로 정렬 가능

importances = [0.05, 0.53, 0.14]
idx = np.argsort(importances)[::-1]  # [1, 2, 0]

importances[idx]          # [0.53, 0.14, 0.05]  ← 중요도 정렬
x.columns[idx]            # ['MedInc', 'AveOccup', 'HouseAge']  ← 컬럼명도 같이 정렬
```

---
# XGBoost 개념 정리

#머신러닝 #XGBoost #앙상블 #Boosting #GradientBoosting

---

## 📌 한 줄 요약

> 약한 모델 여러 개를 **순차적으로 연결**해서 강한 모델을 만드는 앙상블 알고리즘  
> Gradient Boosting을 **병렬처리 + 최적화**해서 빠르게 만든 구현체

---

## 🌳 랜덤포레스트 vs XGBoost

둘 다 결정트리 여러 개를 쓰는 앙상블이지만 **학습 방식**이 완전히 다름

```
랜덤포레스트 (Bagging) — 병렬
┌──────┐  ┌──────┐  ┌──────┐
│ 트리1 │  │ 트리2 │  │ 트리3 │  → 각자 독립 학습 → 다수결
└──────┘  └──────┘  └──────┘
   (동시에 학습, 서로 무관)

XGBoost (Boosting) — 순차
트리1 학습
   ↓ 오답에 가중치↑
트리2 학습 (오답에 집중)
   ↓ 오답에 가중치↑
트리3 학습
   ↓
최종 예측 (가중 합산)
```

| |랜덤포레스트|XGBoost|
|---|---|---|
|방식|Bagging (병렬)|Boosting (순차)|
|트리 관계|독립적|이전 트리의 오답을 다음이 보완|
|속도|빠름|느림 (but 병렬처리로 개선)|
|성능|우수|**더 우수**|
|과적합|강함|가능성 있음|
|튜닝 난이도|낮음|높음|

---

## ⚙️ Gradient Boosting 원리

> 오답에 가중치를 주는 방식이 **경사하강법(Gradient Descent)** 기반

```
① 첫 번째 트리로 예측
② 실제값 - 예측값 = 잔차(오차) 계산
③ 잔차를 줄이는 방향으로 두 번째 트리 학습
④ 반복 → 잔차가 점점 줄어듦
⑤ 모든 트리의 예측을 합산 → 최종 예측
```

> XGBoost는 이 과정을 **병렬처리 + 정규화**로 최적화한 것

---

## 🏆 왜 캐글에서 유명한가

```
정확도  ── 오답을 계속 보완 → 높은 성능
속도    ── 병렬처리 지원 → 빠름
유연성  ── 분류, 회귀 모두 가능
과적합  ── 정규화(L1, L2) 내장
결측치  ── 자동 처리 지원
```

---

## 📐 주요 하이퍼파라미터

### 학습 관련

|파라미터|기본값|설명|
|---|---|---|
|`objective`|reg:linear|손실 함수 정의|
|`eval_metric`|objective별 자동|평가 지표|
|`seed`|0|재현을 위한 난수 고정|

**objective 주요 옵션**

|값|설명|
|---|---|
|`binary:logistic`|이진 분류 — 확률 반환|
|`multi:softmax`|다중 분류 — 클래스 반환|
|`multi:softprob`|다중 분류 — 확률 반환|
|`reg:squarederror`|회귀|

**eval_metric 주요 옵션**

|값|설명|사용 시점|
|---|---|---|
|`logloss`|이진 분류 손실|이진 분류|
|`mlogloss`|다중 분류 손실|**다중 분류**|
|`rmse`|평균 제곱근 오차|회귀|
|`auc`|ROC 곡선 아래 면적|이진 분류|
|`error`|오분류율|이진 분류|

---

### 부스팅 관련

|파라미터|기본값|설명|
|---|---|---|
|`n_estimators`|100|트리(약한 분류기) 개수|
|`max_depth`|6|개별 트리 최대 깊이 (보통 3~10)|
|`eta` (learning_rate)|0.3|학습률 (보통 0.01~0.2)|
|`subsample`|1|트리당 사용할 샘플 비율 (0.5~1)|
|`colsample_bytree`|1|트리당 사용할 feature 비율 (0.5~1)|
|`min_child_weight`|1|과적합 방지 — 높을수록 보수적|
|`gamma`|0|분할에 필요한 최소 손실 감소량|
|`lambda`|1|L2 정규화 (Ridge)|
|`alpha`|0|L1 정규화 (Lasso)|
|`scale_pos_weight`|1|클래스 불균형 보정 (불균형 시 0보다 크게)|

---

### 일반 관련

|파라미터|설명|
|---|---|
|`booster='gbtree'`|트리 기반 모델 (기본)|
|`booster='gblinear'`|선형 모델|
|`n_jobs`|병렬처리 코어 수 (-1 = 전체)|

---

## 🆚 하이퍼파라미터 튜닝 방법 비교

|방법|설명|장점|단점|
|---|---|---|---|
|for 루프|직접 반복문 작성|이해 쉬움|파라미터 많으면 중첩 복잡|
|GridSearchCV|전체 조합 탐색|최적 보장|조합 많으면 느림|
|RandomizedSearchCV|랜덤 일부 탐색|빠름|최적 보장 안 됨|
|베이즈 최적화 (hyperopt)|이전 결과 반영해 탐색|효율적|구현 복잡|

```
조합 수가 적다  → GridSearchCV
조합 수가 많다  → RandomizedSearchCV
실무/대회      → 베이즈 최적화 (hyperopt)
```

---

## 🐍 기본 사용법

```python
import xgboost as xgb

# 이진 분류
xgb_clf = xgb.XGBClassifier(
    booster='gbtree',
    max_depth=6,
    n_estimators=200,
    eval_metric='logloss',   # 이진 분류
    random_state=42
)

# 다중 분류
xgb_clf = xgb.XGBClassifier(
    booster='gbtree',
    max_depth=6,
    n_estimators=200,
    eval_metric='mlogloss',  # 다중 분류
    random_state=42
)

xgb_clf.fit(x_train, y_train)
pred = xgb_clf.predict(x_test)
```

> ⚠️ 다중 분류 시 label이 **0부터 시작하는 연속 정수**여야 함  
> 1~7처럼 중간이 빠진 경우 → `LabelEncoder` 또는 `mapping` 으로 변환 필요

---

## ⚠️ 단점

- 랜덤포레스트보다 **과적합 가능성 높음**
- 하이퍼파라미터가 많아 **튜닝이 복잡**
- 데이터가 **적을 때는** 랜덤포레스트가 나을 수 있음
- 이상치에 **민감**

---

## 💡 XGBoost vs LightGBM

| |XGBoost|LightGBM|
|---|---|---|
|출시|2014|2017|
|속도|보통|**빠름**|
|메모리|많이 사용|**적게 사용**|
|대용량|느림|**강함**|
|소용량|준수|과적합 주의|
|성능|우수|비슷하거나 우수|

> 실무에서는 대용량일수록 **LightGBM** 선호  
> 데이터가 적으면 **XGBoost** 또는 **랜덤포레스트** 권장

---
# 📄 ex31xgboost.py — XGBoost & LightGBM 실습 (유방암 분류)

#머신러닝 #XGBoost #LightGBM #Boosting #BreastCancer #실습 #sklearn #Python

---

## 📌 이번 실습 포인트

- `XGBClassifier` vs `LGBMClassifier` 성능 비교
- **gain 기준** 특성 중요도 추출 & 비교
- 두 모델의 특성 중요도를 **나란히 시각화**
- Boosting 계열 알고리즘 실전 적용

---

## Step 0 — Import

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# pip install xgboost lightgbm
import xgboost as xgb
from lightgbm import LGBMClassifier  # xgboost보다 성능 우수하나 자료가 적으면 과적합 발생
import lightgbm as lgb
```

---

## Step 1 — 데이터 로드 & 확인

```python
data = load_breast_cancer()

# DataFrame으로 변환 (컬럼명 포함)
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0: malignant(악성), 1: benign(양성)

print(x[:3], x.shape)   # (569, 30)
print(y[:3], y.shape)

# 클래스별 샘플 수 확인 — 불균형 여부 체크
print('레이블 분포 : ', {name:(y == i).sum() for i, name in enumerate(data.target_names)})
# 'malignant'(악성): 212, 'benign'(양성): 357
```

**데이터 구성**

|항목|내용|
|---|---|
|샘플 수|569개|
|feature 수|30개 (세포핵 관련 측정값)|
|클래스|0: malignant(악성) 212개 / 1: benign(양성) 357개|
|불균형|약 37% : 63% → `stratify=y` 권장|

---

## Step 2 — Train/Test 분할

```python
# stratify=y : 클래스 비율 유지 (불균형 데이터 대응)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=12, stratify=y
)
print(x_train.shape, x_test.shape)  # (455, 30) (114, 30)
```

---

## Step 3 — XGBoost 모델 생성 & 학습

```python
xgb_clf = xgb.XGBClassifier(
    booster='gbtree',      # 'gbtree':트리 기반, 'gblinear':선형 모델
    max_depth=6,           # 개별 결정트리 최대 깊이
    n_estimators=200,      # 약한 분류기(트리)의 개수
    eval_metric='logloss', # 이진 분류 평가 지표
    random_state=42
)
xgb_clf.fit(x_train, y_train)
```

**주요 파라미터**

|파라미터|값|설명|
|---|---|---|
|booster|gbtree|트리 기반 부스팅|
|max_depth|6|트리 깊이 (기본값, 과적합 시 줄임)|
|n_estimators|200|트리 200개 순차 학습|
|eval_metric|logloss|이진분류 → logloss / 다중분류 → mlogloss|

---

## Step 4 — LightGBM 모델 생성 & 학습

```python
lgb_clf = LGBMClassifier(
    n_estimators=200,
    random_state=42,
    verbose=-1   # 학습 로그 숨기기 (경고 메시지 제거)
)
lgb_clf.fit(x_train, y_train)
```

> `verbose=-1` — LightGBM은 기본적으로 학습 로그를 출력  
> `-1` 로 설정하면 `No further splits` 등 경고 메시지 숨김

---

## Step 5 — 예측 & 성능 비교

```python
pred_xgb = xgb_clf.predict(x_test)
pred_lgb = lgb_clf.predict(x_test)

print(f'XGBClassifier  acc : {accuracy_score(y_test, pred_xgb):.5f}')  # 0.96491
print(f'LGBMClassifier acc : {accuracy_score(y_test, pred_lgb):.5f}')  # 0.99123
```

**결과 비교**

|모델|정확도|
|---|---|
|XGBoost|0.96491|
|**LightGBM**|**0.99123**|

> LightGBM이 약 2.6% 높음  
> 단, breast_cancer는 569개로 **데이터가 적어 과적합 가능성** 있음  
> 교차검증으로 실제 성능 재확인 권장

---

## Step 6 — 특성 중요도 추출 (gain 기준)

```python
# XGBoost : get_booster().get_score()로 gain 값 추출
booster = xgb_clf.get_booster()
xgb_gain = pd.Series(booster.get_score(importance_type='gain'))

# LightGBM : booster_.feature_importance()로 gain 값 추출
lgb_gain = pd.Series(
    lgb_clf.booster_.feature_importance(importance_type='gain'),
    index=x_train.columns  # 컬럼명 인덱스로 설정
)
```

**importance_type 종류**

|값|설명|
|---|---|
|`gain`|해당 feature가 분할 시 가져온 **불순도 감소량** — 품질 기준|
|`weight`|해당 feature가 분할에 **사용된 횟수** — 빈도 기준|
|`cover`|해당 feature가 커버한 **샘플 수**|

> `gain` 이 가장 의미 있는 중요도 지표 — 실제로 예측에 기여한 정도

---

## Step 7 — 중요도 비율(%) 변환 & 정렬

```python
# 각 피처의 중요도를 전체 합 대비 비율(%)로 변환
# 0으로 나누기 방지 : xgb_gain.sum() != 0 조건 추가
xgb_gain_pct = 100 * xgb_gain / (xgb_gain.sum() if xgb_gain.sum() != 0 else 1)
lgb_gain_pct = 100 * lgb_gain / (lgb_gain.sum() if lgb_gain.sum() != 0 else 1)

# XGBoost가 사용하지 않은 피처 → 0으로 채움 (두 모델 컬럼 통일)
xgb_gain_pct = xgb_gain_pct.reindex(x_train.columns).fillna(0)
lgb_gain_pct = lgb_gain_pct.reindex(x_train.columns).fillna(0)

# 두 모델의 중요도를 하나의 DataFrame으로 합치기
comp_df = pd.DataFrame({
    'XGBoost (gain %)': xgb_gain_pct,
    'LightGBM (gain %)': lgb_gain_pct,
}).sort_values('XGBoost (gain %)', ascending=False)

print(comp_df.head(10))  # 중요 피처(변수) top-10
```

**`reindex + fillna(0)` 이 필요한 이유**

```
XGBoost는 중요도가 0인 피처를 아예 결과에서 제외함
→ 컬럼 수가 30개보다 적을 수 있음
→ reindex로 전체 30개 컬럼으로 맞추고
→ fillna(0)으로 빠진 피처는 0으로 채움
→ 두 모델 비교 시 컬럼 수 통일
```

---

## Step 8 — 특성 중요도 시각화 (나란히 비교)

```python
topk = 5                    # 상위 5개 피처만 시각화
top = comp_df.head(topk)[::-1]  # 내림차순 → 뒤집기 (barh는 아래서 위로 출력)

# 1행 2열 서브플롯 : XGBoost / LightGBM 나란히
fig, axes = plt.subplots(1, 2, figsize=(8, 5))

# 두 모델의 최대 중요도값 → x축 범위 통일 (공정한 비교)
xmax = float(np.ceil(top.max().max()))

# zip으로 axes와 컬럼명을 묶어서 반복
for ax, col in zip(axes, ['XGBoost (gain %)', 'LightGBM (gain %)']):
    ax.barh(top.index, top[col])                        # 수평 막대 그래프
    ax.set_title(f'{col.split()[0]} Feature importance') # 'XGBoost' or 'LightGBM'
    ax.set_xlabel('Importance (%)')
    ax.set_xlim(0, xmax)   # x축 범위 통일 → 두 모델 시각적으로 공정 비교

plt.tight_layout()
plt.show()
```

![[ex31xgboost.png]]

**시각화 핵심 포인트**

```python
top = comp_df.head(topk)[::-1]
# head(5) → 상위 5개 (내림차순 정렬된 상태)
# [::-1]  → 뒤집기
# barh는 아래→위 방향으로 출력하기 때문에
# 뒤집지 않으면 중요도 낮은 것이 위에 표시됨

xmax = float(np.ceil(top.max().max()))
# top.max()     → 각 컬럼의 최대값 (XGBoost 최대, LightGBM 최대)
# .max()        → 그 중 가장 큰 값
# np.ceil()     → 올림 → 깔끔한 축 범위
# 두 그래프의 x축을 동일하게 맞춰야 시각적 비교가 공정함
```

---

## 💡 핵심 개념 정리

### XGBoost vs LightGBM

| |XGBoost|LightGBM|
|---|---|---|
|트리 성장 방식|Level-wise (균형)|**Leaf-wise (비균형)**|
|속도|보통|**빠름**|
|메모리|많이 사용|**적게 사용**|
|소용량 데이터|준수|과적합 주의|
|대용량 데이터|느림|**강함**|
|verbose 기본|조용함|로그 많이 출력|

**트리 성장 방식 차이**

```
Level-wise (XGBoost)        Leaf-wise (LightGBM)
       [Root]                      [Root]
      /      \                    /      \
  [L1]      [R1]             [L1]      [R1]
  /  \      /  \                       /  \
[L2][R2] [L3][R3]                   [R2] [R3]
(모든 레벨 균등 분할)          (손실 가장 큰 리프만 분할)
→ 안정적, 느림                 → 빠름, 과적합 주의
```

### gain 기반 중요도 계산

```
각 트리의 모든 분기에서
    해당 feature 사용 시 불순도(loss) 감소량 누적
    → 전체 트리 평균
    → 합계가 100%가 되도록 정규화

gain이 높다 = 예측에 실질적으로 많이 기여함
```

### plt.subplots 활용

```python
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
# 1행 2열 서브플롯 생성
# axes[0] → 왼쪽 그래프 (XGBoost)
# axes[1] → 오른쪽 그래프 (LightGBM)

# zip으로 묶으면 더 간결
for ax, col in zip(axes, ['XGBoost (gain %)', 'LightGBM (gain %)']):
    ax.barh(...)  # 각 ax에 그래프 그리기
```

---
# 📄 ex32boost_iris.py — LightGBM 실습 (Iris 다항분류)

#머신러닝 #LightGBM #Boosting #Iris #다항분류 #실습 #sklearn #Python

---

## 📌 이번 실습 포인트

- `LGBMClassifier` 로 Iris 3개 클래스 분류
- `boosting_type='gbdt'` — Gradient Boosting 방식 지정
- 혼동행렬 3가지 방법으로 정확도 확인
- `joblib` 으로 모델 저장 & 불러오기
- `predict_proba` — softmax 확률값 확인
- 결정경계 시각화

---

## Step 0 — Import

```python
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler    # 트리 기반이라 불필요, 참고용
from sklearn.linear_model import LogisticRegression # 참고용 import
from lightgbm import LGBMClassifier
```

> `LogisticRegression` — 다중 클래스를 지원하도록 일반화됨  
> 이를 **Softmax Regression** 또는 **Multinomial Logistic Regression** 이라고 부름  
> LightGBM은 트리 기반 → **표준화 불필요**

---

## Step 1 — 데이터 로드 & 확인

```python
iris = datasets.load_iris()
print(iris.keys())
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']

print(iris.target)
# [0 0 0 ... 1 1 1 ... 2 2 2]  ← 클래스 0,1,2 각 50개씩

print(iris.data[:3])
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]]

# 꽃잎 길이(col2)와 꽃잎 너비(col3)의 상관계수
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])[0, 1])  # 0.9628654314027961
```

---

## Step 2 — feature / label 선택

```python
# 꽃잎 길이(index 2), 꽃잎 너비(index 3) 두 특성만 사용
x = iris.data[:, [2, 3]]
y = iris.target

print(x.shape, ' ', y.shape)    # (150, 2)   (150,)
print(x[:3], y[:3], set(map(int, y)))
# [[1.4 0.2]
#  [1.4 0.2]
#  [1.3 0.2]] [0 0 0] {0, 1, 2}
```

**Iris 클래스 구성**

|클래스|이름|샘플 수|
|---|---|---|
|0|Setosa|50개|
|1|Versicolor|50개|
|2|Virginica|50개|

> 두 특성 간 상관계수 **0.963** → 거의 같은 정보를 담고 있음

---

## Step 3 — Train/Test 분할

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (105, 2) (45, 2) (105,) (45,)
```

---

## Step 4 — (참고) 표준화 Scaling

```python
"""
# LightGBM은 트리 기반 → 표준화 불필요
# 하지만 로지스틱 회귀, KNN, SVM 등에서는 필수

sc = StandardScaler()
sc.fit(x_train)              # train 기준으로만 fit
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)  # test는 transform만 (fit_transform 금지)

# 원복
ori_x_train = sc.inverse_transform(x_train)
"""
```

> ⚠️ test에 `fit_transform` 사용 시 → **데이터 누수(Data Leakage)** 발생  
> Iris 데이터는 특성 간 크기 차이가 거의 없어 표준화 효과 미미

---

## Step 5 — LightGBM 모델 생성 & 학습

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    boosting_type='gbdt',  # 'gbdt':Gradient Boosting, 'dart':드롭아웃 적용, 'goss':빠른 샘플링
    n_estimators=500,      # 약한 분류기(트리)의 개수
    random_state=0,
    verbose=-1             # 학습 로그 숨기기
)
print(model)
model.fit(x_train, y_train)
```

**boosting_type 옵션**

|값|설명|
|---|---|
|`gbdt`|기본 Gradient Boosting — 가장 많이 사용|
|`dart`|드롭아웃 적용 → 과적합 방지|
|`goss`|빠른 샘플링 → 속도 우선|

---

## Step 6 — 예측

```python
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  2 1 1 2 0 2 0 0]
print('실제값 : ', y_test)
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  1 1 1 2 0 2 0 0]

print(f'총 갯수:{len(y_test)}, 오류수:{(y_test != y_pred).sum()}')
# 총 갯수:45, 오류수:1
```

---

## Step 7 — 정확도 확인 3가지 방법

### 방법 1 — accuracy_score

```python
print(f'{accuracy_score(y_test, y_pred)}')  # 0.9777777777777777
```

### 방법 2 — 혼동행렬 (Confusion Matrix)

```python
con_mat = pd.crosstab(y_test, y_pred, rownames=['실제값'], colnames=['예측값'])
print(con_mat)
# 예측값   0   1   2
# 실제값
# 0    16   0   0
# 1     0  17   1  ← Versicolor → Virginica 1개 오분류
# 2     0   1  11  ← Virginica → Versicolor 1개 오분류

print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))
# 0.9777777777777777
```

**혼동행렬 구조**

```
              예측 0   예측 1   예측 2
실제 0 (Setosa)     [ 16      0      0  ]  ← 완전 정답
실제 1 (Versicolor) [  0     17      1  ]  ← 1개 오분류
실제 2 (Virginica)  [  0      1     11  ]  ← 1개 오분류
```

> Setosa는 완전 분리 — 꽃잎 크기가 다른 두 종과 뚜렷하게 다름  
> Versicolor ↔ Virginica 경계에서 총 2개 오분류

### 방법 3 — model.score

```python
print('test score : ', model.score(x_test, y_test))     # 0.9777
print('train score : ', model.score(x_train, y_train))  # 0.9809
```

> test(0.977) vs train(0.980) 차이 매우 작음 → **과적합 없음**  
> 이전 랜덤포레스트(ex28)와 동일한 결과 — LightGBM도 준수한 성능

---

## Step 8 — 모델 저장 & 불러오기

```python
import joblib   # pickle보다 빠르고 대용량 지원

# 모델 저장
joblib.dump(model, 'logimodel.pkl')

# 기존 모델 삭제 후 불러오기
del model
read_model = joblib.load('logimodel.pkl')
# 이후 코드에서는 read_model 사용
```

---

## Step 9 — 새로운 데이터 예측

```python
new_data = np.array([[5.5, 2.2], [0.6, 0.3], [1.1, 0.5]])
# [꽃잎 길이, 꽃잎 너비] 형식

new_pred = read_model.predict(new_data)
print('예측 결과 : ', new_pred)  # [2 0 0]
# softmax 확률값 중 가장 큰 인덱스가 출력된 값

# softmax 확률값 확인
print(read_model.predict_proba(new_data))
# [[4.50e-06  5.66e-03  9.94e-01]  → Virginica  99.4%
#  [9.99e-01  1.37e-05  2.88e-06]  → Setosa    99.9%
#  [9.99e-01  1.37e-05  2.88e-06]] → Setosa    99.9%
```

**predict vs predict_proba**

| |predict|predict_proba|
|---|---|---|
|출력|클래스 1개|각 클래스별 확률 배열|
|예시|`[2]`|`[0.0, 0.006, 0.994]`|
|합계|-|항상 1.0|
|활용|최종 예측|확신 정도 확인|

---

## Step 10 — 결정경계 시각화

```python
import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib.colors import ListedColormap

def plot_decision_regionFunc(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정경계 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    # meshgrid : 평면 전체 격자점 생성
    xx, yy = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    # 격자점 전체에 predict() 적용 → 결정경계 색칠
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # 클래스별 산점도
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
                    color=cmap(idx), marker=markers[idx], label=cl)

    # test 데이터 별도 마커로 표시
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

# train + test 합쳐서 전체 시각화
x_combined_std = np.vstack((x_train, x_test))  # 수직 결합
y_combined = np.hstack((y_train, y_test))        # 수평 결합

plot_decision_regionFunc(
    X=x_combined_std,
    y=y_combined,
    classifier=read_model,
    test_idx=range(105, 150),  # test 데이터 인덱스
    title='scikit-learn제공'
)
```

**결정경계 시각화 결과**

![[ex32boost_iris.png]]

**시각화 해석**

```
빨간 영역 (class 0, Setosa)
→ 꽃잎 길이 ~2.5 이하 — 완전히 분리됨

파란 영역 (class 1, Versicolor)
→ 꽃잎 길이 2.5~5.0, 너비 낮은 구간

초록 영역 (class 2, Virginica)
→ 꽃잎 길이 5.0 이상 또는 너비 높은 구간
```

> LightGBM도 의사결정트리 기반 → **수직·수평 계단 경계선** 형태  
> Versicolor(파란x) 와 Virginica(초록점) 경계 근처에서 오분류 발생  
> Setosa(빨간■) 는 완전 분리 — 경계가 명확

---

## 💡 핵심 개념 정리

### LightGBM vs 랜덤포레스트 (Iris 결과 비교)

| |랜덤포레스트 (ex28)|LightGBM (ex32)|
|---|---|---|
|정확도|0.9556|**0.9778**|
|오류 수|2개|1개|
|학습 방식|Bagging (병렬)|Boosting (순차)|
|과적합|train 0.990 / test 0.956|train 0.981 / test 0.978|

> LightGBM이 Iris에서 더 높은 정확도  
> 과적합 차이도 LightGBM이 더 작음

### softmax 출력값 해석

```
predict_proba 출력 : [[4.50e-06, 5.66e-03, 9.94e-01]]
                       클래스0    클래스1    클래스2
                        0.0%      0.6%     99.4%
                                           ↑
                                       최종 예측 : 2 (Virginica)
```

> LightGBM 내부에서 각 클래스에 대한 확률을 softmax로 계산  
> `predict()` 는 이 중 가장 높은 확률의 클래스 인덱스를 반환