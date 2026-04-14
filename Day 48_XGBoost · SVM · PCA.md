# Day 47_XGBoost · SVM · PCA
## 📅 2026-04-14

---
# 📄 ex33xgboost.py — XGBoost 실습 (산탄데르 고객만족 분류)

> **데이터셋**: Kaggle Santander Customer Satisfaction  
> **목표**: 고객 불만족 여부 이진 분류 (0: 만족, 1: 불만족)  
> **평가지표**: ROC-AUC (클래스 불균형 데이터에 적합)

---

## 1. 라이브러리 임포트

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier       # XGBoost 분류 모델
from xgboost import plot_importance     # 피처 중요도 시각화
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
```

---

## 2. 데이터 로드 및 탐색

```python
df = pd.read_csv("train_san.csv", encoding='latin-1')

print(df.shape)   # (76020, 371) → 76,020행, 371열(ID + 369 feature + TARGET)
print(df.info())
```

### 클래스 분포 확인

```python
print(df['TARGET'].value_counts())
# 0    73012  (만족)
# 1     3008  (불만족)

unsatisfied_cnt = df[df['TARGET'] == 1].TARGET.count()
total_cnt = df.TARGET.count()
print(f'불만족 비율은 {unsatisfied_cnt / total_cnt}')  # 약 3.96%
```

> ⚠️ **클래스 불균형 주의**: 불만족 비율이 약 4%에 불과 → 정확도보다 **ROC-AUC**로 평가해야 함

---

## 3. 데이터 전처리

```python
# var3 컬럼의 이상치(-999999)를 최빈값(2)으로 대체
df['var3'].replace(-999999, 2, inplace=True)

# ID 컬럼 제거 (식별자이므로 학습에 불필요)
df.drop('ID', axis=1, inplace=True)
```

### Feature / Label 분리

```python
x_features = df.iloc[:, :-1]   # 마지막 열(TARGET) 제외한 모든 열
y_label    = df.iloc[:, -1]    # TARGET 열만
# x_feature shape: (76020, 369)
```

---

## 4. 학습/검증 데이터 분리

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_features, y_label,
    test_size=0.2,    # 80% 학습, 20% 검증
    random_state=0
)
```

### 분포 비율 확인

```python
train_cnt = y_train.count()
test_cnt  = y_test.count()
print('학습데이터 레이블 값 분포 비율:', y_train.value_counts() / train_cnt)
print('검증데이터 레이블 값 분포 비율:', y_test.value_counts() / test_cnt)
# 학습: 0 → 96.1%, 1 → 3.9%
# 검증: 0 → 95.8%, 1 → 4.2%  ← 전체 비율과 유사하게 유지됨
```

---

## 5. 기본 XGBoost 모델 학습

```python
xgb_clf = XGBClassifier(
    n_estimators=500,          # 트리 개수 (실습에서는 시간상 5로 설정)
    random_state=12,
    early_stopping_rounds=10,  # 10라운드 동안 성능 개선 없으면 조기 종료
    eval_metric='auc'          # 평가 기준: AUC
)

xgb_clf.fit(
    x_train, y_train,
    eval_set=[(x_train, y_train), (x_test, y_test)],  # 학습/검증 동시 모니터링
    verbose=50  # 50라운드마다 출력
)
```

> ⚠️ **버전 주의**: XGBoost 최신 버전에서 `early_stopping_rounds`, `eval_metric`은  
> `fit()` 안이 아닌 **생성자(`XGBClassifier()`)에 넣어야 함**

### 성능 평가

```python
xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(x_test)[:, 1])
# predict_proba(x_test)[:, 1] → 불만족(1)일 확률값만 추출
print(f'xgb_roc_score : {xgb_roc_score:.5f}')  # 0.83431

pred = xgb_clf.predict(x_test)
print('예측값 : ', pred[:5])
print('실제값 : ', y_test[:5].values)   # ← x_test가 아닌 y_test !

from sklearn import metrics
print('분류 정확도 : ', metrics.accuracy_score(y_test, pred))  # 0.9583
```

---

## 6. GridSearchCV로 최적 파라미터 탐색

```python
params = {
    'max_depth'        : [5, 7],        # 트리 깊이 (깊을수록 복잡, 과적합 위험)
    'min_child_weight' : [1, 3],        # 리프 노드의 최소 관측치 가중치합 (클수록 보수적)
    'colsample_bytree' : [0.5, 0.75]    # 트리 생성 시 사용할 feature 비율
}

gridcv = GridSearchCV(xgb_clf, param_grid=params)
gridcv.fit(x_train, y_train, eval_set=[(x_test, y_test)])
print('최적 파라미터:', gridcv.best_params_)
# → {'colsample_bytree': 0.75, 'max_depth': 5, 'min_child_weight': 3}
```

### GridSearch 결과 평가

```python
gridcv_roc_score = roc_auc_score(
    y_test,
    gridcv.predict_proba(x_test)[:, 1],
    average='macro'
    # macro  : 클래스별 점수를 동등하게 평균 → 소수 클래스 성능 중요시
    # micro  : 전체 데이터 기반 평균 → 데이터 불균형이 심할 때 사용
)
print(f'gridcv_roc_score : {gridcv_roc_score:.5f}')  # 0.83870
```

---

## 7. 최적 파라미터로 모델 재생성

```python
xgb_clf2 = XGBClassifier(
    n_estimators=5,           # 실습용 (실전에서는 500+)
    random_state=12,
    max_depth=5,              # GridSearch 결과
    min_child_weight=3,       # GridSearch 결과
    colsample_bytree=0.75     # GridSearch 결과
)

xgb_clf2.fit(
    x_train, y_train,
    eval_set=[(x_train, y_train), (x_test, y_test)]
)

xgb_roc_score2 = roc_auc_score(y_test, xgb_clf2.predict_proba(x_test)[:, 1])
print(f'xgb_roc_score2 : {xgb_roc_score2:.5f}')  # 0.83825

pred2 = xgb_clf2.predict(x_test)
print('분류 정확도 :', metrics.accuracy_score(y_test, pred2))  # 0.9583
```

---

## 8. 피처 중요도 시각화

```python
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_importance(xgb_clf2, ax=ax, max_num_features=20)  # 상위 20개 feature만 표시
plt.show()
```

![[ex33xgboost.png]]

---

## 전체 흐름 요약

```
데이터 로드
    ↓
탐색(EDA) - 클래스 불균형 확인 (불만족 ~4%)
    ↓
전처리 - 이상치 처리, ID 제거
    ↓
Feature/Label 분리 → Train/Test Split
    ↓
기본 XGBoost 학습 → AUC: 0.83431
    ↓
GridSearchCV로 최적 파라미터 탐색 → AUC: 0.83870
    ↓
최적 파라미터로 재학습 → AUC: 0.83825
    ↓
피처 중요도 시각화
```

---

## 주요 개념 정리

### ROC-AUC를 쓰는 이유

클래스가 불균형(만족 96%, 불만족 4%)할 때 정확도는 의미 없음.  
모델이 전부 "만족"으로 예측해도 정확도 96% 달성 가능 → AUC로 평가해야 실제 성능 확인 가능.

### early_stopping_rounds

지정한 라운드 수 동안 검증 성능이 개선되지 않으면 학습을 조기 종료.  
`n_estimators`를 넉넉하게 설정하고 early stopping이 자동으로 최적 지점을 찾게 하는 것이 일반적.

### GridSearchCV

지정한 파라미터 조합을 전부 시도해 가장 성능이 좋은 조합을 찾아줌.  
`best_params_`로 최적 파라미터 확인 가능.

### predict_proba()[:, 1]

각 샘플이 클래스 1(불만족)일 확률을 반환.  
`predict()`는 0/1 이진값, `predict_proba()`는 확률값 → AUC 계산에는 확률값 필요.

---
# 📖 SVM (Support Vector Machine) 개념 정리

> **서포트 벡터 머신**: 데이터가 어느 카테고리에 속할지 판단하기 위해  
> **가장 적절한 경계**를 찾는 지도학습 기반 선형 분류 모델

---

## 1. 핵심 아이디어 — 어떤 경계선이 가장 좋을까?

두 집단을 나누는 직선은 수만 가지가 존재한다.  
그 중 **새로운 데이터가 들어와도 여전히 잘 구분하는** 직선이 최적의 경계선이다.

```
[직선 A]  두 집단 사이 거리가 멀다 → 새 데이터도 잘 분류 ✅
[직선 B]  한쪽 집단에 너무 가깝다 → 새 데이터 오분류 위험 ❌
```

> 💡 핵심: 경계선을 중심으로 **양쪽 집단이 최대한 멀리** 떨어져 있어야 한다

---

## 2. 개념도

![[svm.png]]

## 3. 3가지 핵심 개념

### 📌 서포트 벡터 (Support Vector)

각 집단에서 결정 경계에 **가장 가까이 위치한** 데이터 포인트들.  
이 포인트들이 경계선의 위치를 결정하는 핵심 역할을 한다.  
→ 나머지 데이터는 경계에 영향을 주지 않음

### 📌 마진 (Margin)

두 집단의 서포트 벡터 사이의 거리.  
SVM의 목표 = **마진을 최대화**하는 결정 경계를 찾는 것.

### 📌 결정 경계 (Decision Boundary)

마진의 정중앙을 지나는 선.  
두 집단을 분할하는 최종 기준선.

```
Class A  ●  ●  ●
               |  ← 서포트 벡터
          ─────────────  ← plus-plane  (점선)
               ↕  마진
          ══════════════  ← 결정 경계  (실선)
               ↕  마진
          ─────────────  ← minus-plane (점선)
               |  ← 서포트 벡터
Class B  ■  ■  ■
```

---

## 3. 결정 경계 구하는 순서

```
① 각 집단의 경계에 위치한 데이터(서포트 벡터)를 찾는다
        ↓
② 서포트 벡터를 지나는 평행한 두 직선을 긋는다 (plus-plane, minus-plane)
        ↓
③ 두 직선 사이의 거리 = 마진
        ↓
④ 마진의 중앙을 지나는 직선 = 결정 경계 (Decision Boundary)
```

---

## 4. 차원에 따른 결정 경계 형태

|feature 수|공간|결정 경계|
|---|---|---|
|2개|2차원|직선 (Line)|
|3개|3차원|평면 (Plane)|
|N개|N차원|초평면 (Hyperplane)|

---

## 5. 하드 마진 vs 소프트 마진

이상치(Outlier)를 얼마나 허용하느냐에 따라 달라진다.

![[하드마진 vs 소프트마진.png]]

### 🔴 하드 마진 (Hard Margin)

- 이상치를 **전혀 허용하지 않음**
- 모든 데이터를 완벽하게 분류
- 마진이 좁아짐
- **과적합(Overfitting)** 위험 — 새로운 데이터에 취약

### 🔵 소프트 마진 (Soft Margin)

- 이상치를 **어느 정도 허용**
- 몇 개 오분류를 감수하고 나머지를 더 잘 구분
- 마진이 넓어짐
- **과소적합(Underfitting)** 주의 — 패턴을 놓칠 수 있음

### ⚙️ C 파라미터로 조절

```python
SVC(C=1e10)  # C 크면 → 하드 마진 (이상치에 엄격)
SVC(C=0.01)  # C 작으면 → 소프트 마진 (이상치에 유연)
```

---

## 6. 커널 트릭 (Kernel Trick)

### 문제 상황

데이터가 선형으로 분리 불가능한 경우가 존재한다.

```
예) 1차원 직선상의 데이터
  🐯  🐰  🐰  🐯  🐰  🐯
← 직선 하나로는 절대 나눌 수 없음
```

### 해결 방법

기존 예측 변수를 변환하여 **더 높은 차원으로 올린다**.

![[커널트릭.png]]

```
1차원 (분리 불가)              2차원으로 변환 (분리 가능!)
x축: 식사 횟수        →       x축: 식사 횟수
                              y축: 식사 횟수²
                                   ↑ 이 공간에서는 직선으로 분리 가능!
```

> 💡 핵심: 저차원에서 비선형인 데이터를 고차원으로 변환하면 선형 분리가 가능해진다.  
> 실제로 고차원 변환을 하지 않고 **수식으로만 계산**해서 효율적으로 처리 → 이게 "트릭"인 이유

### 커널 종류

|커널|코드|설명|언제 쓰나|
|---|---|---|---|
|선형|`kernel='linear'`|직선으로 분리|선형 분리 가능할 때|
|다항식|`kernel='poly'`|다항식 곡선|곡선 경계가 필요할 때|
|방사형|`kernel='rbf'`|방사형 곡선|**실무에서 가장 많이 사용**|

---

## 7. 주요 파라미터

```python
from sklearn.svm import SVC

SVC(
    kernel='rbf',   # 커널 종류: linear / poly / rbf
    C=1.0,          # 마진 조절: 클수록 하드마진, 작을수록 소프트마진
    degree=3,       # poly 커널 전용: 다항식 차수 (높을수록 복잡한 곡선)
    coef0=1,        # poly 커널 전용: 고차/저차 영향력 비율
)
```

---

## 8. 기본 실습 코드

### 선형 분류

```python
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=40, centers=2, cluster_std=0.5, random_state=4)
y = 2 * y - 1   # 0/1 → -1/+1 변환 (SVM 관례)

model = SVC(kernel='linear', C=1e10).fit(X, y)

# 서포트 벡터 확인
print(model.support_vectors_)
```

### 비선형 분류 (다항식 커널)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

poly_svm = Pipeline([
    ("scaler", StandardScaler()),   # ⚠️ 스케일링 필수!
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_svm.fit(X, y)
```

> ⚠️ **스케일링 주의**: SVM은 거리 기반 알고리즘이므로  
> 반드시 `StandardScaler`로 정규화 후 학습해야 한다

---

## 9. SVM 실생활 활용 예시

|분야|활용 예|
|---|---|
|금융|기업 부도 예측 (매출 변동성, 현금흐름 등)|
|의료|환자 간 질병 분류 및 진단|
|제조|설비 이상 반응 예측 시스템|
|이미지|손글씨 인식, 얼굴 인식|

---

## 핵심 요약

|용어|설명|
|---|---|
|결정 경계|두 클래스를 나누는 선/평면/초평면|
|마진|결정 경계와 서포트 벡터 사이의 거리|
|서포트 벡터|결정 경계에 가장 가까운 데이터 포인트|
|하드 마진|이상치 불허 → 마진 좁음 → 과적합 위험|
|소프트 마진|이상치 허용 → 마진 넓음 → 과소적합 주의|
|커널 트릭|저차원 → 고차원 변환으로 비선형 분리 가능하게 함|
|초평면|고차원에서의 결정 경계|
|C 파라미터|클수록 하드마진, 작을수록 소프트마진|

---
# 📄 ex34svm.py — SVM 실습 (Iris 선형 분류)

> **알고리즘**: SVM (Support Vector Machine)  
> **목표**: 두 클래스를 마진이 최대가 되는 초평면으로 분리  
> **데이터**: make_blobs로 생성한 가상 이진 분류 데이터

---

## 1. 라이브러리 임포트 및 데이터 생성

```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

plt.rc('font', family='malgun gothic')

# 분류용 가상 데이터 생성
#   n_samples  : 총 데이터 수
#   centers    : 클래스(군집) 수
#   cluster_std: 군집의 퍼짐 정도 (작을수록 촘촘)
#   random_state: 재현 가능한 난수 고정
X, y = make_blobs(n_samples=50, centers=2, cluster_std=0.5, random_state=4)

# SVM 관례상 레이블을 0/1 → -1/+1 로 변환
# 수식에서 +1 클래스, -1 클래스로 구분하기 위함
y = 2 * y - 1
```

---

## 2. 학습 데이터 시각화

```python
plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', label="-1 클래스")
plt.scatter(X[y == +1, 0], X[y == +1, 1], marker='x', label="+1 클래스")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("학습용 데이터")
plt.show()
```

![[ex34svm1.png]]

---

## 3. SVM 모델 학습

```python
# SVC: Support Vector Classifier
#   kernel : 결정 경계 형태
#            'linear' → 직선
#            'rbf'    → 방사형 곡선 (실무에서 가장 많이 사용)
#            'poly'   → 다항식 곡선
#   C      : 마진 조절 파라미터
#            C 크면  → 하드 마진 (이상치에 엄격, 과적합 위험)
#            C 작으면 → 소프트 마진 (이상치에 유연, 과소적합 주의)
model = SVC(kernel='linear', C=1.0).fit(X, y)
```

> 💡 `C` 값을 바꿔가며 실행해보면 마진이 달라지는 게 눈으로 보임

---

## 4. 결정 경계 시각화 준비

```python
# 그래프 범위 설정 (데이터의 최솟값~최댓값)
xmin, xmax = X[:, 0].min(), X[:, 0].max()
ymin, ymax = X[:, 1].min(), X[:, 1].max()

# linspace: 범위를 균등하게 10등분한 배열 생성
xx = np.linspace(xmin, xmax, 10)
yy = np.linspace(ymin, ymax, 10)

# meshgrid: xx, yy를 조합해 2D 격자 좌표 생성
# X1, X2는 각각 격자의 x좌표, y좌표를 담은 2D 배열
X1, X2 = np.meshgrid(xx, yy)
```

---

## 5. decision_function 계산

```python
# decision_function: 각 포인트가 결정 경계로부터 얼마나 떨어져 있는지 부호 있는 거리값
#   양수 → +1 클래스 방향
#   0    → 결정 경계 (경계선 위)
#   음수 → -1 클래스 방향
z = np.empty(X1.shape)

for (i, j), val in np.ndenumerate(X1):  # 배열의 (행, 열) 인덱스와 값을 순서대로 꺼냄
    x1 = val
    x2 = X2[i, j]
    p = model.decision_function([[x1, x2]])  # 해당 좌표의 경계 거리값 계산
    z[i, j] = p[0]
```

---

## 6. 결정 경계 및 서포트 벡터 시각화

```python
plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', label="-1 클래스")
plt.scatter(X[y == +1, 0], X[y == +1, 1], marker='x', label="+1 클래스")

# contour: 등고선 그리기
#   levels=[-1, 0, 1] : decision_function 값이 -1, 0, +1인 선을 그림
#     -1 → minus-plane (점선) : -1 클래스 서포트 벡터를 지나는 선
#      0 → 결정 경계   (실선) : 마진의 중앙선
#     +1 → plus-plane  (점선) : +1 클래스 서포트 벡터를 지나는 선
plt.contour(X1, X2, z, levels=[-1, 0, 1], colors='k', linestyles=['dashed', 'solid', 'dashed'])

# 서포트 벡터 강조 표시 (반투명 큰 원)
# support_vectors_ : 학습 후 모델이 찾아낸 서포트 벡터 좌표 배열
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, alpha=0.3)

# 새로운 테스트 데이터 표시
x_new = [10, 2]
plt.scatter(x_new[0], x_new[1], marker='^', s=100)
plt.text(x_new[0] + 0.03, x_new[1] + 0.08, "테스트 데이터")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("SVM 예측 결과")
plt.show()
```

![[ex34svm2.png]]

---

## 7. 서포트 벡터 출력

```python
# 서포트 벡터: 결정 경계에 가장 가까이 위치한 데이터 포인트들
# 이 포인트들만이 경계선(마진)을 결정하며, 나머지 데이터는 영향을 주지 않음
print('서포트 벡터 좌표 :')
print(model.support_vectors_)
```

---

## 전체 흐름 요약

```
가상 데이터 생성 (make_blobs)
    ↓
레이블 변환 (0/1 → -1/+1)
    ↓
SVM 모델 학습 (kernel='linear', C=1.0)
    ↓
격자 좌표마다 decision_function 값 계산
    ↓
등고선으로 결정 경계 / 마진 시각화
    ↓
서포트 벡터 확인 (support_vectors_)
```

---

## 주요 개념 정리

### decision_function

각 데이터 포인트가 결정 경계로부터 얼마나 떨어져 있는지 나타내는 부호 있는 거리값.  
`levels=[-1, 0, 1]`로 등고선을 그릴 때 -1/+1이 점선(마진 경계), 0이 실선(결정 경계)이 됨.

### support_vectors_

학습 후 모델이 자동으로 찾아낸 서포트 벡터의 좌표 배열.  
이 포인트들만으로 경계가 결정되며 나머지 데이터는 영향을 주지 않음.

### kernel 파라미터

|값|설명|언제 쓰나|
|---|---|---|
|`'linear'`|직선 경계|선형 분리 가능할 때|
|`'rbf'`|방사형 곡선|실무에서 가장 많이 사용|
|`'poly'`|다항식 곡선|곡선 경계가 필요할 때|

### C 파라미터

| 값    | 마진          | 특징                |
| ---- | ----------- | ----------------- |
| C 크게 | 하드 마진 (좁음)  | 이상치에 엄격 → 과적합 위험  |
| C 작게 | 소프트 마진 (넓음) | 이상치에 유연 → 과소적합 주의 |

---
# 📄 ex35xor.py — SVM 실습 (AND / OR / XOR 논리 연산 분류)

> **목표**: 논리 연산(AND, OR, XOR)을 SVM과 로지스틱 회귀로 분류  
> **핵심**: XOR은 선형 분리 불가 → SVM이 로지스틱 회귀보다 우수

---

## 1. XOR 데이터 구조

```python
# XOR 진리표
#   p | q | 결과(XOR)
#   0 | 0 |  0
#   0 | 1 |  1
#   1 | 0 |  1
#   1 | 1 |  0
# ※ AND는 [0,0,0], [0,1,0], [1,0,0], [1,1,1]
# ※ OR  는 [0,0,0], [0,1,1], [1,0,1], [1,1,1] 로 바꿔서 실습 가능

x_data = [
    [0, 0, 0],   # p=0, q=0, XOR=0
    [0, 1, 1],   # p=0, q=1, XOR=1
    [1, 0, 1],   # p=1, q=0, XOR=1
    [1, 1, 0]    # p=1, q=1, XOR=0
]
```

> ⚠️ XOR은 직선 하나로 분리가 불가능한 **비선형 문제**  
> → 선형 모델(로지스틱 회귀)로는 완벽하게 풀 수 없음

---

## 2. 라이브러리 임포트

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression   # 선형 분류 모델
from sklearn import svm, metrics
```

---

## 3. Feature / Label 분리

```python
x_df = pd.DataFrame(x_data)

# iloc[:, 0:2] → 0열(p), 1열(q) = feature
# iloc[:, 2]   → 2열(XOR 결과) = label
feature = np.array(x_df.iloc[:, 0:2])
label   = np.array(x_df.iloc[:, 2])

print(feature)
# [[0 0]
#  [0 1]
#  [1 0]
#  [1 1]]

print(label)
# [0 1 1 0]
```

---

## 4. 모델 생성 및 학습

```python
lmodel = LogisticRegression()  # 선형 분류 → XOR 같은 비선형 문제에 취약
smodel = svm.SVC()             # 커널 트릭으로 비선형 문제도 처리 가능
#   SVC 기본 kernel='rbf' (방사형) → 비선형 분리 가능

lmodel.fit(feature, label)
smodel.fit(feature, label)
```

---

## 5. 예측 및 정확도 비교

```python
pred1 = lmodel.predict(feature)
print('lmodel 예측값 : ', pred1)   # [0 0 0 0] ← XOR 패턴을 못 잡음
pred2 = smodel.predict(feature)
print('smodel 예측값 : ', pred2)   # [0 1 1 0] ← 정확하게 분류

acc1 = metrics.accuracy_score(label, pred1)
print('lmodel 정확도 : ', acc1)    # 0.75

acc2 = metrics.accuracy_score(label, pred2)
print('smodel 정확도 : ', acc2)    # 1.0
```

### 결과 비교

|모델|예측값|정확도|이유|
|---|---|---|---|
|LogisticRegression|`[0 0 0 0]`|0.75|선형 경계만 그을 수 있어 XOR 패턴 학습 불가|
|SVM (SVC)|`[0 1 1 0]`|1.0|커널 트릭으로 고차원 변환 → 비선형 분리 가능|

---

## 전체 흐름 요약

```
XOR 진리표 데이터 준비
    ↓
feature(p, q) / label(XOR 결과) 분리
    ↓
로지스틱 회귀 vs SVM 동시 학습
    ↓
예측값 비교
    ↓
정확도 비교 → SVM 완승 (1.0 vs 0.75)
```

---

## 핵심 개념 정리

### 왜 로지스틱 회귀는 XOR을 못 풀까?

XOR은 직선 하나로 두 클래스를 나눌 수 없는 비선형 문제다.  
로지스틱 회귀는 직선(선형 경계)만 그을 수 있어서 XOR 패턴을 학습하지 못한다.

### 왜 SVM은 XOR을 풀 수 있을까?

SVM의 기본 커널인 `rbf`(방사형)가 데이터를 고차원으로 변환(커널 트릭)하여  
고차원 공간에서 선형으로 분리 가능하게 만들기 때문이다.

### AND / OR 로 바꿔서 실습하려면?

```python
# AND 진리표
x_data = [[0,0,0], [0,1,0], [1,0,0], [1,1,1]]

# OR 진리표
x_data = [[0,0,0], [0,1,1], [1,0,1], [1,1,1]]
```

AND, OR은 선형 분리가 가능해서 로지스틱 회귀도 정확도 1.0이 나온다.

---
# 📄 ex36svm_iris.py — SVM 실습 (Iris 다중 분류)

> **알고리즘**: SVM (kernel='rbf')  
> **데이터셋**: Iris (꽃잎 길이/너비 기준 3종 분류)  
> **평가**: 정확도 확인 3가지 방법 + 모델 저장/불러오기

---

## 1. 라이브러리 임포트 및 데이터 로드

```python
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# LogisticRegression: 다중 클래스를 지원하도록 일반화된 모델
# → Softmax Regression = Multinomial Logistic Regression 이라고도 부름

iris = datasets.load_iris()
print(iris.keys())
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', ...]

print(iris.target)   # 0: Setosa, 1: Versicolor, 2: Virginica
print(iris.data[:3])
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]]
```

---

## 2. Feature 선택 및 상관관계 확인

```python
# 꽃잎 길이(2열), 꽃잎 너비(3열)만 사용
x = iris.data[:, [2, 3]]
y = iris.target
print(x.shape, y.shape)   # (150, 2) (150,)

# 두 feature 간의 상관계수 확인
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])[0, 1])
# 0.9628... → 두 feature가 매우 강한 양의 상관관계
```

---

## 3. Train / Test Split

```python
# 7:3 비율로 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0
)
print(x_train.shape, x_test.shape)   # (105, 2) (45, 2)
```

---

## 4. 스케일링 (주석 처리 — 참고용)

```python
"""
# Iris는 feature 간 크기 차이가 거의 없어 표준화 효과 미미
# 하지만 SVM은 거리 기반이므로 일반적으로 스케일링 권장

sc = StandardScaler()
sc.fit(x_train)          # ⚠️ fit은 반드시 train 데이터로만!
# sc.fit(x_test)         # ← 이렇게 하면 안 됨 (train 기준 통계 덮어씌워짐)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)   # test는 transform만

# 원복
ori_x_train = sc.inverse_transform(x_train)
"""
```

> ⚠️ **스케일링 주의**: `fit()`은 반드시 `x_train`으로만, `x_test`는 `transform()`만 적용

---

## 5. SVM 모델 학습

```python
from sklearn import svm

# kernel='rbf' : 방사형 커널 (비선형 분리 가능, 실무에서 가장 많이 사용)
# C=1.0        : 마진 조절 (클수록 하드마진, 작을수록 소프트마진)
model = svm.SVC(kernel='rbf', C=1.0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print(f'총 갯수:{len(y_test)}, 오류수:{(y_test != y_pred).sum()}')
# 총 갯수:45, 오류수:1
```

---

## 6. 정확도 확인 3가지 방법

```python
# 방법 1 — accuracy_score
print(accuracy_score(y_test, y_pred))   # 0.9777...

# 방법 2 — 혼동행렬 (Confusion Matrix)
# crosstab: 행=실제값, 열=예측값으로 교차표 생성
con_mat = pd.crosstab(y_test, y_pred, rownames=['실제값'], colnames=['예측값'])
print(con_mat)
# 예측값   0   1   2
# 실제값
# 0       16   0   0
# 1        0  17   1   ← 1을 2로 오분류 1건
# 2        0   1  11   ← 2를 1로 오분류 1건 (같은 1건)

# 대각선 합 / 전체 = 정확도
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))
# 0.9777...

# 방법 3 — model.score()
print('test score : ', model.score(x_test, y_test))     # 0.9777...
print('train score : ', model.score(x_train, y_train))  # 0.9809...
# ※ test score와 train score 차이가 크면 과적합(Overfitting) 의심
```

---

## 7. 모델 저장 및 불러오기

```python
import joblib   # pickle보다 빠르고 대용량 지원

joblib.dump(model, 'svm_model.pkl')   # 모델 저장 (.pkl, .sav, .model 등 가능)
del model                              # 메모리에서 모델 삭제

read_model = joblib.load('svm_model.pkl')   # 모델 불러오기
# 이후 read_model로 예측 진행
```

---

## 8. 새로운 데이터 예측

```python
new_data = np.array([[5.5, 2.2], [0.6, 0.3], [1.1, 0.5]])
# ⚠️ 표준화된 모델이었다면 new_data도 동일하게 transform 필요

new_pred = read_model.predict(new_data)
print('예측 결과 : ', new_pred)   # [2 0 0]
# → 내부적으로 Softmax 확률값 중 가장 큰 클래스 인덱스를 반환

# predict_proba는 probability=True 옵션 필요
# print(read_model.predict_proba(new_data))
```

---

## 9. 결정 경계 시각화

```python
import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib.colors import ListedColormap

def plot_decision_regionFunc(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')
    colors  = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap    = ListedColormap(colors[:len(np.unique(y))])

    # 격자 좌표 생성
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # ⚠️ 원본은 오타 (0→1)
    xx, yy = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    # 격자 전체 예측 → 배경 색상으로 결정 경계 표현
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)   # 배경 색상 채우기
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
                    color=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

# train + test 합쳐서 시각화
x_combined = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regionFunc(
    X=x_combined, y=y_combined,
    classifier=read_model,
    test_idx=range(105, 150),
    title='scikit-learn 제공'
)
```

![[ex36svm_iris.png]]

---

## 전체 흐름 요약

```
Iris 데이터 로드
    ↓
feature 선택 (꽃잎 길이, 너비) + 상관관계 확인
    ↓
Train / Test Split (7:3)
    ↓
SVM 학습 (kernel='rbf', C=1.0)
    ↓
정확도 확인 3가지 (accuracy_score / 혼동행렬 / model.score)
    ↓
모델 저장 (joblib) → 불러오기 → 새 데이터 예측
    ↓
결정 경계 시각화
```

---

## 주요 개념 정리

### 혼동행렬 (Confusion Matrix)

실제값과 예측값을 교차표로 나타낸 것.  
대각선 = 정분류, 나머지 = 오분류. 대각선 합 / 전체 = 정확도.

### joblib vs pickle

`joblib`은 numpy 배열처럼 대용량 데이터를 포함한 객체를 더 빠르게 저장/불러오기 가능.  
머신러닝 모델 저장에는 `joblib`을 권장.

### train score vs test score

두 값의 차이가 크면 **과적합(Overfitting)** 의심.  
train은 높고 test가 낮으면 학습 데이터에만 특화된 모델.

### ⚠️ 코드 오타 (원본)

```python
# 잘못됨 — x2 범위가 x1과 동일하게 설정됨
x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1

# 올바름
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
```

---
# 📄 ex37svm_bmi.py — SVM 실습 (BMI 비만도 분류)

> **알고리즘**: SVM (kernel='rbf')  
> **데이터셋**: bmi.csv (무작위 생성 50,000건)  
> **목표**: 키와 몸무게로 비만도 3단계 분류 (thin / normal / fat)

---

## BMI란?

**BMI(Body Mass Index)** = 체질량지수  
키와 몸무게로 체지방량을 추정하여 비만도를 간편하게 측정하는 지표

```
공식: 체중(kg) / 키(m)²
예)  키 170cm, 몸무게 68kg → 68 / (1.7)² = 23.5 → normal
```

|BMI 범위|분류|label|
|---|---|---|
|18.5 미만|저체중|thin|
|18.5 ~ 25.0|정상|normal|
|25.0 이상|과체중/비만|fat|

---

## 1. BMI 데이터 생성 (주석 처리된 전처리 코드)

```python
import random
random.seed(12)

def cald_bmiFunc(h, w):
    bmi = w / (h / 100) ** 2
    if bmi < 18.5: return 'thin'
    if bmi < 25.0: return 'normal'
    return 'fat'

# bmi.csv 파일 생성
fp = open('bmi.csv', mode='w')
fp.write('height,weight,label\n')   # 헤더

cnt = {'thin':0, 'normal':0, 'fat':0}

for i in range(50000):
    h = random.randint(150, 200)    # 키: 150~200cm 무작위
    w = random.randint(35, 200)     # 몸무게: 35~200kg 무작위
    label = cald_bmiFunc(h, w)
    cnt[label] += 1
    fp.write('{0},{1},{2}\n'.format(h, w, label))

fp.close()
```

> 💡 이 코드는 최초 1회만 실행하면 됨. 이후엔 bmi.csv 파일을 불러서 사용

---

## 2. 라이브러리 임포트 및 데이터 로드

```python
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('bmi.csv')
print(df.head(2), df.shape)   # (50000, 3)
print(df.info())
```

---

## 3. Feature 전처리 — 정규화

```python
label = df['label']

# 정규화: 값의 범위를 0~1 사이로 줄여줌
# SVM은 거리 기반이므로 feature 간 스케일 차이가 크면 성능 저하
# weight: 35~200 → /100 으로 0.35~2.0 범위로 축소
# height: 150~200 → /200 으로 0.75~1.0 범위로 축소
w = df['weight'] / 100
h = df['height'] / 200

# 두 컬럼 합치기 (axis=1: 열 방향으로 합치기)
wh = pd.concat([w, h], axis=1)
print(wh.head(2))

# label 수치화 (문자 → 숫자)
label = label.map({'thin':0, 'normal':1, 'fat':2})
print(label[:2])
```

> ⚠️ **정규화 주의**: SVM은 거리 기반 알고리즘이므로 feature 간 스케일이 다르면  
> 반드시 정규화 또는 표준화(`StandardScaler`) 처리 필요

---

## 4. Train / Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    wh, label, test_size=0.3, random_state=1
)
print(x_train.shape, x_test.shape)   # (35000, 2) (15000, 2)
```

---

## 5. SVM 모델 학습

```python
# kernel='rbf': 방사형 커널 → 비선형 분리 가능 (실무에서 가장 많이 사용)
# C=0.01: 소프트 마진 (이상치에 유연)
model = svm.SVC(C=0.01, kernel='rbf').fit(x_train, y_train)

pred = model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10].values)
# 예측값 :  [2 1 2 2 2 2 0 1 2 2]
# 실제값 :  [2 1 1 2 2 2 0 1 2 2]

sc_score = metrics.accuracy_score(y_test, pred)
print('sc_score : ', sc_score)   # 0.9549333333333333
```

---

## 6. 교차 검증 (Cross Validation)

```python
from sklearn import model_selection

# cv=3: 데이터를 3등분하여 번갈아가며 검증 → 과적합 여부 확인
# 3회 각각 다른 데이터로 검증해서 평균 정확도가 일정하면 안정적인 모델
cross_vali = model_selection.cross_val_score(model, wh, label, cv=3)
print('3회 각 정확도 : ', cross_vali)
print('평균 정확도 : ', cross_vali.mean())
```

---

## 7. 새로운 데이터 예측

```python
new_data = pd.DataFrame({'weight':[66, 88], 'height':[188, 160]})

# ⚠️ 학습 때 정규화했으면 새 데이터도 동일하게 정규화 필수!
new_data['weight'] = new_data['weight'] / 100
new_data['height'] = new_data['height'] / 200

new_pred = model.predict(new_data)
print('새로운 값 예측 결과 : ', new_pred)
# 0: thin, 1: normal, 2: fat
```

---

## 8. 시각화

```python
# AHD를 인덱스로 설정 → df2.loc['fat'] 처럼 라벨로 바로 필터링 가능
df2 = pd.read_csv('bmi.csv', index_col=2)

def scatterFunc(lbl, color):
    b = df2.loc[lbl]                              # 해당 라벨 행만 추출
    plt.scatter(b['weight'], b['height'], c=color, label=lbl)

scatterFunc('fat',    'red')
scatterFunc('normal', 'yellow')
scatterFunc('thin',   'blue')
plt.xlabel('몸무게 (weight)')
plt.ylabel('키 (height)')
plt.title('BMI 비만도 분류 시각화')
plt.legend()
plt.show()
```

![[ex37svm_bmi.png]]

---

## 전체 흐름 요약

```
BMI 공식으로 50,000건 데이터 생성 → bmi.csv 저장
    ↓
데이터 로드 → feature 정규화 (weight/100, height/200)
    ↓
label 수치화 (thin:0, normal:1, fat:2)
    ↓
Train / Test Split (7:3)
    ↓
SVM 학습 (kernel='rbf', C=0.01)
    ↓
정확도 확인 (0.9549) + 교차 검증
    ↓
새로운 데이터 예측
    ↓
시각화 (index_col=2로 label 인덱스 활용)
```

---

## 주요 개념 정리

### 정규화 vs 표준화

정규화는 값의 범위를 0~1로 줄이는 것 (직접 나누기).  
표준화는 평균 0, 표준편차 1로 변환하는 것 (`StandardScaler`).  
SVM처럼 거리 기반 알고리즘은 반드시 둘 중 하나 적용 필요.

### 교차 검증 (Cross Validation)

데이터를 cv등분하여 번갈아가며 학습/검증을 반복.  
한 번의 train/test split보다 신뢰도 높은 성능 측정 가능.  
3회 정확도가 일정하면 안정적인 모델, 들쑥날쑥하면 과적합 의심.

### index_col=2 활용

label 컬럼을 인덱스로 설정하면 `df.loc['fat']`처럼  
라벨명으로 바로 필터링할 수 있어 시각화 코드가 간결해짐.


---
# 📄 ex37quiz.py — SVM 실습 (심장병 환자 분류)

> **데이터셋**: Heart.csv (흉부외과 환자 303명)  
> **목표**: 심장병(AHD) 여부 이진 분류 (No: 0, Yes: 1)  
> **특이사항**: 문자 컬럼 제외 + 표준화 적용

---

## 1. 데이터 로드 및 전처리

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, metrics

# 데이터 로드
df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Heart.csv", index_col=0)

# 결측값 제거 (Ca 4개, Thal 2개) → 303행 → 297행
df = df.dropna()

# 문자 컬럼 제외 (Feature Selection), label 분리 + Dummy 처리
feature = df.drop(['ChestPain', 'Thal', 'AHD'], axis=1)
label = df['AHD'].map({'No':0, 'Yes':1})
```

> 💡 **특성공학 기법 적용**  
> Feature Selection — 문자 컬럼(ChestPain, Thal) 제거  
> Dummy — AHD: No/Yes → 0/1 변환

---

## 2. Train / Test Split + 표준화

```python
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=0)

# 표준화 — fit은 train으로만, test는 transform만
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```

> ⚠️ `fit_transform` vs `transform`  
> `fit_transform` — 평균/표준편차 계산 + 변환 동시에 (train만)  
> `transform` — 이미 계산된 기준으로 변환만 (test에 적용)

---

## 3. 모델 학습 및 평가

```python
# C=0.01: 소프트 마진, kernel='rbf': 방사형 커널
svc_model = svm.SVC(C=0.01, kernel='rbf')
svc_model.fit(x_train, y_train)

pred = svc_model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10].values)
print('sc_score : ', metrics.accuracy_score(y_test, pred))  # 0.7777...
```

> 💡 **C 값에 따른 성능 차이**  
> C=0.01 → 정확도 0.53 (소프트마진 과하게 적용)  
> 표준화 후 C=0.01 → 정확도 0.77 (표준화 효과)  
> C=1.0 이상으로 올리면 정확도 더 향상 가능

---

## 4. 교차 검증

```python
from sklearn import model_selection

# cv=3: 데이터를 3등분하여 번갈아가며 검증
# 3회 정확도가 일정하면 안정적인 모델, 들쑥날쑥하면 과적합 의심
cross_vali = model_selection.cross_val_score(svc_model, feature, label, cv=3)
print('3회 각 정확도 : ', cross_vali)
print('평균 정확도 : ', cross_vali.mean())
```

> ⚠️ 교차검증 시 정규화 안 된 `feature`를 넣으면 정확도가 낮게 나옴  
> 정확한 교차검증을 하려면 `Pipeline`으로 StandardScaler와 모델을 묶어야 함

---

## 5. 새로운 환자 데이터 예측

```python
new_data = pd.DataFrame({
    'Age'    : [63,  45,  55],
    'Sex'    : [1,   0,   1],       # 1: 남성, 0: 여성
    'RestBP' : [145, 130, 160],
    'Chol'   : [233, 204, 286],
    'Fbs'    : [1,   0,   0],       # 1: 공복혈당 > 120mg/dl
    'RestECG': [2,   0,   2],       # 0: 정상, 1: 이상, 2: 비대
    'MaxHR'  : [150, 172, 108],
    'ExAng'  : [0,   0,   1],       # 1: 운동 시 협심증 있음
    'Oldpeak': [2.3, 1.4, 1.5],
    'Slope'  : [3,   1,   2],
    'Ca'     : [0,   0,   3]        # 형광투시로 확인된 혈관 수
})

new_pred = svc_model.predict(new_data)
for i, result in enumerate(new_pred):
    print(f'환자 {i+1} : {"심장병 있음" if result == 1 else "심장병 없음"}')
```

> ⚠️ 학습 시 표준화를 적용했다면 new_data도 동일하게 변환 필요  
> `new_data = sc.transform(new_data)`

---

## 전체 흐름 요약

```
데이터 로드 (303행)
    ↓
결측값 제거 → 297행
    ↓
Feature Selection (문자 컬럼 제거) + Dummy (AHD 0/1 변환)
    ↓
Train / Test Split (7:3)
    ↓
표준화 (StandardScaler)
    ↓
SVM 학습 (kernel='rbf', C=0.01)
    ↓
정확도 확인 + 교차 검증
    ↓
새로운 환자 데이터 예측
```

---

## 주요 개념 정리

### 특성공학 기법 적용 내역

|기법|내용|
|---|---|
|Feature Selection|ChestPain, Thal 제거|
|Dummy|AHD → No:0, Yes:1|
|Scaling (표준화)|StandardScaler 적용|

### C 파라미터

- C 크면 → 하드 마진 (이상치에 엄격, 과적합 위험)
- C 작으면 → 소프트 마진 (이상치에 유연, 과소적합 주의)
- 현재 C=0.01로 너무 작음 → C=1.0 이상 권장

### 교차 검증 (Cross Validation)

데이터를 cv등분하여 번갈아가며 학습/검증 반복.  
한 번의 train/test split보다 신뢰도 높은 성능 측정 가능.  
단, 표준화된 데이터로 돌려야 의미있는 결과가 나옴.



특성공학기법 - 좋은 성능을 내기 위해 입력 자료를 변형하거나 가공하는 방법
- 차원 축소
	1) feature selection : 변수 선택
	2) feature extraction : 차원 축소(방법: 주성분분석(PCA))
- Scaling (정규화 표준화)
- Transform (변형)
	1) Binning(비닝) : 연속적 자료를 구간으로 분류(연속형 -> 범주형)
	2) Dummy : 범주형을 연속형으로 변환
- feature creation : 특성 생성 - 기존 자료로 의미있는 새로운 변수 생성(예: 날짜로 년,월,일,요일,분기 등의 변수 생성, 연봉으로 보너스 변수 생성 ...)

---
# 📖 특성공학 & 주성분분석 (PCA) 개념 정리

---

## 1. 특성공학 (Feature Engineering)

> 좋은 성능을 내기 위해 입력 자료를 변형하거나 가공하는 방법

### 1-1. 차원 축소

feature가 너무 많으면 오히려 성능이 떨어짐 → 줄여야 함

- **Feature Selection (변수 선택)** — 중요한 변수만 골라서 나머지 버리기  
    예) Heart 데이터에서 문자 컬럼(ChestPain, Thal) 제거
    
- **Feature Extraction (차원 축소)** — 여러 변수를 조합해 더 적은 수의 새 변수로 압축  
    대표적인 방법: **PCA (주성분분석)**  
    예) 국어, 영어, 수학, 과학 4개 → 문과성향, 이과성향 2개로 압축
    

### 1-2. Scaling

SVM처럼 거리 기반 알고리즘은 feature 간 스케일이 다르면 성능 저하

- **정규화** — 0~1 사이로 축소 (BMI 실습에서 weight/100, height/200 한 것)
- **표준화** — 평균 0, 표준편차 1로 변환 (Heart 실습에서 StandardScaler 쓴 것)

### 1-3. Transform (변형)

- **Binning (비닝)** — 연속형 → 범주형  
    BMI 값(연속) → thin / normal / fat (범주)  
    나이(연속) → 청년 / 중년 / 장년 / 노년 (범주)
    
- **Dummy (원핫인코딩)** — 범주형 → 연속형(0/1)  
    AHD: No/Yes → 0/1 (Heart 실습에서 한 것)  
    ChestPain: typical/asymptomatic/… → 각각 0 또는 1로 변환
    

### 1-4. Feature Creation (특성 생성)

기존 자료로 의미있는 새로운 변수 생성

- 날짜 → 년, 월, 일, 요일, 분기, 공휴일 여부
- 연봉 → 월급, 보너스, 세후금액
- MaxHR ÷ Age → 심장건강지수

### 실습에서 적용한 기법

|기법|내용|
|---|---|
|Feature Selection|ChestPain, Thal 제거|
|Dummy|AHD → No:0, Yes:1|
|Scaling (표준화)|StandardScaler 적용|
|Binning|BMI → thin/normal/fat|

---

## 2. 주성분분석 (PCA, Principal Component Analysis)

> 여러 feature 가운데 **대표 특성**을 찾아 분석하는 방식  
> 자료의 차원을 고차원에서 저차원으로 축소하는 **차원축소 기법**

### 2-1. 핵심 아이디어

2차원 데이터를 1차원으로 줄이려면 어느 방향으로 정사영(projection)할지 결정해야 한다.  
이때 기준이 되는 것이 **분산(Variance)** 이다.

> 💡 분산이 가장 큰 방향 = 데이터의 변화가 가장 뚜렷한 방향 = 첫 번째 주성분

### 2-2. 주성분을 찾는 과정

```
① 분산이 가장 큰 방향을 찾는다
        ↓
   → 이것이 첫 번째 주성분 (C1)
        ↓
② 첫 번째 주성분과 직교(90°)하는 방향 중
   분산이 가장 큰 방향을 찾는다
        ↓
   → 이것이 두 번째 주성분 (C2)
        ↓
③ 필요한 차원 수만큼 반복
```

### 2-3. 직교(Orthogonality)란?

두 선이 90°를 이루며 교차하는 상태.  
두 주성분이 직교한다는 것은 서로 **가장 독립적인 상태** 임을 의미한다.  
(내적(Inner Product) = 0)

- 한 방향이 우상향이면 → 직교 방향은 우하향
- 두 주성분이 겹치는 정보 없이 서로 다른 특성을 설명

### 2-4. 차원축소의 3가지 순기능

**① 이해가 쉬워짐**  
차원이 낮아질수록 데이터 구조 파악이 용이  
공간 → 면 → 선 → 점 순으로 이해하기 쉬워짐

**② 연산속도 향상**  
분산값을 유지하면서 데이터 크기 자체를 줄이기 때문에  
데이터 특성을 훼손하지 않고 빠른 연산 가능

**③ 차원의 저주 해결**  
차원이 높아질수록 같은 데이터 수로는 공간을 채우기 어려워짐  
→ 데이터 밀도가 낮아짐 → 과적합(Overfitting) 발생  
→ 차원축소로 이 문제를 예방

### 2-5. 차원의 저주란?

차원이 증가할수록 필요한 데이터 양이 기하급수적으로 늘어나는 현상.  
데이터 증량 없이 차원만 높아지면 → 데이터 밀도 감소 → 과적합 발생.  
차원축소는 데이터가 부족한 상태에서 **과적합을 예방하는 전처리 기법**이 된다.

---

## 핵심 요약

| 용어                 | 설명                              |
| ------------------ | ------------------------------- |
| 특성공학               | 성능 향상을 위해 입력 자료를 변형/가공하는 방법     |
| Feature Selection  | 중요한 변수만 선택해서 나머지 제거             |
| Feature Extraction | 여러 변수를 조합해 새 변수로 압축 (PCA)       |
| PCA                | 분산이 가장 큰 방향을 주성분으로 선택하는 차원축소 기법 |
| 직교                 | 두 주성분이 90°를 이루며 서로 독립적인 상태      |
| 차원의 저주             | 차원 증가 시 필요 데이터가 기하급수적으로 늘어나는 현상 |

---

# 📄 ex38pca.py — PCA 실습 (Iris 차원 축소)

> **알고리즘**: PCA (Principal Component Analysis, 주성분분석)  
> **데이터셋**: Iris (꽃받침 길이/너비 2개 feature)  
> **목표**: 고차원 데이터를 저차원으로 축소하면서 데이터 특성 최대한 보존

---

## PCA란?

선형대수 관점에서, 입력 데이터의 **공분산 행렬을 고윳값 분해**하고  
구한 **고유벡터에 입력 데이터를 선형변환**하는 것.

이 고유벡터가 PCA의 **주성분 벡터**로서 입력 데이터의 **분산이 큰 방향**을 나타낸다.  
→ 입력 데이터의 성질을 최대한 유지한 상태로 고차원을 저차원으로 변환하는 기법

---

## 1. 라이브러리 임포트 및 데이터 로드

```python
from sklearn.decomposition import PCA   # PCA 모델
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
```

---

## 2. 데이터 준비

```python
n = 10
# iris.data 컬럼 구조: [꽃받침길이, 꽃받침너비, 꽃잎길이, 꽃잎너비]
# [:n, :2] → 앞 10개 샘플, 앞 2개 feature(꽃받침 길이/너비)만 선택
x = iris.data[:n, :2]
print('차원 축소 전 : x: ', x, x.shape, type(x))
# shape: (10, 2) → 10개 샘플, 2개 feature

# x.T → 전치행렬 (행↔열 뒤집기)
# (10, 2) → (2, 10): 특성별로 묶어서 보기 위함
print(x.T)
# [[5.1 4.9 4.7 ... ]  ← 꽃받침 길이 10개
#  [3.5 3.0 3.2 ... ]] ← 꽃받침 너비 10개
```

---

## 3. 시각화

```python
# x.T로 전치해서 plot → 각 열(feature)이 x축 기준점이 됨
# 'o:' → 점(o) + 점선(:) 스타일
plt.plot(x.T, 'o:')

# x축 눈금을 숫자 대신 feature 이름으로 표시
plt.xticks(range(2), ['꽃받침길이', '꽃받침너비'])
plt.grid(True)
plt.title('아이리스 크기 특성')
plt.xlabel('특성의 종류')
plt.ylabel('특성값')
plt.xlim(-0.5, 2)
plt.ylim(2.5, 6)

# 표본 1 ~ 표본 10 범례 자동 생성
plt.legend(['표본 {}'.format(i + 1) for i in range(n)])
plt.show()
```

![[ex38pca.png]]

> 💡 그래프 해석  
> 각 선 = 하나의 샘플(표본)  
> 왼쪽 점 = 꽃받침 길이, 오른쪽 점 = 꽃받침 너비  
> 모든 샘플이 우하향 → 꽃받침 길이가 너비보다 전반적으로 큰 경향

---

## 전체 흐름 요약

```
Iris 데이터 로드
    ↓
앞 10개 샘플, 꽃받침 길이/너비 2개 feature 선택
    ↓
전치행렬(x.T)로 feature별로 묶어서 시각화
    ↓
각 샘플의 두 feature 값 변화 패턴 확인
```

---

## 주요 개념 정리

### 공분산 행렬

두 변수가 함께 변하는 정도를 나타낸 행렬.  
PCA는 이 행렬을 분해해서 데이터가 가장 많이 퍼져있는 방향(주성분)을 찾는다.

### 고유벡터 / 고윳값

고유벡터 = 데이터가 퍼져있는 방향 (주성분의 방향)  
고윳값 = 해당 방향으로 데이터가 얼마나 퍼져있는지 (분산의 크기)  
→ 고윳값이 클수록 중요한 주성분

### 전치행렬 (x.T)

행과 열을 뒤집은 행렬.  
`(10, 2)` → `(2, 10)`: 샘플별이 아닌 **feature별**로 묶어서 볼 때 사용  
`plt.plot(x.T)`에서 각 feature가 하나의 x축 포인트가 됨

### n_components

PCA에서 축소할 차원 수 지정  
예) `PCA(n_components=2)` → 2개의 주성분으로 압축