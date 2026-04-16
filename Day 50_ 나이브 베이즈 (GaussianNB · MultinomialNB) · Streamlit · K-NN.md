# Day 50_나이브 베이즈 (GaussianNB · MultinomialNB) · Streamlit · K-NN
## 📅 2026-04-16

---
# 📄 ex40naive.py — 나이브 베이즈 분류 (weather.csv)

---

## 개념 정리

### GaussianNB의 feature 중요도 분석

> `GaussianNB`는 feature importance를 직접 제공하지 않지만, **클래스별 평균(`theta_`)의 차이**로 중요도를 간접 추정할 수 있다.

|속성|설명|
|---|---|
|`model.theta_`|각 클래스별 feature 평균값 (shape: [클래스 수, feature 수])|
|`model.theta_[0]`|RainTomorrow=0 (비 안 오는 날)의 feature 평균|
|`model.theta_[1]`|RainTomorrow=1 (비 오는 날)의 feature 평균|
|`np.abs(mean_1 - mean_0)`|두 클래스 간 평균 차이 → 클수록 중요한 feature|

### 교차 검증 (Cross Validation)

> 데이터를 `cv`개의 fold로 나눠 번갈아가며 검증셋으로 사용하는 기법 → **과적합 방지 + 일반화 성능 측정**

```
전체 데이터 (366개)
├── fold 1: [검증] [train] [train] [train] [train]
├── fold 2: [train] [검증] [train] [train] [train]
├── ...
└── fold 5: [train] [train] [train] [train] [검증]
→ 5번의 accuracy 평균 = 최종 성능 지표
```

---

## 1단계 : 데이터 로드 및 확인

```python
import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv")
print(df.head(2))
#          Date  MinTemp  MaxTemp  Rainfall  Sunshine  WindSpeed  Humidity  Pressure  Cloud  Temp RainToday RainTomorrow
# 0  2016-11-01      8.0     24.3       0.0       6.3         20        29    1015.0      7  23.6        No          Yes

print(df.info())
# Sunshine : 363 non-null → 결측값 3개 존재
# RainToday, RainTomorrow : object → 인코딩 필요
```

---

## 2단계 : 전처리

```python
# Date 컬럼 제거 (학습에 불필요)
df = df.drop('Date', axis=1)

# 범주형 → 수치형 변환
df['RainToday']    = df['RainToday'].map({'Yes': 1, 'No': 0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

# 결측값 처리 (Sunshine 3개 → 평균값으로 대체)
df['Sunshine'] = df['Sunshine'].fillna(df['Sunshine'].mean())
```

---

## 3단계 : feature / label 분리 및 데이터 분할

```python
x = df.drop('RainTomorrow', axis=1)   # feature (10개)
y = df['RainTomorrow']                # label

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
    # stratify=y : 클래스 비율을 유지하며 분리 (불균형 데이터 완화)
)
```

---

## 4단계 : GaussianNB 모델 학습 및 평가

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = GaussianNB()
model.fit(x_train, y_train)

pred = model.predict(x_test)
print('분류 정확도 : ', accuracy_score(y_test, pred))   # 0.8783783783783784
print('confusion_matrix : \n', confusion_matrix(y_test, pred))
# [[55  6]
#  [ 3 10]]
```

### confusion_matrix 해석

| |예측: 비 안 옴(0)|예측: 비 옴(1)|
|---|:-:|:-:|
|**실제: 비 안 옴(0)**|55 (TN)|6 (FP)|
|**실제: 비 옴(1)**|3 (FN)|10 (TP)|

---

## 5단계 : 교차 검증

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, x, y, cv=5)
print(f'각 fold : {scores}')
print(f'평균    : {scores.mean()}')
# 각 fold : [0.72972973 0.82191781 0.79452055 0.8630137  0.83561644]
# 평균    : 0.8089596445760829
```

---

## 6단계 : feature 중요도 분석

```python
# theta_ : GaussianNB의 클래스별 feature 평균
mean_0 = model.theta_[0]   # 비 안 오는 날 평균
mean_1 = model.theta_[1]   # 비 오는 날 평균

# 두 클래스 간 평균 차이 → 클수록 중요한 feature
importance = np.abs(mean_1 - mean_0)

feat_impo = pd.DataFrame({
    'feature'   : x.columns,
    'importance': importance
}).sort_values(by='importance', ascending=False)

print(feat_impo)
#      feature  importance
# 5   Humidity   15.756059   ← 가장 중요
# 6   Pressure    6.070088
# 3   Sunshine    3.698378
# 0    MinTemp    3.448954
# 7      Cloud    2.623589
# 2   Rainfall    1.151417
# 1    MaxTemp    0.745820
# 8       Temp    0.296384
# 9  RainToday    0.157575
# 4  WindSpeed    0.094734   ← 가장 덜 중요
```

---

## 7단계 : feature 중요도 시각화

```python
import matplotlib.pyplot as plt
import koreanize_matplotlib

plt.figure()
plt.bar(feat_impo['feature'], feat_impo['importance'])
plt.xlabel('feature')
plt.ylabel('중요도(평균)')
plt.xticks(rotation=45)   # x축 레이블 45도 기울임 (겹침 방지)
plt.tight_layout()        # 레이블 잘림 방지 (xticks rotation 사용 시 세트로 사용)
plt.show()
```

![[ex40naive.png]]

---

## 8단계 : 새로운 데이터로 예측

```python
newdata = pd.DataFrame([{
    'MinTemp'  : 12.3,
    'MaxTemp'  : 27.0,
    'Rainfall' : 0.0,
    'Sunshine' : 10.0,
    'WindSpeed': 8.0,
    'Humidity' : 40,
    'Pressure' : 1005.0,
    'Cloud'    : 1,
    'Temp'     : 20.0,
    'RainToday': 0
}])

newpred = model.predict(newdata)
print('예측 결과 : ', '비옴' if newpred == 1 else '비안옴')   # 비안옴
print('확률은 ', model.predict_proba(newdata))
# [[0.983... 0.016...]]  → 비 안 올 확률 98.3%, 비 올 확률 1.7%
```

---

## 핵심 정리

### GaussianNB 주요 속성

|속성/메서드|설명|
|---|---|
|`model.theta_`|클래스별 각 feature의 평균 (shape: [n_classes, n_features])|
|`model.var_`|클래스별 각 feature의 분산|
|`model.class_prior_`|각 클래스의 사전 확률|
|`model.predict_proba()`|각 클래스에 속할 확률 반환|

### GaussianNB vs MultinomialNB vs BernoulliNB

|종류|적합한 데이터|주요 용도|
|---|---|---|
|`GaussianNB`|연속형 (정규분포 가정)|날씨, 키/몸무게 등|
|`MultinomialNB`|정수형 빈도 데이터|텍스트 분류, 단어 빈도|
|`BernoulliNB`|이진형 (0 or 1)|스팸 분류|

### 주의사항

|상황|올바른 처리|
|---|---|
|결측값 존재 (`Sunshine` 3개)|`fillna(mean())` 으로 대체 후 학습|
|`label` 분리 후 map 적용|`label = df['class'].map(...)` (Series에 바로 적용)|
|불균형 클래스 분리|`stratify=y` 옵션으로 비율 유지|

---
# 📄 ex41naive_iris.py — GaussianNB 분류 (iris)

---

## 개념 정리

### 결정 경계 (Decision Region)

> 분류 모델이 각 클래스를 어떻게 나누는지 2D 평면에 시각화한 것 `contourf()` 로 배경 색을 칠해 경계를 표현함

### 모델 저장 / 불러오기 (joblib)

> 학습 완료된 모델을 파일로 저장해 재사용하는 방법 `pickle` 보다 빠르고 대용량(numpy 배열 포함) 지원

|함수|설명|
|---|---|
|`joblib.dump(model, 'file.pkl')`|모델 저장|
|`joblib.load('file.pkl')`|모델 불러오기|

### train score vs test score

> 두 값의 차이가 크면 **과적합(overfitting)** 의심

| |의미|
|---|---|
|`model.score(x_train, y_train)`|학습 데이터 정확도|
|`model.score(x_test, y_test)`|검증 데이터 정확도|

---

## 1단계 : 데이터 로드 및 feature 선택

```python
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

# 꽃잎 길이(2열), 꽃잎 너비(3열) 2개 feature만 사용
# → 시각화(2D)를 위해 2개만 선택
x = iris.data[:, [2, 3]]
y = iris.target

# 꽃잎 길이와 너비는 상관관계가 매우 높음
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])[0, 1])   # 0.9628654314027961
```

---

## 2단계 : 데이터 분할

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape)   # (105, 2) (45, 2)
```

> **표준화(StandardScaler)** : iris 데이터는 feature 간 크기 차이가 거의 없어 표준화 효과 미미 → 생략

---

## 3단계 : GaussianNB 학습 및 평가

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

model = GaussianNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(f'총 갯수:{len(y_test)}, 오류수:{(y_test != y_pred).sum()}')
# 총 갯수:45, 오류수:1

print(accuracy_score(y_test, y_pred))   # 0.9777777777777777
```

---

## 4단계 : 정확도 확인 3가지 방법

```python
# 방법 1 : accuracy_score
print(accuracy_score(y_test, y_pred))   # 0.9777777777777777

# 방법 2 : pd.crosstab (confusion matrix 대체)
import pandas as pd
con_mat = pd.crosstab(y_test, y_pred, rownames=['실제값'], colnames=['예측값'])
print(con_mat)
# 예측값   0   1   2
# 실제값
# 0    16   0   0
# 1     0  17   1   ← 클래스 1을 2로 1개 오분류
# 2     0   1  11
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))   # 0.9777...

# 방법 3 : model.score()
print('test score : ', model.score(x_test, y_test))     # 0.9777777777777777
print('train score : ', model.score(x_train, y_train))  # 0.9523809523809523
# test ≈ train → 과적합 없음
```

---

## 5단계 : 모델 저장 및 불러오기

```python
import joblib

joblib.dump(model, 'logimodel.pkl')   # 모델 저장
del model                             # 메모리에서 삭제

read_model = joblib.load('logimodel.pkl')   # 모델 불러오기
# 이후 read_model 사용
```

---

## 6단계 : 새로운 데이터로 예측

```python
new_data = np.array([[5.5, 2.2], [0.6, 0.3], [1.1, 0.5]])
new_pred = read_model.predict(new_data)
print('예측 결과 : ', new_pred)   # [2 0 0]

# 각 클래스에 속할 확률 (softmax 출력값)
print(read_model.predict_proba(new_data))
# [[4.50e-06 5.66e-03 9.94e-01]   → 클래스 2일 확률 99.4%
#  [9.99e-01 1.37e-05 2.88e-06]   → 클래스 0일 확률 99.9%
#  [9.99e-01 1.37e-05 2.88e-06]]  → 클래스 0일 확률 99.9%
```

---

## 7단계 : 결정 경계 시각화

```python
import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib.colors import ListedColormap

def plot_decision_regionFunc(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')
    colors  = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap    = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    # ravel() : 2D → 1D, .T : 전치 → predict() 입력 형태로 변환
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)   # 원래 배열 모양으로 복원

    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)   # 배경 색상으로 경계 표현

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

# train + test 합쳐서 전체 분포 시각화
x_combined = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regionFunc(
    X=x_combined, y=y_combined,
    classifier=read_model,
    test_idx=range(105, 150),   # test 데이터 인덱스 (동그라미로 표시)
    title='scikit-learn제공'
)
```

![[ex41naive_iris.png]]

---

## 핵심 정리

### 주요 함수 정리

|함수|설명|
|---|---|
|`np.corrcoef(a, b)[0,1]`|두 변수 간 상관계수|
|`pd.crosstab(실제, 예측)`|confusion matrix를 DataFrame으로 표현|
|`model.score(x, y)`|정확도 한 번에 계산|
|`model.predict_proba()`|각 클래스 확률 반환 (softmax 출력)|
|`joblib.dump() / .load()`|모델 저장 / 불러오기|
|`np.vstack()`|배열 수직 결합 (행 추가)|
|`np.hstack()`|배열 수평 결합 (열 추가)|
|`np.meshgrid()`|격자 좌표 생성 (결정 경계 그릴 때 사용)|
|`arr.ravel()`|다차원 배열 → 1차원으로 펼치기|

### 오류 주의사항

|상황|주의|
|---|---|
|표준화 후 모델 저장|새 데이터도 반드시 `sc.transform()` 후 예측|
|`x2_min, x2_max`|코드에서 `x[:, 0]` 기준으로 잘못 계산됨 (원래는 `x[:, 1]` 이어야 함) → 결정 경계가 정사각형으로 그려지는 원인|

---
# 📄 ex41quiz.py — GaussianNB 분류 (mushrooms.csv)

---

## 개념 정리

### 전체 흐름

```
mushrooms.csv 로드
    ↓
인코딩 (cat.codes) + label map
    ↓
XGBoost 학습 → plot_importance로 중요 feature 확인
    ↓
상위 10개 feature 선택
    ↓
GaussianNB 학습 → 평가
```

### XGBoost를 feature 선택 도구로 사용하는 이유

> GaussianNB는 feature importance를 직접 제공하지 않음 → XGBoost로 먼저 학습해 중요 변수를 추린 뒤 GaussianNB에 입력하는 **2단계 전략** 사용

---

## 1단계 : 데이터 로드 및 확인

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import plot_importance

df = pd.read_csv('mushrooms.csv')
print(df.info())
# 총 23개 컬럼, 8124행, 전부 object 타입 → 인코딩 필요
```

---

## 2단계 : 인코딩 및 분리

```python
# feature : class 제외 22개
feature = df.drop('class', axis=1)

# object → 숫자 변환 (알파벳 순으로 자동 코드 부여)
for col in feature.columns:
    feature[col] = feature[col].astype('category').cat.codes

# label : e=1(식용), p=0(독버섯)
label = df['class'].map({'e': 1, 'p': 0})

x_train, x_test, y_train, y_test = train_test_split(
    feature, label, test_size=0.3, random_state=42, stratify=label
)
print(x_train.shape, x_test.shape)   # (5686, 22) (2438, 22)
```

---

## 3단계 : XGBoost로 중요 feature 확인

```python
xgb_clf = xgb.XGBClassifier(
    booster='gbtree',       # 트리 기반 부스터
    max_depth=6,            # 트리 최대 깊이
    n_estimators=200,       # 트리 개수
    eval_metric='logloss',  # 손실 함수
    random_state=42
)

xgb_clf.fit(x_train, y_train)
xgb_pred = xgb_clf.predict(x_test)
print("XGBoost 정확도 : ", accuracy_score(y_test, xgb_pred))

# 중요도 시각화 (중요도 0인 feature는 자동 제외 → 19개만 표시됨)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_importance(xgb_clf, ax=ax)
plt.show()
```

---

## 4단계 : 상위 feature 추출

```python
# get_fscore() : feature가 분기에 사용된 횟수 기준 중요도
importance_dict = xgb_clf.get_booster().get_fscore()

importance_df = pd.DataFrame({
    'feature'   : list(importance_dict.keys()),
    'importance': list(importance_dict.values())
}).sort_values('importance', ascending=False)

# 상위 10개 feature 선택
top_features = importance_df['feature'][:10].tolist()
print(top_features)
```

---

## 5단계 : GaussianNB 학습 및 평가

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

nb_model = GaussianNB()
nb_model.fit(x_train[top_features], y_train)

pred = nb_model.predict(x_test[top_features])
print('정확도 : ', accuracy_score(y_test, pred))   # 0.9122231337161608
print(classification_report(y_test, pred, target_names=['poisonous', 'edible']))
#               precision    recall  f1-score   support
#    poisonous       0.94      0.88      0.91      1175
#       edible       0.89      0.94      0.92      1263
#     accuracy                           0.91      2438
```

---

## classification_report 해석

|클래스|precision|recall|f1-score|의미|
|---|:-:|:-:|:-:|---|
|poisonous|0.94|0.88|0.91|독버섯 예측 정밀도 높음|
|edible|0.89|0.94|0.92|식용버섯 재현율 높음|

> **recall이 더 중요한 문제** : 독버섯을 식용으로 잘못 분류(FN)하면 위험 → poisonous의 recall(0.88)을 더 높이는 방향으로 개선 필요

---

## 핵심 정리

### XGBoost 주요 파라미터

|파라미터|설명|
|---|---|
|`booster='gbtree'`|트리 기반 부스터 사용|
|`max_depth`|트리 최대 깊이 (깊을수록 복잡, 과적합 위험)|
|`n_estimators`|생성할 트리 개수|
|`eval_metric`|손실 평가 기준 (`logloss` = 이진 분류 기본값)|

### get_fscore() vs feature_importances_

|방법|설명|
|---|---|
|`get_booster().get_fscore()`|feature가 분기에 사용된 횟수 반환 → dict|
|`xgb_clf.feature_importances_`|동일한 값을 numpy 배열로 반환|

### veil-type 제외 이유

> `veil-type` 컬럼은 전체 8124행이 모두 같은 값('p') → 분류에 기여 없음 → XGBoost가 중요도 0으로 판단, `plot_importance` 및 `get_fscore()`에서 자동 제외 → 19개만 표시되는 원인 (나머지 2개도 동일한 이유)

---
# 📄 ex42naive_mail.py — 스팸 메일 분류기 (MultinomialNB)

---

## 개념 정리

### CountVectorizer

> 텍스트 문서를 **단어 빈도수 기반의 숫자 행렬**로 변환하는 도구 단어의 순서 정보는 버리고, 각 단어가 몇 번 등장했는지만 추출

```
"무료 쿠폰 지금 무료 클릭하면 무료 선물"
        ↓ CountVectorizer
[무료:3, 쿠폰:1, 지금:1, 클릭하면:1, 선물:1, ...]
```

|속성/메서드|설명|
|---|---|
|`fit_transform(texts)`|단어사전 생성 + 행렬 변환 (train에만 사용)|
|`transform(texts)`|기존 사전으로 변환만 (test, 새 데이터에 사용)|
|`get_feature_names_out()`|단어사전 목록 반환|
|`vocabulary_`|단어 → 인덱스 매핑 딕셔너리|

### MultinomialNB (다항 나이브 베이즈)

> 텍스트 분류에서 가장 많이 쓰이는 나이브 베이즈 모델 **단어의 등장 횟수(빈도)** 를 기반으로 클래스 확률 계산

$$P(spam | 문서) \propto P(spam) \times \prod_{i} P(단어_i | spam)^{count_i}$$

|종류|적합한 데이터|특징|
|---|---|---|
|`GaussianNB`|연속형 수치|정규분포 가정|
|`MultinomialNB`|단어 빈도 (정수)|텍스트 분류에 최적|
|`BernoulliNB`|이진형 (0 or 1)|단어 존재 여부만|

### 희소 행렬 (Sparse Matrix)

> 대부분의 값이 0인 행렬 → 0은 저장하지 않고 값이 있는 위치만 저장해 메모리 절약

```
print(x)
# (문서번호, 단어번호)  등장횟수
#   (0, 3)              3    ← 0번 문서에서 3번 단어(무료)가 3번 등장
#   (0, 11)             1
```

---

## 1단계 : 학습 데이터 준비

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "무료 쿠폰 지금 무료 클릭하면 무료 선물",
    "한번만 클릭하면 무료 대박",
    "오늘 회의는 2시야",
    "지금 할인 행사 진행 중",
    "회의 자료는 메일로 보내주세요",
    "지금 바로 쿠폰 확인"
]
labels = ["spam", "spam", "ham", "spam", "ham", "spam"]
```

---

## 2단계 : CountVectorizer로 벡터화

```python
vect = CountVectorizer()
x = vect.fit_transform(texts)   # 단어사전 생성 + 행렬 변환

print(vect.get_feature_names_out())
# ['2시야' '대박' '메일로' '무료' '바로' '보내주세요' '선물' '오늘'
#  '자료는' '지금' '진행' '쿠폰' '클릭하면' '한번만' '할인' '행사' '확인' '회의' '회의는']

print(x.toarray())
# 행 = 문서, 열 = 단어, 값 = 등장 횟수
# [[0 0 0 3 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0]   ← 0번 문서: '무료' 3번
#  [0 1 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0]
#  [1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1]
#  [0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 0 0 0]
#  [0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0]
#  [0 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0 0]]

print(vect.vocabulary_)
# {'무료': 3, '쿠폰': 11, '지금': 9, ...}  ← 단어:인덱스 매핑
```

---

## 3단계 : 모델 학습 및 평가

```python
from sklearn.metrics import accuracy_score

model = MultinomialNB()
model.fit(x, labels)

pred = model.predict(x)
print('정확도 : ', accuracy_score(labels, pred))
# 학습 데이터 자체로 평가했으므로 정확도 매우 높음 (참고용)
```

---

## 4단계 : 새로운 문장 예측

```python
test_text = ["무료 쿠폰 지금 발급", "간부 회의는 언제 시작하나요?"]

# transform() : 기존 단어사전 그대로 사용 (fit 하지 않음)
x_test = vect.transform(test_text)
print(x_test)
# (0, 3)  1  ← '무료'
# (0, 9)  1  ← '지금'
# (0, 11) 1  ← '쿠폰'
# (1, 18) 1  ← '회의는'
# * 새 단어('발급', '간부', '언제' 등)는 사전에 없으므로 무시됨

preds = model.predict(x_test)
probs = model.predict_proba(x_test)
class_names = model.classes_   # ['ham', 'spam'] ← 알파벳 순 정렬

# zip() : 여러 리스트를 같은 인덱스끼리 묶어 동시에 순회
for text, pred, prob in zip(test_text, preds, probs):
    prob_str = ", ".join([f"{cls}:{p:.4f}" for cls, p in zip(class_names, prob)])
    print(f"'{text}' -> 예측:{pred} / 확률:[{prob_str}]")
# '무료 쿠폰 지금 발급'       -> 예측:spam / 확률:[ham:..., spam:...]
# '간부 회의는 언제 시작하나요?' -> 예측:ham  / 확률:[ham:..., spam:...]
```

---

## 핵심 정리

### fit_transform vs transform

|메서드|동작|사용 위치|
|---|---|---|
|`fit_transform()`|단어사전 생성 + 변환|train 데이터에만|
|`transform()`|기존 사전으로 변환만|test / 새 데이터|

> test에 `fit_transform` 쓰면 단어사전이 새로 만들어져 train과 기준이 달라짐 → **오류 또는 엉뚱한 예측**

### 희소 행렬 읽는 법

```
(문서번호, 단어번호)  등장횟수
    (0, 3)              3    ← 0번 문서에서 3번 단어가 3번 등장
```

### 사전에 없는 단어 처리

> 새 문장에 학습 때 없던 단어가 나와도 **그냥 무시**하고 아는 단어만으로 예측 → CountVectorizer의 기본 동작

---
# 📄 ex43naive_mail.py — 스팸 메일 분류기 (CSV + 사용자 입력)

---

## 개념 정리

### ConfusionMatrixDisplay

> `confusion_matrix()` 결과를 **시각화**해주는 sklearn 내장 도구 seaborn heatmap 없이도 혼동행렬을 깔끔하게 출력 가능

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
disp.plot(cmap='Blues')
```

### 스팸 판별 임계값 (Threshold)

> `predict()` 는 확률이 0.5 이상이면 spam으로 분류 임계값을 직접 설정하면 **더 엄격하게** 스팸 판별 가능

```python
# 0.7 이상이어야 스팸으로 판단 (기본 0.5보다 엄격)
result = '스팸이에요' if spam_prob >= 0.7 else '정상 메일입니다'
```

---

## 1단계 : 데이터 로드 및 전처리

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import koreanize_matplotlib

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/mydata.csv')
print(df.head(3), df.shape)
#               text label
# 0    광고성 메일을 확인하세요  spam  (20, 2)

# 공백 제거 + 소문자 통일 (데이터 정제)
df['label'] = df['label'].str.strip().str.lower()

texts  = df['text'].tolist()
labels = df['label'].tolist()
```

---

## 2단계 : 데이터 분할 및 벡터화

```python
x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)  # 단어사전 생성 + 변환
x_test_vec  = vectorizer.transform(x_test)        # 기존 사전으로 변환만
```

---

## 3단계 : 모델 학습 및 평가

```python
model = MultinomialNB()
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)
print('분류 정확도 : ', accuracy_score(y_test, y_pred))   # 0.8
```

---

## 4단계 : Confusion Matrix 시각화

```python
cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
print(cm)
# [[2 1]   ← ham  2개 정답, 1개 spam으로 오분류
#  [0 2]]  ← spam 2개 모두 정답

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
disp.plot(cmap='Blues')
plt.title('Confusion matrix(혼동행렬)')
plt.show()
```

![[ex43naive_mail.png]]

### Confusion Matrix 해석

| |예측: ham|예측: spam|
|---|:-:|:-:|
|**실제: ham**|2 (TN)|1 (FP)|
|**실제: spam**|0 (FN)|2 (TP)|

> FN=0 → 스팸을 정상으로 놓친 경우 없음 (중요!) FP=1 → 정상 메일을 스팸으로 잘못 분류 1건

---

## 5단계 : 사용자 입력 실시간 분류

```python
while True:
    userInput = input('이메일 내용 입력(종료는 q):')
    if userInput.lower() == 'q':
        break

    x_new = vectorizer.transform([userInput])            # 리스트로 감싸야 함
    prob = model.predict_proba(x_new)[0]                 # 각 클래스 확률
    spam_prob = prob[model.classes_.tolist().index('spam')]  # spam 확률만 추출

    # 임계값 0.7 : 기본(0.5)보다 엄격하게 스팸 판별
    result = '스팸이에요' if spam_prob >= 0.7 else '정상 메일입니다'
    print(f'스팸확률은 {spam_prob:.2f} -> {result}')

# 이메일 내용 입력(종료는 q): 광고성 메일을 확인하세요
# 스팸확률은 0.93 -> 스팸이에요
# 이메일 내용 입력(종료는 q): 회의 일정 변경 공지
# 스팸확률은 0.05 -> 정상 메일입니다
# 이메일 내용 입력(종료는 q): 무료 쿠폰 보냈습니다 지금 확인하세요
# 스팸확률은 0.99 -> 스팸이에요
```

---

## 핵심 정리

### ex42 vs ex43 비교

| |ex42|ex43|
|---|---|---|
|데이터|코드에 직접 작성|CSV 파일에서 읽기|
|벡터화 오류|`fit_transform` 버그 있었음|`transform` 올바르게 사용|
|평가|학습 데이터로 평가|train/test 분리 후 평가|
|시각화|없음|`ConfusionMatrixDisplay` 사용|
|예측 방식|일괄 예측|`while` 루프로 실시간 입력|

### 주요 함수 정리

|함수|설명|
|---|---|
|`str.strip()`|앞뒤 공백 제거|
|`str.lower()`|소문자 변환|
|`tolist()`|numpy 배열 → 파이썬 리스트 변환|
|`model.classes_`|클래스 이름 배열 반환 (`['ham', 'spam']`)|
|`list.index('spam')`|리스트에서 'spam'의 인덱스 반환|

---
# 📄 ex44streamlit.py — Streamlit 웹 이메일 분류기

---

## 개념 정리

### Streamlit

> 파이썬 코드만으로 **웹 UI를 만들 수 있는** 라이브러리 HTML/CSS/JS 없이도 브라우저에서 바로 실행 가능

```bash
pip install streamlit          # 설치
streamlit run 파일명.py        # 실행
# → 브라우저에서 http://localhost:8501 자동 오픈
```

### 주요 Streamlit 함수

|함수|설명|
|---|---|
|`st.title("텍스트")`|큰 제목 출력|
|`st.text_input("라벨")`|텍스트 입력창|
|`st.write("텍스트")`|텍스트 출력|
|`st.progress(값)`|0.0~1.0 사이 값으로 진행 바 표시|

### 동작 방식

> Streamlit은 사용자가 입력할 때마다 **전체 코드를 위에서부터 다시 실행**함 → `if user_input:` 으로 입력이 있을 때만 예측 실행

---

## 전체 코드

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# 학습 데이터
texts = [
    '광고성 메일을 확인하세요',
    '회의 일정 변경 공지',
    '무료 쿠폰을 지금 사용하세요',
    '중요한 계약 내용을 확인해주세요',
    '지금 할인 중입니다',
    '오늘 업무 일정 다시 확인해 주세요',
    '지금 바로 확인하세요',
    '사내 공지입니다',
]
labels = ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']

# 벡터화 + 모델 학습 (앱 시작 시 1회 실행)
vect = CountVectorizer()
x = vect.fit_transform(texts)

model = MultinomialNB()
model.fit(x, labels)

# Streamlit UI
st.title("이메일 분류기(나이브베이즈)")

user_input = st.text_input("이메일 내용을 입력하세요")

if user_input:   # 입력값이 있을 때만 예측 실행
    x_new = vect.transform([user_input])   # 리스트로 감싸야 함
    pred  = model.predict(x_new)[0]
    prob  = model.predict_proba(x_new)[0]

    spam_prob = prob[model.classes_.tolist().index('spam')]
    ham_prob  = prob[model.classes_.tolist().index('ham')]

    st.write(f'예측 결과 : {pred}')
    st.progress(spam_prob if pred == 'spam' else ham_prob)   # 확률을 진행 바로 표시
    st.write(f'확률 결과 → spam:{spam_prob:.2%}, ham:{ham_prob:.2%}')
```

---

## 실행 결과

### 미학습 단어만 포함된 경우 → 50:50

![[ex44streamlit.png]]

> "대출 즉시 가능합니다" → 학습 데이터에 없는 단어들 → 확률 50:50 → ham으로 분류

### 학습 단어 포함 시 → 정상 분류

![[ex44streamlit2.png]]

> "대출 즉시 가능합니다 **지금 할인 중** 입니다" → 학습된 spam 단어 포함 → spam 90.30%

---

## 핵심 정리

### 미학습 단어 처리 방식

> CountVectorizer는 학습 때 없던 단어를 **그냥 무시**함 아는 단어가 하나도 없으면 모든 클래스 확률이 **균등(50:50)** 이 됨

### `while` vs Streamlit 비교

| |ex43 (while 루프)|ex44 (Streamlit)|
|---|---|---|
|실행 환경|터미널|브라우저|
|UI|텍스트 입력/출력|웹 UI (입력창, 진행 바)|
|배포|불가|Streamlit Cloud로 배포 가능|
|코드 복잡도|낮음|낮음 (HTML 불필요)|

### 실행 포트 주의

```
기본 포트 : 8501
포트 충돌 시 자동으로 8502, 8503... 으로 변경됨
강제 종료 : 터미널에서 Ctrl + C
```

---
# 📄 K-NN (K-Nearest Neighbors) 개념 정리

---

## 개념

> 예측하려는 임의의 데이터와 가장 가까운 거리의 데이터 K개를 찾아 **다수결**에 의해 데이터를 예측하는 지도학습 알고리즘

![[knn_example.png]]

> K=3일 때 별 모양 주변 3개 중 Class B가 더 많으므로 → **Class B로 분류** K=4로 늘려도 Class B가 더 많으므로 → **Class B로 분류**

---

## K값의 영향

|K값|문제|결과|
|---|---|---|
|너무 작음 (K=1)|노이즈에 민감, 경계가 너무 복잡|과적합 (Overfitting)|
|너무 큼 (K=100)|경계가 너무 모호|과소적합 (Underfitting)|
|적절한 K|교차검증/랜덤샘플링으로 탐색|일반화 성능 좋음|

> 보통 **훈련 데이터 개수의 제곱근**을 초기 K값으로 설정 K는 **홀수**로 설정하는 것이 일반적 → 다수결 시 동점 방지

### K를 홀수로 쓰는 이유

```
K=4 일 때
Class A : 2개
Class B : 2개  ← 동점! 분류 불가

K=3 일 때
Class A : 1개
Class B : 2개  ← Class B로 명확하게 분류
```

> 클래스가 3개 이상이면 홀수여도 동점이 날 수 있어 절대적인 규칙은 아니지만 관례적으로 홀수 K를 많이 사용함

---

## 거리 측정 방식

|방식|공식|특징|
|---|---|---|
|맨해튼 거리|$d = \|a_1' - a_1\| + \|a_2' - a_2\|$|격자 이동 거리 (P=1)|
|유클리드 거리|$d = \sqrt{(a_1'-a_1)^2 + (a_2'-a_2)^2}$|직선 거리 (P=2)|
|민코프스키|$d = \sqrt[p]{\sum_k (a_k' - a_k)^p}$|두 방식을 일반화한 공식|

> P=1 → 맨해튼, P=2 → 유클리드 sklearn 기본값은 **민코프스키 P=2 (유클리드 거리)**

```python
KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
```

---

## 장단점

**장점**

- 알고리즘이 간단하여 구현하기 쉬움
- 수치 기반 데이터 분류 작업에서 성능이 좋음
- 별도의 훈련 과정 불필요 (새 데이터 추가 처리 쉬움)
- 분류 기준을 몰라도 분류 가능

**단점**

- 학습 데이터 양이 많으면 분류 속도 느림
- feature 간 스케일 차이가 크면 성능 저하 → **표준화(StandardScaler) 필수**
- feature 수가 너무 많으면 연산속도 느려짐
- 이상치의 영향을 크게 받음

---

## K-NN vs K-Means 혼동 주의

| |K-NN|K-Means|
|---|---|---|
|학습 종류|지도학습 (분류)|비지도학습 (군집화)|
|K의 의미|참조할 이웃 수|군집 개수|
|레이블|필요함|없음|
|용도|분류 / 회귀|군집화|

> **K-Means++** 는 K-Means의 초기 중심값을 잘 설정하기 위한 알고리즘 → K-NN과 무관

---
# 📄 ex45knn.py — K-NN 기초 실습

---

## 개념 정리

### weights 파라미터

> K개의 이웃을 찾은 후 분류할 때 **각 이웃에 가중치를 부여하는 방식**

|값|설명|
|---|---|
|`'uniform'`|모든 이웃을 동등하게 취급 (기본값)|
|`'distance'`|가까운 이웃일수록 더 높은 가중치 부여|

> `weights='distance'` 사용 시 거리가 가까운 데이터가 분류에 더 큰 영향을 미침

---

## 전체 코드

```python
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 3차원 데이터 (각 행이 하나의 데이터 포인트)
train = [
    [5, 3, 2],   # label 0
    [1, 3, 5],   # label 1
    [4, 5, 6]    # label 1
]
label = [0, 1, 1]

# 시각화 (train을 그대로 plot → 행별로 점 3개씩 찍힘)
plt.plot(train, 'o')
plt.xlim([-1, 5])
plt.ylim([0, 8])
plt.show()
```

![[ex45knn.png]]

```python
# K=3, 거리 기반 가중치 적용
kmodel = KNeighborsClassifier(n_neighbors=3, weights='distance')
kmodel.fit(train, label)

pred = kmodel.predict(train)
print('pred :', pred)
# pred : [0 1 1]

print(f'test acc : {kmodel.score(train, label)}')
# test acc : 1.0  ← 학습 데이터로 평가했으므로 참고용

# 새로운 데이터 예측
new_data = [[1, 2, 9], [6, 2, 1]]
new_pred = kmodel.predict(new_data)
print('new_pred : ', new_pred)
# new_pred : [1 0]
# [1, 2, 9] → label 1에 가까움
# [6, 2, 1] → label 0에 가까움
```

---

## 핵심 정리

### KNeighborsClassifier 주요 파라미터

|파라미터|기본값|설명|
|---|---|---|
|`n_neighbors`|5|참조할 이웃 수 (K값)|
|`weights`|`'uniform'`|가중치 방식|
|`p`|2|민코프스키 거리 (2=유클리드)|
|`metric`|`'minkowski'`|거리 측정 방식|

### 주의사항

> 이 코드는 학습 데이터가 **3개뿐**이고 K=3으로 설정했기 때문에 모든 이웃이 참조되어 `weights='distance'` 의 효과가 명확히 드러남 실제 사용 시에는 데이터가 충분히 많아야 K-NN이 의미있게 동작함

---
# 📄 ex46knn_cancer.py — K-NN 분류 (breast_cancer)

---

## 개념 정리

### 스케일링이 필요한 이유

> K-NN은 **거리 기반** 모델이라 feature 간 크기 차이가 크면 큰 값을 가진 feature가 거리를 지배함 → 반드시 **StandardScaler로 표준화** 후 학습해야 공정한 거리 계산 가능

```python
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # 평균/표준편차 계산 + 변환
x_test_scaled  = scaler.transform(x_test)        # 학습된 기준으로 변환만
# ※ test에 fit_transform 쓰면 기준이 달라지므로 반드시 transform만 사용
```

### 최적 K값 찾기

> K값에 따라 정확도가 달라지므로 **반복문으로 여러 K를 시도**해 최적값 탐색

|K값|특징|
|---|---|
|K=3|test acc 가장 높지만 과적합 의심|
|K=4|불안정 (짝수라 동점 가능)|
|K=7~9|안정적 → **실무에서 권장**|

---

## 1단계 : 데이터 로드

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
x = data.data    # (569, 30) : 30개 feature
y = data.target  # 0:악성(malignant), 1:양성(benign)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 2단계 : 스케일링

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # 학습용 : 기준 생성 + 변환
x_test_scaled  = scaler.transform(x_test)        # 검증용 : 변환만
```

---

## 3단계 : K값 탐색

```python
train_acc = []
test_acc  = []
k_range   = range(3, 12)

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train_scaled, y_train)

    train_acc.append(accuracy_score(y_train, model.predict(x_train_scaled)))
    test_acc.append(accuracy_score(y_test,  model.predict(x_test_scaled)))

# 시각화
plt.figure()
plt.plot(k_range, train_acc, marker='o', label='Train acc')
plt.plot(k_range, test_acc,  marker='s', label='Test acc')
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.title('knn acc comp')
plt.legend()
plt.grid()
plt.show()

# test acc가 가장 높은 K값 출력
best_k = k_range[np.argmax(test_acc)]
print('최적의 k :', best_k)   # 3 (but 과적합 의심)
```

![[ex46knn_cancer.png]]

| K값    | Train acc | Test acc  | 분석                     |
| ----- | --------- | --------- | ---------------------- |
| 3     | 0.978     | 0.985     | test 최고, 불안정 → 과적합 의심  |
| 4     | 0.982     | 0.948     | 짝수 동점 → 급락             |
| 5     | 0.974     | 0.956     | 회복 중                   |
| 6     | 0.978     | 0.956     | 안정적                    |
| 7     | 0.976     | 0.974     | 안정권 진입                 |
| 8     | 0.976     | 0.974     | 안정적                    |
| **9** | **0.974** | **0.974** | **홀수 + 안정적 → 최종 선택** ✅ |
| 10    | 0.976     | 0.965     | 하락                     |
| 11    | 0.971     | 0.974     | 회복                     |

---

## 4단계 : 최종 모델 (K=9)

```python
# test acc 최고점(K=3)보다 안정적인 K=9 선택
best_k = 9
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(x_train_scaled, y_train)

y_pred = final_model.predict(x_test_scaled)
print('정확도 : ', accuracy_score(y_test, y_pred))   # 0.9736842105263158

print(classification_report(y_test, y_pred))
#                precision  recall  f1-score  support
#             0       1.00    0.93      0.96       42   ← 악성
#             1       0.96    1.00      0.98       72   ← 양성
#      accuracy                         0.97      114

print(confusion_matrix(y_test, y_pred))
# [[39  3]   ← 악성 39 정답, 3개 양성으로 오분류
#  [ 0 72]]  ← 양성 72 모두 정답 (FN=0)
```

> **FN=0** : 실제 악성을 양성으로 놓친 경우 없음 → 의료 분류에서 매우 중요한 지표

---

## 5단계 : 새로운 데이터 예측

```python
# 기존 데이터에 약간의 노이즈를 추가해 새 데이터처럼 사용
new_data = x[0].copy()
new_data = new_data + np.random.normal(0, 0.1, size=new_data.shape)

# 반드시 학습 때 사용한 scaler로 변환
new_data_scaled = scaler.transform([new_data])

prediction = final_model.predict(new_data_scaled)
proba       = final_model.predict_proba(new_data_scaled)

print('예측 : ', prediction[0], '  (0:악성, 1:양성)')
# 예측 :  0  (0:악성, 1:양성)
print('확률 : ', proba)
# [[0.555 0.444]]  ← 악성 55.6%, 양성 44.4% → 근소한 차이로 악성 예측
```

---

## 핵심 정리

### 전체 흐름

```
데이터 로드 (569, 30)
    ↓
train/test 분리 (stratify=y)
    ↓
StandardScaler 표준화  ← 거리 기반 모델 필수
    ↓
K값 탐색 (range(3,12)) → 그래프로 최적 K 확인
    ↓
최종 모델 K=9 (안정적) → 평가
    ↓
새로운 데이터 예측 (scaler 동일하게 적용)
```

### np.random.normal()

|파라미터|의미|
|---|---|
|`0`|평균 (노이즈 중심)|
|`0.1`|표준편차 (노이즈 크기)|
|`size`|생성할 배열 크기|

> 기존 데이터에 작은 노이즈를 더해 새 데이터처럼 만드는 테스트 기법

---
# 📄 ex47knn_iris.py — K-NN 분류 (iris)

---

## 개념 정리

> ex41naive_iris.py와 동일한 구조지만 **GaussianNB → KNeighborsClassifier** 로 모델만 교체 같은 데이터, 같은 시각화 함수로 두 모델의 **결정 경계 차이**를 비교할 수 있음

| |ex41 (GaussianNB)|ex47 (K-NN)|
|---|---|---|
|모델|GaussianNB|KNeighborsClassifier|
|정확도|0.9777|0.9777|
|결정 경계|곡선형 (확률 기반)|불규칙 곡선 (거리 기반)|
|스케일링|불필요|iris는 차이 작아 생략|

---

## 1단계 : 데이터 로드 및 feature 선택

```python
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

# 꽃잎 길이(2열), 꽃잎 너비(3열) 2개만 사용 → 시각화(2D) 목적
x = iris.data[:, [2, 3]]
y = iris.target

# 꽃잎 길이와 너비의 상관계수
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])[0, 1])   # 0.9628
```

---

## 2단계 : 데이터 분할

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape)   # (105, 2) (45, 2)

# iris는 feature 간 크기 차이가 거의 없어 표준화 효과 미미 → 생략
```

---

## 3단계 : K-NN 학습 및 평가

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(f'총 갯수:{len(y_test)}, 오류수:{(y_test != y_pred).sum()}')
# 총 갯수:45, 오류수:1

# 정확도 확인 3가지
print(accuracy_score(y_test, y_pred))          # 0.9777 ① accuracy_score

con_mat = pd.crosstab(y_test, y_pred, rownames=['실제값'], colnames=['예측값'])
print(con_mat)
# 예측값   0   1   2
# 실제값
# 0    16   0   0
# 1     0  17   1  ← 클래스 1을 2로 1개 오분류
# 2     0   1  11
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))  # 0.9777 ②

print('test score : ', model.score(x_test, y_test))     # 0.9777 ③
print('train score : ', model.score(x_train, y_train))  # 0.9523
# test ≈ train → 과적합 없음
```

---

## 4단계 : 모델 저장 및 불러오기

```python
import joblib

joblib.dump(model, 'logimodel.pkl')   # 모델 저장
del model                              # 메모리에서 삭제
read_model = joblib.load('logimodel.pkl')   # 불러오기
```

---

## 5단계 : 새로운 데이터 예측

```python
new_data = np.array([[5.5, 2.2], [0.6, 0.3], [1.1, 0.5]])
new_pred = read_model.predict(new_data)
print('예측 결과 : ', new_pred)   # [2 0 0]

# 각 클래스 확률
print(read_model.predict_proba(new_data))
# [[0. 0.2 0.8]   → 클래스 2일 확률 80%
#  [1.  0.  0. ]  → 클래스 0일 확률 100%
#  [1.  0.  0. ]] → 클래스 0일 확률 100%
# K-NN은 이웃 K개의 다수결이므로 확률이 0.2 단위로 나옴 (K=5 기준)
```

---

## 6단계 : 결정 경계 시각화

```python
x_combined = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regionFunc(
    X=x_combined, y=y_combined,
    classifier=read_model,
    test_idx=range(105, 150),
    title='scikit-learn제공'
)
```

![[ex47knn_iris.png]]

> K-NN의 결정 경계는 **거리 기반**이라 GaussianNB보다 불규칙한 형태 클래스 1(파랑)과 2(초록) 경계가 복잡하게 나뉘는 것이 K-NN의 특징

---

## 핵심 정리

### predict_proba 확률 단위

> K=5 일 때 이웃 5개의 다수결이므로 확률은 **0.2(1/5) 단위**로만 나옴 K=3이면 0.333 단위, K=7이면 0.143 단위

### x2_min, x2_max 버그 (ex41과 동일)

```python
# 코드상 버그 : x[:,0] 기준으로 x2도 계산됨
x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # ❌
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # ✅
# → 결정 경계가 정사각형 격자로 그려지는 원인
```