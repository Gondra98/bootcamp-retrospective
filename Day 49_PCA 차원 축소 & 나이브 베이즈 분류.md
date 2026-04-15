# Day 49_PCA 차원 축소 & 나이브 베이즈 분류
## 📅 2026-04-15

---

# 📄 ex38.py — PCA (주성분분석)

## 개념 정리

> **PCA(Principal Component Analysis)** 란? 선형대수 관점에서, 입력데이터의 **공분산 행렬을 고윳값 분해**하고, 이렇게 구한 **고유벡터에 입력 데이터를 선형변환**하는 것이다. 이 고유벡터가 PCA의 **주성분 벡터**로서 입력 데이터의 **분산이 큰 방향**을 나타낸다. → 입력 데이터의 성질을 최대한 유지한 상태로 **고차원을 저차원 데이터로 변환**하는 기법

### PCA 진행 순서

|순서|내용|
|---|---|
|1|입력 데이터의 **공분산 행렬** 생성|
|2|공분산 행렬의 **고유벡터**와 **고윳값** 계산|
|3|고윳값이 큰 순서대로 k개 만큼 고유벡터 추출|
|4|추출된 고유벡터로 입력 데이터를 새롭게 변환|

> `sklearn`의 `PCA`를 이용하면 위 순서가 자동으로 진행됨

### 주요 속성

|속성|설명|
|---|---|
|`components_`|주성분 방향벡터|
|`explained_variance_ratio_`|각 주성분이 설명하는 분산 비율|
|`fit_transform()`|학습 + 차원 축소 동시 수행|
|`inverse_transform()`|축소된 데이터를 원래 차원으로 복원|

---

## 1단계 : iris 데이터 준비 및 시각화

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
n = 10

# 꽃받침 길이(sepal length), 꽃받침 폭(sepal width) 2열만 선택
x = iris.data[:n, :2]
print('차원 축소 전 : x: ', x, x.shape, type(x))
# → (10, 2) shape : 10개 표본, 2개 특성
print(x.T)
```

출력 결과:

```
차원 축소 전 : x:  [[5.1 3.5]
 [4.9 3. ]
 ...
 [4.9 3.1]] (10, 2) <class 'numpy.ndarray'>
[[5.1 4.9 4.7 4.6 5.  5.4 4.6 5.  4.4 4.9]
 [3.5 3.  3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1]]
```

### 시각화1 - 꺾은선 그래프 (주석처리됨)

```python
# plt.plot(x.T, 'o:')
# plt.xticks(range(2), ['꽃받침길이', '꽃받침너비'])
# plt.grid(True)
# plt.title('아이리스 크기 특성')
# plt.xlabel('특성의 종류')
# plt.ylabel('특성값')
# plt.xlim(-0.5, 2)
# plt.ylim(2.5, 6)
# plt.legend(['표본 {}'.format(i + 1) for i in range(n)])
# plt.show()
```

![[ex38pca.png]]

### 시각화2 - 산점도

```python
df = pd.DataFrame(x)

# marker='s' : 사각형 마커 (markers 아님 주의!)
ax = sns.scatterplot(x=0, y=1, data=df, marker='s', s=100, color='b')

# 각 점에 레이블 표시 (좌측 상단으로 약간 오프셋)
for i in range(n):
    ax.text(x[i, 0] - 0.05, x[i, 1] + 0.03, '표본{}'.format(i+1))

plt.xlabel('꽃받침길이')
plt.ylabel('꽃받침폭')
plt.title('아이리스 특성')
plt.axis('equal')   # x, y 축 비율 동일하게 (axis ≠ axes 주의!)
plt.show()
```

![[ex38pca2.png]]

> 그래프 결과: 두 변수(꽃받침 길이, 폭)는 공통된 경향이 있음 → **차원 축소의 근거**가 있다고 판단 → PCA 진행

---

## 2단계 : PCA 차원 축소 (2D → 1D)

```python
# n_components=1 : 1차원으로 축소
pca1 = PCA(n_components=1)

# fit_transform : 공분산 행렬 계산 → 고유벡터 추출 → 선형변환 한번에 수행
x_low = pca1.fit_transform(x)   # (10, 2) → (10, 1)
print('x_low : ', x_low, ' ', x_low.shape)

# inverse_transform : 1차원 → 2차원으로 복원 (손실 있음)
x2 = pca1.inverse_transform(x_low)
print('원복 후 x2: ', x2, ' ', x2.shape)

print('원본  : ', x[0, :])      # [5.1 3.5]
print('주성분: ', x_low[0])     # [0.30270263]
print('원복  : ', x2[0, :])     # [5.06676112 3.53108532] ← 원본과 약간 다름(정보손실)
```

출력 결과:

```
x_low :  [[ 0.30270263]
 [-0.1990931 ]
 ...
 [-0.12605597]]   (10, 1)

원복 후 x2:  [[5.06676112 3.53108532]
 ...
 [4.77389743 3.21793233]]   (10, 2)
```

> **원복 후 값이 원본과 다른 이유**: 2차원 → 1차원으로 압축하면서 일부 정보가 손실되기 때문

---

## 3단계 : 제1주성분 방향 시각화

```python
# components_ : 주성분 방향벡터 (데이터 분산이 가장 큰 방향)
pc1 = pca1.components_[0]

# 데이터의 평균 = PCA 화살표 시작점(중심점)
mean = x.mean(axis=0)

df = pd.DataFrame(x)
ax = sns.scatterplot(x=0, y=1, data=df, marker='s', s=100, color='b')

for i in range(n):
    ax.text(x[i, 0] - 0.05, x[i, 1] + 0.03, f'표본{i+1}')

# quiver : 화살표로 PCA 주성분 방향 시각화
plt.quiver(
    mean[0], mean[1],   # 화살표 시작점 (데이터 평균)
    pc1[0], pc1[1],     # 화살표 방향 (주성분 벡터)
    scale=3, color='r', width=0.01
)

plt.xlabel('꽃받침길이')
plt.ylabel('꽃받침폭')
plt.title('아이리스 특성 + 제1주성분')
plt.axis('equal')
plt.grid(True)
plt.show()
```

![[ex38pca3.png]]

---

## 4단계 : 4차원 → 2차원 PCA + SVM 분류 비교

```python
print('***' * 10)

# 원본 데이터 : 4개 열 (꽃받침길이, 꽃받침폭, 꽃잎길이, 꽃잎폭)
x = iris.data
print(x[0, :])   # [5.1 3.5 1.4 0.2]

# 4차원 → 2차원으로 PCA 변환
pca2 = PCA(n_components=2)
x_low2 = pca2.fit_transform(x)
print('x_low2 : ', x_low2[0, :], ' ', x_low2.shape)
# → [-2.68412563  0.31939725]   (150, 2)

# explained_variance_ratio_ : 각 주성분이 원본 데이터의 분산을 얼마나 설명하는지
print(pca2.explained_variance_ratio_)
# → [0.92461872 0.05306648]
# 제1주성분 92.5% + 제2주성분 5.3% = 약 97.8% 정보 보존

x4 = pca2.inverse_transform(x_low2)
print('최초 자료 : ', x[0])          # [5.1  3.5  1.4  0.2 ]
print('차원축소  : ', x_low2[0])     # [-2.684  0.319]
print('차원복귀  : ', x4[0, :])      # [5.083 3.517 1.403 0.214] ← 근사값

# DataFrame으로 정리
iris1 = pd.DataFrame(x, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
iris2 = pd.DataFrame(x_low2, columns=['var1', 'var2'])
print(iris1.head(3))
print(iris2.head(3))
```

### SVM 분류 모델 비교

```python
from sklearn import svm, metrics

label = iris.target   # 정답 레이블 (0, 1, 2)

# --- 원본 4열로 학습 ---
print('원본 데이터로 SVM 분류 모델 작성')
feature1 = iris1.values                                     # (150, 4)
model1 = svm.SVC(C=0.1, random_state=0).fit(feature1, label)
pred1 = model1.predict(feature1)
print('model1 accuracy : ', metrics.accuracy_score(label, pred1))
# → 0.94

# --- PCA 2열로 학습 ---
print('PCA 데이터로 SVM 분류 모델 작성')
feature2 = iris2.values                                     # (150, 2)
model2 = svm.SVC(C=0.1, random_state=0).fit(feature2, label)
pred2 = model2.predict(feature2)
print('model2 accuracy : ', metrics.accuracy_score(label, pred2))
# → 0.9467
```

### 결과 비교

|        | 특성 수 | 정확도    |
| ------ | ---- | ------ |
| 원본 데이터 | 4개   | 94.0%  |
| PCA 축소 | 2개   | 94.67% |

> 특성을 절반으로 줄였는데 정확도가 소폭 **상승** → PCA가 노이즈를 제거해 **일반화 성능이 향상**된 효과

---

## 핵심 정리

```
원본 데이터 (고차원)
    ↓  fit_transform()
PCA 축소 데이터 (저차원)  ←  분산이 큰 방향 순으로 압축
    ↓  inverse_transform()
복원 데이터 (고차원, 근사값)  ←  일부 정보 손실 발생
```

### 주요 함수 오류 주의

|잘못된 코드|올바른 코드|이유|
|---|---|---|
|`plt.axes('equal')`|`plt.axis('equal')`|`axes()`는 객체 생성, `axis()`가 설정|
|`markers='s'`|`marker='s'`|seaborn은 단수형 `marker` 사용|
|`plt.subplot(3,5)`|`plt.subplots(3,5)`|복수형이 Figure+Axes 배열 반환|
|`x=[i, 1]`|`x[i, 1]`|인덱싱에 `=` 오타|

---
# 📄 ex39pca.py — SVM + PCA 얼굴 이미지 분류 (LFW 데이터셋)

## 개념 정리

> **목표** : PCA로 고차원 이미지를 저차원으로 압축한 뒤 SVM으로 인물을 분류

```
얼굴 이미지 (2914픽셀)
    ↓ PCA (100차원으로 압축)
주성분 특징벡터 (100개)
    ↓ SVM 분류
인물 예측 (8명 중 1명)
```

### 핵심 아이디어

- 2914개 픽셀을 그대로 학습하면 느리고 노이즈가 많음
- PCA로 핵심 특징(얼굴 윤곽, 눈 위치, 코 그림자 등)만 추출
- SVM은 **실제 얼굴이 아니라 특징 패턴**으로 분류 작업을 수행

---

## 1단계 : 데이터 로드 및 시각화

```python
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# min_faces_per_person=60 : 한 사람당 60장 이상인 인물만 사용
# color=False : 흑백, resize=0.5 : 원본의 절반 크기
faces = fetch_lfw_people(min_faces_per_person=60, color=False, resize=0.5)

print(faces.data.shape)       # (1348, 2914) : 1348장, 픽셀 62×47=2914 (1D)
print(faces.images.shape)     # (1348, 62, 47) : 시각화용 (2D)
print(faces.target_names)     # ['Ariel Sharon', 'Colin Powell', ...]
```

### 데이터 구조

|속성|shape|용도|
|---|---|---|
|`faces.data`|(1348, 2914)|모델 학습용 (1D 펼친 형태)|
|`faces.images`|(1348, 62, 47)|시각화용 (2D 이미지)|
|`faces.target`|(1348,)|인물 번호 (정수)|
|`faces.target_names`|(8,)|인물 이름|

### 인물 이름 찾기

```python
# faces.target[1] : 1번 사진의 인물 번호 (예: 3)
# faces.target_names[3] : 3번 인물의 이름 (예: George W Bush)
print(faces.target_names[faces.target[1]])  # George W Bush
```

![[ex39pca.png]]

### 단일 이미지 시각화

```python
# 이미지 1개만 확인
plt.imshow(faces.images[1], cmap='bone')
plt.show()
```

### 15개 이미지 시각화

```python
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],   # x, y 축 숨기기
            xlabel=faces.target_names[faces.target[i]])
plt.show()
```

![[ex39pca1.png]]

---

## 2단계 : PCA 차원 축소

```python
# 설명력 95%가 되는 최소 주성분 개수 확인
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(faces.data)
print(pca.n_components_)    # 184 → 95% 설명력에 184개 필요

# 분석가가 판단해서 100개로 설정
n = 100
m_pca = PCA(n_components=n, whiten=True, random_state=0)
# whiten=True : 주성분 간 스케일을 균일하게 조정 → SVM 성능 향상
x_low = m_pca.fit_transform(faces.data)   # (1348, 2914) → (1348, 100)
```

### Eigenfaces (주성분 얼굴) 시각화

```python
fig, ax = plt.subplots(3, 5, figsize=(10, 6))
for i, axi in enumerate(ax.flat):
    # components_ : 주성분 벡터 (1D) → reshape으로 2D 이미지로 변환
    axi.imshow(m_pca.components_[i].reshape(faces.images[0].shape), cmap='bone')
    axi.axis('off')
    axi.set_title(f'PC {i + 1}')
plt.suptitle('Eigenfaces(주성분 얼굴)', fontsize=12)
plt.tight_layout()
plt.show()
# 출력 이미지는 실제 얼굴이 아니라 특징 패턴 (얼굴 윤곽, 눈 위치, 코 그림자...)
```

![[ex39pca2.png]]

> PC1~PC15로 갈수록 덜 중요한 특징을 담음 SVM은 이 특징 패턴의 조합으로 인물을 분류

### 설명력 확인

```python
print(m_pca.explained_variance_ratio_[:10])   # 상위 10개 주성분의 설명력
print('누적 설명력 : ', m_pca.explained_variance_ratio_.sum())  # 0.9039
# → 100개 주성분으로 원본 정보의 약 90% 유지
```

---

## 3단계 : 원본 vs 복원 이미지 비교

```python
# inverse_transform : 100차원 → 2914차원으로 복원 (일부 정보 손실)
x_reconst = m_pca.inverse_transform(x_low)

fig, ax = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
    ax[0, i].imshow(faces.images[i], cmap='bone')
    ax[0, i].set_title("원본")
    ax[0, i].axis('off')

    ax[1, i].imshow(x_reconst[i].reshape(faces.images[0].shape), cmap='bone')
    ax[1, i].set_title("복원")
    ax[1, i].axis('off')

plt.suptitle('PCA 복원 비교', fontsize=12)
plt.tight_layout()
plt.show()
# 원본과 복원 이미지의 기본 특징은 크게 차이 없음 → 패턴 유지됨
```

![[ex39pca3.png]]

---

## 4단계 : 분류 모델 생성 (Pipeline)

```python
from sklearn.model_selection import train_test_split

svcmodel = SVC(C=1, random_state=1)
# make_pipeline : PCA → SVM 순서로 자동 연결
# fit/predict 한 번에 처리 가능
mymodel = make_pipeline(m_pca, svcmodel)

# stratify : 클래스 비율을 유지하며 train/test 분리 (불균형 데이터 완화)
x_train, x_test, y_train, y_test = train_test_split(
    faces.data, faces.target, random_state=1, stratify=faces.target
)

mymodel.fit(x_train, y_train)
pred = mymodel.predict(x_test)
print('예측값 : ', pred[:10])   # [3 3 4 3 3 0 7 3 3 1]
print('실제값 : ', y_test[:10]) # [3 5 4 2 4 0 6 3 3 1]
```

---

## 5단계 : 정확도 평가

```python
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

confmat = confusion_matrix(y_test, pred)
print('accuracy : ', accuracy_score(y_test, pred))  # 0.8071
print(classification_report(y_test, pred, target_names=faces.target_names))
```

### classification_report 해석

|인물|precision|recall|f1-score|
|---|:-:|:-:|:-:|
|Ariel Sharon|1.00|0.63|0.77|
|Colin Powell|0.93|0.85|0.88|
|Donald Rumsfeld|1.00|0.50|0.67|
|**George W Bush**|0.70|**1.00**|0.82|
|Gerhard Schroeder|0.93|0.52|0.67|
|Hugo Chavez|1.00|0.50|0.67|
|Junichiro Koizumi|1.00|0.73|0.85|
|Tony Blair|0.93|0.78|0.85|

> George W Bush는 recall=1.0 → 실제 Bush를 Bush라고 모두 맞춤 단, precision=0.70 → 다른 인물도 Bush로 잘못 예측하는 경우가 많음 (학습 데이터에서 Bush 사진이 가장 많아서 Bush로 편향되는 경향)

---

## 6단계 : 결과 시각화

```python
# 예측 결과 시각화 (파란색 = 정답, 빨간색 = 오답)
fig, axes = plt.subplots(4, 6)
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].reshape(62, 47), cmap='bone')
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel(
        faces.target_names[pred[i]].split()[-1],   # 성(last name)만 표시
        color='blue' if pred[i] == y_test[i] else 'red',
        fontweight='bold'
    )
fig.suptitle('예측 결과', fontsize=12)
plt.tight_layout()
plt.show()
```

![[ex39pca5.png]]

```python
# Confusion Matrix heatmap
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues',
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('예측')
plt.ylabel('실제')
plt.title('Confusion matrix')
plt.show()
```

![[ex39pca6.png]]

```python
# PCA 누적 설명력 그래프
import numpy as np
plt.plot(np.cumsum(m_pca.explained_variance_ratio_))
plt.xlabel('주성분 개수')
plt.ylabel('누적 설명력')
plt.title('PCA 설명력')
plt.grid(True)
plt.show()
# 처음 몇 개의 주성분이 대부분의 정보를 담고 있음
# 100개 이후로는 설명력 증가가 완만해짐
```

![[ex39pca7.png]]

---

## 7단계 : 새로운 이미지로 예측

### 실습1 : 기존 데이터로 테스트

```python
test_img = faces.data[0].reshape(1, -1)   # (1, 2914) 형태로 변환
test_pred = mymodel.predict(test_img)
print('예측 결과 : ', faces.target_names[test_pred[0]])   # Colin Powell
print('실제값   : ', faces.target_names[faces.target[0]]) # Colin Powell
```

![[ex39pca4.png]]

### 실습2 : 외부 이미지로 테스트 (중요!)

```python
from PIL import Image

# 단계 : 이미지 읽기 → 흑백변환 → 크기 맞추기 → 1차원 변환 → 예측
img = Image.open('bush.jpeg')
img = img.convert('L')      # 흑백 변환 (3채널 → 1채널)
img = img.resize((47, 62))  # PIL은 (width, height) 순서!
# numpy는 (height, width) → 반드시 (47, 62)로 resize해야 numpy에서 (62, 47)이 됨

img_np = np.array(img)      # (62, 47) shape
img_np = img_np / 255.0     # 정규화 (학습 데이터와 맞춰야 함)
img_flat = img_np.reshape(1, -1)   # (1, 2914) → 모델 입력 형태

new_pred = mymodel.predict(img_flat)
print('예측 결과 : ', faces.target_names[new_pred[0]])   # George W Bush

plt.imshow(img_np, cmap='bone')
plt.title(f'예측 : {faces.target_names[new_pred[0]]}')
plt.axis('off')
plt.show()
# 참고 : 정확도를 높이려면 밝기/위치 정렬 등의 전처리가 필요!
```

![[ex39pca8.png]]

### PIL vs numpy 축 순서 주의

|라이브러리|순서|resize 인자|
|---|---|---|
|PIL|(width, height) = (가로, 세로)|`resize((47, 62))`|
|numpy|(height, width) = (세로, 가로)|결과: `(62, 47)`|

---

## 핵심 정리

### 전체 흐름

```
faces.data (1348, 2914)
    ↓ PCA(n=100, whiten=True)
x_low (1348, 100)          ← 90% 정보 유지
    ↓ make_pipeline
SVC 분류
    ↓
accuracy : 80.7%
```

### 주요 속성/함수

|코드|설명|
|---|---|
|`pca.n_components_`|설명력 기준 충족하는 최소 주성분 수|
|`pca.components_`|주성분 방향벡터 (Eigenfaces)|
|`pca.explained_variance_ratio_`|각 주성분의 설명력|
|`make_pipeline(pca, svc)`|PCA → SVM 순서로 자동 연결|
|`stratify=faces.target`|클래스 비율 유지하며 분리|
|`.split()[-1]`|이름에서 성(last name)만 추출|

### 오류 주의사항

| 오류                                         | 원인               | 수정                       |
| ------------------------------------------ | ---------------- | ------------------------ |
| `X has N features, but PCA expecting 2914` | PIL 변환 결과 미저장    | `img = img.convert('L')` |
| `plt.subplot`                              | 단수형은 단일 axes     | `plt.subplots` (복수형)     |
| `ax.ylabel()`                              | plt 함수를 axes에 사용 | `ax.set_ylabel()`        |
# 📄 베이즈 정리 & 나이브 베이즈 분류

## 개념 정리

### 베이즈 정리 (Bayes' Theorem)

> 새로운 데이터(증거)를 보고 기존의 믿음(사전확률)을 업데이트하는 수학 공식

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

$$사후확률 = \frac{우도 \times 사전확률}{증거}$$

|기호|이름|의미|
|---|---|---|
|$P(A\|B)$|사후확률 (Posterior)|데이터를 본 후 업데이트된 믿음|
|$P(B\|A)$|우도 (Likelihood)|원인을 고정했을 때 증거가 나올 확률|
|$P(A)$|사전확률 (Prior)|데이터 보기 전 초기 믿음|
|$P(B)$|증거 (Evidence)|정규화 역할|

---

### 나이브 베이즈 (Naive Bayes)

> 베이즈 정리를 머신러닝 분류에 응용한 알고리즘 **"나이브(순진한)"** = 각 특성이 서로 독립이라고 가정

$$P(클래스|특성들) \propto P(클래스) \times \prod_{i} P(특성_i|클래스)$$

| |베이즈 정리|나이브 베이즈|
|---|---|---|
|종류|수학 공식|머신러닝 알고리즘|
|용도|조건부 확률 계산|분류 문제|
|관계|기반 이론|베이즈 정리 + 독립 가정|

---

## 스팸 메일 예제

### 문제

> '광고'라는 단어가 들어간 메일이 스팸일 확률은?

### 데이터 표 (과거데이터 수집 — 귀납법)

| |광고 포함|광고 미포함|합계|
|---|:-:|:-:|:-:|
|SPAM|4|16|20|
|HAM|1|79|80|
|합계|5|95|100|

### 우도 표 — 행(row) 합계로 나눈 값 4개 모두 우도

| |광고 포함|광고 미포함|합계|
|---|:-:|:-:|:-:|
|SPAM|$P(광고\|SPAM) =$ **4/20**|$P(광고X\|SPAM) = 16/20$|20/100|
|HAM|$P(광고\|HAM) = 1/80$|$P(광고X\|HAM) = 79/80$|80/100|
|합계|5/100|95/100||

> 행 기준으로 나눈 **4개** → 우도 전체(100) 기준으로 나눈 **합계 행/열** → 사전확률 / 증거 이 중 문제에서 필요한 우도는 $P(광고|SPAM) = 4/20$

### 베이즈 정리 적용

|항목|의미|계산|값|
|---|---|:-:|:-:|
|$P(광고\|SPAM)$|우도|4/20|0.2|
|$P(SPAM)$|사전확률|20/100|0.2|
|$P(광고)$|증거|5/100|0.05|

$$P(SPAM|광고) = \frac{P(광고|SPAM) \times P(SPAM)}{P(광고)} = \frac{0.2 \times 0.2}{0.05} = 0.8$$

> **→ 80% 확률로 스팸!**

---

## 나이브 베이즈로 확장

단어가 여러 개일 때 각 단어를 **독립적으로** 판단해서 곱함

$$P(SPAM|광고,대출,이자) \propto P(광고|SPAM) \times P(대출|SPAM) \times P(이자|SPAM) \times P(SPAM)$$

> **독립사건의 조건부 확률** = 나이브 베이즈의 핵심 현실에선 완전한 독립이 아니지만, 실제로는 꽤 잘 동작함 (특히 텍스트 분류)

---

## 핵심 정리

### 우도 찾는 법

```
행(row) 합계로 나눈 값 → 우도
전체 합계로 나눈 값   → 사전확률
```

### 방향성

| |방향|예시|
|---|---|---|
|우도|원인 → 결과|스팸이니까 광고 단어가 있겠지?|
|사후확률|결과 → 원인|광고 단어가 있으니 스팸이겠지?|

> 베이즈 정리 = 우도를 뒤집어서 결과에서 원인을 역추적하는 것

### sklearn 구현

```python
from sklearn.naive_bayes import GaussianNB      # 연속형 데이터
from sklearn.naive_bayes import MultinomialNB   # 텍스트 빈도 데이터
from sklearn.naive_bayes import BernoulliNB     # 이진형 데이터 (0 or 1)

model = GaussianNB()
model.fit(x_train, y_train)
pred = model.predict(x_test)
```

|종류|적합한 데이터|주요 용도|
|---|---|---|
|`GaussianNB`|연속형 (정규분포 가정)|아이리스, 키/몸무게|
|`MultinomialNB`|정수형 빈도 데이터|텍스트 분류, 단어 빈도|
|`BernoulliNB`|이진형 (0 or 1)|스팸 분류|
