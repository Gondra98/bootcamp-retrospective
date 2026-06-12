# Day86_NLP - 희소벡터와 문서 벡터화

## 📅 2026-06-12

---
# 📄 nlp3sparse_vec.ipynb — 희소벡터 · CountVectorizer · TF-IDF

---

## 🧠 핵심 개념 정리

### 자연어를 숫자로 변환하는 세 가지 방법

|방식|핵심 질문|출력 형태|
|---|---|---|
|**CountVectorizer**|그 단어가 몇 번 나왔는가?|정수 희소벡터|
|**TfidfVectorizer**|그 단어가 이 문서에서 얼마나 중요한가?|실수 희소벡터|
|**Word2Vec**|그 단어가 어떤 의미적 위치에 있는가?|실수 밀집벡터|

> **언제 무엇을 쓸까?**
> 
> - 키워드가 정확히 맞아야 하는 검색 → `CountVectorizer`, `TfidfVectorizer`
> - 의미가 비슷한 단어까지 찾고 싶을 때 → `Word2Vec`

---

## 📦 희소벡터 (Sparse Vector) vs 밀집벡터 (Dense Vector)

- **희소벡터**: 대부분의 값이 0인 고차원 벡터. 단어 집합 크기만큼 축이 생김.
- **밀집벡터**: 학습을 통해 의미를 압축한 저차원 벡터. 대부분 0이 아닌 실수값.

---

## 🔢 CountVectorizer

문장 안에 각 단어가 **몇 번 등장했는지** 세어서 벡터로 변환.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

content = ['How to format my hard disk', 'format Hard disk format problems']

count_vec = CountVectorizer(analyzer='word', min_df=1)
# analyzer='word' : 단어 단위로 분석
# min_df=1        : 최소 1번 이상 등장한 단어만 사용

tran = count_vec.fit_transform(content)
# fit_transform : 어휘집(vocabulary) 학습 + 변환을 동시에 수행

print(count_vec.get_feature_names_out())
# ['disk' 'format' 'hard' 'how' 'my' 'problems' 'to']
#    0        1       2     3     4       5          6

print(tran.toarray())
# [[1 1 1 1 1 0 1]   ← 'How to format my hard disk'
#  [1 2 1 0 0 1 0]]  ← 'format Hard disk format problems'
# format이 2번 등장했으므로 index 1 위치 값이 2
```

### 출력 해석

```
           disk  format  hard  how  my  problems  to
문장 0:      1     1      1    1    1      0       1   ← 'How to format my hard disk'
문장 1:      1     2      1    0    0      1       0   ← 'format Hard disk format problems'
```

> **Sparse Matrix 형태**: 0이 많기 때문에 메모리 절약을 위해 0은 저장하지 않고 1인 좌표만 저장. `(0, 3) 1` → 0번 문장, 3번 단어('how') = 1번 등장

---

## 📊 TfidfVectorizer

단순 빈도(Count)에서 나아가, **이 문서에서 얼마나 중요한 단어인지** 가중치를 계산.

- **TF (Term Frequency)**: 해당 문서에서 단어가 얼마나 자주 나왔는가
- **IDF (Inverse Document Frequency)**: 모든 문서에서 흔한 단어일수록 페널티 부여

```
TF-IDF = TF × IDF
```

> **예시**: 'the', 'a' 같은 단어는 모든 문서에 자주 등장 → IDF 낮음 → TF-IDF 낮음 특정 문서에만 나오는 전문 용어 → IDF 높음 → TF-IDF 높음

```python
tfidf_vec = TfidfVectorizer(analyzer='word', min_df=1)
tran2 = tfidf_vec.fit_transform(content)

print(tfidf_vec.get_feature_names_out())
# ['disk' 'format' 'hard' 'how' 'my' 'problems' 'to']

print(tran2.toarray())
# [[0.33471228 0.33471228 0.33471228 0.47042643 0.47042643 0.         0.47042643]
#  [0.35409974 0.70819948 0.35409974 0.         0.         0.49767483 0.        ]]
```

### 출력 해석

```
           disk      format    hard      how       my        problems   to
문장 0:   0.3347    0.3347    0.3347   0.4704   0.4704     0.0        0.4704
문장 1:   0.3541    0.7082    0.3541   0.0      0.0        0.4977     0.0
```

- `disk`, `hard`: 두 문장 모두 1번씩 등장 → TF-IDF 값 비슷 (약 0.33~0.35)
- `format`: 문장 1에서 **2번** 등장 → TF 높음 → 문장 1에서의 TF-IDF가 더 높음 (0.3347 → 0.7082)
- `how`, `my`, `to`: 문장 0에만 등장 → 문장 1에서는 0
- `problems`: 문장 1에만 등장 → 문장 0에서는 0

---

## 🔄 fit_transform vs transform

|메서드|설명|사용 시점|
|---|---|---|
|`fit_transform(X)`|어휘집 학습 + 변환 동시 수행|학습 데이터에 사용|
|`transform(X)`|기존 어휘집으로 변환만 수행|테스트/새 데이터에 사용|

> 새로운 문장을 변환할 때는 반드시 `transform`만 써야 함. `fit_transform`을 쓰면 어휘집이 새로 갱신되어 인덱스가 달라질 수 있음.

---

## ✅ 정리

```
CountVectorizer  → 단어 등장 횟수 기반 → 정수 희소벡터
TfidfVectorizer  → 문서 내 중요도 기반 → 실수 희소벡터
Word2Vec         → 의미적 위치 기반   → 실수 밀집벡터
```

희소벡터 방식(Count, TF-IDF)은 **어떤 단어가 얼마나 나왔는가**에 집중하고, Word2Vec은 **단어의 의미적 관계**를 학습해 비슷한 단어끼리 벡터 공간에서 가깝게 위치시킨다.

---
# 📄 nlp4count_vec.ipynb — 쇼핑몰 리뷰 · CountVectorizer · TfidfVectorizer

---

## 🧠 실습 개요

쇼핑몰 고객 리뷰 14개를 대상으로 `CountVectorizer`와 `TfidfVectorizer`를 적용해 각 리뷰를 숫자 벡터로 변환하고 단어별 중요도를 분석한다.

---

## 📦 사용 라이브러리

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer    # 단어 등장 횟수 기반 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer    # 단어 중요도 기반 벡터화
from sklearn.metrics.pairwise import cosine_similarity         # 문서 간 유사도 계산
```

---

## 📋 데이터 준비

```python
reviews = [
    "배송이 빠르고 포장이 깔끔해서 만족했습니다",
    "상품 품질은 좋았지만 가격이 조금 비쌌습니다",
    "주문한 색상과 다른 제품이 도착했습니다",
    "고객센터 응답이 늦어서 불편했습니다",
    "교환 처리가 빠르게 진행되어 만족했습니다",
    "앱에서 결제가 자꾸 실패해서 주문을 못 했습니다",
    "재구매하고 싶을 만큼 제품이 마음에 들었습니다",
    "배송이 지연되어 선물 날짜를 맞추지 못했습니다",
    "환불 처리가 너무 오래 걸려서 불만족스러웠습니다",
    "제품 설명과 실제 상품이 달라서 실망했습니다",
    "쿠폰 적용이 되지 않아 결제 금액이 다르게 나왔습니다",
    "포장이 훼손되어 상품 일부가 파손된 상태로 도착했습니다",
    "고객센터 상담원이 친절하게 문제를 해결해 주었습니다",
    "반품 신청 과정이 복잡해서 이용하기 어려웠습니다",
]

review_df = pd.DataFrame({
    'review_id': range(1, len(reviews) + 1),
    'review_text': reviews
})
```

---

## 🔢 CountVectorizer 적용

단어가 각 리뷰에 **몇 번 등장했는지** 카운트해서 벡터로 변환.

```python
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(review_df['review_text'])
# fit_transform : 어휘집 학습 + 벡터 변환 동시 수행

count_words = count_vectorizer.get_feature_names_out()
# 어휘집에서 추출된 단어 목록 반환
# ['가격이' '걸려서' '결제' '결제가' '고객센터' ... ]  (총 74개 단어)

count_df = pd.DataFrame(count_matrix.toarray(), columns=count_words)

# 원본 리뷰와 단어 등장 횟수 DataFrame 합치기
count_result = pd.concat([review_df[["review_id", "review_text"]], count_df], axis=1)
```

### 결과 해석

```
review_id  review_text                       가격이  걸려서  결제  ...
0     1     배송이 빠르고 포장이 깔끔해서 만족했습니다     0      0     0  ...
1     2     상품 품질은 좋았지만 가격이 조금 비쌌습니다     1      0     0  ...
```

- 각 셀의 값 = 해당 리뷰에서 해당 단어가 등장한 횟수
- 리뷰 2번에서 '가격이'가 1번 등장 → `가격이` 열의 값이 1
- 등장하지 않은 단어는 모두 0 → 희소행렬(Sparse Matrix)

> **전체 어휘집 크기**: 14개 리뷰에서 총 **74개** 고유 단어 추출 리뷰가 많아질수록 어휘집 크기(= 벡터 차원)도 커짐

---

## 📊 TfidfVectorizer 적용

단순 등장 횟수가 아닌, **이 문서에서 얼마나 중요한 단어인지** 가중치를 부여.

```python
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(review_df['review_text'])

# ⚠️ 어휘집은 count_vectorizer에서 이미 만든 걸 재사용 (같은 단어 순서 유지)
tfidf_words = count_vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_words)

tfidf_result_df = pd.concat([review_df[["review_id", "review_text"]], tfidf_df], axis=1)
```

### 결과 해석

```
review_id  review_text                       가격이    걸려서    결제   ...
0     1     배송이 빠르고 포장이 깔끔해서 만족했습니다  0.000000  0.000000  0.000000
1     2     상품 품질은 좋았지만 가격이 조금 비쌌습니다  0.417061  0.000000  0.000000
...
8     9     환불 처리가 너무 오래 걸려서 불만족스러웠습니다  0.000000  0.417061  0.000000
10   11     쿠폰 적용이 되지 않아 결제 금액이 다르게 나왔습니다  0.000000  0.000000  0.353553
```

- 값이 클수록 해당 문서에서 그 단어의 중요도가 높음
- 리뷰 2번 '가격이' = 0.417 → 이 리뷰에서 꽤 중요한 단어
- 리뷰 11번 '결제' = 0.353 → 두 단어('결제가'도 함께 등장)가 공유하므로 값이 분산되어 낮아짐

---

## 🔄 CountVectorizer vs TfidfVectorizer 비교

|항목|CountVectorizer|TfidfVectorizer|
|---|---|---|
|출력값|정수 (등장 횟수)|실수 (중요도 가중치)|
|공통 단어 처리|많이 나올수록 값이 높아짐|여러 문서에 나오면 패널티|
|활용|빈도 기반 분석|문서 간 핵심 키워드 추출|

> **예시**: '했습니다'는 거의 모든 리뷰에 등장
> 
> - CountVectorizer → 많이 나와서 값이 높음
> - TfidfVectorizer → 모든 문서에 등장하므로 IDF가 낮아져 중요도 낮게 반영됨

---

## 💡 다음 단계 (예정)

- `cosine_similarity`를 이용해 리뷰 간 유사도 계산
- 비슷한 리뷰끼리 묶어 카테고리 분류 (긍정/부정/배송/환불 등)
