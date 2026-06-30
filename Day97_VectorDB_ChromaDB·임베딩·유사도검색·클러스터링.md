# Day97_VectorDB_ChromaDB·임베딩·유사도검색·클러스터링

## 📅 2026-06-30

---
# 📄 vecdb2.ipynb — ChromaDB · CRUD · PersistentClient

## 🧠 개념 정리

ChromaDB는 **벡터 데이터베이스**로, 텍스트를 임베딩(숫자 벡터)으로 변환해 저장하고 유사도 기반으로 검색할 수 있게 해주는 저장소다. 이번 노트북은 ChromaDB의 가장 기본적인 CRUD(Create-Read-Update-Delete) 동작을 `add`, `get`, `update`, `delete` 4개 메서드로 직접 연습한다.

### PersistentClient vs Client

`chromadb.Client()`는 메모리에서만 동작해 세션이 끝나면 데이터가 사라지지만, `PersistentClient(path=...)`는 지정한 경로(`./chroma`)에 데이터를 디스크로 저장해 다음 실행에도 유지된다. 실무/실습에서는 `PersistentClient`를 쓰는 게 일반적이다.

### embedding_function의 역할

`collection.add()`에 `documents`(원문 텍스트)를 넘기면, 컬렉션에 등록된 `embedding_function`이 내부적으로 자동 호출되어 임베딩 벡터로 변환 후 저장한다. 즉 사용자가 직접 임베딩을 만들 필요 없이 텍스트만 넘기면 된다 — 이 방식은 다음 노트(vecdb3embedding)에서 정리한 "방법2"에 해당한다.

### get_or_create_collection

이름이 같은 컬렉션이 이미 있으면 가져오고, 없으면 새로 생성한다. 매번 실행할 때마다 컬렉션이 중복 생성되는 것을 방지해준다.

---

### 1. 패키지 설치 및 클라이언트 준비

```python
# Chromedb에 add, update, delete, get 연습
!pip install chromadb sentence-transformers

import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 클라이언트 및 임베딩 함수
# - SentenceTransformerEmbeddingFunction: 텍스트 -> 벡터 변환을 자동으로 수행하는 함수 객체
# - all-MiniLM-L6-v2: 가볍고 빠른 범용 영어권 임베딩 모델 (384차원 벡터 출력)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# PersistentClient: 데이터를 디스크(.chroma 폴더)에 영구 저장
# (cf. chromadb.Client()는 메모리 전용이라 세션 종료 시 데이터 소실)
client = PersistentClient(path=".chroma")

# get_or_create_collection: "test"라는 이름의 컬렉션이 있으면 불러오고, 없으면 새로 생성
# embedding_function을 등록해두면 add() 시 documents가 자동으로 임베딩됨
collection = client.get_or_create_collection(name="test", embedding_function=embedding_fn)
```

### 2. 데이터 저장 (Create)

```python
# 데이터 저장
# - documents: 원문 텍스트 (embedding_function에 의해 자동으로 벡터화됨)
# - metadatas: 각 문서에 딸린 부가 정보. 추후 where 조건 필터링에 사용 가능
# - ids: 문서를 식별하는 고유 키. 동일 id로 add() 재호출 시 에러 발생 (update를 써야 함)
collection.add(
    documents=[
        "문서1 : 인공지능 기술이 난리가 났네",
        "문서2 : 언제 다 공부하나"
    ],
    metadatas=[
        {"tag": "mes1"},
        {"tag": "mes2"}
    ],
    ids=[
        "doc1",
        "doc2"
    ]
)
```

### 3. 데이터 조회 (Read)

```python
# 데이터 조회
print('\n전체 문서 조회')

# include 파라미터로 어떤 필드를 가져올지 명시적으로 지정해야 함
# (ChromaDB는 기본적으로 embeddings를 응답에서 생략하므로 필요시 직접 명시)
results = collection.get(include=['documents', 'metadatas', 'embeddings'])
# print(results)

# ids는 include 여부와 무관하게 항상 반환됨
for doc, meta, emb, id in zip(
    results['documents'], results['metadatas'], results['embeddings'], results['ids']):
    print(f'id : {id}')
    print(f'documents : {doc}')
    print(f'metadatas : {meta}')
    print(f'embeddings : {len(emb)}, {emb[:5]}')  # 384차원 벡터 중 앞 5개만 출력
    print('-' * 30)
```

**실행 결과**

```
전체 문서 조회
id : doc1
documents : 문서1 : 인공지능 기술이 난리가 났네
metadatas : {'tag': 'mes1'}
embeddings : 384, [-0.03150008  0.00852532  0.03011171 -0.03096524  0.00244242]
------------------------------
id : doc2
documents : 문서2 : 언제 다 공부하나
metadatas : {'tag': 'mes2'}
embeddings : 384, [ 0.00367608  0.06078542  0.05154031 -0.03263206  0.00723875]
------------------------------
```

→ `all-MiniLM-L6-v2` 모델이 각 문서를 **384차원 벡터**로 변환해 저장한 것을 확인할 수 있다.

### 4. 데이터 수정 (Update)

```python
# 데이터 수정
# update()는 해당 ids가 존재하면 내용을 덮어쓰고, 존재하지 않으면 에러 발생
# (= upsert와 다름: upsert는 없으면 새로 생성, update는 반드시 기존 데이터가 있어야 함)
collection.update(
    ids=['doc2'],
    documents=['문서2 : 내용을 일부 수정'],
    metadatas=[{'tag': 'edited message'}]
)

print('\n수정 후 문서 조회')

# where 절: 메타데이터 조건으로 필터링하여 조회
# tag가 'edited message'인 문서만 가져옴 (방금 수정한 doc2가 해당)
upresult = collection.get(where={'tag': 'edited message'}, include=['documents', 'metadatas'])

for doc, meta in zip(upresult['documents'], upresult['metadatas']):
    print(f'documents : {doc}')
    print(f'metadatas : {meta}')
```

**실행 결과**

```
수정 후 문서 조회
documents : 문서2 : 내용을 일부 수정
metadatas : {'tag': 'edited message'}
```

→ documents와 metadatas가 동시에 갱신되었고, `where` 필터로 정확히 매칭됨을 확인.

### 5. 데이터 삭제 (Delete)

```python
# 데이터 삭제
# delete()는 ids 또는 where 조건 둘 중 하나로 삭제 대상을 지정할 수 있음
collection.delete(ids=['doc1'])
# collection.delete(where={'tag': 'mes1'})  # 메타데이터 조건으로 삭제하는 대안 방식

print('\n삭제 후 문서 조회')
delresult = collection.get(include=['documents', 'metadatas'])

for doc, meta in zip(delresult['documents'], delresult['metadatas']):
    print(f'documents : {doc}')
    print(f'metadatas : {meta}')
```

**실행 결과**

```
삭제 후 문서 조회
documents : 문서2 : 내용을 일부 수정
metadatas : {'tag': 'edited message'}
```

→ `doc1`이 삭제되어 남은 문서는 (수정된) `doc2` 하나뿐. 삭제와 수정이 정상적으로 누적 반영된 것을 확인할 수 있다.

---

## 요약

|메서드|역할|비고|
|---|---|---|
|`add()`|신규 문서 추가|이미 존재하는 id면 에러|
|`get()`|조회|`include`로 필드 지정, `where`로 필터링|
|`update()`|기존 문서 수정|존재하지 않는 id면 에러 (upsert와 구분)|
|`delete()`|문서 삭제|`ids` 또는 `where` 조건으로 지정 가능|

---
# 📄 vecdb3embedding.ipynb — ChromaDB · Embedding · SentenceTransformer

## 🧠 개념 정리

ChromaDB는 임베딩을 직접 생성하는 모델이 아니라 **이미 만들어진 임베딩(벡터)을 저장하고 검색하는 데이터베이스**다. 따라서 ChromaDB에 데이터를 넣기 전에는 반드시 "임베딩 생성"이라는 선행 단계가 필요하고, 이 임베딩을 만드는 방법은 한 가지가 아니라 여러 가지다. 이 노트북은 그 방법들을 4가지로 비교한다.

### 4가지 임베딩 생성 방식 비교

|방법|임베딩을 만드는 주체|특징|
|---|---|---|
|방법1|컬렉션 내부의 기본 임베딩 함수를 직접 꺼내 호출|가능은 하지만 캡슐화를 깨는 방식이라 비권장 (`_embedding_function`은 내부 속성, private 컨벤션)|
|방법2|`embedding_functions`를 컬렉션에 등록 → `add`/`upsert` 시 자동 호출|**ChromaDB 표준 방식.** 텍스트만 넘기면 알아서 벡터화됨|
|방법3|`SentenceTransformer`로 직접 인코딩 후 벡터를 수동으로 넘김|ChromaDB와 완전히 독립적으로 임베딩 단계를 통제하고 싶을 때 사용|
|방법4|Hugging Face의 사전학습 모델(한국어 특화)을 임베딩 함수로 등록|방법2의 변형. 모델만 한국어 특화 모델로 교체|

방법1과 방법2의 차이는 "임베딩 함수를 컬렉션이 들고 있느냐"에 있다. 방법2처럼 컬렉션 생성 시 `embedding_function`을 등록해두면, 이후 `add()`나 `upsert()` 호출 시 원문 텍스트만 넘겨도 컬렉션이 알아서 벡터로 변환한다. 반면 방법1은 그 임베딩 함수를 외부로 직접 꺼내와 호출하는 우회적인 방식이라, 정식 API가 아니라 비권장된다.

### upsert란?

`upsert`는 update + insert의 합성어로, 해당 id가 이미 존재하면 갱신(update)하고 존재하지 않으면 새로 추가(insert)한다. 같은 id로 여러 번 실행해도 에러 없이 안전하게 재실행할 수 있어, 노트북처럼 반복 테스트하는 상황에 적합하다 (cf. vecdb2에서 다룬 `update`는 존재하지 않는 id에 대해 에러를 낸다).

### all-MiniLM-L6-v2 vs ko-sroberta-multitask

`all-MiniLM-L6-v2`는 영어 중심으로 학습된 범용 경량 모델이고, `jhgan/ko-sroberta-multitask`는 한국어 문장 의미 파악에 특화된 모델이다. 이 노트북의 예시 문장이 한국어("사과는 과일이야" 등)이므로, 실제 의미 기반 유사도 검색 품질은 방법4가 더 우수할 가능성이 높다. 임베딩 모델 선택은 "어떤 언어로 검색할 것인가"를 가장 먼저 고려해야 한다는 점을 보여주는 예시다.

---

### 0. 준비

```python
# Embedding 방법 정리
# ChromaDB는 임베딩 모델이 아니라 임베딩 결과를 저장하는 DB
# 임베딩이 선행 - 방법이 여러가지

!pip install chromadb sentence-transformers
```

```python
import chromadb
from sentence_transformers import SentenceTransformer

# PersistentClient: 디스크에 영구 저장되는 클라이언트
client = chromadb.PersistentClient(path='.chroma_db')

# 4가지 방법 모두 동일한 예시 문장으로 비교 (한국어 문장)
texts = ['사과는 과일이야', '고양이는 동물이야']
```

### 방법1 — 컬렉션 내부 임베딩 함수를 직접 꺼내 쓰기 (비권장)

```python
print('방법1 : 이런 것도 가능하나 비권장')
# ChromaDB 내부 임베딩 함수를 직접 꺼내 사용하는 방식
# get_or_create_collection에 embedding_function을 따로 지정하지 않으면
# ChromaDB 기본 임베딩 함수(all-MiniLM-L6-v2 기반)가 자동으로 들어감
collection1 = client.get_or_create_collection(name="test")

# _embedding_function: 언더스코어(_)로 시작 -> 비공개(private) 속성이라는 관례
# 정식 API가 아니므로 직접 접근해서 쓰는 건 권장되지 않음
embedding_fn1 = collection1._embedding_function
embeddings1 = embedding_fn1(texts)
print(len(embeddings1), len(embeddings1[0]))   # 2개 문장, 384차원
print(embeddings1[0][:5])

# upsert: update + insert. id가 있으면 수정, 없으면 신규 추가 (에러 없이 안전)
collection1.upsert(
    documents=texts,
    embeddings=embeddings1,  # 직접 만든 임베딩을 명시적으로 전달
    ids=["id1", "id2"]
)
```

### 방법2 — 임베딩 함수를 컬렉션에 등록 (표준 방식)

```python
print('방법2 : ChromaDB에서 가장 일반적인 방법')
# ChromaDB에 임베딩 함수를 등록해 자동 임베딩하는 방식
from chromadb.utils import embedding_functions

embedding_fn2 = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
embeddings2 = embedding_fn2(texts)
print(len(embeddings2), len(embeddings2[0]))
print(embeddings2[0][:5])

# 컬렉션 생성 시 embedding_function을 등록해두면
# 이후 upsert/add에서 documents만 넘겨도 자동으로 벡터화됨 (embeddings 인자 생략 가능)
collection2 = client.get_or_create_collection(name="test2", embedding_function=embedding_fn2)
collection2.upsert(
    documents=texts,
    ids=["id1", "id2"]
    # embeddings를 따로 넘기지 않아도 컬렉션이 등록된 embedding_fn2로 자동 변환
)
```

### 방법3 — SentenceTransformer로 직접 인코딩

```python
print('방법3 : ChromaDB와 별개로 SentenceTransformer 모델을 직접 사용해 임베딩을 만드는 방법')
# SentenceTransformer로 직접 임베딩
# ChromaDB의 embedding_functions 래퍼를 거치지 않고, 라이브러리를 직접 호출
model3 = SentenceTransformer("all-MiniLM-L6-v2")
embeddings3 = model3.encode(texts).tolist()  # numpy array -> list 변환 (ChromaDB는 list 형태를 받음)
print(len(embeddings3), len(embeddings3[0]))
print(embeddings3[0][:5])

# 이 컬렉션은 embedding_function을 등록하지 않았으므로
# documents만 넘기면 에러가 남 -> embeddings를 반드시 직접 넘겨줘야 함
collection3 = client.get_or_create_collection(name="test3")
collection3.upsert(
    documents=texts,
    embeddings=embeddings3,
    ids=["id1", "id2"]
)
```

### 방법4 — Hugging Face 한국어 사전학습 모델 사용

```python
print('방법4 : Hugging Face의 사전학습 모델을 로컬에서 사용하는 방법')
# 방법2와 구조는 동일하나, 모델만 한국어 특화 모델로 교체
# jhgan/ko-sroberta-multitask: 한국어 문장 의미 파악에 특화된 sentence-transformers 모델
embedding_fn4 = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='jhgan/ko-sroberta-multitask'
)
embeddings4 = embedding_fn4(texts)
print(len(embeddings4), len(embeddings4[0]))
print(embeddings4[0][:5])

collection4 = client.get_or_create_collection(name="test4", embedding_function=embedding_fn4)
collection4.upsert(
    documents=texts,
    ids=["id1", "id2"]
)
```

---

## 요약

- ChromaDB는 벡터를 **저장/검색**할 뿐, 임베딩 자체는 외부(sentence-transformers, HF 모델 등)에서 만들어야 한다.
- 실무 표준은 **방법2/방법4**처럼 `embedding_function`을 컬렉션에 등록하는 방식이다 — 코드가 간결해지고, `add`/`upsert` 시 documents만 넘기면 된다.
- 한국어 데이터를 다룰 땐 영어 기반 모델(`all-MiniLM-L6-v2`)보다 한국어 특화 모델(`ko-sroberta-multitask` 등)이 의미 유사도 측면에서 유리하다.
- `upsert`는 id 존재 여부와 무관하게 안전하게 실행 가능해 반복 테스트에 적합하다 (cf. `update`는 id가 없으면 에러).

---
# 📄 vecdb4query.ipynb — ChromaDB · Vector Search · Distance Metric

## 🧠 개념 정리

ChromaDB에 데이터를 저장하는 것만으로는 충분하지 않다. 진짜 가치는 **쿼리(질문) 문장을 벡터로 변환한 뒤, 저장된 문서들 중 가장 가까운 벡터를 찾아내는 검색 단계**에서 나온다. 이 노트북은 문서 저장부터 유사 문장 검색까지의 전체 흐름을 다룬다.

### hnsw:space — 거리 측정 방식

컬렉션 생성 시 `metadata={'hnsw:space': 'l2'}`로 벡터 간 거리를 어떻게 계산할지 지정한다. 세 가지 옵션이 있다.

|옵션|의미|해석|
|---|---|---|
|`l2`|유클리드 거리|값이 작을수록(0에 가까울수록) 유사|
|`cosine`|코사인 거리 (1 - 코사인 유사도)|값이 작을수록 유사, 벡터의 방향만 비교 (크기 무시)|
|`ip`|Inner Product (내적)|임베딩이 정규화되어 있을 때 유사도와 직결|

이 노트북은 `l2`를 사용했으므로, 검색 결과의 `distance` 값이 작을수록 의미적으로 더 유사한 문장이라는 뜻이다.

### query() vs get()

`get()`은 `ids`나 `where` 조건으로 정확히 일치하는 데이터를 가져오는 반면, `query()`는 쿼리 벡터와 저장된 벡터들 간의 **거리를 계산해 가장 가까운 n개를 순위별로 반환**한다. 즉 `get`은 "찾기", `query`는 "유사도 검색"이라는 역할 차이가 있다.

### 임베딩 자동화 없이 수동으로 다루는 이유

이 노트북은 컬렉션에 `embedding_function`을 등록하지 않고, `SentenceTransformer` 모델을 직접 호출해서 만든 벡터를 `add()`와 `query()`에 명시적으로 넘긴다. vecdb3embedding에서 정리한 "방법3"에 해당하는 흐름으로, 저장 시점과 검색 시점 모두 같은 모델을 일관되게 써야 한다는 점이 핵심이다 (저장할 때와 검색할 때 서로 다른 임베딩 모델을 쓰면 벡터 공간이 달라 비교 자체가 무의미해진다).

---

### 1. 클라이언트 및 컬렉션 생성

```python
# ChromaDB에 저장 후 유사 문장 검색
!pip install chromadb sentence-transformers

import chromadb
from sentence_transformers import SentenceTransformer

# PersistentClient: 지정 경로에 디스크 저장 (중첩 경로도 자동 생성됨)
client = chromadb.PersistentClient(path=".aa/bb/ccdb")

# hnsw:space: 벡터 간 거리 계산 방식 지정
# 'l2'(유클리드) / 'cosine'(코사인) / 'ip'(내적) 중 선택
# l2, cosine은 값이 작을수록 유사, ip는 값이 클수록 유사(정규화된 벡터 기준)
collection = client.get_or_create_collection(name="mytest",
                        metadata={'hnsw:space': 'l2'})
```

### 2. 검색 대상 문서 준비

```python
texts = [
    "Apple is a fruit",
    "Python is a programming language",
    "The sun rises in the east",
    "I love to eat mangoes",
]

# ids는 문자열이어야 하므로 정수 인덱스를 str()로 변환
ids = [str(i) for i in range(len(texts))]
print(ids)   # ['0', '1', '2', '3']
```

### 3. SentenceTransformer로 직접 임베딩 생성

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
print(model.get_sentence_embedding_dimension())  # 384
print(model)
print(model.encode(texts).shape)  # (4, 384) -> 문장 4개, 각 384차원 벡터

# numpy array는 ChromaDB가 직접 받지 못하므로 .tolist()로 list[list[float]] 변환 필요
embeddings = model.encode(texts).tolist()
print(embeddings)
```

**실행 결과 (요약)**

```
384
SentenceTransformer(
  (0): Transformer(...)
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', ...})
  (2): Normalize({})
)
(4, 384)
```

→ 모델 구조를 보면 Transformer → Pooling(mean) → Normalize 3단계로 구성되어 있다. 토큰별 벡터를 평균 풀링(mean pooling)해서 문장 전체를 하나의 384차원 벡터로 압축하고, 마지막에 정규화까지 거친다.

### 4. 벡터 DB에 저장

```python
# vectordb에 저장
# documents(원문), ids(식별자), embeddings(이미 계산된 벡터)를 함께 저장
# embedding_function이 등록되지 않은 컬렉션이므로 embeddings를 직접 넘겨야 함
collection.add(documents=texts, ids=ids, embeddings=embeddings)
```

### 5. 저장된 데이터 조회

```python
# vectordb의 자료 조회
record = collection.get(ids=["0"], include=['embeddings', 'documents'])
print('조회된 문서 : ', record['documents'][0])
print('벡터 (앞 10개) : ', record['embeddings'][0][:10])
```

**실행 결과**

```
조회된 문서 :  Apple is a fruit
벡터 (앞 10개) :  [ 0.07331283  0.04409566  0.00852459  0.04129385 -0.03418821  0.03472484
  0.08898374 -0.05657282  0.05579079  0.03283169]
```

→ id `"0"`에 해당하는 `"Apple is a fruit"` 문서가 정상 저장·조회됨을 확인.

### 6. 유사 문장 검색 (핵심)

```python
# vectordb의 자료 유사 문장 검색
query_data = "What is Python?"
# 검색 쿼리도 반드시 저장할 때와 동일한 모델로 임베딩해야 같은 벡터 공간에서 비교 가능
query_vector = model.encode([query_data]).tolist()
print(query_vector)

result = collection.query(
    query_embeddings=query_vector,
    n_results=2,   # 가장 가까운 상위 2개만 반환
    include=['documents', 'distances']   # distances - query_vector와의 거리. 0에 근사하면 유사함
)

print('유사한 문장 검색 결과:')
for doc, dist in zip(result['documents'][0], result['distances'][0]):
    print(f'- 문장:{doc} (유사도거리:{dist:.4f})')
```

**실행 결과**

```
유사한 문장 검색 결과:
- 문장:Python is a programming language (유사도거리:0.2156)
- 문장:Apple is a fruit (유사도거리:1.8316)
```

→ `"What is Python?"`이라는 질문에 대해, 가장 가까운 문서는 `"Python is a programming language"`(거리 0.2156)로 정확히 의미적으로 연관된 문장이 1순위로 검색됐다. 반면 관련 없는 `"Apple is a fruit"`은 거리 1.8316으로 훨씬 멀게 측정됐다. l2 거리가 작을수록 유사하다는 원리가 실제 검색 결과로 입증된 셈이다.

---

## 요약

- 벡터 검색의 핵심 흐름은 **쿼리 텍스트 → 임베딩 → 저장된 벡터들과 거리 계산 → 가까운 순으로 정렬**이다.
- `hnsw:space`로 거리 측정 방식을 선택하며, `l2`/`cosine`은 값이 작을수록 유사, `ip`는 보통 클수록 유사하다(정규화 전제).
- 저장 시점과 검색 시점에 **반드시 동일한 임베딩 모델**을 사용해야 벡터 비교가 의미를 갖는다.
- `query()`는 유사도 기반 랭킹 검색, `get()`은 조건 기반 정확 조회 — 역할이 다르다.
- 이번 실습에서 `"What is Python?"` 쿼리가 `"Python is a programming language"`를 정확히 1순위로 찾아낸 것은, 단순 키워드 매칭이 아니라 **의미 기반 검색**이 실제로 작동함을 보여주는 좋은 예시다.

---
# 📄 vecdb5cluster.ipynb — Embedding · KMeans · Silhouette Score · PCA 시각화

## 🧠 개념 정리

이 노트북은 임베딩을 **비지도학습(KMeans)으로 클러스터링**하고, 그 결과를 **PCA로 2차원 축소해 시각화**하는 흐름을 다룬다. ChromaDB 저장/검색을 다룬 앞선 노트들과 달리, 임베딩 자체를 분석 대상으로 삼아 "의미적으로 비슷한 문장끼리 자동으로 묶이는지"를 확인하는 실습이다.

### 왜 임베딩을 클러스터링하는가

임베딩은 문장의 의미를 고차원 벡터(여기선 384차원)로 표현한 것이라, 의미가 비슷한 문장끼리는 벡터 공간에서 서로 가깝게 위치한다. 이 성질을 이용해 KMeans 같은 클러스터링 알고리즘을 적용하면 사람이 라벨을 달지 않아도 "과일 이야기", "파이썬 이야기", "스포츠 이야기" 같은 주제별 군집이 자동으로 형성된다.

### 실루엣 점수(Silhouette Score)로 k값 정하기

KMeans는 클러스터 개수(k)를 미리 정해줘야 하는데, 몇 개가 적절한지는 보통 데이터를 보기 전엔 알 수 없다. 실루엣 점수는 "같은 군집 내 데이터끼리는 얼마나 가깝고, 다른 군집과는 얼마나 떨어져 있는지"를 -1~1 사이 값으로 측정한다. 1에 가까울수록 군집이 잘 나뉜 것이고, 보통 여러 k값을 시도해서 가장 높은 점수가 나오는 k를 선택한다.

### PCA로 시각화하는 이유

임베딩은 384차원이라 사람이 직접 눈으로 볼 수 없다. PCA(주성분분석)는 고차원 데이터에서 분산이 가장 큰 방향(주성분)을 찾아 2차원으로 압축해주는 차원 축소 기법이다. 정보 손실은 있지만, "클러스터링이 대략적으로 잘 됐는지"를 시각적으로 확인하는 용도로는 충분하다.

---

### 1. 패키지 설치 (오타 주의)

```python
# 임베딩 시각화
#
! pip install senetence-transformers
```

> ⚠️ `senetence-transformers`는 오타다 (sentence-transformers의 e와 n이 뒤바뀜). 실제로 실행 결과에도 `ERROR: Could not find a version that satisfies the requirement`가 떴다 — 패키지 자체가 존재하지 않는 이름이라 설치가 실패했다. 다행히 이미 환경에 `sentence-transformers`가 설치되어 있어서 다음 셀은 정상 동작했지만, 새 환경에서 처음 실행한다면 아래처럼 고쳐야 한다.
> 
> ```python
> !pip install sentence-transformers
> ```

### 2. 예시 문장 준비 및 임베딩

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# 의도적으로 3가지 주제(과일, 파이썬, 스포츠)가 섞인 12개 문장 준비
# -> 클러스터링이 이 주제 구분을 알아서 찾아내는지 확인하는 것이 실습 목적
texts = [
    "나는 사과를 좋아해",
    "바나나는 내가 제일 좋아하는 과일이야",
    "파이썬은 프로그래밍 언어",
    "나는 가끔 파이썬으로 소스를 만들어",
    "사과와 바나나는 모두 맛있어",
    "파이썬 코딩은 즐거워",
    "나는 망고 스무디를 즐겨 마셔",
    "과일은 건강한 간식이야",
    "나는 열대 과일이 좋아",
    "운동은 역시 축구야",
    "재미있는 야구 경기를 보러 가야지",
    "야구 만세",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)  # (12, 384) 형태의 numpy array
print(embeddings[:3])
```

### 3. 실루엣 점수로 최적 k 탐색 + KMeans 클러스터링

```python
# KMeans Clustering : k값 ?
# 클러스터 수 찾기(실루엣 기법)
from sklearn.metrics import silhouette_score

# k=2부터 5까지 시도하며 각각의 실루엣 점수 비교
for k in range(2, 6):
  kmeans = KMeans(n_clusters=k, random_state=42)
  labels = kmeans.fit_predict(embeddings)
  score = silhouette_score(embeddings, labels)
  print(f"k={k}, Silhouette Score: {score: .4f}")

# k=2, Silhouette Score:  0.2042
# k=3, Silhouette Score:  0.2077  <- 가장 높은 점수
# k=4, Silhouette Score:  0.1126
# k=5, Silhouette Score:  0.1254

# 실루엣 점수가 가장 높았던 k=3을 최종 클러스터 개수로 채택
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)
# print(labels)

print('유사도 기반 문장 클러스터링 결과 :')
for idx, (text, label) in enumerate(zip(texts, labels)):
  print(f"[군집 {label}] {text}")

print('\n군집 결과')
from collections import defaultdict
clusters = defaultdict(list)
for text, label in zip(texts, labels):
  clusters[label].append(text)

for cluster_id, docs in clusters.items():
  print(f'\n---cluster {cluster_id} ---')
  for d in docs:
    print(d)
```

**실행 결과**

```
k=2, Silhouette Score:  0.2042
k=3, Silhouette Score:  0.2077
k=4, Silhouette Score:  0.1126
k=5, Silhouette Score:  0.1254

---cluster 0 ---
나는 사과를 좋아해
바나나는 내가 제일 좋아하는 과일이야
사과와 바나나는 모두 맛있어
나는 열대 과일이 좋아
야구 만세

---cluster 2 ---
파이썬은 프로그래밍 언어
나는 가끔 파이썬으로 소스를 만들어
파이썬 코딩은 즐거워

---cluster 1 ---
나는 망고 스무디를 즐겨 마셔
과일은 건강한 간식이야
운동은 역시 축구야
재미있는 야구 경기를 보러 가야지
```

**결과 해석**

군집 2(파이썬 관련 3문장)는 100% 정확하게 묶였다. 반면 군집 0과 군집 1은 과일/스포츠 주제가 서로 살짝 섞였는데, 특히 `"야구 만세"`가 스포츠가 아닌 군집 0(과일류)에 들어간 점이 눈에 띈다. 이는 문장이 매우 짧아서("야구 만세" 단 두 단어) 임베딩이 주제를 충분히 담아내지 못했기 때문일 가능성이 높다 — 임베딩 기반 클러스터링은 문장이 너무 짧거나 모호하면 정확도가 떨어질 수 있다는 점을 보여주는 사례다.

### 4. PCA로 2차원 시각화

```python
# 군집 결과 시각화
# PCA를 사용해 차원 축소 후 시각화
# 각 클러스터별 대표 문장 출력
# !pip install koreanize-matplotlib

import matplotlib.pyplot as plt
import koreanize_matplotlib   # matplotlib 한글 폰트 깨짐 방지용 라이브러리
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)     # 384 -> 2차원으로 축소

plt.figure(figsize=(8,6))
colors = ['red', 'green', 'blue', 'orange', 'purple']
for i in range(n_clusters):
  cluster_points = reduced[labels == i]
  plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
              color=colors[i % len(colors)], label=f'군집 {i}')

plt.title('문장 군집화(PCA 시각)')
plt.xlabel('PCA1')
plt.xlabel('PCA2')   # ⚠️ 오타: ylabel이어야 함. 그대로 두면 PCA1만 x축에 표시되고 y축 라벨은 비게 됨
plt.legend()
plt.grid(True)
plt.show()
```

> ⚠️ 위 코드의 `plt.xlabel('PCA2')`는 `plt.ylabel('PCA2')`로 고쳐야 y축 라벨이 정상적으로 표시된다. 원본 그대로 두면 x축 라벨이 'PCA2'로 덮어써져서 'PCA1'은 사라지고, y축은 라벨이 비어있게 된다.

> 올바른 버전:
> 
> ```python
> plt.xlabel('PCA1')
> plt.ylabel('PCA2')
> ```

**실제 실행 결과**

<img src="images/vecdb5cluster.png" width="500"/>

위 그래프에서 실제로 x축 라벨이 'PCA2'로 찍혀 있고 y축 라벨은 비어있는 것을 확인할 수 있다 — 코드의 오타가 그대로 결과물에 반영된 사례다.

그래프 자체의 군집 분리는 시각적으로도 뚜렷하다. 파란색(군집 2, 파이썬 관련 문장들)은 좌측 상단에 확실히 분리되어 있고, 초록색(군집 1)과 빨간색(군집 0)도 대체로 영역이 나뉘어 있다. 다만 빨간 점 하나가 군집 1 영역과 가까운 위치(약 0.40, 0.18 부근)에 찍혀 있는데, 이게 앞서 언급한 `"야구 만세"` 문장일 가능성이 높다 — 짧은 문장이라 의미 임베딩이 약해 경계에 걸친 위치에 놓인 것으로 보인다.

---

## 요약

- KMeans로 임베딩을 클러스터링하기 전, **실루엣 점수**로 여러 k값을 비교해 최적의 군집 개수를 정하는 것이 일반적인 절차다 (이번 실습에선 k=3이 최고점).
- 클러스터링 결과는 완벽하지 않을 수 있다 — 특히 문장이 짧거나 모호하면("야구 만세") 의미 임베딩이 약해져 잘못된 군집에 배정될 수 있다.
- 고차원 임베딩(384차원)은 사람이 직접 볼 수 없으므로, **PCA로 2차원 축소** 후 산점도로 시각화해 군집이 잘 나뉘었는지 직관적으로 확인한다.
- 코드 작성 시 `plt.xlabel`과 `plt.ylabel`을 혼동하지 않도록 주의가 필요하다.