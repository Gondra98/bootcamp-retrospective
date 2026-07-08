# Day102_RAG파이프라인_임베딩·벡터검색·LLM응답

## 📅 2026-07-08

---
# 📄 rag4text.ipynb — RAG · ChromaDB · LangChain

## 🧠 개념 정리

**RAG (Retrieval-Augmented Generation)** 질문에 답하기 전에 관련 문서를 먼저 "검색(Retrieval)"하고, 그 문서를 근거로 LLM이 "답변을 생성(Generation)"하는 전체 방식. LLM이 모르는 최신/전문 정보도 검색된 문서를 통해 답변에 반영할 수 있게 해줌.

**LangChain** RAG 파이프라인(문서 로딩 → 분할 → 임베딩 → 검색 → 프롬프트 구성)을 부품처럼 조립하기 쉽게 해주는 프레임워크.

**LangGraph** LangChain보다 더 복잡한 흐름(조건 분기, 반복, 여러 에이전트 협업 등)을 그래프의 노드와 엣지로 표현해서 조립하는 도구.

**RAG 전체 흐름 (이번 노트 기준)**

```
문서 로딩 → 청크 분할 → 임베딩 벡터화 → ChromaDB 저장
   ↓
질문 → 질문도 임베딩 → 유사도 검색(Retrieval)
   ↓
검색된 문서 + 질문 → 프롬프트 구성 → LLM 응답 생성(Generation)
```

**거리(distance) vs 유사도(similarity)**

- ChromaDB의 `query()`가 반환하는 `distances`는 값이 **작을수록** 더 유사한 문서.
- `normalize_embeddings=True`로 정규화한 벡터끼리 내적(dot product)하면 코사인 유사도와 같아지고, 값이 **클수록** 더 유사.
- 이 노트에서 `distance ≈ 1 - similarity` 관계로 나타남 (예: distance 0.548 ↔ similarity 0.726).

---

## 📌 셀별 코드 + 주석

### 1) 패키지 설치

```python
# RAG : 검색해서 답변하는 전체 방식
# LangChain : 그 방식을 조립하기 쉽게 해 주는 도구
# LangGraph : 그 방식을 그래프와 엣지를 이용해 조립하기 쉽게 해 주는 도구

!pip install openai sentence-transformers chromadb python-dotenv
!pip install -U langchain langchain-community
!pip install langchain-text-splitters
```

> 💡 이전 시도에서 `python -dotenv`(공백 오타), `langchain-splitters`(존재하지 않는 패키지명) 오류를 겪은 뒤 정상화된 버전.

---

### 2) 클라이언트 및 임베딩 모델 초기화

```python
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 OPENAI_API_KEY 등 환경변수 로드

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o-mini"  # 오타(gtpt-4o-mini) 수정된 버전
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 문장 → 384차원 임베딩 벡터 변환 모델
```

> ⚠️ `.env` 파일명이 정확히 `.env`(점 포함)여야 `load_dotenv()`가 인식함. `env`로 저장 시 인식 실패.

---

### 3) 문서 로딩 및 청크 분할 (RAG 1단계: Text → VectorDB 준비)

```python
# 방법1 : 문서 데이터 직접 작성 (리스트로 하드코딩)
# 방법2 : 문서 파일 읽기 (순수 Python, with open)
# 방법3 : 문서 파일 읽기 (LangChain TextLoader, 줄 단위 분리)

# 방법4 : 문서 파일 읽기 (LangChain, chunk 단위 분할) — 실제 사용된 방식
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("foods.txt", encoding="utf-8")
datas = loader.load()  # Document 객체 리스트로 반환 (문자열 아님)

# text 나누기
splitter = CharacterTextSplitter(separator="\n", chunk_size=100, chunk_overlap=0)
spl_docs = splitter.split_documents(datas)  # Document 객체를 여러 조각(chunk)으로 분할

# Document 객체에서 실제 문자열만 추출
documents = [doc.page_content for doc in spl_docs]
print(documents)
print()
print(len(documents), documents[0])
```

**출력 결과:**

```
['김치찌게는 한국의 대표적인 찌게 요리이다.\n된장찌게는 발효된 된장을 이용해 만든다.\n비빔밥은 여러 가지 나물을 비벼서 먹는 밥 요리이다.', '불고기는 앙념한 소고기를 구워 먹는 전통 음식이다.\n삼계탕은 닭에 인삼, 대추 등을 넣고 푹 끓인 보양식이다.']

2 김치찌게는 한국의 대표적인 찌게 요리이다.
된장찌게는 발효된 된장을 이용해 만든다.
비빔밥은 여러 가지 나물을 비벼서 먹는 밥 요리이다.
```

→ `chunk_size=100` 기준으로 foods.txt 5문장이 2개 청크로 묶임 (청크1: 김치찌개~비빔밥 3문장 / 청크2: 불고기~삼계탕 2문장)

> 🧠 `chunk_size=100`이 문장들보다 넉넉해서, `\n` 구분자 기준으로 문장 여러 개가 한 청크에 묶임. `chunk_size`를 줄이면 더 잘게 쪼개짐. ⚠️ `langchain_community`는 deprecated 경고 발생 — 향후 별도 통합 패키지로 마이그레이션 권장.

---

### 4) 임베딩 변환 + ChromaDB 저장

```python
# 임베딩 벡터로 변환
doc_embedding = embedder.encode(documents)
print(doc_embedding[0][:3])  # [0.02435515 0.088004   0.0844731 ]

# ChromaDB에 저장
# 방법1 : 임시저장 (in-memory, 세션 종료 시 소멸)
# chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

# 방법2 : 영구저장 (디스크에 남음) — 실제 사용된 방식
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False,  # 익명 사용 통계 수집 비활성화
))

collection = chroma_client.get_or_create_collection(name="foods")
# collection = chroma_client.create_collection(name="foods")  # 이미 있으면 에러남

# ChromaDB에 저장
for i, (doc, embedding) in enumerate(zip(documents, doc_embedding)):
    collection.add(
        documents=[doc],
        embeddings=[embedding.tolist()],  # numpy 배열 → 리스트로 변환 필요
        ids=[str(i)]
    )
```

> ⚠️ **트러블슈팅 이력**: `chromadb.Client()`를 서로 다른 설정으로 두 번 호출하면 `ValueError: An instance of Chroma already exists ... with different settings` 발생 (싱글톤 구조 때문). 방법1과 방법2 중 하나만 실행하도록 주석 처리해서 해결. ⚠️ 컬렉션 생성(`get_or_create_collection`)을 빼먹으면 `NotFoundError: Collection [foods] does not exist` 발생.

---

### 5) Retrieval 단계 — 질문과 유사한 문서 검색

```python
# 다음 단계 : 질문 -> 관련문서 검색 -> LLM 응답 생성
# RAG (Retrieval 단계 - 관련문서 검색)
query = "한국의 대표적인 찌게 음식 알려줘"  # 사용자의 기본 질문

# 질문과 유사한 문자열을 ChromaDB에서 검색하기 위해 질문도 동일하게 임베딩
query_embedding = embedder.encode(
    [query],
    normalize_embeddings=True  # 코사인 유사도 계산을 위해 벡터 정규화
)[0]

collection = chroma_client.get_or_create_collection(name="foods")

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=len(documents),  # 저장된 문서 전체를 유사도 순으로 반환
    include=["documents", "distances"]
)

print('results :', results)

# 질문에 대한 거리값 목록 출력 (값이 작을수록 유사)
import numpy as np
distances = results['distances'][0]
print("거리(distance)", distances)

# 직접 계산한 유사도 값으로 확인 (정규화된 벡터의 내적 = 코사인 유사도, 값이 클수록 유사)
similarities = []
for doc_id in results['ids'][0]:
    doc_embed = collection.get(
        ids=[doc_id],
        include=['embeddings'],
    )['embeddings'][0]

    doc_embed = np.array(doc_embed, dtype=np.float64)
    sim = np.dot(query_embedding, doc_embed)
    similarities.append(sim)

print("유사도(similarity)", similarities)

ids_list = results['ids'][0]
docs_list = results['documents'][0]
dista_list = results['distances'][0]

for i, doc_id in enumerate(ids_list):
    print(f"\n문서 {i}")
    print(f" - id:{doc_id}")
    print(f" - 문서내용: \n{docs_list[i]}")
    print(f" - 거리: \n{dista_list[i]:.4f}")
    print(f" - 유사도: \n{similarities[i]:.4f}")
    print("-------------")
```

**출력 결과:**

```
results : {'ids': [['0', '1']], 'embeddings': None, 'documents': [['김치찌게는 한국의 대표적인 찌게 요리이다.\n된장찌게는 발효된 된장을 이용해 만든다.\n비빔밥은 여러 가지 나물을 비벼서 먹는 밥 요리이다.', '불고기는 앙념한 소고기를 구워 먹는 전통 음식이다.\n삼계탕은 닭에 인삼, 대추 등을 넣고 푹 끓인 보양식이다.']], 'uris': None, 'included': ['documents', 'distances'], 'data': None, 'metadatas': None, 'distances': [[0.547970175743103, 0.5921519994735718]]}
거리(distance) [0.547970175743103, 0.5921519994735718]
유사도(similarity) [np.float64(0.7260149779122675), np.float64(0.703924079108683)]

문서 0
 - id:0
 - 문서내용: 
김치찌게는 한국의 대표적인 찌게 요리이다.
된장찌게는 발효된 된장을 이용해 만든다.
비빔밥은 여러 가지 나물을 비벼서 먹는 밥 요리이다.
 - 거리: 
0.5480
 - 유사도: 
0.7260
-------------

문서 1
 - id:1
 - 문서내용: 
불고기는 앙념한 소고기를 구워 먹는 전통 음식이다.
삼계탕은 닭에 인삼, 대추 등을 넣고 푹 끓인 보양식이다.
 - 거리: 
0.5922
 - 유사도: 
0.7039
-------------
```

**요약:**

|문서|거리(distance)|유사도(similarity)|
|---|---|---|
|청크1 (김치찌개·된장찌개·비빔밥)|0.5480|0.7260|
|청크2 (불고기·삼계탕)|0.5922|0.7039|

> 질문이 "찌개"를 물었으므로 찌개 관련 문장이 포함된 청크1이 더 높은 유사도로 상위에 옴 — 의도한 대로 동작.

---

### 6) 프롬프트 구성 (검색된 문서로 질문 보강)

```python
# RAG (generation 단계 - 검색된 문서로 질문을 보강해 LLM에서 답변 요청 후 생성)
retrueved_docs_list = results['documents'][0]
retrueved_docs = "\n\n".join(retrueved_docs_list)
print(retrueved_docs)

prompt = f"""
너는 한국 전통 음식 전문가야.
지금부터 사용자 질문에 답변할 때에는 반드시 아래 문서 내용을 참고해.

[참고문서]
{retrueved_docs}

[사용자질문]
{query}

위 참고 문서를 바탕으로 사용자 질문에 10문장 이내로 답해줘
마크다운 기호없이 평문으로 답해줘
"""
print(prompt)
```

**출력 결과:**

```
김치찌게는 한국의 대표적인 찌게 요리이다.
된장찌게는 발효된 된장을 이용해 만든다.
비빔밥은 여러 가지 나물을 비벼서 먹는 밥 요리이다.

불고기는 앙념한 소고기를 구워 먹는 전통 음식이다.
삼계탕은 닭에 인삼, 대추 등을 넣고 푹 끓인 보양식이다.

너는 한국 전통 음식 전문가야.
지금부터 사용자 질문에 답변할 때에는 반드시 아래 문서 내용을 참고해.

[참고문서]
김치찌게는 한국의 대표적인 찌게 요리이다.
된장찌게는 발효된 된장을 이용해 만든다.
비빔밥은 여러 가지 나물을 비벼서 먹는 밥 요리이다.

불고기는 앙념한 소고기를 구워 먹는 전통 음식이다.
삼계탕은 닭에 인삼, 대추 등을 넣고 푹 끓인 보양식이다.

[사용자질문]
한국의 대표적인 찌게 음식 알려줘

위 참고 문서를 바탕으로 사용자 질문에 10문장 이내로 답해줘
마크다운 기호없이 평문으로 답해줘
```

> 🧠 이게 RAG의 핵심 — 검색된 문서(retrieved context)를 프롬프트 안에 직접 넣어서 LLM이 "모르는 내용"이 아니라 "주어진 문서 기반"으로 답하게 강제함.

---

### 7) Generation 단계 — LLM 응답 생성

```python
# LLM 응답 - OpenAI
response = client.responses.create(
    model=model,
    input=prompt
)

import re
text = response.output_text
print(text)
sentences = re.split(r"(?<=[.!? ])\s+", text)  # 마침표, 물음표, 느낌표 뒤에서 문장 분리
for s in sentences:
    s = s.strip()
    if not s:
        continue
    print(s)
```

**출력 결과:**

```
한국의 대표적인 찌개 음식으로는 김치찌개가 있습니다. 김치찌개는 깊은 맛을 가진 국물 요리로, 김치와 다양한 재료를 사용해 끓이는 전통 음식입니다. 또 다른 찌개로는 된장찌개가 있습니다. 된장찌개는 발효된 된장을 이용해 만든 요리로, 구수한 맛이 특징입니다. 이 두 가지 찌개는 한국 식사에서 자주 제공되며, 고기와 채소 등 다양한 재료와 함께 즐길 수 있습니다. 김치찌개는 매콤하게, 된장찌개는 고소하게 먹는 것이 일반적입니다. 한국의 식탁에서 빠질 수 없는 찌개들로, 다양한 변형이 존재합니다. 이들은 쌀밥과 함께 먹으면 더욱 맛있습니다. 한국의 찌개 문화는 깊은 맛과 풍부한 영양을 자랑합니다.

한국의 대표적인 찌개 음식으로는 김치찌개가 있습니다.
김치찌개는 깊은 맛을 가진 국물 요리로, 김치와 다양한 재료를 사용해 끓이는 전통 음식입니다.
또 다른 찌개로는 된장찌개가 있습니다.
된장찌개는 발효된 된장을 이용해 만든 요리로, 구수한 맛이 특징입니다.
이 두 가지 찌개는 한국 식사에서 자주 제공되며, 고기와 채소 등 다양한 재료와 함께 즐길 수 있습니다.
김치찌개는 매콤하게, 된장찌개는 고소하게 먹는 것이 일반적입니다.
한국의 식탁에서 빠질 수 없는 찌개들로, 다양한 변형이 존재합니다.
이들은 쌀밥과 함께 먹으면 더욱 맛있습니다.
한국의 찌개 문화는 깊은 맛과 풍부한 영양을 자랑합니다.
```

→ RAG로 검색된 문서 내용(김치찌개·된장찌개)을 근거로 정확히 답변이 생성됨.

> ⚠️ **트러블슈팅 이력**: `RateLimitError: insufficient_quota` (OpenAI 계정 결제/크레딧 부족) 발생 이력 있음 — 이후 재실행 시 정상 응답 확인. 결제 문제가 재발하면 Ollama(Gemma3) 등 로컬 LLM으로 대체 가능.

---

## 🔧 이번 실습에서 겪은 주요 트러블슈팅 정리

1. `python -dotenv` → `python-dotenv` (공백 오타)
2. `langchain-splitters` → 존재하지 않는 패키지명, `langchain-text-splitters`가 정식명
3. `documents`가 정의되기 전에 자기 자신을 참조하는 `NameError` (방법3 관련)
4. `chromadb.Client()`를 다른 설정으로 두 번 호출 → 싱글톤 충돌 `ValueError`
5. 컬렉션 미생성 상태에서 `get_collection()` 호출 → `NotFoundError`
6. `.env` 파일명에 점(.)이 빠짐 → `load_dotenv()` 인식 실패
7. `OPENAI_API_KEY` 발급 후에도 `insufficient_quota` (결제/크레딧 문제)
8. `gtpt-4o-mini` → `gpt-4o-mini` 오타
9. `annoymized_telemetry` → `anonymized_telemetry` 오타

---

# 📄 rag5text.ipynb — RAG · ChromaDB · cosine distance

## 🧠 개념 정리

**이번 노트의 특징** 지난 `rag4text.ipynb`가 텍스트 파일(`foods.txt`)을 로딩해서 청크로 쪼개는 실습이었다면, 이번 노트는 **RAG 파이프라인 자체를 더 깔끔하게 재구성**한 버전. 문서를 코드에 직접 리스트로 작성하고, ChromaDB 컬렉션 생성 시 **거리 계산 방식(cosine)을 명시적으로 지정**한 점이 다름.

**HNSW / cosine distance space**

```python
collection = chroma_client.create_collection(
    name="animals",
    metadata={"hnsw:space": "cosine"}
)
```

- ChromaDB는 내부적으로 **HNSW(Hierarchical Navigable Small World)** 알고리즘으로 벡터 근사 최근접 검색을 수행함.
- `hnsw:space`는 이때 사용할 거리 계산 방식을 지정하는 옵션. 기본값은 `l2`(유클리드 거리)인데, 문장 임베딩 유사도 비교에는 보통 `cosine`(코사인 거리)이 더 적합해서 명시적으로 지정함.
- `cosine` 거리 = `1 - cosine similarity`. 즉 **거리값이 작을수록 더 유사한 문서**.

**컬렉션 재생성 패턴**

```python
try:
    chroma_client.delete_collection("animals")
except:
    pass
collection = chroma_client.create_collection(...)
```

실습을 반복 실행하면 같은 `id`(`doc_0`, `doc_1`...)가 중복 추가되면서 데이터가 꼬일 수 있음. 그래서 매번 기존 컬렉션을 지우고 새로 만들어서 **항상 깨끗한 상태에서 시작**하도록 처리한 패턴. (`get_or_create_collection` 대신 `delete` → `create`를 쓴 이유)

**RAG 전체 흐름 (이번 노트 기준)**

```
지식 문서 5개 작성 → 임베딩 벡터화 (normalize) → ChromaDB(cosine) 저장
   ↓
질문 → 질문도 동일하게 임베딩 → 코사인 거리 기준 유사 문서 3개 검색
   ↓
검색된 문서(context) + 질문 → 프롬프트 구성 → LLM 응답 생성
```

---

## 📌 셀별 코드 + 주석

### 1) 패키지 설치

```python
!pip install openai sentence-transformers chromadb python-dotenv
!pip install -U langchain langchain-community
!pip install langchain-text-splitters
```

> 이번 노트는 실제로 `langchain` 관련 기능(TextLoader, Splitter)을 안 쓰기 때문에, 마지막 2줄은 사실 없어도 무방함. 이전 노트와 환경을 통일하기 위해 남겨둔 것으로 보임.

---

### 2) 클라이언트 초기화

```python
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv

load_dotenv()  # .env에서 OPENAI_API_KEY 로드
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o-mini"
```

---

### 3) 지식 문서 구성

```python
# 1. 지식 구성 ----------------------------
# RAG에서 검색 대상이 되는 간단한 지식 문서들
knowledge = [
    "기린은 목이 길다.",
    "코끼리는 귀가 크고 무겁다.",
    "치타는 육상에서 가장 빠른 동물이다.",
    "하마는 물속에서 생활하는 포유류다.",
    "펭귄은 날지 못하지만 수영을 잘한다."
]
```

> 이전 노트(`foods.txt`)와 달리 파일 로딩 없이 리스트로 바로 작성 — 소규모 실습에 더 간단한 방식.

---

### 4) 임베딩 벡터 변환

```python
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 2. 임베딩 모델 로드

# 지식 문서들을 임베딩 벡터로 변환
# normalize_embeddings=True를 사용하면 벡터 크기가 1로 정규화됨
# 정규화된 벡터끼리 내적하면 cosine similarity로 해석하기 쉬움
embeddings = embedder.encode(
    knowledge,
    normalize_embeddings=True
)
print("문서 임베딩:", embeddings)
```

**출력 결과:**

```
문서 임베딩: [[-0.04507371  0.12638244  0.07124437 ...  0.0676651  -0.05525566
  -0.02433011]
 [-0.02226256  0.08212922  0.08384718 ...  0.09879503  0.02848517
   0.01256627]
 [-0.0334919   0.09172806  0.00165735 ...  0.01161214 -0.06550936
  -0.00087902]
 [-0.01816243  0.06918667  0.05138754 ...  0.08816178 -0.07515938
  -0.00698429]
 [-0.04773956  0.16576874  0.03402462 ...  0.10616168 -0.03843705
   0.00790475]]
```

> 5개 문서 × 384차원(`all-MiniLM-L6-v2` 기본 출력 차원) 벡터가 생성됨.

---

### 5) ChromaDB 저장소 구성 및 저장

```python
# 3. Chroma 저장소 구성 ----------------------------------
# ChromaDB를 로컬 디스크에 저장하는 벡터 데이터베이스 클라이언트 생성
chroma_client = chromadb.Client(Settings(
    persist_directory="./rag_demo",
    anonymized_telemetry=False   # 익명 사용 정보 전송 비활성화(권장)
))

# 실습에서는 같은 ID가 중복 추가될 수 있으므로 기존 컬렉션을 삭제하고 다시 생성
try:
    chroma_client.delete_collection("animals")
except:
    pass

# cosine 거리 기준으로 animals 컬렉션 생성
collection = chroma_client.create_collection(
    name="animals",
    metadata={"hnsw:space": "cosine"}
)

# 문서와 임베딩 벡터를 ChromaDB에 저장
for i, (text, emb) in enumerate(zip(knowledge, embeddings)):
    collection.add(
        documents=[text],
        embeddings=[emb.tolist()],
        ids=[f"doc_{i}"]
    )

# 저장된 문서와 ID 확인
all_data = collection.get()
print("\n저장된 전체 데이터:", all_data)
print("\n컬렉션에 저장된 문서 개수:", collection.count())

# 특정 ID의 문서만 조회
doc = collection.get(ids=["doc_0"])
print("\n특정 문서 조회:", doc)
```

**출력 결과:**

```
저장된 전체 데이터: {'ids': ['doc_0', 'doc_1', 'doc_2', 'doc_3', 'doc_4'], 'embeddings': None, 'documents': ['기린은 목이 길다.', '코끼리는 귀가 크고 무겁다.', '치타는 육상에서 가장 빠른 동물이다.', '하마는 물속에서 생활하는 포유류다.', '펭귄은 날지 못하지만 수영을 잘한다.'], 'uris': None, 'included': ['metadatas', 'documents'], 'data': None, 'metadatas': [None, None, None, None, None]}

컬렉션에 저장된 문서 개수: 5

특정 문서 조회: {'ids': ['doc_0'], 'embeddings': None, 'documents': ['기린은 목이 길다.'], 'uris': None, 'included': ['metadatas', 'documents'], 'data': None, 'metadatas': [None]}
```

> 이전 노트에서 겪었던 "컬렉션 미생성" `NotFoundError`나 "설정 다른 Client 중복 생성" `ValueError`가 이번엔 `try/except` + 단일 `Client` 호출 구조 덕분에 발생하지 않음.

---

### 6) 질문 처리 (Retrieval 단계)

```python
# 4. 질문 처리 ---------
query = "목이 긴 동물은?"    # 사용자의 질문

# 질문을 임베딩 벡터로 변환
query_vec = embedder.encode(
    [query],
    normalize_embeddings=True
)[0]

# 질문 벡터와 가까운 문서를 ChromaDB에서 검색
results = collection.query(
    query_embeddings=[query_vec.tolist()],
    n_results=3,
    include=["documents", "distances"]
)
print("\n검색 결과:", results)

# 검색된 문서들을 하나의 참고 문맥으로 합치기
context = "\n".join(results["documents"][0])
print("\n검색된 참고 문서:", context)
```

**출력 결과:**

```
검색 결과: {'ids': [['doc_0', 'doc_2', 'doc_4']], 'embeddings': None, 'documents': [['기린은 목이 길다.', '치타는 육상에서 가장 빠른 동물이다.', '펭귄은 날지 못하지만 수영을 잘한다.']], 'uris': None, 'included': ['documents', 'distances'], 'data': None, 'metadatas': None, 'distances': [[0.18601900339126587, 0.27172964811325073, 0.34308648109436035]]}

검색된 참고 문서: 기린은 목이 길다.
치타는 육상에서 가장 빠른 동물이다.
펭귄은 날지 못하지만 수영을 잘한다.
```

> `n_results=3`이라 5개 문서 중 상위 3개만 반환. "목이 긴 동물"이라는 질문에 `기린`(거리 0.186)이 가장 가깝게 나온 게 핵심 — 코사인 거리 값이 작을수록 유사함을 다시 확인할 수 있음. 나머지 두 개(치타, 펭귄)는 상대적으로 거리가 멀지만 top-3 안에 든 것.

---

### 7) 프롬프트 구성 + LLM 응답 생성 (Generation 단계)

```python
# 5. GPT에 연결하여 답변 생성 --------------------------
# 검색된 문서와 사용자 질문을 함께 프롬프트에 넣음
# 이것이 RAG에서 Generation 단계로 넘어가는 핵심 부분
prompt = f"""
아래 정보를 참고해서 질문에 답해줘.
질문 정보를 바탕으로 질문에 대해 상세하게 설명해 줘.
가능하면 예시나 관련 배경지식도 함께 덧붙여.

[참고 정보]
{context}

[질문]
{query}

추가 사항:
마크다운 기호 없이 평문으로 답해줘.
"""

print("프롬프트 확인 ---")
print(prompt)

# GPT 모델에 프롬프트를 전달하여 답변 생성
response = client.responses.create(
    model=model, input=prompt
)
print("\n답변:", response.output_text)
```

**출력 결과:**

```
프롬프트 확인 ---

아래 정보를 참고해서 질문에 답해줘.
질문 정보를 바탕으로 질문에 대해 상세하게 설명해 줘.
가능하면 예시나 관련 배경지식도 함께 덧붙여.

[참고 정보]
기린은 목이 길다.
치타는 육상에서 가장 빠른 동물이다.
펭귄은 날지 못하지만 수영을 잘한다.

[질문]
목이 긴 동물은?

추가 사항:
마크다운 기호 없이 평문으로 답해줘.

답변: 목이 긴 동물은 기린입니다. 기린은 그 독특한 긴 목으로 유명하며, 이는 주로 나무의 높은 곳에서 잎을 먹기 위해 진화한 특징입니다. 기린의 목은 길어 보이지만, 실제로 목뼈의 개수는 사람과 동일하게 7개입니다. 그러나 각 목뼈의 길이가 길어서 눈에 띄게 긴 목을 형성하게 됩니다.

예를 들어, 기린은 상대적으로 높은 나무에서 자생하는 아카시아 나무의 잎을 쉽게 먹기 위해 긴 목을 발달시켰습니다. 이러한 특성은 다른 초식동물들과의 경쟁에서 유리한 점이 됩니다. 기린의 긴 목은 또한 서로 싸울 때 '넥킹'이라고 불리는 싸움에서 유용하게 사용됩니다. 두 기린이 목을 이용해 서로를 가격하며 우위를 점하기 위해 싸우는 모습은 흥미로운 행동 중 하나입니다.

또한 다른 동물들 중에서는 목이 긴 특징이 두드러지지 않지만, 기린과 같은 생태적 역할을 하는 동물들은 많지 않습니다. 기린은 아프리카의 사바나 지역에서 주로 서식하며, 그들의 긴 목과 긴 다리를 통해 넓은 공간에서 빠르게 이동하고, 먹이를 찾는 데 유리한 삶에 적합하게 진화했습니다.
```

> 참고 문서에 "기린은 목이 길다"라는 한 문장뿐인데도, LLM이 이를 근거 삼아 목뼈 개수·넥킹 행동·서식지 등 **자체 지식으로 살을 붙여 상세하게 답변**함. RAG는 "정답의 핵심 근거"만 제공하고, 그 위에 자연스러운 설명을 붙이는 건 LLM의 사전 지식이 담당한다는 걸 보여주는 예시.

---

## 🔧 rag4text.ipynb 대비 개선된 점

1. `chromadb.Client()`를 **한 번만** 호출 (이전엔 설정 다른 Client를 두 번 호출해 충돌 발생했었음)
2. 컬렉션 생성 시 `try/except delete_collection` 패턴으로 재실행 시 중복 ID 문제 예방
3. `metadata={"hnsw:space": "cosine"}`로 거리 계산 방식을 명시적으로 지정 (기본값 L2 대신)
4. 파일 로딩 없이 리스트로 바로 지식 문서 구성 — 더 간단한 데모 구조