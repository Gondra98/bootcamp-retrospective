# Day 109_LCEL 강의 추천 시스템 RAG 구축

## 📅 2026-07-17

---
# 📄 lang10aiagent.ipynb — LangChain LCEL · RAG · Multi-Agent 강의 추천 시스템

## 🎯 개요

**LCEL(LangChain Expression Language)** 방식으로 강의 추천 시스템을 만드는 실습. 이번 노트북은 크게 두 가지 버전으로 구성될 예정.

|구분|구조|설명|
|---|---|---|
|코드 1|단일 Agent|`CourseAgent` 하나가 검색(Retrieve) + 답변 생성까지 전부 처리|
|코드 2|복수 Agent|`PersonaAgent` → `QueryAgent` → `RetrieveAgent` → `ReplyAgent` 로 역할 분리|

즉, **RAG(Retrieval-Augmented Generation)** 를 먼저 가장 단순한 형태(단일 함수)로 구현해보고, 이후 역할별로 쪼갠 **멀티 에이전트 파이프라인**으로 확장하는 흐름. 이 노트에서는 현재까지 진행된 **환경 설정 → 임베딩 준비 → 벡터스토어 구축** 단계까지 정리.

---

## 1️⃣ 라이브러리 설치

```python
# LangChain의 LCEL 방식으로 강의 추천 시스템
# 코드1 : 단일 Agent - CourseAgent로 검색 + 답변까지 처리
# 코드2 : 복수 Agent - PersonaAgent + QueryAgent + RetrieveAgent + ReplyAgent

!pip install langchain langchain-openai langchain-community langchain-core python-dotenv
!pip install langchain-chroma sentence-transformers
```

### 개념 정리

- **langchain / langchain-core**: LCEL(파이프라인 `|` 연산자), 프롬프트, 파서 등 핵심 골격
- **langchain-openai**: `ChatOpenAI`, `OpenAIEmbeddings` 등 OpenAI 연동 래퍼
- **langchain-community**: 서드파티 통합(벡터DB, 로더 등) 모음
- **langchain-chroma**: LangChain에서 **ChromaDB**를 벡터스토어로 쓰기 위한 커넥터
- **sentence-transformers**: Hugging Face 기반 문장 임베딩 모델 라이브러리 (OpenAI 임베딩 대신 무료 로컬 임베딩 사용 가능)
- **python-dotenv**: `.env` 파일에서 API 키 같은 환경변수를 불러오는 유틸

> 💡 이미 설치돼 있으면 `Requirement already satisfied` 로 스킵되고, `langchain-chroma`, `sentence-transformers` 관련 신규 패키지(`chromadb`, `onnxruntime` 등)만 새로 다운로드됨.

**출력 결과 요약**: 대부분 `Requirement already satisfied` (기설치), `langchain-chroma`, `chromadb`, `pybase64`, `onnxruntime` 등 일부 패키지만 신규 `Collecting/Downloading` 진행.

---

## 2️⃣ 환경 설정 & 임베딩 클래스 정의

```python
from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from chromadb import PersistentClient
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import os
import shutil

from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드 → os.environ에 등록

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
  raise RuntimeError("OPENAI_API_KEY가 없어요")  # 키 없으면 바로 에러로 중단 (fail-fast)

# LLM 객체 생성 (답변 생성에 사용될 모델)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)

# Chroma용 임베딩 클래스
# LangChain의 Chroma 커넥터는 embed_documents / embed_query 두 메서드를 가진
# 객체를 "임베딩 함수"로 기대하기 때문에, SentenceTransformer를 그 규격에 맞게 감싸주는 어댑터 클래스
class STEmbedding:
  def __init__(self, model_name: str):
    self.model = SentenceTransformer(model_name)  # HF 허브에서 모델 로드

  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    # 여러 문서를 한 번에 벡터화 (저장할 때 사용)
    return self.model.encode(texts).tolist()

  def embed_query(self, text: str) -> List[float]:
    # 사용자 질문 1개를 벡터화 (검색할 때 사용)
    return self.model.encode(text).tolist()

# 다국어 지원 경량 임베딩 모델 로드 (한국어 포함, 384차원)
embedding = STEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

### 개념 정리

- **`load_dotenv()`**: `.env` 파일에 있는 `OPENAI_API_KEY=...` 같은 값을 환경변수로 등록. 이후 `os.getenv()`로 읽어옴.
- **fail-fast 패턴**: API 키가 없으면 뒤에서 애매하게 에러 나기 전에, 앞에서 명확한 예외를 바로 던져 원인을 빨리 알 수 있게 함.
- **`ChatOpenAI(temperature=0.5)`**: `temperature`는 답변의 창의성/무작위성 조절 (0에 가까울수록 일관적·보수적, 1에 가까울수록 다양·창의적).
- **왜 임베딩 클래스를 직접 만들었나?**
    - `OpenAIEmbeddings`를 쓰면 API 호출 비용이 들지만, `SentenceTransformer`는 **로컬에서 무료로** 임베딩 생성 가능.
    - 다만 LangChain의 `Chroma`는 특정 인터페이스(`embed_documents`, `embed_query`)를 기대하므로, `SentenceTransformer`를 그대로 못 넣고 **래퍼(adapter) 클래스**로 감싸야 함.
- **`paraphrase-multilingual-MiniLM-L12-v2`**: 50개 이상 언어를 지원하는 경량 다국어 임베딩 모델. 한국어 강의 설명문 임베딩에 적합.

**출력 결과 요약**:

- `Warning: You are sending unauthenticated requests to the HF Hub...` → Hugging Face 토큰 미설정 경고 (기능엔 문제 없음, 속도/제한만 영향)
- 이어서 `modules.json`, `config.json`, `model.safetensors`(약 471MB), `tokenizer.json` 등 모델 구성 파일들이 순차적으로 다운로드되는 진행률 바 출력 (최초 1회만 다운로드, 이후엔 캐시 사용)

---

## 3️⃣ 강의 문서(docs) 생성 & 벡터스토어 구축

```python
# LangChain 강의 추천용 자료
docs = [
    Document(
        page_content="파이썬 기초 과정 / 프로그래밍 입문 / 변수,조건문,반복문,함수,리스트를 배운다.",
        metadata={"title": "파이썬 기초 과정", "level": "초급"}
    ),
    Document(
        page_content="데이터 분석 입문 / pandas,numpy,matplotlib를 사용해 데이터를 정리하고 시각화",
        metadata={"title": "데이터 분석 입문", "level": "초급~중급"}
    ),
    Document(
        page_content="머신러닝 실습 과정 / scikit-learn으로 회귀, 분류, 모델 평가를 실습한다.",
        metadata={"title": "머신러닝 실습 과정", "level": "중급"}
    ),
    Document(
        page_content="자연어 처리 과정 / 텍스트 전처리, 임베딩, 문서 검색, RAG 구조를 학습한다.",
        metadata={"title": "자연어 처리 과정", "level": "중급"}
    ),
    Document(
        page_content="MLOps 엔지니어 과정 / 모델 배포, Docker, FastAPI, CI/CD, 모니터링을 다룬다.",
        metadata={"title": "MLOps 엔지니어 과정", "level": "중급~고급"}
    ),
    Document(
        page_content="생성형 AI Agent 과정 / LangChain, Tool Calling, RAG, Multi-Agent 구조를 실습.",
        metadata={"title": "생성형 AI Agent 과정", "level": "중급~고급"}
    ),
]

CHROMA_DIR = "./chroma_course"
shutil.rmtree(CHROMA_DIR, ignore_errors=True)  # 기존 DB 폴더 삭제 후 새로 생성 (재실행 시 중복 방지)

# docs를 임베딩(embedding)으로 벡터화해서 ChromaDB에 저장
vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding,
    collection_name = "course_recomm",
    persist_directory = CHROMA_DIR
)

# 질문이 들어오면 vectordb에서 유사 문서를 찾아 주는 검색기 역할 수행
retriver = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### 개념 정리

- **`Document`**: LangChain에서 텍스트 데이터를 다루는 표준 단위.
    - `page_content`: 실제 임베딩 대상이 되는 텍스트 (검색 매칭에 사용됨)
    - `metadata`: 검색 결과에 함께 딸려오는 부가 정보 (제목, 난이도 등 필터링/표시용)
- **`shutil.rmtree(..., ignore_errors=True)`**: 같은 셀을 여러 번 실행해도 기존 Chroma DB와 충돌 없이 매번 깨끗하게 새로 만들기 위한 초기화 코드.
- **`Chroma.from_documents()`**: `docs` 리스트의 각 `page_content`를 `embedding` 객체로 벡터화한 뒤, ChromaDB(`persist_directory`에 로컬 저장)에 컬렉션(`collection_name`) 단위로 저장.
- **`as_retriever(search_kwargs={"k": 3})`**: 벡터스토어를 "검색기(Retriever)" 형태로 변환. 질문이 들어오면 **코사인 유사도 기준 상위 3개** 문서를 반환하도록 설정.
- **RAG 흐름과의 연결**: 이 단계는 RAG의 **Indexing(색인)** 파트에 해당 — 문서를 벡터화해서 검색 가능한 형태로 미리 저장해두는 과정. 이후 코드1/코드2에서 `retriver.invoke(질문)`으로 이 인덱스를 검색(Retrieval)하게 됨.

**출력 결과**: 별도 출력 없음 (벡터스토어와 retriever 객체가 정상적으로 생성됨. 에러 없이 실행되면 `./chroma_course` 폴더에 벡터DB 파일이 생성됨)

---

## 📌 다음 단계 (예정)

- **코드 1**: `course_agent(user_question)` 함수 정의 → `retriver.invoke()`로 검색 → LLM에 컨텍스트로 넘겨 답변 생성
- **코드 2**: 역할을 4개 Agent(Persona/Query/Retrieve/Reply)로 분리해 LCEL 체인(`|`)으로 연결

> 오탈자 주의 포인트: 변수명 `retriver`(retriever 오타), 함수 호출 시 `course_agent`와 `coures_agnet` 등 오탈자로 `NameError`가 자주 발생했음. 변수/함수명은 자동완성을 적극 활용하는 게 안전함.