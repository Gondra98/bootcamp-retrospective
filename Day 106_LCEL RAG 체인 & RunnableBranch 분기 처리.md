# Day 106_LCEL RAG 체인 & RunnableBranch 분기 처리

## 📅 2026-07-14

---
# 📄 lang5.ipynb — LCEL · RAG · RunnableBranch

## 개요

LangChain LCEL(LangChain Expression Language)을 사용해 **RAG 체인**과 **조건 분기 체인(RunnableBranch)**을 구성하는 실습.

전체 흐름:

```
문서 준비 → 텍스트 분할 → 임베딩 → ChromaDB 저장 → Retriever 생성
→ RAG 체인(LCEL) 구성 → 조건별 분기 처리(RunnableBranch)
```

---

## 1. 환경 설정 및 초기화

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
import langsmith
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 API 키 등 환경변수 로드

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
  raise RuntimeError("OPENAI_API_KEY가 없어요")  # 키 없으면 바로 에러 발생시켜 이후 단계 낭비 방지

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)

# 문장 임베딩 모델 : 문서를 벡터로 변환할 때 사용 (HuggingFace 로컬 모델)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

**개념 정리**

- `langsmith`는 `langchain-core`의 **필수 의존성**이라 별도 설치 없이 import 가능. 트레이싱을 실제로 켜려면 `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` 환경변수를 `.env`에 등록해야 함.
- `.env` 파일은 `KEY=VALUE` 형식만 인식함. `KEY:VALUE`(콜론)로 쓰면 `python-dotenv`가 파싱하지 못해 값이 로드되지 않음 — 실습 중 실제로 겪은 이슈.

**주의 (Deprecation)**

- `langchain_community.embeddings.HuggingFaceEmbeddings`는 곧 제거 예정. `langchain-huggingface` 패키지의 `HuggingFaceEmbeddings`로 이전 권장.

---

## 2. 문서 준비 → 분할 → 임베딩 → 저장

```python
# 실험용 문서 : LangChain은 모든 문서 타입을 Document() 객체로 감싸서(wrapping) 다룸
docs = [
    Document(page_content="랭체인(LangChain)은 해리슨 체이스가 2022년 10월에 시작한 오픈소스 프로젝트..."),
    Document(page_content="RAG는 Retrieval-Augmented Generation의 약자로, 검색과 생성을 결합한 NLP 기술..."),
]

# 문서를 작은 조각(chunk)으로 분할 — RAG에서 검색 정확도를 위해 권장되는 전처리
text_splitter = CharacterTextSplitter(
    chunk_size = 200,      # 조각 하나의 최대 길이
    chunk_overlap = 20,    # 조각 간 겹치는 길이 (문맥 단절 방지)
    separator="\n\n"
)

split_docs = text_splitter.split_documents(docs)
print('분할된 문서 수 : ', len(split_docs))
print('첫 분할된 문서 : ', split_docs[0].page_content)

# ChromaDB : 벡터 데이터베이스. 분할된 문서를 임베딩하여 저장
db = Chroma.from_documents(
    documents = split_docs,
    embedding = embedding_model,
)

# Retriever : 질문과 의미적으로 유사한 문서를 Chroma에서 검색해오는 객체
retriever = db.as_retriever()
```

**▶ 실행 결과**

```
분할된 문서 수 :  2
첫 분할된 문서 :  랭체인(LangChain)은 해리슨 체이스(Harrison Chase)에 의해 2022년 10월에 오픈 소스 프로젝트로 시작되었습니다. 그는 머신러닝 스타트업인 로버스트 인텔리전스(Robust Intelligence)에서 근무하면서 대규모 언어 모델(LLM)을 활용하여 애플리케이션과 파이프라인을 신속하게 구축할 수 있는 플랫폼의 필요성을 느꼈습니다. 이러한 비전을 가지고 개발자들이 챗봇, 질의응답 시스템, 자동 요약 등 다양한 LLM 애플리케이션을 쉽게 개발할 수 있도록 지원하는 프레임워크를 만들었습니다. LangChain 1.0은 다음과 같은 5계층 아키텍처로 구성되어 있습니다.
```

> chunk_size=200으로 설정했지만 원본 문서 1개(RAG 설명 문서, 약 300자 이상)가 그대로 하나의 chunk로 남아있는 것으로 보임 — `CharacterTextSplitter`는 `separator`(`\n\n`)가 텍스트 안에 없으면 분할하지 않고 원문 그대로 반환하는 특성 때문. 두 `Document`가 각각 `\n\n`을 포함하지 않아 분할 수가 원본 문서 수(2)와 동일하게 나온 것.

**개념 정리**

- **CharacterTextSplitter**: 지정한 구분자(`separator`) 기준으로 텍스트를 자르되, `chunk_size`를 넘지 않게 분할. `chunk_overlap`으로 조각 경계에서 문맥이 끊기지 않도록 일부 겹치게 함.
- **Chroma.from_documents**: 문서 리스트 + 임베딩 모델을 받아 각 문서를 벡터화하고 저장. `persist_directory`를 지정하지 않으면 메모리에만 저장되고 세션 종료 시 사라짐 (영구 저장하려면 경로 지정 필요).
- **Retriever**: 벡터 DB를 "질문 → 유사 문서 검색" 인터페이스로 감싼 것. LCEL 체인 안에서 `retriever | ...` 형태로 파이프라인에 바로 연결 가능.

---

## 3. RAG 체인 구성 (LCEL)

```python
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template = """
        너는 친절하고 똑똑한 AI 어시스턴트야,
        아래 문서 내용을 참고해서 사용자의 질문에 답해 줘,
        문서 내용이 불충분한 경우에는 '문서에 해당 정보가 없어요'라고 답변해.

        문서 내용:
        {context}

        질문:
        {question}

        답변은 5행 정도의 분량이면 좋겠어.
    """
)

# 검색된 Document 리스트를 하나의 문자열로 합치는 함수
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        # "context" 키 : retriever로 질문과 유사한 문서 검색 → format_docs로 문자열 변환
        "context": retriever | RunnableLambda(format_docs),

        # "question" 키 : 원래 질문 값을 그대로 통과시킴
        "question": RunnablePassthrough()
    }
    | prompt_template   # {context}, {question}을 채워서 완성된 프롬프트 생성
    | llm                # 완성된 프롬프트를 LLM에 전달
    | StrOutputParser()  # LLM의 메시지 객체 응답을 순수 문자열로 파싱
)

query = "RAG가 뭔가요?"
result = rag_chain.invoke(query)
```

**▶ 실행 결과 (오류)**

```
RateLimitError: Error code: 429 - {'error': {'message': 'You exceeded your current quota,
please check your plan and billing details. ...', 'type': 'insufficient_quota'}}
```

> `context`/`question` 생성, `retriever` 검색, `prompt_template` 완성까지는 정상 통과. 마지막 `llm.invoke()` 단계에서 OpenAI 계정 쿼터 소진으로 실패 — 코드 로직 문제 아님. 크레딧/결제 확인 필요.

**개념 정리 — LCEL 파이프 구조**

- `{ "context": ..., "question": ... }` 딕셔너리는 LCEL에서 **병렬 실행 후 결과를 하나의 딕셔너리로 합치는 RunnableParallel** 역할을 함 (딕셔너리 형태로 쓰면 자동으로 병렬 러너블로 변환됨).
- `RunnablePassthrough()`: 입력값을 가공 없이 그대로 다음 단계로 넘겨줌. 여기서는 원본 질문 문자열을 `question` 자리에 그대로 전달하는 용도.
- `RunnableLambda(func)`: 일반 파이썬 함수를 LCEL 체인 안에 끼워 넣을 수 있게 감싸는 어댑터. `func`에 실제 함수 객체를 넘겨야 하며, 빈 채로 두면 `TypeError` 발생.
- 파이프(`|`) 연산자는 왼쪽 결과를 오른쪽 입력으로 전달 — Unix 셸의 파이프와 동일한 개념.

**실습 중 겪은 이슈**

- OpenAI API 호출 시 `RateLimitError: 429 insufficient_quota` 발생 → rate limit이 아니라 **계정 쿼터/크레딧 소진** 문제. 체인의 검색·프롬프트 조합 단계(`retriever`, `format_docs`, `prompt_template`)까지는 정상 통과했고, 마지막 `llm` 호출에서 실패한 것으로 확인됨.

---

## 4. 조건 분기 - RunnableBranch (기본)

```python
from langchain_core.runnables import RunnableBranch

# 조건 판별 함수 : 텍스트에 "날씨"가 포함되어 있는지 확인
def is_weather_question(text:str) -> bool:
  return "날씨" in text.lower()

# 분기 A : 날씨 질문이면 고정 응답 반환
weather_chain = RunnableLambda(lambda x: "오늘의 날씨는 흐리고 최고온도는 32도입니다.")

# 분기 B : 그 외 일반 질문
basic_general_chain = RunnableLambda(lambda x: f"일반 질문이군요. '{x}에 대해 설명할게요'")

# (조건함수, 해당 체인) 튜플을 순서대로 검사, 마지막 인자는 기본(default) 체인
basic_branch_chain = RunnableBranch(
    (is_weather_question, weather_chain),
    basic_general_chain
)

print(basic_branch_chain.invoke("오늘 날씨는 어때?"))       # → weather_chain 실행
print(basic_branch_chain.invoke("점심 메뉴 뭐가 좋을까?"))    # → basic_general_chain 실행
```

> ⚠️ 이 셀은 아직 실행되지 않아 노트북에 출력값이 저장되어 있지 않음. (weather_chain은 고정 문자열 반환이라 정상 실행 시 "오늘의 날씨는 흐리고 최고온도는 32도입니다."가 그대로 출력될 것으로 예상)

**개념 정리**

- **RunnableBranch**: `(조건함수, 실행체인)` 쌍을 순서대로 평가하며, 조건이 `True`인 첫 번째 체인을 실행. 어떤 조건도 만족하지 않으면 마지막에 넣은 기본 체인이 실행됨.
- if-elif-else 구조를 LCEL 파이프라인 안에서 표현하는 방법이라고 이해하면 됨.

---

## 5. 조건 분기 - 질문 유형별 LLM 라우팅 (응용)

```python
llm2 = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # 다른 모델로 라우팅하는 예시

def make_math_prompt(question:str) -> str:
  return (
      "당신은 수학 풀이를 잘하는 전문가야.\n"
      "아래 수학 문제를 단계별로 풀고 과정도 적어 줘, 마지막 줄에 정답만 한 번 더 적어 줘.\n\n"
      f"문제 : {question}\n\n"
      "풀이:"
  )

def make_code_prompt(question:str) -> str:
  return (
      "당신은 프로그래머 강사야.\n"
      "아래 요청에 대해 1) 간단한 설명 2) 예제 코드 3) 중요 포인트 순서대로 말해줘.\n\n"
      f"요청 : {question}\n\n"
      "풀이:"
  )

def make_general_prompt(question:str) -> str:
  return (
      "당신은 친절한 AI 설명가야.\n"
      "아래 질문에 대해 초보자도 이해할 수 있도록 5문장 정도로 답을 줘.\n\n"
      f"문제 : {question}\n\n"
      "답변:"
  )

# 각 유형별 체인 : 프롬프트 생성 -> LLM 호출 -> 문자열 파싱
math_chain = RunnableLambda(make_math_prompt) | llm2 | StrOutputParser()
code_chain = RunnableLambda(make_code_prompt) | llm2 | StrOutputParser()
general_chain = RunnableLambda(make_general_prompt) | llm | StrOutputParser()  # 일반 질문만 llm(gpt-4.1-mini) 사용

# 수학 질문 판별 : 키워드 또는 연산 기호 포함 여부
def is_math_question(text:str) -> bool:
  t = text.replace(" ", "").lower()
  math_keywords = ["더하기","빼기","곱하기","나누기","계산","합","차","곱","나눈값","몇","얼마"]
  math_symbols = ["+", "-", "*", "/", "^"]
  return any(k in t for k in math_keywords) or any(s in t for s in math_symbols)

# 코딩 질문 판별 : 관련 키워드 포함 여부
def is_code_question(text:str) -> bool:
  t = text.lower()
  code_keywords = ["코드","함수","클래스","메소드","알고리즘","python","파이썬","java","자바"]
  return any(k in t for k in code_keywords)

# 수학 -> 코드 -> 일반 순으로 조건 검사, 아무것도 해당 안 되면 general_chain
branch_chain = RunnableBranch(
    (is_math_question, math_chain),
    (is_code_question, code_chain),
    general_chain
)

questions = [
    "3 + 5 *2는 얼마야?",
    "파이썬으로 리스트 정렬하는 코드 만들어 줘",
    "짬뽕이야 짜장이야?"
]

for q in questions:
  result = branch_chain.invoke(q)
```

> ⚠️ 이 셀도 아직 실행되지 않음. 특히 `math_chain`/`code_chain`이 `llm2`(gpt-4o-mini)를 쓰지만, `llm`(gpt-4.1-mini) 쪽 쿼터 문제가 계정 전체에 걸린 거라면 세 질문 모두 위와 동일한 `RateLimitError`가 날 가능성이 높음. 크레딧/결제 이슈 해결 후 재실행 필요.

**개념 정리**

- 질문 내용에 따라 **다른 프롬프트 + 다른 LLM(모델)**로 라우팅하는 패턴. 실무에서는 비용/성능이 다른 모델(GPT, Gemini, Claude 등)을 질문 난이도나 유형에 맞게 자동 선택하는 데 활용됨.
- 판별 함수(`is_math_question`, `is_code_question`)는 키워드/기호 매칭 기반의 단순 규칙. 정교하게 하려면 별도 분류 모델이나 LLM 자체를 라우터로 쓰는 방법도 있음.
- `RunnableBranch`의 조건 검사 순서가 중요 — 여기서는 수학 조건을 먼저 검사하므로, 수학+코드 키워드가 동시에 있는 질문은 무조건 `math_chain`으로 감.

---

## LangSmith 트레이싱 관련 메모

- `.env`에 아래 3개 변수 설정 시 LangChain 실행이 자동으로 LangSmith에 트레이스로 기록됨:
    
    ```
    LANGSMITH_TRACING=trueLANGSMITH_API_KEY=lsv2_pt_...LANGSMITH_PROJECT=sample-rag
    ```
    
- `langsmith[claude-agent-sdk]`처럼 대괄호가 붙은 extra 설치는 특정 에이전트 프레임워크(Claude Agent SDK 등) 연동 시에만 필요 — 현재 LangChain 기반 워크플로우에는 불필요.
- API 키는 절대 코드/채팅에 평문으로 노출하지 않기 — 노출 시 즉시 재발급(regenerate) 권장.
