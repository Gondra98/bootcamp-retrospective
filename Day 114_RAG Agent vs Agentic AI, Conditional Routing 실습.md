# Day 114_RAG Agent vs Agentic AI, Conditional Routing 실습

## 📅 2026-07-24

---
# 📄 lgraph11agent.ipynb — langgraph · RAG agent · agentic AI

## LangGraph 기반 RAG Agent vs Agentic AI

같은 RAG(검색 증강 생성) 문제를 **① 단일 노드가 전부 처리하는 AI Agent 방식**과 **② 역할별로 노드를 나눈 Agentic AI 방식** 두 가지로 구현하며 구조 차이를 비교하는 실습.

---

## 🧠 핵심 개념

|구분|AI Agent|Agentic AI|
|---|---|---|
|노드 구성|1개 노드가 전부 처리|역할별로 여러 노드가 분담|
|책임|검색 + 프롬프트 구성 + 답변까지 한 함수 안에서 처리|persona 추출 → 검색어 생성 → 검색 → 답변 생성으로 파이프라인 분리|
|장점|구조가 단순, 빠르게 구현 가능|각 단계 결과를 개별 검증/재사용 가능, 확장(멀티 에이전트) 용이|
|단점|로직이 커질수록 함수 하나가 비대해짐|노드 수만큼 LLM 호출이 늘어 비용·지연시간 증가|

이 노트북은 같은 RAG(검색 증강 생성) 문제를 **① 단일 Agent 방식**과 **② 여러 Agent가 역할을 나눈 Agentic 방식** 두 가지로 구현하며 차이를 비교하는 실습입니다.

---

## 1. 환경 준비

```python
# 공통 준비
!pip install langgraph langchain-openai openai langchain-chroma sentence-transformers python-dotenv
```

- `langgraph` : 상태(State) 기반으로 노드-엣지 그래프를 구성해 LLM 파이프라인을 만드는 프레임워크
- `langchain-openai` : LangChain에서 OpenAI 모델을 `ChatOpenAI`로 감싸 쓰기 위한 패키지
- `langchain-chroma` : 벡터 DB인 Chroma를 LangChain과 연동
- `sentence-transformers` : 문장을 임베딩(숫자 벡터)으로 변환하는 오픈소스 임베딩 모델 라이브러리
- `python-dotenv` : `.env` 파일에 저장한 API 키 등을 코드에서 불러오기 위함

> ⚠️ 패키지명은 반드시 하이픈(`-`)으로 연결해야 함. `sentence=transformers`처럼 `=`를 쓰면 pip가 패키지로 인식하지 못해 에러 발생.

---

## 2. LLM 및 기본 세팅

```python
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수를 읽어와 os.environ에 등록
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

# temperature: 낮을수록 결정적(일관된) 답변, 높을수록 창의적/다양한 답변
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
```

**개념 — `TypedDict`와 LangGraph State** LangGraph의 각 노드는 공통 State(딕셔너리 형태)를 입력받아 일부를 갱신한 딕셔너리를 반환합니다. `TypedDict`로 State의 스키마(어떤 키가 있는지)를 미리 선언해두면, 그래프가 어떤 키를 관리할지 알 수 있습니다.

> 💡 **주의할 점**: `TypedDict(total=False)`는 "선언 안 된 키"를 반환해도 에러 없이 조용히 무시합니다. 즉 노드에서 `return {"context": ...}`를 했는데 State 클래스에 `context` 필드가 없으면, 그 값은 저장되지 않고 나중에 `state["context"]`를 참조할 때 `KeyError`가 발생합니다. → **새 필드를 쓰기 전에 State 클래스부터 갱신하는 습관이 중요**합니다.

---

## 3. 임베딩 래퍼 클래스

```python
# SentenceTransformer를 LangChain Chroma에서 사용하기 위한 래퍼
class STEmbeding:
  def __init__(self, mode_name: str):
    self.model = SentenceTransformer(mode_name)  # 모델 로드

  def embed_documents(self, text: List[str]) -> List[List[float]]:
    # 문서(여러 개) 임베딩 — Chroma가 문서 저장 시 호출
    return self.model.encode(text).tolist()

  def embed_query(self, text: str) -> List[float]:
    # 검색 질의(1개) 임베딩 — Chroma가 검색 시 호출
    return self.model.encode(text).tolist()

embedding = STEmbeding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

**개념 — LangChain Embeddings 인터페이스** LangChain의 `Chroma.from_documents()`는 `embedding` 인자로 `embed_documents(list) -> list[list[float]]`와 `embed_query(str) -> list[float]` 두 메서드를 가진 객체를 기대합니다. `SentenceTransformer`는 이 인터페이스를 직접 구현하지 않기 때문에, 위처럼 얇은 래퍼 클래스로 감싸줘야 Chroma와 호환됩니다.

- `paraphrase-multilingual-MiniLM-L12-v2` : 다국어(한국어 포함) 문장을 지원하는 경량 임베딩 모델. 한국어 텍스트 유사도 검색에 적합.

**실행 결과**

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

→ HuggingFace Hub 인증 토큰 없이 다운로드해서 뜨는 경고. 동작에는 문제없지만, 반복 실습 시 속도 제한에 걸릴 수 있으므로 `HF_TOKEN` 등록을 권장.

---

## 4. 샘플 문서(상품) 데이터 & 벡터스토어 구축

```python
docs = [
    Document(
        page_content="남성 겨울 패딩 / 아우터 / 보온성이 좋은 남성용 롱패딩. 추운 날씨에 적합.",
        metadata={"name": "남성 겨울 패딩", "price": "159000원"}
    ),
    Document(
        page_content="남성 발열 내의 세트 / 이너웨어 / 얇지만 따뜻한 발열 기능 상하의 세트.",
        metadata={"name": "남성 발열 내의 세트", "price": "39000원"}
    ),
    Document(
        page_content="블루투스 무선 이어폰 / 테크 / 노이즈 캔슬링과 긴 배터리를 지원.",
        metadata={"name": "블루투스 무선 이어폰", "price": "129000원"}
    ),
    Document(
        page_content="남성 플리스 자켓 / 아우터 / 가볍고 따뜻한 캐주얼 자켓.",
        metadata={"name": "남성 플리스 자켓", "price": "69000원"}
    ),
    Document(
        page_content="게이밍 키보드 / 테크 / 기계식 스위치와 RGB 백라이트 지원.",
        metadata={"name": "게이밍 키보드", "price": "89000원"}
    ),
]

# 문서 리스트 → 임베딩 → Chroma 벡터DB에 저장
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="shopping_demo",       # 컬렉션(테이블) 이름
    persist_directory="./chroma_shop_db"   # 로컬 디스크에 영구 저장할 경로
)

# k=3 : 검색 시 유사도 상위 3개 문서를 반환
retriver = vectorstore.as_retriever(search_kwargs={"k": 3})
```

**개념 — Document 객체** `page_content`는 임베딩·검색에 실제로 쓰이는 텍스트, `metadata`는 검색 결과에 딸려오는 부가 정보(가격, 이름 등)입니다. 검색 자체는 `page_content`의 의미적 유사도로 이루어지고, `metadata`는 이후 답변 생성 시 참고 자료로 활용됩니다.

**개념 — Retriever** `as_retriever()`는 벡터스토어를 "질의 → 관련 문서 리스트" 형태의 표준 인터페이스로 감싸주는 역할. `search_kwargs={"k": 3}`으로 반환 개수를 제어합니다.

---

## 5. 코드1 — 단일 Agent 방식 (AI Agent)

```python
# 코드1 : RAG 기반의 AI Agent 연습
# 사용자 질문 받기 -> ChromaDB에서 관련 상품 검색 -> LLM에게 답변 요청

class AgentState(TypedDict, total=False):
  question: str
  context: str
  answer: str

def query_agent(state: AgentState) -> AgentState:
  # 검색 + 프롬프트 구성 + 답변 생성을 한 노드가 전부 처리 (= AI Agent 방식)
  question = state["question"]

  found_docs = retriver.invoke(question)  # 벡터 검색 실행

  # 검색된 문서를 "설명 / 가격" 형태의 문자열로 합치기
  context = "\n".join(f' - {doc.page_content} / 가격:{doc.metadata["price"]}' for doc in found_docs)

  prompt = f"""
    너는 쇼핑 추천 도우미야.
    아래 검색 결과를 참고해서 사용자에게 상품을 추천해 줘.

    사용자 질문:
    {question}

    검색 결과:
    {context}

    답변은 한국어로 5행 이내로 작성해 줘.
    절대로 마크업은 출력하지마.
  """

  answer = llm.invoke(prompt).content

  return {
      "context": context,
      "answer": answer
  }

def build_ai_agent_graph():
  graph = StateGraph(AgentState)

  graph.add_node("query_agent", query_agent)
  graph.set_entry_point("query_agent")   # 시작 노드 지정
  graph.add_edge("query_agent", END)     # query_agent 실행 후 바로 종료

  return graph.compile()

ai_agent_app = build_ai_agent_graph()

result = ai_agent_app.invoke({"question": "날씨가 추울 때 입을 남성용 옷을 추천해 줘"})

print("검색 문맥 : ", result["context"])
print("답변 : ", result["answer"])
```

**개념 — StateGraph 기본 흐름** `StateGraph(State클래스)`로 그래프를 만들고 → `add_node(이름, 함수)`로 노드 등록 → `set_entry_point`로 시작점 지정 → `add_edge(A, B)`로 노드 간 흐름(A 다음 B) 연결 → `compile()`로 실행 가능한 그래프 객체 생성. 이 예제는 노드가 `query_agent` 단 하나뿐이라 그래프라기보단 "직선 파이프라인"에 가깝습니다.

**실행 결과**

```
검색 문맥 :   - 남성 플리스 자켓 / 아우터 / 가볍고 따뜻한 캐주얼 자켓. / 가격:69000원
 - 남성 플리스 자켓 / 아우터 / 가볍고 따뜻한 캐주얼 자켓. / 가격:69000원
 - 남성 플리스 자켓 / 아우터 / 가볍고 따뜻한 캐주얼 자켓. / 가격:69000원
답변 :  추운 날씨에 적합한 남성용 옷으로 가볍고 따뜻한 남성 플리스 자켓을 추천합니다. 이 자켓은 캐주얼한 스타일로 다양한 상황에서 활용할 수 있습니다. 가격은 69,000원으로 합리적입니다. 따뜻하게 입고 외출하세요!
```

> 🔎 **관찰 포인트**: 같은 문서가 3번 검색됨(k=3인데 후보 문서가 사실상 겹침). 샘플 데이터가 5개뿐이라 유사도 기반 검색에서 "남성 겨울 패딩"처럼 더 적합한 문서가 상위로 안 뽑히고 특정 문서에 쏠린 것으로 보임 — 실제 서비스라면 문서 다양성/개수를 늘리거나 `search_kwargs`에 `fetch_k`, `lambda_mult`(MMR) 등을 조정해 중복을 줄이는 것을 고려할 만함.

---

## 6. 코드2 — Agentic AI 방식 (역할 분담)

### 6-1. State 정의

```python
# 코드 2 : RAG 기반 Agentic AI
# PersonaAgent, QueryAgent, RetrieverAgent, ReplyAgent
# 여러 Agent가 모여 하나의 목표를 해결하는 구조

class AgenticState(TypedDict, total=False):
  question: str
  persona: str    # 사용자 취향/상황 요약
  query: str      # 검색용 쿼리
  context: str    # 검색된 문서 요약
  answer: str     # 최종 답변
```

코드1의 `AgentState`와 달리, 파이프라인 중간 산출물(`persona`, `query`)까지 State 필드로 명시했습니다. 각 노드가 만든 중간 결과를 다음 노드가 이어받아 쓰는 구조이기 때문입니다.

### 6-2. 노드(Agent) 정의

```python
# agent node
def persona_agent(state: AgenticState) -> AgenticState:
  # 1단계: 사용자 질문에서 취향, 상황, 조건을 추출
  prompt = f"""
  사용자 질문을 보고 쇼핑 취향과 상황을 한 문장으로 정리 해.

  사용자 질문:
  {state["question"]}

  출력:
  """
  persona = llm.invoke(prompt).content
  print("persona : ", persona)
  return {"persona": persona}


def query_agent(state: AgenticState) -> AgenticState:
  # 2단계: persona를 바탕으로 ChromaDB 검색에 사용할 검색어를 생성
  prompt = f"""
    아래 persona를 보고 상품 검색에 사용할 짧은 검색어 하나를 만들어.

  persona:
  {state["persona"]}

  예시 : 반팔 셔츠

  검색어:
  """
  query_result = llm.invoke(prompt).content
  print("검색어 : ", query_result)
  return {"query": query_result}


def retriever_agent(state: AgenticState) -> AgenticState:
  # 3단계: query_agent가 만든 검색어로 Chroma에서 관련 상품을 찾는다.
  found_docs = retriver.invoke(state["query"])
  context = "\n".join(f' - {doc.page_content} / 가격:{doc.metadata["price"]}' for doc in found_docs)
  return {"context": context}


def reply_agent(state: AgenticState) -> AgenticState:
  # 4단계: persona, query, 검색 결과를 종합해 최종 답변 생성
  prompt = f"""
    너는 친절한 쇼핑 추천 도우미야.
    아래 검색 결과를 참고해서 사용자에게 상품을 추천해 줘.

    사용자 질문:
    {state["question"]}

    사용자 persona
    {state["persona"]}

    검색어:
    {state["query"]}

    검색 결과:
    {state["context"]}

    답변은 한국어로 10행 이내로 작성하고 추천 이유도 설명해.
    절대로 마크업은 출력하지마.
  """

  answer = llm.invoke(prompt).content
  return {"answer": answer}
```

**개념 — 왜 노드를 4개로 쪼갰나** `persona_agent → query_agent → retriever_agent → reply_agent` 흐름은 각 단계가 명확한 단일 책임을 가짐:

- `persona_agent` : 자연어 질문 → 사용자 의도/상황 요약 (LLM 호출 1회)
- `query_agent` : 의도 요약 → 벡터 검색용 짧은 쿼리로 변환 (LLM 호출 1회)
- `retriever_agent` : 순수 검색 로직 (LLM 호출 없음, 벡터DB만 사용)
- `reply_agent` : 모든 중간 결과를 종합해 최종 답변 생성 (LLM 호출 1회)

이렇게 나누면 각 단계를 독립적으로 테스트/교체할 수 있고, 나중에 조건 분기(예: 검색 결과가 없으면 재검색)를 추가하기도 쉬워집니다. 대신 LLM 호출이 3번(코드1은 1번)이라 비용과 응답 시간이 늘어나는 트레이드오프가 있습니다.

### 6-3. 그래프 구성 및 실행

```python
# 그래프 구성
def build_ai_agentic_graph():
  aiGraph = StateGraph(AgenticState)

  aiGraph.add_node("persona_agent", persona_agent)
  aiGraph.add_node("query_agent", query_agent)
  aiGraph.add_node("retriever_agent", retriever_agent)
  aiGraph.add_node("reply_agent", reply_agent)

  aiGraph.set_entry_point("persona_agent")

  # 노드 간 순서(엣지) 연결: persona → query → retriever → reply → END
  aiGraph.add_edge("persona_agent", "query_agent")
  aiGraph.add_edge("query_agent", "retriever_agent")
  aiGraph.add_edge("retriever_agent", "reply_agent")
  aiGraph.add_edge("reply_agent", END)

  return aiGraph.compile()

agentic_app = build_ai_agentic_graph()

agentic_result = agentic_app.invoke(
    {"question": "나는 추위를 많이 타는 차도남이야. 매우 추울 때 입을 옷 추천해줘"}
)

print("최종 답변 : ", agentic_result["answer"])
```

**실행 결과**

```
persona :  추위를 많이 타는 차도남이므로, 매우 추운 날씨에 적합한 따뜻한 옷을 추천해달라는 요청입니다.
검색어 :  겨울 외투
최종 답변 :  추위를 많이 타는 차도남이시라면 보온성이 뛰어난 남성용 롱패딩을 추천해 드립니다. 이 롱패딩은 매우 추운 날씨에 적합하며, 몸을 따뜻하게 감싸주기 때문에 체온 유지에 효과적입니다.

가격은 159,000원으로, 품질 대비 적당한 가격이라고 생각됩니다. 롱패딩은 길이가 길어 바람을 막아주고, 스타일리시한 디자인으로 차도남의 이미지를 더욱 돋보이게 할 수 있습니다.

추운 날씨에도 멋을 잃지 않으면서 따뜻함을 유지할 수 있는 이 아이템을 고려해보세요!
```

> 🔎 **코드1과 비교**: 코드1은 원본 질문("날씨가 추울 때...")을 그대로 검색어로 써서 검색 결과가 "남성 플리스 자켓" 하나에 쏠렸지만, 코드2는 `query_agent`가 "겨울 외투"라는 더 일반화된 검색어를 만들어낸 덕분에 "남성 겨울 패딩"이라는 더 적합한 상품이 검색·추천됨. → **질문을 그대로 검색에 쓰는 것보다, 검색 전용 쿼리로 재작성(query rewriting)하는 단계를 거치면 검색 품질이 개선될 수 있음**을 실제로 보여주는 사례.

---

## 📌 실습에서 겪었던 디버깅 포인트 (요약)

노트북 진행 중 실제로 발생했던 오류와 원인을 정리해두면 다음에 비슷한 실수를 줄일 수 있습니다.

|오류|원인|
|---|---|
|`KeyError: 'context'`|`TypedDict` 필드명 오타(`contenxt`)로 인해 반환값이 State에 저장되지 않음|
|`RepositoryNotFoundError` (401)|임베딩 모델명에 `-L12-v2` 접미사 누락|
|`AttributeError` 가능성|`__init__`에서 `self.mode`로 저장하고 메서드는 `self.model` 참조 (변수명 불일치)|
|`NameError: name 'docs' is not defined`|`Document` 리스트를 만드는 셀을 실행하지 않고 다음 셀 실행|
|`SyntaxError`|`return {"context": continue}` — `continue`는 예약어라 값으로 사용 불가|
|함수 중복 정의|같은 이름(`reply_agent`)으로 함수를 두 번 정의 → 뒤의 정의가 앞을 덮어씀|
|`NameError: name 'retriever_agent' is not defined`|함수 정의는 `retriver_agent`(오탈자)인데 그래프에서는 `retriever_agent`로 참조|
|`TypeError: OpenAI.__init__() got an unexpected keyword argument 'model'`|`openai.OpenAI`(클라이언트)와 `langchain_openai.ChatOpenAI`(래퍼) 혼동 — `model`, `temperature`는 클라이언트 생성자가 아니라 `chat.completions.create()` 호출 시 넘기는 인자|

**공통 교훈**: LangGraph의 `TypedDict(total=False)` 스키마는 정의 안 된 키를 조용히 무시하므로, 노드가 새 값을 반환하기 전에 **State 클래스부터 필드를 추가**하는 습관을 들이는 게 가장 중요한 디버깅 예방책이었습니다.

---

# 📄 lgraph12agentic_workflow.ipynb — langgraph · conditional routing · stategraph

## LangGraph 조건부 라우팅으로 만든 회의 스케줄링 Agentic Workflow

날씨(가상 데이터)와 캘린더를 참조해 회의 가능 시간을 자동으로 찾고, 회의 주제를 LLM으로 요약해 결과를 보고하는 파이프라인. `add_conditional_edges`로 상황에 따라 그래프 흐름을 분기시키는 것이 핵심 실습 포인트.

---

## 🧠 핵심 개념 — 조건부 라우팅(Conditional Routing)

일반 `add_edge(A, B)`는 A 다음에 항상 B로 가는 고정 흐름이지만, `add_conditional_edges(A, 분기함수, 매핑)`는 A 실행 후 **분기 함수의 반환값에 따라 다음 노드를 동적으로 선택**합니다.

```python
builder.add_conditional_edges(
    "dry_dates",           # 이 노드 실행 직후
    route_after_weather,   # 이 함수를 호출해서
    {
        "calendar": "calendar",   # 함수가 "calendar"를 반환하면 → calendar 노드로
        "report": "report"        # 함수가 "report"를 반환하면 → report 노드로
    },
)
```

> ⚠️ 매핑 딕셔너리의 **키는 분기 함수가 실제로 반환하는 문자열**과 정확히 일치해야 합니다. 이 노트북 작업 중에도 분기 함수는 `"calendar"`를 반환하는데 매핑 키를 `"calender"`(오타)로 써서 `ValueError: Found edge starting at unknown node 'calender'`가 났던 적이 있었습니다 — 노드 이름과 분기 반환값 스펠링을 항상 맞춰야 합니다.

이번 그래프의 전체 흐름:

```
weather → dry_dates ─(비 없는 날 있음)→ calendar → free_times ─(빈 시간 있음)→ meeting → summarize → report → END
              └─(비 없는 날 없음)→ report ↑                    └─(빈 시간 없음)→ report ┘
```

---

## 1. 환경 준비

```python
# 캘린더(예: 구글 캘린더 API)를 참조해 회의 스케쥴 생성
# 날씨 참조 -> 비가 없는 날짜가 있는가?(없음 - END) 있음 -> 날짜, 시간 검색
#              빈 시간이 있는가?(없음 - END) 있음 -> 회의일정 생성 -> 회의 주제 요약 출력 -> END
# 실제 날씨 대신 가상의 데이터로 대체

!pip install -U langgraph openai python-dotenv
```

- 이번 실습은 `langchain_openai.ChatOpenAI`가 아니라 **openai 패키지의 `OpenAI` 클라이언트**를 직접 사용 → `client.responses.create(...)` 방식으로 LLM 호출
- 날씨·캘린더는 실제 API 대신 가상 데이터로 대체해 로직(분기 처리)에 집중하는 구조

---

## 2. LLM 클라이언트 세팅

```python
import os
from datetime import date, timedelta
from typing import TypedDict, Literal
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 없어요")

llm = OpenAI(api_key=api_key)   # openai 클라이언트 객체. model/temperature는 여기서 지정 안 함
```

**개념 — `openai.OpenAI` vs `langchain_openai.ChatOpenAI`** `OpenAI(api_key=...)`는 클라이언트 "연결"만 생성합니다. `model`, `temperature` 같은 값은 생성자가 아니라 실제 호출 시점(`llm.responses.create(model=..., ...)` 또는 `llm.chat.completions.create(model=..., ...)`)에 넘겨야 합니다. `ChatOpenAI`(LangChain 래퍼)는 반대로 생성자에서 `model`을 받고 `.invoke()`로 호출하는 방식이라 헷갈리기 쉬운 부분입니다.

---

## 3. State 정의

```python
# 그래프 전체에서 공유할 자원(상태)
class ScheduleState(TypedDict, total=False):
  city: str      # 지역명
  weather: list[dict[str, str]]   # 날짜별 날씨 정보
  dry_dates: list[str]      # 비가 오지 않는 날짜 목록
  calendar: dict[str, list[str]]   # 날짜별 이미 예약된 시간 정보
  selected_date: str   # 회의 일정으로 선택된 날짜
  selected_time: str   # 회의 일정으로 선택된 시간

  meeting_topic: str   # 사용자가 입력한 회의 주제
  topic_summary: str   # LLM이 생성한 회의 주제 요약

  meeting_created: bool  # 회의 일정 생성 성공 여부
  result: str    # 최종 결과 메세지
```

파이프라인 각 단계가 만들어내는 중간 산출물(`weather`, `dry_dates`, `calendar`, `selected_date/time`, `topic_summary`, `meeting_created`)을 전부 필드로 선언 — `TypedDict(total=False)`라서 선언 안 된 키를 반환하면 조용히 무시되고 나중에 `KeyError`로 이어지므로, 노드를 추가할 때마다 이 스키마부터 갱신하는 습관이 필요합니다.

---

## 4. 날씨 조회 & 비 없는 날짜 선택

```python
# 노드 정의
# 서울 날씨 조회
def check_weather(state: ScheduleState):
  # 원래는 별도 날씨 제공 서버를 이용해야 하나, 실제 API 대신 5일 간의 가상 날씨를 생성
  start_date = date.today() + timedelta(days=1)
  conditions = ["맑음", "비", "흐림", "비", "맑음"]

  weather = [
      {
          "date": str(start_date + timedelta(days=i)),   # i로 하루씩 증가
          "condition": condition
      }
      for i, condition in enumerate(conditions)
  ]

  print("1. 날씨조회")
  for item in weather:
    print(item["date"], item["condition"])

  return {"weather": weather}

# 단독 테스트: 그래프 없이 함수만 직접 호출해서 검증
test_state: ScheduleState = {}
print(check_weather(test_state))
```

**실행 결과**

```
1. 날씨조회
2026-07-25 맑음
2026-07-26 비
2026-07-27 흐림
2026-07-28 비
2026-07-29 맑음
{'weather': [{'date': '2026-07-25', 'condition': '맑음'}, {'date': '2026-07-26', 'condition': '비'}, {'date': '2026-07-27', 'condition': '흐림'}, {'date': '2026-07-28', 'condition': '비'}, {'date': '2026-07-29', 'condition': '맑음'}]}
```

```python
# 비가 없는 날짜 선택
def select_dry_dates(state: ScheduleState):
  dry_dates = [
      item["date"]
      for item in state["weather"]
      if item["condition"] != "비"
  ]

  print("2. 비가 없는 날짜 선택 : ", dry_dates)
  return {"dry_dates": dry_dates}


# 날씨 조회 결과에 따른 분기 — Literal로 반환 가능한 값을 명시(타입 힌트, 강제성은 없음)
def route_after_weather(state: ScheduleState) -> Literal["calendar", "report"]:
  if state.get("dry_dates"):
    return "calendar"   # 비 없는 날짜가 있으면 캘린더 확인 단계로

  return "report"        # 없으면 바로 결과 보고로
```

**개념 — 분기 함수와 `Literal` 타입 힌트** `Literal["calendar", "report"]`는 "이 함수는 이 두 문자열 중 하나만 반환한다"는 의도를 코드에 명시하는 용도입니다. 실제 런타임 강제력은 없지만(다른 문자열 반환해도 에러 안 남), `add_conditional_edges`의 매핑 딕셔너리 키와 이 반환값이 반드시 일치해야 하므로 협업/가독성 측면에서 유용합니다.

---

## 5. 캘린더 확인 & 빈 시간 탐색

```python
# 사용자 캘린더 확인
def check_calendar(state: ScheduleState):
  # 실제 Calendar API 대신 날짜별 예약 시간을 생성 예:{"2026-07-25":["10:00","15:00"]}
  calendar = {}

  for index, selected_date in enumerate(state["dry_dates"]):
    if index == 0:
      # 첫번째 날짜에는 10시와 15시에 이미 예약이 있다고 가정
      calendar[selected_date] = ["10:00", "15:00"]
    else:
      calendar[selected_date] = ["10:00"]

  print("3. 사용자 캘린더 확인")
  for day, busy_times in calendar.items():
    print(f"{day} 예약시간:{busy_times}")

  return {"calendar": calendar}


# 비어 있는 시간 탐지
def find_free_times(state: ScheduleState):
  meeting_times = ["10:00", "15:00", "16:00"]   # 후보 회의 시간대

  for selected_date in state["dry_dates"]:
    busy_times = state["calendar"].get(selected_date, [])   # 예약 없으면 빈 리스트

    for meeting_time in meeting_times:
      if meeting_time not in busy_times:
        print("4. 비어 있는 시간 탐지")
        print(f"{selected_date} {meeting_time} 가능")

        # 가장 먼저 찾은 (날짜, 시간) 조합을 바로 확정하고 종료
        return {
            "selected_date": selected_date,
            "selected_time": meeting_time,
        }

  print("4. 비어 있는 시간 탐지 했으나 가능한 시간이 없어요")
  return {}   # 못 찾으면 selected_date/time 없이 빈 딕셔너리 반환


# 빈 시간 검색 결과에 따른 분기
def route_after_freetime(state: ScheduleState) -> Literal["meeting", "report"]:
  if state.get("selected_date") and state.get("selected_time"):
    return "meeting"   # 시간 확보됐으면 회의 생성 단계로

  return "report"        # 못 찾았으면 바로 결과 보고로


# 회의 일정 생성
def create_meeting(state: ScheduleState):
  print("5. 회의 일정 생성")
  print(
      f"{state['selected_date']} "
      f"{state['selected_time']} 에 회의를 생성했습니다."
  )
  return {"meeting_created": True}
```

**개념 — `find_free_times`의 이중 반복문** 날짜 순서대로 순회하면서, 각 날짜마다 후보 시간(`10:00`, `15:00`, `16:00`)을 확인해 **처음으로 비어있는 (날짜, 시간) 조합을 찾는 즉시 반환**합니다. 모든 날짜·시간을 다 확인했는데도 못 찾으면 빈 딕셔너리를 반환해 `selected_date`/`selected_time`이 State에 채워지지 않게 하고, 이걸 `route_after_freetime`이 감지해 `report`로 분기시킵니다.

---

## 6. 회의 주제 요약 (LLM 호출)

```python
# LLM을 이용해 회의 주제 요약
def summarize_meeting_topic(state: ScheduleState):
  meeting_topic = state.get("meeting_topic", "").strip()

  # 회의 주제가 입력 되지 않은 경우
  if not meeting_topic:
    return {"topic_summary": "입력된 회의 주제가 없습니다."}

  prompt = f"""
    다음은 회의에서 다룰 주제입니다.

    회의 주제:
    {meeting_topic}

    이 주제에 대해 참석자들이 회의 전에 이해하면 좋은 기본 내용을 한국어로 요약해 줘.

    작성 조건:
    - 핵심 개념을 중심으로 작성
    - 3 ~ 5개 정도의 항목으로 간결하게 작성
    - 불필요한 인사말 등은 작성하면 안돼.
    - 마크업 등은 출력에서 제외 시켜
  """

  try:
    # Responses API: input에 프롬프트를 넣고 output_text로 바로 텍스트 추출
    response = llm.responses.create(model="gpt-4.1-mini", input=prompt)
    topic_summary = response.output_text.strip()

  except Exception as err:
    # LLM 호출 실패 시에도 그래프 흐름이 끊기지 않도록 에러 메시지를 결과에 담아 반환
    topic_summary = (f"회의 주제 요약을 생성하지 못함. \n오류 내용:{err}")

  return {"topic_summary": topic_summary}
```

**개념 — `try/except`로 LLM 호출 감싸기** LLM API 호출은 네트워크 문제, 요금 한도, 응답 지연 등으로 실패할 수 있습니다. 여기서는 실패해도 예외를 그대로 던지지 않고 `topic_summary`에 에러 메시지를 담아 정상적으로 반환 — 그래프 전체가 중단되지 않고 `report` 노드까지 흐름이 이어지도록 방어적으로 설계된 부분입니다.

---

## 7. 결과 보고 (⚠️ 디버깅 포인트)

```python
# 결과 보고
def report_result(state: ScheduleState):
  if not state.get("dry_dates"):
    result = "비가 없는 날씨를 찾을 수 없어요"
  elif not state.get("selected_date"):
    result = "비가 없는 날씨는 있으나 빈 시간을 찾을 수 없어요"
  elif state.get("meeting_created"):
    result = (
        f"회의 일정 생성 완료: "
        f"{state['selected_date']} "
        f"{state['selected_time']}\n"
        f"회의 주제 : {state.get('meeting_topic', '')}\n"
        f"주제 내용 : {state.get('topic_summary', '')}"
    )
  else:
    result = "회의 일정을 생성하지 못함"

  return {"result": result}
```

> 🐛 **발견했던 버그 (수정 완료)**: 원본 코드는 `elif not state.get("meeting_created"):`로 조건이 **뒤집혀** 있었습니다. `create_meeting`이 정상 실행되면 `meeting_created`는 `True`가 되는데, `not True`는 `False`라서 성공 메시지 분기를 건너뛰고 `else`(실패 메시지)로 빠져버렸습니다. 실제로 회의가 정상 생성됐는데도 최종 결과에 `"회의 일정을 생성하지 못함"`이 출력된 원인이었습니다.
> 
> 함께 딸려있던 다른 문제 — `state['"selected_date"']`(f-string 안에서 따옴표가 중첩되어 실제로는 `"selected_date"`라는 리터럴 문자열을 키로 사용하려는 잘못된 코드), `state['meeting_topic', '']`(`.get()`이 아니라 튜플로 인덱싱하는 잘못된 문법) — 도 조건이 뒤집혀 있던 탓에 한 번도 실행되지 않아 드러나지 않았던 문제였습니다. `elif state.get("meeting_created"):`로 조건을 바로잡고, 따옴표 중첩과 `.get()` 호출을 정리해 아래처럼 최종 수정했습니다 (위 코드가 수정 완료본).

---

## 8. 그래프 구성

```python
# 그래프
def build_graph():
  builder = StateGraph(ScheduleState)

  builder.add_node("weather", check_weather)
  builder.add_node("dry_dates", select_dry_dates)
  builder.add_node("calendar", check_calendar)
  builder.add_node("free_times", find_free_times)
  builder.add_node("meeting", create_meeting)
  builder.add_node("summarize", summarize_meeting_topic)
  builder.add_node("report", report_result)

  builder.add_edge(START, "weather")
  builder.add_edge("weather", "dry_dates")

  # 조건 분기(비 여부)
  builder.add_conditional_edges(
      "dry_dates",
      route_after_weather,
      {
          "calendar": "calendar",
          "report": "report"
      },
  )

  builder.add_edge("calendar", "free_times")

  # 조건 분기(빈 시간 여부)
  builder.add_conditional_edges(
      "free_times",
      route_after_freetime,
      {
          "meeting": "meeting",
          "report": "report"
      },
  )

  builder.add_edge("meeting", "summarize")
  builder.add_edge("summarize", "report")
  builder.add_edge("report", END)

  return builder.compile()
```

**개념 — 왜 두 분기 모두 결국 `report`로 모이나** `dry_dates`에서 `report`로 빠지는 경로와, `free_times`에서 `report`로 빠지는 경로, 그리고 정상 흐름(`meeting → summarize → report`)까지 **모든 경로가 결국 `report` 노드 하나로 합류**합니다. 이렇게 설계하면 성공/실패 어떤 경로를 거치든 `report_result`가 항상 `state["result"]`를 채워주므로, 그래프를 호출하는 쪽(`invoke`)은 분기 결과를 신경 쓰지 않고 항상 `final_state["result"]`만 읽으면 됩니다.

---

## 9. 실행

```python
# 실행
if __name__ == "__main__":
  graph = build_graph()
  meeting_topic = input("회의 주제 입력:").strip()

  initial_state: ScheduleState = {
      "city": "서울",
      "meeting_topic": meeting_topic,
      "meeting_created": False
  }

  final_state = graph.invoke(initial_state)
  print(f"최종 결과 : {final_state['result']}")
```

**실행 결과** (입력: `팀 워크플로우 개선 방안`, `report_result` 버그 수정 반영 후)

```
1. 날씨조회
2026-07-25 맑음
2026-07-26 비
2026-07-27 흐림
2026-07-28 비
2026-07-29 맑음
2. 비가 없는 날짜 선택 :  ['2026-07-25', '2026-07-27', '2026-07-29']
3. 사용자 캘린더 확인
2026-07-25 예약시간:['10:00', '15:00']
2026-07-27 예약시간:['10:00']
2026-07-29 예약시간:['10:00']
4. 비어 있는 시간 탐지
2026-07-25 16:00 가능
5. 회의 일정 생성
2026-07-25 16:00 에 회의를 생성했습니다.
최종 결과 : 회의 일정 생성 완료: 2026-07-25 16:00
회의 주제 : 팀 워크플로우 개선 방안
주제 내용 : 1. 팀 워크플로우란 팀 내 업무 프로세스와 협업 방식을 의미하며, 효율적 수행을 위해 역할 분담과 소통 체계가 중요하다.
6. 워크플로우 개선은 업무 병목 현상 해소, 반복 작업 자동화, 커뮤니케이션 간소화를 통해 생산성을 높이는 것을 목표로 한다.
7. 효과적인 워크플로우 관리를 위해서는 업무 프로세스 분석, 문제점 파악, 개선 방안 도출과 전 팀원 동의가 필수적이다.
8. 도구 활용(프로젝트 관리 툴, 협업 소프트웨어 등)과 정기적인 피드백 수집 및 반영이 개선 지속성에 기여한다.
```

`2026-07-25`는 캘린더에 `10:00`, `15:00`이 이미 예약되어 있어 `find_free_times`가 세 번째 후보인 `16:00`을 찾아 확정한 것을 로그로 확인할 수 있습니다. `report_result` 버그를 고친 뒤에는 회의 생성 결과 + 회의 주제 + LLM이 요약한 주제 내용까지 최종 결과 메시지에 정상적으로 담겨 나옵니다.

---

## 📌 이 노트북에서 겪은 디버깅 포인트 요약

|오류/증상|원인|
|---|---|
|`KeyError: 'data'`|`select_dry_dates`에서 `item["data"]` → `item["date"]` 오타|
|`TypeError` (`.get[...]`)|`route_after_weather`에서 `state.get["dry_dates"]` → 소괄호 호출 누락|
|`ValueError: Found edge starting at unknown node 'calender'`|노드 등록명은 `"calendar"`인데 엣지/분기 매핑에서 `"calender"`로 오타|
|조건 분기 매핑 불일치|`route_after_freetime`은 `"meeting"`을 반환하는데 매핑 딕셔너리엔 `"calendar"`/`"report"`만 있었음|
|`add_edge("create", ...)` 참조 오류|노드명은 `"meeting"`인데 `"create"`로 잘못 참조|
|최종 결과 메시지가 항상 실패로 표시|`report_result`의 `elif not state.get("meeting_created")` — 조건 반전, `elif state.get("meeting_created")`로 수정 필요|
|f-string 내 따옴표 중첩 오류|`state['"selected_date"']`처럼 안쪽 따옴표를 바깥과 같은 종류로 써서 의도와 다른 키로 인덱싱됨|
|`.get()`과 인덱싱 혼동|`state['meeting_topic', '']`는 튜플 인덱싱이지 `.get(key, default)`가 아님|

**공통 교훈**: 조건부 라우팅에서는 ① 분기 함수의 실제 반환 문자열, ② `add_conditional_edges` 매핑의 키, ③ `add_node`로 등록한 노드 이름 — 이 세 가지가 철자 단위로 정확히 일치해야 합니다. 하나라도 어긋나면 그래프 컴파일 시점(`ValueError`) 또는 실행 시점(엉뚱한 분기)에 문제가 드러납니다.
