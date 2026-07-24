# Day 112_Router 패턴 비교 (rag·llm / math·code·chat / web RAG)

## 📅 2026-07-22

---
# 📄 lgraph6_chain.ipynb — RAG · LangGraph · 질문 라우팅

## 개요

사용자 질문이 **사내 문서(사규) 관련**인지 **일반 상식**인지 먼저 분류(classify)한 다음, 분류 결과에 따라 서로 다른 체인으로 답을 생성하는 구조.

- **LangChain** : 벡터스토어(Chroma) 구성, RAG 체인, 분류 체인, 일반 답변 체인
- **LangGraph** : 위 체인들을 노드(Node)로 감싸고, 분류 결과에 따라 실행 경로를 분기(conditional edge)

```
사용자 질문
   │
   ▼
[classify] ── 질문이 사규 관련? 일반 상식?
   │
   ├─ rag ─────▶ [rag_answer] ──▶ END
   ├─ llm ─────▶ [llm_answer] ──▶ END
   └─ unknown ─▶ [fallback]   ──▶ END
```

---

## 핵심 개념 정리

### 1. RAG (Retrieval-Augmented Generation)

질문과 관련된 문서를 **먼저 검색(retrieve)**해서 그 내용을 프롬프트에 끼워 넣은 뒤 LLM이 답하게 하는 방식. LLM이 모르는(학습 안 된) 사내 정보도 검색 결과만 넣어주면 답할 수 있게 됨.

### 2. Vectorstore / Retriever (Chroma)

- 문서를 임베딩(숫자 벡터)으로 변환해 저장해두는 DB → `Chroma.from_documents(...)`
- `as_retriever(search_kwargs={'k':3})` : 질문과 가장 유사한 문서 k개를 꺼내오는 역할을 하는 객체로 변환

### 3. LCEL (LangChain Expression Language)

`|` 연산자로 컴포넌트를 이어 붙여 체인을 구성하는 문법.

```python
rag_chain = (
    {"context": retiever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT | llm | StrOutputParser()
)
```

- `RunnablePassthrough()` : 입력값을 그대로 다음 단계로 전달 (여기선 질문 원문을 `question`으로 흘려보냄)
- 딕셔너리 형태로 여러 입력을 병렬 구성 → 각 key가 프롬프트의 `{context}`, `{question}`에 매핑됨

### 4. 라우터(분류기) 패턴

LLM에게 "rag" 또는 "llm" 이라는 **라벨만** 출력하게 시켜서, 그 라벨로 분기 로직을 결정하는 방식. 프롬프트로 "반드시 rag 또는 llm 둘 중 하나로만 답해줘"라고 강하게 제약을 준 것이 포인트.

### 5. LangGraph — StateGraph

- **State**: 그래프 전체에서 공유되는 데이터 스키마. `TypedDict`로 정의.
    
    ```python
    class ChatState(TypedDict):    question:str    route:Literal['rag','llm','unknown']    answer:str
    ```
    
- **Node**: 함수 하나 = 노드 하나. `state`를 받아서 갱신할 부분만 dict로 반환하면 LangGraph가 알아서 병합해줌.
- **Edge**: 노드 간 연결. 고정 흐름은 `add_edge`, 조건 분기는 `add_conditional_edges`.
- **Entry point**: 그래프 시작 노드 지정 (`set_entry_point`).
- **END**: 그래프 종료를 나타내는 특수 노드.

### 6. 조건부 분기 (add_conditional_edges)

```python
graph_builder.add_conditional_edges(
    "classify",       # 어느 노드 다음에 분기할지
    route_decider,     # state를 받아서 문자열(키)을 리턴하는 함수
    {"rag":"rag_answer", "llm":"llm_answer", "unknown":"fallback"}  # 키 → 다음 노드 매핑
)
```

`route_decider`가 리턴한 문자열이 dict의 key와 일치하는 노드로 이동. `classify_node`가 만든 `route` 값이 실질적인 "분기 스위치" 역할.

---

## 섹션별 코드 + 주석

### 1) 환경설정 및 모델 준비

```python
# gpt-4.1-mini를 채팅 모델로, text-embedding-3-small을 임베딩 모델로 사용
llm = ChatOpenAI(model="gpt-4.1-mini", temperature="0.2", api_key=api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
```

### 2) 샘플 사규 문서 & 벡터스토어

```python
# 실무에서는 파일(PDF, txt 등)에서 로드하지만 여기선 샘플 3문장을 직접 Document로 생성
docs = [
    Document(page_content="우리 회사의 정식 근무 시간은 오전 9시부터 오후 6시까지로 한다"),
    Document(page_content="연차 휴가는 1년에 15일이 제공되며, 2년차부터 하루씩 증가한다"),
    Document(page_content="사내 메신저는 슬랙을 사용하며, 중요 공지는 #notice 채널을 이용한다"),
]

# 문서를 임베딩해서 Chroma에 저장 (persist_directory에 로컬로 영속화됨)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="company_poli",
    persist_directory="./chroma_company",
)

retiever = vectorstore.as_retriever(search_kwargs={'k':3})  # 상위 3개 문서 검색
```

> `retiever` 오타(retriever)가 있지만 변수명이라 동작에는 문제 없음. 다만 코드 가독성을 위해 다음에 새로 작성할 땐 `retriever`로 통일하는 게 좋음.

### 3) RAG 체인

```python
# 검색된 문서(Document 리스트)를 하나의 문자열로 합치는 함수
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# system 프롬프트에 {context}(검색된 사규), human 프롬프트에 {question}(질문) 삽입
RAG_PROMPT = ChatPromptTemplate.from_messages([...])

rag_chain = (
    {
        "context": retiever | format_docs,      # 질문 → 검색 → 문자열 결합까지 한 번에
        "question": RunnablePassthrough(),       # 질문 원문 그대로 전달
    }
    | RAG_PROMPT | llm | StrOutputParser()
)
```

### 4) 분류 체인

```python
# 질문을 'rag' 또는 'llm' 라벨로만 답하도록 강하게 제약한 프롬프트
CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([...])
classify_chain = CLASSIFY_PROMPT | llm | StrOutputParser()
```

### 5) 일반 상식 답변 체인

```python
LLM_PROMPT = ChatPromptTemplate.from_messages([...])
llm_chain = LLM_PROMPT | llm | StrOutputParser()
```

### 6) LangGraph 구성

```python
class ChatState(TypedDict):
    question:str
    route:Literal['rag','llm','unknown']
    answer:str

def classify_node(state:ChatState) -> ChatState:
    q = state['question']
    result = classify_chain.invoke({'question':q}).strip().lower()  # 'rag' / 'llm' 문자열

    # LLM이 완전히 rag/llm만 딱 떨어지게 안 줄 수도 있어서 'in'으로 느슨하게 체크
    if 'rag' in result:
        route = 'rag'
    elif 'llm' in result:
        route = 'llm'
    else:
        route = 'unknown'   # 둘 다 아니면 애매한 질문 → fallback으로

    return {"route":route}   # state 전체가 아니라 갱신할 키만 반환하면 LangGraph가 merge

def rag_answer_node(state:ChatState) -> ChatState:
    answer = rag_chain.invoke(state['question'])
    return {"answer":answer}

def llm_answer_node(state:ChatState) -> ChatState:
    answer = llm_chain.invoke({"question": state['question']})
    return {"answer":answer}

def fallback_node(state:ChatState) -> ChatState:
    # 분류가 애매(unknown)할 때는 일단 일반 LLM 답변에 안내 문구를 덧붙여서 응답
    answer = "질문이 애매해서 일단은 일반 LLM으로 답변합니다\n\n" + llm_chain.invoke({"question": state['question']})
    return {"answer":answer}
```

```python
graph_builder = StateGraph(ChatState)
graph_builder.add_node("classify", classify_node)
graph_builder.add_node("rag_answer", rag_answer_node)
graph_builder.add_node("llm_answer", llm_answer_node)
graph_builder.add_node("fallback", fallback_node)

graph_builder.set_entry_point("classify")   # 그래프는 항상 classify부터 시작

def route_decider(state:ChatState) -> str:
    return state['route']   # classify_node가 세팅한 route 값을 그대로 리턴

graph_builder.add_conditional_edges(
    "classify", route_decider,
    {"rag":"rag_answer", "llm":"llm_answer", "unknown":"fallback"}
)

# 세 노드 모두 답변 생성 후 그래프 종료
graph_builder.add_edge("rag_answer", END)
graph_builder.add_edge("llm_answer", END)
graph_builder.add_edge("fallback", END)

app = graph_builder.compile()   # 실행 가능한 그래프 객체로 컴파일
```

### 7) 실행 루프

```python
while True:
    user_q = input("질문 (종료:q)")
    if user_q.lower() == 'q':
        break

    # 매 턴마다 새 state로 그래프 실행 (대화 히스토리는 유지 안 함, 단발성 질문 처리)
    state:ChatState = {"question":user_q, "route":"unknown", "answer":""}
    result = app.invoke(state)
    print("답변 : ")
    print(result["answer"])
```

---

## 실제 실행 결과

입력: `근무 시간은?`

```
질문 (종료:q)근무 시간은?
[classify_node ] : route 결정 : rag
[rag_answer_node ] : RAG 답변 생성
답변 :
우리 회사의 정식 근무 시간은 오전 9시부터 오후 6시까지입니다.
```

→ 질문이 사규 관련으로 정확히 분류(`route: rag`)되어 `rag_answer_node`로 이동했고, 벡터스토어에 저장해둔 근무시간 문서를 검색해 그대로 답변에 반영된 것을 확인. RAG + 라우팅이 의도대로 동작함.

---

## 정리

- **분류 → 분기 → 실행** 3단계 구조가 LangGraph의 가장 기본적인 라우터 패턴.
- state는 "그래프 전체가 공유하는 공용 저장소"이고, 각 노드는 자기가 갱신할 키-값만 반환하면 됨 (전체 state를 다시 만들 필요 없음).
- `add_conditional_edges`의 세 번째 인자(dict)는 "분기 함수의 리턴값" → "다음 노드 이름"을 매핑하는 라우팅 테이블.
- 분류가 애매한 경우(`unknown`)를 별도 노드(`fallback`)로 처리해서 그래프가 죽지 않고 항상 답변을 내도록 방어적으로 설계된 점이 눈여겨볼 부분.
- 다음에 확장한다면: 대화 히스토리를 state에 누적하거나, 분류 라벨을 rag/llm 2개보다 세분화(예: HR/총무/IT 등 카테고리별 RAG)하는 방향으로 발전 가능.

---
# 📄 lgraph7_chain.ipynb — LangGraph · 질문분류 · 라우팅

사용자 질문을 LLM이 math/code/chat 중 하나로 분류 → 알맞은 노드에서 답변 생성 → 마지막에 답변을 요약까지 하는 4단계 LangGraph 파이프라인.

> ⚠️ 이 노트북은 실행 이력(output)이 없는 상태로 업로드돼서, 아래 "실행 결과"는 실제 캡처가 아니라 코드 흐름상 예상되는 동작 설명임. 실제로 돌린 뒤 output 있는 버전을 주면 그걸로 교체 가능.

---

## 핵심 개념 정리

- **분류(Classify) → 분기(Route) → 실행(Execute) → 후처리(Summarize)** 4단계 구조. lgraph6과 다른 점은 분류 라벨이 2개(rag/llm)가 아니라 3개(math/code/chat)이고, 답변 후에 **요약 노드**가 하나 더 붙는다는 것.
- **state 병합 방식의 차이**: lgraph6은 `{"route": route}`처럼 갱신할 키만 반환했는데, 이 노트북은 `{**state, "answer": answer}`처럼 기존 state를 스프레드(`**state`)한 뒤 새 키를 얹어서 반환. 결과적으로는 동일하게 동작하지만(LangGraph가 알아서 merge하므로 `**state`는 사실 불필요), 명시적으로 전체 state를 유지하는 스타일.
- **후처리(Summarize) 노드 합류**: math/code/chat 세 노드가 각자 답변을 만든 뒤 모두 `summarize` 노드로 모이는 구조 (`add_edge("math","summarize")` 등 3개 엣지가 한 노드로 수렴). 여러 분기가 다시 하나로 합쳐지는 "fan-in" 패턴.

---

## 셀별 설명

### Cell 0 — Markdown (제목)

```
# 질문 분류 기반 답변 라우팅 (math / code / chat)
사용자 질문에 대해 LLM이 분류 키워드 생성 후 적당한 답변
```

노트북 전체 목적을 소개하는 타이틀 셀.

### Cell 1 — Code (패키지 설치)

```python
!pip install langchain langchain-openai langchain-community langchain-core python-dotenv
!pip install langchain-chroma sentence-transformers langchain-text-splitters chromadb
!pip install langgraph
```

langchain·langgraph 관련 패키지 설치. (실제로는 Chroma/sentence-transformers까지는 이 노트북에서 안 쓰지만 이전 노트북 템플릿을 그대로 가져와서 남아있는 것으로 보임 — 불필요한 설치라 지워도 무방)

### Cell 2 — Markdown

`## 1. 환경설정 및 모델 준비` 섹션 헤더.

### Cell 3 — Code (환경설정 & LLM 초기화)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
...
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 없어요")   # 키 없으면 바로 에러 발생시켜서 이후 단계 진행 방지

llm = ChatOpenAI(model="gpt-4.1-mini", temperature="0.2", api_key=api_key)
```

`.env`에서 `OPENAI_API_KEY`를 읽어와 LLM 객체 하나만 생성. temperature 0.2로 낮게 설정해 분류/답변이 비교적 일관되게 나오도록 함.

### Cell 4 — Markdown

`## 2. State 정의` 섹션 헤더.

### Cell 5 — Code (State 정의)

```python
class QaState(TypedDict, total=False):
    question:str
    intent:str    # math, code, chat
    answer:str
    summary:str
```

그래프가 공유할 상태 스키마. `total=False`라서 모든 키가 선택적(optional) — 처음엔 `question`만 있고 나머지는 노드를 거치며 하나씩 채워지는 구조.

### Cell 6 — Markdown

`### 3-1. 분류 노드` 섹션 헤더.

### Cell 7 — Code (classify_node)

```python
def classify_node(state: QaState) -> QaState:
    q = state['question']
    prompt = f"""
        너는 질문 분류기야.
        아래 질문이 '수학','코딩','대화' 중 무엇인지 판단해.
        결과는 영어로 math / code / chat 중 하나만 출력해
        질문:{q}
    """
    resp = llm.invoke(prompt)
    text = resp.content.lower()

    # 'in' 체크라 LLM이 "code입니다" 처럼 부연설명을 붙여도 안전하게 걸림
    if "math" in text:intent="math"
    elif "code" in text:intent="code"
    else: intent="chat"   # math/code 둘 다 없으면 기본값 chat

    print(f"[classify_node] : intent = {intent}")
    return {**state, "intent":intent}
```

질문 하나를 받아 LLM에게 라벨링을 시키고, 문자열에 `"math"`/`"code"`가 포함돼 있는지로 느슨하게 판정. **default가 "chat"**이라는 점이 lgraph6의 `fallback` 노드와 다른 부분 — 별도 fallback 노드 없이 분류 실패 시 그냥 잡담으로 처리.

### Cell 8 — Markdown

`### 3-2. 분야별 답변 노드 (math / code / chat)` 섹션 헤더.

### Cell 9 — Code (math_node / code_node / chat_node)

```python
def math_node(state:QaState) -> QaState:
    q = state['question']
    prompt = f"""너는 수학 선생님이야. 아래 수학 문제를 단계별로 풀고 마지막 정답을 적어 줘. 문제:{q}"""
    answer = llm.invoke(prompt).content
    return {**state, "answer":answer}

def code_node(state:QaState) -> QaState:
    q = state['question']
    prompt = f"""너는 파이썬/자바 전문 프로머야. 아래 코딩 질문에 대해 개념을 설명하고 예시 코드를 만들어 줘. 질문:{q}"""
    answer = llm.invoke(prompt).content
    return {**state, "answer":answer}

def chat_node(state:QaState) -> QaState:
    q = state['question']
    prompt = f"""너는 친절한 한국어 챗봇이야. 아래 질문에 대해 10문장 내외로 부드럽고 우아하며 자연스런 문장으로 답해 줘. 질문:{q}"""
    answer = llm.invoke(prompt).content
    return {**state, "answer":answer}
```

세 노드 모두 구조는 동일(질문 받기 → 역할극 프롬프트 구성 → LLM 호출 → answer 저장), **시스템 역할(수학 선생님/개발자/챗봇)만** 다르게 준 패턴. 하나의 함수로 통합하고 프롬프트 딕셔너리만 바꾸는 리팩터링도 가능한 구조.

### Cell 10 — Markdown

`### 3-3. 답변 요약 노드` 섹션 헤더.

### Cell 11 — Code (summarize_node)

```python
def summarize_node(state:QaState) -> QaState:
    answer = state.get('answer', "")
    prompt = f"""아래 답변 내용을 50자 이내로 요약해 줘. [답변] {answer}"""
    summary = llm.invoke(prompt).content.strip()
    return {**state, "summary":summary}
```

직전 노드(math/code/chat 중 하나)가 만든 `answer`를 받아 50자 이내로 재요약. math/code/chat 세 경로 모두 이 노드로 합류하기 때문에, **분기 종류와 무관하게 항상 동일하게 실행되는 공통 후처리 단계**.

### Cell 12 — Markdown

`## 4. 분기 함수 및 그래프 구성` 섹션 헤더.

### Cell 13 — Code (route_by_intent, build_graph)

```python
def route_by_intent(state:QaState) -> str:
    intent = state.get("intent", "chat")
    print(f"[route_by_intent] intent = {intent}")
    return intent

def build_graph():
    graph = StateGraph(QaState)
    graph.add_node("classify", classify_node)
    graph.add_node("math", math_node)
    graph.add_node("code", code_node)
    graph.add_node("chat", chat_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify", route_by_intent,
        {"math":"math", "code":"code", "chat":"chat"},
    )

    # 세 분기 모두 summarize로 합류(fan-in) 후 종료
    graph.add_edge("math", "summarize")
    graph.add_edge("code", "summarize")
    graph.add_edge("chat", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()
```

`route_by_intent`는 `classify_node`가 세팅한 `intent` 값을 그대로 꺼내서 반환하는 얇은 래퍼(라우팅 키 결정 함수). 그래프 구조는: `classify` → (조건분기) → `math`/`code`/`chat` 중 하나 → `summarize` → `END`.

### Cell 14 — Markdown

`## 5. 실행` — 샘플 질문 3개(수학/코딩/대화)를 순서대로 실행하고 그래프 구조를 시각화한다는 설명.

### Cell 15 — Code (실행 루프 + 그래프 시각화)

```python
app = build_graph()

try:
    from IPython.display import Image, display
    graph_obj = app.get_graph()
    png_bytes = graph_obj.draw_mermaid_png()
    display(Image(data=png_bytes))   # 그래프 구조를 mermaid 다이어그램 이미지로 시각화 (Jupyter 전용)
except Exception as e:
    print('Jupyter notebook 환경이 아니므로 그래프 출력 불가')
    print('에러 : ', e)

qs = [
    "2차 방정식의 해를 구하는 공식을 설명해 줘",   # → math
    "파이썬에서 리스트와 튜플의 차이는 뭐야?",        # → code
    "장마철 꿀꿀한 기분을 달래줄 노래 추천해 줘"       # → chat
]

for q in qs:
    print("===========")
    print("질문 : ", q)
    init_state:QaState = {"question":q}
    final = app.invoke(init_state)
    print("intent : ", final.get("intent"))
    print("답변 : ", final.get("answer"))
    print("요약 : ", final.get("summary"))
```

3개 질문이 각각 math/code/chat으로 분류될 것을 노려서 만든 테스트 셋. `draw_mermaid_png()` 호출은 Jupyter/Colab처럼 이미지 렌더링이 되는 환경에서만 성공하고, 일반 스크립트 환경에서는 예외 처리로 안전하게 skip.

---

## 정리

- lgraph6과 골격은 같은 "분류 → 분기 → 답변" 패턴이지만, **분기 종류가 3개로 늘고 뒤에 요약 노드가 fan-in으로 붙은 확장판**.
- fallback 노드를 따로 두는 대신 `route_by_intent`의 default 값(`"chat"`)으로 분류 실패를 처리 — 노드를 줄이는 대신 로직이 함수 안에 숨어있는 트레이드오프.
- `{**state, "key":value}` 방식과 `{"key":value}`만 반환하는 방식 둘 다 LangGraph에서 동일하게 동작함 (내부적으로 state를 merge해주기 때문) — 팀 컨벤션 정할 때 참고.
- 확장한다면: summarize 결과를 다시 state에 누적해 대화형으로 여러 턴을 이어가거나, intent 카테고리를 더 세분화(예: "history", "science") 하는 방향으로 발전 가능.

---
# 📄 lgraph8_web.ipynb — LangGraph · Tavily · Web RAG

Tavily로 실시간 웹 검색 → 검색 결과 요약 → 요약을 근거로 질문에 답변하는 2단계 LangGraph RAG 파이프라인. 사내 문서(Chroma) 대신 실시간 웹이 지식 소스라는 점이 lgraph6/7과 다름.

> ⚠️ 이 노트북도 실행 이력(output)이 없는 상태로 업로드돼서, 아래 "실행 결과"는 실제 캡처가 아니라 코드 흐름상 예상되는 동작 설명임.

---

## 핵심 개념 정리

- **Realtime Web RAG**: 벡터DB에 미리 저장해둔 문서가 아니라, 질문이 들어올 때마다 **Tavily API로 실시간 웹 검색**을 수행해서 최신 정보를 근거로 답하는 방식. LLM 학습 시점 이후의 최신 이슈에도 대응 가능.
- **Tavily**: RAG용으로 최적화된 검색 엔진. 일반 웹 크롤링 결과보다 이미 정리된 JSON 형태로 반환돼서 LLM에 넣기 좋음.
- **2단계 파이프라인(fan-out 없는 선형 구조)**: 이 노트북은 조건 분기(`add_conditional_edges`)가 없고 `search_and_summarize → generate_answer → END`로 **일직선**으로 흐르는 가장 단순한 형태의 그래프. 분류/라우팅이 필요 없는 경우 LangGraph가 이렇게 단순 파이프라인으로도 쓰일 수 있음을 보여주는 예제.
- **검색 결과 요약 후 답변 생성을 분리한 이유**: 검색 결과(raw_results)를 바로 답변 프롬프트에 넣으면 광고/중복 등 노이즈가 많아 토큰 낭비 + 품질 저하. 중간에 "요약 노드"를 하나 끼워 넣어 노이즈를 걸러낸 뒤 최종 답변을 만드는 2단계 정제 구조.

---

## 셀별 설명

### Cell 0 — Markdown (제목)

노트북 목적 소개: LangGraph 노드 구조로 리팩토링한 실시간 웹 RAG 시스템. Tavily API 키가 별도로 필요하다는 안내 포함.

### Cell 1 — Code (패키지 설치)

```python
!pip install -U langgraph langchain-openai openai langchain-tavily python-dotenv
```

`langchain-tavily`가 핵심 추가 패키지 — Tavily 검색을 LangChain 툴 형태로 감싸주는 패키지.

### Cell 2 — Markdown

`## 1. State 정의 (LangGraph State)` 섹션 헤더.

### Cell 3 — Code (State 정의 + import)

```python
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import os, time
from dotenv import load_dotenv
load_dotenv()

class RAGState(TypedDict):
    question: str
    raw_results: Optional[str]   # Tavily 원본 검색 결과 (문자열로 변환해서 저장)
    summary: Optional[str]       # 노이즈 제거 후 요약된 검색 결과
    answer: Optional[str]        # 최종 답변
```

`Optional[str]`로 선언돼 있지만 `total=False`가 아니라서 실제로는 4개 키 모두 항상 존재해야 함(값이 `None`일 수는 있음). 파이프라인 진행에 따라 `raw_results → summary → answer` 순서로 하나씩 채워지는 구조.

### Cell 4 — Markdown

`## 2. 공용 LLM & Search 초기화` 섹션 헤더.

### Cell 5 — Code (search_tool, llm 초기화)

```python
search_tool = TavilySearch(
    max_results=5,  api_key=os.getenv("TAVILY_API_KEY"),
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
```

`max_results=5`: 검색당 상위 5개 결과만 가져옴. `temperature=0.0`: 답변의 일관성을 최우선으로 — 검색 결과 기반 사실 답변이라 창의성보다 정확성이 중요하기 때문으로 보임. (lgraph6/7은 0.2였던 것과 대비)

### Cell 6 — Markdown

`## 3. 프롬프트 정의 (요약용 / 답변용)` 섹션 헤더.

### Cell 7 — Code (summary_prompt, answer_prompt)

```python
summary_prompt = ChatPromptTemplate.from_template(
    """당신은 검색 결과를 정리하는 요약 전문가입니다.
    다음은 웹 검색 결과입니다: {search_results}
    요구사항:
    - 광고/홍보성/중복 내용을 제거하고
    - 서로 겹치는 내용은 합쳐서 핵심 정보만 5~10줄로 간결하게 정리하세요.
    검색 결과 요약:"""
)

answer_prompt = ChatPromptTemplate.from_template(
    """아래는 어떤 질문에 대해 웹 검색을 수행한 뒤 정리한 요약입니다:
    [검색 요약] {summary}
    [질문] {question}
    위 요약에 있는 정보만 사용해서 질문에 답변하세요.
    추측하거나 지어내지 말고, 모르는 내용은 모른다고 하세요.
    최종 답변:"""
)
```

`answer_prompt`의 "위 요약에 있는 정보만 사용해서... 추측하거나 지어내지 말고" 지시가 핵심 — **환각(hallucination) 방지**를 프롬프트 레벨에서 명시적으로 막는 전형적인 RAG 가드레일 문구.

### Cell 8 — Markdown

`### 4-1. 검색 + 요약 노드` 섹션 헤더.

### Cell 9 — Code (node_search_and_summarize)

```python
def node_search_and_summarize(state: RAGState) -> RAGState:
    """노드 A: Tavily 검색 + 요약"""
    question = state["question"]
    print(f" [노드 A] 검색 및 요약 진행 중: {question}")

    raw_results = search_tool.invoke({"query": question})   # 1) Tavily 실시간 웹 검색
    time.sleep(0.2)   # API rate limit 완충용 살짝 대기

    chain = summary_prompt | llm   # 2) 검색 결과 요약
    summary_msg = chain.invoke({"search_results": raw_results})
    summary_text = summary_msg.content

    new_state: RAGState = {
        "question": question,
        "raw_results": str(raw_results),   # dict/list일 수 있는 결과를 문자열로 강제 변환해 state에 저장
        "summary": summary_text,
        "answer": state.get("answer"),      # 아직 답변 전이므로 기존 값(보통 None) 유지
    }
    return new_state
```

한 노드 안에서 "검색"과 "요약" 두 작업을 순차로 처리. `time.sleep(0.2)`는 Tavily 응답 이후 짧은 지연을 둬서 API 호출 부담을 완화하려는 의도로 보임(엄밀한 rate limit 대응이라기보단 안전 마진).

### Cell 10 — Markdown

`### 4-2. 최종 답변 생성 노드` 섹션 헤더.

### Cell 11 — Code (node_generate_answer)

```python
def node_generate_answer(state: RAGState) -> RAGState:
    """노드 B: 요약 + 질문 → 최종 답변"""
    print(f"[노드 B] 최종 답변 생성 중...")

    question = state["question"]
    summary = state["summary"] or ""   # summary가 None일 경우를 대비한 방어 코드

    chain = answer_prompt | llm
    answer_msg = chain.invoke({"summary": summary, "question": question})
    answer_text = answer_msg.content

    new_state: RAGState = {
        "question": question,
        "raw_results": state.get("raw_results"),   # 이전 단계 값 그대로 유지(전달)
        "summary": summary,
        "answer": answer_text,
    }
    return new_state
```

직전 노드가 만든 `summary`만 근거로 최종 답변 생성. `raw_results`는 이 노드에서 쓰이진 않지만 state에 계속 들고 다니는 이유는 디버깅/로그용으로 원본 검색 결과를 남겨두기 위함으로 보임.

### Cell 12 — Markdown

`## 5. LangGraph 그래프 구성` 섹션 헤더.

### Cell 13 — Code (build_rag_graph)

```python
def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("search_and_summarize", node_search_and_summarize)
    graph.add_node("generate_answer", node_generate_answer)

    graph.set_entry_point("search_and_summarize")

    graph.add_edge("search_and_summarize", "generate_answer")   # 조건 분기 없이 순차 연결
    graph.add_edge("generate_answer", END)

    app = graph.compile()
    return app
```

분기 로직이 전혀 없는 가장 단순한 2노드 선형 그래프. `add_conditional_edges` 없이 `add_edge`만으로 구성 가능하다는 걸 보여주는 예시.

### Cell 14 — Markdown

`## 6. 실행` — 샘플 질문 3개에 대해 순차적으로 그래프를 실행한다는 설명.

### Cell 15 — Code (실행 루프)

```python
app = build_rag_graph()

questions = [
    "최신 AI 기술 동향은?",
    "한국에서 가장 인기 있는 빵은?",
    "한국의 방산 관련 최근 이슈는?",
]

for q in questions:
    print("\n===============")
    print("질문:", q)

    final_state = app.invoke({"question": q, "raw_results": None, "summary": None, "answer": None})

    print("\n[검색 요약]")
    print(final_state["summary"])
    print("\n[최종 답변]")
    print(final_state["answer"])
```

`invoke` 호출 시 4개 키를 모두 명시적으로 초기화해서 넘김 — `RAGState`가 `total=False`가 아니라 모든 키가 필수이기 때문에 (Cell 3 참고) 빠뜨리면 타입상 문제가 될 수 있어 매번 전체 초기 state를 채워서 넘기는 것으로 보임. 3개 질문 모두 시사성 있는 주제로, 실시간 검색이 필요함을 보여주려는 의도.

---

## 정리

- lgraph6/7이 "분류 후 분기"였다면, lgraph8은 **분기 없는 순차 파이프라인**(검색→요약→답변) — LangGraph가 라우팅뿐 아니라 단순 워크플로우 오케스트레이션에도 쓰일 수 있음을 보여줌.
- 지식 소스가 정적 벡터DB(Chroma)가 아니라 **실시간 웹 검색(Tavily)** 이라는 점이 가장 큰 차이 — "최신 정보"가 필요한 질문에 강함.
- "요약 → 답변" 2단계 분리는 노이즈 제거 + 프롬프트 품질 향상을 위한 흔한 RAG 정제 패턴으로, 다른 RAG 파이프라인에도 재사용 가능한 구조.
- 확장한다면: 검색 결과가 부실할 때 재검색하는 supervisor 패턴(lgraph10과 유사)을 붙이거나, Tavily 검색 파라미터(날짜 범위, 도메인 필터 등)를 질문 유형별로 다르게 주는 방향으로 발전 가능.