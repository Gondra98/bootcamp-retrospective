# Day 113_쇼핑 도우미 LangGraph 실습: 단일 체인 → Multi-Agent RAG로 확장

## 📅 2026-07-23

---
# 📄 lgraph9shopping.ipynb — langgraph · stategraph · 페르소나 검색어 생성

## 쇼핑 검색 도우미 (페르소나 기반 검색어 생성/검증)

### 개념 요약

LLM으로 사용자의 **페르소나(질문자 성향)**를 추출하고, 그 페르소나에 맞는 **쇼핑 검색어**를 생성 → 검증 → (필요시) 수정하여 최종 검색어와 답변을 제공하는 LangGraph 워크플로우. 여러 턴(질문-응답) 동안 `state`에 페르소나 정보와 대화 히스토리를 계속 누적/유지하는 것이 핵심.

**전체 흐름**

```
사용자 입력
  → 페르소나 추출/갱신 (extract_persona)
  → 검색어 후보 5개 생성 (generate_query_candidates)
  → 검색어 적합성 검증 (validate_queries)
  → 조건 분기: needs_fix?
      - True  → 검색어 수정 (fix_queries)
      - False → 기존 검색어 그대로 채택 (final_accept)
  → 최종 답변 생성 + 히스토리 저장 (build_reply)
  → 다음 사용자 입력 대기
```

---

## Cell 1 — 패키지 설치

```python
!pip install langgraph langchain langchain-openai openai python-dotenv
```

**설명:** LangGraph 기반 워크플로우 구성에 필요한 패키지 설치. `langgraph`(그래프/상태 관리), `langchain-openai`(OpenAI 모델 연동), `python-dotenv`(환경변수 로드)가 핵심.

---

## 1. 환경설정 및 모델 준비

## Cell 2 — LLM 모델 초기화

```python
from typing import TypedDict, List, Dict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 불러오기

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 없어요")  # 키 없으면 바로 예외 발생시켜 조기 확인

llm = ChatOpenAI(model="gpt-4.1-mini", temperature="0.5")  # 창의성과 일관성의 중간 정도 온도
```

**설명:** `.env`에서 API 키를 불러와 `gpt-4.1-mini` 모델 인스턴스 생성. `temperature=0.5`로 설정해 검색어 추천이 너무 획일적이지도, 너무 산만하지도 않게 조절.

---

## 2. LLM 출력에서 리스트/JSON 추출 함수

## Cell 3 — 출력 파싱 유틸 함수

```python
def extract_list_from_llm_output(text: str):
    # LLM이 ```json...``` 같은 코드블록 형식으로 출력하거나, json이 조금 깨져 있어도 최대한 리스트로 변환
    text = re.sub(r"^```(?:json|python)?\s*", "", text, flags=re.IGNORECASE)  # 시작 코드블록 표시 제거
    text = re.sub(r"\s*```$", "", text).strip()  # 마지막 코드블록 표시 제거

    # 1차 시도: json 문자열로 파싱
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2차 시도: 파이썬 리스트 리터럴로 파싱 (json.loads 실패 시)
    try:
        return eval(text)
    except Exception:
        pass

    # 둘 다 실패하면 원문 텍스트를 리스트 형태로 감싸서 반환 (최후의 안전장치)
    return [text]

print(extract_list_from_llm_output('```json\n["안녕", "반가워"]\n```'))
print(extract_list_from_llm_output("```python\n['사과', '바나나']\n```"))
print(extract_list_from_llm_output("안녕"))
```

**출력 결과**

```
['안녕', '반가워']
['사과', '바나나']
['안녕']
```

**설명:** LLM 응답은 마크다운 코드블록(` ```json `)으로 감싸져 오거나 JSON 형식이 미묘하게 깨지는 경우가 많음. 이 함수는 ① 코드블록 표시 제거 → ② `json.loads` 시도 → ③ 실패 시 `eval`로 파이썬 리터럴 파싱 시도 → ④ 그마저 실패하면 원문을 1개짜리 리스트로 반환하는 **3단계 폴백(fallback) 파서**. 이후 노드들에서 LLM 출력을 다룰 때 공통으로 재사용됨.

> ⚠️ `eval` 사용은 신뢰할 수 없는 외부 입력에 대해서는 보안상 위험할 수 있는 방식이라, 실서비스에서는 `ast.literal_eval` 등으로 대체하는 게 더 안전함.

---

## 3. State 정의

## Cell 4 — WorkflowState

```python
class WorkflowState(TypedDict, total=False):
    raw_input: str                  # 사용자 질문
    persona_info: dict              # 누적/갱신되는 사용자 페르소나 정보
    query_candidates: List[str]     # LLM이 생성한 검색 쿼리 후보 (수정 전)
    final_queries: List[str]        # 검증/수정이 끝난 최종 검색 쿼리 목록
    needs_fix: bool                 # 현재 후보 쿼리 수정 필요 여부
    reasoning: str                  # 수정여부에 대한 LLM의 판단 근거
    history: List[Dict[str, str]]   # 전체 대화 히스토리 (역할별 메시지 누적)
    reply: str                      # 이번 턴 에이전트가 사용자에게 답할 문장
```

**설명:** 그래프 전체에서 공유되는 상태 스키마. `total=False`이므로 모든 키가 선택적(optional) — 각 노드는 자신이 갱신할 키만 반환하면 LangGraph가 알아서 병합해줌. `persona_info`와 `history`는 턴이 반복될수록 누적되는 값이라는 점이 이 워크플로우의 핵심 포인트.

---

## 4. 노드 정의

### 4-1. Persona 추출/갱신 노드

## Cell 5 — extract_persona

```python
def extract_persona(state: WorkflowState) -> WorkflowState:
    # 이전까지 알고 있던 persona 정보 + 사용자의 이번 질문을 통한 페르소나 갱신
    prev_persona = state.get("persona_info", {})  # 첫 턴이면 빈 dict

    prompt = f"""
    너는 쇼핑 도우미 에이전트야
    아래의 지금까지 알고 있던 사용자 정보(persona)와 이번에 사용자가 말한 내용을 제공할게.
    1) 이전 persona를 반영하고
    2) 이번 질문 내용을 이용해 필요한 부분은 갱신해
    3) 최종 persona를 JSON(dict) 형식으로 만들어 줘.

    이전 persona는 비어 있을 수 있음.
    {json.dumps(prev_persona, ensure_ascii=False)}

    이번 사용자 질문 내용:
    {state['raw_input']}

    출력 형식:
    {{
        "age":30,
        "gender":"male",
        "style":"...",
        "favorite":"...",
    }}
    """
    persona = llm.invoke(prompt).content

    # 응답을 JSON으로 파싱, 실패하면 원문을 raw 키에 그대로 보관
    try:
        persona_dict = json.loads(persona)
    except Exception:
        persona_dict = {"raw": persona}

    return {"persona_info": persona_dict}
```

**설명:** 이전 페르소나(`prev_persona`)와 이번 사용자 발화를 함께 프롬프트에 넣어 **누적 갱신** 방식으로 페르소나를 업데이트. 노드는 `state` 전체가 아니라 `{"persona_info": ...}`만 반환 — LangGraph가 이 부분 딕셔너리를 기존 state에 merge하는 구조.

---

### 4-2. 검색어 후보 생성 노드

## Cell 6 — generate_query_candidates

```python
def generate_query_candidates(state: WorkflowState) -> WorkflowState:
    # 현재 페르소나 정보를 기반으로 쇼핑몰 검색어 후보 5개 정도 생성
    prompt = f"""
    아래 persona 정보를 참고해서, 이 사람이 쇼핑몰 검색에 사용할 것 같은 검색어를 5개 추천해 줘.

    persona:
    {json.dumps(state["persona_info"], ensure_ascii=False)}

    예:
    ["남성 여름용 반팔 티", "무선 이어폰", ...]
    """
    raw = llm.invoke(prompt).content
    parsed = extract_list_from_llm_output(raw)  # 코드블록/JSON 깨짐 대비 파싱

    return {"query_candidates": parsed}
```

**설명:** 앞 단계에서 갱신된 `persona_info`를 바탕으로 검색어 후보 5개를 생성. LLM 원본 출력(raw)은 형식이 들쭉날쭉할 수 있어 앞서 만든 `extract_list_from_llm_output`으로 안전하게 리스트화.

---

### 4-3. 검색어 검증 노드

## Cell 7 — validate_queries

```python
def validate_queries(state: WorkflowState) -> WorkflowState:
    prompt = f"""
    아래 검색 쿼리들이 persona에 맞는지 검사해 줘.

    persona:
    {json.dumps(state["persona_info"], ensure_ascii=False)}

    candidates:
    {state["query_candidates"]}

    결과는 JSON 형식으로 만들어 줘.
    {{
        "needs_fix":True 또는 False,
        "reson":"왜 그런 판단을 했는지 한글로 간단히 설명"
    }}
    """
    raw = llm.invoke(prompt).content
    try:
        res_dict = json.loads(raw)
    except Exception:
        try:
            res_dict = eval(raw)  # json.loads 실패 시 파이썬 딕셔너리 리터럴로 재시도
        except Exception:
            # 둘 다 실패하면 안전하게 "수정 불필요"로 기본값 처리
            res_dict = {"needs_fix": False, "reason": "파싱실패 -> 일단은 그대로 사용"}

    return {
        "needs_fix": res_dict.get("needs_fix", False),
        "reasoning": res_dict.get("reason", "")
    }
```

**설명:** 검색어 후보가 페르소나에 적합한지 LLM이 자체 평가(self-critique)하는 노드. `needs_fix` 값에 따라 다음 단계에서 그래프가 분기됨. 여기도 파싱 실패에 대비한 이중 폴백(`json.loads` → `eval` → 기본값) 구조.

> 💡 프롬프트의 키 이름이 `"reson"`(오타)으로 되어 있지만, 실제 파싱 시에는 `res_dict.get("reason", "")`로 읽고 있어 오타가 있는 응답이 오면 `reasoning`이 빈 문자열로 채워질 수 있음 — 실제 동작에 영향 줄 수 있는 부분이라 참고.

---

### 4-4. 검색어 수정 노드 / 확정 노드

## Cell 8 — fix_queries / final_accept

```python
def fix_queries(state: WorkflowState) -> WorkflowState:
    # needs_fix가 True일 때만 호출되는 노드: 기존 후보를 참고해 더 적합한 검색어로 재생성
    prompt = f"""
    persona:
    {json.dumps(state["persona_info"], ensure_ascii=False)}

    기존 검색 후보:
    {state["query_candidates"]}

    위 정보를 참고해서, 더 잘 맞는 새로운 검색어 5개를 만들어 줘
    출력은 반드시 파이썬 리스트 형태로 해야 해.
    예:["...", "...", ... ]
    """
    raw = llm.invoke(prompt).content
    parsed = extract_list_from_llm_output(raw)

    return {"final_queries": parsed}


# 검색 쿼리 확정 노드 (needs_fix가 False일 때: 기존 후보를 그대로 최종 채택)
def final_accept(state: WorkflowState) -> WorkflowState:
    return {"final_queries": state["query_candidates"]}
```

**설명:** 조건 분기의 두 갈래를 처리하는 노드 쌍. `fix_queries`는 검증 단계에서 부적합 판정을 받았을 때 새 검색어 5개를 재생성하고, `final_accept`는 적합 판정을 받았을 때 기존 후보를 그대로 `final_queries`로 승격시키는 단순 통과 노드. 두 노드 모두 결과적으로 `final_queries` 키를 채워서 다음 노드(`build_reply`)가 동일한 인터페이스로 처리할 수 있게 함.

---

### 4-5. 최종 응답 생성 노드

## Cell 9 — build_reply

```python
def build_reply(state: WorkflowState) -> WorkflowState:
    # final_queries를 바탕으로 사용자에게 보여줄 한글 응답 생성 후 대화 history를 갱신
    persona = state.get("persona_info", {})
    queries = state.get("final_queries", [])
    user_input = state["raw_input"]

    prompt = f"""
    너는 친절한 쇼핑 검색 도우미야.

    [사용자 정보(persona)]
    {json.dumps(persona, ensure_ascii=False)}

    [사용자 질문(발화)]
    {user_input}

    [추천 검색어 목록]
    {queries}

    위 정보를 바탕으로, 사용자가 쇼핑몰 검색 창에 바로 써 볼 수 있도록 추천 검색어 들을
    자연스럽게 설명해 줘.

    조건:
    - 한글로 5문장 정도.
    - 사용자 입장에서 쉽게 이해할 수 있도록 출력해.
    - 마크업 이런 것은 절대 출력하지 마!
    """
    reply = llm.invoke(prompt).content
    history = state.get("history", [])  # 지금까지의 대화 히스토리 가져오기
    history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": reply},
    ]

    return {
        "reply": reply,
        "history": history
    }
```

**설명:** 최종 검색어 목록(`final_queries`)을 사용자에게 친화적인 자연어 문장으로 풀어주는 노드. 동시에 `history`에 이번 턴의 user/assistant 메시지를 append하여, 다음 턴의 `extract_persona`가 참고할 수 있는 대화 맥락을 계속 축적.

---

## 5. 그래프 구성

## Cell 10 — build_graph

```python
def build_graph():
    graph = StateGraph(WorkflowState)

    # 노드 등록
    graph.add_node("extract_persona", extract_persona)
    graph.add_node("generate_queries", generate_query_candidates)
    graph.add_node("validate_queries", validate_queries)
    graph.add_node("fix_queries", fix_queries)
    graph.add_node("final_accept", final_accept)
    graph.add_node("build_reply", build_reply)

    graph.set_entry_point("extract_persona")  # 그래프 시작 노드 지정

    # 순차 흐름 설정
    graph.add_edge("extract_persona", "generate_queries")
    graph.add_edge("generate_queries", "validate_queries")

    # 조건 분기 함수: needs_fix 값에 따라 다음 노드를 다르게 라우팅
    def check_fix(state: WorkflowState):
        return "fix" if state.get("needs_fix") else "pass"

    graph.add_conditional_edges(
        "validate_queries",
        check_fix,
        {
            "fix": "fix_queries",     # needs_fix=True  → 검색어 재생성
            "pass": "final_accept",   # needs_fix=False → 기존 검색어 확정
        }
    )

    # 두 분기 모두 build_reply로 합류
    graph.add_edge("fix_queries", "build_reply")
    graph.add_edge("final_accept", "build_reply")
    graph.add_edge("build_reply", END)  # 그래프 종료

    return graph.compile()
```

**설명:** `StateGraph`에 6개 노드를 등록하고 엣지로 연결. 핵심은 `add_conditional_edges`로 만든 **조건 분기 구조** — `validate_queries`의 결과(`needs_fix`)에 따라 `fix_queries` 또는 `final_accept`로 갈라졌다가 둘 다 `build_reply`에서 합류하는 다이아몬드 형태 그래프. `graph.compile()`로 실행 가능한 앱 객체를 생성.

---

## 6. 실행 (대화 루프)

## Cell 11 — 대화 루프 실행

```python
app = build_graph()
state: WorkflowState = {"history": []}  # 최초 state는 빈 히스토리로 시작

print("쇼핑 도우미입니다. (종료:q)")
while True:
    user_input = input("사용자:").strip()
    if user_input.lower() in ("exit", "quit", "q"):
        print("대화를 종료합니다")
        break

    state["raw_input"] = user_input   # 1회 턴의 사용자 질문을 state에 주입

    state = app.invoke(state)         # 그래프 실행 → state가 갱신된 결과로 대체됨
    print("답변 : ", state["reply"], "\n")
```

**출력 결과 (실행 예시)**

```
쇼핑 도우미입니다. (종료:q)
사용자:반팔티 추천좀
답변 :  반팔티를 찾고 계시다면 '남성 반팔 티셔츠'라는 검색어로 시작해 보세요.
여름에 입기 좋은 시원한 옷을 원하시면 '여름용 반팔 티'도 추천합니다.
좀 더 편안하고 일상적인 스타일을 원한다면 '캐주얼 반팔 티'를 검색해 보세요.
피부에 부담 없고 부드러운 소재를 찾는다면 '면 반팔 티셔츠'가 딱 맞을 거예요.
마지막으로 깔끔하고 심플한 디자인을 좋아한다면 '심플 반팔 티'도 함께 검색해 보시면 좋습니다.

사용자:남성 반팔 여름용 티셔츠
답변 :  남성 반팔 여름용 티셔츠를 찾고 계시다면 '남성 여름 반팔 티셔츠'라는 검색어를 사용해 보세요.
시원한 소재를 원한다면 '쿨링 반팔 티'도 좋은 선택입니다.
통기성이 좋은 옷을 찾고 싶다면 '통기성 좋은 반팔'로 검색해 보시면 다양한 제품이 나와요.
좀 더 넓은 범위에서 여름용 남성 옷을 보고 싶다면 '남성 여름 옷'을 입력해 보세요.
마지막으로 '여름용 남성 티'도 간편하게 여름 티셔츠를 찾기에 적합한 검색어입니다.

사용자:닌텐도 스위치2
답변 :  닌텐도 스위치2에 관심이 많으시군요. 게임기 본체뿐만 아니라
닌텐도 스위치2 액세서리나 케이스도 함께 찾아보시면 좋을 것 같아요.
여름에 입기 좋은 남성 반팔 티셔츠도 추천해 드릴게요.
특히 쿨링 기능이 있는 티셔츠나 남자 여름 캐주얼 티도 시원하게 입기 좋아요.
이렇게 검색해 보시면 원하는 제품을 더 쉽게 찾으실 수 있을 거예요.
```

**설명:** `while True` 루프로 여러 턴을 이어가며 `app.invoke(state)`를 호출할 때마다 이전 `state`(누적된 `persona_info`, `history` 포함)를 그대로 넘겨 다음 턴에 반영. 실행 예시를 보면 3번째 턴(닌텐도 스위치2)에서 이전 턴의 페르소나(반팔티 관심)가 완전히 대체되지 않고 답변에 섞여 나오는 것을 확인할 수 있음 — `extract_persona`가 이전 정보를 "반영"하는 방식이라 페르소나가 누적/혼합되는 특성이 그대로 드러난 부분.

---

## 핵심 정리

|개념|설명|
|---|---|
|**StateGraph**|여러 노드가 공유 `state`(TypedDict)를 주고받으며 순차/조건 실행되는 LangGraph의 핵심 구조|
|**조건 분기 (add_conditional_edges)**|노드 실행 결과에 따라 다음 노드를 동적으로 선택 (`needs_fix` → fix/pass)|
|**부분 반환 + merge**|각 노드는 state 전체가 아닌 갱신할 키만 반환하고, LangGraph가 기존 state와 병합|
|**폴백 파싱**|LLM 출력이 항상 깔끔한 JSON이 아니므로 `json.loads` → `eval` → 원문 보존의 다단계 방어 로직 필요|
|**상태 누적**|`persona_info`, `history`를 턴마다 계속 갱신/축적하여 멀티턴 대화 맥락 유지|

---
# 📄 lgraph10multiagent.ipynb — LangGraph · Multi-Agent · RAG

## CSV + Chroma + SentenceTransformer + LangGraph Multi-Agent RAG (여성 쇼핑 도우미)

### 개념 요약

CSV 상품 데이터를 임베딩하여 ChromaDB(벡터DB)에 저장하고, LangGraph의 **Multi-Agent(Supervisor 패턴)** 구조로 여러 에이전트가 역할을 나눠 협업하며 대화형 쇼핑 추천을 수행하는 RAG 예제.

**RAG 기본 흐름**

```
data(CSV) loading → embedding → vectorDB 저장 → vectorDB에서 검색 → prompt 강화 → LLM 질문 → 결과
```

**Multi-Agent 구성 (Supervisor 패턴)**

|에이전트|역할|
|---|---|
|PersonaAgent|사용자 발화에서 여성 고객 페르소나(나이/스타일/관심사) 분석|
|QueryAgent|페르소나 + Supervisor 피드백을 반영해 검색어 5개 생성|
|RetrieverAgent|검색어를 임베딩 후 ChromaDB에서 유사 상품 검색|
|SupervisorAgent|검색 결과 품질을 검토해 `retry`(재검색) 또는 `reply`(답변 생성) 결정|
|ReplyAgent|최종 검색 상품을 바탕으로 자연어 추천 답변 생성|

이전 실습(쇼핑 검색 도우미)과의 차이는 **단일 LLM 호출 체인이 아니라, 각 역할별로 별도 에이전트/노드를 두고 Supervisor가 재검색 여부까지 판단**한다는 점. 즉 결정적(deterministic) 코드(ChromaDB 검색)와 LLM 기반 판단(Supervisor)이 그래프 안에서 함께 동작하는 구조.

---

## Cell 1 — 패키지 설치

```python
!pip install -U langgraph langchain-openai openai python-dotenv
!pip install -U sentence-transformers chromadb
```

**설명:** 기존 LangGraph/OpenAI 패키지에 더해 임베딩용 `sentence-transformers`와 벡터DB `chromadb`를 추가 설치. RAG 구현을 위한 필수 조합.

---

## 1. 환경설정 및 모델/DB 초기화

## Cell 2 — LLM/임베딩 모델/ChromaDB 초기화

```python
import os
import csv
import json
from typing import TypedDict, Literal, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 재현성을 위해 온도 0 (일관된 판단/검색어)

CSV_FILE = "women_products.csv"
CHROMA_DIR = "./chroma_women_shopping"          # ChromaDB 로컬 저장 경로
COLLECTION_NAME = "women_shopping_products"

# 다국어(한국어 포함) 지원 문장 임베딩 모델
EMBED_MODEL_NAME = ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

chroma_client = PersistentClient(path=CHROMA_DIR)   # 디스크에 영구 저장되는 Chroma 클라이언트
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"},  # 코사인 유사도 기반 인덱스
)
```

**출력 결과 (요약)**

```
[DB] 여성용 상품 6개 저장 완료
```

(모델 최초 다운로드 시 modules.json, config.json, model.safetensors 등 HuggingFace 파일 다운로드 로그 출력됨 — 이후 캐시되어 재실행 시 생략)

**설명:** `temperature=0`으로 설정해 페르소나/검색어/검토 판단의 일관성을 높임(이전 실습의 `temperature=0.5`보다 더 결정적인 값 사용). `paraphrase-multilingual-MiniLM-L12-v2`는 한국어를 포함한 다국어 문장 임베딩에 적합한 경량 모델. Chroma는 `PersistentClient`로 로컬 디스크에 저장해 세션이 끊겨도 인덱스가 유지되도록 구성.

---

## 2. 여성용 샘플 상품 DB 구성

## Cell 3 — prepare_product_db

```python
def prepare_product_db():
    """여성용 상품 CSV를 만들고 ChromaDB에 저장한다."""
    if not os.path.exists(CSV_FILE):
        rows = [
            ["id", "name", "category", "desc", "price"],
            ["w1", "여성 겨울 롱패딩", "아우터", "보온성이 좋은 여성용 롱패딩으로 한겨울에 적합함.", "159000원"],
            ["w2", "여성 발열 내의 세트", "이너웨어", "얇고 따뜻한 기모 안감의 여성용 발열 내의 세트.", "39000원"],
            ["w3", "여성 플리스 재킷", "아우터", "가볍고 따뜻하며 캐주얼 코디에 잘 어울리는 재킷.", "69000원"],
            ["w4", "여성 가죽 숄더백", "가방", "출근과 일상에서 사용할 수 있는 심플한 가죽 가방.", "119000원"],
            ["w5", "여성 앵클부츠", "신발", "쿠션감이 좋고 정장과 캐주얼에 모두 어울리는 부츠.", "99000원"],
            ["w6", "여성용 로즈골드 무선 이어폰", "테크", "노이즈 캔슬링과 최대 24시간 배터리를 지원함.", "129000원"],
        ]
        # CSV가 없을 때만 샘플 데이터 생성 (이미 있으면 기존 파일 그대로 사용)
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerows(rows)
        print(f"[DB] 여성용 샘플 CSV 생성: {CSV_FILE}")

    with open(CSV_FILE, "r", encoding="utf-8") as file:
        products = list(csv.DictReader(file))

    ids = [product["id"] for product in products]

    # 임베딩 대상 문서: name/category/desc를 하나의 문자열로 결합
    documents = [
        f'{product["name"]} / {product["category"]} / {product["desc"]}'
        for product in products
    ]

    # id를 제외한 나머지 컬럼을 메타데이터로 저장 (검색 결과 표시용)
    metadatas = [
        {key: value for key, value in product.items() if key != "id"}
        for product in products
    ]

    embeddings = embed_model.encode(documents, normalize_embeddings=True)  # 정규화된 임베딩 벡터 생성

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=[vector.tolist() for vector in embeddings],
    )
    print(f"[DB] 여성용 상품 {len(products)}개 저장 완료")
```

**설명:** CSV가 없으면 6개 샘플 상품(패딩/이너웨어/재킷/가방/부츠/이어폰)을 생성하고, 있으면 기존 CSV(업로드된 `women_products.csv`)를 그대로 읽음. `name + category + desc`를 하나의 문서로 합쳐 임베딩하는 것이 포인트 — 상품명만이 아니라 카테고리·설명까지 벡터에 반영해 검색 정확도를 높임. `collection.upsert`는 동일 id가 있으면 덮어쓰고 없으면 추가하므로 재실행해도 중복 저장되지 않음.

---

## 3. LLM 구조화 출력 모델 (Pydantic)

## Cell 4 — 구조화 출력 스키마 정의

```python
class PersonaProfile(BaseModel):
    target: str = "여성"            # 추천 상품 대상
    age: int | None = None       # 사용자가 언급한 나이
    style: str = ""                 # 선호 스타일
    interests: list[str] = Field(default_factory=list)
    needs: list[str] = Field(default_factory=list)


class SearchPlan(BaseModel):
    queries: list[str]                  # 상품 검색어 목록


class ReviewDecision(BaseModel):
    action: Literal["retry", "reply"]        # 재검색 또는 최종 답변
    feedback: str                       # 검색 결과 검토 의견


# with_structured_output: LLM이 Pydantic 스키마에 맞는 형태로만 응답하도록 강제
persona_llm = llm.with_structured_output(PersonaProfile)
query_llm = llm.with_structured_output(SearchPlan)
review_llm = llm.with_structured_output(ReviewDecision)
```

**설명:** 이전 실습(`lgraph9shopping`)에서는 LLM 출력을 문자열로 받아 `json.loads`/`eval`로 직접 파싱했지만, 이번엔 `with_structured_output(Pydantic모델)`을 사용해 **모델 자체가 구조화된 객체를 반환**하도록 강제. 이렇게 하면 별도의 폴백 파서(`extract_list_from_llm_output` 같은) 없이도 파싱 실패 위험이 크게 줄어듦 — 더 안정적인 방식으로 발전한 부분.

---

## 4. LangGraph 공유 상태

## Cell 5 — WorkflowState

```python
class WorkflowState(TypedDict, total=False):
    raw_input: str                               # 현재 사용자 입력
    persona_info: dict[str, Any]                 # 여성 고객 페르소나
    queries: list[str]                           # 검색어 목록
    retrieved_products: list[dict[str, Any]]     # 검색된 상품
    supervisor_feedback: str                     # 감독 에이전트 의견
    next_action: str                             # 다음 실행 경로
    retry_count: int                             # 재검색 횟수
    reply: str                                   # 최종 추천 답변
    history: list[dict[str, str]]                # 대화 기록
```

**설명:** 이전 실습 대비 `retrieved_products`(벡터 검색 결과), `supervisor_feedback`/`next_action`/`retry_count`(Supervisor의 재검색 제어용 필드)가 추가됨. `retry_count`는 무한 재검색 루프를 막기 위한 안전장치용 상태값.

---

## 5. Multi-Agent 노드

### 5-1. PersonaAgent

## Cell 6 — persona_agent

```python
def persona_agent(state: WorkflowState):
    """사용자 발화에서 여성 고객의 취향과 요구사항을 분석한다."""
    print("\n[PersonaAgent] 여성 고객 페르소나 분석")
    previous = state.get("persona_info", {})  # 이전 턴까지 누적된 페르소나

    profile = persona_llm.invoke(
        f"""
너는 여성 쇼핑 고객을 분석하는 PersonaAgent다.

이전 페르소나:
{json.dumps(previous, ensure_ascii=False)}

현재 사용자 입력:
{state["raw_input"]}

대상 고객은 여성으로 설정한다.
나이, 스타일, 관심 상품과 구매 목적을 정리하라.
"""
    )

    return {"persona_info": profile.model_dump()}  # Pydantic 객체 → dict 변환 후 state에 저장
```

**설명:** `persona_llm`이 `PersonaProfile` 스키마로 강제되어 있어 결과가 항상 `target/age/style/interests/needs` 필드를 가진 객체로 반환됨. `profile.model_dump()`로 dict화하여 state에 저장.

---

### 5-2. QueryAgent

## Cell 7 — query_agent

```python
def query_agent(state: WorkflowState):
    """페르소나와 Supervisor 피드백으로 검색어를 생성한다."""
    print("\n[QueryAgent] 여성용 상품 검색어 생성")

    plan = query_llm.invoke(
        f"""
너는 여성 패션·생활 상품 검색을 담당하는 QueryAgent다.

사용자 입력:
{state["raw_input"]}

여성 고객 페르소나:
{json.dumps(state["persona_info"], ensure_ascii=False)}

Supervisor 피드백:
{state.get("supervisor_feedback", "없음")}

한국 쇼핑몰에서 사용할 수 있는 여성용 검색어를
짧은 표현으로 5개 생성하라.
"""
    )

    # 빈 문자열 제거 + 최대 5개로 제한
    queries = [query.strip() for query in plan.queries if query.strip()][:5]

    if not queries:
        queries = [state["raw_input"]]  # 검색어 생성 실패 시 원본 입력을 검색어로 대체

    print("검색어:", queries)
    return {"queries": queries}
```

**설명:** 이 노드는 재검색(`retry`) 시에도 재호출되는데, 이때 `supervisor_feedback`(직전 검토 의견)이 프롬프트에 포함되어 이전 검색어의 문제를 반영한 개선된 검색어를 생성하도록 유도. 즉 QueryAgent는 Supervisor의 피드백 루프 안에서 동작하는 구조.

---

### 5-3. RetrieverAgent

## Cell 8 — retriever_agent

```python
def retriever_agent(state: WorkflowState):
    """검색어를 임베딩하고 ChromaDB에서 상품을 검색한다."""
    print("\n[RetrieverAgent] ChromaDB 상품 검색")

    products = []
    seen_ids = set()  # 여러 검색어에서 같은 상품이 중복 검색되는 것 방지

    for query in state.get("queries", []):
        query_vector = embed_model.encode(query, normalize_embeddings=True).tolist()

        output = collection.query(
            query_embeddings=[query_vector],
            n_results=3, include=["metadatas", "distances"],   # 검색어당 상위 3개
        )

        ids = output.get("ids", [[]])[0]
        metadatas = output.get("metadatas", [[]])[0]
        distances = output.get("distances", [[]])[0]

        for product_id, metadata, distance in zip(ids, metadatas, distances):
            if product_id in seen_ids:
                continue
            seen_ids.add(product_id)
            products.append({**metadata, "score": round(float(distance), 4)})  # 거리값을 score로 저장

    products.sort(key=lambda product: product["score"])  # 거리가 작을수록(유사할수록) 상위 정렬
    products = products[:8]  # 최종 상위 8개만 유지
    print(f"검색된 상품 수: {len(products)}")

    return {"retrieved_products": products}
```

**설명:** 5개 검색어 각각에 대해 ChromaDB에서 상위 3개씩 검색 후, 중복 제거 → 거리(distance, 코사인 거리이므로 작을수록 유사) 기준 정렬 → 상위 8개로 압축하는 **다중 쿼리 검색(multi-query retrieval)** 패턴. 이 방식은 단일 검색어보다 더 넓은 후보군을 확보할 수 있음.

---

### 5-4. SupervisorAgent

## Cell 9 — supervisor_agent

```python
def supervisor_agent(state: WorkflowState):
    """상품 검색 결과를 검토하고 재검색 또는 답변 생성을 결정한다."""
    print("\n[SupervisorAgent] 검색 결과 품질 검토")

    decision = review_llm.invoke(
        f"""
너는 여러 쇼핑 에이전트를 관리하는 SupervisorAgent다.

사용자 요청:
{state["raw_input"]}

여성 고객 페르소나:
{json.dumps(state["persona_info"], ensure_ascii=False)}

검색어:
{json.dumps(state["queries"], ensure_ascii=False)}

검색 결과:
{json.dumps(state.get("retrieved_products", [])[:6], ensure_ascii=False)}

현재 재검색 횟수:
{state.get("retry_count", 0)}

판단 기준:
- 여성 고객의 요청과 관련된 상품이 충분하면 reply
- 상품이 부족하거나 요구사항과 맞지 않으면 retry
- retry인 경우 QueryAgent가 개선할 수 있도록 피드백 작성
"""
    )

    retry_count = state.get("retry_count", 0)

    # 무한 반복 방지를 위해 재검색은 한 번만 허용
    if decision.action == "retry" and retry_count < 1:
        action = "retry"
        retry_count += 1
    else:
        action = "reply"

    print("다음 작업:", action)
    print("검토 의견:", decision.feedback)

    return {
        "next_action": action,
        "supervisor_feedback": decision.feedback,
        "retry_count": retry_count,
    }
```

**설명:** 이 워크플로우의 핵심 노드. LLM이 `retry`를 원하더라도 `retry_count < 1` 조건으로 **재검색은 최대 1회로 코드 레벨에서 강제 제한**함 — LLM의 판단(비결정적)과 파이썬 조건문(결정적)을 함께 써서 무한 루프를 방지하는 대표적인 패턴.

---

### 5-5. ReplyAgent

## Cell 10 — reply_agent

```python
def reply_agent(state: WorkflowState):
    """최종 상품 추천 답변을 생성한다."""
    print("\n[ReplyAgent] 최종 여성용 상품 추천 생성")
    products = state.get("retrieved_products", [])[:4]  # 상위 4개만 답변에 사용

    product_text = "\n".join(
        f'- {product["name"]} / {product["price"]} / {product["desc"]}'
        for product in products
    )

    if not product_text:
        product_text = "관련 상품을 찾지 못했습니다."

    reply = llm.invoke(
        f"""
너는 여성 고객에게 상품을 추천하는 ReplyAgent다.

사용자 요청:
{state["raw_input"]}

여성 고객 페르소나:
{json.dumps(state["persona_info"], ensure_ascii=False)}

검색 상품:
{product_text}

Supervisor 검토 의견:
{state.get("supervisor_feedback", "")}

작성 조건:
1. 사용자 취향을 한 문장으로 정리
2. 적합한 상품 2~4개 추천
3. 각 상품의 추천 이유 설명
4. 검색되지 않은 상품은 만들어 내지 않기
5. 자연스러운 한국어로 작성
"""
    ).content

    # 이번 턴의 user/assistant 메시지를 history에 누적
    history = state.get("history", []) + [
        {"role": "user", "content": state["raw_input"]},
        {"role": "assistant", "content": reply},
    ]

    return {
        "reply": reply,
        "history": history,
    }
```

**설명:** 검색된 상위 4개 상품 정보만 프롬프트에 넣고, "검색되지 않은 상품은 만들어 내지 않기"라는 **환각(hallucination) 방지 지시**를 조건에 명시한 것이 포인트. (아래 실행 예시에서 이 지시가 실제로는 지켜지지 않은 사례가 보여서 참고용으로 짚어둠.)

---

## 6. Multi-Agent 그래프 구성

## Cell 11 — build_graph

```python
def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("persona_agent", persona_agent)
    graph.add_node("query_agent", query_agent)
    graph.add_node("retriever_agent", retriever_agent)
    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("reply_agent", reply_agent)

    graph.add_edge(START, "persona_agent")
    graph.add_edge("persona_agent", "query_agent")
    graph.add_edge("query_agent", "retriever_agent")
    graph.add_edge("retriever_agent", "supervisor_agent")

    # Supervisor가 검색 결과에 따라 재검색 또는 답변 생성을 결정
    graph.add_conditional_edges(
        "supervisor_agent",
        lambda state: state["next_action"],
        {
            "retry": "query_agent",     # 검색어 재생성 → 재검색 루프
            "reply": "reply_agent",     # 최종 답변 생성으로 이동
        },
    )

    graph.add_edge("reply_agent", END)
    return graph.compile()
```

**설명:** `START`(LangGraph 내장 시작 노드) → persona → query → retriever → supervisor로 이어지다가, supervisor의 판단(`next_action`)에 따라 **query_agent로 되돌아가는 루프(retry)** 또는 **reply_agent로 진행(reply)**하는 순환 구조. 이전 실습의 단순 분기(다이아몬드형)와 달리, 이번엔 그래프 안에 실제 **루프(cycle)**가 존재한다는 점이 구조적 차이.

---

## 7. 실행

## Cell 12 — DB 준비 → 그래프 컴파일 → 대화 루프

```python
prepare_product_db()
app = build_graph()

# 선택 사항: Colab/Jupyter에서 그래프 구조를 이미지로 시각화
try:
    from IPython.display import Image, display
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    pass

state: WorkflowState = {"persona_info": {}, "history": []}
print("\n여성용 Multi-Agent RAG 쇼핑 도우미입니다.\n종료: exit / quit / q\n")

while True:
    user_input = input("사용자: ").strip()

    if user_input.lower() in {"exit", "quit", "q"}:
        print("에이전트: 대화를 종료합니다.")
        break

    # 새 대화 턴마다 Supervisor 재검색 상태 초기화 (persona/history는 유지)
    state.update(
        {
            "raw_input": user_input,
            "retry_count": 0,
            "supervisor_feedback": "",
            "next_action": "",
        }
    )

    state = app.invoke(state)
    print("\n에이전트:", state["reply"])
    print()
```

**출력 결과 (실행 예시 — "로션 추천좀")**

```
[DB] 여성용 상품 6개 저장 완료

여성용 Multi-Agent RAG 쇼핑 도우미입니다.
종료: exit / quit / q

사용자: 로션 추천좀

[PersonaAgent] 여성 고객 페르소나 분석
[QueryAgent] 여성용 상품 검색어 생성
검색어: ['자연 성분 로션', '유기농 보습 로션', '피부 진정 로션 추천', '자극 없는 로션', '건강한 피부 로션']

[RetrieverAgent] ChromaDB 상품 검색
검색된 상품 수: 4

[SupervisorAgent] 검색 결과 품질 검토
다음 작업: retry
검토 의견: 검색 결과에 로션 관련 제품이 포함되어 있지 않습니다. ...

[QueryAgent] 여성용 상품 검색어 생성
검색어: ['자연 성분 로션', '유기농 보습 로션', '피부 진정 로션', '자극 없는 로션', '여성용 스킨케어 로션']

[RetrieverAgent] ChromaDB 상품 검색
검색된 상품 수: 4

[SupervisorAgent] 검색 결과 품질 검토
다음 작업: reply
검토 의견: 검색 결과에 로션 관련 제품이 포함되어 있지 않습니다. ...

[ReplyAgent] 최종 여성용 상품 추천 생성

에이전트: 고객님은 자연스럽고 건강한 피부를 추구하며, 보습과 피부 진정에 효과적인
자극 없는 자연 성분의 로션을 원하십니다.

추천 상품은 다음과 같습니다:

1. 유기농 알로에 베라 로션
   - 추천 이유: 알로에 베라 성분이 함유되어 있어 피부를 진정시키고 보습 효과가 뛰어납니다...
2. 자연 유래 오일 보습 로션
   - 추천 이유: 다양한 자연 유래 오일이 포함되어 있어 피부에 깊은 보습을 제공합니다...
3. 허브 추출물 수분 로션
   - 추천 이유: 여러 가지 허브 추출물이 포함되어 있어 피부를 진정시키고 수분을 공급합니다...
```

**설명:** DB 준비 후 그래프를 mermaid 이미지로 시각화(선택 사항)한 뒤 대화 루프 진입. 턴마다 `retry_count`/`supervisor_feedback`/`next_action`은 초기화하지만 `persona_info`와 `history`는 유지해 멀티턴 맥락을 보존.

> ⚠️ **실행 결과에서 확인된 주의점**: `women_products.csv`(샘플 6개 상품: 롱패딩·발열내의·재킷·가방·부츠·이어폰)에는 로션류 상품이 전혀 없음에도, SupervisorAgent가 두 번 모두 "로션 관련 제품 없음"을 확인하고도 결국 `reply`로 넘어갔고, ReplyAgent는 프롬프트 조건 4번("검색되지 않은 상품은 만들어 내지 않기")을 어기고 "유기농 알로에 베라 로션" 등 **DB에 존재하지 않는 상품을 생성(환각)**해서 추천함. Supervisor가 재검색을 1회로 제한한 상태에서 여전히 결과가 부적합할 경우 무조건 reply로 넘어가는 구조라, 이런 경우 "관련 상품 없음"으로 답하도록 하는 별도 처리(예: 검색 결과가 임계 유사도 이하이면 reply_agent에서 명시적으로 "해당 상품 없음" 응답)를 추가하는 게 좋아 보임.

---

## 핵심 정리

| 개념                                       | 설명                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------- |
| **RAG (Retrieval-Augmented Generation)** | 벡터DB 검색 결과로 프롬프트를 보강한 뒤 LLM에 질의하는 구조                                      |
| **Multi-Agent Supervisor 패턴**            | 역할별 에이전트(Persona/Query/Retriever/Reply)를 두고, Supervisor가 재검색 여부를 판단해 라우팅  |
| **with_structured_output**               | Pydantic 스키마로 LLM 출력 형식을 강제 → 수동 JSON 파싱/폴백 로직 불필요                        |
| **그래프 내 루프(cycle)**                      | `add_conditional_edges`로 조건에 따라 이전 노드(query_agent)로 되돌아가는 재검색 루프 구성       |
| **retry_count 가드**                       | LLM의 재검색 판단을 신뢰하되, 코드에서 횟수를 강제 제한해 무한 루프 방지                               |
| **환각(hallucination) 리스크**                | "검색되지 않은 상품은 만들어 내지 않기"라는 프롬프트 지시만으로는 완전히 막지 못할 수 있음 — 코드 레벨 검증이 필요할 수 있음 |
