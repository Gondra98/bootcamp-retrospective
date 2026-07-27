# Day 115_LangGraph 병렬 워크플로

## 📅 2026-07-27

---
# 🔀 LangGraph 병렬 워크플로 — Fan-out · Fan-in · Parallel vs Async

LangGraph 병렬 워크플로 (Parallel Workflow) 하나의 작업을 여러 독립 작업으로 나눠 동시에 실행한 뒤, 결과를 모아 다음 단계로 넘기는 처리 구조. Fan-out/Fan-in 개념과 병렬(Parallel) vs 비동기(Async)의 차이를 함께 정리.

---

## 핵심 구조

```
        START
          ↓
         init
   ↙      ↓      ↘
weather  news   stocks     ← fan-out (병렬 실행)
   ↘      ↓      ↙
   combine_results          ← fan-in (결과 통합)
          ↓
      llm_report
          ↓
         END
```

1. **Fan-out (분기)**: 하나의 노드에서 여러 노드로 작업을 나눔. 서로 결과를 기다릴 필요 없음.
2. **병렬 실행**: 각 노드가 동일한 초기 상태를 받아 각자 다른 상태 키(`weather`, `news`, `stocks`)에 결과를 저장 → 충돌 없음.
3. **Fan-in (통합)**: 모든 병렬 작업이 끝난 후 `combine_results`에서 합침.

## 순차 vs 병렬

| |순차 처리|병렬 처리|
|---|---|---|
|방식|날씨→뉴스→주식 (하나씩)|날씨‖뉴스‖주식 (동시)|
|소요 시간 (각 2초)|약 6초|약 2초 (가장 오래 걸리는 작업 기준)|

**병렬이 적합한 경우**: 서로 의존하지 않는 작업 (API 다중 호출, DB 다중 조회, 문서 다중 검색 등) **순차가 필요한 경우**: 뒤 작업이 앞 작업 결과를 필요로 할 때 (질문 분석 → 검색어 생성 → 문서 검색 → 답변 생성)

## 비유로 이해하기

- 순차: 한 사람이 심부름 3개를 차례로 처리 (날씨 확인 → 뉴스 확인 → 주식 확인)
- 병렬: 사람 3명이 각자 심부름 1개씩 동시에 처리 → 끝나면 "결과 모아서 리포트로 만들어줘" (fan-in)

### 병렬 워크플로우를 만드는 핵심 아이디어 3개

1. **독립적인 작업들은 동시에 실행할 수 있다** — 서로 영향을 주지 않는 작업은 함께 실행 가능 (LangGraph의 fan-out 구조)
2. **각각의 결과는 나중에 한 곳에서 모을 수 있다** — 모든 작업이 끝나면 결과를 합치는 fan-in 단계 등장
3. **전체 상태(State)를 하나로 관리하면 서로 데이터를 주고받을 수 있다** — weather, news, stocks 각각 다른 키에 기록되어 전체 상태 딕셔너리에 모임

## Parallel vs Async (자주 헷갈리는 개념)

|특징|병렬 (Parallel)|비동기 (Async)|
|---|---|---|
|실제 실행 시점|진짜 동시에 실행|동시처럼 "보이는" 실행|
|CPU 코어|여러 코어 사용|한 코어로도 가능|
|목적|속도 향상 (계산량 분산)|대기 시간 절약 (I/O 효율)|
|예시|이미지 10개 동시 처리|HTTP 요청 보내고 기다리지 않음|

- **병렬**: 요리사 2명이 각각 라면/볶음밥을 동시에 조리 → 물리적으로 동시에 실행 (Multi-threading, Multi-processing)
- **비동기**: 라면 끓이는 동안 빨래 대기 안 하고 다른 일 처리 → CPU는 하나지만 대기 시간을 다른 작업에 씀 (`async/await`)

## LangGraph에서의 의미

- LangGraph는 **병렬 실행이 "가능한" 구조(fan-out/fan-in)**를 그래프로 표현해줄 뿐, 실제 CPU 수준 병렬 실행을 보장하지는 않음.
- 진짜 병렬/비동기 실행을 원하면 `async def` 노드, executor/worker 설정, 멀티프로세스/스레드 백엔드가 필요.
- 정리:
    - **LangGraph = 병렬 구조 설계**
    - **Python 실행 방식 = 실제 병렬/비동기 여부 결정**


---

## ⚠️ 꼭 기억할 핵심 개념

> **절대 잊지 말 것**

- **RAG** (Retrieval-Augmented Generation)
- **ReAct** (Reasoning + Acting)
- **Vector** (벡터DB)

---
# 📄 lgraph13parall.ipynb — LangGraph · Parallel Workflow · Fan-out·Fan-in

LangGraph 병렬 워크플로 실습 (날씨/뉴스/주식 병렬 조회 → 통합 → LLM 리포트) 날씨·뉴스·주식 3개 노드를 fan-out으로 동시 실행하고, fan-in으로 결과를 모아 LLM에게 자연어 보고서를 작성시키는 실습 코드. 셀별 코드 + 주석 + 예상 출력 정리.

---

## 개념 정리

- **Fan-out**: 하나의 노드(`init`)에서 여러 노드(`fetch_weather`, `fetch_news`, `fetch_stock`)로 분기 → 서로 의존성 없는 작업은 동시 실행 가능
- **Fan-in**: 분기된 작업이 모두 끝난 뒤 한 노드(`combine_result`)로 모여 결과 통합
- **State**: `TypedDict(total=False)`로 정의 → 병렬 노드들이 각자 다른 키(weather/news/stocks)에 결과를 써도 충돌 없음
- **Checkpointer(MemorySaver)**: `thread_id` 단위로 상태를 저장/복원 → 같은 thread_id면 이어서 실행 가능

```
        START
          ↓
         init
   ↙      ↓      ↘
weather  news   stock      ← fan-out
   ↘      ↓      ↙
   combine_result           ← fan-in
          ↓
      llm_report
          ↓
         END
```

---

## Cell 0 — 패키지 설치

```python
# LangGraph + Parallel Workflow
# Fan-out, Fan-in

!pip install langgraph langchain-openai openai langchain-chroma sentence-transformers python-dotenv
```

**설명**: 이번 실습에 필요한 패키지 설치. `langgraph`(그래프 구성), `langchain-openai`(OpenAI LLM 연동), `python-dotenv`(.env 파일에서 API 키 로드) 가 핵심.

---

## Cell 1 — 환경 설정 & LLM 초기화

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict

import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수(OPENAI_API_KEY 등) 로드

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")  # 키 없으면 바로 실행 중단

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)  # 사용할 LLM 정의
```

**설명**: `.env`에서 API 키를 불러오고, 없으면 명확한 에러로 즉시 종료. `temperature=0.5`로 적당히 창의적이면서도 일관된 응답 유도.

---

## Cell 2 — State 정의

```python
class State(TypedDict, total=False):
    location: str        # 날씨/뉴스 조회 지역
    symbols: List[str]   # 조회할 주식 종목 코드 리스트
    weather: str         # fetch_weather 결과
    news: List[str]      # fetch_news 결과
    stocks: Dict[str, float]  # fetch_stocks 결과
    summary: str         # combine_results가 만든 요약 텍스트
    llm_report: str       # llm_report가 만든 최종 보고서
```

**설명**: `total=False` → 모든 키가 선택적(optional). 병렬 노드들이 실행 순서와 상관없이 자기 키만 채워도 에러 없음. Fan-out 구조에서 필수적인 설계.

---

## Cell 3 — 병렬 노드 정의 (fetch_weather / fetch_news / fetch_stocks)

```python
# Node
def init_state(state: State) -> dict:
    # 초기 진입 노드 - 입력값 확인용 로그만 찍고 상태 변경 없음
    print(f"init_state : 입력 location={state.get('location')}, symbols={state.get('symbols')}")
    return {}

def fetch_weather(state: State) -> dict:
    # 날씨 정보 읽기 (실습용 더미 데이터)
    loc = state.get("location", "Seoul")
    weather_text = f"{loc} : 맑고 가끔 구름. 30도"
    print(f"fetch_weather 날씨 데이터 처리 완료")
    return {"weather": weather_text}  # state의 'weather' 키만 갱신

def fetch_news(state: State) -> dict:
    # 뉴스 정보 읽기 (실습용 더미 데이터)
    loc = state.get("location", "Seoul")
    news_list = [
        f"{loc} : 경제 성장률 상향 조정",
        f"{loc} : IT 기업, AI 투자 확대",
        f"{loc} : 증시, 외국인 차익 실현",
    ]
    print("fetch_news 뉴스 데이터 처리 완료")
    return {"news": news_list}  # state의 'news' 키만 갱신

def fetch_stocks(state: State) -> dict:
    # 주식 정보 읽기 (실습용 더미 데이터)
    symbols = state.get("symbols") or ["AAPL", "GOOG", "TSLA"]
    dummy_prices = {
        "AAPL": 210.5,
        "GOOG": 135.2,
        "TSLA": 186.5,
        "NVDA": 125.5
    }
    print("fetch_stocks 주식 데이터 처리 완료")
    # symbols에 없는 종목은 dummy_prices.get() 기본값 100.0으로 처리
    stocks = {sym: dummy_prices.get(sym, 100.0) for sym in symbols}
    return {"stocks": stocks}  # state의 'stocks' 키만 갱신
```

**설명**: 세 함수 모두 자기 담당 키(`weather` / `news` / `stocks`)만 반환 → LangGraph가 병합할 때 서로 덮어쓰지 않음. 이게 fan-out에서 "충돌 없이 동시 실행" 가능한 이유.

**예상 출력 (symbols=["AAPL","TSLA","NVDA"] 기준)**

```
init_state : 입력 location=Seoul, symbols=['AAPL', 'TSLA', 'NVDA']
fetch_weather 날씨 데이터 처리 완료
fetch_news 뉴스 데이터 처리 완료
fetch_stocks 주식 데이터 처리 완료
```

---

## Cell 4 — 결과 통합(Fan-in) & LLM 리포트 생성

```python
def combine_results(state: State) -> dict:
    # 병렬로 처리된 날씨/뉴스/주식 결과를 하나의 요약 문자열로 합치기
    weather = state.get("weather", "날씨 정보 없음")
    news = state.get("news", [])
    stocks = state.get("stocks", {})

    lines = []
    lines.append("오늘의 종합 브리핑(요약 데이터) ---")
    lines.append(f"[날씨]\n- {weather}")

    lines.append(f"[뉴스]")
    if news:
        for i, n in enumerate(news, start=1):
            lines.append(f" {i}. {n}")
    else:
        lines.append(" - 뉴스 없음")

    lines.append(f"[주식]")
    if stocks:
        for sym, price in stocks.items():  # dict.items() 그대로 순회
            lines.append(f" - {sym}: {price} USD")
    else:
        lines.append(f" - 주식 자료 없음")

    summary_text = "\n".join(lines)
    print("[combine_results] 요약 완료")
    return {"summary": summary_text}


def llm_report(state: State) -> dict:
    # summary + 원본 데이터를 합쳐 LLM에게 자연어 보고서 작성 요청
    weather = state.get("weather", "")
    news = state.get("news", [])
    stocks = state.get("stocks", [])
    summary = state.get("summary", "")

    prompt = (
        "너는 경제/시장 보고서 작성 전문가야.\n"
        "아래에 오늘의 날씨, 뉴스, 주식 데이터가 있어.\n"
        "이를 바탕으로 일반인 이해하기 쉬운 한국어 보고서(양식은 일반 양식)를 작성해 줘.\n"
        "중요 트렌드, 투자자 관점에서 눈여겨볼 만한 포인트도 같이 정리해 줘.\n\n"
        "[원시 데이터]"
        f"{summary}\n"
        "[구조화된 데이터]"
        f" - 날씨 : {weather}\n"
        f" - 뉴스 : {news}\n"
        f" - 주식 : {stocks}\n"
        "위 정보를 모두 참고해서 자연스럽게 서술형 기사 형태로 작성해 줘"
    )

    print("[llm_report] LLM 호출 시작~~~")
    resp = llm.invoke(prompt)  # 실제 OpenAI API 호출
    report_text = str(resp.content)
    return {"llm_report": report_text}
```

**설명**:

- `combine_results`는 fan-in 노드 → weather/news/stocks가 **모두 준비된 후에만** 실행됨.
- `stocks.items()`를 바로 순회(이전에 `enumerate()`로 잘못 감쌌던 버그 수정 반영)해서 `{종목: 가격}` 형태로 정상 출력.
- `llm_report`는 summary(가공된 텍스트) + 원본 구조화 데이터를 함께 프롬프트에 넣어 LLM이 더 풍부한 보고서를 쓰도록 유도.

**예상 출력**

```
[combine_results] 요약 완료
[llm_report] LLM 호출 시작~~~
```

(이후 `final_state["summary"]`, `final_state["llm_report"]` 값은 Cell 6에서 출력됨)

---

## Cell 5 — 그래프 구성 (노드/엣지 연결)

```python
# 실행 구조
# START
#   |
# init
#   |
#   | -> fetch_weather    |
#   | -> fetch_news       | -> combine_result -> llm_report -> END
#   | -> fetch_stocks     |

def build_app():
    graph = StateGraph(State)

    # 노드 등록
    graph.add_node("init", init_state)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("fetch_news", fetch_news)
    graph.add_node("fetch_stock", fetch_stocks)
    graph.add_node("combine_result", combine_results)
    graph.add_node("llm_report", llm_report)

    # 흐름 정의
    graph.add_edge(START, "init")

    # "init" 이후 세 개의 작업을 병렬로 실행 (Fan-out)
    graph.add_edge("init", "fetch_weather")
    graph.add_edge("init", "fetch_news")
    graph.add_edge("init", "fetch_stock")

    # Fan-out이 끝나면 combine_result로 모이는 구조 (Fan-in)
    graph.add_edge("fetch_weather", "combine_result")
    graph.add_edge("fetch_news", "combine_result")
    graph.add_edge("fetch_stock", "combine_result")

    graph.add_edge("combine_result", "llm_report")
    graph.add_edge("llm_report", END)

    # CheckPointer 설정
    # MemorySaver() : state를 메모리에 저장/복원하는 간단한 체크포인터
    # 같은 thread_id로 여러 번 실행하면 '이어서 실행' 패턴 가능
    checkpointer = MemorySaver()

    app = graph.compile(checkpointer=checkpointer)
    return app
```

**설명**:

- `add_node`로 등록한 이름과 실제 함수는 별개 (`fetch_stock` 노드 이름 ↔ `fetch_stocks` 함수 — 이름이 살짝 달라도 정상 동작, 헷갈리지 않게 통일하면 더 좋음).
- `combine_result`로 들어오는 엣지가 3개 → LangGraph는 3개 입력 노드가 **모두 완료될 때까지 자동으로 기다렸다가** `combine_result` 실행 (fan-in의 핵심 동작).
- `MemorySaver`는 그래프 상태를 메모리에 저장 → 같은 `thread_id`로 재실행 시 이전 상태 이어받기 가능 (대화형 세션, 중단 후 재개 등에 활용).

---

## Cell 6 — 실행

```python
if __name__ == "__main__":
    app = build_app()

    # 체크포인터 사용 시 세션 구분용 thread_id 필요
    config = {
        "configurable": {
            "thread_id": "wns-llm-1"   # 임의 문자열, 세션 구분용
        }
    }

    # 초기 입력 상태
    init_state_value: State = {
        "location": "Seoul",
        "symbols": ["AAPL", "TSLA", "NVDA"],
    }

    # 전체 실행 (동기, 최종 상태만 반환)
    final_state: State = app.invoke(init_state_value, config=config)

    print("[최종 summary] : ", final_state.get("summary"))
    print("[LLM이 생성한 보고서] : ", final_state.get("llm_report"))

    # stream()으로 노드별 진행 상황 출력 (선택)
    print("\n노드별 진행 상태 보기 ====")
    for step in app.stream(
        init_state_value,
        config={"configurable": {"thread_id": "stream-demo-llm"}},
        stream_mode="values"  # 각 단계별 state 전체를 dict 형태로 출력
    ):
        print("\n[stream step] 현재 state 키:", list(step.keys()))
```

**설명**:

- `app.invoke()`: 그래프 전체를 한 번에 실행하고 **최종 state**만 반환.
- `app.stream(..., stream_mode="values")`: 각 노드가 끝날 때마다 **그 시점의 전체 state**를 하나씩 yield → 병렬/순차 흐름을 단계별로 눈으로 확인하기 좋음.
- 두 번째 실행은 `thread_id`가 달라서(`stream-demo-llm`) 첫 번째 실행과 별개의 세션으로 처리됨.

**예상 출력 (요약)**

```
init_state : 입력 location=Seoul, symbols=['AAPL', 'TSLA', 'NVDA']
fetch_weather 날씨 데이터 처리 완료
fetch_news 뉴스 데이터 처리 완료
fetch_stocks 주식 데이터 처리 완료
[combine_results] 요약 완료
[llm_report] LLM 호출 시작~~~
[최종 summary] :  오늘의 종합 브리핑(요약 데이터) ---
[날씨]
- Seoul : 맑고 가끔 구름. 30도
[뉴스]
 1. Seoul : 경제 성장률 상향 조정
 2. Seoul : IT 기업, AI 투자 확대
 3. Seoul : 증시, 외국인 차익 실현
[주식]
 - AAPL: 210.5 USD
 - TSLA: 186.5 USD
 - NVDA: 125.5 USD
[LLM이 생성한 보고서] :  (LLM이 생성한 자연어 기사형 보고서 텍스트)

노드별 진행 상태 보기 ====

[stream step] 현재 state 키: ['location', 'symbols']
[stream step] 현재 state 키: ['location', 'symbols']
[stream step] 현재 state 키: ['location', 'symbols', 'weather', 'news', 'stocks']
[stream step] 현재 state 키: ['location', 'symbols', 'weather', 'news', 'stocks', 'summary']
[stream step] 현재 state 키: ['location', 'symbols', 'weather', 'news', 'stocks', 'llm_report', 'summary']
```

> ⚠️ `stream_mode="values"`에서 fan-out 3개 노드는 각각 도착 순서대로 state가 누적되며 출력되므로, 위 출력의 중간 단계 수·순서는 실행 환경에 따라 약간 달라질 수 있음 (병렬 실행 특성).

---

## 핵심 정리 3줄

1. 병렬 노드(`fetch_weather/news/stocks`)는 **서로 다른 state 키**만 건드려야 충돌 없이 fan-out 가능.
2. `combine_result`처럼 여러 엣지가 모이는 노드는 **모든 입력이 끝나야 자동 실행**되는 것이 fan-in.
3. `MemorySaver` + `thread_id`로 세션별 상태 저장/재개가 가능하며, `stream(stream_mode="values")`로 병렬 실행 과정을 단계별로 확인할 수 있다.