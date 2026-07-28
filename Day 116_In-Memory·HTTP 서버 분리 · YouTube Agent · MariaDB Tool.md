# Day 116_In-Memory·HTTP 서버 분리 · YouTube Agent · MariaDB Tool

## 📅 2026-07-28

---
## 🔌 MCP(Model Context Protocol) — Agent to Tool 연결 표준

### MCP란 무엇인가
LLM 애플리케이션이 외부 도구·데이터·시스템에 연결되는 방식을 표준화한 프로토콜. 모델마다 API 연동을 따로 구현하는 대신, MCP 서버가 제공하는 기능을 표준 방식으로 발견·호출하도록 만든 구조다.

#### 핵심 구조
```
사용자 → MCP Host(Claude Desktop, IDE 등) → MCP Client → MCP Server → 외부 시스템(DB, API, VectorDB 등)
```

| 구성요소 | 역할 |
|---|---|
| MCP Host | LLM이 들어있는 애플리케이션 |
| MCP Client | 서버와 통신하는 연결 담당 |
| MCP Server | 외부 기능을 표준 방식으로 제공 |
| Tool | LLM이 호출 가능한 실행 함수 |
| Resource | LLM에게 제공되는 데이터/문서 |
| Prompt | 서버가 제공하는 재사용 가능한 템플릿 |

하나의 MCP 클라이언트는 여러 서버에 동시 접속 가능하다. Web API, DB, GitHub, 로컬 파일시스템처럼 각 기능을 개별 MCP 서버로 두고 하나의 AI 애플리케이션이 이들을 동시에 연결하는 구조.

### Tool이 핵심인 이유
MCP Tool은 이름·설명·입력 스키마를 가진 외부 함수. `@mcp.tool()` 데코레이터로 정의하고, LangChain에서는 `client.get_tools()`로 도구 목록을 가져와 Agent에 연결한다.

```python
tools = await client.get_tools()
agent = create_agent(model=llm, tools=tools)
```

핵심: GPT가 기능을 직접 가진 게 아니라, **MCP 서버의 Tool을 가져와 필요할 때 호출**하는 구조.

### 기존 방식과의 차이

| 구분 | 일반 API 연동 | Function Calling | MCP |
|---|---|---|---|
| 연결 방식 | 기능마다 직접 구현 | LLM API 내부 정의 | 표준 프로토콜 |
| 도구 위치 | 앱 코드 내부 | 앱 코드 내부 | MCP 서버(외부) |
| 재사용성 | 낮음 | 특정 앱 한정 | 여러 Host에서 재사용 |
| 확장성 | 기능 추가마다 수정 | 제한적 | 서버 추가로 확장 |

### MCP와 RAG의 관계 ⭐ (강사님 강조)
MCP는 RAG를 대체하지 않는다. 역할이 다르다.

- **RAG** = 검색 구조 (Document → Embedding → VectorDB → Retriever)
- **MCP** = 연결 구조 (Tool 호출 → 결과 반환)

둘을 결합하면 **RAG 검색기 자체를 MCP Tool로 감싸서** 여러 Host(LangChain Agent, Claude Desktop, Cursor, 사내 챗봇)에서 재사용 가능한 외부 도구로 분리할 수 있다.

```python
@mcp.tool()
def search_documents(query: str, k: int = 3) -> str:
    """VectorDB에서 질문과 관련된 문서를 검색한다."""
    docs = retriever.invoke(query)
    return "\n".join(f"- {doc.page_content}" for doc in docs[:k])
```

**처리 흐름**: 질문 → Agent 판단 → `search_documents` Tool 호출 → MCP 서버가 VectorDB 검색 → 결과 반환 → LLM 최종 답변 생성

이렇게 하면 RAG 검색 기능이 특정 코드에 갇히지 않고, 표준 인터페이스로 여러 애플리케이션에서 공유된다.

### MCP vs A2A(Google)

| 구분 | MCP | A2A |
|---|---|---|
| 공개 | Anthropic | Google |
| 연결 대상 | AI 앱 ↔ 도구·데이터 | AI 에이전트 ↔ AI 에이전트 |
| 목적 | 외부 기능 사용 | 여러 에이전트 협업 |

- **MCP = Agent to Tool**
- **A2A = Agent to Agent**

여행 예약 예시: MCP로 항공권/호텔 API를 호출하고, A2A로 일정 관리 에이전트와 조율하는 식으로 상호 보완적으로 쓰인다.

### 보안 주의점
Tool 권한 최소화, 삭제·결제 같은 위험 작업은 사용자 확인 필수, 서버 반환값 무조건 신뢰 금지, Prompt Injection 주의, 실행 로그 남기기, 민감 정보는 LLM에게 그대로 전달하지 않기.

---

# 📄 mcp1.ipynb — FastMCP · In-Memory Client · Tool 등록

### MCP 기초 실습 — 서버/클라이언트 한 파일에서 운영

별도의 MCP 서버 프로세스를 띄우지 않고, 하나의 파이썬 파일 안에서 `FastMCP` 서버 객체와 `Client`를 함께 만들어 MCP 기본 동작(도구 등록 → 조회 → 호출)을 익히는 가장 단순한 실습이다.

---

### 개념 정리

MCP 실습은 크게 두 가지 방식으로 나뉜다.

|방식|특징|이 노트북|
|---|---|---|
|In-Memory (서버 X)|서버/클라이언트가 같은 프로세스, `Client(mcp)`처럼 서버 객체를 직접 전달|✅ 여기 해당|
|HTTP 서버 분리|서버는 `mcp.run(transport="http", ...)`로 별도 실행, 클라이언트는 URL로 접속|다음 실습(mcp3 등)에서 진행|

이번 실습은 **네트워크 통신 없이** MCP의 핵심 흐름(Tool 등록 → 목록 조회 → 호출 → 결과 반환)만 빠르게 확인하기 위한 단계다.

**처리 구조**

```
외부 프로그램/API/DB/응용프로그램
   ↓ MCP Tool로 감싸서 등록
MCP 서버 실행
   ↓
MCP 클라이언트가 서버에 접속
   ↓ 도구 목록 조회 및 호출
MCP 서버가 실제 기능 실행
   ↓
실행 결과를 클라이언트에 반환
```

---

### 코드 + 주석

**1. 패키지 설치**

```python
# fastmcp: MCP 서버·클라이언트를 쉽게 만들 수 있게 해주는 파이썬 라이브러리
!pip install -q -U fastmcp
```

**2. MCP 서버 정의 및 도구(Tool) 등록**

```python
import asyncio
from fastmcp import FastMCP     # MCP 서버를 만들기 위한 클래스
from fastmcp import Client      # MCP 서버에 접속해 도구를 조회·호출하기 위한 클래스

# MCP 서버 인스턴스 생성 (서버 이름은 임의로 지정 가능)
mcp = FastMCP("Simple MCP Server")

# @mcp.tool 데코레이터를 붙이면 일반 함수가 MCP Tool로 등록되어
# Client가 원격(혹은 인메모리)으로 호출할 수 있게 된다.
@mcp.tool
def addFunc(a: int, b: int) -> int:
    """ 두 수를 더하기하는 도구 """
    return a + b

@mcp.tool
def greetFunc(name: str) -> str:
    """ 이름을 받아 인사말을 반환하는 도구 """
    return f"안녕하세요, {name}님!"
```

- 함수의 **docstring**이 그대로 MCP Tool의 `description`으로 등록된다 (아래 출력 결과 참고).
- 타입 힌트(`a: int, b: int`)는 Tool의 입력 스키마(inputSchema)를 자동 생성하는 데 쓰인다.

**3. MCP 클라이언트로 도구 조회 및 호출**

```python
# 서버 객체(mcp)를 직접 넘겨서 클라이언트 생성 → HTTP 없이 같은 프로세스 안에서 통신
client = Client(mcp)

# MCP 클라이언트의 연결 및 도구 호출은 비동기(async)로 처리해야 한다
async def mainFunc():
    async with client:  # 연결 시작 (블록을 벗어나면 자동 종료)
        # 등록된 MCP 서버 도구 목록 조회
        tools = await client.list_tools()
        for tool in tools:
            print("도구명 : ", tool.name)
            print("설명 : ", tool.description)

        # addFunc 도구 호출 (a=10, b=20 전달)
        add_result = await client.call_tool(
            "addFunc",
            {"a": 10, "b": 20}
        )
        print("addFunc 수행 결과 : ", add_result)

        # greetFunc 도구 호출 (name="홍길동" 전달)
        greet_result = await client.call_tool(
            "greetFunc",
            {"name": "홍길동"}
        )
        print("greetFunc 수행 결과 : ", greet_result)

await mainFunc()
```

---

### 출력 결과

```
도구명 :  addFunc
설명 :  두 수를 더하기하는 도구 
도구명 :  greetFunc
설명 :  이름을 받아 인사말을 반환하는 도구 

addFunc 수행 결과 :  CallToolResult(
    content=[TextContent(type='text', text='30', ...)],
    structured_content={'result': 30},
    data=30,
    is_error=False
)

greetFunc 수행 결과 :  CallToolResult(
    content=[TextContent(type='text', text='안녕하세요, 홍길동님!', ...)],
    structured_content={'result': '안녕하세요, 홍길동님!'},
    data='안녕하세요, 홍길동님!',
    is_error=False
)
```

**결과 해석 포인트**

- 도구 호출 결과는 단순 값이 아니라 `CallToolResult` 객체로 반환된다.
- 실제 값만 꺼내려면 `.data` 속성을 쓰면 된다 (`add_result.data` → `30`).
- `content` 필드는 LLM에게 텍스트 형태로 전달하기 위한 `TextContent` 리스트이고, `structured_content`는 구조화된(JSON형) 결과다. 용도에 따라 둘 중 필요한 걸 꺼내 쓰면 된다.

---

### 정리

- 이 실습은 MCP의 가장 기본 골격 — **Tool 등록 → 목록 조회 → 호출 → 결과 반환** — 을 네트워크 통신 없이 최소 구성으로 확인한 것이다.
- 다음 단계(HTTP 서버 분리 실습)에서는 `mcp.run(transport="http", host=..., port=...)`로 서버를 별도 프로세스로 띄우고, 클라이언트가 URL(`http://127.0.0.1:8000/mcp`)로 접속하는 구조로 확장된다.
- `CallToolResult.data`로 실제 값을 꺼내는 패턴은 이후 YouTube MCP, MariaDB MCP 실습에서도 동일하게 적용된다.

---
# 📄 mcp2.ipynb — FastMCP · HTTP Transport · 서버/클라이언트 파일 분리

### HTTP MCP 서버-클라이언트 분리 실습

`mcp1.ipynb`에서는 서버(`mcp`)와 클라이언트(`Client(mcp)`)가 한 프로세스 안에서 in-memory로 통신했다면, 이번 실습은 **서버를 별도 프로세스로 띄우고, 클라이언트가 HTTP로 접속**하는 실제 MCP 사용 패턴에 가까운 구조다.

---

### 개념 정리

|구분|mcp1 (In-Memory)|mcp2 (HTTP 분리)|
|---|---|---|
|서버-클라이언트 관계|같은 프로세스, 객체 직접 전달|별도 프로세스, 네트워크로 통신|
|Client 생성 방식|`Client(mcp)`|`Client("http://127.0.0.1:8000/mcp")`|
|서버 실행|불필요 (객체만 존재)|`mcp.run(transport="http", ...)`로 명시적 기동 필요|
|실무 유사도|낮음 (학습용)|높음 (실제 MCP 서버 배포 형태)|

**처리 구조**

```
simple_mcp_server.py (백그라운드 실행, 포트 8000)
       ↑ HTTP 통신
simple_mcp_client.py (Client가 URL로 접속)
       ↓
도구 목록 조회 → 도구 호출 → 결과 반환
```

Colab처럼 노트북 환경에서는 서버와 클라이언트를 같은 셀에서 동시에 실행할 수 없기 때문에, `%%writefile`로 각각 `.py` 파일을 만들고 서버는 `nohup ... &`로 **백그라운드 실행**한 뒤 클라이언트를 별도로 실행하는 방식을 쓴다.

---

### 코드 + 주석

**1. 패키지 설치**

```python
# HTTP 서버/클라이언트를 별도 파일로 분리해서 실습하기 위한 설치
!pip install -q -U fastmcp
```

**2. 서버 파일 작성 (`simple_mcp_server.py`)**

```python
# %%writefile은 반드시 셀의 첫 줄이어야 하며, 이 셀에는 이 매직 명령어 하나만 있어야 한다.
%%writefile simple_mcp_server.py

from fastmcp import FastMCP

mcp = FastMCP("Simple HTTP MCP Server")

# MCP 도구 등록 - Client가 HTTP로 접근 가능
@mcp.tool
def addFunc(a: int, b: int) -> int:
    """ 두 수를 더하기하는 도구 """
    return a + b

@mcp.tool
def greetFunc(name: str) -> str:
    """ 이름을 받아 인사말을 반환하는 도구 """
    return f"안녕하세요, {name}님!"

if __name__ == "__main__":
    # transport="http"로 지정하면 실제 HTTP 서버로 동작
    # 기본 MCP 엔드포인트 경로는 /mcp
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8000
    )
```

**3. 서버 백그라운드 실행**

```python
# 기존에 같은 포트로 서버가 떠 있으면 먼저 종료 (포트 충돌 방지)
!pkill -f simple_mcp_server.py || true

# nohup + & 로 백그라운드 실행, 로그는 별도 파일로 저장
!nohup python simple_mcp_server.py > simple_mcp_server.log 2>&1 &
!cat simple_mcp_server.log
```

> ⚠️ 이 셀의 실제 출력은 `^C`만 찍혀 있어 서버 기동 로그가 정상적으로 안 보였다. Colab에서 백그라운드 프로세스가 완전히 뜨기 전에 `cat`이 실행되면 로그가 비어 보일 수 있는데, 이 경우 사이에 `!sleep 2~3`을 넣어 서버가 뜰 시간을 벌어주는 게 안전하다. (다행히 이후 클라이언트 실행 결과 서버는 정상적으로 떠 있었다.)

**4. 클라이언트 파일 작성 (`simple_mcp_client.py`)**

```python
%%writefile simple_mcp_client.py

import asyncio
from fastmcp import Client

# URL로 접속 → HTTP를 통해 원격(혹은 로컬 다른 프로세스) MCP 서버와 통신
client = Client("http://127.0.0.1:8000/mcp")

def print_tool_result(result):
    """ CallToolResult 객체에서 실제 텍스트 값만 뽑아 출력하는 헬퍼 함수 """
    if hasattr(result, "content"):
        for content in result.content:
            if hasattr(content, "text"):
                print(content.text)
            else:
                print(content)
    else:
        print(result)

async def main():
    async with client:
        print("MCP 서버 연결 성공")

        # 등록된 MCP 서버 도구 목록 조회
        tools = await client.list_tools()
        for tool in tools:
            print("도구명 : ", tool.name)
            print("설명 : ", tool.description)

        # addFunc 도구 호출
        add_result = await client.call_tool(
            "addFunc",
            {"a": 10, "b": 20}
        )
        print_tool_result(add_result)

        # greetFunc 도구 호출
        greet_result = await client.call_tool(
            "greetFunc",
            {"name": "홍길동"}
        )
        print_tool_result(greet_result)

if __name__ == "__main__":
    asyncio.run(main())
```

**5. 클라이언트 실행**

```python
!python simple_mcp_client.py
```

---

### 출력 결과

```
MCP 서버 연결 성공
도구명 :  addFunc
설명 :  두 수를 더하기하는 도구 
도구명 :  greetFunc
설명 :  이름을 받아 인사말을 반환하는 도구 
30
안녕하세요, 홍길동님!
```

**결과 해석 포인트**

- `print_tool_result` 헬퍼로 `CallToolResult` 객체 대신 실제 값(`30`, `안녕하세요, 홍길동님!`)만 깔끔하게 출력됐다.
- mcp1과 동일한 도구(`addFunc`, `greetFunc`)를 그대로 재사용했지만, 이번엔 서버와 클라이언트가 **완전히 분리된 프로세스**로 HTTP를 통해 통신했다는 점이 핵심 차이다.

---

### 정리

- 이번 실습으로 **서버 프로세스를 독립적으로 띄우고, 클라이언트가 URL로 접속하는** MCP의 실제 배포 형태를 확인했다.
- `nohup ... &`로 서버를 백그라운드 실행하는 패턴은 이후 YouTube MCP, MariaDB MCP 서버 실습에서도 동일하게 반복된다.
- `%%writefile`은 반드시 셀의 첫 줄에 단독으로 있어야 하며, 다른 `!` 명령어나 두 번째 `%%writefile`과 한 셀에 섞이면 파일이 의도대로 생성되지 않는다는 점을 이번에도 재확인.
- 다음 단계에서는 이 구조 그대로 YouTube 자막 조회, MariaDB 조회 같은 실제 기능을 가진 Tool로 확장된다.

---
# 📄 mcp3.ipynb — YouTube MCP Server · LangChain Agent · 자막 요약

### YouTube 자막 조회 MCP 서버 + GPT Agent 연동 실습

mcp2에서 익힌 "서버 분리 + HTTP 통신" 구조를 실제 기능(YouTube 자막 조회)에 적용하고, 여기에 LangChain `MultiServerMCPClient`와 GPT Agent를 연결해서 **자막을 가져와 요약하는 파이프라인**까지 완성하는 실습이다.

---

### 개념 정리 — 전체 구성

|파일|역할|
|---|---|
|`youtube_mcp_server.py`|MCP 서버. YouTube URL → video_id 추출, 자막 조회 Tool 4종 제공|
|`check_mcp_tools.py`|MCP 도구 목록만 확인하는 테스트용 클라이언트|
|`youtube_agent_client.py`|LangChain + GPT Agent 실행 파일. MCP Tool을 Agent에 연결해 실제 요약 수행|

**핵심 구조**

```
YouTube URL
   ↓
GPT Agent
   ↓
MCP Client (LangChain MultiServerMCPClient)
   ↓
YouTube MCP Server
   ↓
get_youtube_transcript_plain 도구 실행
   ↓
스크립트 반환
   ↓
GPT가 요약
```

> ⚠️ 주의: YouTube 영상에 자막이 없거나 접근이 제한되면 스크립트를 못 가져올 수 있다. 특히 Shorts는 자막이 없는 경우가 많아, 실습용으로는 자막이 확실한 일반 영상을 준비하는 게 좋다.

**전체 실행 순서**

1. 패키지 설치 (`fastmcp`, `youtube-transcript-api`, `langchain`, `langchain-openai`, `langchain-mcp-adapters`)
2. `.env` 파일 준비 (OPENAI_API_KEY 등)
3. `youtube_mcp_server.py` 파일 생성
4. MCP 서버 백그라운드 실행
5. `check_mcp_tools.py`로 도구 목록 확인
6. `youtube_agent_client.py` 실행 → GPT Agent 실습

---

### 코드 + 주석

**1. MCP 서버 — video_id 추출 로직**

```python
%%writefile youtube_mcp_server.py
from fastmcp import FastMCP
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

mcp = FastMCP("YouTube Transcript MCP Server")

def extract_video_id(youtube_url_or_id: str) -> str:
    """
    YouTube URL 또는 video_id에서 video_id만 추출한다.
    지원 형식:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - VIDEO_ID (이미 순수 ID인 경우)
    """
    text = youtube_url_or_id.strip()

    # 이미 video_id만 들어온 경우 (도메인 문자열이 없으면 그대로 반환)
    if "youtube.com" not in text and "youtu.be" not in text:
        return text

    parsed = urlparse(text)

    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        # 일반 영상 URL: ?v=VIDEO_ID
        if "v" in query:
            return query["v"][0]
        # Shorts URL: /shorts/VIDEO_ID
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] == "shorts":
            return path_parts[1]

    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/").split("/")[0]

    return ""
```

- URL 형식 3가지(watch, shorts, youtu.be)를 모두 처리하는 공통 헬퍼 함수. MCP Tool은 아니고, 아래 4개 Tool 내부에서 공통으로 재사용된다.

**2. MCP Tool 4종 등록**

```python
@mcp.tool()
def get_video_id(youtube_url_or_id: str) -> str:
    """ YouTube URL에서 video_id만 추출해서 반환한다. 디버깅용 도구. """
    video_id = extract_video_id(youtube_url_or_id)
    if not video_id:
        return "video_id를 추출하지 못했습니다."
    return video_id


@mcp.tool()
def list_youtube_transcripts(youtube_url_or_id: str) -> str:
    """ YouTube 영상에서 사용 가능한 자막 언어 목록을 확인한다. """
    video_id = extract_video_id(youtube_url_or_id)
    if not video_id:
        return "유효한 YouTube 영상 ID를 찾지 못했습니다."
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        lines = [
            f"language={t.language}, language_code={t.language_code}, "
            f"is_generated={t.is_generated}, is_translatable={t.is_translatable}"
            for t in transcript_list
        ]
        return "\n".join(lines) if lines else "사용 가능한 자막이 없습니다."
    except Exception as e:
        return f"자막 목록을 가져오지 못했습니다. 원인: {type(e).__name__}: {str(e)}"


@mcp.tool()
def get_youtube_transcript(youtube_url_or_id: str, language: str = "ko") -> str:
    """
    시간 정보가 포함된 자막 스크립트를 가져온다.
    language 기본값은 ko, 실패 시 en도 함께 시도한다.
    """
    video_id = extract_video_id(youtube_url_or_id)
    if not video_id:
        return "유효한 YouTube 영상 ID를 찾지 못했습니다."
    try:
        ytt_api = YouTubeTranscriptApi()
        if language == "ko":
            transcript = ytt_api.fetch(video_id, languages=["ko", "en"])
        else:
            transcript = ytt_api.fetch(video_id, languages=[language, "ko", "en"])
        lines = [f"[{round(item.start, 2)}초] {item.text.replace(chr(10), ' ')}" for item in transcript]
        return "\n".join(lines)
    except Exception as e:
        return f"스크립트를 가져오지 못했습니다. 원인: {type(e).__name__}: {str(e)}"


@mcp.tool()
def get_youtube_transcript_plain(youtube_url_or_id: str, language: str = "ko") -> str:
    """ 시간 정보 없이 스크립트 텍스트만 가져온다. 요약용으로 사용하기 좋다. """
    video_id = extract_video_id(youtube_url_or_id)
    if not video_id:
        return "유효한 YouTube 영상 ID를 찾지 못했습니다."
    try:
        ytt_api = YouTubeTranscriptApi()
        if language == "ko":
            transcript = ytt_api.fetch(video_id, languages=["ko", "en"])
        else:
            transcript = ytt_api.fetch(video_id, languages=[language, "ko", "en"])
        text_list = [item.text.replace(chr(10), " ") for item in transcript]
        return " ".join(text_list)
    except Exception as e:
        return f"스크립트를 가져오지 못했습니다. 원인: {type(e).__name__}: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)
```

- 4개 도구 모두 `extract_video_id` → `try/except`로 실패 원인까지 반환하는 구조로 일관성 있게 설계되어 있다.
- `get_video_id`는 디버깅용, `list_youtube_transcripts`는 자막 언어 확인용, `get_youtube_transcript`(타임스탬프 O) / `get_youtube_transcript_plain`(타임스탬프 X, 요약용)은 용도별로 분리되어 있다.

**3. 서버 백그라운드 실행**

```python
!pkill -f youtube_mcp_server.py || true
!nohup python youtube_mcp_server.py > youtube_mcp_server.log 2>&1 &
!sleep 3
!cat youtube_mcp_server.log
```

**4. 도구 목록 확인용 클라이언트**

```python
%%writefile check_mcp_tools.py
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    # LangChain에서 MCP 서버에 접속하기 위한 클라이언트
    # 여러 MCP 서버를 동시에 등록할 수 있어 이름을 붙여 딕셔너리로 관리한다 ("youtube": {...})
    client = MultiServerMCPClient(
        {
            "youtube": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8000/mcp/",
            }
        }
    )

    # 서버가 제공하는 Tool 목록을 LangChain Tool 객체로 변환해서 가져온다
    tools = await client.get_tools()

    print("=== YouTube MCP 서버에서 가져온 도구 목록 ===")
    for tool in tools:
        print("-", tool.name)
        print(" ", tool.description)

if __name__ == "__main__":
    asyncio.run(main())
```

**5. GPT Agent 실행 파일**

```python
%%writefile youtube_agent_client.py
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다.")

    client = MultiServerMCPClient(
        {
            "youtube": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8000/mcp/",
            }
        }
    )

    # MCP 서버의 Tool을 LangChain Tool 형태로 가져와 Agent에 그대로 연결
    tools = await client.get_tools()

    print("=== 연결된 MCP 도구 ===")
    for tool in tools:
        print("-", tool.name)

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
        당신은 유튜브 영상 요약 도우미입니다.

        규칙:
        1. 사용자가 YouTube URL을 주면 반드시 MCP 도구를 사용해 스크립트를 가져오세요.
        2. 스크립트 내용을 바탕으로 한국어로 요약하세요.
        3. 영상에 없는 내용은 임의로 만들어내지 마세요.
        4. 가능하면 핵심 내용, 주요 개념, 학습 포인트로 나누어 설명하세요.
        """
    )

    youtube_url = "https://www.youtube.com/watch?v=8nt3edWLgIg"

    question = f"""
        다음 유튜브 영상의 스크립트를 가져와서 교육생이 이해하기 쉽게 요약해줘.

        URL:
        {youtube_url}
        """

    # Agent가 스스로 MCP Tool 호출이 필요한지 판단하고, 필요하면 자동 실행
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    print("\n최종 요약 결과 : ")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 디버깅 기록 — `%%writefile` 중복/혼합 셀 버그

이 노트북에서 반복적으로 발생한 문제: **`%%writefile`은 셀당 한 번, 반드시 첫 줄이어야 한다.**

|증상|원인|해결|
|---|---|---|
|`youtube_mcp_server.py`가 안 만들어지고 `abc.py`에 내용이 통째로 섞여 들어감|한 셀에 `%%writefile abc.py`와 `%%writefile youtube_mcp_server.py`가 함께 있어, 두 번째 매직이 인식 안 되고 이후 내용 전체가 첫 번째 writefile의 텍스트로 들어감|`%%writefile`이 들어간 셀은 **그 명령어 하나만** 있도록 분리|
|`!python youtube_mcp_server.py` → `No such file or directory`|위 버그로 인해 파일 자체가 생성된 적이 없었음|파일 생성 셀 분리 후 재실행, `!ls`로 파일 존재 확인|
|`check_mcp_tools.py`, `youtube_agent_client.py` 실행 시 `httpx.ConnectError: All connection attempts failed`|서버가 애초에 뜬 적이 없으니 8000번 포트 접속 자체가 실패|서버 파일이 정상 생성·실행됐는지 먼저 확인 (`ps aux`, 로그 확인) 후 재시도|
|`!pip install`과 `%%writefile`을 한 셀에 같이 넣어서 매직 인식 실패|`%%writefile`이 셀 중간에 위치하게 됨|pip install 셀과 writefile 셀을 항상 분리|

**규칙 정리**: `%%writefile`이 들어간 셀에는 그 명령어 하나만 두고, pip install이나 다른 `!`/`%%` 명령어와 절대 섞지 않는다.

---

### 정리

- YouTube MCP 서버는 실무형 도구 4종(video_id 추출, 자막 언어 목록, 타임스탬프 포함 자막, 요약용 순수 텍스트 자막)을 갖춘 완성도 있는 구성이다.
- `MultiServerMCPClient.get_tools()`로 가져온 MCP Tool을 LangChain `create_agent`에 그대로 넘기면, Agent가 자율적으로 어떤 Tool을 언제 호출할지 판단해서 실행한다 — MCP + Agent 결합의 핵심 패턴.
- 오늘 반복해서 겪은 `%%writefile` 관련 노트북 실행 버그는 코드 로직 문제가 아니라 **Jupyter 셀 구조 문제**였다는 점을 기록해둘 것 — 다음에 비슷한 실습할 때 미리 셀을 분리해서 작성하면 같은 실수를 반복하지 않을 수 있다.

---
# 📄 buser_mcp_server.py / buser_mcp_client.py — MariaDB · FastMCP · CRUD Tool

### MariaDB 연동 MCP 서버 실습 (buser 테이블)

지금까지는 YouTube 자막처럼 외부 API를 감싸는 MCP Tool을 만들었다면, 이번엔 **실제 관계형 DB(MariaDB)를 조회하는 MCP 서버**를 만드는 실습이다. Colab이 아닌 로컬 Windows 환경에서 직접 서버를 띄우고 클라이언트로 검증까지 완료했다.

---

### 개념 정리

**전체 구조**

```
buser_mcp_client.py
   ↓ HTTP (streamable_http)
buser_mcp_server.py (FastMCP, port 8000)
   ↓ mariadb 커넥터
MariaDB 'test' DB의 buser 테이블
```

**이번 실습에서 새로 나온 개념**

|개념|설명|
|---|---|
|`mask_error_details=True`|MCP 도구 실행 중 발생한 내부 오류(DB 에러 메시지 등)의 상세 내용을 Client나 LLM에 그대로 노출하지 않도록 가리는 서버 옵션. 보안상 민감한 DB 에러 스택을 감춘다.|
|`annotations`|Tool의 동작 특성을 명시하는 메타데이터. `readOnlyHint`(읽기 전용), `idempotentHint`(같은 인자로 반복 호출해도 부작용 없음), `openWorldHint`(불특정 외부 시스템과 상호작용하지 않음 = False)|
|`ToolError`|MCP 표준 예외. 이 예외를 발생시키면 Client 쪽에 "Tool 실행 오류"로 명확히 전달된다.|
|`?` 플레이스홀더|mariadb 커넥터의 파라미터 바인딩 방식 (SQL Injection 방지)|

---

### 코드 + 주석

## 🖥 서버 코드 — `buser_mcp_server.py`

**1. DB 연결 공통 함수**

```python
# !pip install -U fastmcp mariadb python-dotenv

import os
import mariadb
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

load_dotenv()  # .env에서 DB_HOST, DB_USER, DB_PASSWORD 등을 읽어옴

mcp = FastMCP(
    name="MariaDB Buser Server",
    mask_error_details=True,  # 내부 DB 에러 상세를 Client에 노출하지 않음
)

def get_connection():
    """
    MariaDB 연결 객체를 생성한다.
    MCP Tool이 아니라, 각 Tool 내부에서 공통으로 쓰는 DB 연결 헬퍼 함수다.
    """
    return mariadb.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME", "test"),
    )
```

**2. 전체 부서 조회 Tool** (`get_all_busers`)

```python
@mcp.tool(
    description="MariaDB의 buser 테이블에서 부서 목록을 조회한다.",
    annotations={
        "readOnlyHint": True,     # 읽기 전용, 상태 변경 없음
        "idempotentHint": True,   # 반복 호출해도 결과가 동일
        "openWorldHint": False,   # 내부 DB 범위 안에서만 동작
    },
)
def get_all_busers(limit: int = 100) -> list[dict]:
    """
    buser 테이블의 전체 부서 목록을 조회한다.
    Args: limit: 최대 조회 건수
    Returns: 부서 정보 목록
    """
    limit = max(1, min(limit, 500))  # 1~500건으로 범위 제한 (과도한 조회 방지)

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)  # 결과를 dict 형태로 받기

        sql = "SELECT * FROM buser ORDER BY buserno LIMIT ?"
        cursor.execute(sql, (limit,))
        return cursor.fetchall()

    except mariadb.Error as err:
        print("[get_all_busers 오류]", err)
        raise ToolError("부서 정보를 조회하지 못했습니다.")
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()
```

**3. 부서번호로 단건 조회** (`get_buser_by_num`)

```python
@mcp.tool(
    description="부서번호를 이용하여 buser 테이블에서 특정 부서를 조회한다.",
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
)
def get_buser_by_num(buser_num: int) -> dict:
    """
    부서번호로 특정 부서를 조회한다.
    Args: buser_num: 조회할 부서번호
    Returns: 조회된 부서 정보
    """
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        sql = "SELECT * FROM buser WHERE buserno = ?"
        cursor.execute(sql, (buser_num,))
        row = cursor.fetchone()

        if row is None:
            return {"found": False, "message": f"{buser_num}번 부서를 찾을 수 없습니다."}
        return {"found": True, "buser": row}

    except mariadb.Error as err:
        print("[get_buser_by_num 오류]", err)
        raise ToolError("부서 정보를 조회하지 못했습니다.")
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()
```

- `found: True/False` 형태로 결과를 감싸서 반환하는 패턴이 눈에 띈다. 단순히 `None`을 반환하는 대신, LLM이 "부서를 못 찾았다"는 상황을 명확히 이해할 수 있도록 구조화했다.

**4. 부서명 검색** (`search_busers`, LIKE 검색)

```python
@mcp.tool(
    description="부서명에 포함된 검색어로 buser 테이블을 검색한다.",
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
)
def search_busers(keyword: str) -> list[dict]:
    """
    부서명에 검색어가 포함된 부서를 조회한다.
    Args: keyword: 검색할 부서명
    Returns: 검색된 부서 목록
    """
    keyword = keyword.strip()
    if not keyword:
        raise ToolError("검색어를 입력해야 합니다.")

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        sql = "SELECT * FROM buser WHERE busername LIKE ? ORDER BY buserno"
        cursor.execute(sql, (f"%{keyword}%",))
        return cursor.fetchall()

    except mariadb.Error as err:
        print("[search_busers 오류]", err)
        raise ToolError("부서 검색을 수행하지 못했습니다.")
    finally:
        if cursor is not None: cursor.close()
        if connection is not None: connection.close()


if __name__ == "__main__":
    # Streamable HTTP 방식으로 MCP 서버 실행
    mcp.run(transport="http", host="127.0.0.1", port=8000)
```

---

## 💻 클라이언트 코드 — `buser_mcp_client.py`

```python
import asyncio
from fastmcp import Client
from fastmcp.exceptions import ToolError

client = Client("http://127.0.0.1:8000/mcp")  # 별도로 실행 중인 MCP 서버 주소

async def main():
    try:
        async with client:
            tools = await client.list_tools()
            print("=== 등록된 MCP Tool ===")
            for tool in tools:
                print("-", tool.name)
                print(" 입력 구조:", tool.inputSchema)

            # 1. 전체 부서 조회
            all_result = await client.call_tool("get_all_busers", {"limit": 100})
            print("\n전체 부서 ===")
            print(all_result.data)

            # 2. 부서번호로 조회 (서버 함수 파라미터명 buser_num과 동일하게 전달)
            one_result = await client.call_tool("get_buser_by_num", {"buser_num": 10})
            print("\n10번 부서 ===")
            print(one_result.data)

            # 3. 부서명으로 검색
            search_result = await client.call_tool("search_busers", {"keyword": "총무부"})
            print("\n=== 부서명 검색 ===")
            print(search_result.data)

    except ToolError as err:
        print("\nMCP Tool 실행 오류:", err)
    except Exception as err:
        print("\nMCP Client 오류:", err)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 디버깅 기록 — DB 계정 인증 오류

**증상**

```
[get_all_busers 오류] Access denied for user 'acorn'@'localhost' (using password: NO)
```

클라이언트 쪽엔 `mask_error_details=True` 설정 때문에 상세 원인이 안 보이고 `"부서 정보를 조회하지 못했습니다"`만 표시됨. 실제 원인은 **서버 콘솔 로그**에서 확인.

**원인 추적**

1. `.env`의 `DB_PASSWORD`가 비어있어 "using password: NO"로 접속 시도
2. `.env`에 비밀번호(`123`)를 채워 넣었지만 여전히 실패
3. MariaDB에서 계정 자체를 확인:
    
    ```sql
    SELECT user, host FROM mysql.user;
    ```
    
    결과, `'acorn'@'localhost'` 계정이 애초에 **존재하지 않았음** (root 계정만 등록되어 있었음)

**해결**: `.env`의 `DB_USER`를 `root`로, `DB_PASSWORD`를 실제 root 비밀번호로 변경 → 서버 재시작 → 정상 동작 확인.

**최종 `.env` 설정**

```dotenv
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=123
DB_NAME=test
```

> 💡 실무에서는 애플리케이션이 root로 DB에 붙는 게 권장되지 않는다. 실습 이후 여유 있을 때 `CREATE USER` + `GRANT`로 전용 계정을 만들어 권한을 최소화하는 연습도 해볼 것.

---

### 출력 결과 (정상 동작 확인)

**서버 콘솔**

```
FastMCP 3.4.5
Server: MariaDB Buser Server, 3.4.5
Starting MCP server 'MariaDB Buser Server' with transport 'http' on http://127.0.0.1:8000/mcp
Uvicorn running on http://127.0.0.1:8000
```

**MariaDB 원본 데이터**

```
+---------+-----------+----------+--------------+
| buserno | busername | buserloc | busertel     |
+---------+-----------+----------+--------------+
|      10 | 총무부    | 서울     | 02-100-1111  |
|      20 | 영업부    | 서울     | 02-100-2222  |
|      30 | 전산부    | 서울     | 02-100-3333  |
|      40 | 관리부    | 인천     | 032-200-4444 |
+---------+-----------+----------+--------------+
```

**클라이언트 실행 결과**

```
=== 등록된 MCP Tool ===
- get_all_busers
  입력 구조: {'properties': {'limit': {'default': 100, 'type': 'integer'}}, ...}
- get_buser_by_num
  입력 구조: {'properties': {'buser_num': {'type': 'integer'}}, 'required': ['buser_num'], ...}
- search_busers
  입력 구조: {'properties': {'keyword': {'type': 'string'}}, 'required': ['keyword'], ...}

전체 부서 ===
[{'buserno': 10, 'busername': '총무부', ...}, {'buserno': 20, 'busername': '영업부', ...},
 {'buserno': 30, 'busername': '전산부', ...}, {'buserno': 40, 'busername': '관리부', ...}]

10번 부서 ===
{'found': True, 'buser': {'buserno': 10, 'busername': '총무부', 'buserloc': '서울', 'busertel': '02-100-1111'}}

=== 부서명 검색 ===
[{'buserno': 10, 'busername': '총무부', 'buserloc': '서울', 'busertel': '02-100-1111'}]
```

→ MariaDB 원본 데이터와 MCP를 통해 조회한 결과가 정확히 일치함을 확인.

---

### 정리

- YouTube API처럼 외부 서비스뿐 아니라 **사내 DB 같은 내부 시스템도 MCP Tool로 감싸서 노출**할 수 있다는 걸 실습으로 확인했다. `search_documents`(RAG)와 구조적으로 동일한 패턴 — "기존 기능을 함수로 감싸고 `@mcp.tool` 데코레이터만 붙이면 LLM이 호출 가능한 도구가 된다."
- `mask_error_details=True`처럼 **운영 관점의 보안 설정**이 실습에 처음 등장했다. 실제 서비스에서는 내부 에러 스택이 클라이언트/LLM에 그대로 노출되지 않도록 막는 게 기본이라는 점을 기억해둘 것.
- 오늘 겪은 DB 인증 오류는 코드 문제가 아니라 **환경 설정(계정 존재 여부) 문제**였음 — MCP 서버 자체는 처음부터 정상 코드였다. 에러 원인을 서버 콘솔에서 확인하는 습관이 중요했던 케이스.