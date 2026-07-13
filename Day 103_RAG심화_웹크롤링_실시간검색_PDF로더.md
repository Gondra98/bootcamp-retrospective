# Day 103_RAG심화_웹크롤링_실시간검색_PDF로더

## 📅 2026-07-09

---
# 📄 rag6web.ipynb — 웹크롤링 · ChromaDB · RAG파이프라인

## 🧩 개요

웹 문서(위키백과)를 크롤링해서 **RAG(Retrieval-Augmented Generation)** 파이프라인을 직접 구현한 실습. LLM이 학습하지 않은/모르는 정보라도, 검색된 문서를 근거로 답변하게 만드는 구조.

**전체 흐름**

```
웹페이지 → 텍스트 추출(BeautifulSoup) → 임베딩(SentenceTransformer)
        → ChromaDB 저장 → 질문 임베딩 → 유사 문서 검색(코사인 유사도)
        → 검색 결과를 프롬프트에 삽입 → LLM 답변 생성
```

---

## 🧠 핵심 개념

### RAG (Retrieval-Augmented Generation)

LLM에게 질문만 던지면 모델이 학습한(파라미터에 저장된) 지식으로만 답변한다 — 최신 정보나 특정 도메인 문서는 모를 수 있음. RAG는 질문과 관련된 문서를 **먼저 검색(Retrieval)**해서 프롬프트에 근거로 넣어준 뒤, 그걸 참고해서 **답변을 생성(Generation)**하게 만드는 구조. "검색 + 생성"을 합친 방식.

### 임베딩(Embedding)과 벡터 유사도

텍스트를 숫자 벡터로 변환하는 과정. 의미가 비슷한 문장은 벡터 공간에서 가까운 위치에 놓임. 이 노트북은 `SentenceTransformer("all-MiniLM-L6-v2")`를 사용 — 문장을 384차원 벡터로 변환하는 경량 임베딩 모델.

**코사인 유사도(Cosine Similarity)**: 두 벡터 사이의 각도로 유사도를 측정. 각도가 작을수록(방향이 비슷할수록) 유사한 문서. ChromaDB는 이를 "코사인 거리(distance)"로 반환하는데, **거리가 작을수록 더 유사한 문서**라는 점이 핵심.

### ChromaDB (벡터스토어)

임베딩된 벡터를 저장하고, 질문 벡터와 가장 가까운 벡터들을 빠르게 찾아주는 벡터 데이터베이스.

- `PersistentClient`: 디스크에 영구 저장 (재실행해도 데이터 유지)
- `collection`: 관련 문서들을 묶는 단위 (RDB의 테이블과 비슷한 개념)
- `metadata={"hnsw:space": "cosine"}`: 유사도 계산 방식을 코사인으로 지정 (HNSW는 근사 최근접 이웃 탐색 알고리즘)

---

## 📝 코드 + 주석

### 1) 환경 설정 및 초기화

```python
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

load_dotenv()  # .env 파일에서 API 키 등 환경변수 로드

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # OpenAI 클라이언트 생성
model = "gpt-4o-mini"                                   # 답변 생성에 쓸 LLM 모델
embedder = SentenceTransformer("all-MiniLM-L6-v2")      # 텍스트 -> 384차원 벡터 변환 모델
```

### 2) 웹 크롤링 함수

```python
def extract_from_urlFunc(url):
  headers = {'User-Agent': 'Mozilla/5.0'}  # 일부 사이트는 봇 요청을 차단하므로 브라우저인 척 위장
  resp = requests.get(url, headers=headers)

  if resp.status_code != 200:
    print('요청 실패, html preview', resp.text[:200])
    return []

  soup = BeautifulSoup(resp.text, 'html.parser')  # HTML 파싱
  paragraphs = soup.find_all("p")                  # <p> 태그(본문 단락)만 추출

  texts = [
      p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
      # get_text(strip=True): 태그 제거 + 앞뒤 공백 제거
      # if 조건: 빈 문자열(광고/구분용 빈 <p>)은 제외
  ]
  print('found p count : ', len(texts))
  return texts
```

> 💡 위키백과 같은 페이지는 본문이 대부분 `<p>` 태그 안에 있어서 이 방식이 잘 통함. 다른 사이트는 구조가 다를 수 있어 범용성은 떨어짐.

### 3) ChromaDB 컬렉션 생성 및 문서 저장

```python
chroma_client = PersistentClient(path="./chroma_db")  # 로컬 디스크에 벡터DB 저장

try:
  chroma_client.delete_collection(name="webdata")  # 재실행 시 기존 컬렉션 삭제(중복 방지)
except:
  pass  # 컬렉션이 없으면 삭제 시도 시 에러 -> 무시하고 진행

collection = chroma_client.create_collection(
    name="webdata",
    metadata={"hnsw:space": "cosine"}  # 유사도 측정 방식을 코사인으로 지정
)

url = "https://ko.wikipedia.org/wiki/김치찌개"
web_docs = extract_from_urlFunc(url)

if not web_docs:
  print('추출된 문서가 없어요')
  raise Exception('문서 없음')  # 크롤링 실패 시 이후 단계 진행 방지

web_embeddings = embedder.encode(web_docs)  # 문단 리스트 -> (문단 수, 384) 벡터 배열
print('web_embeddings : ', web_embeddings.shape)

# 문단 하나씩 ChromaDB에 저장
for i, (doc, emb) in enumerate(zip(web_docs, web_embeddings), start=0):
  collection.add(
      documents=[doc],              # 원본 텍스트
      embeddings=[emb],             # 해당 텍스트의 벡터
      metadatas=[{"source": url}],  # 출처 URL을 메타데이터로 기록
      ids=[f'web_{i}']              # 고유 ID (web_0, web_1, ...)
  )

print('저장된 문서 수 : ', collection.count())

result = collection.get(ids=['web_0', 'web_1'])  # 저장한 id 형식과 동일하게 조회해야 함
print('샘플 문서 : ', result['documents'])
```

**▶ 실행 결과**

```
found p count :  9
web_embeddings :  (9, 384)
저장된 문서 수 :  9
샘플 문서 :  ['국물류', '반찬']
```

> 위키 문서에서 `<p>` 태그 9개(문단)를 추출해 9개의 384차원 벡터로 변환, ChromaDB에 9건 저장. `web_0`, `web_1`로 조회한 샘플이 각각 '국물류', '반찬'인 걸 보면 위키 페이지 상단의 카테고리 태그성 텍스트가 첫 문단으로 잡힌 것으로 보임(정제 여지 있는 부분).

> ⚠️ **주의**: `ids`에 저장한 형식(`web_0`)과 조회할 때 넘기는 형식이 반드시 일치해야 함. 형식이 다르면 빈 결과가 반환됨.

### 4) 질문 벡터화 및 유사 문서 검색

```python
query = "김치찌개의 역사와 조리법 알려 줘"

query_vec = embedder.encode(
    [query], normalize_embeddings=True  # 벡터 정규화(길이를 1로) -> 코사인 유사도 계산 정확도↑
)[0]

results = collection.query(
    query_embeddings=[query_vec.tolist()],  # ChromaDB는 리스트 형태를 기대 -> numpy는 .tolist() 필요
    n_results=3,                             # 가장 유사한 상위 3개 문서만 반환
    include=['documents', 'distances']       # 문서 원문과 거리값을 함께 받기 (복수형 표기 주의)
)

retrieved_docs = results['documents'][0]  # query_embeddings가 리스트라 결과도 리스트의 리스트로 반환됨
retrieved_dist = results['distances'][0]

for i, (doc, dist) in enumerate(zip(retrieved_docs, retrieved_dist), 1):
  print(f'문서{i} : {doc}')
  print(f'distances : {dist:.4f}\n')  # 거리가 작을수록 질문과 유사한 문서

# 검색된 문서들을 하나의 텍스트로 합침 -> LLM 프롬프트에 넣을 컨텍스트로 사용
retrieved_text = '\n\n'.join(retrieved_docs)
print(retrieved_text)
```

**▶ 실행 결과**

```
문서1 : 김치찌개는 대표적인한국 요리가운데 하나로,김치를 넣고 얼큰하게 끓인찌개이다.된장찌개·순두부찌개와 함께 가장 널리 알려진 요리이다.
distances : 0.2854

문서2 : 영양 면에서는 김치 외에 다양한 재료를 더해 영양소를 보충할 수 있다. 다만 김치는 높은 온도에서 끓이면유산균이 감소한다. 또한 김치와 부재료로 인해나트륨함량이 높은 편이며, 볶는 과정에서 사용하는 기름 때문에지방함량도 비교적 높다. 이에 따라칼로리도 높은 편에 속한다.
distances : 0.4231

문서3 : 김치찌개에는 어느 정도발효되어 신맛이 나는 김치를 사용하는데, 담근 지 얼마 되지 않은 김치를 쓰면 특유의 깊은 맛이 덜하다. 신 김치를 그대로 사용하기도 하지만, 기름에 볶은 뒤 끓이면 맛이 더 깊어진다. 볶을 때는참기름이나들기름을 주로 사용하며, 돼지고기를 넣을 경우 라드(돼지 비계)로 볶기도 한다.
distances : 0.4584
```

> 질문("역사와 조리법")과 가장 가까운 문서가 distance 0.28로 뽑혔고, 이어서 영양 정보(0.42), 조리 팁(0.46) 순으로 검색됨. 질문에 "역사"가 포함돼 있지만 실제로 역사를 직접 설명하는 문단(김치의 유래 부분)은 상위 3개에 안 들어간 게 눈에 띔 — 임베딩 유사도가 항상 사람이 기대하는 "정답 문단"과 일치하진 않는다는 걸 보여주는 실제 사례.

> 💡 `n_results=3`으로 상위 3개만 뽑는 이유: 관련 없는 문서까지 다 넣으면 LLM 프롬프트가 길어지고 답변 품질도 떨어짐(노이즈 증가). 실제 결과에서도 거리 0.28~0.46 수준으로 상위 3개가 질문과 관련성이 확인됨.

### 5) 검색 결과 기반 LLM 답변 생성

```python
prompt = f"""
너는 한국요리 특히 찌개 전문가야.
다음 문서를 참고하여 '{query}'에 대해 답변해줘

[참고문서]
{retrieved_text}

요구 사항:
김치찌개의 조리법, 재료의 역할까지 설명해.
최소 20문장 이내로 작성해.
마크다운 없이 작성해.
각 문장을 줄 바꿈으로 분리해.
"""

response = client.responses.create(
    model=model,
    input=prompt
)
print('답변 : ', response.output_text)
```

**▶ 실행 결과**

```
답변 :  김치찌개는 한국의 전통적인 대표 요리 중 하나로, 김치가 주재료로 사용됩니다.
이 요리는 얼큰하게 끓여지는 찌개로, 주로 된장찌개나 순두부찌개와 함께 많이 알려져 있습니다.
김치찌개의 기본 재료는 발효된 김치입니다.
김치는 신맛이 나는 것이 좋으며, 담근 지 얼마 되지 않은 김치를 사용할 경우 깊은 맛이 부족합니다.
이 외에도 돼지고기, 두부, 양파, 파, 마늘, 고추 등이 자주 사용됩니다.
돼지고기는 찌개의 풍미를 더해주며, 기름에 볶아 깊은 맛을 내는 데 기여합니다.
김치를 볶을 때는 참기름이나 들기름을 주로 사용하여 향을 살립니다.
볶은 김치를 육수와 함께 끓여내면 더 진한 맛을 느낄 수 있습니다.
또한 영양 면에서도 다양한 재료를 추가할 수 있어 영양소 보충이 가능합니다.
하지만 높은 온도에서 김치를 끓이면 유산균이 감소하는 단점이 있습니다.
김치찌개는 나트륨 함량이 높은 편이며, 볶는 과정에서 사용하는 기름 때문에 지방 함량도 증가합니다.
칼로리 역시 높은 편이라 주의가 필요합니다.
된장찌개와 비슷한 조리법을 따르지만, 김치의 맛과 향이 중심이 됩니다.
조리법은 먼저 김치를 기름에 볶아서 국물의 맛을 강화하는 방식입니다.
볶은 후에 물이나 육수를 추가하고 끓이면서 다른 부재료를 넣습니다.
양파와 마늘은 향을 더해주는 역할을 하며, 고추는 얼큰함을 더해줍니다.
두부는 부드러운 식감을 추가하고, 익히면 더욱 맛있습니다.
모든 재료가 잘 어우러져서 찌개가 완성되면, 마지막에 파를 넣어 신선함을 더합니다.
김치찌개는 강한 맛이 일품이며, 밥과 함께하면 더욱 맛있습니다.
이렇게 조리된 김치찌개는 한국의 식탁에서 빠질 수 없는 중요한 요리로 자리 잡고 있습니다.
```

> 검색된 3개 문단(정의·영양·발효/조리법)의 내용이 답변 전반에 고르게 녹아 있음 — 특히 "유산균 감소", "나트륨/지방 함량" 같은 문서2의 영양 정보와 "참기름/들기름 볶음", "발효 정도" 같은 문서3의 조리 팁이 그대로 반영된 게 확인됨. 다만 요구사항이었던 "역사" 설명은 검색된 문서에 해당 내용이 없어서 답변에도 빠짐 — **검색되지 않은 정보는 LLM이 만들어내지 않는다**는 RAG의 특성이 잘 드러난 부분(할루시네이션 방지 프롬프트가 의도대로 작동).

> 💡 프롬프트에 `[참고문서]` 섹션으로 검색된 텍스트를 직접 삽입 — 이게 RAG의 핵심. LLM은 이 문서를 근거로만 답변하도록 지시받음("문서를 참고하여"). 이렇게 하면 LLM이 모르는 최신/특정 정보도 검색된 문서 기반으로 정확하게 답변 가능.

---

## 🔍 되짚어볼 포인트

|항목|설명|
|---|---|
|`web_embeddings.shape`|`(9, 384)` → 문단 9개, 각 문단이 384차원 벡터로 변환됨|
|`include=['documents', 'distances']`|ChromaDB는 옵션 이름이 전부 **복수형** (documents, distances, embeddings, metadatas)|
|distance 값|코사인 거리 기준, **작을수록 유사도 높음** (0에 가까울수록 거의 동일한 의미)|
|`raise Exception('문서 없음')`|크롤링 실패 시 이후 임베딩/저장 단계로 넘어가지 않도록 조기 종료|
|저장 방식|for loop로 문서 하나씩 `add()` → 문서 수 많아지면 배치 저장이 더 효율적 (개선 여지)|

## 🔗 확장 아이디어

- 이 노트북은 **고정 URL 하나**만 크롤링하는 정적 RAG. 오늘 같이 다룬 Tavily/DuckDuckGo 검색 API를 붙이면 **질문마다 실시간 웹 검색 → RAG** 구조로 확장 가능.
- 여러 URL을 순회하며 저장하면 멀티소스 RAG로 발전시킬 수 있음.

---
# 📄 rag7web.ipynb — DuckDuckGo · 실시간웹검색 · LangChain체인

## 🧩 개요

이전 노트(`rag6web.ipynb`)는 **고정된 위키 문서 1개**를 미리 크롤링해서 ChromaDB에 저장해두고 검색하는 **정적 RAG**였다면, 이 노트북은 질문이 들어올 때마다 **DuckDuckGo로 실시간 웹 검색**을 수행해서 최신 정보를 근거로 답변하는 구조.

**전체 흐름**

```
질문 입력 → DuckDuckGo 실시간 검색(search_results)
         → 프롬프트에 검색결과 삽입 → LLM(ChatOpenAI) 답변 생성
```

정적 RAG(ChromaDB)와 다르게 **벡터 임베딩/저장 과정이 없음** — 검색 결과를 곧바로 프롬프트에 넣는 단순한 구조라 "RAG"라기보단 **Retrieval + Generation**에 가까움 (벡터DB 없이 검색 API로 대체).

---

## 🧠 핵심 개념

### DuckDuckGo Search (No-AI 검색엔진)

개인정보 추적 없이 검색 결과를 제공하는 엔진. `DuckDuckGoSearchResults()`는 LangChain이 제공하는 래퍼로, 검색 API 없이(무료) 텍스트 검색 결과를 바로 가져올 수 있음. 단, 안정성이 Tavily 같은 전용 API보다는 떨어질 수 있음(요청 제한, 파싱 실패 등).

### LangChain의 프롬프트 체인 (`|` 파이프 연산자)

```python
qa_chain = self.qa_prompt | self.llm
```

`ChatPromptTemplate`과 `ChatOpenAI` 모델을 `|`로 연결하면, 프롬프트에 값을 채운 뒤 자동으로 LLM에 전달하는 **체인(Chain)**이 만들어짐. `invoke()`에 딕셔너리로 변수만 넘기면 프롬프트 완성 → LLM 호출까지 한 번에 처리됨.

### `ChatOpenAI` vs `OpenAI` (langchain_openai)

- `OpenAI`: 구형 텍스트 completion 모델용 (chat 모델과 인터페이스가 다름)
- `ChatOpenAI`: gpt-4o-mini 같은 채팅형 모델용, LangChain 체인(`|`)과 호환됨

이전 노트에서 `OpenAI`를 잘못 써서 체인 연결이 깨졌던 것과 달리, 이번 버전은 `ChatOpenAI`로 정확히 수정되어 있음.

---

## 📝 코드 + 주석

### 1) 패키지 설치

```python
# DuckDuckGo. 트래픽 급증 속 'No-AI' 검색 엔진에 더 쉽게 접근할 수 있게 함
# 실시간 웹 검색 구현
!pip install -U langchain-openai langchain-community duckduckgo-search ddgs
```

> 💡 `duckduckgo-search`와 `ddgs`는 실제 검색을 수행하는 하위 라이브러리. `langchain-community`의 `DuckDuckGoSearchResults`가 내부적으로 이걸 사용함.

### 2) 실시간 웹 RAG 클래스

```python
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
import os
import time
from dotenv import load_dotenv

load_dotenv()

class MyRealtimeWebRag:
  def __init__(self):
    self.search = DuckDuckGoSearchResults()  # 검색 도구 초기화 (API 키 불필요)
    self.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,                      # 0.0 = 창의성 최소화, 검색 결과에 최대한 충실하게 답변
        api_key=os.getenv("OPENAI_API_KEY")
    )

    message = """
      웹에서 검색한 최신정보를 바탕으로 답변하세요.

      검색 결과 :
      {search_results}

      질문:
      {question}

      중요:
      검색 결과에 있는 답변만 하세요. 추측하지 마세요.
      모르면 모른다고 하세요.

      답변:
    """
    # "검색 결과에 있는 답변만" -> 할루시네이션(모델이 없는 정보를 지어내는 것) 방지 지시

    self.qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", message)   # 'role':'user'와 동일한 역할 (사용자 메시지로 전달)
        ]
    )

  def answerFunc(self, question):
    print(f'검색 중 : {question}')

    search_results = self.search.run(question)  # DuckDuckGo 검색 실행, 결과를 텍스트로 반환
    time.sleep(1)                                 # 연속 검색 시 요청 제한(rate limit) 방지용 딜레이

    qa_chain = self.qa_prompt | self.llm  # 프롬프트 + LLM을 체인으로 연결

    response = qa_chain.invoke(
        {
            "search_results": search_results,  # {search_results} 자리에 채워짐
            "question": question                # {question} 자리에 채워짐
        }
    )

    return response  # AIMessage 객체 (실제 텍스트는 .content 로 접근)
```

### 3) 실행부

```python
if __name__ == '__main__':
  web_rag = MyRealtimeWebRag()

  question = [
      "최신 AI 기술은 뭐니?",
      "한국에서 가장 인기 있는 빵은?",
      "한국의 여름철 장마에 대해 알려해"
  ]

  for q in question:
    print(f'질문:{q}')
    answer = web_rag.answerFunc(q)
    print(f'답변:{answer.content}')  # response는 AIMessage -> .content로 실제 답변 텍스트 추출
```

**▶ 실행 결과**

```
질문:최신 AI 기술은 뭐니?
검색 중 : 최신 AI 기술은 뭐니?
답변:최신 AI 기술은 생성형 AI, AI 반도체 등으로 발전하고 있으며, 특정 분야에 특화된 AI 모델들이 대거 등장하고 있습니다.
이러한 AI 모델들은 문서 요약, 이미지 생성, 코딩 등 다양한 용도로 활용되고 있습니다.
2025년에는 빅데이터 및 AI 시장이 약 4,563억 달러 규모로 성장할 전망입니다.
각 산업별 AI 도입 현황과 활용 분야도 주목받고 있습니다.

질문:한국에서 가장 인기 있는 빵은?
검색 중 : 한국에서 가장 인기 있는 빵은?
답변:한국에서 가장 인기 있는 빵은 소금빵입니다. 소금빵은 '단짠'의 정석으로 고소한 버터 풍미가 특징이며, 최근에는 크루아상과 같은 다른 빵들도 인기를 끌고 있습니다.

질문:한국의 여름철 장마에 대해 알려해
검색 중 : 한국의 여름철 장마에 대해 알려해
답변:한국의 여름철 장마는 6월 하순에서 7월 하순에 걸쳐 동아시아에서 습한 공기가 전선을 형성하며 많은 비를 내리는 현상을 가리킵니다. 이 시기를 장마철이라고 하며, 구우(久雨)라고도 불립니다. 장마의 사전적 의미는 여름철에 여러 날 계속해서 비가 내리는 현상이나 날씨를 의미합니다. 비가 내리지 않으면 장마라고 할 수 없으며, 자연스럽게 마른장마라는 말은 모순입니다.

기상청에 따르면, 2025년 7월 3일에는 제주 및 남부 지역에 장마가 종료됐다고 선언되었습니다. 이는 북태평양 고기압이 이례적으로 빠르게 세력을 확장한 것의 영향으로 보입니다.
```

> 세 번째 답변에서 **"2025년 7월 3일 제주·남부지역 장마 종료"**라는 매우 구체적인 최신 기상청 발표까지 반영된 게 눈에 띔 — 이는 LLM이 원래 알고 있던 지식이 아니라 DuckDuckGo가 실시간으로 가져온 검색 결과이기 때문에 가능한 답변. 정적 RAG(rag6web)와 달리 **질문 시점의 최신 정보**가 반영된다는 점이 실시간 웹 RAG의 핵심 가치를 보여줌.

---

## 🔍 이전 노트(rag6web) 대비 달라진 점

|항목|rag6web (정적 RAG)|rag7web (실시간 RAG)|
|---|---|---|
|문서 소스|위키 문서 1개, 사전 크롤링|DuckDuckGo 실시간 검색|
|저장소|ChromaDB (벡터DB)|없음 (검색 결과 즉시 사용)|
|유사도 검색|임베딩 + 코사인 거리로 상위 3개 선별|검색 엔진이 반환한 결과 그대로 사용|
|최신성|크롤링 시점 정보에 고정|매 질문마다 최신 웹 정보 반영|
|LLM 클래스|`client.responses.create()` (OpenAI SDK 직접 호출)|`ChatOpenAI` + LangChain 체인(`|

## 🔗 확장 포인트

- 지금은 검색 결과를 그대로 프롬프트에 넣지만, 이전에 논의했던 `MyOptimizedWebRag`처럼 **검색 → 요약 → 답변**의 2단계 체인으로 확장하면 프롬프트 길이도 줄고 답변 품질도 개선됨.
- DuckDuckGo는 무료지만 안정성이 낮을 수 있어, 프로덕션 레벨에서는 Tavily 같은 전용 API로 교체하는 게 안전.

---
# 📄 rag8web.ipynb — Tavily · 요약체인 · 2단계RAG

## 🧩 개요

`rag7web.ipynb`(DuckDuckGo)에서 한 단계 더 발전한 버전. 검색 엔진을 **Tavily**로 교체하고, 검색 결과를 곧바로 프롬프트에 넣는 대신 **"검색 → 요약 → 최종 답변"의 2단계 체인**으로 구조를 개선함.

**전체 흐름**

```
질문 입력 → Tavily 실시간 검색(raw_results)
         → 1단계: 요약 프롬프트로 핵심만 정리(summary_prompt)
         → 2단계: 요약본 + 질문으로 최종 답변 생성(answer_prompt)
```

**왜 2단계로 나눴나?** Tavily는 여러 검색 결과(최대 5개)를 한 번에 반환하는데, 원문 그대로 LLM에 넣으면 광고성 내용/중복 정보가 섞여서 프롬프트가 길고 지저분해짐. **요약 단계를 한 번 거치면** 노이즈가 걸러지고, 최종 답변 프롬프트가 짧고 명확해져서 답변 품질이 안정적으로 나옴.

---

## 🧠 핵심 개념

### Tavily vs DuckDuckGo

| |DuckDuckGo|Tavily|
|---|---|---|
|API 키|불필요|필요 (무료 티어 제공)|
|결과 형태|검색 결과 텍스트(비교적 raw)|LLM 친화적으로 정리된 결과|
|안정성|요청 제한/파싱 실패 가능성 있음|AI 활용 목적으로 설계되어 더 안정적|
|결과 개수 조절|제한적|`max_results` 파라미터로 직접 제어|

### 프롬프트 체인을 두 개로 분리하는 패턴

```python
self.summary_prompt = ChatPromptTemplate.from_template(...)  # 검색결과 -> 요약
self.answer_prompt  = ChatPromptTemplate.from_template(...)  # 요약+질문 -> 최종답변
```

하나의 거대한 프롬프트에 "검색결과 요약해서 답변까지 해줘"라고 몰아넣는 것보다, **역할별로 프롬프트를 쪼개서 체인을 연결**하면 각 단계가 무엇을 하는지 명확해지고 디버깅도 쉬워짐. LangChain에서 자주 쓰는 패턴(파이프라인 분리).

---

## 📝 코드 + 주석

### 1) 패키지 설치 및 초기화

```python
# TavilySearch 사용
# VectorDB없이 웹 검색 결과를 문맥으로 넣어 LLM이 답하도록 하는 웹 검색기반 RAG구조에 적합
# 실시간 웹 검색 구현
!pip install -U langchain-openai langchain-community langchain-tavily
```

```python
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch   # Tavily 공식 LangChain 통합 패키지
from langchain_core.prompts import ChatPromptTemplate
import os
import time
from dotenv import load_dotenv

load_dotenv()
```

### 2) 클래스 구조: 두 프롬프트 + 두 메서드

```python
class MyOptimizedWebRag:
  def __init__(self):
    self.search = TavilySearch(
        max_results=5,                      # 검색 결과 상위 5개까지 가져옴
        api_key=os.getenv('TAVILY_API_KEY')
    )

    self.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,                    # 요약/답변 모두 사실 기반이어야 하므로 창의성 최소화
        api_key=os.getenv('OPENAI_API_KEY')
    )

    # [1단계용] 검색결과 -> 요약 프롬프트
    self.summary_prompt = ChatPromptTemplate.from_template(
        """
        당신은 검색결과를 정리 요약하는 전문가 입니다.

        다음은 웹 검색 결과 입니다.
        {search_results}

        요구사항:
        - 광고/홍보성 내용을 제거하고
        - 서로 겹치는 내용은 합쳐서
        - 핵심 정보만 5줄 내외로 간결하게 정리하세요

        검색 결과 요약:
        """
    )

    # [2단계용] 요약본 + 질문 -> 최종 답변 프롬프트
    self.answer_prompt = ChatPromptTemplate.from_template(
        """
        아래에 어떤 질문에 대한 웹 검색을 수행한 뒤 정리한 요약 내용입니다.

        [검색 요약]
        {mysummary}

        [질문]
        {myquestion}

        위 요약에 있는 정보만 사용해서 질문에 답변하세요.
        추측하거나 지어내지 말고, 모르면 모른다고 말하세요

        최종 답변:
        """
    )
```

### 3) 1단계: 검색 + 요약

```python
  def summarize_search(self, question):
    raw_results = self.search.invoke({"query": question})  # Tavily 검색 실행
    time.sleep(0.5)  # 연속 호출 시 요청 제한 방지

    chain = self.summary_prompt | self.llm  # 요약 프롬프트 + LLM 체인 연결

    summary_msg = chain.invoke({"search_results": raw_results})  # {search_results}에 원본 검색결과 삽입
    return summary_msg.content  # 요약된 텍스트만 반환
```

### 4) 2단계: 요약 기반 최종 답변

```python
  def answer_question(self, question):
    print(f'검색 및 요약 중 : {question}')

    summary = self.summarize_search(question)  # 1단계 실행 -> 요약 텍스트 확보

    chain = self.answer_prompt | self.llm      # 답변용 프롬프트 + LLM 체인 연결

    answer_msg = chain.invoke(
        {
            "mysummary": summary,     # {mysummary}에 요약본 삽입
            "myquestion": question    # {myquestion}에 원본 질문 삽입
        }
    )

    return answer_msg.content
```

> ⚠️ 지난번 초안에서는 여기서 `summary_prompt`와 `answer_prompt`가 서로 뒤바뀌어 있었는데(요약 단계에서 answer_prompt를 쓰거나 반대), 최종적으로 **각 단계에 맞는 프롬프트로 정확히 매칭**되면서 정상 동작함.

### 5) 실행부

```python
if __name__ == '__main__':
  web_rag = MyOptimizedWebRag()

  question = [
      "최신 AI 기술은 뭐니?",
      "한국에서 가장 인기 있는 짬뽕은?",
      "한국의 여름철 장마 기간 중 점심때 강남 근처에서 먹기 좋은 음식을 추천해"
  ]

  for q in question:
    print(f'질문:{q}')
    answer = web_rag.answer_question(q)
    print(f'답변:{answer}')
```

**▶ 실행 결과**

```
질문:최신 AI 기술은 뭐니?
검색 및 요약 중 : 최신 AI 기술은 뭐니?
답변:최신 AI 기술은 기계가 경험을 통해 학습하고 새로운 데이터를 처리하여 인간의 작업을 수행할 수 있도록 하는 혁신적인 기술입니다. 주요 발전 방향으로는 자율주행 기술, 자연어 처리(NLP), AI 기반 개인화 경험, 헬스케어 혁신, AI와 IoT의 융합이 있습니다. AI는 대량의 데이터를 분석하여 패턴을 인식하고, 반복적인 작업을 자동화하여 인간이 더 복잡한 문제에 집중할 수 있도록 돕습니다. 또한, 기계 학습과 딥 러닝 기술이 지속적으로 발전하고 있으며, 다양한 산업에서 활용되고 있습니다.

질문:한국에서 가장 인기 있는 짬뽕은?
검색 및 요약 중 : 한국에서 가장 인기 있는 짬뽕은?
답변:한국에서 가장 인기 있는 짬뽕은 지역별로 다양하며, 특히 강릉의 교동반점, 군산의 복성루, 공주의 동해원이 유명합니다. 강릉 교동반점은 매운 짬뽕으로 해산물이 풍부하게 들어가고, 군산 복성루는 고명 양이 많으며 해물과 돼지고기가 조화를 이루는 짬뽕으로 알려져 있습니다. 공주 동해원은 칼칼한 맛이 특징입니다. 이 외에도 인천 차이나타운과 같은 지역에서도 짬뽕 문화가 발달해 있습니다. 각 지역의 짬뽕은 고유의 맛과 특색을 지니고 있어, 짬뽕 투어를 즐기는 이들이 많습니다.

질문:한국의 여름철 장마 기간 중 점심때 강남 근처에서 먹기 좋은 음식을 추천해
검색 및 요약 중 : 한국의 여름철 장마 기간 중 점심때 강남 근처에서 먹기 좋은 음식을 추천해
답변:여름철 장마 기간 중 강남 근처에서 점심때 먹기 좋은 음식으로는 다음과 같은 것들이 있습니다:

1. **냉면**: 시원하고 가벼운 맛으로 장마철에 적합하며, 무 김치와 함께 먹으면 좋습니다.
2. **해물파전**: 비 오는 날에 잘 어울리는 음식으로, 김치와 오징어의 조화가 매력적입니다.
3. **삼계탕**: 더위에 지친 몸을 회복시켜주는 보양식으로 인기가 높습니다.
4. **콩국수**: 시원하고 담백한 맛으로 여름철에 즐기기 좋은 음식입니다.

이 외에도 장어와 물회도 여름철 인기 음식으로 추천됩니다.
```

> rag7web(DuckDuckGo)과 같은 질문("최신 AI 기술")을 던졌는데, rag8web 답변에는 시장 규모 수치 대신 **자율주행·NLP·헬스케어·IoT 등 기술 카테고리 중심**으로 정리됨 — 이는 검색 엔진 차이(Tavily vs DuckDuckGo)뿐 아니라, **요약 단계를 한 번 거치면서 정보가 재구성**된 영향도 있음. 세 번째 질문(장마철+강남+점심)처럼 조건이 여러 개 섞인 복합 질문에도 번호 매긴 리스트로 구체적 메뉴(냉면, 해물파전, 삼계탕, 콩국수)까지 답변한 걸 보면, 검색→요약→답변 체인이 복합 조건 질문에도 잘 대응하는 것으로 확인됨.

---

## 🔍 rag7web(DuckDuckGo) 대비 개선점

|항목|rag7web|rag8web|
|---|---|---|
|검색 엔진|DuckDuckGo (무료, API키 불필요)|Tavily (API키 필요, LLM 친화적 결과)|
|처리 단계|검색 → 바로 답변 (1단계)|검색 → 요약 → 답변 (2단계)|
|프롬프트 개수|1개|2개 (역할 분리)|
|노이즈 처리|검색 결과 그대로 사용|요약 단계에서 광고/중복 제거|
|결과 개수 조절|불가|`max_results=5`로 명시적 제어|

## 🔗 확장 포인트

- 지금은 매 질문마다 Tavily API를 새로 호출함 → 같은 질문이 반복되면 캐싱(예: 이전 rag6web의 ChromaDB처럼 검색 결과를 저장) 붙이면 API 호출 비용/속도 개선 가능.
- 요약 단계와 답변 단계 사이에 "요약이 질문에 답할 만큼 충분한지" 체크하는 단계를 추가하면 더 견고한 RAG 파이프라인으로 발전 가능.

---
# 🔍 웹검색 도구 — DuckDuckGo · Tavily

## DuckDuckGo

- **일반 사용자용 검색엔진** (구글/네이버와 같은 포지션)
- 최대 특징: **트래킹 없음** — 검색 기록, 개인 프로필 저장 안 함, 모두에게 동일한 결과 제공
- LangChain에서 `DuckDuckGoSearchResults()`로 API 키 없이 바로 사용 가능
- 사람이 보는 HTML 검색결과 기반이라 결과가 다소 raw하고, 요청 제한/파싱 실패 가능성 있음

## Tavily

- **LLM/AI 애플리케이션 전용으로 설계된 검색 API**
- 검색 결과를 사람이 읽는 페이지가 아니라 **LLM이 바로 소화하기 좋은 정리된 텍스트/JSON**으로 반환
- API 키 필요 (무료 티어 제공, 월 1,000회 수준)
- `max_results`로 결과 개수 직접 제어 가능, LangChain에 `langchain_tavily.TavilySearch`로 공식 통합됨
- RAG 파이프라인에서 실시간 웹 검색 소스로 특히 많이 쓰임

## 한눈에 비교

| |DuckDuckGo|Tavily|
|---|---|---|
|용도|일반 웹 검색|AI/LLM 특화 검색|
|API 키|불필요|필요|
|결과 형태|비교적 raw|LLM 친화적으로 정리됨|
|비용|무료|무료 티어 + 유료|
|RAG 적합도|보통 (간단한 실습용)|높음 (프로덕션급)|

> 💡 오늘 실습(`rag7web` vs `rag8web`)에서도 같은 질문("최신 AI 기술은 뭐니?")에 Tavily 쪽이 더 구조화된 정보를 가져온 걸 확인함.


---
# 📄 rag9pdf.ipynb — PDF문서로더 · LangChain · PyPDF·Unstructured·PyMuPDF

## 🧩 개요

LangChain의 `document_loaders` 모듈이 제공하는 **PDF 로더 4종**을 비교 실습. 전부 PDF를 읽어 `Document` 객체(랭체인 표준 문서 형식)로 변환하지만, 방식과 강점이 각각 다름.

|방법|로더|특징|
|---|---|---|
|1|`PyPDFLoader`|페이지 단위로 텍스트 추출, 가장 기본/가벼움|
|2|`UnstructuredPDFLoader`|텍스트를 의미 단위(elements)로 세분화, 레이아웃 정보 포함|
|3|`PyMuPDFLoader`|상세 메타데이터 추출에 강점|
|4|`OnlinePDFLoader`|로컬 파일이 아닌 웹 URL의 PDF를 바로 로드|

**공통 개념 — `Document` 객체** 랭체인에서 텍스트 데이터를 다루는 표준 단위. `page_content`(실제 텍스트)와 `metadata`(출처, 페이지 번호 등 부가정보) 두 속성을 가짐. RAG 파이프라인에서 임베딩/검색의 기본 단위로 쓰임.

---

## 📝 방법 1) `PyPDFLoader` — 페이지 단위 추출

```python
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pdf_filepath = 'sample.pdf'

loader = PyPDFLoader(pdf_filepath)
pages = loader.load()   # PDF -> Document 객체 리스트, 페이지 1개 = Document 1개

print(type(loader), type(pages))
print(len(pages))            # 총 페이지 수

print(pages[1].page_content)  # 2번째 페이지(인덱스 1) 텍스트
print(pages[1].metadata)      # 페이지 메타데이터
```

**▶ 실행 결과**

```
<class 'langchain_community.document_loaders.pdf.PyPDFLoader'> <class 'list'>
10

metadata: {'producer': 'ezPDF Builder 2006', 'creator': 'ezPDF Builder 2006',
'creationdate': '2010-03-19T13:04:08+09:00', 'source': 'sample.pdf',
'total_pages': 10, 'page': 1, 'page_label': '2'}
```

> 소설 "운수 좋은 날"(현진건) PDF, 총 10페이지. `pages[1]`은 두 번째 페이지 내용. 텍스트를 보면 문장 중간중간 띄어쓰기·어순이 살짝 뒤틀려 있는데(예: `이 맺히었다 김첨지의 눈시울도`), 이는 PDF 내부 텍스트 배치 순서를 그대로 따라가는 PyPDFLoader의 한계 — 원본 PDF가 스캔/조판된 방식에 따라 추출 순서가 흐트러질 수 있음.

### 페이지별 요약 + 전체 요약

```python
print("각 페이지 요약")

for i, page in enumerate(pages, start=1):
    prompt = f"다음 PDF 페이지 내용을 2줄로 요약:\n\n{page.page_content}"

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    print(f"페이지 {i} 요약 ---")
    print(response.output_text)

print("\n전체 문서 요약:")

full_text = "\n".join(p.page_content for p in pages)  # 전체 페이지 텍스트를 하나로 결합

prompt = "다음 문서를 10줄로 자연스러운 서술형 문단으로 요약해줘. :\n" + full_text

response_all = client.responses.create(
    model="gpt-4.1-mini",
    input=prompt
)

print(response_all.output_text)
```

> 💡 페이지 단위로 나눠서 요약하는 이유: 문서 전체를 한 번에 프롬프트에 넣으면 길이 제한에 걸리거나 앞부분 정보가 희석될 수 있음. 페이지별로 잘라서 각각 요약 → 전체를 다시 종합하는 방식(Map-Reduce 요약 패턴)의 축소판.

**▶ 실행 결과 (일부)**

```
페이지 1 요약 ---
운수 좋은 날, 인력거꾼 김첨지는 오랜만에 손님을 태워 돈을 벌게 되어 기뻐한다.
그러나 아픈 아내를 돌보며 돈이 절실한 상황이다.
```

> 페이지별 요약이 정상적으로 잘림 없이 생성됨. gpt-4.1-mini로 비용을 절약하면서 요약 태스크를 처리한 것으로 보임 (요약은 상대적으로 가벼운 작업이라 mini 모델로 충분).

---

## 📝 방법 2) `UnstructuredPDFLoader` — 의미 단위(elements) 추출

```python
# poppler-utils: PDF 분석/처리 도구, tesseract-ocr: 이미지 기반 PDF의 OCR용
!apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-kor
!pip install -U "unstructured[pdf]" langchain_community
```

```python
from langchain_community.document_loaders import UnstructuredPDFLoader
import os

pdf_filepath = "sample.pdf"

if not os.path.exists(pdf_filepath):
    raise FileNotFoundError(f"{pdf_filepath} 파일을 찾을 수 없습니다.")  # 파일 부재 시 조기 에러

# --- single 모드: 문서 전체를 하나의 Document로 ---
loader_single = UnstructuredPDFLoader(
    pdf_filepath,
    mode="single",       # 전체를 1개 Document로 합침
    languages=["ko"]     # 한국어 OCR/파싱 힌트
)
docs_single = loader_single.load()
print("single mode 결과")
print("문서 개수:", len(docs_single))
print("metadata:", docs_single[0].metadata)
print("내용 일부:", docs_single[0].page_content[:100])

print("----------")

# --- elements 모드: 문단/줄 단위로 잘게 쪼갠 Document 여러 개 ---
loader_elements = UnstructuredPDFLoader(
    pdf_filepath,
    mode="elements",
    languages=["ko"]
)
docs_elements = loader_elements.load()

print("elements mode 결과")
print("요소 개수:", len(docs_elements))

for idx, e in enumerate(docs_elements[:5], start=1):
    print(f"{idx}번째 요소")
    print(f"Type: {type(e).__name__}")
    print(f"Text: {e.page_content[:50]}")
    print(f"Coordinates: {e.metadata.get('coordinates', '없음')}")  # PDF 상 텍스트의 픽셀 좌표
    print("-" * 40)
```

**▶ 실행 결과**

```
single mode 결과
문서 개수: 1
metadata: {'source': 'sample.pdf'}
내용 일부: 운수 좋은날 현진건 새침하게 흐린 품이 눈이 올 듯하더니 눈은 아니 오고 얼다가 만 비가 추

----------
elements mode 결과
요소 개수: 706

1번째 요소
Type: Document
Text: 운수 좋은날
Coordinates: {'points': ((84.96, 99.13...), ...), 'system': 'PixelSpace',
'layout_width': 595.2, 'layout_height': 841.92}
----------------------------------------
2번째 요소
Type: Document
Text: 현진건
...
```

> **`single` vs `elements` 모드 차이가 뚜렷하게 확인됨:**
> 
> - `single`: PDF 10페이지 전체가 **문서 1개**로 합쳐짐. 게다가 `single` 모드의 `page_content` 미리보기(`운수 좋은날 현진건 새침하게...`)를 보면 PyPDFLoader보다 **띄어쓰기가 훨씬 자연스럽게 복원**된 걸 확인할 수 있음 — unstructured 라이브러리가 레이아웃을 분석해서 읽기 순서를 더 잘 재구성하기 때문.
> - `elements`: 제목, 문단 한 줄 한 줄이 **각각 별도 Document**로 쪼개짐(총 706개). 각 요소마다 PDF 상의 정확한 픽셀 좌표(`Coordinates`)까지 포함 — 표/이미지가 섞인 복잡한 PDF에서 특정 영역만 뽑아낼 때 유용.

---

## 📝 방법 3) `PyMuPDFLoader` — 상세 메타데이터 추출

```python
!pip install pymupdf
```

```python
from langchain_community.document_loaders import PyMuPDFLoader

pdf_filepath = 'sample.pdf'
loader = PyMuPDFLoader(pdf_filepath)
pages = loader.load()
print(len(pages))
print(pages[0].page_content)   # 첫 페이지 텍스트

print(pages[0].metadata)       # 메타데이터 (다른 로더보다 항목이 많음)

import json
print(json.dumps(pages[0].metadata, indent=2, ensure_ascii=False))  # 메타데이터 구조 예쁘게 출력
print(pages[0].metadata["producer"])  # 특정 키만 뽑아 보기
```

**▶ 실행 결과**

```
10
운수좋은날
현진건
새침하게흐린품이눈이올듯하더니눈은아니오고얼다가만비가추
적추적내리는날이었다.
...

{'producer': 'ezPDF Builder 2006', 'creator': 'ezPDF Builder 2006',
'creationdate': '2010-03-19T13:04:08+09:00', 'source': 'sample.pdf',
'file_path': 'sample.pdf', 'total_pages': 10, 'format': 'PDF 1.4',
'title': '', 'author': '', 'subject': '', 'keywords': '',
'moddate': '2010-03-19T13:04:08+09:00', 'trapped': '',
'modDate': "D:20100319130408+09'00'", 'creationDate': "D:20100319130408+09'00'",
'page': 0}
```

> ⚠️ 흥미로운 점: PyMuPDFLoader로 추출한 텍스트는 **띄어쓰기가 아예 사라짐**(`운수좋은날현진건새침하게흐린품이...`) — PyPDFLoader(어순이 뒤틀림)나 UnstructuredPDFLoader(자연스러움)와 또 다른 양상. 같은 PDF도 **어떤 로더를 쓰냐에 따라 텍스트 품질이 크게 달라짐**을 직접 확인할 수 있는 대목. 반면 메타데이터는 `format`, `title`, `author`, `trapped`, ISO 형식 날짜(`modDate`) 등 다른 로더 대비 항목이 훨씬 풍부함 — 문서 속성 분석이 필요한 경우 PyMuPDFLoader가 유리.

---

## 📝 방법 4) `OnlinePDFLoader` — 웹 URL의 PDF 바로 로드

```python
from langchain_community.document_loaders import OnlinePDFLoader

# Transformer 논문("Attention Is All You Need")을 URL에서 직접 로드
loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
pages = loader.load()
print(len(pages))                       # 1
print(pages[0].page_content[:1000])     # 첫 1000자만 미리보기
```

**▶ 실행 결과**

```
1
3 2 0 2

g u A 2

] L C . s c [

7 v 2 6 7 3 0 . 6 0 7 1 : v i X r a

Provided proper attribution is provided, Google hereby grants permission to
reproduce the tables and figures in this paper solely for use in journalistic
or scholarly works.

Attention Is All You Need

Ashish Vaswani∗ Google Brain avaswani@google.com
...
```

> 로컬 파일 경로 대신 **URL만 넘기면** 다운로드 + 파싱까지 한 번에 처리됨. 단, 출력 앞부분의 `3 2 0 2`, `g u A 2` 같은 깨진 텍스트는 논문 좌측 여백에 세로로 인쇄된 arXiv 버전 정보(`2 Aug 2023` 등)가 **글자 단위로 뒤집혀 추출된 것** — PDF 레이아웃이 복잡한 논문류에서 흔히 나타나는 파싱 노이즈. `len(pages)`가 1인 이유는 URL 로더가 논문 전체를 페이지 구분 없이 하나의 Document로 묶어서 반환했기 때문(다만 실제로는 언어 감지 경고와 함께 로드된 것으로 보아, 페이지별 로딩이 아닌 전체 통합 로딩 방식으로 동작).

---

## 🔍 4가지 로더 최종 비교

|로더|텍스트 품질(이 PDF 기준)|문서 단위|메타데이터|적합한 상황|
|---|---|---|---|---|
|`PyPDFLoader`|어순이 다소 뒤틀림|페이지당 1개 (10개)|기본 수준|빠르고 간단한 페이지 단위 처리|
|`UnstructuredPDFLoader` (single)|가장 자연스러움|문서 전체 1개|최소|문서 전체를 하나로 다뤄야 할 때|
|`UnstructuredPDFLoader` (elements)|문장/줄 단위로 분해|706개(세분화)|좌표 포함|표·이미지 등 복잡한 레이아웃 분석|
|`PyMuPDFLoader`|띄어쓰기 소실|페이지당 1개 (10개)|가장 풍부(제목/저자/날짜 등)|문서 속성/메타데이터가 중요한 경우|
|`OnlinePDFLoader`|레이아웃 복잡한 PDF엔 노이즈 발생|URL당 1개|최소|로컬 저장 없이 웹 PDF 즉시 처리|

## 🔗 확장 포인트

- 같은 소설 PDF인데 로더마다 텍스트 정제 품질이 다르게 나온 게 이번 실습의 핵심 발견 — 실제 RAG 파이프라인 구축 시엔 **PDF 성격(스캔본/조판본/텍스트PDF)에 맞는 로더 선택**이 검색 품질에 직접 영향을 줌.
- `elements` 모드로 뽑은 706개 조각을 이전 실습(`rag6web`)처럼 ChromaDB에 임베딩+저장하면, PDF 기반 RAG로 자연스럽게 확장 가능.