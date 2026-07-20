# Day 108_CLIP 멀티모달 이미지 검색

## 📅 2026-07-16

---
# 📄 lang9multimodal.ipynb — CLIP · ChromaDB · LangChain 멀티모달 이미지 검색

## 개요

텍스트로 이미지를 검색하는 CLIP 기반 벡터 검색 파이프라인을 구축하고, 검색된 이미지를 LangChain을 통해 LLM(gpt-4.1-mini)에게 전달해 설명을 생성하는 실습. 크게 3단계로 구성됨:

1. 이미지 → CLIP 임베딩 → ChromaDB 저장
2. 텍스트 쿼리 → CLIP 텍스트 임베딩 → 유사 이미지 검색
3. 검색된 이미지를 base64로 인코딩해 LangChain 멀티모달 메시지로 LLM에 전달

---

## 1. 환경 설정 및 모델 로드

```python
from typing import List, Dict, Any
from pathlib import Path
import base64

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from PIL import Image

# 이미지, 크로마, 컬렉션 경로
IMG_DIR = "images_db"
CHROMA_DIR = "./chroma_image_db"
COLLECTION = "images_docs"

IMG_MODEL_NAME = "clip-VIT-B-32"
TEXT_MODEL_NAME = "sentence-transformers/clip-VIT-B-32-multilingual-v1"   
# ↑ clip 확장 버전, 한국어 강함 (멀티링구얼 텍스트 인코더)

img_model = SentenceTransformer(IMG_MODEL_NAME)     # 이미지 임베딩용 모델
text_model = SentenceTransformer(TEXT_MODEL_NAME)   # 텍스트 임베딩용 모델

client = PersistentClient(path=CHROMA_DIR)
col = client.get_or_create_collection(COLLECTION)
```

**핵심 개념**

- `clip-VIT-B-32`는 이미지 전용 CLIP 인코더, `clip-VIT-B-32-multilingual-v1`은 다국어(한국어 포함) 텍스트 인코더.
- 두 모델은 **같은 벡터 공간**에 매핑되도록 학습되어 있어서, 텍스트 임베딩과 이미지 임베딩 간 코사인 유사도 비교가 가능함.
- `PersistentClient`는 디스크에 벡터DB를 영구 저장 (재실행해도 데이터 유지).

---

## 2. 이미지 벡터DB 구축

```python
# 유틸함수 (이미지 경로 수집용)
def list_imagesFunc(root: str, exts=(".jpg", ".jpeg", ".png")) -> List[str]:
    return [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]

# vectordb 구축 : 이미지들을 CLIP 임베딩으로 저장
def build_image_dbFunc():
    Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
    paths = list_imagesFunc(IMG_DIR)
    if not paths:
        print("해당 이미지 파일이 없어요")
        return

    # VectorDB를 새로 구축할 때 기존 레코드 전체 삭제 후 새로 구축
    existing = col.get(include=[])          # include를 비우면 id만 반환
    ids_to_delete = existing.get("ids", [])  # ids가 없으면 빈 리스트 반환
    if ids_to_delete:
        col.delete(ids=ids_to_delete)
        print(f'기존 {len(ids_to_delete)}개 레코드 삭제 완료')

    print(f"{len(paths)}개 이미지 로딩")

    # PIL.Image 리스트를 이미지용 모델로 임베딩
    imgs = [Image.open(p).convert("RGB") for p in paths]
    vecs = img_model.encode(imgs, normalize_embeddings=True).tolist()
    ids = [f"img_{i}" for i in range(len(paths))]
    metadatas = [{"path": p} for p in paths]

    col.add(
        ids=ids,
        embeddings=vecs,
        metadatas=metadatas,
    )
    print(f"{len(paths)}개 이미지 저장 완료")

build_image_dbFunc()
```

**출력 예시**

```
9개 이미지 로딩
9개 이미지 저장 완료
```

**핵심 개념**

- `normalize_embeddings=True`: 벡터를 정규화해서 코사인 유사도 계산이 안정적으로 되도록 함.
- 재구축 시 기존 레코드를 전부 지우고 새로 넣는 방식 → 중복 저장 방지.
- 메타데이터에 원본 파일 경로를 함께 저장해서, 검색 결과에서 실제 이미지를 다시 불러올 수 있게 함.

---

## 3. 텍스트 → 이미지 검색

```python
# 텍스트로 이미지 검색(vectordb)
def retrieve_imagesFunc(query: str, top_k: int = 3) -> Dict[str, Any]:
    q_vec = text_model.encode([query], normalize_embeddings=True).tolist()
    # encode([query])는 (1, dim) 형태 → tolist() 시 이미 [[...]] 2차원 리스트가 됨
    print("컬렉션 자료수 :", col.count())

    res = col.query(
        query_embeddings=q_vec,   # 이미 2차원이라 다시 감싸지 않음
        n_results=top_k,
        include=['metadatas', 'distances']
    )
    metas = res['metadatas'][0] if res['metadatas'] else []
    distance = res['distances'][0] if res['distances'] else []
    paths: List[str] = [m.get("path", "?") for m in metas]

    return {
        "query": query,
        "image_paths": paths,
        "distances": distance,
    }
```

**출력 예시**

```
컬렉션 자료수 : 9
{'query': '귀여운 동물', 'image_paths': ['images_db/ani2.jpeg', 'images_db/ani3.jpeg', 'images_db/ani1.jpeg'],
 'distances': [1.4775, 1.4822, 1.4858]}
```

**핵심 개념 — distance 값 해석**

- ChromaDB 기본 거리 척도는 **코사인 거리**이며, 값 범위는 0~2 (0에 가까울수록 유사).
- 지금 결과처럼 1.4~1.5대 값은 특이한 게 아니라 정상 범주.

---

## 4. 검색 결과 출력 / 이미지 표시

```python
# 텍스트 출력 보기 좋게
def debug_print_retrieveFunc(result: Dict[str, Any]):
    print("이미지 검색 결과")
    print("질의 : ", result["query"])
    print("검색결과 : ")
    for i, (p, d) in enumerate(zip(result["image_paths"], result["distances"])):
        print(f" {i}){p} (distance={d:.4f})")

# 검색 결과 이미지 출력
def show_images(result: Dict[str, Any], max_cols: int = 3):
    paths = result["image_paths"]
    if not paths:
        print("표시할 이미지 없음")
        return

    for i, p in enumerate(paths, start=1):
        try:
            img = Image.open(p).convert("RGB")
            print(f"{i}. {p}")
            display(img)
        except Exception as e:
            print(f"{i}. {p} 이미지 로딩 실패:{e}")

imsi = retrieve_imagesFunc("귀여운 동물")
debug_print_retrieveFunc(imsi)
show_images(imsi)
```

**출력 예시**

```
컬렉션 자료수 : 9
이미지 검색 결과
질의 :  귀여운 동물
검색결과 : 
 0)images_db/ani2.jpeg (distance=1.4775)
 1)images_db/ani3.jpeg (distance=1.4822)
 2)images_db/ani1.jpeg (distance=1.4858)
```

1. images_db/ani2.jpeg

<img src="images/ani2.jpeg">

2. images_db/ani3.jpeg

<img src="images/ani3.jpeg">

3. images_db/ani1.jpeg

<img src="images/ani1.jpeg">

**핵심 개념**

- `retrieve_imagesFunc`가 반환하는 딕셔너리 키(`image_paths`)와, 이를 소비하는 함수들의 참조 키를 **반드시 일치**시켜야 함.
    - 한쪽만 바꾸면 나머지도 같이 바꿔야 `KeyError` 방지 가능.
- `show_images`는 실제 파일을 다시 열어(`Image.open`) 노트북에 인라인으로 렌더링.

---

## 5. LangChain 멀티모달 메시지 구성 (이미지 → LLM 전달)

```python
from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

# 랭체인 구성
parser = StrOutputParser()

# LLM에게 추천된 이미지 설명용 — 이미지를 base64 data URL로 인코딩
def encode_image_to_data_urlFunc(img_path: str) -> str:
    # jpg -> jpeg 하기를 랭체인이 원함 (OpenAI 멀티모달 API가 요구하는 mime type)
    ext = Path(img_path).suffix.lower().replace(".", "")

    if ext == "jpg":
        ext = "jpeg"
    if not ext:
        ext = "jpeg"

    supportted_exts = {"jpeg", "png", "gif", "webp"}  # openai 입력에서 주로 사용하는 이미지 mime type

    if ext not in supportted_exts:
        raise ValueError(f"지원하지 않는 이미지 형식입니다:{ext}")

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    # data URL 형식: "data:image/{ext};base64,{데이터}" — 콜론이 아니라 세미콜론!
    return f"data:image/{ext};base64,{b64}"
```

**출력 예시**

```
data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD...
```

**핵심 개념**

- `raise` 뒤 코드가 같은 `if` 블록 안에 들여쓰기 되면, 정상 케이스에서 해당 코드가 아예 실행되지 않아 `None`이 반환됨 → 들여쓰기 레벨 주의.
- Data URL 형식은 `data:{mime};base64,{데이터}` — **콜론(:) 아니라 세미콜론(;)**이 정확한 문법. 틀리면 이미지로 인식 안 될 수 있음.

```python
# LLM에게 텍스트 + 여러 이미지가 포함된 HumanMessage 생성
def build_multimodal_messageFunc(query: str):
    res = retrieve_imagesFunc(query, top_k=3)

    contents = [
        {
            "type": "text",
            "text": (
                "사용자 설명 : " + res["query"] + "\n"
                "아래에 표시된 여러 사진을 보고 장면과 분위기를 한국어로 자세히 설명해 줘. \n"
                "사진 순서대로 1번, 2번, 3번 번호를 붙여서 설명해.\n"
                "마크다운은 절대로 표시하지마"
            )
        }
    ]

    for idx, path in enumerate(res["image_paths"], start=1):
        try:
            data_url = encode_image_to_data_urlFunc(path)  # query가 아니라 path를 인코딩
            contents.append({
                "type": "text",
                "text": f"{idx}번 이미지 (파일 경로:{path})"
            })
            contents.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
        except Exception as e:
            # 이미지 인코딩 실패시 로그 텍스트만 출력
            contents.append({
                "type": "text",
                "text": f"{idx}번 이미지 로딩 실패:{path}, 오류 원인은 {e}"
            })

    return [HumanMessage(content=contents)]

# LCEL 체인 구성
image_search_chain = (
    RunnableLambda(build_multimodal_messageFunc)
    | llm
    | parser
)
```

**핵심 개념**

- `HumanMessage(content=[...])`처럼 `content`에 리스트를 넣으면, 텍스트 블록과 이미지 블록을 순서대로 섞어 하나의 메시지로 전달 가능 (OpenAI 멀티모달 메시지 포맷).
- for문 안에서 `encode_image_to_data_urlFunc(query)`처럼 **루프 변수가 아닌 바깥 변수**를 잘못 넣으면, 매 반복마다 엉뚱한 값을 인코딩하게 되는 버그가 발생하므로 인자 확인 필수.

---

## 6. 전체 파이프라인 실행

```python
# 실행
def runFunc():
    print("1) 이미지 구축 DB")
    build_image_dbFunc()

    print("2) 텍스트로 DB의 유사 이미지 검색")
    q = "사랑스럽고 귀여운 동물 이미지를 찾아 줘"

    result = retrieve_imagesFunc(q, top_k=3)
    print('result : ', result)

    debug_print_retrieveFunc(result)
    show_images(result=result)

    print("3) LangChain으로 유사 이미지 검색 결과를 LLM이 설명")
    expl = image_search_chain.invoke(q)
    print('추천 설명 결과 : \n')
    print(expl)

if __name__ == "__main__":
    runFunc()
```

**출력 예시**

```
1) 이미지 구축 DB
기존 9개 레코드 삭제 완료
9개 이미지 로딩
2) 텍스트로 DB의 유사 이미지 검색
컬렉션 자료수 : 9
result :  {'query': '사랑스럽고 귀여운 동물 이미지를 찾아 줘', 
           'image_paths': ['images_db/ani2.jpeg', 'images_db/ani3.jpeg', 'images_db/ani1.jpeg'], 
           'distances': [1.4572, 1.4700, 1.4776]}
이미지 검색 결과
질의 :  사랑스럽고 귀여운 동물 이미지를 찾아 줘
검색결과 : 
 0)images_db/ani2.jpeg (distance=1.4572)
 1)images_db/ani3.jpeg (distance=1.4700)
 2)images_db/ani1.jpeg (distance=1.4776)
```

<img src="images/ani2.jpeg"> <img src="images/ani3.jpeg"> <img src="images/ani1.jpeg">

```
3) LangChain으로 유사 이미지 검색 결과를 LLM이 설명
추천 설명 결과 : 
[gpt-4.1-mini가 생성한 이미지 설명 텍스트]
```

**파이프라인 요약**

1. `build_image_dbFunc()` — 폴더 내 이미지를 CLIP으로 임베딩해 ChromaDB에 저장
2. `retrieve_imagesFunc()` — 한국어 텍스트 쿼리를 임베딩해 유사 이미지 top-k 검색
3. `image_search_chain` (LCEL) — 검색 결과 이미지를 base64로 인코딩 → LangChain 멀티모달 메시지 생성 → LLM(gpt-4.1-mini) 호출 → 이미지 설명 텍스트 생성

---

## 7. 커스텀 임베딩 클래스 (LangChain Chroma 연동용)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 없어요")

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)

# Chroma용 커스텀 임베딩 클래스
class STEmbedding:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 여러 문서를 한 번에 임베딩 (2차원 리스트 반환)
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        # 단일 쿼리 문자열을 임베딩 (1차원 벡터 반환)
        return self.model.encode(text).tolist()

embedding = STEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

**핵심 개념**

- LangChain의 `Chroma`가 요구하는 임베딩 인터페이스는 **두 개의 메서드**를 필요로 함:
    - `embed_documents(texts: List[str]) -> List[List[float]]` — 여러 문서 배치 임베딩
    - `embed_query(text: str) -> List[float]` — 단일 쿼리 임베딩 (1차원 벡터로 반환해야 함)
- `embed_query`에서 `self.model.encode([text])`처럼 리스트로 감싸면 `(1, dim)` 형태가 되어 `.tolist()` 결과가 2차원(`List[List[float]]`)이 되어버림 → 스펙과 어긋나 검색 시 차원 오류 위험. 반드시 `self.model.encode(text)`로 단일 문자열을 그대로 인코딩해야 1차원 벡터가 나옴.
- 모델 리포지토리 이름 오타(`paraphrase-multi-MiniLM...` → 올바른 이름은 `paraphrase-multilingual-MiniLM...`) 시 HuggingFace에서 `RepositoryNotFoundError`(401)가 발생함. 존재하지 않는 리포를 요청한 것이지 인증 문제가 아님.

---

## 요약 정리

|구성 요소|역할|
|---|---|
|`SentenceTransformer(clip-VIT-B-32)`|이미지 → 벡터 임베딩|
|`SentenceTransformer(clip-...multilingual-v1)`|한국어 텍스트 → 동일 벡터 공간 임베딩|
|`ChromaDB (PersistentClient)`|이미지 벡터 영구 저장 및 유사도 검색|
|`base64 data URL 인코딩`|로컬 이미지를 LLM 멀티모달 입력 형식으로 변환|
|`LangChain HumanMessage(content=[...])`|텍스트+이미지 혼합 메시지 구성|
|`LCEL (RunnableLambda \| llm \| parser)`|검색→인코딩→LLM 설명까지 하나의 체인으로 연결|
|`STEmbedding` (커스텀 클래스)|LangChain의 `Chroma`가 요구하는 임베딩 인터페이스에 맞춰 SentenceTransformer 래핑|
