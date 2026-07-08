# Day 101_GPT텍스트이미지질의_멀티모달·스트리밍

## 📅 2026-07-07

---
# 📄 rag2gpt.ipynb — OpenAI · responses.create · GPT4.1-mini

## 🎯 목적

OpenAI GPT LLM(`gpt-4.1-mini`)을 이용한 가장 기본적인 텍스트 질의 실습

## 🧠 개념정리

|용어|설명|
|---|---|
|**OpenAI 클라이언트**|`OpenAI()` 객체 하나로 인증(api_key)~모델 호출까지 처리하는 진입점|
|**`responses.create`**|OpenAI 최신 질의 방식. 기존 `chat.completions.create`의 후속 API. 텍스트/이미지 등 멀티모달 입력도 같은 구조로 처리 가능|
|**`.env` / `load_dotenv()`**|API 키를 코드에 하드코딩하지 않고 환경변수 파일에서 불러오는 패턴. `True` 반환 = 정상 로드|
|**`gpt-4.1-mini`**|가볍고 빠른 모델. 비용·속도 우선인 간단 질의응답에 적합|

## 1. 패키지 설치

```python
!pip install openai
```

OpenAI 파이썬 SDK 설치. Colab엔 대부분 이미 설치돼 있어 "Requirement already satisfied" 로그 뜸

## 2. 환경설정 & API 키 로드

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 로드
```

출력:

```
True
```

`.env` 파일에 `OPENAI_API_KEY=발급받은키` 형식으로 저장돼 있어야 정상 로드됨

## 3. (선택) 사용 가능 모델 목록 확인

```python
# 모델 목록 확인
client = OpenAI()
models = client.models.list()
print(models)
```

실제 서비스엔 불필요, 계정에서 쓸 수 있는 모델 확인용 디버깅 코드 ⚠️ `api_key` 인자 없이 호출 → 환경변수 `OPENAI_API_KEY`를 자동으로 읽어감

## 4. 실제 사용할 클라이언트 생성

```python
# GPT 클라이언트 객체 생성
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
```

`api_key`를 명시적으로 전달 — 3번 셀과 방식만 다르고 결과는 동일

## 5. 질의 실행

```python
response = client.responses.create(
    model = "gpt-4.1-mini",
    input = "강남구에 가성비 좋은 식당 알려줘"
)

print(response.output_text)
```

`input`에 문자열만 입력하는 가장 단순한 형태 (system/user role 구분 없음) `response.output_text` → 모델이 생성한 최종 텍스트만 추출

출력:

```
강남구에는 다양한 가성비 좋은 식당이 많이 있습니다. 몇 가지 추천드릴게요!

1. 봉추찜닭 강남역점 — 찜닭 / 넉넉한 양, 매콤달콤한 맛
2. 명동칼국수 — 칼국수, 만두 / 저렴하고 든든한 한 끼
3. 진미식당 — 백반, 고기구이 / 반찬 다양, 회식·가족모임 적합
4. 다이닝포인트 강남역점 — 한식, 돈까스 / 합리적 가격
5. 하동관 강남점 — 곰탕, 설렁탕 / 깊은 국물맛, 점심 추천

더 구체적인 메뉴나 위치, 분위기에 따라 추천받고 싶으시면 알려주세요!
```

## 메모

- rag3image.ipynb와 비교: 여기는 system_prompt 없이 `input`에 텍스트만 넣는 최소 형태 → 이미지/역할 지정 추가 시 `content` 배열 구조(`input_text`, `input_image`)로 확장됨
- 리스트 형태 답변도 잘 처리하는 것 확인
- 모델 목록 확인 셀(3번)은 실습 참고용, 배포 코드엔 불필요

## 관련

- [[Day101_GPT활용과RAG기초_멀티모달·벡터DB]]
- [[📄 rag3image.ipynb — 멀티모달 · base64 · 스트리밍]]

---
# 📄 rag3image.ipynb — 멀티모달 · base64 · 스트리밍

## 🎯 목적

이미지를 GPT에 함께 전달해 설명/시 작성을 요청하는 멀티모달(텍스트+이미지) 질의 실습. 단일 실행형 스크립트 → 클래스 기반 스트리밍 구조로 확장

## 🧠 개념정리

|용어|설명|
|---|---|
|**멀티모달(Multi-modal)**|텍스트뿐 아니라 이미지 등 여러 형태의 입력을 함께 처리하는 방식. `content` 배열 안에 `input_text`와 `input_image`를 동시에 넣어 구현|
|**base64 인코딩**|이미지(바이너리) 데이터를 텍스트로 변환하는 방식. OpenAI API는 이미지를 URL 문자열(`data:image/jpeg;base64,...`) 형태로 받기 때문에 필수|
|**`role: system` / `role: user`**|system은 모델의 역할·태도·규칙 지정, user는 실제 질문·이미지 전달 담당|
|**스트리밍(`stream=True`)**|응답을 한번에 받지 않고, 생성되는 조각(delta)을 실시간으로 하나씩 받아 출력하는 방식. 챗봇 UI에서 "타이핑되는" 효과의 원리|
|**`event.type == "response.output_text.delta"`**|스트리밍 이벤트 중 실제 텍스트 조각만 필터링하는 조건|

## 1. 패키지 설치

```python
!pip install openai python-dotenv
```

## 2. 환경설정

```python
from openai import OpenAI
import io, os
import base64
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
model = "gpt-4o-mini"
```

`gpt-4o-mini`는 이미지 입력(비전)을 지원하는 모델 — rag2gpt.ipynb의 `gpt-4.1-mini`와 달리 멀티모달용으로 선택

### 사용된 테스트 이미지

<img src="images/pic.jpeg" width="500"/>

산책로 풍경 사진 (자갈길, 잔디밭, 나무들) — 아래 "설명"과 "시" 출력이 이 이미지를 보고 생성된 결과

---

## 1️⃣ 단일 실행형 스크립트

### 이미지 열기 & base64 인코딩

```python
# 프롬프트 작성
system_prompt = "당신은 그림을 보고 내용에 대한 설명을 잘하는 전문가입니다."   # 모델의 역할 지정
user_prompt = "다음 이미지를 보고 무슨 내용인지 작성하시오"    # 실제 사용자 요청 내용

# 이미지 열기
image_path = "pic.jpeg"
with open(image_path, 'rb') as f:
    image_data = f.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    print(image.size)   # (658, 465)

# 이미지를 base64 문자열로 변환 - OpenAI api가 원함
base64_image = base64.b64encode(image_data).decode("utf-8")
# print(base64_image)
```

`Image.open(io.BytesIO(...))`은 실제로 이미지가 정상적으로 열리는지 확인하는 용도 (크기 출력용). API 전달에는 `image_data`(원본 바이트)를 그대로 base64 인코딩해서 사용

출력:

```
(367, 267)
```

### 멀티모달 질의 실행

```python
# 멀티 모달(텍스트 + 이미지) 질의 실행
response = client.responses.create(
    model=model,
    input = [
        {   # 첫번째 전달할 메세지 : system 역할 메세지
            "role":"system",   # 모델의 역할, 태도, 규칙, 답변형식 등 지정
            "content":[
                {
                    "type":"input_text",
                    "text":system_prompt
                }
            ]
        },
        {   # 두번째 전달할 메세지 : user 역할 메세지
            "role":"user",    # 사용자의 실제 질문, 요청, 이미지, 텍스트 전달...
            "content":[
                {
                    "type":"input_text",
                    "text":user_prompt
                },
                {
                    "type":"input_image",
                    "image_url":f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
)

# print(response)
print('이미지 설명 : ')
if response.output_text:
    print(response.output_text)
else:
    print("응답이 비었어요")
```

`content` 배열 안에 `input_text`(질문)와 `input_image`(이미지)를 같이 넣는 게 핵심 — 이 구조 덕분에 하나의 메시지로 텍스트+이미지를 동시에 전달 가능

출력:

```
이미지 설명 : 
이미지에는 산책로가 보이는 풍경이 담겨 있습니다. 길은 고르고 깔끔한 자갈길로 되어 있으며, 
양쪽에는 풀이 자란 넓은 잔디밭과 나무들이 있습니다. 길을 따라 몇 그루의 나무가 흩어져 있으며, 
푸르른 나무숲이 배경을 이루고 있습니다. 하늘은 맑고 푸르른 색으로, 전반적으로 평화롭고 
자연적인 분위기를 느낄 수 있는 장면입니다.
```

---

## 2️⃣ 클래스 기반 구조화 (스트리밍 응답)

### MyMultiModel 클래스

```python
class MyMultiModel:
    def __init__(self, client, model, system_prompt="", user_prompt=""):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    # 이미지 경로를 받아 스트리밍 응답을 생성
    def streamMethod(self, image_path):
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = self.client.responses.create(
            model=self.model,
            input = [
                {"role":"system", "content":[{"type":"input_text","text":self.system_prompt}]},
                {"role":"user", "content":[
                    {"type":"input_text","text":self.user_prompt},
                    {"type":"input_image","image_url":f"data:image/jpeg;base64,{base64_image}"}
                ]}
            ],
            stream=True    # 스트리밍 응답으로 생성되는 텍스트가 한 줄씩 실시간 출력됨
        )
        return response
```

1번의 단일 스크립트를 재사용 가능한 클래스로 리팩토링. `system_prompt`/`user_prompt`를 인스턴스 속성으로 받아서, 매번 다른 역할(설명가/시인 등)로 쉽게 바꿔 쓸 수 있게 함

### 스트리밍 응답 출력 함수

```python
# 스트리밍 응답 출력 함수
def stream_responseFunc(response):
    for event in response:  # 스트리밍으로 전달되는 이벤트를 하나씩 수행

        # delta : 스트리밍 중에 조금씩 도착하는 '응답 조각'
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
```

`stream=True`로 받은 응답은 하나의 텍스트가 아니라 이벤트 스트림 → `response.output_text.delta` 타입인 이벤트만 골라서 `end=""`로 이어붙여 출력 (줄바꿈 없이 실시간 타이핑처럼 보이게)

### 실행 (시 작성 프롬프트로 테스트)

```python
llm = "gpt-4o-mini"
system_prompt = "당신은 시인입니다. 당신의 임무는 주어진 이미지를 가지고 시를 작성하는 것입니다"   # 모델의 역할 지정
user_prompt = "다음 이미지를 보고 무슨 시를 작성하시오"    # 실제 사용자 요청 내용

multimodal_gpt = MyMultiModel(
    client,
    model=model,
    system_prompt=system_prompt,
    user_prompt=user_prompt
)

IMAGE_PATH = "pic.jpeg"
response = multimodal_gpt.streamMethod(IMAGE_PATH)
stream_responseFunc(response)
```

같은 이미지, 다른 system_prompt(설명가 → 시인)로 바꿔서 결과가 어떻게 달라지는지 비교 실습

출력:

```
길이 열려 있네, 고요한 초원 위에  
푸르른 나무들 사이로 뻗어가는  
작고 부드러운 자갈길,   
하늘은 높은 푸른색으로 물들어.  

차분한 흙길을 따라 나아가면  
사과나무들이 속삭이는 듯,  
흩날리는 꽃잎의 향기는  
봄의 속삭임처럼 달콤해.  

푸른 들판의 끝자리,  
여기서 시작되는 나의 여행,  
자연의 품 안에 안기어  
내 마음도 함께 피어나리.  

이 길의 끝에 무언가 있을까?  
미지의 세계가 나를 부르네,  
자연과 나의 조화 속에  
온전한 나를 찾는 여정.
```

## 💡 메모

- 같은 이미지(산책로 풍경)로 "설명" vs "시" 두 프롬프트 결과 비교 → system_prompt만 바꿔도 완전히 다른 톤의 결과물이 나오는 걸 확인
- 1번(단일 스크립트)은 결과를 한번에 받고, 2번(클래스+스트리밍)은 실시간으로 조각조각 받음 — 실제 챗봇 UX엔 스트리밍 방식이 자연스러움
- rag2gpt.ipynb와 비교: `input`이 단순 문자열이 아니라 `role`+`content` 배열 구조로 확장된 형태

## 🔗 관련

- [[Day101_GPT활용과RAG기초_멀티모달·벡터DB]]
- [[📄 rag2gpt.ipynb — OpenAI · responses.create · GPT4.1-mini]]
