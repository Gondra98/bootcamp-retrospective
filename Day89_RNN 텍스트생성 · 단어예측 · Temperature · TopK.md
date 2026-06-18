# Day89_RNN 텍스트생성 · 단어예측 · Temperature · TopK

## 📅 2026-06-17

---
# 📄 rnn4.ipynb — RNN텍스트생성 · 단어예측 · LSTM · TextGeneration

---

## 🌐 핵심 개념 정리

### 1. RNN 기반 텍스트 생성이란?

- **과거 단어 시퀀스**를 입력받아 **다음에 올 단어**를 예측하는 모델
- 분류 문제처럼 접근: 다음 단어를 vocab 전체 중 하나로 분류 (다항 분류)
- 학습 방식: `[A, B, C, D]` → 입력: `[A, B, C]`, 정답: `D`

---

### 2. 텍스트 전처리 파이프라인

```
원문 텍스트
  ↓ Tokenizer.fit_on_texts()       # 단어사전 생성
  ↓ texts_to_sequences()           # 단어 → 정수 인덱스
  ↓ 슬라이딩 윈도우로 시퀀스 생성    # [w1], [w1,w2], [w1,w2,w3], ...
  ↓ pad_sequences()                # 길이 통일 (앞쪽 0으로 패딩)
  ↓ x, y 분리                      # x: 앞 n개, y: 마지막 1개
  ↓ to_categorical(y)              # y를 one-hot encoding
```

---

### 3. Embedding 레이어

- 정수 인덱스 → 밀집 벡터(dense vector)로 변환
- 단어 간 의미적 유사성을 학습
- `input_dim=vocab_size`, `output_dim=임베딩차원`, `input_length=시퀀스길이`

### 4. LSTM vs SimpleRNN

||SimpleRNN|LSTM|
|---|---|---|
|장기 의존성|취약 (기울기 소실)|강함 (Cell State, Gate 구조)|
|파라미터 수|적음|많음|
|텍스트 생성|짧은 문장 가능|긴 문맥 학습 가능|

### 5. loss 선택 기준

|y 형태|loss|
|---|---|
|one-hot encoding (`to_categorical` 사용)|`categorical_crossentropy`|
|정수 인덱스 (`to_categorical` 안 씀)|`sparse_categorical_crossentropy`|

---

## 📝 코드 1: 전처리 — 토크나이저 및 시퀀스 생성

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

text = """
넥슨이 25년간 서비스해온 캐주얼 온라인 게임 '크레이지 아케이드' 서비스를 종료한다. ...
"""

# 단어 단위 토크나이저 생성 (char_level=False가 기본값)
tok = Tokenizer()
tok.fit_on_texts([text])             # 텍스트로 단어사전 구축
encoded = tok.texts_to_sequences([text])[0]  # 단어 → 정수 인덱스 변환
print(encoded)
print(tok.word_index)

# vocab_size: 실제 단어 수 + 1 (인덱스 0은 패딩용으로 비워두기)
vocab_size = len(tok.word_index) + 1
```

```
# 출력
[17, 18, 19, 20, 21, 1, 6, 22, 7, 23, 3, 24, 25, 26, 27, 28, ...]
{'게임': 1, '크레이지': 2, '지난해': 3, '지난': 4, '오는': 5,
 ''크레이지': 6, '서비스를': 7, 'ip': 8, '넥슨은': 9, '8월': 10, ...}
```

> 💡 **word_index 해석**
> 
> - 빈도 높은 단어일수록 낮은 인덱스 부여 (Keras Tokenizer 기본 동작)
> - '게임'(1), '크레이지'(2) 처럼 기사에서 자주 등장하는 단어가 앞쪽에 위치

---

### 슬라이딩 윈도우로 훈련 데이터 생성

```python
sequences = list()
for line in text.split('\n'):        # 문장 단위로 분리
    enco = tok.texts_to_sequences([line])[0]

    # 바로 다음 단어를 label로 사용: 점점 늘어나는 시퀀스 생성
    # 예) [1], [1,2], [1,2,3], [1,2,3,4], ...
    for i in range(1, len(enco)):
        sequ = enco[:i+1]    # 인덱스 0~i까지 (마지막이 정답)
        sequences.append(sequ)

print('학습에 참여할 샘플 수 : ', len(sequences))
print(sequences)
print(max(len(i) for i in sequences))
```

```
# 출력
학습에 참여할 샘플 수 :  249
[[17, 18], [17, 18, 19], [17, 18, 19, 20], [17, 18, 19, 20, 21], ...]
30
```

> 💡 **249개 시퀀스, 최대 길이 30**
> 
> - 문장마다 길이 2부터 시작해서 단어 수만큼 시퀀스 생성
> - 가장 긴 문장이 30 단어 → max_len = 30

---

### 패딩 및 x, y 분리

```python
# 가장 긴 시퀀스 길이로 맞춤 (짧은 건 앞쪽에 0으로 패딩)
max_len = max(len(i) for i in sequences)
psequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(psequences)

# 마지막 요소를 label(정답)로 분리
x = psequences[:, :-1]    # 앞 n-1개: 입력 feature
y = psequences[:, -1]     # 마지막 1개: 정답 label
print(x)
print(y)

# 다항 분류이므로 y를 one-hot encoding 필요
# 예) y=3, vocab_size=240 → [0,0,0,1,0,...,0] (240차원)
y = to_categorical(y, num_classes=vocab_size)
print(y[:2])
```

```
# 출력 (psequences)
[[  0   0   0 ...   0  17  18]
 [  0   0   0 ...  17  18  19]
 [  0   0   0 ...  18  19  20]
 ...
 [  0   0   0 ... 235 236 237]
 [  0   0 211 ... 236 237 238]
 [  0 211 212 ... 237 238 239]]

# y (정수 인덱스)
[ 18  19  20  21   1   6  22   7  23   3  24  25  26  27 ...]

# y[:2] (one-hot encoding 후)
[[0. 0. 0. ... 1. 0. 0. ...]   ← 인덱스 18 위치에 1
 [0. 0. 0. ... 0. 1. 0. ...]]  ← 인덱스 19 위치에 1
```

> 💡 **padding='pre'**: 짧은 시퀀스의 앞쪽을 0으로 채움 → LSTM이 실제 단어부터 읽도록

---

## 📝 코드 2: 모델 구성 및 학습

```python
model = Sequential()

# Embedding: 정수 인덱스 → 32차원 밀집 벡터로 변환
# input_length: x의 시퀀스 길이 = max_len - 1 (y 분리했으니까)
model.add(Embedding(vocab_size, 32, input_length=max_len - 1))

# LSTM: 시계열 패턴 학습 (tanh 활성화가 기본값)
model.add(LSTM(32, activation='tanh'))

# Dense 레이어: 특징 추출
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# 출력층: vocab_size개 중 하나 선택 → softmax (확률 합=1)
model.add(Dense(vocab_size, activation='softmax'))

# 다항 분류 + one-hot y → categorical_crossentropy
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(x, y, epochs=200, verbose=2)
print(model.evaluate(x, y))
```

```
# 출력 (model.summary)
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding (Embedding)           │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm (LSTM)                     │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ ?                      │   0 (unbuilt) │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 0 (0.00 B)    ← Input() 없이 Embedding 쓰면 빌드 전엔 0으로 표시

# 학습 로그 (주요 에폭만)
Epoch 1/200   8/8 - 3s  - accuracy: 0.0000 - loss: 5.4823
Epoch 10/200  8/8 - 0s  - accuracy: 0.0080 - loss: 5.3005
Epoch 50/200  8/8 - 0s  - accuracy: 0.3454 - loss: 2.4930
Epoch 100/200 8/8 - 0s  - accuracy: 0.8594 - loss: 0.8595
Epoch 150/200 8/8 - 0s  - accuracy: 0.9759 - loss: 0.2506
Epoch 200/200 8/8 - 0s  - accuracy: 0.9880 - loss: 0.0966

# evaluate 결과
[0.0877, 0.9920]    ← loss: 0.0877, accuracy: 99.2%
```

> 💡 **학습 곡선 해석**
> 
> - 1에폭: loss 5.48, acc 0% → 완전 랜덤 수준
> - 50에폭: loss 2.49, acc 34.5% → 패턴 학습 시작
> - 100에폭: loss 0.86, acc 85.9% → 꽤 잘 맞춤
> - 200에폭: loss 0.097, acc 98.8% → 훈련 데이터 거의 암기 수준 (과적합 주의)

---

## 📝 코드 3: 텍스트 생성 함수

```python
def sequence_gen_text(model, t, current_word, n):
    init_word = current_word
    sentence = ""

    for _ in range(n):
        # 현재 문장을 정수 시퀀스로 변환
        encoded = t.texts_to_sequences([current_word])[0]

        # 모델 입력 길이에 맞게 앞쪽 패딩
        encoded = pad_sequences([encoded], maxlen=max_len - 1, padding='pre')

        # 모델 예측: 가장 높은 확률의 단어 인덱스 반환
        result = np.argmax(model.predict(encoded, verbose=0), axis=-1)

        # 인덱스 → 단어 역변환 (word_index를 순회해서 검색)
        for word, index in t.word_index.items():
            if index == result:
                break

        current_word = current_word + ' ' + word
        sentence = sentence + ' ' + word

    sentence = init_word + sentence
    return sentence

print(sequence_gen_text(model, tok, '크레이지', 20))
print(sequence_gen_text(model, tok, '카트라이더', 20))
print(sequence_gen_text(model, tok, '넥슨이', 20))
```

```
# 출력
크레이지 아케이드는 물풍선으로 상대를 가두고 터뜨리는 대전 게임으로 '다오'
'배찌' 등 친숙한 캐릭터를 앞세워 pc방 문화를 이끈 작품이다 작품이다 작품이다 도

카트라이더 앞서 2월에는 수집형 게임 '호연'을 정리했다 정리했다 종료한다
지난해 사상 최대 매출을 올린 회사가 2000년대 pc방을 풍미한 대표작까지 정리하는 것이다

넥슨이 25년간 서비스해온 캐주얼 온라인 게임 '크레이지 아케이드' 서비스를
종료한다 지난해 사상 최대 매출을 올린 회사가 2000년대 pc방을 풍미한 대표작까지 정리하는
```

> 💡 **생성 결과 해석**
> 
> - `크레이지` → 기사 본문 내용을 그대로 이어서 생성 (훈련 데이터 암기)
> - `작품이다 작품이다 작품이다` → 반복 루프 발생 (argmax의 단점 — 항상 같은 단어 선택)
> - `카트라이더` → 훈련 데이터에 없는 단어라 다음 단어를 전혀 모름 → 엉뚱한 문장 생성
> - rnn5의 temperature 샘플링으로 반복 문제 해결 가능

---

## 🌐 전체 흐름 요약

```
텍스트 데이터 (넥슨 기사)
  │
  ├─ Tokenizer → 단어사전 (239개 단어, vocab_size=240)
  ├─ texts_to_sequences → 정수 시퀀스
  ├─ 슬라이딩 윈도우 → 249개 시퀀스 생성 (최대 길이 30)
  ├─ pad_sequences → max_len=30으로 앞쪽 PAD 패딩
  ├─ x/y 분리 → x:(249, 29), y:(249,) → to_categorical → y:(249, 240)
  │
  ├─ Embedding(240,32) → LSTM(32) → Dense(32) → Dense(16) → Dense(240, softmax)
  │    200에폭 학습: loss 5.48 → 0.097, acc 0% → 98.8%
  │
  └─ 생성: seed 단어 → pad → predict → argmax → 단어 역변환 → 반복
```

---

## ⚠️ 자주 나오는 실수 및 한계

| 항목                                 | 설명                                                     |
| ---------------------------------- | ------------------------------------------------------ |
| `vocab_size = len(tok.word_index)` | 인덱스 0 미포함 → `+ 1` 필수                                   |
| `input_length=max_len`             | y 분리 후 길이가 `max_len-1` → `max_len - 1` 사용              |
| loss에 `sparse_` 빠뜨림                | y가 one-hot이면 `categorical_`, 정수면 `sparse_categorical_` |
| 생성 결과 반복                           | argmax는 항상 같은 단어 → rnn5의 temperature 샘플링으로 해결          |
| 훈련 데이터 암기                          | 200에폭 acc 98%는 과적합 — 새로운 문장 생성 능력 제한적                  |
| 미등록 단어 입력                          | 훈련에 없는 단어(카트라이더)는 UNK 처리 → 엉뚱한 결과                      |

---
# 📄 rnn5.ipynb — RNN텍스트생성 · 단어예측 · LSTM · TextGeneration

---

## 🌐 핵심 개념 정리

### 1. rnn4 vs rnn5 차이점

|항목|rnn4 (이전)|rnn5 (이번)|
|---|---|---|
|토크나이저|`keras.Tokenizer`|`TextVectorization` (TF2 권장)|
|y 형태|마지막 토큰 1개|시퀀스 전체 (각 시점의 다음 토큰)|
|loss|`categorical_crossentropy`|`SparseCategoricalCrossentropy(from_logits=True)`|
|텍스트 생성|`argmax` (결정적)|`temperature + top_k` 샘플링 (확률적)|
|데이터 파이프라인|단순 numpy|`tf.data.Dataset` (고성능)|

---

### 2. TextVectorization

- TF2에서 권장하는 내장 토크나이저 레이어
- `adapt()` 한 번으로 단어사전 자동 구축
- 인덱스 0 = PAD, 인덱스 1 = UNK (자동 예약)
- `standardize=None`: 이미 정제했으니 기본 소문자 변환 등 비활성화

### 3. [NL] 토큰 처리

```
원문의 줄바꿈(\n) → " [NL] " 문자열로 치환 → 토크나이저가 하나의 단어로 처리
생성 후 복원: "[NL]" 토큰 → 다시 \n으로
```

- 줄바꿈을 그냥 두면 토크나이저가 무시하거나 처리 불가
- 특수 토큰으로 만들어 문단 구조를 학습시키기 위함

### 4. 슬라이딩 윈도우 방식 (rnn5 핵심)

```
토큰 시퀀스: [A, B, C, D, E, F, ...]   SEQ_LEN=3 기준

윈도우1: [A, B, C, D]  → x=[A,B,C], y=[B,C,D]
윈도우2: [B, C, D, E]  → x=[B,C,D], y=[C,D,E]
윈도우3: [C, D, E, F]  → x=[C,D,E], y=[D,E,F]
```

- rnn4는 y가 마지막 1개였지만, rnn5는 **각 시점마다 다음 토큰을 예측** (`return_sequences=True`)
- 데이터 효율이 훨씬 높음 (1개 윈도우에서 SEQ_LEN개 학습)

### 5. Temperature & Top-K 샘플링

```
logits (원시 점수)
  │
  ├─ 금지 토큰 -inf 처리 (PAD, UNK 제거)
  ├─ ÷ temperature (분포 조절)
  │     temperature < 1 → 보수적 (높은 확률에 집중)
  │     temperature > 1 → 창의적 (다양한 단어 선택)
  ├─ top_k 필터링 (상위 k개만 후보로)
  └─ softmax → 확률 분포 → np.random.choice (샘플링)
```

### 6. from_logits=True

- 출력층에 softmax를 붙이지 않고 **raw logits** 그대로 loss 계산
- softmax → log → loss 과정을 한 번에 계산 → **수치 안정성 향상**
- 생성 시에는 `sample_from_logits`에서 직접 softmax 수동 계산

---

## 📝 코드 1: 라이브러리 임포트 및 데이터 로드

```python
import tensorflow as tf
import numpy as np
import re

# 소설 '토지' 데이터 다운로드 (약 677,125 문자)
path_to_file = tf.keras.utils.get_file(
    'toji.txt',
    'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/rnn_test_toji.txt'
)

with open(path_to_file, encoding='utf-8', errors='ignore') as obj:
    raw_text = obj.read()

print(raw_text[:100])
print('문자 수 : ', len(raw_text))
```

```
# 출력
Downloading data from https://raw.githubusercontent.com/.../rnn_test_toji.txt
1588545/1588545 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step

제 1 편 어둠의 발소리
1897년의 한가위.
까치들이 울타리 안 감나무에 와서 아침 인사를 하기도 전에, 무색 옷에 댕기꼬리를 늘인
아이들은 송편을 입에 물고 마을길을 쏘다니며

문자 수 :  677125
```

---

## 📝 코드 2: 텍스트 정제 및 토크나이저 구성

```python
def clean_str(text: str) -> str:
    text = re.sub(r"[^가-힣0-9() \n]", " ", text)  # 한글, 숫자, 괄호, 공백, 개행 외 제거
    text = re.sub(r"\s{2,}", " ", text)              # 연속 공백 하나로 압축
    return text.strip()

cleaned = clean_str(raw_text)
# 줄바꿈을 [NL] 토큰으로 치환 → 토크나이저가 단어처럼 처리 가능
corpus = cleaned.replace("\n", " [NL] ")

# TextVectorization: TF2 권장 내장 토크나이저 레이어
vectorizer = tf.keras.layers.TextVectorization(
    standardize=None,             # 이미 정제했으므로 기본 전처리 비활성화
    split="whitespace",           # 공백 기준 단어 분리
    output_mode="int",            # 단어 → 정수 인덱스
    output_sequence_length=None,  # 시퀀스 길이 고정 안함
    vocabulary=None               # adapt()으로 직접 구축
)

# adapt(): corpus를 분석해 단어사전 자동 생성
vectorizer.adapt(tf.data.Dataset.from_tensor_slices([corpus]))

vocab = vectorizer.get_vocabulary()  # ['', '[UNK]', '[NL]', '그', '안', ...] 형태
PAD, UNK = 0, 1                      # 0: 패딩, 1: 미등록 단어 (자동 예약)
vocab_size = len(vocab)
print(f'어휘 수 : {vocab_size} (PAD={PAD}, UNK={UNK})')
print('샘플 어휘 : ', vocab[:20])

# 전체 corpus를 정수 시퀀스로 변환
token_ids = vectorizer(tf.constant([corpus])).numpy()[0]
print('토큰 수 : ', len(token_ids))

if len(token_ids) <= 50:
    raise ValueError('토큰 수가 너무 적어 작업 안함')
```

```
# 출력
어휘 수 : 51358 (PAD=0, UNK=1)
샘플 어휘 :  ['', '[UNK]', '[NL]', '그', '안', '있었다', '다', '한', '것', '이',
             '것이다', '있는', '수', '없는', '하고', '못', '같은', '와', '그러나', '내']
토큰 수 :  164150
[   51 51341  2059 ...    49  1590   275]
```

> 💡 **샘플 어휘 해석**
> 
> - 인덱스 0(`''`): PAD, 인덱스 1(`[UNK]`): 미등록 단어, 인덱스 2(`[NL]`): 줄바꿈 특수 토큰
> - 빈도순 정렬이라 '그', '안', '있었다' 같은 고빈도 단어가 앞쪽에 위치

---

## 📝 코드 3: tf.data 파이프라인 구성

```python
SEQ_LEN = 15   # 과거 15개 토큰으로 다음 토큰 예측
BATCH = 32     # 배치 크기
BUFFER = 2000  # 셔플 버퍼 (클수록 더 잘 섞임, 메모리 사용↑)

# ① 토큰 하나씩 Dataset으로 변환
ds = tf.data.Dataset.from_tensor_slices(token_ids)

# ② 슬라이딩 윈도우: SEQ_LEN+1 크기, 1칸씩 이동
ds = ds.window(SEQ_LEN + 1, shift=1, drop_remainder=True)

# ③ 각 윈도우(Dataset)를 텐서(배열)로 수집
ds = ds.flat_map(lambda w: w.batch(SEQ_LEN + 1))

# x, y, w 분리 함수
def split_xyFunc(chunk):   # chunk: SEQ_LEN+1 길이 텐서
    x = chunk[:-1]   # 앞 15개: 입력
    y = chunk[1:]    # 뒤 15개: 각 시점의 다음 토큰 (정답)
                     # → x[0]의 정답=y[0], x[1]의 정답=y[1], ...
    w = tf.cast(tf.not_equal(y, PAD), tf.float32)  # PAD=0(무시), 실제토큰=1(학습)
    return x, y, w

# ④ 파이프라인 최종 구축
ds = (ds.map(split_xyFunc, num_parallel_calls=tf.data.AUTOTUNE)  # 병렬 전처리
        .cache()                                                   # 첫 epoch 후 캐시 (이후 빠름)
        .shuffle(BUFFER)                                           # 무작위 섞기
        .batch(BATCH, drop_remainder=True)                         # 배치화
        .prefetch(tf.data.AUTOTUNE))                               # GPU 연산 중 다음 배치 준비

windows = len(token_ids) - SEQ_LEN
steps_per_epoch = min(100, max(1, windows // BATCH))  # 최대 100 스텝으로 제한
print('steps_per_epoch : ', steps_per_epoch)
```

```
# 출력
steps_per_epoch :  100
```

> 💡 실제 윈도우 수는 164,135개지만 `min(100, ...)` 으로 100 스텝만 사용 — 빠른 실험용

---

## 📝 코드 4: 모델 구성

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN,)),

    # Embedding: 정수 인덱스 → 128차원 밀집 벡터 (의미 유사한 단어끼리 가깝게 학습)
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128),

    # LSTM: return_sequences=True → 각 시점마다 출력 (시퀀스 전체 예측용)
    #        return_sequences=False → 마지막 시점만 출력 (단일 예측)
    tf.keras.layers.LSTM(256, return_sequences=True),

    tf.keras.layers.Dense(256, activation='relu'),

    # 출력층: vocab_size개 logits 출력 (softmax 없음 → from_logits=True로 처리)
    tf.keras.layers.Dense(vocab_size)
])

# SparseCategoricalCrossentropy: y가 정수 인덱스일 때 사용 (one-hot 불필요)
# from_logits=True: softmax 없이 logits 직접 받아 수치 안정적으로 loss 계산
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
model.summary()
```

```
# 출력
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding (Embedding)           │ (None, 15, 128)        │     6,573,824 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm (LSTM)                     │ (None, 15, 256)        │       394,240 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 15, 256)        │        65,792 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 15, 51358)      │    13,199,006 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 20,232,862 (77.18 MB)
 Trainable params: 20,232,862 (77.18 MB)
 Non-trainable params: 0 (0.00 B)
```

> 💡 **Output Shape 해석**
> 
> - `(None, 15, 128)`: 배치 × SEQ_LEN × 임베딩차원
> - `(None, 15, 256)`: LSTM이 15개 시점 각각 출력 (`return_sequences=True`)
> - `(None, 15, 51358)`: 각 시점마다 vocab_size개 logits → 총 2000만 파라미터 (77MB)

---

## 📝 코드 5: 텍스트 생성 유틸 함수

```python
# logits → 확률 → 다음 토큰 하나 샘플링
def sample_from_logits(logits, temperature=1.0, top_k=0, forbid_ids=(0, 1)):
    logits = logits.astype(np.float64)

    # 순서1: 금지 토큰(PAD, UNK) → -inf (선택 확률 0)
    for tid in forbid_ids:
        if 0 <= tid < logits.size:
            logits[tid] = -np.inf

    # 순서2: temperature 적용
    # 낮은 값(0.5~) → 보수적, 높은 값(1.5~) → 창의적
    if temperature <= 0:
        temperature = 1e-8
    logits = logits / temperature

    # 순서3: top_k 필터링 (상위 k개만 남기고 나머지 -inf)
    # np.argpartition: 정렬 없이 상위 k개 인덱스를 O(n)으로 빠르게 찾음
    if top_k:
        k = min(int(top_k), logits.size)
        if 0 < k < logits.size:
            idxs = np.argpartition(-logits, k)[:k]
            mask = np.full_like(logits, -np.inf)
            mask[idxs] = logits[idxs]
            logits = mask

    # 순서4: softmax (수치 안정성을 위해 max 먼저 빼기)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs_sum = probs.sum()

    if probs_sum == 0 or not np.isfinite(probs_sum):
        probs = np.ones_like(probs) / probs.size
    else:
        probs /= probs_sum

    return np.random.choice(len(probs), p=probs)


idx2tok = np.array(vocab, dtype=object)  # 인덱스 → 단어 변환용 배열


def ids_to_text(ids):
    toks = idx2tok[ids]
    toks = [("\n" if t == "[NL]" else t) for t in toks]   # [NL] → 실제 개행 복원
    return " ".join(toks).replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")


def generateFunc(seed_text, max_new_tokens=80, temperature=0.9, top_k=30):
    seed = clean_str(seed_text).replace("\n", " [NL] ")
    seed_ids = vectorizer(tf.constant([seed])).numpy()[0].tolist()

    # SEQ_LEN에 맞게 왼쪽 PAD 패딩
    # 예) SEQ_LEN=15, seed_ids=5개 → [PAD]*10 + seed_ids
    context = [PAD] * max(0, SEQ_LEN - len(seed_ids)) + seed_ids[-SEQ_LEN:]

    out_ids = []
    for _ in range(max_new_tokens):
        x = np.array(context, dtype=np.int32)[None, :]  # (1, SEQ_LEN) 배치 차원 추가
        logits = model.predict(x, verbose=0)[0, -1]     # 마지막 시점 logits만 사용

        tid = sample_from_logits(logits, temperature=temperature,
                                  top_k=top_k, forbid_ids=(PAD, UNK))
        out_ids.append(tid)
        context = context[1:] + [tid]  # 슬라이딩: 맨 앞 제거, 새 토큰 추가

    text = ids_to_text(out_ids)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text
```

---

## 📝 코드 6: 학습 및 SamplerCallback

```python
class SamplerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 5 에폭마다 + 마지막 에폭에는 항상 출력
        if epoch % 5 != 0 and epoch != (self.params.get('epochs', 1) - 1):
            return

        seed = "귀녀의 모습을 한번 쳐다보고 떠나려 했다."
        sample = generateFunc(seed, max_new_tokens=80, temperature=0.9, top_k=30)
        print("\n[샘플 생성:", epoch)
        print(seed + " " + sample[:500])

EPOCHS = 2
history = model.fit(
    ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=[SamplerCallback()],  # 괄호 필수 — 인스턴스 전달
    verbose=2
)

print('final loss : ', float(history.history['loss'][-1]))
print('final acc : ', float(history.history['sparse_categorical_accuracy'][-1]))
```

```
# 출력
Epoch 1/2

[샘플 생성: 0
귀녀의 모습을 한번 쳐다보고 떠나려 했다. 그 을 것 구천이의 그는 수 안 것이다 그는
이 있었다 것이다 을
있는 그런 삼수는 다 수
수 돌이
있었다 구천이는
...
100/100 - 25s - 253ms/step - loss: 9.1476 - sparse_categorical_accuracy: 0.0222

Epoch 2/2

[샘플 생성: 1
귀녀의 모습을 한번 쳐다보고 떠나려 했다. 봉순네 있었다 없었다 이 와
수 와 수
봉순네는 그런 같은 못 하고 이 것 서희는 봉순네 와 것이다
...
100/100 - 20s - 198ms/step - loss: 8.8809 - sparse_categorical_accuracy: 0.0283

final loss :  8.880899429321289
final acc :  0.028291668742895126
```

> 💡 **결과 해석**
> 
> - loss: 9.14 → 8.88 (감소 중), accuracy: 2.2% → 2.8% (아직 초기 학습 단계)
> - 2 에폭으론 의미있는 문장 생성 불가 — 실제론 50~100 에폭 이상 필요
> - 고빈도 단어('그', '수', '안', '것이다')만 반복되는 건 학습 부족의 전형적인 증상

---

## 📝 코드 7: 최종 텍스트 생성 테스트

```python
seed = "귀녀의 모습을 한번 쳐다보고 떠나려 했다."
out = generateFunc(seed, max_new_tokens=100, temperature=0.8, top_k=40)
print('최종 결과 : \n')
print(seed + " " + out)
```

```
# 출력
최종 결과 :

귀녀의 모습을 한번 쳐다보고 떠나려 했다. 수 니 그 내가 안 봉순이 할 그
이 말을 그런 않았다 얼굴을 수 누가
고 안 하고 그런 말을 누가 누가 그러나 같은 봉순네는 별당 봉순네는 것을 그런 없었다
봉순네는 별당 것 봉순네는 그
안 안 못 와 안 것
그 얼굴을
것 니 와 안 못 것 같은 않았다
봉순네는 것 없었다 있었다 봉순이는 안 하고 같은 누가 않았다 같은 말이다
것이 서희는 하고
다 있었다 수 있었다 봉순이는
길상이 것이 이 말
없는
```

> 💡 **생성 품질 개선 방법**
> 
> - EPOCHS 늘리기 (50+)
> - SEQ_LEN 늘리기 (더 긴 문맥 학습)
> - LSTM 레이어 추가 (2~3층)
> - temperature 낮추기 (0.6~0.7) → 더 자연스러운 문장

---

## 🌐 전체 파이프라인 흐름

```
소설 토지 raw_text (677,125자)
  │
  ├─ clean_str()        → 한글/숫자/공백 외 제거
  ├─ replace(\n → [NL]) → 줄바꿈 토큰화
  ├─ TextVectorization  → 단어사전 구축(51,358개) + 정수 시퀀스 변환 (164,150 토큰)
  │
  ├─ tf.data.Dataset
  │    window(16, shift=1) → flat_map → split_xyFunc(x,y,w)
  │    → cache → shuffle → batch(32) → prefetch  [steps_per_epoch=100]
  │
  ├─ Embedding(128) → LSTM(256, return_seq=True) → Dense(256) → Dense(51358)
  │    loss: SparseCategoricalCrossentropy(from_logits=True)
  │    총 파라미터: 20,232,862개 (77.18 MB)
  │
  └─ generateFunc(seed)
       seed 정수화 → PAD 패딩 → 반복 {predict → sample_from_logits → context 슬라이딩}
       → ids_to_text → [NL] 복원 → 최종 문장
```

---

## ⚠️ 헷갈리기 쉬운 포인트

| 포인트                            | 설명                                                 |
| ------------------------------ | -------------------------------------------------- |
| `y = chunk[1:]`                | rnn4는 `chunk[-1]` (1개), rnn5는 `chunk[1:]` (시퀀스 전체) |
| `return_sequences=True`        | 시퀀스 예측이므로 각 시점 출력 필요                               |
| `from_logits=True`             | 출력층에 softmax 없음, loss에서 직접 처리                      |
| `SamplerCallback()`            | 괄호 필수 — 클래스가 아닌 인스턴스 전달                            |
| `temperature=0.8~0.9`          | 너무 낮으면 반복, 너무 높으면 무의미한 단어                          |
| `top_k=30~40`                  | 후보를 너무 좁히면 단조로움, 너무 넓히면 품질 저하                      |
| `steps_per_epoch=min(100,...)` | 전체 데이터 다 안쓰고 100스텝만 — 빠른 실험용                       |