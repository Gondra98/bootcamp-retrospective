# Day92_Seq2Seq

## 📅 2026-06-23

---
## 📄 rnn11seq2seq.ipynb — Seq2Seq · Encoder-Decoder · Teacher Forcing 

---

## 🧠 1. 개념 정리

### 🔄 Seq2Seq (Sequence-to-Sequence)

- 가변 길이의 입력 시퀀스를 가변 길이의 출력 시퀀스로 변환하는 모델 구조
- 번역, 챗봇 응답 생성, 요약 등에 쓰임
- **Encoder + Decoder** 두 개의 LSTM으로 구성됨

### 📥 인코더 (Encoder)

- 입력 문장(여기서는 한국어)을 순차적으로 받아서 마지막 시점의 `state_h`(hidden state), `state_c`(cell state) 두 개로 압축
- 이 두 상태가 곧 **문맥 벡터(context vector)** 🧩 — 입력 문장의 의미가 압축된 벡터
- `return_state=True` 옵션을 줘야 LSTM의 최종 상태값을 꺼내올 수 있음 (기본값은 출력 시퀀스만 반환)

### 📤 디코더 (Decoder)

- 인코더가 만든 `[state_h, state_c]`를 **초기 상태(initial_state)** 로 받아서 시작
- `<sos>` 토큰부터 시작해서 한 단어씩 순차적으로 단어를 생성
- `return_sequences=True`로 모든 시점의 출력을 받아서, 각 시점마다 `Dense(softmax)`로 "다음 단어가 무엇일지" 확률분포를 계산

### 👩‍🏫 Teacher Forcing

학습 시 디코더에 **정답 시퀀스를 한 칸씩 밀어서** 입력으로 넣어주는 기법.

```
원본 시퀀스      : <sos>  hi   <eos>
decoder_input    : <sos>  hi          (마지막 토큰 제거 → [:, :-1])
decoder_target   :        hi   <eos>  (첫 토큰 제거     → [:, 1:])
```

이렇게 한 칸씩 어긋나게 만들면 "지금 입력으로 들어온 단어 다음에 어떤 단어가 와야 하는지"를 직접 정답으로 학습시킬 수 있다. 실제 예측값이 아니라 **항상 정답을 다음 입력으로 넣어주기 때문에** 학습이 훨씬 안정적이고 빠르다.

### ⚙️ 학습 모델 vs 추론(inference) 모델을 분리하는 이유

- **학습 시**: Teacher Forcing으로 정답 시퀀스 전체를 한 번에 넣어서 병렬로 빠르게 학습 가능
- **추론(실제 번역) 시**: 정답이 없으므로, `<sos>`부터 시작해서 **한 단어씩 생성 → 그 단어를 다음 입력으로 재사용**하는 반복(loop) 구조가 필요
- 그래서 가중치(레이어)는 공유하되, 동작 방식이 다른 `encoder_model` / `decoder_model`을 별도로 다시 정의함

### 🚦 `<sos>` / `<eos>` 토큰

- `<sos>` (start of sequence): 디코더가 문장 생성을 시작하라는 신호
- `<eos>` (end of sequence): 디코더가 생성을 멈추라는 신호
- ⚠️ 주의: `'<sos>' + eng` 처럼 공백 없이 이어붙이면 `<sos>hi`, `<sos>good`처럼 **매번 다른 토큰**이 만들어져서 "문장 시작" 의미가 사라진다. `'<sos> ' + eng + ' <eos>'`처럼 공백을 둬야 모든 문장에서 동일한 독립 토큰으로 인식된다.

---

## 💻 2. 전체 코드 + 실행 결과

### 1️⃣ Cell 1 — 병렬 문장 데이터 준비

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 병렬 문장 데이터 : 한국어 질문과 영어 답변의 쌍으로 이루어진 리스트
# Seq2Seq 모델이 "입력시퀀스" -> "출력시퀀스" 형태로 학습할 수 있게 해준다
data = [
    ("안녕", "hi"),
    ("잘 지내?", "How are you?"),
    ("고마워", "hi"),
    ("좋은 아침", "good morining"),  # 오타지만 토이 데이터라 그대로 사용
    ("사랑해", "i love you"),
    ("잘 자", "good night")
]
```

---

### 2️⃣ Cell 2 — 토크나이징 & 패딩

```python
# 토크나이저 함수
# 가장 빈도가 높은 100개 단어만 사용하고 나머지는 OOV 처리
def tokenizerFunc(sentences, num_words=100):
    tok = Tokenizer(num_words=num_words, filters='')  # filters='' : 특수기호(<,>,'?')를 단어에서 제거하지 않음
    tok.fit_on_texts(sentences)                # 단어 사전 구축 (단어 -> 정수 인코딩)
    seqs = tok.texts_to_sequences(sentences)   # 문장을 정수 시퀀스로 변환
    return seqs, tok

# 입출력 문장 분리
input_texts = [kor for kor, eng in data]
# <sos>/<eos> 사이에 공백을 둬야 독립된 토큰으로 분리됨 (붙이면 <sos>hi 식으로 토큰이 매번 달라짐)
target_texts = ['<sos> ' + eng + ' <eos>' for kor, eng in data]
print('input_text : ', input_texts)
print('target_text : ', target_texts)

print()
# 인코더용 토크나이징
encoder_seqs, enc_tok = tokenizerFunc(input_texts)
print(encoder_seqs)
print(enc_tok)
print(enc_tok.word_index)
print(enc_tok.index_word)

# 디코더용 토크나이징
decoder_seqs, dec_tok = tokenizerFunc(target_texts)
print(decoder_seqs)
print(dec_tok)
print(dec_tok.word_index)
print(dec_tok.index_word)

# padding : 문장마다 길이가 다르므로 가장 긴 문장 기준으로 0을 채워 동일 길이로 맞춤
encoder_input_data = pad_sequences(encoder_seqs, padding='post')
decoder_sequences = pad_sequences(decoder_seqs, padding='post')
print(encoder_input_data)
print(decoder_sequences)
```

**📊 실행 결과**

```
input_text :  ['안녕', '잘 지내?', '고마워', '좋은 아침', '사랑해', '잘 자']
target_text :  ['<sos> hi <eos>', '<sos> How are you? <eos>', '<sos> hi <eos>', '<sos> good morining <eos>', '<sos> i love you <eos>', '<sos> good night <eos>']

[[2], [1, 3], [4], [5, 6], [7], [1, 8]]
enc_tok.word_index = {'잘': 1, '안녕': 2, '지내?': 3, '고마워': 4, '좋은': 5, '아침': 6, '사랑해': 7, '자': 8}
enc_tok.index_word = {1: '잘', 2: '안녕', 3: '지내?', 4: '고마워', 5: '좋은', 6: '아침', 7: '사랑해', 8: '자'}

decoder_seqs = [[1, 3, 2], [1, 5, 6, 7, 2], [1, 3, 2], [1, 4, 8, 2], [1, 9, 10, 11, 2], [1, 4, 12, 2]]
dec_tok.word_index = {'<sos>': 1, '<eos>': 2, 'hi': 3, 'good': 4, 'how': 5, 'are': 6,
                       'you?': 7, 'morining': 8, 'i': 9, 'love': 10, 'you': 11, 'night': 12}
dec_tok.index_word = {1: '<sos>', 2: '<eos>', 3: 'hi', 4: 'good', 5: 'how', 6: 'are',
                       7: 'you?', 8: 'morining', 9: 'i', 10: 'love', 11: 'you', 12: 'night'}

encoder_input_data =
[[2 0]
 [1 3]
 [4 0]
 [5 6]
 [7 0]
 [1 8]]

decoder_sequences =
[[ 1  3  2  0  0]
 [ 1  5  6  7  2]
 [ 1  3  2  0  0]
 [ 1  4  8  2  0]
 [ 1  9 10 11  2]
 [ 1  4 12  2  0]]
```

> `<sos>`/`<eos>`가 공백으로 분리되니 `<sos>`(=1), `<eos>`(=2)가 모든 문장에서 동일한 토큰으로 잡혔다. (공백 없이 붙였을 때는 `<sos>hi`, `<sos>good`처럼 매번 다른 토큰이 됐었음 — 이전 디버깅에서 고친 부분)

---

### 3️⃣ Cell 3 — 디코더 입력 / 타겟 분리 (Teacher Forcing 준비)

```python
# decoder 입력 / 타겟 분리 (Teacher Forcing)
# 마지막 토큰 <eos>(또는 패딩 0)를 제거한 입력 시퀀스
decoder_input_data = decoder_sequences[:, :-1]
# 첫번째 토큰 <sos>를 제거한 출력(정답) 시퀀스 → 한 칸씩 밀린 형태
decoder_target_data = decoder_sequences[:, 1:]
print('decoder_input_data', decoder_input_data)
print('decoder_target_data', decoder_target_data)
print('차원 확대 전 : ', decoder_target_data.shape)

# sparse_categorical_crossentropy를 사용할 경우 출력 shape은 (batch, timesteps, 1)이어야 함
decoder_target_data = decoder_target_data[..., np.newaxis]
print('차원 확대 후 :', decoder_target_data.shape)

# Seq2Seq 모델의 임베딩 및 LSTM 구조를 설정하는 파라미터 정의
enc_vocab = len(enc_tok.word_index) + 1   # 인코더 Embedding의 input_dim으로 사용 (+1은 패딩용 0번 인덱스)
dec_vocab = len(dec_tok.word_index) + 1   # 디코더 Dense(softmax)의 output_dim으로 사용
```

**📊 실행 결과**

```
decoder_input_data =
[[ 1  3  2  0]
 [ 1  5  6  7]
 [ 1  3  2  0]
 [ 1  4  8  2]
 [ 1  9 10 11]
 [ 1  4 12  2]]

decoder_target_data =
[[ 3  2  0  0]
 [ 5  6  7  2]
 [ 3  2  0  0]
 [ 4  8  2  0]
 [ 9 10 11  2]
 [ 4 12  2  0]]

차원 확대 전 :  (6, 4)
차원 확대 후 : (6, 4, 1)
enc_vocab = 9, dec_vocab = 13
```

---

### 4️⃣ Cell 4 — Seq2Seq 모델 정의 (Functional API, 학습용)

```python
# Seq2Seq 모델 정의 - Functional API 사용

# 인코더 : 입력 시퀀스(예: 한국어 문장)를 받아 요약된 의미 벡터인 state_h(hidden state)와
# state_c(cell state)를 얻는다.
# 이 상태 정보는 디코더가 답변을 생성하는데 필요한 문맥이 된다.

hidden_size = 256   # 임베딩 차원 & LSTM hidden/cell state 차원 (토이 데이터라 작아도 충분)

encoder_inputs = Input(shape=(None,), name='encoder_inputs')
enc_emb_layer = tf.keras.layers.Embedding(enc_vocab, hidden_size, name='enc_emb')   # 임베딩 레이어
encoder_emb = enc_emb_layer(encoder_inputs)

encoder_lstm = LSTM(hidden_size, return_state=True, name='encoder_lstm')
# return_state=True <== 최종 시점의 hidden state(state_h)와 cell state(state_c)를 반환

# _ : 전체 출력 시퀀스(매 스텝 출력)는 사용하지 않음 (인코더는 마지막 상태만 필요)
_, state_h, state_c = encoder_lstm(encoder_emb)
encoder_states = [state_h, state_c]   # 디코더에 넘겨줄 상태 정보를 묶음

# 디코더 : 인코더에서 전달받은 상태로 <sos>부터 시작해 한 단어씩 생성해가는 구조
decoder_inputs = Input(shape=(None,), name='decoder_inputs')
dec_emb_layer = tf.keras.layers.Embedding(dec_vocab, hidden_size, name='dec_emb')   # 임베딩 레이어
decoder_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, name='decoder_lstm')
# decoder_lstm의 initial_state로 인코더 상태를 넘겨받아 "문맥"을 그대로 이어받음
# _, _ : 마지막 시점 상태 (학습 시점에서는 사용하지 않음 — 추론에서만 필요)
decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)

decoder_dense = Dense(dec_vocab, activation='softmax', name='decoder_softmax')
# decoder_outputs를 Dense 레이어에 통과시켜 각 시점마다 단어 예측 확률분포(softmax)를 얻음
decoder_outputs = decoder_dense(decoder_outputs)

# 학습용 모델 : [인코더 입력, 디코더 입력] -> 디코더 출력(단어별 확률분포)
train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
train_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
train_model.summary()

train_model.fit(
    x=[encoder_input_data, decoder_input_data],   # 인코더/디코더 입력
    y=decoder_target_data,                        # 한 칸 밀린 정답 (Teacher Forcing)
    batch_size=2,
    epochs=300,
    verbose=2
)
print('학습 완료')
```

**📊 실행 결과 (모델 구조 요약)**

```
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ encoder_inputs      │ (None, None)      │          0 │ -                 │
│ decoder_inputs      │ (None, None)      │          0 │ -                 │
│ enc_emb (Embedding) │ (None, None, 256) │      2,304 │ encoder_inputs    │
│ dec_emb (Embedding) │ (None, None, 256) │      3,328 │ decoder_inputs    │
│ encoder_lstm (LSTM) │ [(None,256)x3]    │    525,312 │ enc_emb           │
│ decoder_lstm (LSTM) │ [(None,None,256), │    525,312 │ dec_emb,          │
│                     │  (None,256)x2]    │            │ encoder_lstm      │
│ decoder_softmax     │ (None, None, 13)  │      3,341 │ decoder_lstm      │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
Total params: 1,059,597 (4.04 MB)

... (epoch 1~300 loss 로그) ...
학습 완료  (최종 loss ≈ 0.0002)
```

> 데이터가 6문장뿐인 토이 데이터셋이라 300 epoch을 돌리면 loss가 거의 0에 가깝게 떨어진다 (정답을 거의 그대로 외우는 수준의 과적합).

---

### 5️⃣ Cell 5 — 추론 모델 분리 + 번역 테스트

```python
# 학습된 모델을 추론용으로 분리하기
# 인코더 모델(입력 문장 -> 문맥 상태 추출)
# 입력 : encoder_inputs (예: "안녕" -> [2])
# 출력 : encoder_states = [state_h, state_c]
# 목적 : 입력 문장을 인코딩해서 디코더에 넘겨줄 context(문맥 정보)를 생성
encoder_model = Model(encoder_inputs, encoder_states)

# 디코더 모델 (한 스텝씩 단어를 생성하기 위한 별도 모델)
decoder_state_input_h = Input(shape=(hidden_size,), name='dec_state_h')   # 이전 hidden state
decoder_state_input_c = Input(shape=(hidden_size,), name='dec_state_c')   # 이전 cell state
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 학습 때 만든 dec_emb_layer / decoder_lstm / decoder_dense를 "재사용"
# (가중치는 학습된 그대로, 입력 방식만 한 스텝씩으로 바꾸는 것)
dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)

# 확률값이 [0.01, 0.03, ..., 0.88] 이라고 할 때 가장 높은 확률을 가진 인덱스를 추출해 번역 단어 생성
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    # 입력 : 현재 단어 인덱스 (decoder_inputs), 이전 상태 (state_h, state_c)
    [decoder_inputs] + decoder_states_inputs,
    # 출력 : 현재 단어의 softmax 확률분포(decoder_outputs2), 다음 상태(state_h2, state_c2)
    [decoder_outputs2, state_h2, state_c2]
)

# 번역 수행 함수
def translateFunc(sentence):
    seq = enc_tok.texts_to_sequences([sentence])   # 한글 문장을 정수 시퀀스로 변환
    seq = pad_sequences(seq, maxlen=encoder_input_data.shape[1], padding='post')
    states = encoder_model.predict(seq)             # 문맥 벡터 [state_h, state_c] 추출

    # 번역 시작 : <sos> 토큰을 입력으로 넣고 누적 처리
    target_seq = np.array([[dec_tok.word_index['<sos>']]])
    decoded = []

    # 토큰 생성 - 한 단어씩 반복
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states)
        # 가장 확률이 높은 단어 인덱스 선택 -> 단어로 변환
        sampled_idx = np.argmax(output_tokens[0, -1, :])
        sampled_word = dec_tok.index_word[sampled_idx]

        if sampled_word == '<eos>' or len(decoded) > 10:  # 종료 토큰 또는 무한루프 방지
            break

        decoded.append(sampled_word)

        # 다음 step 준비 : 방금 생성한 단어를 다음 입력으로 사용 (실제 추론에서는 정답이 없으므로)
        target_seq = np.array([[sampled_idx]])
        states = [h, c]   # 상태도 갱신해서 이어지는 문맥 유지

    return ' '.join(decoded)

print('번역 테스트')
for s in input_texts:
    print(f"{s} ==> {translateFunc(s)}")
```

**📊 실행 결과**

```
번역 테스트
안녕 ==> hi
잘 지내? ==> how are you?
고마워 ==> hi
좋은 아침 ==> good morining
사랑해 ==> i love you
잘 자 ==> good night
```

> 🎯 학습 데이터 6문장을 그대로 다시 입력해서 테스트했기 때문에 모두 정확히 맞춘다 — 이는 모델이 일반화된 번역 능력을 갖췄다는 뜻이 아니라, 데이터가 너무 적어서(6문장) **암기(과적합)** 한 결과에 가깝다.

---

## 📝 3. 핵심 정리

|구성요소|역할|
|---|---|
|Encoder LSTM|입력 문장을 `state_h`, `state_c`(문맥 벡터)로 압축|
|Decoder LSTM|문맥 벡터를 초기상태로 받아 `<sos>`부터 한 단어씩 생성|
|Teacher Forcing|학습 시 정답을 한 칸 밀어서 입력으로 사용 → 안정적·빠른 학습|
|학습 모델 / 추론 모델 분리|학습은 병렬(Teacher Forcing), 추론은 순차 생성(단어 하나→다음 입력) 이라는 동작 차이 때문|
|`<sos>` / `<eos>`|디코더의 생성 시작/종료 신호 (공백 없이 붙이면 토큰이 깨짐 주의)|

**🚧 한계 및 다음 단계 후보**

- 데이터가 6문장뿐 → 일반화 검증 불가 (학습=테스트라 100% 정답)
- 다음 단계로 `manythings.org/anki`의 실제 kor-eng 병렬 데이터로 교체해서 진짜 번역 성능 확인해볼 수 있음
- Attention 메커니즘 추가, Bidirectional 인코더 적용도 자연스러운 확장 방향

---

## 🐛 4. 오늘 디버깅한 주요 이슈 (히스토리)

1. `<sos>` + 단어 사이 공백 누락 → `<sos>hi`, `<sos>good`처럼 매번 다른 토큰이 생성되던 문제
2. 변수명 불일치 (`decoder_seqs` vs `decoder_sequences`) → `NameError`
3. `tok.word_index`를 `work_index`로 오타 → `AttributeError`
4. 슬라이싱 의도 오류 (`[:, -1]` → `[:, :-1]`로 수정, "마지막 토큰 제거"의 정확한 표현)
5. 정의되지 않은 변수 `hidden_size` 누락 → 추가하여 Embedding/LSTM 차원 통일