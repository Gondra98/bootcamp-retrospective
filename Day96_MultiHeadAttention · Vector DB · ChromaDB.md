# Day96_MultiHeadAttention · Vector DB · ChromaDB

## 📅 2026-06-29

---
# 📄 rnn17multiohead.ipynb — Transformer · MultiHeadAttention · Encoder-Decoder

---

## 개념 정리

### Transformer Encoder-Decoder가 하는 일

이 노트북은 RNN/LSTM 없이 **Attention만으로** 문장을 입력받아 그대로 재생성하는 가장 단순한 형태의 Transformer를 구현한다. (번역이 아니라 "동일 문장 복원" 과제라서 구조 학습 자체에 집중할 수 있음)

```
인코더: 문장을 받아서 문맥 정보가 반영된 벡터로 압축
디코더: <sos> + 일부 문장을 받아서, 인코더 출력을 참고하며 다음 단어를 한 개씩 예측
```

### 핵심 구성요소 3가지

|구성요소|역할|
|---|---|
|**Self-Attention**|문장 내부에서 단어들끼리 서로 얼마나 관련있는지 계산|
|**Cross-Attention** (인코더-디코더 어텐션)|디코더가 출력을 만들 때 인코더의 어떤 부분을 참고할지 계산|
|**Residual + LayerNorm**|각 서브레이어 출력에 입력을 더하고 정규화 → 학습 안정화, 정보 손실 방지|

### 왜 디코더 Self-Attention엔 `use_causal_mask=True`가 필요한가

디코더는 **다음 단어를 예측**하는 역할이라, 학습 시점에 미래 토큰을 보면 안 됨. causal mask는 토큰 i가 1~i까지만 보고 i+1을 예측하도록 강제하는 삼각형 마스크다. 이게 없으면 디코더가 정답을 베껴쓰는 것과 같아져서 학습은 잘 되는 것처럼 보이지만 실제 추론(자기 회귀적 생성)에서는 의미 없는 모델이 된다.

### Residual 연결이 "직전 서브레이어 출력"을 따라가야 하는 이유

Transformer는 서브레이어를 통과할 때마다 残差(residual) 스트림이 끊기지 않고 이어져야 한다. 디코더의 두 번째 서브레이어(cross-attention)는 `dec_emb`가 아니라 **첫 번째 서브레이어 출력인 `out2`**에 더해야, self-attention에서 얻은 정보가 cross-attention 이후로도 계속 전달된다.

### Teacher Forcing 입력/타겟 시프트

```
원문(encoder_input):   I    have   a    pen
decoder_input:        <sos>  I    have   a     ← 한 칸 밀려 들어감
decoder_target:        I    have   a    pen   ← 디코더가 맞춰야 할 정답 (= 원문 그대로!)
```

디코더 입력은 "정답을 한 칸 밀어서 `<sos>`를 앞에 붙인 것" → 학습 시엔 정답을 한 스텝씩 보여주며 다음 단어를 맞히게 학습(Teacher Forcing). **타겟은 원문(encoder_input)과 동일해야 함** — 위치 t의 디코더 입력이 다음 단어(=원문의 t번째 단어)를 예측하도록.

> 🐛 **실제 코드엔 이 부분에 버그가 있음** — 아래 Cell 1 참고. 코드가 타겟을 한 번 더 밀어버려서 위 그림과 다른 값이 만들어짐.

---

## 코드 + 주석

### Cell 0 — 하이퍼파라미터 및 데이터 준비

```python
# Transformer Encoder-Decoder 구조 이해 - MultiHeadAttention

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense
from tensorflow.keras.models import Model

# 하이퍼 파라미터 설정
max_len = None    # 입출력 시퀀스의 최대길이 (다음 셀에서 실제 데이터 보고 계산됨)
d_model = 16      # 임베딩 차원 (각 단어를 16차원 벡터로 표현)
num_heads = 2     # 어텐션 헤드 수 (관점을 2개로 나눠서 어텐션 계산)
dff = 32          # FFN(Feed Forward Network) 내부 은닉층 크기
epochs = 30
batch_size = 4    # 문장이 4개뿐이라 batch_size=4면 1 epoch = 1 step

# 토이 데이터: 문장 4개 (모두 "주어 + 동사 + (관사) + 목적어" 4단어 구조)
sentences = [
    "I have a pen",
    "You are nice",
    "He love her",
    "We eat pizza"
]
```

### Cell 1 — 토크나이징 + 디코더 입력/타겟 생성 (Teacher Forcing)

```python
# 단어 단위 토크나이저: filters='' (구두점 제거 안함), lower=False (대소문자 구분 유지)
tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')   # oov : 단어사전외 단어
tokenizer.fit_on_texts(sentences)
print(tokenizer.word_index)
seqs = tokenizer.texts_to_sequences(sentences)
print(seqs)

# 모든 문장이 4단어라 max_len=4, 패딩 불필요하지만 일반화를 위해 pad_sequences 적용
max_len = max(len(s) for s in seqs)
encoder_input_data = pad_sequences(seqs, maxlen=max_len, padding='post')
print(encoder_input_data)

# 디코더 입력/타겟 생성 : 어휘 사전 크기와 시작 토큰 설정
vocab_size = len(tokenizer.word_index) + 1   # +1은 패딩(0)을 위한 자리
sos_token = vocab_size   # 시작 토큰 <sos>의 정수 인덱스를 어휘 사전 마지막 번호로 지정

tokenizer.word_index['<sos>'] = sos_token
vocab_size += 1   # <sos> 추가했으니 vocab_size 다시 +1

# 디코더 입력 : [<sos>] + encoder_input_data[:,:-1]
# 예) 정답이 [I, have, a, pen]이면 디코더 입력은 [<sos>, I, have, a]
decoder_input_data = np.zeros_like(encoder_input_data)
decoder_input_data[:, 0] = sos_token
decoder_input_data[:, 1:] = encoder_input_data[:, :-1]   # 정답을 한 칸씩 오른쪽으로 밀어서 채움
# 목적 : 디코더는 훈련 중, 정답 문장을 한 토큰씩 왼쪽으로 밀기(shift)한 형태를 입력으로 받아야 함 (Teacher Forcing)

# 디코더 타겟 : 디코더가 각 타임스텝에서 맞춰야 할 정답 (= 원문 그대로, 한 칸 당겨서 정렬)
decoder_target_data = np.zeros_like(encoder_input_data)
decoder_target_data[:, :-1] = encoder_input_data[:, 1:]   # 🐛 버그: 아래 설명 참고
print(decoder_input_data)
```

🐛 **버그 발견**: `decoder_target_data[:, :-1] = encoder_input_data[:, 1:]`는 타겟을 **한 번 더 시프트**해버림. `decoder_target_data`는 원래 `encoder_input_data`와 동일해야 하는데(타겟 = 원문 그대로), 코드가 원문을 한 칸 더 밀어서 넣는 바람에 디코더 입력과 타겟 사이의 위치 정렬이 한 칸 어긋남.

```python
# 수정안
decoder_target_data = encoder_input_data.copy()   # 타겟 = 원문 그대로 (시프트 없음)
```

**출력 (현재 버그 있는 코드 기준)**

```
{'<OOV>': 1, 'I': 2, 'have': 3, 'a': 4, 'pen': 5, 'You': 6, 'are': 7, 'nice': 8, 'He': 9, 'love': 10, 'her': 11, 'We': 12, 'eat': 13, 'pizza': 14}
[[2, 3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]

# encoder_input_data (패딩 적용됨, max_len=4)
[[ 2  3  4  5]
 [ 6  7  8  0]
 [ 9 10 11  0]
 [12 13 14  0]]

# decoder_input_data (최종 결과: <sos>=15 + 한 칸 시프트된 원문)
[[15  2  3  4]
 [15  6  7  8]
 [15  9 10 11]
 [15 12 13 14]]
```

**실제 `decoder_target_data` 값 (버그 vs 정상)**

```
encoder_input_data (= 원문)        : [[ 2  3  4  5] [ 6  7  8  0] [ 9 10 11  0] [12 13 14  0]]
decoder_target_data (버그, 현재값)  : [[ 3  4  5  0] [ 7  8  0  0] [10 11  0  0] [13 14  0  0]]
decoder_target_data (정상값이어야 함): [[ 2  3  4  5] [ 6  7  8  0] [ 9 10 11  0] [12 13 14  0]]  ← encoder_input_data와 동일
```

→ vocab_size = 16 (단어 14개 + `<OOV>` + `<sos>`), `<sos>` 토큰 인덱스 = 15

> ⚠️ **셀 실행 순서 주의**: `decoder_target_data`는 이 셀에서 정의됨. 커널 재시작 후 이 셀을 건너뛰고 모델 정의/학습 셀만 실행하면 `NameError`가 발생함 — 위에서부터 순서대로 실행할 것.

### Cell 2 — Transformer 모델 정의

```python
# Transformer 구성
def get_transformerFunc(vocab_size, max_len):
  # ===== 인코더 =====
  # 인코더 입력 : 정수 시퀀스 형태로 들어옴 (예: [2,5,3,0,0] -> "I like you" + 패딩)
  enc_inputs = Input(shape=(max_len, ))
  emb_layer = Embedding(vocab_size, d_model, mask_zero=True)   # mask_zero=True: 패딩(0) 토큰 무시
  enc_emb = emb_layer(enc_inputs)

  # Multi-head Self-Attention: 문장 내부 단어들끼리 서로 얼마나 관련있는지 계산
  atten_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(enc_emb, enc_emb)
  # 잔차 연결 + Layer Normalization(학습 안정화): 원본(enc_emb) + 어텐션 결과를 더해서 정보 보존
  out1 = LayerNormalization(epsilon=1e-6)(enc_emb + atten_out)

  # FFN (Feed Forward Network): 위치별로 독립적으로 작동하는 2층 신경망
  ffn = Dense(dff, activation='relu')(out1)
  ffn = Dense(d_model)(ffn)

  # 또 잔차 연결 (FFN 입력 out1 + FFN 출력)
  enc_out = LayerNormalization(epsilon=1e-6)(out1 + ffn)    # 최종 인코더 출력 — 디코더가 이걸 참고함

  # ===== 디코더 =====
  # 디코더 입력 : <sos> + 문장일부 : 예 : <sos> I have a -> [10, 1, 2, 3]처럼 정수 인코딩된 입력
  dec_inputs = Input(shape=(max_len, ))
  dec_emb = emb_layer(dec_inputs)   # 인코더와 디코더가 같은 임베딩 레이어 사용(공통된 단어사전 사용)

  # 1단계 : Self-Attention (디코더 내부)
  # use_causal_mask=True 핵심! 디코더가 미래 토큰을 보지 못하게 막아서, 실제 추론 상황(이전 토큰만 보고 다음 토큰 예측)과 동일하게 학습되도록 함
  attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(dec_emb, dec_emb, use_causal_mask=True)
  out2 = LayerNormalization(epsilon=1e-6)(dec_emb + attn1)

  # 2단계 : 인코더-디코더 어텐션 (Cross-Attention)
  # Query=out2(디코더), Key/Value=enc_out(인코더) → 디코더가 인코더 출력 중 어디를 참고할지 결정
  attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(out2, enc_out)
  # residual은 직전 서브레이어 출력인 out2에 더함 (dec_emb가 아님 — residual stream을 끊지 않기 위함)
  out3 = LayerNormalization(epsilon=1e-6)(out2 + attn2)

  # 위치별 독립적으로 작동하는 2층 신경망 (디코더 FFN)
  ffn2 = Dense(dff, activation='relu')(out3)
  ffn2 = Dense(d_model)(ffn2)

  # 마지막으로 잔차 연결 + 정규화 -> 디코더 최종 출력
  dec_out = LayerNormalization(epsilon=1e-6)(out3 + ffn2)

  # 최종 출력층: 각 타임스텝마다 vocab_size 차원의 확률분포 (다음 단어 예측)
  final = Dense(vocab_size, activation='softmax')(dec_out)
  return Model(inputs=[enc_inputs, dec_inputs], outputs=final)  # 입력2 -> 출력 1 형태의 트랜스포머 모델

# 모델 생성 및 컴파일
transformer = get_transformerFunc(vocab_size, max_len)
# sparse_categorical_crossentropy: 타겟이 one-hot이 아니라 정수 인덱스 그대로라 사용
transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
transformer.summary()

print('Encoder input shape : ', encoder_input_data.shape)
print('Decoder input shape : ', decoder_input_data.shape)
print('Decoder target shape : ', decoder_target_data.shape)
```

**출력 — `transformer.summary()`**

```
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer_1       │ (None, 4)         │          0 │ -                 │
│ input_layer         │ (None, 4)         │          0 │ -                 │
│ embedding           │ (None, 4, 16)     │        256 │ input_layer, ...  │
│ multi_head_attenti… │ (None, 4, 16)     │      2,160 │ embedding[0][0] x2│ ← 인코더 self-attn
│ add (Add)           │ (None, 4, 16)     │          0 │ embedding + attn  │
│ layer_normalization │ (None, 4, 16)     │         32 │ add               │ ← out1
│ dense (Dense)       │ (None, 4, 32)     │        544 │ layer_normalizat. │ ← 인코더 FFN 1층
│ multi_head_attenti… │ (None, 4, 16)     │      2,160 │ embedding[1][0] x2│ ← 디코더 self-attn (causal)
│ dense_1 (Dense)     │ (None, 4, 16)     │        528 │ dense             │ ← 인코더 FFN 2층
│ add_2 (Add)         │ (None, 4, 16)     │          0 │ embedding + attn  │ ← out2 잔차
│ add_1 (Add)         │ (None, 4, 16)     │          0 │ out1 + ffn        │
│ layer_normalizatio… │ (None, 4, 16)     │         32 │ add_2             │ ← out2
│ layer_normalizatio… │ (None, 4, 16)     │         32 │ add_1             │ ← enc_out
│ multi_head_attenti… │ (None, 4, 16)     │      2,160 │ out2, enc_out     │ ← cross-attention
│ add_3 (Add)         │ (None, 4, 16)     │          0 │ out2 + attn2      │ ← out3 잔차 (수정된 부분!)
│ layer_normalizatio… │ (None, 4, 16)     │         32 │ add_3             │ ← out3
│ dense_2 (Dense)     │ (None, 4, 32)     │        544 │ out3              │ ← 디코더 FFN 1층
│ dense_3 (Dense)     │ (None, 4, 16)     │        528 │ dense_2           │ ← 디코더 FFN 2층
│ add_4 (Add)         │ (None, 4, 16)     │          0 │ out3 + ffn2       │ ← dec_out 잔차
│ layer_normalizatio… │ (None, 4, 16)     │         32 │ add_4             │ ← dec_out
│ dense_4 (Dense)     │ (None, 4, 16)     │        272 │ dec_out           │ ← final (vocab_size=16)
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 9,312 (36.38 KB)
 Trainable params: 9,312 (36.38 KB)
 Non-trainable params: 0 (0.00 B)

Encoder input shape :  (4, 4)
Decoder input shape :  (4, 4)
Decoder target shape :  (4, 4)
```

→ `add_3`(out3)이 `out2 + attn2`로 연결돼 있는 게 보임 — residual 수정이 구조에 제대로 반영된 것.

> 📌 **참고**: 이 구현엔 **Positional Encoding이 없음**. Self-Attention 자체는 순서 정보가 없는 연산(순서를 섞어도 결과가 동일)이라, 완전한 Transformer라면 enc_emb/dec_emb에 위치 정보(sin/cos 또는 학습형 position embedding)를 더해줘야 함. 이 노트북은 MultiHeadAttention 구조 이해가 목적이라 생략된 것으로 보임.

### Cell 3 — 모델 학습

```python
# 모델 학습
history = transformer.fit(
    [encoder_input_data, decoder_input_data],   # 입력 2개: 인코더 입력 + 디코더 입력(시프트된 정답)
    np.expand_dims(decoder_target_data, -1),    # (4, 4) -> (4, 4, 1): sparse_categorical_crossentropy 형식
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)
```

**출력 — loss 추이 (30 epoch, 일부)**

```
Epoch 1/30   1/1  10s  10s/step - loss: 3.4494
Epoch 5/30   1/1  0s    75ms/step - loss: 2.4142
Epoch 10/30  1/1  0s    72ms/step - loss: 1.9027
Epoch 15/30  1/1  0s    72ms/step - loss: 1.6304
Epoch 20/30  1/1  0s    202ms/step - loss: 1.4191
Epoch 25/30  1/1  0s    53ms/step - loss: 1.1986
Epoch 30/30  1/1  0s    69ms/step - loss: 0.9988
```

→ loss가 3.45 → 1.00까지 떨어졌지만 0 근처까진 안 감. (Cell 4에서 확인되듯, 모델 자체는 거의 다 맞히고 있고 — 단지 `decoder_target_data` 버그 때문에 "맞혀야 할 정답"이 원래 의도와 다르게 설정된 상태. 남은 loss는 4문장 중 1개(`You are nice`)의 첫 단어 예측이 살짝 불안정한 정도)

> 데이터가 문장 4개뿐이라 batch_size=4면 epoch당 step 1번. Cell 1의 `decoder_target_data` 버그를 고치고 다시 학습하면 결과가 달라질 수 있음 — 고친 다음 재학습 결과 비교해보는 것도 좋은 연습.

### Cell 4 — 추론(디코더 테스트): 입력 문장 재생성

```python
# 디코더 테스트(입력 문장 재생성) 입력문장->정수시퀀스-><sos>추가->디코딩->단어복원
def decode_transformerFunc(sentences):
  # 입력 문장을 정수 시퀀스로 변환 + 패딩
  seq = tokenizer.texts_to_sequences([sentences])
  pad = pad_sequences(seq, maxlen=max_len, padding='post')

  # 디코더 입력 생성: <sos> + 원문을 한 칸 밀어서 채움 (학습 때와 동일한 형식)
  dec_in = np.zeros_like(pad)
  dec_in[0, 0] = sos_token
  dec_in[0, 1:] = pad[0, :-1]

  # 모델 예측 (한 번에 전체 시퀀스 예측 — Teacher Forcing 방식 그대로 사용)
  pred = transformer.predict([pad, dec_in])
  tokens = np.argmax(pred[0], axis=-1)   # 각 타임스텝에서 가장 확률 높은 단어 인덱스 선택
  words = [tokenizer.index_word.get(tok, '') for tok in tokens]   # 인덱스 -> 단어 복원
  return ' '.join(words)

print('테스트 결과 ---')
for s in sentences:
  print(f'{s} => {decode_transformerFunc(s)}')

# 학습 데이터에 없는 문장으로 테스트 (일반화 능력 확인용)
print(decode_transformerFunc('She loves pizza'))
```

**출력**

```
테스트 결과 ---
I have a pen   => have a pen
You are nice   => love nice
He love her    => love her
We eat pizza   => eat pizza

She loves pizza => love
```

**🐛 확정 — Cell 1의 `decoder_target_data` 버그 때문임**

정답이 `I have a pen`인데 예측은 `have a pen`(첫 단어 누락). 이걸 Cell 1에서 계산한 **버그 있는 `decoder_target_data`** 값과 비교하면 정확히 일치함:

|문장|버그 있는 target|실제 예측|
|---|---|---|
|I have a pen|`[have, a, pen, pad]`|`have a pen` ✅|
|He love her|`[love, her, pad, pad]`|`love her` ✅|
|We eat pizza|`[eat, pizza, pad, pad]`|`eat pizza` ✅|
|You are nice|`[are, nice, pad, pad]`|`love nice` (1단어만 다름)|

**실제로 재현 실행해서 검증함**: 동일 코드를 다른 random seed로 재학습하면 `You are nice` 행도 버그 target과 정확히 일치(`are nice`)하게 나옴 — 4문장 전부 버그 있는 target을 정확히 맞춤. 즉 원본 노트북에서 그 한 단어가 빗나간 건 **버그와는 별개로, 그 특정 weight 초기화(랜덤 시드)에서 우연히 덜 수렴된 것**일 뿐 — 버그 진단 자체와는 무관함.

**수정본도 실제로 재학습해서 확인함**: `decoder_target_data = encoder_input_data.copy()`로 고치고 재학습하면

```
I have a pen => I have a pen   ✅
You are nice => You are nice   ✅
He love her  => He love her    ✅
We eat pizza => We eat pizza   ✅
```

정확히 원문 그대로 재생성됨 — 버그 진단과 수정 둘 다 실행으로 확정.

`'She loves pizza'`(학습에 없던 문장) 결과가 `love` 한 단어만 나온 것은, 이 함수가 진짜 자기회귀 생성이 아니라 Teacher Forcing 평가 방식(디코더 입력에 정답을 미리 시프트해서 넣어줌)이라 일반화 테스트로는 신뢰하기 어려움. 진짜 생성 능력을 보려면 한 토큰씩 예측 → 그 예측을 다시 디코더 입력에 넣는 **autoregressive 루프**가 필요함.

---

## 한 줄 요약

|항목|핵심|
|---|---|
|Self-Attention|문장 내부 단어 간 관련성 계산|
|Cross-Attention|디코더가 인코더 출력 중 참고할 부분 결정|
|Causal Mask|디코더가 미래 토큰을 못 보게 막음 (필수)|
|Residual + LayerNorm|서브레이어 입력을 출력에 더해서 정보 손실 방지, 학습 안정화|
|Teacher Forcing|디코더 입력 = 정답을 한 칸 시프트 + `<sos>`, 타겟 = 원문 그대로|
|🐛 발견된 버그|`decoder_target_data`가 원문을 한 번 더 시프트해버림 → 출력이 한 칸씩 밀려서 나옴 (Cell 1 수정 필요)|
|누락된 부분|Positional Encoding (순서 정보 없음), 진짜 autoregressive 추론|

---
# 🗄️ Vector Database 개념 — Embedding · ANN · HNSW · RAG

---

## 📐 벡터(Vector)와 임베딩(Embedding)

벡터는 길이와 방향을 가진 수학적 객체. n차원 벡터 공간에서의 위치를 나타내며, 길이 n인 1차원 숫자 배열로 표현됨.

데이터 과학에서는 데이터의 특성/속성을 나타내는 숫자 배열로 이해 — 복잡한 데이터일수록 더 높은 차원의 벡터로 표현됨.

<img src="images/vector_definition.png" width="500"/>

**임베딩(Embedding)**: 데이터를 모델이 이해할 수 있는 벡터 형태로 변환하는 기법.

- 고차원 데이터를 저차원 벡터로 압축
- 비정형 데이터(텍스트, 이미지, 음성)를 정형 데이터(벡터)로 변환
- 핵심 특징: **원본 데이터의 의미를 보존** → 유사한 데이터들이 벡터 공간에서 서로 근접하게 배치됨
- '유사하다'의 기준은 임베딩 목적에 따라 달라짐 (예: 색감 기준 vs 오브젝트 기준 이미지 임베딩)

<img src="images/embedding_example.png" width="500"/>

---

## 🗃️ 벡터 DB란?

벡터 형태로 데이터를 저장하는 DB. 일반 DB와의 차이:

| |일반 DB|벡터 DB|
|---|---|---|
|인덱싱|특정 값 할당|원본 데이터 + 임베딩 벡터 함께 저장|
|검색 결과|쿼리와 **정확히 일치**|ANN 기반 **근사 최근접** 검색|

ANN(Approximate Nearest Neighbor) + 벡터 인덱스 알고리즘 → 벡터 양이 많아져도 검색 속도 유지 가능. RAG(Retrieval-Augmented Generation) 프레임워크의 필수 요소.

<img src="images/vectordb_storage_structure.png" width="500"/>

벡터 DB는 생성 모델의 **장기 기억(long-term memory)** 역할 → 일관되고 정확한 결과 생성에 기여. 자연어뿐 아니라 정형 데이터, 이미지, 영상, 음성도 벡터화하여 RAG 시스템 구축 가능.

<img src="images/multimodal_embedding_search.png" width="500"/>

---

## ⚙️ 내부 작동 원리

### 일반 파이프라인 (3단계)

1. **Indexing**: PQ, LSH, HNSW 등으로 벡터를 빠른 검색용 자료구조에 매핑
2. **Querying**: 쿼리 벡터 ↔ 인덱스 벡터 비교, 유사도 메트릭 적용
3. **Post-processing**: 최종 후보 재정렬(re-ranking) 등 사후 처리

**핵심 트레이드오프**: 정확도 ↔ 속도 (반비례 관계)

### 벡터 인덱싱 알고리즘 4종

|알고리즘|설명|
|---|---|
|**Random Projection**|무작위 투영 행렬로 고차원 → 저차원 압축, 벡터 간 유사성은 유지|
|**Product Quantization (PQ)**|벡터를 작은 청크로 나눠 각각 대표 코드로 압축 (손실 압축)|
|**LSH**|속도 최적화된 해싱 기반 인덱싱, 근사치/비포괄적 결과|
|**HNSW**|계층적 그래프 구조, 노드 간 엣지 = 벡터 유사도|

### 유사도 측정 3종

- **코사인 유사도**: 두 벡터 간 각도의 코사인. 범위 -1~1 (1=동일, 0=직교, -1=정반대)
- **유클리드 거리**: 두 벡터 간 직선 거리. 범위 0~∞ (0=동일, 클수록 다름)
- **내적(Dot Product)**: 크기×각도 코사인. 범위 -∞~∞ (양수=같은방향, 0=직교, 음수=반대방향)

### 메타데이터 필터링

- **Pre-filtering**: 검색 전 필터링 → 검색공간 축소, but 관련 결과 누락 가능 + 계산 오버헤드
- **Post-filtering**: 검색 후 필터링 → 결과 누락 적음, but 추가 오버헤드로 속도 저하

---

## ⚖️ Vector Index vs Vector DB

FAISS 같은 벡터 인덱스는 검색 성능은 개선하지만 **DB 기능 자체는 없음**.

Vector DB가 제공하는 추가 기능:

- 데이터 관리: 삽입/삭제/갱신 용이
- 메타데이터 저장 및 필터링
- 확장성: 분산/병렬처리, 샤딩, Replication
- 실시간 업데이트, 백업/컬렉션
- 에코시스템 연동: ETL(Spark), 시각화(Grafana), AI 도구(LangChain, LlamaIndex)
- 데이터 보안 및 접근 권한 관리

---

## 🔍 활용 분야

**벡터 데이터베이스가 주로 사용되는 분야**

|분야|사용 예시|
|---|---|
|시맨틱 검색 (Semantic Search)|질문: "빨간 과일 뭐 있지?" → 결과: "사과, 체리" (키워드 없이도 의미 기반 검색)|
|RAG (Retrieval-Augmented Generation)|GPT가 답변 생성 전에 관련 문서를 벡터 DB에서 찾아 사용|
|AI 챗봇 / QA 시스템|사용자의 질문과 유사한 기존 문장을 벡터로 검색해 응답 정확도 향상|
|뉴스/문서 분류 & 중복 감지|유사 기사 클러스터링, 중복 뉴스/문장 자동 탐지|
|이미지/영상 검색|이미지 임베딩 → 유사 이미지 찾기 (핀터레스트, 쇼핑몰, 디지털 자산 검색)|
|음성/음악 검색|"이런 멜로디"에 비슷한 노래 찾기 (Spotify, Shazam 등)|
|추천 시스템|사용자/아이템을 벡터로 표현 → 유사도 기반 추천|
|보안 / 이상 탐지|사용자 행동 벡터 분석 → 이상치 탐지, 사기 의심 패턴 추적|
|생물정보학 / 과학 연구|분자 구조, 단백질 패턴 등을 벡터화하여 검색/비교|
|멀티모달 검색|텍스트로 이미지 검색, 이미지로 텍스트 찾기 (텍스트 ↔ 이미지 ↔ 코드)|

**벡터 DB가 특히 빛나는 상황**

|상황|설명|
|---|---|
|정확한 키워드 없이 검색하고 싶을 때|예: "사과가 들어간 디저트 뭐 있어?" ← 의미 기반 검색|
|수천 개 이상의 문서를 빠르게 찾고 싶을 때|수많은 문서 중에서 GPT에 줄 몇 개만 뽑는 상황|
|동일한 뜻인데 표현이 다른 경우를 잡고 싶을 때|"파이썬 루프 만드는 법" = "python 반복문"|
|GPT + 나만의 지식 연결 (RAG)|FAQ, 내부 문서, 정책 자료 등을 검색해서 답변 생성|

**벡터 DB가 없을 때의 한계**

- 검색: "강아지" 검색 시 "개"는 안 나옴 (키워드 매칭 한계)
- GPT: context 없이 엉뚱한 내용 생성
- 추천: 속성만 보고 판단 → "비슷한 취향" 추천 어려움

---

## 📊 시장 현황

- **전문 벡터 DB**: Pinecone, Chroma, Milvus
- **멀티모델 DB의 벡터 검색 기능**: PostgreSQL pgvector, Elasticsearch

---

## 🐍 Chroma DB — 로컬/Python 친화적 벡터 DB

RAG 시스템에서 많이 쓰는 벡터 DB. 로컬에서 쉽게 쓸 수 있고 Python 친화적이라 실험/개발 단계에서 인기.

**저장 구조 5요소**

|구성 요소|설명|
|---|---|
|Collection|문서의 논리적 그룹 (예: "support-docs")|
|Document|저장하고 싶은 텍스트|
|Embedding|문서를 숫자 벡터로 변환한 값|
|Metadata|부가 정보 (출처, 날짜 등)|
|ID|고유 식별자 (생략 시 자동 생성)|

**저장 방식**

- 인메모리 (Memory only): 빠른 테스트용, 종료 시 사라짐
- 로컬 파일 기반: DuckDB + Parquet, 재시작해도 유지됨

```python
chroma_client = chromadb.Client(chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"  # 파일 저장 디렉터리
))
collection = chroma_client.create_collection(name="my_collection")
```

**`.chroma/` 디렉토리 구조 (v0.4.x 이상)**

<img src="images/chroma_directory_structure.png" width="500"/>

|파일|역할|
|---|---|
|`.chroma/index`|DuckDB 기반 메타데이터 인덱스 (벡터 자체는 없음)|
|`.chroma/chroma.sqlite3`|컬렉션/메타정보용 SQLite 백엔드|
|`<UUID>/data_level0.bin`|임베딩 벡터 값이 실제로 저장되는 바이너리 파일|
|`<UUID>/length.bin`|각 벡터의 길이 정보|
|`<UUID>/link_lists.bin`|벡터 인덱스 간 연결 정보 (HNSW 구조용)|
|`<UUID>/header.bin`|벡터 저장 구조의 헤더 정보|

→ 요약: 벡터는 `.chroma/<collection-id>/data_level0.bin`에 압축된 바이너리 형식으로 저장됨

---

## 🔗 참고 자료

- [Pinecone — What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)
- [Chroma DB 공식 사이트](https://www.trychroma.com/)
- [hashdork — 벡터 데이터베이스](https://hashdork.com/ko/%EB%B2%A1%ED%84%B0-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4/)

### Colab 실습 코드

- [Vector DB 실습 (원본)](https://colab.research.google.com/drive/1iIZBCvA1HIAIToxCbyKZLt_5XG7VjhJs)
- [Pinecone + OpenAI](https://colab.research.google.com/gist/IvanCampos/909cc104444f0ce950b7f991bcc016c6/vector-pinecone-openai.ipynb)
- [Chroma + OpenAI](https://colab.research.google.com/gist/IvanCampos/715908e57fc1c1a6acd374e31f8c8aa9/vector-chroma-openai.ipynb)
- [LangChain + OpenAI](https://colab.research.google.com/gist/IvanCampos/e595371a66d96811116e9a444a63816e/vector-langchain-openai.ipynb)

---
# 📄 vecdb1.ipynb — ChromaDB · Embedding · Vector Search

---

## 개념 정리

### 이 노트북의 전체 흐름

```
설치 → 클라이언트 생성 → 컬렉션 생성 → 텍스트 임베딩 → 저장(add) → 조회(get) → 벡터 검색(query)
```

ChromaDB를 이용해 텍스트 2개를 벡터로 변환해 저장하고, 저장된 데이터를 조회 → 마지막엔 새로운 질문 문장을 벡터화해서 "가장 유사한 문서"를 찾아오는 시맨틱 검색까지 한 번에 실습.

### 핵심 개념 3가지

|개념|설명|
|---|---|
|**`PersistentClient`**|디스크에 영구 저장되는 ChromaDB 클라이언트. (메모리 전용 `Client()`와 달리 재시작해도 데이터 유지)|
|**Embedding Function**|텍스트 → 벡터 변환 함수. `get_or_create_collection`에서 따로 지정 안 하면 ChromaDB **자체 내장 기본 모델**(`all-MiniLM-L6-v2`, 384차원) 사용|
|**`include` 파라미터**|`.get()`/`.query()`에서 결과에 어떤 필드를 포함시킬지 선택. `documents`, `embeddings`, `metadatas`, `distances`, `uris`, `data` 중에서 골라야 함 — **`ids`는 항상 기본 포함이라 옵션에 넣으면 에러**|

### 코사인 유사도 vs ChromaDB의 기본 거리(distance) — 중요한 구분

Cell 2에서 `sklearn.metrics.pairwise.cosine_similarity`로 직접 계산한 유사도는 **코사인 유사도**(1에 가까울수록 유사, 범위 -1~1)인데, Cell 4의 `collection.query()`가 반환하는 `distances`는 **기본적으로 코사인이 아니라 L2(유클리드) 거리 기반**임 (ChromaDB 컬렉션의 기본 `hnsw:space`는 `"l2"`). 그래서:

- `distances` 값은 **작을수록 유사**(코사인 유사도와는 정반대 방향 — 코사인은 클수록 유사)
- 코사인 유사도를 쓰고 싶으면 컬렉션 생성 시 `metadata={"hnsw:space": "cosine"}`을 명시해야 함

이 노트북에서는 둘을 같은 의미로 혼용하기 쉬운데, **"직접 계산한 코사인 유사도"와 "컬렉션 검색이 쓰는 기본 거리"는 서로 다른 척도**라는 걸 구분해서 이해해야 함.

---

## 코드 + 주석

### Cell 0 — 패키지 설치

```python
# Vector database : ChromaDB 사용
!pip install ChromaDB sentence-transformers
```

> 이미 설치돼 있어서 `Requirement already satisfied`만 길게 출력됨 (정상). `sentence-transformers`는 설치는 되지만, 아래 코드에서 실제로 호출되진 않음 — ChromaDB가 자체 내장 임베딩 모델을 쓰기 때문(Cell 2 참고).

### Cell 1 — 모듈 import 및 설치 경로 확인

```python
import chromadb
from chromadb import PersistentClient   # DB 클라이언트 객체(DB 접속 관리자 역할)
# ! pip show chromadb
print(chromadb.__file__)   # __file__: 모듈이 실제로 설치된 경로 (밑줄 2개씩 — dunder 속성)
```

**출력**

```
/usr/local/lib/python3.12/dist-packages/chromadb/__init__.py
```

### Cell 2 — 클라이언트/컬렉션 생성 + 임베딩 + 저장

```python
client = PersistentClient(path=".chroma")   # 디스크에 ".chroma" 폴더로 영구 저장되는 클라이언트
!ls -a

# 컬렉션 생성 (이미 있으면 가져오고, 없으면 새로 생성)
collection = client.get_or_create_collection("test")

print("데이터 벡터화 후 저장 ---")
texts = ["Hello world", "Chroma is cool"]
ids = ["doc1", "doc2"]
metas = [{"source":"greeting"}, {"source":"statement"}]

# 현재 컬렉션에 설정된 임베딩 함수 가져오기
embedding_fn = collection._embedding_function   # 기본 내장 임베딩 모델: all-MiniLM-L6-v2 지원
# ⚠️ _embedding_function은 private(밑줄 시작) 속성 — 공식 API는 아니지만 현재 버전에서는 동작함

embeddings = embedding_fn(texts)   # 텍스트 2개를 벡터로 변환
print(embeddings[0][:5])
print(type(embeddings), len(embeddings), len(embeddings[0]))
print()

for i, vector in enumerate(embeddings):
  print(f'문서 : {texts[i]}')
  print(f'임베딩 값 5개만 : {vector[:5]}')  # 384차원에 float32 벡터로 변환
  print(f'차원수 : {len(vector)}')
  print("-" * 30)

print('참고 : 두 문장 간 유사도 계산')
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f'두 문장 간 코사인 유사도 {sim:.4f}')

# 컬렉션에 저장 (이 줄이 핵심! — 실제로 DB에 영구 저장하는 부분)
collection.add(
    documents=texts,
    embeddings=embeddings,   # 이미 계산해둔 임베딩을 그대로 넘김
    metadatas=metas,
    ids=ids
)
print(collection.count())   # 저장된 문서 개수 확인
```

**출력**

```
.  ..  .chroma  .config  sample_data

데이터 벡터화 후 저장 ---
[-0.03447729  0.03102317  0.00673498  0.02610895 -0.03936207]
<class 'list'> 2 384

문서 : Hello world
임베딩 값 5개만 : [-0.03447729  0.03102317  0.00673498  0.02610895 -0.03936207]
차원수 : 384
------------------------------
문서 : Chroma is cool
임베딩 값 5개만 : [-0.11315749 -0.00511692 -0.01389769 -0.02509309 -0.04774468]
차원수 : 384
------------------------------
참고 : 두 문장 간 유사도 계산
두 문장 간 코사인 유사도 0.0580

2
```

→ 임베딩 차원 **384** = `all-MiniLM-L6-v2` 모델의 고정 출력 차원. `Hello world`와 `Chroma is cool`의 코사인 유사도는 **0.058로 매우 낮음** — 두 문장이 의미적으로 거의 무관하다는 뜻 (둘 다 짧고 평범한 문장이라 당연한 결과). `collection.count()` 결과 **2** → 저장이 정상적으로 됐음을 확인.

### Cell 3 — 저장된 문서 조회

```python
# 저장된 문서 조회
results = collection.get(include=['documents', 'metadatas'])   # 'ids'는 include에 넣으면 에러 — 항상 기본 포함됨
print(results)

for doc, meta, id in zip(results['documents'], results['metadatas'], results['ids']):
  print(f'id : {id}')
  print(f'documents : {doc}')
  print(f'metadatas : {meta}')
  print(f'**' * 30)

print()
print('저장된 문서 id 목록 : ', collection.get()['ids'])   # include 없이 호출해도 ids는 항상 반환됨

results_vec = collection.get(include=['embeddings'])   # 임베딩 벡터까지 같이 조회
first_embedding = results_vec['embeddings'][0]
print('first_embedding : ', first_embedding[:5])
print('임베딩 벡터 차원 수 : ', len(first_embedding))

for id, emb in zip(results_vec['ids'], results_vec['embeddings']):
  print(f'id : {id}')
  print(f'embeddings(앞 5개) : {emb[:5]}')
```

**출력**

```
{'ids': ['doc1', 'doc2'], 'embeddings': None, 'documents': ['Hello world', 'Chroma is cool'], 'uris': None, 'included': ['documents', 'metadatas'], 'data': None, 'metadatas': [{'source': 'greeting'}, {'source': 'statement'}]}
id : doc1
documents : Hello world
metadatas : {'source': 'greeting'}
************************************************************
id : doc2
documents : Chroma is cool
metadatas : {'source': 'statement'}
************************************************************

저장된 문서 id 목록 :  ['doc1', 'doc2']
first_embedding :  [-0.03447729  0.03102317  0.00673498  0.02610895 -0.03936207]
임베딩 벡터 차원 수 :  384
id : doc1
embeddings(앞 5개) : [-0.03447729  0.03102317  0.00673498  0.02610895 -0.03936207]
id : doc2
embeddings(앞 5개) : [-0.11315749 -0.00511692 -0.01389769 -0.02509309 -0.04774468]
```

→ `include=['documents','metadatas']`로 가져온 결과의 `'embeddings'` 필드는 `None`(요청 안 했으니까), `include=['embeddings']`로 다시 조회하니 정상적으로 384차원 벡터가 나옴. Cell 2에서 직접 계산했던 임베딩 값과 **완전히 동일** — 저장된 게 그대로 다시 불러와지는 것 확인.

### Cell 4 — 벡터 기반 시맨틱 검색

```python
# 벡터 기반 검색
# 검색용 질문
query_text = "What's the status of chroma?"

# 검색 문장 임베딩 (저장할 때 썼던 것과 동일한 embedding_fn 사용 — 검색 정확도를 위해 반드시 같은 모델 써야 함)
query_embedding = embedding_fn([query_text])[0]

# Chroma에서 유사 문서 검색
search_result = collection.query(
    query_embeddings=[query_embedding],   # 질문을 벡터화한 것을 그대로 전달
    n_results=2,                          # 가장 유사한 상위 2개 반환
    include = ['documents', 'metadatas', 'distances']
)

for i, (doc, meta, dist) in enumerate(zip(search_result['documents'][0], search_result['metadatas'][0], search_result['distances'][0])):
  print(f'\n결과 {i + 1}')
  print(f' documents {doc}')
  print(f' metadatas {meta}')
  print(f' distances {dist}')   # 기본은 L2 거리 — 값이 작을수록 더 유사함 (코사인 유사도와 반대 방향!)
```

**출력**

```
결과 1
 documents Chroma is cool
 metadatas {'source': 'statement'}
 distances 0.5813290476799011

결과 2
 documents Hello world
 metadatas {'source': 'greeting'}
 distances 1.9882818460464478
```

→ `"What's the status of chroma?"`라는 질문에 대해, `distances`가 더 작은 **`Chroma is cool`(0.58)이 1순위**, `Hello world`(1.99)가 2순위로 나옴. "chroma"라는 키워드가 직접 겹치는 문장이 더 가깝게 나온 것 — 단순 키워드 매칭이 아니라 임베딩 공간에서의 의미적 거리로 판단한 결과이지만, 이 예시에서는 키워드 겹침이 곧 의미적 유사성으로도 이어진 케이스.

> `search_result['documents'][0]`처럼 맨 앞에 `[0]`이 붙는 이유: `collection.query()`는 **여러 개의 쿼리를 동시에** 넣을 수 있게 설계돼서, 결과가 "쿼리별 리스트의 리스트" 구조임. 지금은 쿼리를 1개만 넣었으니 `[0]`이 그 1개 쿼리에 대한 결과.

---

## 한 줄 요약

|항목|핵심|
|---|---|
|`PersistentClient`|디스크에 영구 저장되는 클라이언트|
|기본 임베딩 모델|`all-MiniLM-L6-v2`, 384차원 (sentence-transformers 설치해도 명시적으로 지정 안 하면 안 쓰임)|
|`include` 파라미터|`documents/embeddings/metadatas/distances/uris/data`만 가능, `ids`는 항상 기본 포함|
|`collection.add()`|실제 저장을 일으키는 핵심 호출 (빠뜨리기 쉬움)|
|코사인 유사도 vs `distances`|직접 계산한 코사인 유사도(클수록 유사) ≠ `query()`의 기본 거리(L2, 작을수록 유사) — 서로 다른 척도|
|`collection.query()` 결과 구조|쿼리 여러 개를 한 번에 처리할 수 있어서 결과가 "리스트의 리스트" (`['documents'][0]`처럼 인덱싱 필요)|