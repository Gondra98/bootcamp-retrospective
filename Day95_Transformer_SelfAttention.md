# Day95_Transformer_SelfAttention

## 📅 2026-06-26

---
# 🧠 Transformer 개념 — Self-Attention · Multi-Head Attention · Positional Encoding

## 🎯 왜 Transformer가 필요했나

|기존 Seq2Seq의 문제|Attention의 등장|Transformer의 발상|
|---|---|---|
|인코더 출력을 고정 크기 context vector 1개로 압축 → 문장 길어지면 정보 손실|인코더의 각 시점 hidden state를 디코더가 매 스텝마다 동적으로 참조 (가중합)|"RNN 없이 Attention만으로 인코딩/디코딩이 가능하지 않을까?" → 2017 "Attention is All You Need"(Google)|

> RNN 셀을 완전히 제거하고, 단어 벡터 간 직접적인 가중치 행렬곱(attention)만으로 문맥을 인코딩. 그 결과 **순차 연산(sequential computation)이 사라지고 병렬 처리**가 가능해짐.

---

## ⚙️ 주요 하이퍼파라미터 (논문 기준값)

|이름|값|의미|
|---|---|---|
|`d_model`|512|인코더/디코더 입출력 차원 (임베딩 차원도 동일)|
|`num_layers`|6|인코더/디코더를 쌓는 층 개수|
|`num_heads`|8|병렬로 수행하는 어텐션 헤드 개수|
|`d_ff`|2048|Position-wise FFNN 은닉층 크기|

`d_k = d_v = d_model / num_heads = 512 / 8 = 64`

---

## 📍 Positional Encoding

RNN은 순차 입력 구조 덕분에 위치 정보를 자연스럽게 가졌지만, Transformer는 모든 단어를 한꺼번에 처리하므로 **위치 정보를 별도로 주입**해야 함.

$$PE_{(pos,\ 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$ $$PE_{(pos,\ 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- `pos` : 문장 내 단어의 위치
- `i` : 임베딩 벡터 내 차원 인덱스
- 짝수 차원 → sin, 홀수 차원 → cos
- 임베딩 벡터 + Positional Encoding → 같은 단어도 위치에 따라 입력값이 달라짐

```python
def get_angles(position, i, d_model):
    angles = 1 / (10000 ** ((2 * (i // 2)) / d_model))
    return position * angles

# 짝수 인덱스: sin, 홀수 인덱스: cos 적용 후 결합
```

---

## 🔍 트랜스포머의 세 가지 어텐션

|종류|Query|Key / Value|위치|
|---|---|---|---|
|인코더 셀프 어텐션|입력 문장 자신|입력 문장 자신 (Q=K=V)|인코더|
|디코더 마스크드 셀프 어텐션|출력 문장 자신|출력 문장 자신 (Q=K=V, 미래 단어 마스킹)|디코더|
|인코더-디코더 어텐션|디코더 벡터|인코더 벡터 (Q ≠ K=V)|디코더|

> ⚠️ "셀프"라는 말은 벡터의 **값**이 같다는 게 아니라 벡터의 **출처**가 같다는 뜻.

---

## 🧠 Self-Attention 동작 원리

### 1) Q, K, V 벡터 만들기

각 단어 벡터(`d_model`=512차원)에 서로 다른 가중치 행렬 `W^Q, W^K, W^V`를 곱해서 더 작은 차원(64)의 Q, K, V 벡터 생성.

```
Q = 입력 문장의 모든 단어 벡터들 × W^Q
K = 입력 문장의 모든 단어 벡터들 × W^K
V = 입력 문장의 모든 단어 벡터들 × W^V
```

기존 Seq2Seq Attention과 다른 점: 기존엔 Q(디코더)≠K,V(인코더)였지만, Self-Attention은 **Q=K=V 출처가 전부 같은 문장**.

### 2) Scaled Dot-Product Attention

$$Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- 기존 dot-product attention(`q·k`)에 `√d_k`로 나누는 스케일링 추가
- 논문 기준 `d_k=64` → `√d_k=8`

```python
def scaled_dot_product_attention(query, key, value, mask):
    # 어텐션 스코어 = Q와 K^T의 행렬곱
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # 스케일링: sqrt(d_k)로 나누기
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # 패딩 마스크 적용
    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights
```

**행렬 크기 정리**

|행렬|크기|
|---|---|
|문장 행렬|`(seq_len, d_model)`|
|Q, K 행렬|`(seq_len, d_k)`|
|V 행렬|`(seq_len, d_v)`|
|최종 Attention Value|`(seq_len, d_v)`|

---

## 🧩 Multi-Head Attention

한 번의 큰 어텐션보다 **작은 차원으로 나눠 여러 번 병렬 어텐션**을 하는 게 더 효과적이라는 발상.

- `d_model(512)` → `num_heads(8)`개로 분할 → 각 헤드는 64차원 Q,K,V로 독립적인 어텐션 수행
- 각 헤드마다 `W^Q, W^K, W^V`가 전부 다름 → **서로 다른 관점**에서 단어 간 관계를 포착
    - 예: "그것(it)"이라는 단어를 두고 1번 헤드는 "동물"과의 연관도를, 2번 헤드는 "피곤하다"와의 연관도를 더 높게 볼 수 있음
- 모든 헤드의 결과를 concat → `(seq_len, d_model)` → 가중치 행렬 `W^O`를 곱해 최종 출력

**구현 5단계**

|단계|내용|
|---|---|
|1|Q,K,V용 Dense layer 통과 (`d_model` 크기)|
|2|`num_heads` 개수만큼 split|
|3|Scaled Dot-Product Attention 수행|
|4|헤드들을 concat|
|5|`W^O` Dense layer 통과|

```python
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)  # W^O

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    # call()에서 split → scaled_dot_product_attention → concat → dense(W^O)
```

---

## 🚧 Padding Mask

`<PAD>` 토큰은 의미 없는 단어이므로 어텐션 스코어 계산에서 제외해야 함.

- 마스킹 위치에 매우 작은 음수(`-1e9`)를 더해줌
- softmax를 지나면 해당 위치 값이 거의 0이 되어 attention weight에 반영되지 않음

```python
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]
```

---

## 🔁 Position-wise FFNN (두 번째 서브층)

$$FFNN(x) = \max(0,\ xW_1+b_1)W_2 + b_2$$

- `W_1`: `(d_model, d_ff)`, `W_2`: `(d_ff, d_model)`
- 한 인코더 층 내에서는 모든 단어에 동일한 `W_1, b_1, W_2, b_2`를 적용 (층마다는 다름)

```python
outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
outputs = tf.keras.layers.Dense(units=d_model)(outputs)
```

---

## ➕ Residual Connection & Layer Normalization (Add & Norm)

|개념|수식|의미|
|---|---|---|
|Residual Connection|`H(x) = x + Sublayer(x)`|서브층 입력을 출력에 더해줌 → 학습 안정화 (CV 분야에서 유래)|
|Layer Normalization|`LN = LayerNorm(x + Sublayer(x))`|`d_model` 차원(마지막 차원) 기준으로 평균·분산 정규화|

$$\hat{x}_{i,k} = \frac{x_{i,k}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}} \quad\rightarrow\quad ln_i = \gamma \hat{x}_i + \beta$$

- `γ`(초기값 1), `β`(초기값 0)는 학습 가능한 파라미터

---

## 🏗️ 인코더 한 층 = 2개 서브층 + Add&Norm

```
입력 → [Multi-Head Self-Attention] → Dropout → Add&Norm
     → [Position-wise FFNN]        → Dropout → Add&Norm → 출력
```

- 입력과 출력의 크기(`seq_len, d_model`)는 항상 동일하게 유지 → 인코더를 `num_layers`개 쌓을 수 있는 이유
- 마지막 인코더 층의 출력이 디코더로 전달됨

```python
def encoder_layer(dff, d_model, num_heads, dropout):
    inputs = tf.keras.Input(shape=(None, d_model))
    padding_mask = tf.keras.Input(shape=(1, 1, None))

    attention = MultiHeadAttention(d_model, num_heads)({
        'query': inputs, 'key': inputs, 'value': inputs,  # Q=K=V
        'mask': padding_mask
    })
    attention = tf.keras.layers.Dropout(dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(d_model)(outputs)
    outputs = tf.keras.layers.Dropout(dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs)
```

---

## 📌 오늘의 핵심 키워드

`Self-Attention` · `Q/K/V` · `Scaled Dot-Product Attention` · `Multi-Head Attention` · `Positional Encoding` · `Padding Mask` · `Position-wise FFNN` · `Residual Connection` · `Layer Normalization`

## ➡️ 다음 학습 방향

- 디코더 구조: Masked Self-Attention + 인코더-디코더 Attention 결합 방식
- Look-ahead Mask (디코더가 미래 단어를 보지 못하게 막는 마스킹)
- 인코더 출력이 디코더의 K, V로 어떻게 전달되는지

## 🔗 참고 자료

- [위키독스 16-01 트랜스포머](https://wikidocs.net/31379) — 수식·코드 기반 상세 설명
- [Transformer Explainer (Georgia Tech)](https://poloclub.github.io/transformer-explainer/) — 브라우저에서 GPT-2 실시간 시각화, 어텐션 가중치 직접 확인 가능

---
# 📄 rnn15attention_lstm.ipynb — Bidirectional LSTM · Bahdanau Attention · IMDB 감성분류

## 🎯 개념 정리: 왜 분류 문제에도 Attention을 쓸까

|기존 BiLSTM 단독 분류|Attention 결합|
|---|---|
|마지막 시점의 은닉 상태(state_h)만 분류에 사용 → 중간 시점들의 정보가 손실됨|모든 시점의 은닉 상태를 다시 참고해서, 분류에 중요한 시점에 더 큰 가중치를 부여|

> RNN이 time step을 지나며 잃어버린 정보를 다시 참고하기 위해, **모든 시점의 은닉 상태(hidden states)** 를 한 번 더 들여다보는 것이 Attention의 역할.

### 이전 Day95 Transformer 노트와의 차이점

| |Seq2Seq Attention (번역)|이 노트북 (분류)|
|---|---|---|
|Query|디코더의 **매 시점** 은닉 상태 (계속 바뀜)|BiLSTM의 **마지막** 은닉 상태(state_h) 1개만 사용|
|목적|매 시점마다 다른 단어를 생성|문장 전체에 대해 **단 하나의 context vector**만 생성|
|Attention 호출 횟수|디코더 시점 수만큼 반복|**1회**|

즉, 이번 구조는 "Self-Attention"도 "Seq2Seq Attention"도 아니고, **BiLSTM의 마지막 상태가 자기 자신의 전체 시퀀스를 한 번 돌아보는** 형태에 가깝다.

---

## ⚙️ Bahdanau Attention 수식

$$score = V\big(\tanh(W_1 \cdot values + W_2 \cdot query)\big)$$ $$attention_weights = softmax(score,\ axis=1)$$ $$context_vector = \sum_{t} attention_weights_t \cdot values_t$$

- `W1`, `W2` : `values`와 `query`를 같은 차원(`units`)으로 투영하는 Dense층
- `V` : 투영된 벡터를 스칼라 점수 1개로 압축하는 Dense(1)층 → **덧셈(additive) 방식 어텐션** (Q·K 내적 방식이 아님)
- `key`와 `value`가 동일 → 별도의 K 프로젝션 없이 `values` 자체가 점수 계산과 가중합에 둘 다 사용됨

---

## 🏗️ 전체 모델 구조

```
Input (max_len=500)
  └─ Embedding (vocab_size=10000, dim=128, mask_zero=True)
       └─ Bidirectional LSTM #1 (64유닛, return_sequences=True)   → (batch, 500, 128)
            └─ Bidirectional LSTM #2 (64유닛, return_sequences=True, return_state=True)
                 ├─ lstm (전체 시퀀스 출력)        → (batch, 500, 128)
                 ├─ forward_h, forward_c           → (batch, 64) 각각
                 └─ backward_h, backward_c         → (batch, 64) 각각
                      └─ state_h = concat(forward_h, backward_h)   → (batch, 128)  ← Attention의 Query
                      └─ state_c = concat(forward_c, backward_c)   → (batch, 128)  (이후 미사용)

BahdanauAttention(64)(values=lstm, query=state_h)
  └─ context_vector   → (batch, 128)
  └─ attention_weights → (batch, 500, 1)

context_vector → Dense(20, relu) → Dropout(0.5) → Dense(1, sigmoid) → 이진분류 출력
```

---

## 💻 주석 추가한 전체 코드

```python
# IMDB 리뷰 감성 분류 : LSTM + Attention

# 단방향 LSTM으로 텍스트 분류를 수행할 수도 있지만 때로는 양방향 LSTM을 사용하는 것이 더 강력하다.
# 여기에 추가적으로 어텐션 메커니즘을 사용할 수도 있다.
# 양방향 LSTM과 어텐션 메커니즘으로 IMDB 리뷰 감성 분류하기를 수행

from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
# IMDB는 케라스 내장 데이터셋이라 이미 정수 인코딩된 상태로 제공됨 (단어 → 빈도순 인덱스)

# IMDB 리뷰 데이터는 이미 정수 인코딩이 된 상태므로 남은 전처리는 패딩뿐이다.
# 리뷰의 최대 길이와 평균 길이를 확인
print('리뷰의 최대 길이 : {}'.format(max(len(l) for l in X_train)))          # 2494
print('리뷰의 평균 길이 : {}'.format(sum(map(len, X_train)) / len(X_train)))  # 238.71364

# 데이터 패딩
max_len = 500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
# 훈련용 리뷰와 테스트용 리뷰의 길이가 둘 다 500이 됨.
# 평균 길이(238)보다 넉넉하게 500으로 잡아 정보 손실을 줄임


# 바다나우 어텐션(Bahdanau Attention) -------------------
# 텍스트 분류에서 어텐션 메커니즘을 사용하는 이유는 무엇일까?
# RNN의 마지막 은닉 상태는 예측을 위해 사용된다.
# 그런데 이 RNN의 마지막 은닉 상태는 몇 가지 유용한 정보들을 손실한 상태다.
# 그래서 RNN이 time step을 지나며 손실했던 정보들을 다시 참고
# 이는 다시 말해 RNN의 모든 은닉 상태들을 다시 한 번 참고하겠다는 것이다.
# 그리고 이를 위해서 어텐션 메커니즘을 사용

import tensorflow as tf

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)   # values(인코더 전체 시점)를 투영하는 가중치
        self.W2 = Dense(units)   # query(현재 기준 은닉 상태)를 투영하는 가중치
        self.V = Dense(1)        # 투영된 벡터를 스칼라 score 1개로 압축

    def call(self, values, query):  # 단, key와 value는 같음 → 별도 key 프로젝션 없음
        # query shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # (batch, hidden) → (batch, 1, hidden)으로 만들어
        # values의 시점 축(axis=1, 길이 500)에 브로드캐스팅 덧셈이 가능하게 함

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        # 모든 시점(500개)에 대해 "이 시점이 현재 query와 얼마나 관련 있는가"를 스칼라로 계산

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # 시점(axis=1) 방향으로 정규화 → 500개 시점에 대한 가중치 합 = 1

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # 가중치를 곱한 뒤 시점 축으로 합산 → 시퀀스 전체를 하나의 벡터로 압축

        return context_vector, attention_weights


# 양방향 LSTM + 어텐션 메커니즘(BiLSTM with Attention Mechanism)
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
import os

# 모델 설계 : 여기서는 케라스의 함수형 API를 사용. 우선 입력층과 임베딩층을 설계
sequence_input = Input(shape=(max_len,), dtype='int32')
embedded_sequences = Embedding(vocab_size, 128,
                                input_length=max_len, mask_zero=True)(sequence_input)
# 10,000개의 단어들을 128차원의 벡터로 임베딩하도록 설계하였다.
# mask_zero=True → 패딩(0)을 이후 LSTM 연산에서 무시하도록 마스킹

# 이제 양방향 LSTM을 설계. 단, 여기서는 양방향 LSTM을 두 층을 사용하겠다.
# 우선, 첫번째 층이다. 두번째 층을 위에 쌓을 예정이므로 return_sequences를 True로 해주어야 한다.
lstm = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True))(embedded_sequences)
# 출력: (batch, 500, 128) ← forward 64 + backward 64 가 concat된 결과

# 두번째 층을 설계: 상태를 리턴받아야 하므로 return_state를 True로 해주어야 한다.
lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional \
    (LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)
# lstm        : (batch, 500, 128)  → Attention의 values로 사용될 전체 시점 은닉 상태
# forward_h/c : (batch, 64) 정방향 LSTM의 마지막 은닉/셀 상태
# backward_h/c: (batch, 64) 역방향 LSTM의 마지막 은닉/셀 상태

# 각 상태의 크기(shape)를 출력
print(lstm.shape, forward_h.shape, forward_c.shape, backward_h.shape, backward_c.shape)
# (None, 500, 128) (None, 64) (None, 64) (None, 64) (None, 64)

# 순방향 LSTM의 은닉 상태와 셀상태를 forward_h, forward_c에 저장하고,
# 역방향 LSTM의 은닉 상태와 셀 상태를 backward_h, backward_c에 저장한다.
# 각 은닉 상태나 셀 상태의 경우에는 128차원을 가지는데, lstm의 경우에는 (500 × 128)의 크기를 가진다.
# forward 방향과 backward 방향이 연결된 hidden state 벡터가 모든 시점에 대해서 존재함을 의미한다.
# 양방향 LSTM을 사용할 경우에는 순방향 LSTM과 역방향 LSTM 각각 은닉 상태와 셀 상태를 가지므로,
# 양방향 LSTM의 은닉 상태와 셀 상태를 사용하려면 두 방향의 LSTM의 상태들을 연결(concatenate)해주면 된다.

state_h = Concatenate()([forward_h, backward_h])  # 은닉 상태 → (batch, 128), Attention의 Query로 사용
state_c = Concatenate()([forward_c, backward_c])  # 셀 상태   → (batch, 128), 이 모델에서는 이후 미사용

# 어텐션 메커니즘에서는 은닉 상태를 사용한다. 이를 입력으로 컨텍스트 벡터(context vector)를 얻는다.
attention = BahdanauAttention(64)              # 가중치 크기(units) 정의
context_vector, attention_weights = attention(lstm, state_h)
# values=lstm(전체 500시점), query=state_h(마지막 시점 요약) → 시퀀스 전체를 하나의 context_vector로 압축

# 컨텍스트 벡터를 밀집층(dense layer)에 통과시키고, 이진 분류이므로
# 최종 출력층에 1개의 뉴런을 배치하고, 활성화 함수로 시그모이드 함수를 사용한다.
dense1 = Dense(20, activation="relu")(context_vector)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation="sigmoid")(dropout)
model = Model(inputs=sequence_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=3, batch_size=256,
                     validation_data=(X_test, y_test), verbose=1)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
```

---

## 📊 결과

|Epoch|accuracy|loss|val_accuracy|val_loss|
|---|---|---|---|---|
|1|0.7612|0.4974|0.8771|0.2990|
|2|0.9046|0.2689|0.8865|0.2705|
|3|0.9239|0.2181|0.8859|0.2863|

**최종 테스트 정확도 : 0.8859 (88.59%)**

> ⚠️ 실행 로그에 `BahdanauAttention layer가 mask 정보를 받았지만 지원하지 않아 버린다`는 경고가 있었음 — `mask_zero=True`로 생성된 패딩 마스크가 Attention 층까지는 전달되지 않는다는 의미. 패딩된 0 부분도 어텐션 score 계산에 포함되긴 하지만, 학습이 진행되며 의미 없는 시점의 가중치는 자연스럽게 낮아지는 경향이 있어 성능에 큰 영향은 없었던 것으로 보임 (정확도 88%대로 정상 범위).

## 📌 핵심 키워드

`Bidirectional LSTM` · `Bahdanau Attention` · `Additive Attention` · `return_state` · `Concatenate` · `Context Vector` · `IMDB 감성분류`

---
# 📄 rnn16selfattention.ipynb — Self-Attention · Scaled Dot-Product · Coreference 학습

## 🎯 개념 정리: 이 노트북이 보여주는 것

구글 AI 블로그의 유명한 예시 문장으로 **"it"이 어떤 단어를 가리키는지(coreference resolution)를 Self-Attention이 학습으로 알아낼 수 있는가**를 직접 구현해서 검증.

> "The animal didn't cross the street because it was too tired." → "it"은 문법적으로는 "street"도 될 수 있지만, 의미상 "animal"을 가리킴. 이 의미적 연결을 Self-Attention이 **학습 전엔 모르고, 학습 후엔 알아내는지** 확인하는 실험.

### rnn15(BiLSTM+Bahdanau) 노트와의 구조적 차이

| |rnn15 (Bahdanau Attention)|rnn16 (이 노트)|
|---|---|---|
|점수 계산 방식|덧셈(additive): `V(tanh(W1·values + W2·query))`|내적(multiplicative): `Q·K^T / √d_k` — Transformer 방식|
|Query 출처|BiLSTM의 마지막 상태 (다른 출처)|문장 자기 자신의 한 토큰("it") — **Self**-Attention|
|목적|감성 분류용 context vector 생성|"it"이 어느 단어에 주목하는지 그 자체가 목적(분석)|

### 핵심 트릭: Self-Mask

자기 자신("it")에게 어텐션이 쏠리는 것을 막기 위해, `it_pos` 위치의 score에 `-1e9`를 더해 softmax 후 거의 0이 되게 만듦 (Day95 Transformer 노트의 Padding Mask와 같은 원리, 단 막는 대상이 패딩이 아니라 **자기 자신**).

### 학습 방식: 어텐션을 "분류 문제"처럼 학습

- `target = animal_pos` → "it의 attention이 가장 커야 할 위치는 animal"이라고 정답을 직접 지정
- `SparseCategoricalCrossentropy(from_logits=True)`로, **마치 분류 문제처럼** `query_scores`(softmax 적용 전 raw score)를 학습
- 이건 일반적인 언어모델 학습이 아니라, "어텐션이 원하는 위치를 가리키도록 직접 지도학습 시키면 실제로 가능한가"를 보여주는 **개념 증명(proof of concept)** 실험. 단 한 쌍(it→animal)만 학습하므로 500 epoch만에 loss가 0으로 수렴(과적합에 가까움).

---

## 🏗️ 전체 흐름

```
tokens (12개, "the" 중복 포함) → word_to_id → input_ids (1차원 정수 텐서)
  └─ MySelfAttention(vocab_size, embed_dim=8)
       ├─ Embedding                         → x: (12, 8)
       ├─ Wq, Wk, Wv (Dense, bias 없음)      → Q, K, V: (12, 8)
       ├─ scores = Q·Kᵀ / √8                → (12, 12)
       ├─ query_scores = scores[it_pos]      → (12,)   "it" 기준 한 행만 추출
       ├─ self_mask (it 자신 위치 -1e9)
       ├─ attention_weights = softmax(query_scores + self_mask)
       └─ context_vector = Σ attention_weights · V

학습: target=animal_pos로 500 epoch Adam 최적화
  → "it"의 attention이 "animal"에 100% 쏠리도록 Wq, Wk, Wv, Embedding을 갱신
```

---

## 💻 주석 추가한 전체 코드

### 1) 토큰화 및 인덱싱

```python
# 트랜스포머에 대한 구글 AI 블로그 포스트에서 가져온 예시 문장

import numpy as np
import tensorflow as tf

# "The animal didn't cross the street because it was too tired"
tokens = ["the", "animal", "did", "not", "cross", "the", "street",
          "because", "it", "was", "too", "tired"]

np.set_printoptions(precision=4, suppress=True)

vocab = list(dict.fromkeys(tokens))   # 중복 제거하면서 순서는 유지 ("the"는 한 번만 등록)
print(vocab)
word_to_id = {word: i for i, word in enumerate(vocab)}   # 단어 → 인덱스
print(word_to_id)
id_to_word = {i: word for word, i in word_to_id.items()}  # 인덱스 → 단어 (역방향)
print(id_to_word)

input_ids = tf.constant(             # 문자열을 정수 인덱스 텐서로 변환
    [word_to_id[word] for word in tokens],   # 각 단어를 숫자로 변환 (중복 "the"는 같은 인덱스)
)
# shape (12,) ← tokens 길이 그대로 (vocab 길이 11과는 다름, "the" 중복 때문)

it_pos = tokens.index("it")        # 8
animal_pos = tokens.index("animal")  # 1
print("it 위치 : ", it_pos)
print("animal 위치 : ", animal_pos)
```

**출력 결과**

```text
['the', 'animal', 'did', 'not', 'cross', 'street', 'because', 'it', 'was', 'too', 'tired']
{'the': 0, 'animal': 1, 'did': 2, 'not': 3, 'cross': 4, 'street': 5, 'because': 6, 'it': 7, 'was': 8, 'too': 9, 'tired': 10}
{0: 'the', 1: 'animal', 2: 'did', 3: 'not', 4: 'cross', 5: 'street', 6: 'because', 7: 'it', 8: 'was', 9: 'too', 10: 'tired'}
tf.Tensor([ 0  1  2  3  4  0  5  6  7  8  9 10], shape=(12,), dtype=int32)
it 위치 :  8
animal 위치 :  1
```

### 2) Self-Attention 모델 정의

```python
import math

# Self-Attention 모델
class MySelfAttention(tf.keras.Model):
  def __init__(self, vocab_size, embed_dim):    # 매개변수 : 단어 수와 임베딩차원
    super(MySelfAttention, self).__init__()

    # 단어 번호를 단어 벡터로 바꾸는 임베딩 레이어
    self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)

    self.embed_dim = embed_dim

    # Q, K, V를 만들기 위한 가중치 행렬 (Self-Attention이므로 Q=K=V 출처는 동일)
    self.Wq = tf.keras.layers.Dense(embed_dim, use_bias=False)
    self.Wk = tf.keras.layers.Dense(embed_dim, use_bias=False)
    self.Wv = tf.keras.layers.Dense(embed_dim, use_bias=False)

  def call(self, input_ids, query_pos=None):    # 모델이 호출될 때 자동 실행
    # 단어 인덱스를 임베딩벡터로 변환
    x = self.embedding(input_ids)               # (seq_len, embed_dim) = (12, 8)

    # Q, K, V 벡터 생성 (Q=K=V 출처는 x로 동일, 가중치만 다름)
    Q = self.Wq(x)
    K = self.Wk(x)
    V = self.Wv(x)

    # attention score: Q·K^T를 √d_k로 스케일링 (Scaled Dot-Product Attention)
    scores = tf.matmul(
        Q, K, transpose_b=True   # K를 전치하여 계산 → (12, 12) 전체 단어 간 관계 행렬
    ) / math.sqrt(K.shape[-1])      # score가 너무 커지지 않도록 루트 d_k로 나눔

    # "it" 위치의 Query가 모든 단어 Key를 본 점수만 추출 (12x12 행렬 중 한 행)
    query_scores = scores[query_pos]   # shape (12,)

    # 자기 자신("it") 위치를 마스킹: it이 it 자신에 주목하는 건 의미 없으므로 차단
    self_mask = tf.one_hot(
        query_pos,                      # 마스크할 위치(it 위치)
        depth=tf.shape(input_ids)[0]    # 전체 토큰 갯수 (12)
    ) * -1e9   # 해당 위치 점수를 매우 작은 값으로 만들어 softmax 결과가 거의 0이 되게 함

    query_scores = query_scores + self_mask   # "it" 자기 자신 위치의 score를 작게 만듦
    attention_weights = tf.nn.softmax(query_scores)   # (12,) 합이 1인 분포

    # attention_weight 만큼 Value 가중합 구하기 → "it"을 대표하는 context vector
    context_vector = tf.reduce_sum(attention_weights[:, tf.newaxis] * V, axis=0)

    return query_scores, attention_weights, context_vector, x, Q, K, V
```

### 3) 학습 전 상태 확인

```python
# 모델 생성 및 학습 전 상태 확인
vocab_size = len(vocab)
embed_dim = 8

model = MySelfAttention(vocab_size=vocab_size, embed_dim=embed_dim)

# query_scores, attention_weights, context_vector, x, Q, K, V
_, weights_before, _, embeddings_before, _, _, _ = model(input_ids, query_pos=it_pos)

Wq_before = model.Wq.get_weights()[0].copy()    # 학습 전 Wq 가중치 복사 (이후 비교용)
Wk_before = model.Wk.get_weights()[0].copy()    # 학습 전 Wk 가중치 복사
Wv_before = model.Wv.get_weights()[0].copy()    # 학습 전 Wv 가중치 복사

print('[학습전 임베딩 일부]')
print('animal embedding : ', embeddings_before[animal_pos].numpy())
print('it embedding : ', embeddings_before[it_pos].numpy())

print('[학습전 it의 attention]')
for_print_before = weights_before.numpy()
for token, w in zip(tokens, for_print_before):  # 각 토큰과 weight를 순회
  print(f'{token:>8} -> attention weight : {w:.4f}')
```

**학습 전 출력 결과**

```text
[학습전 임베딩 일부]
animal embedding :  [ 0.0336 -0.0414  0.023   0.0362 -0.0433 -0.0112 -0.0338  0.0495]
it embedding :  [-0.0019 -0.0216 -0.0335 -0.0123  0.0447  0.0487  0.0319  0.0255]
[학습전 it의 attention]
     the -> attention weight : 0.0910
  animal -> attention weight : 0.0909
     did -> attention weight : 0.0910
     not -> attention weight : 0.0909
   cross -> attention weight : 0.0909
     the -> attention weight : 0.0910
  street -> attention weight : 0.0909
 because -> attention weight : 0.0908
      it -> attention weight : 0.0000
     was -> attention weight : 0.0909
     too -> attention weight : 0.0910
   tired -> attention weight : 0.0909
```

가중치가 거의 균등(0.0909~0.0910)하게 분산됨 — 랜덤 초기화 상태라 "it"이 어느 단어에 특별히 더 주목하지 않음 (자기 자신만 마스킹으로 0).

### 4) 학습 (attention이 animal을 가리키도록 지도학습)

```python
# 학습
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
target = tf.constant([animal_pos], dtype=tf.int32)    # 정답 클래스는 animal 위치

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for epoch in range(500):
  with tf.GradientTape() as tape:
    query_scores, attention_weights, context_vector, x, Q, K, V = model(
        input_ids, query_pos=it_pos
    )
    loss = loss_fn(target, query_scores[tf.newaxis, :])   # batch 차원 추가: shape(1, seq_len)으로 만듦

  gradients = tape.gradient(    # loss를 기준으로 학습 가능한 변수들의 기울기 계산
      loss,                          # 줄이고 싶은 손실값
      model.trainable_variables     # embedding, Wq, Wk, Wv
  )

  optimizer.apply_gradients(   # 계산된 기울기를 이용해 가중치를 갱신
      zip(gradients, model.trainable_variables)
  )

  if epoch % 100 == 0:
    print(f'epoch {epoch:3d} | loss:{loss.numpy():.5f}')
```

**학습 로그**

```text
epoch   0 | loss:2.39303
epoch 100 | loss:0.00000
epoch 200 | loss:0.00000
epoch 300 | loss:0.00000
epoch 400 | loss:0.00000
```

epoch 0에서 loss 2.39 → epoch 100부터 loss 0.00000으로 수렴. (예제 1개만 지도학습하는 구조라 매우 빠르게 과적합 수준으로 수렴함 — 일반적인 언어모델 학습과는 다른, 어텐션 메커니즘 자체의 학습 가능성을 보여주는 toy example)

### 5) 학습 후 결과 확인

```python
# 학습 후 결과
query_scores, attention_weights, context_vector, embeddings_after, Q, K, V = model(
  input_ids, query_pos=it_pos
)

print('[학습후 임베딩 일부]')
print('animal embedding : ', embeddings_after[animal_pos].numpy())
print('it embedding : ', embeddings_after[it_pos].numpy())

print('[학습후 it의 attention]')
for token, score, w in zip(tokens, query_scores.numpy(), attention_weights.numpy()):
  print(f'{token:>8} -> score:{score:9.4f}, attention weight:{w:.4f}')

print()
max_index = np.argmax(attention_weights.numpy())
print("[결론]")
print(f"'it'이 가장 많이 참고한 단어는 '{tokens[max_index]}'")
print('it의 context vector ', context_vector.numpy())
```

**출력 결과**

```text
[학습후 임베딩 일부]
animal embedding :  [-0.6065 -0.8545 -0.7936  0.723  -0.8062  0.8074 -0.7542  0.8808]
it embedding :  [ 0.7873 -0.8391 -0.8464  0.8189  0.8427  0.5018  0.7333  0.6538]
[학습후 it의 attention]
     the -> score: -90.5905, attention weight:0.0000
  animal -> score:  67.3249, attention weight:1.0000
     did -> score: -65.4635, attention weight:0.0000
     not -> score: -65.7570, attention weight:0.0000
   cross -> score: -66.0432, attention weight:0.0000
     the -> score: -90.5905, attention weight:0.0000
  street -> score: -65.4482, attention weight:0.0000
 because -> score: -66.4922, attention weight:0.0000
      it -> score:-1000000000.0000, attention weight:0.0000
     was -> score: -66.8059, attention weight:0.0000
     too -> score: -64.9459, attention weight:0.0000
   tired -> score: -67.3951, attention weight:0.0000

[결론]
'it'이 가장 많이 참고한 단어는 'animal'
it의 context vector  [-0.2404 -0.5363  0.1304 -0.7452 -1.2434  0.1117 -0.7771  0.5691]
```

---

## 📊 결과: 학습 전 vs 학습 후

|토큰|학습 전 attention|학습 후 score|학습 후 attention|
|---|---|---|---|
|the|0.0910|-90.59|0.0000|
|**animal**|0.0909|**+67.32**|**1.0000**|
|did|0.0910|-65.46|0.0000|
|not|0.0909|-65.76|0.0000|
|cross|0.0909|-66.04|0.0000|
|street|0.0909|-65.45|0.0000|
|because|0.0908|-66.49|0.0000|
|it (자기 자신, 마스킹)|0.0000|-1e9|0.0000|
|was|0.0909|-66.81|0.0000|
|too|0.0910|-64.95|0.0000|
|tired|0.0909|-67.40|0.0000|

**결론: "it"이 가장 많이 참고한 단어는 'animal'** — score가 +67.32로 압도적으로 높고, attention weight는 1.0000(100%)으로 수렴.

> 학습 전엔 11개 후보 단어에 거의 균등하게(약 9%씩) 분산되어 있던 attention이, "정답은 animal"이라는 단 하나의 지도 신호만으로 500 epoch 만에 완전히 한 곳에 쏠리는 걸 확인. 임베딩 값도 학습 전 -0.04~0.05의 작은 랜덤값에서, 학습 후 -0.85~0.88까지 크게 벌어짐 → "animal"과 "it"의 벡터가 서로 강하게 연관되도록 임베딩 자체도 재구성됨.

## 📌 핵심 키워드

`Self-Attention` · `Q/K/V` · `Scaled Dot-Product Attention` · `Self-Mask` · `Coreference Resolution` · `SparseCategoricalCrossentropy` · `GradientTape`