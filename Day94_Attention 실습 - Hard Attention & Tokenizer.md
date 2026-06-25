# Day94_Attention 실습 - Hard Attention & Tokenizer

## 📅 2026-06-25

---
# 📄 rnn13attention.ipynb — Dot-Product Attention · One-hot Q(Hard Attention) · Softmax 가중치 분산

---

## 개념 정리

이 노트북은 `rnn12attention.ipynb`에서 만든 Dot-Product Attention 함수를 그대로 가져와, **"I love you → 나는 너를 사랑해"** 번역 상황에 적용해본 예제다. 핵심은 Q를 학습값이 아니라 **사람이 직접 one-hot으로 지정**했을 때, Attention이 실제로 어떤 weight를 만들어내는지 확인하는 데 있다.

### Q·K·V 역할 정리 (소스 토큰 3개 기준)

|개념|의미|이 노트북에서의 값|
|---|---|---|
|Key(K), Value(V)|입력 단어(`I`, `love`, `you`)의 표현|`np.eye(3)` — 단위행렬로 단순화 (실제로는 임베딩 벡터)|
|Query(Q)|출력 단어가 "어디에 집중할지" 미리 지정한 기준 벡터|출력 단어마다 한 곳만 1.0 (직접 하드코딩 = Hard Attention)|
|Output(생성 단어)|실제 번역 결과|`tgt_tokens`를 그대로 사용 (Attention 결과와는 무관)|

### Q를 직접 지정한 매핑

|출력 단어 (i)|집중 대상 (Q index)|대응 입력 토큰|
|---|---|---|
|나는 (i=0)|`Q[0,0]=1.0`|`I`|
|너를 (i=1)|`Q[1,2]=1.0`|`you`|
|사랑해 (i=2)|`Q[2,1]=1.0`|`love`|

### 왜 weight가 1.0/0.0이 아니라 0.471/0.264로 나올까?

Q를 정확히 한 곳에만 1.0으로 줘도, 최종 weight가 완전히 0/1로 갈리지 않는 이유는 **softmax의 특성** 때문이다.

```
score = Q · Kᵗ / √d_k
      = [1,0,0] · [[1,0,0],[0,1,0],[0,0,1]] / √3
      ≈ [0.577, 0, 0]

softmax([0.577, 0, 0])
 = exp(0.577) / (exp(0.577) + exp(0) + exp(0))
 ≈ 1.781 / 3.781 ≈ 0.471      (나머지 두 자리는 각각 1/3.781 ≈ 0.264)
```

score 차이가 크지 않으면 softmax는 한쪽에 weight를 "몰아주되" 완전히 0으로 만들진 않는다. 실제로 학습된 Attention에서도 마찬가지라, 모델이 특정 토큰에 강하게 집중해도 다른 토큰의 정보가 항상 조금씩은 섞여 들어간다.

> **주의할 점:** 이 노트북은 `generated.append(tgt)`로 **정답 토큰(tgt_tokens)을 그대로** 결과에 넣는다. 즉 Attention이 계산한 `weights`/`context`는 화면 출력용일 뿐, 실제 번역 단어를 고르는 데는 쓰이지 않는다. (`rnn12attention.ipynb`에서는 `np.argmax(weight)`로 weight 결과를 실제 단어 선택에 사용했던 것과 차이가 있는 부분.) 실제 Seq2Seq + Attention 모델이라면 이 context 벡터가 디코더의 다음 단어 예측에 직접 들어간다.

---

## 코드 + 주석 정리

```python
# 번역 Dot-Product Attention 모델 작성
# "I love you" -> "나는 너를 사랑해"

import numpy as np
import math

src_tokens = ["I", "love", "you"]      # 인코더 입력 토큰
tgt_tokens = ["나는", "너를", "사랑해"]    # 디코더가 생성해야 할 정답 토큰

n_src = len(src_tokens)
K = np.eye(n_src)   # Key: 입력 토큰을 one-hot으로 단순화 (실제로는 임베딩 벡터)
V = np.eye(n_src)   # Value: Key와 동일하게 단순화

n_tgt = len(tgt_tokens)
Q = np.zeros((n_tgt, n_src))   # Query: 출력 단어 개수만큼 한 행씩 생성

# 출력 단어(tgt_tokens) 하나하나에, 어느 입력 단어(src_tokens)에 집중할지를 직접 설정
for i in range(n_tgt):
  if i == 0:        # 출력 첫단어(i==0)는 입력 첫단어(src[0])에 100% 집중
    Q[i, 0] = 1.0
  elif i == 1:      # "너를" → you (마지막 입력 단어)
    Q[i, 2] = 1.0
  elif i == 2:      # "사랑해" → love
    Q[i, 1] = 1.0

print(Q)

def scaled_atten_func(q, K, V):
  scores = q.dot(K.T) / math.sqrt(K.shape[1])   # 유사도 점수 계산 / 스케일링 (Scaled Dot-Product)
  exp = np.exp(scores - np.max(scores))         # softmax 분자 (오버플로 방지용 max값 빼기)
  weights = exp / exp.sum()                     # 가중치 합이 1이 되도록 정규화 (Attention weight)
  context = (weights[:, None] * V).sum(axis=0)  # weight로 V를 가중합 → 동적 컨텍스트 벡터
  return context, weights   # context는 이 노트북에서는 출력만 하고 실제로 쓰이진 않음

# 각 목표 단어마다 attention 실행 후 결과 확인
generated = []    # 생성된 단어 저장용
for i, tgt in enumerate(tgt_tokens):
  context, weights = scaled_atten_func(Q[i], K, V)   # i번째 출력 위치의 Q로 attention 계산
  print(f'생성 단어 : {tgt}')
  print('입력 단어별 어텐션 가중치:')
  for src, w in zip(src_tokens, weights):
    print(f'{src:>7} : {w:.3f}')
  generated.append(tgt)   # ※ attention 결과가 아니라 정답 토큰을 그대로 추가

print('최종 번역 결과')
print(' '.join(generated))
```

---

## 실행 결과

```
[[1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]]
생성 단어 : 나는
입력 단어별 어텐션 가중치:
      I : 0.471
   love : 0.264
    you : 0.264
생성 단어 : 너를
입력 단어별 어텐션 가중치:
      I : 0.264
   love : 0.264
    you : 0.471
생성 단어 : 사랑해
입력 단어별 어텐션 가중치:
      I : 0.264
   love : 0.471
    you : 0.264
최종 번역 결과
나는 너를 사랑해
```

---

## 📌 핵심 정리

- Q를 직접 one-hot으로 지정해도, softmax 특성상 weight는 완전히 0/1로 갈리지 않고 **(0.471, 0.264, 0.264)처럼 부드럽게 분산**된다.
- 이 노트북의 `context`, `weights`는 **시각화 목적**으로만 출력되며, 실제 번역 단어 선택(`generated.append(tgt)`)에는 사용되지 않는다 — `rnn12attention.ipynb`의 `np.argmax(weight)` 방식과 대비되는 부분.
- `K = V = np.eye(n_src)`로 단순화했기 때문에, context 벡터도 결국 "각 입력 토큰에 대한 attention 비율"과 동일한 값이 된다.
- 실제 Transformer/Seq2Seq에서는 Q, K, V가 학습된 임베딩에서 나오고, context 벡터가 디코더의 다음 단어 예측에 직접 입력으로 들어간다.

#NLP #Attention #DotProductAttention

---
# 📄 rnn14attention.ipynb — Tokenizer 기반 Encoder-Decoder · Scaled Dot-Product Attention · 디코더 출력 복원

---

## 개념 정리

`rnn13attention.ipynb`에서는 `src_tokens`/`tgt_tokens`를 그냥 리스트로 직접 적었지만, 이 노트북은 실제 문장(`input_text`, `output_text`)을 **Keras `Tokenizer` + `pad_sequences`로 전처리**한 뒤 동일한 Scaled Dot-Product Attention을 적용한다. 추가로 출력 시퀀스를 다시 자연어로 복원하는 **디코딩 단계**까지 포함되어 있어, "토큰화 → 인코딩 → Attention → 디코딩"의 전체 흐름을 한 번에 볼 수 있는 노트북이다.

### 전체 흐름 정리

|단계|코드 영역|설명|
|---|---|---|
|토큰화|`Tokenizer`|문장을 단어 단위로 쪼개 `word_index` 사전 생성|
|시퀀스 변환|`texts_to_sequences`|단어 → 정수 id로 변환|
|패딩|`pad_sequences`|가변 길이 시퀀스를 고정 길이로 통일 (`padding='post'`)|
|K, V 구성|`np.eye(n_src)`|입력 토큰 표현을 one-hot으로 단순화 (실제로는 임베딩 벡터)|
|Q 구성 (디코더)|수동 지정|출력 위치별로 어디에 집중할지 직접 정함|
|Attention 계산|`attentionFunc`|scaled dot-product → softmax → weighted sum|
|디코딩|`reverse_output_index`|정수 id → 단어, padding/`<sos>`/`<eos>` 제거|

### Q 행렬이 만들어내는 attention 패턴

```
n_src=4 (I, have, a, pen),  n_tgt=6 (<sos>, 나는, 펜을, 가지고, 있다, <eos>)

[[1.  0.  0.  0. ]   ← 출력0 : 입력 0번(I)에 100%
 [0.  0.  0.  0.5]   ← 출력1 : 입력 3번(pen)에 50%만
 [0.  0.  0.  0.5]   ← 출력2 : 입력 3번(pen)에 50%만
 [0.  0.  0.  1. ]   ← 출력3 : 입력 3번(pen)에 100%
 [0.  0.  0.  1. ]   ← 출력4 : 입력 3번(pen)에 100%
 [0.  0.  0.  1. ]]  ← 출력5 : 입력 3번(pen)에 100%
```

규칙을 풀어보면:

- `i == 0` → 출력 첫 위치는 입력 첫 토큰(`I`)에 100% 집중
- `i == n_tgt-1` (마지막 출력) → 입력 마지막 토큰(`pen`)에 100% 집중
- `i < n_src-1` (즉 i=1,2) → 입력 마지막 토큰(`pen`)에 50%만 집중
- 그 외(i=3,4) → 입력 마지막 토큰(`pen`)에 100% 집중

> 이 규칙대로면 출력 중간 단어들(`펜을`, `가지고`, `있다`)이 전부 입력의 `pen` 토큰에만 (부분적 또는 전체로) 집중하고, `have`·`a` 같은 중간 입력 토큰에는 전혀 집중하지 않는다. 실제 단어 의미 대응(`가지고`→`have`, `있다`→`a`)과는 다른 패턴인데, 코드 주석에 "개념 이해용 예제이므로 Q를 직접 지정"이라고 적혀 있는 것처럼, 정확한 단어 매핑보다는 **"이런 Q를 주면 attention weight가 이렇게 분산된다"**를 보여주는 데 목적이 있는 예제다.

### Score 계산 예시 (Output 위치 1 기준)

`n_src=4`라서 스케일링 값은 `√4 = 2`. `K=np.eye(4)`이므로 `Q·Kᵗ`는 그냥 `Q`값 자체가 된다.

```
Q[1] = [0, 0, 0, 0.5]
scores = Q[1] / 2 = [0, 0, 0, 0.25]

softmax([0, 0, 0, 0.25])
 = exp([0,0,0,0.25]) / sum
 ≈ [1, 1, 1, 1.284] / 4.284
 ≈ [0.233, 0.233, 0.233, 0.300]
```

→ 실행 결과의 `[Output 위치 1]` 값(I:0.233, have:0.233, a:0.233, pen:0.300)과 정확히 일치한다.

---

## 코드 + 주석 정리

```python
# Scaled Dot-Product Attention (Tokenizer + Attention)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
import numpy as np

# 입출력 문장
input_text = "I have a pen"
output_text = "<sos> 나는 펜을 가지고 있다 <eos>"
```

```python
# 인코더

# 토큰 처리
input_tokenizer = Tokenizer(filters='', lower=False)   # 입력(영어) 문장용 토크나이저
output_tokenizer = Tokenizer(filters='', lower=False)  # 출력(한국어) 문장용 토크나이저

input_tokenizer.fit_on_texts([input_text])              # 입력 문장으로 vocab 생성
print('input_tokenizer : ', input_tokenizer.word_index)
print('input_tokenizer : ', input_tokenizer.word_counts)
output_tokenizer.fit_on_texts([output_text])            # 출력 문장으로 vocab 생성
print('output_tokenizer : ', output_tokenizer.word_index)
print('output_tokenizer : ', output_tokenizer.word_counts)

# 문장을 숫자 시퀀스화 (단어 -> 정수 id)
input_seq = input_tokenizer.texts_to_sequences([input_text])
print('input_seq : ', input_seq)
output_seq = output_tokenizer.texts_to_sequences([output_text])
print('output_seq : ', output_seq)

# 시퀀스 패딩 (여기서는 문장이 1개씩이라 max길이 = 그 문장 길이 그대로)
max_input_len = max(len(s) for s in input_seq)
max_output_len = max(len(s) for s in output_seq)
print(max_input_len, max_output_len)
input_pad = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
output_pad = pad_sequences(output_seq, maxlen=max_output_len, padding='post')

# Q, K, V 구성
n_src = input_pad.shape[1]   # 입력 길이 (4)
n_tgt = output_pad.shape[1]  # 출력 길이 (6)

K = np.eye(n_src)   # Key: 입력 토큰을 one-hot으로 단순화
V = np.eye(n_src)   # Value: Key와 동일하게 단순화
```

```python
# 디코더
# 개념 이해용 예제이므로 Q를 어디에 집중할지 직접 지정
Q = np.zeros((n_tgt, n_src))

for i in range(n_tgt):
  if i == 0:
    Q[i, 0] = 1.0          # 출력 첫 위치 -> 입력 첫 토큰(I)에 100% 집중
  elif i == n_tgt - 1:
    Q[i, -1] = 1.0         # 출력 마지막 위치 -> 입력 마지막 토큰(pen)에 100% 집중
  elif i < n_src - 1:
    Q[i, -1] = 0.5         # 출력 중간 일부 -> 입력 마지막 토큰(pen)에 50%만 집중
  else:
    Q[i, -1] = 1.0         # 나머지 출력 위치 -> 입력 마지막 토큰(pen)에 100% 집중
print(Q)

def attentionFunc(q, K, V):
  scores = q.dot(K.T) / math.sqrt(K.shape[1])  # 유사도 점수 / 스케일링 (Scaled Dot-Product)
  exp = np.exp(scores - np.max(scores))        # softmax 분자 (오버플로 방지)
  weights = exp / exp.sum()   # 실제로 디코더의 각 입력 단어에 대해 얼마나 주목했는지를 나타냄
  context = (weights[:, None] * V).sum(axis=0) # weight로 V를 가중합 → context 벡터
  return context, weights
```

```python
# Attention 실행 (디코더가 한 스텝씩 생성하는 과정)
print('Attention 결과 : \n')
for i in range(n_tgt):
  context, weights = attentionFunc(Q[i], K, V)   # i번째 출력 위치의 Q로 attention 계산
  print(f'[Output 위치 {i}]')
  for src_word, w in zip(input_tokenizer.word_index.keys(), weights):
    print(f' - {src_word:>5} -> Attention:{w:.3f}')

# 최종 출력 시퀀스
print('입력 시퀀스 : ', input_pad)
print('출력 시퀀스 : ', output_pad)
```

```python
# 디코더 출력 복원 (불필요한 토큰 제거 : 0, <sos>, <eos>)
reverse_output_index = { v:k for k, v in output_tokenizer.word_index.items()}  # id -> 단어 역매핑
print(reverse_output_index)   # {1: '<sos>', 2: '나는', 3: '펜을', 4: '가지고', 5: '있다', 6: '<eos>'}
reconstructed = []

for idx in output_pad[0]:
    if idx == 0:
        continue                                # 패딩(0) 무시
    word = reverse_output_index[idx]
    if word in ('<sos>', '<eos>'):
        continue                                # sos/eos 토큰 무시
    reconstructed.append(word)

print("복원 결과 : \n")
print(" ".join(reconstructed))
```

---

## 실행 결과

```
input_tokenizer :  {'I': 1, 'have': 2, 'a': 3, 'pen': 4}
output_tokenizer :  {'<sos>': 1, '나는': 2, '펜을': 3, '가지고': 4, '있다': 5, '<eos>': 6}
input_seq :  [[1, 2, 3, 4]]
output_seq :  [[1, 2, 3, 4, 5, 6]]

Attention 결과 :

[Output 위치 0]
 -     I -> Attention:0.355
 -  have -> Attention:0.215
 -     a -> Attention:0.215
 -   pen -> Attention:0.215
[Output 위치 1]
 -     I -> Attention:0.233
 -  have -> Attention:0.233
 -     a -> Attention:0.233
 -   pen -> Attention:0.300
[Output 위치 2]
 -     I -> Attention:0.233
 -  have -> Attention:0.233
 -     a -> Attention:0.233
 -   pen -> Attention:0.300
[Output 위치 3]
 -     I -> Attention:0.215
 -  have -> Attention:0.215
 -     a -> Attention:0.215
 -   pen -> Attention:0.355
[Output 위치 4]
 -     I -> Attention:0.215
 -  have -> Attention:0.215
 -     a -> Attention:0.215
 -   pen -> Attention:0.355
[Output 위치 5]
 -     I -> Attention:0.215
 -  have -> Attention:0.215
 -     a -> Attention:0.215
 -   pen -> Attention:0.355

입력 시퀀스 :  [[1 2 3 4]]
출력 시퀀스 :  [[1 2 3 4 5 6]]

복원 결과 :

나는 펜을 가지고 있다
```

---

## 📌 핵심 정리

- `rnn13attention.ipynb`와 같은 attention 계산 로직을 쓰지만, 이번엔 실제 문장을 **Tokenizer → pad_sequences**로 전처리하는 단계가 추가됐다.
- Q를 수동으로 지정한 규칙 때문에, 출력 중간 단어들이 모두 `pen`(입력 마지막 토큰)에만 집중하는 패턴이 나온다 — 의미적으로 정확한 단어 매핑은 아니지만, "Q 설계에 따라 weight 분산이 어떻게 달라지는지" 보여주는 게 목적인 예제.
- 마지막 셀에서 `output_pad`의 정수 id를 다시 단어로 복원(`reverse_output_index`)하면서, padding(0)·`<sos>`·`<eos>` 토큰을 걸러내는 디코딩 로직까지 완성했다.
- 이 노트북으로 "문장 → 토큰화 → 패딩 → Attention → 문장 복원"의 한 사이클이 전부 동작하는 최소 단위 파이프라인이 완성된 셈이다.

#NLP #Attention #Tokenizer #Seq2Seq