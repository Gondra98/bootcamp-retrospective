# Day 04_파이썬 기초 (반복문 & 함수)
## 📅 2026-02-02
---
## 1. 반복문 while

### 1-1. 기본 while 구조

```python
a = 10          # 변수의 초기값
while a >= 5:   # 조건
    print(a, end=' ')
    a -= 1      # 변수 변화
else:
    print('\n수행 성공!')
```

**출력**
```
10 9 8 7 6 5
수행 성공!
```
- while은 **조건이 참(True)** 인 동안 반복
- 조건이 거짓이 되면 `else` 블록 실행 (break 없이 정상 종료 시)

---
### 1-2. 중첩 while

```python
i = 1
while i <= 3:
    j = 1
    while j <= 4:
        print('i=' + str(i) + '/j=' + str(j))
        j += 1
    i += 1
```

- while 안에 또 다른 while 사용 가능
- **중첩 반복문** 구조

---
### 1-3. 누적 합 계산 (조건 포함)
```python
su = 1
hap = 0

while su <= 100:
    if su % 3 == 0:
        hap += su
    su += 1

print(f'1~100 사이의 정수 중 3의 배수의 합은 {hap} 이다.')
```

- 조건 필터링 + 누적 계산
- 데이터 처리 기본 패턴 (ML에서도 자주 등장)

---
### 1-4. 리스트 반복 (while)
```python
colors = ['빨강','파랑','노랑','초록']
print(len(colors))

i = 0
while i < len(colors):
    print(colors[i])
    i += 1
```

- while로 리스트 순회 시 **len() + 인덱스** 조합 필수

---
### 1-5. 별 찍기 (패턴 출력)
```python
print('\n별 찍기------')
i = 1
while i <= 10:
    j = 1
    msg = ''
    while j <= i:
        msg += '*'
        j += 1
    print(msg)
    i += 1
```
- 문자열 누적 + 중첩 반복
- 패턴 문제의 기본 형태

---
## 2. break / continue / 무한 반복

### 2-1. continue와 break
```python
a = 0
while a < 10:
    a += 1
    if a == 3 or a == 5:
        continue   # 아래 문장 무시
    if a == 7:
        break      # 반복문 즉시 탈출
    print(a)
else:
    print('정상종료')

print('while 수행 후 a값은 %d이다.' % a)
```

- `continue` : 다음 반복으로 이동
- `break` : 반복문 즉시 종료
- break 발생 시 `else` 실행 ❌

---
### 2-2. 무한 반복 + 입력 처리
```python
while True:
    key = input('확인할 숫자를 입력하세요. [q:종료]')
    
    if key == 'Q' or key == 'q':
        break
    elif int(key) % 2 == 0:
        print('%d는 짝수입니다.' % int(key))
        continue
    elif int(key) % 2 == 1:
        print('%d는 홀수입니다.' % int(key))
        continue

print('end')
```
- `while True` → 무한 반복
- 종료 조건은 `break`로 제어

---

## 3. 반복문 for

### 3-1. 기본 for 구조

```python
for i in [1,2,3,4,5]:
    print(i, end=' ')
```

- 리스트, 튜플, 집합, 문자열 등 **반복 가능한 객체(iterable)** 사용 가능

---
### 3-2. 분산과 표준편차 계산

```python
numbers1 = [1,3,5,7,9]
numbers2 = [3,4,5,6,7]
numbers3 = [-3,4,5,7,12]

numbers = [numbers1, numbers2, numbers3]

for j in range(3):
    tot = 0
    for a in numbers[j]:
        tot += a
    avg = tot / len(numbers[j])
    print(f"합은 {tot}, 평균은 {avg}")

    tmp = 0
    for i in numbers[j]:
        tmp += (i - avg) ** 2

    print(f"분산은 {tmp/len(numbers[j])}, 표준편차는 {(tmp/len(numbers[j]))**0.5}")
```

- **중첩 for + 통계 계산**
- ML 기초 개념 (분산, 표준편차)

---

### 3-3. iter(), enumerate()

```python
colors = {'r','g','b'}

iterator = iter(colors)
for v in iterator:
    print(v, end=' ')
```

```python
for idx, d in enumerate(colors):
    print(idx, d)
```

- `iter()` : 반복자 생성
- `enumerate()` : 인덱스 + 값 동시에 반환

---

### 3-4. dict 반복

```python
datas = {'python':'만능언어', 'java':'웹용언어', 'mariadb':'db언어'}

for k, v in datas.items():
    print(k, v)

for val in datas.values():
    print(val, end=' ')
```

---

### 3-5. 다중 for (구구단)

```python
for n in range(2,10):
    print(f'---{n}단---')
    for i in range(1,10):
        print(f'{n} * {i} = {n*i}')
```

---
## 4. 정규표현식 + 반복문

### 4-1. 한글 단어 빈도수 계산

```python
import re

text = '''이 최고위원은 이날 국회에서 열린 최고위원회의에서 ...'''

text2 = re.sub(r'[^가-힣\s]', '', text)
words = text2.split()

cou = {}
for w in words:
    if w in cou:
        cou[w] += 1
    else:
        cou[w] = 1

print(cou)
```
- `re.sub()` : 패턴 치환
- 단어 빈도수 → **텍스트 데이터 전처리 기본**

---
### 4-2. 전화번호 패턴 검사
```python
for test_ss in ['111-1234','일이삼-일이삼사','222-1234','333&1234']:
    if re.match(r'^\d{3}-\d{4}$', test_ss):
        print(test_ss, '전화번호 맞아요')
    else:
        print(test_ss, '전화번호 아니에요')
```

---
### 5. Comprehension (한 줄 반복문)
```python
a = range(1,11)
print([i for i in a if i % 2 == 0])
```

```python
datas = [1,2,'a',True,3.0]
li2 = [i*i for i in datas if type(i) == int]
print(li2)
```

```python
id_name = {1:'tom', 2:'lucy'}
name_id = {val:key for key, val in id_name.items()}
print(name_id)
```
- 반복 + 조건 + 값 생성 → **한 줄 표현**
- 가독성 & 생산성 ↑

---
## 6. range 활용
```python
print(list(range(-10, -100, -20)))
print(set(range(1,6,2)))
print(tuple(range(1,6,2)))
```

```python
for _ in range(6):
    print('반복')
```
`_` : 값 사용 안 할 때 관례적 변수

---
## 7. 함수 (Function)
### 7-1. 기본 함수 정의
```python
def doFunc1():
    print('doFunc1 수행')
```

```python
def doFunc3(arg1, arg2):
    return arg1 + arg2
```

- 함수는 **독립된 실행 공간**
- `return` 없으면 `None`

---
### 7-2. 다중 반환 (튜플)

```python
def swapFunc(a, b):
    return b, a

print(swapFunc(10, 20))
```

---
### 7-3. 함수 중첩

```python
def funcTest():
    def funcInner():
        print('내부 함수')
    funcInner()

funcTest()
```

---
### 7-4. 함수 + comprehension

```python
def isOdd(x):
    return x % 2 == 1

mydict = {x:x*x for x in range(11) if isOdd(x)}
print(mydict)
```