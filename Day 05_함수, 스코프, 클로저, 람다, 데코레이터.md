# Day 05_함수, 스코프, 클로저, 람다, 데코레이터

# 📅 2026-02-03
1. 공모전 참여하기 (하는거랑 안하는거랑 차원이 다름)
2. https://cafe.daum.net/flowlife/F6Jx/339

---

## 1. 스코프(Scope)와 변수

```python
# 모듈(파일) 단위 변수
a, b, c = 10, 20, 30

def Foo():
    a = 7      # 지역 변수
    b = 100

    def Bar():
        global c     # 모듈 단위 전역 변수 c 참조
        nonlocal b   # Foo 함수의 b를 참조
        b = 8
        print(f'Bar 수행 후 a:{a}, b:{b}, c:{c}')
        c = 9
        b = 200

    Bar()
    print(f'Foo 수행 후 a:{a}, b:{b}, c:{c}')

Foo()
print(f'모듈 단위 a:{a}, b:{b}, c:{c}')

```


**출력**

```
Bar 수행 후 a:7, b:8, c:30
Foo 수행 후 a:7, b:200, c:9
모듈 단위 a:10, b:20, c:9
```

- `global` → 모듈(파일) 단위 변수 참조/수정
- `nonlocal` → **바깥 함수의 지역 변수** 참조
- 지역 변수 → 함수 내에서만 유효

---

## 2. 매개변수 유형

### 2-1. 위치/기본/키워드 매개변수

```python
def showGugu(start, end=5):        # end 기본값 5
    for dan in range(start, end+1):
        print(f'{dan}단 출력')
        for i in range(1, 10):
            print(f'{dan}*{i}={dan*i}', end=' ')
        print()

showGugu(2, 3)       # 위치 인수
showGugu(2)          # 기본값 사용
showGugu(start=7, end=9)  # 키워드 인수
showGugu(end=9, start=7)  # 순서 바꿔도 키워드 가능
```

---

### 2-2. 가변 인수 `*args` / `**kwargs`

```python
# 튜플로 받는 가변 인수
def func1(*args):
    print(args)
    for i in args:
        print('밥:', i)

func1('김밥','비빔밥','볶음밥')

# 튜플 + 일반 인수
def func2(a, *args):
    print(a)
    print(args)

func2('김밥','비빔밥','볶음밥','공기밥')

# 딕셔너리로 받는 가변 키워드 인수
def func3(w, h, **other):
    print(f'몸무게: {w}, 키: {h}')
    print(f'기타: {other}')

func3(80, 180, irum='신기루', nai=23)

# 혼합: 일반 + *튜플 + **딕트
def func4(a, b, *c, **d):
    print(a, b)
    print(c)
    print(d)

func4(1, 2)
# a=1, b=2  
# c=()   -> *c에 전달된 값 없음, 빈 튜플  
# d={}   -> **d에 전달된 값 없음, 빈 딕셔너리

func4(1, 2, 3, 4, 5)
# a=1, b=2  
# c=(3, 4, 5)  -> *c에 3,4,5 전달  
# d={}         -> **d에 전달된 값 없음, 빈 딕셔너리

func4(1, 2, 3, 4, 5, kbs=9, mbc=11)
# a=1, b=2  
# c=(3, 4, 5)          -> *c에 3,4,5 전달  
# d={'kbs':9,'mbc':11} -> **d에 kbs, mbc 키워드 인수 전달
```

**출력**
```
('김밥', '비빔밥', '볶음밥')
밥: 김밥
밥: 비빔밥
밥: 볶음밥

김밥
('비빔밥', '볶음밥', '공기밥')

몸무게: 80, 키: 180
기타: {'irum': '신기루', 'nai': 23}

1 2
(3, 4, 5)
{'kbs': 9, 'mbc': 11}
```

- `*args` → 튜플 형태로 여러 인수 받음
- `**kwargs` → 딕셔너리 형태로 키워드 인수 받음
- 일반 인수 + `*args` → 일반 인수가 먼저, 나머지 인수는 튜플로
- 일반 + `*args` + `**kwargs` → 순서 중요: 일반 → 튜플 → 딕셔너리

---
## 2-3. 타입 힌트(Type Hint)

```python
# typeFunc 함수 정의
# num: int -> 정수값을 받을 거라는 힌트
# data: list[str] -> 문자열 리스트를 받을 거라는 힌트
# -> dict[str, int] -> 반환값은 {문자열: 정수} 형태라는 힌트
def typeFunc(num: int, data: list[str]) -> dict[str, int]:
    print(num)   # 전달받은 num 출력
    print(data)  # 전달받은 리스트 출력
    result = {}
    # enumerate로 index와 값 가져오기, start=1 → 1부터 시작
    for idx, item in enumerate(data, start=1):
        print(f'idx:{idx}, item:{item}')
        result[item] = idx  # 딕셔너리에 {값:순서} 저장
    return result

# 정상 타입 입력
rdata = typeFunc(1, ['일','이','삼'])
print(rdata)
# 출력:
# 1
# ['일', '이', '삼']
# idx:1, item:일
# idx:2, item:이
# idx:3, item:삼
# {'일': 1, '이': 2, '삼': 3}

# 잘못된 타입 입력도 오류 발생하지 않음 (Type Hint는 가독성용)
rdata = typeFunc("한개", [10,20,30])
print(rdata)
# 출력:
# 한개
# [10, 20, 30]
# idx:1, item:10
# idx:2, item:20
# idx:3, item:30
# {10: 1, 20: 2, 30: 3}

```

- `num: int` → num은 int형
- `data: list[str]` → data는 문자열 리스트
- `-> dict[str,int]` → 반환값은 key:str, value:int 딕셔너리
- **실행 시 타입 체크 X**, 가독성 향상용

---
## 3. 함수 객체와 일급 함수

```python
# 함수는 객체, 변수에 저장 가능
def funcTimes(a, b):
    return a * b

kbs = funcTimes      # 함수 객체 저장
print(kbs(2, 3))    # 6

# 함수 삭제 후에도 객체로 호출 가능
del funcTimes
print(kbs(4, 5))    # 20
```

- **일급 함수 특징**
    
    1. 변수에 저장 가능
        
    2. 인자로 전달 가능
        
    3. 반환값으로 반환 가능

---

## 4. 클로저(Closure)

```python
# 외부 함수
def outer():
    count = 0  # 외부 함수의 지역 변수
    # 내부 함수
    def inner():
        nonlocal count  # 외부 변수 count 사용
        count += 1
        return count
    return inner  # inner 함수 자체를 반환 → 클로저 생성

# var1이라는 클로저 생성
var1 = outer()  

print(var1())  # 1 → count=1
print(var1())  # 2 → count=2, 이전 호출 때 값 기억

# var2라는 새로운 클로저 생성
var2 = outer()  

print(var2())  # 1 → var1과는 별개, 새로운 count=0부터 시작
print(var2())  # 2 → var2 내부 count 값 증가
print(var1())  # 3 → var1의 count 값 계속 유지
```

**출력**
```
1
2
1
2
3
```

- 내부 함수가 **외부 함수의 지역 변수**를 기억함
- 서로 다른 클로저는 **서로 영향을 주지 않음**
- 외부에서 직접 못 보는 **숨겨진 변수**를 가질 수 있음

---
### 4-1. 클로저 활용 예제

```python
def outer2(tax):
    def inner2(su, dan):
        return su * dan * tax
    return inner2

q1 = outer2(0.1)   # 1분기 세금 10%
print(q1(5, 50000))  # 25000.0
print(q1(2, 10000))  # 2000.0

q2 = outer2(0.05)  # 2분기 세금 5%
print(q2(5, 50000))  # 12500.0
```

---

## 5. 람다(lambda) 함수

```python
# 일반 함수
def hapFunc(x, y):
    return x + y
print(hapFunc(1, 2))  # 3

# 람다 함수
gg = lambda x, y: x + y
print(gg(1, 2))       # 3

# 기본값, 가변 인수 가능
kbs = lambda a, su=10: a + su
print(kbs(5))          # 15
print(kbs(5, 6))       # 11

sbs = lambda a, *tu, **di: print(a, tu, di)
sbs(1, 2, 3, var1=4, var2=5) 
# 1 (2, 3) {'var1': 4, 'var2': 5}
```

- 람다: **이름 없는 한 줄 함수**, 반환식만 사용
- filter/map/sort 등 함수형 프로그래밍에 활용

```python
# filter 예제: 1~100 사이 5 또는 7 배수
print(list(filter(lambda a: a%5==0 or a%7==0, range(1,101))))
```

---
## 6. 함수 장식자(Decorator)

```python
# 장식자 기본 예제
def make2(fn):
    return lambda: "안녕 " + fn()

def make1(fn):
    return lambda: "반가워 " + fn()

def hello():
    return "홍길동"

hi = make2(make1(hello))
print(hi())  # 안녕 반가워 홍길동
```

---
### 6-1. @표현식 사용

```python
@make2
@make1
def hello2():
    return "신기해"

print(hello2())  # 안녕 반가워 신기해
```
- `@데코레이터` → 함수를 감싸는 기능
- 실행 순서: **아래부터 위로 감싸짐**
	1. `hello2` → `make1(hello2)`
	2. `make1(hello2)` 결과 → `make2(...)`
	3. 최종 `hello2()` 호출 → `"안녕 반가워 신기해 신기해"`
- 함수 앞에 `@장식자` → 원래 함수 수정 없이 기능 추가

---
### 6-2. 매개변수 있는 함수 장식자

```python
def traceFunc(func):
    def wrapperFunc(a, b):
        r = func(a, b)
        print(f'함수명:{func.__name__} (a={a}, b={b} -> {r})')
        return r
    return wrapperFunc

@traceFunc
def addFunc(a, b):
    return a + b

print(addFunc(10, 20))
```

**출력**

```
함수명:addFunc (a=10, b=20 -> 30)
30
```

- 함수 호출 전/후로 코드 추가 가능
- 메타 정보(`func.__name__`) 확인 가능
