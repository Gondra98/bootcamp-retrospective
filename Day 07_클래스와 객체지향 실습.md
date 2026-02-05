# Day 07_클래스와 객체지향 실습

# 📅 2026-02-07

---
## 1. 클래스(Class)와 객체(Object)

```python
class Car:                          # 공유 가능, 대문자로 시작
    handle = 1                      # 클래스 변수
    speed = 0                       # 클래스 변수

    def __init__(self, name, speed):
        self.name = name            # 인스턴스 변수
        self.speed = speed          # 인스턴스 변수

    def showData(self):
        km = '킬로미터'             # 지역 변수
        msg = '속도:' + str(self.speed) + km
        return msg

    def printHandle(self):
        return self.handle
```

- `class Car` → 객체를 만들기 위한 설계도(원형, prototype)
- 클래스 변수는 모든 객체가 공유
- 인스턴스 변수는 객체마다 개별 보유
- 메서드는 객체 상태(`self`)를 다루는 함수

---

### 1-1. 객체 생성과 self 동작

```python
print(Car.handle)       # 클래스 변수 접근

car1 = Car('tom', 10)
print('car1 객체 주소:', car1)
print('car1 :', car1.name, car1.speed, car1.handle)

car1.color = '파랑'     # 동적 멤버 추가
print('car1.color :', car1.color)

car2 = Car('john', 20)
print('car2 객체 주소:', car2)
print('car2 :', car2.name, car2.speed, car2.handle)
```

- 객체 생성 시 `__init__` 자동 호출
    
- `self` → 현재 객체 자신
    
- 인스턴스에 변수가 없으면 클래스 변수 참조
    
- 파이썬 객체는 실행 중 멤버 추가 가능(해당 객체만 적용)
    

---

### 1-2. 객체 탐색 구조와 메모리

```python
print(Car, car1, car2)
print(id(Car), id(car1), id(car2))

print(car1.__dict__)
print(car2.__dict__)
```

- 클래스와 객체는 서로 다른 메모리 공간 사용
    
- 각 객체는 고유한 주소를 가짐
    
- `__dict__`에는 객체가 실제로 가진 멤버만 저장됨
    

---

### 1-3. 동적 멤버 추가

```python
car1.color = '파랑'
print(car1.color)
```

- 파이썬 객체는 실행 중에도 멤버 추가 가능
    
- **추가한 객체에만 적용**
    

```python
# 아래는 오류
# print(car2.color)
```

---

### 1-4. 객체와 메모리 주소

```python
print(Car, car1, car2)
print(id(Car), id(car1), id(car2))
```

- 클래스와 객체는 **서로 다른 메모리 공간**
    
- 각 객체는 고유한 주소(id)를 가짐
    

---

### 1-5. __dict__로 실제 멤버 확인

```python
print(car1.__dict__)
print(car2.__dict__)
```

출력 예:

```text
{'name': 'tom', 'speed': 10, 'color': '파랑'}
{'name': 'john', 'speed': 20}
```

- 객체가 **직접 가지고 있는 멤버만 출력**
    
- 클래스 변수는 포함되지 않음
    

---

### 1-6. 메서드 호출과 상태 변경

```python
print('---메소드------------------')
print('car1 speed :', car1.showData())
print('car2 speed :', car2.showData())

car1.speed = 80
car2.speed = 110

print('car1 speed :', car1.showData())
print('car2 speed :', car2.showData())

print('car1 handle :', car1.printHandle())
print('car2 handle :', car2.printHandle())
```

- `car1.showData()` 내부 동작 → `Car.showData(car1)`
    
- 인스턴스 변수 값은 객체별로 독립 유지
    
- 클래스 변수는 모든 객체에서 동일하게 참조
    

---

### 1-7. 클래스 변수 접근 메서드

```python
def printHandle(self):
    return self.handle
```

```python
print(car1.printHandle())
print(car2.printHandle())
```

- `self.handle`
    
    - 인스턴스에 없으면 클래스 변수 참조
        

---

### 1-8. 인스턴스 변수 값 변경

```python
car1.speed = 80
car2.speed = 110
```

```python
print(car1.showData())
print(car2.showData())
```

- 객체별로 **독립적인 값 유지**
    
- 다른 객체에 영향 없음

---
## 2. UML 개요 (Unified Modeling Language)

- 클래스를 그림으로 표현하면 구조 이해가 훨씬 쉬워짐
    
- **설계 단계에서 매우 중요**

---
###  2-1. UML에서 중요한 다이어그램 종류

- **클래스 다이어그램**
    
    - 클래스 구조, 속성, 메서드 표현
        
    - 객체지향 설계의 핵심
    
- **시퀀스 다이어그램**
    
    - 객체 간 메시지 흐름, 호출 순서 표현
        
    - 메서드 호출 흐름 이해에 중요
    
- **유스 케이스 다이어그램**
    
    - 사용자(Actor)와 시스템 간의 기능 관계 표현
        
    - 요구사항 정리 단계에서 사용

---
#### 📌 클래스 다이어그램(Class Diagram) 예시

- 클래스 표현 구조

![Car 클래스 다이어그램](https://www.plantuml.com/plantuml/png/SoWkIImgAStDuKhEIImkLd1EB5AevkBIpaZCIyb9LR1IoCmh0NAB2r9JKn2yZFnobyIIH0rD8AUW22Ze0LOEujToEQJcfG0D1000)

- 위쪽: **속성(Attribute)**
- 아래쪽: **메서드(Method)**

- 접근 지정자
    - `+` : public (외부 접근 가능)
    - `-` : private (외부 접근 불가)

---

#### 📌 시퀀스 다이어그램(Sequence Diagram) 예시

- 객체 사이의 **시간 흐름에 따른 메시지 호출 순서** 표현
- 실제 코드 실행 흐름을 그림으로 나타냄

![시퀀스 다이어그램](https://www.plantuml.com/plantuml/png/SoWkIImgAStDuU9AJ2x9Br88BKujuk8g00fc9cSM9EQLA3WdeWcuvgLdvgLoSIaeS761b2lese4KALWfW0tJqEJY0dA1eb2Lo19G4Lsu5dzli6gG2CXkc913QbuAq5K0)

- 왼쪽에서 오른쪽으로 객체 배치
- 위 → 아래 방향이 **시간 흐름**
- 메서드 호출과 반환 관계를 한눈에 확인 가능

---

#### 📌 유스 케이스 다이어그램(Use Case Diagram) 예시

- 사용자(Actor)와 시스템 기능 간의 관계 표현
- "무엇을 할 수 있는가"에 집중

![유스 케이스 다어이그램](https://www.plantuml.com/plantuml/png/SoWkIImgAStDuSf9JIjHACbNACfCpoXHICaiIaqkoSpFu-9AJ2x9Br9uCslBcmKjR-PDuE82oIJcfUUaAhpPiEBr_CxuDzrptdGjUTiwHw6QIm6XKa5NLq43AFUwVwR2guqxNktOe8aX_Mf3mvksthTJP-2GdO17zis2gM-MDy1aQxaSKlDIW3u10000)

- Actor: 시스템을 사용하는 외부 주체
- Use Case: 시스템이 제공하는 기능
- 요구사항 분석 단계에서 주로 사용

---
## 3. 이름 탐색 우선순위 (Scope 우선순위)

```python
kor = 100   # 모듈 전역 변수

def abc():
    kor = 0     # 함수 지역 변수
    print('모듈의 멤버 함수')

class My:
    kor = 80    # 클래스 변수

    def abc(self):
        print('My 멤버 메소드')

    def show(self):
        print(kor)        # 전역 변수
        print(self.kor)  # 인스턴스 → 없으면 클래스
        self.abc()       # 클래스 메소드
        abc()            # 전역 함수
```

- `print(kor)` → **전역 변수**
- `print(self.kor)` → **인스턴스 → 클래스 변수**
- `self.abc()` ≠ `abc()`

```python
tom = My()
print(tom.kor)   # 80
tom.kor = 88     # 인스턴스 변수 생성
print(tom.kor)   # 88

oscar = My()
print(oscar.kor) # 80
```

`이름 탐색 순서: 지역 → 인스턴스 → 클래스 → 전역`

---
## 4. 클래스 변수 공유와 객체 참조 (Singer 예제)

**ex22class.py**
```python
# 클래스는 새로운 타입을 만들어 자원을 공유 가능

# class Singer:
#     title_song = "빛나라 대한민국"
#
#     def sing(self):
#         msg = "노래는 "
#         print(msg, self.title_song)

#import ex22singer             # 모듈 전체를 import (모듈명.클래스명으로 접근)
from ex22singer import Singer  # 모듈 안의 Singer 클래스만 import

# bts = ex22singer.Singer()  # 생성자 호출해 객체를 생성 후 주소를 bts에 치환
bts = Singer()
bts.sing()
print(type(bts))

# 인스턴스 변수가 생성되며 클래스 변수 가림
bts.title_song = "Permission to Dance"
bts.sing()

# 동적 멤버 추가 (bts 객체에만 존재)
bts.co = '빅히트 엔터테인먼트'
print('소속사 : ', bts.co)

print()

ive = Singer()
ive.sing()
print(type(ive))

# print('소속사 : ', ive.co)
# AttributeError: 'Singer' object has no attribute 'co'

# 클래스 변수 변경
Singer.title_song = "아 대한민국"

bts.sing()          # bts 객체에 있는 인스턴스 변수 참조
ive.sing()          # Singer 클래스 변수 참조

# 객체 주소 치환
niceGroup = ive
niceGroup.sing()
```

**ex22singer.py**
```python

class Singer:
    title_song = "빛나라 대한민국"

    def sing(self):
        msg = "노래는 "
        print(msg, self.title_song)
```

**출력**
```
노래는  빛나라 대한민국
<class 'ex22singer.Singer'>
노래는  Permission to Dance
소속사 :  빅히트 엔터테인먼트

노래는  빛나라 대한민국
<class 'ex22singer.Singer'>
노래는  Permission to Dance
노래는  아 대한민국
노래는  아 대한민국
```

---
## 5. 클래스 포함 관계 (Composition, HAS-A 관계)

- 객체지향에서는 **클래스가 다른 클래스를 멤버로 가지는 구조**를 자주 사용함
    
- “~이다(IS-A)”가 아니라  
    **“~을 가진다(HAS-A)” 관계**
    
- 부품 클래스를 재사용하여 **완성 객체를 조립**하는 개념

---
### 5-1. 부품 클래스 정의 (Handle)

**ex23pohamhandle.py**

```python
# 어딘가에서 재사용될 부품 클래스 (핸들)
class PohamHandle:
    quantity = 0    # 핸들 회전량 (공유 자원)

    def leftTurn(self, quantity):   # quantity는 지역 변수
        self.quantity = quantity
        return "좌회전"
    
    def rightTurn(self, quantity):
        self.quantity = quantity
        return "우회전"
```

- `PohamHandle`
    
    - 자동차에 **포함될 부품 클래스**
    
- `quantity`
    
    - 회전량 저장
        
    - 실제 사용은 `self.quantity` → **객체 기준**

---
### 5-2. 포함 관계를 사용하는 완성차 클래스

**ex23pohamcar.py**
```python
# 여러 개의 부품 객체를 조립해서 완성차 생성
# 클래스의 포함 관계(Composition) 사용

from ex23pohamhandle import PohamHandle

class PohamCar:
    turnShowMessage = "정지"

    def __init__(self, ownerName):
        self.ownerName = ownerName
        self.handle = PohamHandle()     # 핸들 객체 포함 (HAS-A 관계)

    def turnHandle(self, q):
        if q > 0:                       # 양수 → 우회전
            self.turnShowMessage = self.handle.rightTurn(q) 
            # 내부적으로 PohamHandle.rightTurn(self.handle, q) 형태로 실행됨
        elif q < 0:                    # 음수 → 좌회전
            self.turnShowMessage = self.handle.leftTurn(q)
        else:
            self.turnShowMessage = "직진"
```
#### 📌 핵심 포인트

- `self.handle = PohamHandle()`
    - **PohamCar HAS-A PohamHandle**
    
- `self.handle.rightTurn(q)`
    - `객체.객체.메서드`
    - 점(`.`)이 두 개 나오면 **포함 관계 의심**
    
- 실제 호출 구조  
    → `rightTurn(self.handle, q)`

---
### 5-3. 포함 관계 객체 실행 예제

```python
if __name__ == '__main__':
    tom = PohamCar('미스터 톰')
    tom.turnHandle(10)
    print(tom.ownerName + '의 회전량은 ' +
          tom.turnShowMessage + ' ' + str(tom.handle.quantity))

    john = PohamCar("미스터 존")
    john.turnHandle(-20)
    print(john.ownerName + '의 회전량은 ' +
          john.turnShowMessage + ' ' + str(john.handle.quantity))

    john.turnHandle(0)
    print(john.ownerName + '의 회전량은 ' +
          john.turnShowMessage + ' 0')
```

**출력**
```
미스터 톰의 회전량은 우회전 10
미스터 존의 회전량은 좌회전 -20
미스터 존의 회전량은 직진 0
```

- 자동차 객체마다 **각자 다른 핸들 객체 보유**
- `tom.handle.quantity` ≠ `john.handle.quantity`
- 부품 객체는 독립적으로 동작

---
### 5-4. 자동차 부품 클래스 다이어그램

![자동차 부품 클래스 다이어그램](https://www.plantuml.com/plantuml/png/NKyz2y8m4DtpAuvELUdWcgdWug88-WSE6vj2UgLtLug8_ztMK3Lc2VSUx-ELeiWwjC6OQq0HE7KUsprI5Hmy34nlqmz8skWTB3Ia4GlkffU1AL_8LSIvgVq_yKoyTyYBSJUIuoEs3Yo8SWHr4fzzDnwS2DO9vMCj-rloAuftthy3Fr7PDeDrKSn_iis2Hp6beseU_m80)

---
## 6. 클래스 포함 관계 연습 (냉장고 – 음식 객체)

- **클래스 포함 관계(Composition, HAS-A)** 실습 예제
    
- 냉장고 객체가 음식 객체를 **담아서 관리**하는 구조
    
- “냉장고는 음식이다(X)” ❌  
    → “냉장고는 음식을 가진다(O)” ✅

---
### 6-1. 냉장고 클래스 (포함하는 쪽)

```python
class Fridge:
    isOpened = False      # 냉장고 문 상태 (클래스 변수)
    foods = []            # 냉장고에 담긴 음식 목록 (FoodData 객체 저장)

    def open(self):
        self.isOpened = True      # 냉장고 문 열림 상태로 변경
        print("냉장고 문을 열기")

    def close(self):
        self.isOpened = False     # 냉장고 문 닫힘 상태로 변경
        print("냉장고 문을 닫기")

    def foodsList(self):
        # 냉장고 안에 들어있는 음식 목록 출력
        for f in self.foods:
            print(f' - {f.name} {f.expiry_date}')
        print()

    def put(self, thing):
        # 냉장고 문이 열려 있을 때만 음식 추가 가능
        if self.isOpened:
            self.foods.append(thing)   # 음식 객체를 리스트에 저장
            print(f'냉장고에 {thing.name} 넣음')
            self.foodsList()
        else:
            print('냉장고 문이 닫혀있음')
```

#### 📌 핵심 포인트

- `foods` 리스트에 **FoodData 객체 저장**
    
- `put()` 메서드는 **객체 자체를 담음**
    
- 상태(`isOpened`)에 따라 동작 제어

---

### 6-2. 음식 데이터 클래스 (포함되는 쪽)

```python
class FoodData:
    def __init__(self, name, expiry_date):
        self.name = name
        self.expiry_date = expiry_date
```

- 음식 하나를 표현하는 **데이터 객체**
    
- 냉장고에 들어갈 부품 클래스

---

### 6-3. 실행 흐름

```python
fObj = Fridge()

apple = FoodData('사과', '2026-8-1')
fObj.put(apple)        # 문 닫힘 → 실패
fObj.open()
fObj.put(apple)        # 성공
fObj.close()

print()

cola = FoodData('콜라', '2027-11-1')
fObj.open()
fObj.put(cola)
fObj.close()
```

**출력**
```
냉장고 문이 닫혀있음
냉장고 문을 열기
냉장고에 사과 넣음
 - 사과 2026-8-1

냉장고 문을 닫기

냉장고 문을 열기
냉장고에 콜라 넣음
 - 사과 2026-8-1
 - 콜라 2027-11-1

냉장고 문을 닫기
```

---
### 6-4 다이어그램 구조


![냉장고 클래스 다이어그램](https://www.plantuml.com/plantuml/png/HO-n2i8m48RtFCMHgQrqSEtKGPm47q5YSzP0cYkvAoZYktiGGpFbFhxyVsvaSLcs9PefEcOuv-1dX8y1FOV0rnKJUXZWJXGBV11vLX83Io6aKjEM-nI9KOTTlQXNmRf98y-GvjPyJQrKwUJ4rTBa5jHubbncVAqXls_UISNwzFzFUxJGzJtOpkuv0qoKn8N4PiJaTRaV)


---
## 7. 클래스 포함 관계 연습 – 로또 시뮬레이션

### 7-1. 로또 공 클래스

```python
import random

class LottoBall:
    """로또 공 객체 하나를 표현"""
    def __init__(self, num):
        self.num = num
```

---
### 7-2. 로또 기계 클래스

```python
class LottoMachine:
    """로또 공 45개를 관리하고, 6개를 무작위로 선택"""

    def __init__(self):
        self.ballist = []
        for i in range(1, 46):
            self.ballist.append(LottoBall(i))   # LottoBall 객체 포함

    def selectBalls(self):
	    # for a in range(45):
	    #     print(self.ballist[a].num, end=' ')
	    # → 초기 상태에서 모든 번호 출력, 확인용
    
    
        print('------------')
	    random.shuffle(self.ballist)  # 번호 섞기
	    
	    # for a in range(45):
	    #     print(self.ballist[a].num, end=' ')
	    # → 섞은 후 번호 출력, 확인용
        return self.ballist[0:6]        # 상위 6개 공 선택
```

- **HAS-A 관계**: `LottoMachine` HAS-A `LottoBall`
- `selectBalls()` → 공 섞기 + 선택

---
### 7-3. 사용자 인터페이스 클래스

```python
class LottoUI:
    """로또 게임 사용자 인터페이스"""

    def __init__(self):
        self.machine = LottoMachine()   # LottoUI HAS-A LottoMachine

    def playLotto(self):
        input('로또를 시작하려면 엔터키를 누르세요')
        selectedBalls = self.machine.selectBalls()
        for ball in selectedBalls:
            print("%d" % (ball.num))
```

- **HAS-A 관계**: `LottoUI` HAS-A `LottoMachine`
- 사용자 입력 후 번호 출력

---
### 7-4. 프로그램 시작점

```python
if __name__ == '__main__':
    # 아래 두 줄은 테스트용, 클래스 동작 확인
    # machine = LottoMachine()
    # print(machine.selectBalls())

    # LottoUI 객체로 실제 게임 실행
    # lot = LottoUI()
    # lot.playLotto()

    # 바로 LottoUI 객체 생성 + 게임 실행
    LottoUI().playLotto()
```
- `__name__ == '__main__'` → 직접 실행 시만 게임 시작
- 클래스별로 **역할 분리** + **포함 관계** 명확

---
### 7-5. 실행 흐름

1. `LottoUI` 객체 생성 → 내부적으로 `LottoMachine` 생성
    
2. `playLotto()` 호출 → 사용자 엔터 대기
    
3. `selectBalls()` 호출 → 1~45번 공 섞기 → 6개 선택
    
4. 선택된 6개 번호 출력

---
### 7-6. 출력 예시
```
로또를 시작하려면 엔터키를 누르세요
------------
23
5
41
12
33
7
```

- 매 실행 시마다 랜덤하게 다른 번호 선택

---
### 7-7. 로또 클래스 다이어그램

![로또 클래스 다이어그램](//www.plantuml.com/plantuml/png/SoWkIImgAStDuL9NUDkuvVMy6M-wbYYyMJ3rpTmPNCavYSN52Zxv9INvJeavEGhLN0f0eAkGLvfhfP2PLmBcQYl4nsVcPPR4nsi0nJMvQhcGzVac9cTavgN2jIO1pSaiBh5I098O-ZMX0iMfEQd99I0hYpKq8KhHZ0trX9kO2x712iK-N2ONv2HMWjLfW1mAydF_chTJLsXuE0RhEceglDhIy6fp2nUAovKCbHJoTNKLbBIKa8B2IY4vFwyaCJElc0lg89X2C8ri0b1jHc8nbqDgNWeed040)

---
## 8. 데이터 전처리 – 로그 변환(Log Transformation)

- 첨도(kurtosis)와 왜도(skewness)가 큰 데이터, 즉 **편차가 큰 데이터**를  
    로그 변환하면 분포 안정, 범위 축소 → 모델 학습 안정화

---
### 8-1. 로그 변환 클래스 정의

```python
import math

class LogTrans:
    # 로그 변환과 역변환을 수행하는 클래스

    def __init__(self, offset: float = 1.0):
        """
        offset : 0 또는 음수 입력 방지용
                 모든 값에 더해서 로그 적용
        """
        self.offset = offset

    def transform(self, x_list: list[float]) -> list[float]:
        # 리스트 데이터를 로그 변환
        return [math.log(x + self.offset) for x in x_list]
    
    def inverse_trans(self, x_list: list[float]) -> list[float]:
        # 로그 변환된 데이터를 원래 값으로 역변환
        return [math.exp(x_log) - self.offset for x_log in x_list]
```

---
### 8-2. 실행 예제

```python
def main():
    # 편차가 큰 예제 데이터
    data = [10.0, 100.0, 1000.0, 10000.0]

    # 로그 변환 객체 생성
    log_trans = LogTrans(offset=1.0)
    
    # 1️⃣ 로그 변환
    data_log_scaled = log_trans.transform(data)
    
    # 2️⃣ 역변환
    reversed_data = log_trans.inverse_trans(data_log_scaled)
    reversed_data_round = [round(val, 1) for val in reversed_data]

    # 결과 출력
    print('원본 자료         :', data)
    print('로그 변환 자료    :', data_log_scaled)
    print('역변환 자료       :', reversed_data_round)


if __name__ == "__main__":
    main()
```

---
### 8-3. 출력 예시
```
원본 자료         : [10.0, 100.0, 1000.0, 10000.0]
로그 변환 자료    : [2.398, 4.615, 6.909, 9.211]
역변환 자료       : [10.0, 100.0, 1000.0, 10000.0]
```

1. 로그 변환 → **큰 수치 차이를 줄여 데이터 안정화**
    
2. `offset` → 로그 입력값이 0 이하가 되지 않도록 방지
    
3. 역변환 가능 → 모델 예측 후 원래 스케일로 복원 가능
