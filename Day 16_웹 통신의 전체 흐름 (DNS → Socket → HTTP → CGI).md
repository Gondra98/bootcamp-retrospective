# Day 16_웹 통신의 전체 흐름 (DNS → Socket → HTTP → CGI)
# 📅 2026-02-25

---
# 1️⃣ HTML에서 시작

```html
<html>    
<body>    
    
<h1>환영합니다</h1>    
<a href="https://www.naver.com">가자 네이버로</a>    
    
</body>    
</html>
```

## ✔ 브라우저에서 일어나는 과정

1. 도메인 확인 ([www.naver.com](http://www.naver.com))
    
2. DNS 서버에 IP 요청
    
3. IP 주소 획득
    
4. TCP 연결 시도
    
5. HTTP 요청 전송
    
6. 서버 응답 수신
    
7. 화면에 출력
    

👉 HTML도 결국 **네트워크 통신**이다.

---

# 2️⃣ 네트워크 기본 개념

- IPv4 주소: `192.168.0.9`
    
- 서브넷 마스크: `255.255.255.0`
    
- 기본 게이트웨이: `192.168.0.1`
    

### ping 예시

- 외부 확인: `ping 8.8.8.8` → 정상 응답
    
- 내부 장치 확인: `ping 192.168.0.214` → 응답 없음
    
- 도메인 확인: `ping www.naver.com` → IP로 변환됨
    

---

# 3️⃣ DNS (Domain Name System)

- **DNS = Domain Name System**  
    → 사람이 기억하기 쉬운 **도메인 이름**을  
    컴퓨터가 이해할 수 있는 **IP 주소**로 변환해주는 시스템
    
- 사람은 `www.naver.com`처럼 도메인 이름을 사용
    
- 컴퓨터/네트워크 장치는 `223.xxx.xxx.xxx` 같은 IP를 사용
    

---

## 🔹 예시 (Python)

```python
import socket  
  
# 도메인 → IP  
ip_address = socket.gethostbyname('www.naver.com')  
print(ip_address)
```


출력 예시:

```
223.130.200.236
```


✔ 즉, `www.naver.com`이라는 사람이 기억하기 쉬운 이름이  
컴퓨터가 이해할 수 있는 **숫자 주소(IP)**로 바뀌었다는 의미

---

## 🔹 DNS 요청 과정 (실제 네트워크 흐름)

1. 브라우저/프로그램에서 `www.naver.com` 요청
    
2. OS가 로컬 DNS 캐시 확인
    
3. 없으면, **ISP DNS 서버**에 질의 → IP 요청
    
4. DNS 서버가 IP 반환
    
5. 프로그램이 받은 IP로 TCP 연결 시도

```
[사용자 프로그램] → DNS 요청 → [DNS 서버] → IP 반환 → 연결
```

---

# 4️⃣ Socket 개념

- **소켓(Socket)** = 프로세스가 네트워크로 데이터를 주고받는 창구
    
- TCP/IP 기반, 클라이언트/서버 구조
    
- 포트 번호로 프로그램 구분 → 같은 IP에서도 여러 통신 가능

### 4-1. 데이터 흐름 (양방향)

```
[클라이언트]           [서버]  
send() ─────────▶ recv()  
recv() ◀───────── send()
```

- TCP = 연결 지향, 신뢰성 높음, 순서 보장
    
- 문자열 전송 시 encode/decode 필요

### 4-2. Python 예시

```python
from socket import *  
  
# 클라이언트 소켓 생성  
sock = socket(AF_INET, SOCK_STREAM)  
sock.connect(('127.0.0.1', 8888))  
sock.send('안녕 서버!'.encode('utf_8'))  
print(sock.recv(1024).decode())  
sock.close()
```

- `AF_INET` → IPv4
    
- `SOCK_STREAM` → TCP
    
- 문자열 → 바이트로 변환 필요

---

# 5️⃣ TCP vs UDP

|구분|TCP|UDP|
|---|---|---|
|연결|O (연결 지향)|X (비연결)|
|신뢰성|높음 (재전송, 오류검사)|낮음 (빠르지만 오류검사 최소)|
|순서보장|O|X|
|사용 예시|웹, 이메일, 파일 전송|게임, 실시간 방송, 스트리밍|

---
### 🔹 TCP 예시 (주고받는 관계)

- **웹 브라우저 ↔ 서버**
    
    - 브라우저가 서버에 요청 → 서버가 응답
        
    - 문자열, 이미지, HTML 모두 **순서대로 정확히 전달**
        
    - 예시 데이터 흐름:
        

```
[클라이언트] → GET /index.html → [서버]  
[클라이언트] ← 200 OK + HTML 파일 ← [서버]
```

- **이메일 송수신 (SMTP/POP3)**
    
    - 메일 전송 요청 → 서버 확인 → 정상 전송 확인
        
    - **응답 확인 후 다음 단계 진행**


✔ 핵심: TCP는 항상 **양방향 주고받음(Full Duplex)** → 신뢰성 있는 통신

---
### 🔹 UDP 예시 (빠르게 한쪽으로만)

- **게임**
    
    - 위치/속도 정보 전송 → 바로 사용, 일부 손실 허용
        
- **실시간 방송**
    
    - 음성/영상 전송 → 약간 끊겨도 스트리밍 계속


✔ 핵심: UDP는 **빠르지만 오류 복구 없음** → 실시간에 유리

---

# 6️⃣ socket_test.py

```python
import socket  
  
# 포트 확인  
print(socket.getservbyname('http', 'tcp'))  # 80  
print(socket.getservbyname('ssh', 'tcp'))   # 22  
  
# 도메인 → IP 확인  
print(socket.getaddrinfo('www.naver.com', 80, proto=socket.SOL_TCP))
```

---

# 7️⃣ socket1 (일회용 서버, 단방향)

## 서버 코드

```python
# 소켓 객체 생성 (IPv4, TCP)
serversock = socket(AF_INET, SOCK_STREAM)  

# 소켓을 IP + 포트에 바인딩
serversock.bind(('127.0.0.1', 8888))  

# 연결 요청 대기 (최대 5개 대기열)
serversock.listen(5)  

# 클라이언트 연결 수락 (여기서 멈추고 기다림)
conn, addr = serversock.accept()  

# 연결된 클라이언트 주소 출력
print('client addr : ', addr)  

# 클라이언트가 보낸 메시지 수신 (최대 1024바이트)
# 수신 데이터는 bytes → decode로 문자열 변환
print('from client message : ', conn.recv(1024).decode())  

# 연결 종료
conn.close()  

# 서버 소켓 종료
serversock.close()
```

- bind() **소켓을 특정 IP 주소와 포트 번호에 연결**시키는 역할
## 클라이언트 코드

```python
# 소켓 객체 생성 (IPv4, TCP)
clientsock = socket(AF_INET, SOCK_STREAM)  

# 서버에 연결 요청 (127.0.0.1:8888)
clientsock.connect(('127.0.0.1', 8888))  

# 서버로 메시지 전송 (문자열 → 바이트 변환 필요)
clientsock.send('안녕 반가워'.encode('utf_8'))  

# 소켓 종료 (서버와 연결 종료)
clientsock.close()
```

### 🔹 실행 흐름 (Full Duplex)

1. **서버 먼저 실행** → 연결 대기 상태
    
2. **클라이언트 실행** → 서버에 연결 요청
    
3. 서버 `accept()` → 클라이언트 연결 수락
    
4. 클라이언트 `send()` → 서버 수신
    
5. 서버 `recv()` → 메시지 확인
    
6. (양쪽 모두 필요 시) 서버 → 클라이언트 `send()` 가능 → 양방향 통신 가능
    
7. **연결 종료**

## 🔹 실제 데이터 이동 (단방향)

```
[클라이언트] → send("안녕 반가워") → [서버] recv()
```

## 🖥 실제 콘솔 결과

### 서버

```
서버 서비스 중...  
client addr :  ('127.0.0.1', 53421)  
from client message : 안녕 반가워
```

### 클라이언트

```
# 보내기만 하고 받지 않음
```

✔ 단방향: 클라이언트 → 서버

---

# 8️⃣ socket2 (무한 루프 서버, 양방향)

## 서버 코드

```python
import socket  

# 모든 IP에서 접속 허용
HOST = ''  
PORT = 7788  

# TCP 소켓 생성
serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  

# IP + 포트에 소켓 바인딩
serversock.bind((HOST, PORT))  

# 연결 요청 대기 (최대 5개 대기열)
serversock.listen(5)  

# 무한 루프 → 여러 클라이언트 처리 가능
while True:  
    # 클라이언트 연결 수락 (블록 상태, 연결되면 conn과 addr 반환)
    conn, addr = serversock.accept()  

    # 클라이언트 IP, 포트 출력
    print('client info : ', addr[0], ' ', addr[1])  

    # 클라이언트 메시지 수신 (최대 1024바이트, decode 필수)
    print(conn.recv(1024).decode())  

    # 서버 → 클라이언트 응답 (문자열 → 바이트 변환)
    conn.send(('from server : ' + str(addr[1]) + ' 너도 잘지내라').encode('utf_8'))

```

**특징:**

- 무한 루프 → 여러 번 접속 가능
    
- 양방향 통신 → 서버가 클라이언트 메시지 받음 + 응답 전송
    
- 순차 처리 → 한 번에 하나씩 처리 (동시 접속 X)

## 클라이언트 코드

```python
from socket import *  

# TCP 소켓 생성
clientsock = socket(AF_INET, SOCK_STREAM)  

# 서버 접속
clientsock.connect(('127.0.0.1', 7788))  

# 서버로 메시지 전송 (문자열 → 바이트)
clientsock.send('안녕 반가워'.encode('utf_8'))  

# 서버가 보내온 메시지 수신 후 출력
print('수신자료:', clientsock.recv(1024).decode())  

# 소켓 종료 (연결 종료)
clientsock.close()
```
## 🔹 실제 데이터 이동 (Full Duplex, 양방향)

```
[클라이언트]                      [서버]
   connect() ─────────────────────▶ accept()
   send("안녕 반가워") ───────────▶ recv()
                                     send("from server : 54321 너도 잘지내라")
   recv() ◀────────────────────────
close()                              close()
```

## 🖥 실제 콘솔 결과

### 서버

```
서버(무한 루핑) 서비스 중...  
client info :  127.0.0.1   54321  
안녕 반가워
```

### 클라이언트

```
수신자료: from server : 54321 너도 잘지내라
```

✔ 양방향: 클라이언트 ↔ 서버

---

# 9️⃣ 전체 통신 흐름

클라이언트:  
`socket() → connect() → send() → recv() → close()`

서버:  
`socket() → bind() → listen() → accept() → recv() → send() → close()`

---
## 1️0. httpserver1.py – 단순 웹 서버

```python
from http.server import SimpleHTTPRequestHandler, HTTPServer  
  
PORT = 7777  
  
# 클라이언트 요청 처리 핸들러 지정 (GET 요청 시 파일 전송)  
handler = SimpleHTTPRequestHandler    
  
# HTTP 서버 객체 생성  
# ('192.168.0.9', PORT) → 서버 IP와 포트 지정  
serv = HTTPServer(('192.168.0.9', PORT), handler)  
  
print('웹 서비스 시작...')  
  
# 웹 서버 무한 루프  
serv.serve_forever()
```
### 🔹 핵심 설명

- **SimpleHTTPRequestHandler**  
    → GET 요청 들어오면 해당 경로의 파일 읽어서 클라이언트로 전송
    
- **HTTPServer**  
    → 지정 IP + 포트에서 요청 대기  
    → bind + listen + accept를 내부적으로 처리
    
- **serve_forever()**  
    → 무한 루프  
    → 브라우저가 요청할 때마다 자동으로 처리

---
## 10-1. abc.html – 기본 HTML 예제

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>연습</title>  
</head>  
<body>  
    <h1>환영합니다</h1>  
    <a href="https://www.naver.com">가자 네이버로</a>  
    <br/>  
    <a href="https://www.daum.net">가자 다음으로</a>  
</body>  
</html>
```
### 🔹 특징

- `<a>` 태그 클릭 → 브라우저가 새 요청 생성
    
- `br` → 줄바꿈
    
- 단순 웹 페이지 + 외부 링크 테스트 가능
    

---

## 10-2. def.html – 버튼과 스타일 적용

```html
<!DOCTYPE html>     <!-- HTML5 선언 -->  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>연습2</title>  
</head>  
<body>  
    <b style="font-size: 30px;">단순 웹 적용</b>  
    <hr>  
    <input type="button" value="이전" onclick="history.back()">  
</body>  
</html>
```

### 🔹 특징

- `<b>` + `style` → 글씨 굵게 + 폰트 크기 적용
    
- `<hr>` → 구분선
    
- `<input type="button">` + `onclick="history.back()"`  
    → 이전 페이지로 돌아가는 버튼 기능

---

## 🔹 실행 흐름 예시

1. 서버 실행:

```
python httpserver1.py  
웹 서비스 시작...
```

2. 브라우저에서 접속:

```
http://192.168.0.9:7777/abc.html  
http://192.168.0.9:7777/def.html
```

3. 브라우저가 GET 요청 → 서버가 HTML 파일 전송 → 화면 출력
    
4. abc.html → 외부 링크 클릭 가능
    
5. def.html → 버튼 클릭 시 이전 페이지로 이동

---

# 11️. httpserver2.py – CGI 지원 웹 서버

이제는 단순 파일 전송이 아니라  
👉 **Python 파일을 실행해서 결과를 보내는 단계**

---
## ✅ 📌 CGI란?

> CGI(Common Gateway Interface)는  
> 웹 서버가 외부 프로그램(파이썬 등)을 실행하고  
> 그 실행 결과를 브라우저에 전달하는 방식이다.

즉,
```
브라우저 → 웹서버 → 파이썬 실행 → 결과 출력 → 브라우저
```

---

## 📁 폴더 구조

```
pack3http/
│
├─ httpserver2.py
├─ index.html
├─ friend.html
│
├─ images/
│   └─ dog.jpeg
│
└─ cgi-bin/
    ├─ hello.py
    ├─ world.py
    ├─ my.py
    └─ friend.py
```

- `images/` → 정적 파일 저장
    
- `cgi-bin/` → 실행용 Python 파일 위치
    

---

# 🔹 11-1. httpserver2.py 코드

```python
# httpserver2.py

from http.server import HTTPServer, CGIHTTPRequestHandler

PORT = 8888  # 사용할 포트 번호

class Handler(CGIHTTPRequestHandler):
    # 🔥 매우 중요
    # URL이 /cgi-bin 으로 시작하면 해당 파일을 "실행"한다
    # (그 외는 정적 파일 전송)
    cgi_directories = ['/cgi-bin']

def run():
    # 127.0.0.1:8888 에 서버 생성
    server = HTTPServer(('127.0.0.1', PORT), Handler)

    print(f"웹 서비스 진행 중... http://127.0.0.1:{PORT}")
    try:
        # 요청을 계속 대기
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료")
    finally:
        # 서버 자원 정리
        server.server_close()

if __name__ == '__main__':
    run()
```

### 🔎 핵심 설명

|요소|설명|
|---|---|
|HTTPServer|웹 요청을 받아주는 서버 객체|
|CGIHTTPRequestHandler|CGI 파일 실행 가능 핸들러|
|cgi_directories|이 경로는 실행, 나머지는 파일 전송|

## ✅ 🔎 CGI 서버 동작 원리 추가 설명

|요청 경로|서버 동작|
|---|---|
|`/index.html`|파일 그대로 전송|
|`/images/dog.jpeg`|이미지 파일 그대로 전송|
|`/cgi-bin/hello.py`|Python 실행 후 결과 전송|

👉 `/cgi-bin` 으로 시작하면 실행  
👉 그 외는 정적 파일

## ✅ 📌 CGI 응답 형식 (중요 추가)

CGI 프로그램은 반드시 아래 형식을 지켜야 한다:

```
[HTTP 헤더]  
(빈 줄)  
[HTML 본문]
```

예:
```python
print("Content-Type: text/html; charset=utf-8")  
print()   # 🔥 반드시 필요
```

⚠ 빈 줄이 없으면  
→ 500 Internal Server Error 발생 가능

---

# 🔹 11-2. index.html

👉 메인 페이지 (GET 테스트)

```html
<!DOCTYPE html>  
<html lang="ko">  
<head>  
    <meta charset="UTF-8">  
    <title>메인</title>  
</head>  
<body>  
    <h1>메인 페이지</h1>  
  
    <!-- hello.py 실행 -->  
    <a href="cgi-bin/hello.py">hello</a>  
    <br/>  
  
    <!-- world.py 실행 -->  
    <a href="cgi-bin/world.py">world</a>  
    <br/>  
  
    <!-- GET 방식으로 데이터 전달 -->  
    <!-- ? 뒤는 파라미터 -->  
    <a href="cgi-bin/my.py?name=tom&age=23">my(get방식)</a>  
    <br/>  
  
    <!-- POST 폼 페이지 -->  
    <a href="friend.html">friend</a>  
</body>  
</html>
```



---

# 🔹 11-3. cgi-bin/hello.py

👉 단순 출력 테스트

```python
# -*- coding: utf-8 -*-  
import sys  
sys.stdout.reconfigure(encoding='utf-8')  
  
# 테스트 변수  
msg = "파이썬 CGI 실행 성공!"  
  
# 🔥 반드시 먼저 출력해야 하는 HTTP 헤더  
print("Content-Type: text/html; charset=utf-8")  
print()  # 헤더와 본문 구분 (매우 중요)  
  
print(f"""  
<html>  
<head>  
<meta charset="UTF-8">  
<title>hello</title>  
</head>  
<body>  
<h2>{msg}</h2>  
<a href="../index.html">메인으로</a>  
</body>  
</html>  
""")
```


---

# 🔹 11-4. cgi-bin/world.py

👉 이미지 포함 페이지

```python
# -*- coding: utf-8 -*-  
import sys  
sys.stdout.reconfigure(encoding='utf-8')  
  
data1 = "자료1"  
data2 = "두번째 자료"  
  
print("Content-Type: text/html; charset=utf-8")  
print()  
  
print(f"""  
<html>  
<head>  
<meta charset="UTF-8">  
<title>world</title>  
</head>  
<body>  
<h1>world 페이지</h1>  
자료 출력 : {data1}, {data2}  
<br/><br/>  
  
<!-- 🔥 현재 위치는 /cgi-bin/world.py -->  
<!-- ../ 은 상위 폴더로 이동 -->  
<img src="../images/dog.jpeg" width="300">  
  
<br/><br/>  
<a href="../index.html">메인으로</a>  
</body>  
</html>  
""")
```

# 🔹 world.py 이미지 부분 아래에 추가

현재 실행 위치:

```
/cgi-bin/world.py
```

이미지 실제 위치:

```
/images/dog.jpeg
```

그래서:

```
../images/dog.jpeg
```

👉 `../` = 상위 폴더로 이동

---

# 🔹 11-5. cgi-bin/my.py (GET 처리)

```python
# -*- coding: utf-8 -*-  
import sys  
sys.stdout.reconfigure(encoding='utf-8')  
  
import os  
import urllib.parse  
  
# 🔥 GET 방식 데이터는 QUERY_STRING에 저장됨  
query = os.environ.get("QUERY_STRING", "")  
  
# 문자열 → 딕셔너리 변환  
params = urllib.parse.parse_qs(query)  
  
# 값 꺼내기 (리스트로 반환되므로 [0])  
name = params.get("name", [""])[0]  
age = params.get("age", [""])[0]  
  
print("Content-Type: text/html; charset=utf-8")  
print()  
  
print(f"""  
<html>  
<head>  
<meta charset="UTF-8">  
<title>my</title>  
</head>  
<body>  
넘겨 받은 값 : 이름은 {name}, 나이는 {age}  
<br/><br/>  
<a href="../index.html">메인으로</a>  
</body>  
</html>  
""")
```


### 🔎 핵심 원리

- `?name=tom&age=23`
    
- `QUERY_STRING`에 저장됨
    
- `parse_qs()`로 딕셔너리 변환

## ✅ 🔎 GET vs POST 차이

|구분|GET|POST|
|---|---|---|
|데이터 위치|URL 뒤 (? 뒤)|요청 Body|
|길이 제한|있음|거의 없음|
|보안|낮음|상대적으로 안전|
|서버에서 읽는 위치|QUERY_STRING|stdin|

예:
```
/cgi-bin/my.py?name=tom&age=23
```

👉 `QUERY_STRING`에 저장됨  
👉 `parse_qs()`로 딕셔너리 변환

---

# 🔹 11-6. friend.html (POST 폼)

```html
<!DOCTYPE html>  
<html lang="ko">  
<head>  
    <meta charset="UTF-8">  
    <title>friend 입력</title>  
</head>  
<body>  
    <h2>친구 정보 입력</h2>  
  
    <!-- 🔥 method="post" -->  
    <form action="cgi-bin/friend.py" method="post">  
        이름: <input type="text" name="name"><br/>  
        전화: <input type="text" name="phone"><br/>  
        성별:  
        <input type="radio" name="gen" value="남" checked>남자  
        <input type="radio" name="gen" value="여">여자  
        <br/><br/>  
        <input type="submit" value="서버로 자료 전송">  
    </form>  
</body>  
</html>
```


<html lang="ko">  
<head>  
    <meta charset="UTF-8">  
    <title>friend 입력</title>  
</head>  
<body>  
    <h2>친구 정보 입력</h2>  
  
    <!-- 🔥 method="post" -->  
    <form action="cgi-bin/friend.py" method="post">  
        이름: <input type="text" name="name"><br/>  
        전화: <input type="text" name="phone"><br/>  
        성별:  
        <input type="radio" name="gen" value="남" checked>남자  
        <input type="radio" name="gen" value="여">여자  
        <br/><br/>  
        <input type="submit" value="서버로 자료 전송">  
    </form>  
</body>  
</html>


---

# 🔹 11-7. cgi-bin/friend.py (POST 처리)

```python
# -*- coding: utf-8 -*-  
import sys  
sys.stdout.reconfigure(encoding='utf-8')  
  
import os  
import urllib.parse  
  
# 🔥 요청 방식 확인  
method = os.environ.get("REQUEST_METHOD", "GET")  
  
if method == "POST":  
    # POST 데이터 길이  
    length = int(os.environ.get("CONTENT_LENGTH", 0))  
      
    # 🔥 POST 데이터는 표준입력(stdin)으로 들어옴  
    body = sys.stdin.read(length)  
else:  
    body = os.environ.get("QUERY_STRING", "")  
  
# 문자열 → 딕셔너리  
params = urllib.parse.parse_qs(body)  
  
name = params.get("name", [""])[0]  
phone = params.get("phone", [""])[0]  
gen = params.get("gen", [""])[0]  
  
print("Content-Type: text/html; charset=utf-8")  
print()  
  
print(f"""  
<html>  
<head>  
<meta charset="UTF-8">  
<title>friend 결과</title>  
</head>  
<body>  
<h2>입력 결과</h2>  
이름 : {name}<br/>  
전화번호 : {phone}<br/>  
성별 : {gen}  
<br/><br/>  
<a href="../index.html">메인으로</a>  
</body>  
</html>  
""")
```

## ✅ 🔎 CGI 환경 변수 설명

CGI가 자동으로 만들어주는 주요 환경변수:

| 환경변수           | 의미               |
| -------------- | ---------------- |
| REQUEST_METHOD | 요청 방식 (GET/POST) |
| QUERY_STRING   | GET 데이터          |
| CONTENT_LENGTH | POST 데이터 길이      |
| CONTENT_TYPE   | 데이터 타입           |

---
# 🔥 전체 동작 흐름 정리

브라우저  
↓  
index.html 요청  
↓  
링크 클릭  
↓

- `/cgi-bin` → Python 실행 후 HTML 생성
    
- `.html` → 파일 그대로 전송
    
- `/images` → 이미지 파일 그대로 전송

---

## ✅ 📌 CGI의 특징과 한계 (마무리 설명 추가 추천)

CGI는 요청이 올 때마다:

1. Python 프로세스를 새로 생성
    
2. 실행 후 종료

👉 속도가 느림  
👉 대규모 서비스에 부적합

그래서 이후 등장한 기술:

- WSGI
    
- Flask
    
- Django
