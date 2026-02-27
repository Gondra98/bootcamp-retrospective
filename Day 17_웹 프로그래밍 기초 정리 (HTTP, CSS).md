# Day 17_웹 프로그래밍 기초 정리 (HTTP, CSS)
# 📅 2026-02-26

---

# HTTP URL 구조 이해

브라우저 주소창에 입력하는 예:

http://127.0.0.1:8888/index.html

구조 분석:
```
http://          → 프로토콜    
127.0.0.1        → 서버 주소 (localhost)    
8888             → 포트번호    
/index.html      → 요청 경로 (Request Path)
```
### ✔ 핵심 개념

`/index.html` 은

👉 **서버에게 무엇을 달라고 요청하는 이름**

즉,
- `/index.html` → index.html 파일 달라
- `/cgi-bin/sangpum.py` → sangpum.py 실행 결과 달라

---
# 1️⃣ 프로젝트 폴더 구조

```
web_project/  
│  
├── index.html  
├── friend.html  
│  
└── cgi-bin/  
    ├── hello.py  
    ├── world.py  
    ├── my.py  
    ├── sangpum.py  
    └── mydb.dat
```

- `index.html` → 메인 페이지 (정적 파일)
    
- `cgi-bin` → CGI 실행 폴더 (Python 실행 위치)
    
- `mydb.dat` → DB 접속 정보 저장 (pickle 사용)


---

# 2️⃣ index.html 코드 + 설명

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>메인</title>  
</head>  
<body>  
    <h1>메인 페이지</h1>  
  
    <!-- 단순 CGI 실행 -->  
    <a href="cgi-bin/hello.py">hello</a>  
    <br/>  
  
    <!-- 단순 CGI 실행 -->  
    <a href="cgi-bin/world.py">world</a>  
    <br/>  
  
    <!-- GET 방식으로 데이터 전달 -->  
    <a href="cgi-bin/my.py?name=tom&age=23">my(get방식)</a>  
    <br/>  
  
    <!-- 일반 정적 HTML 파일 -->  
    <a href="friend.html">friend</a>  
    <br/>  
  
    <!-- DB 조회 프로그램 -->  
    <a href="cgi-bin/sangpum.py">sangpum</a>  
</body>  
</html>
```

### ✔ 핵심 설명

- `friend.html` → 그냥 파일 열기 (정적)
    
- `cgi-bin/*.py` → 서버에서 Python 실행 후 결과 반환
    
- `?name=tom&age=23` → GET 방식 (URL에 데이터 포함)
    

---

# 3️⃣ sangpum.py 코드 + 상세 설명

```python
import sys  
sys.stdout.reconfigure(encoding='utf-8')
```

👉 브라우저에 한글이 깨지지 않도록 UTF-8 설정

---

```python
import MySQLdb  
import pickle
```

- `MySQLdb` → MariaDB 연결용 라이브러리
    
- `pickle` → 파일에서 객체(설정값) 읽기
    

---

```python
with open("cgi-bin/mydb.dat", mode="rb") as obj:  
    config = pickle.load(obj)
```

👉 DB 접속 정보를 파일에서 불러옴  
👉 아이디/비밀번호를 코드에 직접 쓰지 않음 (보안 목적)

---
```python
print("Content-Type: text/html; charset=utf-8")  
print()
```

🚨 CGI에서 가장 중요

- HTTP 헤더 먼저 출력
    
- 빈 줄 반드시 필요

---
```python
print("<html><body><b>** 상품 정보 **</b><br/>")  
print("<table border='1'>")  
print("<tr><td>코드</td><td>상품명</td><td>수량</td><td>단가</td></tr>")
```

👉 HTML 시작 + 테이블 헤더 생성

---
```python
try:  
    conn = MySQLdb.connect(**config)  
    cursor = conn.cursor()
```

- DB 연결
    
- 커서 객체 생성 (SQL 실행용)

---
```python
    cursor.execute("select * from sangdata order by code desc")
```

👉 `sangdata` 테이블 전체 조회  
👉 코드 내림차순 정렬

---
```python
    datas = cursor.fetchall()
```

👉 조회 결과를 전부 가져옴  
👉 결과는 튜플 리스트 형태

예:
```
[(3,'모자',5,10000),  
 (2,'장갑',3,8000)]
```


---
```python
    for code,sang,su,dan in datas:  
        print("""  
            <tr>  
                <td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td>  
            </tr>  
        """.format(code,sang,su,dan))
```

👉 한 행씩 꺼내서 HTML `<tr>` 생성  
👉 Python에서 HTML을 직접 만들어 출력

---
```python
except Exception as e:  
    print("err : ", e)
```

👉 예외 발생 시 에러 출력

---
```python
finally:  
    cursor.close()  
    conn.close()
```

👉 DB 자원 반드시 닫기  
👉 서버 프로그램에서는 특히 중요

---
```python
print("</table>")  
print("</body></html>")  
  
print("</table>")  
print("</body></html>")
```
👉 HTML 종료  
(※ 여기서는 table/body가 두 번 출력됨 → 중복)


---
# 4. WEB 기초
## 4-1. 폴더 구조 이해

```
pro3web/  
│  
└── pack1/  
     │  
     ├── a.html        # HTML 태그, 목록, pre, Block/Inline 연습  
     ├── b.html        # 이미지, 링크, iframe 연습  
     ├── god.txt       # 단순 텍스트 파일  
     ├── mydb.dat      # DB 접속 정보(pickle)  
     │  
     ├── images/         
     │     └── car.jpeg # 이미지 파일  
     │  
     ├── kbs/            
     │     ├── aa.html  # kbs 단순 페이지  
     │     └── sbs/  
     │           └── cc.html # sbs, 상대/절대 경로 실습  
     │  
     └── mbc/            
           └── bb.html  # mbc 단순 페이지
```

### 🔹 이해 포인트

- **pack1** → 실습용 HTML/텍스트/이미지 모아둔 폴더
    
- **images/** → 이미지 파일 저장, HTML에서 `<img>`로 불러옴
    
- **kbs/mbc/** → 사이트별 폴더, 링크 연습용
    
- **상대경로/절대경로**를 이해하기 쉽게 폴더 구조 설계
    

---

## 4-2. a.html – HTML 태그 실습

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
</head>  
<body>  
    Tag: 문서의 구조와 형태를 표현하는 명령어로 <>안에 적는다.<br/><br>  
    aaa&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bbb  ccc<br>  
    -- 출력 형태에 태그 종류 ---<br>  
    1) Block 방식 : 행 분리가 이루어짐 : p, h, div, ol, li ...<br>  
    2) inline 방식 : 같은 행에 계속 출력 : b, i, span ...      
    <!-- html 주석 -->  
    <hr>  
    <p><b>문단 나누기</b></p>  
    <i>이텔릭으로 표시</i>  
    작업 계속  
    <h1>표제용 제목 가장 큼</h1>  
    <h6>표제용 제목 가장 작음</h6>  
    <div style="background-color: yellow;">구역 설정 태그(행 전체)</div>good  
    <span style="background-color: silver;">구역 설정 태그(일부 영역)</span>good  
    <b>&lt;특수문자 표시&gt;</b><br>  
    <hr size="10" width="200" color="blue">  
    <hr width="60%" color="#0000ff">  
    <pre>  
        내가    입력한  
        대로    그  대  로  
                보  여  줘  
    </pre>  
    <br>  
    --- 목록 만들기 ---<br>  
    <ul>  
        <li>파이썬</li>  
        <li>마리아디비</li>  
    </ul>  
    <ol>  
        <li>파이썬</li>  
        <li>마리아디비</li>  
        <ul>  
            <li>파이썬</li>  
            <li>마리아디비</li>  
        </ul>  
    </ol>  
</body>  
</html>
```

### 🔹 상세 설명

1. **Block vs Inline**
    
    - **Block** → 한 줄 전체 차지, 줄 바꿈 자동: `<p>`, `<h1>`~`<h6>`, `<div>`, `<ol>`, `<li>`
        
        - `<p>` → 문단, 자동 줄바꿈
            
        - `<h1>`~`<h6>` → 제목, 숫자 작을수록 크기 작음 (`h1` 가장 큼)
            
        - `<div>` → 블록 단위 영역, 스타일 적용 가능
            
        - `<ol>`/`<ul>` → 순서 있는/없는 목록, `<li>`와 함께 사용
            
    - **Inline** → 같은 줄에 계속 출력: `<b>`(굵게), `<i>`(기울임), `<span>`(부분 스타일)
        
2. **특수문자 표시**
    
    - `<b>&lt;특수문자&gt;</b>` → `<`와 `>`를 화면에 그대로 출력
        
3. **`<pre>` 태그**
    
    - 줄바꿈/공백 그대로 유지, 코드나 시각적 정렬용
        
4. **목록 만들기**
    
    - `<ul>` → 순서 없는 목록, 점(`•`)으로 표시
        
    - `<ol>` → 순서 있는 목록, 숫자로 표시
        
    - `<li>` → 항목, `<ul>`이나 `<ol>` 안에 사용 가능
        
    - `<li>` 안에 `<ul>` → 하위 목록 구현 가능
        

---
## 4-3. b.html – 이미지, 링크, iframe

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>이미지와 링크</title>  
</head>  
<body>  
    이미지 삽입  
    <img src="images/car.jpeg" width="300" title="자동차" alt="난 자동차야"/>  
    <br>  
    <img src="https://fizz-download.playnccdn.com/download/v2/buckets/preorder/files/19c8df2ff51-830e1829-122e-4607-8f35-770080cd6ca9"/>  
    <br>  
    ** a tag 연습 **<br>  
    <a href="a.html">문서 보기</a><br>  
    <a href="a.html" target="_new">문서 보기(새탭)</a><br>  
    <a href="a.html" target="_blank">문서 보기(새탭)</a><br>  
    <a href="https://www.kbs.co.kr">kbs 보기</a><br>  
    <a href="a.html"><img src="images/car.jpeg" width="100"/></a><br>  
    <a href="god.txt">text 문서 보기</a><br>  
    <a href="mydb.dat">mydb 문서 보기</a><br>  
    <a href="mailto:asseo1@naver.com">이메일</a><br>  
    <a href="kbs/sbs/cc.html">sbs의 cc보기(상대경로)</a><br>  
    <a href="/pro3web/pack1/kbs/sbs/cc.html">sbs의 cc보기(절대경로)</a><br>  
    <hr>  
    --- iframe 연습 ---<br>  
    <a href="https://www.daum.net">daum</a>  
    <br><br>  
    <a href="a.html" target="tiger">문서 보기</a>  
    <iframe src="https://www.daum.net" width="98%" height="150" name="tiger"/>  
</body>  
</html>
```

### 🔹 상세 설명

1. **`<img>`**
    
    - `src` → 이미지 경로 (`images/car.jpeg` 로컬, URL 가능)
        
    - `width` → 이미지 너비 지정
        
    - `title` → 마우스 올리면 툴팁 표시
        
    - `alt` → 이미지가 안 보일 때 텍스트 표시, 접근성 필수
        
2. **링크 `<a>`**
    
    - `href` → 이동할 URL
        
    - `target="_blank"` → 새 탭에서 열기
        
    - `<a>` 안에 `<img>` → 이미지를 클릭하면 링크 이동
        
    - `mailto:` → 이메일 앱 열기
        
3. **iframe**
    
    - 외부 페이지 삽입
        
    - `width`/`height` → 크기 지정
        
    - `name` → 타겟 지정 가능, 다른 링크에서 `target="tiger"`로 iframe에 표시
        

---

## 4-4. aa.html – kbs 단순 페이지

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>KBS</title>  
</head>  
<body>  
    kbs 파일  
</body>  
</html>
```

### 🔹 설명

- 단순 텍스트 페이지, 링크 실습용
    
- HTML 최소 구조: `<!DOCTYPE html>`, `<html>`, `<head>`, `<body>`
    
- 브라우저에 “kbs 파일”만 표시
    

---

## 4-5. bb.html – mbc 단순 페이지

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>MBC</title>  
</head>  
<body>  
    mbc 파일  
</body>  
</html>
```

- aa.html과 구조 동일
    
- “mbc 파일” 표시, 링크 연습용
    

---

## 4-6. cc.html – sbs 페이지, 상대/절대 경로

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>SBS</title>  
</head>  
<body>  
    sbs 파일<br>  
    <a href="../aa.html">kbs : aa</a><br>  
    <a href="../../mbc/bb.html">mbc : bb</a><br>  
</body>  
</html>
```

### 🔹 상세 설명

1. **상대 경로**
    
    - `../` → 상위 폴더 한 단계 이동
        
    - `../../` → 상위 2단계 이동
        
2. **절대 경로**
    
    - `/pro3web/pack1/kbs/sbs/cc.html` → 루트부터 전체 경로 지정
        
3. **핵심 학습 포인트**
    
    - 폴더 구조와 링크 관계 이해
        
    - 사이트 네비게이션 구조 설계 연습 가능

---
## 4-7. c.html – 테이블(table)과 폼(form) 태그

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>테이블과 폼</title>  
</head>  
<body>  
    --- table(표) tag ---  
    <table border="1">  
        <tr>  
            <th>번호</th><th>이름</th><th>주소</th>  
        </tr>  
        <tr>  
            <td width="100">1</td><td>가나다</td><td>강남구 역삼동</td>  
        </tr>  
        <tr>  
            <td>2</td><td></td><td></td>  
        </tr>  
    </table>
```

### 🔹 Table 기본 설명

1. `<table>` → 표 시작
    
    - `border="1"` → 테두리 굵기 지정
        
2. `<tr>` → 행(row) 생성
    
3. `<th>` → 헤더 셀, 기본 **굵은 글씨, 중앙 정렬**
    
4. `<td>` → 데이터 셀, 일반 내용
    
5. `width` → 셀 너비 지정
    
6. 빈 `<td>` → 비어있어도 셀 유지

---

### 🔹 열병합 / 행병합 예시

```html
   열병합, 행병합<br>  
    <table border="1" width="200">  
        <tr>  
            <td colspan="2">a</td>  
            <td>b</td>  
            <td rowspan="2">c</td>  
        </tr>  
        <tr>  
            <td>d</td>  
            <td>cell(셀)</td>  
            <td>  
                <table border="2">  
                    <tr>  
                        <td>kor</td>  
                        <td>eng</td>  
                    </tr>  
                </table>  
            </td>  
        </tr>  
    </table>
```

 
- `colspan="2"` → 가로 2칸 합침
    
- `rowspan="2"` → 세로 2칸 합침
    
- `<td>` 안에 `<table>` → **중첩 테이블**, 복잡한 표 구현 가능

---
### 🔹 Form 태그 – 사용자 입력

```html
    <hr>  
    --- form 태그(자료 입력용 태그) ---  
    <form method="get">  
        1. 이름 : <input type="text" name="name" id="irum" value="이기자" size="30" readonly><br>  
        2. 비밀번호 : <input type="password" name="pwd"><br>  
        3. 메세지 : <textarea name="msg" cols="50" rows="5">멀티라인 텍스트 박스</textarea><br>  
        4. 학년 : <input type="radio" name="hak" value="1" checked>1학년  
                    <input type="radio" name="hak" value="2">2학년  
                    <input type="radio" name="hak" value="3">3학년  
                    <input type="radio" name="hak" value="4">4학년<br>  
        5. 과목 : <input type="checkbox" name="gwa" value="python">파이썬  
                    <input type="checkbox" name="gwa" value="c">C언어  
                    <input type="checkbox" name="gwa" value="java">자바<br>  
        6. 지역 : <select name="addr">  
                    <option value="c" selected>시청</option>  
                    <option value="j">종로</option>  
                    <option value="k">강남</option>  
                </select>  
        7. <input type="file" name="fname"><br>  
        8. 숨김 : <input type="hidden" name="ji" value="대한민국" ><br><br>  
        <input type="submit" value="전송확인">  
        <input type="reset" value="입력 취소">  
    </form>  
</body>  
</html>
```

### 🔹 Form 상세 설명

1. `<form>`
    
    - `method="get"` → URL 쿼리로 전송
        
    - `method="post"` → 데이터 안전 전송
        
2. 입력 요소
    
    - `<input type="text">` → 한 줄 입력
        
    - `<input type="password">` → 글씨 가림
        
    - `<textarea>` → 멀티라인 입력
        
    - `<input type="radio">` → 1개 선택 가능
        
    - `<input type="checkbox">` → 다중 선택 가능
        
    - `<select>` + `<option>` → 드롭다운 메뉴
        
    - `<input type="file">` → 파일 업로드
        
    - `<input type="hidden">` → 숨김 데이터 전송
        
3. 속성
    
    - `name` → 서버에서 변수명으로 사용
        
    - `id` → CSS/JS 연결용
        
    - `value` → 초기값
        
    - `size`, `cols`, `rows` → 크기 조절
        
    - `checked`, `selected`, `readonly` → 상태 지정
        
4. 제출/초기화
    
    - `<input type="submit">` → 데이터 전송
        
    - `<input type="reset">` → 입력 초기화

---
## 4-8. d_css.html – HTML / CSS 실습

### 🔹 HTML 구조
```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>CSS 실습</title>  
  
    <!-- 1. Internal Style -->  
    <style type="text/css">  
        /* h1 : 큰 제목, 궁서 글꼴, 빨강 */  
        h1 { font-size: 50px; font-family: 궁서; color: red; }  
  
        /* h2, strong : 작은 글씨, 파랑 */  
        h2,strong { font-size: 8px; color: blue; }  
  
        /* 클래스 선택자 .c1 : 여러 요소 적용 가능 */  
        .c1 { color: silver; font-family: 돋움, 궁서, serif, "Times New Roman"; }  
  
        /* 아이디 선택자 #myid1 : 단수 적용 */  
        #myid1 { color: cyan; font-style: italic; }  
    </style>  
  
    <!-- 2. External Style -->  
    <link rel="stylesheet" type="text/css" href="css/ex.css" />  
</head>  
<body>  
  
    <pre>  
** 스타일 시트 **  
HTML 문서에 다양한 레이아웃과 꾸미기를 위한 표현 언어 CSS 사용  
선택자 종류:  
1) 태그 선택자 : 특정 태그 직접 선택  
2) .클래스 선택자 : 여러 요소 적용 가능  
3) #아이디 선택자 : 단일 요소 적용  
    </pre>  
  
    <hr>  
  
    <span style="font-size: 30px;color: aqua">인라인 스타일</span> (해당 요소 직접 스타일 적용)  
  
    <h1>인터널 방식(Internal)</h1>  
    <h2 style="color: yellow;">안녕하세요</h2>  
    <h1>반가워요</h1>  
  
    <p class="c1">클래스 사용1 - 중복 가능</p>  
    <h2 class="c1">클래스 사용2</h2>  
    <p id="myid1">아이디 사용 - 중복 불가</p>  
    <div id="myid2">아이디 적용 div</div>  
  
    <p>* 입력 양식 작성 *</p>  
    <form>  
        <table>  
            <tr>  
                <td>아이디 :</td>  
                <td><input type="text" name="name"></td>  
            </tr>  
            <tr>  
                <td>비밀번호 :</td>  
                <td><input type="text" name="pwd"></td>  
            </tr>  
        </table>  
    </form>  
  
    <br>  
    <a href="https://naver.com">네이버</a><br>  
    <a href="https://kbs.co.kr">KBS</a> 공영방송  
    <hr>  
  
    <b>b 태그 적용 예제</b>  
    <div><b>div-b tag</b></div>  
    <div><span><b>div-span-b tag</b></span></div>  
    <div><b id="hi">div-b tag : id-hi</b></div>  
    <div><b class="hello">div-b tag : class-hello</b></div>  
  
</body>  
</html>
```

---
### 🔹 HTML 설명

1. **스타일 적용 방식**
    
    - **Inline**: `<h2 style="color: yellow;">` → 해당 요소만 직접 적용
    - **Internal**: `<style>` → head 내에서 선언, 같은 HTML 파일 안에서 사용
    - **External**: `<link rel="stylesheet" href="ex.css">` → 여러 HTML에서 공유 가능
    
2. **선택자 구분**
    
    - **태그 선택자**: `h1`, `h2`, `strong`
    - **클래스 선택자**: `.c1` → 여러 요소 적용 가능, 반드시 `.` 필요
    - **아이디 선택자**: `#myid1` → 단일 요소만 적용
    
3. **폼 & 동적 스타일**
    
    - `<input>` → 사용자 입력창
    - `input:focus` → 입력창 클릭 시 배경색 변경 (External CSS)
    
4. **b 태그 실습**
    
    - div 내부, 직계 자식, id, class 별로 글자 크기와 색상 다르게 적용
    - 우선순위와 상속 확인 가능

---

## 4-9. ex.css – 외부 CSS

```css
/* ex.css : 외부 스타일 */  
  
/* 1. ID 선택자 */  
#myid2 {  
    background-color: yellow; /* 배경색 */  
    color: red;               /* 글자색 */  
    font-size: 20px;          /* 글자 크기 */  
}  
  
/* 2. 동적 선택자 : input 클릭 시 */  
input:focus {  
    background-color: rgb(100%, 100%, 0%); /* 노란색 강조 */  
}  
  
/* 3. 링크 스타일 */  
a:link { text-decoration: none; }          /* 밑줄 제거 */  
a:hover { color: orange; font-size: 30px; } /* 마우스 올리면 스타일 변경 */  
  
/* 4. b 태그 선택 연습 */  
/* 모든 b 태그 기본 8px */  
b { font-size: 8px; }  
  
/* div 내부 모든 후손 b 태그 16px */  
div b { font-size: 16px; }  
  
/* div 직계 자식 b 태그 20px, 파랑색 */  
div > b { font-size: 20px; color: blue; }  
  
/* div 내부 id="hi" b 태그 30px */  
div b#hi { font-size: 30px; }  
  
/* div 내부 class="hello" b 태그 60px */  
div b.hello { font-size: 60px; }
```

### 🔹 CSS 설명

1. **선택자**
    
    - `#id` → 단일 요소
        
    - `.class` → 여러 요소 가능
        
    - `태그` → 모든 동일 태그
        
    - `div b` → 후손 선택
        
    - `div > b` → 직계 자식만 선택
        
    - `div b#hi` → 후손 + id 선택
        
    - `div b.hello` → 후손 + class 선택
        
2. **의사 클래스**
    
    - `:hover` → 마우스 올렸을 때
        
    - `:focus` → 입력창 클릭 시
        
3. **스타일 속성**
    
    - `color` → 글자색
        
    - `background-color` → 배경색
        
    - `font-size` → 글자 크기
        
    - `font-family` → 글꼴
        
4. **우선순위**
    
    - Inline > ID > Class > Tag
        
    - 직계 선택자 > 후손 선택자
        
5. **b 태그 우선순위 예시**
    

|선택자|글자 크기|색상|
|---|---|---|
|b|8px|기본|
|div b|16px|기본|
|div > b|20px|파랑|
|div b#hi|30px|기본|
|div b.hello|60px|기본|

6. **상속**
    
    - 부모 요소 스타일이 자식 요소로 전달될 수 있음
        
    - 예: div에 color 지정 → div 내부 p, span 등 글자색 상속 가능

---
# 🌐 DOM (Document Object Model) 이해

## 1️⃣ DOM 정의

- **DOM**은 **HTML 문서를 자바스크립트가 이해할 수 있는 객체 형태로 변환한 트리 구조**
    
- 브라우저는 HTML 파일을 로드하면서 **문서를 메모리 안 트리 구조**로 만들어 JS와 CSS가 접근할 수 있게 함
    

즉, HTML = 문서 → 브라우저 = 메모리에서 DOM 트리 → JS/CSS로 조작 가능

---

## 2️⃣ DOM 구조

HTML 문서:

```html
<html>  
  <head>  
    <title>My Page</title>  
  </head>  
  <body>  
    <h1>안녕하세요</h1>  
    <p id="intro">첫 번째 문단</p>  
    <div class="box">  
      <b>굵은 글씨</b>  
      <span>일부 영역</span>  
    </div>  
  </body>  
</html>
```


### 트리 구조 (DOM 관점)

```
document  
│  
├─ html  
   ├─ head  
   │   └─ title : "My Page"  
   └─ body  
       ├─ h1 : "안녕하세요"  
       ├─ p#intro : "첫 번째 문단"  
       └─ div.box  
           ├─ b : "굵은 글씨"  
           └─ span : "일부 영역"
```

- **document** → 최상위 객체
    
- **html** → 루트 엘리먼트
    
- 각 태그 → **노드(node)**
    
- 텍스트 → **텍스트 노드**
    
- 속성(id, class) → **속성 노드**
    

---

## 3️⃣ DOM 노드 종류

|노드 종류|설명|예시|
|---|---|---|
|Element Node|태그 자체|`<div>` `<p>`|
|Text Node|태그 안의 글자|`"안녕하세요"`|
|Attribute Node|태그 속성|`id="intro"` `class="box"`|
|Comment Node|주석|`<!-- html 주석 -->`|

---

## 4️⃣ DOM 접근 방법 (JS 기준)

1. **태그 선택**
    
```javascript
const h1 = document.querySelector("h1");  // 첫 번째 h1  
const intro = document.getElementById("intro"); // id 선택  
const box = document.querySelector(".box"); // class 선택
```

2. **내용 변경**
    
```javascript
intro.textContent = "두 번째 문단";  
h1.innerHTML = "<span>안녕하세요</span>";
```

3. **속성 변경**
	
```javascript
box.setAttribute("data-num", "123"); // div에 data-num 추가
```


4. **스타일 변경**
    
```javascript
h1.style.color = "red";  
box.style.backgroundColor = "yellow";
```

5. **DOM 트리 이동**
    
```javascript
const parent = box.parentNode;  // 부모 div  
const child = box.childNodes;   // 자식 노드 목록  
const first = box.firstChild;   // 첫 번째 자식
```

---

## 5️⃣ DOM 핵심 포인트

1. **트리 구조** → 부모/자식/형제 관계 이해 필수
    
2. **노드 종류** → Element, Text, Attribute, Comment
    
3. **조작 가능** → JS로 내용, 속성, 스타일, 구조 변경 가능
    
4. **동적 반응** → 이벤트와 결합 → 클릭, 입력, 마우스 등
    
5. **CSS 적용** → DOM 노드 기준으로 스타일 적용
    
6. **HTML ↔ DOM ↔ JS/CSS** → 브라우저 내부의 중간 형태
    

---

💡 **비유**

- HTML 파일 = 설계도
    
- DOM = 브라우저가 만든 **실제 건물 구조** (방, 층, 창문 각각 객체화)
    
- JS/CSS = **인테리어와 장식**, 문서 구조를 자유롭게 변경 가능

---
## 1️⃣ MIME 정의

- 원래는 **메일 첨부파일 타입 지정**용으로 만들어짐
    
- 지금은 **HTTP에서 서버가 브라우저에 보내는 데이터 형식 지정**에도 사용됨
    
- 브라우저가 어떤 방식으로 처리할지 MIME 타입 보고 결정
    

---

## 2️⃣ MIME 타입 구조

```
type/subtype
```

예시:

|MIME 타입|설명|
|---|---|
|text/html|HTML 문서|
|text/plain|일반 텍스트|
|text/css|CSS 파일|
|text/javascript|JS 파일|
|image/jpeg|JPEG 이미지|
|image/png|PNG 이미지|
|application/json|JSON 데이터|
|application/pdf|PDF 파일|
|application/octet-stream|알 수 없는 바이너리 파일, 다운로드용|
|multipart/form-data|폼 전송 시 파일 포함|

---

## 3️⃣ HTTP와 MIME

HTTP에서는 **Content-Type** 헤더에 MIME 타입 사용:

```http
HTTP/1.1 200 OK  
Content-Type: text/html; charset=utf-8  

<html>...</html>
```

  
- 서버 → 브라우저에게 “이건 HTML이니까 HTML로 처리해”라고 알려줌
    
- 만약 이미지라면:
    

```http
Content-Type: image/jpeg
```

- 브라우저가 자동으로 렌더링함

---
## 4️⃣ MIME과 CGI 연관

- CGI 프로그램에서 출력 시 **가장 먼저 Content-Type 헤더 출력**해야 함
    
```python
print("Content-Type: text/html; charset=utf-8")  
print()  # 반드시 빈 줄
```

- 이렇게 해야 브라우저가 **어떤 타입으로 해석할지** 알 수 있음
    
- 예: HTML, JSON, 이미지 등 다양하게 처리 가능