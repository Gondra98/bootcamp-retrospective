# Day 19_자바스크립트 함수와 이벤트, DOM, Ajax 기초
# 📅 2026-03-03

---
# 1️⃣ ES6 화살표 함수 (Arrow Function)

## 1-1. 개념

ES6부터는 `function` 키워드 대신  
**화살표(=>)** 를 사용하여 함수를 만들 수 있다.

✔ 더 짧은 문법  
✔ 함수 표현식을 간결하게 작성 가능  
✔ 코드 가독성 향상

---

# 2️⃣ 기존 함수 vs 화살표 함수 비교

## 2-1. 일반 함수 선언

```javascript
function sum0(a, b){  
    return a + b;  
}
```

---

## 2-2. 함수 표현식

```javascript
let sum1 = function(a, b){  
    return a + b;  
}
```

---

## 2-3. 화살표 함수 (기본형)

```javascript
let sum2 = (a, b) => a + b;
```

✔ `function` 생략  
✔ `{}` 생략 가능 (한 줄일 경우)  
✔ `return` 자동 포함

---

## 2-4. 중괄호 사용하는 경우

```javascript
let sum3 = (a, b) => {  
    return a + b;  
}
```

✔ 여러 줄일 경우 `{}` 사용  
✔ 이때는 반드시 `return` 작성

---

# 3️⃣ 매개변수 규칙

## 3-1. 매개변수 1개일 경우

괄호 생략 가능

```javascript
let double1 = n => n * 2;
```


---

## 3-2. 매개변수 없을 경우

반드시 괄호 필요

```javascript
let sayHi = () => document.write("안녕하세요");
```

---

# 4️⃣ 삼항 연산자 + 화살표 함수

```javascript
let age = 25;  
  
let welcome = (age < 30)  
    ? () => document.write("안녕")  
    : () => document.write("반갑습니다.");
```

✔ 조건에 따라 다른 함수를 저장  
✔ 함수 자체를 변수에 넣을 수 있다

👉 함수도 하나의 값이다 (중요)

---

# 5️⃣ 블록 사용 예제

```javascript
let sumFunc = (a, b) => {  
    let result = a + b;  
    return result;  
}
```

✔ 여러 줄 작성 가능  
✔ 내부 변수 선언 가능

---

# 6️⃣ 화살표 함수 특징 정리

|구분|일반 함수|화살표 함수|
|---|---|---|
|키워드|function 필요|불필요|
|문법|길다|짧다|
|return|반드시 작성|한 줄이면 생략 가능|
|this|자신만의 this 가짐|자신만의 this 없음 (상위 this 사용)|

※ this 부분은 나중에 더 배울 가능성 있음

---
7. 이벤트 처리 함수 전체 코드 (js10event.html)

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
    <script>  
        function func(){  
            // 이벤트 처리용 함수  
            let dateTime = "2026-3-3"  
            let arrData = dateTime.split("-")  
            console.log(`${arrData[0]}년 ${arrData[1]}월 ${arrData[2]}일`)  
        }  
    </script>  
</head>  
<body>  
    <h2>이벤트 처리 이해</h2>  
    <a href="https://www.naver.com">네이버</a><br>  
    <a href="javascript:func()">이벤트 처리 함수 호출(이벤트 핸들러 X)</a>  
    <br>이벤트 핸들러는 on이벤트종료=핸들러함수()  
    <br>  
    <a href="javascript:onclick=func()">이벤트 처리 함수 호출(이벤트 핸들러 O)</a>  
    <br>  
    <button onclick="func()">이벤트 처리 함수 호출3</button>  
    <br>  
    <input type="button">이벤트 처리 함수 호출3</button>  
</body>  
</html>
```


설명:

- func()는 이벤트 발생 시 실행되는 함수
    
- split("-")으로 문자열을 배열로 분리
    
- 배열 값을 이용해 날짜 형식으로 출력
    
- onclick 속성을 통해 이벤트와 함수 연결
    

---

8. 색상 변경 이벤트 전체 코드 (js11event.html)
    
```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
    <script type="text/javascript">  
        function colorFunc(){  
            document.body.bgColor = document.frm.bc.value;  
            document.body.text = document.frm.tc.value;  
        }  
    </script>  
</head>  
<body>  
    이벤트 연습 : 색 변경<br>  
    <form name="frm">  
        <table border="1">  
            <tr>  
                <td colspan="2" style="text-align: center;">색상 선택</td>  
            </tr>  
            <tr>  
                <td><b>배경색 : </b></td>  
                <td>  
                    <select name="bc">  
                        <option value="black">검정</option>  
                        <option value="blue">파랑</option>  
                        <option value="green">초록</option>  
                        <option value="white">하양</option>  
                    </select>  
                </td>  
            </tr>  
            <tr>  
                <td><b>글자색 : </b></td>  
                <td>  
                    <select name="tc">  
                        <option value="black" selected>검정</option>  
                        <option value="blue">파랑</option>  
                        <option value="green">초록</option>  
                        <option value="white">하양</option>  
                    </select>  
                </td>  
            </tr>  
            <tr>  
                <td colspan="2" style="text-align: center;">  
                    <input type="button" value="변 경" onclick="colorFunc()">  
                </td>  
            </tr>  
        </table>  
    </form>  
</body>  
</html>
```


설명:

- form name="frm"으로 폼 객체 지정
    
- select name="bc" → 배경색 선택
    
- select name="tc" → 글자색 선택
    
- onclick="colorFunc()"으로 버튼 클릭 시 함수 실행
    
- document.frm.bc.value로 선택값 읽기
    
- document.body 속성 변경으로 화면 즉시 반영
    

---

9. 이벤트 처리 핵심 정리
    

- 이벤트는 사용자 행동이다.
    
- 이벤트 핸들러는 on이벤트명="함수()" 형식이다.
    
- split()은 문자열을 배열로 분리한다.
    
- form은 name으로 접근 가능하다.
    
- value로 선택값을 읽는다.
    
- document.body를 통해 화면을 직접 변경할 수 있다.

---

10. innerHTML 전체 코드 (js12inner.html)
    
```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
    <script>  
        function func1(){  
            // alert("a");  
            // let str1 = document.getElementById("test1").innerHTML;  
            let str1 = document.querySelector("#test1").innerHTML;  
            console.log(str1);  
            document.querySelector("#show1").innerHTML = str1;  
  
            let tag1 = "이름:<input type='text' name='irum'>";  
            let tag2 = "나이:<input type='text name=nai'>";  
            document.querySelector("#test1").innerHTML = tag1 + "<br>" + tag2;  
        }  
  
        function func2(){  
            // alert("b");  
  
            let str2 = document.querySelector("#test2").innerHTML;  
            console.log(str2);  
            document.querySelector("#show2").innerHTML = str2;  
        }  
    </script>  
</head>  
<body>  
    ** 실행 중 tag의 생성, 삭제, 데이터의 참조, 추가, 수정, 삭제 처리 가능 명령 **<br>  
    element.innerText : 이 속성은 element 안의 text 값들만 대상으로 처리가  
    <br>  
    element.innerHTML : element 안의 텍스트 뿐 아니라 HTML이나 XML을 대상으로 처리가  
    <hr>  
    <div id="test1"><h2>innerHTML 연습</h2></div>  
    <div id="test2"><h2>안녕하세요</h2></div>  
    <input type="button" value="처리1" onclick="func1()">  
    <input type="button" value="처리2" onclick="func2()">  
    <br>  
    <div id="show1">출력 1</div>  
    <div id="show2">출력 2</div>  
</body>  
</html>
```

---

11. innerText와 innerHTML 차이

	1. element.innerText
	- 요소 내부의 텍스트만 처리
	- HTML 태그는 문자 그대로 출력

	2. element.innerHTML

	- 요소 내부의 HTML 구조까지 처리
	- 태그를 실제 태그로 인식하여 화면에 반영

innerHTML은 HTML 구조 자체를 변경할 수 있다.

---

12. func1() 동작 과정 상세 분석
    
```javascript
function func1(){  
    let str1 = document.querySelector("#test1").innerHTML;  
    console.log(str1);  
    document.querySelector("#show1").innerHTML = str1;  
  
    let tag1 = "이름:<input type='text' name='irum'>";  
    let tag2 = "나이:<input type='text name=nai'>";  
    document.querySelector("#test1").innerHTML = tag1 + "<br>" + tag2;  
}
```

1단계. 기존 내용 읽기

- id="test1" 요소 선택
    
- 내부 HTML 내용 저장
    
- 현재 값: `<h2>innerHTML 연습</h2>`
    

2단계. 읽은 내용 출력

- show1 영역에 동일하게 출력
    

3단계. 새로운 태그 문자열 생성

- 입력창 2개를 문자열로 생성
    

4단계. 기존 내용 교체

- test1 영역 내부 전체가 새 HTML로 변경
    
- 기존 h2 태그는 삭제됨
    

중요: innerHTML에 값을 다시 대입하면 기존 내용은 완전히 교체된다.

---

13. func2() 동작 과정
    
```javascript
function func2(){  
    let str2 = document.querySelector("#test2").innerHTML;  
    console.log(str2);  
    document.querySelector("#show2").innerHTML = str2;  
}
```

1단계. test2 내부 내용 읽기  
2단계. show2에 동일하게 출력

HTML 구조가 그대로 복사되어 표시된다.

---

14. querySelector 설명
    
```javascript
document.querySelector("#test1")
```

- CSS 선택자 방식 사용
    
- # → id 선택
    
- . → class 선택 가능
    

기존 방식:

```javascript
document.getElementById("test1")
```

querySelector는 CSS 문법을 그대로 사용하므로 더 유연하다.


---

15. js13event.html
    
```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
<meta charset="UTF-8">  
<meta name="viewport" content="width=device-width, initial-scale=1.0">  
<title>Document</title>  
  
<script>  
function func1(){  
    document.getElementById("show").innerHTML = "전통적인 이벤트 처리";  
}  
  
let a = 10;  
  
let b = {  
    funcNew1:function(){  
        document.getElementById("show").innerHTML = "b 객체의 메소드 처리";  
    },  
    funcNew2:function(){  
        document.getElementById("show").innerHTML = "<h1>b 객체의 멤버 수행</h1>";  
    }  
}  
</script>  
  
<script>  
window.onload = function(){  
  
    const exBtn1 = document.getElementById("btn1");  
    exBtn1.onclick = function(){  
        document.getElementById("show").innerHTML = "클릭1 선택";  
    }  
  
    const exBtn2 = document.getElementsByName("myBtn");  
    exBtn2[0].onclick = function(){  
        document.getElementById("show").innerHTML = "<i>클릭2 누름</i>";  
    }  
    exBtn2[1].onclick = function(){  
        document.getElementById("show").innerHTML = "<i>클릭3 누름</i>";  
    }  
  
    const exBtn4 = document.getElementsByTagName("button")[3];  
    exBtn4.onclick = function(){  
        document.getElementById("show").innerHTML = "<b>클릭4 선택함</b>";  
    }  
  
    document.getElementById("btn5").addEventListener("click", abcFunc);  
  
    function abcFunc(){  
        document.getElementById("show").innerHTML = "클릭5 이벤트 연결 처리";  
        document.getElementById("btn5").removeEventListener("click", abcFunc);  
    }  
}  
</script>  
  
</head>  
<body>  
  
<h2>이벤트 처리 : 이벤트 핸들러 연결</h2>  
  
<input type="button" value="버튼1" onclick="func1()"><br>  
<input type="button" value="버튼2" onclick="b.funcNew1()"><br>  
<input type="button" value="버튼3" onclick="b.funcNew2()"><br>  
<hr>  
  
<button id="btn1">클릭1</button><br>  
<button id="btn2" name="myBtn">클릭2</button><br>  
<button id="btn3" name="myBtn">클릭3</button><br>  
<button id="btn4">클릭4</button><br>  
<button id="btn5">클릭5</button><br>  
  
<div id="show"></div>  
  
</body>  
</html>
```


설명

1. func1()은 전통적인 inline 이벤트 방식이다.
    
2. b 객체는 객체 내부 메소드를 이용한 이벤트 호출 예시이다.
    
3. window.onload는 문서 로딩 완료 후 실행된다.
    
4. getElementById는 단일 요소 선택.
    
5. getElementsByName은 배열로 반환된다.
    
6. getElementsByTagName도 배열 반환이며 index 접근 필요.
    
7. addEventListener는 가장 권장되는 이벤트 연결 방식이다.
    
8. removeEventListener는 동일한 함수 참조를 전달해야 제거 가능하다.
    

---

16. js14event.html
    
```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
<meta charset="UTF-8">  
<meta name="viewport" content="width=device-width, initial-scale=1.0">  
<title>Document</title>  
  
<script>  
window.onload = () => {  
  
    document.getElementById("colorW").onclick = function(){  
        document.body.style.backgroundColor = "white";  
    }  
  
    document.getElementById("colorG").onclick = () => {  
        document.body.style.backgroundColor = "#00ff00";  
    }  
  
    document.querySelector("#colorR").onclick = () =>{  
        document.body.style.backgroundColor = "rgb(255,0,0)";  
    }  
  
    document.querySelector(".hello").onclick = () =>{  
        document.querySelector("#show").innerText = "안녕하세요";  
    }  
  
    document.querySelector("#calc").ondblclick = () =>{  
        let ss = `연산 결과는 ${10 + 20}`;  
        document.querySelector("#show").innerText = ss;  
    }  
  
    document.querySelector("#name").onmouseover = () =>{  
        document.querySelector("#show").innerHTML = "<h2>난 홍길동</h2>";  
    }  
  
    document.querySelector("#daum").onclick = function(event){  
        event.preventDefault();  
        document.title = "새로운 제목";  
    }  
  
    const sdata = document.querySelector("#sel");  
    sdata.onchange = function(){  
        document.querySelector("#show").innerHTML = sdata.value;  
    }  
}  
</script>  
  
</head>  
<body>  
  
<input type="button" value="흰색" id="colorW">  
<span id="colorG">녹색</span>  
<b id="colorR" style="border:2px solid black; cursor:pointer;">적색</b>  
  
<hr>  
  
<button class="hello">인사하기(click)</button>  
<button id="calc">계산하기(dblclick)</button>  
<button id="name">이름알기(mouseover)</button>  
  
<br>  
<a href="https://m.daum.net" id="daum">앵커 태그 클릭</a>  
  
<br>  
<select id="sel">  
<option value="프로그래머">프로그래머</option>  
<option value="웹퍼블리셔">웹퍼블리셔</option>  
<option value="프로게이머">프로게이머</option>  
</select>  
  
<hr>  
<div id="show">출력장소</div>  
  
</body>  
</html>
```

설명

1. onclick, ondblclick, onmouseover 등 다양한 이벤트 사용.
    
2. document.body.style.backgroundColor 방식이 권장.
    
3. 템플릿 문자열을 이용한 계산 결과 출력.
    
4. preventDefault()는 a 태그 기본 이동 기능을 차단.
    
5. select는 onchange 이벤트를 사용해야 선택 변경 감지 가능.
    
6. querySelector는 CSS 선택자 방식으로 요소 선택 가능.

---

1. js15errcheck.html
    
```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
<meta charset="UTF-8">  
<meta name="viewport" content="width=device-width, initial-scale=1.0">  
<title>Document</title>  
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">  
  
<script>  
window.onload = () => {  
    document.querySelector("#btnSend").onclick = chkDataFunc;  
    document.querySelector("#btnClear").onclick = clsDataFunc;  
}  
  
function chkDataFunc(event){  
    event.preventDefault();     // submit 기능 해제  
  
    if(frm.name.value === "" || isNaN(frm.name.value) === false){  
        frm.name.focus();  
        alert("이름을 입력하세요");  
        return;  
    }  
  
    if(frm.id.value.length < 3){  
        frm.id.focus();  
        alert("아이디는 3글자 이상 입력하세요");  
        return;  
    }  
  
    let email_form = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/i;  
    if(!frm.email.value.match(email_form)){  
        frm.email.focus();  
        alert("email 형식에 맞게 입력하세요");  
        return;  
    }  
  
    let age_form = /^[0-9]{1,2}$/;  
    if(!frm.age.value.match(age_form)){  
        frm.age.focus();  
        alert("나이 형식에 맞게 입력하세요");  
        return;  
    }  
  
    frm.action = "my.py";  
    frm.method = "get";  
    frm.submit();  
}  
  
function clsDataFunc(){  
    document.querySelector("#name").focus();  
}  
</script>  
</head>  
  
<body>  
** 개인 자료 입력 **<br>  
  
<form name="frm">  
<table class="table table-dark table-hover">  
<tr>  
    <td>이름</td>  
    <td><input type="text" name="name" id="name" value="홍길동"></td>  
</tr>  
<tr>  
    <td>아이디</td>  
    <td><input type="text" name="id" id="id" placeholder="3글자 이상"></td>  
</tr>  
<tr>  
    <td>이메일</td>  
    <td><input type="text" name="email" id="email"></td>  
</tr>  
<tr>  
    <td>나이</td>  
    <td><input type="text" name="age" id="age"></td>  
</tr>  
<tr>  
    <td colspan="2" style="text-align: center;">  
        <input type="submit" id="btnSend" value="자료 전송" class="btn btn-primary">  
        <input type="reset" id="btnClear" value="자료 삭제" class="btn btn-info">  
    </td>  
</tr>  
</table>  
</form>  
  
</body>  
</html>
```

설명

1. Bootstrap 5.3.3 CDN을 사용하여 테이블과 버튼 스타일 적용.
    
2. window.onload에서 버튼 이벤트 연결.
    
3. submit 버튼 클릭 시 chkDataFunc 실행.
    
4. event.preventDefault()로 기본 submit 기능 차단.
    
5. 이름 검사
    
    - 공백이거나 숫자일 경우 오류 처리
        
    - isNaN()으로 숫자 여부 판단.
        
6. 아이디 검사
    
    - length 속성으로 3글자 이상 확인.
        
7. 이메일 검사
    
    - 정규표현식 사용
        
    - 아이디@도메인.확장자 형식 체크.
        
8. 나이 검사
    
    - 숫자 1~2자리만 허용.
        
9. 모든 검사를 통과하면
    
    - action과 method 설정 후 submit 실행.
        
10. reset 버튼 클릭 시 clsDataFunc 실행
    
    - 입력 초기화 후 name 입력란에 focus 이동.

---
18. js16ajax.html
    
```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
<meta charset="UTF-8">  
<meta name="viewport" content="width=device-width, initial-scale=1.0">  
<title>Document</title>  
  
<script>  
window.onload = () => {  
    document.querySelector("#btnGet").onclick = getFunc;  
    document.querySelector("#btnPost").onclick = postFunc;  
}  
  
let xhr;  
  
function getFunc(){  
    let irum = frm.name.value;  
    let nai = frm.age.value;  
  
    xhr = new XMLHttpRequest();   // Ajax 통신 객체 생성  
  
    let fname = "js.py?irum=" + irum + "&nai=" + nai;  
    xhr.open("GET", fname, true);   // true = 비동기 방식  
  
    xhr.onreadystatechange = function(){  
        if(xhr.readyState === 4 && xhr.status === 200){  
            process1();  
        }  
    }  
  
    xhr.send();  
}  
  
function postFunc(){  
    let irum = frm.name.value;  
    let nai = frm.age.value;  
  
    xhr = new XMLHttpRequest();  
    xhr.open("POST", "js.py", true);  
    xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");  
  
    xhr.onreadystatechange = function(){  
        if(xhr.readyState === 4 && xhr.status === 200){  
            process1();  
        }  
    }  
  
    xhr.send("irum=" + irum + "&nai=" + nai);  
}  
  
function process1(){  
    alert(xhr.responseText);  
}  
</script>  
  
</head>  
<body>  
  
<h2>Ajax 이해 : get, post</h2>  
  
<form name="frm">  
이름 : <input type="text" name="name" value="tom"><br>  
나이 : <input type="text" name="age" value="33"><br>  
<input type="button" value="get 방식" id="btnGet">  
<input type="button" value="post 방식" id="btnPost">  
</form>  
  
</body>  
</html>
```

---

설명

1. Ajax란  
    페이지 전체를 새로고침하지 않고 서버와 비동기 통신하는 기술이다.
    
2. XMLHttpRequest 객체  
    서버와 통신하기 위한 핵심 객체이다.
    
```javascript
xhr = new XMLHttpRequest();
```
    
    
3. GET 방식

```javascript
xhr.open("GET", "js.py?irum=tom&nai=33", true);
```
    
    - 데이터가 URL 뒤에 붙는다.
        
    - 주소창에 데이터 노출
        
    - 길이 제한 존재
        
    - 조회용에 적합
        
4. POST 방식

```javascript
xhr.open("POST", "js.py", true);  
    xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");  
    xhr.send("irum=tom&nai=33");
```
    
    - 데이터가 body에 담겨 전송
        
    - 주소창에 노출되지 않음
        
    - 길이 제한 적음
        
    - 저장/수정 처리에 적합
        
5. readyState 값
    
    0 → 객체 생성  
    1 → open 호출  
    2 → send 호출  
    3 → 데이터 수신 중  
    4 → 수신 완료
    
6. status 코드
    
    200 → 정상 응답  
    404 → 파일 없음  
    500 → 서버 오류
    
7. 원본 코드 오류 부분
    
    기존 코드에는 process1()이 두 번 선언되어 있었다.

```javascript
   function process1(){  
        alert(xhr.responseText);  
    }  
      
    function process1(){  
    }
```
    
    → 아래 함수가 위 함수를 덮어씀  
    → alert가 실행되지 않음
    
    반드시 하나만 사용해야 한다.