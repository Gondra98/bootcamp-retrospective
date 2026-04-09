# Day 20_AJAX 활용 (JSON, CSV, Fetch, 배열 처리)
# 📅 2026-03-04

---

# 1. 전체 코드

## js16ajax.html

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
<meta charset="UTF-8">  
<meta name="viewport" content="width=device-width, initial-scale=1.0">  
<title>Document</title>  
  
<script>  
window.onload = () => {  
    document.querySelector('#btnGet').onclick = getFunc;  
    document.querySelector('#btnPost').onclick = postFunc;  
}  
  
let xhr;  
  
function getFunc(){  
    let irum = frm.name.value;  
    let nai = frm.age.value;  
  
    xhr = new XMLHttpRequest();    
    let fname = "js16.py?irum="+irum+"&nai="+nai;  
  
    xhr.open("GET", fname, true);  
  
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
    let fname = "js16.py";  
  
    xhr.open("POST", fname, true);  
  
    xhr.onreadystatechange = function(){  
        if(xhr.readyState === 4 && xhr.status === 200){  
            process1();  
        }  
    }  
  
    xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");  
    xhr.send("?irum=" + irum + "&nai=" + nai);  
}  
  
function process1(){  
    alert(xhr.responseText);  
}  
</script>  
  
</head>  
<body>  
<h2>Ajax 이해 : get, post</h2>  
<br>Ajax는 비동기 HTTP 통신  
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

## js16.py (서버 응답 예시)

```python
{  
    "status": "success",  
    "name": "tom",  
    "age": "33",  
    "msg": "tom님의 나이는 33세 입니다."  
}
```

---

# 2. Ajax 기본 개념

Ajax는 **비동기 HTTP 통신 방식**이다.

특징:

- 페이지 이동이 없다.
    
- 새로고침 없이 서버와 통신한다.
    
- 필요한 데이터만 받아온다.
    
- 사용자 경험이 향상된다.
    

---

# 3. 전체 동작 흐름

1. 사용자가 버튼 클릭
    
2. JavaScript가 XMLHttpRequest 객체 생성
    
3. 서버(js16.py)로 요청 전송
    
4. 서버가 JSON 데이터 반환
    
5. 브라우저가 응답을 받아 처리
    

---

# 4. 코드 흐름 분석

## 1) 버튼과 함수 연결

```javascript
window.onload = () => {  
    document.querySelector('#btnGet').onclick = getFunc;  
    document.querySelector('#btnPost').onclick = postFunc;  
}
```

- 페이지 로딩 완료 후 실행
    
- GET 버튼 → getFunc()
    
- POST 버튼 → postFunc()
    

submit 버튼이 아니기 때문에  
페이지 이동이 발생하지 않는다.

---

## 2) XMLHttpRequest 객체

```javascript
xhr = new XMLHttpRequest();
```

- Ajax 통신을 담당하는 객체
    
- 서버와 HTTP 통신을 수행
    

이 객체를 통해 GET, POST 요청을 보낸다.

---

# 5. GET 방식 동작 원리

```javascript
let fname = "js16.py?irum="+irum+"&nai="+nai;  
xhr.open("GET", fname, true);  
xhr.send();
```

### 특징

- 데이터를 URL 뒤에 붙인다.
    
- Query String 형태
    

예:

js16.py?irum=tom&nai=33

- 길이 제한이 있다.
    
- 주소창에 데이터가 노출된다.
    
- 간단한 데이터 전송에 사용
    

---

# 6. POST 방식 동작 원리

```javascript
xhr.open("POST", fname, true);  
xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");  
xhr.send("?irum=" + irum + "&nai=" + nai);
```

### 특징

- 데이터를 Body 영역에 포함한다.
    
- URL에 데이터가 보이지 않는다.
    
- 대용량 데이터 전송 가능
    
- 파일 업로드 등에 사용
    

### Content-Type 의미

application/x-www-form-urlencoded

폼 데이터를 서버가 이해할 수 있도록 형식을 지정해주는 역할이다.

---

# 7. 비동기 처리 핵심

```javascript
xhr.onreadystatechange = function(){  
    if(xhr.readyState === 4 && xhr.status === 200){  
        process1();  
    }  
}
```

### readyState

| 값   | 의미      |
| --- | ------- |
| 0   | 초기화     |
| 1   | open 완료 |
| 2   | send 완료 |
| 3   | 응답 수신 중 |
| 4   | 응답 완료   |

### status

|값|의미|
|---|---|
|200|정상 처리|
|404|페이지 없음|
|500|서버 오류|

조건:

readyState === 4 && status === 200

응답 완료 + 정상 성공일 때만 처리한다.

---

# 8. 응답 처리

```javascript
function process1(){  
    alert(xhr.responseText);  
}
```

- 서버가 보낸 데이터를 그대로 출력
    
- 현재는 JSON 문자열 그대로 표시됨
    

실무에서는 보통:

```javascript
let data = JSON.parse(xhr.responseText);
```

JSON 객체로 변환하여 사용한다.

---

# 9. GET vs POST 비교

|구분|GET|POST|
|---|---|---|
|데이터 위치|URL|Body|
|길이 제한|있음|거의 없음|
|보안성|낮음|상대적으로 높음|
|대용량 전송|불가|가능|

---
# 10. js17ajax.html 전체 코드

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
      
    <script>  
        window.onload = () => {  
            document.querySelector('#btn1').onclick = funcJson;  
            document.querySelector('#btn2').onclick = funcCsv;  
        }  
        function funcJson(){  
            xhr = new XMLHttpRequest();           
            xhr.open("GET", "sangpum.json", true);        
            xhr.onreadystatechange = function(){  
                if(xhr.readyState === 4 && xhr.status === 200){  
                    let data = JSON.parse(xhr.responseText);  
                    let output = "JSON 반환 결과<br>";  
                    output += "코드&nbsp;&nbsp;&nbsp;품명<br>";  
                    for(let i = 0; i < data.length; i++){  
                        output += data[i].code + "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + data[i].sang + "<br>";  
                    }  
                    document.querySelector("#result").innerHTML = output;  
                }else{  
                    document.querySelector("#result").innerHTML = "Loading Fail : " + xhr.status;   
                }  
            }  
            xhr.send();  
        }  
        function funcCsv(){  
            xhr = new XMLHttpRequest();           
            xhr.open("GET", "sangpum.csv", true);         
            xhr.onreadystatechange = function(){  
                if(xhr.readyState === 4 && xhr.status === 200){  
                    let text = xhr.responseText.trim();  
                    let lines = text.split("\n");  
                    let output = "CSV 반환 결과<br>";  
                    for(let i = 0; i < lines.length; i++){  
                        if(lines[i] !== ""){  
                            let parts = lines[i].split(",");  
                            let code = parts[0].trim();  
                            let sang = parts[1].trim();  
                            output += "코드 " + code + ", 상품명 " + sang + "<br>";  
                        }  
                    }  
                    document.querySelector("#result").innerHTML = output;  
                }  
            }  
            xhr.send();  
        }  
    </script>  
</head>  
<body>  
    <h3>전통 AJAX 처리</h3>  
    <button id="btn1">JSON 불러오기</button>  
    <button id="btn2">CSV 불러오기</button>  
    <hr>  
    <div id="result"></div>  
</body>  
</html>
```

---

# 11. 전체 동작 구조

이 페이지는 두 가지 파일을 Ajax로 읽는다.

1. sangpum.json → JSON 데이터 처리
    
2. sangpum.csv → CSV 데이터 처리
    

공통 구조:

1. 버튼 클릭
    
2. XMLHttpRequest 객체 생성
    
3. open()으로 파일 요청
    
4. readyState와 status 확인
    
5. 응답 데이터 가공
    
6. div 영역에 출력
    

---

# 12. JSON 처리 흐름 분석

```javascript
function funcJson(){  
    xhr = new XMLHttpRequest();           
    xhr.open("GET", "sangpum.json", true);        
    xhr.onreadystatechange = function(){  
        if(xhr.readyState === 4 && xhr.status === 200){  
            let data = JSON.parse(xhr.responseText);  
            let output = "JSON 반환 결과<br>";  
            output += "코드&nbsp;&nbsp;&nbsp;품명<br>";  
            for(let i = 0; i < data.length; i++){  
                output += data[i].code + "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + data[i].sang + "<br>";  
            }  
            document.querySelector("#result").innerHTML = output;  
        }else{  
            document.querySelector("#result").innerHTML = "Loading Fail : " + xhr.status;   
        }  
    }  
    xhr.send();  
}
```

## 핵심 포인트

- JSON은 구조화된 데이터
    
- JSON.parse()를 사용하면 자바스크립트 객체로 변환된다.
    
- data는 배열 형태이므로 data.length 사용 가능
    
- data[i].code, data[i].sang처럼 속성 접근 가능
    

예시 JSON 구조:

```json
[  
  {"code":"101","sang":"모니터"},  
  {"code":"102","sang":"키보드"}  
]
```

---

# 13. CSV 처리 흐름 분석

```javascript
function funcCsv(){  
    xhr = new XMLHttpRequest();           
    xhr.open("GET", "sangpum.csv", true);         
    xhr.onreadystatechange = function(){  
        if(xhr.readyState === 4 && xhr.status === 200){  
            let text = xhr.responseText.trim();  
            let lines = text.split("\n");  
            let output = "CSV 반환 결과<br>";  
            for(let i = 0; i < lines.length; i++){  
                if(lines[i] !== ""){  
                    let parts = lines[i].split(",");  
                    let code = parts[0].trim();  
                    let sang = parts[1].trim();  
                    output += "코드 " + code + ", 상품명 " + sang + "<br>";  
                }  
            }  
            document.querySelector("#result").innerHTML = output;  
        }  
    }  
    xhr.send();  
}
```
## 핵심 포인트

CSV는 구조화된 객체가 아니다.  
단순 문자열이다.

따라서:

1. 줄 단위로 split("\n")
    
2. 쉼표 기준 split(",")
    
3. 배열 인덱스로 접근


예시 CSV:

```
101,모니터  
102,키보드

↓

["101,모니터", "102,키보드"]

↓

["101", "모니터"]
```

---

# 14. JSON vs CSV 비교

|구분|JSON|CSV|
|---|---|---|
|데이터 구조|객체/배열|단순 문자열|
|처리 방법|JSON.parse()|split()|
|가독성|높음|낮음|
|실무 활용|매우 많음|제한적|

---

# 15. 전통 AJAX 방식 특징

이 코드는 XMLHttpRequest 기반 전통 Ajax 방식이다.

특징:

- readyState, status 직접 확인
    
- 코드 길이가 길다
    
- 에러 처리 수동 작성
    
- 최근에는 fetch() 방식이 더 많이 사용된다.

---
# 16. js18ajax.html 전체 코드

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
    <!-- axios cdn -->  
    <script src= "https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>  
</head>  
<body>  
    <h3>AJAX 3가지 방식 비교</h3>  
    <button id="btn1">1) fetch then</button>  
    <button id="btn2">2) async fetch</button>  
    <button id="btn3">3) axios</button>  
    <hr>  
    <div id="result"></div>  
  
    <script>  
        document.querySelector("#btn1").onclick = funcFetch;  
        document.querySelector("#btn2").onclick = funcAsync;  
        document.querySelector("#btn3").onclick = funcAxios;  
  
        // 1. fetch + then 방식  
        function funcFetch(){  
            fetch("sangpum.json")  
                .then(response => {  
                    if(!response.ok){  
                        throw new Error("서버 오류");  
                    }  
                    return response.json();  
                })  
                .then(data => {  
                    let output = "<b>fetch then 결과</b><br>";  
                    data.forEach(item => {  
                        output += "코드:" + item.code + " / 품명:" + item.sang + "<br>";  
                    });  
                    document.getElementById("result").innerHTML = output;  
                })  
                .catch(error => {  
                    document.getElementById("result").innerHTML = "에러 발생 : " + error;  
                });  
        }  
  
        // 2. async + await fetch 방식  
        async function funcAsync(){  
            try{  
                const response = await fetch("sangpum.json")  
  
                if(!response.ok){  
                        throw new Error("서버 오류");  
                    }  
                  
                const data = await response.json();  
  
                let output = "<b>fetch then 결과</b><br>";  
                data.forEach(item => {  
                    output += "코드:" + item.code + " / 품명:" + item.sang + "<br>";  
                });  
                document.getElementById("result").innerHTML = output;  
  
            }catch(error){  
                document.getElementById("result").innerHTML = "에러 발생 : " + error;  
            }  
        }  
  
        // 3. axios 방식  
        async function funcAxios(){  
            try{  
                const response = await axios.get("sangpum.json");  
                const data = response.data;     // json 처리를 해줌  
  
                let output = "<b>axios 결과</b><br>";  
                data.forEach(item => {  
                    output += "코드:" + item.code + " / 품명:" + item.sang + "<br>";  
                });  
                document.getElementById("result").innerHTML = output;  
  
            }catch(error){  
                document.getElementById("result").innerHTML = "에러 발생 : " + error;  
            }  
        }  
  
    </script>  
  
</body>  
</html>
```

---

# 17. AJAX 3가지 방식 비교 개요

이 페이지는 같은 JSON 파일(sangpum.json)을  
3가지 방법으로 불러온다.

1. fetch + then
    
2. async + await fetch
    
3. axios
    

모두 비동기 처리 방식이다.

---

# 18. fetch + then 방식

```javascript
function funcFetch(){  
    fetch("sangpum.json")  
        .then(response => {  
            if(!response.ok){  
                throw new Error("서버 오류");  
            }  
            return response.json();  
        })  
        .then(data => {  
            let output = "<b>fetch then 결과</b><br>";  
            data.forEach(item => {  
                output += "코드:" + item.code + " / 품명:" + item.sang + "<br>";  
            });  
            document.getElementById("result").innerHTML = output;  
        })  
        .catch(error => {  
            document.getElementById("result").innerHTML = "에러 발생 : " + error;  
        });  
}
```
## 핵심 개념

- fetch()는 Promise 객체를 반환한다.
    
- then()으로 성공 시 처리.
    
- catch()로 에러 처리.
    

### 처리 흐름

1. fetch 요청
    
2. 응답 객체(response) 반환
    
3. response.json()으로 JSON 변환
    
4. 두 번째 then에서 데이터 사용
    
5. catch에서 예외 처리
    

---

# 19. async + await fetch 방식

```javascript
async function funcAsync(){  
    try{  
        const response = await fetch("sangpum.json")  
  
        if(!response.ok){  
                throw new Error("서버 오류");  
            }  
          
        const data = await response.json();  
  
        let output = "<b>fetch then 결과</b><br>";  
        data.forEach(item => {  
            output += "코드:" + item.code + " / 품명:" + item.sang + "<br>";  
        });  
        document.getElementById("result").innerHTML = output;  
  
    }catch(error){  
        document.getElementById("result").innerHTML = "에러 발생 : " + error;  
    }  
}
```

## 핵심 개념

- async 함수는 Promise를 반환한다.
    
- await는 Promise가 끝날 때까지 기다린다.
    
- try-catch로 에러 처리.
    

### 장점

- 코드가 동기식처럼 보인다.
    
- 가독성이 좋다.
    
- 복잡한 비동기 처리에 유리하다.
    

---

# 20. axios 방식

```javascript
async function funcAxios(){  
    try{  
        const response = await axios.get("sangpum.json");  
        const data = response.data;  
  
        let output = "<b>axios 결과</b><br>";  
        data.forEach(item => {  
            output += "코드:" + item.code + " / 품명:" + item.sang + "<br>";  
        });  
        document.getElementById("result").innerHTML = output;  
  
    }catch(error){  
        document.getElementById("result").innerHTML = "에러 발생 : " + error;  
    }  
}
```

## 핵심 개념

- axios는 외부 라이브러리이다.
    
- JSON 변환을 자동으로 처리한다.
    
- response.data로 바로 사용 가능.


CDN 추가 부분:

```html
<script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
```


이 스크립트가 있어야 axios 사용 가능하다.

---

# 21. 3가지 방식 비교

|구분|fetch then|async/await|axios|
|---|---|---|---|
|문법|Promise 체인|동기식처럼 표현|라이브러리|
|가독성|보통|좋음|매우 좋음|
|JSON 처리|직접 변환|직접 변환|자동 변환|
|에러 처리|catch|try-catch|try-catch|

---
# 22. js19buser.html – 직원 부서별 검색 (Ajax + JSON 실전 예제)

## 22-1. js19buser.html 전체 코드
```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
    <style>  
        .error {color:red; margin-top:10px;}  
        table{  
            border-collapse:collapse;       /*table의 테두리 함치기*/  
            margin-top: 15px;  
            width: 500px;  
        }  
  
        th, td{  
            border: 1px solid #333;  
            text-align: center;  
        }  
  
        th {background-color: #eee;}  
    </style>  
</head>  
<body>  
    <h3>직원 부서별 검색(Ajax)</h3>  
    <input type="text" id="deptKeyword" placeholder="부서명 입력">  
    <button id="btnSearch">검색</button>  
  
    <div id="loading"></div>  
    <div id="message"></div>  
    <div id="result"></div>  
  
    <script>  
        document.getElementById("btnSearch").addEventListener("click", loadJikwons);  
  
        async function loadJikwons(){  
            // alert("A")  
            const keyword = document.getElementById("deptKeyword").value.trim();  
            const loading = document.getElementById("loading");  
            const message = document.getElementById("message");  
            const resultDiv = document.getElementById("result");  
  
            loading.innerHTML = "데이터 로딩 중";  
            message.innerHTML = "";  
            resultDiv.innerHTML = "";  
  
            try{  
                const response = await fetch("js19jikwon.json");  
                // alert(response.ok);  
                if(!response.ok){  
                    throw new Error("서버 오류 발생");  
                }  
  
                const jsonData = await response.json();  
                  
                if (jsonData.status !== "success"){  
                    throw new Error("서버 응답 오류 발생");  
                }  
  
                const jikwons = jsonData.data;  
  
                // 부서 필터링  
                const filtered = jikwons.filter(jik =>   
                    jik.dept.includes(keyword)  
                );  
  
                if (filtered.length === 0){  
                    loading.innerHTML = "";  
                    message.innerHTML = "<b>검색 결과가 없어요</b>";  
                    return;  
                }  
  
                renderTableFunc(filtered);  
  
                // 검색 인원수와 연봉 평균 - reduce() : 배열 축소  
                // filtered 배열에 있는 모든 직원의 연봉을 더해서 인원수로 나눔.  
                const avgSalary = filtered.reduce((sum, emp) => sum + emp.salary, 0) / filtered.length;  
  
                message.innerHTML = "검색 인원 : " + filtered.length + " / 평균 연봉 : " + avgSalary.toFixed(2);        // toFixed(2) : 소수점 이하 둘째자리까지 표시  
                loading.innerHTML = "";  
  
            }catch(error){  
                loading.innerHTML = "";  
                message.innerHTML = "<span class='error'>" + error.message + "</span>";  
            }  
        }  
  
        function renderTableFunc(data){  
            let table = "<table>";  
            table += "<tr><th>ID</th><th>이름</th><th>부서</th><th>연봉</th></tr>";  
            data.forEach(emp => {  
                table += "<tr>";  
                table += "<td>" + emp.id + "</td>";  
                table += "<td>" + emp.name + "</td>";  
                table += "<td>" + emp.dept + "</td>";  
                table += "<td>" + emp.salary + "</td>";  
                table += '</tr>';  
            });  
            table += "</table>";  
  
            document.getElementById("result").innerHTML = table;  
        }  
    </script>  
</body>  
</html>
```

---

# 22-2. js19jikwon.json

```json
{  
    "status":"success",  
    "count":5,  
    "data":[  
        {"id":1, "name":"홍길동", "dept":"영업부", "salary":6500},  
        {"id":2, "name":"이순신", "dept":"총무부", "salary":4200},  
        {"id":3, "name":"강감찬", "dept":"전산부", "salary":5900},  
        {"id":4, "name":"강나루", "dept":"인사부", "salary":6700},  
        {"id":5, "name":"신선해", "dept":"영업부", "salary":7700}  
    ]  
}
```

### JSON 구조 분석

- status : 서버 처리 결과
    
- count : 전체 인원수
    
- data : 실제 직원 배열
    

data는 배열이며, 각 요소는 객체이다.

---
# 22-3. 실행 흐름 정리

1. 검색 버튼 클릭
    
2. loadJikwons() 실행
    
3. fetch로 JSON 요청
    
4. response.ok 검사
    
5. JSON 파싱
    
6. status 검사
    
7. filter로 부서 검색
    
8. 결과 없으면 메시지 출력
    
9. 결과 있으면 테이블 출력
    
10. reduce로 평균 연봉 계산
    

---
# 22-4. 핵심 문법 설명

## 1) 입력값 가져오기

```javascript
const keyword = document.getElementById("deptKeyword").value.trim();
```

trim()  
→ 앞뒤 공백 제거

---

## 2) fetch 요청

```javascript
const response = await fetch("js19jikwon.json");
```

- Promise 반환
    
- await로 동기처럼 사용
    

---

## 3) 서버 상태 검사

```javascript
if(!response.ok){  
    throw new Error("서버 오류 발생");  
}
```

HTTP 상태코드가 200이 아니면 에러 발생

---

## 4) JSON 파싱

```javascript
const jsonData = await response.json();
```

문자열 → 객체 변환

---

## 5) filter() – 조건 검색

```javascript
const filtered = jikwons.filter(jik =>   
    jik.dept.includes(keyword)  
);
```

- 배열에서 조건 만족하는 요소만 추출
    
- includes() → 부분 문자열 검색
    

---

## 6) reduce() – 평균 계산

```javascript
const avgSalary = filtered.reduce((sum, emp) => sum + emp.salary, 0) / filtered.length;
```

reduce 구조:

- sum : 누적값
    
- emp.salary 더하기
    
- 초기값 0
    

---

## 7) toFixed(2)

```javascript
avgSalary.toFixed(2)
```

소수점 둘째 자리까지 표시

---

## 8) 동적 테이블 생성

```javascript
function renderTableFunc(data)
```

- 문자열로 table 태그 생성
    
- forEach로 행 추가
    
- innerHTML로 화면 출력
    

---

# 23. sangpum.json

```json
[  
    {"code":"1","sang":"안경"},  
    {"code":"2","sang":"가방"}  
]
```

이 파일은 배열 구조만 존재한다.  
status 같은 래핑 객체가 없다.

접근 방법 예:

```javascript
data[0].code  
data[0].sang
```

---

# 24. sangpum.csv

```csv
1,물안경  
2,손가방
```

CSV 특징:

- 쉼표로 구분
    
- JSON.parse 사용 불가
    
- 직접 split()으로 나눠야 함
    

예:

```javascript
let lines = text.split("\n");  
let cols = lines[i].split(",");
```

---
