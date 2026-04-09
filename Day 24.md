# Day 24_
# 📅 2026-03-10
---
# SPA / MPA 차이

## 1. MPA (Multi Page Application)

### 개념

여러 개의 HTML 페이지로 구성된 웹 애플리케이션  
페이지 이동 시 **서버에서 새로운 HTML을 받아와 전체 페이지가 다시 로드됨**

### 동작 방식

사용자 요청 → 서버 → 새로운 HTML 페이지 반환 → 브라우저 전체 새로고침

### 특징

- 페이지 이동 시 전체 페이지 새로고침 발생
    
- 서버에서 HTML을 생성하여 전달
    
- 전통적인 웹사이트 구조
    

### 장점

- SEO(검색엔진 최적화)에 유리
    
- 구조가 비교적 단순
    
- 서버 중심 개발 가능
    

### 단점

- 페이지 이동 시 속도가 느릴 수 있음
    
- 매번 전체 페이지를 다시 로드해야 함
    

### 예

- 전통적인 쇼핑몰
    
- 게시판 사이트
    
- 기업 홈페이지
    

---

# 2. SPA (Single Page Application)

### 개념

하나의 HTML 페이지에서 **JavaScript가 동적으로 화면을 변경하는 방식**

### 동작 방식

처음: HTML 1번 로드  
이후: 서버(API)에서 데이터만 받아 화면을 변경

### 특징

- 페이지 새로고침 없이 화면 변경
    
- 클라이언트(JavaScript)가 화면 제어
    
- 앱처럼 동작하는 웹
    

### 장점

- 사용자 경험(UX)이 빠르고 부드러움
    
- 페이지 전환 속도가 빠름
    
- 앱과 비슷한 인터페이스 구현 가능
    

### 단점

- 초기 로딩이 무거울 수 있음
    
- SEO 처리 어려움
    
- JavaScript 의존도 높음
    

### 예

- Gmail
    
- Google Maps
    
- React / Vue 기반 웹사이트
    

---

# 3. SPA vs MPA 비교

|구분|SPA|MPA|
|---|---|---|
|페이지 구조|하나의 페이지|여러 페이지|
|화면 전환|데이터만 변경|전체 페이지 새로고침|
|속도|빠름|상대적으로 느림|
|개발 방식|JavaScript 중심|서버 중심|
|대표 기술|React, Vue|JSP, Flask, Django|

---

# 4️⃣ index.html (화면)

역할

- 사용자 입력 받기
    
- JS 연결
    
- 결과 출력
    

전체 코드
```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
  
    <!--  
    <link rel="stylesheet" href="{{url_for('static', filename='css/style.css')}}">  
    -->  
  
    <link rel="stylesheet" href="/static/css/style.css">  
  
    <script src="{{url_for('static', filename='js/app.js')}}" defer></script>  
  
</head>  
<body>  
  
<h2>FLASK API + Ajax(fetch) GET 방식</h2>  
  
<div>  
  
    <div>  
        <label for="name">이름</label>  
        <input id="name" type="text" value="길동"/>  
    </div>  
  
    <div>  
        <label for="age">나이</label>  
        <input id="age" type="number" value="23" min="1"/>  
    </div>  
  
    <button id="sendBtn">GET(REST) 요청</button>  
  
</div>  
  
<div id="result"></div>  
  
</body>  
</html>
```


설명

|코드|설명|
|---|---|
|`/static/css/style.css`|CSS 파일 연결|
|`url_for('static', filename='js/app.js')`|Flask에서 JS 파일 연결|
|`defer`|HTML 로딩 후 JS 실행|

---

# 5️⃣ style.css (디자인)

화면 스타일을 담당하는 CSS 파일

전체 코드
```html
body{margin: 24px;}  
  
button{  
    padding: 8px 12px;  
    margin-top: 8px;  
}  
  
#result{  
    margin-top: 20px;  
    padding: 10px;  
    border: 1px dotted #ddd;  
}  
  
.error{  
    color: red;  
}
```


설명

| CSS      | 설명           |
| -------- | ------------ |
| body     | 페이지 전체 여백    |
| button   | 버튼 크기와 간격    |
| # result | 결과 출력 영역 스타일 |
| .error   | 에러 메시지 색상    |

---

# 6️⃣ app.js (Ajax 요청)

브라우저에서 Flask API로 요청을 보내고  
응답을 받아 화면에 출력하는 JavaScript 파일

전체 코드

```javascript
// 함수(화살표함수) 객체 생성 후 $에 할당  
  
const $ = (sel) => document.querySelector(sel);  
  
// function $(sel){  
//     return document.querySelector(sel);  
// }  
  
// ex) $("#sendBtn") 하면 document.querySelector(sel)가 실행  
  
  
$("#sendBtn").addEventListener("click", async() => {    // 비동기 처리  
  
    const name = $("#name").value.trim();  
  
    //const age = $("#age").value.trim();  
    const age = document.querySelector("#age").value.trim();  
  
  
    const params = new URLSearchParams({name, age});      //공백, 한글이 포함된 경우 자동 인코딩  
  
    const url = `/api/friend?${params.toString()}`        // 최종 URL 생성  
  
  
    $("#result").textContent = "요청 중...";        // 요청 중 메시지  
  
  
    try{  
  
        const res = await fetch(url, {  
            method:"GET",  
            headers:{"Accept":"application/json"}  
        });  
  
  
        const data = await res.json();  // JSON → JS 객체 변환  
  
  
        if(!res.ok || data.ok === false){  
  
            $("#result").innerHTML =  
            `<span class="error">에러 : ${data.error}</span>`;  
  
            return;  
        }  
  
  
        // 요청 성공  
  
        $("#result").innerHTML = `  
            <div>이름 : ${data.name}</div>  
            <div>나이 : ${data.age}</div>  
            <div>연령대 : ${data.age_group}</div>  
            <div>메세지 : ${data.message}</div>  
        `  
  
    }catch(err){  
  
        $("#result").innerHTML =  
        `<span class="error">네트워크, 파싱 오류 : ${err}</span>`  
  
    }  
  
});
```

설명

|코드|설명|
|---|---|
|`$()`|querySelector 축약 함수|
|`addEventListener`|버튼 클릭 이벤트|
|`URLSearchParams`|GET 파라미터 생성|
|`fetch()`|서버 요청|
|`res.json()`|JSON → JS 객체 변환|

---

# 7️⃣ app.py (Flask 서버)

Flask를 이용하여 REST API 서버를 구현

전체 코드

from flask import Flask, render_template, request, jsonify  
  
app = Flask(__name__)  
  
  
@app.get("/")  
def home():  
    return render_template("index.html")  
  
  
@app.get("/api/friend")  
def api_friendFunc():  
  
    name = request.args.get("name", "").strip()  
    age_str = request.args.get("age", "").strip()  
  
    # 입력 검증  
  
    if not name:  
        return jsonify({"ok":False,"error":"name is required"}), 400  
  
    if not age_str.isdigit():  
        return jsonify({"ok":False,"error":"age is required"}), 400  
  
  
    age = int(age_str)  
  
    age_group = f"{(age // 10) * 10}대"     # 23 -> 20대  
  
  
    return jsonify({  
  
        "ok":True,  
        "name":name,  
        "age":age,  
        "age_group":age_group,  
        "message":f"{name}님은 {age}살 {age_group}입니다"  
  
    })  
  
  
if __name__=="__main__":  
    app.run(debug=True)

설명

|코드|설명|
|---|---|
|Flask|웹 서버 생성|
|render_template|HTML 렌더링|
|request.args|GET 파라미터 받기|
|jsonify|JSON 응답 생성|

