# Day 29_Flask REST API (Join & Dynamic Table)
# 📅 2026-03-17
---
## 📁 프로젝트 구조

```Plaintext
fpro17rest/
├── app.py              # Flask 서버 (REST API 엔드포인트)
├── db.py               # MariaDB 연결 설정
├── static/
│   └── js/
│       └── app.js      # Fetch API 및 동적 테이블 렌더링
└── templates/
    └── index.html      # 메인 UI 레이아웃
```

---

## 🗄️ db.py — DB 커넥션 설정

```Python
import os
import pymysql

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123")
DB_NAME = os.getenv("DB_NAME", "test")

def get_connFunc():
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )
```

### 💡 코드 설명

- **DictCursor**: SELECT 결과를 리스트 내 딕셔너리(`[{'key': 'value'}]`) 형태로 반환합니다. 프론트엔드에서 `data.jikwonname`과 같이 키 값으로 접근하기 위해 필수적입니다.
    
- **Context Manager**: `with get_connFunc() as conn:` 구문을 통해 사용 후 DB 연결이 자동으로 닫히도록 관리합니다.
    

---

## 🐍 app.py — Flask REST API (Backend)

```Python
from flask import Flask, jsonify, request, render_template
from db import get_connFunc

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

# 1) 전체 직원 조회 (Join 활용)
@app.get("/api/jikwon")
def jikwon_list():
    sql = """
        select jikwonno, jikwonname, busername, jikwonjik, jikwonpay, year(jikwonibsail) as ibsayear
        from jikwon
        inner join buser on jikwon.busernum = buser.buserno
        order by jikwonno
    """
    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return jsonify({"ok":True, "data":rows})

# 2) 직원 1명 상세 조회 (Path Variable)
@app.get("/api/jikwon/<int:no>")
def jikwon_one(no):
    sql = """
        select jikwonno, jikwonname, busername, jikwonjik, jikwonpay, year(jikwonibsail) as ibsayear
        from jikwon
        inner join buser on jikwon.busernum = buser.buserno
        where jikwonno=%s
    """
    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (no,))
            row = cur.fetchone()
    return jsonify({"ok":True, "data":row})

# 3) 특정 부서 소속 직원 조회
@app.get("/api/buser/<int:bno>/jikwon")
def buser_jikwon(bno):
    sql = "select jikwonno, jikwonname, jikwonjik, jikwonpay from jikwon where busernum=%s"
    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (bno,))
            rows = cur.fetchall()
    return jsonify({"ok":True, "data":rows})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
```

### 💡 코드 설명

- **SQL Inner Join**: `jikwon` 테이블의 부서 번호(`busernum`)를 `buser` 테이블의 부서명(`busername`)으로 변환하여 가져옵니다.
    
- **SQL 가공 (`year()`)**: DB 단에서 날짜 데이터의 연도만 추출하여 가공된 데이터를 전송함으로써 프론트엔드 연산 부담을 줄입니다.
    
- **경로 파라미터**: `<int:no>` 형식을 사용하여 RESTful한 URL 구조를 설계하고, 파라미터를 SQL 쿼리에 바인딩하여 보안(SQL Injection 방지)을 강화했습니다.
    

---

## 🌐 app.js — Vanilla JS (Frontend)

```JavaScript
const msg = document.querySelector("#msg");
const thead = document.querySelector("#thead");
const tbody = document.querySelector("#tbody");

function setMsg(text){ msg.textContent = text; }
function clearTable(){ thead.innerHTML = ""; tbody.innerHTML = ""; }

// [핵심] 동적 테이블 생성 함수
function makeTable(rows){
    clearTable();
    if(!rows || rows.length === 0 || rows[0] === null){
        setMsg("자료 없음");
        return;
    }
    
    // 1. 헤더 생성: 첫 번째 데이터 객체의 Key들을 추출하여 <th> 생성
    let headers = "<tr>" + Object.keys(rows[0]).map(k => `<th>${k}</th>`).join("") + "</tr>";
    thead.innerHTML = headers;

    // 2. 데이터 바디 생성: 각 행의 Value들을 추출하여 <td> 생성
    rows.forEach(r => {
        let tr = "<tr>" + Object.values(r).map(v => `<td>${v}</td>`).join("") + "</tr>";
        tbody.innerHTML += tr;
     });
}

// 비동기 데이터 로드 예시 (부서별 직원)
async function loadDept() {
    const bno = document.querySelector("#buserno").value;
    const res = await fetch(`/api/buser/${bno}/jikwon`);
    const result = await res.json();
    makeTable(result.data);
    setMsg(`${bno}번 부서 조회 완료`);
}
```

### 💡 코드 설명

- **`Object.keys()` & `Object.values()`**: 서버에서 응답받은 데이터의 구조가 바뀌더라도(컬럼 추가 등), JS 코드를 수정할 필요 없이 테이블 제목과 내용을 자동으로 렌더링하는 고도의 유연성을 제공합니다.
    
- **`fetch` API**: 비동기 통신을 통해 페이지 새로고침 없이 서버 데이터를 요청하고 UI를 업데이트합니다.
    

---
## 🖥️ index.html

```HTML
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RESTful 직원 조회</title>
    <script src="/static/js/app.js" defer></script>
</head>
<body>
    <h2>RESTful 직원 / 부서 정보 조회</h2>
    <hr>
    <h3>1. 직원 전체 조회</h3>
    <button id="btnJikwon">전체 직원 보기</button>

    <h3>2. 직원 1명 조회</h3>
    직원 번호 : <input id="Jikwonno">
    <button id="btnOne">일부 직원 보기</button>

    <h3>3. 부서 전체 조회</h3>
    <button id="btnBuser">전체 부서 보기</button>

    <h3>4. 부서별 직원 조회</h3>
    부서 번호 : <input id="buserno">
    <button id="btnDept">일부 부서 직원 보기</button>

    <hr>
    <div id="msg"></div>

    <h3>조회 결과</h3>
    <table border="1">
        <thead id="thead"></thead>
        <tbody id="tbody"></tbody>
    </table>
</body>
</html>
```

### 💡 코드 설명

- **`defer` 속성**: `<script src="..." defer>`를 사용하여 HTML 파싱이 완전히 끝난 후 자바스크립트가 실행되도록 합니다. 이를 통해 JS에서 DOM 요소(`button`, `input` 등)에 접근할 때 발생할 수 있는 `null` 참조 에러를 방지합니다.
    
- **RESTful UI 설계**:
    
    - **조회 방식의 다양화**: 버튼 클릭(전체 조회)과 입력값 전달(조건 조회)을 구분하여 구성했습니다.
        
    - **비어있는 테이블 구조**: `<thead>`와 `<tbody>`를 비워둔 채 `id`만 부여했습니다. 이는 데이터가 도착했을 때 자바스크립트(`app.js`)가 동적으로 내용을 채워 넣을 "무대" 역할을 합니다.
        
- **사용자 경험(UX)**:
    
    - `<div id="msg">`: 데이터 로딩 중, 완료, 에러 메시지 등을 사용자에게 시각적으로 전달하는 상태 표시창 역할을 합니다.
        
    - `input type="number"` 등의 제약 조건을 통해 잘못된 데이터 입력을 일차적으로 방지합니다.

---
## 🔁 데이터 흐름 (Data Flow)

1. **이벤트 발생 (Client Side)**
    
    - 사용자가 `index.html`에서 버튼을 클릭하면 `app.js`에 등록된 이벤트 리스너가 작동합니다.
        
2. **데이터 추출 및 요청 (JS → Flask)**
    
    - `input` 태그에 입력된 값(직원번호 등)을 읽어와서 `fetch('/api/jikwon/10')`와 같이 **RESTful한 경로**로 서버에 비동기 요청을 보냅니다.
        
3. **서버 로직 수행 (Flask → DB)**
    
    - `app.py`는 경로 파라미터(10)를 받아 `db.py`를 통해 MariaDB에 접속합니다.
        
    - **Join 쿼리**를 실행하여 여러 테이블에 흩어진 데이터를 하나로 합칩니다.
        
4. **데이터 반환 (DB → Flask → JS)**
    
    - DB가 보낸 결과(DictCursor 덕분에 딕셔너리 형태)를 Flask가 `jsonify()`를 통해 **JSON 문자열**로 변환하여 브라우저로 응답합니다.
        
5. **동적 렌더링 (JS → UI)**
    
    - `app.js`의 `makeTable()` 함수가 응답받은 JSON 데이터의 **Key를 분석해 제목(`<th>`)**을 만들고, **Value를 순회하며 내용(`<td>`)**을 생성하여 HTML 테이블을 완성합니다.