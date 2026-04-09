# Day 28_Flask REST API CRUD
# 📅 2026-03-16

---
# fpro15crud — Flask + MariaDB CRUD 수업 정리

## 📁 프로젝트 구조

```
fpro15crud/
├── app.py
├── db.py
├── templates/
│   └── index.html
└── static/
    └── js/
        └── myapp.js
```

---

## 🗄️ db.py

- 환경변수(`os.getenv`)로 DB 접속 정보 관리 → 보안상 하드코딩 지양
- `DictCursor` : SELECT 결과를 딕셔너리 형태로 반환 (컬럼명으로 접근 가능)
- `autocommit=True` : INSERT/UPDATE/DELETE 시 자동 커밋
- `charset="utf8mb4"` : 전세계 문자 + 이모지 처리
- `with get_connFunc() as conn:` 형태로 사용 → 자동으로 연결 해제

```python
import pymysql
import os

# MariaDB 연결 정보
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PWD = os.getenv("DB_PWD", "123")
DB_NAME = os.getenv("DB_NAME", "test")

def get_connFunc():
    return pymysql.connect(
        host = DB_HOST,
        port = DB_PORT,
        user = DB_USER,
        password = DB_PWD,
        database = DB_NAME,
        charset = "utf8mb4",                            # utf8mb4 : 전세계 문자 + 이모지 처리 가능
        cursorclass = pymysql.cursors.DictCursor,       # Dictcursor : select 결과를 "dict type" 형태로 받게 해줌
        autocommit = True
    )
```

---

## 🐍 app.py — Flask REST API

### API 엔드포인트 정리

|메서드|경로|기능|
|---|---|---|
|GET|`/`|index.html 렌더링|
|GET|`/api/sangdata`|전체 조회|
|POST|`/api/sangdata`|새 상품 추가|
|PUT|`/api/sangdata/<int:code>`|상품 수정|
|DELETE|`/api/sangdata/<int:code>`|상품 삭제|

```python
from flask import Flask, render_template, request, jsonify
from db import get_connFunc

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

# 전체 조회
@app.get("/api/sangdata")
def list_sangdata():
    sql = "select code, sang, su, dan from sangdata order by code asc"
    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    return jsonify({"ok":True, "data":rows})


# 새 상품 추가
@app.post("/api/sangdata")
def create_sangdata():
    data = request.get_json()
    code = data["code"]
    sang = data["sang"]
    su = int(data["su"])
    dan = int(data["dan"])
    isql = "insert into sangdata(code, sang, su, dan) values(%s, %s, %s, %s)"

    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(isql, (code, sang, su, dan))

    return jsonify({"ok":True})


# 상품 수정
@app.put("/api/sangdata/<int:code>")
def update_sangdata(code):
    data = request.get_json()
    sang = data["sang"]
    su = int(data["su"])
    dan = int(data["dan"])
    usql = "update sangdata set sang=%s, su=%s, dan=%s where code=%s"

    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(usql, (sang,su,dan,code))

    return jsonify({"ok":True})


# 상품 삭제
@app.delete("/api/sangdata/<int:code>")
def delete_sangdata(code):
    try:
        dsql = "delete from sangdata where code=%s"

        with get_connFunc() as conn:
            with conn.cursor() as cur:
                cur.execute(dsql, (code,))
                if cur.rowcount == 0:
                    # rowcount : 실제로 영향받은 행의 수. 0이면 해당 코드 없음
                    return jsonify({"ok":False, "msg":"해당 자료 없음"})

        return jsonify({"ok":True, "msg":"삭제 완료"})
    except Exception as err:
        return jsonify({"ok":False, "msg":str(err)})


if __name__ == "__main__":
    app.run(debug=True)
```

---

## 🌐 myapp.js — 프론트엔드 (Vanilla JS)

### 전체 흐름

```
사용자 버튼 클릭
    → fetch()로 API 호출 (async/await)
    → 응답 JSON 파싱
    → 테이블 갱신 + 메시지 표시
```

### 핵심 포인트

- POST/PUT 시 `JSON.stringify(data)` 로 JS 객체를 문자열로 변환해 body에 전달
- `headers: {"Content-Type": "application/json"}` 반드시 명시
- `Number()` : input value는 기본 문자열 → 숫자 필드 변환 필요
- `async/await` : fetch는 비동기 → await 없으면 빈 응답 받음
- 추가/수정/삭제 후 `loadAll()` 호출해서 테이블 갱신

```javascript
// 각종 요소
const code = document.querySelector("#code");
const sang = document.querySelector("#sang");
const su = document.querySelector("#su");
const dan = document.querySelector("#dan");

const msg = document.querySelector("#msg");
const tbody = document.querySelector("#tbody");

const btnAdd = document.querySelector("#btnAdd");
const btnUpdate = document.querySelector("#btnUpdate");
const btnDelete = document.querySelector("#btnDelete");
const btnReload = document.querySelector("#btnReload");

function setMsg(text){
    msg.textContent = text;
}

// 입력 폼 초기화
function clearForm(){
    code.value = "";
    sang.value = "";
    su.value = "";
    dan.value = "";
}

// 전체 자료 읽기
async function loadAll(){
    const res = await fetch("/api/sangdata");
    const data = await res.json();

    tbody.innerHTML = "";

    data.data.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = "<td>" + r.code + "</td>" +
            "<td>" + r.sang + "</td>" +
            "<td>" + r.su + "</td>" +
            "<td>" + r.dan + "</td>";
        tbody.appendChild(tr);
    });

    clearForm();
    setMsg("조회 완료");
}

// 추가
async function addData(){
    const data = {
        code:Number(code.value),
        sang:sang.value,
        su:Number(su.value),
        dan:Number(dan.value)
    };

    const res = await fetch("/api/sangdata", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body : JSON.stringify(data)     // js 객체를 문자열로 변환해 전송
    });

    await res.json();
    setMsg("추가 완료");
    clearForm();
    loadAll();      // 추가 후 전체 자료 보기
}

// 수정
async function updateData(){
    const data = {
        sang:sang.value,
        su:Number(su.value),
        dan:Number(dan.value)
    };

    const res = await fetch("/api/sangdata/" + code.value , {
        method:"PUT",
        headers:{"Content-Type":"application/json"},
        body : JSON.stringify(data)
    });

    const imsi = await res.json();
    if(imsi.ok)
        setMsg("수정 완료");
    else
        setMsg("수정 실패");

    clearForm();
    loadAll();
}

// 삭제
async function deleteData(){
    const res = await fetch("/api/sangdata/" + code.value , {
        method:"DELETE"
    });

    const imsi = await res.json();
    if(imsi.ok)
        setMsg(imsi.msg);
    else
        setMsg("삭제 실패 : " + imsi.msg);

    clearForm();
    loadAll();
}

window.onload = loadAll;

btnAdd.onclick = addData;
btnUpdate.onclick = updateData;
btnDelete.onclick = deleteData;
btnReload.onclick = loadAll;
```

---

## 🖥️ index.html

- `defer` : HTML 파싱 완료 후 JS 실행 → DOM 요소 접근 가능
- `id` 속성으로 JS에서 `querySelector`로 각 요소 접근

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <script src="/static/js/myapp.js" defer></script>
</head>
<body>
    <h2>상품 CRUD(sangdata table)</h2>
    코드 : <input id="code"><br>
    품명 : <input id="sang"><br>
    수량 : <input id="su"><br>
    단가 : <input id="dan"><br>
    <br>
    <button id="btnAdd">추가</button>
    <button id="btnUpdate">수정</button>
    <button id="btnDelete">삭제</button>
    <button id="btnReload">조회</button>
    <br><br>
    <div id="msg"></div>
    <table border="1">
        <tr>
            <th>코드</th><th>품명</th><th>수량</th><th>단가</th>
        </tr>
        <tbody id="tbody"></tbody>
    </table>
</body>
</html>
```

---

## ⚠️ 주요 포인트 / 자주 실수하는 것

| 포인트                | 설명                                   |
| ------------------ | ------------------------------------ |
| `JSON.stringify()` | fetch body에 JS 객체 넣을 때 문자열 변환 필수     |
| `Content-Type` 헤더  | POST/PUT 시 반드시 `application/json` 명시 |
| `Number()` 변환      | 입력값은 기본 문자열 → 숫자 필드는 변환 필요           |
| `async/await`      | fetch는 비동기 → await 없으면 빈 응답 받음       |
| `DictCursor`       | 없으면 SELECT 결과가 튜플로 와서 컬럼명 접근 불가      |
| `rowcount`         | DELETE 후 0이면 해당 데이터 없음               |

---

## 🔁 전체 데이터 흐름

```
브라우저 (index.html + myapp.js)
    ↕ fetch (JSON)
Flask app.py (REST API)
    ↕ pymysql
MariaDB (sangdata 테이블)
```

---
# fpro16crud — Flask + MariaDB CRUD 수업 정리 (개선판)

## 📁 프로젝트 구조

```
fpro16crud/
├── app.py
├── db.py
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

---

## fpro15crud와 달라진 점

|항목|fpro15|fpro16|
|---|---|---|
|JS 요소 선택|개별 변수|`els` 객체로 캐싱|
|API 주소|JS에 하드코딩|`window.API_LIST`로 HTML에서 주입|
|에러 처리|없음|`try/catch` + HTTP 상태코드|
|입력값 검증|없음|서버/클라이언트 모두 검증|
|행 클릭|없음|클릭 시 폼 자동 채우기|
|1건 조회|없음|`GET /api/sangdata/<code>` 추가|
|CSS|없음|`style.css` 분리|

---

## 🗄️ db.py

```python
import os
import pymysql

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123")
DB_NAME = os.getenv("DB_NAME", "test")

def get_connFunc():
    """
    MariaDB 연결 객체 반환
    - cursorclass=DictCursor : 결과를 dict로 받기 (JSON 만들기 편함)
    - autocommit=True : 편의상 자동 커밋
    """
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )
```

---

## 🐍 app.py — Flask REST API

### API 엔드포인트 정리

|메서드|경로|기능|상태코드|
|---|---|---|---|
|GET|`/`|index.html 렌더링|200|
|GET|`/api/sangdata`|전체 조회|200|
|GET|`/api/sangdata/<int:code>`|1건 조회|200 / 404|
|POST|`/api/sangdata`|새 상품 추가|201 / 400|
|PUT|`/api/sangdata/<int:code>`|상품 수정|200 / 400 / 404|
|DELETE|`/api/sangdata/<int:code>`|상품 삭제|200 / 404|

### 핵심 변화 — 에러 처리 & 입력값 검증

```python
# silent=True : JSON 파싱 실패해도 예외 안 터지고 None 반환
data = request.get_json(silent=True) or {}

# 필수값 체크 후 400 반환
try:
    code = int(data.get("code"))
except Exception:
    return jsonify({"ok": False, "error": "code is required(int)"}), 400

# rowcount로 존재 여부 확인 후 404 반환
if cur.rowcount == 0:
    return jsonify({"ok": False, "error": "NOT_FOUND"}), 404
```

### 전체 코드

```python
# pip install flask pymysql
from flask import Flask, jsonify, request, render_template
from db import get_connFunc

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")


# 1) 전체 조회
@app.get("/api/sangdata")
def list_sangdata():
    sql = "SELECT code, sang, su, dan FROM sangdata ORDER BY code"
    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return jsonify({"ok": True, "data": rows})


# 2) 1건 조회(선택)
@app.get("/api/sangdata/<int:code>")
def get_one(code: int):
    sql = "SELECT code, sang, su, dan FROM sangdata WHERE code=%s"
    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (code, ))
            row = cur.fetchone()
    if not row:
        return jsonify({"ok": False, "error": "NOT_FOUND"}), 404
    return jsonify({"ok": True, "data": row})


# 3) 추가 (POST / JSON)
@app.post("/api/sangdata")
def create_sangdata():
    data = request.get_json(silent=True) or {}

    try:
        code = int(data.get("code"))
    except Exception:
        return jsonify({"ok": False, "error": "code is required(int)"}), 400

    sang = (data.get("sang") or "").strip()
    if not sang:
        return jsonify({"ok": False, "error": "sang is required"}), 400

    try:
        su = int(data.get("su", 0))
        dan = int(data.get("dan", 0))
    except Exception:
        return jsonify({"ok": False, "error": "su/dan must be int"}), 400

    sql = "INSERT INTO sangdata(code, sang, su, dan) VALUES(%s, %s, %s, %s)"
    try:
        with get_connFunc() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (code, sang, su, dan))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400    # 예: PK 중복 등

    return jsonify({"ok": True, "message": "CREATED", "code": code}), 201


# 4) 수정 (PUT / JSON)
@app.put("/api/sangdata/<int:code>")
def update_sangdata(code: int):
    data = request.get_json(silent=True) or {}

    sang = (data.get("sang") or "").strip()
    if not sang:
        return jsonify({"ok": False, "error": "sang is required"}), 400

    try:
        su = int(data.get("su", 0))
        dan = int(data.get("dan", 0))
    except Exception:
        return jsonify({"ok": False, "error": "su/dan must be int"}), 400

    sql = "UPDATE sangdata SET sang=%s, su=%s, dan=%s WHERE code=%s"
    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (sang, su, dan, code))
            if cur.rowcount == 0:
                return jsonify({"ok": False, "error": "NOT_FOUND"}), 404

    return jsonify({"ok": True, "message": "UPDATED", "code": code})


# 5) 삭제 (DELETE)
@app.delete("/api/sangdata/<int:code>")
def delete_sangdata(code: int):
    sql = "DELETE FROM sangdata WHERE code=%s"
    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (code,))
            if cur.rowcount == 0:
                return jsonify({"ok": False, "error": "NOT_FOUND"}), 404

    return jsonify({"ok": True, "message": "DELETED", "code": code})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
```

---

## 🌐 app.js — 프론트엔드 (Vanilla JS)

### 핵심 패턴 — 요소 캐싱 (els 객체)

```javascript
const $ = (sel) => document.querySelector(sel);

// DOM 요소들을 한 번에 찾아 변수(객체)에 담아두고, 이후에 편하게 쓰려고 만든 요소 캐싱 패턴.
// els는 각 DOM 요소를 key-value로 모아 둔 객체 변수명. els.code, els.sang 이렇게 호출
const els = {
  code: $("#code"),
  sang: $("#sang"),
  su: $("#su"),
  dan: $("#dan"),
  msg: $("#msg"),
  tbody: $("#tbody"),
  btnAdd: $("#btnAdd"), btnUpdate: $("#btnUpdate"), btnDelete: $("#btnDelete"),
  btnClear: $("#btnClear"),
  btnReload: $("#btnReload"),
};
```

### 핵심 패턴 — classList.toggle로 에러 스타일 처리

```javascript
// <div id="msg" class="msg">메시지 영역</div>에 메세지 출력용 함수
function setMsg(text, isError = false) {
  els.msg.textContent = text;
  els.msg.classList.toggle("error", isError);
  // classList.toggle(클래스명, 조건)
  // 조건이 true면 → 해당 클래스를 추가(add)
  // 조건이 false면 → 해당 클래스를 제거(remove)
}
```

### 핵심 패턴 — renderRows (map + join으로 tbody 한번에 렌더링)

```javascript
// 서버에서 받은 상품 목록(rows)을 tr로 만들어서 테이블 tbody에 한 번에 뿌려주는 함수
function renderRows(rows) {
  els.tbody.innerHTML = rows.map(r => `
    <tr data-code="${r.code}">
      <td>${r.code}</td>
      <td>${r.sang ?? ""}</td>
      <td>${r.su ?? ""}</td>
      <td>${r.dan ?? ""}</td>
    </tr>`).join("");
}
// <tr data-code="${r.code}"> : tr(한 행)에 data-code="1" 같은 커스텀 데이터 속성을 달아둠
// 나중에 '행 클릭했을 때 code를 쉽게 꺼내기'에 좋음. 예: tr.dataset.code 로 읽음
```

### 핵심 패턴 — 행 클릭 시 폼 채우기 (이벤트 위임)

```javascript
// 행 클릭 → 폼 채우기
els.tbody.addEventListener("click", (e) => {
  // 클릭된 위치에서 가장 가까운(자기 자신 포함) 조상 요소 중 <tr>을 찾는 코드
  const tr = e.target.closest("tr");
  if (!tr) return;
  const tds = tr.querySelectorAll("td");
  els.code.value = tds[0].textContent;
  els.sang.value = tds[1].textContent;
  els.su.value   = tds[2].textContent;
  els.dan.value  = tds[3].textContent;
  setMsg(`선택됨: code=${els.code.value}`);
});
```

### 전체 코드

```javascript
const $ = (sel) => document.querySelector(sel);

const els = {
  code: $("#code"),
  sang: $("#sang"),
  su: $("#su"),
  dan: $("#dan"),
  msg: $("#msg"),
  tbody: $("#tbody"),
  btnAdd: $("#btnAdd"), btnUpdate: $("#btnUpdate"), btnDelete: $("#btnDelete"),
  btnClear: $("#btnClear"),
  btnReload: $("#btnReload"),
};

function setMsg(text, isError = false) {
  els.msg.textContent = text;
  els.msg.classList.toggle("error", isError);
}

function getForm() {
  return {
    code: els.code.value.trim(),
    sang: els.sang.value.trim(),
    su: els.su.value.trim(),
    dan: els.dan.value.trim(),
  };
}

function clearForm() {
  els.code.value = "";
  els.sang.value = "";
  els.su.value = "";
  els.dan.value = "";
  setMsg("초기화 완료");
}

function renderRows(rows) {
  els.tbody.innerHTML = rows.map(r => `
    <tr data-code="${r.code}">
      <td>${r.code}</td>
      <td>${r.sang ?? ""}</td>
      <td>${r.su ?? ""}</td>
      <td>${r.dan ?? ""}</td>
    </tr>`).join("");
}

async function loadAll() {
  setMsg("조회 중...");
  try {
    // window.API_LIST는 index.html에서 주입한 API 주소
    const res = await fetch(window.API_LIST, { headers: { "Accept": "application/json" } });
    const data = await res.json();
    if (!res.ok || data.ok === false) throw new Error(data.error || "조회 실패");

    renderRows(data.data);
    setMsg(`조회 완료: ${data.data.length}건`);
  } catch (e) {
    setMsg(`조회 오류: ${e.message}`, true);
  }
}

// 추가
async function addOne() {
  const f = getForm();
  if (!f.code || !f.sang) return setMsg("code, sang는 필수!", true);
  try {
    const res = await fetch(window.API_LIST, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({
        code: Number(f.code),
        sang: f.sang,
        su: Number(f.su || 0),
        dan: Number(f.dan || 0),
      })
    });
    const data = await res.json();
    if (!res.ok || data.ok === false) throw new Error(data.error || "추가 실패");

    setMsg(`추가 완료: code=${data.code}`);
    await loadAll();
  } catch (e) {
    setMsg(`추가 오류: ${e.message}`, true);
  }
}

// 수정
async function updateOne() {
  const f = getForm();
  if (!f.code) return setMsg("수정하려면 code가 필요!", true);
  try {
    const res = await fetch(`${window.API_LIST}/${Number(f.code)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({
        sang: f.sang,
        su: Number(f.su || 0),
        dan: Number(f.dan || 0),
      })
    });
    const data = await res.json();
    if (!res.ok || data.ok === false) throw new Error(data.error || "수정 실패");

    setMsg(`수정 완료: code=${data.code}`);
    await loadAll();
  } catch (e) {
    setMsg(`수정 오류: ${e.message}`, true);
  }
}

// 삭제
async function deleteOne() {
  const f = getForm();
  if (!f.code) return setMsg("삭제하려면 code가 필요!", true);
  if (!confirm(`code=${f.code}를 삭제할까요?`)) return;
  try {
    const res = await fetch(`${window.API_LIST}/${Number(f.code)}`, {
      method: "DELETE",
      headers: { "Accept": "application/json" }
    });
    const data = await res.json();
    if (!res.ok || data.ok === false) throw new Error(data.error || "삭제 실패");

    setMsg(`삭제 완료: code=${data.code}`);
    clearForm();
    await loadAll();
  } catch (e) {
    setMsg(`삭제 오류: ${e.message}`, true);
  }
}

// 행 클릭 → 폼 채우기
els.tbody.addEventListener("click", (e) => {
  const tr = e.target.closest("tr");
  if (!tr) return;
  const tds = tr.querySelectorAll("td");
  els.code.value = tds[0].textContent;
  els.sang.value = tds[1].textContent;
  els.su.value   = tds[2].textContent;
  els.dan.value  = tds[3].textContent;
  setMsg(`선택됨: code=${els.code.value}`);
});

// 버튼 이벤트 장착
els.btnReload.addEventListener("click", loadAll);
els.btnClear.addEventListener("click", clearForm);
els.btnAdd.addEventListener("click", addOne);
els.btnUpdate.addEventListener("click", updateOne);
els.btnDelete.addEventListener("click", deleteOne);

// 최초 로딩 시 전체조회
window.addEventListener("DOMContentLoaded", loadAll);
```

---

## 🖥️ index.html

### 핵심 포인트 — window.API_LIST 주입

```html
<!-- API base를 템플릿에서 주입(하드코딩 방지) -->
<script>
  // app.py의 @app.get("/api/sangdata") 했으므로 /api/sangdata가 됨
  window.API_LIST = "{{ url_for('list_sangdata') }}";
</script>
```

> JS에 URL을 하드코딩하지 않고, Flask의 `url_for`로 생성한 주소를 `window.API_LIST`에 담아 JS에서 사용

### 전체 코드

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>MariaDB + Flask CRUD</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
  <script>
    window.API_LIST = "{{ url_for('list_sangdata') }}";
  </script>
  <script src="{{ url_for('static', filename='js/app.js') }}" defer></script>
</head>
<body>
  <h2>상품 CRUD (sangdata)</h2>
  <section class="panel">
    <h2>입력/수정 폼</h2>
    <div>
      <label>코드:</label>
      <input id="code" type="number" placeholder="1">
      <br>
      <label>품명:</label>
      <input id="sang" type="text" placeholder="연필">
      <br>
      <label>수량:</label>
      <input id="su" type="number" placeholder="10">
      <br>
      <label>단가:</label>
      <input id="dan" type="number" placeholder="500">
    </div>

    <div class="btns">
      <button id="btnAdd">추가(POST)</button>
      <button id="btnUpdate">수정(PUT)</button>
      <button id="btnDelete">삭제(DELETE)</button>
      <button id="btnClear">초기화</button>
      <button id="btnReload">전체조회(GET)</button>
    </div>
    <div id="msg" class="msg">메시지 영역</div>
  </section>

  <section class="panel">
    <h2>전체 목록</h2>
    <table>
      <thead>
        <tr><th>코드</th><th>품명</th><th>수량</th><th>단가</th></tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
    <p class="hint">행을 클릭하면 폼에 값이 채워짐. (수정/삭제 편하게)</p>
  </section>
</body>
</html>
```

---

## 🎨 style.css

```css
.panel {
  margin: 10px 0;  padding: 16px;  border: 1px solid #ddd;
}

/* 버튼 묶음 */
.btns {
  margin-top: 10px;  display: flex;  flex-wrap: wrap;  gap: 10px;
}

/* 메시지 영역 */
.msg {
  margin-top: 12px;  padding: 10px 12px;  border: 1px dashed #bbb;
  display: flex;  align-items: center;
}

/* JS에서 메시지 종류별 클래스를 붙일 수 있게 준비 */
.msg.ok    { border-style: solid; }
.msg.error { border-style: solid; }

/* 안내 문구 */
.hint { margin-top: 10px; font-size: 13px; opacity: 0.85; }
```

---

## ⚠️ 주요 포인트 정리

|포인트|설명|
|---|---|
|`els` 객체 캐싱|DOM 요소를 매번 querySelector로 찾지 않고 한 번에 모아둠|
|`window.API_LIST`|Flask `url_for`로 API 주소 생성 → JS에 하드코딩 방지|
|`silent=True`|JSON 파싱 실패해도 예외 안 터지고 None 반환|
|HTTP 상태코드|성공 201, 실패 400, 없음 404 등 명확하게 구분|
|`classList.toggle`|`toggle(클래스, 조건)` → true면 추가, false면 제거|
|`data-code` 속성|`tr.dataset.code`로 행의 code 값을 쉽게 읽을 수 있음|
|`e.target.closest("tr")`|클릭된 td에서 가장 가까운 tr 조상을 찾음 (이벤트 위임)|
|`?? ""` (널 병합)|null/undefined면 빈 문자열로 대체|
|`DOMContentLoaded`|HTML 파싱 완료 시점에 초기 조회 실행|

---

## 🔁 전체 데이터 흐름

```
브라우저 (index.html + app.js)
    ↕ fetch (JSON) — window.API_LIST 사용
Flask app.py (REST API) — 에러 처리 + 상태코드
    ↕ pymysql
MariaDB (sangdata 테이블)
```