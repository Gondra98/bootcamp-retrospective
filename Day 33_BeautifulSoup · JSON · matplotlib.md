# Day 33_BeautifulSoup · JSON · matplotlib

## 📅 2026-03-23

---

## 1. 📡 `pandas9soup.py` — 기본 웹 요청 & a태그 추출

### 1-1. 웹 페이지 요청

```python
baseurl = "https://www.naver.com"
headers = {"User-Agent":"Mozilla/5.0"}
source = requests.get(baseurl, headers=headers)
```

- `User-Agent` 헤더 : 브라우저로 위장해 차단 우회
- `status_code` : 200 = 요청 성공
- `source.text` → 문자열(str) / `source.content` → 바이트(bytes)

### 1-2. BeautifulSoup 파싱 객체 생성

```python
conv_data = BeautifulSoup(source.text, 'lxml')
```

- `lxml` 파서로 HTML 문자열을 분석 가능한 객체로 변환
- 타입 : `<class 'bs4.BeautifulSoup'>`

### 1-3. a태그 링크 추출

```python
for atag in conv_data.find_all('a'):
    href = atag.get('href')
    title = atag.get_text(strip=True)
    if title:
        print(href)
        print(title)
```

|메서드|설명|
|---|---|
|`find_all('a')`|모든 `<a>` 태그 리스트 반환|
|`.get('href')`|href 속성값(링크 주소) 추출|
|`.get_text(strip=True)`|태그 내 텍스트 추출 + 공백 제거|
|`if title:`|텍스트 없는 태그 제외|

---

## 2. 🔍 `pandas10find.py` — find() / find_all() 메서드

### 2-1. HTML 직접 파싱

```python
soup = BeautifulSoup(html_page, 'html.parser')
```

- `html.parser` : 외부 라이브러리 없이 파이썬 내장 파서 사용

### 2-2. 태그 탐색 (트리 방식)

```python
h1 = soup.html.body.h1             # 태그를 . 으로 직접 접근
p1 = soup.html.body.p              # 첫 번째 p 태그
p2 = p1.next_sibling.next_sibling  # 다음 형제 태그로 이동
```

- `.next_sibling` 을 두 번 쓰는 이유 : 태그 사이 **줄바꿈(공백 텍스트)** 도 sibling으로 취급되기 때문

### 2-3. find() 메서드

```python
soup2.find('p')                      # 첫 번째 p 태그
soup2.find('p', id="my")             # id가 "my"인 p 태그
soup2.find(id="title")               # id로만 검색
soup2.find(class_="our")             # class로 검색 (class_ 로 써야 함)
soup2.find(attrs={"class":"our"})    # attrs 딕셔너리로 검색
```

### 2-4. find_all() 메서드

```python
soup3.find_all(['a'])           # a태그 전체
soup3.find_all(['a', 'p'])      # a, p태그 모두
```

- 리스트로 반환 → `for`문으로 순회

### 2-5. 정규표현식 활용

```python
import re
links2 = soup3.find_all(href=re.compile(r'^https'))
```

- `href` 속성이 `https`로 시작하는 태그만 필터링

### 2-6. 실습 — bugs 음악 차트 읽기

```python
url = "https://music.bugs.co.kr/chart"
response = requests.get(url)
bsoup = BeautifulSoup(response.text, 'html.parser')
musics = soup.find_all("td", class_="check")
for idx, music in enumerate(musics):
    print(f"{idx + 1}위) {music.input['title']}")
```

---

## 3. 🎯 `pandas11select.py` — CSS 셀렉터 활용

### 3-1. select_one() — 단일 태그 선택

```python
soup.select_one("div")           # div 태그 하나
soup.select_one("div#hello")     # id="hello"인 div
soup.select_one("div.good")      # class="good"인 div
soup.select_one("div#hello > a") # id=hello인 div의 직계 a태그
```

### 3-2. select() — 여러 태그 선택

```python
soup.select("div#hello ul.world > li")  # 리스트 반환
for i in bb:
    print(i.text)
```

|셀렉터 문법|설명|
|---|---|
|`태그#id`|특정 id를 가진 태그|
|`태그.class`|특정 class를 가진 태그|
|`A > B`|A의 **직계 자식** B|
|`A B`|A의 **모든 하위** B|

### 3-3. 실습 ① — 위키백과 이순신 검색

```python
url = "https://ko.wikipedia.org/wiki/이순신"
soup = BeautifulSoup(wiki.text, 'html.parser')
result = soup.select("p#mwHw")
for s in result:
    for sup in s.find_all("sup"):
        sup.decompose()       # 각주 태그 삭제
    print(s.get_text(strip=True))
```

- `decompose()` : 태그를 HTML에서 완전히 제거

### 3-4. 실습 ② — 교촌치킨 메뉴/가격 + pandas 통계

```python
names  = [tag.text.strip() for tag in soup2.select("dl.txt>dt")]
prices = [int(tag.text.strip().replace(',','')) for tag in soup2.select("p.money strong")]
df = pd.DataFrame({"상품명":names, "가격":prices})
print(f"가격 평균 : {df['가격'].mean():.2f}")
print(f"가격 표준편차 : {df['가격'].std():.2f}")
cv = df['가격'].std() / df['가격'].mean() * 100
print(f"가격 변동계수(CV) : {cv:.2f}%")
# 해석 : CV ≈ 28% → 평균 대비 적당히 퍼져 있는 편
```

|통계값|설명|
|---|---|
|평균 `mean()`|전체 가격의 평균|
|표준편차 `std()`|가격의 퍼짐 정도|
|변동계수 CV|표준편차 ÷ 평균 × 100, 상대적 분산도|

---

## 4. ⏱️ `pandas12bs.py` — 주기적 실시간 데이터 수집

### 4-1. 반복 수집 구조

```python
import time
cnt = 2
while cnt:
    res = requests.get(url=url, headers=headers)
    soup = BeautifulSoup(res.content, 'lxml')
    # ... 데이터 추출 ...
    cnt -= 1
    time.sleep(10)   # 10초 대기 후 재요청
```

- `time.sleep(N)` : N초 대기 → 일정 주기로 데이터 갱신

### 4-2. 네이버 금융 환율 추출

```python
nation  = soup.select_one("h3.h_lst span.blind").get_text(strip=True)
price   = soup.select_one(".value").get_text(strip=True)
unit    = soup.select_one(".txt_krw .blind").get_text(strip=True)
change  = soup.select_one(".change").get_text(strip=True)
updown  = soup.select("div.head_info.point_up span.blind")[-1].get_text(strip=True)
```

- `[-1]` : 리스트의 마지막 요소 선택

### 4-3. 인코딩 설정

```python
sys.stdout.reconfigure(encoding="utf-8")
```

- 한글 출력 깨짐 방지

---

## 5. 💾 `pandas12quiz.py` — 시가총액 CSV 저장

### 5-1. 여러 페이지 반복 수집

```python
urls = [
    "https://finance.naver.com/sise/sise_market_sum.naver?&page=1",
    "https://finance.naver.com/sise/sise_market_sum.naver?&page=2"
]
for url in urls:
    res = requests.get(url=url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
```

### 5-2. CSV 파일로 저장

```python
with open(file_name, mode='w', encoding='utf-8') as f:
    f.write("종목명,시가총액\n")
    for row in rows:
        if not row.select_one("a.tltle"): continue
        name  = row.select_one("a.tltle").get_text(strip=True)
        price = row.select(".number")[4].get_text(strip=True).replace(',', '')
        f.write(f"{name},{price}\n")
```

- `if not ... : continue` : 빈 행(광고/공백) 건너뜀
- `.select(".number")[4]` : 숫자 셀 중 5번째(시가총액) 선택

### 5-3. pandas로 후처리

```python
df = pd.read_csv(file_name)
df['시가총액'] = pd.to_numeric(df['시가총액'])
df.index = df.index + 1   # 인덱스 1부터 시작
print(df[['종목명', '시가총액']].head(5))
```

---

## 6. 🗂️ `pandas13xml.py` — XML 문서 처리

### 6-1. XML 파일 읽기

```python
with open('my.xml', mode='r', encoding='utf-8') as f:
    xmlfile = f.read()
soup = BeautifulSoup(xmlfile, 'lxml')
```

- HTML과 동일하게 `lxml` 파서로 파싱 가능

### 6-2. XML 구조 (`my.xml`)

```xml
<items>
    <item>
        <name id="ks1">홍길동</name>
        <tel>010-111-1111</tel>
        <exam kor="100" eng="90" />
    </item>
    <item>
        <name id="ks2">고길동</name>
        <tel>010-111-2222</tel>
        <exam kor="88" eng="92" />
    </item>
</items>
```

### 6-3. 태그 및 속성 추출

```python
itemTag = soup.find_all('item')
nameTag = soup.find_all('name')
print(nameTag[0]['id'])         # 속성값 추출 : ks1

for i in itemTag:
    for j in i.find_all('name'):
        print("id:" + j["id"] + " name:" + j.string)
        print("tel:", i.find("tel").string)
    for j in i.find_all('exam'):
        print("kor:" + j["kor"] + ", eng:" + j["eng"])
```

- `태그["속성명"]` : 속성값 접근 (딕셔너리 방식)
- `태그.string` : 태그 내 텍스트 추출

---

## 7. 📦 `pandas14json.py` — JSON 데이터 처리

### 7-1. JSON이란?

- XML에 비해 **경량**
- 딕셔너리 + 배열 개념만 알면 처리 가능
- 현대 API의 표준 데이터 형식

### 7-2. 인코딩 / 디코딩

```python
import json

dict = {'name':'tom', 'age':25, 'score':['90','80','88']}

str_val  = json.dumps(dict)    # 인코딩 : dict → str
json_val = json.loads(str_val) # 디코딩 : str  → dict
```

|함수|방향|결과 타입|
|---|---|---|
|`json.dumps()`|dict → JSON 문자열|`str`|
|`json.loads()`|JSON 문자열 → dict|`dict`|

> ⚠️ `str` 상태에서는 키 접근 불가 → 슬라이싱만 가능  
> ✅ `dict` 상태에서만 `json_val['name']` 접근 가능

### 7-3. dict 순회

```python
for k in json_val.keys():   # 키 순회
    print(k)
for v in json_val.values(): # 값 순회
    print(v)
```

### 7-4. 서울시 도서관 JSON API 파싱

```python
url = "http://openapi.seoul.go.kr:8088/sample/json/SeoulLibraryTimeInfo/1/5/"
plainText = req.urlopen(url).read().decode()
jsonData = json.loads(plainText)
```

응답 JSON 구조:

```json
{
  "SeoulLibraryTimeInfo": {
    "row": [
      {"LBRRY_NAME": "BIBLIOTECA", "ADRES": "...", "TEL_NO": ""}
    ]
  }
}
```

데이터 접근 방법 2가지:

```python
# 방법 1: 키 직접 접근
jsonData["SeoulLibraryTimeInfo"]["row"][0]["LBRRY_NAME"]

# 방법 2: get() — 키 없을 때 None 반환 (안전)
jsonData.get("SeoulLibraryTimeInfo").get("row")
```

### 7-5. DataFrame 변환

```python
datas = []
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel  = ele.get('TEL_NO')
    addr = ele.get('ADRES')
    datas.append([name, tel, addr])

df = pd.DataFrame(datas, columns=['도서관명','전화','주소'])
```

### 🔁 XML vs JSON 비교

|항목|XML (BeautifulSoup)|JSON|
|---|---|---|
|파서|`BeautifulSoup` 필요|내장 `json` 모듈|
|데이터 접근|`.find()`, `.select()`|딕셔너리 키 접근|
|코드 복잡도|복잡|단순|
|무게|무거움|경량|

---

## 8. 📊 `plot1.py` — matplotlib 시각화

### 8-1. 기본 설정

```python
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      # 한글 폰트 (Mac: AppleGothic)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
```

### 8-2. 기본 꺾은선 그래프

```python
x = ("서울", "인천", "수원")
y = [5, 3, 7]
plt.xlim([-1, 3])                      # x축 범위
plt.ylim([0, 10])                      # y축 범위
plt.yticks(list(range(0, 11, 3)))      # y축 눈금 : 0, 3, 6, 9
plt.plot(x, y)
plt.show()
```

> ✅ x에 list, tuple 사용 가능 / set은 순서 없어 ❌

### 8-3. 데이터 포인트에 값 표시

```python
data = np.arange(1, 11, 2)   # [1, 3, 5, 7, 9]
plt.plot(data)
for a, b in zip(x, data):
    plt.text(a, b, str(b))   # (x위치, y위치, 텍스트)
plt.show()
```

- `plt.text()` : 그래프 위에 직접 텍스트 표시

### 8-4. 선 스타일 / 마커 지정

```python
plt.plot(x, y, 'go--', linewidth=2, markersize=12)
```

|코드|의미|
|---|---|
|`g`|초록색(green)|
|`o`|원형 마커|
|`--`|점선|
|`linewidth`|선 두께|
|`markersize`|마커 크기|

### 8-5. 복수 그래프 겹치기 (hold)

```python
x = np.arange(0, np.pi * 3, 0.1)
plt.figure(figsize=(10, 5))     # 그래프 전체 크기(가로, 세로)
plt.plot(x, y_sin, 'r')         # 꺾은선
plt.scatter(x, y_cos)           # 산점도
plt.xlabel('x 축')
plt.ylabel('y 축')
plt.title('sine & cosine')
plt.legend(['sine', 'cosine'])  # 범례
plt.show()
```

|함수|설명|
|---|---|
|`plt.plot()`|꺾은선 그래프|
|`plt.scatter()`|산점도|
|`plt.legend()`|범례 표시|
|`plt.figure(figsize=)`|캔버스 크기 설정|

### 8-6. subplot — 여러 그래프 분할 배치

```python
plt.subplot(2, 1, 1)   # (행수, 열수, 현재위치)
plt.plot(x, y_sin)
plt.title('sine')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('cosine')
plt.show()
```

> `plt.subplot(2, 1, 1)` → 2행 1열 구조의 **첫 번째** 칸

### 🔁 주요 함수 정리

|함수|설명|
|---|---|
|`plt.plot()`|꺾은선 그래프|
|`plt.scatter()`|산점도|
|`plt.subplot()`|그래프 분할|
|`plt.xlim() / ylim()`|축 범위 설정|
|`plt.xticks() / yticks()`|축 눈금 설정|
|`plt.xlabel() / ylabel()`|축 이름|
|`plt.title()`|그래프 제목|
|`plt.legend()`|범례|
|`plt.text()`|텍스트 표시|
|`plt.show()`|그래프 출력|

---

## 🔁 전체 흐름 요약

```
requests.get()          → 웹 페이지 요청
    ↓
BeautifulSoup()         → HTML/XML 파싱 객체 생성
    ↓
find() / find_all()     → 태그명, id, class, 정규식으로 탐색
select() / select_one() → CSS 셀렉터로 탐색
    ↓
.get_text() / .string / .attrs["속성"] → 데이터 추출
    ↓
pandas / CSV            → 정리 및 저장
```