---
aliases:
  - Day 10_Github
  - 파일 처리 (File I/O)
  - 데이터베이스
---
# Day 10_Github, 파일 처리 (File I/O), 데이터베이스 

# 📅 2026-02-10

---
## 1. Github

### 1-1. Remote Repository란?

지금까지 만든 Git 저장소는 전부 **내 컴퓨터(Local)** 에만 존재했다.

👉 **Remote Repository(원격 저장소)** 는  
로컬 저장소를 **인터넷에 백업하고 공유**하기 위한 공간이다.

대표적인 원격 저장소 서비스:

- GitHub
    
- GitLab
    
- Bitbucket
    

한 줄 요약:

> **Git = 로컬 버전 관리**  
> **GitHub = 원격 저장소**

---

### 1-2. Local ↔ Remote 전체 구조

```
[Local PC]
Working Directory
   ↓ git add
Staging Area
   ↓ git commit
Local Repository
   ↓ git push
-----------------
[GitHub]
Remote Repository
```

📌 중요

- GitHub는 **commit된 것만** 받는다
    
- `git add`만으로는 업로드되지 않음 ❌
    

---

### 1-3. GitHub 저장소를 로컬로 가져오기 (clone)

이미 GitHub에 저장소가 있을 경우  
👉 **처음부터 로컬에 내려받는 방법**

```
git clone 저장소주소
```

예:

```
git clone https://github.com/USERNAME/git-practice.git
```

📌 clone 결과

- 폴더 자동 생성
    
- `.git` 포함
    
- remote(origin) 자동 등록
    

👉 `git init` + `remote add`를 **한 번에 처리**

---

### 1-4. clone 후 기본 상태 확인

clone 후 폴더 이동:

```
cd git-practice
```

상태 확인:

```
git status
```

출력 예:

```
On branch main
Your branch is up to date with 'origin/main'
```

👉 이미 원격(main)과 연결된 상태

---

### 1-5. Remote 이름(origin) 이해하기

원격 저장소는 **별명(alias)** 으로 관리된다.

기본 이름:

```
origin
```

확인:

```
git remote -v
```

출력 예:

```
origin  https://github.com/USERNAME/git-practice.git (fetch)
origin  https://github.com/USERNAME/git-practice.git (push)
```

👉 origin = 기본 원격 저장소

---

### 1-6. 로컬 변경 사항을 GitHub로 올리기 (push)

파일 수정 후:

```
git add .
git commit -m "update file"
```

원격으로 전송:

```
git push
```

📌 최초 push 시:

```
git push -u origin main
```

- `-u` : upstream 설정
    
- 이후에는 `git push`만 사용
    

---

### 1-7. GitHub 변경 사항 가져오기 (pull)

다른 사람이 수정했거나  
GitHub에서 직접 수정한 경우:

```
git pull
```

👉 원격 변경 사항을 로컬로 가져와 병합

📌 실무 습관:

> **작업 시작 전 pull**

---

### 1-8. 브랜치와 원격 저장소

브랜치는 로컬에서만 쓰는 개념이 아니다.

### 브랜치 생성

```
git switch -c feature-test
```

### 브랜치 push

```
git push -u origin feature-test
```

👉 GitHub에 브랜치 생성됨

---

### 1-9. Pull Request 개념 이해

Pull Request(PR)는:

> “내 브랜치를 main에 합쳐주세요”라는 요청

📌 목적

- 코드 리뷰
    
- 실수 방지
    
- 협업 기록 관리
    

---

### 1-10. .gitignore (가볍게)

Git이 **추적하지 않아야 할 파일**을 지정한다.

예:

```
node_modules/
.env
.DS_Store
```

📌 특징

- 추적 제외
    
- GitHub에도 안 올라감
    
- 초기에 설정 권장
    

---

### 1-11. git restore (commit 전 되돌리기)

commit 전에 수정한 내용을 되돌릴 때:

```
git restore 파일명
```

전체:

```
git restore .
```

⚠️ 변경 내용은 복구 불가

---

### 1-12. Git Bash vs VS Code

Git 사용 위치:

- Git Bash
    
- VS Code 터미널
    
- VS Code Source Control
    

👉 **기능 동일, 방식만 다름**

---

### 1-13. Remote 중심 Git 최종 흐름

```
clone 또는 init
 → 작업
 → add
 → commit
 → push
 → pull
```

---
## 2. 파일 처리 (File I/O)

프로그램에서 파일 처리는  
**데이터를 읽고, 저장하고, 다시 사용하는 작업**이다.

파일 작업은 외부 환경에 영향을 받기 때문에  
👉 **예외 처리와 함께 사용하는 것이 기본**

---
### 2-1. 파일 처리 전체 예제

```python
print("파일 처리 -----")
import os

try:
    print("파일 읽기 ------")
    print(os.getcwd())  # 현재 작업 경로 확인

    # 파일 읽기
    f1 = open("ftext.txt", mode='r', encoding='utf-8')
    print(f1)
    print(f1.read())
    f1.close()

    print("파일 저장 ------")
    # 파일 쓰기 (기존 내용 덮어쓰기)
    f2 = open('ftext.txt', mode='w', encoding='utf-8')
    f2.write('내 친구들\n')
    f2.write('홍길동, 한국인')
    f2.close()
    print('파일 저장 성공')

    print("파일 내용 추가 ------")
    # 파일 내용 추가
    f3 = open('ftext2.txt', mode='a', encoding='utf-8')
    f3.write('\n사오정')
    f3.write('\n저팔계')
    f3.write('\n손오공')
    f3.close()
    print('파일 추가 성공')

    # 추가된 파일 다시 읽기
    f4 = open('ftext2.txt', mode='r', encoding='utf-8')
    print(f4.read())
    f4.close()

except Exception as e:
    print('파일 처리 오류 : ', e)
```

---
### 2-2. 코드 흐름 설명

#### 1️⃣ 현재 작업 경로 확인

```
os.getcwd()
```

- 파일을 열 때 기준이 되는 위치
    
- 상대 경로(`"ftext.txt"`)가 어디를 의미하는지 확인용
    

---

#### 2️⃣ 파일 읽기 (`mode='r'`)

```
open("ftext.txt", mode='r', encoding='utf-8')
```

- 파일 내용을 읽기 전용으로 열기
    
- 파일이 없으면 예외 발생
    

---

#### 3️⃣ 파일 쓰기 (`mode='w'`)

```
open('ftext.txt', mode='w', encoding='utf-8')
```

- 파일에 내용 저장
    
- 기존 내용은 **모두 삭제 후 새로 작성**
    

⚠️ `w` 모드는 항상 초기화됨

---

#### 4️⃣ 파일 내용 추가 (`mode='a'`)

```
open('ftext2.txt', mode='a', encoding='utf-8')
```

- 기존 내용 유지
    
- 파일 끝에 내용 추가
    

---

#### 5️⃣ 파일은 반드시 close()

`f.close()`

- 파일 사용 종료
    
- 리소스 해제
    

---

#### 6️⃣ 파일 처리 + 예외 처리

```
try:
    ...
except Exception as e:
    ...
```

- 파일 없음
    
- 경로 오류
    
- 인코딩 문제  
    👉 전부 예외 발생 가능


그래서 파일 처리는 **try-except와 세트**

---

## 2-3. 파일 처리 모드 요약

|mode|의미|
|---|---|
|r|읽기|
|w|쓰기 (덮어쓰기)|
|a|내용 추가|
|b|바이너리|

---
### 2-4. with 구문을 이용한 파일 처리

파일을 열었으면 반드시 `close()`를 해야 한다.  
하지만 `with` 구문을 사용하면  
👉 **블록을 벗어나는 순간 자동으로 close()** 된다.

---
#### with 구문 파일 처리 예제

```python
# with 구문 사용 - 내부적으로 close() 함
try:
    # 파일 저장
    with open('ftext3.txt', mode='w', encoding='utf-8') as fobj1:
        fobj1.write('파이썬에서 문서저장\n')
        fobj1.write('with구문은\n')
        fobj1.write('명시적으로 close() 할 필요 없다\n')
    print('저장 완료')

    # 파일 읽기
    with open('ftext3.txt', mode='r', encoding='utf-8') as fobj2:
        print(fob제
```python
# 우편정보 파일 자료 읽기
# 키보드에서 입력한 동이름으로 해당 주소 정보 출력

def zipProcess():
    dongIrum = input('동이름 입력:')

    with open(r'zipcode.txt', mode='r', encoding='euc-kr') as f:
        line = f.readline()     # 한 행 읽기

        while line:
            # tab(\t) 기준으로 분리
            lines = line.split(chr(9))

            # 동 이름이 일치하면 출력
            if lines[3].startswith(dongIrum):
                print(
                    '우편번호:' + lines[0] + ', ' +
                    lines[1] + ' ' +
                    lines[2] + ' ' +
                    lines[3]
                )

            line = f.readline()

if __name__ == '__main__':
    zipProcess()
```

---
#### 코드 흐름 설명 (위 → 아래)

#### 1️⃣ 동 이름 입력

```python
dongIrum = input('동이름 입력:')
```

- 사용자가 검색할 **동 이름** 입력
    
- 예: `도곡`, `개포`
    

---

### 2️⃣ 파일 열기 (with 구문)

```python
with open(r'zipcode.txt', mode='r', encoding='euc-kr') as f:
```

- 우편번호 파일 읽기
    
- 한글 데이터 → `euc-kr` 인코딩
    
- `with` 사용 → 자동 close
    

---

### 3️⃣ 한 줄씩 읽기

```python
line = f.readline()
while line:
    ...
    line = f.readline()
```

- 파일 끝까지 반복
    
- 한 행씩 처리 → 대용량 파일에도 안전

---

### 4️⃣ 탭(tab) 기준 분리

```python
lines = line.split(chr(9))
```

- 우편번호 파일은 **탭으로 구분된 구조**
    
- `chr(9)` = `\t`

**아스키 코드 기본 암기**
```
97 → a
65 → A
10, 13 → Enter 역할
```

구조 예:

```
135-806	서울	강남구	개포1동 경남아파트
```

---

### 5️⃣ 동 이름 비교

```
if lines[3].startswith(dongIrum):
```

- 주소 항목에서 동 이름으로 시작하는 행만 선택
    
- 부분 검색 가능 (`도곡`, `개포` 등)
    

---

### 6️⃣ 검색 결과 출력

```python
print('우편번호:' + lines[0] + ', ' + lines[1] + ' ' + lines[2] + ' ' + lines[3])
```

- 우편번호 + 시 + 구 + 동 출력

---

### 7️⃣ 프로그램 시작점

```python
if __name__ == '__main__':
    zipProcess()
```

- 직접 실행할 때만 함수 호출
    
- 모듈화 기본 패턴

---
## 3. 데이터베이스 (MariaDB 기초)

### 📌 데이터베이스와 MariaDB 개념

- **Database(DB)**  
    : 데이터를 **체계적으로 저장·관리**하는 공간
    
- **DBMS(Database Management System)**  
    : 데이터베이스를 **생성, 저장, 조회, 관리**해주는 소프트웨어

| 구분    | RDBMS          | NoSQL          |
| ----- | -------------- | -------------- |
| 구조    | 테이블(행·열)       | 비정형 구조         |
| 스키마   | 고정             | 유연             |
| 사용 언어 | SQL            | 전용 쿼리          |
| 대표 DB | MariaDB, MySQL | MongoDB, Redis |

---
### 📌 MariaDB란?

- MySQL 기반의 **오픈소스 RDBMS**
    
- SQL 언어 사용
    
- 빠른 성능과 높은 호환성
    
- MySQL과 사용법 거의 동일
    

👉 **MariaDB = RDBMS + SQL 사용**

---

### 3-1. MariaDB 설치 및 접속

- MariaDB 설치
    
- root 비밀번호 설정
    
- DB 서버 접속
    

```
mysql -u root -비밀번호
```

비밀번호 입력 후 MariaDB 접속

---

### 3-2. 데이터베이스 생성 및 선택

`CREATE DATABASE testdb; USE testdb;`

- `DATABASE` : 데이터를 저장하는 공간
    
- `USE` : 사용할 데이터베이스 선택
    

---

### 3-3. 테이블 생성 실습

```sql
CREATE TABLE test_table (
    id INT,
    name VARCHAR(10)
);
```

사용한 자료형:

- `INT` : 정수형 데이터
    
- `VARCHAR(10)` : 문자열 (최대 10자)
    

👉 컬럼 2개짜리 테이블 생성 실습

---

### 3-4. 테이블 확인

```sql
SHOW TABLES;
DESC test_table;
```

- 테이블 목록 확인
    
- 컬럼 구조 확인
    

---

### 3-5. 데이터 입력 및 조회 (간단)

```sql
INSERT INTO test_table VALUES (1, 'kim');
SELECT * FROM test_table;
```

- 데이터 추가
    
- 전체 데이터 조회