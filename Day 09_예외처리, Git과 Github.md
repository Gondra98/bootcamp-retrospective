# Day 09_예외처리, Git과 Github

# 📅 2026-02-08

---
## 1. 예외처리 (try - except)
프로그램 실행 중 발생할 수 있는 **에러(예외)** 에 대비하여 프로그램이 **중단되지 않도록 처리**하는 문법이다.

파일, 네트워크, DB 작업, 계산 오류 등 **외부 요인이나 런타임 오류**에서 주로 사용한다.

### 기본개념
- **try** : 예외가 발생할 가능성이 있는 코드
    
- **except** : 예외 발생 시 실행할 코드
    
- **finally** : 예외 발생 여부와 관계없이 항상 실행
    

> try–except는 "쓸까 말까 고민되면 쓰는 게 맞다

### 1-1 기본문법 구조
```python
try:
    # 실행 코드 (예외 발생 가능)
    pass
except ExceptionType:
    # 예외 처리 코드
    pass
finally:
    # 항상 실행되는 코드
    pass
```

### 1-2 예제 코드
```python
def divide(a, b):
    return a / b

print('이런 저런 작업 진행...')

try:
    c = divide(5, 2)
    print(c)

    aa = [1, 2]
    # print(aa[3])

    open("c:/work/abc.txt")

except ZeroDivisionError:
    print('두번째 값은 0을 주면 안돼요')

except IndexError as err:
    print('참조 범위 오류 :', err)

except Exception as e:
    print('에러 :', e)

finally:
    print('에러 유무에 상관없이 반드시 수행됨')

print('end')

```

### 1-3. 예외 종류와 처리 순서
- `ZeroDivisionError` : 0으로 나눌 때
    
- `IndexError` : 리스트 인덱스 범위 초과
    
- `FileNotFoundError` : 파일이 없을 때
    
- `Exception` : 모든 예외의 상위 클래스

⚠️ **except 순서 중요**

- 구체적인 예외 → 추상적인 예외(Exception)
    
- `Exception`을 먼저 쓰면 아래 예외는 실행되지 않음

### 1-4. finally의 역할

- 예외 발생 여부와 관계없이 **항상 실행**
    
- 주 사용처
    
    - 파일 닫기
        
    - DB 연결 해제
        
    - 네트워크 종료
        
    - 리소스 정리

---
## 2. Git
### 2-1. Git 개요

- Git은 **분산 버전 관리 시스템(DVCS)**
    
- 여러 사람이 동시에 작업해도 각자의 로컬에서 독립적으로 버전 관리 가능
    
- 파일의 변경 사항을 **스냅샷 단위**로 저장
    

### 특징

- 로컬 저장소와 원격 저장소 생성 가능
    
- 파일 생성, 수정 내역 추적
    
- 브랜치(branch)를 사용한 독립적인 작업
    
- 브랜치 병합(merge) 지원
    
- master(main) 브랜치에 영향을 주지 않는 작업 가능
    

---

### 2-2. Git은 언제 쓰는 거야?

👉 **이 순간부터 Git을 쓴다**

- 코드/문서 작업을 **되돌릴 가능성**이 있을 때
    
- “어제 코드로 돌아가고 싶은데…?” 같은 상황
    
- 혼자 작업하지만 **실무처럼 관리**하고 싶을 때
    
- 나중에 **GitHub에 올릴 예정**일 때
    

📌 **GitHub 없어도 Git은 쓸 수 있음**  
👉 GitHub는 _나중에 붙이는 저장소_

---
### 2-3. Git 다운로드

1. 브라우저 열기
    
2. 아래 주소 접속  
    👉 [https://git-scm.com/](https://git-scm.com/)
    
3. **Download for Windows** 클릭
    
4. 설치 파일 다운로드 완료
    

---

### 2-4. Git 설치 (중요 옵션만 체크)

설치 중 옵션 많아서 당황함 😵  
👉 **아래만 지키면 됨**

#### ✔ 기본값으로 그냥 `Next` 누르기

딱 하나만 확인:

- **Adjusting your PATH environment**  
    👉 `Git from the command line and also from 3rd-party software` 선택되어 있는지
    

(보통 기본 선택임)

➡️ 끝까지 `Install`

---

### 2️-5. Git 설치 확인 (제일 중요)

#### 터미널 열기

아무거나 하나만 써도 됨:

- **Git Bash** (추천 👍)
    
- PowerShell
    
- VS Code 터미널
    

👉 처음엔 **Git Bash**가 제일 편함

---

#### 설치 확인

```
git --version
```

#### 정상 출력 예시

```
git version 2.44.0.windows.1
```

❗ 이거 나오면 설치 성공  
❌ 안 나오면 설치 실패

---

### 2-6. Git 최초 설정 (한 번만 하면 끝)

Git은 기록을 남기기 때문에  
**“이 기록 누가 남겼냐”**를 알아야 함

#### 이름 설정

```
git config --global user.name "본인이름"
```

예:

```
git config --global user.name "Sangwoo"
```

---

### 2-7. 이메일 설정

```
git config --global user.email "이메일"
```

예:

```
git config --global user.email "test@gmail.com"
```

👉 GitHub랑 같으면 좋음 (지금은 몰라도 됨)

---

### 2-8. 설정 확인

```
git config --global --list
```

출력 예:

```
user.name=Sangwoo
user.email=test@gmail.com
```

---

### 2-9. 이제 진짜 Git 시작 (폴더 준비)

#### 연습용 폴더 만들기

```
mkdir git-practice
cd git-practice
```

📂 지금 상태:

```
git-practice/
```

---

### 3-10. Git에게 말 걸기 (이 순간부터 시작)

```
git init
```

출력 예:

```
Initialized empty Git repository in ...
```

📂 폴더 안에 생긴 것:

```
.git/
```

❗ 이 순간부터:

> “이 폴더는 Git이 기억하는 폴더”

---

### 3-11. Git 상태 확인 (습관!)

```
git status
```

출력 예:

```
On branch main
No commits yet
nothing to commit
```

👉 해석:

- 아직 저장된 기록 없음
    
- 파일도 아직 없음
    

---

### 3-12. 버전으로 관리할 문서 작성 (Working Directory)

Git 저장소를 만들었다고 해서  
자동으로 파일이 저장되는 건 아님 ❌

👉 **먼저 파일을 직접 만든다**

예시:

```
echo hello > ex1.txt
```

📂 현재 상태

```
git-practice/
 ├─ ex1.txt
 └─ .git/
```

이 상태에서 Git은 이렇게 생각함:

> “파일이 하나 생겼네?  
> 근데 이걸 기록할지는 아직 모르겠어”

---

### 3-13. 저장소 상태 다시 확인

```
git status
```

출력 예:

```
Untracked files:
  ex1.txt
```

👉 의미:

- `ex1.txt`는 **아직 Git이 추적하지 않는 파일**
    
- 그냥 폴더에 있는 일반 파일 상태
    

📌 **중요 개념**

- Git은 **허락받은 파일만** 관리함
    
- 허락 = `git add`
    

---

### 3-14. 추적시킬 파일 등록 (Staging Area)

```
git add ex1.txt
```

다시 상태 확인:

```
git status
```

출력 예:

```
Changes to be committed:
  new file: ex1.txt
```

👉 의미:

- `ex1.txt`를 **저장해도 되는 후보**로 올려둔 상태
    
- 아직 저장(commit)된 건 아님
    

---

### 📦 Staging Area란?

👉 **“이 파일은 이번에 저장할 거야”라고 고르는 공간**

정리하면:

```
[Working Directory]  ← 파일 수정
        ↓ git add
[Staging Area]       ← 저장 후보
        ↓ git commit
[Repository]         ← 진짜 저장됨
```

📌 왜 굳이 이 단계를 나눌까?

- 파일 10개 중 **3개만 저장하고 싶을 때**
    
- 실수한 파일은 add 안 하면 됨
    
- 실무에서 진짜 많이 쓰임
    

---

### 3-15. 버전 만들기 (commit)

```
git commit -m "Add ex1.txt"
```

출력 예:

```
1 file changed
create mode 100644 ex1.txt
```

👉 이 순간:

- 하나의 **버전(스냅샷)** 이 생성됨
    
- Git이 이 시점을 기억함
    

📌 commit 메시지:

- “이 버전에서 뭘 했는지” 설명
    
- 나중에 **내가 나를 살리는 메모**
    

---

### 3-16. 버전 확인하기

```
git log
```

또는 간단하게:

```
git log --oneline
```

출력 예:

```
a1b2c3d Add ex1.txt
```

👉 의미:

- 커밋 ID (고유 번호)
    
- 커밋 메시지
    
- 이력이 쌓이는 구조
    

---

### 3-17. 수정 → 다시 저장 흐름

파일 수정:

```
echo world >> ex1.txt
```

상태 확인:

```
git status
```

출력 예:

```
modified: ex1.txt
```

👉 Git이 말함:

> “이 파일 바뀌었어.  
> 근데 아직 저장하진 않았어.”

---

### 3-18. 변경 내용 확인 (diff)

```
git diff
```

👉 **이전 버전과 현재 상태의 차이**를 보여줌

- 실무에서 진짜 많이 씀
    
- “내가 뭘 바꿨지?” 할 때 필수
    

---

### 3-19. 다시 add → commit

```
git add ex1.txt
git commit -m "Update ex1.txt"
```

👉 새로운 버전 하나 추가됨

---

### 🔁 지금까지 Git 핵심 흐름 요약

```
파일 생성/수정
   ↓
git status        (상태 확인)
   ↓
git add           (저장 후보로 올림)
   ↓
git commit        (버전 생성)
   ↓
git log / diff    (이력 확인)
```

---
### 3-20. staging과 commit을 한꺼번에 처리

내용:

```
git commit -am "메시지"
```

설명 포인트:

- `-a` : **이미 추적 중인 파일만** 자동으로 add
    
- 새 파일은 `git add` 먼저 해야 함 ❗
    

📌 실무 팁:

- 수정만 할 때는 `-am` 많이 씀
    
- 새 파일엔 안 먹힘 (중요)

---

### 3-21. 커밋 메시지 수정하기

```
git commit --amend
```

설명 포인트:

- **가장 최근 커밋만 수정 가능**
    
- 메시지 잘못 썼을 때 사용
    
- 커밋 ID가 바뀜 (중요)

---
### 3-22. Git 저장소 구조 요약

```
Working Directory
   ↓ git add
Staging Area
   ↓ git commit
Local Repository (.git)
```

한 줄 요약:

- Working: 내가 작업하는 곳
    
- Staging: 저장할 파일 고르는 곳
    
- Repository: 저장된 버전 창고

---
## 4. GitHub와 연결하기 (Remote Repository)

### 4-1. GitHub는 뭐하는 애야?

📌 지금까지 한 Git 작업은 전부 **내 컴퓨터 안(Local)** 에만 있음

👉 GitHub는:

- 내 로컬 저장소를 **인터넷에 백업**
    
- 다른 사람과 **공유**
    
- 다른 PC에서도 **이어 작업**
    

한 줄 요약:

> **Git = 버전 관리 도구**  
> **GitHub = 원격 저장소 서비스**

---

### 4-2. Git 작업 흐름 한 장 요약 (핵심)

```
[내 PC]
Working Directory
   ↓ git add
Staging Area
   ↓ git commit
Local Repository
   ↓ git push
------------------
[GitHub]
Remote Repository
```

👉 ❗중요

- GitHub로 **바로 저장되는 건 절대 아님**
    
- 반드시 `commit` → `push` 순서
    

---

### 4-3. GitHub 원격 저장소 생성

1. GitHub 로그인
    
2. **New Repository** 클릭
    
3. Repository name: `git-practice`
    
4. ❗ **README 생성 체크 해제**
    
5. Create repository
    

👉 왜 README 체크 해제?

- 로컬이랑 충돌 방지 (초보자 필수)
    

---

### 4-4. 원격 저장소 등록 (remote add)

GitHub에서 제공한 HTTPS 주소 복사 후:

```
git remote add origin https://github.com/USERNAME/git-practice.git
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

👉 의미:

- `origin` = 원격 저장소 별명
    
- 이제 **로컬 ↔ GitHub 연결 완료**
    

---

### 4-5. 로컬 커밋을 GitHub로 올리기 (push)

```
git push -u origin main
```

설명:

- `origin` : 원격 저장소 이름
    
- `main` : 로컬 브랜치
    
- `-u` : 앞으로 기본 연결로 기억
    

📌 처음 한 번만 `-u` 쓰면 됨  
이후엔:

```
git push
```

---

### 4-6. GitHub 기준 최종 흐름 정리

```
파일 수정
 → git add
 → git commit
 → git push
```

👉 이 3단계가 **실무에서도 그대로 쓰임**

---
### 4-7. README 파일과 원격 저장소 충돌 개념

GitHub에서 README를 먼저 만들면  
**원격 저장소에 이미 커밋이 존재**하게 된다.

👉 이 상태에서 발생할 수 있는 문제:

- 로컬에는 없는 커밋이 원격에 존재
    
- push 시 충돌 가능성 발생
    

📌 해결 방법 요약:

- 처음엔 README 체크 해제 권장
    
- 이미 만들었다면:
    
    - pull로 먼저 가져오거나
        
    - GitHub 가이드대로 초기 연결 진행
        

---

### 4-8. GitHub 인증 방식 변화 (중요)

과거 방식 ❌:

- 아이디 + 비밀번호
    

현재 방식 ✅:

- **Personal Access Token (PAT)** 사용
    

#### 토큰 인증이 필요한 이유

- 보안 강화
    
- 자동화/외부 접근 제어
    

👉 GitHub는 이제 **비밀번호 push 불가**

---

### 4-9. GitHub 토큰 인증 흐름 정리

```
git push 실행
 → GitHub 로그인 요청
 → 비밀번호 대신 토큰 입력
 → 인증 성공
```

📌 토큰 특징:

- 한 번 저장되면 매번 입력 안 할 수도 있음
    
- 비밀번호처럼 취급해야 함 (노출 금지)
    

---

### 4-10. 로컬 ↔ GitHub 전체 흐름 최종 정리

```
[Local]
파일 수정
 → git add
 → git commit
 → git push
-----------------
[GitHub]
Repository에 반영
```

👉 GitHub는 **commit된 것만** 받는다  
👉 add만 해서는 절대 안 올라감 ❌

---

### 4-11. 실무 기준 Git 사용 패턴 (혼자 작업)

```
1. 작업 시작
2. 파일 수정
3. git status
4. git add
5. git commit
6. git push
```

✔ 하루에 여러 번 반복  
✔ push는 “백업 버튼” 느낌