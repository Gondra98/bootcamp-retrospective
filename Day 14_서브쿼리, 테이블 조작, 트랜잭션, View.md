# Day 14_서브쿼리, 테이블 조작, 트랜잭션, View

# 📅 2026-02-23

---

## 1️⃣ 서브쿼리(Subquery)

### 1-1. 총무부 직원이 관리하는 고객 (Subquery)

```sql
SELECT gogekno, gogekname, gogektel   
FROM gogek  
WHERE gogekdamsano IN (  
    SELECT jikwonno FROM jikwon  
    WHERE busernum = (  
        SELECT buserno FROM buser WHERE busername='총무부'  
    )  
);
```

**출력값 예시:**

|gogekno|gogekname|gogektel|
|---|---|---|
|101|한송이|333-3333|
|102|장영희|444-4444|

> ✅ Subquery 3단계로 조건 적용

---

### 1-2. JOIN으로 동일 결과

```sql
SELECT gogekno, gogekname, gogektel   
FROM gogek  
INNER JOIN jikwon ON jikwon.jikwonno=gogek.gogekdamsano  
INNER JOIN buser ON jikwon.busernum=buser.buserno  
WHERE busername='총무부';
```

**출력값:** Subquery와 동일

**그림 설명 (JOIN 조건)**

```
jikwon                 buser                 gogek  
---------              --------             ---------  
jikwonno, busernum     buserno, busername    gogekno, gogekdamsano  
김철수, 10             10, 총무부           101, 10  
홍길동, 10             10, 총무부           102, 10
```

---

## 2️⃣ ANY / ALL 연산자

```sql
-- '대리'의 최대값보다 작은 연봉  
SELECT jikwonno, jikwonname, jikwonpay  
FROM jikwon  
WHERE jikwonpay < ANY (SELECT jikwonpay FROM jikwon WHERE jikwonjik='대리');
```

|jikwonno|jikwonname|jikwonpay|
|---|---|---|
|7|김하나|28000|
|8|박철수|27000|

```sql
-- 30번 부서 최고 연봉보다 높은 직원  
SELECT jikwonno, jikwonname, jikwonpay  
FROM jikwon  
WHERE jikwonpay > ALL (SELECT jikwonpay FROM jikwon WHERE busernum=30);
```

|jikwonno|jikwonname|jikwonpay|
|---|---|---|
|4|박민준|40000|
|5|최수진|42000|

---

## 3️⃣ EXISTS / NOT EXISTS

```sql
-- 직원이 있는 부서  
SELECT busername, buserloc  
FROM buser bu  
WHERE EXISTS (  
    SELECT 'imsi' FROM jikwon WHERE jikwon.busernum = bu.buserno  
);
```

|busername|buserloc|
|---|---|
|총무부|서울|
|영업부|부산|
|전산부|대전|

```sql
-- 직원 없는 부서  
SELECT busername, buserloc  
FROM buser bu  
WHERE NOT EXISTS (  
    SELECT 'imsi' FROM jikwon WHERE jikwon.busernum = bu.buserno  
);
```

|busername|buserloc|
|---|---|
|기획실|광주|

---

## 4️⃣ 상관 서브쿼리

```sql
-- 각 부서 최대 연봉자  
SELECT *   
FROM jikwon a  
WHERE a.jikwonpay = (  
    SELECT MAX(b.jikwonpay)   
    FROM jikwon b  
    WHERE a.busernum = b.busernum  
);
```

|jikwonno|jikwonname|jikwonjik|jikwonpay|busernum|
|---|---|---|---|---|
|4|박민준|과장|40000|10|
|5|최수진|과장|42000|20|
|6|이영희|대리|35000|30|

---

## 5️⃣ 테이블 생성 / 삽입 / 수정 / 삭제

### 5-1. 테이블 생성

```sql
CREATE TABLE jiktab1 AS SELECT * FROM jikwon;  
CREATE TABLE jiktab2 AS SELECT * FROM jikwon WHERE 1=0;
```

**그림: 테이블 구조**

```
jikwon  
------  
jikwonno, jikwonname, jikwonjik, jikwonpay, busernum  
  
jiktab1  
------  
동일 구조 + 데이터 포함  
  
jiktab2  
------  
동일 구조, 데이터 없음
```

### 5-2. INSERT + Subquery

```sql
INSERT INTO jiktab2 SELECT * FROM jikwon WHERE jikwonjik='과장';
```

|jikwonno|jikwonname|jikwonjik|jikwonpay|busernum|
|---|---|---|---|---|
|4|박민준|과장|40000|10|
|5|최수진|과장|42000|20|

### 5-3. UPDATE + Subquery

```sql
UPDATE jiktab1  
SET jikwonjik = (SELECT jikwonjik FROM jikwon WHERE jikwonname='이순신')  
WHERE jikwonno = 2;
```

- 2번 직원 직급이 **이순신과 동일**으로 변경
    

### 5-4. DELETE + Subquery

```sql
DELETE FROM jiktab1  
WHERE jikwonno IN (SELECT DISTINCT gogekdamsano FROM gogek);
```

- 고객 담당 직원 삭제
    
- 남은 직원만 테이블에 존재
    

---

## 6️⃣ 트랜잭션(Transaction)

- - **트랜잭션**: DB에서 상태를 변경하는 **논리적 작업 단위**
    
- 예: `INSERT`, `UPDATE`, `DELETE` 등
    
- 트랜잭션 특징: **ACID**
    
    1. **Atomicity (원자성)** – 작업 전체가 모두 수행되거나, 모두 취소됨
        
    2. **Consistency (일관성)** – DB 규칙/제약 조건 유지
        
    3. **Isolation (격리성)** – 다른 트랜잭션 간 간섭 방지
        
    4. **Durability (지속성)** – 완료된 트랜잭션은 영구 저장

```sql
SET autocommit = FALSE;  
  
DELETE FROM jiktab3 WHERE jikwonno=2;  
ROLLBACK;   -- 삭제 취소  
COMMIT;     -- 확정  
SET autocommit = TRUE;
```

**그림: 트랜잭션 흐름**

```
트랜잭션 시작  
   |  
   |-- 작업1: DELETE / UPDATE  
   |  
   |-- SAVEPOINT (부분 저장)  
   |  
   |-- 작업2: UPDATE  
   |  
   |-- ROLLBACK TO SAVEPOINT  -> 작업2만 취소  
   |  
   |-- COMMIT                 -> 작업1 확정
```

### 6-1 COMMIT / ROLLBACK 개념

| 명령어          | 설명                                   |
| ------------ | ------------------------------------ |
| **COMMIT**   | 현재 트랜잭션에서 수행한 작업을 **확정**. DB에 영구 반영  |
| **ROLLBACK** | 현재 트랜잭션에서 수행한 작업을 **취소**. 이전 상태로 되돌림 |

- 트랜잭션이 시작되면, DB는 **작업을 임시 공간(버퍼/트랜잭션 로그)에 저장**함
    
- `COMMIT` → 임시 공간 내용을 DB에 확정 반영
    
- `ROLLBACK` → 임시 공간 내용 폐기, DB 원래 상태 유지

### 6-2 예시 테이블
| jikwonno | jikwonname | jikwonpay |
| -------- | ---------- | --------- |
| 1        | 김철수        | 30000     |
| 2        | 이순신        | 32000     |
| 3        | 이미라        | 28000     |
| 4        | 박민준        | 40000     |
| 5        | 최수진        | 42000     |
> 이 테이블을 `jiktab3`라고 가정

### 6-3 트랜잭션 실행 예시

```sql
SET autocommit = FALSE;    -- 자동 커밋 OFF

DELETE FROM jiktab3 WHERE jikwonno=2;  
SELECT * FROM jiktab3;    -- 임시 상태 확인
ROLLBACK;                  -- 삭제 취소
SELECT * FROM jiktab3;    -- 원래 상태로 복원
SET autocommit = TRUE;     -- 자동 커밋 ON
```
**변화 그림**

1. 트랜잭션 시작

|jikwonno|jikwonname|jikwonpay|
|---|---|---|
|1|김철수|30000|
|2|이순신|32000|
|3|이미라|28000|
|4|박민준|40000|
|5|최수진|42000|

2. DELETE 실행 (임시 상태, DB 서버에는 아직 반영 X)

|jikwonno|jikwonname|jikwonpay|
|---|---|---|
|1|김철수|30000|
|3|이미라|28000|
|4|박민준|40000|
|5|최수진|42000|

3. `ROLLBACK` → 원래 상태로 복원

|jikwonno|jikwonname|jikwonpay|
|---|---|---|
|1|김철수|30000|
|2|이순신|32000|
|3|이미라|28000|
|4|박민준|40000|
|5|최수진|42000|
✅ DELETE가 취소됨 → **원자성 보장**


### 6-4 UPDATE + SAVEPOINT + ROLLBACK TO SAVEPOINT

```sql
SET autocommit = FALSE;

UPDATE jiktab3 SET jikwonpay=7777 WHERE jikwonno=4;  -- 작업1
SAVEPOINT a;                                        -- 저장점 생성
UPDATE jiktab3 SET jikwonpay=8888 WHERE jikwonno=5;  -- 작업2
ROLLBACK TO SAVEPOINT a;                             -- 작업2만 취소
COMMIT;                                              -- 작업1 확정
SET autocommit = TRUE;
```

**변화 그림**

1. 초기 상태

|jikwonno|jikwonname|jikwonpay|
|---|---|---|
|4|박민준|40000|
|5|최수진|42000|

2. UPDATE 작업1 → 임시 상태

|jikwonno|jikwonname|jikwonpay|
|---|---|---|
|4|박민준|7777|
|5|최수진|42000|

3. SAVEPOINT a 생성
    
4. UPDATE 작업2 → 임시 상태
    

|jikwonno|jikwonname|jikwonpay|
|---|---|---|
|4|박민준|7777|
|5|최수진|8888|

5. `ROLLBACK TO SAVEPOINT a` → 작업2만 취소

|jikwonno|jikwonname|jikwonpay|
|---|---|---|
|4|박민준|7777|
|5|최수진|42000|

6. `COMMIT` → 작업1 확정

|jikwonno|jikwonname|jikwonpay|
|---|---|---|
|4|박민준|7777|
|5|최수진|42000|

> ✅ SAVEPOINT 활용 → 부분 롤백 가능  
> ✅ COMMIT → 트랜잭션 최종 확정


**요약 그림**
```
트랜잭션 시작
   |
   |-- 작업1: DELETE / UPDATE
   |-- SAVEPOINT (부분 저장)
   |-- 작업2: UPDATE
   |-- ROLLBACK TO SAVEPOINT  -> 작업2만 취소
   |-- COMMIT                 -> 작업1 확정
```

---

## 7️⃣ Deadlock (교착상태)

- **정의**:  
    두 개 이상의 트랜잭션이 서로 상대방이 가진 **락(Lock)**을 기다리면서 **영원히 진행하지 못하는 상태**
    
- **원인**:
    
    1. 여러 트랜잭션이 **동일 테이블, 서로 다른 Row를 동시에 수정**
        
    2. 트랜잭션 **Row 접근 순서가 서로 다름**
        
    3. 트랜잭션이 **너무 오래 지속**
        
- **발생 예시**:
    

```
트랜잭션 A                   트랜잭션 B  
-------------------          -------------------  
UPDATE jiktab3 SET ...        UPDATE jiktab3 SET ...  
WHERE jikwonno=7              WHERE jikwonno=8  
Row 7 Lock                     Row 8 Lock  
WAIT Row 8                     WAIT Row 7  <-- 서로 대기
```

- **MariaDB / MySQL 상황**:
    

```sql
SET autocommit = FALSE;  
  
-- 트랜잭션 A  
UPDATE jiktab3 SET jikwonpay=1234 WHERE jikwonno=7;  
  
-- 트랜잭션 B (동시 접근)  
DELETE FROM jiktab3 WHERE jikwonno=7;  
-- ERROR 1205: Lock wait timeout exceeded
```

- **특징**:
    
    - Lock Wait Timeout 발생 가능 (`ERROR 1205`)
        
    - DBMS는 보통 **Deadlock 감지 후 한 트랜잭션 롤백**
        
- **해결/예방 방법**:
    
    1. 트랜잭션을 **짧게 유지**
        
    2. **Row 접근 순서를 일정하게**
        
    3. **SAVEPOINT**로 부분 롤백 가능
        
    4. 작업 완료 후 **COMMIT / ROLLBACK**으로 락 해제
        
- **트랜잭션 + Deadlock 흐름 그림**:
    

트랜잭션 A  
-----------------  
```
SELECT * FROM jiktab3 WHERE jikwonno=7  
UPDATE jiktab3 SET jikwonpay=1234  <-- Row 7 Lock  
SAVEPOINT a  
UPDATE jiktab3 SET jikwonpay=8888  <-- Row 8 Lock  
ROLLBACK TO SAVEPOINT a            <-- 부분 취소  
COMMIT                              <-- Row 7 Lock 해제  
```
  
트랜잭션 B (동시)  
-----------------  
```
DELETE FROM jiktab3 WHERE jikwonno=7  
  --> Row 7 Lock 필요  
  --> LOCK WAIT 발생  
  --> 트랜잭션 A COMMIT 후 수행 가능
```

> 🔹 Deadlock = 서로 기다림 → 무한 대기  
> 🔹 해결 = COMMIT / ROLLBACK  
> 🔹 예방 = 트랜잭션 짧게, Row 접근 순서 일정, SAVEPOINT 활용


---

## 8️⃣ VIEW (뷰)

### ✅ View 개념

- **물리 테이블을 기반으로 한 SELECT 문을 저장한 가상 테이블**
    
- 실제 데이터를 저장하지 않음 (논리적 객체)
    
- 장점:
    
    - ✔ 복잡한 쿼리 단순화
        
    - ✔ 보안 강화 (필요 컬럼만 노출)
        
    - ✔ 데이터 독립성 확보
        
    - ✔ 메모리 사용 거의 없음
        

---

### ✅ 기본 문법

```sql
CREATE OR REPLACE VIEW 뷰이름 AS SELECT문;  
ALTER VIEW 뷰이름 ...  
DROP VIEW 뷰이름;
```

---

### 8-1. 단순 View 생성

```sql
CREATE OR REPLACE VIEW v_a AS  
SELECT jikwonno, jikwonname, jikwonpay   
FROM jikwon   
WHERE jikwonibsail < '2010-12-31';
```

#### 🔹 실행 결과

```sql
SELECT * FROM v_a;
```

|jikwonno|jikwonname|jikwonpay|
|---|---|---|
|1|김철수|3000|
|2|이영희|3500|

(※ 실제 데이터에 따라 달라짐)

---

#### 🔹 View 확인

```sql
SHOW FULL TABLES IN test WHERE table_type LIKE 'VIEW';
```

출력 예:

```
v_a  
v_b  
v_c
```

---

### 8-2. View는 테이블처럼 조회 가능

```sql
SELECT SUM(jikwonpay) AS 연봉합 FROM v_a;
```

출력 예:

연봉합  
6500

---

### 8-3. View와 원본 테이블 관계

```sql
ALTER TABLE jikwon RENAME kbs;  
SELECT * FROM v_b;   -- 오류 발생
```

❗ 이유:

- View는 **원본 테이블 구조에 의존**
    
- 테이블 이름 바꾸면 View 깨짐
    

다시 복구:

```sql
ALTER TABLE kbs RENAME jikwon;
```

→ 정상 작동

---

### 8-4. 계산 컬럼 View

```sql
CREATE VIEW v_d AS   
SELECT jikwonno, jikwonname, jikwonpay * 10000 AS ypay   
FROM jikwon;
```

### 🔹 결과

|jikwonno|jikwonname|ypay|
|---|---|---|
|1|김철수|30000000|

---

#### ❗ 계산 컬럼은 직접 수정 불가

```sql
UPDATE v_d SET ypay=1111 WHERE jikwonname='홍길동';
```

➡ 오류 발생  
이유:

- `ypay`는 계산된 컬럼
    
- 실제 테이블 컬럼이 아님
    

---

### 8-5. View를 통한 UPDATE / DELETE

```sql
DELETE FROM v_d WHERE jikwonname = '최미숙';
```

✔ 가능  
→ 실제 jikwon 테이블에서도 삭제됨

DELETE FROM v_d WHERE ypay=41000000;

✔ 가능  
→ 조건에 계산 컬럼 사용은 가능

---

### 8-6. 조건 있는 View (삽입 주의)

```sql
CREATE VIEW v_f AS  
SELECT * FROM jikwon  
WHERE jikwonibsail < '2015-1-1';

INSERT INTO v_f VALUES(33, '주먹밥', 10, 7000, '2025-5-6');
```

✔ INSERT 성공  
❗ 하지만 View에는 안보임

왜?

조건: jikwonibsail < '2015-1-1'  
입력값: 2025-5-6 → 조건 불충족

→ jikwon에는 들어가지만 v_f에서는 조회 안됨

---

### 8-7. GROUP BY View

```sql
CREATE VIEW v_group AS  
SELECT jikwonjik, SUM(jikwonpay) AS hap, AVG(jikwonpay) AS ave  
FROM jikwon GROUP BY jikwonjik;
```

✔ 조회 가능  
❌ INSERT / UPDATE / DELETE 불가

이유:

- 집계 함수 사용 → 단일 Row가 아님
    

---

### 8-8. JOIN View

```sql
CREATE VIEW v_join AS  
SELECT jikwonno, jikwonname, busername, jikwonjik   
FROM jikwon  
INNER JOIN buser ON jikwon.busernum=buser.buserno  
WHERE jikwon.busernum IN (10, 20);
```

### 🔹 조회 결과

|jikwonno|jikwonname|busername|jikwonjik|
|---|---|---|---|
|1|김철수|영업부|사원|

---

### 🔹 수정 가능 조건

```sql
UPDATE v_join   
SET jikwonname = '손오공'   
WHERE jikwonname = '박명화';
```

✔ 가능 (한 테이블 컬럼만 수정 시)

---

```sql
UPDATE v_join   
SET jikwonname='사오정', busername='영업부'  
WHERE jikwonname='손오공';
```

❌ 오류 발생

이유:

- Join View는 **한 개 테이블만 수정 가능**
    
- 두 테이블 동시 수정 불가
    

---

```sql
DELETE FROM v_join WHERE jikwonname='손오공';
```

❌ MariaDB에서는 삭제 불가  
(Oracle은 가능)

---

## 🔥 View 핵심 정리

|종류|조회|INSERT|UPDATE|DELETE|
|---|---|---|---|---|
|단순 View|O|O|O|O|
|계산 컬럼|O|O|일부불가|O|
|조건 View|O|O|O|O|
|GROUP BY|O|X|X|X|
|JOIN|O|제한적|제한적|제한적|

---
## 📌 문1) v_exam1

```sql
/*  
 문1) 사번   이름    부서    직급     근무년수   고객확보  
       1   홍길동  영업부   사원        6          O 또는 X  
  
 조건 :  
 1. 직급이 없으면 '임시직'  
 2. 전산부 자료는 제외  
*/  
  
CREATE OR REPLACE VIEW v_exam1 AS  
SELECT   
    jikwonno AS 사번,  
    jikwonname AS 이름,  
    busername AS 부서,  
    NVL(jikwonjik, '임시직') AS 직급,  
    DATE_FORMAT(NOW(), '%Y') - DATE_FORMAT(jikwonibsail, '%Y') AS 근무년수,  
    CASE NVL(gogekname, 'a')  
        WHEN 'a' THEN 'X'  
        ELSE 'O'  
    END AS 고객확보  
FROM jikwon  
LEFT OUTER JOIN buser   
    ON busernum = buserno  
LEFT OUTER JOIN gogek   
    ON jikwonno = gogekdamsano  
WHERE busername <> '전산부'   
   OR busername IS NULL;
```


### 🔎 핵심 정리

- `NVL(jikwonjik, '임시직')` → 직급 NULL 처리
    
- 고객 존재 여부 → CASE 처리
    
- 전산부 제외 → WHERE 조건
    
- LEFT JOIN → 고객 없는 직원도 포함
    

---
## 📌 문2) v_exam2

```sql
/*  
 문2) 부서명   인원수  
  
 조건 :  
 직원수가 가장 많은 부서 1개 출력  
*/  
  
CREATE OR REPLACE VIEW v_exam2 AS  
SELECT   
    busername AS 부서명,  
    COUNT(busername) AS 인원수  
FROM jikwon  
INNER JOIN buser   
    ON jikwon.busernum = buser.buserno  
GROUP BY busername  
ORDER BY COUNT(busername) DESC  
LIMIT 1;
```

### 🔎 핵심 정리

- GROUP BY → 부서별 집계
    
- ORDER BY COUNT DESC → 인원 많은 순
    
- LIMIT 1 → 가장 많은 부서 1개
    

✔ 실전에서 이렇게 작성해도 충분히 정답 처리됨

---
## 📌 문3) v_exam3

```sql
/*  
 문3) 가장 많은 직원이 입사한 요일에 입사한 직원 출력  
  
 출력:  
 직원명   요일   부서명   부서전화  
*/  
  
CREATE OR REPLACE VIEW v_exam3 AS  
SELECT   
    jikwonname AS 직원명,  
    DATE_FORMAT(jikwonibsail, '%a') AS 요일,  
    busername AS 부서명,  
    busertel AS 부서전화  
FROM jikwon  
INNER JOIN buser   
    ON jikwon.busernum = buser.buserno  
WHERE DATE_FORMAT(jikwonibsail, '%a') = (  
    SELECT DATE_FORMAT(jikwonibsail, '%a')  
    FROM jikwon  
    GROUP BY DATE_FORMAT(jikwonibsail, '%a')  
    ORDER BY COUNT(*) DESC  
    LIMIT 1  
);
```

### 🔎 핵심 정리

1. 서브쿼리에서 요일별 인원수 집계
    
2. 가장 많은 요일 1개 추출
    
3. 그 요일과 같은 직원만 출력
    

✔ 상관서브쿼리 아님 (단일값 반환 서브쿼리)  
✔ 시험용으로 완전 적절한 구조
