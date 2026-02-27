# Day 13_집계, JOIN, 서브쿼리
# 📅 2026-02-13
---
## `SELECT` 동작 원리 (DB → RAM)

- `SELECT`는 **DB 서버에 저장된 테이블 데이터를 조회**하는 것
    
- 내부적으로:
    
    1. **테이블에서 필요한 행/컬럼 읽기** (`FROM`, `WHERE`)
        
    2. **필요한 계산/집계 수행** (`GROUP BY`, `HAVING`, `SUM`, `AVG` 등)
        
    3. **최종 결과를 클라이언트(또는 쿼리 실행 환경)로 전송**
        
- 실제로는 **DB 서버 메모리(RAM)에 데이터를 올리고 처리** → 결과만 가져옴
    
- 그래서 대용량 테이블을 그냥 SELECT * 하면 서버 부담 + 네트워크 부담이 커짐
---
## 데이터 타입 관점

- **정형(Structured)** → 행/열 구조, 예: `jikwon` 테이블
    
- **반정형(Semi-structured)** → 키-값, JSON 등, 일부 구조 있음
    
- **비정형(Unstructured)** → 이미지, 영상, 텍스트, 로그 등
    
- SELECT는 **정형/반정형** 데이터는 바로 쿼리 가능
    
- 비정형은 별도 변환/처리가 필요

---

## 1️⃣ 집계함수 & 조건 (Aggregate Functions)

- **집계함수**: `COUNT`, `SUM`, `AVG`, `MAX`, `MIN`
    
- 조건절 `WHERE`와 함께 사용 가능
    
- NULL 값은 집계 제외됨 (단, `COUNT(*)`는 NULL 포함)
    

### 예시1. 과장 인원수

```sql
SELECT COUNT(*) AS 인원수
FROM jikwon
WHERE jikwonjik = '과장';
```

**출력 예시:**

|인원수|
|---|
|5|

> ✅ `COUNT(*)` → 조건에 맞는 **행의 수**를 계산

---

### 예시2. 2010년 이전 입사 남직원

```sql
SELECT COUNT(*) AS 인원수  
FROM jikwon  
WHERE jikwonibsail < '2010-01-01'  
  AND jikwongen = '남';
```

**출력 예시:**

|인원수|
|---|
|12|

> ✅ 날짜 비교 가능, `WHERE` 절로 조건 걸기

---

### 예시3. 2015년 이후 입사 여직원 연봉 합/평균/인원수

```sql
SELECT SUM(jikwonpay) AS 연봉합,  
       AVG(jikwonpay) AS 연봉평균,  
       COUNT(*) AS 인원수  
FROM jikwon  
WHERE jikwonibsail > '2015-01-01'  
  AND jikwongen = '여';
```

**출력 예시:**

|연봉합|연봉평균|인원수|
|---|---|---|
|78000|26000|3|

> ✅ `SUM` = 합계, `AVG` = 평균, `COUNT` = 행 수  
> ✅ 필터 조건 적용 후 집계 가능

---

## 2️⃣ GROUP BY & HAVING

- `GROUP BY` → 그룹별 집계
    
- `HAVING` → 그룹 결과에 조건 적용 (`WHERE`와 다름!)
    

### 예시1. 성별 연봉 평균, 인원수

```sql
SELECT jikwongen, AVG(jikwonpay), COUNT(*)
FROM jikwon
GROUP BY jikwongen;
```

**출력 예시:**

|jikwongen|AVG(jikwonpay)|COUNT(*)|
|---|---|---|
|남|32000|15|
|여|28000|10|

> ✅ 그룹별 집계 → 성별별 평균/인원 계산

---

### 예시2. 부서별 연봉합 35000 이상

```sql
SELECT busernum, SUM(jikwonpay)
FROM jikwon
GROUP BY busernum
HAVING SUM(jikwonpay) >= 35000;
```

**출력 예시:**

|busernum|SUM(jikwonpay)|
|---|---|
|10|42000|
|20|36000|

> ✅ `HAVING`은 그룹 집계 결과에 조건 적용 가능

---

## 3️⃣ NULL 처리 / NVL

- `NVL(column, '대체값')` → NULL 대신 지정값 사용
    
- MariaDB / MySQL은 `IFNULL(column, '대체값')` 사용
    

```sql
SELECT NVL(jikwonjik,'임시직') AS 직급, SUM(jikwonpay)  
FROM jikwon  
GROUP BY 직급  
HAVING SUM(jikwonpay) >= 7000;
```

**출력 예시:**

|직급|SUM(jikwonpay)|
|---|---|
|사원|15000|
|대리|12000|
|과장|18000|
|임시직|7500|

> ✅ NULL 직급도 **임시직**으로 처리하고 집계 가능

---

## 4️⃣ JOIN 활용

### 4-1. EQUI JOIN (동등 조건)

```sql
SELECT jikwonname, busername  
FROM jikwon, buser  
WHERE jikwon.busernum = buser.buserno;
```

**출력값**

| jikwonname | busername |
| ---------- | --------- |
| 김철수        | 총무부       |
| 이순신        | 영업부       |
| 박영수        | 전산부       |


**그림 설명:**

```
조건: jikwon.busernum = buser.buserno

jikwon         buser
---------      ---------
김철수,10      10,총무부
이순신,20      20,영업부
박영수,30      30,전산부
홍길동,NULL    -
김영희,NULL    -
```

> ✅ 조건 일치 행만 출력

---
### 4-2. CROSS JOIN (모든 조합)

```sql
SELECT jikwonname, busername  
FROM jikwon CROSS JOIN buser;
```

**출력값 (일부)**

| jikwonname | busername |
| ---------- | --------- |
| 김철수        | 총무부       |
| 김철수        | 영업부       |
| 김철수        | 전산부       |
| 김철수        | 기획실       |
| 이순신        | 총무부       |
| 이순신        | 영업부       |
| 이순신        | 전산부       |
| 이순신        | 기획실       |
| 홍길동        | 총무부       |
| 홍길동        | 영업부       |
| 홍길동        | 전산부       |
| 홍길동        | 기획실       |
| 김영희        | 총무부       |
| 김영희        | 영업부       |
| 김영희        | 전산부       |
| 김영희        | 기획실       |


**그림 설명:**

```
모든 행 조합 생성

jikwon         buser
---------      ---------
김철수         총무부
김철수         영업부
김철수         전산부
김철수         기획실
이순신         총무부
이순신         영업부
이순신         전산부
이순신         기획실
홍길동         총무부
홍길동         영업부
홍길동         전산부
홍길동         기획실
김영희         총무부
김영희         영업부
김영희         전산부
김영희         기획실
```

> ✅ 모든 행 조합 생성

---

### 4-3. INNER JOIN + 조건 필터

```sql
SELECT jikwonname, busername  
FROM jikwon  
INNER JOIN buser ON jikwon.busernum = buser.buserno  
WHERE jikwongen = '남';
```

**출력값**

| jikwonname | busername |
| ---------- | --------- |
| 김철수        | 총무부       |
| 이순신        | 영업부       |
| 박영수        | 전산부       |


**그림 설명:**

```
남직원만 + 양쪽 데이터 모두 존재해야 출력

jikwon (남)     buser
---------      ---------
김철수,10      10,총무부
이순신,20      20,영업부
박영수,30      30,전산부
홍길동,NULL    -
```

---

### 4-4. LEFT OUTER JOIN (왼쪽 테이블 모두 포함)

```sql
SELECT jikwonname, busername  
FROM jikwon  
LEFT JOIN buser ON jikwon.busernum = buser.buserno;
```

**출력값**

|jikwonname|busername|
|---|---|
|김철수|총무부|
|이순신|영업부|
|박영수|전산부|
|홍길동|NULL|
|김영희|NULL|


**그림 설명:**

```
왼쪽 테이블(jikwon) 모든 행 포함
부서 없는 직원 → busername=NULL

jikwon         buser
---------      ---------
김철수,10      10,총무부
이순신,20      20,영업부
박영수,30      30,전산부
홍길동,NULL    -
김영희,NULL    -
```

---

### 4-5. RIGHT OUTER JOIN (오른쪽 테이블 모두 포함)

```sql
SELECT jikwonname, busername  
FROM jikwon  
RIGHT JOIN buser ON jikwon.busernum = buser.buserno;
```

**출력값**

| jikwonname | busername |
| ---------- | --------- |
| 김철수        | 총무부       |
| 이순신        | 영업부       |
| 박영수        | 전산부       |
| NULL       | 기획실       |


**그림 설명:**

```
오른쪽 테이블(buser) 모든 행 포함
직원 없는 부서 → jikwonname=NULL

jikwon         buser
---------      ---------
김철수,10      10,총무부
이순신,20      20,영업부
박영수,30      30,전산부
NULL          50,기획실
```


---

### 4-6. FULL OUTER JOIN (LEFT + RIGHT UNION)

```sql
SELECT jikwonname, busername  
FROM jikwon  
LEFT JOIN buser ON jikwon.busernum = buser.buserno  
UNION  
SELECT jikwonname, busername  
FROM jikwon  
RIGHT JOIN buser ON jikwon.busernum = buser.buserno;
```

**출력값**

| jikwonname | busername |
| ---------- | --------- |
| 김철수        | 총무부       |
| 이순신        | 영업부       |
| 박영수        | 전산부       |
| 홍길동        | NULL      |
| 김영희        | NULL      |
| NULL       | 기획실       |


**그림 설명:**

```
모든 직원 + 모든 부서 포함
부서 없는 직원 → NULL
직원 없는 부서 → NULL

jikwon         buser
---------      ---------
김철수,10      10,총무부
이순신,20      20,영업부
박영수,30      30,전산부
홍길동,NULL    -
김영희,NULL    -
NULL          50,기획실
```

---

## 5️⃣ SUBQUERY (서브쿼리)

### 5-1. '이미라'와 같은 직급 직원

```sql
SELECT *  
FROM jikwon  
WHERE jikwonjik = (SELECT jikwonjik FROM jikwon WHERE jikwonname='이미라');
```

**출력 예시:**

|jikwonno|jikwonname|jikwonjik|jikwonpay|
|---|---|---|---|
|3|이미라|대리|30000|
|7|김하나|대리|28000|

---

### 5-2. 과장 중 최대/최소 급여

```sql
SELECT *  
FROM jikwon  
WHERE jikwonjik = '과장' AND jikwonpay IN (  
    SELECT MAX(jikwonpay) FROM jikwon WHERE jikwonjik='과장'  
    UNION  
    SELECT MIN(jikwonpay) FROM jikwon WHERE jikwonjik='과장'  
);
```

**출력 예시:**

|jikwonno|jikwonname|jikwonjik|jikwonpay|
|---|---|---|---|
|4|박민준|과장|40000|
|5|최수진|과장|30000|

> ✅ 서브쿼리로 **MAX/MIN 계산 후 메인쿼리에서 조건 적용**

---

## 6️⃣ UNION (두 테이블 결과 합치기)

```sql
SELECT bun AS 번호, pummok AS 품명 FROM pum1  
UNION  
SELECT mum, sangpum FROM pum2;
```

**출력 예시:**

|번호|품명|
|---|---|
|1|귤|
|2|한라봉|
|3|바나나|
|10|토마토|
|20|딸기|
|30|참외|
|40|수박|

> ✅ 구조가 같은 두 쿼리 결과 합치기, 중복 제거 (UNION ALL은 중복 포함)

---

## 7️⃣ 고객-직원 JOIN 예시

### 7-1. 고객 확보 직원, 입사일 순

```sql
SELECT jikwonname AS 직원명,  
       jikwonjik AS 직급,  
       busername AS 부서명,  
       jikwonibsail AS 입사일  
FROM jikwon  
LEFT OUTER JOIN buser ON busernum = buserno  
WHERE jikwonno IN (SELECT DISTINCT gogekdamsano FROM gogek)  
ORDER BY jikwonibsail;
```

**출력 예시:**

|직원명|직급|부서명|입사일|
|---|---|---|---|
|김철수|사원|총무부|2008-03-12|
|이미라|대리|전산부|2012-07-01|
|박민준|과장|영업부|2014-05-20|

---

### 7-2. 이순신 부서 직원 + 고객 정보

```sql
SELECT jikwonname AS 직원명,  
       busername AS 부서명,  
       busertel AS 부서전화,  
       jikwonjik AS 직급,  
       gogekname AS 고객명,  
       gogektel AS 고객전화,  
       CASE  
         WHEN YEAR(CURDATE()) - (CASE WHEN SUBSTR(gogekjumin,8,1) IN ('1','2')   
                                       THEN 1900+LEFT(gogekjumin,2)  
                                       ELSE 2000+LEFT(gogekjumin,2) END) <=30   
         THEN '청년'  
         WHEN YEAR(CURDATE()) - (CASE WHEN SUBSTR(gogekjumin,8,1) IN ('1','2')   
                                       THEN 1900+LEFT(gogekjumin,2)  
                                       ELSE 2000+LEFT(gogekjumin,2) END) <=50   
         THEN '중년'  
         ELSE '노년'  
       END AS 고객구분  
FROM jikwon  
INNER JOIN buser ON busernum = buserno  
INNER JOIN gogek ON jikwonno = gogekdamsano  
WHERE busernum = (SELECT busernum FROM jikwon WHERE jikwonname='이순신')  
ORDER BY SUBSTR(gogekjumin,1,6);
```

**출력 예시:**

|직원명|부서명|부서전화|직급|고객명|고객전화|고객구분|
|---|---|---|---|---|---|---|
|한송이|총무부|123-1111|사원|백송이|333-3333|청년|
|한송이|총무부|123-1111|사원|장영희|444-4444|중년|

> ✅ 서브쿼리 + JOIN + CASE 활용  
> ✅ 고객 나이 계산 후 구분까지 처리
