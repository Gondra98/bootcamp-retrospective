# Day 12_SELECT, 함수, 집계함수 종합 정리
# 📅 2026-02-12

---

# 1. 테이블 생성

## 🔹 1-1. sangdata (상품 테이블)

```sql
CREATE TABLE sangdata(
  code INT PRIMARY KEY,
  sang VARCHAR(20),
  su INT,
  dan INT
);
```

### ✔ 데이터 입력

```sql
INSERT INTO sangdata VALUES(1,'장갑',3,10000);
INSERT INTO sangdata VALUES(2,'벙어리장갑',2,12000);
INSERT INTO sangdata VALUES(3,'가죽장갑',10,50000);
INSERT INTO sangdata VALUES(4,'가죽점퍼',5,650000);
```

### ✔ 전체조회

```sql
SELECT * FROM sangdata;
```

### ✅ 출력결과

|code|sang|su|dan|
|---|---|---|---|
|1|장갑|3|10000|
|2|벙어리장갑|2|12000|
|3|가죽장갑|10|50000|
|4|가죽점퍼|5|650000|

---

# 2️. SELECT 기본

## 🔹 전체 컬럼 조회

```sql
SELECT * FROM jikwon;
```

## 🔹 특정 컬럼 조회

```sql
SELECT jikwonno, jikwonname FROM jikwon;
```

## 🔹 별칭 사용

```sql
SELECT jikwonno AS 직원번호, jikwonname AS 직원명 
FROM jikwon;
```

---

# 3️. 연산 & 계산

```sql
SELECT 10, '안녕', 12/3 AS 결과 FROM DUAL;
```

### ✅ 출력

|10|안녕|결과|
|---|---|---|
|10|안녕|4|

---

## 🔹 세금 계산

```sql
SELECT jikwonname, jikwonpay, jikwonpay * 0.05 AS tax 
FROM jikwon;
```

### 예시 출력

|이름|연봉|tax|
|---|---|---|
|홍길동|9900|495|
|한송이|8800|440|

---

# 4️. 정렬 (ORDER BY)

## 🔹 오름차순

```sql
SELECT * FROM jikwon ORDER BY jikwonpay;
```

→ 급여 낮은 사람부터 출력

## 🔹 내림차순

```sql
SELECT * FROM jikwon ORDER BY jikwonpay DESC;
```

→ 급여 높은 사람부터 출력

---

# 5️. DISTINCT (중복 제거)

```sql
SELECT DISTINCT jikwonjik FROM jikwon;
```

### ✅ 출력

|jikwonjik|
|---|
|이사|
|부장|
|과장|
|대리|
|사원|

---

# 6️. WHERE 조건

---

## 🔹 직급이 대리

```sql
SELECT * FROM jikwon WHERE jikwonjik = '대리';
```

### ✅ 결과 예시

|번호|이름|직급|
|---|---|---|
|4|이미라|대리|
|9|채송화|대리|
|13|박명화|대리|
|19|이유라|대리|
|24|유비|대리|
|28|박명화|대리|

---

## 🔹 범위 조건 (BETWEEN)

```sql
SELECT * FROM jikwon 
WHERE jikwonno BETWEEN 5 AND 10;
```

→ 5번 ~ 10번 직원

---

## 🔹 IN 조건

```sql
SELECT * FROM jikwon 
WHERE jikwonjik IN('대리','과장','부장');
```

→ 해당 직급만 조회

---

# 7️. LIKE (문자 패턴 검색)

## 🔹 이씨 성

```sql
SELECT * FROM jikwon 
WHERE jikwonname LIKE '이%';
```

### ✅ 결과 예시

이순신  
이순라  
이유가  
이순기  
이유라

---

## 🔹 이름이 '라'로 끝나는 사람

```sql
SELECT * FROM jikwon 
WHERE jikwonname LIKE '%라';
```

예: 이미라, 이유라, 김유라

---

# 8️. NULL 처리

```sql
UPDATE jikwon SET jikwonjik = NULL WHERE jikwonno = 5;
```

### ❌ 틀린 방식

```sql
SELECT * FROM jikwon WHERE jikwonjik = NULL;
```

### ✅ 올바른 방식

```sql
SELECT * FROM jikwon WHERE jikwonjik IS NULL;
```

---

# 9️. LIMIT

## 🔹 상위 3명

```sql
SELECT * FROM jikwon LIMIT 3;
```

## 🔹 급여 높은 3명

```sql
SELECT * FROM jikwon 
ORDER BY jikwonpay DESC 
LIMIT 3;
```

### ✅ 결과

|이름|급여|
|---|---|
|홍길동|9900|
|한송이|8800|
|김부만|8600|

---

# 10. 종합 문제 쿼리

```sql
SELECT 
jikwonno AS 직원번호, 
jikwonname AS 직원명, 
jikwonjik AS 직급, 
jikwonpay AS 연봉,
jikwonpay / 12 AS 보너스, 
jikwonibsail AS 입사일 
FROM jikwon
WHERE jikwonjik IN('과장', '부장', '사원')
AND jikwonpay >= 4000 
AND jikwonibsail BETWEEN '2015-1-1' AND '2019-12-31' 
ORDER BY jikwonjik, jikwonpay DESC 
LIMIT 3;
```

### ✅ 실제 결과

| 직원번호 | 직원명 | 직급  | 연봉          | 보너스 |
| ---- | --- | --- | ----------- | --- |
| 11   | 김부해 | 사원  | 3900 ❌ (제외) |     |
| 15   | 채미리 | 사원  | 4000        |     |
| 25   | 박혁기 | 사원  | 3800 ❌ (제외) |     |

👉 조건 충족자 예시:

- 채미리
    
- 김유라
    
- 장비 등
    

(정확 결과는 실행환경 기준)

---
# 1️1. 문자 함수

---

## 🔹 LOWER / UPPER

```sql
SELECT LOWER('Hello'), UPPER('Hello') FROM DUAL;
```

### ✅ 출력

|lower|upper|
|---|---|
|hello|HELLO|

---

## 🔹 SUBSTR (문자열 자르기)

```sql
SELECT 
SUBSTR('hello world', 3), 
SUBSTR('hello world', 3, 3),
SUBSTR('hello world', -3, 3) 
FROM DUAL;
```

### ✅ 출력

|결과1|결과2|결과3|
|---|---|---|
|llo world|llo|rld|

✔ 3 → 3번째부터 끝까지  
✔ 3,3 → 3번째부터 3글자  
✔ -3,3 → 뒤에서 3번째부터 3글자

---

## 🔹 LENGTH / INSTR

```sql
SELECT LENGTH('hello world'), 
INSTR('hello world', 'e') 
FROM DUAL;
```

### ✅ 출력

|length|instr|
|---|---|
|11|2|

✔ INSTR → 문자열 위치 반환

---

## 🔹 REPLACE

```sql
SELECT REPLACE('010.111.1234', '.', '-') FROM DUAL;
```

### ✅ 출력

`010-111-1234`

---

## 🔹 실전 예제 (오타 수정 완료)

👉 이름에 '이'가 포함된 직원 → '이'부터 두 글자 출력

```sql
SELECT jikwonname,
SUBSTR(jikwonname, INSTR(jikwonname, '이'), 2) AS 결과
FROM jikwon
WHERE jikwonname LIKE '%이%';
```

### ✅ 출력 예시

|이름|결과|
|---|---|
|이순신|이순|
|이순라|이순|
|이유가|이유|
|이유라|이유|

---

# 12. 숫자 함수

---

## 🔹 ROUND (반올림)

```sql
SELECT 
ROUND(45.678, 2),
ROUND(45.678),
ROUND(45.678, 0),
ROUND(45.678, -1)
FROM DUAL;
```

### ✅ 출력

| 45.68 | 46 | 46 | 50 |

✔ -1 → 십의 자리 반올림

---

## 🔹 급여의 25% 계산

```sql
SELECT 
jikwonname, 
jikwonpay, 
jikwonpay * 0.25 AS tax,
ROUND(jikwonpay * 0.25, 0) AS 반올림
FROM jikwon;
```

### 예시

|이름|연봉|tax|반올림|
|---|---|---|---|
|홍길동|9900|2475|2475|
|한송이|8800|2200|2200|

---

## 🔹 TRUNCATE (버림)

```sql
SELECT 
TRUNCATE(45.678,0),
TRUNCATE(45.678,1),
TRUNCATE(45.678,-1)
FROM DUAL;
```

### ✅ 출력

| 45 | 45.6 | 40 |

---

## 🔹 MOD (나머지)

```sql
SELECT MOD(15, 2), 15 / 2 FROM DUAL;
```

### ✅ 출력

| 1 | 7.5 |

---

## 🔹 GREATEST / LEAST

```sql
SELECT GREATEST(23,25,5,1,12),
LEAST(23,25,5,1,12)
FROM DUAL;
```

### ✅ 출력

| 25 | 1 |

---

# 13. 날짜 함수

---

## 🔹 현재 시간

```sql
SELECT NOW(), CURDATE() FROM DUAL;
```

### ✅ 출력 예시

| NOW()               | CURDATE()  |
| ------------------- | ---------- |
| 2026-02-12 13:20:00 | 2026-02-12 |

---

## 🔹 날짜 더하기

```sql
SELECT 
ADDDATE('2020-08-01', 3),
ADDDATE('2020-08-01', -3),
SUBDATE('2020-08-01',3)
FROM DUAL;
```

### ✅ 출력

| 2020-08-04 | 2020-07-29 | 2020-07-29 |

---

## 🔹 날짜 차이

```sql
SELECT DATEDIFF(NOW(), '2025-05-05');
```

→ 오늘 기준 며칠 차이인지 반환

---

# 14. 형변환 함수

---

## 🔹 날짜 포맷 변경

```sql
SELECT 
NOW(),
DATE_FORMAT(NOW(), '%Y%m%d'),
DATE_FORMAT(NOW(), '%Y년%m월%d일');
```

### ✅ 출력 예시

| 2026-02-12 13:20 | 20260212 | 2026년02월12일 |

---

## 🔹 입사 요일 확인

```sql
SELECT 
jikwonname, 
jikwonibsail, 
DATE_FORMAT(jikwonibsail, '%W') 
FROM jikwon
WHERE busernum = 10;
```

---

## 🔹 문자열 → 날짜 변환

```sql
SELECT STR_TO_DATE('2026-02-12', '%Y-%m-%d');
```

---

# 15. 순위 함수 (윈도우 함수)

---

## 🔹 오름차순 순위

```sql
SELECT 
jikwonno, 
jikwonname, 
jikwonpay,
RANK() OVER (ORDER BY jikwonpay) AS 순위,
DENSE_RANK() OVER (ORDER BY jikwonpay) AS 밀집순위
FROM jikwon;
```

✔ RANK → 공동순위 다음 숫자 건너뜀  
✔ DENSE_RANK → 건너뛰지 않음

---

## 🔹 내림차순 (급여 높은 순)

```sql
SELECT 
jikwonno, 
jikwonname, 
jikwonpay,
RANK() OVER (ORDER BY jikwonpay DESC) AS 순위
FROM jikwon;
```

---

# 16. NULL 관련 함수

---

## 🔹 NVL (MySQL은 IFNULL 사용)

```sql
SELECT 
jikwonname, 
jikwonjik, 
IFNULL(jikwonjik, '임시직')
FROM jikwon;
```

---

## 🔹 NULLIF

```sql
SELECT 
jikwonname, 
jikwonjik, 
NULLIF(jikwonjik, '대리')
FROM jikwon;
```

→ 직급이 대리면 NULL

---

# 17. CASE 표현식

---

## 🔹 형식 1 (값 비교)

```sql
SELECT 
CASE 10/5
WHEN 5 THEN '안녕'
WHEN 2 THEN '반가워'
ELSE '잘가'
END AS 결과
FROM DUAL;
```

### ✅ 출력

`반가워`

---

## 🔹 직급별 기부금 계산

```sql
SELECT 
jikwonname,
jikwonpay,
jikwonjik,
CASE jikwonjik
WHEN '이사' THEN jikwonpay * 0.05
WHEN '부장' THEN jikwonpay * 0.04
WHEN '과장' THEN jikwonpay * 0.03
ELSE jikwonpay * 0.02
END AS donation
FROM jikwon;
```

---

## 🔹 형식 2 (조건식)

```sql
SELECT 
jikwonname,
CASE 
WHEN jikwongen='남' THEN '남성'
WHEN jikwongen='여' THEN '여성'
END AS gender
FROM jikwon;
```

---

## 🔹 급여 등급 분류

```sql
SELECT 
jikwonname,
jikwonpay,
CASE
WHEN jikwonpay >= 7000 THEN '우수연봉'
WHEN jikwonpay >= 5000 THEN '보통연봉'
ELSE '저조'
END AS result
FROM jikwon
WHERE jikwonjik IN ('대리','과장');
```

---

# 18. IF 함수

```sql
SELECT 
jikwonname,
jikwonpay,
IF(TRUNCATE(jikwonpay/1000,0) >= 5, 'good', 'normal') AS result
FROM jikwon;
```

✔ 5000 이상 → good  
✔ 미만 → normal


---
# 19. 응용 문제 정리
## ✅ 문제 1

### 🔹 요구사항

- 근무년수 계산
    
- 10년 이상 → "감사합니다" + 5% 특별수당
    
- 미만 → "열심히" + 3% 특별수당
    

---

### 🔹 작성 코드

```sql
SELECT 
jikwonname AS 직원명,
YEAR(NOW()) - YEAR(jikwonibsail) AS 근무년수,
CASE
WHEN YEAR(NOW()) - YEAR(jikwonibsail) >= 10 THEN '감사합니다'
ELSE '열심히'
END AS 표현,
CASE
WHEN YEAR(NOW()) - YEAR(jikwonibsail) >= 10 
     THEN ROUND(jikwonpay * 0.05, 0)
ELSE ROUND(jikwonpay * 0.03, 0)
END AS 특별수당
FROM jikwon;
```

---

### ✅ 출력 예시

|직원명|근무년수|표현|특별수당|
|---|---|---|---|
|홍길동|12|감사합니다|495|
|한송이|7|열심히|264|
|김유라|3|열심히|120|

---

### 💡 실무 개선 (정확한 근속년수 계산)

현재 방식은 단순 연도 차이임.  
정확히 하려면 `TIMESTAMPDIFF()` 사용 👇

`TIMESTAMPDIFF(YEAR, jikwonibsail, NOW())`

---

## ✅ 문제 2

### 🔹 요구사항

- 입사일 포맷 변경
    
- 근속년수 기준 등급
    
- 부서번호 → 부서명 변환
    

---

### 🔹 작성 코드

```sql
SELECT 
jikwonname AS 직원명,
jikwonjik AS 직급,
DATE_FORMAT(jikwonibsail, '%Y.%m.%d') AS 입사년월일,

CASE
WHEN YEAR(NOW()) - YEAR(jikwonibsail) >= 8 THEN '왕고참'
WHEN YEAR(NOW()) - YEAR(jikwonibsail) >= 5 THEN '고참'
WHEN YEAR(NOW()) - YEAR(jikwonibsail) >= 3 THEN '보통'
ELSE '일반'
END AS 구분,

CASE
WHEN busernum = 10 THEN '총무부'
WHEN busernum = 20 THEN '영업부'
WHEN busernum = 30 THEN '전산부'
WHEN busernum = 40 THEN '관리부'
END AS 부서

FROM jikwon;
```

---
### ✅ 출력 예시

|직원명|직급|입사년월일|구분|부서|
|---|---|---|---|---|
|홍길동|부장|2013.05.01|왕고참|총무부|
|한송이|과장|2018.03.15|고참|영업부|
|이유라|대리|2022.02.01|일반|전산부|

---

### 💡 실무 팁

CASE 대신 부서 테이블 만들어서  
JOIN 하는 것이 정석 구조임.

---

## ✅ 문제 3

### 🔹 요구사항

- 부서별 인상률 적용
    
- 장기근속 여부 표시
    

---
### 🔹 작성 코드

```sql
SELECT 
jikwonno AS 사번,
jikwonname AS 직원명,
busernum AS 부서,
jikwonpay AS 연봉,

CASE
WHEN busernum = 10 THEN ROUND(jikwonpay * 1.01, 0)
WHEN busernum = 30 THEN ROUND(jikwonpay * 1.02, 0)
ELSE jikwonpay
END AS 인상연봉,

CASE
WHEN YEAR(NOW()) - YEAR(jikwonibsail) >= 8 THEN 'O'
ELSE 'X'
END AS 장기근속

FROM jikwon;
```

---
### ✅ 출력 예시

|사번|직원명|부서|연봉|인상연봉|장기근속|
|---|---|---|---|---|---|
|1|홍길동|10|9900|9999|O|
|2|한송이|20|8800|8800|X|
|3|김유라|30|6000|6120|X|

---
# 20. 집계함수 정리 (출력 포함)

---
## 20-1. 합계 / 평균

```sql
SELECT 
SUM(jikwonpay) AS 합,
AVG(jikwonpay) AS 평균
FROM jikwon;
```

### ✅ 출력 예시

|합|평균|
|---|---|
|156300|5389.66|

✔ NULL 1명 제외 후 평균 계산됨  
✔ 실제 계산: SUM / 29명

---

## 20-2. 최대값 / 최소값

```sql
SELECT 
MAX(jikwonpay) AS 최대값,
MIN(jikwonpay) AS 최소값
FROM jikwon;
```

### ✅ 출력 예시

|최대값|최소값|
|---|---|
|9900|2500|

---

## 20-3. NULL 평균 비교

```sql
SELECT 
AVG(jikwonpay) AS NULL제외,
AVG(IFNULL(jikwonpay, 0)) AS NULL포함
FROM jikwon;
```

### ✅ 출력 예시

|NULL제외|NULL포함|
|---|---|
|5389.66|5210.00|

✔ NULL을 0으로 넣으면 평균이 낮아짐

---

## 20-4. 직접 나누기 비교

```sql
SELECT 
SUM(jikwonpay) / 29 AS NULL제외계산,
SUM(jikwonpay) / 30 AS 전체계산
FROM jikwon;
```

### ✅ 출력 예시

|NULL제외계산|전체계산|
|---|---|
|5389.66|5210.00|

---

## 20-5. COUNT 차이

```sql
SELECT 
COUNT(jikwonno) AS 사번개수,
COUNT(jikwonpay) AS 급여개수,
COUNT(jikwonjik) AS 직급개수
FROM jikwon;
```

### ✅ 출력 예시

|사번개수|급여개수|직급개수|
|---|---|---|
|30|29|30|

✔ 급여 1명 NULL → 29

---

### 🔹 전체 인원

```sql
SELECT COUNT(*) AS 인원수 FROM jikwon;
```

### ✅ 출력

|인원수|
|---|
|30|

---

## 20-6. 분산 / 표준편차

```sql
SELECT 
STDDEV(jikwonpay) AS 표준편차,
VAR_SAMP(jikwonpay) AS 분산
FROM jikwon;
```

### ✅ 출력 예시

|표준편차|분산|
|---|---|
|1850.32|3423684.15|

✔ 급여 차이가 클수록 값이 커짐

---

## 20-7. 조건별 집계

### 🔹 부서 10번

```sql
SELECT 
COUNT(*) AS 인원,
VAR_SAMP(jikwonpay) AS 분산
FROM jikwon
WHERE busernum = 10;
```

### ✅ 출력 예시

|인원|분산|
|---|---|
|8|2100000|

---

### 🔹 부서 20번

```sql
SELECT 
COUNT(*) AS 인원,
VAR_SAMP(jikwonpay) AS 분산
FROM jikwon
WHERE busernum = 20;
```

### ✅ 출력 예시

|인원|분산|
|---|---|
|7|1500000|

---

## 20-8. GROUP BY (부서별 평균)

```sql
SELECT 
busernum AS 부서,
COUNT(*) AS 인원수,
AVG(jikwonpay) AS 평균급여
FROM jikwon
GROUP BY busernum;
```

### ✅ 출력 예시

|부서|인원수|평균급여|
|---|---|---|
|10|8|6200|
|20|7|5400|
|30|9|5800|
|40|6|5000|
