# Day 18_자바스크립트 기초
# 📅 2026-02-27

---
## 1️. 자바스크립트란?

- **클라이언트 사이드 스크립트**
    
- 브라우저에서 실행되는 프로그래밍 언어
    
- HTML과 결합하여 동적인 웹 페이지 제작 가능
    

---

## 2️. HTML에 JS 삽입 방법

1. `<script>` 태그 안에 직접 작성
    
2. 외부 JS 파일을 `<script src="파일명.js"></script>`로 불러오기
    

---

## 3️. 화면 출력과 콘솔 출력

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
</head>  
<body>  
    ** 자바 스크립트 : 클라이언트 사이드 스크립트 **<br>  
    <script>  
        window.document.write("환영합니다. JS 세상");   // 브라우저 화면으로 출력  
        console.log("개발자를 위해 표준 출력 장치로 출력")  // 브라우저 콘솔에 출력  
        document.write("<h2>브라우저에 문자열 출력</h2>");  
    </script>  
    다시 html로 나옴  
    <br>  
    <script>  
        document.write("여기는 자바스크립트");  
          
        let a = 10;  
        let b = a + 10;  
        document.write("<br>b는 ", b)  
    </script>  
</body>  
</html>
```


### ✔ 실행 순서

1. 첫 번째 `<script>` 실행
    
    - 화면 출력: `환영합니다. JS 세상`
        
    - 콘솔 출력: `개발자를 위해 표준 출력 장치로 출력`
        
    - 화면 출력: `<h2>브라우저에 문자열 출력</h2>` → 큰 글씨
        
2. 일반 HTML 텍스트: `다시 html로 나옴`
    
3. 두 번째 `<script>` 실행
    
    - 화면 출력: `여기는 자바스크립트`
        
    - 변수 연산: `let a = 10; let b = a + 10;`
        
    - 화면 출력: `b는 20`

---
## 4️. 핵심 포인트

- `<script>` 안 코드 → HTML 읽는 중 즉시 실행
    
- `document.write()` → 브라우저 화면 출력
    
- `console.log()` → 개발자 콘솔 출력 (화면에는 안 보임)
    
- JS 변수와 연산 가능 (`let a = 10`, `b = a + 10`)
    
- HTML 텍스트와 JS 출력이 섞여도 순서대로 표시됨

---
## 5️. 변수와 연산자(js2.html)

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
</head>  
<body>  
    -- 변수와 연산자 --<br/>  
    <script type = "text/javascript">  
        // var aa;  // 전통적인 변수 선언방식  
        let aa;     // 모던(modern)한 변수 선언  
        aa = 100;   // 변수에 값 할당  
        let bb = 20.5;   // 선언과 동시에 값 할당  
        const cc = 300;  // 상수 선언, 값 변경 불가  
        bb = bb + 10     // 기존 값에 10 더하기, bb = 30  
        // cc = cc + 10     // err : 상수는 수정 불가  
        let dd = false;   // boolean 값  
        let msg = "변수 선언 이해" // 문자열 값  
        document.write(aa, " " , bb, " " , cc, " ", dd, " ", msg);   
        // 화면에 변수 값 순서대로 출력  
        document.write("<hr>"); // 가로줄  
        document.write("type 확인 :", typeof(1), ' ', typeof(1.5), ' ',  
                        typeof('안녕'), ' ', typeof(true), ' ',  
                        typeof(null), ' ', typeof(undefined));  
        // typeof() → 변수 타입 확인 (number, string, boolean, object, undefined)  
  
        {   // 블록 영역 시작  
            let v1 = 1;     // 지역변수 v1 선언  
            document.write("<br>", v1); // v1 출력  
            {  
                let v2 = 2; // 블록 내부 지역변수 v2  
                document.write("<br>", v2);  
                {  
                    let v3 = 3; // 블록 내부 지역변수 v3  
                    document.write("<br>", v3);  
                    document.write("<br>", v1, ' ', v2, ' ', v3);   
                    // 모두 접근 가능: 내부 블록에서는 상위 블록 변수 참조 가능  
                }  
            }  
        }  
        // document.write("<br>", v1, ' ', v2, ' ', v3);      
        // err: Uncaught ReferenceError: v1, v2, v3 is not defined  
        // 블록 밖에서는 let으로 선언한 변수 접근 불가  
    </script>  
  
    <br>-- 연산자 --<br/>  
    <script>  
        let x = 5, y = 2;  
        document.write("<br>", x, " ", y);  
  
        // 산술 연산  
        document.write("<br>산술 연산 : ", x + y, " ", x / y, " ", x % y);  
        // + : 덧셈, / : 나누기, % : 나머지  
  
        // 관계(비교) 연산  
        document.write("<br>관계(비교) 연산 : ", x > y, " ", x <= y, " ", x == y, " ", x != y);  
        // >, <=, ==, !=  
  
        // 논리 연산  
        document.write("<br>논리 연산(&&-and, ||-or) : ", x > y && x <= y, " ", x > y || x <= y);  
        // && : AND, || : OR  
  
        // 누적 연산  
        document.write("<br>누적 연산 : ", x = x + 1, " ", ++x, " ", x += 1);  
        // x = x + 1 : 증가, ++x : 1 증가 후 값 반환, x += 1 : 누적 증가  
  
        // 삼항 연산  
        document.write("<br>삼항 연산 : ", (x > y)?1000:1000+2000);  
        // (조건) ? 참일때 값 : 거짓일때 값  
  
        // 문자열 더하기  
        document.write("<br>", "a" + "b" + "c")  // 문자열 연결  
  
        // 할당 연산 체이닝  
        let m, b, c;  
        m = b = c = 6 + 5  
        document.write("<br>", m, " ", b, " ", c)  
        // c = 11 → b = 11 → m = 11  
    </script>  
  
</body>  
</html>
```

---

## 6️. 상세 설명

### 변수 선언

- `let`
    
    - 블록 단위 스코프
        
    - 같은 블록 안에서는 중복 선언 불가
        
- `const`
    
    - 상수, 한 번 값 할당 후 변경 불가
        
- `var`
    
    - 전통적인 변수 선언, 함수 단위 스코프, 블록 밖에서도 접근 가능

### 타입 확인

- `typeof()` → 값의 타입 확인
    
    - 예: `number`, `string`, `boolean`, `object`, `undefined`
        

### 연산자

- **산술** : `+`, `-`, `*`, `/`, `%`
    
- **관계** : `>`, `<`, `>=`, `<=`, `==, !=
    
- **논리** : `&&`(AND), `||`(OR)
    
- 누적 : `x += 1`, `++x`, `x = x + 1`
    
- 삼항 : `(조건) ? 참값 : 거짓값`
    
- **문자열 연결** : `+`
    
- **할당 체이닝** : `a = b = c = 값`
    

### 블록 스코프

- `{ }` 내부에 `let` 변수 선언 → 블록 밖에서는 접근 불가
    
- 블록 내부에서는 상위 블록 변수 접근 가능

---
## 7️. 조건문: if / else if / else

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>조건문 예제</title>  
</head>  
<body>  
    조건 판단문 : if<br>  
    <script>  
        // let ave = prompt("평균 점수 입력:", "80");  
        // prompt() → 사용자 입력받기, 문자열 반환  
  
        let ave = 85; // 점수 변수  
  
        if(ave >= 90)   
            document.write("점수 : ", ave + "이므로 <b style='color:blue'>우수</b>");  
        else if(ave >= 70) {  
            // 수행 명령문이 2개 이상이면 중괄호 필요  
            document.write("점수 : " + ave + "이므로 <b style='color:red'>보통</b>");  
            document.write("&nbsp;&nbsp;와우");  
        }  
        else  
            document.write("점수 : " + ave + "이므로 <b style='color:red'>저조</b>");  
    </script>  
</body>  
</html>
```

### ✔ 상세 설명

- `if(조건)` → 조건이 참이면 실행
    
- `else if(조건)` → 첫 번째 조건이 거짓이고 이 조건이 참이면 실행
    
- `else` → 위 조건 모두 거짓이면 실행
    
- 중괄호 `{ }` → 여러 명령문을 묶을 때 필요
    
- `document.write()` → HTML 화면에 출력
    

---

## 8️. 조건문: switch

```html
<br>조건 판단문 : switch<br>  
<script>  
    let result = "A";    // 학점 변수  
  
    switch(result){  
        case "A": {  
            document.write("매우 우수");  
            document.write("good");  
            break;  // switch 종료  
        }  
        case "B":   
            document.write("우수");   
            break;  
        case "C":   
            document.write("중간");   
            break;  
        // case "D": document.write("저조"); break;  
        case "D":  
        case "F":   
            document.write("불량");   
            break;  
        default:  
            document.write("기타");  // 어떤 case도 해당하지 않을 때  
    }  
</script>
```

### ✔ 상세 설명

- `switch(변수)` → 변수 값에 따라 case 선택
    
- `case 값:` → 값이 맞으면 해당 블록 실행
    
- `break;` → switch 탈출, 없으면 아래 case까지 실행
    
- `default:` → 모든 case에 해당하지 않을 때 실행
    
- 여러 case를 묶을 수 있음 (`case "D": case "F":`)

---
## 9️. 반복문: for

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>반복문 예제</title>  
</head>  
<body>  
    반복문 for(초기값;조건;증감값) <br>  
    <script>  
        let a;  
  
        // 1 ~ 5까지 증가  
        for(let a = 1; a <= 5; a++){    // ++a 가능  
            document.write("a 값은 " + a + "<br>");  
        }  
        // for 블록 밖에서는 let a를 다시 선언해야 함 (블록 스코프)  
        document.write("for 블럭 수행 후 a 값은 " + a + "<br>");  
  
        document.write("<br>");  
  
        // 5 ~ 1까지 감소  
        for(let a = 5; a >= 1; a--){  
            document.write("a 값은 " + a + "<br>");  
        }  
  
        document.write("<br>");  
  
        // 3 ~ 10까지 2씩 증가  
        for(let a = 3; a < 11; a+=2){  
            document.write("a 값은 " + a + "<br>");  
        }  
  
        document.write("<br>다중 for<br>");  
  
        // 다중 for문 예제  
        for(let i = 1; i <= 3; i++){  
            document.write("i 값은 => " + i + "&nbsp;&nbsp");  
            for(let j = 1; j <= 4; j++){  
                document.write("j:" + j + "&nbsp;&nbsp");  
            }  
            document.write("<br>");  
        }  
  
        document.write("<br>1 ~ 10 까지의 정수 합은?<br>");  
        let hap = 0;    // 누적 변수 초기화  
        for(let su=1; su <= 10; su++){  
            hap = hap + su;     // hap += su 가능  
        }  
        document.write(hap);  
        document.write("<br>");  
  
        document.write("<br>1 ~ 100 까지의 정수 중 2의 배수이나 5의 배수가 아닌 수의 합 출력<br>");  
        let sum = 0;     // 합 변수 초기화  
        for(let i=1; i<=100; i++){  
            // ===는 엄격한 비교, 권장  
            if (i%2 === 0 && i%5 !== 0) {  
                sum = sum + i;     // 조건을 만족하는 값 누적  
            }   
        }  
        document.write(sum);  
  
        document.write("<hr>2 ~ 9 구구단 출력 - table에 출력<br>");  
        document.write("<table border='1'>");  
        for(let a=2; a < 10; a++){  
            document.write("<tr>");  
            for(let b=1; b<10; b++){  
                document.write(`<td>${a}*${b}=${a*b}&nbsp;&nbsp;</td>`);    // 템플릿 문자열  
            }  
            document.write("</tr>");  
        }  
        document.write("</table>");  
  
    </script>  
</body>  
</html>

```
---

## 9-1. 상세 설명

### 기본 for문

```javascript
for(let a = 1; a <= 5; a++) { }
```

- `초기값; 조건; 증감값`
    
    - 초기값: 반복 변수 시작값
        
    - 조건: 참일 동안 반복
        
    - 증감값: 반복 변수 증가/감소
        
- `++a` → 전위 증가, `a++` → 후위 증가
    
- `a += 2` → 2씩 증가
    

---

### 다중 for문

```javascript
for(let i=1;i<=3;i++){  
    for(let j=1;j<=4;j++){  
        // 내부 반복  
    }  
}
```

- 바깥쪽 반복 변수 i, 안쪽 반복 변수 j
    
- 내부 for문이 끝날 때마다 바깥쪽 반복문이 한 단계 진행
    

---

### 누적 합

```javascript
let hap = 0;  
for(let su=1; su<=10; su++){  
    hap += su;  
}
```


- `hap = hap + su` → 누적 연산
    
- 특정 조건 합산도 가능

```javascript
    if(i%2===0 && i%5!==0) sum += i;
```

---

### 구구단 출력

- `<table>`과 `<tr>`, `<td>` 사용
    
- 템플릿 문자열 사용: `` `${a}*${b}=${a*b}` ``
    
- 2중 반복문으로 2~9단 출력

---
### 템플릿 문자열

```javascript
document.write(`<td>${a}*${b}=${a*b}</td>`); // `${}` 안에 변수나 계산식 삽입
```

- **백틱(`` ` ``)** → 문자열 + 변수/계산식 결합 가능
    
- `${a}*${b}=${a*b}` → a, b 값과 계산 결과를 바로 문자열에 넣음
    
- `<td>...</td>` → 테이블 칸 출력

즉 **템플릿 문자열 한 줄로 변수 출력 + 계산 + HTML 태그 삽입** 가능

---

## 10. 반복문: while (js5while.html)

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>while 예제</title>  
</head>  
<body>  
    반복문 while <br>  
    <script>  
        let k = 0;                 // 반복 변수 초기화  
        while(k < 10){              // 조건이 참인 동안 반복  
            k++;                    // 반복 변수 증가  
              
            if(k === 3 || k === 5) continue;   // k가 3 또는 5이면 출력 건너뛰기  
              
            document.write(k + " ");           // 화면에 k 값 출력  
  
            if(k === 9) break;                 // k가 9이면 반복 종료  
        }  
    </script>  
</body>  
</html>
```

---

### 상세 설명

1. **반복 변수 초기화**

```javascript
let k = 0;
```

- 반복 시작 전에 반드시 초기값 설정
    

2. **while 조건**

```javascript
while(k < 10)
```

- 조건이 **참(true)**일 때 반복
    
- 거짓이면 반복 종료
    

3. **continue 문**

```javascript
if(k === 3 || k === 5) continue;
```

- 현재 반복 **건너뛰기**
    
- 3과 5는 출력 없이 바로 다음 반복으로 이동
    

4. **document.write() 출력**
    

```javascript
document.write(k + " ");
```

- 조건에 맞으면 화면에 값 출력
    
- 결과: 1 2 4 6 7 8 9
    

5. **break 문**

```javascript
if(k === 9) break;
```

- 반복문 **즉시 종료**
    
- 9가 출력된 후 while문 종료

---

### 실행 결과

```
1 2 4 6 7 8 9
```

- 3, 5는 continue로 건너뜀
    
- 9에서 break로 반복 종료

---
## 11. 배열(Array)과 활용(js6array.html)

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Array 예제</title>  
</head>  
<body>  
    js Array(배열) : 순서가 있는 여러 개의 자료를 대표명을 주고 첨자로 값을 구분하는 자료구조<br>  
    반복처리가 효과적<br>  
    형식 : 배열명 = new Array();  
    <hr>  
    <script type="text/javascript">  
        // 배열 선언과 요소 추가  
        let aa = new Array();  
        aa[0] = 10; aa[1] = 5.6; aa[3] = "결과는";   
        aa[4] = "안녕"; aa[5] = true;      
        aa[6] = {kbs:9, sbs:5} // 객체도 가능  
  
        document.write(aa, ' ', typeof(aa));  
        document.write(`<br>${aa[0]}  ${aa[1]} 전체 크기:${aa.length}`);  
  
        // 변수로 첨자 접근  
        let su = 0, su2 = 0;  
        document.write(`<br>${aa[0]} ${aa[su]} ${aa[su2]}`);  
  
        // 초기값 부여하며 배열 생성  
        let bb = new Array(100, 200, 300);  
        document.write(`<br>${bb[0]} ${bb[1]} ${bb[2]}`);  
  
        // 배열 리터럴과 push  
        let cc = [];  
        cc[0] = "tom";  
        cc.push(23);  
        cc.push("seoul");  
        cc.push(82, 1234, 5678);  
        document.write(`<br>${cc[0]} ${cc[1]} ${cc[2]} ${cc[3]} ${cc[4]} ${cc[5]}`);  
        cc[99] = 'wow'  
        document.write(`<br>cc 배열의 크기는 ${cc.length}`);  
  
        // for를 이용한 배열 초기화 및 출력  
        let dd = new Array();  
        for(let m=0; m<10; m++){  
            dd[m] = m + 1;  
        }  
        document.write("dd[0] : ", dd[0]);  
        document.write("<br>");  
        for(let m=0; m<10; m++){  
            document.write(dd[m] + "&nbsp;&nbsp;&nbsp;");  
            console.log(dd[m]);  
        }  
  
        // 배열에 함수, 객체, 값 혼합 가능  
        let myarr = [  
            '안녕', true, 3.5, {name:'신기해'},  
            function(){ document.write('난 함수'); }  
        ]  
        document.write(myarr[0]);  
        document.write(myarr[3].name);  
        document.write("<br>");  
        myarr[4]();  // 배열 안 함수 실행  
  
        // 배열 요소 출력 방법  
        let korea = ['연필', '지우개', '노트'];  
        for(let i=0; i<korea.length; i++) document.write(korea[i]+" ");  
        for(let i of korea) document.write(i+" ");  
        for(let i in korea) document.write(korea[i]+" ");  
  
        // 배열 요소 제거  
        let ar = ['i', 'go', 'home'];  
        delete ar[1]; // 값 삭제, 자리 남음  
        document.write("<br>", ar, " ", ar.length);  
        ar = ['i', 'go', 'home'];  
        ar.splice(1,1); // 인덱스 1부터 1개 제거  
        document.write("<br>", ar, " ", ar.length);  
  
        // 구조분해 할당  
        let nums = [1,2,3,4];  
        let [a1, a2, a3] = nums;  
        document.write("<br>", `${a1} ${a2} ${a3}`);  
  
        // 전개 연산자  
        const fruits = ['apple', 'peach', 'melon'];  
        const imsi = [...fruits];  
        document.write("<br>", imsi);  
  
    </script>  
</body>  
</html>
```

---

### 상세 설명

#### 배열 선언

- `new Array()` → 가변 크기 배열 생성
    
- `[]` → 배열 리터럴, 현대적 권장 방식
    

#### 배열 요소 접근

- 첨자(index) 사용: `배열명[0]`, `배열명[su]`
    
- 첨자는 상수 또는 변수 가능
    
- 배열은 `0`부터 시작
    

#### 혼합 자료형 저장 가능

- 숫자, 문자열, boolean, 객체, 함수 등 한 배열에 혼합 가능
    
```javascript
let arr = [1, "문자", true, {k:5}, function(){...}];
```

#### 배열 요소 출력

1. 전통 for문
    
```javascript
for(let i=0; i<korea.length; i++) console.log(korea[i]);
```

2. for…of (값 직접)
    
```javascript
for(let i of korea) console.log(i);
```

3. for…in (인덱스 접근)
    
```javascript
for(let i in korea) console.log(korea[i]);
```

#### 배열 요소 제거

- `delete 배열[인덱스]` → 값만 삭제, 자리 남음
    
- `배열.splice(시작인덱스, 개수)` → 값 삭제 및 배열 크기 감소

#### 구조분해 할당

- 배열/객체의 값을 변수에 쉽게 추출
    
```javascript
let [a1,a2,a3] = [1,2,3,4]; // a1=1, a2=2, a3=3
```

#### 전개 연산자

- 배열을 다른 배열로 복사하거나 결합
    
```javascript
const arr2 = [...arr1];
```


---

## 12. 함수(Function) 개념(js7func.html)

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>함수 예제</title>  
    <script type="text/javascript">  
        // 사용자 정의 함수  
        function aa(){  
            count += 1;  
            document.write(count + "번 수행<br>");  
        }  
  
        function bb(){  
            document.write(`bb 함수 실행 성공<br>`);  
            cc(5); // 다른 함수 호출 가능  
  
            let re = cc(5); // 반환값 받기  
            document.write(`re는 ${re}<br>`);  
  
            // 매개변수 전달  
            dd(7, 8);  
            dd(7, 8, 9); // 세번째 인자는 무시  
            dd(7);       // 두번째 인자는 undefined  
        }  
  
        function cc(para){  
            let kk = para + 10;  
            document.write(`kk는 ${kk}<br>`);  
            return kk; // 반환값 1개  
        }  
  
        function dd(para1, para2){  
            document.write(`dd함수에서 ${para1} ${para2}<br>`);  
        }  
  
        // 전역, 지역 변수 예제  
        let a = 100;   // 전역 변수  
        const b = 200; // 전역 상수  
  
        function func(){  
            let c = 300;   // 지역 변수  
            const d = 400; // 지역 상수  
            document.write(`func 내부 a:${a} b:${b}<br>`);  
            document.write(`func 내부 c:${c} d:${d}<br>`);  
        }  
    </script>  
</head>  
<body>  
    <div>함수(function) : 특정 목적의 작업을 수행하도록 설계된 독립적 코드 블록</div>  
  
    <hr>  
    <script>  
        // 전역 변수 초기화  
        let count = 0;  
  
        // 함수 호출 예시  
        aa(); // count 1 증가  
        document.write("뭔가를 하다가...<br>");  
        aa(); // count 2 증가  
        document.write("함수는 참조형 타입 : " + typeof(aa) + "<br>");  
  
        bb(); // 여러 함수 호출 및 반환값 처리  
  
        document.write("전역, 지역 변수 ---<br>");  
        func(); // 지역 변수 출력  
        document.write(`전역 a:${a} b:${b}<br>`);  
    </script>  
</body>  
</html>
```

---

### 상세 설명

#### 함수 정의

- `function 함수명(){ ... }` → 코드 블록을 함수로 정의
    
- 매개변수 전달 가능
    
```javascript
function cc(para){ return para+10; }
```

- 반환값은 `return`으로 1개 가능
    

#### 함수 호출

- `함수명()` → 정의된 함수 실행
    
- JS 함수는 **일급 객체** 지원
    
    - 함수 자체를 변수에 저장 가능
        
    - 함수가 다른 함수의 매개변수로 전달 가능
        
    - 함수가 다른 함수의 반환값이 될 수 있음
        

#### 매개변수 처리

- 정의된 매개변수보다 적게 전달 → 부족한 인자는 `undefined`
    
- 정의된 매개변수보다 많이 전달 → 초과 인자는 무시
    

#### 전역 vs 지역 변수

- 전역 변수: 함수 외부에서 선언, 어디서든 접근 가능
    
- 지역 변수: 함수 내부에서 선언, 외부 접근 불가

```javascript
let a = 100; // 전역  
function func(){  
    let c = 300; // 지역  
}
```

- 상수도 마찬가지: `const b=200;` 전역, `const d=400;` 지역

#### 내장 함수 사용 예

- 문자열 함수: `str.charAt(index)`, `str.bold()`
    
- 수학 함수: `Math.abs(-7)`
    
- JS는 다양한 내장 함수를 지원, 개발 편리

---
### ✔ 보일러플레이트(Boilerplate)란?

- **기본 뼈대 코드**: 웹 개발에서 HTML, CSS, JS를 시작할 때 반드시 들어가는 구조
    
- 예시:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>문서 제목</title>
</head>
<body>
    <script>
        // JS 코드
    </script>
</body>
</html>
```
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <title>문서 제목</title>  
</head>  
<body>  
    <script>  
        // JS 코드  
    </script>  
</body>  
</html>

- 거의 모든 HTML/JS 문서에서 반복됨 → **보일러플레이트**

---

### 왜 알아야 하는가?

1. **빠르게 시작 가능**
    
    - 매번 처음부터 구조를 만들지 않고, 최소한의 뼈대만 복사하면 바로 코딩 가능
        
2. **코드 안정성 확보**
    
    - HTML, JS, CSS가 충돌 없이 제대로 동작하는 최소 환경 제공
        
3. **함수·클래스 작성과 연관**
    
    - 함수나 클래스를 만들려면 **어디에 작성할지, 어디서 호출할지** 알아야 함
        
    - 보일러플레이트를 이해하면 **전역/지역 변수, 함수 호출 위치** 등을 정확히 잡을 수 있음
        

---

💡 **요약**:

> “보일러플레이트를 알면, JS나 함수·클래스를 어디서 어떻게 작성해야 하는지 바로 감이 오고, 안정적인 코드 작성이 가능하다.”


---
## 13. JS 대화상자(Dialog) 함수

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
</head>  
<body>  
    <h2>js의 대화상자 관련 함수</h2>  
    <script>  
        let irum = "홍길동";  
        let tel = "111-1111";  
        alert("이름은 " + irum + "\n전화는 " + tel);    // 메시지창  
        console.log("이름은 " + irum + "\n전화는 " + tel);  
  
        document.write("<br>alert 함수 수행 후 계속");  
  
        let result = confirm("계속할까요?");    // 선택 창  
        document.write("<br>선택한 값은 " + result);    // true, false  
  
        let juso = prompt("사는 곳을 입력", "테헤란로");  
        document.write("<br>사는 곳 : ", juso);  
  
    </script>  
</body>  
</html>
```

---
### 상세 설명

### alert()

- **역할**: 단순 알림 메시지 창
    
- 확인 버튼만 있음
    
- 코드 실행은 **창이 닫힐 때까지 멈춤**
    

예시:

alert("이름은 " + irum + "\n전화는 " + tel);

### confirm()

- **역할**: 확인/취소 선택 창
    
- 반환값: **true / false**
    

let result = confirm("계속할까요?");

- 확인 → `true`, 취소 → `false`
    

### prompt()

- **역할**: 사용자로부터 문자열 입력 받기
    
- 반환값: **문자열**
    
- 기본값 설정 가능:
    

let juso = prompt("사는 곳을 입력", "테헤란로");

---

### 요약

|함수|기능|반환값|
|---|---|---|
|alert()|메시지 창 출력|없음|
|confirm()|확인/취소 선택|true/false|
|prompt()|사용자 입력 받기|문자열|

💡 **왜 알아야 할까?**

> 웹 페이지에서 **사용자와 상호작용**할 때 가장 기본적인 방법.  
> 간단한 테스트, 알림, 입력 확인 용도로 자주 쓰이며, JS를 배우면서 **이벤트와 사용자 반응 처리**의 기초가 됨.