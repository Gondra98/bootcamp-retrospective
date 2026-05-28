# Day75_딥러닝 : YOLO Classification · ByteTrack · Multi Object Tracking

## 📅 2026-05-27

---
# 📄 yolo11classify.ipynb — YOLO11-cls · Flower Dataset · Confusion Matrix

## 🧠 개념 정리

### YOLO 이미지 분류 (Classification)

YOLO는 원래 객체 탐지(Object Detection) 모델이지만, `-cls` 버전은 **이미지 분류** 전용 모델이다.

|구분|Object Detection|Classification|
|---|---|---|
|목적|객체 위치 + 클래스 탐지|이미지 전체의 클래스 판별|
|출력|Bounding Box + Class|Class + Confidence Score|
|모델 예시|yolo11n.pt|yolo11n-cls.pt|

> 즉, YOLO-cls는 "이 이미지가 무슨 꽃이야?" 처럼 **전체 이미지 단위**로 분류한다.

---

### 데이터셋 구조 (YOLO Classification 형식)

YOLO 분류 모델은 아래 폴더 구조를 요구한다.

```
flower_dataset/
├── train/
│   ├── daisy/
│   ├── dandelion/
│   ├── roses/
│   ├── sunflowers/
│   └── tulips/
├── val/
│   └── (동일 구조)
└── test/
    └── (동일 구조)
```

- 폴더 이름 자체가 클래스 레이블 역할을 한다
- 별도의 yaml 파일 없이 폴더 구조만으로 학습 가능

---

### 평가 지표

|지표|설명|
|---|---|
|Accuracy|전체 중 맞춘 비율|
|Precision|예측 양성 중 실제 양성 비율|
|Recall|실제 양성 중 맞게 예측한 비율|
|F1-Score|Precision과 Recall의 조화 평균|
|Confusion Matrix|클래스별 예측 결과를 행렬로 시각화|

---

## 1. 환경 설치

```python
# YOLO 및 OpenCV 설치
!pip install ultralytics opencv-python
```

---

## 2. 라이브러리 로딩 및 데이터셋 다운로드

```python
import random
import shutil
from pathlib import Path
import tensorflow as tf
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow 유틸로 꽃 이미지 데이터셋 다운로드 및 압축 해제
# flower_photos: daisy / dandelion / roses / sunflowers / tulips 5개 클래스
dataset_path = tf.keras.utils.get_file(
    fname="flower_photos",
    origin="http://download.tensorflow.org/example_images/flower_photos.tgz",
    untar=True  # tgz 파일을 자동으로 압축 해제
)

SOURCE_DIR = Path(dataset_path) / "flower_photos"
print('SOURCE_DIR = ', SOURCE_DIR)

# 하위 폴더(클래스) 목록 출력: ['sunflowers', 'roses', 'tulips', 'daisy', 'dandelion']
classes = [p.name for p in SOURCE_DIR.iterdir() if p.is_dir()]
print('classes : ', classes)
```

---

## 3. YOLO용 데이터셋 폴더 생성 및 분할

```python
# YOLO 학습용 폴더 구조 생성: flower_dataset/train|val|test/클래스명/
DATASETS_DIR = Path('flower_dataset')

# 기존 폴더가 있으면 삭제 후 새로 생성 (중복 방지)
if DATASETS_DIR.exists():
    shutil.rmtree(DATASETS_DIR)

# train / val / test 경로 설정
TRAIN_DIR = DATASETS_DIR / 'train'
VAL_DIR   = DATASETS_DIR / 'val'
TEST_DIR  = DATASETS_DIR / 'test'

for class_dir in SOURCE_DIR.iterdir():
    if not class_dir.is_dir():  # 파일이면 스킵 (LICENSE.txt 등)
        continue

    class_name = class_dir.name             # 클래스 이름 (ex: daisy)
    images = list(class_dir.glob("*.*"))    # 해당 클래스의 모든 이미지 파일 리스트

    if len(images) == 0:
        continue

    random.shuffle(images)  # 학습 편향 방지를 위해 순서 무작위 섞기

    total = len(images)
    train_end = int(total * 0.7)   # 70% → train
    val_end   = int(total * 0.9)   # 20% → val

    # 비율에 따라 분할 (7 : 2 : 1)
    splits = {
        'train': images[:train_end],
        'val':   images[train_end:val_end],
        'test':  images[val_end:]           # 나머지 10% → test
    }
    print(len(splits['train']), " ", len(splits['val']), " ", len(splits['test']))

    # 각 분할 데이터를 폴더에 복사
    for split_name, split_images in splits.items():
        target_dir = DATASETS_DIR / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)  # 중간 폴더 없어도 자동 생성

        for img in split_images:
            shutil.copy2(img, target_dir / img.name)   # 메타데이터 포함 복사

# 생성된 폴더 구조 및 이미지 수 확인
for split in ['train', 'val', 'test']:
    print(f'[{split}]')
    for class_dir in (DATASETS_DIR / split).iterdir():
        count = len(list(class_dir.glob("*.*")))
        print(f'{class_dir.name}의 건수는 : {count}')
```

---

## 4. YOLO11 분류 모델 학습

```python
# 사전 학습된 YOLO11 nano classification 모델 로딩
# yolo11n-cls.pt: ImageNet으로 사전 학습된 경량 모델 (n = nano)
model = YOLO('yolo11n-cls.pt')

model.train(
    data=str(DATASETS_DIR.resolve()),  # 데이터셋 절대 경로 (상대 경로는 오류 발생 가능)
    epochs=5,                          # 전체 데이터를 5번 반복 학습
    imgsz=224,                         # 입력 이미지 크기 224x224 (YOLO-cls 기본값)
    batch=16,                          # 한 번에 16장씩 학습
    device='cpu'                       # GPU 없을 경우 cpu 사용 (Colab은 'cuda' 또는 0)
)

print('학습 완료')  # 학습 후 정확도 약 0.956 달성
```

---

## 5. 모델 성능 평가 (test 데이터)

```python
# 학습 중 val 기준으로 가장 성능 좋았던 가중치 로딩
best_model = YOLO("runs/classify/train/weights/best.pt")

y_true = []  # 실제 정답 레이블 리스트
y_pred = []  # 예측 레이블 리스트

test_dir = DATASETS_DIR / 'test'

for class_dir in test_dir.iterdir():
    true_label = class_dir.name  # 폴더명 = 실제 클래스

    for img_path in class_dir.glob('*.*'):
        results = best_model.predict(
            source=str(img_path),
            imgsz=224,
            verbose=False  # 예측 로그 출력 억제
        )

        r = results[0]
        pred_idx   = r.probs.top1           # 가장 높은 확률의 클래스 인덱스
        pred_label = r.names[pred_idx]      # 인덱스 → 클래스 이름으로 변환

        y_true.append(true_label)
        y_pred.append(pred_label)

# 전체 정확도 출력
acc = accuracy_score(y_true, y_pred)
print(f'test acc : {acc * 100:.2f}%')

# 클래스별 precision / recall / f1-score 출력
print(classification_report(y_true, y_pred))

# Confusion Matrix 히트맵 시각화
# 행 = 실제 클래스 (True), 열 = 예측 클래스 (Predicted)
# 대각선 값이 클수록 정확한 분류
cm = confusion_matrix(y_true, y_pred, labels=classes)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,          # 각 셀에 숫자 표시
    fmt='d',             # 정수 형식
    xticklabels=classes,
    yticklabels=classes,
    cmap='Blues'         # 파란색 계열 색상맵
)
plt.title('Confusion Matrix(test data)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

---

## 6. 새 이미지 추론 (단일 이미지 분류)

```python
print('새 꽃 이미지 분류')
sample_path = Path('myflower.jpeg')

if sample_path.exists():
    results = best_model.predict(
        source=str(sample_path),
        imgsz=224,
        verbose=False
    )
    r = results[0]

    pred_idx   = r.probs.top1               # 예측 클래스 인덱스
    pred_label = r.names[r.probs.top1]      # 예측 클래스 이름
    confidence = float(r.probs.top1conf)    # 예측 확신도 (0.0 ~ 1.0)

    print(sample_path)
    print('Predicted class : ', pred_label)
    print('confidence : ', round(confidence, 3))
else:
    print('파일 없음')  # myflower.jpeg 파일이 없을 경우
```

**📌 입력 이미지 — myflower.jpeg**

<img src="images/myflower.jpeg" width="400">

**📌 출력 결과 예시**

```
새 꽃 이미지 분류
myflower.jpeg
Predicted class :  roses
confidence :  0.978
```

---

## 📌 핵심 정리

- `yolo11n-cls.pt` : 이미지 분류 전용 YOLO 모델, 폴더 구조만으로 학습 가능
- 데이터 분할 비율 : train 70% / val 20% / test 10%
- `r.probs.top1` : 예측 결과에서 가장 높은 확률의 클래스 인덱스
- `r.probs.top1conf` : 해당 클래스의 신뢰도 (Confidence Score)
- Confusion Matrix : 대각선 = 정답, 비대각선 = 오분류 → 어떤 클래스끼리 혼동되는지 파악

---
# 📄 yolo12tracking.ipynb — YOLO11 Tracking · ByteTrack · Multi Object Tracking

---

## 📌 핵심 개념 정리

### 🔷 Detection vs Tracking

|항목|Detection|Tracking|
|---|---|---|
|동작|매 프레임 독립적으로 객체 탐지|이전 프레임과 현재 프레임을 비교해 동일 객체 추적|
|ID 부여|없음|같은 객체에 고유 ID 유지|
|활용|정지 이미지 분석|영상 내 이동 객체 추적, CCTV, 자율주행|

```
Detection만 사용 시
Frame 1: [car], [car], [car]
Frame 2: [car], [car], [car]   ← 같은 차인지 알 수 없음

Tracking 사용 시
Frame 1: [id:1 car], [id:2 car], [id:3 car]
Frame 2: [id:1 car], [id:2 car], [id:3 car]   ← 동일 객체 유지
```

---

### 🔷 ByteTrack 알고리즘

ByteTrack은 YOLO와 함께 가장 많이 사용되는 다중 객체 추적 알고리즘이다.

- 신뢰도 높은 탐지 + **낮은 탐지도 버리지 않고** 재매칭에 활용 → ID 유지율 향상
- 이전 프레임 트랙과 현재 탐지 결과를 **IoU 기반**으로 매칭
- `bytetrack.yaml` 파일로 파라미터 조정 가능

```
주요 파라미터 (bytetrack.yaml)
- track_high_thresh : 1차 매칭 신뢰도 임계값
- track_low_thresh  : 2차 재매칭 신뢰도 임계값
- new_track_thresh  : 신규 트랙 생성 임계값
- track_buffer      : 트랙 유지 버퍼 (프레임 수)
```

---

### 🔷 출력 데이터 구조

|항목|설명|
|---|---|
|`result.boxes.id`|추적 ID 배열. 탐지 객체 없으면 `None`|
|`result.boxes.xywh`|박스 좌표 (center_x, center_y, width, height)|
|`result.boxes.cls`|클래스 인덱스 배열|
|`result.names[idx]`|클래스 인덱스 → 이름 변환 (ex: 2 → car)|
|`result.plot()`|탐지 결과를 이미지 위에 자동 시각화해 반환|

> `boxes.xywh`는 center_x, center_y, w, h 형식 → x1y1x2y2와 다름에 주의

---

### 🔷 OpenCV 영상 처리 흐름

```
VideoCapture(영상 열기)
    ↓
cap.read() → 프레임 1장 읽기
    ↓
model.track() → YOLO 추적 실행
    ↓
result.plot() → 결과 시각화
    ↓
cv2.imshow() → 화면 출력
    ↓
waitKey(1) → 키 입력 감지 (q: 종료)
    ↓
cap.release() + destroyAllWindows()
```

---

### 🔷 `persist=True` 의 역할

```python
model.track(source=frame, persist=True, ...)
```

- `persist=False` : 매 프레임마다 트래커 초기화 → ID 계속 바뀜
- `persist=True` : 이전 프레임의 트랙 상태를 현재 프레임으로 이어서 유지 → ID 연속성 보장

---

## 🗺️ 전체 흐름

```
STEP 1. 환경 설치 + 라이브러리 로딩
        ↓
STEP 2. YOLO 모델 로드 + 영상 열기
        ↓
STEP 3. 프레임 읽기 + 추적 실행 (while 루프)
        ↓
STEP 4. 추적 결과 파싱 (ID, 클래스, 박스 좌표)
        ↓
STEP 5. 시각화 + 화면 출력
        ↓
STEP 6. 자원 해제
```

---

## 💻 전체 코드

### STEP 1. 환경 설치 + 라이브러리 로딩

```python
# Ultralytics YOLO를 사용한 다중 객체 추적
# - Detection : 사람 / 동물 / 자동차 등 감지
# - Tracking  : 같은 객체에 id를 부여해 프레임이 바뀌어도 동일 객체 유지

!pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# 사전 학습된 YOLO11 nano 모델 로딩 (COCO 80개 클래스 학습됨)
model = YOLO("yolo11n.pt")
```

---

### STEP 2. 영상 열기

```python
video_path = "road_car.mp4"

# VideoCapture : 지정한 동영상 파일을 프레임 단위로 읽기 위한 객체 생성
cap = cv2.VideoCapture(video_path)

# 파일 열기 실패 시 종료 (경로 오류, 파일 없음 등)
if not cap.isOpened():
    print('동영상 열기 실패')
    exit()
```

---

### STEP 3. 프레임 읽기 + 추적 실행

```python
while True:
    # ret   : 프레임 읽기 성공 여부 (True / False)
    # frame : 읽어온 실제 이미지 데이터 (numpy array, BGR 형식)
    ret, frame = cap.read()

    if not ret:   # 영상 끝 또는 읽기 실패 시 루프 종료
        break

    results = model.track(
        source  = frame,             # 분석할 입력 이미지 (현재 프레임 1장)
        persist = True,              # 이전 프레임의 트랙 ID를 현재 프레임에서도 유지
        tracker = "bytetrack.yaml",  # ByteTrack 알고리즘 사용
        verbose = False,             # 콘솔 로그 출력 억제
        show    = False,             # YOLO 내장 창 비활성화
    )

    result = results[0]   # 첫 번째(유일한) 결과 추출
```

---

### STEP 4. 추적 결과 파싱

```python
    # 추적된 객체가 있을 때만 처리 (탐지 결과 없으면 boxes.id = None)
    if result.boxes is not None and result.boxes.id is not None:
        boxes     = result.boxes.xywh.cpu().numpy()            # 박스 좌표 (cx, cy, w, h)
        ids       = result.boxes.id.cpu().numpy().astype(int)  # 추적 ID 배열
        class_ids = result.boxes.cls.cpu().numpy().astype(int) # 클래스 인덱스 배열

        # 박스 좌표 / 추적 ID / 클래스를 묶어 반복 처리
        for box, track_id, class_id in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = map(int, box)      # xywh 값을 정수로 변환
            class_name = result.names[class_id]  # 클래스 인덱스 → 이름 (ex: 2 → car)
            print(f"id : {track_id} class : {class_name} box : ", x1, y1, x2, y2)
```

**📌 출력 결과 — 프레임 1**

```
id : 1 class : car box :  1401 1550 489 481
id : 2 class : car box :  1098 1674 673 602
id : 3 class : car box :  838 2058 665 192
id : 4 class : car box :  2490 1454 572 572
id : 5 class : car box :  2236 1048 392 341
id : 6 class : car box :  1497 1223 491 419
id : 7 class : car box :  179 835 358 291
id : 8 class : car box :  1477 945 364 219
id : 9 class : truck box :  2781 1925 1066 456
id : 10 class : truck box :  2271 574 291 251
id : 11 class : car box :  1637 910 306 293
id : 12 class : car box :  705 680 224 203
id : 13 class : truck box :  442 706 320 333
id : 14 class : car box :  1550 749 286 171
id : 15 class : bus box :  442 705 320 330
id : 16 class : car box :  2278 783 296 187
id : 17 class : truck box :  199 595 285 193
id : 18 class : person box :  2682 2080 152 141
```

**📌 출력 결과 — 프레임 2**

```
id : 1 class : car box :  1398 1559 491 483
id : 2 class : car box :  1092 1688 685 614
id : 3 class : car box :  830 2069 616 177
id : 4 class : car box :  2490 1462 572 573
id : 5 class : car box :  2240 1052 398 346
id : 6 class : car box :  1495 1222 481 410
id : 7 class : car box :  185 832 353 286
id : 8 class : car box :  1475 948 369 222
id : 9 class : truck box :  2804 1932 1024 438
id : 10 class : truck box :  2270 574 290 250
id : 11 class : car box :  1636 917 318 305
id : 12 class : car box :  707 678 222 201
id : 13 class : bus box :  444 704 315 327
id : 14 class : car box :  1547 750 288 173
id : 15 class : truck box :  444 705 317 327
id : 16 class : car box :  2260 781 307 193
```

---

### STEP 5. 시각화 + 화면 출력

```python
    # YOLO 탐지 결과(바운딩박스, ID, 클래스, 신뢰도)를 이미지 위에 자동으로 그리기
    annotated_frame = result.plot()

    # 화면 출력용으로 960x540 크기로 리사이즈
    display_frame = cv2.resize(annotated_frame, (960, 540))

    cv2.imshow("YOLO Tracking", display_frame)

    # waitKey(1) : 1ms 대기하며 키 입력 감지
    # & 0xFF     : 하위 8비트만 추출 (운영체제 호환성)
    # q 키 누르면 루프 탈출
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```

**📌 출력 결과 — 트래킹 시각화**

<img src="images/yolo12tracking.png" width="500"> <img src="images/yolo12tracking2.png" width="500">

---

### STEP 6. 자원 해제

```python
cap.release()           # VideoCapture 객체 해제 (파일 핸들 반환)
cv2.destroyAllWindows() # 열려 있는 모든 OpenCV 창 닫기
```

---

## 📊 실행 결과 요약

|항목|값|
|---|---|
|모델|YOLO11n (nano)|
|트래커|ByteTrack|
|입력 영상|road_car.mp4|
|탐지 클래스|car, truck, bus, person|
|최대 추적 ID|18 (프레임 기준)|
|출력 해상도|960 × 540|

---

## 📌 핵심 정리

- `model.track()` : `model.predict()`에 추적 기능 추가. `persist=True` 가 핵심
- `persist=True` : 이전 프레임의 트랙 ID를 현재 프레임으로 이어서 유지
- `result.boxes.id` : 추적 ID 배열. 탐지 객체 없으면 `None` → 조건 확인 필수
- `result.boxes.xywh` : center_x, center_y, width, height 형식 (x1y1x2y2 아님)
- `result.plot()` : 탐지 결과를 이미지에 자동 시각화해 반환
- Colab 환경에서는 `cv2.imshow()` 대신 `from google.colab.patches import cv2_imshow` 필요


