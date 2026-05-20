# Day 71_딥러닝 : YOLO 파인튜닝 · Detection vs Segmentation · 마스크 처리

## 📅 2026-05-20

---
# 📄 yolo7finetunning.ipynb — YOLO11 · Fine-Tuning · Object Detection

---

## 📌 핵심 개념 정리

### 🔷 파인튜닝(Fine-Tuning)이란?

- 이미 대규모 데이터로 학습된 **사전학습 모델(pretrained model)** 의 가중치를 가져와서
- 내가 원하는 **새로운 도메인 데이터**에 맞게 추가 학습하는 것
- 처음부터 학습(scratch)하는 것보다 훨씬 빠르고 성능이 좋음
- 오늘 실습: YOLO11n이 기본 제공하는 **COCO 80 클래스 → 수족관 동물 7 클래스**로 파인튜닝

```
COCO 80 클래스 모델 (yolo11n.pt)
        ↓  파인튜닝 (내 데이터로 추가 학습)
수족관 7 클래스 모델 (best.pt)
: fish, jellyfish, penguin, puffin, shark, starfish, stingray
```

---

### 🔷 YOLO(You Only Look Once)란?

- 이미지를 **단 한 번(One pass)** 에 객체 탐지(bounding box + 클래스 분류)를 동시에 수행하는 모델
- 속도가 매우 빠르고 실시간 탐지에 적합
- YOLO11n = YOLO11 **nano** 버전 → 가장 가볍고 빠른 모델

|모델|파라미터 수|특징|
|---|---|---|
|yolo11n|약 259만|가장 가볍고 빠름|
|yolo11s|약 940만|균형형|
|yolo11m|약 2,000만|중간|
|yolo11l/x|그 이상|고성능, 느림|

---

### 🔷 Object Detection 주요 평가 지표

|지표|설명|
|---|---|
|**Precision**|탐지한 것 중 실제로 맞은 비율 (오탐 적을수록 높음)|
|**Recall**|실제 객체 중 탐지한 비율 (미탐 적을수록 높음)|
|**mAP50**|IoU 0.5 기준 평균 정밀도. 가장 기본적인 탐지 성능 지표|
|**mAP50-95**|IoU 0.5~0.95 구간 평균. 더 엄격한 기준|
|**box_loss**|바운딩박스 위치 예측 오차|
|**cls_loss**|클래스 분류 오차|
|**dfl_loss**|Distribution Focal Loss — 박스 경계 정밀도 오차|

> **IoU(Intersection over Union)**: 예측 박스와 정답 박스가 겹치는 비율. 1에 가까울수록 정확.

---

### 🔷 YAML 환경설정 파일

- YOLO 학습 시 데이터 경로와 클래스 정보를 담는 설정 파일
- `nc` = number of classes (클래스 수)
- `names` = 클래스 이름 리스트 (인덱스 순서가 라벨 번호와 일치해야 함)

**Roboflow 원본 `data.yaml` vs 직접 만든 `Aquarium_Data.yaml` 비교**

|항목|Roboflow 원본 (`data.yaml`)|직접 생성 (`Aquarium_Data.yaml`)|
|---|---|---|
|경로 방식|상대경로 `../train/images`|절대경로 `/content/Aquarium_Data/train/images/`|
|클래스 정보|동일|동일|
|추가 메타데이터|roboflow workspace/project 정보 포함|없음|

```yaml
# Roboflow 원본 data.yaml
train: ../train/images
val: ../valid/images
test: ../test/images
nc: 7
names: ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
roboflow:
  workspace: brad-dwyer
  project: aquarium-combined
  version: 2
  license: CC BY 4.0
  url: https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/2
```

> 사실 `data.yaml`의 경로를 Colab 절대경로로만 바꿔도 그대로 사용 가능 직접 yaml을 생성하는 방법도 알아두면 커스텀 데이터셋 구성 시 유용

---

### 🔷 세그멘테이션(Segmentation)이란?

- Object Detection: 객체를 **사각형 박스**로 감싸는 것
- Segmentation: 객체를 **픽셀 단위**로 구분하는 것 (더 정밀)
- `yolo11n-seg.pt` = segmentation 전용 모델
- `res.masks.data` = 각 객체의 마스크 이미지 (0~1 float)
- `res.masks.xy` = 객체 윤곽선 좌표

---

## 🗺️ 전체 흐름

```
STEP 1. 데이터셋 다운로드 및 압축 해제
        ↓
STEP 2. YAML 환경설정 파일 생성
        ↓
STEP 3. YOLO 모델 로드 및 파인튜닝 (100 epochs)
        ↓
STEP 4. 모델 성능 평가 (test 데이터)
        ↓
STEP 5. 새 이미지 추론 (Inference)
        ↓
STEP 6. 탐지 결과 분석 (클래스별 신뢰도)
        ↓
STEP 7. 세그멘테이션 (픽셀 단위 마스크)
```

### 📷 테스트 이미지 (추론에 사용한 수족관 사진)

<img src="images/yolo7_newimg.jpg" width="400">

---

## 💻 전체 코드

### STEP 1. 데이터셋 다운로드 및 압축 해제

```python
# Roboflow Aquarium 데이터셋 다운로드 (public 도메인 사용)
# public.roboflow.com : 로그인 없이 공개 데이터셋 다운 가능
# ?key= 값은 시간 제한이 있어 만료되면 Roboflow에서 재발급 필요
!wget -O Aquarium_Data.zip https://public.roboflow.com/ds/lh43HXLtGX?key=EwWAf7C2T7

import zipfile

# 다운받은 zip 파일 압축 해제
with zipfile.ZipFile('/content/Aquarium_Data.zip') as file:
    file.extractall('/content/Aquarium_Data/')
```

**📌 출력 결과**

```
Aquarium_Data.zip   100%[===================>]  66.94M  72.4MB/s    in 0.9s
2026-05-20 01:02:55 (72.4 MB/s) - 'Aquarium_Data.zip' saved [70190866/70190866]
```

---

### STEP 2. YAML 환경설정 파일 생성

```python
import yaml

# 학습에 사용할 데이터 경로와 클래스 정보 정의
data = {
    'train': '/content/Aquarium_Data/train/images/',
    'val': '/content/Aquarium_Data/valid/images/',
    'test': '/content/Aquarium_Data/test/images/',
    'names': ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'],
    'nc': 7  # number of classes
}

# yaml 파일로 저장
with open('/content/Aquarium_Data/Aquarium_Data.yaml', 'w') as f:
    yaml.dump(data, f)

# 저장된 내용 확인 (딕셔너리로 불러오기)
with open('/content/Aquarium_Data/Aquarium_Data.yaml', 'r') as f:
    aquarium_yaml = yaml.safe_load(f)
    display(aquarium_yaml)

# 저장된 내용 확인 (터미널 출력)
!cat /content/Aquarium_Data/Aquarium_Data.yaml
```

**📌 출력 결과**

```
names:
- fish
- jellyfish
- penguin
- puffin
- shark
- starfish
- stingray
nc: 7
test: /content/Aquarium_Data/test/images/
train: /content/Aquarium_Data/train/images/
val: /content/Aquarium_Data/valid/images/
```

---

### STEP 3. YOLO 모델 로드 및 파인튜닝

```python
import ultralytics
ultralytics.checks()  # GPU, 버전, 환경 정보 출력

from ultralytics import YOLO

# 사전학습된 YOLO11n 모델 로드 (COCO 80 클래스)
model = YOLO('yolo11n.pt')
print(type(model.names), len(model.names))  # <class 'dict'> 80
print(model.names)  # {0: 'person', 1: 'bicycle', ...}

# 파인튜닝 실행
# - data     : 위에서 만든 yaml 파일 경로
# - epochs   : 전체 학습 반복 횟수
# - patience : 성능 개선 없을 때 조기 종료 기준 (Early Stopping)
# - batch    : 한 번에 처리할 이미지 수 (GPU 메모리에 맞게 설정)
# - imgsz    : 입력 이미지 크기 (YOLO 계열 권장: 416, 512, 640)
model.train(
    data='/content/Aquarium_Data/Aquarium_Data.yaml',
    epochs=100,
    patience=30,
    batch=32,
    imgsz=416
)
```

**📌 출력 결과 (주요 부분)**

```
Ultralytics 8.4.51 🚀 Python-3.12.13 torch-2.10.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
Overriding model.yaml nc=80 with nc=7        ← 80 클래스 헤드를 7 클래스로 교체
YOLO11n summary: 182 layers, 2,591,205 parameters
Transferred 448/499 items from pretrained weights   ← 나머지 레이어 가중치 재사용 (파인튜닝 핵심)

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  mAP50
      1/100      2.06G      1.677      4.158      1.308  0.019
     50/100      2.66G      1.355      0.987      1.061  0.669
    100/100      3.10G      1.049      0.681      0.953  0.721

100 epochs completed in 0.158 hours.
Results saved to /content/runs/detect/train
```

---

### STEP 4. 모델 성능 평가 (test 데이터)

```python
# 저장된 best 가중치로 모델 로드
model = YOLO('/content/runs/detect/train/weights/best.pt')

# test 데이터셋 기준으로 평가
# split='test' : yaml에 정의된 test 경로 사용
metrics = model.val(
    data='/content/Aquarium_Data/Aquarium_Data.yaml',
    split='test'
)

# Metric 객체 속성명 주의: .precision(X) → .mp(O), .recall(X) → .mr(O)
print("Precision:", metrics.box.mp)    # 평균 정밀도
print("Recall:",    metrics.box.mr)    # 평균 재현율
print("mAP50:",     metrics.box.map50) # IoU 0.5 기준 mAP
print("mAP50-95:",  metrics.box.map)   # IoU 0.5~0.95 기준 mAP
```

**📌 출력 결과**

```
Class       Images  Instances    Box(P      R    mAP50  mAP50-95)
all             63        584    0.856  0.645    0.762     0.446
fish            30        249    0.732  0.526    0.637     0.347
jellyfish       11        154    0.827  0.857    0.891     0.538
penguin          7         82    0.888  0.476    0.703     0.291
puffin           6         35    0.689  0.371    0.404     0.197
shark           14         38    0.992  0.711    0.838     0.534
starfish         5         11    0.942  0.909    0.976     0.591
stingray        10         15    0.925  0.667    0.885     0.622

Precision: 0.8563454786811129
Recall: 0.6452229274404307
mAP50: 0.7619597273328912
mAP50-95: 0.44555091112438405
```

**📊 val vs test 비교**

|지표|val|test|
|---|---|---|
|Precision|0.797|**0.856**|
|Recall|0.657|**0.645**|
|mAP50|0.721|**0.762**|
|mAP50-95|0.416|**0.446**|

> test가 val보다 높게 나온 것 → 과적합 없이 일반화 잘 됨 👍 puffin이 가장 낮은 이유 → 학습 데이터가 74장으로 가장 적음

---

### STEP 5. 새 이미지 추론 (Inference)

```python
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

model = YOLO('/content/runs/detect/train/weights/best.pt')

image_path = '/content/yolo7_newimg.jpg'

# save=True : 탐지 결과 이미지를 runs/detect/predict/ 에 자동 저장
results_pred = model.predict(source=image_path, save=True, imgsz=416)

# 저장된 결과 이미지 경로 자동 추출 후 시각화
result_img_path = results_pred[0].save_dir + '/' + results_pred[0].path.split('/')[-1]
img = Image.open(result_img_path)
plt.imshow(img)
plt.axis('off')
plt.show()
```

**📌 추론 결과 이미지 (바운딩박스 표시)**

<img src="images/yolo7finetunning.png" width="400">

---

### STEP 6. 탐지 결과 분석

```python
import numpy as np
from collections import defaultdict  # 키가 없을 때 자동으로 기본값 생성하는 딕셔너리

detected_classes = []
conf_dict = defaultdict(list)  # 클래스별 신뢰도 리스트 자동 생성

# 탐지된 각 박스에서 클래스와 신뢰도 추출
for box in results_pred[0].boxes:
    cls_id = int(box.cls)              # 클래스 인덱스 (정수)
    cls_name = model.names[cls_id]     # 인덱스 → 클래스 이름 변환
    conf = float(box.conf)             # 신뢰도 (0~1)
    detected_classes.append(cls_name)
    conf_dict[cls_name].append(conf)

print('탐지된 클래스 전체:', detected_classes)
print('고유 클래스:', sorted(set(detected_classes)))

# 클래스별 요약 출력
for cls_name, confs in conf_dict.items():
    print(f' - {cls_name}: 갯수={len(confs)}, 평균 신뢰도={np.mean(confs):.3f}')
```

**📌 출력 결과**

```
탐지된 클래스 전체: ['fish', 'fish', 'fish', 'fish', 'shark', 'fish', 'fish', 'fish', 'fish', 'fish', 'fish', 'shark', 'fish', 'fish', 'fish']
고유 클래스: ['fish', 'shark']
 - fish: 갯수=13, 평균 신뢰도=0.826
 - shark: 갯수=2, 평균 신뢰도=0.855
```

---

### STEP 7. 세그멘테이션 (Segmentation)

```python
import os, cv2, numpy as np
from ultralytics import YOLO

IMG_PATH = 'yolo_image1.jpg'
OUT_DIR = 'seg_out'
os.makedirs(OUT_DIR, exist_ok=True)

im = cv2.imread(IMG_PATH)
assert im is not None, f'이미지 없음: {IMG_PATH}'

H, W = im.shape[:2]  # 원본 높이/너비 → 마스크 리사이즈에 사용

# segmentation 전용 모델 로드 (yolo11n-seg.pt)
model = YOLO('yolo11n-seg.pt')
res = model(im)[0]

print(res.boxes)  # 바운딩박스 정보
print(res.masks)  # 세그멘테이션 마스크 정보
# masks.data : 개별 마스크 이미지 데이터
# masks.xy   : 객체 윤곽선 좌표

# sanity check: 바운딩박스 + 마스크 오버레이 저장
cv2.imwrite(os.path.join(OUT_DIR, 'OO_anno.jpg'), res.plot())

if res.masks is None or len(res.masks.data) == 0:
    print('마스크 객체 없음')
    raise SystemExit

# (N, h, w) float 텐서 → numpy 배열 변환 (값 범위 0~1)
m_small = res.masks.data.cpu().numpy()

# 각 마스크를 원본 이미지 크기로 리사이즈
# INTER_NEAREST : 마스크 경계가 흐려지지 않도록 최근접 이웃 보간 사용
masks = np.stack([
    cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) for m in m_small
])
```

```python
# 컬러 오버레이 + 외곽선 최종 세그멘테이션

def color(i):
    # 객체마다 고유한 색상 생성 (소수 곱셈으로 색상 분산)
    return ((37 * i) % 256, (17 * i) % 256, (91 * i) % 256)

final = im.copy()          # 최종 합성용 캔버스
blend = np.zeros_like(im)  # 컬러 오버레이용 빈 캔버스 (im과 동일 크기)

for i, m in enumerate(masks):
    blend[m > 0.5] = color(i)  # 마스크 영역에 고유 색상 채우기

    # 마스크를 uint8(0 or 255)로 변환 후 외곽선 추출
    cnts, _ = cv2.findContours(
        (m.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,       # 가장 바깥쪽 윤곽선만 검출
        cv2.CHAIN_APPROX_SIMPLE  # 꼭지점만 저장해 좌표 단순화
    )
    # 흰색 외곽선을 final 이미지에 덧그리기
    cv2.drawContours(final, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA)

# 반투명 합성: final(원본+외곽선) 100% + blend(컬러 마스크) 45%
final = cv2.addWeighted(final, 1.0, blend, 0.45, 0.0)
cv2.imwrite(os.path.join(OUT_DIR, '02_final_seg.png'), final)

# Colab에서는 cv2.imshow() 사용 불가 → cv2_imshow 사용
from google.colab.patches import cv2_imshow
cv2_imshow(final)
```

> `cv2.imshow()` 는 Colab에서 세션 크래시를 유발하므로 비활성화됨 Colab 전용 대안: `from google.colab.patches import cv2_imshow`

---

## 📊 학습 결과 요약

|항목|값|
|---|---|
|모델|YOLO11n (nano)|
|파라미터 수|약 259만|
|학습 데이터|train 448장, val 127장|
|클래스 수|7 (COCO 80 → 커스텀 7)|
|Epochs|100|
|이미지 크기|416×416|
|GPU|Tesla T4|
|학습 시간|약 9.5분|
|최종 mAP50 (test)|**0.762**|

---

## ⚠️ 오늘의 주요 오류 모음

|오류|원인|해결|
|---|---|---|
|`NameError: tfile`|변수명 오타|`tfile` → `file`|
|`403 Forbidden`|URL 키 만료|Roboflow에서 새 URL 재발급|
|`ModuleNotFoundError: ultralytics`|패키지 미설치|`!pip install ultralytics`|
|`AttributeError: 'Metric' has no 'precision'`|속성명 오류|`.precision` → `.mp`, `.recall` → `.mr`|
|`np.unit8`|오타|`np.uint8`|
|`DisabledFunctionError: cv2.imshow()`|Colab 제한|`from google.colab.patches import cv2_imshow`|
|경로 앞 `/` 누락|`'content/...'`|`'/content/...'`|

---
# 📄 yolo8dec_vs_seg.ipynb — Object Detection · Segmentation · 비교 시각화

---

## 📌 핵심 개념 정리

### 🔷 Object Detection vs Segmentation 비교

|항목|Object Detection|Segmentation|
|---|---|---|
|출력 형태|**사각형 바운딩박스**|**픽셀 단위 마스크**|
|정밀도|낮음 (박스로 근사)|높음 (실제 윤곽선)|
|속도|빠름|느림|
|모델|`yolo11n.pt`|`yolo11n-seg.pt`|
|결과 속성|`result.boxes`|`result.boxes` + `result.masks`|
|활용|객체 위치 파악, 카운팅|의료 영상, 자율주행, 배경 분리|

```
Object Detection     Segmentation
┌──────────┐         ░░░░░░░░░░░
│  [shark] │   vs    ░ (shark) ░   ← 픽셀 단위로 경계 구분
└──────────┘         ░░░░░░░░░░░
  바운딩박스              마스크
```

---

### 🔷 `.plot()` 메서드

- YOLO 결과 객체에서 바로 시각화된 이미지(numpy 배열)를 반환
- Detection이면 바운딩박스 + 라벨 + 신뢰도를 그려서 반환
- Segmentation이면 마스크 + 바운딩박스 + 라벨을 그려서 반환
- 반환값은 **BGR 형식** → matplotlib 출력 시 `cv2.cvtColor()` 변환 필요

```python
det_img = det_results.plot()   # BGR numpy array 반환
plt.imshow(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))  # RGB로 변환 후 출력
```

---

### 🔷 `plt.subplot(행, 열, 순서)`

- 하나의 figure 안에 여러 그래프를 나란히 배치할 때 사용
- `plt.subplot(1, 2, 1)` → 1행 2열 배치에서 첫 번째 칸
- `plt.subplot(1, 2, 2)` → 1행 2열 배치에서 두 번째 칸

```python
plt.figure(figsize=(12, 6))    # 전체 캔버스 크기 설정

plt.subplot(1, 2, 1)           # 왼쪽 칸
plt.imshow(det_img)
plt.title("Detection")

plt.subplot(1, 2, 2)           # 오른쪽 칸
plt.imshow(seg_img)
plt.title("Segmentation")

plt.tight_layout()             # 서브플롯 간 여백 자동 조정
plt.show()
```

---

## 🗺️ 전체 흐름

```
STEP 1. 패키지 설치 및 라이브러리 import
        ↓
STEP 2. 이미지 로드 + 두 모델 로드
        ↓
STEP 3. Detection 추론 → .plot()으로 시각화 이미지 생성
        ↓
STEP 4. Segmentation 추론 → .plot()으로 시각화 이미지 생성
        ↓
STEP 5. subplot으로 두 결과 나란히 비교 출력
```

---

### 📷 테스트 이미지

<img src="images/yolo8_newimg.jpg" width="500">

---

## 💻 전체 코드

### STEP 1. 환경 설치 및 Import

```python
# YOLO 및 OpenCV 설치
!pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
```

---

### STEP 2. 이미지 및 모델 로드

```python
# 테스트용 이미지 경로
image_path = "yolo8_newimg.jpg"

# Detection 전용 모델 — 바운딩박스만 출력
det_model = YOLO("yolo11n.pt")

# Segmentation 전용 모델 — 바운딩박스 + 픽셀 마스크 출력
seg_model = YOLO("yolo11n-seg.pt")

# 이미지 로드 (OpenCV → BGR 형식)
img = cv2.imread(image_path)
assert img is not None, "이미지 로드 실패!"  # 파일 없으면 즉시 오류 발생
```

---

### STEP 3~4. Detection & Segmentation 추론

```python
# 1) Object Detection 추론
det_results = det_model(img)[0]   # [0] : 첫 번째 이미지 결과 추출
det_img = det_results.plot()      # 바운딩박스 + 라벨 시각화된 BGR 이미지 반환

# 2) Image Segmentation 추론
seg_results = seg_model(img)[0]   # [0] : 첫 번째 이미지 결과 추출
seg_img = seg_results.plot()      # 마스크 + 바운딩박스 시각화된 BGR 이미지 반환
```

---

### STEP 5. 나란히 비교 출력

```python
plt.figure(figsize=(12, 6))   # 전체 캔버스 크기 (가로 12인치, 세로 6인치)

# 왼쪽: Detection 결과
plt.subplot(1, 2, 1)                                        # 1행 2열 중 1번째 칸
plt.imshow(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))        # BGR → RGB 변환 필수
plt.title("Object Detection (Bounding Boxes)")
plt.axis("off")                                             # 축 눈금 숨기기

# 오른쪽: Segmentation 결과
plt.subplot(1, 2, 2)                                        # 1행 2열 중 2번째 칸
plt.imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))        # BGR → RGB 변환 필수
plt.title("Image Segmentation (Pixel Masks)")
plt.axis("off")

plt.tight_layout()   # 서브플롯 간 여백 자동 조정 (겹침 방지)
plt.show()
```

**📌 입력 이미지**

<img src="images/yolo8_newimg.jpg" width="500">

**📌 출력 결과 (Detection vs Segmentation 비교)**

<img src="images/yolo8dec_vs_seg.png" width="700">

**📌 실제 결과 관찰 포인트**

이 이미지는 수족관 동물 사진인데, YOLO가 **COCO 80 클래스 기준**으로 판단하다 보니 흥미로운 오탐이 발생했다.

|실제 객체|YOLO 예측|이유|
|---|---|---|
|상어, 돌고래, 거북이, 물고기, 오리 등|`bird`|COCO에 물고기/상어 클래스 없음 → 가장 유사한 동물로 대체|
|거북이 등껍질|`cake` (0.26)|둥글고 패턴 있는 텍스처가 케이크와 유사하게 보임|
|주황색 물고기|`kite` (0.28)|납작하고 밝은 색상이 연 모양과 유사|

> 이것이 바로 **파인튜닝이 필요한 이유** — COCO 기본 모델은 수족관 동물을 모름 앞 실습(yolo7finetunning)에서 Aquarium 데이터로 파인튜닝한 모델은 fish/shark를 정확히 탐지했던 것과 대조됨

---

## 📊 Detection vs Segmentation 결과 차이

|항목|Detection|Segmentation|
|---|---|---|
|표현 방식|사각형 박스|픽셀 마스크 (색상 오버레이)|
|객체 경계|근사|정밀|
|겹치는 객체 처리|박스가 겹침|마스크로 구분 가능|
|연산 비용|낮음|높음|

---

## ⚠️ 주의사항

|항목|내용|
|---|---|
|BGR vs RGB|OpenCV는 BGR, matplotlib은 RGB → `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 필수|
|`.plot()` 반환값|numpy 배열 (BGR) — 파일 저장은 `cv2.imwrite()`, 화면 출력은 변환 후 `plt.imshow()`|
|Colab `cv2.imshow()`|사용 불가 → `from google.colab.patches import cv2_imshow` 사용|

---
# 📄 yolo9seg.ipynb — Segmentation · 마스크 처리 · 컬러 오버레이 · 외곽선

---

## 📌 핵심 개념 정리

### 🔷 Segmentation 결과 데이터 구조

```
res = model(im)[0]
├── res.boxes          → 바운딩박스 정보
│   ├── .xyxy          → 절대 픽셀 좌표 [x1, y1, x2, y2]
│   ├── .cls           → 클래스 인덱스
│   └── .conf          → 신뢰도 (0~1)
└── res.masks          → 세그멘테이션 마스크 정보
    ├── .data          → (N, h, w) 마스크 텐서 — 객체별 이진 마스크 이미지
    └── .xy            → 객체별 윤곽선 좌표 리스트 (픽셀 좌표)
```

> Detection은 박스로 위치를 표현하지만, Segmentation은 **픽셀 단위**로 객체 영역을 표현

---

### 🔷 마스크 처리 파이프라인

YOLO가 반환하는 마스크는 **모델 입력 크기(416×640)** 기준이라 원본 이미지 크기와 다름. 그래서 원본 크기로 리사이즈하는 과정이 필요하다.

```
res.masks.data           → (N, 416, 640) float 텐서  ← 모델 내부 크기
        ↓ .cpu().numpy()
m_small                  → (N, 416, 640) numpy 배열
        ↓ cv2.resize() × N개
masks                    → (N, H, W) bool 배열       ← 원본 이미지 크기
```

---

### 🔷 `np.stack()` vs `np.array()`

```python
# np.stack : 새로운 축을 추가하며 배열을 쌓음
masks = np.stack([m1, m2, m3], axis=0)   # → (3, H, W)

# np.array : 단순히 배열로 변환 (기존 축 사용)
masks = np.array([m1, m2, m3])           # → (3, H, W) (결과 동일하지만 의미 다름)
```

> `axis=0` : 가장 앞에 새 축 추가 → (H, W) 배열 N개 → (N, H, W)

---

### 🔷 `masks.any(axis=0)` — 마스크 합산

```python
mask_union = masks.any(axis=0)
# any(axis=0) : N개 마스크 중 하나라도 True인 픽셀을 True로
# 결과: (H, W) bool 배열 — 어떤 객체든 있으면 True
# → 모든 객체 마스크를 하나로 합친 미리보기 이미지 생성에 사용
```

---

### 🔷 `cv2.INTER_NEAREST` (최근접 이웃 보간법)

|보간법|특징|
|---|---|
|`INTER_LINEAR` (기본)|주변 픽셀 평균 → 경계가 부드럽지만 **흐려짐**|
|`INTER_NEAREST`|가장 가까운 픽셀값 그대로 → **경계 선명**|

마스크는 0 또는 1(True/False)만 있어야 하는데, 보간 시 중간값이 생기면 경계가 흐려진다. 그래서 마스크 리사이즈에는 항상 `INTER_NEAREST` 사용.

---

### 🔷 `cv2.findContours()` — 윤곽선 검출

```python
cnts, _ = cv2.findContours(
    (m.astype(np.uint8) * 255),  # 입력: 0/255 이진 이미지 (uint8 필수)
    cv2.RETR_EXTERNAL,           # 검출 모드: 가장 바깥쪽 윤곽선만
    cv2.CHAIN_APPROX_SIMPLE      # 근사 방법: 직선 구간은 꼭지점만 저장
)
# 반환: (윤곽선 리스트, 계층 정보)
# _ 로 계층 정보 무시
```

|옵션|설명|
|---|---|
|`RETR_EXTERNAL`|가장 바깥쪽 윤곽선만 검출 (내부 구멍 무시)|
|`RETR_TREE`|모든 윤곽선 + 계층 구조|
|`CHAIN_APPROX_SIMPLE`|꼭지점만 저장 (좌표 압축, 메모리 절약)|
|`CHAIN_APPROX_NONE`|모든 경계 픽셀 저장|

---

### 🔷 `cv2.addWeighted()` — 반투명 합성

```python
final = cv2.addWeighted(src1, alpha, src2, beta, gamma)
# result = src1 × alpha + src2 × beta + gamma

cv2.addWeighted(final, 1.0, blend, 0.45, 0.0)
# final(원본+외곽선) 100% + blend(컬러 마스크) 45% 합성
# → 원본이 비쳐보이는 반투명 컬러 오버레이 효과
```

---

### 🔷 `assert` — 조건 검증

```python
assert im is not None, f'이미지 없음: {IMG_PATH}'
# 조건이 False이면 AssertionError 발생 + 메시지 출력
# 파일 경로 오류나 이미지 로드 실패를 즉시 감지하는 용도
```

---

## 🗺️ 전체 흐름

```
STEP 1. 이미지 로드 + Segmentation 모델 추론
        ↓
STEP 2. 마스크 추출 → 원본 크기로 리사이즈 → bool 배열 변환
        ↓
STEP 3. 마스크 미리보기 저장 (O1_mask_preview.png)
        ↓
STEP 4. 객체별 컬러 오버레이 + 외곽선 그리기
        ↓
STEP 5. 반투명 합성 → 최종 결과 저장 + 출력 (02_final_seg.png)
```

---

### 📷 테스트 이미지

<img src="images/yolo_image1.jpg" width="400">

---

## 💻 전체 코드

### STEP 1. 모델 로드 및 추론

```python
import os, cv2, numpy as np
from ultralytics import YOLO

IMG_PATH = 'yolo_image1.jpg'
OUT_DIR = 'seg_out'                      # 결과 이미지 저장 폴더
os.makedirs(OUT_DIR, exist_ok=True)      # 폴더 없으면 자동 생성

im = cv2.imread(IMG_PATH)
assert im is not None, f'이미지 없음: {IMG_PATH}'  # 로드 실패 시 즉시 오류

H, W = im.shape[:2]   # 원본 높이/너비 저장 (마스크 리사이즈 기준 크기)

# Segmentation 전용 모델 로드
model = YOLO('yolo11n-seg.pt')
res = model(im)[0]    # [0]: 첫 번째 이미지 결과

# 탐지 결과 확인
print(res.boxes)   # 바운딩박스 정보 (cls, conf, xyxy 등)
print(res.masks)   # 마스크 정보 (data, xy)

# sanity check: YOLO 기본 시각화 이미지 저장 (정상 작동 확인용)
cv2.imwrite(os.path.join(OUT_DIR, 'OO_anno.jpg'), res.plot())
```

**📌 출력 결과**

```
0: 416x640 2 persons, 315.7ms
cls: tensor([0., 0.])
conf: tensor([0.8409, 0.7079])
xyxy: tensor([[120.0181, 0.9432, 283.0374, 173.3571],
              [ 15.4300, 11.7173, 165.8862, 173.1279]])
shape: torch.Size([2, 416, 640])   ← 마스크 2개, 모델 내부 크기 416×640
```

**📌 OO_anno.jpg (YOLO 기본 시각화)**

<img src="images/yolo9_OO_anno.jpg" width="400">

---

### STEP 2. 마스크 추출 및 리사이즈

```python
# 마스크 없으면 종료
if res.masks is None or len(res.masks.data) == 0:
    print('마스크 객체 없음')
    raise SystemExit

# (N, h, w) float 텐서 → numpy 배열로 변환 (값: 0~1)
m_small = res.masks.data.cpu().numpy()

# 각 마스크를 원본 이미지 크기(H, W)로 리사이즈
# INTER_NEAREST : 마스크 경계가 흐려지지 않도록 최근접 이웃 보간 사용
# > 0.5 : float → bool 변환 (임계값 0.5 기준 이진화)
# np.stack(..., axis=0) : N개의 (H,W) bool 배열 → (N, H, W) 배열
masks = np.stack([
    cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
    for m in m_small
], axis=0)
```

---

### STEP 3. 마스크 미리보기 저장

```python
# any(axis=0) : N개 마스크 중 하나라도 True인 픽셀 → True
# 모든 객체 영역을 하나의 흑백 이미지로 합산
mask_union = (masks.any(axis=0).astype(np.uint8) * 255).astype(np.uint8)
# bool → uint8 변환 후 × 255 : OpenCV 이미지 저장 범위 0~255에 맞춤

cv2.imwrite(os.path.join(OUT_DIR, 'O1_mask_preview.png'), mask_union)
```

**📌 O1_mask_preview.png (마스크 미리보기 — 흰색=객체 영역)**

<img src="images/yolo9_O1_mask_preview.png" width="400">

---

### STEP 4~5. 컬러 오버레이 + 외곽선 + 최종 합성

```python
def color(i):
    # 객체 번호 i마다 고유한 BGR 색상 생성
    # 소수(37, 17, 91) 곱셈으로 색이 골고루 퍼지게 분산
    # % 256 : BGR 범위(0~255) 초과 방지
    return ((37 * i) % 256, (17 * i) % 256, (91 * i) % 256)

final = im.copy()          # 외곽선을 그릴 캔버스 (원본 복사)
blend = np.zeros_like(im)  # 컬러 마스크를 채울 빈 캔버스 (원본과 동일 크기, 검정)

for i, m in enumerate(masks):
    # 마스크 영역(True 픽셀)에 고유 색상 채우기
    # blend[m] 은 float 배열이라 불안정 → m > 0.5 로 명시적 bool 인덱싱
    blend[m > 0.5] = color(i)

    # 마스크를 uint8(0 or 255) 이진 이미지로 변환 후 윤곽선 검출
    cnts, _ = cv2.findContours(
        (m.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,       # 가장 바깥쪽 윤곽선만 검출
        cv2.CHAIN_APPROX_SIMPLE  # 직선 구간은 꼭지점만 저장 (좌표 압축)
    )
    # final 위에 흰색 윤곽선 그리기
    # -1 : 모든 윤곽선, (255,255,255) : 흰색, 2 : 두께, LINE_AA : 안티앨리어싱
    cv2.drawContours(final, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA)

# 반투명 합성
# final(외곽선 그린 원본) × 1.0 + blend(컬러 마스크) × 0.45
# → 원본이 비쳐보이는 반투명 컬러 오버레이 효과
final = cv2.addWeighted(final, 1.0, blend, 0.45, 0.0)

cv2.imwrite(os.path.join(OUT_DIR, '02_final_seg.png'), final)

# Colab에서는 cv2.imshow() 사용 불가 → cv2_imshow 사용
from google.colab.patches import cv2_imshow
cv2_imshow(final)
```

**📌 02_final_seg.png (최종 세그멘테이션 결과)**

<img src="images/yolo9_02_final_seg.png" width="400">

---

## 📊 출력 파일 정리

|파일명|내용|
|---|---|
|`OO_anno.jpg`|YOLO 기본 시각화 (sanity check용)|
|`O1_mask_preview.png`|모든 마스크 합산한 흑백 미리보기|
|`02_final_seg.png`|컬러 오버레이 + 흰색 외곽선 최종 결과|

---

## ⚠️ 주요 주의사항

|항목|내용|
|---|---|
|`blend[m]` vs `blend[m > 0.5]`|float 마스크는 bool 인덱싱 불안정 → `> 0.5` 명시 필수|
|`np.uint8`|`np.unit8` 으로 오타 주의 (unit ❌ → uint ✅)|
|마스크 크기|YOLO 내부 크기(416×640) ≠ 원본 크기 → 반드시 리사이즈 필요|
|`cv2.imshow()`|Colab에서 비활성화 → `from google.colab.patches import cv2_imshow`|
|`findContours` 입력|`uint8` 타입 필수, `float`으로 넣으면 오류|
