# 📅 2026-01-29

## 오늘 한 일
- Python 설치 및 Python Installer Manager 이해
- CMD / Anaconda Prompt 사용 실습
- venv 가상환경 생성 및 프로젝트별 활성화/비활성화 실습
- ProjectA_HR (인사 관리), ProjectB_Admin (행정 관리) 프로젝트별 독립 환경 구성
- Anaconda 설치 및 conda 가상환경 생성 실습

---

## 핵심 이해

### 1. Python 설치 구조
- Python Installer Manager: Python 설치를 돕는 도구. 설치만 해서는 개발 환경이 만들어지지 않음.  
- 시스템 Python: Installer Manager로 설치한 Python 본체 + 표준 라이브러리 + venv 포함. 직접 개발하지 않고, 프로젝트별 venv의 기반으로 사용.  
- 프로젝트별 venv: 시스템 Python을 기반으로 독립된 개발 환경 생성 → 패키지/버전 충돌 방지, 프로젝트 단위 개발 가능.  
- 실제 개발: 프로젝트별 가상환경에서 진행  
  - Python 버전: 최신 버전 X → 1~2단계 이전 안정화 버전 사용  
    - 이유: 최신 버전은 일부 라이브러리와 호환성 문제 발생 가능  
    - 단점: 멀티스레드 성능 제한 → 동시에 여러 CPU 작업 시 효율이 떨어질 수 있음

---

### 2️. venv 가상환경 운영

#### 2-1. 프로젝트 구조 예시
```
C:\Users\acorn\work\
 ├─ ProjectA\
 │    └─ venv314\      ← Python 3.14
 └─ ProjectB\
      └─ venv313\      ← Python 3.13
```

- 각 프로젝트마다 **독립된 Python 환경** 확보  
- 패키지 충돌 방지 및 Python 버전 관리 용이

---

#### 2-2. 가상환경 생성 / 활성화 / 비활성화

##### ProjectA_HR
```cmd
cd C:\Users\acorn\work\ProjectA_HR
py -3.14 -m venv venv314   # 가상환경 생성
venv314\Scripts\activate   # 활성화
python                     # >>> 파이썬 인터프리터 실행
deactivate                 # 비활성화
```

##### ProjectB_Admin
```cmd
cd C:\Users\acorn\work\ProjectB_Admin
py -3.13 -m venv venv313
venv313\Scripts\activate
python
deactivate
```

---

#### 2-3. 핵심 포인트
- 프로젝트마다 가상환경을 따로 만들어 **패키지/버전 충돌 방지**  
- 가상환경 내에서만 pip/conda install 수행  
- 프로젝트 변경 시 **deactivate → 다음 프로젝트 activate**  
- IDE(VSCode 등)에서 자동 인식되어 개발 편리

---

### 3. Anaconda / conda 가상환경
- Anaconda = 데이터 과학용 독립 플랫폼
  - ipython, numpy, pandas, matplotlib, scipy, scikit-learn, jupyter 등 기본 설치
  - TensorFlow, PyTorch 등 ML/DL 패키지도 conda 또는 pip로 설치 가능

#### 실무에서의 정석 선택
- 풀 세트 설치 (비권장)
```
conda create -n myproject python=3.13.9 anaconda
```
- 표준 방식 / 권장 (필요한 것만 설치)
```
conda create -n myproject python=3.13.9
conda activate myproject
conda install numpy pandas matplotlib
conda install numpy pandas matplotlib scipy scikit-learn jupyter
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**왜 Anaconda 풀 세트를 피하는가?**
- 편하지만 무겁고 관리 어려움
- 필요 없는 패키지 많음 → 환경 복제/공유 어려움
- 의존성 충돌 증가
- 디스크 낭비

#### 주요 명령
```
conda env list               # 가상환경 목록 확인
conda activate myproject     # 환경 진입
conda deactivate             # 환경 나가기
conda list                   # 설치된 패키지 확인
```

---

### 4. pip / conda 패키지 관리
- pip: Python 패키지 설치용
- conda: Anaconda 환경에서 설치, 의존성 자동 해결
- 가상환경 활성화 상태에서만 설치
```
conda activate myproject
conda install numpy pandas matplotlib
pip install pygame   # conda에 없는 패키지
```

---

### 5. site-packages
- 라이브러리 실제 설치 위치
- 가상환경마다 독립 → 프로젝트별 패키지 충돌 방지
- 설치 확인:
```
python -m site
```

---

### 6. 주의 사항 / 강사님 포인트
- 초반에는 ChatGPT 등 도구 의존 금지 → 개념 이해가 먼저
- Python 버전 관리 ≠ 패키지 관리 → 분리된 사고 필요
- 프로젝트마다 독립 가상환경 필수
- base 환경만 사용하면 패키지 충돌, 버전 문제, 관리 어려움
- 필요 없는 풀 세트 설치 금지 → 디스크 낭비, 환경 관리 복잡
- Python 최신 버전보다 1~2단계 이전 안정화 버전 권장 → 호환성 문제 최소화

---

## 느낀 점
- 프로젝트별 venv/conda 환경을 만드는 습관이 중요함
- activate/deactivate로 시스템 Python과 프로젝트 Python을 자유롭게 전환 가능
- Python 버전과 패키지 관리 분리를 체감
- 안정화된 Python 버전을 사용하는 것이 실습/개발 환경 안정에 도움이 됨
