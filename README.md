# road-cctv-calib

> **도로 CCTV 카메라 자동 캘리브레이션 도구**
>
> 체커보드 없이, **차량 궤적**과 **수직 구조물(전봇대·가로등)**에서 추출한 소실점을 이용해
> 카메라 내부 파라미터(Intrinsics)를 자동으로 추정합니다.

---

## 📌 핵심 원리

도로 CCTV는 현장 접근이 어려워 체커보드 캘리브레이션이 불가능합니다.
본 도구는 아래 기하학적 제약만으로 초점거리를 추정합니다.

| 입력               | 추출 정보          | 역할                          |
| ------------------ | ------------------ | ----------------------------- |
| 차량 직진 궤적     | 수평 소실점 (VP_h) | 초점거리 추정 (1차 근거)      |
| 수직 구조물 (pole) | 수직 소실점 (VP_v) | **직교 소실점 쌍**으로 f 계산 |

**직교 소실점 공식:**
```
f² = -(VP_h - C) · (VP_v - C)
```
- `C`: 주점(이미지 중심으로 고정)
- 두 소실점이 없으면 제공된 카메라 FOV(D/H/V)로 폴백

### 추정 파라미터

| 파라미터   | 추정 방법                       | 상태     |
| ---------- | ------------------------------- | -------- |
| `fx`, `fy` | 직교 소실점 또는 FOV 기반       | ✅ 구현됨 |
| `cx`, `cy` | 이미지 중심으로 고정            | ✅        |
| `k1`, `k2` | 차량 궤적 직선성 최적화 (scipy) | ✅ 구현됨 |
| `p1`, `p2` | — (고정값 0)                    | —        |

---

## 🏗️ 프로젝트 구조

```
road-cctv-calib/
├── main.py                      # 진입점
├── requirements.txt
├── models/
│   └── sam3.pt                  # SAM3 세그멘테이션 모델
├── data/
│   └── result/                  # 캘리브레이션 결과 JSON 저장
└── src/
    ├── core/
    │   ├── calibrator.py        # 소실점 계산 + 초점거리·왜곡 추정
    │   └── trajectory_filter.py # 직진 궤적 필터링
    ├── detection/
    │   ├── vehicle_tracker.py   # RT-DETR + BoT-SORT 차량 추적
    │   ├── vehicle_detector.py  # RT-DETR 차량 검출
    │   └── pole_detector.py     # SAM3 기반 수직 구조물 검출
    └── ui/
        ├── main_window.py       # PySide6 메인 윈도우
        └── video_widget.py      # 영상 뷰어 및 시각화
```

---

## 🔧 모듈 설명

### 1. `TrajectoryFilter` — 직진 궤적 필터링

`VehicleTracker`가 수집한 궤적에서 **소실점 계산에 사용할 수 있는 직진 궤적**만 추출합니다.

| 파라미터        | 기본값 | 설명                         |
| --------------- | ------ | ---------------------------- |
| `min_points`    | 20     | 최소 좌표 포인트 수          |
| `min_distance`  | 100 px | 최소 이동 거리               |
| `min_r_squared` | 0.98   | 직선 피팅 최소 결정계수 (R²) |

- PCA 기반 직선 피팅으로 수직 궤적도 안정적으로 처리
- `FilteredTrajectory` 데이터클래스로 궤적 메타데이터 제공

### 2. `PoleDetector` — 수직 구조물 검출

SAM3 모델로 **전봇대·가로등·신호등 기둥** 등 수직 구조물을 세그멘테이션 후 직선 파라미터를 추출합니다.

| 파라미터          | 기본값 | 설명                    |
| ----------------- | ------ | ----------------------- |
| `conf_threshold`  | 0.3    | 검출 신뢰도 임계값      |
| `min_height`      | 50 px  | 최소 구조물 높이        |
| `angle_tolerance` | ±15°   | 90° 기준 허용 각도 오차 |

- 프롬프트: `pole`, `traffic light pole`, `street lamp`, `utility pole`
- IoU 기반 중복 제거 (`_remove_duplicates`)
- `VerticalStructure` 데이터클래스로 중심선·직선 파라미터 제공

### 3. `Calibrator` — 소실점 기반 캘리브레이션

```
직진 궤적  →  수평 소실점 (RANSAC)  ─┐
                                      ├→  초점거리 f 추정  →  왜곡계수 최적화
수직 구조물 →  수직 소실점 (RANSAC)  ─┘
```

**주요 메서드:**

| 메서드                    | 설명                              |
| ------------------------- | --------------------------------- |
| `compute_horizontal_vp()` | 차량 궤적 → 수평 소실점           |
| `compute_vertical_vp()`   | 수직 구조물 → 수직 소실점         |
| `estimate_focal_length()` | 직교 VP로 f 추정 (폴백: FOV 기반) |
| `calibrate()`             | 전체 파이프라인 실행              |

- RANSAC으로 아웃라이어에 강건한 소실점 추정
- 수직 소실점 없을 때 카메라 FOV(D/H/V) 입력으로 폴백
- `scipy.optimize`로 차량 궤적 직선성 최소화 → k1, k2 추정

### 4. `VehicleTracker` — 차량 검출 및 추적

- **RT-DETR-L** 모델로 차량 검출
- **BoT-SORT** 알고리즘으로 프레임 간 ID 유지 및 연속 궤적 추출
- 바운딩 박스 하단 중앙 좌표를 궤적으로 저장

### 5. UI (`MainWindow` / `VideoWidget`)

- **PySide6(Qt)** 기반 GUI
- 영상 로드 → 차량/Pole 실시간 추적 시각화 → 캘리브레이션 실행
- 카메라 FOV(D/H/V) 수동 입력 지원
- 결과를 JSON 파일로 저장 (`data/result/`)

---

## 📤 출력 형식

캘리브레이션 결과는 `data/result/calib_result_<timestamp>.json`에 저장됩니다.

```json
{
    "camera_id": "20260305_155655",
    "image_size": { "width": 1920, "height": 1080 },
    "intrinsic": {
        "fx": 1728.986,
        "fy": 1728.986,
        "cx": 960.0,
        "cy": 540.0
    },
    "distortion": {
        "k1": -0.05976,
        "k2": -0.02050,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0
    }
}
```

---

## 💻 설치 및 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 실행
python main.py
```

### 요구 사항

- Python 3.9+
- CUDA 지원 GPU 권장 (CPU로도 동작하나 속도 저하)
- SAM3 모델 파일: `models/sam3.pt`

### 의존 패키지

| 패키지                  | 용도               |
| ----------------------- | ------------------ |
| `ultralytics`           | RT-DETR, SAM3 추론 |
| `opencv-contrib-python` | 영상 처리          |
| `numpy`                 | 수치 연산          |
| `scipy`                 | 왜곡계수 최적화    |
| `PySide6`               | GUI                |
| `pyyaml`, `tqdm`        | 유틸리티           |

---

## 🗺️ 향후 계획

- **단일 소실점만 있을 때** 초점거리 추정 신뢰도 향상
- **칼만 필터** 기반 소실점 실시간 안정화
- **PTZ 카메라** 동적 캘리브레이션 (Zoom/Pan/Tilt 대응)
- p1, p2 (접선 왜곡) 추정

---

## 📚 참고 문헌

1. Dubská, M. et al. *"Fully Automatic Roadside Camera Calibration for Traffic Surveillance."* IEEE TITS, 2014. — https://ieeexplore.ieee.org/document/6909022/
2. *"Automatic Roadside Camera Calibration with Transformers."* Sensors, 2023. — https://pmc.ncbi.nlm.nih.gov/articles/PMC10708783/
3. *"Traffic Camera Calibration via Vehicle Vanishing Point Detection."* arXiv, 2021. — https://ar5iv.labs.arxiv.org/html/2103.11438