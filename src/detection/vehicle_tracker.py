"""차량 추적 모듈"""

import numpy as np
from collections import defaultdict
from ultralytics import RTDETR
from src.detection.vehicle_detector import Detection


class VehicleTracker:
    """RT-DETR + BoT-SORT 기반 차량 추적기"""

    # 차량 관련 COCO 클래스 ID
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
    }

    def __init__(self, model_path: str = 'models/rtdetr-l.pt', conf_threshold: float = 0.5):
        """
        Args:
            model_path: RT-DETR 모델 파일 경로
            conf_threshold: 검출 신뢰도 임계값
        """
        self.model = RTDETR(model_path)
        self.conf_threshold = conf_threshold

        # 차량별 궤적 저장 {track_id: [(frame_idx, x, y), ...]}
        self.trajectories = defaultdict(list)
        self.frame_idx = 0

    def track(self, frame: np.ndarray) -> list[Detection]:
        """
        프레임에서 차량 검출 및 추적

        Args:
            frame: BGR 이미지 (numpy array)

        Returns:
            Detection 객체 리스트 (track_id 포함)
        """
        # BoT-SORT 추적기 사용, persist=True로 이전 프레임 정보 유지
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            persist=True,
            tracker='botsort.yaml',
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                # 차량 클래스만 필터링
                if class_id not in self.VEHICLE_CLASSES:
                    continue

                bbox = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].item()

                # 추적 ID (없으면 -1)
                track_id = -1
                if boxes.id is not None:
                    track_id = int(boxes.id[i].item())

                det = Detection(
                    bbox=tuple(bbox),
                    confidence=conf,
                    class_id=class_id,
                    class_name=self.VEHICLE_CLASSES[class_id],
                    track_id=track_id
                )
                detections.append(det)

                # 궤적 저장 (바닥 중심점)
                if track_id >= 0:
                    bx, by = det.bottom_center
                    self.trajectories[track_id].append((self.frame_idx, float(bx), float(by)))

        self.frame_idx += 1
        return detections

    def get_trajectory(self, track_id: int) -> list[tuple]:
        """
        특정 차량의 궤적 반환

        Args:
            track_id: 추적 ID

        Returns:
            [(frame_idx, x, y), ...] 형태의 궤적
        """
        return self.trajectories.get(track_id, [])

    def get_all_trajectories(self) -> dict:
        """모든 차량 궤적 반환"""
        return dict(self.trajectories)

    def get_recent_points(self, track_id: int, n: int = 30) -> list[tuple]:
        """
        특정 차량의 최근 n개 포인트 반환 (궤적 시각화용)

        Args:
            track_id: 추적 ID
            n: 반환할 포인트 수

        Returns:
            [(x, y), ...] 형태의 좌표 리스트
        """
        traj = self.trajectories.get(track_id, [])
        recent = traj[-n:] if len(traj) > n else traj
        return [(x, y) for _, x, y in recent]

    def reset(self):
        """추적 상태 초기화"""
        self.trajectories.clear()
        self.frame_idx = 0
        # 모델의 추적 상태도 초기화
        self.model.predictor = None

    def set_confidence(self, threshold: float):
        """신뢰도 임계값 설정"""
        self.conf_threshold = max(0.0, min(1.0, threshold))
