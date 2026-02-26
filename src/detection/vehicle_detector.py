"""차량 검출 모듈"""

import numpy as np
from ultralytics import RTDETR
from dataclasses import dataclass


@dataclass
class Detection:
    """검출 결과 데이터 클래스"""
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    track_id: int = -1  # 추적 ID (-1: 미추적)

    @property
    def center(self) -> tuple:
        """바운딩 박스 중심점"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bottom_center(self) -> tuple:
        """바운딩 박스 하단 중심점 (차량 바닥 추정)"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def is_tracked(self) -> bool:
        """추적 중인지 여부"""
        return self.track_id >= 0


class VehicleDetector:
    """RT-DETR 기반 차량 검출기"""

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

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        프레임에서 차량 검출

        Args:
            frame: BGR 이미지 (numpy array)

        Returns:
            Detection 객체 리스트
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
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

                detections.append(Detection(
                    bbox=tuple(bbox),
                    confidence=conf,
                    class_id=class_id,
                    class_name=self.VEHICLE_CLASSES[class_id]
                ))

        return detections

    def set_confidence(self, threshold: float):
        """신뢰도 임계값 설정"""
        self.conf_threshold = max(0.0, min(1.0, threshold))
