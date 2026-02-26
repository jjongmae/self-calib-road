"""SAM3 기반 차선 검출 모듈"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class LaneSegment:
    """차선 세그먼트 데이터 클래스"""
    mask: np.ndarray           # 세그먼트 마스크
    polygon: List[List[int]]   # 폴리곤 좌표
    score: float               # 신뢰도
    centroid: Tuple[float, float] = field(default=(0, 0))  # 중심점
    angle: float = 0.0         # 주 방향 각도
    is_valid: bool = True      # 유효한 차선인지 (노이즈 필터링 결과)

    def __post_init__(self):
        """중심점과 각도 계산 및 노이즈 필터링"""
        if self.polygon and len(self.polygon) >= 2:
            points = np.array(self.polygon)
            self.centroid = (float(points[:, 0].mean()), float(points[:, 1].mean()))
            
            # Y 길이 계산 (너무 짧은 경우 노이즈)
            y_span = points[:, 1].max() - points[:, 1].min()
            if y_span < 20: 
                self.is_valid = False
                
            # PCA로 주 방향 계산
            if len(points) >= 2:
                centered = points - points.mean(axis=0)
                cov = np.cov(centered.T)
                if cov.shape == (2, 2):
                    eigenvalues, eigenvectors = np.linalg.eig(cov)
                    
                    # 방향성이 애매한지(blob 형태) 확인
                    max_eigen = np.max(eigenvalues)
                    min_eigen = np.min(eigenvalues)
                    if min_eigen > 0 and (max_eigen / min_eigen) < 3.0:
                        self.is_valid = False
                        
                    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
                    angle = np.degrees(np.arctan2(main_axis[1], main_axis[0]))
                    
                    # 수평선 필터링 (가로 방향 차선은 노이즈로 간주)
                    # angle 0도 주변이거나 180도 주변인 경우 (-30~30, 150~180, -150~-180)
                    abs_angle = abs(angle)
                    if abs_angle < 25 or abs_angle > 155:
                        self.is_valid = False
                        
                    self.angle = angle



@dataclass
class Lane:
    """병합된 차선 데이터 클래스"""
    segments: List[LaneSegment]  # 병합된 세그먼트들
    merged_mask: np.ndarray      # 병합된 마스크
    center_line: List[Tuple[int, int]]  # 중심선 좌표
    angle: float                 # 평균 각도
    direction_group: int = 0     # 방향성 그룹 ID

    @property
    def score(self) -> float:
        """평균 신뢰도"""
        if not self.segments:
            return 0.0
        return sum(s.score for s in self.segments) / len(self.segments)


class LaneDetector:
    """SAM3 기반 차선 검출기"""

    def __init__(
        self,
        model_path: str = 'models/sam3.pt',
        device: str = 'cuda',
        conf_threshold: float = 0.3,
        imgsz: int = 1024,
        prompt: str = 'lane line'
    ):
        """
        Args:
            model_path: SAM3 모델 파일 경로
            device: 디바이스 (cuda/cpu)
            conf_threshold: 신뢰도 임계값
            imgsz: 이미지 크기
            prompt: 검출 프롬프트
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.prompt = prompt
        self.predictor = None

        # 병합 파라미터
        self.angle_threshold = 20.0    # 각도 차이 임계값 (degree)
        self.distance_threshold = 100  # 거리 임계값 (pixel)

    def _init_predictor(self):
        """Predictor 초기화 (지연 로딩)"""
        if self.predictor is None:
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = dict(
                conf=self.conf_threshold,
                task="segment",
                mode="predict",
                model=self.model_path,
                device=self.device,
                save=False,
                imgsz=self.imgsz
            )
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            print(f"[LaneDetector] SAM3 모델 로드 완료: {self.model_path}")

    def detect(self, frame: np.ndarray) -> List[Lane]:
        """
        프레임에서 차선 검출

        Args:
            frame: BGR 이미지

        Returns:
            Lane 객체 리스트 (병합된 차선들)
        """
        self._init_predictor()

        # SAM3 추론
        results = self.predictor(source=frame, text=[self.prompt], stream=False)

        if not results:
            return []

        result = results[0]

        # 결과 없으면 빈 리스트
        if result.masks is None or len(result.masks) == 0:
            return []

        # 세그먼트 추출
        masks = result.masks.data.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(masks))

        h, w = frame.shape[:2]
        segments = []

        for mask, score in zip(masks, scores):
            # 마스크 리사이즈
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

            # 폴리곤 추출
            polygon = self._mask_to_polygon(mask)
            if polygon is None:
                continue

            segments.append(LaneSegment(
                mask=mask,
                polygon=polygon,
                score=float(score)
            ))

        if not segments:
            return []

        # 차선 병합
        lanes = self._merge_segments(segments, (h, w))

        return lanes

    def _mask_to_polygon(self, mask: np.ndarray, min_area: int = 100) -> Optional[List[List[int]]]:
        """마스크에서 중심 폴리라인 추출 (auto-map-matching 로직 참고)"""
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 가장 큰 컨투어 선택
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            return None

        x, y, w, h = cv2.boundingRect(largest)

        # 세로 방향으로 픽셀 간격 기준 샘플링 (auto-map-matching 기준)
        pixel_interval = 20
        min_width = 3
        max_x_delta = 50
        poly_degree = 1  # 2차 다항식은 곡리가 과도해져서(삑사리 현상) 1차(직선) 피팅으로 변경

        raw_points = []
        y_values = range(y, y + h, pixel_interval)

        for yi in y_values:
            row = mask_uint8[yi, :]
            x_indices = np.where(row > 0)[0]

            # 마스크 두께 필터링
            if len(x_indices) >= min_width:
                x_center = int(np.mean(x_indices))
                raw_points.append([x_center, int(yi)])

        # 이상치 제거
        points = self._filter_outliers(raw_points, max_x_delta)

        # 다항식 피팅으로 스무딩
        if poly_degree > 0 and len(points) >= poly_degree + 1:
            points = self._smooth_with_polynomial(points, poly_degree)

        if len(points) < 2:
            return None

        return points

    def _filter_outliers(self, points: List[List[int]], max_x_delta: int) -> List[List[int]]:
        """연속된 포인트 간 x 변화가 급격한 이상치 제거."""
        if len(points) < 2:
            return points

        filtered = [points[0]]
        for i in range(1, len(points)):
            prev_x = filtered[-1][0]
            curr_x = points[i][0]

            # 이전 포인트와 x 차이가 허용 범위 내인 경우만 추가
            if abs(curr_x - prev_x) <= max_x_delta:
                filtered.append(points[i])

        return filtered

    def _smooth_with_polynomial(
        self,
        points: List[List[int]],
        degree: int = 2
    ) -> List[List[int]]:
        """다항식 피팅으로 포인트를 스무딩."""
        if len(points) < degree + 1:
            return points

        y_coords = np.array([p[1] for p in points])
        x_coords = np.array([p[0] for p in points])

        try:
            coeffs = np.polyfit(y_coords, x_coords, degree)
            poly = np.poly1d(coeffs)
            smoothed_x = poly(y_coords)
            
            smoothed_points = [
                [int(round(x)), int(y)]
                for x, y in zip(smoothed_x, y_coords)
            ]
            
            return smoothed_points
        except Exception:
            return points

    def _merge_segments(self, segments: List[LaneSegment], img_shape: Tuple[int, int]) -> List[Lane]:
        """
        세그먼트들을 차선으로 병합
        비슷한 각도와 가까운 위치에 있는 세그먼트들을 하나의 차선으로 병합하고, 방향성 그룹 할당
        """
        valid_segments = [s for s in segments if s.is_valid]
        if not valid_segments:
            return []

        h, w = img_shape
        used = [False] * len(valid_segments)
        lanes = []

        for i, seg in enumerate(valid_segments):
            if used[i]:
                continue

            # 새 차선 그룹 시작
            group = [seg]
            used[i] = True

            # 병합할 수 있는 세그먼트 찾기
            for j, other in enumerate(valid_segments):
                if used[j]:
                    continue

                # 각도 차이 확인
                angle_diff = abs(seg.angle - other.angle)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff

                if angle_diff > self.angle_threshold:
                    continue

                # 거리 확인 (중심점 간 단순 거리 대신, 측면 거리와 종방향 거리 분리)
                dx = other.centroid[0] - seg.centroid[0]
                dy = other.centroid[1] - seg.centroid[1]
                
                angle_rad = np.radians(seg.angle)
                dir_x = np.cos(angle_rad)
                dir_y = np.sin(angle_rad)
                
                # 측면 수직 거리 (lateral distance)
                lateral_dist = abs(dx * dir_y - dy * dir_x)
                # 단순 유클리디안 거리
                dist = np.sqrt(dx**2 + dy**2)

                # 같은 차선의 점선이거나 끊어진 부분은 종방향 거리가 멀어도 (dist < 400), 측면 거리가 가까우면(lateral < 30) 병합
                if lateral_dist < 30 and dist < 400:
                    group.append(other)
                    used[j] = True

            # 병합된 마스크 생성
            merged_mask = np.zeros((h, w), dtype=np.float32)
            for s in group:
                merged_mask = np.maximum(merged_mask, s.mask)

            # 중심선 추출
            center_line = self._extract_center_line(merged_mask)

            # 평균 각도
            avg_angle = np.mean([s.angle for s in group])

            lanes.append(Lane(
                segments=group,
                merged_mask=merged_mask,
                center_line=center_line,
                angle=avg_angle
            ))

        # 방향성 그룹 할당 (방향성이 비슷한 차선끼리 묶어 줌)
        if lanes:
            angles = np.array([l.angle for l in lanes])
            # 180도를 0도로 맞춰 계산 (-180~180 -> 0~180 변환 후 그룹화)
            norm_angles = np.where(angles < 0, angles + 180, angles)
            
            group_id = 0
            assigned = [False] * len(lanes)
            
            for i in range(len(lanes)):
                if assigned[i]:
                    continue
                lanes[i].direction_group = group_id
                assigned[i] = True
                
                # 같은 각도를 가지는 차선 찾기 (15도 이내 등)
                for j in range(i + 1, len(lanes)):
                    if assigned[j]:
                        continue
                    diff = abs(norm_angles[i] - norm_angles[j])
                    if diff > 90:
                        diff = 180 - diff
                        
                    if diff < 15.0:  # 비슷한 방향성을 가진다면 같은 그룹에 배정
                        lanes[j].direction_group = group_id
                        assigned[j] = True
                group_id += 1

        # 신뢰도 순 정렬
        lanes.sort(key=lambda l: l.score, reverse=True)

        return lanes

    def _extract_center_line(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """마스크에서 중심선 추출 (poly line 방식 적용)"""
        polygon = self._mask_to_polygon(mask, min_area=50)
        if polygon is None:
            return []
            
        # (x, y) 형식의 튜플 리스트로 변환
        return [(int(p[0]), int(p[1])) for p in polygon]

    def detect_raw_segments(self, frame: np.ndarray) -> List[LaneSegment]:
        """
        병합 없이 원본 세그먼트만 반환 (디버그용)
        """
        self._init_predictor()

        results = self.predictor(source=frame, text=[self.prompt], stream=False)

        if not results:
            return []

        result = results[0]

        if result.masks is None or len(result.masks) == 0:
            return []

        masks = result.masks.data.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(masks))

        h, w = frame.shape[:2]
        segments = []

        for mask, score in zip(masks, scores):
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

            polygon = self._mask_to_polygon(mask)
            if polygon is None:
                continue

            segments.append(LaneSegment(
                mask=mask,
                polygon=polygon,
                score=float(score)
            ))

        return segments

    def release(self):
        """리소스 해제"""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None

            import gc
            import torch
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
