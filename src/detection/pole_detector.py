"""SAM3 기반 수직 구조물(pole) 검출 모듈"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class VerticalStructure:
    """수직 구조물 데이터 클래스"""
    mask: np.ndarray                      # 세그먼트 마스크
    center_line: List[Tuple[int, int]]    # 수직 중심선 좌표
    top_point: Tuple[int, int]            # 상단 점
    bottom_point: Tuple[int, int]         # 하단 점
    score: float                          # 신뢰도
    line_params: Tuple[float, float, float] = field(default=(0, 1, 0))  # ax + by + c = 0
    angle: float = 90.0                   # 수직 각도 (90도가 완벽한 수직)

    @property
    def height(self) -> float:
        """수직 구조물 높이 (픽셀)"""
        return abs(self.bottom_point[1] - self.top_point[1])


class PoleDetector:
    """SAM3 기반 수직 구조물 검출기"""

    # 수직 구조물 검출 텍스트 프롬프트
    PROMPTS = ['pole', 'traffic light pole', 'street lamp', 'utility pole']

    def __init__(
        self,
        model_path: str = 'models/sam3.pt',
        device: str = 'cuda',
        conf_threshold: float = 0.3,
        imgsz: int = 1024,
        min_height: int = 50,
        angle_tolerance: float = 15.0  # 허용 각도 오차: 90 ± 15도
    ):
        """
        Args:
            model_path: SAM3 모델 파일 경로
            device: 연산 디바이스 (cuda/cpu)
            conf_threshold: 신뢰도 임계값
            imgsz: 입력 이미지 크기
            min_height: 검출 최소 높이 (픽셀)
            angle_tolerance: 수직 각도 허용 오차 (도)
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.min_height = min_height
        self.angle_tolerance = angle_tolerance
        self.predictor = None

    def _init_predictor(self):
        """Predictor 초기화 (지연 로딩)"""
        if self.predictor is None:
            print(f"[PoleDetector] SAM3 Predictor 초기화 중... (모델: {self.model_path}, 디바이스: {self.device})")
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
            print(f"[PoleDetector] SAM3 모델 로드 완료: {self.model_path}")

    def detect(self, frame: np.ndarray) -> List[VerticalStructure]:
        """
        프레임에서 수직 구조물 검출

        Args:
            frame: BGR 이미지

        Returns:
            VerticalStructure 객체 리스트
        """
        self._init_predictor()

        h, w = frame.shape[:2]
        print(f"[PoleDetector] 검출 시작 - 프레임 크기: {w}x{h}")

        all_structures = []

        # 각 텍스트 프롬프트마다 검출 수행
        for prompt in self.PROMPTS:
            print(f"[PoleDetector] 프롬프트 '{prompt}' 로 SAM3 추론 중...")
            results = self.predictor(source=frame, text=[prompt], stream=False)

            if not results:
                print(f"[PoleDetector]   → '{prompt}': 결과 없음")
                continue

            result = results[0]

            if result.masks is None or len(result.masks) == 0:
                print(f"[PoleDetector]   → '{prompt}': 마스크 없음")
                continue

            masks = result.masks.data.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(masks))

            print(f"[PoleDetector]   → '{prompt}': 마스크 {len(masks)}개 검출됨 (신뢰도: {[f'{s:.2f}' for s in scores]})")

            valid_count = 0
            for mask, score in zip(masks, scores):
                # 마스크 크기가 다르면 원본 크기로 리사이즈
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

                # 수직 중심선 추출 및 수직 구조물 생성
                structure = self._mask_to_vertical_structure(mask, float(score))
                if structure is not None:
                    all_structures.append(structure)
                    valid_count += 1

            print(f"[PoleDetector]   → '{prompt}': 수직 구조물 {valid_count}개 유효")

        print(f"[PoleDetector] 전체 검출 후 중복 제거 전: {len(all_structures)}개")

        # IoU 기반 중복 제거
        structures = self._remove_duplicates(all_structures)
        print(f"[PoleDetector] 중복 제거 후: {len(structures)}개")

        # 신뢰도 내림차순 정렬
        structures.sort(key=lambda s: s.score, reverse=True)

        # 최종 결과 요약 출력
        for i, s in enumerate(structures):
            print(f"[PoleDetector]   pole #{i+1}: 상단={s.top_point}, 하단={s.bottom_point}, "
                  f"높이={s.height:.0f}px, 각도={s.angle:.1f}°, 신뢰도={s.score:.2f}")

        return structures

    def _mask_to_vertical_structure(
        self,
        mask: np.ndarray,
        score: float
    ) -> Optional[VerticalStructure]:
        """
        마스크에서 수직 구조물 정보 추출

        핵심 알고리즘:
        1. 각 y좌표에서 마스크 픽셀의 x 중심 계산
        2. 1차 다항식 피팅 (수직선이므로 거의 x=상수)
        3. 각도 검증 (75~105도)
        """
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 가장 큰 컨투어 선택
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 100:
            return None

        x, y, bw, bh = cv2.boundingRect(largest)

        # 최소 높이 검사
        if bh < self.min_height:
            return None

        # y축 방향으로 중심점 샘플링
        center_points = []
        pixel_interval = max(5, bh // 50)  # 적응적 샘플링 간격

        for yi in range(y, y + bh, pixel_interval):
            row = mask_uint8[yi, :]
            x_indices = np.where(row > 0)[0]

            if len(x_indices) >= 2:
                x_center = int(np.mean(x_indices))
                center_points.append([x_center, yi])

        if len(center_points) < 3:
            return None

        points = np.array(center_points)

        # 1차 다항식 피팅: x = a*y + b (수직선이므로 y를 독립변수로)
        try:
            coeffs = np.polyfit(points[:, 1], points[:, 0], 1)
            a_slope = coeffs[0]  # dx/dy

            # 수직 각도 계산 (y축 기준, 완벽한 수직 = 90도)
            angle = 90 - np.degrees(np.arctan(a_slope))

            # 수직 여부 검증 (75~105도 범위)
            if not self._validate_vertical(angle):
                return None

            # 스무딩된 중심선 생성
            y_coords = np.array([p[1] for p in center_points])
            poly = np.poly1d(coeffs)
            x_smoothed = poly(y_coords)

            center_line = [(int(x), int(y)) for x, y in zip(x_smoothed, y_coords)]

            # 상단/하단 끝점
            top_point = center_line[0]
            bottom_point = center_line[-1]

            # 직선 파라미터 계산 (ax + by + c = 0)
            line_params = self._fit_line_params(center_line)

            return VerticalStructure(
                mask=mask,
                center_line=center_line,
                top_point=top_point,
                bottom_point=bottom_point,
                score=score,
                line_params=line_params,
                angle=angle
            )

        except Exception as e:
            print(f"[PoleDetector] 마스크→구조물 변환 중 오류: {e}")
            return None

    def _validate_vertical(self, angle: float) -> bool:
        """
        수직 여부 검증

        Args:
            angle: 각도 (90도가 완벽한 수직)

        Returns:
            75~105도 범위 내이면 True
        """
        return abs(angle - 90) <= self.angle_tolerance

    def _fit_line_params(
        self,
        points: List[Tuple[int, int]]
    ) -> Tuple[float, float, float]:
        """
        포인트들로부터 직선 파라미터 계산 (ax + by + c = 0)

        Returns:
            (a, b, c) 정규화된 직선 파라미터
        """
        pts = np.array(points)

        if len(pts) < 2:
            return (1.0, 0.0, 0.0)

        # PCA로 직선 방향 계산
        centroid = pts.mean(axis=0)
        centered = pts - centroid

        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 법선 방향 = 가장 작은 고유값에 대응하는 벡터
        normal_idx = np.argmin(eigenvalues)
        normal = eigenvectors[:, normal_idx]

        a, b = normal
        c = -np.dot(normal, centroid)

        # 정규화
        norm = np.sqrt(a**2 + b**2)
        if norm > 1e-10:
            a, b, c = a/norm, b/norm, c/norm

        return (float(a), float(b), float(c))

    def _remove_duplicates(
        self,
        structures: List[VerticalStructure],
        iou_threshold: float = 0.5
    ) -> List[VerticalStructure]:
        """IoU 기반 중복 구조물 제거"""
        if len(structures) <= 1:
            return structures

        # 신뢰도 내림차순 정렬
        sorted_structures = sorted(structures, key=lambda s: s.score, reverse=True)
        kept = []

        for structure in sorted_structures:
            is_duplicate = False

            for kept_structure in kept:
                iou = self._compute_mask_iou(structure.mask, kept_structure.mask)
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(structure)

        return kept

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """두 마스크의 IoU 계산"""
        m1 = (mask1 > 0.5).astype(np.uint8)
        m2 = (mask2 > 0.5).astype(np.uint8)

        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def get_vertical_lines(
        self,
        structures: List[VerticalStructure]
    ) -> List[Tuple[float, float, float]]:
        """
        수직 구조물로부터 직선 파라미터 리스트 추출

        Args:
            structures: VerticalStructure 리스트

        Returns:
            직선 파라미터 리스트 [(a, b, c), ...]
        """
        return [s.line_params for s in structures]

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
            print("[PoleDetector] 리소스 해제 완료")
