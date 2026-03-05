"""직진 궤적 필터링 모듈"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class FilteredTrajectory:
    """필터링된 직진 궤적"""
    track_id: int
    points: List[Tuple[float, float]]
    line_params: Tuple[float, float, float]  # ax + by + c = 0 (정규화됨)
    direction_angle: float  # 이동 방향 각도 (도)
    r_squared: float  # 직선 피팅 결정계수

    @property
    def length(self) -> float:
        """궤적 길이 (픽셀)"""
        if len(self.points) < 2:
            return 0.0
        pts = np.array(self.points)
        return float(np.sqrt(np.sum((pts[-1] - pts[0]) ** 2)))


class TrajectoryFilter:
    """차량 궤적 필터링 - 직진 궤적만 추출"""

    def __init__(
        self,
        min_points: int = 20,
        min_distance: float = 100.0,
        min_r_squared: float = 0.98
    ):
        """
        Args:
            min_points: 최소 포인트 개수
            min_distance: 최소 이동 거리 (픽셀)
            min_r_squared: 직선 피팅 최소 결정계수
        """
        self.min_points = min_points
        self.min_distance = min_distance
        self.min_r_squared = min_r_squared

    def filter_trajectories(
        self,
        trajectories: Dict[int, List[Tuple[float, float]]]
    ) -> List[FilteredTrajectory]:
        """
        직진 궤적 필터링

        Args:
            trajectories: {track_id: [(x, y), ...]} 형태의 궤적 데이터

        Returns:
            필터링된 직진 궤적 리스트
        """
        filtered = []

        for track_id, points in trajectories.items():
            result = self._process_trajectory(track_id, points)
            if result is not None:
                filtered.append(result)

        return filtered

    def _process_trajectory(
        self,
        track_id: int,
        points: List[Tuple[float, float]]
    ) -> Optional[FilteredTrajectory]:
        """
        단일 궤적 처리

        Returns:
            직진 조건을 만족하면 FilteredTrajectory, 아니면 None
        """
        # 최소 포인트 수 확인
        if len(points) < self.min_points:
            return None

        pts = np.array(points)

        # 최소 이동 거리 확인
        distance = np.sqrt(np.sum((pts[-1] - pts[0]) ** 2))
        if distance < self.min_distance:
            return None

        # 직선 피팅 및 R² 계산
        r_squared, line_params = self._fit_line(pts)
        if r_squared < self.min_r_squared:
            return None

        # 이동 방향 계산
        direction = pts[-1] - pts[0]
        angle = np.degrees(np.arctan2(direction[1], direction[0]))

        return FilteredTrajectory(
            track_id=track_id,
            points=[(float(p[0]), float(p[1])) for p in pts],
            line_params=line_params,
            direction_angle=float(angle),
            r_squared=float(r_squared)
        )

    def _fit_line(
        self,
        points: np.ndarray
    ) -> Tuple[float, Tuple[float, float, float]]:
        """
        직선 피팅 (최소제곱법)

        수직에 가까운 선도 처리하기 위해 주성분 분석(PCA) 사용

        Args:
            points: (N, 2) 형태의 좌표 배열

        Returns:
            (R², (a, b, c)) - 결정계수와 직선 파라미터 (ax + by + c = 0, 정규화됨)
        """
        # 중심점
        centroid = points.mean(axis=0)
        centered = points - centroid

        # PCA로 주 방향 계산
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 가장 큰 고유값에 해당하는 방향이 직선 방향
        main_idx = np.argmax(eigenvalues)
        main_direction = eigenvectors[:, main_idx]

        # 법선 방향 (직선에 수직)
        normal = eigenvectors[:, 1 - main_idx]

        # 직선 방정식: normal · (p - centroid) = 0
        # ax + by + c = 0 형태로 변환
        a, b = normal
        c = -np.dot(normal, centroid)

        # 정규화
        norm = np.sqrt(a**2 + b**2)
        a, b, c = a/norm, b/norm, c/norm

        # R² 계산
        # 총 분산 대비 주축 방향 분산의 비율
        total_variance = eigenvalues.sum()
        if total_variance < 1e-10:
            return 0.0, (a, b, c)

        r_squared = eigenvalues[main_idx] / total_variance

        return r_squared, (float(a), float(b), float(c))

    @staticmethod
    def line_intersection(
        line1: Tuple[float, float, float],
        line2: Tuple[float, float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        두 직선의 교차점 계산

        Args:
            line1, line2: (a, b, c) 형태의 직선 파라미터 (ax + by + c = 0)

        Returns:
            (x, y) 교차점 또는 평행이면 None
        """
        a1, b1, c1 = line1
        a2, b2, c2 = line2

        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            return None  # 평행

        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det

        return (x, y)
