"""소실점 계산 및 카메라 캘리브레이션 모듈"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import cv2
import scipy.optimize

from src.core.trajectory_filter import FilteredTrajectory, TrajectoryFilter


@dataclass
class VanishingPoint:
    """소실점"""
    x: float
    y: float
    confidence: float  # 신뢰도 (0~1)
    inlier_count: int = 0  # 인라이어 개수

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class CalibrationResult:
    """캘리브레이션 결과"""
    focal_length: float  # 추정 초점거리 (픽셀)
    horizontal_vp: VanishingPoint  # 수평 소실점 (차량 직진 방향)
    vertical_vp: VanishingPoint  # 수직 소실점 (pole 방향)
    principal_point: Tuple[float, float]  # 주점 (이미지 중심으로 가정)
    image_size: Tuple[int, int]  # (width, height)
    distortion_coeffs: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # (k1, k2, p1, p2)

    @property
    def is_valid(self) -> bool:
        """유효한 캘리브레이션인지 확인"""
        # 초점거리가 양수이고 합리적인 범위인지
        return 100 < self.focal_length < 10000


class Calibrator:
    """
    소실점 기반 카메라 캘리브레이션

    핵심 원리:
    - 차량 직진 궤적 → 수평 소실점 (VP_horizontal)
    - 수직 구조물 (pole) → 수직 소실점 (VP_vertical)
    - 직교 소실점으로 초점거리 계산: f² = -VP_h · VP_v (주점 중심 좌표계)
    """

    def __init__(
        self,
        ransac_iterations: int = 1000,
        ransac_threshold: float = 5.0,
        min_lines: int = 3
    ):
        """
        Args:
            ransac_iterations: RANSAC 반복 횟수
            ransac_threshold: RANSAC 인라이어 판정 임계값 (픽셀)
            min_lines: 소실점 계산 최소 직선 개수
        """
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold
        self.min_lines = min_lines

    def compute_horizontal_vp(
        self,
        trajectories: List[FilteredTrajectory]
    ) -> Optional[VanishingPoint]:
        """
        차량 직진 궤적으로부터 수평 소실점 계산

        Args:
            trajectories: 필터링된 직진 궤적 리스트

        Returns:
            수평 소실점 또는 None
        """
        if len(trajectories) < self.min_lines:
            return None

        # 직선 파라미터 추출
        lines = [t.line_params for t in trajectories]

        return self._ransac_vp(lines)

    def compute_vertical_vp(
        self,
        vertical_lines: List[Tuple[float, float, float]],
        image_size: Tuple[int, int] = None
    ) -> Optional[VanishingPoint]:
        """
        수직 구조물(pole)로부터 수직 소실점 계산

        물리적 제약:
        - 도로 CCTV는 아래를 향하므로 수직 소실점은 이미지 위쪽에 위치해야 함
        - 즉, VP의 y좌표는 이미지 상단보다 위 (y < 0) 이어야 정상

        Args:
            vertical_lines: 수직선 파라미터 리스트 [(a, b, c), ...]
            image_size: (width, height) - 방향 검증용 (없으면 skip)

        Returns:
            수직 소실점 또는 None
        """
        if len(vertical_lines) < self.min_lines:
            print(f"[Calibrator] 수직 소실점 계산 불가: 수직선 {len(vertical_lines)}개 (최소 {self.min_lines}개 필요)")
            return None

        vp = self._ransac_vp(vertical_lines)
        if vp is None:
            return None

        # 물리적 방향 검증: 수직 소실점은 이미지 위쪽에 있어야 함
        if image_size is not None:
            _, h = image_size
            if vp.y > 0:
                print(f"[Calibrator] ⚠️  수직 VP 방향 이상! y={vp.y:.1f} (이미지 아래쪽)")
                print(f"[Calibrator]    정상 범위: y < 0 (이미지 위쪽, 하늘 방향)")
                print(f"[Calibrator]    원인 추정: pole 검출 오류 (차선/가드레일이 pole로 오검출)")
                print(f"[Calibrator]    → 수직 소실점 무효 처리")
                return None
            elif vp.y > -100:
                print(f"[Calibrator] ⚠️  수직 VP가 이미지 경계 근처: y={vp.y:.1f} (불안정할 수 있음)")

        return vp

    def estimate_focal_length(
        self,
        h_vp: VanishingPoint,
        v_vp: Optional[VanishingPoint],
        image_size: Tuple[int, int],
        dfov: float = 0.0,
        hfov: float = 0.0,
        vfov: float = 0.0
    ) -> Optional[CalibrationResult]:
        """
        초점거리 추정 파이프라인

        - 1지망: 두 소실점(수평, 수직)의 직교 성질을 이용 (f² = -(vh-c)·(vv-c))
        - 2지망: 수직 소실점이 없거나 실패하면 단일 소실점 기반 휴리스틱 적용

        Args:
            h_vp: 수평 소실점
            v_vp: 수직 소실점 (없을 경우 None 허용)
            image_size: (width, height)

        Returns:
            캘리브레이션 결과 또는 None
        """
        w, h = image_size
        cx, cy = w / 2, h / 2  # 주점 (이미지 중심으로 가정)
        
        focal_length = -1.0
        used_v_vp = v_vp

        # 1. 두 소실점이 모두 주어진 경우 직교 조건 테스트
        if v_vp is not None:
            vh = np.array([h_vp.x - cx, h_vp.y - cy])
            vv = np.array([v_vp.x - cx, v_vp.y - cy])

            print(f"[Calibrator] === 두 소실점 기반 직교 추정 시도 ===")
            print(f"[Calibrator] 이미지 크기: {w}x{h}, 주점: ({cx}, {cy})")
            print(f"[Calibrator] 수평 소실점 (원본): ({h_vp.x:.1f}, {h_vp.y:.1f})")
            print(f"[Calibrator] 수직 소실점 (원본): ({v_vp.x:.1f}, {v_vp.y:.1f})")

            dot_product = np.dot(vh, vv)
            f_squared = -dot_product
            
            if f_squared > 0:
                focal_length = np.sqrt(f_squared)
                print(f"[Calibrator] ✅ 직교 조건 성공! f = √{f_squared:.2f} = {focal_length:.2f} px")
            else:
                print(f"[Calibrator] ⚠️ 직교 조건 실패! f²={f_squared:.2f} (음수). 수직 소실점 기하학적 오류.")
                used_v_vp = None  # 직교 실패 시 수직 VP 무효화
        else:
            print(f"[Calibrator] 수직 소실점(v_vp) 정보가 없어 단일 소실점 추정으로 우회합니다.")

        # 2. 직교 조건이 실패했거나 수직 소실점이 애초에 없는 경우 (Fallback)
        if used_v_vp is None:
            focal_length = self._estimate_f_fallback(h_vp, image_size, dfov, hfov, vfov)
            # 가상의 수직 소실점 생성 (기하학적 일관성 유지)
            # 수직 소실점은 이미지 수직축(y축)을 따라 위쪽에 존재한다고 가정
            # h_vp.y를 기준으로 보편적 틸트를 반영하여 추정
            estimated_vp_y = cy - (focal_length ** 2) / max(abs(h_vp.y - cy), 1.0)
            if h_vp.y - cy > 0:
                estimated_vp_y = cy - (focal_length ** 2) / (h_vp.y - cy)
            
            used_v_vp = VanishingPoint(x=cx, y=float(estimated_vp_y), confidence=0.0, inlier_count=0)
            print(f"[Calibrator] 가상 수직 소실점(가조정) 생성: (x={used_v_vp.x:.1f}, y={used_v_vp.y:.1f})")

        if focal_length <= 0:
            return None

        # 화각 참고값 출력
        import math
        hfov = 2 * math.degrees(math.atan(cx / focal_length))
        vfov = 2 * math.degrees(math.atan(cy / focal_length))
        print(f"[Calibrator] 참고 화각: 수평={hfov:.1f}°, 수직={vfov:.1f}°")

        return CalibrationResult(
            focal_length=float(focal_length),
            horizontal_vp=h_vp,
            vertical_vp=used_v_vp,
            principal_point=(cx, cy),
            image_size=image_size,
            distortion_coeffs=(0.0, 0.0, 0.0, 0.0)
        )

    def _estimate_f_fallback(
        self,
        h_vp: VanishingPoint,
        image_size: Tuple[int, int],
        dfov: float = 0.0,
        hfov: float = 0.0,
        vfov: float = 0.0
    ) -> float:
        """
        단일 주행 궤적(수평 소실점)과 제공된 카메라 화각(D/H/V)들을 이용한 초점거리 추정
        """
        w, h = image_size
        cx, cy = w / 2, h / 2

        print(f"[Calibrator] --- 카메라 화각(FOV) 바탕 초점거리 수학적 계산 ---")
        
        # 주점에서 소실점까지 Y축으로 얼마나 내려왔는지가 카메라의 틸트(Tilt, 숙인 각도)에 비례
        vp_dy = abs(h_vp.y - cy)
        
        print(f"[Calibrator] 지평선 낙하 거리(Tilt 크기 지표): {vp_dy:.1f} px")

        import math
        f_estimates = []
        
        if dfov > 0:
            dfov_rad = math.radians(dfov)
            d = math.sqrt(w**2 + h**2)
            f_d = (d / 2) / math.tan(dfov_rad / 2)
            f_estimates.append(f_d)
            print(f"[Calibrator] ✅ DFOV={dfov}° → f ≈ {f_d:.1f} px")

        if hfov > 0:
            hfov_rad = math.radians(hfov)
            f_h = (w / 2) / math.tan(hfov_rad / 2)
            f_estimates.append(f_h)
            print(f"[Calibrator] ✅ HFOV={hfov}° → f ≈ {f_h:.1f} px")
            
        if vfov > 0:
            vfov_rad = math.radians(vfov)
            f_v = (h / 2) / math.tan(vfov_rad / 2)
            f_estimates.append(f_v)
            print(f"[Calibrator] ✅ VFOV={vfov}° → f ≈ {f_v:.1f} px")

        if not f_estimates:
            print("[Calibrator] 입력된 화각이 없어, CCTV 기본 화각 58° 가정 사용")
            focal_length = (w / 2) / math.tan(math.radians(58.0) / 2)
        else:
            focal_length = sum(f_estimates) / len(f_estimates)
            if len(f_estimates) > 1:
                print(f"[Calibrator] 최종 결정 초점거리(평균): f ≈ {focal_length:.1f} px")

        return float(focal_length)

    def _ransac_vp(
        self,
        lines: List[Tuple[float, float, float]]
    ) -> Optional[VanishingPoint]:
        """
        RANSAC 기반 소실점 추정

        Args:
            lines: 직선 파라미터 리스트 [(a, b, c), ...]

        Returns:
            소실점 또는 None
        """
        if len(lines) < 2:
            return None

        best_vp = None
        best_inliers = 0
        best_score = float('inf')

        # 모든 직선 쌍의 교차점을 후보로 사용
        candidates = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                point = TrajectoryFilter.line_intersection(lines[i], lines[j])
                if point is not None:
                    candidates.append(point)

        if not candidates:
            return None

        # RANSAC
        for _ in range(min(self.ransac_iterations, len(candidates) * 10)):
            # 무작위 교차점 선택
            vp = random.choice(candidates)

            # 인라이어 카운트 (각 직선까지의 거리가 임계값 이하)
            inliers = 0
            total_dist = 0

            for a, b, c in lines:
                # 점과 직선 사이 거리
                dist = abs(a * vp[0] + b * vp[1] + c)
                if dist < self.ransac_threshold:
                    inliers += 1
                total_dist += dist

            # 인라이어가 많고 평균 거리가 작을수록 좋음
            if inliers > best_inliers or (inliers == best_inliers and total_dist < best_score):
                best_inliers = inliers
                best_score = total_dist
                best_vp = vp

        if best_vp is None or best_inliers < self.min_lines:
            return None

        # 신뢰도 계산
        confidence = best_inliers / len(lines)

        return VanishingPoint(
            x=float(best_vp[0]),
            y=float(best_vp[1]),
            confidence=float(confidence),
            inlier_count=best_inliers
        )

    def calibrate(
        self,
        trajectories: List[FilteredTrajectory],
        vertical_lines: List[Tuple[float, float, float]],
        image_size: Tuple[int, int],
        dfov: float = 0.0,
        hfov: float = 0.0,
        vfov: float = 0.0
    ) -> Optional[CalibrationResult]:
        """
        전체 캘리브레이션 파이프라인

        Args:
            trajectories: 필터링된 직진 궤적
            vertical_lines: 수직선 파라미터
            image_size: (width, height)

        Returns:
            캘리브레이션 결과 또는 None
        """
        print(f"[Calibrator] === calibrate() 시작 ===")
        print(f"[Calibrator] 입력 - 직진 궤적: {len(trajectories)}개, 수직선: {len(vertical_lines)}개, 이미지: {image_size}")

        # 각 수직선 방향 진단 출력
        print(f"[Calibrator] --- 수직선(pole) 방향 진단 ---")
        for i, (a, b, c) in enumerate(vertical_lines):
            # ax + by + c = 0 에서 직선의 방향벡터: (-b, a)
            # 수직선이므로 |a|이 크면 거의 수직
            import math
            angle_from_vertical = math.degrees(math.atan2(abs(b), abs(a))) if abs(a) > 1e-6 else 90.0
            print(f"[Calibrator]   수직선 #{i+1}: a={a:.4f}, b={b:.4f}, c={c:.4f} → 수직 편차={angle_from_vertical:.1f}°")
        print(f"[Calibrator] ---------------------------------")

        # 수평 소실점 계산
        h_vp = self.compute_horizontal_vp(trajectories)
        if h_vp is None:
            print("[Calibrator] 수평 소실점 계산 실패: 직진 궤적 부족")
            return None
        print(f"[Calibrator] 수평 소실점: ({h_vp.x:.1f}, {h_vp.y:.1f}), 신뢰도={h_vp.confidence:.0%}, 인라이어={h_vp.inlier_count}개")

        # 수직 소실점 계산 (방향 검증 포함)
        # 이제 v_vp 계산이 실패해도 캘리브레이션을 중단하지 않습니다
        v_vp = self.compute_vertical_vp(vertical_lines, image_size)
        if v_vp is None:
            print("[Calibrator] 수직 소실점이 없습니다 (하지만 캘리브레이션은 계속 진행됩니다)")
        else:
            print(f"[Calibrator] 수직 소실점: ({v_vp.x:.1f}, {v_vp.y:.1f}), 신뢰도={v_vp.confidence:.0%}, 인라이어={v_vp.inlier_count}개")

        # 초점거리 추정
        result = self.estimate_focal_length(h_vp, v_vp, image_size, dfov, hfov, vfov)
        if result is None:
            print("[Calibrator] 초점거리 추정 실패")
            return None

        # k1, k2 왜곡 최적화
        cx, cy = image_size[0] / 2, image_size[1] / 2
        k1, k2 = self._optimize_distortion(trajectories, result.focal_length, cx, cy)
        
        # 튜플 재할당으로 distortion 업데이트
        result.distortion_coeffs = (k1, k2, 0.0, 0.0)

        print(f"[Calibrator] === 캘리브레이션 완료 ===")
        print(f"[Calibrator]   초점거리: {result.focal_length:.1f} px")
        print(f"[Calibrator]   수평 소실점: ({result.horizontal_vp.x:.1f}, {result.horizontal_vp.y:.1f})")
        print(f"[Calibrator]   수직 소실점: ({result.vertical_vp.x:.1f}, {result.vertical_vp.y:.1f})")
        print(f"[Calibrator]   왜곡 계수(k1,k2): {k1:.5f}, {k2:.5f}")

        return result

    def _optimize_distortion(self, trajectories: List[FilteredTrajectory], f: float, cx: float, cy: float) -> Tuple[float, float]:
        """차량 궤적이 최대한 직선이 되도록 k1, k2를 최적화"""
        print("[Calibrator] --- 왜곡 계수(k1, k2) 최적화 시작 ---")
        
        # 길이가 5포인트 이상인 궤적만 추출하여 최적화에 사용
        valid_trajs = [np.array(t.points, dtype=np.float32).reshape(-1, 1, 2) for t in trajectories if len(t.points) >= 5]
        if not valid_trajs:
            print("[Calibrator] ⚠️ 왜곡 보정용 궤적 데이터 부족(5포인트 이상). 왜곡 추정 생략.")
            return 0.0, 0.0

        camera_matrix = np.array([[f, 0, cx],
                                  [0, f, cy],
                                  [0, 0, 1]], dtype=np.float32)

        def objective(params):
            k1, k2 = params
            dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float32)
            total_error = 0.0
            
            for pts in valid_trajs:
                try:
                    # undistortPoints는 Normalized Coordinates를 반환, 이를 바탕으로 다시 영상 좌표계를 만들고 싶다면 P 인자 전달
                    undist = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
                    undist = undist.reshape(-1, 2)
                except Exception:
                    return 1e9

                # 직선 핏 에러 계산 (점들의 공분산 행렬에서 가장 작은 고윳값)
                # 고윳값이 작다는 것은 분산의 최소 방향(즉 직선과 직교하는 방향)으로의 퍼짐이 적어 완벽한 선에 가깝다는 의미
                mean = np.mean(undist, axis=0)
                centered = undist - mean
                cov = np.dot(centered.T, centered)  # 2x2 매트릭스
                
                # 2x2 행렬 최소 고윳값
                A = cov[0, 0]
                B = cov[0, 1]
                C = cov[1, 1]
                trace = A + C
                det = A * C - B * B
                desc = trace**2 - 4 * det
                if desc < 0: desc = 0
                max_eig = (trace + np.sqrt(desc)) / 2
                min_eig = (trace - np.sqrt(desc)) / 2
                
                # 영상 스케일 축소에 따른 '가짜 정답(꼼수)' 방지: 고윳값 비율로 직선도(Straightness) 평가
                if max_eig > 1e-5:
                    total_error += min_eig / max_eig
                else:
                    total_error += 1.0  # 의미 없는 궤적 페널티
                
            # 과적합 방지를 위한 정규화 (L2)
            regularization = 0.5 * (k1**2 + k2**2)
            return total_error + regularization

        print(f"[Calibrator] {len(valid_trajs)}개 궤적으로 K1, K2 찾기 진행 중 (Differential Evolution)...")
        # Differential Evolution은 노이즈가 많거나 국소 최적해(Local Minima)가 많은 비선형 문제에서 
        # 전역 최적해(Global Minimum)를 찾는 데 매우 강력한 방법입니다.
        result = scipy.optimize.differential_evolution(
            func=objective,
            bounds=[(-0.5, 0.5), (-0.5, 0.5)],  # k1, k2 탐색 공간 (-0.5 ~ 0.5 범위로 충분)
            strategy='best1bin',
            maxiter=50,      # DE는 세대(generation) 반복이므로 과도하게 높일 필요 없음
            popsize=10,      # 개체군 크기
            tol=1e-4,
            disp=False,
            polish=True      # 전역 탐색 후 수렴점 근처에서 국소 최적화(L-BFGS-B)를 한 번 더 실행하여 정밀도 향상
        )
        
        k1, k2 = result.x
        print(f"[Calibrator] ✅ 왜곡 추정 완료 (DE): k1={k1:.5f}, k2={k2:.5f} (직선 잔차 합: {result.fun:.2f})")
        return float(k1), float(k2)
