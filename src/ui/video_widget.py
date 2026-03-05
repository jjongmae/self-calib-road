"""영상 표시 위젯 모듈"""

import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QFont, QBrush

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.calibrator import CalibrationResult
    from src.core.trajectory_filter import FilteredTrajectory
    from src.detection.pole_detector import VerticalStructure


class VideoWidget(QWidget):
    """
    영상 표시 위젯

    - 종횡비 유지하면서 위젯 크기에 맞게 스케일링
    - 레터박스(검은 여백) 처리
    - 마우스 좌표를 원본 영상 좌표로 변환 지원
    - 캘리브레이션 결과 시각화 (소실점, 궤적, 수직선)
    """

    # 마우스 클릭 시그널 (원본 영상 좌표)
    clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #1a1a1a;")

        self._frame = None          # 원본 프레임 (numpy array)
        self._qimage = None         # 변환된 QImage
        self._display_rect = None   # 실제 영상이 그려지는 영역
        self._detections = []       # 검출 결과 리스트
        self._trajectories = {}     # 궤적 데이터 {track_id: [(x, y), ...]}
        self._lanes = []            # 차선 검출 결과
        self._poles = []            # Pole 검출 결과 (추출 중 시각화용)

        # 캘리브레이션 결과 시각화
        self._calibration_result = None
        self._filtered_trajectories = []
        self._vertical_structures = []

    def set_frame(self, frame: np.ndarray):
        """
        프레임 설정

        Args:
            frame: BGR 포맷의 numpy array (OpenCV 기본 포맷)
        """
        if frame is None:
            self._frame = None
            self._qimage = None
            self.update()
            return

        self._frame = frame

        # BGR -> RGB 변환
        rgb = frame[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        self._qimage = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
        )

        self.update()

    def get_frame(self) -> np.ndarray:
        """현재 프레임 반환"""
        return self._frame

    def set_detections(self, detections: list, trajectories: dict = None):
        """
        검출 결과 설정

        Args:
            detections: Detection 객체 리스트
            trajectories: 궤적 데이터 {track_id: [(x, y), ...]}
        """
        self._detections = detections
        self._trajectories = trajectories or {}
        # print(f"[VideoWidget] set_detections: 검출={len(detections)}개, 궤적={len(self._trajectories)}개")
        self.update()

    def set_lanes(self, lanes: list):
        """
        차선 검출 결과 설정

        Args:
            lanes: Lane 객체 리스트
        """
        self._lanes = lanes
        self.update()

    def set_poles(self, poles: list):
        """
        Pole 검출 결과 설정 (추출 중 시각화용)

        Args:
            poles: VerticalStructure 객체 리스트
        """
        self._poles = poles
        # print(f"[VideoWidget] set_poles: pole={len(poles)}개")
        self.update()

    def set_calibration_result(
        self,
        result: Optional['CalibrationResult'] = None,
        trajectories: Optional[List['FilteredTrajectory']] = None,
        vertical_structures: Optional[List['VerticalStructure']] = None
    ):
        """
        캘리브레이션 결과 설정

        Args:
            result: CalibrationResult 객체 또는 None (해제)
            trajectories: 필터링된 직진 궤적 리스트
            vertical_structures: 수직 구조물 리스트
        """
        self._calibration_result = result
        self._filtered_trajectories = trajectories or []
        self._vertical_structures = vertical_structures or []
        self.update()

    def paintEvent(self, event):
        """페인트 이벤트 - 영상 그리기"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)

        # 배경 채우기
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        if self._qimage is None:
            # 영상 없을 때 안내 텍스트
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "영상 파일을 열어주세요\n(Ctrl+O)"
            )
            return

        # 종횡비 유지하면서 스케일링
        self._display_rect = self._calc_display_rect()
        painter.drawImage(self._display_rect, self._qimage)

        # 캘리브레이션 결과 시각화 (우선)
        if self._calibration_result:
            self._draw_calibration_result(painter)
        else:
            # 기존 시각화 (캘리브레이션 결과가 없을 때만)
            # 차선 그리기
            self._draw_lanes(painter)

            # Pole 그리기 (추출 중)
            self._draw_poles(painter)

            # 궤적 그리기
            self._draw_trajectories(painter)

            # 검출 결과 그리기
            self._draw_detections(painter)

    def _draw_calibration_result(self, painter: QPainter):
        """캘리브레이션 결과 시각화"""
        if not self._calibration_result or self._display_rect is None:
            return

        result = self._calibration_result

        # 1. 직진 궤적 그리기 (연장선 포함)
        self._draw_filtered_trajectories(painter)

        # 2. 수직선 그리기 (연장선 포함)
        self._draw_vertical_structures(painter)

        # 3. 수평 소실점 그리기
        h_vp = result.horizontal_vp
        self._draw_vanishing_point(
            painter, h_vp.x, h_vp.y,
            QColor(255, 100, 100),  # 빨강
            "H-VP",
            h_vp.confidence
        )

        # 4. 수직 소실점 그리기
        v_vp = result.vertical_vp
        self._draw_vanishing_point(
            painter, v_vp.x, v_vp.y,
            QColor(100, 100, 255),  # 파랑
            "V-VP",
            v_vp.confidence
        )

        # 4-5. 단일 소실점 휴리스틱에 사용된 주요 정보(지평선, 화면 중심 등)
        self._draw_heuristic_info(painter, result)

        # 5. 초점거리 정보 표시
        self._draw_focal_length_info(painter, result)

    def _draw_filtered_trajectories(self, painter: QPainter):
        """필터링된 직진 궤적 시각화"""
        if not self._filtered_trajectories:
            return

        h_vp = self._calibration_result.horizontal_vp

        for i, traj in enumerate(self._filtered_trajectories):
            # 궤적별 색상 (황색 계열)
            hue = (i * 30 + 30) % 60 + 30  # 30~90 (노랑~주황)
            color = QColor.fromHsv(hue, 200, 255)

            # 궤적 포인트 그리기
            pen = QPen(color, 2)
            painter.setPen(pen)

            points = traj.points
            if len(points) < 2:
                continue

            prev_pt = None
            for x, y in points:
                widget_pt = self.frame_to_widget_coords(x, y)
                if widget_pt is None:
                    prev_pt = None
                    continue

                if prev_pt is not None:
                    painter.drawLine(prev_pt[0], prev_pt[1], widget_pt[0], widget_pt[1])
                prev_pt = widget_pt

            # 소실점으로의 연장선 (점선)
            if len(points) >= 2:
                # 마지막 점에서 소실점 방향으로 연장
                last_pt = self.frame_to_widget_coords(points[-1][0], points[-1][1])
                vp_pt = self.frame_to_widget_coords(h_vp.x, h_vp.y)

                if last_pt and vp_pt:
                    dash_pen = QPen(color, 1, Qt.DashLine)
                    painter.setPen(dash_pen)
                    painter.drawLine(last_pt[0], last_pt[1], vp_pt[0], vp_pt[1])

    def _draw_vertical_structures(self, painter: QPainter):
        """수직 구조물 시각화"""
        if not self._vertical_structures:
            return

        v_vp = self._calibration_result.vertical_vp

        for i, structure in enumerate(self._vertical_structures):
            # 구조물별 색상 (청색 계열)
            hue = (i * 30 + 200) % 60 + 180  # 180~240 (시안~파랑)
            color = QColor.fromHsv(hue, 200, 255)

            # 중심선 그리기
            pen = QPen(color, 3)
            painter.setPen(pen)

            center_line = structure.center_line
            if len(center_line) < 2:
                continue

            prev_pt = None
            for x, y in center_line:
                widget_pt = self.frame_to_widget_coords(x, y)
                if widget_pt is None:
                    prev_pt = None
                    continue

                if prev_pt is not None:
                    painter.drawLine(prev_pt[0], prev_pt[1], widget_pt[0], widget_pt[1])
                prev_pt = widget_pt

            # 소실점으로의 연장선 (점선)
            top_pt = self.frame_to_widget_coords(structure.top_point[0], structure.top_point[1])
            vp_pt = self.frame_to_widget_coords(v_vp.x, v_vp.y)

            if top_pt and vp_pt:
                dash_pen = QPen(color, 1, Qt.DashLine)
                painter.setPen(dash_pen)
                painter.drawLine(top_pt[0], top_pt[1], vp_pt[0], vp_pt[1])

    def _draw_vanishing_point(
        self,
        painter: QPainter,
        x: float,
        y: float,
        color: QColor,
        label: str,
        confidence: float
    ):
        """소실점 그리기"""
        widget_pt = self.frame_to_widget_coords(x, y)
        if widget_pt is None:
            # 화면 밖에 있어도 표시
            # 화면 경계에 화살표로 표시
            self._draw_offscreen_vp_indicator(painter, x, y, color, label)
            return

        wx, wy = widget_pt

        # 십자선 그리기
        pen = QPen(color, 2)
        painter.setPen(pen)
        size = 20
        painter.drawLine(wx - size, wy, wx + size, wy)
        painter.drawLine(wx, wy - size, wx, wy + size)

        # 원 그리기
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(wx - 15, wy - 15, 30, 30)

        # 라벨 그리기
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        label_text = f"{label} ({confidence:.0%})"

        # 라벨 배경
        fm = painter.fontMetrics()
        label_w = fm.horizontalAdvance(label_text) + 8
        label_h = fm.height() + 4
        label_x = wx + 20
        label_y = wy - label_h // 2

        painter.fillRect(label_x, label_y, label_w, label_h, color)

        # 라벨 텍스트
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(label_x + 4, label_y + label_h - 4, label_text)

    def _draw_offscreen_vp_indicator(
        self,
        painter: QPainter,
        x: float,
        y: float,
        color: QColor,
        label: str
    ):
        """화면 밖 소실점 인디케이터"""
        if self._display_rect is None:
            return

        rect = self._display_rect

        # 화면 중심에서 소실점 방향 계산
        cx = rect.x() + rect.width() / 2
        cy = rect.y() + rect.height() / 2

        # 프레임 좌표를 위젯 좌표로 (클리핑 없이)
        if self._qimage is None:
            return

        vp_wx = int(x * rect.width() / self._qimage.width()) + rect.x()
        vp_wy = int(y * rect.height() / self._qimage.height()) + rect.y()

        # 화면 경계와의 교차점 계산
        dx = vp_wx - cx
        dy = vp_wy - cy

        if abs(dx) < 1 and abs(dy) < 1:
            return

        # 경계에 표시할 위치 계산
        if abs(dx) > abs(dy):
            # 좌우 경계
            if dx > 0:
                edge_x = rect.x() + rect.width() - 30
            else:
                edge_x = rect.x() + 30
            edge_y = int(cy + dy * (edge_x - cx) / dx)
            edge_y = max(rect.y() + 20, min(rect.y() + rect.height() - 20, edge_y))
        else:
            # 상하 경계
            if dy > 0:
                edge_y = rect.y() + rect.height() - 30
            else:
                edge_y = rect.y() + 30
            edge_x = int(cx + dx * (edge_y - cy) / dy)
            edge_x = max(rect.x() + 20, min(rect.x() + rect.width() - 20, edge_x))

        # 화살표 그리기
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(QBrush(color))

        # 방향 계산
        import math
        angle = math.atan2(dy, dx)
        arrow_size = 15

        # 삼각형 화살표
        p1 = (edge_x + arrow_size * math.cos(angle),
              edge_y + arrow_size * math.sin(angle))
        p2 = (edge_x + arrow_size * 0.6 * math.cos(angle + 2.5),
              edge_y + arrow_size * 0.6 * math.sin(angle + 2.5))
        p3 = (edge_x + arrow_size * 0.6 * math.cos(angle - 2.5),
              edge_y + arrow_size * 0.6 * math.sin(angle - 2.5))

        from PySide6.QtGui import QPolygon
        from PySide6.QtCore import QPoint
        polygon = QPolygon([QPoint(int(p1[0]), int(p1[1])),
                           QPoint(int(p2[0]), int(p2[1])),
                           QPoint(int(p3[0]), int(p3[1]))])
        painter.drawPolygon(polygon)

        # 라벨
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(color)
        painter.drawText(edge_x - 20, edge_y - 20, label)

    def _draw_focal_length_info(self, painter: QPainter, result: 'CalibrationResult'):
        """초점거리 정보 표시"""
        if self._display_rect is None:
            return

        rect = self._display_rect

        # 정보 패널 배경
        panel_w = 230
        panel_h = 115
        panel_x = rect.x() + 10
        panel_y = rect.y() + 10

        painter.fillRect(panel_x, panel_y, panel_w, panel_h, QColor(0, 0, 0, 180))

        # 텍스트
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))

        k1, k2, _, _ = result.distortion_coeffs
        
        lines = [
            f"초점거리: {result.focal_length:.1f} px",
            f"궤적: {len(self._filtered_trajectories)}개 | 수직선: {len(self._vertical_structures)}개",
            f"수직VP: ({result.vertical_vp.x:.0f}, {result.vertical_vp.y:.0f})",
            f"왜곡 계수:",
            f"  k1 = {k1:.5f}, k2 = {k2:.5f}"
        ]

        for i, line in enumerate(lines):
            painter.drawText(panel_x + 10, panel_y + 18 + i * 18, line)

    def _draw_heuristic_info(self, painter: QPainter, result: 'CalibrationResult'):
        """단일 소실점 휴리스틱 사용 시 사용되는 추가 정보 시각화"""
        if not self._qimage or not self._display_rect:
            return

        cx = self._qimage.width() / 2
        cy = self._qimage.height() / 2
        h_vp_y = result.horizontal_vp.y

        color_center = QColor(0, 255, 0)      # 화면 중심: 녹색
        color_horizon = QColor(0, 255, 255)   # 지평선: 시안색

        # 1. 화면 중심 십자 (Principal Point)
        center_pt = self.frame_to_widget_coords(cx, cy)
        if center_pt is not None:
            pen = QPen(color_center, 2)
            painter.setPen(pen)
            r = 15
            painter.drawLine(center_pt[0] - r, center_pt[1], center_pt[0] + r, center_pt[1])
            painter.drawLine(center_pt[0], center_pt[1] - r, center_pt[0], center_pt[1] + r)
            
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(center_pt[0] + 10, center_pt[1] + 20, "센터(PP)")

        # 2. 지평선 (Horizon Line) - 수평 소실점(H-VP) 높이의 수평선
        left_pt = self.frame_to_widget_coords(0, h_vp_y)
        right_pt = self.frame_to_widget_coords(self._qimage.width(), h_vp_y)

        if left_pt and right_pt:
            # 반투명 점선으로 지평선 그리기
            horizon_pen = QPen(QColor(0, 255, 255, 200), 2, Qt.DashLine)
            painter.setPen(horizon_pen)
            painter.drawLine(left_pt[0], left_pt[1], right_pt[0], right_pt[1])
            
            painter.setPen(color_horizon)
            painter.drawText(right_pt[0] - 120, right_pt[1] - 10, "지평선(Horizon)")

        # 3. 중심에서 지평선까지 떨어지는 선 (Tilt Range / 지평선 낙하)
        vp_y_pt = self.frame_to_widget_coords(cx, h_vp_y)
        if center_pt and vp_y_pt:
            tilt_pen = QPen(QColor(255, 255, 0, 200), 2, Qt.DotLine) # 노란 점선
            painter.setPen(tilt_pen)
            painter.drawLine(center_pt[0], center_pt[1], vp_y_pt[0], vp_y_pt[1])
            
            # 낙하 거리 텍스트
            tilt_dist = cy - h_vp_y
            mid_y = (center_pt[1] + vp_y_pt[1]) / 2 + 5
            painter.setPen(QColor(255, 255, 0))
            painter.drawText(center_pt[0] + 10, int(mid_y), f"낙하 거리: {tilt_dist:.1f}px")

    def _draw_lanes(self, painter: QPainter):
        """차선 오버레이 그리기 (SAM3 마스크 + 중심선)"""
        if not self._lanes or self._display_rect is None:
            return

        for lane in self._lanes:
            # 방향성 그룹 인덱스에 따른 색상 (같은 방향은 같은 색상)
            direction_group = getattr(lane, 'direction_group', 0)
            hue = (direction_group * 90 + 30) % 360 # 그룹별로 눈에 띄게 색상 차이 줌 (30도 오프셋)
            color = QColor.fromHsv(hue, 255, 200)

            # 중심선 그리기
            if hasattr(lane, 'center_line') and lane.center_line:
                # 같은 그룹이라도 조금씩 구분되도록 선 두께나 스타일을 유지
                pen = QPen(color, 4)
                painter.setPen(pen)

                prev_pt = None
                for x, y in lane.center_line:
                    widget_pt = self.frame_to_widget_coords(x, y)
                    if widget_pt is None:
                        prev_pt = None
                        continue

                    if prev_pt is not None:
                        painter.drawLine(prev_pt[0], prev_pt[1], widget_pt[0], widget_pt[1])
                    prev_pt = widget_pt

            # 세그먼트 폴리곤 그리기 (반투명)
            if hasattr(lane, 'segments'):
                for seg in lane.segments:
                    if seg.polygon and len(seg.polygon) > 2:
                        pen = QPen(color, 1)
                        painter.setPen(pen)

                        # 폴리곤 그리기
                        points = []
                        for pt in seg.polygon:
                            widget_pt = self.frame_to_widget_coords(pt[0], pt[1])
                            if widget_pt:
                                points.append(widget_pt)

                        if len(points) > 1:
                            for j in range(len(points) - 1):
                                p1 = points[j]
                                p2 = points[j + 1]
                                painter.drawLine(p1[0], p1[1], p2[0], p2[1])

    def _draw_poles(self, painter: QPainter):
        """Pole 구조물 오버레이 그리기 (추출 중 시각화)"""
        if not self._poles or self._display_rect is None:
            # print(f"[VideoWidget] _draw_poles: pole={len(self._poles) if self._poles else 0}개, display_rect={self._display_rect is not None}")
            return

        # print(f"[VideoWidget] _draw_poles: pole {len(self._poles)}개 그리는 중")

        for i, pole in enumerate(self._poles):
            # Pole별 색상 (청록색 계열)
            hue = (i * 30 + 150) % 60 + 150  # 150~210 (청록~파랑)
            color = QColor.fromHsv(hue, 255, 255)

            # 중심선 그리기 (굵게)
            if hasattr(pole, 'center_line') and pole.center_line:
                pen = QPen(color, 4)
                painter.setPen(pen)

                prev_pt = None
                for x, y in pole.center_line:
                    widget_pt = self.frame_to_widget_coords(x, y)
                    if widget_pt is None:
                        prev_pt = None
                        continue

                    if prev_pt is not None:
                        painter.drawLine(prev_pt[0], prev_pt[1], widget_pt[0], widget_pt[1])
                    prev_pt = widget_pt

            # 상단/하단 점 표시
            if hasattr(pole, 'top_point') and hasattr(pole, 'bottom_point'):
                top_pt = self.frame_to_widget_coords(pole.top_point[0], pole.top_point[1])
                bottom_pt = self.frame_to_widget_coords(pole.bottom_point[0], pole.bottom_point[1])

                if top_pt:
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(top_pt[0] - 5, top_pt[1] - 5, 10, 10)

                if bottom_pt:
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(bottom_pt[0] - 5, bottom_pt[1] - 5, 10, 10)

                painter.setBrush(Qt.NoBrush)

            # 라벨 표시
            if hasattr(pole, 'center_line') and pole.center_line:
                top_pt = self.frame_to_widget_coords(pole.center_line[0][0], pole.center_line[0][1])
                if top_pt:
                    font = QFont()
                    font.setPointSize(9)
                    font.setBold(True)
                    painter.setFont(font)

                    label = f"Pole #{i+1}"
                    if hasattr(pole, 'score'):
                        label += f" ({pole.score:.0%})"

                    # 라벨 배경
                    fm = painter.fontMetrics()
                    label_w = fm.horizontalAdvance(label) + 6
                    label_h = fm.height() + 4
                    painter.fillRect(top_pt[0] - label_w // 2, top_pt[1] - 25, label_w, label_h, color)

                    # 라벨 텍스트
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(top_pt[0] - label_w // 2 + 3, top_pt[1] - 25 + label_h - 4, label)

    def _draw_detections(self, painter: QPainter):
        """검출 결과 오버레이 그리기"""
        if not self._detections or self._display_rect is None:
            # print(f"[VideoWidget] _draw_detections: 검출={len(self._detections) if self._detections else 0}개, display_rect={self._display_rect is not None}")
            return

        # print(f"[VideoWidget] _draw_detections: 검출 {len(self._detections)}개 그리는 중")

        # 클래스별 색상
        colors = {
            'car': QColor(0, 255, 0),        # 녹색
            'motorcycle': QColor(255, 255, 0),  # 노란색
            'bus': QColor(0, 128, 255),      # 파란색
            'truck': QColor(255, 128, 0),    # 주황색
        }
        default_color = QColor(255, 0, 255)  # 마젠타

        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        for det in self._detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, default_color)

            # 프레임 좌표 -> 위젯 좌표 변환
            pt1 = self.frame_to_widget_coords(x1, y1)
            pt2 = self.frame_to_widget_coords(x2, y2)
            if pt1 is None or pt2 is None:
                continue

            # 바운딩 박스 그리기
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(pt1[0], pt1[1], pt2[0] - pt1[0], pt2[1] - pt1[1])

            # 라벨 텍스트 구성
            if det.is_tracked:
                label = f"#{det.track_id} {det.class_name}"
            else:
                label = f"{det.class_name} {det.confidence:.0%}"

            # 라벨 배경
            fm = painter.fontMetrics()
            label_w = fm.horizontalAdvance(label) + 6
            label_h = fm.height() + 4
            painter.fillRect(pt1[0], pt1[1] - label_h, label_w, label_h, color)

            # 라벨 텍스트
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(pt1[0] + 3, pt1[1] - 4, label)

    def _draw_trajectories(self, painter: QPainter):
        """궤적 그리기"""
        if not self._trajectories or self._display_rect is None:
            # print(f"[VideoWidget] _draw_trajectories: 궤적={len(self._trajectories) if self._trajectories else 0}개, display_rect={self._display_rect is not None}")
            return

        # print(f"[VideoWidget] _draw_trajectories: 궤적 {len(self._trajectories)}개 그리는 중")

        # track_id별 색상 (해시 기반)
        def get_track_color(track_id: int) -> QColor:
            hue = (track_id * 67) % 360  # 분산된 색상
            return QColor.fromHsv(hue, 255, 255)

        for track_id, points in self._trajectories.items():
            if len(points) < 2:
                continue

            color = get_track_color(track_id)
            pen = QPen(color, 2)
            painter.setPen(pen)

            # 궤적 선 그리기
            prev_pt = None
            for x, y in points:
                widget_pt = self.frame_to_widget_coords(x, y)
                if widget_pt is None:
                    prev_pt = None
                    continue

                if prev_pt is not None:
                    painter.drawLine(prev_pt[0], prev_pt[1], widget_pt[0], widget_pt[1])

                prev_pt = widget_pt

            # 마지막 점에 원 표시
            if prev_pt is not None:
                painter.setBrush(color)
                painter.drawEllipse(prev_pt[0] - 4, prev_pt[1] - 4, 8, 8)
                painter.setBrush(Qt.NoBrush)  # 브러시 초기화

    def _calc_display_rect(self) -> QRect:
        """영상 표시 영역 계산 (종횡비 유지, 레터박스)"""
        if self._qimage is None:
            return QRect()

        widget_w = self.width()
        widget_h = self.height()
        img_w = self._qimage.width()
        img_h = self._qimage.height()

        # 스케일 비율 계산
        scale_w = widget_w / img_w
        scale_h = widget_h / img_h
        scale = min(scale_w, scale_h)

        # 스케일링된 크기
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)

        # 중앙 정렬을 위한 오프셋
        offset_x = (widget_w - scaled_w) // 2
        offset_y = (widget_h - scaled_h) // 2

        return QRect(offset_x, offset_y, scaled_w, scaled_h)

    def widget_to_frame_coords(self, widget_x: int, widget_y: int) -> tuple:
        """
        위젯 좌표를 원본 프레임 좌표로 변환

        Args:
            widget_x: 위젯 내 x 좌표
            widget_y: 위젯 내 y 좌표

        Returns:
            (frame_x, frame_y) 또는 영역 밖이면 None
        """
        if self._display_rect is None or self._qimage is None:
            return None

        rect = self._display_rect

        # 표시 영역 내인지 확인
        if not rect.contains(widget_x, widget_y):
            return None

        # 상대 좌표 계산
        rel_x = widget_x - rect.x()
        rel_y = widget_y - rect.y()

        # 원본 좌표로 변환
        frame_x = int(rel_x * self._qimage.width() / rect.width())
        frame_y = int(rel_y * self._qimage.height() / rect.height())

        return (frame_x, frame_y)

    def frame_to_widget_coords(self, frame_x: float, frame_y: float) -> tuple:
        """
        원본 프레임 좌표를 위젯 좌표로 변환

        Args:
            frame_x: 프레임 내 x 좌표
            frame_y: 프레임 내 y 좌표

        Returns:
            (widget_x, widget_y) 또는 영역 밖이면 None
        """
        if self._display_rect is None or self._qimage is None:
            return None

        rect = self._display_rect

        # 위젯 좌표로 변환
        widget_x = int(frame_x * rect.width() / self._qimage.width()) + rect.x()
        widget_y = int(frame_y * rect.height() / self._qimage.height()) + rect.y()

        # 표시 영역 내인지 확인
        if not (rect.x() <= widget_x <= rect.x() + rect.width() and
                rect.y() <= widget_y <= rect.y() + rect.height()):
            return None

        return (widget_x, widget_y)

    def mousePressEvent(self, event):
        """마우스 클릭 이벤트"""
        if event.button() == Qt.LeftButton:
            coords = self.widget_to_frame_coords(event.x(), event.y())
            if coords:
                self.clicked.emit(coords[0], coords[1])

    def get_display_scale(self) -> float:
        """현재 표시 스케일 반환"""
        if self._display_rect is None or self._qimage is None:
            return 1.0
        return self._display_rect.width() / self._qimage.width()
