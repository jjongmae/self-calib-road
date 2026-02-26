"""영상 표시 위젯 모듈"""

import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QFont


class VideoWidget(QWidget):
    """
    영상 표시 위젯

    - 종횡비 유지하면서 위젯 크기에 맞게 스케일링
    - 레터박스(검은 여백) 처리
    - 마우스 좌표를 원본 영상 좌표로 변환 지원
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
        self.update()

    def set_lanes(self, lanes: list):
        """
        차선 검출 결과 설정

        Args:
            lanes: Lane 객체 리스트
        """
        self._lanes = lanes
        self.update()

    def paintEvent(self, event):
        """페인트 이벤트 - 영상 그리기"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

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

        # 차선 그리기
        self._draw_lanes(painter)

        # 궤적 그리기
        self._draw_trajectories(painter)

        # 검출 결과 그리기
        self._draw_detections(painter)

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

    def _draw_detections(self, painter: QPainter):
        """검출 결과 오버레이 그리기"""
        if not self._detections or self._display_rect is None:
            return

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
            return

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

    def frame_to_widget_coords(self, frame_x: int, frame_y: int) -> tuple:
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
