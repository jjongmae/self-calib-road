"""메인 윈도우 모듈"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QSlider, QStatusBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence

from src.ui.video_widget import VideoWidget
from src.utils.video_io import VideoReader
from src.detection.vehicle_tracker import VehicleTracker
from src.detection.lane_detector import LaneDetector


class MainWindow(QMainWindow):
    """캘리브레이션 도구 메인 윈도우"""

    def __init__(self):
        super().__init__()
        self.video_reader = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)
        self.is_playing = False

        # 차량 추적기
        self.tracker = None
        self.tracking_enabled = False

        # 차선 검출기 (지연 로딩 - SAM3 모델이 무거움)
        self.lane_detector = None
        self.lane_detection_enabled = False

        self._init_ui()
        self._init_menu()
        self._init_tracker()

    def _init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("도로 카메라 캘리브레이션")
        self.setFixedSize(1280, 720)

        # 중앙 위젯
        central = QWidget()
        self.setCentralWidget(central)

        # 메인 레이아웃
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 영상 위젯
        self.video_widget = VideoWidget()
        layout.addWidget(self.video_widget, stretch=1)

        # 하단 컨트롤 패널
        control_layout = QHBoxLayout()
        control_layout.setSpacing(8)

        # 재생/일시정지 버튼
        self.btn_play = QPushButton("재생")
        self.btn_play.setFixedWidth(80)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_play.setEnabled(False)
        control_layout.addWidget(self.btn_play)

        # 프레임 슬라이더
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_changed)
        control_layout.addWidget(self.slider, stretch=1)

        # 프레임 정보 라벨
        self.lbl_frame = QLabel("0 / 0")
        self.lbl_frame.setFixedWidth(100)
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.lbl_frame)

        # 차량 추적 토글 버튼
        self.btn_track = QPushButton("차량 추적")
        self.btn_track.setFixedWidth(80)
        self.btn_track.setCheckable(True)
        self.btn_track.clicked.connect(self._toggle_tracking)
        control_layout.addWidget(self.btn_track)

        # 차선 검출 토글 버튼
        self.btn_lane = QPushButton("차선 검출")
        self.btn_lane.setFixedWidth(80)
        self.btn_lane.setCheckable(True)
        self.btn_lane.clicked.connect(self._toggle_lane_detection)
        control_layout.addWidget(self.btn_lane)

        layout.addLayout(control_layout)

        # 상태바
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("영상 파일을 열어주세요 (Ctrl+O)")

    def _init_menu(self):
        """메뉴 초기화"""
        menubar = self.menuBar()

        # 파일 메뉴
        file_menu = menubar.addMenu("파일(&F)")

        open_action = QAction("열기(&O)", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("종료(&Q)", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _init_tracker(self):
        """차량 추적기 초기화"""
        try:
            self.tracker = VehicleTracker('models/rtdetr-l.pt')
            self.status_bar.showMessage("차량 추적기 로드 완료")
        except Exception as e:
            self.status_bar.showMessage(f"추적기 로드 실패: {e}")
            self.btn_track.setEnabled(False)

    def _toggle_tracking(self, checked: bool):
        """차량 추적 토글"""
        self.tracking_enabled = checked

        if checked:
            self.btn_track.setText("추적 중...")
            # 추적기 초기화 (새로 시작)
            if self.tracker:
                self.tracker.reset()
            # 현재 프레임에 추적 적용
            self._track_current_frame()
        else:
            self.btn_track.setText("차량 추적")
            self.video_widget.set_detections([])

    def _toggle_lane_detection(self, checked: bool):
        """차선 검출 토글"""
        self.lane_detection_enabled = checked

        if checked:
            self.btn_lane.setText("로딩...")
            self.btn_lane.setEnabled(False)
            self.status_bar.showMessage("SAM3 모델 로딩 중...")

            # 지연 로딩
            if self.lane_detector is None:
                try:
                    self.lane_detector = LaneDetector()
                except Exception as e:
                    self.status_bar.showMessage(f"차선 검출기 로드 실패: {e}")
                    self.lane_detection_enabled = False
                    self.btn_lane.setChecked(False)
                    self.btn_lane.setText("차선 검출")
                    self.btn_lane.setEnabled(True)
                    return

            self.btn_lane.setText("검출 중...")
            self.btn_lane.setEnabled(True)
            self._detect_lanes_current_frame()
        else:
            self.btn_lane.setText("차선 검출")
            self.video_widget.set_lanes([])

    def _detect_lanes_current_frame(self):
        """현재 프레임에서 차선 검출"""
        if not self.lane_detection_enabled or self.lane_detector is None:
            return

        frame = self.video_widget.get_frame()
        if frame is None:
            return

        try:
            lanes = self.lane_detector.detect(frame)
            self.video_widget.set_lanes(lanes)

            # 상태바에 검출 개수 표시
            total_segments = sum(len(l.segments) for l in lanes)
            self.status_bar.showMessage(f"차선 {len(lanes)}개 검출 (세그먼트 {total_segments}개 병합)")
        except Exception as e:
            self.status_bar.showMessage(f"차선 검출 오류: {e}")

    def _track_current_frame(self):
        """현재 프레임에서 차량 추적"""
        if not self.tracking_enabled or self.tracker is None:
            return

        frame = self.video_widget.get_frame()
        if frame is None:
            return

        detections = self.tracker.track(frame)

        # 현재 검출된 차량들의 궤적 수집
        trajectories = {}
        for det in detections:
            if det.is_tracked:
                traj = self.tracker.get_recent_points(det.track_id, n=50)
                if traj:
                    trajectories[det.track_id] = traj

        self.video_widget.set_detections(detections, trajectories)

    def _open_file(self):
        """영상 파일 열기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 파일 열기",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )

        if file_path:
            self._load_video(file_path)

    def _load_video(self, file_path: str):
        """영상 로드"""
        # 기존 영상 정리
        if self.video_reader:
            self.timer.stop()
            self.video_reader.release()

        # 추적기 리셋
        if self.tracker:
            self.tracker.reset()

        self.video_reader = VideoReader(file_path)

        if not self.video_reader.is_opened():
            self.status_bar.showMessage(f"영상 열기 실패: {file_path}")
            return

        # UI 업데이트
        info = self.video_reader.get_info()
        self.slider.setRange(0, info['frame_count'] - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.btn_play.setEnabled(True)

        # 첫 프레임 표시
        frame = self.video_reader.read_frame()
        if frame is not None:
            self.video_widget.set_frame(frame)

        self._update_frame_label()
        self.status_bar.showMessage(
            f"{file_path} | {info['width']}x{info['height']} | "
            f"{info['fps']:.1f}fps | {info['frame_count']}프레임"
        )

    def _toggle_play(self):
        """재생/일시정지 토글"""
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        """재생 시작"""
        if not self.video_reader:
            return

        info = self.video_reader.get_info()
        interval = int(1000 / info['fps'])
        self.timer.start(interval)
        self.is_playing = True
        self.btn_play.setText("일시정지")

    def _pause(self):
        """일시정지"""
        self.timer.stop()
        self.is_playing = False
        self.btn_play.setText("재생")

    def _on_timer(self):
        """타이머 콜백 - 다음 프레임 표시"""
        if not self.video_reader:
            return

        frame = self.video_reader.read_frame()
        if frame is None:
            # 영상 끝
            self._pause()
            self.video_reader.seek(0)
            return

        self.video_widget.set_frame(frame)

        # 차량 추적
        self._track_current_frame()

        # 차선 검출
        self._detect_lanes_current_frame()

        # 슬라이더 업데이트 (시그널 차단)
        self.slider.blockSignals(True)
        self.slider.setValue(self.video_reader.get_position())
        self.slider.blockSignals(False)

        self._update_frame_label()

    def _on_slider_pressed(self):
        """슬라이더 드래그 시작"""
        if self.is_playing:
            self.timer.stop()

    def _on_slider_released(self):
        """슬라이더 드래그 종료"""
        if self.is_playing:
            info = self.video_reader.get_info()
            interval = int(1000 / info['fps'])
            self.timer.start(interval)

    def _on_slider_changed(self, value: int):
        """슬라이더 값 변경"""
        if not self.video_reader:
            return

        self.video_reader.seek(value)
        frame = self.video_reader.read_frame()
        if frame is not None:
            self.video_widget.set_frame(frame)
            # seek 후 read하면 위치가 +1 되므로 다시 seek
            self.video_reader.seek(value)
            # 차량 추적
            self._track_current_frame()
            # 차선 검출
            self._detect_lanes_current_frame()

        self._update_frame_label()

    def _update_frame_label(self):
        """프레임 라벨 업데이트"""
        if self.video_reader:
            info = self.video_reader.get_info()
            pos = self.video_reader.get_position()
            self.lbl_frame.setText(f"{pos} / {info['frame_count']}")

    def closeEvent(self, event):
        """윈도우 닫기 이벤트"""
        self.timer.stop()
        if self.video_reader:
            self.video_reader.release()
        if self.lane_detector:
            self.lane_detector.release()
        event.accept()
