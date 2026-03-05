"""메인 윈도우 모듈"""
import os
import json
from datetime import datetime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QStatusBar,
    QProgressBar, QMessageBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence

from src.ui.video_widget import VideoWidget
from src.utils.video_io import VideoReader
from src.detection.vehicle_tracker import VehicleTracker
from src.core.trajectory_filter import TrajectoryFilter
from src.core.calibrator import Calibrator, CalibrationResult


class ExtractionWorker(QThread):
    """특징 추출 워커 스레드"""
    progress = Signal(int, str)  # (진행률, 메시지)
    finished = Signal(object, object)  # (궤적 dict, 수직구조물 list)
    error = Signal(str)
    frame_update = Signal(object, object, object)  # (프레임, 검출결과 list, 궤적 dict)
    pole_update = Signal(object, object)  # (프레임, pole구조물 list)

    def __init__(self, video_path: str, tracker: VehicleTracker):
        super().__init__()
        self.video_path = video_path
        self.tracker = tracker
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            from src.utils.video_io import VideoReader
            import cv2

            reader = VideoReader(self.video_path)
            if not reader.is_opened():
                self.error.emit("영상 열기 실패")
                return

            info = reader.get_info()
            total_frames = info['frame_count']
            all_trajectories = {}
            vertical_structures = []

            # 추적기 초기화
            self.tracker.reset()

            # 전체 영상 처리
            frame_idx = 0
            while not self._is_cancelled:
                frame = reader.read_frame()
                if frame is None:
                    break

                # 차량 추적
                detections = self.tracker.track(frame)

                # 궤적 수집
                for det in detections:
                    if det.is_tracked:
                        track_id = det.track_id
                        cx = (det.bbox[0] + det.bbox[2]) / 2
                        cy = (det.bbox[1] + det.bbox[3]) / 2

                        if track_id not in all_trajectories:
                            all_trajectories[track_id] = []
                        all_trajectories[track_id].append((cx, cy))

                frame_idx += 1

                # 시각화 업데이트 (5프레임마다)
                if frame_idx % 5 == 0:
                    # 현재 궤적 데이터 (최근 50포인트만)
                    current_trajectories = {}
                    for tid, traj in all_trajectories.items():
                        current_trajectories[tid] = traj[-50:] if len(traj) > 50 else traj

                    self.frame_update.emit(frame, detections, current_trajectories)

                # 진행률 업데이트 (10프레임마다)
                if frame_idx % 10 == 0:
                    progress = int(frame_idx * 100 / total_frames)  # 0~100%
                    self.progress.emit(progress, f"프레임 {frame_idx}/{total_frames} 특징 추출 중...")

            reader.release()

            if self._is_cancelled:
                return

            self.progress.emit(100, "완료")
            self.finished.emit(all_trajectories, vertical_structures)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """캘리브레이션 도구 메인 윈도우"""

    def __init__(self):
        super().__init__()
        self.video_reader = None
        self.video_path = None

        # 차량 추적기
        self.tracker = None

        # 특징 데이터
        self.trajectories = {}  # 전체 궤적 {track_id: [(x, y), ...]}
        self.filtered_trajectories = []  # 필터링된 직진 궤적
        self.vertical_structures = []  # 수직 구조물

        # 캘리브레이션 결과
        self.calibration_result: CalibrationResult = None

        # 결과 표시 상태
        self.show_result = False

        # 추출 워커
        self.extraction_worker = None

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

        self.btn_calibrate = QPushButton("캘리브레이션1 시작")
        self.btn_calibrate.setFixedWidth(150)
        self.btn_calibrate.clicked.connect(self._start_calibration_pipeline)
        self.btn_calibrate.setEnabled(False)
        control_layout.addWidget(self.btn_calibrate)
        
        control_layout.addStretch()

        # 화각 입력 필드 추가
        fov_layout = QHBoxLayout()
        fov_layout.addWidget(QLabel("화각(D/H/V):"))

        self.spin_dfov = QDoubleSpinBox()
        self.spin_dfov.setRange(0.0, 180.0)
        self.spin_dfov.setValue(65.56)
        self.spin_dfov.setDecimals(2)
        self.spin_dfov.setToolTip("대각 화각 (DFOV). 모르면 0.0")
        self.spin_dfov.setFixedWidth(70)
        fov_layout.addWidget(self.spin_dfov)

        self.spin_hfov = QDoubleSpinBox()
        self.spin_hfov.setRange(0.0, 180.0)
        self.spin_hfov.setValue(58.44)
        self.spin_hfov.setDecimals(2)
        self.spin_hfov.setToolTip("수평 화각 (HFOV). 모르면 0.0")
        self.spin_hfov.setFixedWidth(70)
        fov_layout.addWidget(self.spin_hfov)

        self.spin_vfov = QDoubleSpinBox()
        self.spin_vfov.setRange(0.0, 180.0)
        self.spin_vfov.setValue(34.11)
        self.spin_vfov.setDecimals(2)
        self.spin_vfov.setToolTip("수직 화각 (VFOV). 모르면 0.0")
        self.spin_vfov.setFixedWidth(70)
        fov_layout.addWidget(self.spin_vfov)

        control_layout.addLayout(fov_layout)

        layout.addLayout(control_layout)

        # 진행률 표시 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

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
            if hasattr(self, 'btn_calibrate'):
                self.btn_calibrate.setEnabled(False)

    def _start_calibration_pipeline(self):
        """특징 추출 및 캘리브레이션 시작 / 중지 토글"""
        if self.extraction_worker and self.extraction_worker.isRunning():
            # 중지 로직
            self.extraction_worker.cancel()
            self.extraction_worker.wait() # 스레드 완전 종료 대기
            self.btn_calibrate.setEnabled(True)
            self.btn_calibrate.setText("캘리브레이션1 시작")
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("캘리브레이션1 작업이 중지되었습니다.")
            return

        if self.video_path is None:
            return

        # 기존 데이터 초기화
        self.trajectories = {}
        self.filtered_trajectories = []
        self.vertical_structures = []
        self.calibration_result = None
        self.show_result = False
        self.video_widget.set_calibration_result(None)

        # UI 상태 업데이트
        self.btn_calibrate.setText("캘리브레이션1 중지")
        self.progress_bar.setValue(0)

        # 워커 스레드 시작
        self.extraction_worker = ExtractionWorker(
            self.video_path, self.tracker
        )
        self.extraction_worker.progress.connect(self._on_extraction_progress)
        self.extraction_worker.finished.connect(self._on_extraction_finished)
        self.extraction_worker.error.connect(self._on_extraction_error)
        self.extraction_worker.frame_update.connect(self._on_frame_update)
        self.extraction_worker.pole_update.connect(self._on_pole_update)
        self.extraction_worker.start()

    def _on_extraction_progress(self, progress: int, message: str):
        """추출 진행 상황 업데이트"""
        self.progress_bar.setValue(progress)
        self.status_bar.showMessage(message)

    def _on_extraction_finished(self, trajectories: dict, structures: list):
        """추출 완료"""
        self.trajectories = trajectories
        self.vertical_structures = structures

        # 직진 궤적 필터링
        trajectory_filter = TrajectoryFilter()
        self.filtered_trajectories = trajectory_filter.filter_trajectories(trajectories)

        # 추출 중 시각화 클리어
        self.video_widget.set_detections([])
        self.video_widget.set_poles([])

        # 최소 요구사항 검사
        if len(self.filtered_trajectories) >= 3:
            self.status_bar.showMessage(f"특징 추출 완료. 캘리브레이션 계산 수행 중...")
            self._run_calibration()
        else:
            self.progress_bar.setValue(0)
            self.btn_calibrate.setEnabled(True)
            self.btn_calibrate.setText("캘리브레이션1 시작")
            QMessageBox.warning(
                self, "주의",
                f"캘리브레이션에 필요한 직진 궤적 데이터 부족\n"
                f"- 직진 궤적: {len(self.filtered_trajectories)}개 (최소 3개 필요)"
            )

    def _on_extraction_error(self, error_msg: str):
        """추출 오류"""
        self.btn_calibrate.setEnabled(True)
        self.btn_calibrate.setText("캘리브레이션1 시작")
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "오류", f"특징 추출 실패: {error_msg}")

    def _on_frame_update(self, frame, detections, trajectories):
        """차량 추적 실시간 시각화"""
        # print(f"[MainWindow] 프레임 업데이트: 검출={len(detections)}개, 궤적={len(trajectories)}개")
        self.video_widget.set_frame(frame)
        self.video_widget.set_detections(detections, trajectories)

    def _on_pole_update(self, frame, structures):
        """Pole 검출 실시간 시각화"""
        # print(f"[MainWindow] Pole 업데이트: 구조물={len(structures)}개")
        self.video_widget.set_frame(frame)
        self.video_widget.set_poles(structures)

    def _run_calibration(self):
        """캘리브레이션 실행"""
        if not self.filtered_trajectories:
            QMessageBox.warning(self, "경고", "차량 직진 궤적 데이터가 부족합니다")
            return

        # 수직선 파라미터 추출
        vertical_lines = [s.line_params for s in self.vertical_structures]

        # 이미지 크기
        if self.video_reader:
            info = self.video_reader.get_info()
            image_size = (info['width'], info['height'])
        else:
            image_size = (1920, 1080)  # 기본값

        # 캘리브레이션 실행
        calibrator = Calibrator()
        dfov = self.spin_dfov.value()
        hfov = self.spin_hfov.value()
        vfov = self.spin_vfov.value()

        result = calibrator.calibrate(
            self.filtered_trajectories,
            vertical_lines,
            image_size,
            dfov, hfov, vfov
        )

        if result is None:
            self.progress_bar.setValue(0)
            self.btn_calibrate.setEnabled(True)
            self.btn_calibrate.setText("캘리브레이션1 시작")
            QMessageBox.warning(self, "실패", "캘리브레이션 실패\n소실점 계산에 실패했습니다")
            return

        self.calibration_result = result
        self.show_result = True
        
        # 바로 화면에 결과 표출 (토글 해제 없이)
        self.video_widget.set_calibration_result(
            self.calibration_result,
            self.filtered_trajectories,
            self.vertical_structures
        )

        # UI 상태 업데이트
        self.progress_bar.setValue(100)
        self.btn_calibrate.setEnabled(True)
        self.btn_calibrate.setText("캘리브레이션1 시작")

        # 결과 저장 (data/result 폴더)
        try:
            result_dir = "data/result"
            os.makedirs(result_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calib_result_{timestamp}.json"
            filepath = os.path.join(result_dir, filename)
            
            k1, k2, p1, p2 = result.distortion_coeffs
            
            output_data = {
                "camera_id": timestamp,
                "image_size": {
                    "width": image_size[0],
                    "height": image_size[1]
                },
                "intrinsic": {
                    "fx": round(result.focal_length, 3),
                    "fy": round(result.focal_length, 3),
                    "cx": round(result.principal_point[0], 4),
                    "cy": round(result.principal_point[1], 4)
                },
                "distortion": {
                    "k1": round(k1, 5),
                    "k2": round(k2, 5),
                    "k3": 0.0,
                    "p1": round(p1, 5),
                    "p2": round(p2, 5)
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
                
            print(f"[MainWindow] 캘리브레이션 결과 저장 완료: {filepath}")
        except Exception as e:
            print(f"[MainWindow] 결과 저장 실패: {e}")

        # 결과 메시지
        msg = (
            f"✅ 완료! 초점거리: {result.focal_length:.1f}px | "
            f"수평VP: ({result.horizontal_vp.x:.0f}, {result.horizontal_vp.y:.0f}) | "
            f"수직VP: ({result.vertical_vp.x:.0f}, {result.vertical_vp.y:.0f})"
        )
        self.status_bar.showMessage(msg)

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
            self.video_reader.release()

        # 추적기 리셋
        if self.tracker:
            self.tracker.reset()

        # 기존 데이터 초기화
        self.trajectories = {}
        self.filtered_trajectories = []
        self.vertical_structures = []
        self.calibration_result = None
        self.show_result = False
        self.video_widget.set_calibration_result(None)

        self.video_path = file_path
        self.video_reader = VideoReader(file_path)

        if not self.video_reader.is_opened():
            self.status_bar.showMessage(f"영상 열기 실패: {file_path}")
            return

        # UI 업데이트
        info = self.video_reader.get_info()
        self.btn_calibrate.setEnabled(True)
        self.btn_calibrate.setText("캘리브레이션1 시작")

        # 첫 프레임 표시
        frame = self.video_reader.read_frame()
        if frame is not None:
            self.video_widget.set_frame(frame)

        self.status_bar.showMessage(
            f"{file_path} | {info['width']}x{info['height']} | "
            f"{info['fps']:.1f}fps | {info['frame_count']}프레임"
        )

    def closeEvent(self, event):
        """윈도우 닫기 이벤트"""
        # 워커 스레드 중지
        if self.extraction_worker and self.extraction_worker.isRunning():
            self.extraction_worker.cancel()
            self.extraction_worker.wait()

        if self.video_reader:
            self.video_reader.release()
        event.accept()
