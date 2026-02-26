"""영상 입출력 유틸리티"""

import cv2
import numpy as np


class VideoReader:
    """영상 파일 읽기 클래스"""

    def __init__(self, file_path: str):
        """
        Args:
            file_path: 영상 파일 경로
        """
        self._cap = cv2.VideoCapture(file_path)
        self._file_path = file_path

        if self._cap.isOpened():
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self._width = 0
            self._height = 0
            self._fps = 0
            self._frame_count = 0

    def is_opened(self) -> bool:
        """영상이 정상적으로 열렸는지 확인"""
        return self._cap.isOpened()

    def get_info(self) -> dict:
        """영상 정보 반환"""
        return {
            'file_path': self._file_path,
            'width': self._width,
            'height': self._height,
            'fps': self._fps,
            'frame_count': self._frame_count,
        }

    def read_frame(self) -> np.ndarray:
        """
        다음 프레임 읽기

        Returns:
            BGR 포맷의 numpy array, 실패 시 None
        """
        ret, frame = self._cap.read()
        if ret:
            return frame
        return None

    def seek(self, frame_idx: int):
        """
        특정 프레임으로 이동

        Args:
            frame_idx: 이동할 프레임 인덱스 (0-based)
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def get_position(self) -> int:
        """현재 프레임 위치 반환"""
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def release(self):
        """리소스 해제"""
        if self._cap:
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
