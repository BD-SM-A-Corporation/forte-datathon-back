import cv2
from pathlib import Path
from typing import Generator, Tuple
import numpy as np


class VideoDecoder:
    """Decodes video files using OpenCV/ffmpeg at target frame rate."""

    def __init__(self, target_fps: int = 5):
        self.target_fps = target_fps

    def extract_frames(self, video_path: Path) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yields (frame_index, timestamp_seconds, frame_bgr)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, round(source_fps / self.target_fps))

        frame_idx = 0
        extracted_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / source_fps
                yield extracted_count, timestamp, frame
                extracted_count += 1

            frame_idx += 1

        cap.release()
