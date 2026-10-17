from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float
    name: str


@dataclass
class PoseDetection:
    person_id: Optional[int]
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2] normalized or absolute
    keypoints: Dict[str, Keypoint]
    confidence: float
    is_cashier: bool = False


@dataclass
class ObjectDetection:
    label: str  # e.g., 'cash', 'cash_stack', 'drawer_open', 'drawer_closed', 'safe'
    bbox: Tuple[float, float, float, float]
    confidence: float
    track_id: Optional[int] = None


@dataclass
class FrameDetections:
    frame_idx: int
    timestamp_sec: float
    poses: List[PoseDetection] = field(default_factory=list)
    objects: List[ObjectDetection] = field(default_factory=list)


class Detector:
    """
    Modular detector supporting YOLO-Pose (skeleton without face recognition)
    and custom object detection (cash, safe, drawer).
    """

    def __init__(self, weights_dir: Optional[Path] = None, device: str = "cpu"):
        self.device = device
        self.weights_dir = weights_dir
        self.pose_model = None
        self.object_model = None
        self._load_models()

    def _load_models(self):
        """Attempts to load Ultralytics YOLO models if weights exist."""
        try:
            from ultralytics import YOLO
            pose_path = (self.weights_dir / "yolov8n-pose.pt") if self.weights_dir else None
            obj_path = (self.weights_dir / "yolov8n-custom.pt") if self.weights_dir else None

            if pose_path and pose_path.exists():
                self.pose_model = YOLO(str(pose_path))
            if obj_path and obj_path.exists():
                self.object_model = YOLO(str(obj_path))
        except Exception:
            # Fallback gracefully during early template stage
            self.pose_model = None
            self.object_model = None

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> FrameDetections:
        """
        Runs detection on a single frame.
        """
        detections = FrameDetections(frame_idx=frame_idx, timestamp_sec=timestamp)

        if self.pose_model is not None:
            results = self.pose_model(frame, verbose=False)
            for r in results:
                if r.keypoints is not None:
                    # Parse keypoints (wrists, elbows, shoulders, hips)
                    pass

        return detections
