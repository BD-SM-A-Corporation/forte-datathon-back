import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class EvidenceRenderer:
    """
    Renders visual evidence: bounding boxes, skeleton annotations, timestamps,
    and encodes to JPEG / base64 for API and frontend display.
    """

    @staticmethod
    def draw_evidence(
        frame: np.ndarray,
        timestamp_sec: float,
        title: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        h, w, _ = annotated.shape

        # Draw banner at bottom
        banner_h = 40
        cv2.rectangle(annotated, (0, h - banner_h), (w, h), (20, 20, 20), -1)

        time_text = f"T: {timestamp_sec:.2f}s | {title}"
        cv2.putText(
            annotated,
            time_text,
            (15, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return annotated

    @classmethod
    def frame_to_base64(cls, frame: np.ndarray, quality: int = 85) -> str:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, buf = cv2.imencode(".jpg", frame, encode_param)
        if not ret:
            return ""
        return base64.b64encode(buf).decode("utf-8")

    @classmethod
    def save_evidence_frame(cls, frame: np.ndarray, dest_path: Path, quality: int = 85) -> str:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(str(dest_path), frame, encode_param)
        return str(dest_path)
