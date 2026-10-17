import base64
from collections.abc import Callable
from pathlib import Path

import cv2

from app.core.errors import VideoError
from app.schemas.analysis import EvidenceFrame
from app.services.video import VideoMetadata, decode_frames


def extract_evidence(
    path: Path,
    metadata: VideoMetadata,
    timestamps: list[float],
    check: Callable[[], None] = lambda: None,
) -> dict[float, EvidenceFrame]:
    """One decoding pass for all consolidated events. Original pixels, no ROI/overlays."""
    if any(not 0 <= t < metadata.duration_sec for t in timestamps):
        raise VideoError("Evidence timestamp outside recording")
    wanted = set(timestamps)
    if not wanted <= set(metadata.timestamps):
        raise VideoError("Evidence must reference actual decoded frame timestamps")
    output = {}
    if not wanted:
        return output
    for frame in decode_frames(path, metadata, check):
        if frame.timestamp_sec in wanted:
            ok, encoded = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise VideoError("Evidence JPEG encoding failed")
            output[frame.timestamp_sec] = EvidenceFrame(
                timestamp_sec=frame.timestamp_sec,
                image_base64=base64.b64encode(encoded).decode("ascii"),
            )
            if len(output) == len(wanted):
                break
    if len(output) != len(wanted):
        raise VideoError("Evidence frames could not be decoded")
    return output
