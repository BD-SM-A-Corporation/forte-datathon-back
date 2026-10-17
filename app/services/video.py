"""Bounded-memory decoding, with frame times from FFmpeg presentation timestamps."""

import itertools
import json
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.core.errors import VideoError


@dataclass(frozen=True)
class VideoMetadata:
    duration_sec: float
    fps: float
    width: int
    height: int
    frame_count: int
    timestamps: tuple[float, ...]


@dataclass
class Frame:
    timestamp_sec: float
    image: np.ndarray  # BGR, full frame unless explicitly transformed


@dataclass(frozen=True)
class Interval:
    start_sec: float
    end_sec: float
    motion_score: float = 0.0


def _probe(path: Path, entries: str, frames: bool = False) -> dict:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if frames:
        command += ["-show_frames"]
    command += ["-show_entries", entries, "-of", "json", str(path.resolve())]
    try:
        result = subprocess.run(command, capture_output=True, timeout=30, check=True)
        if result.stderr.strip():
            raise VideoError("Video contains decoding errors")
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise VideoError("Video is unreadable or FFmpeg is unavailable") from exc


def inspect_video(path: Path, settings: Settings) -> VideoMetadata:
    data = _probe(path, "stream=width,height,duration,avg_frame_rate:format=duration")
    try:
        stream = data["streams"][0]
        duration = float(stream.get("duration", data.get("format", {}).get("duration", 0)))
        width, height = int(stream["width"]), int(stream["height"])
        numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
        fps = numerator / denominator
        if not (0 < duration <= settings.max_duration_sec and 0 < fps <= 120):
            raise ValueError("duration or frame rate")
        if not (0 < width * height <= settings.max_video_pixels):
            raise ValueError("resolution")
        if duration * fps > settings.max_decoded_frames:
            raise ValueError("frame count")
        data = _probe(path, "frame=best_effort_timestamp_time", frames=True)
        pts = [float(f["best_effort_timestamp_time"]) for f in data["frames"]]
        times = tuple(t - pts[0] for t in pts)
        if not times or len(times) > settings.max_decoded_frames:
            raise ValueError("frame count")
        if (
            any(not np.isfinite(t) for t in times)
            or any(b <= a for a, b in itertools.pairwise(times))
            or times[-1] >= duration
        ):
            raise ValueError("presentation timestamps")
        return VideoMetadata(duration, fps, width, height, len(times), times)
    except (KeyError, IndexError, ValueError, ZeroDivisionError, TypeError) as exc:
        raise VideoError(
            "Invalid video metadata, timing, or configured video limits exceeded"
        ) from exc


def decode_frames(
    path: Path, metadata: VideoMetadata, check: Callable[[], None] = lambda: None
) -> Iterator[Frame]:
    # ffprobe and OpenCV both decode the first video stream in presentation order.
    # Rotation is disabled so ROI and original evidence use identical coordinates.
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    try:
        if not cap.isOpened():
            raise VideoError("OpenCV cannot decode this recording")
        for timestamp in metadata.timestamps:
            check()
            ok, image = cap.read()
            if not ok:
                raise VideoError("Video decoding ended before its presentation timestamps")
            yield Frame(timestamp, image)
        if cap.read()[0]:
            raise VideoError("Video decoder and presentation timestamps disagree")
    finally:
        cap.release()


def sample_frames(
    path: Path,
    metadata: VideoMetadata,
    fps: float,
    start: float = 0,
    end: float | None = None,
    check: Callable[[], None] = lambda: None,
) -> Iterator[Frame]:
    end = min(end if end is not None else metadata.duration_sec, metadata.duration_sec)
    target = start
    for frame in decode_frames(path, metadata, check):
        if frame.timestamp_sec > end:
            break
        if frame.timestamp_sec + 1e-7 >= target:
            yield frame
            target = start + (int((frame.timestamp_sec - start) * fps + 1e-7) + 1) / fps


def transform_frame(frame: Frame, settings: Settings) -> Frame:
    image = frame.image
    if settings.roi:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = settings.roi
        image = image[
            int(y1 * h) : max(int(y2 * h), int(y1 * h) + 1),
            int(x1 * w) : max(int(x2 * w), int(x1 * w) + 1),
        ]
    h, w = image.shape[:2]
    ratio = min(1.0, settings.inference_max_side / max(h, w))
    image = cv2.resize(image, (max(1, round(w * ratio)), max(1, round(h * ratio))))
    return Frame(frame.timestamp_sec, image)


def windows(duration: float, length: float, overlap: float) -> list[Interval]:
    result = []
    start = 0.0
    while start < duration:
        end = min(duration, start + length)
        result.append(Interval(start, end))
        if end >= duration:
            break
        start += length - overlap
    return result


def merge_intervals(intervals: list[Interval], gap: float = 1) -> list[Interval]:
    merged: list[Interval] = []
    for current in sorted(intervals, key=lambda i: i.start_sec):
        if merged and current.start_sec <= merged[-1].end_sec + gap:
            old = merged.pop()
            current = Interval(
                old.start_sec,
                max(old.end_sec, current.end_sec),
                max(old.motion_score, current.motion_score),
            )
        merged.append(current)
    return merged
