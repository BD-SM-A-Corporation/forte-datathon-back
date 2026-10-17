import base64

import cv2
import numpy as np
import pytest

from app.core.errors import VideoError
from app.services.evidence import extract_evidence
from app.services.motion import MotionDetector
from app.services.video import (
    Frame,
    Interval,
    decode_frames,
    inspect_video,
    merge_intervals,
    sample_frames,
    transform_frame,
    windows,
)


def test_metadata(video, settings):
    meta = inspect_video(video, settings)
    assert (meta.width, meta.height, meta.frame_count, meta.fps) == (160, 120, 40, 10)
    assert meta.duration_sec == pytest.approx(4)
    assert meta.timestamps[0] == 0


def test_invalid(tmp_path, settings):
    path = tmp_path / "bad.mov"
    path.write_bytes(b"not a video")
    with pytest.raises(VideoError):
        inspect_video(path, settings)


def test_limits(video, settings):
    with pytest.raises(VideoError):
        inspect_video(video, settings.model_copy(update={"max_duration_sec": 1}))


def test_sampling(video, settings):
    meta = inspect_video(video, settings)
    sampled = list(sample_frames(video, meta, 1))
    assert [f.timestamp_sec for f in sampled] == [0, 1, 2, 3]
    detail = list(sample_frames(video, meta, 3, start=1, end=2))
    assert [f.timestamp_sec for f in detail] == pytest.approx([1, 1.4, 1.7, 2])


def test_vfr_timestamps(vfr_video, settings):
    meta = inspect_video(vfr_video, settings)
    assert np.ptp(np.diff(meta.timestamps)) > 0.2
    frames = list(decode_frames(vfr_video, meta))
    assert [f.timestamp_sec for f in frames] == list(meta.timestamps)
    assert [int(f.image.mean()) for f in frames][:4] == pytest.approx([0, 60, 120, 180], abs=5)
    assert meta.timestamps[:4] == pytest.approx([0, 0.12, 0.6, 0.84], abs=0.04)


def test_motion(video, settings):
    detector = MotionDetector(settings)
    for frame in sample_frames(video, inspect_video(video, settings), 1):
        detector.add(frame)
    intervals = detector.result()
    assert intervals and all(0 < i.motion_score <= 1 for i in intervals)
    assert len(intervals) == 1


def test_static_and_noise(settings):
    detector = MotionDetector(settings)
    rng = np.random.default_rng(42)
    for t in range(5):
        detector.add(Frame(float(t), rng.integers(0, 5, (120, 160, 3), dtype=np.uint8)))
    assert detector.result() == []


def test_intervals():
    assert windows(10, 5, 1) == [Interval(0, 5), Interval(4, 9), Interval(8, 10)]
    assert merge_intervals([Interval(0, 1, 0.3), Interval(1.5, 2, 0.8)]) == [Interval(0, 2, 0.8)]


def test_roi(settings):
    frame = Frame(1, np.zeros((120, 160, 3), dtype=np.uint8))
    result = transform_frame(frame, settings.model_copy(update={"roi": (0, 0, 0.5, 0.5)}))
    assert result.image.shape == (60, 80, 3)
    assert frame.image.shape == (120, 160, 3)


def test_evidence(video, settings):
    meta = inspect_video(video, settings)
    evidence = extract_evidence(video, meta, [1, 2])
    image = cv2.imdecode(np.frombuffer(base64.b64decode(evidence[1].image_base64), np.uint8), 1)
    assert image.shape == (120, 160, 3)
    assert evidence[1].timestamp_sec == 1
    with pytest.raises(VideoError):
        extract_evidence(video, meta, [4])
    with pytest.raises(VideoError):
        extract_evidence(video, meta, [1.01])
