import cv2

from app.core.config import Settings
from app.services.video import Frame, Interval, merge_intervals


class MotionDetector:
    """Score = fraction of blurred 160px grayscale pixels changed by > threshold.

    A 3x3 morphological opening removes isolated noise. Merged intervals retain
    the peak changed-pixel fraction, never a probability of a violation.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.previous = None
        self.previous_time = 0.0
        self.intervals: list[Interval] = []

    def add(self, frame: Frame) -> None:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        gray = cv2.resize(gray, (160, max(1, round(h * 160 / w))))
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.previous is not None:
            delta = cv2.absdiff(gray, self.previous)
            mask = cv2.threshold(
                delta, self.settings.motion_pixel_threshold, 255, cv2.THRESH_BINARY
            )[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            score = cv2.countNonZero(mask) / mask.size
            if score >= self.settings.motion_threshold:
                self.intervals.append(Interval(self.previous_time, frame.timestamp_sec, score))
        self.previous, self.previous_time = gray, frame.timestamp_sec

    def result(self) -> list[Interval]:
        return merge_intervals(self.intervals, self.settings.motion_merge_gap_sec)
