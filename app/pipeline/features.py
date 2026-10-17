import math
from typing import Dict, Optional, Tuple
from app.pipeline.detector import FrameDetections, PoseDetection, ObjectDetection


class FeatureExtractor:
    """
    Computes invariant spatial and relative features:
    - Distance between cashier wrists/hands and objects (cash, drawer, counter)
    - Cashier posture relative to counter (present, away, facing away)
    - Avoiding absolute pixel coordinates to remain robust to camera angle changes.
    """

    @staticmethod
    def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @classmethod
    def extract_frame_features(cls, frame_data: FrameDetections) -> Dict:
        cashier = next((p for p in frame_data.poses if p.is_cashier), None)
        cash_items = [obj for obj in frame_data.objects if "cash" in obj.label]
        drawers = [obj for obj in frame_data.objects if "drawer" in obj.label]

        features = {
            "cashier_present": cashier is not None,
            "cash_count": len(cash_items),
            "open_drawer_count": sum(1 for d in drawers if d.label == "drawer_open"),
            "min_hand_to_cash_dist": None,
            "min_hand_to_drawer_dist": None,
        }

        if cashier and cashier.keypoints:
            left_wrist = cashier.keypoints.get("left_wrist")
            right_wrist = cashier.keypoints.get("right_wrist")
            hands = [pt for pt in [left_wrist, right_wrist] if pt is not None]

            if hands and cash_items:
                dists = [
                    cls.euclidean_distance((h.x, h.y), (c.bbox[0], c.bbox[1]))
                    for h in hands
                    for c in cash_items
                ]
                features["min_hand_to_cash_dist"] = min(dists) if dists else None

        return features
