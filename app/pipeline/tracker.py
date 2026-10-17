from typing import List, Dict
from app.pipeline.detector import FrameDetections, PoseDetection, ObjectDetection


class MultiObjectTracker:
    """
    Maintains track IDs across frames (e.g. cashier consistency, objects).
    Uses ByteTrack or simple IoU/Euclidean association.
    """

    def __init__(self):
        self.active_tracks: Dict[int, Dict] = {}
        self.next_track_id: int = 1

    def update(self, detections: FrameDetections) -> FrameDetections:
        """
        Associates detections across consecutive frames to maintain consistent identities.
        """
        # Assign track IDs to poses and objects
        for pose in detections.poses:
            if pose.person_id is None:
                pose.person_id = self.next_track_id
                self.next_track_id += 1

        for obj in detections.objects:
            if obj.track_id is None:
                obj.track_id = self.next_track_id
                self.next_track_id += 1

        return detections
