import asyncio
from datetime import datetime
from pathlib import Path
from sqlalchemy.future import select
from app.config import settings
from app.database import async_session
from app.models import Job, Violation, JobStatus, Verdict
from app.pipeline.video import VideoDecoder
from app.pipeline.detector import Detector
from app.pipeline.tracker import MultiObjectTracker
from app.pipeline.features import FeatureExtractor
from app.pipeline.classifier import ComplianceClassifier
from app.pipeline.evidence import EvidenceRenderer


async def process_video_job(job_id: str):
    """
    Asynchronously processes a video analysis job from beginning to end.
    """
    async with async_session() as session:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return

        job.status = JobStatus.PROCESSING
        job.progress = 5
        await session.commit()

        video_path = Path(job.file_path)
        if not video_path.exists():
            job.status = JobStatus.FAILED
            job.error = f"Video file not found at: {video_path}"
            await session.commit()
            return

        try:
            decoder = VideoDecoder(target_fps=settings.FPS_SAMPLE_RATE)
            detector = Detector(weights_dir=settings.BASE_DIR / settings.WEIGHTS_DIR, device=settings.DEVICE)
            tracker = MultiObjectTracker()
            classifier = ComplianceClassifier(fps=settings.FPS_SAMPLE_RATE)

            frame_features = []
            cached_frames = {}

            # Process frames
            for idx, timestamp, frame in decoder.extract_frames(video_path):
                # Detection and tracking
                dets = detector.process_frame(frame, idx, timestamp)
                tracked_dets = tracker.update(dets)

                # Feature extraction
                feats = FeatureExtractor.extract_frame_features(tracked_dets)
                frame_features.append({
                    "frame_idx": idx,
                    "timestamp": timestamp,
                    "features": feats
                })

                # Cache every few frames for potential evidence export
                if idx % 5 == 0:
                    cached_frames[idx] = (timestamp, frame)

            # Evaluate state machine rules
            verdict, violations_detected = classifier.evaluate_sequence(frame_features)

            # Store detected violations with evidence
            for v in violations_detected:
                frame_filename = f"{job_id}_{v.event_type}_{int(v.timestamp_sec)}.jpg"
                frame_dest = settings.BASE_DIR / settings.FRAMES_DIR / frame_filename

                sample_frame = cached_frames.get(v.frame_idx)
                if sample_frame:
                    t, img = sample_frame
                    annotated = EvidenceRenderer.draw_evidence(img, t, v.title)
                    EvidenceRenderer.save_evidence_frame(annotated, frame_dest)

                violation_record = Violation(
                    job_id=job.id,
                    event_type=v.event_type,
                    title=v.title,
                    description=v.description,
                    timestamp_sec=v.timestamp_sec,
                    end_timestamp_sec=v.end_timestamp_sec,
                    confidence=v.confidence,
                    frame_path=str(frame_dest) if frame_dest.exists() else None,
                    rule_source=v.rule_source,
                    rule_section=v.rule_section,
                    rule_url=v.rule_url,
                )
                session.add(violation_record)

            job.status = JobStatus.COMPLETED
            job.verdict = verdict
            job.progress = 100
            job.completed_at = datetime.utcnow()
            await session.commit()

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            await session.commit()
