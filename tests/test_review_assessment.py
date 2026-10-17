import pytest
from pydantic import ValidationError

from app.schemas.analysis import Result, ReviewAssessment


@pytest.mark.parametrize("confidence", [None, 0, 0.8, 1])
def test_confidence_never_approves_a_result(confidence):
    # Older stored results have no assessment or confidence metadata.
    violation = {
        "type": "test",
        "timecode_sec": 0,
        "description": "Observed action",
        "evidence_frames": [{"timestamp_sec": 0, "image_base64": "test"}],
    }
    if confidence is not None:
        violation["confidence"] = confidence
    result = Result.model_validate({"verdict": "violation", "violations": [violation]})
    assert result.violations[0].confidence == confidence
    assert result.violations[0].confidence_kind == "self_reported_observation"
    assert result.assessment.status == "potential_violation"
    assert result.assessment.review_required is True
    assert result.assessment.decision_status == "pending_human_review"


def test_old_empty_result_still_requires_review():
    result = Result.model_validate({"verdict": "no_violation", "violations": []})
    assert result.assessment.status == "no_supported_match"
    assert result.assessment.review_required is True


def test_automatic_review_bypass_is_rejected():
    with pytest.raises(ValidationError):
        ReviewAssessment(status="no_supported_match", review_required=False)
