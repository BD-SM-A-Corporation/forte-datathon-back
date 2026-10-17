import json

import pytest
from lmformatenforcer import JsonSchemaParser

from app.services.prompts import observation_schema


def accepts(text: str, timestamps: list[float]) -> bool:
    parser = JsonSchemaParser(observation_schema(timestamps))
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


@pytest.fixture
def output():
    return {"events": [{
        "event_type": "object_transfer",
        "start_sec": 4.0,
        "end_sec": 9.0,
        "description": "An object appears to move across the counter.",
        "objects": ["unknown_object"],
        "confidence": 0.7,
        "evidence_timestamps": [4.0, 9.0],
        "facts": {"work_area_and_counterparty_visible": [4.0]},
        "uncertainties": ["Object is unclear."],
    }]}


def test_accepts_complete_grounded_observation(output):
    assert accepts(json.dumps(output), [4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert accepts('{"events": []}', [4.0, 9.0])


@pytest.mark.parametrize("field", ["start_sec", "end_sec", "evidence_timestamps", "facts"])
def test_blocks_unsupplied_timestamps_in_every_location(output, field):
    event = output["events"][0]
    if field == "facts":
        event[field]["work_area_and_counterparty_visible"] = [10.0]
    elif field == "evidence_timestamps":
        event[field] = [10.0]
    else:
        event[field] = 10.0
    assert not accepts(json.dumps(output), [4.0, 5.0, 6.0, 7.0, 8.0, 9.0])


def test_preserves_fractional_frame_times(output):
    event = output["events"][0]
    event.update(start_sec=1.033333, end_sec=2.066667,
                 evidence_timestamps=[1.033333, 2.066667], facts={})
    assert accepts(json.dumps(output), [1.033333, 2.066667])
    assert not accepts(json.dumps(output), [1.0, 2.0])


def test_rejects_incomplete_json_and_missing_required_fields():
    assert not accepts('{"events": [', [4.0, 9.0])
    assert not accepts('{"events": [{}]}', [4.0, 9.0])
