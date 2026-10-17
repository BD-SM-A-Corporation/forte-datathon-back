import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from lmformatenforcer import JsonSchemaParser

from app.core.errors import InferenceError
from app.schemas.analysis import Observation
from app.services.inference import MockObserver, TimestampGroundingError
from app.services.pipeline import Pipeline
from app.services.rules import RuleEngine
from app.services.verification import (
    RuleVerifier,
    Verification,
    context_timestamps,
    parse_verification,
    rank_candidates,
    select_spread_intervals,
    verification_schema,
    verified_observation,
)
from app.services.video import Frame, Interval


@pytest.fixture
def engine():
    return RuleEngine.load(Path(__file__).resolve().parents[1] / "rules/regulations.json")


@pytest.fixture
def rule(engine):
    return engine.catalogue.rules[2]


@pytest.fixture
def candidate():
    return Observation(event_type="operator_leaving_workstation", start_sec=1, end_sec=2,
                       description="UNTRUSTED first-pass claim", objects=["banknotes"],
                       confidence=0.99, evidence_timestamps=[1, 2])


def response(rule, status="confirmed"):
    fact = {"status": status, "evidence_timestamps": [1, 2],
            "explanation": "A visible action supports this answer."}
    return {"checks": {name: dict(fact) for name in rule.conditions.facts_all},
            "episode_link": dict(fact), "confidence": 0.9, "uncertainties": []}


@pytest.fixture
def frames():
    return [Frame(t, np.zeros((56, 56, 3), np.uint8)) for t in range(4)]


def test_schema_and_parser_accept_grounded_complete_answers(rule, frames):
    raw = json.dumps(response(rule))
    parser = JsonSchemaParser(verification_schema(rule, [0, 1, 2, 3]))
    for char in raw:
        assert char in parser.get_allowed_characters()
        parser = parser.add_character(char)
    assert parser.can_end()
    assert parse_verification(raw, rule, frames).episode_link.status == "confirmed"


@pytest.mark.parametrize("status", ["unknown", "contradicted"])
def test_any_unconfirmed_fact_prevents_match(rule, candidate, status):
    data = response(rule)
    data["checks"][rule.conditions.facts_all[0]]["status"] = status
    assert verified_observation(candidate, rule, Verification(**data), [0, 1, 2, 3]) is None


@pytest.mark.parametrize("blocker", ["episode", "confidence", "uncertainty"])
def test_complete_facts_still_require_episode_and_quality(rule, candidate, blocker):
    data = response(rule)
    if blocker == "episode":
        data["episode_link"]["status"] = "unknown"
    elif blocker == "confidence":
        data["confidence"] = 0.1
    else:
        data["uncertainties"] = ["Cash identity uncertain"]
    assert verified_observation(candidate, rule, Verification(**data), [0, 1, 2, 3]) is None


@pytest.mark.parametrize("status", ["confirmed", "contradicted"])
def test_nonunknown_answers_need_evidence(rule, status):
    data = response(rule, status)
    data["checks"][rule.conditions.facts_all[0]]["evidence_timestamps"] = []
    with pytest.raises(ValueError):
        Verification(**data)


def test_missing_extra_or_ungrounded_facts_rejected(rule, frames):
    data = response(rule)
    del data["checks"][rule.conditions.facts_all[0]]
    with pytest.raises(ValueError):
        parse_verification(json.dumps(data), rule, frames)
    data = response(rule)
    data["checks"]["invented_fact"] = data["episode_link"]
    with pytest.raises(ValueError):
        parse_verification(json.dumps(data), rule, frames)
    data = response(rule)
    data["episode_link"]["evidence_timestamps"] = [22]
    with pytest.raises(TimestampGroundingError):
        parse_verification(json.dumps(data), rule, frames)


def test_verifier_uses_frames_not_the_first_pass_story(settings, candidate, rule, frames):
    captured = []

    def generate(frames, deadline, **kwargs):
        captured.append(kwargs)
        return json.dumps(response(rule))

    verifier = RuleVerifier(settings, SimpleNamespace(generate_structured=generate))
    verifier.verify(frames, rule, candidate, time.monotonic()+10)
    assert "UNTRUSTED" not in captured[0]["prompt"]
    assert "0.99" not in captured[0]["prompt"]
    assert "operator_departure_visible" in captured[0]["prompt"]
    assert captured[0]["task"] == "rule_verification"


def test_context_keeps_before_after_and_candidate_evidence(candidate):
    candidate.start_sec, candidate.end_sec = 20, 25
    candidate.evidence_timestamps = [22]
    times = context_timestamps(list(range(30)), candidate, 6, 12)
    assert len(times) == 12
    assert times[0] == 14 and times[-1] == 29
    assert 22 in times
    assert times == sorted(set(times))


def test_refinement_cap_includes_late_recording():
    windows = [Interval(i, i+2) for i in range(25)]
    selected = select_spread_intervals(windows, 4)
    assert windows[0] in selected and windows[-1] in selected
    assert len(selected) == 4


def test_rule_candidates_cover_rules_and_time(engine, candidate):
    events = [candidate.model_copy(update={"event_type": "cash_like_object_handling",
                                          "start_sec": i, "end_sec": i+1}) for i in [1, 10, 23]]
    events += [candidate.model_copy(update={"start_sec": 12, "end_sec": 13})]
    selected = rank_candidates(events, engine.catalogue.rules, 4)[:4]
    assert len({r.id for _, r in selected}) == 4
    assert any(e.start_sec == 23 for e, _ in selected)


@pytest.mark.parametrize("status,expected", [("confirmed", "violation"),
                                            ("unknown", "no_violation"),
                                            ("contradicted", "no_violation")])
def test_pipeline_verification_controls_matches(video, settings, engine, rule, candidate,
                                                status, expected):
    candidate.facts = {name: [1, 2] for name in rule.conditions.facts_all}

    class Verifier:
        def verify(self, frames, requested_rule, event, deadline):
            assert frames[0].timestamp_sec == 0
            assert frames[-1].timestamp_sec == 3
            return Verification(**response(requested_rule, status))

    output = Pipeline(settings, MockObserver([candidate]), engine, verifier=Verifier()).run(video)
    assert output.result.verdict == expected
    assert output.details["verification_summary"]["checked_count"] == 1
    assert output.details["rule_checks"][0]["facts"][0]["status"] == status
    assert output.result.assessment.review_required is True


def test_invalid_verifier_never_uses_first_pass_claim_as_fallback(video, settings, engine,
                                                               rule, candidate):
    candidate.facts = {name: [1, 2] for name in rule.conditions.facts_all}

    class Verifier:
        def verify(self, *args):
            raise InferenceError("Invalid verification")

    output = Pipeline(settings, MockObserver([candidate]), engine, verifier=Verifier()).run(video)
    assert output.result.violations == []
    assert output.details["verification_summary"]["unavailable_count"] == 1
    assert output.details["rule_checks"][0]["status"] == "unavailable"
