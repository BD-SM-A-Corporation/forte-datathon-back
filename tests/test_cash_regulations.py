"""Production rule semantics, not tests of visual-model accuracy or legal liability."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.analysis import Observation
from app.services.events import consolidate
from app.services.inference import MockObserver, parse_observations
from app.services.pipeline import Pipeline
from app.services.rules import Catalogue, RuleEngine

CATALOGUE = Path(__file__).resolve().parents[1] / "rules/regulations.json"


@pytest.fixture
def production_rules():
    return RuleEngine.load(CATALOGUE)


def observation(rule, **updates):
    data = {
        "event_type": rule.observable_event_types[0],
        "start_sec": 1,
        "end_sec": 2,
        "description": "The visible cash-service action spans these supplied frames.",
        "objects": ["banknotes"],
        "confidence": 0.9,
        "evidence_timestamps": [1, 2],
        "facts": {name: [1, 2] for name in rule.conditions.facts_all},
    }
    data.update(updates)
    return Observation(**data)


def test_source_and_configured(production_rules):
    assert production_rules.configured
    assert len(production_rules.catalogue.rules) == 4
    assert {r.source.section for r in production_rules.catalogue.rules} == {
        "Глава 2, пункт 20",
        "Глава 2, пункт 22",
        "Глава 2, пункт 32",
    }
    assert all(r.source.url.endswith("V1900019680") for r in production_rules.catalogue.rules)
    source = json.loads((CATALOGUE.parent / "sources/V1900019680.json").read_text())
    assert source["is_actual"] is True
    assert all(
        r.source.excerpt == source["clauses"][r.source.section.split()[-1]]
        for r in production_rules.catalogue.rules
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_all_facts_required(production_rules, index):
    rule = production_rules.catalogue.rules[index]
    event = observation(rule)
    assert production_rules.match(event) == [rule]
    for missing in rule.conditions.facts_all:
        facts = {key: value for key, value in event.facts.items() if key != missing}
        incomplete = observation(rule, facts=facts)
        assert production_rules.match(incomplete) == [], missing
        assert production_rules.needs_review(incomplete)
    assert (
        production_rules.match(observation(rule, uncertainties=["Service boundary unclear"])) == []
    )
    assert production_rules.match(observation(rule, confidence=0.6)) == []


@pytest.mark.parametrize(
    "event_type",
    [
        "cash_storage",
        "cash_counting",
        "person_leaving",
        "operator_leaving_workstation",
        "object_transfer",
    ],
)
def test_action_label_alone_never_flags(production_rules, event_type):
    event = observation(production_rules.catalogue.rules[0], event_type=event_type, facts={})
    assert production_rules.match(event) == []


def test_p32_not_a_violation_even_without_visible_counting(production_rules):
    rule = production_rules.catalogue.rules[3]
    for facts in ({}, {"package_appears_intact": [1]}, {"counting_not_visible": [1]}):
        event = observation(rule, facts=facts)
        assert production_rules.match(event) == []
        assert production_rules.context(event) == [rule]
        assert production_rules.needs_review(event)


def test_context_only_is_not_a_configured_detector(production_rules):
    assert not RuleEngine(Catalogue(rules=[production_rules.catalogue.rules[3]])).configured


def test_consolidation_cannot_manufacture_combined_conditions(production_rules):
    rule = production_rules.catalogue.rules[2]
    names = rule.conditions.facts_all
    left = observation(rule, facts={name: [1] for name in names[:3]})
    right = observation(rule, facts={name: [2] for name in names[3:]})
    events = consolidate([left, right], 4)
    assert len(events) == 2
    assert all(not production_rules.match(e) for e in events)


def test_consolidation_retains_one_complete_fact_witness(production_rules):
    rule = production_rules.catalogue.rules[2]
    first = observation(rule)
    second = observation(
        rule, facts={name: [2] for name in rule.conditions.facts_all}, confidence=0.95
    )
    result = consolidate([first, second], 4)
    assert len(result) == 1
    assert result[0].facts == second.facts


def test_fact_timestamps_validated(production_rules):
    import numpy as np

    from app.services.video import Frame

    rule = production_rules.catalogue.rules[2]
    with pytest.raises(ValidationError):
        observation(rule, facts={"operator_departure_visible": [3]})
    with pytest.raises(ValidationError):
        observation(rule, facts={"operator_departure_visible": []})
    frames = [Frame(t, np.zeros((56, 56, 3), np.uint8)) for t in [1, 2]]
    event = observation(rule, facts={"operator_departure_visible": [1.5]})
    with pytest.raises(ValueError, match="grounded"):
        parse_observations(json.dumps({"events": [event.model_dump()]}), frames)


def test_default_catalogue_full_pipeline(video, settings, production_rules):
    event = observation(production_rules.catalogue.rules[2])
    output = Pipeline(settings, MockObserver([event]), production_rules).run(video)
    assert output.result.verdict == "violation"
    assert output.details["matches"][0]["rule"]["id"] == "KZ_231_P22_DEPARTURE"
    assert output.result.violations[0].evidence_frames


def test_p32_full_pipeline_is_review_context(video, settings, production_rules):
    event = observation(production_rules.catalogue.rules[3])
    output = Pipeline(settings, MockObserver([event]), production_rules).run(video)
    assert output.result.verdict == "no_violation"
    assert output.details["context_matches"][0]["rule"]["id"] == "KZ_231_P32_PACKAGING_CONTEXT"
    assert output.details["regulatory_assessment"] == "no_supported_match"
    assert output.details["review_required"] is True
