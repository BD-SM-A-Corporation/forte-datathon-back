import pytest
from pydantic import ValidationError

from app.core.errors import ConfigurationError
from app.services.events import consolidate
from app.services.rules import Catalogue, RuleEngine


def test_matching(rules, event):
    assert rules.match(event)[0].id == "TEST_ONLY"
    assert rules.match(event.model_copy(update={"objects": []})) == []
    assert rules.match(event.model_copy(update={"confidence": 0.1})) == []
    assert rules.match(event.model_copy(update={"event_type": "phone_interaction"})) == []


def test_unconfigured(event):
    rules = RuleEngine(Catalogue(rules=[]))
    assert not rules.configured
    with pytest.raises(ConfigurationError, match="verdict unavailable"):
        rules.match(event)


def test_rule_file(rule_file):
    assert RuleEngine.load(rule_file).configured


@pytest.mark.parametrize("raw", ['{"rules": [{}]}', "not json", '{"rules": [], "unknown": 1}'])
def test_invalid_file(tmp_path, raw):
    path = tmp_path / "rules.json"
    path.write_text(raw)
    with pytest.raises(ConfigurationError):
        RuleEngine.load(path)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigurationError):
        RuleEngine.load(tmp_path / "missing.json")


def test_unknown_conditions_rejected(catalogue_data):
    catalogue_data["rules"][0]["conditions"]["invented"] = True
    with pytest.raises(ValidationError):
        Catalogue.model_validate(catalogue_data)


def test_source_rejected(catalogue_data):
    catalogue_data["rules"][0]["source"]["url"] = "https://adilet.zan.kz.evil.example/rule"
    with pytest.raises(ValidationError):
        Catalogue.model_validate(catalogue_data)


def test_duplicate_rules(catalogue_data):
    catalogue_data["rules"] *= 2
    with pytest.raises(ValidationError):
        Catalogue.model_validate(catalogue_data)


def test_multiple_events(rules, event):
    other = event.model_copy(update={"start_sec": 3, "end_sec": 4, "evidence_timestamps": [3]})
    assert len([r for e in [event, other] for r in rules.match(e)]) == 2


def test_aggregation(event):
    duplicate = event.model_copy(
        update={"start_sec": 2, "end_sec": 3, "evidence_timestamps": [2, 3]}
    )
    distinct = event.model_copy(update={"start_sec": 5, "end_sec": 6, "evidence_timestamps": [5]})
    result = consolidate([event, duplicate, distinct], 7)
    assert len(result) == 2
    assert result[0].end_sec == 3
    assert result[0].evidence_timestamps == [1, 2, 3]


def test_distinct_descriptions(event):
    other = event.model_copy(update={"description": "A different object moves into a bag."})
    assert len(consolidate([event, other], 4)) == 2


def test_invalid_boundary(event):
    with pytest.raises(ValueError):
        consolidate([event], 1)
