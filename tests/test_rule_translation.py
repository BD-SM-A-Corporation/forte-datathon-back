import json
from pathlib import Path

import pytest

from app.core.errors import ConfigurationError
from app.schemas.analysis import Observation
from app.services.rules import RuleEngine

RULES = Path(__file__).resolve().parents[1] / "rules"


def test_english_model_catalogue_preserves_russian_matching():
    engine = RuleEngine.load(RULES / "regulations.json", model_path=RULES / "regulations.en.json")
    assert engine.catalogue.language == "ru"
    assert engine.model_catalogue.language == "en"
    assert len(engine.model_catalogue.rules) == len(engine.catalogue.rules) == 11
    assert "Unofficial" in engine.model_catalogue.translation_note
    assert not any("\u0400" <= char <= "\u04ff" for char in engine.model_catalogue.model_dump_json())
    rule = engine.catalogue.rules[0]
    event = Observation(
        event_type=rule.observable_event_types[0], start_sec=1, end_sec=2,
        description="Visible action", objects=["banknotes"], confidence=0.9,
        evidence_timestamps=[1, 2], facts={fact: [1, 2] for fact in rule.conditions.facts_all},
    )
    assert engine.match(event) == [rule]
    assert engine.match(event)[0].name.startswith("Возможный")
    assert engine.match(event)[0].source.excerpt.startswith("20. Наличные")


def test_worker_reads_guidance_from_english_file(settings, tmp_path, monkeypatch):
    from app.workers import analysis

    translated = json.loads((RULES / "regulations.en.json").read_text())
    translated["rules"][0]["observation_guidance"] = "English-only test observation target."
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(translated))
    monkeypatch.setattr(analysis, "_pipeline", None)
    monkeypatch.setattr(analysis, "Settings", lambda: settings.model_copy(update={
        "rules_path": RULES / "regulations.json", "model_rules_path": model_path,
    }))
    pipeline = analysis.get_pipeline()
    assert "English-only test observation target." in pipeline.observer.rule_guidance
    assert pipeline.rules.catalogue.rules[0].name.startswith("Возможный")


@pytest.mark.parametrize("change", ["id", "kind", "events", "conditions", "url", "provenance"])
def test_rejects_divergent_translations(tmp_path, change):
    translated = json.loads((RULES / "regulations.en.json").read_text())
    rule = translated["rules"][0]
    if change == "id":
        rule["id"] = "different_id"
    elif change == "kind":
        rule["kind"] = "context"
    elif change == "events":
        rule["observable_event_types"] = ["person_entering"]
    elif change == "conditions":
        rule["conditions"]["facts_all"] = []
    elif change == "url":
        rule["source"]["url"] = "https://adilet.zan.kz/rus/docs/V2000020075"
    else:
        translated["translated_from_sha256"] = "0" * 64
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(translated))
    with pytest.raises(ConfigurationError):
        RuleEngine.load(RULES / "regulations.json", model_path=model_path)


def test_russian_text_change_requires_translation_review(tmp_path):
    original = json.loads((RULES / "regulations.json").read_text())
    original["rules"][0]["applicability"] += " Уточнение области применения."
    path = tmp_path / "regulations.json"
    path.write_text(json.dumps(original, ensure_ascii=False))
    with pytest.raises(ConfigurationError, match="outdated"):
        RuleEngine.load(path, model_path=RULES / "regulations.en.json")


def test_missing_english_file_does_not_silently_fall_back(tmp_path):
    with pytest.raises(ConfigurationError, match="cannot be read"):
        RuleEngine.load(RULES / "regulations.json", model_path=tmp_path / "missing.json")
