import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator

from app.core.errors import ConfigurationError
from app.schemas.analysis import EventType, Observation, StrictModel

logger = logging.getLogger(__name__)


class Source(StrictModel):
    document: str = Field(min_length=1)
    section: str = Field(min_length=1)
    url: str
    jurisdiction: Literal["KZ"]
    verified_on: date
    verified_by: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def official(self) -> "Source":
        url = urlparse(self.url)
        host = url.hostname or ""
        if url.scheme != "https" or not any(
            host == domain or host.endswith("." + domain)
            for domain in ("adilet.zan.kz", "gov.kz", "nationalbank.kz")
        ):
            raise ValueError("Source must be an official Kazakhstan public regulatory URL")
        if self.verified_on > date.today():
            raise ValueError("Verification cannot be dated in the future")
        return self


class Conditions(StrictModel):
    objects_all: list[str] = Field(default_factory=list)
    min_duration_sec: float = Field(0, ge=0)
    min_confidence: float = Field(0, ge=0, le=1)
    facts_all: list[EventType] = Field(default_factory=list)
    require_no_uncertainties: bool = False


class Rule(StrictModel):
    kind: Literal["potential_violation", "context"] = "potential_violation"
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source: Source
    observable_event_types: list[EventType] = Field(min_length=1)
    conditions: Conditions
    applicability: str = Field(min_length=1)
    explanation_template: str = Field(min_length=1)
    observation_guidance: str = Field(default="", max_length=700)


class Catalogue(StrictModel):
    language: Literal["ru", "en"] = "ru"
    translation_note: str = ""
    translated_from_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    rules: list[Rule]

    @model_validator(mode="after")
    def unique(self) -> "Catalogue":
        if len({r.id for r in self.rules}) != len(self.rules):
            raise ValueError("Duplicate rule IDs")
        return self


class CatalogueResponse(StrictModel):
    rules: list[Rule]
    configured: bool
    review_required: Literal[True] = True


class RuleEngine:
    def __init__(self, catalogue: Catalogue, model_catalogue: Catalogue | None = None):
        self.catalogue = catalogue
        self.model_catalogue = model_catalogue

    @classmethod
    def load(cls, path: Path, *, model_path: Path | None = None) -> "RuleEngine":
        try:
            source_bytes = path.read_bytes()
            raw = source_bytes.decode("utf-8")
        except OSError as exc:
            logger.error("Cannot read regulation catalogue %s: %s", path.absolute(), exc)
            raise ConfigurationError(
                "Regulation catalogue cannot be read; check RULES_PATH"
            ) from exc
        except UnicodeError as exc:
            logger.error("Regulation catalogue %s is not UTF-8: %s", path.absolute(), exc)
            raise ConfigurationError("Regulation catalogue must be UTF-8 encoded") from exc
        try:
            catalogue = Catalogue.model_validate_json(raw)
        except ValueError as exc:
            logger.error("Invalid regulation catalogue %s: %s", path.absolute(), exc)
            raise ConfigurationError(
                "Regulation catalogue is invalid; check logs for validation details"
            ) from exc
        model_catalogue = None
        if model_path is not None:
            model_catalogue = cls.load(model_path).catalogue
            if (
                catalogue.language != "ru"
                or model_catalogue.language != "en"
                or not model_catalogue.translation_note
                or model_catalogue.translated_from_sha256
                != hashlib.sha256(source_bytes).hexdigest()
            ):
                raise ConfigurationError(
                    "English rule translation is missing provenance or outdated; "
                    "check MODEL_RULES_PATH against RULES_PATH"
                )
            original = {rule.id: rule for rule in catalogue.rules}
            translated = {rule.id: rule for rule in model_catalogue.rules}
            if original.keys() != translated.keys():
                raise ConfigurationError("Russian and English rule IDs do not match")
            for rule_id, rule in original.items():
                counterpart = translated[rule_id]
                if (
                    rule.kind != counterpart.kind
                    or rule.observable_event_types != counterpart.observable_event_types
                    or rule.conditions != counterpart.conditions
                    or rule.source.url != counterpart.source.url
                    or rule.source.jurisdiction != counterpart.source.jurisdiction
                    or rule.source.verified_on != counterpart.source.verified_on
                    or not counterpart.observation_guidance
                ):
                    raise ConfigurationError(
                        "Russian and English rule logic or source references do not match"
                    )
        return cls(catalogue, model_catalogue)

    @property
    def configured(self) -> bool:
        return any(rule.kind == "potential_violation" for rule in self.catalogue.rules)

    def require_configured(self) -> None:
        if not self.configured:
            raise ConfigurationError(
                "Verified regulations are not configured; regulatory verdict unavailable"
            )

    def observation_guidance(self) -> str:
        """Give the observer visible action targets, not legal verdict instructions."""
        return "\n".join(
            f"- {', '.join(rule.observable_event_types)}: {rule.observation_guidance}"
            for rule in (self.model_catalogue or self.catalogue).rules
            if rule.observation_guidance
        )

    def match(self, event: Observation) -> list[Rule]:
        self.require_configured()
        return [
            rule
            for rule in self.catalogue.rules
            if rule.kind == "potential_violation"
            and event.event_type in rule.observable_event_types
            and set(rule.conditions.objects_all) <= set(event.objects)
            and event.end_sec - event.start_sec >= rule.conditions.min_duration_sec
            and event.confidence >= rule.conditions.min_confidence
            and set(rule.conditions.facts_all) <= set(event.facts)
            and (not rule.conditions.require_no_uncertainties or not event.uncertainties)
        ]

    def needs_review(self, event: Observation) -> bool:
        """A relevant action with missing context must not be presented as cleared."""
        return (
            bool(event.uncertainties)
            or bool(self.context(event))
            or (
                any(
                    rule.kind == "potential_violation"
                    and event.event_type in rule.observable_event_types
                    for rule in self.catalogue.rules
                )
                and not self.match(event)
            )
        )

    def context(self, event: Observation) -> list[Rule]:
        """Relevant source context never contributes to the jury violation verdict."""
        return [
            rule
            for rule in self.catalogue.rules
            if rule.kind == "context" and event.event_type in rule.observable_event_types
        ]


def write_schema(path: Path) -> None:
    path.write_text(json.dumps(Catalogue.model_json_schema(), indent=2) + "\n")
