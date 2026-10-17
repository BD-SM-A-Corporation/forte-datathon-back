from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Time = Annotated[float, Field(ge=0, allow_inf_nan=False)]
EventType = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Observation(StrictModel):
    event_type: EventType
    start_sec: Time
    end_sec: Time
    description: str = Field(min_length=1, max_length=1500)
    objects: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(max_length=20)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evidence_timestamps: list[Time] = Field(min_length=1, max_length=3)

    # Positive, visually supported facts only. Missing facts mean unknown, never false.
    facts: dict[EventType, Annotated[list[Time], Field(min_length=1, max_length=3)]] = Field(
        default_factory=dict, max_length=16
    )
    uncertainties: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(
        default_factory=list, max_length=8
    )

    @field_validator("objects")
    @classmethod
    def normalize_objects(cls, values: list[str]) -> list[str]:
        return sorted({value.strip().lower().replace(" ", "_") for value in values})

    @model_validator(mode="after")
    def boundaries(self) -> "Observation":
        if self.end_sec < self.start_sec:
            raise ValueError("Reversed event interval")
        if any(t < self.start_sec or t > self.end_sec for t in self.evidence_timestamps):
            raise ValueError("Evidence outside event")
        if any(
            t < self.start_sec or t > self.end_sec for times in self.facts.values() for t in times
        ):
            raise ValueError("Fact evidence outside event")
        return self


class Observations(StrictModel):
    events: list[Observation] = Field(max_length=30)


class EvidenceFrame(StrictModel):
    timestamp_sec: Time
    image_base64: str


class Violation(StrictModel):
    type: str
    timecode_sec: Time
    description: str
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class Result(StrictModel):
    verdict: Literal["violation", "no_violation"]
    violations: list[Violation]


class JobResponse(StrictModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["violation", "no_violation"] | None = None
    violations: list[Violation] | None = None
    error: str | None = None


class AnalyzeResponse(StrictModel):
    job_id: str
