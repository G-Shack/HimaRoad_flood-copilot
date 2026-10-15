"""Trust-boundary contracts. Unavailable observations are never filled with zero."""

from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
COMMIT = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Status(StrEnum):
    OBSERVED_WATER = "observed_water"
    POSSIBLE_DEBRIS = "possible_debris"
    POTENTIALLY_AFFECTED = "potentially_affected"
    UNKNOWN = "unknown"
    NO_DETECTED_CHANGE = "no_detected_change"
    MODELLED_DISCONNECTED = "modelled_disconnected"
    BASELINE_UNREACHABLE = "baseline_unreachable"
    MAPPED_CONNECTION = "mapped_road_connection_remains"
    REROUTED = "rerouted"


class SourceRef(Contract):
    file: str
    field: str
    feature_id: str | None = None
    sha256: SHA256


class Acquisition(Contract):
    timestamp: datetime
    scene_id: str | None = None
    source: str
    relative_orbit: int | None = None
    orbit_pass: str | None = None
    platform: str | None = None


class HimaRoadRun(Contract):
    run_path: Path
    producing_commit: COMMIT | None = None
    producer_identity: Literal["recorded", "unknown"] = "unknown"
    producer_tree_dirty: bool | None = None
    producer_commit_source: Literal["git", "sync_stamp", "unavailable"] | None = None
    package_version: str | None = None
    event_time: datetime | None = None
    event_date_boundary: datetime | None = None
    temporal_offsets: dict[str, Any] | None = None
    model_metadata: dict[str, Any] | None = None
    component_synthetic: dict[str, bool | None] | None = None
    adapter_pin: COMMIT
    schema_version: str | None = None
    status: Literal["complete", "incomplete"]
    status_code: str | None = None
    reason: str | None = None
    synthetic: bool
    event_date: date
    cutoff: datetime | None = None
    bbox: tuple[float, float, float, float] | None = None
    aoi: list[Any] | dict[str, Any] | None = None
    acquisitions: tuple[Acquisition, ...] = ()
    source_datasets: dict[str, str | None] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    scenarios: dict[str, str] = Field(default_factory=dict)
    limitations: tuple[str, ...]
    files: dict[str, SHA256]
    summary: dict[str, int | float] | None = None

    @model_validator(mode="after")
    def truthful_availability(self) -> "HimaRoadRun":
        if (self.producing_commit is None) != (self.producer_identity == "unknown"):
            raise ValueError("producer identity must match recorded commit availability")
        if self.status == "incomplete":
            if self.summary is not None:
                raise ValueError("incomplete runs cannot provide impact summary counts")
            if not self.status_code or not self.reason:
                raise ValueError("incomplete runs require a status code and reason")
        elif self.bbox is None or self.aoi is None or self.summary is None:
            raise ValueError("complete runs require bbox, AOI and summary")
        return self


class EvidenceClaim(Contract):
    id: SHA256
    run_id: SHA256
    kind: str
    label: str
    value: Any
    unit: str
    subject: str
    scenario: str
    status: Status
    sources: tuple[SourceRef, ...] = Field(min_length=1)
    method: str
    parameters: dict[str, Any]
    observation_times: tuple[datetime, ...]
    quality_flags: tuple[str, ...]
    derived_by: Literal["himaroad", "copilot-analysis"]


class RouteResult(Contract):
    settlement_id: str
    scenario: str
    target_id: str | None
    cost_m: float | None = Field(ge=0)
    removed_link_ids: tuple[str, ...]
    geometry: dict[str, Any] | None
    result: Literal["found", "none", "unassessable"]
    assumptions: tuple[str, ...]


class FlowPath(Contract):
    start_point: tuple[float, float]
    dem_product: str
    dem_version: str
    geometry: dict[str, Any]
    settlements: tuple[tuple[str, float], ...]
    assumptions: tuple[str, ...]


class Answer(Contract):
    language: Literal["en", "ne"]
    segments: tuple[str, ...]
    claim_ids: tuple[SHA256, ...]
    feature_refs: tuple[str, ...]
    limitations: tuple[str, ...]
    rendered_by: Literal["llm", "template"]
    validation_result: Literal["passed", "failed"]


SectionID = Literal[
    "context", "acquisitions", "method", "coverage", "hazards", "impact", "access", "limitations"
]


class ReportSection(Contract):
    id: SectionID
    claim_ids: tuple[SHA256, ...]


class ReportFacts(Contract):
    schema_version: Literal["report-facts-1.0"] = "report-facts-1.0"
    evidence_id: SHA256
    run_id: SHA256
    availability: Literal["available", "unavailable"]
    sections: tuple[ReportSection, ...]
    claims: tuple[EvidenceClaim, ...]
    native_review: str = "not yet reviewed by a native speaker"
