"""Isolated display of HimaRoad evaluation summaries; never part of evidence import.

Only metrics.json is read. Raw references and paths named inside metrics are
never opened. The caller must record exposure before parsing the summary.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import Field

from backend.app.evidence.claims import make_claim, source
from backend.app.schemas import SHA256, Contract, EvidenceClaim, HimaRoadRun, SourceRef
from backend.app.settings import Settings
from backend.app.upstream.paths import RunContractError, authorised_path

Fraction = Annotated[float, Field(ge=0, le=1)]
Count = Annotated[int, Field(ge=0, strict=True)]


class FloodMetrics(Contract):
    iou: Fraction | None
    precision: Fraction | None
    recall: Fraction | None
    f1: Fraction | None
    true_positive: Count
    false_positive: Count
    false_negative: Count
    evaluated_pixels: Count
    valid_area_coverage: Fraction


class InfrastructureMetrics(Contract):
    reference_features: Count
    matched_reference_features: Count
    unmatched_reference_features: Count
    matching_coverage: Fraction
    affected_road_recall: Fraction | None
    false_alarm_reference_features: Count | None
    reliable_negative_features: Count
    unobservable_road_fraction: Fraction


class DisplayMetrics(Contract):
    reference_source: str
    reference_observation_timestamp: datetime
    reference_manifest_sha256: SHA256
    limitations: tuple[str, ...]
    thresholds: dict[str, float]
    flood: FloodMetrics | None = None
    infrastructure: InfrastructureMetrics | None = None
    # Isolation has no validation ground truth in the reviewed satellite workflow.
    connectivity: dict[str, str | float | int | None]
    runtime_seconds: Annotated[float, Field(ge=0)]


class AgreementMetric(Contract):
    value: Annotated[float, Field(strict=True, ge=0, le=1)] | None
    availability: Literal["available", "unavailable"]
    unit: Literal["fraction"] = "fraction"
    reason: str | None = None


def case_study_metrics(raw: dict) -> dict:
    """Registered positive-footprint aggregates only; never matching labels.

    No unavailable count/score is converted to zero. A zero in-grid footprint
    cannot establish agreement. Unsupported precision/IoU remain unavailable.
    """
    if raw.get("purpose") != "evaluation":
        raise ValueError("evaluation purpose required")
    disturbance = raw["disturbance"]
    water = raw["water"]
    if water.get("iou") is not None:
        raise ValueError("landslide comparison cannot establish water IoU")
    known_negatives = disturbance.get("known_negatives", False)
    if not known_negatives and any(
        disturbance.get(k) is not None for k in ("precision", "iou", "f1")
    ):
        raise ValueError("positive-only comparison cannot establish precision/IoU/F1")
    assessed = disturbance.get("observed_assessed_reference_area_m2")
    available = disturbance.get("status") == "agreement" and assessed is not None and assessed > 0
    recall = disturbance.get("area_recall") if available else None
    if available and recall is None:
        raise ValueError("assessed footprint lacks area recall")
    metrics = {
        "landslide_footprint_area_recall": AgreementMetric(
            value=recall,
            availability="available" if recall is not None else "unavailable",
            reason=None
            if recall is not None
            else disturbance.get(
                "reason", "No observed assessed reference footprint inside prediction grid"
            ),
        ),
        "water_iou": AgreementMetric(
            value=None, availability="unavailable", reason=water["reason"]
        ),
    }
    for name in ("precision", "iou", "f1"):
        value = disturbance.get(name) if available and known_negatives else None
        metrics["disturbance_" + name] = AgreementMetric(
            value=value,
            availability="available" if value is not None else "unavailable",
            reason=None if value is not None else "No explicit negative assessment coverage",
        )
    roads = raw["infrastructure"]
    recall = roads.get("affected_road_recall")
    metrics["affected_road_recall"] = AgreementMetric(
        value=recall,
        availability="available" if recall is not None else "unavailable",
        reason=None if recall is not None else roads.get("reason", "No assessed affected roads"),
    )
    return {name: metric.model_dump(mode="json") for name, metric in metrics.items()}


def load_validation_presentation(
    run: HimaRoadRun, settings: Settings, exposure_log: Path
) -> EvidenceClaim | None:
    """Call only after freezing evidence; no output feeds mapping or evidence code."""
    path = authorised_path(run.run_path / "metrics.json", settings, root=run.run_path)
    if not path.exists():
        return None
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    stamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M IST")
    # Record even malformed or drifted summaries. No reference geometry is opened.
    with exposure_log.open("a") as log:
        log.write(
            f"\n| {stamp} | codex validation presentation | metrics.json SHA256 {digest} | "
            f"Evaluation summary parsed for display (synthetic run: {run.synthetic}) | "
            "No raw reference files opened; derived matching labels may be present | "
            "Display only; evidence frozen before viewing; no tuning |\n"
        )
    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise RunContractError("metrics.json: expected object")
    # Matches are reference-associated labels, not impact facts; never expose them to Q&A.
    infrastructure = raw.get("infrastructure")
    if isinstance(infrastructure, dict):
        raw["infrastructure"] = {k: v for k, v in infrastructure.items() if k != "matches"}
    parsed = DisplayMetrics.model_validate(raw)
    if parsed.reference_observation_timestamp.tzinfo is None:
        raise RunContractError("metrics.json: reference timestamp must have timezone")
    return make_claim(
        run,
        kind="external_validation",
        label="himaroad_evaluation_summary",
        value={
            "availability": "available",
            "metrics": parsed.model_dump(mode="json"),
            "scope": "Validation only; does not establish confirmed damage or isolation.",
        },
        sources=(
            SourceRef(file="metrics.json", field="/", sha256=digest),
            source(run, "report.json", "/"),
            source(run, "manifest.json", "/"),
        ),
        method="reviewed-legacy-evaluation-summary-display-only",
        derived=True,
        flags=("validation_only", "evaluation_identity_by_run_directory_only"),
    )
