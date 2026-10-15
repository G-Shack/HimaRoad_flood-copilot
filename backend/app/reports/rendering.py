"""Language-specific presentation of identical ordered evidence references."""

import json
import re

from backend.app.copilot.offline import Language
from backend.app.copilot.rendering import _access, _card
from backend.app.reports.facts import ReportFacts, verify_report
from backend.app.reports.formatting import event_day, mapped_name, number, timestamp
from backend.app.reports.templates import text
from backend.app.schemas import Contract, EvidenceClaim


class ReportRow(Contract):
    claim_id: str
    text: str


class RenderedSection(Contract):
    id: str
    title: str
    rows: tuple[ReportRow, ...]


class RenderedReport(Contract):
    language: str
    fact_ids: tuple[str, ...]
    sections: tuple[RenderedSection, ...]
    native_review: str


def project(claim: EvidenceClaim, language: Language) -> str:
    field = claim.label.strip("/").split("/")[-1]
    if claim.kind == "bbox":
        # Coordinates and source field identifiers always remain Latin.
        return f"{claim.label}: {json.dumps(claim.value, ensure_ascii=True)}"
    if claim.kind == "event_date":
        return text("context", language) + ": " + event_day(claim.value, language)
    if claim.kind == "detector" and claim.value in {"baseline", "unet"}:
        return text("method", language) + ": " + claim.value
    if claim.kind in {"thresholds", "source", "temporal_offsets", "observation_age_hours"}:
        if re.fullmatch(r"[A-Za-z0-9_./-]{1,150}", claim.label):
            if type(claim.value) in {int, float}:
                unit = claim.unit if claim.unit != "none" else ""
                return f"{claim.label}: {number(claim.value, language)} {unit}"
            if claim.value is None:
                return f"{claim.label}: {number(None, language)}"
            if field in {"timestamp", "event_time", "snapshot", "cutoff", "max_edit_timestamp"}:
                return f"{claim.label}: {timestamp(claim.value, language)}"
    if claim.kind == "scenarios":
        meanings = {
            "1": ("Recorded asserted closures only", "अभिलेखित बन्द भनिएका खण्ड मात्र"),
            "2": ("Adds corroborated impact links", "थप प्रमाण भएका प्रभावित खण्ड समावेश"),
            "3": ("Adds observed, possible and unknown links", "अवलोकन, सम्भावित र अज्ञात खण्ड समावेश"),
        }
        if claim.parameters.get("schema_version") == "3.0":
            meanings = {
                "1": ("Recorded asserted closures only", "अभिलेखित बन्द भनिएका खण्ड मात्र"),
                "2": (
                    "Adds SAR observed-change and corroborated links (primary)",
                    "SAR अवलोकित परिवर्तन र थप प्रमाण भएका खण्ड (प्रमुख)",
                ),
                "3": (
                    "Adds corroborated links; subset of scenario2",
                    "थप प्रमाण भएका खण्ड; परिदृश्य२को उपसमूह",
                ),
                "4": (
                    "Adds possible and unknown links (worst case)",
                    "सम्भावित र अज्ञात खण्ड (अत्यन्त प्रतिकूल अवस्था)",
                ),
            }
        scenario = claim.scenario
        if scenario in meanings:
            return (
                ("Hypothetical scenario " if language == "en" else "काल्पनिक परिदृश्य ")
                + scenario
                + ": "
                + meanings[scenario][0 if language == "en" else 1]
            )
    if claim.kind == "feature":
        name, fallback = mapped_name(claim.value["properties"], language)
        # Imported names are data, never instructions or asserted facts.
        # Display as a quoted bounded string, HTML escaped by the PDF layer.
        suffix = ""
        if name:
            suffix = " " + json.dumps(name[:100], ensure_ascii=False)
            if fallback:
                suffix += " (" + text("name_fallback", language) + ")"
        if claim.label == "settlements":
            for scenario, field in [
                ("baseline", "baseline_accessible"),
                ("1", "scenario1_accessible"),
                ("2", "scenario2_accessible"),
                ("3", "scenario3_accessible"),
                *(
                    ([("4", "scenario4_accessible")])
                    if claim.parameters.get("schema_version") == "3.0"
                    else []
                ),
            ]:
                suffix += (
                    "; " + scenario + ": " + _access(claim.value["properties"].get(field), language)
                )
        return _card(claim, language).rsplit(" [", 1)[0] + suffix
    return _card(claim, language).rsplit(" [", 1)[0]


def render_report(facts: ReportFacts, language: Language) -> RenderedReport:
    verify_report(facts)
    if language not in {"en", "ne"}:
        raise ValueError("Unsupported language")
    by_id = {c.id: c for c in facts.claims}
    sections = tuple(
        RenderedSection(
            id=s.id,
            title=text(s.id, language),
            rows=tuple(
                ReportRow(claim_id=i, text=project(by_id[i], language)) for i in s.claim_ids
            ),
        )
        for s in facts.sections
    )
    return RenderedReport(
        language=language,
        fact_ids=tuple(i for s in facts.sections for i in s.claim_ids),
        sections=sections,
        native_review=facts.native_review,
    )
