"""Shared ordered report facts. External validation is excluded from impact reports."""

from backend.app.copilot.tools import EvidenceContext, ToolRejected, tool_definitions
from backend.app.reports.native_review import review_notice
from backend.app.schemas import EvidenceClaim, ReportFacts, ReportSection, SectionID


def section_for(claim: EvidenceClaim) -> SectionID | None:
    from backend.app.evidence.model_policy import MODEL_FIELDS, validation_prose

    if "model_metadata/" in claim.label:
        leaf = claim.label.split("model_metadata/", 1)[1].split("/")[0]
        if leaf not in MODEL_FIELDS:
            return None
    if (
        claim.kind == "limitations"
        or claim.kind == "source"
        and claim.label.startswith("/limitations/")
    ) and validation_prose(claim.value):
        return None
    if (
        claim.kind == "limitation"
        and isinstance(claim.value, dict)
        and validation_prose(claim.value.get("text"))
        and claim.method == "upstream-field"
    ):
        return None
    if claim.kind == "external_validation" or "validation_only" in claim.quality_flags:
        return None
    if claim.kind == "limitation":
        return "limitations"
    if claim.kind == "source":
        root = claim.label.strip("/").split("/")[0]
        return (
            "acquisitions"
            if root
            in {
                "pre",
                "post",
                "optical_pre",
                "optical_post",
                "temporal_gaps_hours",
                "temporal_offsets",
                "pairing",
                "osm",
                "cutoff",
            }
            else "method"
        )
    if claim.kind in {
        "event_date",
        "event_time",
        "bbox",
        "aoi",
        "status",
        "status_code",
        "reason",
        "synthetic",
        "evidence_validation",
    }:
        return "context"
    if claim.kind in {"observation_timestamp", "temporal_offsets", "observation_age_hours"}:
        return "acquisitions"
    if claim.kind == "feature" and claim.label == "settlements":
        return "access"
    if claim.kind == "feature" and claim.label in {
        "flood_extent",
        "disturbance",
        "possible_debris",
        "unobservable",
    }:
        return "hazards"
    if claim.kind == "area":
        return "hazards"
    if claim.kind == "sensitivity" or claim.label.endswith("_percent"):
        return "coverage"
    if claim.kind in {"summary", "feature"}:
        return "impact"
    if claim.kind in {"scenarios", "route", "primary_care_access", "repair_priority", "flow_path"}:
        return "access"
    return "method"


def build_report(context: EvidenceContext, authorised_ids: frozenset[str]) -> ReportFacts:
    if context.evidence_id not in authorised_ids:
        raise ToolRejected("run_not_authorised")
    tool_definitions(context)  # Snapshot digest, membership and content hashes.
    integrity = [c for c in context.claims if c.kind == "evidence_validation"]
    if len(integrity) != 1:
        raise ValueError("Report requires exactly one integrity claim")
    failed = integrity[0].value["result"] == "failed"
    incomplete = any(c.kind == "status" and c.value == "incomplete" for c in context.claims)
    ordered = sorted(context.claims, key=lambda c: (c.kind, c.label, c.subject, c.scenario, c.id))
    selected = tuple(
        c
        for c in ordered
        if section_for(c) is not None
        and (
            not failed
            or (
                c.kind in {"limitation", "evidence_validation"}
                and c.label != "zero_scenario2_without_corroboration"
            )
        )
        and (
            not (failed or incomplete)
            or c.kind
            in {
                "limitation",
                "evidence_validation",
                "status",
                "status_code",
                "reason",
                "event_date",
                "event_time",
                "bbox",
                "aoi",
                "synthetic",
                "source",
                "detector",
                "thresholds",
                "producer_commit",
                "schema_version",
            }
        )
    )
    sections = tuple(
        ReportSection(id=key, claim_ids=tuple(c.id for c in selected if section_for(c) == key))
        for key in (
            "context",
            "acquisitions",
            "method",
            "coverage",
            "hazards",
            "impact",
            "access",
            "limitations",
        )
    )
    return ReportFacts(
        native_review=review_notice(),
        evidence_id=context.evidence_id,
        run_id=selected[0].run_id,
        availability="unavailable" if failed or incomplete else "available",
        sections=sections,
        claims=selected,
    )


def verify_report(facts: ReportFacts) -> None:
    from backend.app.evidence.claims import verify_claim

    ids = [c.id for c in facts.claims]
    assigned = [i for s in facts.sections for i in s.claim_ids]
    if len(ids) != len(set(ids)) or sorted(ids) != sorted(assigned):
        raise ValueError("Report section membership differs from claims")
    if tuple(s.id for s in facts.sections) != (
        "context",
        "acquisitions",
        "method",
        "coverage",
        "hazards",
        "impact",
        "access",
        "limitations",
    ):
        raise ValueError("Invalid report section order")
    membership = {s.id: set(s.claim_ids) for s in facts.sections}
    for c in facts.claims:
        verify_claim(c)
        section = section_for(c)
        if c.run_id != facts.run_id or section is None:
            raise ValueError("Foreign or validation-only report fact")
        if c.id not in membership[section]:
            raise ValueError("Claim assigned to wrong section")
    integrity = [c for c in facts.claims if c.kind == "evidence_validation"]
    if len(integrity) != 1:
        raise ValueError("Missing integrity result")
    unavailable = integrity[0].value["result"] == "failed" or any(
        c.kind == "status" and c.value == "incomplete" for c in facts.claims
    )
    if integrity[0].value["result"] == "failed" and any(
        (
            c.kind not in {"limitation", "evidence_validation"}
            or c.label == "zero_scenario2_without_corroboration"
        )
        for c in facts.claims
    ):
        raise ValueError("Failed-integrity report contains facts")
    if unavailable != (facts.availability == "unavailable"):
        raise ValueError("Availability differs from evidence")
    if unavailable and any(
        c.kind in {"summary", "area", "feature", "route", "sensitivity"} for c in facts.claims
    ):
        raise ValueError("Unavailable report contains impact facts")
