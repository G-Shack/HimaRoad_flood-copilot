"""Pure query tools return cited claims and applicable limitations."""

from typing import Literal

from backend.app.schemas import Contract, EvidenceClaim, Status


class QueryResult(Contract):
    availability: Literal["available", "unavailable"]
    claims: tuple[EvidenceClaim, ...]
    limitations: tuple[EvidenceClaim, ...]
    coverage: tuple[EvidenceClaim, ...]


def _result(
    claims: tuple[EvidenceClaim, ...],
    selected: tuple[EvidenceClaim, ...],
    *,
    available: bool = True,
) -> QueryResult:
    if len({c.run_id for c in claims}) != 1:
        raise ValueError("queries require exactly one run")
    integrity = tuple(c for c in claims if c.kind == "evidence_validation")
    if len(integrity) != 1:
        raise ValueError("queries require one evidence integrity result")
    failed = integrity[0].value["result"] == "failed"
    if failed:
        selected = integrity
        available = False
    return QueryResult(
        availability="available" if available else "unavailable",
        claims=selected,
        limitations=tuple(c for c in claims if c.kind == "limitation"),
        coverage=() if failed else tuple(c for c in claims if c.kind == "sensitivity"),
    )


def summarize_run(claims: tuple[EvidenceClaim, ...]) -> QueryResult:
    return _result(
        claims,
        tuple(
            c
            for c in claims
            if c.kind
            in {
                "summary",
                "status",
                "status_code",
                "reason",
                "detector",
                "event_date",
                "observation_timestamp",
                "scenarios",
                "sensitivity",
                "primary_care_access",
            }
        ),
    )


def list_affected_assets(
    claims: tuple[EvidenceClaim, ...], layer: Literal["roads", "buildings"] = "roads"
) -> QueryResult:
    return _result(
        claims,
        tuple(
            c
            for c in claims
            if c.kind == "feature" and c.label == layer and c.status == Status.POTENTIALLY_AFFECTED
        ),
        available=any(c.kind == "feature" for c in claims)
        or any(c.kind == "summary" for c in claims),
    )


def list_potentially_isolated_settlements(claims: tuple[EvidenceClaim, ...]) -> QueryResult:
    return _result(
        claims,
        tuple(
            c
            for c in claims
            if c.kind == "feature"
            and c.label == "settlements"
            and c.status == Status.MODELLED_DISCONNECTED
        ),
        available=any(c.kind == "summary" for c in claims),
    )


def inspect_feature(claims: tuple[EvidenceClaim, ...], subject: str) -> QueryResult:
    selected = tuple(c for c in claims if c.subject == subject and c.kind in {"feature", "route"})
    return _result(claims, selected, available=bool(selected))


def get_route(
    claims: tuple[EvidenceClaim, ...],
    settlement_id: str,
    scenario: str = "2",
    mode: Literal["vehicle", "pedestrian"] = "vehicle",
) -> QueryResult:
    selected = tuple(
        c
        for c in claims
        if c.kind == "route"
        and c.label == mode
        and c.subject == "settlements:" + settlement_id
        and c.scenario == scenario
    )
    return _result(
        claims,
        selected,
        available=bool(selected) and all(c.value["result"] != "unassessable" for c in selected),
    )


def get_repair_priority(claims: tuple[EvidenceClaim, ...]) -> QueryResult:
    selected = tuple(c for c in claims if c.kind == "repair_priority")
    return _result(
        claims,
        selected,
        available=bool(selected) and all(c.value["availability"] == "available" for c in selected),
    )


def trace_flow_path(claims: tuple[EvidenceClaim, ...]) -> QueryResult:
    selected = tuple(c for c in claims if c.kind == "flow_path")
    return _result(
        claims,
        selected,
        available=bool(selected) and all(c.value["availability"] == "available" for c in selected),
    )


def get_sources_and_limitations(claims: tuple[EvidenceClaim, ...]) -> QueryResult:
    return _result(
        claims, tuple(c for c in claims if c.kind in {"source", "detector", "limitation"})
    )


def get_validation_results(
    claims: tuple[EvidenceClaim, ...], validation: EvidenceClaim | None = None
) -> QueryResult:
    """External model evaluation only; never label integrity checks mapping accuracy."""
    selected = tuple(c for c in claims if c.kind == "external_validation")
    if validation is not None:
        from backend.app.evidence.claims import verify_claim

        verify_claim(validation)
        if (
            validation.kind != "external_validation"
            or "validation_only" not in validation.quality_flags
            or validation.run_id not in {c.run_id for c in claims}
        ):
            raise ValueError("unlabelled or wrong-run validation presentation")
        selected = (validation,)
    return _result(
        claims,
        selected,
        available=bool(selected) and all(c.value["availability"] == "available" for c in selected),
    )


def get_evidence_validation(claims: tuple[EvidenceClaim, ...]) -> QueryResult:
    selected = tuple(c for c in claims if c.kind == "evidence_validation")
    return _result(claims, selected, available=bool(selected))


def get_primary_care_access(claims: tuple[EvidenceClaim, ...]) -> QueryResult:
    selected = tuple(c for c in claims if c.kind == "primary_care_access")
    return _result(claims, selected, available=bool(selected))


def get_hazard_extent(claims: tuple[EvidenceClaim, ...]) -> QueryResult:
    return _result(
        claims,
        tuple(
            c
            for c in claims
            if c.kind in {"area", "feature"}
            and c.label in {"flood_extent", "disturbance", "unobservable"}
        ),
        available=any(c.kind == "area" for c in claims),
    )
