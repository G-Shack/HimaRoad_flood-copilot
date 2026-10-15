"""Explicit partial display projection; full exports retain the complete facts."""

from collections import Counter

from backend.app.reports.facts import verify_report
from backend.app.schemas import Contract, ReportFacts, ReportSection

MAX_LAYER_FEATURES = 100
MAX_VERTICES = 20_000
MAX_ROUTES = 20


class EvidenceView(Contract):
    facts: ReportFacts
    total_claims: int
    displayed_claims: int
    partial: bool
    settlements: tuple[str, ...]


def vertices(value) -> int:
    if not isinstance(value, (list, tuple)):
        return 0
    if len(value) >= 2 and all(type(v) in {int, float} for v in value):
        return 1
    return sum(vertices(v) for v in value)


def evidence_view(facts: ReportFacts, selected: str | None = None) -> EvidenceView:
    verify_report(facts)
    return _project_verified_view(facts, selected)


def _project_verified_view(facts: ReportFacts, selected: str | None = None) -> EvidenceView:
    """Server-only projection of build_report output from an immutable context.

    The API verifies the complete SQLite snapshot before freezing it; requests
    cannot supply ReportFacts here. The smaller returned projection is rechecked.
    """
    chosen = []
    counts = Counter()
    remaining = MAX_VERTICES
    routes = 0
    settlements = []
    for c in sorted(facts.claims, key=lambda c: c.id != selected):
        if c.kind == "feature" and c.label == "settlements":
            settlements.append(c.subject)
        geometry = c.value.get("geometry") if isinstance(c.value, dict) else None
        cost = vertices(geometry.get("coordinates")) if geometry else 0
        if c.kind == "feature":
            if counts[c.label] >= MAX_LAYER_FEATURES or cost > remaining:
                continue
            counts[c.label] += 1
            remaining -= cost
        elif c.kind == "route":
            if routes >= MAX_ROUTES or cost > remaining:
                continue
            routes += 1
            remaining -= cost
        elif c.kind == "flow_path":
            if cost > remaining:
                continue
            remaining -= cost
        chosen.append(c)
    # A selected citation is fetched separately with exact provenance; membership
    # never authorizes an outside claim, and large geometries never exceed budgets.
    ids = {c.id for c in chosen}
    chosen = [c for c in facts.claims if c.id in ids]
    projection = facts.model_copy(
        update={
            "claims": tuple(chosen),
            "sections": tuple(
                ReportSection(id=s.id, claim_ids=tuple(i for i in s.claim_ids if i in ids))
                for s in facts.sections
            ),
        }
    )
    verify_report(projection)
    return EvidenceView(
        facts=projection,
        total_claims=len(facts.claims),
        displayed_claims=len(chosen),
        partial=len(chosen) != len(facts.claims),
        settlements=tuple(sorted(set(settlements))),
    )
