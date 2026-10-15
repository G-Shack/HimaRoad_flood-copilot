"""Run-to-evidence assembly; all derived output remains content-addressed."""

from backend.app.analysis.repair import repair_priority
from backend.app.analysis.routes import ReconstructionUnavailable, build_network, compute_routes
from backend.app.analysis.sensitivity import exposure_band
from backend.app.evidence.claims import build_claims, make_claim, source
from backend.app.evidence.limitations import limitation_claims
from backend.app.evidence.persisted import flow_claims, route_claims
from backend.app.evidence.validation import validate_evidence
from backend.app.schemas import EvidenceClaim, HimaRoadRun, Status
from backend.app.settings import Settings


def assemble_evidence(run: HimaRoadRun, settings: Settings) -> tuple[EvidenceClaim, ...]:
    base = build_claims(run, settings)
    claims = [*base, *limitation_claims(run, settings), *flow_claims(run, settings)]
    if run.status == "complete" and run.schema_version == "3.0":
        claims.extend(route_claims(run, settings))
        refs = tuple(
            source(run, name, "/")
            for name in (
                "routes.geojson",
                "roads.geojson",
                "settlements.geojson",
                "destinations.geojson",
                "manifest.json",
                "pre.tif",
            )
        )
        priorities = None
        reason = "Rebuilt graph disagrees with persisted routes; reopening gains unavailable."
        try:
            rebuilt = compute_routes(run, settings)
            recorded = {(c.label, c.subject, c.scenario): c for c in claims if c.kind == "route"}
            matches = all(
                (
                    original := recorded.get(
                        (mode, "settlements:" + result.settlement_id, result.scenario)
                    )
                )
                is not None
                and original.value["result"] == result.result
                and original.value["target_id"] == result.target_id
                and (
                    original.value["cost_m"] is None
                    and result.cost_m is None
                    or original.value["cost_m"] is not None
                    and result.cost_m is not None
                    and abs(original.value["cost_m"] - result.cost_m)
                    <= max(0.01, result.cost_m * 1e-6)
                )
                for mode, result in rebuilt
            )
            eligible = (
                bool(rebuilt) and matches and all(r.result != "unassessable" for _, r in rebuilt)
            )
            for _, result in rebuilt:
                if result.result == "unassessable" and len(result.assumptions) == 1:
                    reason = result.assumptions[0]
                    break
            if eligible:
                priorities = repair_priority(build_network(run, settings, "vehicle"))
        except ReconstructionUnavailable as exc:
            reason = str(exc)
        claims.append(
            make_claim(
                run,
                kind="repair_priority",
                label="single_link_reopening",
                value={
                    "availability": "available" if priorities is not None else "unavailable",
                    "rankings": [p.model_dump(mode="json") for p in priorities]
                    if priorities is not None
                    else None,
                    "reason": None if priorities is not None else reason,
                },
                sources=refs,
                scenario="2",
                method="persisted-route-crosscheck-single-link-reopening-v3",
                derived=True,
                flags=(
                    () if priorities is not None else ("reconstruction_mismatch_or_unassessable",)
                ),
            )
        )
    elif run.status == "complete":
        refs = tuple(
            source(run, name, "/")
            for name in (
                "roads.geojson",
                "settlements.geojson",
                "destinations.geojson",
                "manifest.json",
                "pre.tif",
            )
        )
        parents = tuple(
            c.id
            for c in base
            if c.kind == "feature" and c.label in {"roads", "settlements", "destinations"}
        )
        results = compute_routes(run, settings)
        baseline = {(mode, r.settlement_id): r for mode, r in results if r.scenario == "baseline"}
        for mode, result in results:
            status = Status.UNKNOWN
            origin = baseline.get((mode, result.settlement_id))
            if origin and origin.result == "none" and result.result != "unassessable":
                status = Status.BASELINE_UNREACHABLE
            elif result.result == "none" and origin and origin.result == "found":
                status = Status.MODELLED_DISCONNECTED
            elif result.result == "found":
                status = Status.MAPPED_CONNECTION
                if (
                    origin
                    and origin.cost_m is not None
                    and result.cost_m is not None
                    and result.cost_m > origin.cost_m
                ):
                    status = Status.REROUTED
            claims.append(
                make_claim(
                    run,
                    kind="route",
                    label=mode,
                    value=result.model_dump(mode="json"),
                    subject="settlements:" + result.settlement_id,
                    scenario=result.scenario,
                    status=status,
                    sources=refs,
                    method="pinned-build_graph-and-snap-directed-dijkstra-v1",
                    derived=True,
                    derived_from=parents,
                )
            )
        eligible = all(r.result != "unassessable" for mode, r in results if mode == "vehicle")
        priorities = None
        reason = "Routes are unassessable; repair gains unavailable."
        if eligible:
            try:
                priorities = repair_priority(build_network(run, settings, "vehicle"))
            except ReconstructionUnavailable as exc:
                reason = str(exc)
        claims.append(
            make_claim(
                run,
                kind="repair_priority",
                label="single_link_reopening",
                value={
                    "availability": "available" if priorities is not None else "unavailable",
                    "rankings": [p.model_dump(mode="json") for p in priorities]
                    if priorities is not None
                    else None,
                    "reason": None if priorities is not None else reason,
                },
                scenario="2",
                sources=refs,
                method="single-link-reopening-reachability-gain-v1",
                derived=True,
                derived_from=parents,
            )
        )
    for layer in ("roads", "buildings"):
        band = exposure_band(base, layer, complete=run.status == "complete")
        refs = (
            (source(run, layer + ".geojson", "/features"),)
            if run.status == "complete"
            else (source(run, "report.json", "/status"),)
        )
        claims.append(
            make_claim(
                run,
                kind="sensitivity",
                label=layer,
                value=band.model_dump(mode="json"),
                sources=refs,
                unit="count",
                method="water-versus-water-or-disturbance-unique-features-v1",
                derived=True,
                derived_from=tuple(c.id for c in base if c.kind == "feature" and c.label == layer),
            )
        )
    claims.append(
        make_claim(
            run,
            kind="external_validation",
            label="external_validation_unavailable",
            value={
                "availability": "unavailable",
                "reason": "External validation metrics have not been imported.",
            },
            sources=(source(run, "report.json", "/"),),
            method="external-validation-not-in-evidence-import",
            derived=True,
            flags=("validation_only",),
        )
    )
    validation = validate_evidence(run, tuple(claims), settings)
    claims.append(
        make_claim(
            run,
            kind="evidence_validation",
            label="evidence_integrity",
            value=validation.model_dump(mode="json"),
            sources=(source(run, "report.json", "/"), source(run, "manifest.json", "/")),
            method="evidence-integrity-validator-v1",
            derived=True,
            derived_from=tuple(c.id for c in claims),
        )
    )
    return tuple(sorted(claims, key=lambda c: c.id))
