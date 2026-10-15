"""Normalize recorded3.0 routes and flow paths, without computing flood paths."""

from backend.app.evidence.claims import make_claim, read_json, source
from backend.app.schemas import EvidenceClaim, HimaRoadRun, RouteResult, Status
from backend.app.settings import Settings


def removed_sets(properties: list[dict]) -> dict[str, frozenset[str]]:
    closed = {p["link_id"] for p in properties if p["operational_status"] == "confirmed_closed"}
    primary = {
        p["link_id"]
        for p in properties
        if p["operational_status"] != "confirmed_open"
        and p["evidence_category"] in {"observed_change", "likely_impact"}
    }
    corroborated = {
        p["link_id"]
        for p in properties
        if p["operational_status"] != "confirmed_open" and p["evidence_category"] == "likely_impact"
    }
    worst = {
        p["link_id"]
        for p in properties
        if p["operational_status"] != "confirmed_open"
        and (
            p["operational_status"] == "unknown"
            or p["evidence_category"] in {"possible_impact", "unobservable"}
        )
    }
    return {
        "baseline": frozenset(),
        "1": frozenset(closed),
        "2": frozenset(closed | primary),
        "3": frozenset(closed | corroborated),
        "4": frozenset(closed | primary | worst),
    }


def route_claims(run: HimaRoadRun, settings: Settings) -> tuple[EvidenceClaim, ...]:
    features = read_json(run, settings, "routes.geojson")["features"]
    roads = read_json(run, settings, "roads.geojson")["features"]
    removals = removed_sets([f["properties"] for f in roads])
    removed_ids = {scenario: tuple(sorted(ids)) for scenario, ids in removals.items()}
    baseline = {
        (p["settlement_id"], p["mode"], p["target"]): p
        for p in (f["properties"] for f in features)
        if p["scenario"] == "baseline"
    }
    claims = []
    for index, feature in enumerate(features):
        props = feature["properties"]
        origin = baseline[props["settlement_id"], props["mode"], props["target"]]
        result = RouteResult(
            settlement_id=props["settlement_id"],
            scenario=props["scenario"],
            target_id=props["destination_id"],
            cost_m=props["cost_m"],
            geometry=feature["geometry"],
            result=props["status"],
            removed_link_ids=removed_ids[props["scenario"]],
            assumptions=(
                "Persisted schema3.0 mapped route; not travel-condition or safety evidence.",
                "Scenario2 is primary observed-change; scenario3 is corroborated; "
                "scenario4 includes unknown links.",
            ),
        )
        status = Status.UNKNOWN
        if result.result != "unassessable":
            if origin["status"] == "none":
                status = Status.BASELINE_UNREACHABLE
            elif result.result == "none" and origin["status"] == "found":
                status = Status.MODELLED_DISCONNECTED
            elif result.result == "found":
                status = Status.MAPPED_CONNECTION
                if (
                    origin["cost_m"] is not None
                    and result.cost_m is not None
                    and result.cost_m > origin["cost_m"]
                ):
                    status = Status.REROUTED
        claims.append(
            make_claim(
                run,
                kind="route",
                label=props["mode"]
                if props["target"] == "town_hospital"
                else props["mode"] + "/primary_care",
                subject="settlements:" + props["settlement_id"],
                scenario=result.scenario,
                value=result.model_dump(mode="json"),
                status=status,
                sources=(
                    source(run, "routes.geojson", f"/features/{index}"),
                    source(run, "roads.geojson", "/features"),
                    source(run, "manifest.json", "/"),
                ),
                method="persisted-routes-schema3-with-conservative-status-v1",
                parameters={"target_set": props["target"], "persisted_link_ids": props["link_ids"]},
                derived=True,
            )
        )
    return tuple(claims)


def flow_claims(run: HimaRoadRun, settings: Settings) -> tuple[EvidenceClaim, ...]:
    if run.schema_version != "3.0":
        from backend.app.evidence.flow import flow_claim

        return (flow_claim(run),)
    if run.status == "incomplete" or "flowpaths.geojson" not in run.files:
        return (
            make_claim(
                run,
                kind="flow_path",
                label="flow_path_unavailable",
                value={
                    "availability": "unavailable",
                    "reason": "No admitted persisted flow path in this run",
                },
                sources=(
                    source(run, "report.json", "/status"),
                    source(run, "manifest.json", "/dem"),
                ),
                method="schema3-persisted-flow-unavailable",
                derived=True,
            ),
        )
    claims = []
    for index, feature in enumerate(read_json(run, settings, "flowpaths.geojson")["features"]):
        props = feature["properties"]
        claims.append(
            make_claim(
                run,
                kind="flow_path",
                label=props["kind"],
                subject="flowpaths:" + props["id"],
                value={"availability": "available", "geometry": feature["geometry"], **props},
                sources=(
                    source(run, "flowpaths.geojson", f"/features/{index}", props["id"]),
                    source(run, "manifest.json", "/dem"),
                ),
                method=props.get("method", "persisted-flow-corridor"),
                flags=("not_an_inundation_forecast",),
            )
        )
    return tuple(claims)
