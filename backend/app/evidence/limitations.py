"""Keyed deterministic limitations, with upstream text retained as source data."""

import hashlib

from backend.app.evidence.claims import make_claim, read_json, source
from backend.app.evidence.model_policy import validation_prose
from backend.app.schemas import EvidenceClaim, HimaRoadRun
from backend.app.settings import Settings

CATALOGUE = {
    "mapped_network": (
        "Mapped settlements, roads and destinations may be incomplete; unmapped assets "
        "are excluded."
    ),
    "hypothetical": (
        "Scenario closures and disconnections are hypothetical, not confirmed conditions."
    ),
    "acquisition_boundary": (
        "Satellite acquisition times differ from event time; offsets use the supplied "
        "event-date boundary."
    ),
    "synthetic": "Synthetic demonstration; not real-event evidence.",
    "synthetic_components": (
        "One or more processing components are labelled synthetic. "
        "Results are demonstrations, not real-event impact evidence."
    ),
    "unet_himalayan_candidate": (
        "UNet water is an unvalidated candidate in Himalayan valleys. "
        "Recorded model identity and development calibration do not demonstrate transfer."
    ),
    "model_evaluation": (
        "Recorded segmentation-model identity and operating point do not establish "
        "independent accuracy or confirmed impacts."
    ),
    "producer_unknown": "Producer commit is unknown; adapter pin does not identify the producer.",
    "incomplete": "Run is incomplete; impact observations and counts are unavailable.",
    "optical_absent": (
        "No optical post-event acquisition is recorded; optical corroboration is unavailable."
    ),
    "zero_scenario2_without_corroboration": (
        "Scenario 2 has zero modelled cut-offs and no radar/optical corroborated links; "
        "observed-change links remain. This does not establish absence of affected "
        "settlements."
    ),
    "terrain_observability": (
        "The reviewed contract does not establish layover or radar shadow coverage."
    ),
    "hazard3_observability": (
        "Recorded layover/shadow masks are approximate; unobservable areas remain unknown. "
        "Fused unions span several satellite dates; "
        "feature times are earliest flagging observations."
    ),
    "hazard3_optical": (
        "Optical-only change belongs to the worst-case scenario; "
        "long-gap signals are suppressed. "
        "The recorded NDBI threshold is unused; optical support does not validate possible debris."
    ),
    "flow_not_forecast": (
        "DEM flow paths are terrain traces, not inundation forecasts or travel recommendations."
    ),
    "resolution": (
        "Detection, exposure and snapping depend on recorded spatial resolution and thresholds."
    ),
    "nonallowed_source": (
        "Recorded source requires challenge compliance review; this run is not approved "
        "as compliant."
    ),
}


def limitation_claims(run: HimaRoadRun, settings: Settings) -> tuple[EvidenceClaim, ...]:
    manifest = read_json(run, settings, "manifest.json")
    report = read_json(run, settings, "report.json")
    records: list[EvidenceClaim] = []

    def add(key: str, value: object = None, refs=()):
        records.append(
            make_claim(
                run,
                kind="limitation",
                label=key,
                value={"key": key, "text": CATALOGUE[key], "details": value},
                sources=refs
                or (source(run, "report.json", "/"), source(run, "manifest.json", "/")),
                method="limitation-catalogue-v1",
                derived=True,
            )
        )

    for key in (
        "mapped_network",
        "hypothetical",
        "acquisition_boundary",
        "resolution",
    ):
        add(key)
    if run.synthetic:
        add("synthetic")
    if any(value is True for value in (run.component_synthetic or {}).values()):
        add("synthetic_components", run.component_synthetic)
    if run.model_metadata is not None:
        add("model_evaluation")
        add("unet_himalayan_candidate")
    if run.producing_commit is None:
        add("producer_unknown")
    if run.status == "incomplete":
        add("incomplete", run.reason)
    else:
        if run.schema_version == "3.0":
            add("hazard3_observability")
            add("hazard3_optical")
            add("flow_not_forecast")
        else:
            add("terrain_observability")
        if not manifest.get("optical_post"):
            add("optical_absent")
        roads = read_json(run, settings, "roads.geojson")["features"]
        corroborated = sum(f["properties"]["evidence_category"] == "likely_impact" for f in roads)
        changed = sum(f["properties"]["evidence_category"] == "observed_change" for f in roads)
        if (
            run.schema_version != "3.0"
            and run.summary
            and run.summary["potentially_road_cut"] == 0
            and corroborated == 0
        ):
            add(
                "zero_scenario2_without_corroboration",
                {"observed_change_links": changed, "corroborated_links": corroborated},
                (
                    source(run, "report.json", "/summary/potentially_road_cut"),
                    source(run, "roads.geojson", "/features"),
                ),
            )
    for key, value in run.source_datasets.items():
        upper = value.upper() if value else ""
        if any(term in upper for term in ("SRTM", "MERIT", "GEE", "COPERNICUS/S1_GRD")):
            add(
                "nonallowed_source",
                {"field": key, "source": value},
                (source(run, "manifest.json", "/" + key),),
            )
    for index, text in enumerate(report["limitations"]):
        if validation_prose(text):
            continue
        key = "upstream." + hashlib.sha256(text.encode()).hexdigest()
        records.append(
            make_claim(
                run,
                kind="limitation",
                label=key,
                value={"key": key, "text": text},
                sources=(source(run, "report.json", f"/limitations/{index}"),),
            )
        )
    for text in run.limitations:
        if text not in report["limitations"] and not validation_prose(text):
            key = "adapter." + hashlib.sha256(text.encode()).hexdigest()
            records.append(
                make_claim(
                    run,
                    kind="limitation",
                    label=key,
                    value={"key": key, "text": text},
                    sources=(source(run, "report.json", "/"), source(run, "manifest.json", "/")),
                    method="validated-adapter-condition",
                    derived=True,
                )
            )
    return tuple(sorted({c.id: c for c in records}.values(), key=lambda c: c.id))
