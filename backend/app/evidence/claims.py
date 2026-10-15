"""Content-addressed claims from an already validated, unchanged run."""

import hashlib
import json
from typing import Any

import numpy as np
from rasterio.io import MemoryFile

from backend.app.evidence.model_policy import MODEL_FIELDS, validation_prose
from backend.app.evidence.vocabulary import map_status
from backend.app.schemas import EvidenceClaim, HimaRoadRun, SourceRef, Status
from backend.app.settings import Settings
from backend.app.storage import canonical_run
from backend.app.upstream import contract
from backend.app.upstream.paths import RunContractError, authorised_path

METHOD_VERSION = "evidence-v3"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def run_id(run: HimaRoadRun) -> str:
    return hashlib.sha256(canonical_run(run)).hexdigest()


def read_bytes(run: HimaRoadRun, settings: Settings, name: str) -> bytes:
    if name not in run.files or name.startswith("input:"):
        raise RunContractError(f"{name}: not a validated run output")
    path = authorised_path(run.run_path / name, settings, root=run.run_path)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != run.files[name]:
        raise RunContractError(f"{name}: changed after validation")
    return content


def read_json(run: HimaRoadRun, settings: Settings, name: str) -> dict[str, Any]:
    return json.loads(read_bytes(run, settings, name))


def source(run: HimaRoadRun, name: str, field: str, feature: str | None = None) -> SourceRef:
    return SourceRef(file=name, field=field, feature_id=feature, sha256=run.files[name])


def make_claim(
    run: HimaRoadRun,
    *,
    kind: str,
    label: str,
    value: Any,
    unit: str = "none",
    subject: str = "run",
    scenario: str = "observation",
    status: Status = Status.UNKNOWN,
    sources: tuple[SourceRef, ...],
    method: str = "upstream-field",
    parameters: dict[str, Any] | None = None,
    flags: tuple[str, ...] = (),
    derived: bool = False,
    derived_from: tuple[str, ...] = (),
) -> EvidenceClaim:
    quality = set(flags)
    if run.synthetic:
        quality.add("synthetic")
    if run.producing_commit is None:
        quality.add("producer_commit_unknown")
    if run.producer_tree_dirty:
        quality.add("producer_tree_dirty")
    if run.status == "incomplete":
        quality.add("incomplete_run")
    if scenario not in {"observation", "baseline"}:
        quality.add("hypothetical")
    for name, synthetic in (run.component_synthetic or {}).items():
        if synthetic is True:
            quality.add(f"synthetic_{name}")
    params = {
        "method_version": METHOD_VERSION,
        "producer_commit": run.producing_commit,
        "adapter_pin": run.adapter_pin,
        "schema_version": run.schema_version,
        "source_datasets": run.source_datasets,
        "producer_tree_dirty": run.producer_tree_dirty,
        "producer_commit_source": run.producer_commit_source,
        "package_version": run.package_version,
        "analysis": parameters or {},
        "derived_from": sorted(set(derived_from)),
    }
    if run.model_metadata is not None or run.component_synthetic is not None:
        params["method_version"] = "evidence-v3-model-provenance-v1"
        params["component_synthetic"] = run.component_synthetic
        params["model_identity"] = (
            {
                key: run.model_metadata[key]
                for key in (
                    "checkpoint_sha256",
                    "model_version",
                    "preprocessing_id",
                    "operating_threshold",
                    "temperature",
                    "synthetic",
                )
            }
            if run.model_metadata
            else None
        )
    times = tuple(sorted({a.timestamp for a in run.acquisitions}))
    if run.schema_version == "3.0":
        from backend.app.upstream.v1 import time

        params["method_version"] = "evidence-hazard3-v1"
        quality.update({"hazard_v2", "possible_debris_unvalidated"})
        if kind == "feature" and isinstance(value, dict):
            props = value["properties"]
            stamps = props.get("sar_post_timestamps") or (
                [props["observation_timestamp"]] if props.get("observation_timestamp") else []
            )
            times = tuple(sorted({time(stamp) for stamp in stamps}))
    draft = EvidenceClaim(
        id="0" * 64,
        run_id=run_id(run),
        kind=kind,
        label=label,
        value=value,
        unit=unit,
        subject=subject,
        scenario=scenario,
        status=status,
        sources=sources,
        method=method,
        parameters=params,
        observation_times=times,
        quality_flags=tuple(sorted(quality)),
        derived_by="copilot-analysis" if derived else "himaroad",
    )
    digest = hashlib.sha256(canonical(draft.model_dump(mode="json", exclude={"id"}))).hexdigest()
    return draft.model_copy(update={"id": digest})


def verify_claim(claim: EvidenceClaim) -> None:
    expected = hashlib.sha256(canonical(claim.model_dump(mode="json", exclude={"id"}))).hexdigest()
    if claim.id != expected:
        raise ValueError("claim content does not match ID")


def _leaves(value: Any, field: str):
    if isinstance(value, dict) and value:
        for key in sorted(value):
            yield from _leaves(value[key], f"{field}/{key}")
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from _leaves(item, f"{field}/{index}")
    else:
        yield field, value


def build_claims(run: HimaRoadRun, settings: Settings) -> tuple[EvidenceClaim, ...]:
    if run.schema_version not in {None, "1.0", "3.0"}:
        raise RunContractError(f"Unreviewed output schema_version: {run.schema_version}")
    report = read_json(run, settings, "report.json")
    manifest = read_json(run, settings, "manifest.json")
    claims: list[EvidenceClaim] = []
    report_fields = (
        "status",
        "status_code",
        "reason",
        "event_date",
        "bbox",
        "aoi",
        "detector",
        "synthetic",
        "observation_timestamp",
        "evidence_sources",
        "observation_age_hours",
        "source_dataset",
        "limitations",
        "thresholds",
        "scenarios",
        "summary",
        "priority_inspection",
        "producer_commit",
        "producer_tree_dirty",
        "producer_commit_source",
        "package_version",
        "event_time",
        "temporal_offsets",
        "model_metadata",
        "schema_version",
    )
    if run.schema_version == "3.0":
        report_fields += (
            "hazard",
            "hazard_version",
            "sar_geometry",
            "hazard_quality_flags",
            "primary_scenario",
        )
    for key in report_fields:
        if key not in report or (key == "summary" and run.status == "incomplete"):
            continue
        for field, value in _leaves(report[key], f"/{key}"):
            if key == "model_metadata" and field.split("/")[2] not in MODEL_FIELDS:
                continue
            if key == "limitations" and validation_prose(value):
                continue
            scenario = "2" if key in {"summary", "priority_inspection"} else "observation"
            if key == "scenarios":
                scenario = field.split("/")[2]
            if run.schema_version == "3.0" and key == "summary":
                leaf = field.rsplit("/", 1)[-1]
                scenario = {
                    "corroborated_road_cut": "3",
                    "depends_only_on_unknown": "4",
                    "settlements_total": "observation",
                    "settlements_from_building_clusters": "observation",
                }.get(leaf, "2")
            unit = (
                "count"
                if key == "summary"
                and field.split("/")[-1]
                in (*contract.SUMMARY_COUNTS, "primary_care_potentially_cut")
                else "none"
            )
            if field.endswith("_percent"):
                unit = "percent"
            claims.append(
                make_claim(
                    run,
                    kind=key,
                    label=field,
                    value=value,
                    unit=unit,
                    scenario=scenario,
                    sources=(source(run, "report.json", field),),
                )
            )
    for key in (
        "pre",
        "post",
        "optical_pre",
        "optical_post",
        "dem",
        "hand",
        "osm",
        "cutoff",
        "configuration",
        "temporal_gaps_hours",
        "software_versions",
        "processing_timestamp",
        "provider",
        "pairing",
        "road_segmentation",
        "limitations",
        "temporal_offsets",
        "model_metadata",
        *(
            ("tracks", "hazard", "sar_geometry", "hazard_quality_flags", "settlement_model")
            if run.schema_version == "3.0"
            else ()
        ),
    ):
        if key in manifest:
            for field, value in _leaves(manifest[key], f"/{key}"):
                if key == "model_metadata" and field.split("/")[2] not in MODEL_FIELDS:
                    continue
                if key == "limitations" and validation_prose(value):
                    continue
                claims.append(
                    make_claim(
                        run,
                        kind="source",
                        label=field,
                        value=value,
                        sources=(source(run, "manifest.json", field),),
                        flags=("offset_from_supplied_date_boundary",)
                        if key == "temporal_gaps_hours"
                        else (),
                    )
                )
    if run.status == "complete":
        for layer in contract.vectors(run.schema_version):
            name = f"{layer}.geojson"
            features = read_json(run, settings, name)["features"]
            ids: set[str] = set()
            for index, feature in enumerate(features):
                props = feature["properties"]
                asset_id = props.get("id", f"feature/{index}")
                if not isinstance(asset_id, str) or asset_id in ids:
                    raise RunContractError(f"{name}: missing or duplicate string feature ID")
                ids.add(asset_id)
                if layer in {"roads", "affected_infrastructure", "buildings"}:
                    status = map_status("evidence", props["evidence_category"])
                elif layer == "settlements":
                    status = map_status("settlement", props["category"])
                elif layer in {
                    "flood_extent",
                    "disturbance",
                    "unobservable",
                    "optical_water",
                    "optical_disturbance",
                    "possible_debris",
                }:
                    status = map_status("hazard", layer)
                else:
                    status = Status.UNKNOWN
                if layer == "settlements" and run.schema_version in {"1.0", "3.0"}:
                    baseline = props["primary_care_baseline_accessible"]
                    care_status = Status.UNKNOWN
                    if baseline is False:
                        care_status = Status.BASELINE_UNREACHABLE
                    elif baseline is True:
                        care_status = (
                            Status.MODELLED_DISCONNECTED
                            if props["primary_care_scenario2_accessible"] is False
                            else Status.MAPPED_CONNECTION
                        )
                    claims.append(
                        make_claim(
                            run,
                            kind="primary_care_access",
                            label="primary_care",
                            value={
                                k: props[k]
                                for k in (
                                    "primary_care_baseline_accessible",
                                    "primary_care_scenario1_accessible",
                                    "primary_care_scenario2_accessible",
                                    "primary_care_scenario3_accessible",
                                    "primary_care_coverage",
                                )
                            },
                            subject=f"settlements:{asset_id}",
                            scenario="2",
                            status=care_status,
                            sources=(source(run, name, f"/features/{index}", asset_id),),
                            method="primary-care-mapped-reachability-not-service-availability",
                        )
                    )
                # Raw physical/operational strings are producer assertions, not verified condition.
                public = {
                    k: v
                    for k, v in props.items()
                    if k
                    not in {
                        "physical_condition",
                        "operational_status",
                        "category",
                        "evidence_category",
                    }
                }
                public["status"] = status.value
                if "operational_status" in props:
                    public["operational_status"] = map_status(
                        "operational", props["operational_status"]
                    ).value
                scenario = "2" if layer == "settlements" else "observation"
                claims.append(
                    make_claim(
                        run,
                        kind="feature",
                        label=layer,
                        value={"properties": public, "geometry": feature["geometry"]},
                        subject=f"{layer}:{asset_id}",
                        scenario=scenario,
                        status=status,
                        sources=(source(run, name, f"/features/{index}", asset_id),),
                        method="upstream-feature-with-conservative-status-mapping",
                    )
                )
        for layer in (
            "flood_extent",
            "disturbance",
            "unobservable",
            "optical_water",
            "optical_disturbance",
            *(("possible_debris",) if run.schema_version == "3.0" else ()),
        ):
            name = f"{layer}.tif"
            with MemoryFile(read_bytes(run, settings, name)) as memory, memory.open() as raster:
                data = raster.read(1, masked=True)
                covered = ~np.ma.getmaskarray(data) & np.isfinite(data.filled(np.nan))
                count = int(np.count_nonzero(covered & (data.filled(0) > 0)))
                pixel_m2 = abs(
                    raster.transform.a * raster.transform.e
                    - raster.transform.b * raster.transform.d
                )
                value = {
                    "area_m2": count * pixel_m2,
                    "observed_area_m2": int(covered.sum()) * pixel_m2,
                }
                if not covered.any():
                    value = {"area_m2": None, "observed_area_m2": 0.0}
                claims.append(
                    make_claim(
                        run,
                        kind="area",
                        label=layer,
                        value=value,
                        unit="m2",
                        status=map_status("hazard", layer),
                        sources=(source(run, name, "band/1"),),
                        method="positive-finite-pixels-times-affine-determinant",
                        parameters={"crs": str(raster.crs), "pixel_area_m2": pixel_m2},
                        derived=True,
                        flags=("no_observation",) if not covered.any() else (),
                    )
                )
    return tuple(sorted(claims, key=lambda c: c.id))
