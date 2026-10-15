"""Read-only import of pinned HimaRoad outputs, with explicit unavailable provenance."""

import hashlib
import json
import math
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from backend.app.schemas import Acquisition, HimaRoadRun
from backend.app.settings import Settings
from backend.app.upstream import contract as c
from backend.app.upstream.model import model_provenance
from backend.app.upstream.paths import RunContractError, authorised_path

PIN_FILE = Path(__file__).resolve().parents[3] / "upstream/himaroad.ref"
EVALUATION_NAMES = {"metrics.json", "comparison.md", "reference.json"}


def _require(obj: dict[str, Any], field: str, kind: type, where: str) -> Any:
    if field not in obj:
        raise RunContractError(f"{where}.{field}: required field missing")
    value = obj[field]
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise RunContractError(f"{where}.{field}: expected {kind.__name__}")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise RunContractError(f"{path.name}: unreadable JSON") from exc
    if not isinstance(value, dict):
        raise RunContractError(f"{path.name}: expected object")
    return value


def _hash(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise RunContractError(f"{path.name}: required file unavailable") from exc


def _time(value: str, where: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("timezone missing")
        return result
    except (ValueError, AttributeError) as exc:
        raise RunContractError(f"{where}: expected timezone-aware timestamp") from exc


def _path(value: str, base: Path, settings: Settings, *, root: Path | None = None) -> Path:
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else base / candidate
    result = authorised_path(candidate, settings, root=root)
    if result.name in EVALUATION_NAMES:
        raise RunContractError("evaluation files are forbidden in evidence imports")
    return result


def _check_paths(value: Any, base: Path, settings: Settings) -> None:
    """Authorise every declared input before any declared input is opened."""
    if isinstance(value, dict):
        if value.get("purpose") == "evaluation":
            raise RunContractError("evaluation-purpose inputs are forbidden")
        for key, item in value.items():
            if key in {"source", "source_dataset", "provider"} and isinstance(item, str):
                if re.search(
                    r"\b(?:EMSR\d*|UNOSAT|GFM)\b|COPERNICUS\s+(?:EMS\b|EMERGENCY)",
                    item,
                    re.IGNORECASE,
                ):
                    raise RunContractError(
                        "validation-only source is forbidden in evidence imports"
                    )
            if key == "path" and isinstance(item, str):
                _path(item, base, settings)
            _check_paths(item, base, settings)
    elif isinstance(value, list):
        for item in value:
            _check_paths(item, base, settings)


def _vector(path: Path, layer: str, schema_version: str | None = None) -> list[str]:
    limitations: list[str] = []
    obj = _json(path)
    if obj.get("type") != "FeatureCollection":
        raise RunContractError(f"{path.name}.type: expected FeatureCollection")
    crs = obj.get("crs")
    if isinstance(crs, dict) and not isinstance(crs.get("properties"), dict):
        raise RunContractError(f"{path.name}.crs: expected CRS properties object")
    if crs and (
        not isinstance(crs, dict)
        or crs.get("properties", {}).get("name")
        not in {
            "EPSG:4326",
            "urn:ogc:def:crs:OGC:1.3:CRS84",
            "urn:ogc:def:crs:EPSG::4326",
        }
    ):
        raise RunContractError(f"{path.name}.crs: expected WGS84")
    features = _require(obj, "features", list, path.name)
    for i, feature in enumerate(features):
        where = f"{path.name}.features[{i}]"
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise RunContractError(f"{where}: expected Feature")
        props = _require(feature, "properties", dict, where)
        geometry = _require(feature, "geometry", dict, where)
        from backend.app.upstream.v3 import numeric_coordinates

        if not numeric_coordinates(geometry.get("coordinates")):
            raise RunContractError(f"{where}.geometry: coordinates must be finite numbers")
        from shapely.geometry import shape

        try:
            geom = shape(geometry)
            if geom.is_empty:
                raise ValueError("invalid geometry")
            if not geom.is_valid:
                if geom.geom_type == "LineString" and geom.length == 0:
                    limitations.append(
                        f"Zero-length upstream link {props.get('id', i)} in {path.name}; "
                        "route suitability has not been assessed."
                    )
                else:
                    raise ValueError("invalid geometry")
            x0, y0, x1, y1 = geom.bounds
            if not (-180 <= x0 <= x1 <= 180 and -90 <= y0 <= y1 <= 90):
                raise ValueError("non-WGS84 coordinates")
        except (ValueError, KeyError, TypeError) as exc:
            raise RunContractError(f"{where}.geometry: invalid WGS84 geometry") from exc
        fields = (
            c.ASSET_FIELDS if layer in {"roads", "affected_infrastructure", "buildings"} else ()
        )
        if layer in {"roads", "affected_infrastructure"}:
            fields += (
                "osm_id",
                "link_id",
                "highway",
                "layer",
                "source_dataset",
                "evidence_sources",
            )
        if layer == "settlements":
            fields = c.SETTLEMENT_FIELDS
        if layer == "destinations":
            fields = ("id", "place", "name")
        if schema_version in {"1.0", "3.0"}:
            from backend.app.upstream.v1 import validate_properties

            validate_properties(layer, props)
            if layer in {"roads", "affected_infrastructure"} and geom.length == 0:
                raise RunContractError("contract 1.0: zero-length road exported")
        for field in fields:
            if field not in props:
                raise RunContractError(f"{where}.properties.{field}: required field missing")
        if "evidence_category" in props and props["evidence_category"] not in c.EVIDENCE:
            raise RunContractError(f"{where}.properties.evidence_category: unknown status")
        if "operational_status" in props and props["operational_status"] not in {
            "unknown",
            "confirmed_open",
            "confirmed_closed",
        }:
            raise RunContractError(f"{where}.properties.operational_status: unknown status")
        if layer == "settlements" and props["category"] not in c.SETTLEMENT_CATEGORIES:
            raise RunContractError(f"{where}.properties.category: unknown status")
        if "valid_observation_fraction" in props:
            fraction = props["valid_observation_fraction"]
            if type(fraction) not in {int, float} or not 0 <= fraction <= 1:
                raise RunContractError(
                    f"{where}.properties.valid_observation_fraction: expected [0,1]"
                )
        for key in fields:
            if key.endswith("accessible") or key in {
                "potentially_exposed",
                "baseline_incomplete",
                "potentially_isolated",
                "depends_on_unknown_links",
                "flood_intersection",
                "disturbance_intersection",
            }:
                _require(props, key, bool, f"{where}.properties")
            if key.endswith("_ids"):
                ids = _require(props, key, list, f"{where}.properties")
                if any(not isinstance(item, str) for item in ids):
                    raise RunContractError(f"{where}.properties.{key}: expected string IDs")

    return limitations


def load_run(path: Path, settings: Settings) -> HimaRoadRun:
    """Validate complete or incomplete output; never replace unavailable counts with zero."""
    run = authorised_path(path, settings)
    report_path = _path("report.json", run, settings, root=run)
    manifest_path = _path("manifest.json", run, settings, root=run)
    bundle = None
    if (run / "bundle.json").exists() or (run / "bundle.json").is_symlink():
        from backend.app.upstream.bundle import verify_bundle

        bundle = verify_bundle(run, settings)
    report, manifest = _json(report_path), _json(manifest_path)
    schema_version = manifest.get("schema_version", report.get("schema_version"))
    if (
        "schema_version" in manifest
        and "schema_version" in report
        and manifest["schema_version"] != report["schema_version"]
    ):
        raise RunContractError("report/manifest.schema_version: conflicting versions")
    if schema_version is not None and not isinstance(schema_version, str):
        raise RunContractError("schema_version must be a string")
    if schema_version not in {None, "1.0", "3.0"}:
        raise RunContractError(f"Unreviewed output schema_version: {schema_version}")
    model_metadata, component_synthetic = model_provenance(report, manifest)
    if schema_version in {"1.0", "3.0"} and bundle is None:
        raise RunContractError("versioned contracts require a verified portable bundle")
    if bundle is not None:
        for key in ("schema_version", "producer_commit", "synthetic"):
            if bundle.get(key) != report.get(key):
                raise RunContractError(f"bundle/report: conflicting {key}")
    _check_paths(manifest, run, settings)
    status = _require(report, "status", str, "report.json")
    if status not in {"complete", "incomplete"}:
        raise RunContractError("report.json.status: expected complete or incomplete")
    synthetic = _require(report, "synthetic", bool, "report.json")
    if "synthetic" in manifest and manifest["synthetic"] is not synthetic:
        raise RunContractError("manifest.json.synthetic: differs from report")
    event = _require(report, "event_date", str, "report.json")
    try:
        event_date = (
            _time(event, "report.event_date").astimezone(UTC).date()
            if "T" in event
            else date.fromisoformat(event)
        )
    except ValueError as exc:
        raise RunContractError("report.json.event_date: expected ISO date") from exc
    if manifest.get("event_date", event) != event:
        raise RunContractError("manifest.json.event_date: differs from report")
    limitations = _require(report, "limitations", list, "report.json")
    if any(not isinstance(item, str) for item in limitations):
        raise RunContractError("report.json.limitations: expected strings")
    source_datasets: dict[str, str | None] = {}
    for key in ("dem", "hand", "post", "optical_pre", "optical_post", "osm"):
        record = manifest.get(key)
        if isinstance(record, dict) and isinstance(record.get("source"), str):
            source_datasets[key] = record["source"]
    recorded_pre = manifest.get("pre", [])
    if not isinstance(recorded_pre, list):
        raise RunContractError("manifest.json.pre: expected list")
    for i, record in enumerate(recorded_pre):
        if isinstance(record, dict) and isinstance(record.get("source"), str):
            source_datasets[f"pre[{i}]"] = record["source"]
    for key in ("dem", "hand"):
        record = manifest.get(key)
        if not record:
            continue
        source = source_datasets.get(key)
        upper = source.upper() if source else ""
        if "SYNTHETIC" in upper:
            limitations.append(f"{key} is labelled synthetic; it is not real-event evidence.")
        elif "SRTM" in upper or "MERIT" in upper:
            limitations.append(
                f"Recorded {key} source {source} is not an allowed challenge input; "
                "this run requires compliance review."
            )
        elif "COPERNICUS" not in upper:
            limitations.append(
                f"Recorded {key} source is unverified; challenge input compliance requires review."
            )
    files = {"report.json": _hash(report_path), "manifest.json": _hash(manifest_path)}
    exports = report.get("exports", {})
    if not isinstance(exports, dict):
        raise RunContractError("report.json.exports: expected object")
    # Authorise export and OSM paths before opening any declared file.
    for value in exports.values():
        if isinstance(value, str):
            _path(value, run, settings, root=run)
    osm_paths = manifest.get("osm", {})
    if isinstance(osm_paths, dict):
        for key in ("roads", "buildings", "settlements", "destinations"):
            if isinstance(osm_paths.get(key), str):
                _path(osm_paths[key], run, settings)
    for name, value in exports.items():
        if not isinstance(value, str):
            raise RunContractError(f"report.json.exports.{name}: expected path")
        export = _path(value, run, settings, root=run)
        files[export.relative_to(run).as_posix()] = _hash(export)
    osm = manifest.get("osm", {})
    if not isinstance(osm, dict):
        raise RunContractError("manifest.json.osm: expected object")
    for name in ("roads", "buildings", "settlements", "destinations"):
        if name in osm:
            if not isinstance(osm[name], str):
                raise RunContractError(f"manifest.json.osm.{name}: expected path")
            _path(osm[name], run, settings)
    inputs = manifest.get("inputs", [])
    if not isinstance(inputs, list):
        raise RunContractError("manifest.json.inputs: expected list")
    input_hashes: dict[Path, str] = {}
    for i, item in enumerate(inputs):
        where = f"manifest.json.inputs[{i}]"
        if not isinstance(item, dict):
            raise RunContractError(f"{where}: expected object")
        source = _path(_require(item, "path", str, where), run, settings)
        expected = _require(item, "sha256", str, where)
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RunContractError(f"{where}.sha256: expected SHA256")
        if _hash(source) != expected:
            raise RunContractError(f"{where}.sha256: checksum mismatch")
        input_hashes[source] = expected
        files[f"input:{source.relative_to(settings.himaroad_data_dir).as_posix()}"] = expected

    def check_declared_inputs(value: Any) -> None:
        if isinstance(value, dict):
            if "path" in value:
                source = _path(_require(value, "path", str, "manifest.json"), run, settings)
                if source not in input_hashes:
                    raise RunContractError(
                        f"manifest.json: declared input {source.name} "
                        "absent from checksummed inputs"
                    )
                if "sha256" in value and value["sha256"] != input_hashes[source]:
                    raise RunContractError(f"manifest.json: conflicting checksum for {source.name}")
            for item in value.values():
                check_declared_inputs(item)
        elif isinstance(value, list):
            for item in value:
                check_declared_inputs(item)

    check_declared_inputs(manifest)
    summary = None
    bbox = report.get("bbox", manifest.get("bbox"))
    aoi = report.get("aoi", bbox)
    acquisitions = []
    if status == "complete":
        for field, kind in (
            ("bbox", list),
            ("thresholds", dict),
            ("scenarios", dict),
            ("summary", dict),
        ):
            _require(report, field, kind, "report.json")
        if not isinstance(report.get("aoi"), (list, dict)):
            raise RunContractError("report.json.aoi: expected bounds or geometry")
        for field in ("pre", "post", "osm", "inputs", "configuration", "software_versions"):
            if field not in manifest:
                raise RunContractError(f"manifest.json.{field}: required field missing")
        summary = report["summary"]
        for field in c.SUMMARY_COUNTS:
            if _require(summary, field, int, "report.json.summary") < 0:
                raise RunContractError(f"report.json.summary.{field}: negative count")
        for field in c.SUMMARY_PERCENTAGES:
            value = summary.get(field)
            if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 100:
                raise RunContractError(f"report.json.summary.{field}: expected percentage [0,100]")
        for scenario in ("1", "2", "3"):
            _require(report["scenarios"], scenario, str, "report.json.scenarios")
        for name in (
            "map.html",
            "report.pdf",
            "satellite_context.png",
            "satellite_context.json",
            *(f"{layer}.kml" for layer in c.vectors(schema_version)),
        ):
            output = _path(name, run, settings, root=run)
            files[name] = _hash(output)
        for name in c.vectors(schema_version):
            vector = _path(f"{name}.geojson", run, settings, root=run)
            limitations.extend(_vector(vector, name, schema_version))
            files[f"{name}.geojson"] = _hash(vector)
        import rasterio
        from rasterio.errors import RasterioError

        grid = None
        for name in c.rasters(schema_version):
            raster = _path(f"{name}.tif", run, settings, root=run)
            try:
                with rasterio.open(raster) as dataset:
                    if not dataset.crs or not dataset.crs.is_projected:
                        raise RunContractError(f"{raster.name}.crs: expected projected metric CRS")
                    if dataset.crs.linear_units != "metre":
                        raise RunContractError(f"{raster.name}.crs: expected metre units")
                    current = (dataset.crs, dataset.transform, dataset.width, dataset.height)
                    if grid is not None and current != grid:
                        raise RunContractError(f"{raster.name}: raster grid differs from pre.tif")
                    grid = current
            except RasterioError as exc:
                raise RunContractError(f"{raster.name}: unavailable or invalid raster") from exc
            files[f"{name}.tif"] = _hash(raster)
        snapshot = _time(
            _require(osm, "snapshot", str, "manifest.json.osm"), "manifest.json.osm.snapshot"
        )
        event_midnight = (
            _time(event, "event_date")
            if "T" in event
            else datetime.fromisoformat(f"{event_date}T00:00:00+00:00")
        )
        if snapshot >= event_midnight:
            raise RunContractError("manifest.json.osm.snapshot: must precede event date")
        pre = _require(manifest, "pre", list, "manifest.json")
        if not pre:
            raise RunContractError("manifest.json.pre: expected at least one acquisition")
        scenes = pre + [_require(manifest, "post", dict, "manifest.json")]
        scenes += [manifest[k] for k in ("optical_pre", "optical_post") if manifest.get(k)]
        for i, scene in enumerate(scenes):
            where = f"manifest.json.acquisitions[{i}]"
            if not isinstance(scene, dict):
                raise RunContractError(f"{where}: expected object")
            stamp = _time(_require(scene, "timestamp", str, where), where)
            try:
                acquisition = Acquisition(
                    timestamp=stamp,
                    scene_id=scene.get("scene_id"),
                    source=_require(scene, "source", str, where),
                    relative_orbit=scene.get("relative_orbit"),
                    orbit_pass=scene.get("pass"),
                    platform=scene.get("platform"),
                )
            except ValidationError as exc:
                raise RunContractError(f"{where}: invalid acquisition metadata") from exc
            acquisitions.append(acquisition)
        for i, scene in enumerate(pre):
            if _time(scene["timestamp"], "manifest.json.pre.timestamp") >= event_midnight:
                raise RunContractError(f"manifest.json.pre[{i}].timestamp: must precede event date")
        post_time = _time(manifest["post"]["timestamp"], "manifest.json.post.timestamp")
        event_midnight = (
            _time(event, "event_date")
            if "T" in event
            else datetime.fromisoformat(f"{event_date}T00:00:00+00:00")
        )
        if post_time <= event_midnight:
            raise RunContractError(
                "manifest.json.post.timestamp: must follow event-date midnight UTC"
            )
        if manifest.get("cutoff") and post_time > _time(manifest["cutoff"], "manifest.json.cutoff"):
            raise RunContractError("manifest.json.post.timestamp: exceeds cutoff")
        for field in ("relative_orbit", "pass", "instrument_mode", "polarization", "units"):
            expected = manifest["post"].get(field)
            if expected is None or any(scene.get(field) != expected for scene in pre):
                raise RunContractError(f"manifest.json.pre/post.{field}: missing or mismatched")
    else:
        _require(report, "status_code", str, "report.json")
        _require(report, "reason", str, "report.json")
        if report.get("summary"):
            raise RunContractError("report.json.summary: incomplete run must not expose counts")
    if bbox is not None and (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(type(v) not in {int, float} or not math.isfinite(v) for v in bbox)
        or not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90)
    ):
        raise RunContractError("report.json.bbox: expected WGS84 bounds")
    schema_version = manifest.get("schema_version", report.get("schema_version"))
    if (
        "schema_version" in manifest
        and "schema_version" in report
        and manifest["schema_version"] != report["schema_version"]
    ):
        raise RunContractError("report/manifest.schema_version: conflicting versions")
    metadata = {}
    if schema_version == "1.0":
        from backend.app.upstream.v1 import validate_metadata, verify_osm_proof

        try:
            metadata = validate_metadata(report, manifest)
            if status == "complete":
                verify_osm_proof(run, manifest, settings)
        except (KeyError, TypeError) as exc:
            raise RunContractError("contract 1.0: malformed metadata") from exc
        if status == "complete":
            if (
                _require(report["summary"], "primary_care_potentially_cut", int, "report.summary")
                < 0
            ):
                raise RunContractError("contract 1.0: negative primary-care count")
        files["bundle.json"] = _hash(run / "bundle.json")
    if schema_version == "3.0":
        from backend.app.upstream.v3 import validate_metadata as validate_v3

        try:
            metadata, scenes = validate_v3(run, report, manifest, settings)
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            if isinstance(exc, RunContractError):
                raise
            raise RunContractError("contract 3.0: malformed recorded fields") from exc
        files["bundle.json"] = _hash(run / "bundle.json")
        if status == "complete":
            files["routes.geojson"] = _hash(run / "routes.geojson")
            if (run / "flowpaths.geojson").exists():
                files["flowpaths.geojson"] = _hash(run / "flowpaths.geojson")
            acquisitions = [
                Acquisition(
                    timestamp=_time(scene["timestamp"], "scene.timestamp"),
                    source=scene["source"],
                    scene_id=scene.get("scene_id"),
                    relative_orbit=scene.get("relative_orbit"),
                    orbit_pass=scene.get("pass"),
                    platform=scene.get("platform"),
                )
                for scene in scenes
            ]
    producer = metadata.pop("producing_commit", manifest.get("producing_commit"))
    cutoff = _time(manifest["cutoff"], "manifest.json.cutoff") if manifest.get("cutoff") else None
    try:
        return HimaRoadRun(
            run_path=run,
            **metadata,
            producing_commit=producer,
            producer_identity="recorded" if producer else "unknown",
            adapter_pin=PIN_FILE.read_text().strip(),
            schema_version=schema_version,
            status=status,
            status_code=report.get("status_code"),
            reason=report.get("reason"),
            synthetic=synthetic,
            model_metadata=model_metadata,
            component_synthetic=component_synthetic,
            event_date=event_date,
            cutoff=cutoff,
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]) if bbox is not None else None,
            aoi=aoi,
            acquisitions=tuple(acquisitions),
            thresholds=report.get("thresholds", {}),
            scenarios=report.get("scenarios", {}),
            limitations=tuple(limitations),
            source_datasets=source_datasets,
            files=files,
            summary=summary,
        )
    except (ValidationError, OSError) as exc:
        raise RunContractError("run provenance: invalid typed metadata or adapter pin") from exc
