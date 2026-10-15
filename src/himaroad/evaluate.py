"""Held-out evaluation only; never called by inference."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from .common import REPO_ROOT, checksum, data_root, now, producer_provenance, read_config, utc, write_json
from .preprocess import align
from .validation_registry import begin_evaluation, committed_config


def binary_metrics(predicted, reference, valid):
    p, r = predicted[valid].astype(bool), reference[valid].astype(bool)
    tp, fp, fn = int((p & r).sum()), int((p & ~r).sum()), int((~p & r).sum())

    def ratio(a, b):
        return a / b if b else None

    return dict(
        iou=ratio(tp, tp + fp + fn),
        precision=ratio(tp, tp + fp),
        recall=ratio(tp, tp + fn),
        f1=ratio(2 * tp, 2 * tp + fp + fn),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        evaluated_pixels=int(valid.sum()),
    )


def evaluate(run, reference):
    start = perf_counter()
    run, reference = Path(run), Path(reference)
    spec = read_config(
        reference / "reference.json" if reference.is_dir() else reference
    )
    root = reference if reference.is_dir() else reference.parent
    if (
        spec.get("purpose") != "evaluation"
        or not spec.get("observation_timestamp")
        or not spec.get("source")
    ):
        raise ValueError(
            "Reference must record purpose=evaluation, source and observation_timestamp"
        )
    results = {
        "reference_source": spec["source"],
        "reference_observation_timestamp": spec["observation_timestamp"],
        "reference_manifest_sha256": checksum(
            reference / "reference.json" if reference.is_dir() else reference
        ),
        "limitations": [
            "Reference interpretation is independent satellite evidence, not guaranteed ground truth. Unassessed assets are never negatives. Road recall associates exposed bounded road segments with reference geometry; it is not confirmed physical damage."
        ],
        "thresholds": {"road_match_m": float(spec.get("road_match_m", 20))},
    }
    with rasterio.open(run / "flood_extent.tif") as src:
        profile = src.profile
        predicted = src.read(1)
        valid = np.isfinite(predicted)
    if spec.get("flood"):
        labels = align(root / spec["flood"], profile, categorical=True)
        assessment = (
            align(root / spec["coverage"], profile, categorical=True)
            if spec.get("coverage")
            else np.isfinite(labels)
        )
        eligible = valid & np.isin(labels, [0, 1]) & (assessment == 1)
        results["flood"] = binary_metrics(predicted > 0, labels == 1, eligible)
        results["flood"]["valid_area_coverage"] = float(
            eligible.sum() / max(1, int((assessment == 1).sum()))
        )
    if spec.get("roads"):
        refs = gpd.read_file(root / spec["roads"]).to_crs(profile["crs"])
        roads = gpd.read_file(run / "roads.geojson").to_crs(profile["crs"])
        if "assessment" not in refs:
            raise ValueError(
                "Reference roads require assessment=affected|unaffected|unknown"
            )
        if not refs["assessment"].isin(["affected", "unaffected", "unknown"]).all():
            raise ValueError("Invalid reference road assessment")
        tolerance = results["thresholds"]["road_match_m"]
        matches = []
        matched = set()
        for i, row in refs.iterrows():
            candidates = roads[
                roads.geometry.intersects(row.geometry.buffer(tolerance))
            ]
            ids = []
            for j, road in candidates.iterrows():
                # Require appreciable linear overlap, not a nearby road crossing.
                if row.geometry.geom_type in {"Point", "MultiPoint"}:
                    if str(road.get("bridge", "no")).lower() not in {
                        "no",
                        "none",
                        "nan",
                        "",
                    }:
                        ids.append(str(road["id"]))
                        matched.add(str(road["id"]))
                    continue
                overlap = road.geometry.intersection(
                    row.geometry.buffer(tolerance)
                ).length
                if (
                    overlap >= min(road.geometry.length, row.geometry.length) * 0.5
                    and overlap > 0
                ):
                    ids.append(str(road["id"]))
                    matched.add(str(road["id"]))
            matches.append(
                dict(
                    reference_id=str(row.get("id", i)),
                    assessment=row["assessment"],
                    link_ids=ids,
                    osm_ids=sorted(
                        set(roads.loc[roads["id"].isin(ids), "osm_id"].astype(str))
                    )
                    if "osm_id" in roads
                    else ids,
                )
            )
        affected = [
            m for m in matches if m["assessment"] == "affected" and m["link_ids"]
        ]
        negatives = [
            m for m in matches if m["assessment"] == "unaffected" and m["link_ids"]
        ]
        predicted_ids = set(
            roads.loc[roads["potentially_exposed"].astype(bool), "id"].astype(str)
        )
        recalled = sum(bool(set(m["link_ids"]) & predicted_ids) for m in affected)
        results["infrastructure"] = dict(
            reference_features=len(refs),
            matched_reference_features=sum(bool(m["link_ids"]) for m in matches),
            unmatched_reference_features=sum(not m["link_ids"] for m in matches),
            matching_coverage=sum(bool(m["link_ids"]) for m in matches)
            / max(1, len(matches)),
            affected_road_recall=recalled / len(affected) if affected else None,
            false_alarm_reference_features=sum(
                bool(set(m["link_ids"]) & predicted_ids) for m in negatives
            )
            if negatives
            else None,
            reliable_negative_features=len(negatives),
            unobservable_road_fraction=float(
                (roads["evidence_category"] == "unobservable").sum()
                / max(1, len(roads))
            ),
            matches=matches,
        )
    results["connectivity"] = {
        "status": "not_scored",
        "reason": "Independent closure observations required; satellite-derived reference maps cannot validate inferred isolation.",
    }
    if spec.get("closures"):
        closure = read_config(root / spec["closures"])
        if not spec.get("closure_source"):
            raise ValueError("Closure labels require an independent closure_source")
        roads = gpd.read_file(run / "roads.geojson")
        id_column = "osm_id" if "osm_id" in roads else "id"
        likely = set(
            roads.loc[roads["evidence_category"] == "likely_impact", id_column].astype(
                str
            )
        )
        mapped = set(roads[id_column].astype(str))
        closed = {
            str(item["osm_id"])
            for item in closure
            if item["status"] == "confirmed_closed"
        }
        opened = {
            str(item["osm_id"])
            for item in closure
            if item["status"] == "confirmed_open"
        }
        eligible_closed = closed & mapped
        results["connectivity"] = {
            "status": "link_comparison_only",
            "independent_closure_count": len(closed),
            "source": spec["closure_source"],
            "matched_closed_links": len(eligible_closed),
            "unmatched_closed_links": len(closed - mapped),
            "suspected_link_closure_recall": len(likely & eligible_closed)
            / len(eligible_closed)
            if eligible_closed
            else None,
            "false_alarms_on_confirmed_open": len(likely & opened & mapped),
        }
    if "flood" not in results and "infrastructure" not in results:
        raise ValueError("No eligible reference flood or road labels supplied")
    results["runtime_seconds"] = perf_counter() - start
    write_json(run / "metrics.json", results)
    lines = [
        "# Independent evaluation",
        f"Reference: {spec['source']}",
        f"Observation: {spec['observation_timestamp']}",
        "TerraMind comparison unavailable: no compatible validated model checkpoint/run.",
        "Graph assessment is traceable functionality, not a claimed performance gain.",
        str(results.get("flood", {})),
        str(
            {
                k: v
                for k, v in results.get("infrastructure", {}).items()
                if k != "matches"
            }
        ),
        *results["limitations"],
    ]
    (run / "comparison.md").write_text("\n\n".join(lines))
    return results


def _reference_log(log, material, action, freeze_hash, independent=False):
    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    if not str(material).startswith("https://"):
        path = Path(material).resolve()
        root = data_root().resolve()
        material = ("HIMAROAD_DATA_DIR/" + str(path.relative_to(root))
                    if path.is_relative_to(root) else path.name)
    material = str(material).replace("|", "\\|").replace("\n", " ")
    kind = "independent human reference access" if independent else "automated reference access"
    purpose = "Independent image-label evaluation; no tuning" if independent else "Agreement only; no tuning"
    log = Path(log)
    # A blank line would end the Markdown table, so only add a newline the file lacks.
    lead = "\n" if log.exists() and log.stat().st_size and not log.read_bytes().endswith(b"\n") else ""
    with log.open("a") as handle:
        handle.write(f"{lead}| {timestamp} | evaluate_himalaya | {material} | {action}; frozen receipt {freeze_hash} | attempted {kind} | {purpose} |\n")


def _heldout_metrics(counts):
    tp, fp, fn, tn = map(int, counts)
    def ratio(a, b):
        return a / b if b else None
    iou = ratio(tp, tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "iou": iou, "csi": iou, "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def _heldout_summary(events, replicates=2000):
    names = ("precision", "recall", "iou", "csi", "f1")
    counts = {method: np.array([event[method] for event in events.values()], dtype="int64")
              for method in ("baseline", "unet")}
    result = {}
    for method, rows in counts.items():
        per_event = [_heldout_metrics(row) for row in rows]
        macro = {}
        for name in names:
            values = [row[name] for row in per_event if row[name] is not None]
            macro[name] = float(np.mean(values)) if values else None
        result[method] = {"pooled": _heldout_metrics(rows.sum(axis=0)), "event_macro": macro, "ci95": None}
    result["paired_difference_ci95"] = None
    if len(events) >= 2:
        draws = np.random.default_rng(2026).integers(len(events), size=(replicates, len(events)))
        sampled = {method: [_heldout_metrics(row) for row in rows[draws].sum(axis=1)]
                   for method, rows in counts.items()}
        def interval(values):
            return np.percentile(values, [2.5, 97.5]).tolist() if all(value is not None for value in values) else None
        for method in counts:
            result[method]["ci95"] = {name: interval([row[name] for row in sampled[method]]) for name in names}
        result["paired_difference_ci95"] = {name: interval([
            left[name] - right[name] if left[name] is not None and right[name] is not None else None
            for left, right in zip(sampled["unet"], sampled["baseline"])]) for name in names}
    return result


def _check_himalaya_freeze(path):
    from .datasets import DEFAULT_SPLITS, validate_splits
    from .flood import DEFAULTS
    from .unet import load_checkpoint

    path = Path(path).resolve()
    receipt = read_config(path)
    if (receipt.get("freeze_schema_version") != "himaroad-himalaya-freeze-v1"
            or not isinstance(receipt.get("synthetic"), bool) or not receipt.get("cases")):
        raise ValueError("Invalid Himalayan freeze receipt")
    expected_sources = [*sorted((REPO_ROOT / "src/himaroad").glob("*.py")), REPO_ROOT / "scripts/model/freeze_validation.py"]
    if set(receipt.get("source_hashes", {})) != {str(item.relative_to(REPO_ROOT)) for item in expected_sources}:
        raise ValueError("Freeze must pin every inference source file")
    if any(checksum(item) != receipt["source_hashes"][str(item.relative_to(REPO_ROOT))] for item in expected_sources):
        raise ValueError("Source changed since predictions were frozen")
    split_path = Path(receipt.get("split_config", DEFAULT_SPLITS)).resolve()
    if not receipt["synthetic"] and not split_path.is_relative_to(REPO_ROOT / "configs"):
        raise ValueError("Real test splits must use a registered repository config")
    splits = validate_splits(read_config(split_path))
    if checksum(split_path) != receipt.get("split_sha256"):
        raise ValueError("Registered test split changed since freezing")
    cfg = receipt["model_config"]
    checkpoint_hash = receipt.get("checkpoint_sha256")
    if (not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint_hash))
            or cfg.get("checkpoint_sha256") != checkpoint_hash or checksum(cfg["checkpoint"]) != checkpoint_hash):
        raise ValueError("Frozen checkpoint pin mismatch")
    _, checkpoint = load_checkpoint(cfg["checkpoint"], allow_synthetic=receipt["synthetic"] and cfg.get("allow_synthetic") is True)
    metadata = checkpoint["metadata"]
    canonical_split = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    if metadata["synthetic"] != receipt["synthetic"] or (not receipt["synthetic"] and (
            metadata.get("split_hash") != canonical_split or cfg.get("protocol_commit") != metadata["protocol_commit"])):
        raise ValueError("Checkpoint split/protocol/synthetic metadata mismatch")
    loaded, seen = {}, set()
    for case in receipt["cases"]:
        event = case["event_id"]
        info = splits["events"].get(event, {})
        if (event in seen or event not in splits["splits"]["test"] or not event.startswith("himalaya-")
                or info.get("country") not in {"Nepal", "India"} or info.get("terrain") != "mountain"
                or case.get("synthetic") != receipt["synthetic"]):
            raise ValueError("Freeze must contain unique registered Himalayan test events")
        seen.add(event)
        provenance = case["model_metadata"]
        keys = ("model_version", "preprocessing_id", "threshold", "temperature", "synthetic", "protocol_commit", "calibration_split", "training_data_versions")
        if (any(provenance.get(key) != metadata[key] for key in keys)
                or provenance.get("checkpoint_sha256") != checkpoint_hash
                or provenance.get("operating_threshold") != metadata["threshold"]
                or case.get("baseline_thresholds") != DEFAULTS):
            raise ValueError("Frozen calibration or baseline thresholds differ from pinned parameters")
        if not receipt["synthetic"] and any(provenance.get(key) != metadata[key] for key in ("tile_size", "overlap", "permanent_water_db")):
            raise ValueError("Frozen tiling/permanent-water parameters differ from the checkpoint")
        arrays, profile = {}, None
        if set(case["rasters"]) != {"baseline", "unet", "valid"}:
            raise ValueError("Freeze requires both predictions and their common valid mask")
        for name, item in case["rasters"].items():
            candidate = path.parent / item["path"]
            resolved = candidate.resolve()
            if Path(item["path"]).is_absolute() or not resolved.is_relative_to(path.parent):
                raise ValueError("Frozen prediction path escapes the receipt directory")
            if checksum(resolved) != item["sha256"]:
                raise ValueError("Frozen prediction checksum mismatch")
            with rasterio.open(resolved) as src:
                grid = (src.crs, src.transform, src.height, src.width)
                if not src.crs or src.count != 1 or src.height * src.width > 1_000_000:
                    raise ValueError("Frozen raster must be one georeferenced band with at most 1M pixels")
                if profile is not None and grid != (profile["crs"], profile["transform"], profile["height"], profile["width"]):
                    raise ValueError("Frozen predictions and support must share an exact grid")
                profile = src.profile.copy()
                arrays[name] = src.read(1, window=Window(0, 0, src.width, src.height), masked=True).filled(np.nan)
        valid = arrays["valid"] == 1
        if (not np.isin(arrays["valid"], [0, 1]).all() or int(valid.sum()) != case["valid_pixels"]
                or any(not np.array_equal(np.isfinite(arrays[name]), valid)
                       or not np.isin(arrays[name][valid], [0, 1]).all() for name in ("baseline", "unet"))):
            raise ValueError("Frozen prediction/support values are inconsistent")
        scenes = [item for item in case["input_hashes"] if item.get("units") == "dB"]
        before = sorted([item for item in scenes if utc(item["timestamp"]) < utc(case["event_timestamp"])], key=lambda item: utc(item["timestamp"]), reverse=True)
        after = [item for item in scenes if utc(item["timestamp"]) > utc(case["event_timestamp"])]
        if len(before) != 2 or len(after) != 1 or utc(after[0]["timestamp"]) != utc(case["observation_timestamp"]):
            raise ValueError("Agreement requires two pre-event and one matched post-event observation")
        roles = {"pre1": before[0], "pre2": before[1], "post": after[0]}
        if any(item.get("role", role) != role for role, item in roles.items()):
            raise ValueError("Frozen observation roles conflict with acquisition times")
        loaded[event] = {"case": case, "arrays": arrays, "profile": profile, "roles": roles}
    return receipt, loaded, checksum(path)


def _watermap_location(item, base, scene):
    if bool(item.get("url")) == bool(item.get("path")):
        raise ValueError("Reference must specify exactly one local path or anonymous URL")
    if item.get("url"):
        parsed = urlsplit(item["url"])
        match = re.fullmatch(r"/HKHwatermaps/(S1[AB])_IW_(\d{8}T\d{6})_DVR_RTC30_G_gpufed_[0-9A-F]{4}_WM\.tif", parsed.path)
        if (parsed.scheme != "https" or parsed.netloc not in {"asf-event-data.s3.us-west-2.amazonaws.com", "asf-event-data.s3-us-west-2.amazonaws.com"}
                or parsed.query or parsed.fragment or not match):
            raise ValueError("Only anonymous ASF HKHwatermaps WM COG URLs are permitted")
        if (match.group(2) != utc(scene["timestamp"]).strftime("%Y%m%dT%H%M%S")
                or not scene["scene_id"].startswith(f"{match.group(1)}_IW_SLC__1SDV_{match.group(2)}_")):
            raise ValueError("Water map URL identity/time differs from frozen acquisition")
        source_urls = scene.get("source_urls", [])
        if not any(url.removesuffix("_VV.tif") + "_WM.tif" == item["url"] for url in source_urls):
            raise ValueError("Water map must belong to the frozen raw RTC product")
        return item["url"]
    root = data_root() / "reference"
    candidate = Path(item["path"]).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    if root.is_symlink() or not candidate.is_relative_to(root) or not resolved.is_relative_to(root):
        raise ValueError("Local water maps must remain inside HIMAROAD_DATA_DIR/reference")
    if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
        raise ValueError("Local water maps require pinned SHA256")
    return resolved


def _read_watermap(location, profile, log, freeze_hash):
    _reference_log(log, location, "attempted metadata and bounded water-map pixel read", freeze_hash)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="2"):
        with rasterio.open(location) as src:
            if not src.crs or src.count != 1 or src.transform.b or src.transform.d:
                raise ValueError("Water maps require one georeferenced north-up band")
            bounds = transform_bounds(profile["crs"], src.crs, *array_bounds(profile["height"], profile["width"], profile["transform"]))
            requested = from_bounds(*bounds, src.transform)
            col, row = max(0, int(np.floor(requested.col_off))), max(0, int(np.floor(requested.row_off)))
            end_col = min(src.width, int(np.ceil(requested.col_off + requested.width)))
            end_row = min(src.height, int(np.ceil(requested.row_off + requested.height)))
            width, height = end_col - col, end_row - row
            if width <= 0 or height <= 0 or width * height > 1_000_000:
                raise ValueError("Water-map crop must overlap and stay within 1M pixels")
            window = Window(col, row, width, height)
            values = src.read(1, window=window, masked=True)
            finite = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
            if np.any(finite & ~np.isin(values.data, [0, 1, 255])):
                raise ValueError("Water maps permit only 0/1/255 codes")
            native = np.where(finite & (values.data != 255), values.data, np.nan).astype("float32")
            aligned = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")
            reproject(native, aligned, src_transform=src.window_transform(window), src_crs=src.crs,
                      src_nodata=np.nan, dst_transform=profile["transform"], dst_crs=profile["crs"],
                      dst_nodata=np.nan, resampling=Resampling.nearest)
            native_hash = hashlib.sha256(np.where(np.isfinite(native), native, 255).astype("uint8").tobytes()).hexdigest()
            return aligned, {"window": [col, row, width, height], "window_pixels": width * height,
                             "window_values_sha256": native_hash, "source_crs": str(src.crs)}


def evaluate_himalaya(freeze_path, reference_manifest, output, leakage_log=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Evaluation output must be fresh; refusing overwrite")
    receipt, frozen, freeze_hash = _check_himalaya_freeze(freeze_path)
    begin_evaluation(receipt, freeze_hash)
    log = Path(leakage_log) if leakage_log is not None else REPO_ROOT / "docs/LEAKAGE_LOG.md"
    manifest_path = Path(reference_manifest).resolve()
    _reference_log(log, manifest_path, "attempted automated reference manifest/metadata read", freeze_hash)
    spec = read_config(manifest_path)
    _reference_log(log, manifest_path, "reference manifest checksum read", freeze_hash)
    manifest_hash = checksum(manifest_path)
    cases = spec.get("cases", [])
    if (spec.get("purpose") != "evaluation" or spec.get("source") != "HydroSAR"
            or spec.get("label_kind") != "automated" or spec.get("synthetic") != receipt["synthetic"]
            or len(cases) != len(frozen) or {case["event_id"] for case in cases} != set(frozen)):
        raise ValueError("Reference manifest must declare matching unique HydroSAR automated evaluation cases")
    plans = []
    for case in cases:
        event = case["event_id"]
        for role in ("pre1", "pre2", "post"):
            item, scene = case[role], frozen[event]["roles"][role]
            if item.get("scene_id") != scene["scene_id"] or utc(item["timestamp"]) != utc(scene["timestamp"]):
                raise ValueError("Reference scene identity/timestamp differs from frozen raw input")
            plans.append((event, role, item, _watermap_location(item, manifest_path.parent, scene)))
    aligned, provenance = {}, {}
    for event, role, item, location in plans:
        if isinstance(location, Path):
            _reference_log(log, location, "local water-map checksum read", freeze_hash)
            if checksum(location) != item["sha256"]:
                raise ValueError("Reference water-map SHA256 mismatch")
        values, details = _read_watermap(location, frozen[event]["profile"], log, freeze_hash)
        aligned.setdefault(event, {})[role] = values
        provenance.setdefault(event, {})[role] = {**item, **details}
    events, counts = {}, {}
    for event, item in frozen.items():
        refs = aligned[event]
        observable = np.isfinite(refs["pre1"]) & np.isfinite(refs["pre2"]) & np.isfinite(refs["post"])
        stable = refs["pre1"] == refs["pre2"]
        target = (refs["post"] == 1) & (refs["pre1"] == 0) & (refs["pre2"] == 0)
        valid = (item["arrays"]["valid"] == 1) & observable & stable
        rows = {}
        for method in ("baseline", "unet"):
            p, r = item["arrays"][method][valid].astype(bool), target[valid]
            rows[method] = [int((p & r).sum()), int((p & ~r).sum()), int((~p & r).sum()), int((~p & ~r).sum())]
        counts[event] = rows
        events[event] = {**{method: _heldout_metrics(row) for method, row in rows.items()},
            "evaluated_pixels": int(valid.sum()), "frozen_valid_pixels": item["case"]["valid_pixels"],
            "evaluation_coverage": float(valid.sum() / item["case"]["valid_pixels"]),
            "reference_observable_pixels": int(observable.sum()),
            "inconsistent_pre_pixels": int((observable & ~stable).sum()),
            "stable_pre_water_pixels": int((valid & (refs["pre1"] == 1)).sum()),
            "reference_provenance": provenance[event], "processing_domain": item["case"]["processing_domain"],
            "event_timestamp": item["case"]["event_timestamp"], "observation_timestamp": item["case"]["observation_timestamp"]}
    result = {**producer_provenance(), "evaluation_schema_version": "himaroad-himalaya-agreement-v1",
        "evaluated_at": now(), "synthetic": receipt["synthetic"], "split": "test",
        "label_kind": "automated", "interpretation": "automated_reference_agreement", "reference_source": "HydroSAR",
        "freeze_sha256": freeze_hash, "reference_manifest_sha256": manifest_hash,
        "checkpoint_sha256": receipt["checkpoint_sha256"], "events": events, "summary": _heldout_summary(counts),
        "bootstrap": {"unit": "event", "replicates": 2000, "seed": 2026},
        "limitations": ["Agreement with an automated HydroSAR water map is not independently reviewed Himalayan accuracy.",
            "HydroSAR references share Sentinel-1 data/processing with inputs; errors can be correlated.",
            "Event-water requires post water and both pre observations nonwater; stable pre water is negative, inconsistent pre labels are unknown.",
            "Frozen gamma0 processing is not exact A-D05 Sentinel Hub transfer validation.",
            "One event cannot support event-bootstrap confidence intervals; unknown/nodata is excluded."]}
    if checksum(Path(freeze_path)) != freeze_hash or checksum(manifest_path) != manifest_hash:
        raise ValueError("Freeze or reference manifest changed during evaluation")
    # Recheck predictions and source pins before publishing a completed result.
    _check_himalaya_freeze(freeze_path)
    for _, _, item, location in plans:
        if isinstance(location, Path):
            _reference_log(log, location, "final local water-map checksum read", freeze_hash)
            if checksum(location) != item["sha256"]:
                raise ValueError("Reference changed during evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


def evaluate_himalaya_human(freeze_path, reference_manifest, output, leakage_log=None):
    """Once-only accuracy on independently reviewed, frozen blind chips."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Evaluation output must be fresh; refusing overwrite")
    receipt, frozen, freeze_hash = _check_himalaya_freeze(freeze_path)
    begin_evaluation(receipt, freeze_hash)
    log = Path(leakage_log) if leakage_log else REPO_ROOT / "docs/LEAKAGE_LOG.md"
    root = data_root() / "reference"
    verified_files = {}

    def location(item, base):
        candidate = Path(item["path"])
        candidate = candidate if candidate.is_absolute() else base / candidate
        path = candidate.resolve()
        if root.is_symlink() or not path.is_relative_to(root.resolve()):
            raise ValueError("Human labels must remain inside the reference directory")
        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
            raise ValueError("Every human review file requires a SHA256 pin")
        _reference_log(log, path, "independent human review file checksum read; no tuning", freeze_hash, independent=True)
        if checksum(path) != item["sha256"]:
            raise ValueError("Human review file SHA256 mismatch")
        verified_files[path] = item["sha256"]
        return path

    manifest_path = Path(reference_manifest).resolve()
    if not manifest_path.is_relative_to(root.resolve()):
        raise ValueError("Human reference manifest must remain inside reference directory")
    _reference_log(log, manifest_path, "independent human review manifest read; no tuning", freeze_hash, independent=True)
    manifest_hash = checksum(manifest_path)
    spec = read_config(manifest_path)
    if (spec.get("purpose") != "evaluation" or spec.get("label_kind") != "independent_human"
            or spec.get("label_status") != "completed" or spec.get("freeze_sha256") != freeze_hash
            or spec.get("synthetic") is not receipt["synthetic"]):
        raise ValueError("Human evaluation requires completed independent labels for this freeze")
    reviewers = spec.get("reviewers", [])
    if (len(reviewers) != 2 or len({r.get("id") for r in reviewers}) != 2
            or any(not r.get("id") or r.get("independent") is not True or not r.get("completed_at") for r in reviewers)):
        raise ValueError("Two distinct completed independent reviewers are required")
    for reviewer in reviewers:
        utc(reviewer["completed_at"])
    package_path = location(spec["package"], manifest_path.parent)
    if not receipt["synthetic"]:
        registry = committed_config()
        if any(registry["events"][event].get("review_package_sha256") != spec["package"]["sha256"] for event in frozen):
            raise ValueError("Blind package must be committed in the freeze registry before review")
    package = read_config(package_path)
    if package.get("freeze_sha256") != freeze_hash:
        raise ValueError("Blind review package belongs to another freeze")
    expected = {item["chip_id"]: item for item in package["chips"]}
    if len(expected) != len(package["chips"]):
        raise ValueError("Blind package chip IDs must be unique")
    chips = spec.get("chips", [])
    if len(chips) != len(expected) or {item["chip_id"] for item in chips} != set(expected):
        raise ValueError("Review must contain every predetermined blind chip exactly once")
    # Amendment A2: registered review sets with fixed roles, evaluated in this one run.
    roles = package.get("sets")
    if not receipt["synthetic"] and any(
        case.get("model_metadata", {}).get("model_version", "").startswith("unet-kuro-v2")
        for case in receipt.get("cases", [])
    ):
        registered_roles = read_config(REPO_ROOT / "configs/model_v2_himalaya_selection.json")["review"]["sets"]
        if roles != {key: registered_roles[key] for key in ("primary", "secondary")}:
            raise ValueError("Real v2 review must preserve the registered A2 set roles")
        windows_hash = package.get("windows_sha256")
        if (not re.fullmatch(r"[0-9a-f]{64}", str(windows_hash))
                or any(registry["events"][event].get("review_windows_sha256") != windows_hash for event in frozen)):
            raise ValueError("Real v2 review windows must match the committed registry pin")
    if roles is not None:
        if (set(roles) != {"primary", "secondary"} or roles["primary"] == roles["secondary"]
                or any(not isinstance(name, str) or not name for name in roles.values())):
            raise ValueError("Review sets require distinct named primary and secondary sets")
        if any(item.get("set") not in roles.values() for item in package["chips"]):
            raise ValueError("Every blind chip must belong to a registered review set")
        if {item["set"] for item in package["chips"]} != set(roles.values()):
            raise ValueError("Each registered review set needs at least one chip")
    names = list(roles.values()) if roles else [None]
    counts = {name: {event: {method: np.zeros(4, dtype="int64") for method in ("baseline", "unet")} for event in frozen}
              for name in names}
    agreement = {name: {event: np.zeros(4, dtype="int64") for event in frozen} for name in names}
    coverage = {name: {event: 0 for event in frozen} for name in names}
    provenance = []
    occupied = {name: {event: np.zeros(item["arrays"]["valid"].shape, bool) for event, item in frozen.items()}
                for name in names}
    for chip in chips:
        info = expected[chip["chip_id"]]
        event = info["event_id"]
        name = info.get("set") if roles else None
        if event not in frozen:
            raise ValueError("Blind chip event is outside the freeze")
        col, row, width, height = info["window"]
        profile = frozen[event]["profile"]
        if (any(type(v) is not int for v in (col, row, width, height)) or min(col, row) < 0
                or min(width, height) <= 0 or col + width > profile["width"] or row + height > profile["height"]):
            raise ValueError("Blind chip window is outside the frozen grid")
        selected = np.s_[row:row + height, col:col + width]
        if occupied[name][event][selected].any():
            raise ValueError("Blind chip windows cannot overlap within a review set")
        occupied[name][event][selected] = True
        labels = {}
        for role in ("reviewer_a", "reviewer_b", "adjudicated"):
            path = location(chip[role], manifest_path.parent)
            _reference_log(log, path, "independent human label pixels read; no tuning", freeze_hash, independent=True)
            if path.suffix == ".json":
                if path.stat().st_size > 1_000_000:
                    raise ValueError("Blind annotation JSON exceeds its size budget")
                annotation = read_config(path)
                if (annotation.get("chip_id") != chip["chip_id"] or annotation.get("width") != width
                        or annotation.get("height") != height or len(annotation.get("values", [])) != width * height):
                    raise ValueError("Human labels must preserve the exact blind chip grid")
                values = np.asarray(annotation["values"])
                if values.dtype.kind not in "iu" or values.ndim != 1:
                    raise ValueError("Blind annotation values must be integer label codes")
                values = values.reshape(height, width)
            else:
                with rasterio.open(path) as source:
                    expected_transform = profile["transform"] * rasterio.Affine.translation(col, row)
                    if (source.count != 1 or source.shape != (height, width)
                            or source.crs != profile["crs"] or source.transform != expected_transform):
                        raise ValueError("Human labels must preserve the exact blind chip grid")
                    values = source.read(1, masked=True).filled(255)
            if not np.isin(values, [0, 1, 255]).all():
                raise ValueError("Human label codes must be 0/1/255")
            labels[role] = values
        a, b, target = (labels[role] for role in ("reviewer_a", "reviewer_b", "adjudicated"))
        observable = frozen[event]["arrays"]["valid"][selected] == 1
        if any(np.any(~observable & (v != 255)) for v in labels.values()):
            raise ValueError("Unobservable frozen pixels must remain unknown in human labels")
        agreed = observable & (a != 255) & (b != 255) & (a == b)
        if np.any(agreed & (target != a)):
            raise ValueError("Adjudication cannot silently change agreed independent labels")
        common = observable & (a != 255) & (b != 255)
        agreement[name][event] += [int((common & (a == 1) & (b == 1)).sum()),
                             int((common & (a == 1) & (b == 0)).sum()),
                             int((common & (a == 0) & (b == 1)).sum()),
                             int((common & (a == 0) & (b == 0)).sum())]
        valid = observable & (target != 255)
        coverage[name][event] += int(valid.sum())
        for method in counts[name][event]:
            p = frozen[event]["arrays"][method][selected][valid] == 1
            t = target[valid] == 1
            counts[name][event][method] += [int((p & t).sum()), int((p & ~t).sum()),
                                     int((~p & t).sum()), int((~p & ~t).sum())]
        provenance.append({"chip_id": chip["chip_id"], "event_id": event,
                           **({"set": name} if roles else {}),
                           **{role: chip[role]["sha256"] for role in labels}})
    per_set = {}
    for name in names:
        events = {}
        for event, rows in counts[name].items():
            if coverage[name][event] == 0:
                raise ValueError("Each reviewed event needs observable adjudicated labels")
            tp, fp, fn, tn = map(int, agreement[name][event])
            total = tp + fp + fn + tn
            observed = (tp + tn) / total if total else None
            chance = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / total ** 2 if total else None
            kappa = (observed - chance) / (1 - chance) if chance is not None and chance < 1 else None
            events[event] = {**{method: _heldout_metrics(row) for method, row in rows.items()},
                             "evaluated_pixels": coverage[name][event],
                             "evaluation_coverage": coverage[name][event] / frozen[event]["case"]["valid_pixels"],
                             "reviewer_agreement": {**_heldout_metrics(agreement[name][event]),
                                                   "common_pixels": total, "agreement_fraction": observed, "kappa": kappa}}
        per_set[name] = {"events": events, "summary": _heldout_summary(counts[name])}
    primary = roles["primary"] if roles else None
    result = {**producer_provenance(), "evaluation_schema_version": "himaroad-himalaya-human-v1",
              "evaluated_at": now(), "synthetic": receipt["synthetic"], "split": "test",
              "label_kind": "independent_human", "freeze_sha256": freeze_hash,
              "reference_manifest_sha256": manifest_hash, "package_sha256": spec["package"]["sha256"],
              "checkpoint_sha256": receipt["checkpoint_sha256"], "events": per_set[primary]["events"],
              "summary": per_set[primary]["summary"], "label_hashes": provenance,
              "bootstrap": {"unit": "event", "replicates": 2000, "seed": 2026},
              "limitations": ["Accuracy covers only predetermined reviewed chips, not whole-event footprints.",
                              "Image interpretation remains uncertain; unknown labels are excluded.",
                              "Two or three events provide unstable event-bootstrap intervals."]}
    if roles:
        result["review_sets"] = {name: {"role": role, **per_set[name]} for role, name in roles.items()}
        result["limitations"].append(f"Top-level events and summary are the primary review set ({primary}); "
                                     "the secondary set is reported under review_sets and roles were fixed before labels.")
        if primary == "A3":
            result["limitations"].append(
                "Primary A3 windows were drawn from the top quartile of valley-floor fraction after the A1 "
                "feasibility check failed: this is performance on valley-enriched Himalayan windows, not "
                "representative Himalayan accuracy; HAND was computed on the AOI-sized DEM (truncated catchments).")
    if checksum(manifest_path) != manifest_hash or any(checksum(path) != digest for path, digest in verified_files.items()):
        raise ValueError("Human review files changed during evaluation")
    _, _, final_freeze_hash = _check_himalaya_freeze(freeze_path)
    if final_freeze_hash != freeze_hash:
        raise ValueError("Freeze changed during evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
