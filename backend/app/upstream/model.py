"""Validate recorded UNet identity, without loading weights or asserting accuracy."""

import json
import math
import re

from backend.app.upstream.paths import RunContractError


def model_provenance(report: dict, manifest: dict) -> tuple[dict | None, dict | None]:
    metadata = manifest.get("model_metadata")
    if "model_metadata" in report and json.dumps(
        report["model_metadata"], sort_keys=True, allow_nan=False
    ) != json.dumps(metadata, sort_keys=True, allow_nan=False):
        raise RunContractError("report/manifest.model_metadata: conflicting identity")
    components = {}
    for name in ("dem", "hand"):
        record = manifest.get(name)
        if isinstance(record, dict) and "synthetic" in record:
            value = record["synthetic"]
            if value is not None and type(value) is not bool:
                raise RunContractError("component.synthetic: expected boolean or unknown")
            components[name] = value
    if metadata is None:
        if report.get("status") == "complete" and report.get("detector") == "unet":
            raise RunContractError("complete UNet run requires recorded model_metadata")
        return None, components or None
    if not isinstance(metadata, dict) or report.get("detector") != "unet":
        raise RunContractError("model_metadata: expected recorded UNet metadata")
    if metadata.get("name") != "unet" or type(metadata.get("synthetic")) is not bool:
        raise RunContractError("model_metadata: missing model name or synthetic flag")
    for key, pattern in (
        ("checkpoint_sha256", r"[0-9a-f]{64}"),
        ("model_version", r"[A-Za-z0-9_.-]{1,128}"),
    ):
        if not isinstance(metadata.get(key), str) or not re.fullmatch(pattern, metadata[key]):
            raise RunContractError(f"model_metadata.{key}: invalid identity")
    if metadata.get("preprocessing_id") not in {"s1-db-six-channel-v1", "s1-db-four-channel-v2"}:
        raise RunContractError("model_metadata.preprocessing_id: unreviewed method")
    for key in ("threshold", "operating_threshold", "temperature"):
        value = metadata.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise RunContractError(f"model_metadata.{key}: expected finite number")
    if not (
        0 < metadata["temperature"]
        and 0 <= metadata["threshold"] <= 1
        and metadata["threshold"] == metadata["operating_threshold"]
    ):
        raise RunContractError("model_metadata: inconsistent operating point")
    tile, overlap = metadata.get("tile_size"), metadata.get("overlap")
    if type(tile) is not int or type(overlap) is not int or not 0 <= overlap < tile:
        raise RunContractError("model_metadata: invalid inference tiling")
    versions = metadata.get("training_data_versions")
    if (
        not isinstance(versions, dict)
        or not versions
        or any(
            not isinstance(k, str) or not isinstance(v, str) or not v for k, v in versions.items()
        )
    ):
        raise RunContractError("model_metadata: training provenance unavailable")
    if not metadata["synthetic"] and (
        not isinstance(metadata.get("protocol_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", metadata["protocol_commit"])
        or metadata.get("calibration_split") != "dev"
    ):
        raise RunContractError("model_metadata: real model needs registered dev calibration")
    components["model"] = metadata["synthetic"]
    # Keep only reviewed identity/configuration; source files remain hash-backed.
    from backend.app.evidence.model_policy import MODEL_FIELDS

    return {key: value for key, value in metadata.items() if key in MODEL_FIELDS}, components
