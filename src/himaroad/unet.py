"""Six-channel SAR baseline, shared preprocessing and calibrated inference."""

import math
import os
from pathlib import Path

import numpy as np
from pyproj import CRS

from .common import DataUnavailable, checksum, data_root, inference_path, read_config

PREPROCESSING_ID = "s1-db-six-channel-v1"
PREPROCESSING_IDS = {6: PREPROCESSING_ID, 4: "s1-db-four-channel-v2"}
CHECKPOINT_FORMAT = "himaroad-unet-v1"


def validate_device(device):
    if str(device).startswith("cuda"):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or visible == "-1" or "," in visible or str(device) not in {"cuda", "cuda:0"}:
            raise ValueError("CUDA requires one explicit CUDA_VISIBLE_DEVICES device and logical cuda:0")


def preprocess(arrays, in_channels=6):
    if in_channels not in PREPROCESSING_IDS:
        raise ValueError("UNet supports four raw or six raw/change SAR channels")
    vv_pre = arrays.get("pre_vv", arrays.get("pre"))
    vv_post = arrays.get("post_vv", arrays.get("post"))
    if vv_pre is None or vv_post is None or any(
        key not in arrays for key in ("pre_vh", "post_vh")
    ):
        raise DataUnavailable("unsupported_model_inputs", "UNet requires pre/post VV and VH in dB")
    bands = [np.asarray(band, dtype="float32") for band in
             (vv_pre, arrays["pre_vh"], vv_post, arrays["post_vh"])]
    shape = bands[0].shape
    if len(shape) != 2 or not all(shape) or any(band.shape != shape for band in bands):
        raise ValueError("SAR bands must share a two-dimensional grid")
    valid = np.isfinite(np.stack(bands)).all(axis=0)
    for key in ("valid", "aoi"):
        if key in arrays:
            mask = np.asarray(arrays[key])
            if mask.shape != shape:
                raise ValueError(f"{key} mask must match the SAR grid")
            valid &= np.isfinite(mask) & (mask != 0)
    normalized = [2 * (np.clip(band, low, high) - low) / (high - low) - 1
                  for band, (low, high) in zip(bands, [(-30, 0), (-35, -5)] * 2)]
    if in_channels == 6:
        normalized.extend(np.clip(bands[index + 2] - bands[index], -20, 20) / 20
                          for index in (0, 1))
    inputs = np.stack(normalized).astype("float32")
    inputs[:, ~valid] = 0
    return inputs, valid


try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


class UNet(nn.Module if nn else object):
    def __init__(self, in_channels=6, base_channels=32, depth=4):
        if nn is None:
            raise DataUnavailable("model_dependency_unavailable", "Install the torch model dependency")
        super().__init__()
        if in_channels not in PREPROCESSING_IDS or base_channels < 2 or not 1 <= depth <= 5:
            raise ValueError("UNet requires four/six channels, base_channels >= 2 and depth 1..5")
        self.depth = depth
        self.in_channels = in_channels

        def block(source, target):
            return nn.Sequential(
                nn.Conv2d(source, target, 3, padding=1, bias=False),
                nn.GroupNorm(max(1, math.gcd(8, target) // 2), target), nn.ReLU(inplace=True),
                nn.Conv2d(target, target, 3, padding=1, bias=False),
                nn.GroupNorm(max(1, math.gcd(8, target) // 2), target), nn.ReLU(inplace=True),
            )

        widths = [base_channels * 2 ** index for index in range(depth + 1)]
        self.encoders = nn.ModuleList(block(source, target) for source, target in
                                     zip([in_channels, *widths[:-1]], widths))
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(nn.ConvTranspose2d(widths[index + 1], widths[index], 2, 2)
                                for index in reversed(range(depth)))
        self.decoders = nn.ModuleList(block(2 * widths[index], widths[index])
                                    for index in reversed(range(depth)))
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, inputs):
        skips = []
        for encoder in self.encoders[:-1]:
            inputs = encoder(inputs)
            skips.append(inputs)
            inputs = self.pool(inputs)
        inputs = self.encoders[-1](inputs)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            inputs = up(inputs)
            if inputs.shape[-2:] != skip.shape[-2:]:
                inputs = nn.functional.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            inputs = decoder(torch.cat((skip, inputs), dim=1))
        return self.head(inputs)


def load_checkpoint(path, allow_synthetic=False):
    if torch is None:
        raise DataUnavailable("model_dependency_unavailable", "Install the torch model dependency")
    path = inference_path(path)
    if not path.is_file():
        raise DataUnavailable("model_checkpoint_unavailable", "Configured UNet checkpoint does not exist")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported UNet checkpoint format")
    metadata = checkpoint.get("metadata", {})
    required = {"model_version", "preprocessing_id", "training_data_versions", "threshold",
                "temperature", "synthetic", "protocol_commit", "calibration_split"}
    channels = checkpoint.get("model_kwargs", {}).get("in_channels", 6)
    if not required <= metadata.keys() or metadata["preprocessing_id"] != PREPROCESSING_IDS.get(channels):
        raise ValueError("Checkpoint metadata is incomplete or preprocessing is incompatible")
    if not metadata["model_version"] or not isinstance(metadata["training_data_versions"], dict):
        raise ValueError("Checkpoint model version and training-data versions are required")
    if not isinstance(metadata["synthetic"], bool):
        raise ValueError("Checkpoint synthetic flag must be a boolean")
    if metadata["calibration_split"] != "dev":
        raise ValueError("Checkpoint operating point must be calibrated on dev only")
    if not 0 < float(metadata["threshold"]) < 1 or not 0 < float(metadata["temperature"]) < float("inf"):
        raise ValueError("Invalid checkpoint calibration parameters")
    if metadata["synthetic"]:
        if not allow_synthetic:
            raise ValueError("Synthetic checkpoint requires explicit allow_synthetic=true")
    else:
        import re
        if not re.fullmatch(r"[0-9a-f]{40}", str(metadata["protocol_commit"])):
            raise ValueError("Trained checkpoints require a pre-registered protocol commit")
        if not metadata["training_data_versions"]:
            raise ValueError("Trained checkpoint must identify training data")
        if not {"tile_size", "overlap", "permanent_water_db"} <= metadata.keys():
            raise ValueError("Trained checkpoint must identify calibrated tiling parameters")
    model = UNet(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint


def checkpoint_path(cfg, config=None):
    path = Path(cfg["checkpoint"]).expanduser()
    if not path.is_absolute():
        if path.parts[0] == "data":
            path = data_root() / Path(*path.parts[1:])
        elif isinstance(config, (str, Path)):
            path = Path(config).resolve().parent / path
    return inference_path(path)


def predict_logits(model, inputs, valid, tile_size=256, overlap=32, device="cpu"):
    """Apply identical padded, overlapping tiles for dev calibration and deployment."""
    validate_device(device)
    inputs = np.asarray(inputs, dtype="float32")
    valid = np.asarray(valid, dtype=bool)
    height, width = valid.shape
    if inputs.shape != (getattr(model, "in_channels", 6), height, width):
        raise ValueError("Prediction inputs must match the model channel grid")
    tile, overlap = int(tile_size), int(overlap)
    if tile < 2 ** model.depth or tile % (2 ** model.depth) or not 0 <= overlap < tile:
        raise ValueError("Tile must fit UNet depth and overlap must be in [0, tile_size)")
    model = model.to(device).eval()
    logits_sum = np.zeros((height, width), dtype="float32")
    counts = np.zeros((height, width), dtype="float32")
    with torch.inference_mode():
        for y in range(0, height, tile - overlap):
            for x in range(0, width, tile - overlap):
                h, w = min(tile, height - y), min(tile, width - x)
                if not valid[y:y+h, x:x+w].any():
                    continue
                patch = np.pad(inputs[:, y:y+h, x:x+w], ((0, 0), (0, tile-h), (0, tile-w)), mode="edge")
                logits = model(torch.from_numpy(patch[None]).to(device))
                logits_sum[y:y+h, x:x+w] += logits.cpu().numpy()[0, 0, :h, :w]
                counts[y:y+h, x:x+w] += 1
    return logits_sum / np.maximum(counts, 1)


def infer(arrays, profile, config):
    if not config:
        raise DataUnavailable("model_checkpoint_unavailable", "Provide a trained UNet checkpoint config")
    cfg = read_config(config) if isinstance(config, (str, Path)) else config
    validate_device(cfg.get("device", "cpu"))
    if cfg.get("preprocessing_id", PREPROCESSING_ID) not in PREPROCESSING_IDS.values():
        raise ValueError("Configured preprocessing is incompatible")
    if not cfg.get("checkpoint"):
        raise DataUnavailable("model_checkpoint_unavailable", "Configure a UNet checkpoint path")
    path = checkpoint_path(cfg, config)
    model, checkpoint = load_checkpoint(path, cfg.get("allow_synthetic", False))
    digest = checksum(path)
    if cfg.get("checkpoint_sha256") and cfg["checkpoint_sha256"] != digest:
        raise ValueError("UNet checkpoint checksum mismatch")
    channels = checkpoint["model_kwargs"].get("in_channels", 6)
    if cfg.get("preprocessing_id", PREPROCESSING_IDS[channels]) != checkpoint["metadata"]["preprocessing_id"]:
        raise ValueError("Configured channel preprocessing differs from checkpoint")
    inputs, valid = preprocess(arrays, channels)
    height, width = valid.shape
    if profile.get("height", height) != height or profile.get("width", width) != width:
        raise ValueError("UNet inputs must match the pipeline grid")
    metadata = checkpoint["metadata"]
    if not metadata["synthetic"] and metadata["model_version"].startswith("unet-kuro-v2"):
        crs = CRS.from_user_input(profile.get("crs")) if profile.get("crs") else None
        transform = profile.get("transform")
        if (crs is None or crs.utm_zone is None or transform is None or transform.b or transform.d
                or transform.a != 20 or transform.e != -20):
            raise DataUnavailable("model_grid_incompatible", "V2 requires a north-up UTM grid at 20 metres")
    tile = int(cfg.get("tile_size", metadata.get("tile_size", 256)))
    overlap = int(cfg.get("overlap", metadata.get("overlap", 32)))
    if not metadata["synthetic"] and (tile != metadata["tile_size"] or overlap != metadata["overlap"]):
        raise ValueError("Inference tiling must match the calibrated checkpoint")
    permanent_water_db = float(cfg.get("permanent_water_db", metadata.get("permanent_water_db", -20)))
    if not metadata["synthetic"] and permanent_water_db != metadata["permanent_water_db"]:
        raise ValueError("Permanent-water threshold must match the calibrated checkpoint")
    logits = predict_logits(model, inputs, valid, tile, overlap, cfg.get("device", "cpu"))
    probabilities = torch.sigmoid(torch.from_numpy(logits) / metadata["temperature"]).numpy()
    permanent = np.asarray(arrays.get("pre_vv", arrays.get("pre"))) < permanent_water_db
    if "permanent_water" in arrays:
        permanent |= arrays["permanent_water"] == 1
    predicted = (probabilities >= metadata["threshold"]) & valid & ~permanent
    if metadata["model_version"].startswith("unet-kuro-v2") and "dem" in arrays:
        from .preprocess import slope_degrees
        slope = slope_degrees(arrays["dem"], profile)
        predicted &= ~np.isfinite(slope) | (slope <= 12)
    return predicted, {
        **metadata, "name": "unet", "checkpoint_sha256": digest,
        "operating_threshold": metadata["threshold"], "torch_version": str(torch.__version__),
        "tile_size": tile, "overlap": overlap,
        "limitations": ["Educational prototype; water segmentation does not establish road closure or damage."]
        + (["Synthetic weights: no measured flood-detection performance."] if metadata["synthetic"] else [])
        + list(cfg.get("evaluation_limitations", [])),
    }
