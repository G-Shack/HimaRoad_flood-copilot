"""Inference for an explicitly compatible trained TerraMind-small GRD checkpoint."""

from pathlib import Path

import numpy as np

from .common import DataUnavailable, checksum, read_config


def infer(arrays, profile, config):
    if not config:
        raise DataUnavailable(
            "model_checkpoint_unavailable",
            "Provide a validated TerraMind-small S1GRD Lightning checkpoint/config. Public TerraMind-base-Flood expects RTC and four timesteps and is incompatible.",
        )
    cfg = read_config(config) if isinstance(config, (str, Path)) else config
    checkpoint = Path(cfg.get("checkpoint", ""))
    if not checkpoint.is_file():
        raise DataUnavailable(
            "model_checkpoint_unavailable",
            "Configured trained checkpoint does not exist",
        )
    if (
        cfg.get("modalities") != ["S1GRD"]
        or cfg.get("channels") != ["VV", "VH"]
        or cfg.get("backbone") != "terramind_v1_small"
    ):
        raise ValueError(
            "Model requires explicit TerraMind-small S1GRD VV,VH compatibility declaration"
        )
    if "post_vh" not in arrays:
        raise DataUnavailable(
            "unsupported_model_inputs", "TerraMind GRD adapter requires VV and VH bands"
        )
    try:
        from terratorch.tasks import SemanticSegmentationTask
    except ImportError as exc:
        raise DataUnavailable(
            "model_dependency_unavailable",
            "Install optional TerraTorch only when supplying a compatible checkpoint",
        ) from exc
    import torch

    model_args = dict(
        backbone="terramind_v1_small",
        backbone_pretrained=False,
        backbone_modalities=["S1GRD"],
        backbone_merge_method="mean",
        necks=[
            {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
            {"name": "ReshapeTokensToImage", "remove_cls_token": False},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        decoder="UNetDecoder",
        num_classes=2,
        rescale=True,
    )
    task = SemanticSegmentationTask.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
        model_factory="EncoderDecoderFactory",
        model_args=model_args,
    )
    device = cfg.get("device", "cpu")
    task = task.to(device).eval()
    mean = np.array([-12.599, -20.293], dtype="float32")[:, None, None]
    std = np.array([5.195, 5.890], dtype="float32")[:, None, None]
    data = np.stack([arrays["post"], arrays["post_vh"]])
    valid = arrays["valid"] & np.isfinite(data).all(axis=0)
    height, width = valid.shape
    tile = int(cfg.get("tile_size", 256))
    if tile <= 0 or tile % 16:
        raise ValueError("Tile size must be positive and divisible by 16")
    predicted = np.zeros(valid.shape, bool)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            h, w = min(tile, height - y), min(tile, width - x)
            if not valid[y : y + h, x : x + w].any():
                continue
            patch = data[:, y : y + h, x : x + w]
            patch = np.where(np.isfinite(patch), patch, mean)
            normalized = (patch - mean) / std
            normalized = np.pad(
                normalized, ((0, 0), (0, tile - h), (0, tile - w)), mode="edge"
            )
            with torch.inference_mode():
                logits = task.model(
                    {"S1GRD": torch.from_numpy(normalized[None]).to(device)}
                ).output
            if tuple(logits.shape) != (1, 2, tile, tile):
                raise ValueError(
                    "Model output is not spatially aligned two-class logits"
                )
            predicted[y : y + h, x : x + w] = logits.argmax(1).cpu().numpy()[
                0, :h, :w
            ] == int(cfg.get("water_class", 1))
    permanent = arrays["pre"] < cfg.get("permanent_water_db", -20)
    if "permanent_water" in arrays:
        permanent |= arrays["permanent_water"] == 1
    return predicted & valid & ~permanent, {
        "name": "terramind_v1_small",
        "checkpoint_sha256": checksum(checkpoint),
        "inference_configuration": cfg,
        "torch_version": torch.__version__,
        "limitations": [
            "No trained model evaluation performed; tile-edge effects are not quantified."
        ],
    }
