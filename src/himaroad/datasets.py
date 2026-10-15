"""Explicit, event-split training chips from the permitted datasets."""
import hashlib
import json
import re
from pathlib import Path

from pyproj import CRS

import numpy as np
import rasterio

from .common import checksum, data_root, inference_path
from .preprocess import slope_degrees

IGNORE_INDEX = -1
DATASETS = {"kurosiwo", "sen1floods11", "himalaya_labels"}
DEFAULT_SPLITS = Path(__file__).resolve().parents[2] / "configs/model_splits.json"


def dataset_path(path, dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported training dataset: {dataset}")
    root = data_root() / "datasets" / "himaroad" / dataset
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    # Check lexical and resolved names: a benign symlink must not hide a reference.
    for value in (candidate, candidate.resolve()):
        for part in value.parts:
            lower = part.lower()
            if any(token in lower for token in ("reference", "emsr927", "unosat", "gfm")):
                raise ValueError("Validation-only location refused for model data")
            # Permitted Kuro events use EMSR IDs; general EMS products stay blocked.
            if re.search(r"ems(?!r\d+(?:[_-]|$))", lower):
                raise ValueError("EMS reference location refused for model data")
    resolved = inference_path(candidate)
    if not candidate.is_relative_to(root) or not resolved.is_relative_to(root.resolve()):
        raise ValueError("Model data must stay inside its explicit dataset root")
    if root.is_symlink():
        raise ValueError("Dataset root must not be a symlink")
    return resolved


def validate_splits(config):
    splits = config["splits"]
    events = config["events"]
    if set(splits) != {"train", "dev", "test"}:
        raise ValueError("Exactly train, dev and test splits are required")
    seen_events, seen_regions = set(), {}
    for split, ids in splits.items():
        for event in ids:
            if event in seen_events:
                raise ValueError(f"Event appears in multiple splits: {event}")
            seen_events.add(event)
            info = events[event]
            region = info["region"]
            if not region or info.get("terrain") not in {"lowland", "mountain", "unknown"}:
                raise ValueError("Each event needs a region and declared terrain")
            if region in seen_regions and seen_regions[region] != split:
                raise ValueError(f"Region appears in multiple splits: {region}")
            seen_regions[region] = split
            if split == "train" and (
                info.get("country", "").lower() == "nepal"
                or "nepal" in event.lower()
                or info["terrain"] != "lowland"
            ):
                raise ValueError("Nepal, mountain and unknown-terrain events cannot train")
    if seen_events != set(events):
        raise ValueError("Every declared event must have exactly one split")
    return config


def map_labels(labels, valid, dataset):
    """Map native water labels to binary targets, keeping nodata as ignore."""
    valid = valid.copy()
    if dataset == "kurosiwo":
        codes, positive, ignored = [0, 1, 2, 3], 2, 3
    elif dataset in {"sen1floods11", "himalaya_labels"}:
        codes, positive, ignored = [-1, 0, 1], 1, IGNORE_INDEX
    else:
        raise ValueError("Unsupported label dataset")
    if np.any(valid & ~np.isin(labels, codes)):
        raise ValueError(f"Unknown {dataset} label code")
    valid &= labels != ignored
    return np.where(valid, labels == positive, IGNORE_INDEX).astype("int64"), valid


def _config(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def _read(path, dataset, band=1):
    path = dataset_path(path, dataset)
    if path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2 or values.dtype.kind not in "biuf":
            raise ValueError("Training NumPy arrays must be two-dimensional numeric data")
        values = values.astype("float32")
        # Webdataset strips georeferencing; never invent a CRS for these chips.
        return values, np.isfinite(values), (None, None, values.shape), {"height": values.shape[0], "width": values.shape[1], "crs": None, "transform": None}
    with rasterio.open(path) as src:
        values = src.read(band, masked=True)
        valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
        return values.data.astype("float32"), valid, (src.crs, src.transform, values.shape), src.profile.copy()


def terrain_p95_slope(dem, valid, profile):
    """Measure slopes in ground metres; Web Mercator spacing needs latitude correction."""
    crs = CRS.from_user_input(profile["crs"])
    transform = profile["transform"]
    if not crs.is_projected or transform.b or transform.d or dem.shape[0] < 3 or dem.shape[1] < 3:
        raise ValueError("Terrain screening requires an axis-aligned projected metric DEM")
    values = np.where(valid & np.isfinite(dem), dem, np.nan).astype("float64")
    slopes = slope_degrees(values, profile)
    slopes = slopes[valid & np.isfinite(slopes)]
    if slopes.size == 0:
        raise ValueError("No observable DEM pixels for terrain screening")
    return float(np.percentile(slopes, 95))


class ManifestDataset:
    """Load chips whose event IDs were frozen before training.

    A manifest contains dataset, dataset_version and chips (id, event_id, files).
    Kuro files use pre1/pre2/post_vv/vh, label and optional valid keys; values are
    linear sigma0. Himalayan chips use pre/post_vv/vh in dB, label 0/1 and -1
    ignore. Post-only Sen1Floods11 cannot feed the temporal six-channel model.
    """

    def __init__(self, manifest, split, split_config=DEFAULT_SPLITS, augment=False, seed=0,
                 in_channels=6, tile_size=None, terrain_policy=False):
        if split not in {"train", "dev", "test"}:
            raise ValueError("Unknown model split")
        self.manifest = _config(manifest)
        self.split_config = validate_splits(_config(split_config))
        self.split = split
        self.dataset = self.manifest["dataset"]
        if self.dataset not in DATASETS:
            raise ValueError("Training dataset is not permitted")
        if self.dataset == "sen1floods11":
            raise ValueError("Sen1Floods11 is post-only; use a separately registered post-only model")
        if self.dataset == "himalaya_labels" and self.manifest.get("label_status") != "completed":
            raise ValueError("Himalaya labels must be explicitly marked completed")
        self.training_data_versions = {self.dataset: self.manifest["dataset_version"]}
        self.split_hash = hashlib.sha256(json.dumps(self.split_config, sort_keys=True).encode()).hexdigest()
        self.augment, self.seed, self.epoch = augment, seed, 0
        self.in_channels, self.tile_size, self.terrain_policy = in_channels, tile_size, terrain_policy
        chips = self.manifest["chips"]
        ids, fingerprints = set(), set()
        for chip in chips:
            if chip["event_id"] not in self.split_config["events"]:
                raise ValueError(f"Unregistered event: {chip['event_id']}")
            info = self.split_config["events"][chip["event_id"]]
            if "allowed_aois" in info and f"{int(chip.get('aoi_id', -1)):02}" not in info["allowed_aois"]:
                raise ValueError("Chip AOI is not permitted by the terrain-reviewed event subset")
            if chip["id"] in ids:
                raise ValueError("Duplicate chip ID")
            ids.add(chip["id"])
            required = {"post_vv", "post_vh", "label"}
            required |= {f"pre{n}_{pol}" for n in (1, 2) for pol in ("vv", "vh")} if self.dataset == "kurosiwo" else {"pre_vv", "pre_vh"}
            if not required.issubset(chip["files"]):
                raise ValueError("Missing temporal input or label")
            paths = {name: dataset_path(path, self.dataset) for name, path in chip["files"].items()}
            fingerprint = tuple(str(paths[name]) for name in sorted(required - {"label"}))
            if fingerprint in fingerprints:
                raise ValueError("Duplicate imagery registered as multiple chips")
            fingerprints.add(fingerprint)
        allowed = set(self.split_config["splits"][split])
        self.samples = [chip for chip in chips if chip["event_id"] in allowed]

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def verify_files(self):
        """Verify only this split against the persisted download inventory."""
        expected = {item["path"]: item for item in self.manifest.get("files", [])}
        paths = {path for chip in self.samples for path in chip["files"].values()}
        paths |= {chip["source_metadata"] for chip in self.samples if "source_metadata" in chip}
        for name in paths:
            if name not in expected or "sha256" not in expected[name] or "size_bytes" not in expected[name]:
                raise ValueError("Real training requires SHA256 and size for every selected file")
            path = dataset_path(name, self.dataset)
            if path.stat().st_size != expected[name]["size_bytes"] or checksum(path) != expected[name]["sha256"]:
                raise ValueError("Training file differs from its frozen manifest")
        return True

    def raw_sample(self, index):
        """Return unaugmented dB arrays, binary/ignore target and source profile."""
        chip = self.samples[index]
        rasters = {name: _read(path, self.dataset) for name, path in chip["files"].items()}
        grids = [value[2] for value in rasters.values()]
        if any(grid != grids[0] for grid in grids[1:]):
            raise ValueError("Training rasters must share shape, CRS and transform")
        valid = np.logical_and.reduce([value[1] for name, value in rasters.items() if name != "dem"])
        arrays = {}
        if self.dataset == "kurosiwo":
            if self.manifest["dataset_version"] == "kuro-cdse-gamma0-utm20-v2":
                crs, transform, _ = grids[0]
                crs = CRS.from_user_input(crs)
                if (crs.utm_zone is None or transform.b or transform.d
                        or transform.a != 20 or transform.e != -20):
                    raise ValueError("V2 data requires a north-up UTM grid at 20 metres")
            for pol in ("vv", "vh"):
                first, second = rasters[f"pre1_{pol}"][0], rasters[f"pre2_{pol}"][0]
                post = rasters[f"post_{pol}"][0]
                if self.manifest.get("sar_units", "linear_power") == "dB":
                    first, second = 10 ** (first / 10), 10 ** (second / 10)
                    post = 10 ** (post / 10)
                elif self.manifest.get("sar_units", "linear_power") != "linear_power":
                    raise ValueError("Kuro imagery requires declared power or dB units")
                valid &= (first > 0) & (second > 0) & (post > 0)
                # Median must be computed in power, before the logarithm.
                arrays[f"pre_{pol}"] = 10 * np.log10(np.maximum((first + second) / 2, np.finfo(np.float32).tiny))
                arrays[f"post_{pol}"] = 10 * np.log10(np.maximum(post, np.finfo(np.float32).tiny))
            labels = rasters["label"][0]
            if "valid" in rasters:
                valid &= rasters["valid"][0] == 1
            target, valid = map_labels(labels, valid, self.dataset)
        else:
            arrays = {name: rasters[name][0].copy() for name in ("pre_vv", "pre_vh", "post_vv", "post_vh")}
            labels = rasters["label"][0]
            target, valid = map_labels(labels, valid, self.dataset)
        profile = next(iter(rasters.values()))[3]
        screening = self.split_config.get("terrain_screening", {})
        if self.split in screening.get("required_splits", []):
            if "dem" not in rasters:
                raise ValueError("Terrain-screened splits require the source DEM")
            terrain_valid = rasters["dem"][1].copy()
            if self.dataset == "kurosiwo":
                if "valid" not in rasters:
                    raise ValueError("Kuro terrain screening requires the native validity mask")
                terrain_valid &= rasters["valid"][1] & (rasters["valid"][0] == 1)
            slope = terrain_p95_slope(rasters["dem"][0], terrain_valid, profile)
            if slope > screening["max_p95_slope_degrees"]:
                raise ValueError("Chip exceeds the registered low-slope terrain limit")
        if not valid.any():
            raise ValueError("Chip has no observable labelled pixels")
        target = np.where(valid, target, IGNORE_INDEX)
        for name in ("pre_vv", "pre_vh", "post_vv", "post_vh"):
            arrays[name] = np.where(valid, arrays[name], np.nan).astype("float32")
        arrays["pre"] = arrays["pre_vv"]
        arrays["post"] = arrays["post_vv"]
        if "dem" in rasters:
            arrays["dem"] = np.where(rasters["dem"][1], rasters["dem"][0], np.nan)
        arrays["valid"] = valid
        arrays["aoi"] = np.ones(valid.shape, dtype=bool)
        return arrays, target, profile

    def __getitem__(self, index):
        import torch
        from .unet import preprocess

        arrays, target, profile = self.raw_sample(index)
        predictable = np.ones(target.shape, bool)
        if self.terrain_policy and "dem" in arrays:
            slope = slope_degrees(arrays["dem"], profile)
            predictable &= ~np.isfinite(slope) | (slope <= 12)
        valid = arrays.pop("valid")
        arrays.pop("aoi")
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            # A single temporal offset preserves physically measured change.
            for pol in ("vv", "vh"):
                offset = rng.uniform(-2, 2)
                arrays[f"pre_{pol}"] += offset
                arrays[f"post_{pol}"] += offset
            turns = int(rng.integers(4))
            flip = bool(rng.integers(2))
            def transform(value):
                rotated = np.rot90(value, turns)
                return np.flip(rotated, axis=1) if flip else rotated
            arrays = {name: transform(value) for name, value in arrays.items()}
            target, valid = transform(target), transform(valid)
            predictable = transform(predictable)
        arrays["valid"] = valid
        inputs, valid = preprocess(arrays, self.in_channels)
        target = np.where(valid, target, IGNORE_INDEX)
        if self.tile_size is not None:
            h, w = valid.shape
            if max(h, w) > self.tile_size:
                raise ValueError("Training chip exceeds the registered tile; do not resize it")
            padding = ((0, self.tile_size-h), (0, self.tile_size-w))
            inputs = np.pad(inputs, ((0, 0), *padding), mode="edge")
            target = np.pad(target, padding, constant_values=IGNORE_INDEX)
            valid = np.pad(valid, padding, constant_values=False)
            predictable = np.pad(predictable, padding, constant_values=False)
        return {"inputs": torch.from_numpy(np.ascontiguousarray(inputs)),
                "target": torch.from_numpy(np.ascontiguousarray(target)),
                "valid": torch.from_numpy(np.ascontiguousarray(valid)),
                "predictable": torch.from_numpy(np.ascontiguousarray(predictable))}
