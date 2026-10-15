import warnings

import numpy as np

from .preprocess import slope_degrees

DEFAULTS = dict(
    post_db=-16,
    decrease_db=3,
    permanent_water_db=-20,
    max_slope_deg=12,
    max_hand_m=20,
    disturbance_db=5,
    neighborhood_size=3,
    mndwi_post=0,
    mndwi_increase=0.2,
    ndvi_decrease=0.25,
    ndbi_increase=0.2,
)


def _median(values):
    values = values[np.isfinite(values)]
    return np.median(values) if len(values) else np.nan


def baseline(arrays, profile, thresholds=None):
    t = DEFAULTS | (thresholds or {})
    size = int(t["neighborhood_size"])
    if size < 1 or size % 2 == 0:
        raise ValueError("Neighborhood size must be a positive odd integer")
    # Same values as generic_filter(_median, mode="constant", cval=nan); vectorised (P04).
    pre = nan_median_filter(arrays["pre"], size)
    post = nan_median_filter(arrays["post"], size)
    valid = arrays["valid"].copy()
    decrease = pre - post
    permanent = pre < t["permanent_water_db"]
    if "permanent_water" in arrays:
        permanent |= arrays["permanent_water"] == 1
    flood = valid & (post < t["post_db"]) & (decrease >= t["decrease_db"]) & ~permanent
    limitations = [
        "SAR darkening is candidate inundation, not confirmed flooding or closure.",
        "Geospatial reprojection aligns grids; residual subpixel co-registration error is not estimated.",
        "SAR layover and shadow are not explicitly modeled; steep-terrain observations may be unreliable.",
    ]
    if "dem" in arrays:
        dem = arrays["dem"]
        slope = slope_degrees(dem, profile)
        flood &= ~np.isfinite(slope) | (slope <= t["max_slope_deg"])
        if np.any(valid & ~np.isfinite(slope)):
            limitations.append(
                "DEM gaps: slope suppression unavailable at those pixels; SAR candidates retained."
            )
    else:
        limitations.append("DEM missing: terrain suppression unavailable.")
    quality_flags = {}
    if "hand" in arrays:
        # Soft flag only (D32, HR-X-001): debris floods can rise far above the channel.
        hand = arrays["hand"]
        high = flood & np.isfinite(hand) & (hand > t["max_hand_m"])
        quality_flags["hand_soft_flag"] = {
            "policy": "candidates above max_hand_m are kept unchanged and only counted",
            "max_hand_m": t["max_hand_m"],
            "water_candidates": int(flood.sum()),
            "above_max_hand": int(high.sum()),
            "hand_unknown": int((flood & ~np.isfinite(hand)).sum()),
        }
        limitations.append(
            f"HAND is a soft flag only: {int(high.sum())} of {int(flood.sum())} water-candidate "
            f"pixels lie more than {t['max_hand_m']} m above the drainage network; they are kept "
            "unchanged and only counted (not marked individually); small tributaries may be "
            "absent from the network."
        )
    disturbance = valid & (np.abs(decrease) >= t["disturbance_db"]) & ~flood
    optical_water = np.zeros(valid.shape, bool)
    optical_disturbance = np.zeros(valid.shape, bool)
    optical_valid = np.zeros(valid.shape, bool)
    if "optical_pre" in arrays and "optical_post" in arrays:
        indices = []
        for key in ("optical_pre", "optical_post"):
            b = arrays[key]
            good = (
                arrays["aoi"]
                & np.isfinite(b["green"])
                & np.isfinite(b["nir"])
                & np.isfinite(b["swir"])
                & np.isfinite(b["red"])
            )
            if "quality" in b:
                good &= np.isin(b["quality"], [2, 4, 5, 6])
            else:
                good[:] = False
                limitations.append(
                    "Optical quality mask absent: optical evidence unsupported."
                )

            def index(a, c):
                return np.divide(
                    a - c,
                    a + c,
                    out=np.full_like(a, np.nan),
                    where=np.abs(a + c) > 1e-8,
                )

            indices.append(
                (
                    index(b["green"], b["swir"]),
                    index(b["nir"], b["red"]),
                    good,
                    index(b["swir"], b["nir"]),
                )
            )
        before, after = indices
        optical_valid = before[2] & after[2]
        optical_water = (
            optical_valid
            & (after[0] > t["mndwi_post"])
            & (after[0] - before[0] > t["mndwi_increase"])
        )
        optical_disturbance = optical_valid & (
            (before[1] - after[1] > t["ndvi_decrease"])
            | (after[3] - before[3] > t["ndbi_increase"])
        )
    return dict(
        flood=flood,
        disturbance=disturbance,
        valid=valid,
        unobservable=arrays["aoi"] & ~valid,
        optical_water=optical_water,
        optical_disturbance=optical_disturbance,
        optical_valid=optical_valid,
        thresholds=t,
        limitations=limitations,
        quality_flags=quality_flags,
    )


def detect(backend, arrays, profile, thresholds=None, model_config=None):
    result = baseline(arrays, profile, thresholds)
    if backend == "baseline":
        return result
    if backend == "unet":
        from .unet import infer
    elif backend == "terramind":
        from .terramind import infer
    else:
        raise ValueError("Unknown detector")

    result["flood"], result["model_metadata"] = infer(arrays, profile, model_config)
    # Model-level limitations (e.g. measured held-out results) travel with every run's report.
    result["limitations"] += [line for line in result["model_metadata"].get("limitations", [])
                              if line not in result["limitations"]]
    result["valid"] &= np.isfinite(arrays["post_vh"])
    if backend == "unet":
        if result["model_metadata"].get("model_version", "").startswith("unet-kuro-v2") and result["thresholds"]["max_slope_deg"] != 12:
            raise ValueError("V2 slope threshold must match the registered calibration policy (12 degrees)")
        result["valid"] &= np.isfinite(arrays["pre_vh"])
        if "dem" in arrays:
            slope = slope_degrees(arrays["dem"], profile)
            result["flood"] &= ~np.isfinite(slope) | (slope <= result["thresholds"]["max_slope_deg"])
        result["flood"] &= result["valid"] & arrays["aoi"]
        size = int(result["thresholds"]["neighborhood_size"])
        change = nan_median_filter(arrays["pre"], size) - nan_median_filter(arrays["post"], size)
        result["disturbance"] = (result["valid"] & (np.abs(change) >= result["thresholds"]["disturbance_db"])
                                 & ~result["flood"])
        if "hand" in arrays:
            high = result["flood"] & np.isfinite(arrays["hand"]) & (arrays["hand"] > result["thresholds"]["max_hand_m"])
            result["quality_flags"]["hand_soft_flag"].update(
                water_candidates=int(result["flood"].sum()), above_max_hand=int(high.sum()),
                hand_unknown=int((result["flood"] & ~np.isfinite(arrays["hand"])).sum()))
            result["limitations"] = [line for line in result["limitations"] if not line.startswith("HAND is a soft flag")]
            result["limitations"].append(f"HAND is a soft flag: {int(high.sum())} of {int(result['flood'].sum())} model candidates are high and retained.")
    result["unobservable"] = arrays["aoi"] & ~result["valid"]
    result["limitations"].append(
        "Model predicts post-event water; stable pre-event water excluded where supported."
    )
    return result


# hazard-v2 (P04, A-D22): added layers and filters on top of the registered rule baseline.
# Values are justified in docs/HAZARD_SCIENCE.md (physics and pre-event no-change data only).
HAZARD_V2_DEFAULTS = dict(
    permanent_water_fraction=0.5,  # water-like (pre VV < permanent_water_db) on at least half the dates
    z_min=3.0,                     # temporal z-score magnitude for change (pre-stack mean and spread)
    z_sd_floor_db=1.0,             # lower bound on the per-pixel temporal spread
    min_pre_for_z=3,               # fewer pre dates: z unavailable, log-ratio only
    corridor_hand_m=30.0,          # valley-bottom corridor for possible_debris
    mmu_m2=1600.0,                 # minimum mapping unit (4 pixels at 20 m)
    optical_clear_classes=(4, 5, 6),  # SCL vegetation, not vegetated, water; 2 = topographic shadow
    bsi_increase=0.1,              # bare-soil index rise (supporting evidence only)
    optical_max_gap_days=30,       # longer optical pre/post gaps are seasonal-prone
)


def nan_median_filter(values, size):
    """Vectorised equivalent of generic_filter(values, _median, size, mode="constant",
    cval=nan): the median of the finite values in each window, NaN if none."""
    pad = size // 2
    finite = np.where(np.isfinite(values), values, np.nan).astype("float64")  # _median drops +-inf
    padded = np.pad(finite, pad, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows stay NaN
        median = np.nanmedian(windows.reshape(*values.shape, size * size), axis=-1)
    return median.astype(values.dtype if np.issubdtype(values.dtype, np.floating) else "float64")


def _remove_small(mask, min_pixels):
    """Drop 8-connected components smaller than min_pixels."""
    from scipy.ndimage import label

    if min_pixels <= 1 or not mask.any():
        return mask
    labels, _ = label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labels]


def hazard_v2(evidence, arrays, profile, thresholds=None, settings=None):
    """Two-sided change, persistence-based permanent water, valley-bottom debris class, MMU.

    `evidence` is the detector result after observability masks; flood_extent keeps its
    detector's definition, minus persistent water and sub-MMU fragments.
    """
    t = DEFAULTS | (thresholds or {})
    h = HAZARD_V2_DEFAULTS | (settings or {})
    valid = evidence["valid"]
    size = int(t["neighborhood_size"])
    stack = arrays.get("pre_stack")
    if stack is None:
        stack = arrays["pre"][np.newaxis]
    filtered = np.stack([nan_median_filter(layer, size) for layer in stack])
    post = nan_median_filter(arrays["post"], size)
    pre = nan_median_filter(arrays["pre"], size)
    finite = np.isfinite(filtered)
    dates = finite.sum(axis=0)
    with np.errstate(invalid="ignore"):
        water_dates = (finite & (filtered < t["permanent_water_db"])).sum(axis=0)
        fraction = np.where(dates > 0, water_dates / np.maximum(dates, 1), np.nan)
    permanent = valid & (fraction >= h["permanent_water_fraction"])
    if "permanent_water" in arrays:
        permanent |= valid & (arrays["permanent_water"] == 1)
    ephemeral = valid & (water_dates > 0) & ~permanent
    log_ratio = post - pre  # dB; negative = darkening (water-like), positive = brightening
    limitations = []
    z = np.full(post.shape, np.nan, dtype="float32")
    if len(stack) >= h["min_pre_for_z"]:
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels stay NaN
            mean = np.nanmean(filtered, axis=0)
            spread = np.nanstd(filtered, axis=0, ddof=1)
        ok = dates >= h["min_pre_for_z"]
        z[ok] = ((post - mean) / np.maximum(spread, h["z_sd_floor_db"]))[ok]
    else:
        limitations.append(
            f"Only {len(stack)} pre-event date(s): temporal z-score unavailable; change uses the "
            "log-ratio alone."
        )
    confirmed = ~np.isfinite(z) | (np.abs(z) >= h["z_min"])
    res = abs(profile["transform"].a) * abs(profile["transform"].e)
    min_pixels = int(np.ceil(h["mmu_m2"] / res))
    flood = _remove_small(evidence["flood"] & ~permanent, min_pixels)
    strong = valid & (np.abs(log_ratio) >= t["disturbance_db"]) & confirmed & ~flood & ~permanent
    if "hand" in arrays:
        corridor = np.isfinite(arrays["hand"]) & (arrays["hand"] <= h["corridor_hand_m"])
    else:
        corridor = np.zeros(valid.shape, bool)
        limitations.append("HAND unavailable: no valley-bottom corridor, so no possible_debris layer.")
    disturbance = _remove_small(evidence["disturbance"] & ~flood & ~permanent, min_pixels)
    # possible_debris is a named subset of disturbance (audit M2), so every debris pixel is
    # also SAR disturbance for roads and scenarios.
    debris = _remove_small(strong & corridor & disturbance, min_pixels)
    corridor_sensitivity = {
        f"hand_le_{int(limit)}m": int(_remove_small(
            strong & disturbance & np.isfinite(arrays["hand"]) & (arrays["hand"] <= limit), min_pixels
        ).sum()) if "hand" in arrays else None
        for limit in (20.0, 30.0, 50.0)
    }
    stats = {
        "hazard_version": "hazard-v2",
        "settings": h,
        "pre_dates": int(len(stack)),
        "z_available": bool(np.isfinite(z).any()),
        "mmu_pixels": min_pixels,
        "possible_debris_corridor_sensitivity": corridor_sensitivity,  # reported, not selected
        "pixels": {
            "flood_extent": int(flood.sum()),
            "flood_removed_by_mmu_or_permanent_water": int((evidence["flood"] & ~flood).sum()),
            "disturbance": int(disturbance.sum()),
            "possible_debris": int(debris.sum()),
            "permanent_water": int(permanent.sum()),
            "ephemeral_water": int(ephemeral.sum()),
            "brightening": int((strong & (log_ratio > 0)).sum()),
            "darkening_not_water": int((strong & (log_ratio < 0)).sum()),
        },
    }
    limitations.append(
        "possible_debris is an unvalidated class: strong two-sided SAR change "
        f"(|log-ratio| >= {t['disturbance_db']} dB and |z| >= {h['z_min']} where available) within the "
        f"valley-bottom corridor (HAND <= {h['corridor_hand_m']} m), outside water candidates. It can "
        "also be landslides, scour, construction, vegetation or moisture change; never confirmed debris."
    )
    return dict(
        flood=flood,
        disturbance=disturbance,
        possible_debris=debris,
        permanent_water=permanent,
        ephemeral_water=ephemeral,
        z_score=z,
        log_ratio=log_ratio,
        stats=stats,
        limitations=limitations,
    )


def _normalized(a, b):
    return np.divide(a - b, a + b, out=np.full_like(a, np.nan), where=np.abs(a + b) > 1e-8)


def optical_corroboration(arrays, hazard, thresholds=None, settings=None, gap_days=None):
    """Sentinel-2 support for SAR layers on clear pixels only (T04.05).

    Optical change never creates a SAR layer; it only corroborates one. Water support:
    MNDWI rise; ground support: NDVI loss or a bare-soil index (BSI) rise.
    """
    t = DEFAULTS | (thresholds or {})
    h = HAZARD_V2_DEFAULTS | (settings or {})
    pre, post = arrays["optical_pre"], arrays["optical_post"]
    classes = list(h["optical_clear_classes"])
    clear = arrays["aoi"].copy()
    for scene in (pre, post):
        clear &= np.isin(scene.get("quality", np.full(clear.shape, np.nan)), classes)
        for band in ("green", "red", "nir", "swir"):
            clear &= np.isfinite(scene[band])
    mndwi = [_normalized(s["green"], s["swir"]) for s in (pre, post)]
    ndvi = [_normalized(s["nir"], s["red"]) for s in (pre, post)]
    has_blue = all("blue" in s for s in (pre, post))
    limitations = []
    with np.errstate(invalid="ignore"):
        water = clear & (mndwi[1] > t["mndwi_post"]) & (mndwi[1] - mndwi[0] > t["mndwi_increase"])
        ground = clear & (ndvi[0] - ndvi[1] > t["ndvi_decrease"])
        if has_blue:
            bsi = [
                _normalized(s["swir"] + s["red"], s["nir"] + s["blue"]) for s in (pre, post)
            ]
            ground |= clear & (bsi[1] - bsi[0] > h["bsi_increase"])
        else:
            limitations.append("Sentinel-2 blue band absent: bare-soil index unavailable; NDVI loss only.")
    seasonal = gap_days is not None and gap_days > h["optical_max_gap_days"]
    if seasonal:
        limitations.append(
            f"Optical pre and post scenes are {gap_days:.0f} days apart (> {h['optical_max_gap_days']}): "
            "vegetation and water indices change with season, so optical support is seasonal-prone."
        )

    def share(layer, support):
        observed = layer & clear
        return {
            "pixels": int(layer.sum()),
            "clear_pixels": int(observed.sum()),
            "supported_pixels": int((observed & support).sum()),
            "supported_fraction_of_clear": round(float((observed & support).sum()) / max(1, int(observed.sum())), 4),
        }

    stats = {
        "clear_classes": classes,
        "clear_percent_of_aoi": round(100 * float(clear.sum()) / max(1, int(arrays["aoi"].sum())), 2),
        "gap_days": None if gap_days is None else round(float(gap_days), 1),
        "seasonal_prone": bool(seasonal),
        "bare_soil_index": has_blue,
        "flood_extent": share(hazard["flood"], water),
        "possible_debris": share(hazard["possible_debris"], ground),
        "water_support_pixels": int(water.sum()),
        "ground_support_pixels": int(ground.sum()),
    }
    return {"clear": clear, "water": water, "ground": ground, "stats": stats, "limitations": limitations}
