"""Coverage-qualified layer inclusion ranges, never confidence intervals."""

from backend.app.schemas import Contract, EvidenceClaim


class ExposureBand(Contract):
    layer: str
    water_only: int | None
    water_or_disturbance: int | None
    unobservable_features: int | None
    feature_count: int | None
    partial_coverage_features: int | None
    interpretation: str = "Layer inclusion sensitivity, not a statistical confidence interval."


def exposure_band(claims: tuple[EvidenceClaim, ...], layer: str, *, complete: bool) -> ExposureBand:
    if layer not in {"roads", "buildings"}:
        raise ValueError("unsupported exposure layer")
    if not complete:
        return ExposureBand(
            layer=layer,
            water_only=None,
            water_or_disturbance=None,
            unobservable_features=None,
            feature_count=None,
            partial_coverage_features=None,
        )
    features = [c for c in claims if c.kind == "feature" and c.label == layer]
    if len({c.subject for c in features}) != len(features):
        raise ValueError("duplicate feature claims")
    properties = [c.value["properties"] for c in features]
    return ExposureBand(
        layer=layer,
        water_only=sum(p["flood_intersection"] for p in properties),
        water_or_disturbance=sum(
            p["flood_intersection"] or p["disturbance_intersection"] for p in properties
        ),
        unobservable_features=sum(p["status"] == "unknown" for p in properties),
        feature_count=len(properties),
        partial_coverage_features=sum(p["valid_observation_fraction"] < 1 for p in properties),
    )
