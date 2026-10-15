"""Exhaustive mapping of the reviewed HimaRoad vocabulary.

Independent closure records stay unknown here: an upstream status string alone
does not authenticate an independent source. Routing assumptions are separate.
"""

from types import MappingProxyType

from backend.app.schemas import Status

TABLES = MappingProxyType(
    {
        "hazard": MappingProxyType(
            {
                "flood_extent": Status.OBSERVED_WATER,
                "optical_water": Status.OBSERVED_WATER,
                "disturbance": Status.POSSIBLE_DEBRIS,
                "possible_debris": Status.POSSIBLE_DEBRIS,
                "optical_disturbance": Status.POSSIBLE_DEBRIS,
                "unobservable": Status.UNKNOWN,
            }
        ),
        "evidence": MappingProxyType(
            {
                "observed_change": Status.POTENTIALLY_AFFECTED,
                "likely_impact": Status.POTENTIALLY_AFFECTED,
                "possible_impact": Status.POTENTIALLY_AFFECTED,
                "unobservable": Status.UNKNOWN,
                "no_detected_change": Status.NO_DETECTED_CHANGE,
            }
        ),
        "settlement": MappingProxyType(
            {
                "potentially_road_cut": Status.MODELLED_DISCONNECTED,
                "access_uncertain": Status.UNKNOWN,
                "baseline_network_incomplete": Status.BASELINE_UNREACHABLE,
                "mapped_road_connection_remains": Status.MAPPED_CONNECTION,
            }
        ),
        "operational": MappingProxyType(
            {
                "unknown": Status.UNKNOWN,
                "confirmed_open": Status.UNKNOWN,
                "confirmed_closed": Status.UNKNOWN,
            }
        ),
    }
)


def map_status(domain: str, category: str) -> Status:
    """Unknown categories fail closed; no default can upgrade evidence."""
    try:
        return TABLES[domain][category]
    except KeyError as exc:
        raise ValueError(f"Unreviewed HimaRoad vocabulary: {domain}.{category}") from exc
