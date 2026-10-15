"""Hypothetical single-link reopening gains, excluding baseline-unreachable places."""

from typing import cast

from backend.app.analysis.routes import Network, route
from backend.app.schemas import Contract


class RepairPriority(Contract):
    link_id: str
    reconnected_settlement_ids: tuple[str, ...]
    settlement_count: int
    exposed_buildings_assigned_to_reconnected_settlements: int | None
    scenario: str
    assumptions: tuple[str, ...]


def repair_priority(network: Network, scenario: str = "2") -> tuple[RepairPriority, ...]:
    if scenario == "baseline" or scenario not in network.removed:
        raise ValueError("repair requires a hypothetical closure scenario")
    eligible = [
        s
        for s in sorted(network.sources)
        if route(network, s, "baseline").result == "found"
        and route(network, s, scenario).result == "none"
    ]
    records = []
    for link in sorted(network.removed[scenario]):
        reconnected = tuple(
            s for s in eligible if route(network, s, scenario, frozenset({link})).result == "found"
        )
        counts = [
            network.settlements.get(s, {}).get("potentially_exposed_buildings") for s in reconnected
        ]
        buildings = (
            sum(cast(list[int], counts)) if all(type(n) is int and n >= 0 for n in counts) else None
        )
        records.append(
            RepairPriority(
                link_id=link,
                reconnected_settlement_ids=reconnected,
                settlement_count=len(reconnected),
                exposed_buildings_assigned_to_reconnected_settlements=buildings,
                scenario=scenario,
                assumptions=(
                    *network.assumptions,
                    "Single-link gains only; joint repairs may help when each single gain is zero.",
                    "Buildings use nearest-settlement assignment; affiliation unknown.",
                ),
            )
        )
    return tuple(
        sorted(
            records,
            key=lambda r: (
                -r.settlement_count,
                -(r.exposed_buildings_assigned_to_reconnected_settlements or 0),
                r.link_id,
            ),
        )
    )
