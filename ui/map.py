"""Tile-free claim map. Only cited admitted geometries are displayed."""

from html import escape

import plotly.graph_objects as go
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.geometry.base import BaseMultipartGeometry

from backend.app.reports.facts import verify_report
from backend.app.schemas import ReportFacts

COLORS = {
    "flood_extent": ("#2584dd", "Candidate floodwater"),
    "disturbance": ("#c39442", "Candidate disturbance"),
    "possible_debris": ("#a86e25", "possible debris (unvalidated)"),
    "unobservable": ("#929a9e", "Unknown / unobservable"),
    "roads": ("#475569", "Mapped roads"),
    "buildings": ("#86997b", "Mapped buildings"),
    "settlements": ("#007f78", "Mapped settlements"),
    "destinations": ("#7d688d", "Mapped destinations"),
}


def claim_map(facts: ReportFacts, selected: str | None = None) -> go.Figure:
    verify_report(facts)
    figure = go.Figure()
    if facts.availability == "unavailable":
        return figure
    seen = set()
    for claim in facts.claims:
        if claim.kind == "feature" and claim.label in COLORS:
            color, label = COLORS[claim.label]
            geometry = claim.value["geometry"]
        elif claim.kind in {"route", "flow_path"}:
            geometry = claim.value.get("geometry")
            color = "#9a568d" if claim.kind == "route" else "#b9803c"
            label = (
                (
                    f"Mapped baseline route · {claim.label}"
                    if claim.scenario == "baseline"
                    else f"Hypothetical route · S{claim.scenario} · {claim.label}"
                )
                if claim.kind == "route"
                else "DEM flow path · not an inundation forecast"
            )
        else:
            continue
        if geometry is None:
            continue
        geom = shape(geometry)
        parts = list(geom.geoms) if isinstance(geom, BaseMultipartGeometry) else [geom]
        for part in parts:
            if isinstance(part, Polygon):
                x, y = part.exterior.xy
            elif isinstance(part, Point):
                x, y = [part.x], [part.y]
            elif isinstance(part, LineString):
                x, y = part.xy
            else:
                continue
            figure.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(y),
                    name=label,
                    legendgroup=label,
                    showlegend=label not in seen,
                    mode="markers" if part.geom_type == "Point" else "lines",
                    line={
                        "color": "#cf3c54" if claim.id == selected else color,
                        "width": 4 if claim.id == selected else 1.5,
                        "dash": "dash" if claim.kind in {"route", "flow_path"} else "solid",
                    },
                    marker={"color": color, "size": 10 if claim.id == selected else 6},
                    customdata=[[claim.id] for _ in x],
                    hovertext=escape(claim.subject),
                    hoverinfo="text",
                )
            )
            seen.add(label)
    figure.update_layout(
        height=480,
        margin={"l": 12, "r": 12, "t": 12, "b": 12},
        xaxis_title="Longitude (WGS84)",
        yaxis_title="Latitude (WGS84)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        legend={"orientation": "h", "y": -0.2},
        dragmode="select",
        template="plotly_white",
    )
    return figure
