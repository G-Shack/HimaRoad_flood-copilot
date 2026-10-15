"""Self-contained Cartesian vector map, KML, and one-page evidence report."""

import base64
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import KeepInFrame, Paragraph, SimpleDocTemplate, Spacer

from .translations import NEPALI_REVIEW_STATUS, TEMPLATES

ATTRIBUTIONS = (
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30.",
    "© OpenStreetMap contributors.",
    "Model training: Bountos et al., Kuro Siwo, NeurIPS Datasets and Benchmarks 2024.",
)

LAYERS = {
    "flood_extent": "#2584dd",
    "disturbance": "#da9a27",
    "possible_debris": "#8b3a0e",  # hazard-v2, unvalidated class
    "unobservable": "#999999",
    "optical_water": "#1266a0",
    "optical_disturbance": "#d0a510",
    "roads": "#666666",
    "affected_infrastructure": "#e24b39",
    "settlements": "#9229b5",
    "buildings": "#a78056",
    "destinations": "#16864b",
    "flowpaths": "#0b3d91",  # DEM-derived flow path, not an inundation forecast
}


def _parts(geometry):
    kind, coordinates = geometry["type"], geometry.get("coordinates")
    if kind == "GeometryCollection":
        for member in geometry["geometries"]:
            yield from _parts(member)
    elif kind == "Point":
        yield kind, [coordinates]
    elif kind in ("MultiPoint", "LineString"):
        yield "Point" if kind == "MultiPoint" else kind, coordinates
    elif kind == "Polygon":
        for ring in coordinates:
            yield "LineString", ring
    elif kind in ("MultiLineString", "MultiPolygon"):
        for part in coordinates:
            yield from _parts(
                {
                    "type": "LineString" if kind == "MultiLineString" else "Polygon",
                    "coordinates": part,
                }
            )


# Hover labels are bounded (HR-X-007): only these fields, lists shown as counts.
# The run's GeoJSON layers stay the complete, authoritative record.
HOVER_FIELDS = (
    "id", "name", "name:ne", "kind", "place", "amenity", "healthcare", "highway",
    "destination_tier", "category", "evidence_category", "operational_status",
    "physical_condition", "valid_observation_fraction", "potentially_exposed",
    "potentially_isolated", "isolation_confidence", "cut_in_scenarios",
    "building_count", "potentially_exposed_buildings", "observation_timestamp",
    "label", "source_name", "length_m", "stop_reason",
    "hazard_class", "confidence", "area_m2", "hand_soft_flag", "max_track_votes",
    "optical_support_fraction", "debris_intersection",
)
HOVER_VALUE_CHARS = 80
# Dashboard view (T08.01): fields that are the same for every feature of a run
# are stated once on the page, and roads are drawn by evidence category with
# hover labels only where change was observed (a real run has ~10^5 quiet
# segments). Exports (map.html) keep the full view.
COMPACT_OMIT = ("operational_status", "physical_condition", "observation_timestamp")
ROAD_EVIDENCE = {
    "observed_change": ("#e24b39", 3),
    "no_detected_change": ("#666666", 1.5),
    "unobservable": ("#bbbbbb", 1),
}
QUIET_ROADS = ("no_detected_change", "unobservable")
# Optical change between scenes weeks apart is seasonal-prone and can cover whole
# AOIs; the dashboard view starts with it hidden (one legend click shows it).
HIDDEN_LAYERS = ("optical_water", "optical_disturbance")


def hover_label(properties, compact=False):
    """Short, bounded hover text; long lists become counts."""
    lines = []
    for key in HOVER_FIELDS:
        if key not in properties or properties[key] is None or (compact and key in COMPACT_OMIT):
            continue
        value = properties[key]
        if isinstance(value, (list, tuple)) and len(value) > 6:
            value = f"{len(value)} items (see the GeoJSON layer)"
        text = str(value)
        if len(text) > HOVER_VALUE_CHARS:
            text = text[: HOVER_VALUE_CHARS - 1] + "…"
        lines.append(f"{escape(key)}: {escape(text)}")
    lists = [k for k, v in properties.items() if isinstance(v, list) and k not in HOVER_FIELDS]
    if lists and not compact:
        lines.append(escape(f"also in the layer: {', '.join(sorted(lists)[:6])}"))
    return "<br>".join(lines)


def map_figure(run, compact=False):
    """Longitude/latitude vector view: no geographic basemap or remote assets.
    compact=True is the lighter dashboard view (COMPACT_OMIT, ROAD_EVIDENCE, HIDDEN_LAYERS)."""
    figure = go.Figure()
    report_path = Path(run) / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    layers = []  # (legend name, legend group, color, width, features, hover anchors)
    for name, color in {**LAYERS, "pedestrian_network": "#15a5a0"}.items():
        path = (
            Path(run) / f"{'roads' if name == 'pedestrian_network' else name}.geojson"
        )
        if not path.exists():
            continue
        features = json.loads(path.read_text())["features"]
        if name == "pedestrian_network":
            features = [
                f
                for f in features
                if f.get("properties", {}).get("highway")
                in ("path", "footway", "steps", "pedestrian", "bridleway", "track")
            ]
        if compact and name == "roads":
            groups = {}
            for feature in features:
                category = feature.get("properties", {}).get("evidence_category")
                groups.setdefault(category, []).append(feature)
            for category, members in groups.items():
                road_color, width = ROAD_EVIDENCE.get(category, (color, 2))
                layers.append((
                    f"roads: {str(category).replace('_', ' ')}", f"roads:{category}",
                    road_color, width, members, category not in QUIET_ROADS,
                ))
        else:
            # The pedestrian layer redraws roads features: their labels already exist.
            layers.append((
                name.replace("_", " "), name, color, 2, features, name != "pedestrian_network",
            ))
    for display, group, color, width, features, anchor in layers:
        # Lines carry no hover text; each line/polygon feature gets one small
        # hover marker at its middle vertex, so a label is stored once (HR-X-007).
        points, lines, anchors = ([], [], []), ([], []), ([], [], [])
        for feature in features:
            if not feature.get("geometry"):
                continue
            label = hover_label(feature.get("properties", {}), compact)
            anchored = False
            for kind, coords in _parts(feature["geometry"]):
                if not coords:
                    continue
                # 6 decimals ~ 0.1 m: enough for display, far smaller HTML.
                xs, ys = [round(c[0], 6) for c in coords], [round(c[1], 6) for c in coords]
                if kind == "Point":
                    points[0].extend(xs)
                    points[1].extend(ys)
                    points[2].extend([label] * len(xs))
                    continue
                lines[0].extend(xs + [None])
                lines[1].extend(ys + [None])
                if not anchored and anchor:
                    middle = len(xs) // 2
                    anchors[0].append(xs[middle])
                    anchors[1].append(ys[middle])
                    anchors[2].append(label)
                    anchored = True
        if lines[0]:
            figure.add_trace(
                go.Scattergl(
                    x=lines[0], y=lines[1], mode="lines", hoverinfo="skip",
                    name=display, legendgroup=group, line={"color": color, "width": width},
                )
            )
        for data, size, legend in ((points, 7, not lines[0]), (anchors, 4, False)):
            if data[0]:
                figure.add_trace(
                    go.Scattergl(
                        x=data[0], y=data[1], text=data[2],
                        hovertemplate="%{text}<extra></extra>", mode="markers",
                        name=display, legendgroup=group, showlegend=legend,
                        marker={"color": color, "size": size},
                    )
                )
    if compact:
        figure.for_each_trace(
            lambda trace: trace.update(visible="legendonly"),
            selector=lambda trace: trace.legendgroup in HIDDEN_LAYERS,
        )
    title = (
        "SYNTHETIC TEST — NOT REAL DISASTER RESULTS"
        if report.get("synthetic")
        else "HimaRoad evidence layers"
    )
    context_path = Path(run) / "satellite_context.json"
    if context_path.exists():
        context = json.loads(context_path.read_text())
        west, south, east, north = context["bounds"]
        encoded = base64.b64encode(
            (Path(run) / "satellite_context.png").read_bytes()
        ).decode()
        figure.add_layout_image(
            source="data:image/png;base64," + encoded,
            xref="x",
            yref="y",
            x=west,
            y=north,
            sizex=east - west,
            sizey=north - south,
            sizing="stretch",
            layer="below",
            opacity=0.6,
        )
    figure.update_layout(
        title=title + " — click legend to toggle",
        xaxis_title="Longitude (WGS84)",
        yaxis_title="Latitude (WGS84)",
        template="plotly_white",
        height=650,
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def _kml_geometry(parent, geometry):
    kind = geometry["type"]
    if kind in ("Point", "LineString"):
        node = ET.SubElement(parent, kind)
        coords = (
            [geometry["coordinates"]] if kind == "Point" else geometry["coordinates"]
        )
        ET.SubElement(node, "coordinates").text = " ".join(
            ",".join(map(str, c)) for c in coords
        )
    elif kind == "Polygon":
        node = ET.SubElement(parent, "Polygon")
        for i, ring in enumerate(geometry["coordinates"]):
            boundary = ET.SubElement(
                node, "outerBoundaryIs" if i == 0 else "innerBoundaryIs"
            )
            linear = ET.SubElement(boundary, "LinearRing")
            ET.SubElement(linear, "coordinates").text = " ".join(
                ",".join(map(str, c)) for c in ring
            )
    else:
        node = ET.SubElement(parent, "MultiGeometry")
        if kind == "GeometryCollection":
            members = geometry["geometries"]
        else:
            single = {
                "MultiPoint": "Point",
                "MultiLineString": "LineString",
                "MultiPolygon": "Polygon",
            }[kind]
            members = [
                {"type": single, "coordinates": c} for c in geometry["coordinates"]
            ]
        for member in members:
            _kml_geometry(node, member)


def export_run(run, language="en"):
    run = Path(run)
    if language not in TEMPLATES:
        raise ValueError("Report language must be en or ne")
    report = json.loads((run / "report.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    html = map_figure(run).to_html(include_plotlyjs=True, full_html=True)
    credits = "<footer>" + "".join(f"<p>{escape(text)}</p>" for text in ATTRIBUTIONS) + "</footer>"
    (run / "map.html").write_text(html.replace("</body>", credits + "</body>"))
    artifacts = {"map": str(run / "map.html")}
    for name in LAYERS:
        source = run / f"{name}.geojson"
        if not source.exists():
            continue
        root = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        document = ET.SubElement(root, "Document")
        ET.SubElement(document, "description").text = "\n".join(ATTRIBUTIONS)
        for feature in json.loads(source.read_text())["features"]:
            if not feature.get("geometry"):
                continue
            mark = ET.SubElement(document, "Placemark")
            props = feature.get("properties", {})
            ET.SubElement(mark, "name").text = str(
                props.get("name", props.get("id", name))
            )
            ET.SubElement(mark, "description").text = json.dumps(
                props, ensure_ascii=False
            )
            _kml_geometry(mark, feature["geometry"])
        path = run / f"{name}.kml"
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        artifacts[name + "_kml"] = str(path)
    # Devanagari needs both a licensed font and verified shaping, unavailable here.
    if language == "ne":
        status = {
            "status": "unsupported_font_rendering",
            "language": "ne",
            "translation_status": NEPALI_REVIEW_STATUS,
            "reason": "No verified Devanagari font and shaping engine configured. English PDF remains available.",
        }
        (run / "report.ne.status.json").write_text(json.dumps(status, indent=2))
        artifacts["nepali"] = status
        return artifacts
    headings = TEMPLATES["en"]
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 8
    styles["BodyText"].leading = 10
    story = [Paragraph(headings["title"], styles["Title"])]
    if report.get("synthetic"):
        story.append(
            Paragraph("SYNTHETIC TEST — NOT REAL DISASTER RESULTS", styles["Heading2"])
        )

    def add(label, value):
        text = (
            json.dumps(value, ensure_ascii=True, default=str)
            if not isinstance(value, str)
            else value
        )
        story.append(
            Paragraph(f"<b>{escape(label)}</b>: {escape(text)}", styles["BodyText"])
        )
        story.append(Spacer(1, 5))

    for key in (
        "event_date",
        "bbox",
        "status",
        "summary",
        "observation_timestamp",
        "observation_age_hours",
        "source_dataset",
        "priority_inspection",
        "evidence_categories",
        "scenarios",
    ):
        if key in report:
            add(key.replace("_", " ").capitalize(), report[key])
    fields = (
        "source",
        "dataset",
        "scene_id",
        "id",
        "timestamp",
        "snapshot",
        "attribution",
        "license",
    )
    sources = {
        k: manifest[k]
        for k in ("provider", "source", "dataset", "attribution")
        if k in manifest
    }
    for key in ("pre", "post", "osm"):
        if key in manifest:
            values = (
                manifest[key] if isinstance(manifest[key], list) else [manifest[key]]
            )
            sources[key] = [{k: v[k] for k in fields if k in v} for v in values]
    add("Sources and acquisition dates", sources)
    add("Attribution", " ".join(ATTRIBUTIONS))
    add(
        "Evidence interpretation",
        "Observed change: satellite surface-change signal. Likely impact: corroborating change evidence. "
        "Possible impact: ambiguous or limited evidence. "
        "Unobservable: insufficient valid observations. No detected change: no signal in observed pixels. "
        "All are screening evidence; operational closure and physical destruction remain unknown.",
    )
    add(
        "Limitations",
        report.get(
            "limitations",
            ["Satellite change is not confirmation of closure or destruction."],
        ),
    )
    add(
        "Use",
        "Research prototype. Suspected disruptions require independent inspection. Satellite grayscale context appears only when persisted; no online basemap.",
    )
    pdf = run / "report.pdf"
    document = SimpleDocTemplate(str(pdf), pagesize=A4, topMargin=24, bottomMargin=24)
    document.build(
        [KeepInFrame(document.width, document.height - 12, story, mode="shrink")]
    )
    artifacts["pdf"] = str(pdf)
    return artifacts
