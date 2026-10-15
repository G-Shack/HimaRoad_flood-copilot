"""Offline one-page report. No arbitrary HTML, remote URLs or filesystem fetches."""

import base64
import hashlib
import io
import json
from html import escape
from pathlib import Path

from backend.app.copilot.offline import Language
from backend.app.reports.attributions import ATTRIBUTIONS
from backend.app.reports.facts import ReportFacts, verify_report
from backend.app.reports.rendering import ReportRow, project
from backend.app.reports.templates import text
from backend.app.schemas import EvidenceClaim


class ReportLayoutError(ValueError):
    pass


def font_bytes(font: Path) -> bytes:
    config = json.loads(Path(__file__).with_name("font.json").read_text())
    if "reference" in {part.casefold() for part in font.resolve().parts}:
        raise ValueError("Reference font path refused before reading")
    data = font.read_bytes()
    if hashlib.sha256(data).hexdigest() != config["sha256"]:
        raise ValueError("Noto font checksum mismatch")
    return data


def map_claims(facts: ReportFacts) -> tuple[EvidenceClaim, ...]:
    verify_report(facts)
    return _map_claims(facts)


def _map_claims(facts: ReportFacts) -> tuple[EvidenceClaim, ...]:
    return tuple(
        c
        for c in facts.claims
        if c.kind == "feature"
        and c.label
        in {
            "flood_extent",
            "disturbance",
            "possible_debris",
            "unobservable",
            "roads",
            "settlements",
        }
    )


def map_thumbnail(facts: ReportFacts) -> bytes | None:
    return _draw_map(map_claims(facts), facts.availability)


def _draw_map(features: tuple[EvidenceClaim, ...], availability: str) -> bytes | None:
    if not features or availability == "unavailable":
        return None
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure
    from shapely.geometry import LineString, Point, Polygon, shape
    from shapely.geometry.base import BaseMultipartGeometry

    figure = Figure(figsize=(5, 1.7), dpi=150)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    colors = {
        "flood_extent": "#278ec4",
        "disturbance": "#d09b45",
        "possible_debris": "#a86e25",
        "unobservable": "#bbbbbb",
        "roads": "#40515e",
        "settlements": "#0a6653",
    }
    # Preserve every vertex and the existing layer order; batch artists only.
    for label in dict.fromkeys(c.label for c in features):
        polygons, polygon_colors, lines, points = [], [], [], []
        for claim in features:
            if claim.label != label:
                continue
            geometry = shape(claim.value["geometry"])
            parts = geometry.geoms if isinstance(geometry, BaseMultipartGeometry) else [geometry]
            for part in parts:
                if isinstance(part, Polygon):
                    polygons.append([coordinate[:2] for coordinate in part.exterior.coords])
                    polygon_colors.append(to_rgba(colors[label], 0.5))
                    for ring in part.interiors:
                        polygons.append([coordinate[:2] for coordinate in ring.coords])
                        polygon_colors.append(to_rgba("white"))
                elif isinstance(part, LineString):
                    lines.append([coordinate[:2] for coordinate in part.coords])
                elif isinstance(part, Point):
                    points.append((part.x, part.y))
        if polygons:
            axis.add_collection(PolyCollection(polygons, facecolors=polygon_colors, zorder=1))
        if lines:
            axis.add_collection(
                LineCollection(
                    lines, colors=colors[label], linewidths=0.7, capstyle="projecting", zorder=2
                )
            )
        if points:
            x, y = zip(*points, strict=True)
            axis.scatter(x, y, s=8, color=colors[label])
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="datalim")
    axis.tick_params(labelsize=6)
    axis.set_xlabel("Longitude / latitude (WGS84)", fontsize=6)
    figure.tight_layout(pad=0.4)
    stream = io.BytesIO()
    figure.savefig(stream, format="png")
    return stream.getvalue()


def synopsis_ids(facts: ReportFacts) -> tuple[str, ...]:
    """One deterministic selection for both languages; complete facts remain exported."""
    selected = []
    has_offsets = any(c.kind == "temporal_offsets" for c in facts.claims)
    schema3 = any(c.parameters.get("schema_version") == "3.0" for c in facts.claims)
    settlements = 0
    for c in facts.claims:
        if schema3:
            if c.kind in {"bbox", "sensitivity", "temporal_offsets"}:
                continue
            if c.kind == "thresholds" and c.label.rsplit("/", 1)[-1] not in {
                "post_db",
                "decrease_db",
                "max_slope_deg",
            }:
                continue
            if c.kind == "source" and c.label.startswith("/optical_"):
                continue
            if (
                c.kind == "source"
                and c.label.endswith("/timestamp")
                and not (c.label == "/post/timestamp" or c.label.startswith("/pre/"))
            ):
                # Secondary passes remain in complete exports; a synopsis uses
                # primary acquisition timestamps, without duplicating tracks.
                continue
            if c.kind == "area" and c.label.startswith("optical_"):
                continue
            if c.kind == "summary" and c.label.rsplit("/", 1)[-1] in {
                "corroborated_road_cut",
                "depends_only_on_unknown",
                "settlements_total",
                "settlements_from_building_clusters",
            }:
                continue
            if c.kind == "feature" and c.label == "settlements":
                settlements += 1
                if settlements > 3:
                    continue
        if c.kind == "feature" and c.label == "settlements":
            selected.append(c.id)
        elif c.kind in {
            "event_date",
            "bbox",
            "status",
            "detector",
            "area",
            "summary",
            "sensitivity",
            "scenarios",
        }:
            selected.append(c.id)
        elif c.kind == "limitation":
            # Authored caveats all retained; arbitrary source prose is in full export.
            from backend.app.evidence.limitations import CATALOGUE

            if c.label in CATALOGUE:
                selected.append(c.id)
        elif c.kind == "source" and c.label.endswith("/timestamp"):
            selected.append(c.id)
        elif c.kind in {"thresholds", "temporal_offsets"} and type(c.value) in {int, float}:
            selected.append(c.id)
        elif c.kind == "source" and c.label.startswith("/temporal_gaps_hours/") and not has_offsets:
            selected.append(c.id)
    return tuple(selected)


def report_html(facts: ReportFacts, language: Language, font: Path, *, full: bool = False) -> str:
    verify_report(facts)
    if language not in {"en", "ne"}:
        raise ValueError("Unsupported language")
    data = base64.b64encode(font_bytes(font)).decode()
    by_id = {c.id: c for c in facts.claims}
    identifiers = tuple(i for s in facts.sections for i in s.claim_ids)
    citations = {
        identifier: f"C{index:03d}" for index, identifier in enumerate(identifiers, start=1)
    }
    selected = set(identifiers if full else synopsis_ids(facts))
    blocks = []
    for section in facts.sections:
        rows = [
            ReportRow(claim_id=i, text=project(by_id[i], language))
            for i in section.claim_ids
            if i in selected
        ]
        compact = [r for r in rows if by_id[r.claim_id].kind in {"thresholds", "bbox"}]
        rows = [r for r in rows if r not in compact]
        compact_text = "; ".join(
            escape(row.text.replace("/thresholds/", "").replace("/bbox/", "bbox/"))
            + f" [{citations[row.claim_id]}]"
            for row in compact
        )
        lines = (f"<p>{compact_text}</p>" if compact else "") + "".join(
            f"<p>{escape(row.text)} <b>[{citations[row.claim_id]}]</b></p>" for row in rows
        )
        if not rows and not compact:
            lines = (
                f"<p>{escape(text('unavailable', language))}</p>" if not section.claim_ids else ""
            )
        blocks.append(f"<section><h2>{escape(text(section.id, language))}</h2>{lines}</section>")
    image = _draw_map(_map_claims(facts), facts.availability)
    map_html = (
        f'<img alt="{escape(text("map", language))}"'
        f""
        f' src="data:image/png;base64,{base64.b64encode(image).decode()}">'
        if image
        else f"<p>{escape(text('unavailable', language))}</p>"
    )
    synthetic = any(
        "synthetic" in c.quality_flags or any(f.startswith("synthetic_") for f in c.quality_flags)
        for c in facts.claims
    )
    stamp = f"<strong>{escape(text('synthetic', language))}</strong>" if synthetic else ""
    credits = "".join(f"<p>{escape(line)}</p>" for line in ATTRIBUTIONS)
    review = f'<p class="review">{escape(facts.native_review)}</p>'
    appendix = ""
    if full:
        appendix = "<h2>Claim reference index</h2>" + "".join(
            f"<p>{citations[i]}: {i}</p>" for i in identifiers
        )
    return f'''<!doctype html><html lang="{language}"><meta charset="utf-8"><style>
    @font-face {{font-family:Noto;src:url(data:font/ttf;base64,{data})}}
    @page {{size:A4;margin:8mm}}
    body {{font-family:Noto,sans-serif;font-size:7.25pt;color:#193441;line-height:1.5}}
    h1 {{font-size:18pt;margin:0;color:#126b67}}
    h2 {{font-size:9pt;margin:5px 0 3px;color:#126b67;break-after:avoid}}
    p {{margin:2px 0;overflow-wrap:anywhere}}
    header {{border-bottom:2px solid #126b67;padding-bottom:5px}}
    .columns {{columns:2;column-gap:7mm}} section p {{break-inside:avoid}}
    figure {{margin:5px 0}}
    img {{width:100%;height:20mm;object-fit:contain}} .detail {{font-size:6.5pt;color:#53636b}}
    .warning {{background:#edf4f2;padding:5px}} .credits {{font-size:6.5pt;margin-top:6px}}
    .review {{color:#8c4b13}} .identity {{font-size:6pt}}
    </style><header>{stamp}<h1>{escape(text("title", language))}</h1>
    <p class="identity">Run: {facts.run_id}<br>
    Evidence: {facts.evidence_id}</p>{review}</header>
    <div class="warning">{escape(text("warning", language))}</div>
    <figure>{map_html}
    <div class="detail">{escape(text("legend", language))}</div>
    <figcaption class="detail">{escape(text("map", language))}</figcaption>
    </figure>
    <div class="columns">{"".join(blocks)}</div>
    <p class="detail">{escape(text("appendix", language))}</p>
    <div class="credits">{credits}</div>{appendix}</html>'''


def pdf_bytes(facts: ReportFacts, language: Language, font: Path) -> bytes:
    from weasyprint import HTML
    from weasyprint.urls import URLFetcher

    fetch = URLFetcher(allowed_protocols={"data"}, allow_redirects=False, fail_on_errors=True)

    document = HTML(string=report_html(facts, language, font), url_fetcher=fetch).render()
    if len(document.pages) != 1:
        raise ReportLayoutError(
            f"Report needs {len(document.pages)} pages; use complete export or reduce scope"
        )
    # WeasyPrint 70 can keep an oversized unbreakable box on one page.
    # Inspect the pinned renderer's box tree so page count cannot hide clipping.
    page = document.pages[0]
    for box in page._page_box.descendants():
        if box.element_tag is not None and (
            box.position_y < 0
            or box.position_y + box.height > page.height
            or box.position_x < 0
            or box.position_x + box.width > page.width
        ):
            raise ReportLayoutError("Report content exceeds page bounds")
    result = document.write_pdf()
    if result is None:
        raise ReportLayoutError("PDF was not generated")
    return result
