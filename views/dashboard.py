"""Evidence map page of the combined app (streamlit run app.py)."""

import json
from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from himaroad.common import data_root
from himaroad.export import map_figure
from himaroad.pipeline import run


# Saved runs: dashboard/CLI outputs, drill outputs (two levels deeper) and
# extracted release bundles (docs/BUNDLES.md).
RUN_PATTERNS = (
    "outputs/himaroad/*/report.json",
    "outputs/himaroad/*/*/*/report.json",
    "bundles/*/report.json",
)
COMPARED = {
    "potentially_road_cut": "Potentially road-cut settlements",
    "access_uncertain": "Uncertain access",
    "potentially_exposed_buildings": "Potentially exposed mapped buildings",
    "suspected_affected_links": "Suspected affected links",
    "valid_change_observation_percent": "Valid change observations (%)",
}


@st.cache_data(max_entries=4)
def _routes(path, mtime_ns):
    """Route features, parsed once per file version (audit L6)."""
    return json.loads(Path(path).read_text())["features"]


@st.cache_resource(max_entries=2)
def _base_figure(run_dir, mtime_ns):
    """Layer figure, built once per run version (3 s and 30 MB on a real run).
    Callers add traces to a copy, never to this object."""
    return map_figure(run_dir, compact=True)


def _scenario_table(settlements, routes, scenarios):
    """Per-scenario counts read from the persisted settlements and routes."""
    rows = []
    for name in ["baseline", *scenarios]:
        status = {}
        for feature in routes:
            p = feature["properties"]
            if (p["scenario"], p["target"]) == (name, "town_hospital"):
                status[(p["mode"], p["status"])] = status.get((p["mode"], p["status"]), 0) + 1
        rows.append({
            "scenario": name,
            "settlements potentially cut": None if name == "baseline" else sum(
                int(name) in (s.get("cut_in_scenarios") or []) for s in settlements
            ),
            "vehicle route to town/hospital: found": status.get(("vehicle", "found"), 0),
            "vehicle: none": status.get(("vehicle", "none"), 0),
            "vehicle: unassessable": status.get(("vehicle", "unassessable"), 0),
            "foot route: found": status.get(("pedestrian", "found"), 0),
            "foot: none": status.get(("pedestrian", "none"), 0),
        })
    return rows


st.markdown('<div class="kicker">Mapping engine</div>', unsafe_allow_html=True)
st.title("Evidence map")
with st.sidebar:
    st.header("Run configuration")
    with st.form("analysis"):
        bbox_text = st.text_input(
            "WGS84 bounding box: west,south,east,north", "85.30,28.12,85.42,28.30"
        )
        polygon_upload = st.file_uploader(
            "Optional AOI polygon GeoJSON (overrides bounding box)",
            type=["geojson", "json"],
        )
        event = st.date_input("Event date", date(2026, 8, 26))
        cutoff = st.text_input("Optional analysis cutoff (ISO UTC timestamp)")
        provider = st.selectbox(
            "Imagery provider",
            ["cdse", "local", "gee"],
            format_func=lambda name: {
                "cdse": "CDSE (default: Sentinel Hub gamma0, Copernicus DEM)",
                "local": "Local manifest",
                "gee": "Earth Engine (flagged fallback, not for judging)",
            }[name],
        )
        detector = st.selectbox("Detector", ["baseline", "unet"])
        if detector == "unet":
            st.caption("Experimental v2 model: recall and IoU were 0 on the three reviewed Himalayan valley windows. See MODEL_CARD.md.")
        with st.expander("Advanced Settings"):
            manifest = st.text_input("Local input manifest path")
            project = st.text_input(
                "Earth Engine project (or use EE_PROJECT environment variable)"
            )
            model_config = st.text_input(
                "UNet inference configuration path", "configs/model_unet.yaml"
            )
            output = st.text_input(
                "Output directory", str(data_root() / "outputs/himaroad/dashboard")
            )
        submitted = st.form_submit_button("Run analysis")
    st.page_link("ui/app.py", label="Ask the copilot about a saved run (EN/NE)")
with st.sidebar:
    # I-A-02-001: availability before a run (no imagery download).
    if st.button("Check satellite availability (no download)"):
        try:
            from himaroad.cdse import precheck
            from himaroad.common import aoi_geometry, read_config

            config = read_config(Path(__file__).resolve().parents[1] / "src/himaroad/default.yaml")
            check = precheck(
                aoi_geometry([float(v) for v in bbox_text.split(",")]),
                event.isoformat(),
                cutoff or None,
                config,
            )
            pair = check.get("selected_pair") or {}
            post = pair.get("post") or {}
            st.write(
                f"Sentinel-1 post scene: {post.get('timestamp', 'none')} "
                f"({post.get('hours_after_event_date_boundary', '–')} h after the event-date "
                f"boundary), pre-stack {len(pair.get('pre') or [])} scenes"
            )
            st.json({t["track"]: len(t["post_candidates"]) for t in check.get("tracks", [])})
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
if submitted:
    try:
        polygon = None
        if polygon_upload is not None:
            upload_dir = Path(output)
            upload_dir.mkdir(parents=True, exist_ok=True)
            polygon = upload_dir / "input_aoi.geojson"
            polygon.write_bytes(polygon_upload.getvalue())
        bbox = (
            [float(v.strip()) for v in bbox_text.split(",")]
            if polygon is None
            else None
        )
        with st.spinner("Processing observations and mapped access…"):
            result = run(
                provider=provider,
                manifest=manifest or None,
                bbox=bbox,
                polygon=polygon,
                event_date=event.isoformat(),
                cutoff=cutoff or None,
                detector=detector,
                output=output,
                project=project or None,
                model_config=model_config or None,
            )
        st.session_state["run_directory"] = str(result)
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(str(exc))
report_paths = [p for pattern in RUN_PATTERNS for p in data_root().glob(pattern)]
selected = st.session_state.get("run_directory")
if selected:
    selected_report = Path(selected) / "report.json"
    if selected_report.exists() and selected_report not in report_paths:
        report_paths.append(selected_report)
reports = {}
for path in sorted(
    report_paths, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True
):
    try:
        reports[str(path.parent.resolve())] = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        st.sidebar.warning(f"Cannot read {path.parent.name}: {exc}")
if reports:
    # ?run=<directory name> opens that saved run (direct links).
    wanted = st.query_params.get("run")
    by_name = {Path(path).name: path for path in reports}
    if wanted in by_name and "run_from_link" not in st.session_state:
        st.session_state["run_directory"] = by_name[wanted]
        st.session_state["run_from_link"] = True
    elif st.session_state.get("run_directory") not in reports:
        st.session_state["run_directory"] = next(
            (
                path
                for path, report in reports.items()
                if report.get("status") == "complete"
            ),
            next(iter(reports)),
        )
    existing = st.selectbox(
        "Saved analysis",
        list(reports),
        key="run_directory",
        format_func=lambda path: (
            f"{Path(path).name} · {reports[path].get('status', 'unknown')}"
            f" · {reports[path].get('detector', 'baseline')}"
            + (" · SYNTHETIC" if reports[path].get("synthetic") else "")
        ),
    )
    st.query_params["run"] = Path(existing).name
else:
    existing = None
    st.info(
        "No saved analyses yet. Configure an analysis in the sidebar to get started."
    )
if existing:
    run_dir = Path(existing)
    report_path = run_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        st.subheader(
            f"{run_dir.name}: event date {report.get('event_date')}, "
            f"{report.get('detector', 'baseline')} detector"
        )
        manifest_path = run_dir / "manifest.json"
        try:
            saved_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        except (OSError, json.JSONDecodeError) as exc:
            saved_manifest = {}
            st.warning(f"Run provenance is unavailable: {exc}")
        st.caption(
            f"Area (WGS84 west, south, east, north): {report.get('bbox')}. "
            f"Sentinel-1 post-event scene: {(saved_manifest.get('post') or {}).get('timestamp', 'unavailable')}; "
            f"latest observation of any sensor: {report.get('observation_timestamp')} (UTC)."
        )
        if report.get("synthetic"):
            st.warning("Synthetic test fixture — these are not real disaster results.")
        if report.get("status") != "complete":
            st.warning(
                "Incomplete analysis: "
                + str(
                    report.get("reason", "Insufficient observations or map evidence.")
                )
            )
        model = saved_manifest.get("model_metadata") or {}
        st.caption(
            "Counts are screening results from the saved run: potentially exposed or cut, never confirmed "
            "destroyed or closed. Operational status is unknown. Scenarios are hypothetical; satellite time "
            "is not event time."
        )
        labels = {
            "potentially_road_cut": "Potentially road-cut settlements",
            "access_uncertain": "Uncertain access",
            "trail_access_possible": "Mapped trail alternatives",
            "potentially_exposed_buildings": "Potentially exposed mapped buildings",
            "suspected_affected_links": "Suspected affected links",
            "valid_post_observation_percent": "Valid post-event observations (%)",
        }
        columns = st.columns(3)
        for i, (key, label) in enumerate(labels.items()):
            value = report.get("summary", {}).get(key, "Unavailable")
            columns[i % 3].metric(label, round(value, 1) if isinstance(value, float) else value)
        twins = {
            path: other
            for path, other in reports.items()
            if path != existing
            and other.get("status") == "complete"
            and other.get("bbox") == report.get("bbox")
            and other.get("event_date") == report.get("event_date")
            and other.get("detector", "baseline") != report.get("detector", "baseline")
        }
        if twins:
            with st.expander("Detector comparison: same area and event date", expanded=True):
                compared = {existing: report, **twins}
                st.dataframe(
                    [
                        {
                            "run": Path(path).name,
                            "detector": other.get("detector", "baseline"),
                            **{
                                label: round(v, 1) if isinstance(v := other.get("summary", {}).get(key), float) else v
                                for key, label in COMPARED.items()
                            },
                        }
                        for path, other in compared.items()
                    ],
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "The v2 UNet detected none of the human-labelled new water on the three reviewed Himalayan "
                    "windows (MODEL_CARD.md); equal counts here do not validate either detector."
                )
                for path, other in twins.items():
                    st.button(
                        f"Show the {other.get('detector', 'baseline')} run",
                        key=f"twin:{path}",
                        on_click=st.session_state.update,
                        kwargs={"run_directory": path},
                    )
        scenarios = report.get("scenarios", {})
        primary = report.get("primary_scenario", "2")
        scenario = st.radio(
            "Hypothetical scenario",
            list(scenarios) or [primary],
            index=(list(scenarios).index(primary) if primary in scenarios else 0),
            horizontal=True,
            help="All scenarios are hypothetical removal sets; mapped reachability is not proof of access.",
        )
        st.caption(scenarios.get(scenario, ""))
        figure = go.Figure(_base_figure(str(run_dir), report_path.stat().st_mtime_ns))
        settlements_path, routes_path = run_dir / "settlements.geojson", run_dir / "routes.geojson"
        legacy = str(report.get("schema_version") or "1").split(".")[0] < "2"
        routes = (
            _routes(str(routes_path), routes_path.stat().st_mtime_ns)
            if routes_path.exists() and not legacy
            else []
        )
        if legacy:
            # 1.x runs number scenarios differently and lack cut_in_scenarios (audit M1).
            st.info("Scenario counts unavailable: this run uses output schema 1.x, whose scenario numbers differ.")
        elif settlements_path.exists():
            rows = [f["properties"] for f in json.loads(settlements_path.read_text())["features"]]
            with st.expander("Scenario comparison (persisted settlements and routes)"):
                st.dataframe(_scenario_table(rows, routes, scenarios), width="stretch", hide_index=True)
                st.caption(
                    "Routes to the nearest town or hospital per settlement. 'none' means no mapped route "
                    "remains in that hypothetical scenario; 'unassessable' means map evidence is missing."
                )
            cut = [r for r in rows if int(scenario) in (r.get("cut_in_scenarios") or [])]
            st.metric(f"Settlements potentially cut in scenario {scenario}", len(cut))
            if cut:
                with st.expander(f"List of the {len(cut)} settlements"):
                    st.dataframe(
                        [
                            {k: r.get(k) for k in (
                                "id", "name", "kind", "isolation_confidence", "cut_in_scenarios",
                                "nearest_destination_id", "scenario2_detour_m",
                            )}
                            for r in cut
                        ],
                        width="stretch",
                    )
        xs, ys = [], []
        for feature in routes:
            p = feature["properties"]
            if (p["scenario"], p["mode"], p["target"], p["status"]) == (
                scenario, "vehicle", "town_hospital", "found"
            ) and feature.get("geometry"):
                for x, y in feature["geometry"]["coordinates"]:
                    xs.append(x)
                    ys.append(y)
                xs.append(None)
                ys.append(None)
        if xs:
            figure.add_trace(go.Scattergl(
                x=xs, y=ys, mode="lines", hoverinfo="skip", name=f"routes (scenario {scenario})",
                line={"color": "#111111", "width": 3, "dash": "dot"},
            ))
        st.subheader("Evidence map")
        flow = st.session_state.get(f"flow:{run_dir}")
        if flow:
            path = next(f for f in flow["features"] if f["properties"]["kind"] == "flow_path")
            coords = path["geometry"]["coordinates"]
            figure.add_trace(go.Scattergl(
                x=[c[0] for c in coords], y=[c[1] for c in coords], mode="lines",
                name="traced flow path (DEM-derived)", line={"color": "#d6008f", "width": 4},
            ))
        west, south, east, north = report.get("bbox") or [85.3, 28.12, 85.42, 28.3]
        pad = 0.1 * max(east - west, north - south)
        # Open on the AOI (the road network extends 5 km beyond it); keep the
        # user's zoom across reruns of the same run.
        figure.update_xaxes(range=[west - pad, east + pad])
        figure.update_yaxes(range=[south - pad, north + pad])
        figure.update_layout(uirevision=str(run_dir))
        event_data = st.plotly_chart(
            figure, width="stretch", on_select="rerun", selection_mode="points", key=f"map:{run_dir}"
        )
        st.caption(
            "Roads are coloured by evidence; hover labels are on observed-change segments only. "
            "Optical (Sentinel-2) layers start hidden: their scenes can be weeks apart, so change is "
            "seasonal-prone; click them in the legend. "
            "Every feature with all fields: map.html and the GeoJSON layers. No basemap: the "
            "background is the run's own post-event Sentinel-1 radar backscatter image."
        )
        picked = (event_data or {}).get("selection", {}).get("points") or []
        st.subheader("Flow path (DEM-derived, not an inundation forecast)")
        columns = st.columns(3)
        lon = columns[0].number_input(
            "Longitude", value=float(picked[0]["x"]) if picked else (west + east) / 2, format="%.5f"
        )
        lat = columns[1].number_input(
            "Latitude", value=float(picked[0]["y"]) if picked else (south + north) / 2, format="%.5f"
        )
        if columns[2].button("Trace flow path from this point"):
            from himaroad.flowpath import trace_for_run

            try:
                st.session_state[f"flow:{run_dir}"] = trace_for_run(run_dir, lon, lat)
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        st.caption("Click a map point or type coordinates, then trace. Uses the run's Copernicus DEM only.")
        if flow:
            props = path["properties"]
            st.write(
                f"{props['label']}: {props['length_m'] / 1000:.1f} km, stops at {props['stop_reason']}; "
                f"settlements within {props['corridor_m']:g} m, downstream order:"
            )
            st.dataframe(props["settlements"], width="stretch")
        limits = [*report.get("limitations", []), *model.get("limitations", [])]
        with st.expander(f"Limitations of this run: {len(limits)} items (read with every number)"):
            for text in limits:
                st.markdown(f"- {text}")
        with st.expander("Run provenance and model limits"):
            post = saved_manifest.get("post") or {}
            osm = saved_manifest.get("osm") or {}
            st.json({
                "producer_commit": report.get("producer_commit", saved_manifest.get("producer_commit")),
                "schema_version": report.get("schema_version", saved_manifest.get("schema_version")),
                "detector": report.get("detector"),
                "event_date": report.get("event_date"),
                "sentinel1_pre_scene_ids": [s.get("scene_id") for s in saved_manifest.get("pre") or []],
                "sentinel1_post_scene_id": post.get("scene_id"),
                "sentinel1_post_time_utc": post.get("timestamp"),
                "sentinel1_orbit": {k: post.get(k) for k in ("relative_orbit", "pass")},
                "sentinel2_post_scene_id": (saved_manifest.get("optical_post") or {}).get("scene_id"),
                "latest_observation_utc": report.get("observation_timestamp"),
                "observation_age_hours": report.get("observation_age_hours"),
                "osm_snapshot": osm.get("snapshot"),
                "osm_cutoff_proof_passed": (osm.get("cutoff_proof") or {}).get("passed"),
                "model_version": model.get("model_version"),
                "checkpoint_sha256": model.get("checkpoint_sha256"),
                "preprocessing_id": model.get("preprocessing_id"),
                "operating_point": {
                    "threshold": model.get("operating_threshold", model.get("threshold")),
                    "temperature": model.get("temperature"),
                    "calibration_split": model.get("calibration_split"),
                },
                "thresholds": report.get("thresholds"),
                "model_limits": model.get("limitations", []),
                "run_limits": report.get("limitations", []),
            })
            if not model:
                st.caption("No trained-model provenance is recorded for this run.")
            st.json(
                {
                    key: report.get(key)
                    for key in (
                        "source_dataset",
                        "scenarios",
                        "evidence_categories",
                    )
                }
            )
        for filename in ("map.html", "report.pdf", "report.json", "manifest.json"):
            path = run_dir / filename
            if path.exists():
                # Read on click only: a real map.html is tens of MB.
                st.download_button(
                    "Download " + filename, path.read_bytes, file_name=filename, key=f"dl:{path}"
                )
    else:
        st.info("No persisted report.json in this directory.")
