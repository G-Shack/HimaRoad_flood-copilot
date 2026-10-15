"""Run: PYTHONPATH=. uv run --no-sync streamlit run ui/app.py."""

import base64
import os
from pathlib import Path
from urllib.parse import urlsplit

import requests
import streamlit as st

from backend.app.copilot.offline import NATIVE_REVIEW, Language, offline_menu
from backend.app.reports.pdf import font_bytes
from backend.app.reports.rendering import render_report
from backend.app.schemas import ReportFacts
from ui.map import claim_map

if os.environ.get("REPORT_FONT"):
    encoded_font = base64.b64encode(font_bytes(Path(os.environ["REPORT_FONT"]))).decode()
    st.markdown(
        "<style>@font-face{font-family:CopilotNoto;src:url(data:font/ttf;base64,"
        + encoded_font
        + ")} .stApp,.stText,.stAlert{font-family:CopilotNoto,sans-serif}</style>",
        unsafe_allow_html=True,
    )
st.markdown('<div class="kicker">Situation-report copilot</div>', unsafe_allow_html=True)
st.title("Ask the copilot")
st.caption("Every number comes from the saved maps. Independent inspection is required.")
st.info(NATIVE_REVIEW)
base = os.environ.get("COPILOT_API_URL", "http://127.0.0.1:8000").rstrip("/")
url = urlsplit(base)
if (
    url.scheme not in {"http", "https"}
    or not url.hostname
    or url.username
    or url.password
    or url.query
    or url.fragment
):
    st.error("Configure a credential-free companion API URL.")
    st.stop()


def call(method: str, path: str, *, accepted: tuple[int, ...] = (200, 204), **kwargs):
    headers = {}
    if st.session_state.get("token"):
        headers["Authorization"] = "Bearer " + st.session_state.token
    try:
        response = requests.request(
            method, base + path, headers=headers, timeout=(10, 600), allow_redirects=False, **kwargs
        )
    except requests.RequestException:
        st.error("Companion API unavailable. Check the local service.")
        st.stop()
    if response.status_code not in accepted:
        # Fixed API error codes only; no transport diagnostics, paths or tokens displayed.
        st.error(f"Request unavailable (HTTP {response.status_code}).")
        st.stop()
    return response


listing = call("GET", "/api/runs").json()
if listing["truncated"]:
    st.warning("Run picker is incomplete: narrow the configured run root.")
if not listing["paths"]:
    st.info("No saved runs available under the configured HimaRoad root.")
    st.stop()
with st.sidebar:
    language_choice = st.selectbox("Language / भाषा", ["en", "ne"])
    language: Language = "ne" if language_choice == "ne" else "en"
    if st.session_state.get("answer_language") != language:
        st.session_state.pop("answer", None)
        st.session_state.pop("download", None)
        st.session_state.answer_language = language
    with st.form("load"):
        default = "bundles/trishuli-2026-v3.0-r2"
        path = st.selectbox(
            "Saved HimaRoad run", listing["paths"],
            index=listing["paths"].index(default) if default in listing["paths"] else 0,
        )
        submitted = st.form_submit_button("Load run")
    if submitted:
        st.info("Loading saved evidence. Large runs must be prepared through the launcher first.")
        with st.spinner("Verifying the saved snapshot; large runs can take several minutes."):
            loaded = call("POST", "/api/runs/load", json={"path": path}).json()
        if st.session_state.get("token"):
            call("DELETE", "/api/session", accepted=(204, 401))
        st.session_state.token = loaded["session_token"]
        st.session_state.run = loaded["run"]
        for key in ("answer", "download", "selected_claim", "views"):
            st.session_state.pop(key, None)
    st.caption("Session stays on this server and expires after one hour. Reload a run to renew.")
if "token" not in st.session_state:
    st.info("Select and load a saved run.")
    st.stop()
run = st.session_state.run
if run["synthetic"]:
    st.warning("SYNTHETIC software demonstration — no real-event result")
if run["status"] == "incomplete":
    st.warning(
        "INCOMPLETE: impact counts and routes are unavailable; missing observations are unknown."
    )
selected = st.session_state.get("selected_claim")
views = st.session_state.setdefault("views", {})
view_key = selected or "default"
if view_key not in views:
    with st.spinner("Building a partial evidence view; large snapshots can take several minutes."):
        view = call("GET", "/api/view", params={"selected": selected} if selected else {}).json()
    if len(views) >= 2:
        views.clear()
    views[view_key] = view
else:
    view = views[view_key]
    # Cached display never extends authorization or hides SQLite corruption.
    anchor = next(
        (c for c in view["facts"]["claims"] if c["kind"] == "status"), view["facts"]["claims"][0]
    )
    card = call("GET", "/api/claims/" + anchor["id"]).json()
    if card["run_id"] != view["facts"]["run_id"]:
        st.error("Cached evidence no longer belongs to this session.")
        st.stop()
facts = ReportFacts.model_validate(view["facts"])
if view["partial"]:
    st.info(
        "Partial evidence view and map. Full JSON/CSV exports retain all evidence. "
        "Use the settlement selector and citations to inspect individual records."
    )
st.caption(f"Evidence snapshot: {facts.evidence_id}")
st.caption(
    f"Output schema: {run['schema_version'] or 'legacy'}; "
    f"producer: {run['producing_commit'] or 'unknown'}"
)
rendered = render_report(facts, language)
features = [c for c in facts.claims if c.kind == "feature"]
subjects = ["No selected asset", *sorted({c.subject for c in features} | set(view["settlements"]))]
subject = st.selectbox("Selected mapped asset (for asset/route questions)", subjects)
selected_subject = None if subject == subjects[0] else subject
selected = st.session_state.get("selected_claim")
if facts.availability == "unavailable":
    st.info("Map/impact evidence unavailable for this snapshot.")
else:
    event = st.plotly_chart(
        claim_map(facts, selected),
        key="claim-map-" + facts.evidence_id,
        on_select="rerun",
        selection_mode="points",
    )
    points = event.get("selection", {}).get("points", [])
    if points:
        picked = points[0].get("customdata", [None])[0]
        if picked in {c.id for c in facts.claims} and picked != selected:
            st.session_state.selected_claim = picked
            st.rerun()
    st.caption(
        "Satellite observations and hypothetical routes have distinct styles. "
        "Routes are mapped reachability only."
    )
summary_ids = {c.id for c in facts.claims if c.kind == "summary" and c.unit == "count"}
with st.expander("Summary evidence / सारांश प्रमाण"):
    for section in rendered.sections:
        for row in section.rows:
            if row.claim_id in summary_ids and st.button(row.text, key="summary-" + row.claim_id):
                st.session_state.selected_claim = row.claim_id
                st.rerun()
if selected:
    card = call("GET", "/api/claims/" + selected).json()
    with st.expander("Cited evidence / provenance", expanded=True):
        st.json(card, expanded=2)

if selected_subject and selected_subject.startswith("settlements:"):
    with st.form("route"):
        scenario = st.selectbox(
            "Hypothetical route scenario",
            ["baseline", "1", "2", "3", *(["4"] if run["schema_version"] == "3.0" else [])],
        )
        route_mode = st.selectbox("Mapped route mode", ["vehicle", "pedestrian"])
        show_route = st.form_submit_button("Show modelled route")
    if show_route:
        st.session_state.answer = call(
            "POST",
            "/api/route",
            json={
                "selected_subject": selected_subject,
                "scenario": scenario,
                "mode": route_mode,
                "language": language,
            },
        ).json()

menu = offline_menu(language)
with st.form("question"):
    mode = st.radio("Question mode", ["Predefined evidence question", "Free text"])
    question_id = st.selectbox(
        "Evidence question",
        [q.id for q in menu.questions],
        format_func=lambda key: next(q.label for q in menu.questions if q.id == key),
    )
    question = st.text_input("Question", max_chars=8000)
    ask = st.form_submit_button("Ask / सोध्नुहोस्")
if ask:
    payload = {"language": language, "selected_subject": selected_subject}
    payload["question_id" if mode == "Predefined evidence question" else "question"] = (
        question_id if mode == "Predefined evidence question" else question
    )
    st.session_state.answer = call("POST", "/api/ask", json=payload).json()
if "answer" in st.session_state:
    result = st.session_state.answer
    answer = result["answer"]
    if answer.get("outcome") == "refused":
        st.info(answer["text"])
    else:
        st.caption("Citations below open the original evidence and highlight mapped geometry.")
        with st.expander("Cited answer and evidence links", expanded=len(answer["segments"]) <= 12):
            for segment in answer["segments"]:
                st.text(segment)
            for i, identifier in enumerate(answer["claim_ids"], start=1):
                if st.button(f"Citation {i}: {identifier[:12]}", key="cite-" + identifier):
                    st.session_state.selected_claim = identifier
                    st.rerun()
    if answer.get("reason") == "llm_unavailable":
        st.info(menu.notice)
        st.caption("Use the predefined evidence questions above.")
# The identical ordered report projection supplies scenarios and limitations.
for section in rendered.sections:
    if section.id not in {"access", "limitations"}:
        continue
    with st.expander(section.title, expanded=section.id == "limitations"):
        for row in section.rows:
            st.text(row.text)
flow = [c for c in facts.claims if c.kind == "flow_path"]
if not flow or all(c.value.get("availability") == "unavailable" for c in flow):
    st.info("DEM flow-path evidence unavailable in this run.")
else:
    st.info("Persisted DEM flow path — not an inundation forecast.")
with st.form("report"):
    format = st.selectbox(
        "Report format",
        ["json", "csv", "pdf"],
        format_func=lambda value: value if value == "pdf" else value + " (.gz)",
    )
    prepare = st.form_submit_button("Prepare report")
if prepare:
    compression = "none" if format == "pdf" else "gzip"
    response = call(
        "GET", f"/api/report?language={language}&format={format}&compression={compression}"
    )
    st.session_state.download = (
        response.content,
        language,
        format if format == "pdf" else format + ".gz",
        response.headers["content-type"],
    )
if "download" in st.session_state:
    content, report_language, report_format, mime = st.session_state.download
    st.download_button(
        "Download prepared report", content, f"report-{report_language}.{report_format}", mime=mime
    )
