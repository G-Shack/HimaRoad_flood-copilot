// HimaRoad Flood Copilot web app: map from the saved run, copilot from the same evidence.
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const state = { run: null, summary: null, settlements: [], scenario: "2", lang: "en", token: null,
  menu: {}, layers: {}, pick: null, selected: null, loading: null };
const CAT = { observed_change: "#E3872D", likely_impact: "#B5452B", possible_impact: "#E8B04A",
  unobservable: "#A9D6E0", no_detected_change: "#FFFFFF" };
const CAT_LABEL = { observed_change: "Observed change", likely_impact: "Likely impact",
  possible_impact: "Possible impact", unobservable: "Unobservable", no_detected_change: "No detected change" };
const PLACE = { "trishuli-2026-v3.0-r2": "Trishuli valley · Rasuwa, Nepal",
  "trishuli-2026-v3.0-r2-unet": "Trishuli valley · Rasuwa, Nepal · UNet detector",
  "melamchi-2021-v3.0-r2": "Melamchi valley · Sindhupalchok, Nepal" };
const SCN = { "1": "S1: only links with an independently sourced confirmed closure are removed. This run has none.",
  "2": "S2, primary: links with observed change in valid radar or optical imagery are removed. Hypothetical.",
  "3": "S3: only radar-optical corroborated (likely impact) links are removed. Hypothetical.",
  "4": "S4, worst case: S2 plus possible-impact, unobservable and unknown links are removed. Hypothetical." };
const KEYWORDS = [["flow", /flow|downstream|river path|बहाव/i], ["repair", /repair|priorit|मर्मत/i],
  ["integrity", /integrity|hash|tamper|अखण्ड/i], ["sources", /source|limit|reliab|sure|trust|स्रोत|सीमा/i],
  ["buildings", /building|house|home|भवन|घर/i], ["roads", /road|link|bridge|सडक/i],
  ["isolation", /cut|isolat|disconnect|settlement|village|access|reach|बस्ती|सम्पर्क/i],
  ["summary", /summar|overview|what|show|सारांश/i]];

const fmt = (n, d = 0) => (n === null || n === undefined ? "–" : Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function toast(text) { const t = $("#toast"); t.textContent = text; t.classList.remove("hidden"); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.add("hidden"), 3500); }
async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (opts.body && !(opts.body instanceof Blob)) headers["Content-Type"] = "application/json";
  if (opts.auth && state.token) headers.Authorization = `Bearer ${state.token}`;
  const res = await fetch(path, { method: opts.method || "GET", headers, body: opts.body ? JSON.stringify(opts.body) : undefined });
  if (!res.ok) { let d = res.statusText; try { d = (await res.json()).detail || d; } catch (e) { /* no JSON body */ } throw new Error(`${res.status} ${d}`); }
  return opts.raw ? res : res.json();
}

// ---------- tabs ----------
function show(tab) {
  $$(".nav").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  ["overview", "evidence", "report", "about"].forEach((id) => $("#" + id).classList.toggle("hidden", id !== tab));
  if (tab === "overview") setTimeout(() => map.invalidateSize(), 50);
  history.replaceState(null, "", `?run=${state.run || ""}#${tab}`);
}
$$(".nav").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
$$("[data-goto]").forEach((b) => b.addEventListener("click", () => show(b.dataset.goto)));

// ---------- map ----------
const map = L.map("map", { preferCanvas: true, zoomSnap: 0.25, zoomControl: false });
L.control.zoom({ position: "topright" }).addTo(map);
map.attributionControl.setPrefix(false);
map.attributionControl.addAttribution("Contains modified Copernicus Sentinel data 2026 · © OpenStreetMap contributors");
const canvas = undefined; // all vectors share the map's one default canvas (preferCanvas), so marker clicks are not caught by a second canvas

function settlementStatus(p, n) {
  if (p.baseline_accessible === false) return { t: "No mapped access before event", k: "unk" };
  if (p[`scenario${n}_accessible`] === false) return { t: "Potentially cut", k: "cut" };
  if (n !== "4" && p.depends_only_on_unknown) return { t: "Uncertain (unknown links)", k: "unk" };
  return { t: "Reachable on mapped roads", k: "ok" };
}
function settlementName(p) {
  if (p.name) return p.name;
  const id = String(p.id || "");
  return p.kind === "building_cluster" ? `Building cluster ${id.split("/")[1]}` : id;
}

function nameOf(id) { const f = state.settlements.find((s) => s.properties.id === id); return f ? settlementName(f.properties) : null; }
function clearLayers() { Object.values(state.layers).forEach((l) => l && map.removeLayer(l)); state.layers = {}; }
function addLayer(name, layer) { state.layers[name] = layer; const box = $(`#legend input[data-layer="${name}"]`); if (!box || box.checked) layer.addTo(map); }

async function loadRadar(run) {
  const [pre, post] = await Promise.all([api(`/api/map/${run}/radar/pre.json`), api(`/api/map/${run}/radar/post.json`)]);
  addLayer("radar_pre", L.imageOverlay(`/api/map/${run}/radar/pre.png`, pre.bounds, { opacity: 1, interactive: false }));
  state.layers.radar_post = L.imageOverlay(`/api/map/${run}/radar/post.png`, post.bounds, { opacity: 1, interactive: false }).addTo(map);
  state.layers.radar_post.on("load", applyCompare);
}
function applyCompare() {
  const el = state.layers.radar_post && state.layers.radar_post.getElement();
  if (el) el.style.clipPath = `inset(0 0 0 ${100 - Number($("#compare").value)}%)`;
}
$("#compare").addEventListener("input", applyCompare);
map.on("zoomend moveend", applyCompare);

async function loadVectors(run) {
  const style = { disturbance: { color: "#E3872D", fillColor: "#E3872D", weight: 1, fillOpacity: 0.55 },
    possible_debris: { color: "#B5452B", fillColor: "#B5452B", weight: 1, fillOpacity: 0.6 },
    flood_extent: { color: "#2E9BD6", fillColor: "#2E9BD6", weight: 1, fillOpacity: 0.55 },
    unobservable: { color: "#6B7A82", fillColor: "#8A9AA2", weight: 0, fillOpacity: 0.35 } };
  const hazards = await Promise.all(Object.keys(style).map((n) => api(`/api/map/${run}/layer/${n}`)));
  Object.keys(style).forEach((n, i) => addLayer(n, L.geoJSON(hazards[i], { renderer: canvas, style: style[n], interactive: false })));
  const roads = await api(`/api/map/${run}/roads`);
  roads.features.sort((a, b) => (a.properties.c === "observed_change") - (b.properties.c === "observed_change"));
  addLayer("roads", L.geoJSON(roads, { renderer: canvas,
    style: (f) => ({ color: CAT[f.properties.c] || "#D2DBDF", weight: f.properties.c === "observed_change" || f.properties.c === "likely_impact" ? 4 : (["trunk", "primary", "secondary"].includes(f.properties.highway) ? 2.4 : 1.4), opacity: 0.95, dashArray: f.properties.c === "unobservable" ? "4 3" : null }),
    onEachFeature: (f, l) => l.bindTooltip(`<b>${esc(f.properties.name || f.properties.highway)}</b><br>${CAT_LABEL[f.properties.c] || f.properties.c}<br>Operational status: unknown`, { sticky: true }) }));
  const bld = await api(`/api/map/${run}/buildings`);
  addLayer("buildings", L.geoJSON(bld, { renderer: canvas, style: { color: "#B5452B", weight: 1, fillColor: "#B5452B", fillOpacity: 0.8 },
    onEachFeature: (f, l) => l.bindTooltip(`Potentially exposed building<br>${CAT_LABEL[f.properties.c] || ""}`) }));
  const flow = await api(`/api/map/${run}/layer/flowpaths`);
  addLayer("flowpaths", L.geoJSON(flow, { renderer: canvas, style: (f) => f.geometry.type.includes("Polygon") ? { color: "#7B3FA0", weight: 0, fillOpacity: 0.08 } : { color: "#7B3FA0", weight: 3, dashArray: "6 4" },
    onEachFeature: (f, l) => f.properties.source_name && l.bindTooltip(`Flow path from ${esc(f.properties.source_name)}<br>${fmt((f.properties.length_m || 0) / 1000, 1)} km, DEM trace`) }));
  const dest = await api(`/api/map/${run}/layer/destinations`);
  addLayer("destinations", L.geoJSON(dest, { pointToLayer: (f, ll) => L.marker(ll, { icon: L.divIcon({ className: "", html: '<div style="width:16px;height:16px;background:#0E2A3B;border:2px solid #fff;border-radius:3px;color:#fff;font:700 11px/12px sans-serif;text-align:center">+</div>', iconSize: [16, 16] }) }).bindTooltip(`Mapped ${esc(f.properties.destination_tier || f.properties.amenity || "destination")}`) }));
}

function drawSettlements() {
  if (state.layers.settlements) map.removeLayer(state.layers.settlements);
  const color = { cut: "#E3872D", ok: "#1F7A8C", unk: "#8A9AA2" };
  const group = L.featureGroup();
  state.settlements.forEach((f) => {
    const p = f.properties; const s = settlementStatus(p, state.scenario);
    const [lon, lat] = f.geometry.coordinates;
    const m = L.circleMarker([lat, lon], { radius: s.k === "cut" ? 8 : 6, color: "#fff", weight: 2, fillColor: color[s.k], fillOpacity: 1, renderer: canvas });
    m.on("click", (e) => { L.DomEvent.stopPropagation(e); selectSettlement(p.id); });
    m.bindTooltip(`<b>${esc(settlementName(p))}</b><br>${s.t}`);
    f._marker = m; group.addLayer(m);
  });
  state.layers.settlements = group;
  if ($('#legend input[data-layer="settlements"]').checked) group.addTo(map);
}

function renderScenario() {
  const n = state.scenario; const rows = state.settlements.map((f) => ({ f, s: settlementStatus(f.properties, n) }));
  const order = { cut: 0, unk: 1, ok: 2 };
  rows.sort((a, b) => order[a.s.k] - order[b.s.k] || (b.f.properties.building_count || 0) - (a.f.properties.building_count || 0));
  const cut = rows.filter((r) => r.s.k === "cut").length;
  $("#mCut").innerHTML = `${cut}<em> of ${rows.length}</em>`;
  $("#mCutNote").textContent = `Scenario S${n}${n === state.summary.primary_scenario ? " (primary)" : ""}, hypothetical`;
  $("#scenarioNote").textContent = SCN[n];
  $("#tableNote").textContent = `${cut} potentially cut in S${n} · click a row to find it on the map`;
  $("#rows").innerHTML = rows.map(({ f, s }) => { const p = f.properties; return `<tr data-id="${esc(p.id)}"><td>${esc(settlementName(p))}<br><span style="color:var(--muted);font-size:11px">${fmt(p.building_count)} mapped buildings</span></td><td><span class="pill ${s.k}">${s.t}</span></td><td>${p.pedestrian_accessible ? "Trail possible" : "None mapped"}</td><td>${CAT_LABEL[p.evidence_category] || "–"}</td><td>${fmt(p.potentially_exposed_buildings)}</td></tr>`; }).join("");
  $$("#rows tr").forEach((tr) => tr.addEventListener("click", () => selectSettlement(tr.dataset.id)));
  drawSettlements();
}
$("#scenario").addEventListener("change", (e) => { state.scenario = e.target.value; renderScenario(); if (state.selected) selectSettlement(state.selected, false); });

function selectSettlement(id, fly = true) {
  const f = state.settlements.find((x) => x.properties.id === id); if (!f) return;
  state.selected = id; const p = f.properties; const s = settlementStatus(p, state.scenario);
  const [lon, lat] = f.geometry.coordinates;
  if (fly) map.flyTo([lat, lon], Math.max(map.getZoom(), 13.5), { duration: 0.8 });
  $$("#rows tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.id === id));
  const row = $(`#rows tr[data-id="${CSS.escape(id)}"]`);
  if (row) { // scroll the table only, never the page (zoom-safe ratio)
    const box = row.closest(".tablebox"); const b = box.getBoundingClientRect(); const r = row.getBoundingClientRect();
    box.scrollTop += (r.top - b.top - 40) * (box.clientHeight / (b.height || 1));
  }
  const cutIn = (p.cut_in_scenarios || []).map((x) => "S" + x).join(", ") || "none";
  $("#mapTitle").textContent = settlementName(p);
  $("#mapDetail").innerHTML = `<span class="pill ${s.k}">${s.t}</span> in S${state.scenario}<br>Cut in: ${cutIn} · ${fmt(p.building_count)} buildings, ${fmt(p.potentially_exposed_buildings)} potentially exposed<br>Nearest mapped ${esc(p.nearest_destination_tier || "destination")}: ${p.nearest_destination_cost_m ? fmt(p.nearest_destination_cost_m / 1000, 1) + " km by road (before event)" : "not reachable"}<br>Trail: ${p.pedestrian_accessible ? "possible on mapped paths" : "none mapped"} · Operational status: unknown`;
}

map.on("click", (e) => {
  state.pick = e.latlng;
  if (state.layers.pick) map.removeLayer(state.layers.pick);
  state.layers.pick = L.circleMarker(e.latlng, { radius: 7, color: "#7B3FA0", weight: 3, fillColor: "#fff", fillOpacity: 1, interactive: false }).addTo(map);
  $("#flowPick").textContent = `Start: ${e.latlng.lat.toFixed(4)}° N, ${e.latlng.lng.toFixed(4)}° E`;
  $("#traceBtn").disabled = false;
});
$("#traceBtn").addEventListener("click", async () => {
  if (!state.pick) return; const btn = $("#traceBtn"); btn.disabled = true; btn.textContent = "Tracing…";
  try {
    const fc = await api(`/api/map/${state.run}/flowpath`, { method: "POST", body: { lon: state.pick.lng, lat: state.pick.lat } });
    if (state.layers.trace) map.removeLayer(state.layers.trace);
    state.layers.trace = L.geoJSON(fc, { interactive: false, style: (f) => f.geometry.type.includes("Polygon") ? { color: "#7B3FA0", weight: 0, fillOpacity: 0.12 } : { color: "#7B3FA0", weight: 5 } }).addTo(map);
    const line = (fc.features || []).find((f) => f.geometry && f.geometry.type.includes("LineString")) || {};
    const p = line.properties || {}; const along = p.settlements || [];
    $("#flowOut").innerHTML = `<p style="margin:10px 0 4px;color:var(--ink)"><b>${fmt((p.length_m || 0) / 1000, 1)} km</b> downstream, stops at ${esc(p.stop_reason || "–")}. ${along.length} settlements within ${fmt(p.corridor_m)} m, in downstream order:</p><ol>${along.slice(0, 40).map((x) => `<li>${esc(nameOf(x.id || x.settlement_id) || x.name || x.id)}${x.distance_along_m !== undefined ? ` · ${fmt(x.distance_along_m / 1000, 1)} km` : ""}</li>`).join("")}</ol>`;
    map.fitBounds(state.layers.trace.getBounds(), { padding: [30, 30] });
  } catch (err) { $("#flowOut").innerHTML = `<p style="color:#8A4A10">Trace refused: ${esc(err.message)}</p>`; }
  btn.disabled = false; btn.textContent = "Trace flow path";
});

$$("#legend input").forEach((box) => box.addEventListener("change", () => {
  const l = state.layers[box.dataset.layer]; if (!l) return; box.checked ? l.addTo(map) : map.removeLayer(l);
  if (box.dataset.layer === "roads" || box.dataset.layer === "buildings") state.layers.settlements && state.layers.settlements.bringToFront && state.layers.settlements.eachLayer((m) => m.bringToFront());
}));
function bboxBounds() { const b = state.summary.bbox; return [[b[1], b[0]], [b[3], b[2]]]; }
$("#resetView").addEventListener("click", () => { map.fitBounds(bboxBounds()); $("#compare").value = 100; applyCompare(); });

// ---------- run ----------
async function loadRun(run) {
  state.run = run; state.selected = null; state.token = null;
  clearLayers();
  const s = await api(`/api/map/${run}/summary`); state.summary = s;
  const sum = s.summary || {};
  $("#placeKicker").textContent = `${PLACE[run] || run} · event date ${s.event_date}`;
  $("#runMeta").innerHTML = `<b>${esc(PLACE[run] || run)}</b><br>Event date: ${esc(s.event_date)}<br>Detector: ${esc(s.detector)}<br>Status: ${esc(s.status)}${s.status !== "complete" ? ' <span class="pill unk">see limits</span>' : ""}`;
  $("#mapSub").textContent = `${s.detector === "unet" ? "UNet" : "Rule baseline"} detector · radar after event ${String(s.post_time_utc || "").slice(0, 16).replace("T", " ")} UTC`;
  $("#mBld").textContent = fmt(sum.potentially_exposed_buildings);
  $("#mLinks").textContent = fmt(sum.suspected_affected_links);
  $("#mCov").innerHTML = sum.valid_change_observation_percent !== undefined ? `${fmt(sum.valid_change_observation_percent, 1)}<em>%</em>` : "–";
  $("#scenario").value = s.primary_scenario || "2"; state.scenario = $("#scenario").value;
  map.fitBounds(bboxBounds());
  state.settlements = (await api(`/api/map/${run}/settlements`)).features;
  renderScenario();
  renderEvidence();
  loadCopilot(run);
  await loadRadar(run);
  await loadVectors(run);
  state.layers.settlements && state.layers.settlements.eachLayer((m) => m.bringToFront());
  history.replaceState(null, "", `?run=${run}${location.hash || ""}`);
}
$("#runSelect").addEventListener("change", (e) => loadRun(e.target.value));

// ---------- copilot ----------
function setStatus(text, kind) { const el = $("#cstatus"); el.className = "cstatus" + (kind ? " " + kind : ""); el.textContent = text; }
async function loadCopilot(run) {
  const mine = Symbol(); state.loading = mine;
  setStatus("Loading the run and checking every file hash…", "busy"); renderChips(true);
  try {
    const loaded = await api("/api/runs/load", { method: "POST", body: { path: `bundles/${run}` } });
    if (state.loading !== mine) return;
    state.token = loaded.session_token;
    setStatus(`Evidence verified · run ${run} · evidence ${loaded.evidence_id.slice(0, 10)}…`);
    renderChips(false);
  } catch (err) {
    if (state.loading !== mine) return;
    setStatus(`The copilot cannot load this run (${err.message}). The map stays available.`, "err");
  }
}
async function renderChips(disabled) {
  if (!state.menu[state.lang]) state.menu[state.lang] = await api(`/api/map/menu?language=${state.lang}`);
  $("#chips").innerHTML = state.menu[state.lang].questions.map((q) => `<button class="chip" data-q="${q.id}" ${disabled ? "disabled" : ""}>${esc(q.label)}</button>`).join("");
  $$("#chips .chip").forEach((c) => c.addEventListener("click", () => ask(c.dataset.q, c.textContent)));
}
function addMsg(html, cls = "") { const d = document.createElement("div"); d.className = "msg " + cls; d.innerHTML = html; $("#chat").appendChild(d); $("#chat").scrollTop = $("#chat").scrollHeight; return d; }
async function ask(qid, label) {
  if (!state.token) { toast("The copilot is still loading the run."); return; }
  addMsg(esc(label), "user");
  const wait = addMsg('<span style="color:var(--muted)">Reading the cited evidence…</span>', state.lang);
  try {
    const r = await api("/api/ask", { method: "POST", auth: true, body: { question_id: qid, language: state.lang } });
    const a = r.answer;
    if (a.outcome === "refused") { wait.innerHTML = `<p>${esc(a.text)}</p>`; return; }
    // Group repeated record lines; show the first records and fold the rest. The text is unchanged.
    let refs = 0; const groups = new Map();
    for (const s of a.segments || []) {
      const plain = s.replace(/\s*\[[0-9a-f]{64}(?:,\s*[0-9a-f]{64})*\]/g, "").trim();
      const g = groups.get(plain) || { html: esc(s).replace(/\s*\[[0-9a-f]{64}(?:,\s*[0-9a-f]{64})*\]/g, () => `<sup class="ref">${++refs}</sup>`), n: 0 };
      g.n += 1; groups.set(plain, g);
    }
    const items = [...groups.values()].map((g) => `<p>${g.html}${g.n > 1 ? ` <span class="times">×${g.n}</span>` : ""}</p>`);
    const parts = items.slice(0, 7).join("") + (items.length > 7 ? `<details><summary>${state.lang === "ne" ? `थप ${items.length - 7} अभिलेख हेर्नुहोस्` : `Show ${items.length - 7} more records`}</summary>${items.slice(7).join("")}</details>` : "");
    const lim = (a.limitations || []).filter((l) => !/^[0-9a-f]{64}$/.test(l)).slice(0, 2).map((l) => `<div class="lim">${esc(l)}</div>`).join("");
    wait.innerHTML = `${parts}<div class="cite"><span>${a.claim_ids.length} cited evidence claims</span><span>${a.feature_refs.length} map features</span><span>check: ${esc(a.validation_result)}</span><span>${state.lang === "ne" ? "सुरक्षित प्रमाणबाट" : "from saved evidence"}</span></div>${lim}`;
  } catch (err) { wait.innerHTML = `<p style="color:#8A4A10">The copilot refused the request: ${esc(err.message)}</p>`; }
  $("#chat").scrollTop = Math.max(0, wait.offsetTop - $("#chat").offsetTop - 70);
}
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault(); const text = $("#question").value.trim(); if (!text) return; $("#question").value = "";
  const hit = KEYWORDS.find(([, re]) => re.test(text));
  if (hit) { const q = state.menu[state.lang].questions.find((x) => x.id === hit[0]); addMsg(`<span style="color:var(--muted);font-size:11px">Matched to the predefined question: “${esc(q.label)}”</span>`, ""); ask(hit[0], text); }
  else { addMsg(esc(text), "user"); addMsg(`<p>${state.lang === "ne" ? "कृपया तलका प्रश्नमध्ये एउटा छान्नुहोस्।" : "Please choose one of the questions below."}</p>`, state.lang); }
});
$$(".lang button").forEach((b) => b.addEventListener("click", () => {
  state.lang = b.dataset.lang; $$(".lang button").forEach((x) => x.classList.toggle("on", x === b));
  if (typeof setSiteLang === "function" && siteLang !== state.lang) setSiteLang(state.lang);
  $("#question").placeholder = state.lang === "ne" ? "प्रश्न लेख्नुहोस्, जस्तै: कुन सडक प्रभावित छन्?" : "Type a question, for example: which roads are affected?";
  renderChips(!state.token);
}));

// ---------- report ----------
$$("[data-report]").forEach((b) => b.addEventListener("click", async () => {
  if (!state.token) { toast("The copilot is still loading the run."); return; }
  const lang = b.dataset.report; $$("[data-report]").forEach((x) => (x.disabled = true));
  $("#reportStatus").textContent = "Preparing the PDF from the cited evidence…"; const t0 = performance.now();
  try {
    const res = await api(`/api/report?language=${lang}&format=pdf`, { auth: true, raw: true });
    const url = URL.createObjectURL(await res.blob());
    $("#pdfframe").src = url + "#navpanes=0&view=FitH"; $("#pdfframe").classList.remove("hidden"); $("#pdfEmpty").classList.add("hidden");
    const dl = $("#pdfDownload"); dl.href = url; dl.download = `situation-report-${state.run}-${lang}.pdf`; dl.classList.remove("hidden");
    $("#reportStatus").textContent = `Ready in ${((performance.now() - t0) / 1000).toFixed(1)} s · ${lang === "ne" ? "Nepali" : "English"} · one page`;
  } catch (err) { $("#reportStatus").textContent = `The report was refused: ${err.message}`; }
  $$("[data-report]").forEach((x) => (x.disabled = false));
}));

// ---------- evidence page ----------
async function renderEvidence() {
  const s = state.summary; const sum = s.summary || {}; const g = s.sar_geometry || {}; const th = s.thresholds || {};
  const row = (num, sub, body) => `<div class="row"><div class="num">${num}<small>${sub}</small></div><div>${body}</div></div>`;
  const kv = (o) => `<div class="kv">${Object.entries(o).map(([k, v]) => `<div><b>${esc(v)}</b>${esc(k)}</div>`).join("")}</div>`;
  const offs = s.temporal_offsets || {};
  let cmp = "";
  if (state.run.startsWith("trishuli")) {
    try {
      const [a, b] = await Promise.all(["trishuli-2026-v3.0-r2", "trishuli-2026-v3.0-r2-unet"].map((r) => api(`/api/map/${r}/summary`)));
      const keys = [["potentially_road_cut", "Potentially road-cut settlements (S2)"], ["potentially_exposed_buildings", "Potentially exposed buildings"], ["suspected_affected_links", "Suspected affected links"], ["valid_change_observation_percent", "Valid change observation, %"]];
      cmp = row("05", "Detector comparison", `<p>The same area and date, with the rule baseline and with the UNet trained from scratch on Kuro Siwo.</p><table class="cmp"><thead><tr><th>Measure</th><th>Rule baseline</th><th>UNet</th></tr></thead><tbody>${keys.map(([k, l]) => `<tr><td>${l}</td><td>${fmt(a.summary[k], k.includes("percent") ? 2 : 0)}</td><td>${fmt(b.summary[k], k.includes("percent") ? 2 : 0)}</td></tr>`).join("")}</tbody></table>`);
    } catch (e) { cmp = ""; }
  }
  $("#evidenceBody").innerHTML = [
    row("01", "Satellite observations", `<p>Sentinel-1 before the event: ${(s.pre_scene_ids || []).map((x) => `<code>${esc(x)}</code>`).join(" ")}</p><p>Sentinel-1 after the event: <code>${esc(s.post_scene_id)}</code> at ${esc(s.post_time_utc)} (orbit ${esc((s.orbit || {}).relative_orbit)}, ${esc((s.orbit || {}).pass)}). That is ${fmt(offs.post_hours_after_event_date_boundary, 1)} h after the start of the event date. Satellite time is not event time.</p><p>Sentinel-2 after the event: <code>${esc(s.optical_post_scene_id || "none")}</code></p>`),
    row("02", "Observation gaps", `${kv({ "Radar layover, %": fmt(g.layover_percent, 2), "Geometric shadow, %": fmt(g.geometric_shadow_percent, 2), "Provider shadow, %": fmt(g.provider_shadow_percent, 2), "Masked in total, %": fmt(g.masked_percent, 2), "Valid change observation, %": fmt(sum.valid_change_observation_percent, 2) })}<p style="margin-top:8px">Masked areas are unobservable. The system reports them as unknown, never as no damage.</p>`),
    row("03", "OpenStreetMap before the event", `<p>Snapshot: <code>${esc(s.osm_snapshot)}</code> · cut-off proof passed: <b>${esc(s.osm_cutoff_proof_passed)}</b>. No edit after the event is used.</p>`),
    row("04", "Hazard thresholds", `${kv(Object.fromEntries(Object.entries(th).map(([k, v]) => [k.replaceAll("_", " "), v])))}<p style="margin-top:8px">Thresholds come from development data only. No reference map tuned them.</p>`),
    cmp,
    row("06", "EMSR927 comparison", `<p>The Copernicus EMS maps of this event (EMSR927) are used only to check the results, after the run is frozen. They are never an input. Results: REPORT.pdf, section 5, and <code>results/emsr927/</code>.</p>`),
    row("07", "Hypothetical scenarios", `<ul class="limits">${Object.entries(s.scenarios || {}).map(([k, v]) => `<li><b>S${k}.</b> ${esc(v)}</li>`).join("")}</ul>`),
    row("08", "Run notes", `<details><summary style="cursor:pointer;color:var(--teal-d);font-weight:600">${(s.limitations || []).length} notes recorded by the run</summary><ul class="limits" style="margin-top:8px">${(s.limitations || []).map((l) => `<li>${esc(l)}</li>`).join("")}</ul></details>`),
    row("09", "Provenance", `${kv({ "Producer commit": String(s.producer_commit || "").slice(0, 10), "Schema version": s.schema_version, Detector: s.detector, Status: s.status, "Run time, s": fmt(Object.values(s.timings_seconds || {}).reduce((x, y) => x + y, 0), 0) })}`),
  ].join("");
}

// ---------- start ----------
(async () => {
  const { runs } = await api("/api/map/runs");
  const want = new URLSearchParams(location.search).get("run");
  const first = runs.includes(want) ? want : (runs.includes("trishuli-2026-v3.0-r2") ? "trishuli-2026-v3.0-r2" : runs[0]);
  $("#runSelect").innerHTML = runs.map((r) => `<option value="${r}" ${r === first ? "selected" : ""}>${esc(PLACE[r] ? PLACE[r].split(" · ")[0] + (r.endsWith("unet") ? " (UNet)" : "") : r)}</option>`).join("");
  const tab = location.hash.slice(1); if (["evidence", "report", "about"].includes(tab)) show(tab);
  try { if (localStorage.getItem("siteLang") === "ne" || new URLSearchParams(location.search).get("lang") === "ne") setSiteLang("ne"); } catch (e) { /* storage unavailable */ }
  if (first) await loadRun(first); else toast("No saved run found. Run scripts/get_data.sh first.");
})();
