# HimaRoad Flood Copilot: satellite evidence for flood mapping, access screening and bilingual reports

Multimodal AI Hackathon 2026, Track B "Mapping Flood Damage from Space". AI component: situation-report copilot.

## 1. Purpose and system overview

The system takes an area and a flood date. It uses free satellite data to answer three questions for rescue teams:

1. **Where did the flood hit?** The system compares Sentinel-1 radar images from before and after the event. It adds Sentinel-2 optical images when the sky is clear.
2. **What was damaged?** The system overlays buildings, roads and bridges from OpenStreetMap (OSM) as it was before the event. It marks the features that overlap a hazard as **potentially exposed**. Overlap is not confirmed damage.
3. **Who is cut off?** The system builds road and path graphs. It finds settlements that can lose their connection to the nearest town or health facility in four hypothetical scenarios.

The system has two parts and one web app in one repository:

| Part | What it does |
|---|---|
| **HimaRoad** (mapping engine) | Acquisition and pairing, hazard layers, exposure, settlement access, DEM flow path, a UNet segmentation model, map HTML, a one-page PDF, GeoJSON and KML |
| **flood-copilot** (companion) | A FastAPI service. It reads HimaRoad's saved evidence and answers predefined questions in English (EN) and Nepali (NE). It writes one-page situation reports. Every number comes from the maps |
| **Web app** | One page joins the map dashboard and the copilot on one port, in English and Nepali. A Streamlit workbench starts new live analyses |

**AI component: situation-report copilot.** The copilot answers rescuers' questions and writes one-page situation reports in English and Nepali. Every number comes from the system's maps. Our own flood segmentation model (a UNet trained from scratch on Kuro Siwo) gives the copilot its water evidence, together with the radar change detector. Section 4 gives the results of both.

**Bonus.** From any point upstream, the system traces a flow path down the valley on the Copernicus DEM. It lists the settlements along the path.

## 2. Data and compliance

**Inputs.** The system uses only the allowed inputs:

- Sentinel-1 gamma0, processed on request from raw GRD by the Copernicus Data Space Ecosystem (CDSE) Sentinel Hub, with Copernicus-30 orthorectification (`GAMMA0_TERRAIN`, `COPERNICUS_30`).
- Sentinel-2 L2A with cloud masks.
- Copernicus GLO-30 DEM, and a height-above-nearest-drainage (HAND) layer that we derive from it.
- OSM from the ohsome v2 API, at 00:00 UTC on the day before the event (2026-08-25T00:00Z for the case study). Each run records an edit-timestamp proof.
- Kuro Siwo for model training.

Earth Engine is a flagged fallback. We did not use it for the case study or for judging.

**Validation only.** Copernicus EMS (EMSR927) and other published flood or damage maps never enter inference, thresholds, training or model selection. We froze the case-study runs first (tag `case-study-frozen`, 02:17:40 IST). The reference folder appeared after the freeze (02:17:51.87 IST). We logged every read of reference material in a leakage log. We did not tune anything on the reference after the comparison.

**Provenance.** Each run saves its producer commit, schema version, scene IDs, thresholds, model version and input checksums. The full attributions are in Section 8.

## 3. Method

### 3.1 Acquisition and pairing

The system searches the CDSE catalogue for Sentinel-1 scenes before and after the event. It pairs only scenes from the same orbit track, because scenes from different tracks view the terrain from different angles. If no same-track pair exists, the run ends as `incomplete` with the reason. It does not make a map that looks valid. All layers align on a 20 m UTM grid. Per-date provider masks define where the data is valid on all dates.

### 3.2 Hazard layers

- **New water.** The default rule marks a pixel when the post-event value is below -16 dB. The drop from the pre-event median must also be at least 3 dB. The rule excludes permanent water and steep slopes.
- **Disturbance and possible debris.** A two-sided change test finds darkening and brightening. Brightening in the valley-bottom corridor becomes `possible_debris`. This class is **unvalidated** rule evidence. It can also be a landslide, scour, construction or vegetation change.
- **Unobservable areas.** Radar layover and shadow hide steep terrain. We trace them along each range line of the DEM and add the provider's shadow mask. In the case study, 24.9% of the area of interest (AOI) is masked. Unobservable means **unknown**, never zero.
- **Other steps.** The system removes polygons smaller than 1,600 m² (4 pixels). HAND is only a soft flag. Sentinel-2 indices give separate corroboration. All thresholds are in `configs/default.yaml` and in each output.

### 3.3 Exposure

The system intersects hazard layers with OSM roads, bridges and buildings. Each road link gets one state: potentially exposed, observed with no detected change, or unobservable. Operational status stays `unknown` unless an independent source gives it.

### 3.4 Cut-off settlement analysis

The system builds directed vehicle and pedestrian graphs. It first checks access before the event, so a settlement that had no connection is not a new cut-off. Then it removes links in four **hypothetical** scenarios:

| Scenario | Links removed |
|---|---|
| 1 | Independently sourced confirmed closures |
| 2 (primary) | Scenario 1 plus links with observed change or likely impact |
| 3 | Scenario 1 plus links with radar and optical corroboration (a subset of scenario 2) |
| 4 (worst case) | Scenario 2 plus possible-impact, unobservable and operationally unknown links |

Settlements are clusters of OSM buildings. Destinations are the nearest town and health facilities. The output gives the nearest reachable destination for each settlement and a candidate route. A route is mapped reachability. It is **not** a safe route.

### 3.5 Flow path (bonus)

The system fills the DEM with priority-flood conditioning. Then it follows D8 steepest descent from the start point. It lists the settlements near the path. The path shows where water from the point goes on the DEM surface. It does not give depth, timing or extent, and it is not a forecast.

## 4. AI component

### 4.1 UNet flood segmentation model

**Training.** We trained a UNet from scratch on Kuro Siwo labels only. We did not use a pretrained encoder. We put the labels again on CDSE gamma0 at 20 m. After the registered checks, 310 training chips (Houston, Togo, Peru) and 134 development chips (Larkana, Illinois) remained. We trained four registered variants for 30 epochs each, with seed 42, BCE+Dice loss, AdamW, learning rate 0.0003 and batch 4. The lowest development loss selected the variant with base width 32 and four channels (pre and post VV/VH). We fixed the temperature (1.4940904) and threshold (0.57) on development data at a precision floor of 0.80. We did this before the Himalayan test.

**Development results** (1,167,702 valid development pixels):

| Method | Precision | Recall | IoU | F1 |
|---|---:|---:|---:|---:|
| Rule baseline | 0.8835 | 0.6629 | 0.6096 | 0.7575 |
| Selected UNet | 0.8008 | 0.9457 | 0.7656 | 0.8672 |

These events selected the model. Thus, these numbers are development evidence, not independent accuracy. Only two development events exist, and Larkana dominates the pixel counts.

**Himalayan test (once only).** We froze predictions for three held-out Himalayan floods: Teesta 2023, Melamchi 2021 and Chamoli 2021. Two independent reviewers labelled blind windows and then adjudicated together. The primary set had one valley-enriched 128×128 window per event.

| Event | Labelled new water (px) | UNet recall / IoU | Baseline precision / recall / IoU | Reviewer κ |
|---|---:|---|---|---:|
| Teesta 2023 | 81 | 0.000 / 0.000 | 0.688 / 0.136 / 0.128 | 0.000 |
| Melamchi 2021 | 970 | 0.000 / 0.000 | 1.000 / 0.013 / 0.013 | 0.679 |
| Chamoli 2021 | 247 | 0.000 / 0.000 | undefined / 0.000 / 0.000 | 0.102 |

**The UNet detected 0 of 1,298 labelled new-water pixels.** Its recall and IoU are 0. Its precision is undefined because it made no positive predictions. The rule baseline detected 24 of 1,298 (precision 0.828, recall 0.018, IoU 0.018). In the six secondary windows, neither method detected labelled new water. The model does not yet transfer to narrow Himalayan valleys. We did not change any threshold, model or checkpoint after the test. The three events are now consumed.

The test is small. It has one window per event, and we chose the windows for valley floor, so they are not representative. Reviewer agreement was low on Teesta and Chamoli.

### 4.2 Situation-report copilot

**Grounding.** The copilot never makes a number. It reads each HimaRoad run and makes a set of evidence claims. Each claim has a source file, a path in that file, a method and a checksum. Every answer and every report cites these claims by ID. Missing observations stay unknown. The copilot refuses requests for safe routes.

**Predefined questions.** The submission uses deterministic predefined questions and templates. Thirty-two predefined cases (eight question types × EN/NE × two areas) passed validation and exact claim-ID parity between English and Nepali. Deterministic route forms also passed parity and cited-card binding.

**Nepali.** Siku, a native speaker, reviewed the 165-entry wording inventory on 2026-10-15. The record has 162 approvals and 3 corrections. This review covers the authored templates only.

**Reports.** The copilot makes a one-page PDF report in English and Nepali. The final proof shows all four PDFs (Trishuli and Melamchi, EN and NE) as one page each. Each PDF has correct glyphs, the native-review marker, all eight authored caveats and full credits.

## 5. Case study: Trishuli, 26 August 2026

**Runs.** We froze two runs before we opened EMSR927: a rule baseline run and a UNet run (producer `23a2527`). The two runs are **identical on every scored item**. The UNet had one water candidate, and the minimum-mapping-unit and permanent-water step removed it. Thus, the disturbance layer in the case study is **rule evidence, not model output**. The flood extent is empty in both runs.

**What we compared.** EMSR927 is a grading product. Its only observed-event class in these AOIs is "Landslide". Thus, water IoU is unavailable, not zero. We compared the landslide footprint with our disturbance layer (positive-only area recall). We also compared graded transport features with our potentially exposed road links (20 m match).

| EMSR927 AOI | Area recall (observed footprint) | Affected transport recalled | "No visible damage" flagged |
|---|---:|---:|---:|
| AOI01 Syapru Besi | **0.012** | **6 of 19** | 0 of 8 |
| AOI02 Timure | **0.038** | **7 of 13** | none graded |
| AOI03 Bidur, AOI05 Phosretar | unavailable (outside our AOI) | unavailable | unavailable |
| AOI04 Bharatpur, AOI06 Kyundi | unavailable (reference not feasible) | unavailable | unavailable |

**Failure analysis.** We observed most of the landslide footprint (93.1% in AOI01, 76.1% in AOI02). Thus, the low recall is a detection miss, not a visibility gap. In AOI01, 11 of the 13 missed affected features lie on links that we observed with no detected change. In AOI02, 5 of the 6 misses are fully or partly unobservable. Across the run, 56.1% of road links are unobservable and stay unknown. The reference interprets very-high-resolution optical images by eye. We measure 20 m radar change.

**Result.** The map shows where links are potentially exposed. It **does not reliably delineate this landslide event**. Overlap is agreement between two satellite products. It never confirms damage or closure.

**Time.** The event time is unknown. Our primary radar post-event scene is from 2026-08-28 12:21 UTC, 60.4 h after the event-date boundary. Satellite time is not event time.

## 6. Live robustness and latency

**Worldwide drills.** We ran six public AOIs outside training, development and the Himalayan test. Each run used the UNet detector and a 180 s budget. The drills test robustness only. They make no accuracy claim.

| Case | Result |
|---|---|
| Lokoja, Nigeria | Complete. Cold run 99.7 s. Later runs 56.7 s and 59.9 s |
| Lismore, Australia | Complete. Runs 45.9–69.7 s |
| Valencia, Spain | Timed out at the 180 s budget |
| Porto Alegre, Brazil | `incomplete`: no same-track pre-event scene |
| Manila (2010, before Sentinel-1) | `incomplete`: no scenes |
| Germany, 7,807 km² | `incomplete`, `aoi_too_large`, nothing downloaded |

Two of six AOIs completed end to end. The other runs stopped with a stated reason. They did not make a map that looks valid. One Lokoja rerun took 159.8 s, because the Sentinel-2 search is not cached (optical stage 20.7 s to 104.9 s).

**Detector latency.** On real inputs of 600 km² (1,433,870 observable pixels), the warm median detector time was 4.929 s on CPU and 1.702 s on GPU. This excludes acquisition, OSM, network analysis and export. It is not end-to-end latency.

**Copilot latency** (saved runs, individual measurements, not service levels):

| Operation | Trishuli | Melamchi |
|---|---:|---:|
| Claims verified | 62,263 | 178,648 |
| Offline preparation (s) | 67.57 | 524.46 |
| Predefined answer (s) | 0.20–0.35 | 0.62–1.02 |
| Final PDF, EN / NE (s) | 8.15 / 7.53 | 45.01 / 53.58 |

## 7. Limitations

**What the system cannot do:**

- **No early warning.** Sentinel satellites revisit an area only every few days. Thus, the system cannot warn of a sudden glacier collapse minutes ahead. It maps what the next pass shows, after the event.
- **Unobservable areas.** Radar layover and shadow hide steep valley walls. In the case study, 24.9% of the AOI is masked and 56.1% of road links are unobservable. These areas are unknown, not safe and not zero.
- **Exposure is not damage.** Overlap with a hazard means potentially exposed. The system cannot confirm destruction, collapse or closure. Operational status stays unknown.
- **Scenarios are hypothetical.** Cut-off results are graph scenarios. Routes are mapped reachability, not safe routes. OSM gaps, tags and the finite graph buffer limit the results.
- **Himalayan transfer.** The UNet detected 0 of 1,298 labelled new-water pixels on the blind Himalayan test. Its development results do not yet transfer to narrow valleys.
- **No validated debris class.** The UNet maps water only. `possible_debris` is unvalidated rule evidence. In the case study, the disturbance layer did not reliably delineate the landslides (area recall 0.012 and 0.038).
- **Sensing limits.** Mixed pixels, vegetation, snow, cloud gaps in optical images and features narrower than about 20 m limit detection. Satellite time is not event time.
- **Dense cities are slow.** The Valencia diagnostic took 518 s against the 180 s budget. Infrastructure exposure took 246.9 s, graph construction took 126.6 s and export took 72.6 s.
- **Data gaps.** If no same-track pair exists, or the date is before Sentinel-1, the run ends as `incomplete`.

**Copilot limits:**

- Route reopening is unavailable above 50 settlements or 2,000 links.
- The map view is partial (at most 100 features per layer). The full exports stay authoritative.

**Responsibility.** The system makes no casualty figures, no claim about the cause of the event, no confirmed-collapse claim and no safe-route claim. Independent inspection is necessary before any use.

## 8. Attributions and citations

- "Contains modified Copernicus Sentinel data 2026."
- "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."
- "The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30."
- "© OpenStreetMap contributors."
- Kuro Siwo: Bountos et al., NeurIPS Datasets and Benchmarks 2024 (training data).
- Sen1Floods11: Bonafilia et al., CVPR Workshops 2020 (surveyed, not used).
- "European Union, Copernicus Emergency Management Service data" (EMSR927, validation only).

Code licence: Apache-2.0. Data licences apply separately.
