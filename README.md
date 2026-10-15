# HimaRoad Flood Copilot

Satellite evidence for flood mapping, cut-off settlement screening and bilingual situation reports.
Multimodal AI Hackathon 2026, Track B "Mapping Flood Damage from Space". AI component: situation-report copilot.

The copilot answers rescuers' questions and writes one-page situation reports in English and Nepali. Every number comes from the system's maps. Our own flood segmentation model (a UNet trained from scratch on Kuro Siwo) gives the copilot its water evidence, together with the radar change detector.

## What the system does

You give an area and a flood date. The system uses Sentinel-1 radar, Sentinel-2 optical images, the Copernicus DEM and OpenStreetMap from before the event. It answers three questions:

- **Where did the flood hit?** It maps possible water and debris-like change from images before and after the event.
- **What was damaged?** It marks buildings and roads that overlap a hazard as **potentially exposed**. Overlap is not confirmed damage.
- **Who is cut off?** It finds settlements that can lose the road connection to the nearest town or hospital, in four hypothetical scenarios.

The system has two parts:

| Part | Folder | What it gives |
|---|---|---|
| HimaRoad (mapping engine) | `src/himaroad/`, `configs/` | UNet segmentation model, hazard layers, cut-off analysis, DEM flow path from any upstream point, map HTML, PDF, GeoJSON and KML |
| flood-copilot (companion) | `backend/`, `upstream/` | Predefined questions and answers in English and Nepali, and the one-page situation report (PDF). Every number comes from the saved maps |
| Web app | `web/` | One page for the map, the copilot, the evidence and the report, on one port (8000) |
| Analysis workbench | `app.py`, `views/`, `ui/` | Streamlit pages that start a new live analysis (port 8501) |

## Deliverables

| Deliverable | Where |
|---|---|
| Map dashboard | The web app in `web/`, at http://127.0.0.1:8000 after `scripts/start.sh`. Pages: Overview (map and copilot), Evidence, Situation report, About and limits |
| One-page situation report from the system | "Situation report" page of the web app (English and Nepali PDF). Examples from the Trishuli case study: `results/situation-report/` |
| How to run the system | This README |
| Report (max. 6 pages) with limitations | [REPORT.md](REPORT.md), designed print version [REPORT.pdf](REPORT.pdf) (6 A4 pages) |
| 3-minute video | Link: *to be added* |
| Presentation deck | [deck/HimaRoad_Flood_Copilot.pdf](deck/HimaRoad_Flood_Copilot.pdf) (15 slides). The PowerPoint file is in the [v1.0 release](https://github.com/G-Shack/HimaRoad_flood-copilot/releases/tag/v1.0) |
| Data attributions and training-dataset citations | [Attributions](#attributions-and-citations) below, and REPORT.md section 8 |
| Case-study results compared with EMSR927 | REPORT.md section 5, and the result files in `results/emsr927/` |

## Requirements

- macOS or Linux, with Python 3.12 and [uv](https://docs.astral.sh/uv/).
- About 24 GB of RAM to prepare the Trishuli case study for the copilot.
- The Pango library for the PDF reports. On macOS: `brew install pango`. On Ubuntu: `sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0`.

## Install

1. Get the code:
   ```bash
   git clone https://github.com/G-Shack/HimaRoad_flood-copilot.git
   cd HimaRoad_flood-copilot
   ```
2. Install the locked environment (add `--extra unet` to use the UNet model):
   ```bash
   uv sync --locked --extra unet
   ```
3. Make your settings file. Then write absolute paths in it:
   ```bash
   cp .env.example .env
   chmod 600 .env
   ```
4. Download the pinned Nepali font (Noto Sans Devanagari, SIL Open Font License). The script refuses a file with the wrong SHA256:
   ```bash
   .venv/bin/python scripts/copilot/fetch_font.py --output data/copilot/cache/fonts
   ```
5. Download the saved case-study runs and the UNet weights from the release. The script checks every SHA256 sum:
   ```bash
   REPO=G-Shack/HimaRoad_flood-copilot scripts/get_data.sh
   ```

## Run the dashboard and the copilot

1. Prepare the case study for the copilot. This step takes about 40 s and about 12 GB of RAM:
   ```bash
   set -a; source .env; set +a
   PYTHONPATH=. .venv/bin/python -m scripts.copilot.prepare "$HIMAROAD_DATA_DIR/bundles/trishuli-2026-v3.0-r2" --output data/copilot/trishuli.json
   ```
2. Start the web app and the analysis workbench:
   ```bash
   scripts/start.sh
   ```
3. Open the web app: http://127.0.0.1:8000/?run=trishuli-2026-v3.0-r2. The copilot loads the run and checks the file hashes. This step takes about 15 s.

**On the "Overview" page:**

- The metric cards show the saved run. "Potentially road-cut settlements" changes with the hypothetical scenario.
- The map shows the run's own Sentinel-1 radar image and the hazard layers. It also shows the roads in a color for each evidence class, the potentially exposed buildings, the settlements and the mapped hospitals. The layer boxes below the map show and hide each layer.
- The "Radar before / after" slider compares the radar images from before and after the event.
- The scenario menu selects one of the four hypothetical scenarios. The settlement table and the map markers change with it.
- Click a settlement on the map or in the table to see its access, its exposed buildings and the nearest mapped hospital.
- The copilot panel answers the predefined questions in English (EN) or Nepali. Each answer shows numbered citations and the count of cited evidence claims. A typed question goes to the predefined question that it matches.
- "Flow path from an upstream point": click the map, then click "Trace flow path". The app traces the path down the valley on the Copernicus DEM and lists the settlements along it. This is a terrain trace, not an inundation forecast.


**English and Nepali:** the "English / नेपाली" switch in the header changes the whole page and the copilot language. The copilot answers and the PDF report use Nepali templates that a native speaker reviewed. The run records (scene IDs, limitations, thresholds) stay in English.

**On the other pages:**

- "Evidence": the scene IDs, the observation gaps, the OpenStreetMap snapshot and the thresholds. It also shows the detector comparison, the EMSR927 comparison, the scenarios, the limitations and the provenance of the run.
- "Situation report": click "Prepare English PDF" or the Nepali button to make the one-page PDF report. Then download it.
- "About and limits": the measured results, the limits of the system and the attributions.

The saved runs work without a network connection.

## Make a new analysis (live satellite data)

A new analysis needs free accounts. Put these values in `.env`:

- `SH_CLIENT_ID` and `SH_CLIENT_SECRET`: an OAuth client of the Copernicus Data Space Ecosystem (Sentinel Hub).
- `OHSOME_API_KEY`: the ohsome API v2, for OpenStreetMap as it was before the event.

Then:

1. Check that the credentials work. The script does not show secret values:
   ```bash
   set -a; source .env; set +a
   .venv/bin/python scripts/check_credentials.py
   ```
2. Check the satellite availability for the area and date. This step downloads no images:
   ```bash
   .venv/bin/himaroad precheck --bbox 85.30,28.12,85.42,28.30 --event-date 2026-08-26
   ```
3. Run the analysis. Use a new output folder for each run:
   ```bash
   .venv/bin/himaroad run --provider cdse --bbox 85.30,28.12,85.42,28.30 --event-date 2026-08-26 --output "$HIMAROAD_DATA_DIR/outputs/himaroad/my-run"
   ```
   Add `--detector unet --model-config configs/model_unet.yaml` to use the UNet model.
4. The workbench (http://127.0.0.1:8501) shows the new run in "Saved analysis". To use the run in the web app and the copilot, make a bundle of it first:
   ```bash
   .venv/bin/himaroad bundle --run "$HIMAROAD_DATA_DIR/outputs/himaroad/my-run" --out "$HIMAROAD_DATA_DIR/bundles/my-run"
   ```
   Then prepare the bundle (step 1 of "Run the dashboard and the copilot"). The web app lists every bundle in `$HIMAROAD_DATA_DIR/bundles` in "Current run".

If the system cannot pair before and after images of the same orbit track, the run ends as `incomplete` with the reason. It never shows a map that looks valid without evidence. An area above 600 km² ends as `aoi_too_large`. A dense city can take several minutes (REPORT.md section 7).

## Attributions and citations

- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30.
- © OpenStreetMap contributors.
- Kuro Siwo: Bountos et al., NeurIPS Datasets and Benchmarks 2024 (training data of the UNet).
- Sen1Floods11: Bonafilia et al., CVPR Workshops 2020 (surveyed, not used).
- European Union, Copernicus Emergency Management Service data (EMSR927, validation only).

The web app uses Leaflet 1.9.4 (BSD-2-Clause, in `web/vendor/`). The saved runs contain OpenStreetMap data from before the event. That data stays under the Open Database License (ODbL). The Nepali font is Noto Sans Devanagari under the SIL Open Font License.

## Licence and sources

The code is under the Apache License 2.0 ([LICENSE](LICENSE)). Data, model weights and reference products keep their own terms.

This repository contains clean copies of two development repositories:

| Part | Source commit |
|---|---|
| HimaRoad | `exceptnull/himaroad` `2c2619f22faf5d1e60ee5d75f09d5c998d6f5a1e` |
| flood-copilot | `G-Shack/flood-copilot` `8159771` |

The saved case-study runs record their own producer commits (`23a2527` for Trishuli, `7c1b74a` for Melamchi). A new run records the commit of this repository. The copilot adapter was reviewed against HimaRoad `7c1b74a` (`upstream/himaroad.ref`). The HimaRoad functions that the copilot uses did not change after that commit.
