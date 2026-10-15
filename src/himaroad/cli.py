import argparse
import importlib.util
import os

from .common import data_root


def main():
    parser = argparse.ArgumentParser(prog="himaroad")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    r = sub.add_parser("run")
    r.add_argument(
        "--provider",
        choices=["cdse", "local", "gee"],
        default=None,
        help="default: cdse, or local when --manifest is given; gee is a flagged fallback",
    )
    r.add_argument("--manifest")
    r.add_argument("--bbox", type=lambda s: [float(x) for x in s.split(",")])
    r.add_argument("--polygon")
    r.add_argument("--event-date")
    r.add_argument(
        "--event-time", help="optional ISO UTC event time; offsets otherwise use the date"
    )
    r.add_argument("--cutoff")
    r.add_argument("--detector", choices=["baseline", "terramind", "unet"], default=None)
    r.add_argument("--output")
    r.add_argument("--config")
    r.add_argument("--project")
    r.add_argument("--model-config")
    pc = sub.add_parser("precheck", help="Sentinel-1 availability for an AOI/date; no download")
    pc.add_argument("--bbox", type=lambda s: [float(x) for x in s.split(",")])
    pc.add_argument("--polygon")
    pc.add_argument("--event-date", required=True)
    pc.add_argument("--cutoff")
    pc.add_argument("--event-time", help="optional ISO UTC event time")
    fp = sub.add_parser(
        "flowpath", help="DEM-derived flow path from a point (not an inundation forecast)"
    )
    fp.add_argument("--run", required=True)
    fp.add_argument("--lon", type=float, required=True)
    fp.add_argument("--lat", type=float, required=True)
    fp.add_argument("--corridor-m", type=float, default=500.0)
    fp.add_argument("--name")
    fp.add_argument("--out", help="write GeoJSON here; default: stdout (never into the run)")
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True)
    e.add_argument("--reference", required=True)
    b = sub.add_parser("bundle", help="portable copy of a run with its inputs")
    b.add_argument("--run", required=True)
    b.add_argument("--out", required=True)
    v = sub.add_parser("verify-bundle")
    v.add_argument("bundle")
    x = sub.add_parser("export")
    x.add_argument("--run", required=True)
    x.add_argument("--language", choices=["en", "ne"], default="en")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "doctor":
        import json

        deps = {
            n: bool(importlib.util.find_spec(n))
            for n in [
                "numpy",
                "pandas",
                "rasterio",
                "geopandas",
                "shapely",
                "pyproj",
                "networkx",
                "ee",
                "streamlit",
                "plotly",
                "reportlab",
                "pytest",
                "torch",
                "terratorch",
            ]
        }
        print(
            json.dumps(
                {
                    "dependencies": deps,
                    "ee_project_configured": bool(os.environ.get("EE_PROJECT")),
                    "earth_engine_authenticated": "not_checked",
                    "terramind": "compatible_checkpoint_required",
                    "data_directory": str(data_root()),
                },
                indent=2,
            )
        )
    elif command == "run":
        from .pipeline import run

        output = run(**args)
        print(output)
        from .common import read_config

        if read_config(output / "report.json")["status"] == "incomplete":
            raise SystemExit(2)
    elif command == "bundle":
        from .bundle import create_bundle, verify_bundle

        print(verify_bundle(create_bundle(args["run"], args["out"])))
    elif command == "verify-bundle":
        from .bundle import verify_bundle

        print(verify_bundle(args["bundle"]))
    elif command == "precheck":
        import json

        from .cdse import precheck
        from .common import aoi_geometry, read_config
        from pathlib import Path

        config = read_config(Path(__file__).with_name("default.yaml"))
        if args["event_time"]:
            from .pipeline import validate_event_time

            validate_event_time(args["event_time"], args["event_date"])
            config["event_time"] = args["event_time"]
        geom = aoi_geometry(args["bbox"], args["polygon"])
        print(json.dumps(precheck(geom, args["event_date"], args["cutoff"], config), indent=2))
    elif command == "flowpath":
        import json
        from pathlib import Path

        from .flowpath import trace_for_run

        collection = trace_for_run(
            args["run"], args["lon"], args["lat"], args["corridor_m"], args["name"]
        )
        text = json.dumps(collection, indent=2)
        if args["out"]:
            Path(args["out"]).write_text(text)
            print(args["out"])
        else:
            print(text)
    elif command == "evaluate":
        from .evaluate import evaluate

        print(evaluate(args["run"], args["reference"]))
    else:
        from .export import export_run

        print(export_run(args["run"], args["language"]))


if __name__ == "__main__":
    main()
