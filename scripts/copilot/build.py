"""Trusted local report builder. HTTP/UI permission binding belongs to Phase05."""

import argparse
import hashlib
import json
from pathlib import Path

from backend.app.copilot.tools import open_context
from backend.app.reports.exports import export_csv, export_json
from backend.app.reports.facts import ReportFacts, build_report
from backend.app.reports.pdf import map_claims, pdf_bytes, report_html
from backend.app.settings import Settings
from backend.app.storage import import_evidence
from backend.app.upstream.paths import authorised_path


def write_report(facts: ReportFacts, output: Path, font: Path) -> dict[str, str]:
    # Render both completely before publishing any artifact.
    products = {"report-facts.json": export_json(facts), "report-claims.csv": export_csv(facts)}
    products["map-claims.json"] = (
        json.dumps(
            [c.model_dump(mode="json") for c in map_claims(facts)],
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    for language in ("en", "ne"):
        products[f"report-{language}.pdf"] = pdf_bytes(facts, language, font)
        products[f"evidence-{language}.html"] = report_html(
            facts, language, font, full=True
        ).encode()
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, data in products.items():
        path = output / name
        if path.is_symlink():
            raise ValueError("Report destination is a symlink")
        path.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (output / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    settings = Settings(
        _env_file=None,
        data_root=args.data_root,
        himaroad_data_dir=args.data_root,
        collab_agent="codex",
    )
    identifier, _ = import_evidence(args.run, settings)
    context = open_context(identifier, frozenset({identifier}), settings)
    facts = build_report(context, frozenset({identifier}))
    output = authorised_path(args.data_root / "outputs" / identifier / "reports", settings)
    hashes = write_report(facts, output, args.font)
    print(json.dumps({"output": str(output), "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
