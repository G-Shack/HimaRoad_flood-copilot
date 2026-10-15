"""Lossless canonical exports. No language-specific fact rewriting."""

import csv
import io
import json
import zlib
from collections.abc import Iterator

from backend.app.reports.facts import ReportFacts, verify_report


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def iter_json(facts: ReportFacts) -> Iterator[bytes]:
    # Validate before returning an iterator, hence before HTTP headers are sent.
    verify_report(facts)
    metadata = facts.model_dump(mode="json", exclude={"claims"})

    def chunks() -> Iterator[bytes]:
        yield b"{\n"
        for index, key in enumerate(sorted((*metadata, "claims"))):
            yield ((",\n" if index else "") + "  " + json.dumps(key) + ": ").encode()
            if key != "claims":
                yield _json(metadata[key]).replace("\n", "\n  ").encode()
                continue
            yield b"["
            for number, claim in enumerate(facts.claims):
                yield b",\n" if number else b"\n"
                yield (
                    "    " + _json(claim.model_dump(mode="json")).replace("\n", "\n    ")
                ).encode()
            yield b"\n  ]" if facts.claims else b"]"
        yield b"\n}\n"

    return chunks()


def export_json(facts: ReportFacts) -> bytes:
    return b"".join(iter_json(facts))


def iter_csv(facts: ReportFacts) -> Iterator[bytes]:
    verify_report(facts)

    def chunks() -> Iterator[bytes]:
        stream = io.StringIO(newline="")
        fields = [
            "reference",
            "section",
            "id",
            "run_id",
            "kind",
            "label",
            "value",
            "unit",
            "subject",
            "scenario",
            "status",
            "sources",
            "method",
            "parameters",
            "observation_times",
            "quality_flags",
            "derived_by",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        yield stream.getvalue().encode()
        by_id = {c.id: c for c in facts.claims}
        reference = 0
        for section in facts.sections:
            for identifier in section.claim_ids:
                row = by_id[identifier].model_dump(mode="json")
                for key in ("value", "sources", "parameters", "observation_times", "quality_flags"):
                    row[key] = json.dumps(
                        row[key], ensure_ascii=False, sort_keys=True, allow_nan=False
                    )
                reference += 1
                row["reference"] = f"C{reference:03d}"
                row["section"] = section.id
                # Quote arbitrary strings as JSON to prevent spreadsheet formulas.
                for key in ("label", "subject", "method"):
                    row[key] = json.dumps(row[key], ensure_ascii=False)
                stream.seek(0)
                stream.truncate(0)
                writer.writerow(row)
                yield stream.getvalue().encode()

    return chunks()


def export_csv(facts: ReportFacts) -> bytes:
    return b"".join(iter_csv(facts))


def gzip_chunks(chunks: Iterator[bytes]) -> Iterator[bytes]:
    compressor = zlib.compressobj(wbits=31)
    for chunk in chunks:
        compressed = compressor.compress(chunk)
        if compressed:
            yield compressed
    yield compressor.flush()
