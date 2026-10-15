"""Fetch pinned OFL font/license outside git; refuse any checksum mismatch."""

import argparse
import hashlib
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path("backend/app/reports/font.json").read_text())
    products = []
    for filename, url_key, hash_key in [
        ("NotoSansDevanagari.ttf", "url", "sha256"),
        ("OFL.txt", "license_url", "license_sha256"),
    ]:
        response = requests.get(config[url_key], timeout=30)
        response.raise_for_status()
        if (
            len(response.content) > 4 * 1024 * 1024
            or hashlib.sha256(response.content).hexdigest() != config[hash_key]
        ):
            raise ValueError("Font/license size or checksum mismatch")
        products.append((filename, response.content))
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, data in products:
        path = args.output / filename
        if path.is_symlink():
            raise ValueError("Font destination is a symlink")
        path.write_bytes(data)
    print("Pinned font and OFL license hashes verified")


if __name__ == "__main__":
    main()
