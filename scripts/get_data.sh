#!/bin/bash
# Download the saved case-study runs and the UNet weights from the v1.0 release,
# check the SHA256 sums, and put each file where the system reads it.
# After this step, the dashboard, the copilot and the reports work offline.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${REPO:-G-Shack/HimaRoad_flood-copilot}"
TAG="${TAG:-v1.0}"
DATA="${HIMAROAD_DATA_DIR:-$PWD/data}"
URL="https://github.com/$REPO/releases/download/$TAG"
FILES="trishuli-2026-v3.0-r2.tar.gz trishuli-2026-v3.0-r2-unet.tar.gz melamchi-2021-v3.0-r2.tar.gz unet-kuro-v2-w32-c4.pt SHA256SUMS"

mkdir -p "$DATA/release"
for file in $FILES; do
  [ -s "$DATA/release/$file" ] || curl -fL --retry 3 -o "$DATA/release/$file" "$URL/$file"
done
(cd "$DATA/release" && grep -E "\.tar\.gz$|\.pt$" SHA256SUMS | shasum -a 256 -c -)

for run in trishuli-2026-v3.0-r2 trishuli-2026-v3.0-r2-unet melamchi-2021-v3.0-r2; do
  if [ ! -e "$DATA/bundles/$run" ]; then
    mkdir -p "$DATA/bundles/$run"
    tar -xzf "$DATA/release/$run.tar.gz" -C "$DATA/bundles/$run"
  fi
  HIMAROAD_DATA_DIR="$DATA" .venv/bin/himaroad verify-bundle "$DATA/bundles/$run"
done

MODEL="$DATA/models/himaroad/unet/unet-kuro-v2-w32-c4"
mkdir -p "$MODEL"
[ -s "$MODEL/model.pt" ] || cp "$DATA/release/unet-kuro-v2-w32-c4.pt" "$MODEL/model.pt"
echo "Data is ready in $DATA"
