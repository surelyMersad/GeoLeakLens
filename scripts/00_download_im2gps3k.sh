#!/usr/bin/env bash
# Download the Im2GPS3k test set (3000 images, ~455 MB) from the canonical
# Vo et al. ICCV 2017 MediaFire mirror.
#
# Pipeline: scrape the MediaFire interstitial → extract direct CDN URL →
# download → verify SHA-256.
#
# If the auto-scrape breaks because MediaFire's HTML changes, this script
# falls back to printing manual instructions. The expected SHA-256 below
# was computed from a verified 2026-04 download.

set -euo pipefail

DEST_DIR="data/raw/im2gps3k"
ZIP_PATH="$DEST_DIR/im2gps3ktest.zip"
EXTRACT_DIR="$DEST_DIR/im2gps3ktest"
PAGE_URL="https://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip"
EXPECTED_SHA256="c3c069a7632983998efcfed52f77687bf6375984fcfb02ec0c5d21fc78ff1892"

mkdir -p "$DEST_DIR"

# 1. Download (skip if zip already present and valid).
if [[ -f "$ZIP_PATH" ]]; then
    actual=$(shasum -a 256 "$ZIP_PATH" | awk '{print $1}')
    if [[ "$actual" == "$EXPECTED_SHA256" ]]; then
        echo "[skip] $ZIP_PATH already present with matching SHA-256"
    else
        echo "[warn] $ZIP_PATH present but hash mismatch ($actual); re-downloading"
        rm -f "$ZIP_PATH"
    fi
fi

if [[ ! -f "$ZIP_PATH" ]]; then
    echo "[1/2] resolving MediaFire CDN URL..."
    page=$(curl -skL "$PAGE_URL")
    direct=$(echo "$page" | grep -oE 'https://download[^"]*\.zip[^"]*' | head -1 || true)
    if [[ -z "$direct" ]]; then
        cat <<EOF
[error] could not auto-resolve MediaFire CDN URL — their interstitial may
        have changed. Manual fallback:
          1. Open: $PAGE_URL
          2. Click "Download (455 MB)"
          3. Save to: $ZIP_PATH
          4. Re-run this script (it will skip the download step)
EOF
        exit 1
    fi
    echo "[2/2] downloading from $direct"
    curl -skL "$direct" -o "$ZIP_PATH"
fi

# 2. Verify SHA-256.
actual=$(shasum -a 256 "$ZIP_PATH" | awk '{print $1}')
if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
    echo "[error] SHA-256 mismatch:"
    echo "  expected: $EXPECTED_SHA256"
    echo "  actual:   $actual"
    exit 1
fi
echo "[ok] SHA-256 verified"

# 3. Extract.
if [[ -d "$EXTRACT_DIR" ]] && [[ $(find "$EXTRACT_DIR" -name '*.jpg' | head -1) ]]; then
    n=$(find "$EXTRACT_DIR" -name '*.jpg' | wc -l | tr -d ' ')
    echo "[skip] $EXTRACT_DIR already contains $n .jpg files"
else
    echo "[extract] unzipping into $DEST_DIR/"
    unzip -q -o "$ZIP_PATH" -d "$DEST_DIR"
    n=$(find "$EXTRACT_DIR" -name '*.jpg' | wc -l | tr -d ' ')
    echo "[ok] extracted $n images"
fi

echo
echo "Next: build the manifest"
echo "  .venv/bin/python -m geoleaklens.data.ingest_im2gps3k \\"
echo "    --raw-dir $EXTRACT_DIR \\"
echo "    --out data/processed/manifests/im2gps3k_test.parquet"
