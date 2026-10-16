#!/usr/bin/env bash
# Download the TU datasets used in the study (TUDataset collection, Morris et al. 2020) and lay them out the way the
# code expects:
#   $GLAD_DATA_ROOT/raw/<DS>/...                 read by common/ (fold caches, training-free references)
#   $GLAD_DATA_ROOT/ubgold_run/data/<DS>/<DS>/raw/...   UB-GOLD's loaders (PyG TUDataset root), TU datasets
#   $GLAD_DATA_ROOT/ubgold_run/data/<Tox21 file>/...    UB-GOLD's Tox21 reader (fixed _training / _testing files)
# Each zip is checked against the sha256 of the copy used in the study (data/raw_zip_sha256.txt).
#
#   export GLAD_DATA_ROOT=/path/to/glad_data
#   bash data/download_tu.sh [DS ...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${GLAD_DATA_ROOT:?set GLAD_DATA_ROOT first}"
BASE="${TU_BASE_URL:-https://www.chrsmrrs.com/graphkerneldatasets}"
ALL="AIDS BZR COX2 DHFR ENZYMES NCI1 PROTEINS_full IMDB-BINARY REDDIT-BINARY DD COLLAB MUTAG
     Tox21_p53_training Tox21_p53_testing Tox21_MMP_training Tox21_MMP_testing
     Tox21_HSE_training Tox21_HSE_testing Tox21_PPAR-gamma_training Tox21_PPAR-gamma_testing"
LIST="${*:-$ALL}"
RAW="$GLAD_DATA_ROOT/raw"; UB="$GLAD_DATA_ROOT/ubgold_run/data"
mkdir -p "$RAW" "$UB"
for DS in $LIST; do
  z="$RAW/$DS.zip"
  [ -s "$z" ] || curl -fsSL -o "$z" "$BASE/$DS.zip"
  want="$(awk -v f="$DS.zip" '$2==f {print $1}' "$HERE/raw_zip_sha256.txt")"
  got="$(sha256sum "$z" | cut -d' ' -f1)"
  if [ -n "$want" ] && [ "$want" != "$got" ]; then
    echo "WARNING: $DS.zip sha256 $got differs from the study's copy ($want)" >&2
  fi
  rm -rf "$RAW/$DS" && unzip -q -o "$z" -d "$RAW"
  case "$DS" in
    Tox21_*) rm -rf "$UB/$DS" && cp -r "$RAW/$DS" "$UB/$DS" ;;
    *)       mkdir -p "$UB/$DS/$DS/raw" && cp "$RAW/$DS"/*.txt "$UB/$DS/$DS/raw/" ;;
  esac
  echo "$DS ok"
done
