#!/usr/bin/env bash
# Fetch the official ProtoGLAD code at the commit we evaluated into ./src, then rebuild our files that reuse its lines.
# The repository has no licence, so none of its code is redistributed here.
#
#   bash fetch_src.sh
#
# src/ itself is never edited. Our changes live in the adapter files next to this script; the ones that reuse
# upstream lines are shipped as recipes in rebuild/ and rebuilt from src/ by scripts/rebuild_from_upstream.py, which
# checks the sha256 of every rebuilt file.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
URLS="https://github.com/IsolationKernel/ProtoGLAD"        # tried in order
COMMIT="f7d8ce953de2c559abb9502c3a2b8195f3da380a"    # 2026-03-28
DEST="$HERE/src"
if [ ! -d "$DEST/.git" ]; then
  for URL in $URLS; do
    GIT_TERMINAL_PROMPT=0 git clone --quiet "$URL" "$DEST" && break || rm -rf "$DEST"
  done
  [ -d "$DEST/.git" ] || { echo "could not clone ProtoGLAD from: $URLS" >&2; exit 1; }
fi
git -C "$DEST" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$DEST" checkout --quiet --detach "$COMMIT"
test "$(git -C "$DEST" rev-parse HEAD)" = "$COMMIT" || { echo "wrong commit in $DEST" >&2; exit 1; }
test -z "$(git -C "$DEST" status --porcelain)" || { echo "$DEST has local changes" >&2; exit 1; }
python3 "$ROOT/scripts/rebuild_from_upstream.py" "$HERE"/rebuild/*.recipe
echo "ProtoGLAD: src/ at $COMMIT, adapter files rebuilt"
