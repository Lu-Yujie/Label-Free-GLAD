#!/usr/bin/env bash
# Fetch the official SUL-GUL code at the commit we evaluated into ./src, then rebuild our files that reuse its lines.
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
URLS="https://github.com/ZhenyuYangMQ/UncertaintyGLAD"        # tried in order
COMMIT="1a1f2e42ce48fe14360a03963785081347a3e929"    # 2026-01-16
DEST="$HERE/src"
if [ ! -d "$DEST/.git" ]; then
  for URL in $URLS; do
    GIT_TERMINAL_PROMPT=0 git clone --quiet "$URL" "$DEST" && break || rm -rf "$DEST"
  done
  [ -d "$DEST/.git" ] || { echo "could not clone SUL-GUL from: $URLS" >&2; exit 1; }
fi
git -C "$DEST" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$DEST" checkout --quiet --detach "$COMMIT"
test "$(git -C "$DEST" rev-parse HEAD)" = "$COMMIT" || { echo "wrong commit in $DEST" >&2; exit 1; }
test -z "$(git -C "$DEST" status --porcelain)" || { echo "$DEST has local changes" >&2; exit 1; }
python3 "$ROOT/scripts/rebuild_from_upstream.py" "$HERE"/rebuild/*.recipe
echo "SUL-GUL: src/ at $COMMIT, adapter files rebuilt"
