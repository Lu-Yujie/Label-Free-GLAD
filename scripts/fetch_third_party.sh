#!/usr/bin/env bash
# Fetch the five method repositories that have no licence (MUSE, GLADMamba, HC-GLAD, ProtoGLAD, SUL/GUL) at the
# commits we evaluated and rebuild the adapter files that reuse their lines. UB-GOLD, GLADPro, GLASS, LGKDE and SEGO
# are MIT-licensed and included.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for m in MUSE GLADMamba HC-GLAD ProtoGLAD SUL-GUL; do
  bash "$ROOT/evalpaper/ext/$m/fetch_src.sh"
done
