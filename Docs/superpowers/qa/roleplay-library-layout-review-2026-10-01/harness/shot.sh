#!/usr/bin/env bash
# Usage: shot.sh <socket> <out-path-without-extension>   -> <out>.txt (plain) + <out>.ansi (SGR)
# Relative <out> paths land under $HARNESS_STATE/captures/ (never in a checkout).
# PNG=1 also renders <out>.png via shot_png.py (Rich -> HTML -> headless Chromium via playwright; slower).
set -euo pipefail
. "$(dirname "$0")/env.sh"
[ $# -eq 2 ] || rp_die "usage: shot.sh <socket> <out-path-without-extension>"
S="$1"; O="$2"
case "$O" in /*) ;; *) O="$HARNESS_STATE/captures/$O" ;; esac
mkdir -p "$(dirname "$O")"
tmux -L "$S" capture-pane -p > "$O.txt"
tmux -L "$S" capture-pane -e -p -J > "$O.ansi"
if [ "${PNG:-0}" = "1" ]; then "$PY" -B "$HARNESS_DIR/shot_png.py" "$O.ansi" "$O.png" >/dev/null 2>&1 || echo "png failed" >&2; fi
echo "$O.txt"
