#!/usr/bin/env bash
# Regenerate the CSS sheets (see tldw_chatbook/css/CSS-MODULAR.md).
# Runs build_css.py with $PYTHON when set, else with the first of the repo's
# .venv, python3, python3.14, python3.13, python3.12 that meets the project's
# Python 3.12 floor. On macOS, a bare python3 can be the system 3.9.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

if [ -n "${PYTHON:-}" ]; then
  candidates=("$PYTHON")
else
  candidates=(.venv/bin/python python3 python3.14 python3.13 python3.12)
fi
for candidate in "${candidates[@]}"; do
  if "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' 2>/dev/null; then
    exec "$candidate" tldw_chatbook/css/build_css.py
  fi
done
echo "build_css.sh: no Python 3.12 or newer found (tried: ${candidates[*]})." \
  "Re-run as PYTHON=/path/to/python3.12-or-newer ./build_css.sh" >&2
exit 1
