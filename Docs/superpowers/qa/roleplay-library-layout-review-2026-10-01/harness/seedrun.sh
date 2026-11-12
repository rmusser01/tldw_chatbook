#!/usr/bin/env bash
# Run a python script against a harness profile with the app's isolated environment.
# Usage: seedrun.sh <profile-dir> <script.py> [args...]
#   <profile-dir> must be inside HARNESS_STATE (a master such as $HARNESS_STATE/golden, or a run
#   copy $HARNESS_STATE/runs/<socket>) and already hold a config.toml (mkprofile.sh / launch.sh).
# The code comes from APP_WT (PYTHONPATH), exactly as launch.sh runs it; the environment is
# cleared (env -i + allowlist), HOME / XDG_* / TLDW_CONFIG_PATH point into the profile, and
# HARNESS_STATE / APP_WT are passed so the script's harness_guard can refuse anything else.
# Never import tldw_chatbook.config outside this wrapper: config readers can write the real
# ~/.config/tldw_cli.
set -euo pipefail
. "$(dirname "$0")/env.sh"
[ $# -ge 2 ] || rp_die "usage: seedrun.sh <profile-dir> <script.py> [args...]"
P="$(rp_under_state "$1")"
[ -f "$P/config.toml" ] || rp_die "$P has no config.toml (run mkprofile.sh or launch.sh first)"
S="$2"; shift 2
[ -f "$S" ] || rp_die "no script $S"
S="$(cd "$(dirname "$S")" && pwd -P)/$(basename "$S")"
mkdir -p "$P/cwd"
rp_app_env "$P"
cd "$P/cwd"
if [ $# -gt 0 ]; then exec "${RP_ENV[@]}" "$PY" "$S" "$@"; else exec "${RP_ENV[@]}" "$PY" "$S"; fi
