#!/usr/bin/env bash
# Write an isolated tldw_chatbook profile skeleton + a FRESH config.toml at <dir> (overwrites
# an existing config). Usage: mkprofile.sh <profile-dir>
# Layout: <dir>/{home,xdg_config,xdg_data,xdg_cache,xdg_state,data,cwd}, <dir>/config.toml.
# Every DB path and [paths].data_dir point inside <dir>; users_name = rp_review
# => user data dir = <dir>/data/rp_review/. Content: see profile_config.py (init).
# EXTRA_TOML=<file> appends a TOML fragment (e.g. mock_llm.toml); empty/unset = none.
# <dir> must be inside HARNESS_STATE. To keep an existing config, use `profile_config.py repath`.
set -euo pipefail
. "$(dirname "$0")/env.sh"
[ $# -eq 1 ] || rp_die "usage: mkprofile.sh <profile-dir>"
P="$(rp_under_state "$1")"
if [ -n "${EXTRA_TOML:-}" ]; then
  [ -f "$EXTRA_TOML" ] || rp_die "EXTRA_TOML=$EXTRA_TOML does not exist"
  "$PY" -B "$HARNESS_DIR/profile_config.py" init "$P" --extra "$EXTRA_TOML"
else
  "$PY" -B "$HARNESS_DIR/profile_config.py" init "$P"
fi
