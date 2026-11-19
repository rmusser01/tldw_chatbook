#!/bin/bash
set -euo pipefail
export HARNESS_STATE=/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/prompt-mixins-pr-d18dfic_/live-current
export APP_WT=/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook
export PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
. "$APP_WT/Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/harness/env.sh"
"$PY" -B "$HARNESS_DIR/profile_config.py" init "$HARNESS_STATE/fresh" > "$HARNESS_STATE/profile-init.log"
rp_app_env "$HARNESS_STATE/fresh"
tmux -f /dev/null -L mixins-dev-port new-session -d -x 160 -y 45 -c "$HARNESS_STATE/fresh/cwd" "${RP_ENV[@]}" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/tldw-cli --focus
tmux -L mixins-dev-port set-option -g window-size manual
tmux -L mixins-dev-port list-panes -F '#{pane_pid}' > "$HARNESS_STATE/app-pid.txt"
