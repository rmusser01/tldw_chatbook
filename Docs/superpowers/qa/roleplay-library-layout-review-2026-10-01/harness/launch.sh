#!/usr/bin/env bash
# Isolated live launch of tldw_chatbook in its own tmux server (one socket per run).
#
# Usage: launch.sh <socket> <cols> <rows> [master]      master: golden (default) | empty | volume | <dir name>
#        launch.sh --self-test                         see launch_selftest.sh; prints PASS/FAIL
#
# The master profile $HARNESS_STATE/<master> is COPIED to $HARNESS_STATE/runs/<socket>/ (fresh each
# launch), a fresh config.toml is written there with every path pointing into that copy, and the
# app runs with a cleared environment (env -i + allowlist: disposable HOME / XDG_* /
# TLDW_CONFIG_PATH, no inherited provider keys) and cwd = runs/<socket>/cwd, where trace exports
# land (never in a checkout).
#
# Env (see env.sh for HARNESS_STATE, APP_WT, PY, EXTRA_PYTHONPATH):
#   REUSE=1      keep an existing runs/<socket>/ (persistence across restarts). When its config.toml
#                exists it is NOT regenerated: only the path keys are rewritten (profile_config.py
#                repath), so sections added by the app or a test (e.g. [roleplay.reader]) survive.
#                EXTRA_TOML is not re-applied on REUSE.
#   INPLACE=1    boot the master profile itself (maintenance only: make_profiles.sh first boot).
#                Same config rule as REUSE (path keys only, if a config exists).
#   EXTRA_TOML=<file>  TOML fragment appended to a FRESH run config. Unset = harness mock_llm.toml
#                (local mock LLM on 127.0.0.1:18765); set to "" for none.
#
# NEVER add a stderr redirect to the app command: Textual draws the UI on stderr.
# Stop: tmux -L <socket> send-keys C-q ; tmux -L <socket> kill-server
set -euo pipefail
if [ "${1:-}" = "--self-test" ]; then shift; exec bash "$(dirname "$0")/launch_selftest.sh" "$@"; fi
. "$(dirname "$0")/env.sh"

[ $# -ge 3 ] && [ $# -le 4 ] || rp_die "usage: launch.sh <socket> <cols> <rows> [golden|empty|volume|<master>]"
SOCK="$1"; COLS="$2"; ROWS="$3"; KIND="${4:-golden}"
case "$SOCK" in *[!A-Za-z0-9._-]*|"") rp_die "socket name must be [A-Za-z0-9._-]+" ;; esac
case "$KIND" in *[!A-Za-z0-9._-]*|""|.|..|runs) rp_die "bad master name: $KIND" ;; esac
case "$COLS$ROWS" in *[!0-9]*) rp_die "cols/rows must be integers" ;; esac

SRC="$HARNESS_STATE/$KIND"
[ -d "$SRC" ] || rp_die "no master profile $SRC (build it with make_profiles.sh)"

# Stop any previous server on this socket and wait for its app process to exit, so nothing is
# still writing the run dir while we copy or rewrite it.
rp_stop_socket() {
  local pids p i
  pids="$(tmux -L "$1" list-panes -a -F '#{pane_pid}' 2>/dev/null || true)"
  tmux -L "$1" kill-server 2>/dev/null || true
  for p in $pids; do
    for i in $(seq 1 100); do kill -0 "$p" 2>/dev/null || break; sleep 0.1; done
  done
}
rp_stop_socket "$SOCK"

KEEP_CONFIG=0
if [ "${INPLACE:-0}" = "1" ]; then
  RUN="$SRC"
  [ -f "$RUN/config.toml" ] && KEEP_CONFIG=1
else
  REUSABLE=0; [ "${REUSE:-0}" = "1" ] && [ -d "$HARNESS_STATE/runs/$SOCK/data" ] && REUSABLE=1
  RUN="$(rp_under_state "$HARNESS_STATE/runs/$SOCK")"
  if [ "$REUSABLE" = "1" ]; then
    [ -f "$RUN/config.toml" ] && KEEP_CONFIG=1
  else
    rm -rf "$RUN"; mkdir -p "$RUN"
    # copy data + home (minus recovery-bootstrap enrollment, which is bound to the source location)
    if [ -d "$SRC/data" ]; then cp -Rp "$SRC/data" "$RUN/data"; fi
    if [ -d "$SRC/home" ]; then cp -Rp "$SRC/home" "$RUN/home"; fi
    rm -rf "$RUN/home/.config/tldw_cli/recovery-bootstrap"
  fi
fi

if [ "$KEEP_CONFIG" = "1" ]; then
  if [ -n "${EXTRA_TOML:-}" ]; then echo "launch.sh: EXTRA_TOML ignored (existing config kept)" >&2; fi
  "$PY" -B "$HARNESS_DIR/profile_config.py" repath "$RUN" >/dev/null
  mkdir -p "$RUN/cwd" "$RUN/home" "$RUN/xdg_config" "$RUN/xdg_data" "$RUN/xdg_cache" "$RUN/xdg_state"
else
  if [ -z "${EXTRA_TOML+x}" ] && [ -f "$HARNESS_DIR/mock_llm.toml" ]; then EXTRA_TOML="$HARNESS_DIR/mock_llm.toml"; fi
  EXTRA_TOML="${EXTRA_TOML:-}" "$HARNESS_DIR/mkprofile.sh" "$RUN" >/dev/null
fi

rp_app_env "$RUN"
# -f /dev/null: ignore any ~/.tmux.conf. The command is passed as argv (no shell, no quoting);
# -c sets the app's cwd. Do NOT redirect stderr.
tmux -f /dev/null -L "$SOCK" new-session -d -x "$COLS" -y "$ROWS" -c "$RUN/cwd" \
  "${RP_ENV[@]}" "$PY" -m tldw_chatbook.app
tmux -L "$SOCK" set-option -g window-size manual >/dev/null 2>&1 || true
tmux -L "$SOCK" resize-window -x "$COLS" -y "$ROWS" >/dev/null 2>&1 || true
echo "launched socket=$SOCK size=${COLS}x${ROWS} profile=$KIND run=$RUN app=$APP_WT log=$RUN/data/rp_review/tldw_cli_app.log"
