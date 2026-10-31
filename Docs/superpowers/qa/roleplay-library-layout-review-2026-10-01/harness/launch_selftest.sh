#!/usr/bin/env bash
# launch.sh --self-test : prove the harness is isolated and that a REUSE=1 restart keeps the
# run's config (the B7 "restart: preferred state restored" check depends on it).
#
# 1. builds a temporary state dir $HARNESS_STATE/selftest.XXXXXX with an `empty` master and a
#    mock-LLM fragment on a random local port (starts mock_llm.py there);
# 2. boot 1: fresh run in tmux on a unique socket -> nav bar -> Ctrl+Q;
# 3. writes a sentinel key into [roleplay.reader] and moves [paths].data_dir elsewhere;
# 4. boot 2: REUSE=1 -> asserts the sentinel survived launch.sh AND the app's own boot and exit,
#    and the path key was rewritten back into the run; nav bar; one pane capture; a Console send
#    answered by the mock (the placeholder key is accepted); Ctrl+Q; kills the tmux server;
# 5. asserts the real profile (~/.config/tldw_cli, ~/.local/share/tldw_cli) is byte/mtime
#    unchanged and no tldw_chatbook login-keychain item appeared or changed (isolation_snap.py
#    before/after), the app log never names the real profile, and the app code came from APP_WT.
# Prints one PASS/FAIL line per check and a final "SELF-TEST: PASS|FAIL"; exit 0 only on PASS.
# The temporary state dir is removed on PASS (KEEP=1 keeps it) and kept on FAIL.
# Env: HARNESS_STATE, APP_WT, PY (env.sh); BOOT_TIMEOUT (default 240 s).
set -uo pipefail
. "$(dirname "$0")/env.sh"

BOOT_TIMEOUT="${BOOT_TIMEOUT:-240}"
ST="$(mktemp -d "$HARNESS_STATE/selftest.XXXXXX")" || rp_die "cannot create a temporary state dir"
ST="$(cd "$ST" && pwd -P)"; chmod 700 "$ST"
export HARNESS_STATE="$ST"           # every sub-script below re-validates and uses the temp state
SOCK="rpst-$$-$RANDOM"
PORT=$((20000 + RANDOM % 20000))
RUN="$ST/runs/$SOCK"
FAILS=0
MOCK_PID=""
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }
cleanup() {
  tmux -L "$SOCK" kill-server 2>/dev/null || true
  # kill-server leaves the socket file behind; remove ours once no server answers on it
  tmux -L "$SOCK" has-session 2>/dev/null || rm -f "${TMUX_TMPDIR:-/tmp}/tmux-$(id -u)/$SOCK"
  if [ -n "$MOCK_PID" ]; then kill "$MOCK_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
pane() { tmux -L "$SOCK" capture-pane -p 2>/dev/null; }
wait_text() { "$PY" -B "$HARNESS_DIR/waitfor.py" "$SOCK" "$1" "$2" >/dev/null 2>&1; }
wait_nav() { wait_text "4 Roleplay" "$BOOT_TIMEOUT" && wait_text "Ctrl+Q" 30; }
cfg_get() { "$PY" -B "$HARNESS_DIR/profile_config.py" get "$RUN/config.toml" "$1" 2>/dev/null; }
quit_app() { # Ctrl+Q, wait (<= 60 s) for the app process to exit; returns 1 if it had to be killed
  local pid i rc=0
  pid="$(tmux -L "$SOCK" list-panes -a -F '#{pane_pid}' 2>/dev/null | head -1)"
  tmux -L "$SOCK" send-keys C-q 2>/dev/null || true
  for i in $(seq 1 600); do
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then rc=1; fi
  tmux -L "$SOCK" kill-server 2>/dev/null || true
  if [ -n "$pid" ]; then for i in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done; fi
  return $rc
}

echo "self-test: state=$ST socket=$SOCK app=$APP_WT py=$PY"
"$PY" -B "$HARNESS_DIR/isolation_snap.py" > "$ST/real-profile.before.txt"

# 1. temporary state: empty master + mock fragment + mock server
if EXTRA_TOML="" "$HARNESS_DIR/mkprofile.sh" "$ST/empty" >/dev/null; then
  pass "temporary state dir built ($ST, master 'empty')"
else
  fail "could not build the empty master"
fi
sed "s|127.0.0.1:18765|127.0.0.1:$PORT|" "$HARNESS_DIR/mock_llm.toml" > "$ST/mock_llm.toml"
"$PY" -B "$HARNESS_DIR/mock_llm.py" "$PORT" > "$ST/mock_llm.out" 2>&1 &
MOCK_PID=$!
disown "$MOCK_PID" 2>/dev/null || true   # no "Terminated" job notice when cleanup kills it
for _ in $(seq 1 50); do
  "$PY" -B -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=1)" 2>/dev/null && break
  sleep 0.1
done

# 2. boot 1 (fresh run)
if EXTRA_TOML="$ST/mock_llm.toml" "$HARNESS_DIR/launch.sh" "$SOCK" 120 36 empty >/dev/null && wait_nav; then
  pass "boot 1 (fresh run): nav bar rendered"
else
  fail "boot 1: nav bar not seen within ${BOOT_TIMEOUT}s"; pane | head -20
fi
if quit_app; then pass "boot 1: app exited on Ctrl+Q"; else fail "boot 1: app did not exit on Ctrl+Q (killed)"; fi

# 3. sentinel + displaced path key
TOKEN="rp-selftest-$(date +%s)-$RANDOM"
if [ -f "$RUN/config.toml" ] \
   && "$PY" -B "$HARNESS_DIR/profile_config.py" set "$RUN/config.toml" roleplay.reader harness_selftest_sentinel "$TOKEN" >/dev/null \
   && "$PY" -B "$HARNESS_DIR/profile_config.py" set "$RUN/config.toml" paths data_dir "$ST/displaced-data-dir" >/dev/null \
   && [ "$(cfg_get roleplay.reader.harness_selftest_sentinel)" = "$TOKEN" ]; then
  pass "sentinel written: [roleplay.reader] harness_selftest_sentinel = \"$TOKEN\""
else
  fail "could not write the sentinel into $RUN/config.toml"
fi

# 4. boot 2 (REUSE=1)
REUSE=1 "$HARNESS_DIR/launch.sh" "$SOCK" 120 36 empty >/dev/null || fail "REUSE=1 launch.sh exited non-zero"
if [ "$(cfg_get roleplay.reader.harness_selftest_sentinel)" = "$TOKEN" ]; then
  pass "REUSE=1: launch.sh kept the config (sentinel present before the app booted)"
else
  fail "REUSE=1: launch.sh regenerated the config (sentinel gone)"
fi
if [ "$(cfg_get paths.data_dir)" = "$RUN/data" ]; then
  pass "REUSE=1: path keys rewritten into the run ([paths].data_dir = $RUN/data)"
else
  fail "REUSE=1: [paths].data_dir = $(cfg_get paths.data_dir), expected $RUN/data"
fi
if wait_nav; then
  pass "boot 2 (REUSE=1): nav bar rendered"
else
  fail "boot 2: nav bar not seen within ${BOOT_TIMEOUT}s"; pane | head -20
fi
pane > "$ST/selftest-pane.txt"
if grep -q "4 Roleplay" "$ST/selftest-pane.txt"; then
  pass "pane captured: $(grep -m1 '4 Roleplay' "$ST/selftest-pane.txt" | sed 's/  */ /g' | cut -c1-100)"
else
  fail "pane capture has no nav bar"
fi
if grep -q "Ready" "$ST/selftest-pane.txt"; then
  pass "provider readiness reads Ready (placeholder key accepted by resolve_provider_api_key)"
else
  fail "provider readiness does not read Ready"
fi
# Focus can land on the composer a few seconds after the nav bar (seen on a REUSE boot that
# restores the previous chat): click it and wait for the focused cursor before typing, because
# keys typed outside a text box are hotkeys.
for _ in 1 2 3; do
  "$PY" -B "$HARNESS_DIR/where.py" "$SOCK" "Ask, command, or paste task" --click B >/dev/null 2>&1 || true
  if wait_text "▌Ask, command" 10; then break; fi
done
if wait_text "▌Ask, command" 1; then
  tmux -L "$SOCK" send-keys -l "harness self-test ping" 2>/dev/null
  wait_text "harness self-test ping" 10 || true
  tmux -L "$SOCK" send-keys Enter 2>/dev/null
fi
if wait_text "mock reply" 90; then
  pass "mock LLM answered a Console send ($(grep -c 'POST /v1/chat/completions' "$ST/mock/mock_llm.$PORT.requests.log" 2>/dev/null) POST, auth header present: $(grep -q 'auth=yes' "$ST/mock/mock_llm.$PORT.requests.log" 2>/dev/null && echo yes || echo no))"
else
  fail "no mock reply on screen within 90s"; pane | tail -15
fi
if [ "$(cfg_get roleplay.reader.harness_selftest_sentinel)" = "$TOKEN" ]; then
  pass "sentinel survived the app's boot (app config writes kept [roleplay.reader])"
else
  fail "sentinel lost after the app booted"
fi
if quit_app; then pass "boot 2: app exited on Ctrl+Q"; else fail "boot 2: app did not exit on Ctrl+Q (killed)"; fi
if tmux -L "$SOCK" has-session 2>/dev/null; then fail "tmux server $SOCK still running"; else pass "tmux server $SOCK killed"; fi
if [ "$(cfg_get roleplay.reader.harness_selftest_sentinel)" = "$TOKEN" ]; then
  pass "sentinel survived the app's exit"
else
  fail "sentinel lost after the app exited"
fi

# 5. isolation
"$PY" -B "$HARNESS_DIR/isolation_snap.py" > "$ST/real-profile.after.txt"
if diff -q "$ST/real-profile.before.txt" "$ST/real-profile.after.txt" >/dev/null; then
  pass "real profile byte/mtime-unchanged ($(grep -vc '^keychain' "$ST/real-profile.before.txt") entries under ~/.config/tldw_cli + ~/.local/share/tldw_cli; tldw_chatbook keychain items unchanged: $(grep -c '^keychain svce=' "$ST/real-profile.before.txt"))"
else
  fail "real profile changed:"; diff "$ST/real-profile.before.txt" "$ST/real-profile.after.txt" | head -20
fi
LOG="$RUN/data/rp_review/tldw_cli_app.log"
REAL_HOME="$("$PY" -B -c 'import os,pwd;print(pwd.getpwuid(os.getuid()).pw_dir)')"
if [ -f "$LOG" ]; then
  n="$(grep -c -e "$REAL_HOME/.config/tldw_cli" -e "$REAL_HOME/.local/share/tldw_cli" "$LOG" || true)"
  if [ "$n" = "0" ]; then pass "app log never names the real profile ($LOG)"; else fail "app log names the real profile $n times"; fi
else
  fail "no app log at $LOG"
fi
rp_app_env "$RUN"
ORIGIN="$(cd "$RUN/cwd" && "${RP_ENV[@]}" "$PY" -B -c 'import importlib.util as u; print(u.find_spec("tldw_chatbook").origin)')"
case "$ORIGIN" in
  "$APP_WT"/*) pass "app code imported from APP_WT ($ORIGIN)" ;;
  *) fail "app code imported from $ORIGIN, not APP_WT=$APP_WT" ;;
esac

if [ "$FAILS" -eq 0 ]; then
  echo "SELF-TEST: PASS"
  cleanup
  if [ "${KEEP:-0}" != "1" ]; then rm -rf "$ST"; else echo "kept: $ST"; fi
  exit 0
fi
echo "SELF-TEST: FAIL ($FAILS failed checks; state kept at $ST)"
exit 1
