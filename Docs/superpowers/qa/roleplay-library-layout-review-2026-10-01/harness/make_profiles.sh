#!/usr/bin/env bash
# Build the harness master profiles under HARNESS_STATE from scratch.
#
# Usage: make_profiles.sh [--volume] [--force]
#   empty/              skeleton + isolated config, never booted. Each launch copies it and the
#                       app's first boot creates the DBs and its 3 built-in characters.
#   golden/             skeleton -> first boot of the master itself (INPLACE launch, waits for the
#                       nav bar, quits with Ctrl+Q) -> golden.preseed.bak/ snapshot -> seed.py.
#   golden.preseed.bak/ golden after first boot, before seeding (re-seed from a copy of it).
#   volume/   --volume: copy of golden -> fresh config -> scale_seed.py -> lore_fix.py
#                       (~320 characters, 60 personas, 43 lore books, 25 dictionaries).
# Existing masters are refused unless --force (which deletes and rebuilds them).
# Masters get NO mock-LLM fragment; launch.sh adds it to each run copy.
# Env: HARNESS_STATE, APP_WT, PY (env.sh); BOOT_TIMEOUT (seconds, default 240).
set -euo pipefail
. "$(dirname "$0")/env.sh"

VOLUME=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --volume) VOLUME=1 ;;
    --force) FORCE=1 ;;
    *) rp_die "usage: make_profiles.sh [--volume] [--force]" ;;
  esac
done
BOOT_TIMEOUT="${BOOT_TIMEOUT:-240}"
SOCK="rpmk-$$"
cleanup() {
  tmux -L "$SOCK" kill-server 2>/dev/null || true
  # kill-server leaves the socket file behind; remove ours once no server answers on it
  tmux -L "$SOCK" has-session 2>/dev/null || rm -f "${TMUX_TMPDIR:-/tmp}/tmux-$(id -u)/$SOCK"
}
trap cleanup EXIT

fresh_master() { # fresh_master <name> : refuse or wipe, then print its path
  local d="$HARNESS_STATE/$1"
  if [ -e "$d" ]; then
    [ "$FORCE" = "1" ] || rp_die "$d exists (use --force to rebuild it)"
    rm -rf "$d"
  fi
  rp_under_state "$d"
}

wait_nav() { # wait_nav <socket> : nav bar + footer rendered (cold start is 10-60 s)
  "$PY" -B "$HARNESS_DIR/waitfor.py" "$1" "4 Roleplay" "$BOOT_TIMEOUT" >/dev/null \
    && "$PY" -B "$HARNESS_DIR/waitfor.py" "$1" "Ctrl+Q" 30 >/dev/null
}

quit_app() { # quit_app <socket> : Ctrl+Q, wait for the app process to exit, then kill the server
  local pid i
  pid="$(tmux -L "$1" list-panes -a -F '#{pane_pid}' 2>/dev/null | head -1 || true)"
  tmux -L "$1" send-keys C-q 2>/dev/null || true
  for i in $(seq 1 600); do   # Ctrl+Q shutdown can take ~20 s
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  tmux -L "$1" kill-server 2>/dev/null || true
  if [ -n "$pid" ]; then for i in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done; fi
}

echo "HARNESS_STATE=$HARNESS_STATE"
echo "APP_WT=$APP_WT"

# --- empty
E="$(fresh_master empty)"
EXTRA_TOML="" "$HARNESS_DIR/mkprofile.sh" "$E" >/dev/null
echo "empty:  $E"

# --- golden: skeleton -> first boot -> preseed snapshot -> seed
G="$(fresh_master golden)"
if [ -e "$HARNESS_STATE/golden.preseed.bak" ]; then rm -rf "$HARNESS_STATE/golden.preseed.bak"; fi
EXTRA_TOML="" "$HARNESS_DIR/mkprofile.sh" "$G" >/dev/null
INPLACE=1 "$HARNESS_DIR/launch.sh" "$SOCK" 160 45 golden >/dev/null
if ! wait_nav "$SOCK"; then
  tmux -L "$SOCK" capture-pane -p >&2 || true
  rp_die "golden first boot: nav bar not seen within ${BOOT_TIMEOUT}s"
fi
sleep 5   # let deferred start-up work (DB creation, built-in characters) settle
quit_app "$SOCK"
[ -f "$G/data/rp_review/tldw_chatbook_ChaChaNotes.db" ] || rp_die "golden first boot created no ChaChaNotes DB"
cp -Rp "$G" "$HARNESS_STATE/golden.preseed.bak"
echo "golden.preseed.bak: $HARNESS_STATE/golden.preseed.bak"
SEED_OUT="$HARNESS_STATE/golden.seed.log"
"$HARNESS_DIR/seedrun.sh" "$G" "$HARNESS_DIR/seed.py" > "$SEED_OUT" 2>&1 || { cat "$SEED_OUT" >&2; rp_die "seed.py failed"; }
grep -q '^ERRORS: \[\]$' "$SEED_OUT" || { cat "$SEED_OUT" >&2; rp_die "seed.py reported errors (see above)"; }
echo "golden: $G (seed log: $SEED_OUT)"

# --- volume (optional)
if [ "$VOLUME" = "1" ]; then
  V="$(fresh_master volume)"
  cp -Rp "$G/data" "$V/data"
  cp -Rp "$G/home" "$V/home"
  rm -rf "$V/home/.config/tldw_cli/recovery-bootstrap"
  EXTRA_TOML="" "$HARNESS_DIR/mkprofile.sh" "$V" >/dev/null   # paths must point at volume, not golden
  VOL_OUT="$HARNESS_STATE/volume.seed.log"
  { "$HARNESS_DIR/seedrun.sh" "$V" "$HARNESS_DIR/scale_seed.py" \
      && "$HARNESS_DIR/seedrun.sh" "$V" "$HARNESS_DIR/lore_fix.py"; } > "$VOL_OUT" 2>&1 \
    || { cat "$VOL_OUT" >&2; rp_die "volume seeding failed"; }
  # scale_seed.py's lore step stops at its first duplicate generated book name (a ConflictError
  # listed under ERRORS); lore_fix.py then tops the books up to 43. That is the gap-round dataset,
  # so verify the result instead of the error list.
  COUNTS="$("$PY" -B -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); q=lambda s: c.execute(s).fetchone()[0]; print(q("select count(*) from character_cards where deleted=0"), q("select count(*) from world_books where deleted=0"), q("select count(*) from chat_dictionaries where deleted=0"))' "$V/data/rp_review/tldw_chatbook_ChaChaNotes.db")"
  set -- $COUNTS
  [ "$1" -ge 348 ] && [ "$2" -ge 43 ] && [ "$3" -ge 28 ] || rp_die "volume counts off: characters=$1 books=$2 dictionaries=$3 (see $VOL_OUT)"
  echo "volume: $V (characters=$1 lore books=$2 dictionaries=$3; seed log: $VOL_OUT)"
fi
rm -rf "$HARNESS_STATE/corpus_tmp"   # metrics.py rebuilds its corpus copy from the new golden
echo "done"
