#!/usr/bin/env bash
# TASK-15810 Task 1 — reproduce Library RAG Answer's first-query stall on demand.
#
# Builds a scratch profile the way the 15400/15700 live checks built theirs
# (36 real Docs/User_Guide pages seeded through the app's own `add_note` +
# `index_entries`; the embedding model already on disk; HF_HUB_OFFLINE=1 so a
# download is impossible), launches the real TUI in tmux, drives it to the
# FIRST RAG Answer query, and times it.
#
#   ./reproduce.sh <scratch-dir> [timeout-seconds]
#
# Requires: the worktree venv (uv venv .venv --python 3.12 + the pinned
# extras) and an on-disk copy of all-MiniLM-L6-v2 under the REAL profile's
# models/embeddings cache (see MODEL_CACHE below).
set -euo pipefail

SCRATCH="${1:?usage: reproduce.sh <scratch-dir> [timeout-seconds]}"
TIMEOUT="${2:-420}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="$REPO/.venv/bin/python"
SOCK="${TMUX_SOCK:-rag15810}"
USER_NAME="verify15810"
# The app's embedding cache is PROFILE-LOCAL (<data_dir>/<user>/models/embeddings),
# NOT $HF_HOME -- a scratch profile therefore starts with an EMPTY model cache and,
# under HF_HUB_OFFLINE=1, cannot load the model at all. Copy the real profile's
# cache in BEFORE first launch (same rule as `chromadb/` in lessons-live-verification).
MODEL_CACHE="${MODEL_CACHE:-$HOME/.local/share/tldw_cli/default_user/models/embeddings}"
# A run-gate blocks RAG Answer mode outright without a provider credential, so
# retrieval never starts. This key is deliberately NOT live: it opens the gate so
# the RETRIEVAL path runs; the answer step afterwards is expected to fail.
# NOTE: deliberately NOT an `sk-`-shaped literal -- committing one trips secret
# scanners and invites accidental reuse. The gate only needs a non-empty value.
FAKE_KEY="${OPENAI_API_KEY:-task15810-placeholder-not-a-credential}"

echo "== repo:    $REPO"
echo "== scratch: $SCRATCH"

rm -rf "$SCRATCH"
mkdir -p "$SCRATCH/home/.config/tldw_cli" "$SCRATCH/data/$USER_NAME/models"

REAL_CONFIG="$HOME/.config/tldw_cli/config.toml"
[ -f "$REAL_CONFIG" ] && shasum -a 256 "$REAL_CONFIG" > "$SCRATCH/real_config.sha256.before"

cat > "$SCRATCH/home/.config/tldw_cli/config.toml" <<EOF
[general]
users_name = "$USER_NAME"
default_tab = "chat"

[paths]
data_dir = "$SCRATCH/data"

[first_run]
setup_started = true
setup_completed = true

[splash_screen]
enabled = false

[embeddings]
default_model_id = "all-MiniLM-L6-v2"

[rag]
enabled = true
EOF
"$PY" -c "import tomllib,sys; tomllib.load(open(sys.argv[1],'rb')); print('scratch config parses OK')" \
  "$SCRATCH/home/.config/tldw_cli/config.toml"

cp -R "$MODEL_CACHE" "$SCRATCH/data/$USER_NAME/models/embeddings"

echo "== seeding 36 User Guide pages (add_note + index_entries)"
env -i PATH=/usr/bin:/bin TERM=xterm-256color \
  "$PY" "$(dirname "${BASH_SOURCE[0]}")/seed_profile.py" "$SCRATCH" "$REPO" \
  > "$SCRATCH/seed.log" 2>&1
grep -E "RESOLVED data_dir|pages found|Notes written|index_entries summary|vector store stats" "$SCRATCH/seed.log"

echo "== launching the TUI"
tmux -L "$SOCK" kill-server 2>/dev/null || true
tmux -L "$SOCK" new-session -d -x 235 -y 52 -c "$REPO" \
  "env HOME=$SCRATCH/home XDG_CONFIG_HOME=$SCRATCH/home/.config \
       XDG_DATA_HOME=$SCRATCH/home/.local/share XDG_CACHE_HOME=$SCRATCH/home/.cache \
       TLDW_CONFIG_PATH=$SCRATCH/home/.config/tldw_cli/config.toml \
       HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OPENAI_API_KEY=$FAKE_KEY \
       TERM=xterm-256color $PY -m tldw_chatbook.app"
sleep 20

# NOTE: `pgrep -f tldw_chatbook.app` -- and any plain substring match -- also
# hits the TMUX SERVER, whose own command line contains the whole launch string.
# Measuring that PID reports ~0.5% CPU and hides the app's ~98% spin entirely
# (this cost a full round on the first reproduction). Match the FIELDS instead:
# the app is the process whose argv is exactly `<...>/python -m tldw_chatbook.app`.
# `ps -eo` can exit nonzero on macOS when a process disappears mid-listing, which
# under `pipefail` would kill this script silently, hence the `|| true`.
# Matching on THIS worktree's interpreter path as well, so a concurrent session
# running the app from another checkout cannot be measured by mistake.
APP_PID="$(ps -eo pid,command 2>/dev/null \
  | awk -v py="$PY" '$2 == py && $3 == "-m" && $4 == "tldw_chatbook.app" {print $1; exit}' \
  || true)"
[ -n "$APP_PID" ] || { echo "FATAL: could not find the app process"; exit 1; }
echo "== app pid: $APP_PID (tmux server is a DIFFERENT pid)"

TMUX_SOCK="$SOCK" "$PY" "$(dirname "${BASH_SOURCE[0]}")/drive_tui.py" "$APP_PID" "$TIMEOUT"

echo "== teardown"
tmux -L "$SOCK" kill-server 2>/dev/null || true
if [ -f "$REAL_CONFIG" ]; then
  echo "real config BEFORE: $(cat "$SCRATCH/real_config.sha256.before")"
  echo "real config AFTER : $(shasum -a 256 "$REAL_CONFIG")"
fi
