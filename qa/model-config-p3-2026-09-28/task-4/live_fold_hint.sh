#!/bin/bash
# TASK-33003.4 live check: open Chat settings on a scratch profile (the focused
# provider picker's results make the body overflow), then wheel the body down
# until the painted content stops moving (bottom), one notch back up, and down
# to the bottom again, capturing each state. Usage: live_fold_hint.sh <cols> <rows>
set -u
# W=<tree> TAG=before- runs another checkout (e.g. the base) into the same dir.
Q=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p3/qa/model-config-p3-2026-09-28/task-4
W=${W:-${Q%/qa/*}}; TAG=${TAG:-}
L=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/73fb7a69-fb3c-49ea-81ff-f711a75d6336/scratchpad/t4/live
S=t33003p4
cols=$1; rows=$2; P=$L/prof
rm -rf $P; mkdir -p $P/home $P/xdg-data $P/xdg-config $P/xdg-cache
cat > $P/config.toml <<TOML
[general]
users_name = "t33003p4_verify"
default_tab = "chat"

[first_run]
setup_started = true
setup_completed = true

[model_catalog]
auto_refresh_enabled = false

[chat_defaults]
provider = "llama_cpp"
model = "model-a"

[api_settings.llama_cpp]
api_url = "http://127.0.0.1:9099"
model = "model-a"
TOML
tmux -L $S kill-server 2>/dev/null || true
tmux -L $S new-session -d -x $cols -y $rows -c $W \
  "env -i PATH=/usr/bin:/bin TERM=xterm-256color COLORTERM=truecolor LANG=en_US.UTF-8 \
   HOME=$P/home XDG_DATA_HOME=$P/xdg-data XDG_CONFIG_HOME=$P/xdg-config XDG_CACHE_HOME=$P/xdg-cache \
   TLDW_CONFIG_PATH=$P/config.toml PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
   PYTHONPATH=$W /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.app; sleep 600"
cap() { tmux -L $S capture-pane -p; }
save() { cap > $Q/live-$TAG$1-${cols}x${rows}.txt; tmux -L $S capture-pane -p -e > $Q/live-$TAG$1-${cols}x${rows}.ansi.txt; }
stable() { local a b i; a=""; for i in $(seq 1 20); do b=$(cap); [ "$a" = "$b" ] && return 0; a=$b; sleep 0.7; done; }
pos() { cap | /usr/bin/python3 -c 'import sys
n=sys.argv[1]
for r,l in enumerate(sys.stdin.read().split("\n"),1):
    i=l.find(n)
    if i>=0: print(r, i+1); break' "$1"; }
sgr() { printf '%s' "$1" | xxd -p | fold -w2 | tr '\n' ' '; }
click() { tmux -L $S send-keys -H $(sgr $'\x1b'"[<0;$1;$2M"); sleep 0.15; tmux -L $S send-keys -H $(sgr $'\x1b'"[<0;$1;$2m"); }
wheel() { tmux -L $S send-keys -H $(sgr $'\x1b'"[<$1;$2;$3M"); }
clickat() { local p; p=$(pos "$1"); [ -n "$p" ] || return 1; click $(( ${p#* } + ${2:-2} )) ${p% *}; }
for i in $(seq 1 60); do cap 2>/dev/null | grep -q "Ready — type a message" && break; sleep 1; done
sleep 6
for attempt in 1 2 3 4 5 6; do
  clickat "   New tab     Settings" 16; sleep 4
  cap | grep -q "Conversation settings" && break
done
cap | grep -q "Conversation settings" || { echo "modal did not open"; cap > $L/fail-${cols}x${rows}.txt; exit 1; }
stable; save open
# Wheel over a mid-body row, just right of the view tabs' left edge.
p=$(pos "Model and generation"); col=$(( ${p#* } + 4 )); row=$(( rows / 2 ))
prev=""; steps=0
for i in $(seq 1 80); do
  cur=$(cap); [ "$cur" = "$prev" ] && break
  echo "$cur" | grep -q "▼ more — scroll" && echo "step $i: hint shown" || echo "step $i: hint hidden"
  prev=$cur; wheel 65 $col $row; steps=$i; sleep 0.5
done
stable; save bottom
echo "bottom after $steps wheel steps; hint: $(cap | grep -c '▼ more — scroll')"
wheel 64 $col $row; sleep 0.8; stable; save one-up
echo "one notch up; hint: $(cap | grep -c '▼ more — scroll')"
prev=""
for i in $(seq 1 10); do
  cur=$(cap); [ "$cur" = "$prev" ] && break
  prev=$cur; wheel 65 $col $row; sleep 0.5
done
stable; save bottom-again
echo "back at the bottom; hint: $(cap | grep -c '▼ more — scroll')"
tmux -L $S kill-server 2>/dev/null
