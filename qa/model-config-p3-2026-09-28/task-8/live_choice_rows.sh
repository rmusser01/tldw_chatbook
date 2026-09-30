#!/bin/bash
# TASK-33003.8 live check: open Chat settings on a scratch profile, expand
# Advanced generation, wheel the body until the provider-choice rows show,
# capture them at rest, then pick the first option of the first choice row
# with real keys (Enter, Down, Enter) and capture again.
# Usage: live_choice_rows.sh <cols> <rows> <chat: llama|custom>
# W=<tree> TAG=before- runs another checkout (e.g. the base) into the same dir.
set -u
Q=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p3/qa/model-config-p3-2026-09-28/task-8
W=${W:-${Q%/qa/*}}; TAG=${TAG:-}
L=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/73fb7a69-fb3c-49ea-81ff-f711a75d6336/scratchpad/t8/live
S=t33003p8
cols=$1; rows=$2; chat=$3; P=$L/prof
rm -rf $P; mkdir -p $P/home $P/xdg-data $P/xdg-config $P/xdg-cache
if [ "$chat" = custom ]; then
  provider='custom-ep:gpu-box'; first_row='Reasoning effort'
  extra='[custom_endpoints.gpu-box]
display_name = "GPU box"
family = "llama_cpp"
base_url = "http://127.0.0.1:9099"
models = ["model-a"]'
else
  provider='llama_cpp'; first_row='Reasoning effort'; extra=''
fi
cat > $P/config.toml <<TOML
[general]
users_name = "t33003p8_verify"
default_tab = "chat"

[first_run]
setup_started = true
setup_completed = true

[model_catalog]
auto_refresh_enabled = false

[chat_defaults]
provider = "$provider"
model = "model-a"

[api_settings.llama_cpp]
api_url = "http://127.0.0.1:9099"
model = "model-a"

$extra
TOML
tmux -L $S kill-server 2>/dev/null || true
tmux -L $S new-session -d -x $cols -y $rows -c $W \
  "env -i PATH=/usr/bin:/bin TERM=xterm-256color COLORTERM=truecolor LANG=en_US.UTF-8 \
   HOME=$P/home XDG_DATA_HOME=$P/xdg-data XDG_CONFIG_HOME=$P/xdg-config XDG_CACHE_HOME=$P/xdg-cache \
   TLDW_CONFIG_PATH=$P/config.toml PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
   PYTHONPATH=$W /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.app; sleep 600"
cap() { tmux -L $S capture-pane -p; }
save() { cap > $Q/live-$TAG$chat-$1-${cols}x${rows}.txt; tmux -L $S capture-pane -p -e > $Q/live-$TAG$chat-$1-${cols}x${rows}.ansi.txt; }
stable() { local a b i; a=""; for i in $(seq 1 20); do b=$(cap); [ "$a" = "$b" ] && return 0; a=$b; sleep 0.7; done; }
pos() { cap | /usr/bin/python3 -c 'import sys
n=sys.argv[1]
for r,l in enumerate(sys.stdin.read().split("\n"),1):
    i=l.find(n)
    if i>=0: print(r, i+1); break' "$1"; }
sgr() { printf '%s' "$1" | xxd -p | fold -w2 | tr '\n' ' '; }
click() { tmux -L $S send-keys -H $(sgr $'\x1b'"[<0;$1;$2M"); sleep 0.15; tmux -L $S send-keys -H $(sgr $'\x1b'"[<0;$1;$2m"); }
clickat() { local p; p=$(pos "$1"); [ -n "$p" ] || return 1; click $(( ${p#* } + ${2:-2} )) ${p% *}; }
wheel() { tmux -L $S send-keys -H $(sgr $'\x1b'"[<$1;$2;$3M"); }
modal_open() { cap | grep -q "Model and generation"; }
for i in $(seq 1 60); do cap 2>/dev/null | grep -q "Ready — type a message\|type a message" && break; sleep 1; done
sleep 6
for attempt in 1 2 3 4 5 6; do
  clickat "   New tab     Settings" 16; sleep 4
  modal_open && break
done
modal_open || { echo "modal did not open"; cap > $L/fail-$chat-${cols}x${rows}.txt; exit 1; }
stable
# The focused provider picker owns the first Escape (closes its list).
tmux -L $S send-keys Escape; sleep 1; stable
modal_open || { echo "first Escape closed the modal"; exit 1; }
for attempt in 1 2 3 4; do
  cap | grep -q "▼ Advanced generation" && break
  clickat "▶ Advanced generation" 4; sleep 1.5; stable
done
p=$(pos "Model and generation"); wcol=$(( ${p#* } + 4 )); wrow=$(( rows / 2 ))
# Wheel until the last shown choice row (Thinking budget follows them) is on screen.
for i in $(seq 1 30); do [ -n "$(pos '  Thinking budget ')" ] && break; [ -n "$(pos '▶ Conversation identity')" ] && break; wheel 65 $wcol $wrow; sleep 0.4; done
for i in 1 2 3; do wheel 65 $wcol $wrow; sleep 0.4; done
stable; save rest
echo "rest rows:"; cap | grep -E "Reasoning effort|Reasoning summary|Verbosity|Thinking " | sed 's/  */ /g'
# Click the first choice row's Select (right of the 23-col label): the click
# focuses it and opens its list with nothing highlighted, so Down lands on
# the blank prompt and a second Down on the first choice; Enter picks it.
clickat "  $first_row " 27; sleep 1; stable; save open
tmux -L $S send-keys Down; sleep 1; tmux -L $S send-keys Down; sleep 1; stable; save highlight
tmux -L $S send-keys Enter; sleep 1.5; stable; save chosen
echo "chosen rows:"; cap | grep -E "Reasoning effort|Reasoning summary|Verbosity|Thinking " | sed 's/  */ /g'
tmux -L $S kill-server 2>/dev/null
