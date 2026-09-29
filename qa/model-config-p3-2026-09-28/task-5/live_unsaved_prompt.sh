#!/bin/bash
# TASK-33003.5 live check: open Chat settings on a scratch profile, capture the
# clean Esc hint, edit Temperature, capture the "asks" hint, press Esc (prompt),
# Esc again (keep editing), Cancel (prompt again), then d (discard, closed).
# Usage: live_unsaved_prompt.sh <cols> <rows>
# W=<tree> TAG=before- runs another checkout (e.g. the base) into the same dir.
set -u
Q=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p3/qa/model-config-p3-2026-09-28/task-5
W=${W:-${Q%/qa/*}}; TAG=${TAG:-}
L=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/73fb7a69-fb3c-49ea-81ff-f711a75d6336/scratchpad/t5/live
S=t33003p5
cols=$1; rows=$2; P=$L/prof
rm -rf $P; mkdir -p $P/home $P/xdg-data $P/xdg-config $P/xdg-cache
cat > $P/config.toml <<TOML
[general]
users_name = "t33003p5_verify"
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
clickat() { local p; p=$(pos "$1"); [ -n "$p" ] || return 1; click $(( ${p#* } + ${2:-2} )) ${p% *}; }
modal_open() { cap | grep -q "Model and generation"; }
for i in $(seq 1 60); do cap 2>/dev/null | grep -q "Ready — type a message" && break; sleep 1; done
sleep 6
for attempt in 1 2 3 4 5 6; do
  clickat "   New tab     Settings" 16; sleep 4
  modal_open && break
done
modal_open || { echo "modal did not open"; cap > $L/fail-${cols}x${rows}.txt; exit 1; }
stable
# Temperature sits under the collapsed "Advanced generation" disclosure:
# expand it, wheel the body until the row is on screen, then click into its
# field (right of the 23-col label) so Esc starts from an edited field.
# The focused provider picker owns the first Escape (closes its list).
tmux -L $S send-keys Escape; sleep 1; stable
cap | grep -q "Model and generation" || { echo "first Escape closed the modal"; exit 1; }
for attempt in 1 2 3 4; do
  cap | grep -q "▼ Advanced generation" && break
  clickat "▶ Advanced generation" 4; sleep 1.5; stable
done
wheel() { tmux -L $S send-keys -H $(sgr $'\x1b'"[<$1;$2;$3M"); }
p=$(pos "Model and generation"); wcol=$(( ${p#* } + 4 )); wrow=$(( rows / 2 ))
for i in $(seq 1 30); do pos "  Temperature " >/dev/null && [ -n "$(pos '  Temperature ')" ] && break; wheel 65 $wcol $wrow; sleep 0.4; done
clickat "  Temperature " 27; sleep 1; stable; save clean
echo "clean hint: $(cap | grep -o 'Esc close[^ ]*\( (asks: [0-9]* unsaved)\)\?' | head -1)"
tmux -L $S send-keys End; for i in 1 2 3 4 5 6 7 8; do tmux -L $S send-keys BSpace; done
tmux -L $S send-keys -l "0.9"; sleep 1.5; stable; save edited
echo "edited hint: $(cap | grep -o 'Esc close (asks: [0-9]* unsaved)' | head -1)"
tmux -L $S send-keys Escape; sleep 1.5; stable; save prompt
echo "after Esc: prompt=$(cap | grep -c 'unsaved edit') modal=$(cap | grep -c 'Model and generation')"
tmux -L $S send-keys Escape; sleep 1.5; stable; save keep-editing
echo "after 2nd Esc (keep editing): prompt=$(cap | grep -c 'unsaved edit to this chat') modal=$(cap | grep -c 'Model and generation')"
clickat "Cancel" 2; sleep 1.5; stable; save prompt-from-cancel
echo "after Cancel: prompt=$(cap | grep -c 'unsaved edit to this chat')"
tmux -L $S send-keys d; sleep 2; stable; save discarded
echo "after d: modal=$(cap | grep -c 'Model and generation')"
tmux -L $S kill-server 2>/dev/null
