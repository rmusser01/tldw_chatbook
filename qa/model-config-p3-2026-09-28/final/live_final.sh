#!/bin/bash
# TASK-33003 parent AC#13, re-captured once at the rebased final head (final
# review M2): Chat settings Model view collapsed and expanded, an edit and the
# unsaved-edits prompt, then Settings rail focus, on a fully scratch profile.
# Usage: FINAL_SCRATCH=<scratch dir> PYTHON=<venv python> live_final.sh <cols> <rows>
# Writes live-<state>-<cols>x<rows>.{txt,ansi.txt} next to this script.
set -u
Q=$(cd "$(dirname "$0")" && pwd)
W=$(git -C "$Q" rev-parse --show-toplevel)
L=${FINAL_SCRATCH:?export FINAL_SCRATCH}/live-final
S=t33003final
cols=$1; rows=$2; P=$L/prof-${cols}x${rows}
rm -rf $P; mkdir -p $P/home $P/xdg-data $P/xdg-config $P/xdg-cache
cat > $P/config.toml <<TOML
[general]
users_name = "t33003final_verify"
default_tab = "chat"

[splash_screen]
enabled = false

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
T="tmux -L $S"
$T kill-server 2>/dev/null || true
$T new-session -d -x $cols -y $rows -c $W \
  "env -i PATH=/usr/bin:/bin TERM=xterm-256color COLORTERM=truecolor LANG=en_US.UTF-8 \
   HOME=$P/home XDG_DATA_HOME=$P/xdg-data XDG_CONFIG_HOME=$P/xdg-config XDG_CACHE_HOME=$P/xdg-cache \
   TLDW_CONFIG_PATH=$P/config.toml PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
   PYTHONPATH=$W ${PYTHON:?} -m tldw_chatbook.app; sleep 600"
cap() { $T capture-pane -p; }
save() { cap > $Q/live-$1-${cols}x${rows}.txt; $T capture-pane -p -e > $Q/live-$1-${cols}x${rows}.ansi.txt; }
stable() { local a b i; a=""; for i in $(seq 1 20); do b=$(cap); [ "$a" = "$b" ] && return 0; a=$b; sleep 0.7; done; }
pos() { cap | /usr/bin/python3 -c 'import sys
n=sys.argv[1]
for r,l in enumerate(sys.stdin.read().split("\n"),1):
    i=l.find(n)
    if i>=0: print(r, i+1); break' "$1"; }
sgr() { printf '%s' "$1" | xxd -p | fold -w2 | tr '\n' ' '; }
click() { $T send-keys -H $(sgr $'\x1b'"[<0;$1;$2M"); sleep 0.15; $T send-keys -H $(sgr $'\x1b'"[<0;$1;$2m"); }
clickat() { local p; p=$(pos "$1"); [ -n "$p" ] || return 1; click $(( ${p#* } + ${2:-2} )) ${p% *}; }
wheel() { $T send-keys -H $(sgr $'\x1b'"[<$1;$2;$3M"); }
modal_open() { cap | grep -q "Model and generation"; }
for i in $(seq 1 60); do cap 2>/dev/null | grep -q "Ready — type a message" && break; sleep 1; done
sleep 4
for attempt in 1 2 3 4 5 6; do
  clickat "   New tab     Settings" 16; sleep 3
  modal_open && break
done
modal_open || { echo "modal did not open"; cap > $L/fail-${cols}x${rows}.txt; exit 1; }
stable
# The focused provider picker owns the first Escape (closes its list).
$T send-keys Escape; sleep 1; stable
modal_open || { echo "first Escape closed the modal"; exit 1; }
save model-collapsed
echo "collapsed hint: $(cap | grep -o 'Esc close[^│]*' | head -1 | sed 's/ *$//')"
for attempt in 1 2 3 4; do
  cap | grep -q "▼ Advanced generation" && break
  clickat "▶ Advanced generation" 4; sleep 1.5; stable
done
save model-expanded
p=$(pos "Model and generation"); wcol=$(( ${p#* } + 4 )); wrow=$(( rows / 2 ))
for i in $(seq 1 30); do [ -n "$(pos '  Temperature ')" ] && break; wheel 65 $wcol $wrow; sleep 0.4; done
clickat "  Temperature " 27; sleep 1; stable
$T send-keys End; for i in 1 2 3 4 5 6 7 8; do $T send-keys BSpace; sleep 0.1; done
for k in 0 . 9; do $T send-keys -l "$k"; sleep 0.2; done; sleep 1.5; stable; save edited
echo "edited hint: $(cap | grep -o 'Esc close (asks: [^)]*)' | head -1)"
$T send-keys Escape; sleep 1.5; stable; save unsaved-prompt
echo "prompt: $(cap | grep -o '[0-9]* unsaved edits\? to this chat: [^.]*' | head -1)"
$T send-keys d; sleep 2; stable
modal_open && { echo "d did not close the modal"; exit 1; }
$T send-keys F4; sleep 5; stable; save rail-rest
$T send-keys Tab; sleep 0.8; stable; save rail-active-focus
$T send-keys Tab; sleep 0.8; stable; save rail-inactive-focus
$T kill-server 2>/dev/null
echo "done ${cols}x${rows}"
