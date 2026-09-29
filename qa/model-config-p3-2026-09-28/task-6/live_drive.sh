#!/bin/bash
# Drive one tree/theme/size through the TASK-33003.6 AC#7 states; writes
# <out>/<tree>-<theme>-<cols>x<rows>-<state>.{txt,ansi.txt}
# Usage: drive.sh <tree> <theme> <cols> <rows> <outdir>
L=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/73fb7a69-fb3c-49ea-81ff-f711a75d6336/scratchpad/t6/live
tree=$1; theme=$2; cols=$3; rows=$4; O=$5; mkdir -p $O
T="tmux -L t33003p6"
cap() { $T capture-pane -p; }
snap() { cap > $O/$tree-$theme-${cols}x${rows}-$1.txt; $T capture-pane -p -e > $O/$tree-$theme-${cols}x${rows}-$1.ansi.txt; }
stable() { a=""; for i in $(seq 1 20); do b=$(cap); [ "$a" = "$b" ] && return 0; a=$b; sleep 0.7; done; }
underlined() {  # exit 0 when the last row holding $1 shows it underlined (focused)
  $T capture-pane -p -e > $L/_u.ansi.txt
  (cd $L && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c '
import sys
from cells import rows
R = rows("_u.ansi.txt"); T = ["".join(c[0] for c in r) for r in R]
ys = [i for i, t in enumerate(T) if sys.argv[1] in t]
sys.exit(0 if ys and "u" in R[ys[-1]][T[ys[-1]].index(sys.argv[1])][3] else 1)' "$1")
}
press_until() { for i in $(seq 1 $3); do $T send-keys $1; sleep 0.5; underlined "$2" && return 0; done; echo "FAIL: $2 never focused"; return 1; }
$L/launch.sh $tree $theme $cols $rows
for i in $(seq 1 60); do cap 2>/dev/null | grep -q "Ready — type a message" && break; sleep 1; done
sleep 5
pos=$(cap | awk '{ i = index($0, "   New tab     Settings"); if (i) { print NR, i + 15; exit } }')
for a in 1 2 3 4 5; do $L/click.sh $(( ${pos#* } + 1 )) ${pos% *}; sleep 3; cap | grep -q "Conversation settings" && break; done
cap | grep -q "Conversation settings" || { echo "modal did not open"; exit 1; }
stable; snap modal-open
pos=$(cap | python3 -c 'import sys
for n,l in enumerate(sys.stdin.read().split("\n"),1):
    i=l.find("Context and memory")
    if i>=0: print(n,i+3); break')
$L/click.sh ${pos#* } ${pos% *}; sleep 2; stable
cap | grep -q "Budget strategy        █" || echo "WARN budget select not focused"
$T send-keys Enter; sleep 1.5; stable; snap modal-select
$T send-keys Escape; sleep 1; stable; snap modal-button-rest
press_until BTab "Use for this conversation" 6 && { stable; snap modal-button-focus; }
$T send-keys Escape; sleep 1.5
cap | grep -q "Conversation settings" && { echo "modal still open"; }
$T send-keys F4; sleep 5; stable; snap rail-rest
$T send-keys Tab; sleep 0.8; stable; snap rail-active-focus
$T send-keys Tab; sleep 0.8; stable; snap rail-inactive-focus
$T send-keys Enter; sleep 4; stable; snap pm-rest
$T send-keys F6; sleep 0.8
press_until Tab "Test Provider" 25 && { stable; snap pm-button-focus; }
$T kill-server 2>/dev/null
echo "done $tree $theme ${cols}x${rows}"
