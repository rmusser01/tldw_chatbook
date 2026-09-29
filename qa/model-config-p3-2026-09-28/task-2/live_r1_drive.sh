#!/bin/bash
# Drive one theme/size: open Chat settings, expand Advanced generation, focus
# Temperature with real keys, type 0.42, capture plain + -e.
# Usage: drive.sh <theme> <cols> <rows>
L=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/73fb7a69-fb3c-49ea-81ff-f711a75d6336/scratchpad/live-r1
Q=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p3/qa/model-config-p3-2026-09-28/task-2
theme=$1; cols=$2; rows=$3
T="tmux -L t33003p2r1"
cap() { tmux -L t33003p2r1 capture-pane -p; }
$L/launch.sh $theme $cols $rows
for i in $(seq 1 60); do cap 2>/dev/null | grep -q "Ready — type a message" && break; sleep 1; done
sleep 6
# control-bar Settings button: row/col from the capture
pos=$(cap | awk '{ i = index($0, "   New tab     Settings"); if (i) { print NR, i + 15; exit } }')
row=${pos% *}; col=${pos#* }
for attempt in 1 2 3 4 5 6; do
  $L/click.sh $((col + 1)) $row; sleep 4
  cap | grep -q "Conversation settings" && break
done
cap | grep -q "Conversation settings" || { echo "modal did not open ($theme)"; exit 1; }
for k in 1 2 3 4 5 6 7 8; do tmux -L t33003p2r1 send-keys Tab; sleep 0.6; done
stable() { a=""; for i in $(seq 1 20); do b=$(cap); [ "$a" = "$b" ] && return 0; a=$b; sleep 1; done; }
for attempt in 1 2 3 4; do
  stable
  apos=$(cap | python3 -c 'import sys
for n, line in enumerate(sys.stdin.read().split("\n"), 1):
    i = line.find("▶ Advanced generation")
    if i >= 0:
        print(n, i + 1); break')
  [ -n "$apos" ] || break
  $L/click.sh $(( ${apos#* } + 2 )) ${apos% *}; sleep 2
  cap | grep -q "▼ Advanced generation" && break
done
cap | grep -q "▼ Advanced generation" || { echo "advanced not expanded ($theme)"; cap > $L/fail-$theme-${cols}x${rows}.txt; exit 1; }
stable
tmux -L t33003p2r1 send-keys Tab; sleep 0.8
tmux -L t33003p2r1 send-keys End BSpace BSpace BSpace; tmux -L t33003p2r1 send-keys -l "0.42"; sleep 1.5
out=$Q/live-r1-$theme-temperature-focused-${cols}x${rows}
cap > $out.txt
tmux -L t33003p2r1 capture-pane -p -e > $out.ansi.txt
grep -q "Temperature *█ 0.42" $out.txt && echo "ok $theme ${cols}x${rows}" || echo "FOCUS NOT CONFIRMED $theme ${cols}x${rows}"
tmux -L t33003p2r1 kill-server 2>/dev/null
