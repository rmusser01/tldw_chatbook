#!/bin/bash
# Open Chat settings, expand Advanced generation and Conversation identity,
# then Tab through every field of both views capturing each focused state
# (the focus fill paints exactly the field's cells). Usage: drive.sh <cols> <rows>
L=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/73fb7a69-fb3c-49ea-81ff-f711a75d6336/scratchpad/t3/live
Q=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p3/qa/model-config-p3-2026-09-28/task-3
F=$L/focus-${1}x${2}; rm -rf $F; mkdir -p $F
cols=$1; rows=$2
PY=/usr/bin/python3
cap() { tmux -L t33003p3 capture-pane -p; }
save() { cap > $Q/live-$1-${cols}x${rows}.txt; tmux -L t33003p3 capture-pane -p -e > $Q/live-$1-${cols}x${rows}.ansi.txt; }
stable() { local a b i; a=""; for i in $(seq 1 20); do b=$(cap); [ "$a" = "$b" ] && return 0; a=$b; sleep 1; done; }
clickat() { p=$(cap | $PY $L/pos.py "$1"); [ -n "$p" ] || return 1; $L/click.sh $(( ${p#* } + ${2:-2} )) ${p% *}; }
$L/launch.sh $cols $rows
for i in $(seq 1 60); do cap 2>/dev/null | grep -q "Ready — type a message" && break; sleep 1; done
sleep 6
for attempt in 1 2 3 4 5 6; do
  clickat "   New tab     Settings" 16; sleep 4
  cap | grep -q "Conversation settings" && break
done
cap | grep -q "Conversation settings" || { echo "modal did not open"; cap > $L/fail-${cols}x${rows}.txt; exit 1; }
for k in 1 2 3 4 5 6 7 8; do tmux -L t33003p3 send-keys Tab; sleep 0.6; done
stable; save chat-settings-model-settled
for title in "Advanced generation" "Conversation identity"; do
  for attempt in 1 2 3 4; do
    stable
    clickat "▶ $title" 2 || break
    sleep 2
    cap | grep -q "▼ $title" && break
  done
done
stable; save chat-settings-model-expanded
clickat "Model and generation" 2; sleep 1.5
for i in $(seq -w 1 32); do
  stable; tmux -L t33003p3 capture-pane -p -e > $F/model-$i.ansi.txt
  tmux -L t33003p3 send-keys Tab; sleep 0.7
done
stable; save chat-settings-model-scrolled
clickat "▶ Conversation identity" 2; sleep 2; stable
clickat "Your name in this chat" 26; sleep 1.5; stable
tmux -L t33003p3 capture-pane -p -e > $F/model-identity.ansi.txt
clickat "Context and memory" 2; sleep 2
stable; save chat-settings-context
for i in $(seq -w 1 20); do
  stable; tmux -L t33003p3 capture-pane -p -e > $F/context-$i.ansi.txt
  tmux -L t33003p3 send-keys Tab; sleep 0.7
done
echo done ${cols}x${rows}
tmux -L t33003p3 kill-server 2>/dev/null
