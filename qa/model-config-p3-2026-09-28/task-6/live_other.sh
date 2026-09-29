#!/bin/bash
# Review round 1: paths come from the environment. T6_SCRATCH = a scratch dir
# holding a `git archive <rev> | tar -x` tree per non-head label (base, pre);
# PYTHON = the venv interpreter; HEAD_TREE = the checkout run as "head". Copy
# live_*.sh to T6_SCRATCH/live without the prefix, and ansi_cells.py as cells.py.
# Capture destinations outside Settings/Chat settings. Usage: other.sh <tree> <theme> <outdir>
L=${T6_SCRATCH:?export T6_SCRATCH}/live
tree=$1; theme=$2; O=$3; T="tmux -L t33003p6"
cap() { $T capture-pane -p; }
stable() { a=""; for i in $(seq 1 30); do b=$(cap); [ "$a" = "$b" ] && return 0; a=$b; sleep 0.8; done; }
$L/launch.sh $tree $theme 211 44
for i in $(seq 1 90); do cap 2>/dev/null | grep -q "Ready — type a message" && break; sleep 1; done
sleep 5
for label in Library MCP; do
  col=$(cap | sed -n 2p | python3 -c "import sys; l=sys.stdin.read(); print(l.index(' $label')+2)")
  $L/click.sh $col 2; sleep 4; stable
  cap > $O/$tree-$theme-211x44-screen-$label.txt; $T capture-pane -p -e > $O/$tree-$theme-211x44-screen-$label.ansi.txt
done
$T kill-server 2>/dev/null
echo "done $tree $theme"
