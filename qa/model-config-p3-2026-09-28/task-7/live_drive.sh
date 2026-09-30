#!/bin/bash
# usage: live_drive.sh <tree> <label> <cols> <rows> <outdir>
# Settings Overview at rest, the inspector wheel-scrolled to its end, then
# Privacy & Security (an overflowing inspector) at rest and scrolled to its end.

L=$(dirname "$0")
W=$1; label=$2; cols=$3; rows=$4; O=$5; mkdir -p $O
T="tmux -L t33003p7"
cap() { $T capture-pane -p; }
snap() { cap > $O/$label-${cols}x${rows}-$1.txt; $T capture-pane -p -e > $O/$label-${cols}x${rows}-$1.ansi.txt; }
stable() { a=""; for i in $(seq 1 20); do b=$(cap); [ "$a" = "$b" ] && return 0; a=$b; sleep 0.7; done; }
hx() { printf '%s' "$1" | xxd -p | fold -w2 | tr '\n' ' '; }
wheel() {  # wheel down N times at 1-based COL ROW
  for i in $(seq 1 $3); do $T send-keys -H $(hx $'\x1b'"[<65;$1;$2M"); sleep 0.1; done
}
inspector_point() {  # 1-based col,row inside the inspector body (a few rows under the pinned note)
  cap | python3 -c 'import sys
L = sys.stdin.read().split("\n")
for n, l in enumerate(L, 1):
    i = l.find("Scope Inspector")
    if i >= 0:
        print(i + 4, n + 16); break'
}
$L/live_launch.sh $W $label $cols $rows
for i in $(seq 1 90); do cap 2>/dev/null | grep -q "Ready — type a message" && break; sleep 1; done
sleep 5
$T send-keys F4; sleep 5; stable; snap overview
p=$(inspector_point); wheel ${p% *} ${p#* } 30; sleep 1; stable; snap overview-scrolled-end
$T send-keys /; sleep 1; $T send-keys -l "Privacy"; sleep 1.5; $T send-keys Enter; sleep 4; stable; snap privacy
p=$(inspector_point); wheel ${p% *} ${p#* } 40; sleep 1; stable; snap privacy-scrolled-end
$T kill-server 2>/dev/null
echo "done $label ${cols}x${rows}"
