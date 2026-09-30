#!/bin/bash
# usage: live_drive.sh <tree> <label> <outdir>
# A: Console Behavior at 235x52 -- click the Temperature fallback, resize to
#    211x44 (the Phase 2 capture's route), Tab to Top P.
# B: 211x44 -- "/" temperature Enter (lands on Providers & Models), resize
#    to 235x52, Tab to the next model default.
L=$(cd "$(dirname "$0")" && pwd)
W=$1; label=$2; O=$3; mkdir -p $O
T="tmux -L t33003p9"
cap() { $T capture-pane -p -t s; }
snap() { cap > $O/$label-$1.txt; $T capture-pane -p -e -t s > $O/$label-$1.ansi.txt; }
stable() { a=""; for i in $(seq 1 20); do b=$(cap); [ "$a" = "$b" ] && return 0; a=$b; sleep 0.7; done; }
hx() { printf '%s' "$1" | xxd -p | fold -w2 | tr '\n' ' '; }
click() {  # 1-based COL ROW
  $T send-keys -t s -H $(hx $'\x1b'"[<0;$1;$2M"); sleep 0.05
  $T send-keys -t s -H $(hx $'\x1b'"[<0;$1;$2m")
}
field_point() {  # the input cell right of a detail-pane label ($1), 1-based
  cap | LABEL="$1" python3 -c 'import os, sys
label = "│ " + os.environ["LABEL"] + " "
for n, l in enumerate(sys.stdin.read().split("\n"), 1):
    i = l.find(label)
    if 0 <= i < 150:
        print(i + 30, n); break'
}
wheel_to() {  # wheel the detail pane until label $1's row shows (at most 80 ticks)
  for i in $(seq 1 80); do
    [ -n "$(field_point "$1")" ] && { sleep 0.5; stable; return 0; }
    $T send-keys -t s -H $(hx $'\x1b'"[<65;80;30M"); sleep 0.15
  done
}
boot() {
  $L/live_launch.sh $W $label-$1 $2 $3
  for i in $(seq 1 90); do cap 2>/dev/null | grep -q "Ready — type a message" && break; sleep 1; done
  sleep 5; $T send-keys -t s F4; sleep 5; stable
}
boot a 235 52
$T send-keys -t s /; sleep 1; $T send-keys -t s -l "Console Behavior"; sleep 1.5; $T send-keys -t s Enter; sleep 4; stable
wheel_to Temperature; wheel_to Temperature
p=$(field_point Temperature); click ${p% *} ${p#* }; sleep 2; stable; snap cb-click-temperature-235x52
$T resize-window -t s -x 211 -y 44; sleep 3; stable; snap cb-temperature-resized-211x44
$T send-keys -t s Tab; sleep 2; stable; snap cb-tab-top-p-211x44
$T kill-server 2>/dev/null
boot b 211 44
$T send-keys -t s /; sleep 1; $T send-keys -t s -l "temperature"; sleep 1.5; $T send-keys -t s Enter; sleep 4; stable; snap pm-search-temperature-211x44
$T resize-window -t s -x 235 -y 52; sleep 3; stable; snap pm-temperature-resized-235x52
$T send-keys -t s Tab; sleep 2; stable; snap pm-tab-next-235x52
$T kill-server 2>/dev/null
echo "done $label"
