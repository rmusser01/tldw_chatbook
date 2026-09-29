#!/bin/bash
# SGR left click at 1-based COL ROW.
col=$1; row=$2
hx() { printf '%s' "$1" | xxd -p | fold -w2 | tr '\n' ' '; }
tmux -L t33003p6 send-keys -H $(hx $'\x1b'"[<0;${col};${row}M")
sleep 0.15
tmux -L t33003p6 send-keys -H $(hx $'\x1b'"[<0;${col};${row}m")
