#!/usr/bin/env bash
# SGR left-click at 1-based CHARACTER column/row. Usage: click.sh <socket> <col> <row>
tmux -L "$1" send-keys -l $'\x1b[<0;'"$2;$3"'M'; tmux -L "$1" send-keys -l $'\x1b[<0;'"$2;$3"'m'
