#!/usr/bin/env bash
# Usage: drive_library.sh <socket> <size-suffix> [full|key]   (app already launched with golden, past cold start)
. "$(dirname "$0")/drive_lib.sh" "$1" "$2"; LVL="${3:-full}"
clk "⌃3 Library"; sleep 4
has "Library |" || { pal "Switch to Library"; sleep 4; }
shot library-landing
navto(){ # navto "<rail entry>" : expand the collapsed Nav grip first when the entry is not on screen
  has "$1" || { clk "--->" L; sleep 2; }
  has "$1" || { clk "‹" L; sleep 2; }
  clk "$1" L; }
navto "Conversations (6)"; sleep 4; shot library-conversations-item-open
if [ "$LVL" = full ]; then
  navto "Notes (5)"; sleep 3; shot library-notes-list
  clk "Aetheria session 12 prep" B; sleep 3; shot library-notes-editor
  key Escape; sleep 1; key Escape; sleep 1
  navto "Prompts (4)"; sleep 3; clk "Roleplay scene setter" B; sleep 3; shot library-prompts-item-open
  navto "Media (2)"; sleep 3; shot library-media-list
fi
