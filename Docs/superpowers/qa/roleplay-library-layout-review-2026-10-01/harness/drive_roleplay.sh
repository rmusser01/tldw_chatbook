#!/usr/bin/env bash
# Usage: drive_roleplay.sh <socket> <size-suffix> [full|key]   (app already launched with golden, past cold start)
. "$(dirname "$0")/drive_lib.sh" "$1" "$2"; LVL="${3:-full}"
clk "⌃4 Roleplay"; sleep 4
has "Modes:" || { pal "Switch to Roleplay"; sleep 4; }
shot roleplay-characters-arrival
clk "Captain Isolde Varga"; sleep 3; shot roleplay-character-selected
if [ "$LVL" = full ]; then
  clk "  Edit  " B; sleep 3; shot roleplay-character-editor
  r=$(rowof "Character Editor"); c=$("$PY" "$R/where.py" "$S" "Character Editor" | head -1 | cut -d' ' -f2)
  wheel down $((c+20)) $((r+8)) 8; sleep 1; shot roleplay-character-editor-scrolled
  clk "Cancel"; sleep 2
  clk "Try a test chat"; sleep 3
  has "▾ Try a test chat" || { clk "Try a test chat"; sleep 3; }
  shot roleplay-testchat-open
  # the input may be clipped to its top border at short heights: click whatever row of it is visible
  r=$(rowof "Test message..."); [ -n "$r" ] || r=$(tmux -L "$S" capture-pane -p | grep -n "Greeting:" -A8 | grep "▊▔" | head -1 | cut -d- -f1)
  if [ -n "$r" ]; then "$R/click.sh" "$S" 70 "$r"; sleep 0.8; fi
  if [ -n "$r" ] && tmux -L "$S" capture-pane -p | sed -n "$((r-1)),$((r))p" | grep -q "┌"; then
    typ "Captain, the derelict is hailing us."; sleep 0.3; key Enter; sleep 6; shot roleplay-testchat-reply
  else echo "  !! test-chat input not focusable; skipped typing"; fi
  clk "Try a test chat"; sleep 1.5
  r=$("$PY" "$R/where.py" "$S" "Library " | awk '$1>3' | head -1 | cut -d' ' -f1); c=$(colafter "$r" "Library" "<"); [ "$c" -gt 0 ] && "$R/click.sh" "$S" "$c" "$r"; sleep 1.5
  r=$("$PY" "$R/where.py" "$S" "Inspector " | awk '$1>3' | head -1 | cut -d' ' -f1); c=$(colafter "$r" "Inspector" ">"); [ "$c" -gt 0 ] && "$R/click.sh" "$S" "$c" "$r"; sleep 1.5
  shot roleplay-rails-collapsed
  clk " Library " L; sleep 1.5; clk " Inspector " B; sleep 1.5
fi
clk "Personas  "; sleep 3; clk "Default Narrator"; sleep 3; shot roleplay-personas-selected
clk "Dictionaries  "; sleep 3; clk "Fantasy Anachron"; sleep 3
r=$(rowof "Try it — substitution"); c=$("$PY" "$R/where.py" "$S" "Try it — substitution" | head -1 | cut -d' ' -f2)
"$R/click.sh" "$S" $((c+10)) $((r+2)); sleep 0.7
if has "Try it — substitution" && tmux -L "$S" capture-pane -p | sed -n "$((r+1))p" | grep -q "┌"; then
  key Down Down Down Down Down Down End; tmux -L "$S" send-keys -N 700 BSpace; sleep 1
  typ "Grab your phone and the car keys, OK? The police took my computer."; sleep 0.4; clk "Run preview"; sleep 3
else echo "  !! Try-it textarea not focused; skipped typing (single-letter mode keys would fire)"; fi
shot roleplay-dictionaries-tryit
clk "    Lore"; sleep 3; clk "Aetheria Atlas"; sleep 3; shot roleplay-lore-selected
if [ "$LVL" = full ]; then
  clk "Characters  "; sleep 3; key C-f; sleep 0.5; typ "zzqx"; sleep 2; shot roleplay-characters-search-nomatch
  key BSpace BSpace BSpace BSpace; sleep 1.5
fi
