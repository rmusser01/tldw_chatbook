---
id: TASK-40659
title: 'Roleplay lore entries table fills on the event loop'
status: To Do
assignee: []
created_date: '2026-11-12 19:29'
labels:
  - roleplay
  - performance
dependencies: []
references:
  - tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py
  - tldw_chatbook/UI/Screens/personas_screen.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Selecting a large lore book stalls Roleplay. `PersonasLoreDetailWidget.update_entries` clears `#personas-lore-entries-table` and adds every entry as a row in one pass on the event loop. In the ADR-011 capture of TASK-33910.44 (real `TldwCli` at 120x36, a 9.0 MB lore book of 22,000 entries imported from Characters, so the landing selects it), that pass held the loop for 1.1-3.0 s on both arms. Nothing painted and no key was handled during it.

This exists on dev today. The code is unchanged on dev `26b2e677dd`; the measurement was taken on burn-down PR 5's base `cec10bbc93`. Burn-down PR 5 (TASK-33910.44) does not fix it: the owner split this stall out of TASK-33910.44 on 2026-11-12 ("Split it out (Rec.)").
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selecting a 22,000-entry lore book shows its first entries at once, and the screen keeps handling keys while the rest load
- [ ] #2 The ADR-011 capture with the 9 MB, 22,000-entry book shows no loop gap over 50 ms while the entries table fills
- [ ] #3 Entry selection, editing and reordering behave as before on a large book
<!-- AC:END -->
