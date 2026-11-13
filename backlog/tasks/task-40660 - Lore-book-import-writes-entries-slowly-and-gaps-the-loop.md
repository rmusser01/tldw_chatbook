---
id: TASK-40660
title: 'Lore book import writes entries slowly and gaps the event loop'
status: To Do
assignee: []
created_date: '2026-11-12 19:29'
labels:
  - roleplay
  - performance
dependencies: []
references:
  - tldw_chatbook/Character_Chat/world_book_manager.py
  - tldw_chatbook/UI/Screens/personas_screen.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Importing a large lore book takes minutes and stalls the screen it runs beside. `WorldBookManager.import_world_book` creates the book, then calls `create_world_book_entry` once per entry. The Roleplay import runs it in a thread. In the ADR-011 capture of TASK-33910.44 (real `TldwCli` at 120x36, a 9.0 MB lore book of 22,000 entries, load average 14-63):
- the write took 100-205 s (149 s standalone);
- the event loop still saw gaps of 115-210 ms, dozens of times, on both arms;
- the gaps come from the GIL and garbage-collection pressure of the per-entry writes.

This exists on dev today. The code is unchanged on dev `26b2e677dd`; the measurement was taken on burn-down PR 5's base `cec10bbc93`. Burn-down PR 5 (TASK-33910.44) does not fix it: the owner split this stall out of TASK-33910.44 on 2026-11-12 ("Split it out (Rec.)"). Why each entry is slow (one transaction per entry or another cost) is not yet measured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Importing a 9 MB, 22,000-entry lore book completes within a stated time budget on the reference machine, measured before and after
- [ ] #2 The ADR-011 capture of that import shows no loop gap over 50 ms during the write
- [ ] #3 The import's other behaviour is unchanged: a name clash is renamed, and a malformed file is refused before any write
<!-- AC:END -->
