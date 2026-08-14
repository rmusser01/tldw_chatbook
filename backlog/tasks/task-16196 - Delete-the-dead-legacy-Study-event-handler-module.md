---
id: TASK-16196
title: 'Delete the dead legacy Study event-handler module'
status: To Do
assignee: []
created_date: '2026-08-14 03:05'
labels:
  - cleanup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Event_Handlers/Study_Events/study_events.py`'s `STUDY_BUTTON_HANDLERS` / `study_event_handler` are referenced nowhere outside the module at current dev HEAD — flashcard handling moved to `Study_Modules/flashcards_handler.py` during the Study rebuild, and TASK-15471's implementer and reviewer independently verified the unreachability (grep + dispatch-path read). The module still contains synchronous ChaChaNotes writes that the input-latency audit flagged — dead code shaped like a loaded gun. Delete the dead handler surface (owner ruling: delete dead code rather than leave loaded guns), preserving anything still genuinely imported (verify each symbol, not just the table). Coordinate with TASK-16195, which may want one piece of it (add-topic) resurrected in the modern location first. Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The unreferenced handler table and its dead handlers are removed
- [ ] #2 Anything still imported elsewhere is identified and preserved (evidence: import graph or grep per symbol)
- [ ] #3 Test collection and the Study suites stay green
<!-- AC:END -->
