---
id: TASK-32858
title: Delete the dead legacy chat tail in Chat_Functions
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies:
  - TASK-32807.5
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The pre-Console chat era left a dead tail inside `Chat/Chat_Functions.py`: zero production callers for `save_chat_history_to_db_wrapper` (:2209, ~413 LOC), `save_chat_history` (:2622), `get_conversation_name` (:2691), `generate_chat_history_content` (:2719), `extract_media_name` (:2882), `update_chat_content` (:2946), `save_character` (:3160), `load_characters` (:3362), `get_character_names` (:3436) — ~1,270 LOC. The legacy `chat()` orchestration (:1443-2118, ~676 LOC) has exactly one production caller chain (`app.py:19859` → `Event_Handlers/worker_events.py:41`, which bans streaming → `UI/MediaWindow_v2.py:1881`); up to ~1,950 LOC (56% of the file) goes if that caller migrates to `chat_api_call` — but `chat()` carries media-content RAG injection the gateway deliberately doesn't own, so migration is a decision, not an assumption.

TASK-32807.5's census (~2,350 dead Chat lines) may already include part of this tail — AC#1 is the reconciliation so the board never double-claims. Tests reference the dead helpers (21 refs across 8+ suites); they retire with the code. Also fixes the stale `chat_events.py` docstring references at `UI/Screens/chat_screen.py:14496/:14507` (the module is gone). ADR required: no — deletion; the MediaWindow_v2 option, if taken, records its decision in the implementation notes.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Census reconciled with TASK-32807.5: the overlap is recorded in both tasks, and neither claims the other's lines
- [ ] #2 The dead helpers and their tests are deleted (~1,270 LOC baseline)
- [ ] #3 The MediaWindow_v2 path decision is recorded: `chat()` kept with its single documented caller chain, or migrated — with tests
- [ ] #4 Stale `chat_events.py` docstring references removed
- [ ] #5 Targeted Chat/UI test runs green after deletion
<!-- AC:END -->
