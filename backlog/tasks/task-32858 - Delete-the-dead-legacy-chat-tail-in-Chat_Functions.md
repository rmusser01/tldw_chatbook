---
id: TASK-32858
title: Delete the dead legacy chat tail in Chat_Functions
status: In Progress
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
- [x] #1 Census reconciled with TASK-32807.5: the overlap is recorded in both tasks, and neither claims the other's lines
- [ ] #2 The dead helpers and their tests are deleted (~1,270 LOC baseline)
- [ ] #3 The MediaWindow_v2 path decision is recorded: `chat()` kept with its single documented caller chain, or migrated — with tests
- [ ] #4 Stale `chat_events.py` docstring references removed
- [ ] #5 Targeted Chat/UI test runs green after deletion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile the census against TASK-32807.5 (done — see notes).
2. Re-verify each helper's zero-production-caller claim at the working HEAD (grep + import trace), since the review baseline was `d6e2a46384`.
3. Delete the helpers and their referencing tests in one slice; keep `chat()` and its caller chain untouched in this slice.
4. Record the MediaWindow_v2 keep-or-migrate decision (AC#3) — default keep; migration needs its own slice because `chat()` owns media-content RAG injection.
5. Remove the stale `chat_events.py` docstring references; run targeted Chat/UI suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Census reconciliation (AC#1, 2026-09-19): **disjoint — zero overlap.** TASK-32807.5 owns five dead MODULES (whole files with zero production importers: `console_visual_evaluation.py` 1,286, `console_visual_benchmark.py` 242, `document_generator.py` 601, `prompt_template_manager.py` 138, `server_chat_loop_service.py` 82 = 2,349 lines, per `qa/core-code-review-2026-09-17/report.md` CHAT-rest-3). This task owns dead FUNCTIONS inside the LIVE `Chat_Functions.py` module — a different granularity the module census cannot see, and `Chat_Functions.py` itself has production importers, so it is not in 32807.5's table. Neither task claims the other's lines. Reconciliation note also added to 32807.5.
<!-- SECTION:NOTES:END -->
