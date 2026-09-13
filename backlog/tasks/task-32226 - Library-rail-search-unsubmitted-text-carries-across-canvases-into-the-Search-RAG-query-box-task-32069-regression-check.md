---
id: TASK-32226
title: >-
  Library rail search: unsubmitted text carries across canvases into the
  Search/RAG query box (task-32069 regression check)
status: Done
assignee: []
created_date: '2026-09-10 14:55'
updated_date: '2026-09-10 19:04'
labels:
  - library
  - rail
  - regression
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-32069 claimed the rail box is emptied on canvas change; assessor B saw unsubmitted text persist across canvases and land in the RAG query box. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 25.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Re-verify; if reproducible, the rail box is emptied on canvas change and never seeds the RAG query box
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep the pins on the rail-search Changed path before changing it.
2. Failing test: type on Media, switch to Search/RAG, assert both the query box and the state are empty.
3. Gate the commit on the selected row; re-run the named neighbour files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
REPRODUCED, and task-32069 was not the whole story. `_library_rail_search_value` does empty the BOX off the Search/RAG canvas, so the widget was never the carrier. What carried was the STATE: `handle_library_search_changed` wrote `_rag_search_state.query` on every `Input.Changed` regardless of the selected row, so a word typed into the rail on Media and abandoned was sitting in the Search/RAG query box the next time that canvas opened -- exactly what assessor B saw.

Fix: off the Search/RAG row the keystrokes stay in the widget; only a submit (which selects that row first) commits them. The task-4023 RC-08 mirror contract is untouched on the row it applies to -- its pin (`test_library_shell_stale_mirror_events_do_not_replenish_changed_traffic`) runs on the Search/RAG row and stays green.

Neighbour files re-run, no new failing names: test_library_rag_keystroke.py 6 passed, test_library_crit8_polish_shell.py 28 passed.

Live at 235x52: typed "draft" into the rail box on Media, opened Search/RAG -- the query box shows its placeholder and the rail box is empty.

Files: tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py, Tests/UI/test_library_crit9_rail.py
<!-- SECTION:NOTES:END -->
