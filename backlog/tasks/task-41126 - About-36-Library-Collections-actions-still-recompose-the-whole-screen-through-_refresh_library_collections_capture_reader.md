---
id: TASK-41126
title: "About 36 Library Collections actions still recompose the whole screen through _refresh_library_collections_capture_reader"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- library
- performance
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_collections_controller.py
- tldw_chatbook/Widgets/Library/library_collections_capture_reader.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40328 made the Collections entry load repaint its panes in place (one recompose per press instead of four, about 0.4-0.6 s less UI-loop CPU per press). Every other Collections action still rebuilds the whole Library screen, with the same cost and the same focus and stale-rail risks that task had to fix.

**Evidence.** `library_collections_controller.py:424-432` (`_refresh_library_collections_capture_reader`) is a whole-screen `refresh(recompose=True)`; it is called from 36 places in that file (`:448` through `:1656`). TASK-40328 routed only the entry loader through `_sync_library_collections_capture_panes` (`recompose_capture_panes`) and left the rest: "Other Collections actions still use the whole-screen helper (out of scope)."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Collections actions that change only the panes' presentation repaint the panes in place; each remaining whole-screen recompose is listed with its reason
- [ ] #2 A test counts compose_content passes for a representative set of actions (capture, annotate, scope change, delete), at most one each, red before the change
- [ ] #3 UI-loop CPU per action is measured before and after
<!-- AC:END -->
