---
id: TASK-38946
title: "Console widgets mounted into an already collapsed rail are not marked hidden and lay out the screen while it is closed"
status: To Do
created_date: 2026-11-05 21:35
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Widgets/Console/console_inspector_section.py
- tldw_chatbook/Widgets/Console/console_run_inspector.py
- backlog/docs/lessons-textual.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

- The Console rail collapse cascade, `_sync_console_rail_descendant_visibility` (`tldw_chatbook/UI/Screens/chat_screen.py:12633`), marks only the widgets present when it runs. It walks `widget.query("*")` (`:12635`) and sets `CONSOLE_RAIL_PRIOR_DISPLAY` on each one (`:12652`).
- It runs only when the rail state changes (`_sync_console_rail_visibility_if_changed`, `:12598`). A widget mounted into a rail that is already collapsed is never marked.
- `hidden_by_console_rail` (`tldw_chatbook/Widgets/Console/console_inspector_section.py:85-99`) reads that mark. Its docstring notes that each layout a hidden widget requests "lays out the whole screen". The section deferral (`:730`) and the run inspector (`tldw_chatbook/Widgets/Console/console_run_inspector.py:308`) depend on the mark, so an unmarked late mount pays full-screen layouts while the rail is closed.
- TASK-38698's probe saw this in 4 of 16 parallel runs. The first rail sync ran before the right rail mounted its sections, so the cascade marked nothing (`backlog/docs/lessons-textual.md:2502-2511`).
- Related but different, both To Do: TASK-37000.1 (direct display writes to widgets that are already marked) and TASK-37000.3 (moving the cascade and its helpers into one module).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A widget mounted inside a Console rail after the rail collapsed counts as hidden by that rail, and its updates request no layout while the rail stays closed
- [ ] #2 When the rail opens, that widget shows exactly as it would in a rail that was open throughout, with any held rebuild applied
- [ ] #3 A deterministic test mounts an Inspector section into an already collapsed rail, in the startup order TASK-38698 recorded, and counts layout requests from inside the rail while it is closed; it is red on dev and reads zero after the fix
- [ ] #4 The TASK-37000 and TASK-38698 deferral and parity tests pass at -n 4, with the load recorded
<!-- AC:END -->
