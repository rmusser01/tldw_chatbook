---
id: TASK-38946
title: "Console: widgets mounted into an already collapsed rail are not marked hidden, so they lay out the screen while it is closed"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 17:03
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
- [x] #1 A widget mounted inside a Console rail after the rail collapsed counts as hidden by that rail, and its updates request no layout while the rail stays closed
- [x] #2 When the rail opens, that widget shows exactly as it would in a rail that was open throughout, with any held rebuild applied
- [x] #3 A deterministic test mounts an Inspector section into an already collapsed rail, in the startup order TASK-38698 recorded, and counts layout requests from inside the rail while it is closed; it is red on dev and reads zero after the fix
- [x] #4 The TASK-37000 and TASK-38698 deferral and parity tests pass at -n 4, with the load recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce TASK-38698's startup order in `Tests/UI/test_console_inspector_rail_deferral.py`'s rail harness: collapse an empty rail through the production cascade, mount an Inspector section into it, then change the section's state and count layout requests from inside the rail. Show it red on the base.
2. Have the cascade record on the rail itself whether it is collapsed (`CONSOLE_RAIL_COLLAPSED`), cleared when it opens.
3. In `hidden_by_console_rail`, a widget with no mark looks up its ancestors for a collapsed rail. If it finds one, it marks itself with its current display (the same mark the cascade writes), so the opening restores it and runs its held rebuild like any widget that was present at the collapse. Every caller (the section deferral, the run inspector, `set_display_under_console_rail`, the readiness rows) gets the fix through this one helper.
4. Check the opened rail against one that was open throughout, widget for widget and box for box.
5. Run the TASK-37000 and TASK-38698 deferral and parity files at `-n 4`, recording the load, plus the covering tests and the size ratchet.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** A widget mounted into a rail after the rail collapsed now counts as hidden by that rail.
- The collapse cascade (`ChatScreen._sync_console_rail_descendant_visibility`) also records on the rail itself whether it is collapsed (`CONSOLE_RAIL_COLLAPSED`). It sets the flag before it walks the descendants, and it clears the flag on the opening before it restores them.
- `hidden_by_console_rail` (`Widgets/Console/console_inspector_section.py`) keeps its fast path for a marked widget. An unmarked widget looks up its ancestors for a collapsed rail. If it finds one, the widget marks itself with its current display, the same mark the cascade writes, and counts as hidden.
- The rail's opening already restores every marked descendant and runs its `rebuild_if_deferred`, so an adopted widget reopens exactly like one present at the collapse. A display preference written meanwhile through `set_display_under_console_rail` lands in the mark and shows on the opening.
- Every caller goes through the one helper: the section deferral, the run inspector, `set_display_under_console_rail`, and the readiness rows' `layout=` choice.
- Marking happens on the widget's first ask, not at mount. Textual has no descendant-mount event to hook, and a widget that never asks does nothing that the mark gates. The mark leaves the widget's own `display` alone: the rail is `display: none`, and writing `display` would itself request the layout this task removes.

**Evidence.**
- New test: `Tests/UI/test_console_inspector_rail_deferral.py::test_a_section_mounted_into_a_collapsed_rail_waits_for_the_opening`. It uses TASK-38698's startup order: the production cascade collapses an empty rail, and an Agents section mounts into it afterwards. The section's "Run history" summary then appears, and the test counts layout requests from inside the rail, opens the rail, and compares it widget for widget and box for box with a rail that was open throughout.
  - Base: red, `the late-mounted section asked for layouts`, 18 requests (the section, its header, body, rows, toggle, summary and title).
  - Fix: zero requests, and the opened rail matches.
- AC#4, `-n 4`: `test_console_inspector_rail_deferral.py`, `test_console_collapsed_inspector_send_layout.py` and `test_console_character_section_collapsed_render.py` gave 23 passed (load 50-53). `test_console_pending_interrupt_projection.py` gave 15 passed (load 47-48).
- Covering files, head against base with the same command (`-n 4`, `-p no:randomly`): `test_console_rail_refresh_scope`, `_inspector_rail_minors`, `_inspector_section`, `_right_rail`, `_environment_section`, `_resize_reflow`, `_run_tick_layout_requests`, `_inspector_focus_carriers`, `_shell_regions`, `_environment_wiring`.
  - Base: 152 failed, 118 passed. Head: 153 failed, 117 passed. Nearly all of the failures are the local `RecoveryRequired` trap (TASK-38111).
  - The one node that failed only on head, `test_resize_priority_hands_inspector_focus_to_reveal_button`, passed on head and failed on the base when rerun alone. It is a load flake.
- `Tests/Architecture/test_screen_size_ratchet.py` passed: `chat_screen.py` gained 2 lines and stays under its ceiling.

**Files:** `tldw_chatbook/Widgets/Console/console_inspector_section.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/UI/test_console_inspector_rail_deferral.py`.
<!-- SECTION:NOTES:END -->
