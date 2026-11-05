---
id: TASK-38698
title: "Console: check the other render-on-open deferrals for the paint-before-recompose race TASK-38650 fixed"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 11:40
updated_date: 2026-11-05 10:12
labels:
- follow-up
- console
- tests
priority: low
references:
- tldw_chatbook/Widgets/Console/console_character_context.py
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/UI/test_console_inspector_rail_deferral.py
dependencies:
- TASK-35970
- TASK-38650
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-38650's review, PR #3373, 2026-11-03).

**What TASK-38650 found.** When the Inspector rail opens, `rebuild_if_deferred` queues its held recompose on the widget's own message pump. The screen's update-timer task can already be awake in the same event-loop pass (`Timer.pause()` only clears an Event), so its tick paints the held rows before the recompose runs. #3373 holds paints for the widgets the rail cascade rebuilds. Its frame test queues the screen's tick ahead of the opening (`loop.call_soon(screen._on_timer_update)`).

**What may share the shape.** Other widgets also render a deferred snapshot when they are shown. The known one is TASK-35970's Character browser. `ConsoleCharacterContext.render_deferred` (console_character_context.py:225-252) starts the render as an asyncio task when the Character section opens, and nothing holds paints until it lands.

Checked by reading the code on dev 9bbd5c2515; no stale frame has been observed. TASK-37021 covers a different Character case: the rail reveal from `on_show`, which renders after the first frame.

Depends on TASK-38650 (in PR #3373).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Console widget that renders a deferred snapshot when it opens is listed in the task notes
- [x] #2 Each is driven by a test that queues the screen's update tick ahead of the opening, as TASK-38650's frame test does; any that paints stale content is fixed, and its test fails on the base build
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read TASK-38650's cause and fix (the cascade's held rebuild and its paint hold) and the lesson in `lessons-textual.md`.
2. Inventory every Console path that keeps a snapshot unrendered while hidden and renders it when it opens: grep for deferral flags, `rebuild_if_deferred`/`render_deferred`, `on_show`, and the rail-open seams in `chat_screen.py`, `left_rail.py` and `right_rail.py`. Classify each as a held snapshot or as a geometry pass that needs the opening's own layout.
3. For each held snapshot, write a frame test that queues the screen's tick ahead of the opening (`loop.call_soon(screen._on_timer_update)`). Run it on the base build, and probe in the full Console where the harness cannot show the path.
4. Fix each one that paints stale content at its opening seam, using TASK-38650's hold (an end-once release plus the asyncio 1 s net). Show its test red on the base and green with the fix.
5. Run the covering tests (head against base by node id), the ratchets and preflight. Then write the inventory into the notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Inventory (AC#1).** These are the Console paths that keep a snapshot unrendered while hidden and render it as the surface opens. They were found by grepping for deferral flags, `rebuild_if_deferred`/`render_deferred`, `on_show`, and the rail-open seams in `chat_screen.py`, `left_rail.py` and `right_rail.py`. Two of them were probed in the full Console.

| # | Widget | What it holds while hidden | Where it renders on open | Races? | Status |
|---|---|---|---|---|---|
| 1 | `ConsoleRunInspector` | its rebuild while the Inspector rail's cascade hides it (TASK-37000) | `rebuild_if_deferred`, from `ChatScreen._sync_console_rail_descendant_visibility` | yes | fixed by TASK-38650; pinned by `test_the_held_rebuild_lands_before_the_openings_first_paint[run inspector]` |
| 2 | `ConsoleInspectorSection` (Agents, Environment, Tasks) | the same | the same | yes | fixed by TASK-38650; pinned by `[section]` |
| 3 | `ConsoleCharacterContext` | the newest snapshot while the Character section is collapsed (TASK-35970) | `render_deferred`, from `ConsoleLeftRail.apply_section_open` (an asyncio task) | **yes**: two frames of the old chats | **fixed here** |
| 4 | Inspector Environment/Tasks sections | their compose-time rows: the controller does not poll while the rail is closed | `_land_console_environment(controller.snapshot)` in `_sync_console_rail_visibility_if_changed`, after the cascade | **yes**: two header-only frames | **fixed here** |
| 5 | `ConsoleCharacterContext` again | the same snapshot, when the whole Context rail was hidden with the section open | `on_show` | renders after the first frame by construction | owned by TASK-37021 (To Do); not driven or changed here, as the description scopes it out |

Checked and not a held snapshot rendered on open:
- `ConsoleBoundedSection.set_presented`/`on_show` and the right rail's outer-fold reconcile. These are geometry passes that measure the opened layout; TASK-37000.5 tracks the outer fold.
- The Character avatar re-fit on open (`invalidate_refresh_scope`). It re-measures the visible body on the next sync tick and needs the opening's layout.
- `ConsoleInspectorSection.set_open` and the Inspector's More (`set_more_open`). These are display toggles; content stays current while closed.
- `chat_screen.py`'s in-place row writes with `layout=False` while hidden: the content is current.
- The hooks indicator's `_refresh_deferred` waits for a send, not an opening. The workspace-files modal's `_status_sync_deferred` waits for its mount. The Terminal center swap mounts fresh from current state.

**Causes.**
- **(3) Character.** The disclosure starts the render as an asyncio task and shows the body in the same callback. A screen tick already woken in that loop pass, with a layout pending, runs before the task enters its recompose's batch, and paints the body's old children. The recorded shape matches TASK-38650's: two frames, one for the layout and one for the repaint.
- **(4) Environment/Tasks.** The opening lands the captured snapshot after the cascade has shown the sections. On a first open that changes the row set (compose-time `()` to the pending row), so `sync_state` queues a rebuild that nothing holds. A queued tick then paints the section header with no rows, twice. The cascade's own hold did not help, for two reasons. The section had no deferred rebuild. And under load the first rail sync can run before the right rail mounts its sections, so the collapse cascade marks nothing and `hidden_by_console_rail` stays False. A probe showed that state in 4 of 16 parallel runs, and every one of those runs failed.

**Fix.** TASK-38650's hold, the end-once release plus asyncio's 1 s `call_later` net, moved from a closure in the cascade into `Widgets/Console/console_inspector_section.py` beside the other rail helpers. `hold_paints(app)` returns the release, and `hold_paints_until_rebuilt(widget)` queues it on the widget's pump behind its rebuild. The cascade now calls the shared helper, with the same behaviour.
- (3): `apply_section_open` holds paints from the render's start and releases them in the render task's done callback, before `_character_render_settled`.
- (4): the opening reads each `ConsoleInspectorSection.recompose_count` before and after `_land_console_environment`, and holds paints until every rebuild the landing queued has run.
- **Cost (3), measured over 10 full-Console runs.** The render's recompose starts 4.6-8.4 ms after the opening and runs 15-25 ms, which the recompose already held. The release follows 0.2-0.6 ms after it. The new pause is therefore about 5-9 ms per disclosure open that has a deferred snapshot.
- (4) pauses only when the opening's landing actually rebuilds a section.
- **An earlier version of (4)** landed the snapshot before the cascade, so the hidden section would defer to the cascade's held rebuild. It passed 8 of 8 runs at load 10 and failed 17 of 24 at eight-way parallel load: the unmarked-section case above. It was replaced.

**Evidence (AC#2).** Each new test queues `screen._on_timer_update` with `call_soon`, plus `screen.refresh(layout=True)` for the pending layout the Character disclosure does not set itself, and records every painted frame.
- `Tests/UI/test_console_character_section_collapsed_render.py::test_a_screen_tick_queued_before_the_opening_paints_no_old_chats`
  - Base: red 16 of 16 (`2 of 6 frames painted old chats`), at eight-way parallel load 27-29.
  - Fix: green 8 of 8 alone and 24 of 24 at eight-way load up to 27.
  - With `call_soon` alone, or the resume/pause nudge, the base passed: the tick had no layout to paint. Lesson below.
- `Tests/UI/test_console_inspector_rail_deferral.py::test_the_inspectors_first_opening_paints_the_environment_rows_it_holds`
  - Base: red 16 of 16 (`((), ('env-pending',))` twice, then the held rows).
  - Fix: green 24 of 24 at eight-way load up to 27.
  - The test skips the opening's controller refresh and stops recording when a later snapshot lands, so only the opening's own landing is judged.
- Both files whole, `-n 4`: 18/18 passed. Serial: 16.6 s and 36.0 s (the 60 s per-file rule holds). Both are census whole-file entries, so the census is unchanged.
- Covering tests: 67 files that use the touched seams or `ConsoleInspectorSection`/`ConsoleRunInspector`, `-n 4`, run on head and then on base sources in one pass.
  - Head: 801 failed, 1624 passed. Base: 801 failed, 1624 passed. Base's failures include the two new tests.
  - On head, 706 of the 801 failures are the local `RecoveryRequired` trap (TASK-38111).
  - 10 nodes failed only on head. Each either passed in 2 head reruns or failed in 2 base reruns as well. They are local load flakes, mostly in `test_console_workspace_context_rail.py`.
- `Tests/Architecture`, serial, head against base: the same 43 failures on both, 4098 passed.
- Size ratchet green.
- Preflight passed.

**Not changed.**
- The on_show path (TASK-37021).
- `Tests/UI/test_console_collapsed_inspector_send_layout.py`.
- No User Guide page describes these one-frame glitches.
- `chat_screen.py` shrank from 24601 to 24589 lines (its ratchet ceiling is 24601). The change is in the rail-opening seam and the cascade, not the full-sync deferral gate PR #3404 edits.

**Lesson:** `lessons-textual.md`, on a queued tick that paints nothing without a pending layout, judging frames against the held state, and rail marks that exist only on children present at the collapse.

**Files:** `Widgets/Console/console_inspector_section.py`, `UI/Screens/chat_screen.py`, `UI/Console_Modules/left_rail.py`, `Tests/UI/test_console_character_section_collapsed_render.py`, `Tests/UI/test_console_inspector_rail_deferral.py`, `backlog/docs/lessons-textual.md`.
<!-- SECTION:NOTES:END -->
