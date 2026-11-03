---
id: TASK-38650
title: Console a late-opened Inspector painted pre-send rows once under load
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 05:51
updated_date: 2026-11-03 07:25
labels:
- follow-up
- console
- tests
priority: low
references:
- Tests/UI/test_console_collapsed_inspector_send_layout.py
- tldw_chatbook/Widgets/Console/console_run_inspector.py
dependencies:
- TASK-37000
- TASK-38644
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-38644 implementer (2026-11-03).

`Tests/UI/test_console_collapsed_inspector_send_layout.py::test_an_inspector_opened_after_a_send_matches_one_open_throughout` pins every frame the Inspector paints when it is opened after a first send (TASK-37000). It failed once at `assert set(frames) == {((False,), ("Run history",))}` (:357 on dev 18482c1cf7). The late-opened rail's first two painted frames showed the rows from before the send, so the rebuild landed after the first paint. That is the TASK-35970 failure the assert exists to catch.

Counts so far: 1 failure in 8 whole-file runs at load about 50 (on the TASK-38644 branch), then none in 24 more runs there and none in 48 runs on dev. The TASK-38644 change runs only after the frames are captured, so it is unlikely to be the cause.

If the race is real, a user who opens the Inspector right after a first send under a busy event loop may see one frame of stale rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The failure is reproduced on dev, under load or with a temporary source edit that delays the rebuild, or the notes record the run counts that did not reproduce it and the ordering the code guarantees
- [x] #2 If it is reproduced: the cause is named, and the fix makes the rebuild land before the opened rail's first paint (product) or corrects the test's frame capture (test), with a test shown red before the fix
- [ ] #3 If it cannot be reproduced: the notes show why the code orders the rebuild before the first paint, and the task closes with no change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the opening path end to end: `action_toggle_console_inspector_rail` -> `_sync_console_rail_visibility` -> the cascade's `rebuild_if_deferred` (run inspector, Agents section) against Textual 8.2.8's paint path (`Screen._on_timer_update`, the update timer, `App.batch_update`, `Pilot.pause`), and list every way a `_compositor_refresh` can run before the held recompose enters its batch.
2. Tell the candidate causes apart with temporary instrumentation in the test (state at the open: held flags, the inspector's and section's own state; per frame: batch count, caller), run the node repeatedly under `-n 4` load, and record counts and load.
3. Force each candidate with a temporary source edit (a delayed rebuild; a late post-send state delivery) and compare the failure's shape with the recorded one (`((True,), ())` twice, then the post-send rows).
4. If reproduced: fix at the root (product ordering or the test's frame capture / precondition), pin with a test shown red first; covering tests, ratchets, preflight.
5. If not: record the counts and the ordering the code guarantees (AC#3) and close with no change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause: a product race between the screen's update tick and the held rebuild.** The rail's opening cascade calls `rebuild_if_deferred`, which queues the recompose on the widget's own message pump (`refresh(recompose=True)` goes through `call_next`). The screen paints from its update timer, which runs on a task of its own. `Pilot.pause()` pauses that timer just before the test opens the rail, but `Timer.pause()` only clears an `asyncio.Event`. When the screen's idle had resumed the timer earlier in the same loop pass, the timer task was already woken. The loop then ran its tick before the pumps of the run inspector and the Agents section, which the opening woke later. The recompose's own batch holds paints only once it has started, so that tick painted the held rows. One `Screen._on_timer_update` calls `_compositor_refresh` twice (`_refresh_layout`, then the repaint), which gives exactly the recorded failure: two `((True,), ())` frames, then the post-send rows. In the app the same thing can happen on any opening while the update timer is running.

**Fix.** `ConsoleRunInspector.rebuild_if_deferred` and `ConsoleInspectorSection.rebuild_if_deferred` call `self.app.delay_update()` before they queue the rebuild. That starts an app batch at once. Its end is a 50 ms timer, which runs only after the pump step queued with it, and by then the recompose holds its own batch. A tick that comes first skips painting. Only an opening with a held rebuild is affected. The run inspector's recompose under load takes 0.15-1.5 s, so the delay does not lengthen the opening.

**AC#1, reproduction (dev 18482c1cf7, `-n 4`).** The instrumentation was temporary and is reverted. It recorded the frame facts, the held flags, the batch count and the paint's caller.
- Natural runs did not reproduce it: 130 frame checks, 0 failures. These were 1 single run, 24 + 90 repeats of the node (load 24-46), and 15 in 16 whole-file repeats (load 27-37).
- Every run had both rebuilds held with the post-send state at the opening. That state landed 1.9-5.1 s before the opening, which rules out a late state delivery.
- Deterministic, test-side nudge before the toggle. `loop.call_soon(screen._on_timer_update)` failed 1 of 1. `_update_timer.resume()` then `.pause()` failed 4 of 4; that is the screen's idle and Pilot's pause in one loop pass. Each failure had the recorded frames `[((True,), ()), ((True,), ()), ((False,), ('Run history',)), ...]`. At the stale frames the batch count was 0, and both widgets had a recompose queued but not started.

**AC#2, fix and pin.**
- New test `test_the_held_rebuild_lands_before_the_openings_first_paint[run inspector|section]` in `Tests/UI/test_console_inspector_rail_deferral.py`, a census whole-file entry. It queues the screen's tick ahead of the production cascade's opening and records every frame.
  - Red on dev in 16 of 16 runs (8 repeats x 2 params), on the frame assertion: the held rows were painted twice, then the rebuilt rows.
  - Green with the fix in 16 of 16.
- Full app with the fix:
  - The resume/pause nudge passed 8 of 8.
  - The plain node had 95 of 95 frame checks green in 96 repeats (load 15-49). The one failure was the outer fold (`console-inspector-rail-body` 23 rows against 22), TASK-37000.5. Dev's 96 repeats had 5 fold failures and 2 `assert not rail.display` failures (TASK-38644).
- Covering tests: the 30 test files that use either widget, by node id against dev. The fix head failed 242 nodes and dev failed 244: the same 242 plus the new test's 2. 235 of the failures are the local `RecoveryRequired` trap (TASK-38111). Five rail-opening cases were run under a temporary `@private_profile_test` marker, since removed: 2 passed and 3 failed on both builds, with the same assertions.
- The sibling chat-switch case (`test_an_inspector_opened_after_chat_switches_matches_one_open_throughout`), 16 node repeats each at load 18-25: the fix head passed 16 of 16, and dev failed 1 of 16 on the staged-context tray's height. In dev's 16 whole-file repeats at load 27-37 it failed 9 of 16; proposed to the coordinator as a follow-up.
- `Tests/Architecture` size ratchets 205/205. Preflight passed.

**AC#3** does not apply: the failure was reproduced and fixed.

**Not changed.** `Tests/UI/test_console_collapsed_inspector_send_layout.py` is untouched (PR #3358 edits it). No User Guide page describes this one-frame glitch.

**Lesson:** `lessons-textual.md`, on `Timer.pause()` not stopping an already woken screen tick.

**Files:** `Widgets/Console/console_run_inspector.py`, `Widgets/Console/console_inspector_section.py`, `Tests/UI/test_console_inspector_rail_deferral.py`, `backlog/docs/lessons-textual.md`.
<!-- SECTION:NOTES:END -->
