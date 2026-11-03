---
id: TASK-38650
title: Console a late-opened Inspector painted pre-send rows once under load
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 05:51
updated_date: 2026-11-03 08:26
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

**Fix (after review round 1).** `rebuild_if_deferred` on both widgets now returns whether it queued a rebuild. Its one caller, the opening cascade `ChatScreen._sync_console_rail_descendant_visibility`, holds paints for each queued rebuild: `app._begin_batch()` when the rebuild is queued, and `child.call_next(app._end_batch)` right behind it. The widget's own pump ends the hold right after its recompose, so no timer is involved. A tick that comes first skips painting.
- **Cost, measured in the deferral test's 40-column host (10 runs).** The recompose starts 0.6-1.3 ms after the opening and runs 9-17 ms. The hold ends 0.02-1.9 ms after it. App paints pause for those 10-19 ms. Dev already held paints for the recompose itself, so the new pause is the 0.6-1.3 ms before it starts plus that tail.
- **Ceiling.** A widget pump busy in a slow handler would hold every paint until it reached the rebuild. None of the rail's held widgets has such a handler. A pruned widget still flushes its next callbacks after its `Prune` message, so the hold ends.
- **The first fix (60e8db5578).** It called `app.delay_update()` (50 ms) in each widget. An earlier version of this note said the opening was not slower because "the recompose under load already takes 0.15-1.5 s". That was wrong: the 0.15-1.5 s was the full app's time from the opening to its first frame, while the recompose itself runs about 10 ms.
- **Review round 1 found that the 50 ms hold made `test_run_inspector_holds_a_rebuild_for_the_rails_opening` flaky** (3 of 32). While a batch is open, `Screen._on_idle` runs `call_after_refresh` callbacks before any layout, and the helper sampled after three fixed pauses.
- **The requested smallest-positive delay would have frozen the app.** `set_timer` skips a one-shot timer's only tick when its first two clock reads straddle the delay. Probe results: `delay_update(1e-6)` and `delay_update(1e-9)` left `_batch_count` at 1 for good; 1e-4 through 0.05 ended normally; 0.05 with a patched 60 ms stall between those reads stayed at 1. That is why the hold is now ended by the pump, not by a timer.

**AC#1, reproduction (dev 18482c1cf7, `-n 4`).** The instrumentation was temporary and is reverted. It recorded the frame facts, the held flags, the batch count and the paint's caller.
- Natural runs did not reproduce it: 130 frame checks, 0 failures. These were 1 single run, 24 + 90 repeats of the node (load 24-46), and 15 in 16 whole-file repeats (load 27-37).
- Every run had both rebuilds held with the post-send state at the opening. That state landed 1.9-5.1 s before the opening, which rules out a late state delivery.
- Deterministic, test-side nudge before the toggle. `loop.call_soon(screen._on_timer_update)` failed 1 of 1. `_update_timer.resume()` then `.pause()` failed 4 of 4; that is the screen's idle and Pilot's pause in one loop pass. Each failure had the recorded frames `[((True,), ()), ((True,), ()), ((False,), ('Run history',)), ...]`. At the stale frames the batch count was 0, and both widgets had a recompose queued but not started.

**AC#2, fix and pin.**
- New test `test_the_held_rebuild_lands_before_the_openings_first_paint[run inspector|section]` in `Tests/UI/test_console_inspector_rail_deferral.py`, a census whole-file entry. It queues the screen's tick ahead of the production cascade's opening and records every frame.
  - Red on dev in 16 of 16 runs (8 repeats x 2 params), on the frame assertion: the held rows were painted twice, then the rebuilt rows.
  - Green with the fix in 16 of 16.
- New head, with the pump-ended hold:
  - The new test, with the cascade's two hold lines removed: red 16 of 16. With them: green 16 of 16.
  - The full-app resume/pause nudge, without the hold: red 4 of 4 with the recorded frames. With the hold: green 8 of 8.
  - The deferral file, 32 whole-file repeats at `-n 4`, load about 17: 256 of 256 passed.
  - `_changed_while_hidden` and `_never_hidden` now wait for a settled rail: no batch open, no bounded-section reconcile pending, and the witness unchanged across two 50 ms pauses. They no longer take three fixed pauses. With a scratch 0.25 s `delay_update` added to the cascade, the six other deferral tests passed 48 of 48.
  - The parity file, 8 whole-file repeats at load 16-18: 23 of 24. The late-opened node passed 8 of 8, and the send-layout node 8 of 8. The chat-switch node failed once, at "temporary tab": the open-throughout arm sampled a row ("Setup: Not ready · no key") before it was laid out (region (-126, -6, 0, 0)). That arm never opens the rail, so it does not run the changed code. It is the chat-switch flake below.
- Full app with the first fix (60e8db5578):
  - The resume/pause nudge passed 8 of 8.
  - The plain node had 95 of 95 frame checks green in 96 repeats (load 15-49). The one failure was the outer fold (`console-inspector-rail-body` 23 rows against 22), TASK-37000.5. Dev's 96 repeats had 5 fold failures and 2 `assert not rail.display` failures (TASK-38644).
- Covering tests at the new head: the same 30 files failed 241 nodes. Every one also fails on dev. One dev failure, in `test_console_pending_display_owner_controls`, passed this time.
- Covering tests with the first fix: the 30 test files that use either widget, by node id against dev. The fix head failed 242 nodes and dev failed 244: the same 242 plus the new test's 2. 235 of the failures are the local `RecoveryRequired` trap (TASK-38111). Five rail-opening cases were run under a temporary `@private_profile_test` marker, since removed: 2 passed and 3 failed on both builds, with the same assertions.
- The sibling chat-switch case (`test_an_inspector_opened_after_chat_switches_matches_one_open_throughout`), 16 node repeats each at load 18-25: the fix head passed 16 of 16, and dev failed 1 of 16 on the staged-context tray's height. In dev's 16 whole-file repeats at load 27-37 it failed 9 of 16; proposed to the coordinator as a follow-up.
- `Tests/Architecture` size ratchets 205/205. Preflight passed.

**AC#3** does not apply: the failure was reproduced and fixed.

**Not changed.** `Tests/UI/test_console_collapsed_inspector_send_layout.py` is untouched (PR #3358 edits it). No User Guide page describes this one-frame glitch.

**Lesson:** `lessons-textual.md`, on `Timer.pause()` not stopping an already woken screen tick. It also covers why the hold must not end on a timer, and why a test must not sample after fixed pauses while a batch may be open.

**Files:** `UI/Screens/chat_screen.py` (the cascade; its size row was raised 24590 -> 24596 with a dated comment in `Tests/Architecture/test_screen_size_ratchet.py`), `Widgets/Console/console_run_inspector.py`, `Widgets/Console/console_inspector_section.py`, `Tests/UI/test_console_inspector_rail_deferral.py`, `backlog/docs/lessons-textual.md`.
<!-- SECTION:NOTES:END -->
