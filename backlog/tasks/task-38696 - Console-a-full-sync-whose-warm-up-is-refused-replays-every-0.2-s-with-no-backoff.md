---
id: TASK-38696
title: "Console: a full sync whose warm-up is refused replays every 0.2 s with no backoff"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 11:40
updated_date: 2026-11-04 05:54
labels:
- follow-up
- performance
- console
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
dependencies:
- TASK-38649
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-38649's review, PR #3372, 2026-11-03). Checked by reading the code on dev 9bbd5c2515 and at #3372's head; not measured.

A full Console sync awaits `prepare_console_presentation_sync` (config_sync.py:100-140) before it renders. That function returns False at once while the screen is maintenance-paused or closing (:127-129). Otherwise, for the stock disk-loaded projection (it returns True early for a config not loaded from disk, :131-132, or a non-standard projection, :134-138), it returns `ConsoleReadinessConfigProjection.warm()` (:140, console_spend_projection.py:462). When it returns False, the sync sets `_console_control_bar_replay_whole_sync` and asks for a delayed control-bar sync (chat_screen.py:17847-17850). That request is a fixed `set_timer(0.2, ...)` (chat_screen.py:20847), so the whole full sync replays.

`warm()` itself refuses up front only for a closing screen or a disposed runtime (console_spend_projection.py:464-467). After its read settles, it returns False when the key has moved or the projection has no fresh value.

Each replay calls `warm()` again, which calls `self.run(...)` and so can start another config read. So while a refusal lasts (the cases TASK-38649 lists: `RecoveryRequired` from a storage pause, a non-stock reader, replaced pricing sources), the screen can pay a full sync and a read about 5 times a second.

During a maintenance pause the same 0.2 s timer is also the resume path. `_run_coalesced_control_bar_sync` re-arms itself while the screen is paused (chat_screen.py:20888-20896), and `_console_sync_maintenance_resume` leaves the replay to that timer (:17665-17672). A backoff must not delay or drop that resume.

TASK-38649 bounds the projection's own retry (backoff from 0.2 s to 5 s, one shared timer). This replay path is separate, and #3372 does not touch it.

Depends on TASK-38649 (in PR #3372).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 While warm-up keeps refusing, full-sync replays back off instead of repeating every 0.2 s, and a pin counts replays over a refusal of several seconds and fails on the base build
- [x] #2 A refusal that clears still lets the next full sync render promptly (pinned), a closing screen schedules no replay, and a full sync requested during a maintenance pause still runs exactly once after resume (pinned)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Count consecutive warm-up refusals in `prepare_console_presentation_sync` (`config_sync.py`); any True return resets the count; pause/closing returns leave it alone.
2. The full sync's refusal replay asks for a backed-off delay (0.2 s doubling to 5 s, TASK-38649's constants); a maintenance pause keeps the fixed 0.2 s path.
3. A pending backoff timer is cut short by an immediate control-bar request (the projection's own retry) and by maintenance resume, so neither a clearing refusal nor the resume waits out the backoff. Logic lives in `config_sync.py`; `chat_screen.py` stays within its size ratchet.
4. Pins (in a PR-lane file): replays over a multi-second refusal (red on base), prompt render once the refusal clears, a closing screen arms no replay, and a sync requested during a pause runs exactly once after resume.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach (all logic in `UI/Console_Modules/config_sync.py`).**
- `prepare_console_presentation_sync` counts consecutive warm-up refusals on the screen (`_console_warm_refusals`); any True return resets it, and the pause/closing early returns leave it alone.
- `full_sync_replay_delay` picks the refused FULL's replay delay: the first refusal and a maintenance pause keep the fixed 0.2 s timer (so a cold mount's single refusal, and the pause's resume poll, behave exactly as before); each further refusal doubles the wait on TASK-38649's constants (0.4, 0.8, 1.6, 3.2, then 5 s).
- `request_control_bar_sync` now holds the coalescing that `ChatScreen._request_console_control_bar_sync` did, plus a numeric delay. A pending backoff timer absorbs delayed requests as before, but an immediate request (the readiness projection's own backed-off `_retry`, any user-driven control-bar request, and `_console_sync_maintenance_resume`) stops it and runs the replay now. That is how a refusal that clears renders without waiting out the backoff, and how a resume is not delayed by one.
- `chat_screen.py`: the refusal site asks for `full_sync_replay_delay`, `_request_console_control_bar_sync` delegates (a torn-down screen still schedules nothing), and the resume makes an immediate request instead of returning. Net 24,601 -> 24,600 lines, 741 methods.

**Decision.** The first replay stays non-preemptible: an earlier cut made every refusal's timer preemptible, and an immediate request then ran a fresh screen's replay earlier than base, which changed the order of work in the H3 settlement tests. The backoff only needs to be cut short once it is longer than the fixed timer.

**Evidence** (new tests in `Tests/UI/test_console_projection_reads_off_loop.py`, a UI-census whole-file entry; load 16-30 on 18 CPUs):
- `test_a_refused_full_sync_backs_off_and_renders_once_the_refusal_clears` (AC#1, AC#2 promptness and reset): red on base c2cdb6b82e, "a 3 s warm-up refusal ran 14 full syncs at [0.03, 0.25, 0.46, ...]" (`assert 14 <= 6`); green with the fix (4-5). With preemption disabled (temporary edit) the promptness part is red: "the cleared refusal rendered 5.05 s after it was asked, after its pending 5.0 s backoff had run out".
- `test_a_full_sync_asked_for_during_a_pause_runs_once_after_resume` (AC#2 resume): base has no backoff to delay it; with the fix but without the resume's immediate request (temporary edit) it is red, "the resume's sync waited 0.80 s, out to its pending 0.8 s backoff"; green with the fix, exactly one replay after resume.
- `test_a_closing_console_arms_no_replay_and_counts_no_refusal` (AC#2 closing): an invariant base keeps; red with the torn-down check removed (temporary edit), "a closing Console armed a replay".
- Whole file: 11 passed (`-n 4`). The 44 files that name the replay/coalescing symbols (plus the cards file): 332 failed / 1080 passed on the branch against 328 failed at base, and the difference is nodes that also flake at base. `test_console_pending_attachment_stash.py::test_fresh_mounted_screen_settles_late_h3_outcome_in_dom_and_controls` fails 2-3 of 3 at base and 1-3 of 3 here, always with its own recorder's `TypeError: ... unexpected keyword argument 'delayed'`. Three `test_console_native_chat_flow.py` nodes (`more_delete_double_activation`, `original_attempt_delete`, `rejected_regenerate`) failed in 5 of 9 node runs at base and 4 of 9 here. Most of the 328 base failures are in files that no lane runs and that fail locally.

**Review fix (2026-11-04, after a9fd04edbf).**
- **Problem.** While the replay flag was set, every explicit full sync deferred to the pending backoff. About 90 direct callers do this, including a user's switch and the readiness projection's publication of a fresh read. So after a refusal cleared, the Console could stay unrendered for up to 5 s. The projection's own retry backs off on the same doubling, so it rarely cut in first. The first promptness pin missed this because it made the immediate request by hand.
- **Fix.** `config_sync.cut_replay_backoff` is called from the full sync's deferral gate (one line in `chat_screen.py`, now 24,601 lines, at its row; 741 methods). When only the replay deferred an explicit, non-poll full sync and a backoff timer is pending (0.4 s or more), it makes an immediate control-bar request, which runs the replay now. The first 0.2 s replay, maintenance pauses, durable-commit holds and the cold core-state replay are unchanged.
- **Decision: the run poll's ticks do not cut.** They come every 0.2 s while a run is live, so cutting would bring back the 0.2 s replay loop (a full sync and a config read each time) for any refusal that overlaps a run. A refused FULL also renders nothing either way: preparation refuses before the transcript projection. Painting resumes on the first explicit full sync, the projection's own retry, or the backoff timer, whichever comes first.
- **Decision: a chat or model switch restarts the count.** The count now covers only consecutive refusals of one unmoved readiness owner key (session, its settings revision, workspace, config identity). A refusal for another chat or model, or one whose key moved during its own warm-up read, starts over at the fixed 0.2 s. TASK-38649's projection retry likewise restarts its delay for a new key.
- **Evidence** (red at a9fd04edbf, green after):
  - The promptness part of the first test now uses a plain full-sync worker: "rendered 5.04 s after it was asked, after its pending 5.0 s backoff had run out".
  - New `test_a_successful_readiness_read_during_a_backoff_renders_at_once` (the projection's real read publishes): "3.23 s ... pending 3.2 s backoff".
  - New `test_a_chat_switch_restarts_the_refusal_backoff[before_the_read|during_the_read]` asserts the refused FULL's own delay: `[1.6]` in both.
  - The AC#1 backoff pin stays green.
  - Whole file: 14 passed in 4 runs.
  - **Re-review Minor 1 (fleet survivor tick).** The one-second survivor tick (`fleet.py`, wired in `wiring.py`) ran its full sync as an explicit one, so while a background sub-agent outlived the run poll it cut a lasting refusal's backoff every second. Its sync now goes through `config_sync.run_periodic_console_sync` (`wiring._run_fleet_survivor_sync`), so it is a poll. The tick runs only while the run poll is stopped, so `_held_for_durable_commit(poll=True)` never holds it, and its projection follows the poll's unchanged-input rule. New `test_the_fleet_survivor_tick_does_not_cut_a_warm_up_backoff`, red at a28249f8ed: "the survivor tick ran the replay 1 time(s) inside its pending 3.2 s backoff". `test_console_fleet_survivor_tick.py` passes 8 of 8 twice, under a temporary `@private_profile_test` marker that was then removed (plain, 7 of its 8 tests hit the known local `RecoveryRequired` trap). `test_console_fleet_wake_ui_freshness.py::test_wake_turn_in_a_nonviewed_session_flips_the_tab_glyph_off_running` fails 2 of 2 at base and here.
  - Covering set after the fix: 332 failed / 1083 passed. No new failure beyond nodes that also fail at base: `test_console_delete_confirmation_resets_when_selection_changes` failed 3 of 3 at base, and the delete/regenerate/H3 nodes listed above. Serial `Tests/Architecture`: 43 failed, the same 43 nodes as base, including the wave6 inventories; `test_screen_size_ratchet` passes.

**Files.** `tldw_chatbook/UI/Console_Modules/config_sync.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/UI/test_console_projection_reads_off_loop.py`.
<!-- SECTION:NOTES:END -->
