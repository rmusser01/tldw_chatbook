---
id: TASK-39162
title: "test_console_fleet_survivor_tick.py waits on fixed 0.5-2.5 s pauses before it asserts timer state, so its nodes fail under load"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 16:49
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/test_console_fleet_survivor_tick.py
dependencies:
- TASK-32470
- TASK-39120
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39120 on round-12 branch fu/r12-ctests; its notes propose this follow-up). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08. Line numbers are dev's; the file is the same on the stack.

**The fixed pauses.** The file waits a fixed time and then asserts:
- `pilot.pause(0.5)` at `:162` ("let the mount hedge fire against idle"), `:179`, `:222` and `:249`;
- `pilot.pause(2.5)` at `:186` and `:258` (">= 2 survivor-tick beats");
- `pilot.pause(1.5)` at `:226` and `pilot.pause(0.6)` at `:282`.

**What failed.** TASK-39120's runs, on base 8b762b52fa:
- `test_survivor_tick_observes_new_published_usage_without_a_new_timer` is `@private_profile_test`, so it runs plain. It failed at `_console_fleet_survivor_timer is None` after `pilot.pause(0.5)`.
- Under a temporary `bootstrap_profile` marker at load 50-60, `test_the_tick_stops_itself_with_one_final_settle_paint` and `test_survivor_elapsed_advances_with_no_other_interaction` failed their timing preconditions.

**Scope.** TASK-39120 owns the file's seven `RecoveryRequired` nodes (its AC#1); this task is only the waits. TASK-32470 (To Do) also lists this file's survivor-elapsed and unseen-mark tests as harness drift. No PR lane runs the file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each fixed pause before an assertion is replaced by a bounded wait on the state that assertion reads (the survivor timer, the poll or the paint), which fails naming that state
- [x] #2 Under a temporary `bootstrap_profile` marker, the file passes 10 of 10 whole-file runs at `-n 4`, with load recorded
- [x] #3 A temporarily broken wait condition fails on its bounded wait, not on a later assertion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a file-local bounded `_until(predicate, what, timeout)` that polls with `asyncio.sleep` (not `pilot.pause()`, which waits on every pump) and fails naming the state.
2. Observe the 0.3 s mount hedge itself: wrap the hedge callback `ChatScreen.on_mount` schedules (`set_timer(0.3, _maybe_start_console_fleet_survivor_tick)`) so the test can wait for it to have fired against idle.
3. Replace each fixed pause before an assertion: hedge fired (`:162`), poll self-stopped (`:179`, `:282`), survivor timer armed (`:222`, `:249`), the advanced paint (`:186`), the published usage (`:226`), the timer stopped and the terminal glyph painted (`:258`).
4. Negative control: break one wait condition temporarily and show the run fails on that bounded wait, not on a later assertion.
5. Whole file 10 of 10 at `-n 4` under a temporary `bootstrap_profile` marker, load recorded.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test-only. Base dev ae9f9e8d21, 2026-11-10.

**What changed.** `Tests/UI/test_console_fleet_survivor_tick.py` gains `_until(predicate, what, timeout=15)`, which polls with `asyncio.sleep` (not `pilot.pause()`) and fails naming the state, and `_observe_mount_hedge(monkeypatch)`, which wraps the one `ChatScreen.set_timer` callback named `_maybe_start_console_fleet_survivor_tick` (the 0.3 s mount hedge) so a test can see it has run. Each fixed pause is now a wait on what the next assertion reads:
- `:162` and `:282`: the mount hedge has fired (against idle);
- `:179`, `:222`, `:249`, `:282`: the transcript poll has self-stopped (`_until_poll_self_stops`);
- `:222`, `:249`: the survivor timer is armed;
- `:186`: the row paints `1m `; `:226`: the row shows the new usage, then the same timer is asserted;
- `:258`: the survivor timer is gone, then the row paints `✓`.

**Red before (base, temporary module `bootstrap_profile` marker, `-n 4`).** Two whole-file runs at load 47-62 each failed 3 of 8: the poll still running after 0.5 s (`Timer(0.2, name='set_interval#1') is None`, at `:182` and `:285`) and the survivor timer not yet armed (`:223`, `:252`).

**AC#3.** A temporary edit made `_maybe_start_console_fleet_survivor_tick` return at once (load 41-54). Each timer test failed on its bounded wait: "precondition: the survivor tick armed at the poll's stop edge (waited 15 s)", "the survivor tick never armed at the poll's stop edge (waited 15 s)" (private-profile child), and "the elapsed segment froze with no other interaction: '● researcher · 3s' (was '● researcher · 1s') -- task-15664's defect (waited 15 s)". Reverted.

**AC#2.** Under the temporary marker, 10 of 10 whole-file runs at `-n 4` passed (8 passed each, 78-142 s), one-minute load 44-101 across the runs (`run.sh`, label r13d-e-rep). The marker is removed again: plain, the seven non-private nodes still fail locally with `RecoveryRequired: raw_source_selection_changed` (the known local trap; TASK-39120 owns them), and the private-profile node passes.

No PR lane runs the file; no criterion asks for one.
<!-- SECTION:NOTES:END -->
