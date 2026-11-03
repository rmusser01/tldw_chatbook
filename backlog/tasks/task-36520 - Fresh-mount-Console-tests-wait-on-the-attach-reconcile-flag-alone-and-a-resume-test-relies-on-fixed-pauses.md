---
id: TASK-36520
title: Fresh-mount Console tests wait on the attach-reconcile flag alone, and a resume test relies on fixed pauses
status: In Progress
assignee:
- '@claude'
created_date: 2026-10-24 00:26
updated_date: 2026-11-03 08:33
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/pump_probe.py
- Tests/UI/test_console_cost_chip_screen.py
- Tests/UI/test_console_send_draft_snapshot.py
- Tests/UI/test_console_approval_compact_layout.py
- Tests/UI/test_console_command_composer.py
dependencies:
- TASK-36389
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36389's review (PR #3233, 2026-10-23). TASK-36389 audited tests that reopen a cached Console. It added `console_attach_ended(screen)` to `Tests/UI/pump_probe.py`, which waits for the reconcile to end and for the Console's queue to drain. Tests on a fresh mount, and one resume test, were outside its scope.

These wait only for `_console_attach_reconciled` before they watch or assert attach-time work (dev 5b5920b8e2):
- `test_console_cost_chip_screen.py:1174-1180` polls the flag, then wraps `_sync_console_settings_summary` and `_sync_console_cost_chip` in counting mocks. Its own comment says the attach's resume refresh must land first "or it races into the watched window".
- `test_console_send_draft_snapshot.py:355-359` polls the flag before it arms the send it measures.
- `test_console_approval_compact_layout.py:35-37`: `_pending_card` asserts the flag before it injects a synthetic approval.

The reconcile runs `on_screen_resume` and then sets the flag (chat_screen.py:15360-15371). `on_screen_resume` queues more work with `call_after_refresh`, such as the hook refresh worker (:22731-22737), so that work can still be pending when the flag reads True.

`test_console_command_composer.py::test_console_resume_triggered_prompt_insert_survives_stale_session_switch` (:1859) relies on fixed sleeps:
- `pilot.pause(0.1)` (:1886) is meant to let the mount-time sync settle the draft tracker.
- `pilot.pause(0.25)` (:1904) is described as "past the 0.15s consumption timer".

Of the four files, only `test_console_approval_compact_layout.py` is in the UI census.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each named test waits for the attach to end and the Console's queue to drain before the work it watches or asserts, or the task notes record why it cannot race
- [x] #2 The prompt-insert test waits for the states it asserts instead of fixed pauses
- [ ] #3 A negative control that delays the reconcile's start, as TASK-36389's probes did, fails each old form and passes each new one
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read what each named test watches or asserts after its attach wait, and what the attach (the reconcile and the `on_screen_resume` it runs) still has queued when `_console_attach_reconciled` flips.
2. Put `Tests/UI/pump_probe.py::console_attach_ended(console)` before that work in the cost-chip burst test, the armed-send snapshot test and each compact-approval journey; keep `_pending_card`'s flag check as a guard.
3. In the prompt-insert resume test, wait for the attach to end and then for the draft tracker on the first session, instead of `pause(0.1)`; after the resume, poll the handoff's consumption and the inserted draft instead of `pause(0.25)`.
4. Negative control: a temporary edit in `chat_screen.py` that starts the attach reconcile late (as TASK-36389's `HA_DEFER` probe did, without holding the screen's pump); run each old form (temporary copies) and each new form under it; a second temporary edit lengthens the 0.15 s consumption timer for the prompt-insert test; revert both.
5. Run the four files' named nodes and the changed files with `-n 4`, the plain nodes through a temporary `@private_profile_test` marker if `RecoveryRequired` blocks them; preflight; notes, ticks, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** Every named test now calls the existing `Tests/UI/pump_probe.py::console_attach_ended(console)` (drain the Console's queue, wait for `_console_attach_reconciled and not _console_attach_reconcile_running`, drain again; raises naming the stage) before the work it watches or asserts. `pump_probe.py` is unchanged.
- `test_console_cost_chip_screen.py::test_idle_draft_edit_burst_coalesces_one_cost_refresh`: replaces the flag poll, before the spend callbacks are wrapped and watched.
- `test_console_send_draft_snapshot.py::test_console_armed_unknown_mouse_send_snapshots_before_skill_await`: the wait moved ahead of the arming (draft, stored draft, `_console_unknown_send_armed`); it used to poll the flag after arming. Placed after the arming instead, the drain changed the measured send: it took the unknown-command hint path (no notice) in 4 of 4 runs, so the arming has to be the last thing before the send.
- `test_console_approval_compact_layout.py`: each of the three journeys waits after `_wait_for_production_console_ready`, before its first `_pending_card`; `_pending_card` keeps its flag check as a guard.
- `test_console_command_composer.py::test_console_resume_triggered_prompt_insert_survives_stale_session_switch` (AC#2): `pause(0.1)` became the attach wait plus a wall-clock poll (`pump_probe.until`, 5 s) for the visible-draft tracker on the first session; `pause(0.25)` became a poll for the handoff consumed and the insert in the draft.

**Why two of them cannot race (AC#1).** The negative control below never failed their old forms (at 1.0 s, 3 of 3 serial and 1 of 1 with `-n 4` each; at 2.5 s, 1 of 1 each).
- Cost chip: the only call that fails the test is an unattributed one, and the attach's own direct `_sync_console_cost_chip()` is in `on_screen_resume`, which the reconcile runs before it sets the flag. What the attach queues after the flag (hook refresh, control-bar sync) reaches the spend callbacks through `_sync_console_rail_and_controls` / `_sync_console_control_bar_under_config`, which the test attributes; a probe printed those rows in the watched window for the old and the new form alike, and both passed.
- Approval journeys: `_wait_for_production_console_ready` already waits up to 10 s for the flag and for no Console sync in progress, requested or running, and `_pending_card` sets the card's state synchronously; nothing the attach queues afterwards rewrites it.
The waits stay converted: they are what the attach owes these tests, and they can fail. With the reconcile start deferred 8 s, the new cost-chip and prompt-insert forms raised `Console attach reconcile never ended ... reconciled=False`.

**Negative controls** (temporary edits in `chat_screen.py`, reverted; patches in the scratchpad). C: `_start_console_attach_reconciliation` returns at once for the first N seconds of each visit and a `set_timer` calls it again, so the reconcile starts late without holding the screen's pump (TASK-36389's `HA_DEFER`). H: `CONSUMER_SETTLE_HEDGE_SECONDS` 0.15 s -> 1.0 s. Old forms ran from temporary copies of the base files (plain nodes under a temporary `@private_profile_test` copy), one node per pytest, load 11-21.
- Prompt insert, C at 1.0 s: old red 5 of 5 (`assert not True`: the handoff was still pending after `pause(0.25)`, because the resume ran before the attach had ended and only re-armed it); new green 5 of 5.
- Prompt insert, H at 1.0 s: old red 5 of 5 (same line); new green 5 of 5. At 0.45 s the old form failed only 1 of 5: its pause plus Pilot's screen wait usually outlasted the timer.
- Snapshot, C at 1.0 s: old red 5 of 5 with notices `[]` (the send was diverted to the unknown-command hint and never reached dispatch); new red 5 of 5 with `['Draft changed; Send again.']`, the same failure as dev without any control (below).
- Cost chip and approval, C at 1.0 s: old and new green 3 of 3 each (see "cannot race").

**AC#3 is not met, so this task stays In Progress.** The snapshot test is red on dev 66155632ed, 4 of 4 runs, with no control: `assert await send_task, notices` fails with `['Draft changed; Send again.']`. The test types " suffix" while the skill context is held, and `Console_Modules/wiring.py` (from 3e618d7119) refuses a send whose live draft generation or edit serial differs from the stash. That is a product/test disagreement outside this task. No lane runs the file. Until it is settled, the new form cannot pass under the control. The control does tell the two forms apart, though: the old form's send never reaches dispatch, and the new one's behaves as on dev. The cost-chip and approval old forms cannot be made to fail by a delayed start, since they cannot race (above). AC#3 is shown for the prompt-insert test.

**Runs (no control).** Named nodes with `-n 4`: 4 passed, snapshot red as on dev. Serial: cost chip 5 of 5, prompt insert (temporary private-profile copy) 5 of 5, approval journeys 8 of 8 (load 19-29). The four files whole with `-n 4`, head and base copies in one run (load ~24): the snapshot file's 5 reds are the same on both; the cost-chip file had 11 reds at head and 6 at base. Three further paired runs of those cost-chip nodes (`-n 4`, load 11-12) failed the same nodes on both trees: `test_active_edit_cancels_an_already_armed_idle_cost_timer` 3 of 3 on both, the two system-prompt revert tests intermittently on both, and `test_excluded_historical_media_keeps_next_send_priced` on neither. That last test did fail at head in two earlier runs (4 params in one `-n 4` run at load 33, 1 param in 1 of 2 serial runs) and never at base in the runs paired with them; it shares only the module with this change, which touches the burst test and adds an import (`assert None is not None` in `_next_send_dollars`: no priced next-send label yet). **A separate flake in the burst test, in both forms.** In pytest processes that ran the approval journeys after it, the burst test failed with `assert not errors` / `["KeyError: 'self'"]`: the base copy 5 of 6, the new form 3 of 6 (load 8-12). Run alone it passed every time. A frame dump of one failure showed the cause. `_sync_console_identity_surfaces` is decorated by `retry_after_config_rebuild`, so the test's `ordinary` map holds that decorator's shared `wrapped` code, whose frame has no `self`. A config-rebuild retry (`retry_after_config_rebuild.<locals>.wrapped.<locals>.retry`, run from `_flush_next_callbacks`) that refreshes the summary inside the watched window trips the lookup. This is not an attach effect, so it is a proposed follow-up, not part of this task. The composer file's 52 plain nodes fail locally on both trees: the `RecoveryRequired` trap (TASK-38111). `ruff check` finds the same counts as base in all four files. `test_console_command_composer.py` is `ruff format`ted as at base; the other three were not formatted at base and only the touched lines changed.

**Files.** `Tests/UI/test_console_cost_chip_screen.py`, `Tests/UI/test_console_send_draft_snapshot.py`, `Tests/UI/test_console_approval_compact_layout.py`, `Tests/UI/test_console_command_composer.py`. Census untouched.
<!-- SECTION:NOTES:END -->
