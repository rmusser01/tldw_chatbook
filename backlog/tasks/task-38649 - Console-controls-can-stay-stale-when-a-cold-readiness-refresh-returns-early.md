---
id: TASK-38649
title: Console controls can stay stale when a cold readiness refresh returns early
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 05:51
updated_date: 2026-11-03 08:03
labels:
- follow-up
- console
- bug
priority: medium
references:
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
- Tests/UI/test_console_readiness_config_projection.py
- Tests/UI/test_console_send_ready_attachment.py
dependencies:
- TASK-38645
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-38645 (2026-11-03); not yet shown to be reachable.

When the readiness display is cold, `ConsoleReadinessConfigProjection.run` (console_spend_projection.py:384 on dev 18482c1cf7) starts a refresh worker and returns False without asking for a retry. The refresh (`_refresh`, :479) has several early returns: the screen is closing, the key moved during the read, the source or owner changed, or an exception (`except Exception: return`, :612). On those paths it neither publishes a whole sync nor asks for a retry. A lone `_sync_console_control_bar` that hit a cold display could then leave Send and the Workbench actions in their old state until some unrelated trigger runs another sync.

Whole syncs re-warm through `prepare_console_presentation_sync`, and `retry_after_config_rebuild` schedules a 0.2 s retry for some callers, so the gap may already be closed in practice. The task is to find out, and close it if it is open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test or probe establishes, for each early-return path of a cold refresh, whether the Console controls stay stale with no further trigger
- [x] #2 If any path leaves them stale: that path requests a retry or a sync, so the controls converge on their own; a test pins it and is shown red before the fix
- [x] #3 If no path does: the evidence is recorded in the notes and no product code changes
- [x] #4 A closing screen or disposed runtime still never schedules more work
<!-- AC:END -->

## Implementation Plan

1. Enumerate every return of `ConsoleReadinessConfigProjection._refresh` that ends without a publication: setup/read exceptions, closing/disposed (before and after the read), runtime rebound, source moved, key moved, pricing sources not current, owner moved after convergence, and any later exception.
2. In the census-listed mounted file `Tests/UI/test_console_send_ready_attachment.py`, mount the real Console, make the display cold with a real owner change, change what the controls must show, run one lone `_sync_console_control_bar()`, and inject a one-shot early return for each path. Then wait with no further trigger and record whether the Workbench and Send controls converge.
3. If a path stays stale, make that path request the module's existing coalesced retry (`_request_console_control_bar_sync(delayed=True)`, the same retry `run_console_presentation_sync` uses). Closing and disposed returns keep scheduling nothing.
4. Show the new mounted test red before the fix and green after it. Keep a closing-screen case that asserts no retry is scheduled.
5. Run the covering tests (`test_console_send_ready_attachment.py`, `test_console_readiness_config_projection.py`), the size ratchets for touched modules, and preflight. Update `Docs/Architecture/console.md` only where it describes this path.

## Implementation Notes

**Finding (AC#1).** All nine non-closing early returns of a cold `_refresh` leave the Console controls stale when nothing else triggers a sync. A lone `_sync_console_control_bar()` on a cold display returns False and relies on the refresh's own whole-sync publication. None of these returns publishes or asks for a retry.

The probe is the mounted test `test_a_cold_refresh_that_returns_early_still_converges_the_controls` in `Tests/UI/test_console_send_ready_attachment.py` (a census whole-file entry). Each case:
- makes the display cold with a real settings change (the owner key's settings revision);
- flips Send readiness;
- runs one lone sync with a one-shot fault injected inside the real refresh;
- watches the Workbench Send and composer Send for 3 s.

On dev 18482c1cf7 every case stayed stale, and `_refresh` requested nothing (`requests == []`). A diagnostic run that logged every `run()` caller showed zero further `run()` calls in 8 of 9 cases. The ninth was rescued once by an incidental context-snapshot read about 0.9 s later, which other runs did not have.

| Path (`_refresh` return) | On dev | After fix |
| --- | --- | --- |
| setup raises (session closed before entry) | stale | retry, converges |
| read raises | stale | retry, converges |
| runtime rebound after read | stale | retry, converges |
| reader returned no checked read | stale | retry, converges |
| config source moved during read | stale | retry, converges |
| owner key moved during read | stale | retry, converges |
| pricing sources moved | stale | retry, converges |
| pristine-default handoff raises | stale | retry, converges |
| owner moved during that handoff | stale | retry, converges |
| runtime disposed before read | nothing scheduled (correct) | nothing scheduled |
| screen closing (before or after read) | nothing scheduled (correct) | nothing scheduled |

**Fix (AC#2).** When `_refresh` fails or is refused, it now schedules `projection._retry` on the screen's own timer (`screen.set_timer`). `_retry` asks for the existing coalesced sync, `_request_console_control_bar_sync()`, which is the same sync `run_console_presentation_sync` defers to. That sync re-enters `run()`, which starts a new refresh, and that refresh's publication repaints the controls.
- The first wait is 0.2 s, the delay the coalesced retry already used.
- The wait doubles for each consecutive failure on the same owner key, up to 5 s (`RETRY_FIRST_DELAY`, `RETRY_MAX_DELAY`).
- A successful read or a change of owner key resets the wait to 0.2 s.
- Cancellation schedules nothing, because it belongs to whoever cancelled.
- A successful refresh is unchanged: it publishes when something changed, and otherwise publishes and schedules nothing.

**Review round 1.**
- **Important: the retry was unbounded.** The first version asked for the 0.2 s coalesced retry after every failure, with no limit. While a refusal lasted (for example `RecoveryRequired` from a storage pause, a non-stock reader, or replaced pricing sources), the screen paid a native config read plus a control-bar sync five times a second, even when idle. The backoff above replaces it.
- **Docs.** The `Docs/Architecture/console.md` paragraph now says "fails or is refused", states the backoff, and says that a successful refresh with nothing changed schedules nothing.
- **Census.** The closed-owner test and the new backoff test are appended at the `scripts/ui_pr_gate_census.txt` tail, and `MINIMUM_FILES` is now 1339.
- **Vacuous disposed case.** The mounted test's `disposed` case now records that the refresh actually reached its disposed return (`disposed:_refresh`) and its retry guard (`disposed:_retired`), and asserts both.

**AC#3** does not apply: paths were stale, so the product code changed.

**Closing and disposed (AC#4).** A closing screen or disposed runtime schedules nothing. `_retired()` makes the same check `run()` makes, both before the timer is scheduled and again when `_retry` fires. Textual also stops a screen's timers on unmount, and `_request_console_control_bar_sync` refuses a torn-down screen.
- Disposed: the mounted test's `disposed` case (runtime disposed after the worker is scheduled, before it starts) asserts that the refresh reached the guard and scheduled no timer.
- Closing: `test_closed_readiness_owner_never_reads_or_publishes[held]` now records `set_timer` calls. It is now census-listed. A mounted screen cannot be marked `_closing`, because Textual's message loop exits on it.

**Red and green evidence.**
- New mounted test red on dev: `stale after a cold early return: {'setup_raises': [], 'read_raises': [], ... 'owner_moved_in_handoff': []}`; green with the fix.
- AC#4 negative control (first version): dropping the closing/disposed guard turned the disposed case red (`[('retry', True)]`) and `closing=held` red (`[{'delayed': True}]`).
- A first version of the fix also retried on cancellation. `test_repeated_readiness_cancellation_waits_for_actual_raw_native_retirement` caught it: the fake screen's missing retry attribute masked the CancelledError. Cancellation was then excluded.
- Backoff pin `test_a_failing_cold_refresh_backs_off_and_a_publish_resets_it`. Its reader always fails, and its fake retry re-enters `run()` the way the screen's sync does. It is red against a22c39efcc (`assert 4952 <= 5` reads in 2.5 s, because the fake runs the requested sync at once) and green with the backoff: 2 to 5 reads, delays 0.2, 0.4, 0.8 and 1.6, and back to 0.2 after a successful read.
- The reviewer's probe (`cs-rev1-loop-probe.py`, whose fake gained one `set_timer` line, now `cs-impl2-loop-probe.py`) read 15 times in 3 s (5.0/s) at a22c39efcc. With the backoff it read 4 times in 3 s (1.3/s) and 8 times in 20 s (0.4/s); the steady state is one read per 5 s. Both read nothing after `_closing`.

**Verification, review round 1 (backoff code).** Load averages were 19 to 32 on 18 CPUs.
- The two edited files, plus `test_console_checked_display_scope.py` and `test_console_pricing_display.py`, ran three times: 100, 100 and 99 passed. Every run failed `mode_bar`, which fails on dev every run, and one run also failed `composer_resync`, a known flake that also failed on dev.
- 30-file covering set plus `Tests/Performance/test_console_keystroke_work_census.py`: 385 passed, 7 failed.
  - `test_console_pricing_display.py::test_changed_adapter_equality_is_not_invoked_after_background_read` reaches the refusal path through the shared `test_console_checked_display_scope._screen` fake, which had no `set_timer`. That fake now has one, and the test passes.
  - `test_console_storage_units_stay_within_their_ratchets[untested]` and `[tested]` fail identically at 18482c1cf7, a22c39efcc and the backoff code: "trace GC pass os_opens: 57 > ceiling 54 x 1.05". The rest of the keystroke census passes.
  - `mode_bar`, `syncs_never_wait…[full-ui-held]`, `settings_open_waits…` and `active_run_copy…` also fail on dev.
- Retry-recording files run again, together with `test_console_full_config_preparation.py` and `test_console_pending_attachment_stash.py`: every failure also fails on dev with the same command. No `set_timer` AttributeError appeared.
- Census and lane CI tests (`Tests/CI` shard replay, census checker, shard count, P0 gate, workflow): 353 passed.
- Size ratchets: 205 passed. `preflight.sh` passed; the census check reports 1339 entries against a floor of 1339.
- Ruff on the touched files reports only findings that dev already has.

**Verification, first version (a22c39efcc).** Load averages were 24 to 45.
- 30-file covering set: 5 failed, 375 passed, and all five also fail on dev.
- `mount_refreshes[roleplay]` flakes under load on both: fix 4 of 10 runs failed, dev 3 of 8, with the same assertion text.
- Second batch: 32 failed on each side, 31 of them the same tests, plus 2 flaky MCP approval tests.
- `test_a_staged_attachment_gets_the_same_send_readiness_from_either_sync_site` failed once with the fix (its direct, un-awaited sync found an expired display), then passed 5 of 5 in alternating fix/dev runs.

**Files.**
- `tldw_chatbook/UI/Console_Modules/console_spend_projection.py`
- `Tests/UI/test_console_send_ready_attachment.py`
- `Tests/UI/test_console_readiness_config_projection.py`
- `Tests/UI/test_console_checked_display_scope.py` (the fake gains `set_timer`)
- `scripts/ui_pr_gate_census.txt` and `scripts/check_ui_pr_gate_census.py` (two tail nodes; floor 1339)
- `Docs/Architecture/console.md` (the paragraph that describes this path)
- `backlog/docs/lessons-testing-evidence.md`
