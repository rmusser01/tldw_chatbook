---
id: TASK-38649
title: Console controls can stay stale when a cold readiness refresh returns early
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 05:51
updated_date: 2026-11-03 06:58
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

**Fix (AC#2).** When `_refresh` ends without publishing, it now asks for the existing coalesced retry, `screen._request_console_control_bar_sync(delayed=True)`. This is the same retry `run_console_presentation_sync` defers to, and no new scheduler was added. The retry's sync re-enters `run()`, which starts a new refresh, and that refresh's publication repaints the controls. Cancellation is excluded because it belongs to whoever cancelled. Success paths are unchanged.

**AC#3** does not apply: paths were stale, so the product code changed.

**Closing and disposed (AC#4).** A closing screen or disposed runtime asks for nothing. The guard is the same check `run()` makes.
- Disposed: pinned by the mounted test's `disposed` case (runtime disposed after scheduling, before worker entry; `requests == []`).
- Closing: pinned by `test_closed_readiness_owner_never_reads_or_publishes[held]`, which now records retry requests. A mounted screen cannot be marked `_closing`, because Textual's message loop exits on it.

**Red and green evidence.**
- New test red on dev: `stale after a cold early return: {'setup_raises': [], 'read_raises': [], ... 'owner_moved_in_handoff': []}`; green with the fix.
- AC#4 negative control: dropping the closing/disposed guard turned the disposed case red (`[('retry', True)]`) and `closing=held` red (`[{'delayed': True}]`).
- A first version of the fix also retried on cancellation. `test_repeated_readiness_cancellation_waits_for_actual_raw_native_retirement` caught it: the fake screen's missing retry attribute masked the CancelledError. Cancellation was then excluded.

**Verification.** Load averages were 24 to 45 on 18 CPUs.
- Covering batch (30 files that exercise the projection), final code: 5 failed, 375 passed. All five also fail on dev:
  - `mode_bar` fails on dev every run (a fake lacks `.content`);
  - `fixture_failure_retires_held_transport[primary]`, `syncs_never_wait…[full-ui-held]`, `settings_open_waits…` and `active_run_copy…` failed in the dev rerun of the same nodes.
- An earlier covering run, with the first fix version, also flagged three more tests:
  - `repeated_readiness_cancellation…`, the cancellation case, which was fixed;
  - `composer_resync`, which also failed in the first dev run of the two edited files;
  - `mount_refreshes[roleplay]`, which flakes under load on both: fix 4 of 10 runs failed, dev 3 of 8, with the same assertion text.
- Second batch (control-bar quiescence users), run with the first fix version: 32 failed with the fix and 32 on dev, 31 of them the same tests. The two that differ are flaky MCP approval tests.
- `test_a_staged_attachment_gets_the_same_send_readiness_from_either_sync_site` failed once with the fix at load about 31: its direct, un-awaited sync found an expired display. It then passed 5 of 5 in alternating fix/dev runs (dev 9 of 9).
- Module and screen size ratchets: 205 passed. `preflight.sh`: all checks passed.

**Files.**
- `tldw_chatbook/UI/Console_Modules/console_spend_projection.py`
- `Tests/UI/test_console_send_ready_attachment.py`
- `Tests/UI/test_console_readiness_config_projection.py`
- `Docs/Architecture/console.md` (the paragraph that describes this path)
- `backlog/docs/lessons-testing-evidence.md`

