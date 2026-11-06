---
id: TASK-38753
title: "Archive-recovery receipt refresh source-switch case asserts before the view syncs to the new chat"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-05 07:11
updated_date: 2026-11-05 22:06
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/test_console_archive_recovery_flow.py
- tldw_chatbook/UI/Console_Modules/archive_recovery.py
dependencies:
- TASK-38701
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

`Tests/UI/test_console_archive_recovery_flow.py::test_receipt_refresh_refusal_keeps_native_source_safe` (:1569), case `[source_switch]`, does this:
1. It holds the receipt Refresh's search on a thread.
2. It activates a new chat and runs `_sync_native_console_chat_ui()`.
3. It releases the search and waits until no search is left.
4. After a single `pilot.pause()`, it asserts that the old chat's receipt and results are gone (`assert not rig.console.query(".archive-recovery-receipt")`).

Nothing waits for the archive-recovery view to sync to the new chat. The one pause stands in for that sync.

TASK-38701's notes (PR #3404) record this case failing 2 of 9 runs at base, at that assertion. It never failed in 19 runs on the #3404 branch, and the mechanism was not established. If the view can keep the old chat's receipt mounted after a switch, this is a product bug, not only a test race.

The file runs on no PR lane on dev; #3404 adds two of its other nodes to the UI census.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes name the cause of the intermittent failure: either the test reads before the view's post-switch sync, or the product leaves the old chat's receipt mounted after the switch
- [x] #2 The case waits for the state it asserts instead of a fixed pause, or the product fix lands; either way the failure is reproduced before the change, under load or with a temporary delay
- [x] #3 The case passes 20 of 20 `-n 4` runs, with the load average recorded in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Instrument the case: the awaited sync's return value, and whether the archive-recovery view is bound to the new chat right after that sync and at the assertion.
2. Replace the single pause with a wait until the view is bound to the new chat, then assert the old chat's cards are gone.
3. Reproduce with a temporary deferral of the native sync before the view sync, on the old and new case.
4. 20 `-n 4` runs with the load recorded.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause (AC#1): the test reads before the view's post-switch sync.** It is not a product bug. A temporary probe in the `[source_switch]` case recorded three things: what the awaited `_sync_native_console_chat_ui()` returned after the switch, whether the archive-recovery view was bound to the new chat (`_session_id`) right after that call, and the same again at the assertion. It ran 18 times, at load 22-74.
- The awaited sync returned False in 10 of 18 runs.
- In 3 of 18, the view was still on the old chat right after it, with the old receipt mounted. The native sync had returned before reaching `_archive_recovery_view().sync()`. It has several early returns that leave the rest to a later sync, such as a refused presentation warm-up handing the whole sync to the delayed control-bar replay. The probe did not record which one fired.
- In all 18 runs a later sync had bound the view to the new chat by the assertion, with no receipt or results left.

The view never re-mounted the old chat's cards after binding to the new one: `_mount_receipt` and `_mount_results` refuse a session that is no longer active. So the base failure (2 of 9 runs) was the single `pilot.pause()` landing before that later sync.

**Change (AC#2).** After the search drains, the case waits (the file's `wait_until`, 15 s) until the view is bound to the new chat. Then it asserts that the old chat's receipt and results are gone. A product that kept the old cards after its sync would still fail those asserts.

**Reproduced with a temporary delay.** A temporary edit made `config_sync.prepare_console_presentation_sync` return False from the switch until 1.5 s after the search drained, so the view's sync landed late. The old case failed 2 of 2 runs at `assert not rig.console.query(".archive-recovery-receipt")`, the recorded failure. The new case passed 2 of 2 under the same edit (load 20-26). Releasing the hold right as the search drained did not reproduce it: the pending replay then landed inside the one pause.

**AC#3.** The `[source_switch]` case passed 20 of 20 `-n 4` runs, at load 22.1-42.0 (1-minute average).

**Files.** `Tests/UI/test_console_archive_recovery_flow.py`.
<!-- SECTION:NOTES:END -->
