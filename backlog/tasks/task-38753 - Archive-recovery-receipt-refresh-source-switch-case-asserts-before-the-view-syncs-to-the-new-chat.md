---
id: TASK-38753
title: "Archive-recovery receipt refresh source-switch case asserts before the view syncs to the new chat"
status: To Do
assignee: []
created_date: 2026-11-05 07:11
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
- [ ] #1 The notes name the cause of the intermittent failure: either the test reads before the view's post-switch sync, or the product leaves the old chat's receipt mounted after the switch
- [ ] #2 The case waits for the state it asserts instead of a fixed pause, or the product fix lands; either way the failure is reproduced before the change, under load or with a temporary delay
- [ ] #3 The case passes 20 of 20 `-n 4` runs, with the load average recorded in the notes
<!-- AC:END -->
