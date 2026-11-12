---
id: TASK-40322
title: "Lane test_console_sync_tick_storage.py and test_personas_subscription_readiness.py, whose 8 nodes fail under the plain command"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- console
- personas
- lanes
priority: medium
references:
- Tests/UI/test_console_sync_tick_storage.py
- Tests/UI/test_personas_subscription_readiness.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-38948
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (send stream, TASK-38948 reviews r13n-rev1 and r13n-rev4). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Both files guard round-13 fixes and run on no PR lane, so a regression would land silently.
- `Tests/UI/test_personas_subscription_readiness.py::test_handoff_readiness_refreshes_expiry_and_stale_snapshot_without_edits` (`:207`) is the only guard against the Personas Start-chat gate flickering to "Checking Claude", which round 13 fixed. All 8 nodes of the file (5 functions, one parametrized 4 ways) fail under the plain command with RecoveryRequired locally, on base and head alike, so the file was not added to the census: TASK-38948 verified the re-pin only under a temporary `@private_profile_test` marker (green 3 of 3 at load 37-50).
- `Tests/UI/test_console_sync_tick_storage.py` (2 tests, module `bootstrap_profile`) guards the once-a-second Character-scope check; review r13n-rev4 fixed its `_tick` to wait out an owed whole-sync replay (f85a33f521; 0 of 48 fails against 17 of 48 unfixed).

Neither path appears in `scripts/ui_pr_gate_census.txt`, `.github/` or `scripts/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Personas file passes under its plain command at `-n 4` (a profile marker or a harness fix, with the RecoveryRequired cause recorded)
- [ ] #2 Both files run on a required PR lane at the census tail, with lane times recorded
- [ ] #3 Both pass 3 of 3 whole-file runs at `-n 4`
<!-- AC:END -->
