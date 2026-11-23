---
id: TASK-42468
title: "The character presentation cadence test's cancelled case is red on dev: a cancelled refresh lets the old display fingerprint pass as current"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- console
- characters
- tests
- dev-red
priority: medium
references:
- Tests/Backup_Recovery/test_character_presentation_cadence.py
- tldw_chatbook/UI/Console_Modules/character_context.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23): TASK-42426's covering runs found it red 1 of 3 parameters, on base and branch alike, twice each. Re-run on the round-22 stack before its rebase (142e821116) at `-n 4`: still red. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches). Paths are under `tldw_chatbook/` unless they start with `Tests/`.

**The failure.** `Tests/Backup_Recovery/test_character_presentation_cadence.py::test_direct_refresh_outcome_cannot_certify_an_old_display[cancelled]` fails at line 472 with `AssertionError: old fingerprint certified a refresh that never successfully published`. On `ConsoleCharacterContextController` (`UI/Console_Modules/character_context.py`), `_successful_refresh_generation` is None and `_generation` is 2, yet the old fingerprint is still accepted. The test's other two parameters pass. The file last changed in 12c803b093 (2026-10-16).

**Why it matters.** If this is a product defect, a cancelled character refresh lets a display that was never republished be treated as current, so the Console could show stale character context as fresh. It may instead be a test whose expectation drifted; that is not known. The file is in no PR lane (not in `scripts/ui_pr_gate_census.txt` or a `derived-artifacts.yml` step), so nothing reports it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is recorded: test drift after a named product commit, or a product defect in how a cancelled refresh certifies the display
- [ ] #2 A defect is fixed in product code (a cancelled refresh never certifies the old fingerprint), with the `[cancelled]` case as its red-first test; drift is repaired in the test with its assertion kept
- [ ] #3 All three parameters pass, alone and in a `-n 4` run of the file
- [ ] #4 The file runs in a PR lane, with its serial time under the lane's flags recorded in the notes
<!-- AC:END -->
