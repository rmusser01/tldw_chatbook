---
id: TASK-37121
title: 'CI: reconcile the UI census ledger''s R11 rule with its bootstrap_profile files'
status: To Do
assignee: []
created_date: '2026-10-27 17:45'
labels:
  - ci
  - testing
  - docs
  - p3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
scripts/check_ui_pr_gate_census.py's ledger (around R11, TASK-32873) and several FAST_LANE / admission comments say bootstrap_profile tests belong in the PR Fast Lane's admission-sensitive step, not the UI census, because keep_bootstrap_profile enrollment can poison sandboxed suites sharing a process. In practice the census now lists many module-level bootstrap_profile files (86 by a pytestmark grep on 2026-10-27, including the three TASK-33620.15.15 added), presumably made safe since by the per-test profile-config restore. A reviewer reading the written rule would move or reject correct entries. Pre-existing; found in review of TASK-33620.15.15.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census checker's ledger and the workflow / Tests/CI comments state one rule for where a bootstrap_profile Tests/UI file goes, and it matches what the census actually lists
- [ ] #2 If bootstrap_profile files may sit in the census, the rule names what makes that safe and which test or guard enforces it; if they may not, the entries that break it are moved or named in a follow-up
- [ ] #3 A reviewer can tell from the checker alone where a new bootstrap_profile UI test belongs
<!-- AC:END -->
