---
id: TASK-35570
title: Decide whether bootstrap_profile tests may run in the UI PR gate census, and enforce it
status: To Do
created_date: 2026-10-18 23:20
dependencies: []
labels:
- model-config-redesign
- follow-up
- ci
priority: low
references:
- scripts/ui_pr_gate_census.txt
- scripts/check_ui_pr_gate_census.py
- Tests/CI/test_ci_queue_pressure_contract.py
- Tests/CI/test_console_p0_regression_gate.py
- .github/workflows/derived-artifacts.yml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 9's ledger ruling R11 (TASK-33002.17) says a bootstrap_profile test goes to the PR Fast Lane's admission step (ADMISSION_TARGETS), never to the UI PR gate census. The reason is TASK-32873: enrolling keep_bootstrap_profile in a shared pytest run can poison sandboxed suites' config access, which shows up as RecoveryRequired.

The repo does not follow that rule as a whole, and nothing decides it.
- At Phase 9's head the census holds 73 bootstrap_profile entries, all from dev, some of them whole files. Some are there on purpose: the TASK-34785 block says "Three node entries cover 13 bootstrap-profile cases".
- The Console P0 gate's `_BOOTSTRAP_OK` (Tests/CI/test_console_p0_regression_gate.py) lists the UI Fast Lane census as an allowed lane for bootstrap tests.
- So the census shards already mix both kinds and pass. Whether the TASK-32873 hazard applies to census shards at all is unproved.

During Phase 9, this phase's own new census entries were made to follow R11: eleven nodes were moved in b4a4c8db7c, and the whole-PR review moved a few more. One more stays: test_settings_discovered_model_toggle.py is listed whole, with one bootstrap node inside.

Moving all 73 would push them into the single 35-minute admission job, while the census is sharded for time. This task first decides the rule, then enforces whichever rule is chosen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rule is decided and recorded in the census header and the testing lessons file. Either census shards may hold bootstrap_profile tests, with the TASK-32873 hazard shown not to apply to them, or they may not, with a timed plan for the existing entries
- [ ] #2 A Tests/CI contract enforces the decided rule. If bootstrap tests may not be in the census, it fails on any bootstrap_profile census entry outside a pinned allowlist of today's entries
- [ ] #3 A negative control in the contract's test shows that it fails when an entry breaking the rule is added
<!-- AC:END -->
