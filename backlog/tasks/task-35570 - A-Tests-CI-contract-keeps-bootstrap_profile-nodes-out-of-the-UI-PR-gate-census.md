---
id: TASK-35570
title: A Tests/CI contract keeps bootstrap_profile nodes out of the UI PR gate census
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

Nothing enforces the rule. Tests/CI checks only the P0 targets, and `_BOOTSTRAP_OK` still admits the census. During Phase 9, eleven bootstrap_profile nodes across six tasks entered the census unnoticed. The capture-phase review found them, and they were moved by hand in b4a4c8db7c.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Tests/CI contract fails when any node or whole file listed in scripts/ui_pr_gate_census.txt carries the bootstrap_profile marker, and names each offender and the admission step it belongs in
- [ ] #2 The contract passes on the census as this task leaves it (any offender it finds is moved to the admission step first), and a negative control shows it fails when a known bootstrap_profile node is added to the census
- [ ] #3 The lessons file or the census header says where a bootstrap_profile UI test goes, so a new test lands in the right lane first time
<!-- AC:END -->
