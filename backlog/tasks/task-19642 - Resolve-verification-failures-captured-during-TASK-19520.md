---
id: TASK-19642
title: Resolve verification failures captured during TASK-19520
status: To Do
assignee: []
created_date: '2026-08-21 23:22'
labels:
  - testing
  - regression
dependencies: []
references:
  - backlog/docs/task-19520-verification-failure-inventory.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track and close every failing or erroring pytest node observed while verifying TASK-19520. The broad run was stopped at user direction after a partial result; these failures are not attributed to TASK-19520 without focused evidence. This parent coordinates symptom-scoped triage tasks and defensible mappings to existing open work; each child must establish its actual root cause before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All 301 observed nodes are mapped to a defensibly symptom-scoped child task or an existing open task.
- [ ] #2 Each mapped task either restores its focused tests or documents reproducible environment-only behavior with an appropriate marker or fixture contract.
- [ ] #3 The permanent failure inventory remains self-contained and updated as child tasks are resolved.
<!-- AC:END -->

## Initial Triage

- `TASK-19642.1` through `TASK-19642.28`, with atomic grandchildren under `.8`, `.19`, and `.21`, cover 297 nodes by the narrowest defensible shared symptom.
- `TASK-18801` already covers the two summarization diagnostic-boundary failures.
- `TASK-3070` / `TASK-3070.11` already cover the `chat_screen.py` size-ratchet failure.
- `TASK-18610` acceptance criterion 6 already covers the server-client provider migration-audit failure.
- The interrupted broad run is evidence of observed failures only; each child must establish a focused root cause before changing production behavior.

## Progress note (2026-10-11, t1-19642 slice)

Fifteen of this family's tasks were worked in the t1-19642 worktree (PR vs dev):
DONE with evidence -- 19642.5, .6, .7 (includes one genuine product fix in
console_agent_bridge.py), .8.1, .8.5, .8.7, .8.8, .14, .16, .18, .24, .25, .27.
Parents .8 and 19642 stay open: .8.2/.8.4/.8.6 and the non-Console children
(.9-.13, .15, .17, .19-.23, .26, .28) are still To Do / owned elsewhere.
Dominant root cause across the slice: the config-participant admission mask
(RecoveryRequired("raw_source_selection_changed"), TASK-32628/ADR-179 class),
fixed via Tests/conftest.py keep_bootstrap_profile enrollments; plus two
test-harness shell drifts, one genuine bridge bug, one import-chain defect
(TASK-19642.24), one stale-suite reconciliation (TASK-19642.14), and one
wall-clock race (TASK-19642.27).
