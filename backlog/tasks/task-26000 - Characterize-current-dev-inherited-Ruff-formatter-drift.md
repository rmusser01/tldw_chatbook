---
id: TASK-26000
title: Characterize current-dev inherited Ruff formatter drift
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-30 15:39'
updated_date: '2026-08-30 16:22'
labels:
  - maintenance
  - formatting
  - quality
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md
documentation:
  - Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-22514 proved that its closeout introduced no Ruff formatter regressions while leaving a historical 61-file residue on its pinned base. Re-census current origin/dev and define conflict-safe atomic cleanup batches so formatter debt stops obscuring feature-owned changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pinned current `origin/dev` census records every Python file failing the repository-supported Ruff format check.
- [ ] #2 The current census is compared with the 61-file TASK-22514 historical residue and every difference is explained.
- [ ] #3 A mechanically checked batch manifest assigns every current failure exactly once to one atomic cleanup record; every record requires behavior preservation, and one final record requires an explicit repository-wide zero-exit Ruff format check after its lower-ID dependencies.
- [ ] #4 TASK-26000 changes no Python source; `git diff --check` over its recorded task boundary and `Tests/CI/test_backlog_task_id_uniqueness.py` pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recheck duplicate work, rebase only the TASK-26000 range onto current `origin/dev`, and record the exact task base, current pin, and common ancestor with TASK-22514's closeout branch.
2. Build and self-test a temporary standard-library census tool that records exact revision-local Git paths, blob IDs, Ruff exit-code status, blockers, configuration provenance, and aggregate control results without modifying any checkout.
3. Run whole-repository base/pre-closeout/closeout/common/current censuses, reconstruct TASK-22514's scoped `M/B/C/H` identity sets, resolve revision-path lineage, and prove the projected final-closeout invariant.
4. Generate one canonical point-in-time JSON manifest, mechanically derive every current classification, define owner-aligned stable batches, prove validator negative cases, and append the exact counts and stable labels to both plans before cleanup records exist.
5. Allocate collision-safe Backlog IDs, create every non-final cleanup record before the lower-ID-dependent final record, bind records to batches, and make the positive manifest checker plus task-ID guard pass.
6. Obtain independent subagent approval of the evidence, lineage, batches, and task contracts; verify and correct every finding before re-review.
7. Recheck current `origin/dev`, run documentation/evidence closeout gates, check all TASK-26000 criteria, add implementation notes, and mark only the characterization task Done.

Task 1 authority state (2026-08-30):

- `task_base`: `0ec518610cb50c4fa749bc97bc32761d4754cb81`
- `current_pin`: `0ec518610cb50c4fa749bc97bc32761d4754cb81`
- `common_ancestor`: `f0e8961222fe1a7a3ac7566f7f78142e717358f3`

ADR required: no.

ADR path: N/A.

Reason: the task records and schedules behavior-preserving formatter cleanup without changing runtime, storage, security, dependency, or cross-module architecture.
<!-- SECTION:PLAN:END -->

## Renumbering provenance

This formatter characterization task renumbered from `TASK-24653` to
`TASK-26000` under TASK-19601. The older holder,
`backlog/tasks/task-24653 - Network-TLS-trust-policy-corp-DPI.md` (Network TLS
trust policy (corp DPI)), keeps `TASK-24653`: it was created on 2026-08-29 22:51,
while this formatter task was created on 2026-08-30 15:39. Per the owner rule, the
younger task renumbers regardless of status. Citations to `TASK-24653` in
pre-renumber branch commits or documentation refer to this formatter task.
