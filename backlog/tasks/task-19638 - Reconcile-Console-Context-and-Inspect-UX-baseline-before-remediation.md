---
id: TASK-19638
title: Reconcile Console Context and Inspect UX baseline before remediation
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:00'
updated_date: '2026-08-20 07:13'
labels:
  - console
  - ux
  - planning
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish a trustworthy latest-dev baseline for the Console Context and Inspect improvement programme so confirmed defects can be fixed without reversing accepted rail, source-ownership, scrolling, focus, or responsive decisions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A latest-dev ledger maps every critique finding to reproduced evidence, accepted decisions, existing tasks, and one disposition: implement now, research first, remove, or coordinate.
- [x] #2 TASK-14810's board status is reconciled with the implementation already present on dev, without claiming completion if its recorded Definition-of-Done blockers remain unresolved.
- [x] #3 Console user documentation accurately describes the current Sessions, Workspaces, Conversations, rail-control, and compact-width behavior without selecting the future IA.
- [x] #4 The next confirmed implementation slices are scoped as independent Backlog tasks with explicit dependencies and no overlap with TASK-18911 mobile ownership.
- [x] #5 No production UI behavior, persistence contract, or accepted ADR decision changes in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Freeze the review against the fetched `origin/dev` commit and classify all 24 findings by evidence, accepted decision, existing ownership, and disposition.
2. Reconcile TASK-14810 honestly by checking whether its recorded repository-level blockers are now resolved; leave it open with an explicit blocker note if they are not.
3. Correct only factual Console documentation for the current rail sections, directional controls, and compact-width floors.
4. File independent follow-up tasks for the exact 100-column geometry defect and Inspector overflow hint, each depending on this baseline task and excluding TASK-18911 mobile ownership.
5. Run backlog integrity, documentation, diff, and focused Console regression checks; record results and close only when the task's Definition of Done is supportable.

ADR required: no

ADR path: N/A

Reason: this task reconciles evidence, documentation, and Backlog ownership without changing UI behavior, persistence, application structure, or an accepted architectural decision.

Detailed plan and finding ledger: `Docs/superpowers/plans/2026-08-20-console-context-inspect-phase0-baseline.md`.
<!-- SECTION:PLAN:END -->

## Definition of Done

<!-- DOD:BEGIN -->
- [x] Acceptance criteria are complete and evidence-backed.
- [x] Relevant Console regressions and scoped Backlog/documentation guards pass or inherited baseline failures are identified precisely.
- [x] User documentation and the durable finding ledger are updated.
- [x] ADR applicability is recorded and accepted decisions are preserved.
- [x] The diff is self-reviewed and contains no production UI change.
- [x] Task status is set to Done.
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audited all 24 Context/Inspector findings against latest `origin/dev`, rendered
Textual evidence, current tests, and accepted decisions. The resulting ledger
separates two confirmed implementation defects from research hypotheses,
protected contracts, and TASK-18911 mobile coordination. It explicitly protects
TASK-400's Inspector ownership of staged Sources, TASK-15110's nested-scroll
ruling, TASK-16001's directional ASCII controls, ADR-043's responsive and focus
contracts, and ADR-017's Library-owned source staging.

Corrected the Console guide's obsolete Session-era information architecture,
rail labels, and compact-width promise. Reconciled TASK-14810 without closing it:
31 Confluence tests now collect and all eight rail-width-budget tests pass, but
the architecture gate remains red at 21,292 lines against a 17,727-line budget.
Filed independent follow-ups TASK-19639 (exact 100-column grid containment) and
TASK-19428 (bounded Context/Inspector sections and fold hints), both dependent
on this baseline and
excluding TASK-18911's mobile scope.

Verification: 15 focused Console rail/narrow-layout tests passed; scoped
frontmatter checks prove TASK-19638, TASK-19639, and TASK-19428 each occur exactly
once; documentation drift and `git diff --check` guards passed. The repository's
global Backlog-ID test remains red on unchanged `origin/dev` because TASK-17165
intentionally carries `id: taREDACTED-17165`; this task neither introduced nor
altered that security-redacted baseline. No Python, CSS, persistence, or runtime
behavior changed.

ADR required: no. ADR path: N/A. This was evidence, documentation, and Backlog
reconciliation only.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-18912, colliding with the older
"Reconcile-Console-Context-and-Inspect-UX-baseline-before-remediation" task that arrived on dev first.
Per the owner rule decided 2026-08-21 in TASK-19601 (**older id keeps it;
the younger task renumbers with a provenance note, regardless of Done
status**), it renumbered to TASK-19638. Citations to TASK-18912
in already-merged commit messages, ADRs, or code comments written before
2026-08-21 refer to THIS task; the other TASK-18912 holder is the
older arrival and keeps the id.
