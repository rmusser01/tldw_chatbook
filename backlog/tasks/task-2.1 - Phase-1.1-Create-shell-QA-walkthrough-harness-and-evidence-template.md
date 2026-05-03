---
id: TASK-2.1
title: 'Phase 1.1: Create shell QA walkthrough harness and evidence template'
status: Done
assignee: []
created_date: '2026-05-03 14:48'
updated_date: '2026-05-03 15:02'
labels:
  - unified-shell
  - phase-1
  - qa
  - harness
dependencies:
  - TASK-1.1
documentation:
  - Docs/superpowers/specs/2026-05-03-unified-shell-maturity-tracking-design.md
  - Docs/superpowers/qa/unified-shell/phase-1/walkthrough-protocol.md
  - Docs/superpowers/qa/unified-shell/phase-1/walkthrough-template.md
  - >-
    Docs/superpowers/qa/unified-shell/phase-1/2026-05-03-phase-1-protocol-smoke.md
  - Docs/superpowers/trackers/unified-shell-maturity-roadmap.md
parent_task_id: TASK-2
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the repeatable running-app walkthrough protocol and evidence template needed before shell destinations can be marked usable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Walkthrough template captures visual, keyboard, click, functional, severity, evidence, and residual-risk fields.
- [x] #2 Harness or manual protocol can be run against the actual Textual app.
- [x] #3 At least one smoke walkthrough summary is stored under Docs/superpowers/qa/unified-shell/phase-1/.
- [x] #4 The roadmap links the harness/template evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression test that validates the Phase 1 QA protocol/template contains the required walkthrough fields and product-QA guardrails.
2. Run the focused test and confirm it fails before the template exists.
3. Add the reusable Phase 1 walkthrough protocol, QA template, and an initial smoke summary under Docs/superpowers/qa/unified-shell/phase-1/.
4. Link the Phase 1.1 evidence from the roadmap.
5. Run the focused test and Backlog/docs verification commands.
6. Check acceptance criteria, add implementation notes, and commit the slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a Phase 1 shell QA walkthrough protocol, reusable evidence template, and protocol smoke summary under Docs/superpowers/qa/unified-shell/phase-1/. Added a focused docs regression test that fails when the protocol/template/smoke evidence or roadmap links are missing. Verified the protocol regression and supporting mounted master-shell navigation test under the repo Python 3.12 venv. Product destination workflow audit remains in TASK-2.2.
<!-- SECTION:NOTES:END -->
