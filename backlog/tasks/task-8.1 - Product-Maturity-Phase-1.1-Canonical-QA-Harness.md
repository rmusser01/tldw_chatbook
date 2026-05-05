---
id: TASK-8.1
title: 'Product Maturity Phase 1.1: Canonical QA Harness'
status: In Progress
assignee: []
created_date: '2026-05-05 15:11'
updated_date: '2026-05-05 15:12'
labels:
  - product-maturity
  - phase-1-qa-baseline
dependencies: []
parent_task_id: TASK-8
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the reusable product-maturity QA protocol, template, evidence index, severity mapping, and smoke evidence so later usability work can be verified against the running app rather than render-only checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Product-maturity QA protocol defines clean-run setup, entry commands, terminal-size matrix, severity mapping, and evidence rules.
- [ ] #2 Product-maturity QA template captures environment, entry path, steps, visual/focus notes, functional result, defects, evidence, residual risk, and exit decision.
- [ ] #3 Product-maturity tracker links the spec, Backlog tasks, Phase 1.1 evidence, and residual risks.
- [ ] #4 Focused pytest coverage verifies the protocol, template, tracker, and Backlog anchors exist and preserve the harness-only boundary.
- [ ] #5 Harness smoke evidence states that Phase 1.1 verifies the QA harness only and does not complete the full product walkthrough.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the focused product-maturity QA harness regression test and confirm it fails before harness docs exist.
2. Create the product-maturity tracker, QA evidence root, Phase 1 protocol, template, README, and harness smoke evidence.
3. Run the focused harness test and adjacent Unified Shell QA protocol regression.
4. Update Phase 1.1 tracker and evidence status after verification.
5. Mark Phase 1.1 acceptance criteria complete with implementation notes while leaving Phase 1 open.
<!-- SECTION:PLAN:END -->
