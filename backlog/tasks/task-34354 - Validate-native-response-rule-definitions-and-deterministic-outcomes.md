---
id: TASK-34354
title: Validate native response rule definitions and deterministic outcomes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:53'
updated_date: '2026-10-12 23:14'
labels: []
dependencies: []
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/228-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 1. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Closed native definitions reject unsupported fields and enforce all size limits.
- [ ] #2 Chat and Workspace exclusions mask logical rules while revisions remain pinned.
- [ ] #3 Literal and Markdown heading checks distinguish violations, unavailable checks and inapplicable rules.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/228-console-learned-response-rules.md
Reason: direct implementation of accepted ADR-228.
Follow implementation plan Task 1: failing boundary/scope/check tests, immutable models and scope resolution, deterministic evaluation, targeted verification and commit.

Current-dev PR3010 integration: follow Docs/superpowers/plans/2026-10-11-pr3010-current-dev-integration.md on the actual post-prior-merge base. ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: preserve the approved local storage/recovery, helper custody and shared native/hook acceptance; source ADR219 remapped after the older claimant. Reproduce reviewed failures before fixes; qualify actual next schema, six voice residues, exact private/index/table catalogs, preparation-through-transport custody, lazy first-use, current Goal/Personal Context owners and separately labelled real-provider/native UI evidence. Historical source qualification above is retained; reopened acceptance awaits current integrated verification and independent review.

2026-10-12 released Task1 on private START13b204cb638ca891175610f5e687a80def71c21e. ADR required: yes (existing). ADR path: backlog/decisions/228-console-learned-response-rules.md; preserves ADR227 Goal and current Personal Context/native ownership. Implement complete91path source workflow by current symbol. Qualify actual schema80-to81/four catalog variants/all predecessors, six voice residues, seven tables/three query/index pins/private import purpose, preparation-through-transport physical custody and original-owner usage, current Goal/hook/queue atomic receipts, lazy first-send/global-before-Chat, manager/draft/Stop/currentness and finite native worker retirement. Preserve all bounds and source history. Use targeted RED/GREEN; measure bounded critical cases for existing required job capacity, preserving inherited targets/order/caps/processes with executable membership negatives. Separate actual local-model/external-hook/native paint from scripted evidence. Root owns actual-base reconciliation, independent reviews, hosted proof and task closure; all AC remain open.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented closed immutable rule definitions, scope precedence and deterministic literal/Markdown-heading checks. Unknown and skipped results remain distinct from pass; private input bodies stay out of repr. Added rule package and three test files plus fixtures. ADR-228 governs the change. Watched the missing boundary fail; 43 feature cases and 36 command regression cases pass; black py312, ruff and mypy pass. Self-review completed.

2026-10-12 current integration (historical source evidence above remains unchanged): Integrated the closed response-rule definition and deterministic detector contracts with current Console owners. Declared behavior fixtures remain separately labelled from real model outcomes. Existing ADR228 governs, with ADR097 live-revision GC and ADR227 Goal constraints applied. Current report: E/pr3010-current-integration/task-1-report.md. All tasks stay In Progress and acceptance remains open pending independent review, actual-base/hosted qualification and unmet external/model checks.
<!-- SECTION:NOTES:END -->
