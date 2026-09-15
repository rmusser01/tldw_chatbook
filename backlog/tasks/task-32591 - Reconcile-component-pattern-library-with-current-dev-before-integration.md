---
id: TASK-32591
title: Reconcile component-pattern library with current dev before integration
status: To Do
assignee: []
created_date: '2026-09-15 00:18'
labels:
  - design-system
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-component-first-ui-audit.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The completed design-system branch cannot merge cleanly into current dev. The audit found five conflicts, including the source stylesheet and the obsolete generated Console sheet. Reconcile the histories while preserving the design-system contracts and intervening feature behavior before broader UI changes depend on this branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The merge candidate includes current dev and has no unresolved conflicts; the integration base and preserved feature changes are recorded.
- [ ] #2 Canonical ownership, source literal floors and generated bundle/split reproducibility pass on the merge candidate; current upstream Console rules remain represented in their owning sources.
- [ ] #3 Targeted Console and Library shell/file-notes tests covering the merged changes pass, with any demonstrated pre-existing failures identified separately.
- [ ] #4 The boot-CSS budget is met without raising its ratchet, and the proposed merge has unique Backlog task IDs.
<!-- AC:END -->

## ID allocation provenance

The local CLI initially offered TASK-32533. Before any references or publication, this audit reassigned it to TASK-32591 after a fresh all-local/remote-ref and worktree sweep found IDs through 32590. No existing task was changed.
