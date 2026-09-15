---
id: TASK-32591
title: Reconcile component-pattern library with current dev before integration
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 00:18'
updated_date: '2026-09-15 00:45'
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
- [x] #1 The merge candidate includes current dev and has no unresolved conflicts; the integration base and preserved feature changes are recorded.
- [x] #2 Canonical ownership, source literal floors and generated bundle/split reproducibility pass on the merge candidate; current upstream Console rules remain represented in their owning sources.
- [x] #3 Targeted Console and Library shell/file-notes tests covering the merged changes pass, with any demonstrated pre-existing failures identified separately.
- [x] #4 The boot-CSS budget is met without raising its ratchet, and the proposed merge has unique Backlog task IDs.
- [x] #5 Python files changed by the integration pass fatal syntax and undefined-name checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/161-component-pattern-library.md (existing); backlog/decisions/150-design-token-system-and-design-language.md
Reason: reconcile existing implementations without changing their architecture or design-language contracts.

1. Refresh origin/dev and merge its exact commit into the existing component-pattern worktree with no automatic merge commit.
2. Preserve upstream Library test coverage and behavior; transplant upstream Console styling into the decomposed owning sources and tokenize any newly introduced fixed values.
3. Rebuild generated bundle/split styles; run ownership, literal-floor, reproducibility and boot-budget checks plus the affected Console and Library tests.
4. Review the integration diff against both parents, check Backlog ID uniqueness, record exact evidence and commit the verified merge candidate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled origin/dev fd30614dcdc1e6cbd39b1532769d3e10be9b12b6 with design head 705a39bdb8, preserving all 89 added/changed upstream style declarations in their decomposed owners and rebuilding generated artifacts. Corrected two incoming style-floor violations, the retired Console split reference, multiline-comment selector parsing, thirteen Library harness stylesheet pins, and one type-checking import. Refreshed the diagnostic manifest for five previously deleted widgets (ten removed calls). Older dev TASK-32532 retained its ID; the younger component-pattern family and twelve subtasks are now TASK-32596, with references updated.

Evidence: 68 successful governance/build checks followed by the repaired guard in a 30-pass targeted harness run; 36 targeted layout checks; fatal Ruff across 277 changed Python files. Boot CSS is 612,733/634,050 bytes without a raised budget. All derived-artifact checks pass (Mermaid checked with network access to pinned inputs). Native scratch-profile startup rendered at 120x40 and 80x24; no provider request sent. Existing incoming documentation whitespace is preserved, and historical design-branch whitespace remains tracked by the audit. No full test suite run.

ADR check: existing ADR-161 and ADR-150 apply; no new architecture decision. Full details, limitations and selected evidence: Docs/superpowers/reports/2026-09-14-component-integration.md.
<!-- SECTION:NOTES:END -->
