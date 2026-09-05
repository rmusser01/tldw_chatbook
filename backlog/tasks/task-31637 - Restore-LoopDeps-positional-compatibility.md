---
id: TASK-31637
title: Restore LoopDeps positional compatibility
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 09:01'
updated_date: '2026-09-05 09:20'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the established LoopDeps positional constructor contract intact after token-budgeted tool disclosure added a new callback dependency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The eighth positional LoopDeps argument remains `call_model_with_continuation`.
- [x] #2 Token-budgeted tool disclosure still receives and invokes `replace_disclosed_names` when supplied by keyword.
- [x] #3 The complete Agents test directory passes.
- [x] #4 Scoped static and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the positional compatibility failure and inspect all LoopDeps construction sites.
2. Move the new defaulted disclosure callback to the append-only portion of the dataclass without changing runtime behavior.
3. Run the focused regression, the complete Agents test directory, and scoped static checks.

ADR required: no

ADR path: backlog/decisions/104-token-budgeted-agent-tool-disclosure.md

Reason: ADR-104 governs the disclosure callback behavior; this repair only restores the dataclass append-only compatibility convention and changes no runtime architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved `replace_disclosed_names` to the append-only tail of `LoopDeps`, preserving every established positional slot while retaining keyword-based disclosure updates. Added the type-check-only fallback import needed for scoped Ruff validation. Verification: the focused positional/disclosure regressions passed, `Tests/Agents` passed 2,581 tests, Ruff passed on all touched Python files, and `git diff --check` passed.
<!-- SECTION:NOTES:END -->
