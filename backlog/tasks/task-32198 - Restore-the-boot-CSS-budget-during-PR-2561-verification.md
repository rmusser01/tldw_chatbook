---
id: TASK-32198
title: Restore the boot CSS budget during PR 2561 verification
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 04:14'
updated_date: '2026-09-10 04:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 2561 inherits a dev CSS byte-budget breach that prevents its required performance check from passing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Boot CSS remains below the existing byte limit with identical CSS rules and source ownership.
- [x] #2 Generated CSS reproduces and the targeted boot budget check passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/097-boot-budget-ratchets.md. Reason: comment-only source trim preserves existing stylesheet rules and runtime boundaries. 1. Record the failing 805669-byte census. 2. Compact explanatory comments in an existing boot stylesheet without changing tokens. 3. Regenerate CSS and verify identical non-comment content, bundle sync, and the unchanged budget guard.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Compacted decorative headers and a verbose choice-list comment in features/_wizards.tcss, then regenerated tldw_cli_modular.tcss. Non-comment content is byte-identical against dev in both files. Boot CSS is 803696 bytes, below the unchanged 804000-byte cap (previously805669). Targeted guard and all derived-artifact checks pass; no CSS rule, ownership, dependency or limit changed. Existing ADR097 boot-budget ratchet applies.
<!-- SECTION:NOTES:END -->
