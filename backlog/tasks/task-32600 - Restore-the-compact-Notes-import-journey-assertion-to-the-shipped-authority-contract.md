---
id: TASK-32600
title: >-
  Restore the compact Notes import journey assertion to the shipped authority
  contract
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 02:48'
updated_date: '2026-09-15 04:46'
labels:
  - library
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-library-workflow-audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 60x20 variant of test_database_notes_import_once_journey_is_painted_focused_and_retained stops at an obsolete requirement that library-notes-authority repeat Library notes. Below 64 columns, _authority_prefix intentionally omits that prefix because the immediately preceding source strip already names Library notes and Folder files (task-32360). The 120x36 variant passes. This early stale assertion leaves the compact import journey unverified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The 60x20 journey verifies the visible source strip and meaningful authority status under the existing compact contract, then completes its import, focus and retained-receipt assertions.
- [x] #2 The 120x36 journey continues to verify its complete authority copy and import behavior.
- [x] #3 The repair retains assertions that distinguish Library-owned notes from folder files and does not restore redundant product copy simply to satisfy an obsolete test.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the 60x20 authority assertion failure and confirm the 120x36 chooser and real import baseline.
2. Assert compositor-painted source ownership and complete status at both sizes, preserving the below-64-column copy contract and chooser focus/retention checks.
3. Run the real SQLite review/cancel/import journey at both sizes and verify the saved receipt can be reopened without another import.
4. Run targeted import and compact-copy regressions, lint/format changed tests, self-review, and record bounded evidence before closing the task.

ADR required: no
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md (existing); backlog/decisions/150-design-token-system-and-design-language.md (existing); backlog/decisions/161-component-pattern-library.md (existing)
Reason: test-only alignment with the shipped task-32360 compact authority contract and stronger coverage of existing import behavior; no new UI, storage, or architectural contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the stale compact authority assertion without changing product code or copy. Both terminal sizes now verify the painted source strip, selected Library authority and complete status. Existing ownership copy, chooser focus and mounted-pane retention assertions remain.

Extended the adjacent real SQLite review/cancel/import test to 60x20 and 120x36. It also returns to Notes and reopens the identical painted receipt through focused keyboard actions, with one persisted note and no duplicate import. This supplies the receipt coverage that the original chooser-only test did not contain.

Verification: baseline 1 failed / 2 passed; final targeted selection 10 passed. Changed test functions pass Ruff formatting; whole-file diagnostic comparison exactly matches six pre-existing Ruff findings (none added). Diff whitespace check passed. Two unrelated pytest cleanup warnings concern old Kokoro test directories. Production CSS and disposable SQLite authorities were used; file-picker return and cancellation timing remain controlled fixtures. No full suite or native file-picker claim.

Modified the two journey tests and appended evidence/scope to Docs/superpowers/reports/2026-09-14-library-workflow-audit.md. Self-review completed. No product, performance, security, dependency or license behavior changed. No new general lesson or ADR needed; existing ADR-086, ADR-150 and ADR-161 govern the unchanged contract.
<!-- SECTION:NOTES:END -->
