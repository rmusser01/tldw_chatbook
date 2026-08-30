---
id: TASK-25706
title: Make submitted-log regression coverage truthful on Windows
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 17:52'
updated_date: '2026-08-30 18:50'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct Windows-only failures exposed by the native submitted-log validation so the matrix distinguishes portable behavior from intentionally POSIX-only security contracts and reports actionable evidence without weakening ADR-029.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The submitted-log regression matrix completes on native Windows without pytest internal errors.
- [x] #2 POSIX-only profile-migration security tests are capability-gated and Windows retains a tested fail-closed contract.
- [x] #3 Windows-illegal filename fixtures do not mask SQLite URI coverage.
- [x] #4 The optional-datasets notice test ignores unrelated expected platform diagnostics.
- [x] #5 The compact selection-menu case is independently diagnosed and either fixed or proven stable on Windows.
- [x] #6 Durable CI coverage exercises the corrected Windows contract.
- [x] #7 Generated diagnostic inventories have the same deterministic path order on Windows and POSIX hosts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the native Windows run as RED evidence and add focused platform-contract tests for timeout selection, filenames, optional-datasets logging, and profile-migration fail-closed behavior.
2. Apply narrow test capability gates without changing ADR-029 or weakening production privacy checks.
3. Run the compact selection-menu case independently on Windows with actionable geometry diagnostics and fix only a reproduced product defect.
4. Make diagnostic inventory ordering platform-independent and keep the floating menu contained after late layout measurement.
5. Add durable native-Windows submitted-log coverage with uploaded structured results.
6. Run targeted local tests, the native Windows matrix, static checks, and self-review; document results and close the task.

ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: ADR-029 already defines Windows privacy as unverified pending separately approved native ACL work; this task corrects test and CI portability while preserving that runtime boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the submitted-log regression surface platform-truthful without changing ADR-029's fail-closed Windows privacy boundary. POSIX descriptor/ACL migration tests now use a capability gate, Windows directly proves both migration destinations reject without residue, illegal `?` filename cases no longer obscure pure URI-builder coverage, pytest chooses its native timeout method, and the optional-datasets notice test asserts only the notice it owns.

Native follow-up exposed two additional cross-platform defects: diagnostic inventory generation inherited Windows case-folded `Path` ordering, and the floating selection menu could finish its first clamp before late CSS measurement. Inventory paths now sort by their serialized POSIX spelling with regression coverage, and menus re-clamp after resize; transcript event assertions also wait for the owning async handler to finish bubbling.

The existing nightly Windows leg now runs under `PYTHONIOENCODING=cp1252`, while other legs remain UTF-8. Native Windows validation at commit `ee83b50b4c` passed the isolated compact-menu probe and the complete submitted-log matrix: 601 passed, 145 intentionally skipped, 746 total ([run 33328585489](https://github.com/rmusser01/tldw_chatbook/actions/runs/33328585489)). Local CP1252-targeted UI, SQLite/runtime, manifest-boundary, and inventory-order tests passed; Ruff, workflow YAML parsing, diagnostic inventory verification, Backlog task-ID validation, and `git diff --check` passed.

ADR required: no. Existing ADR: `backlog/decisions/029-local-private-data-boundary.md`.

Updated tests and infrastructure in `.github/workflows/nightly-deep.yml`, `Tests/`, `scripts/check_persistent_diagnostic_inventory.py`, `tldw_chatbook/Widgets/Console/console_selection_menu.py`, and the testing-evidence lessons ledger.
<!-- SECTION:NOTES:END -->
