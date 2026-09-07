---
id: TASK-15483
title: 'Harden File Notes navigation, warnings, and review-transition focus'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 19:38'
updated_date: '2026-08-11 20:19'
labels:
  - notes
  - library
  - ux
  - accessibility
dependencies: []
references:
  - >-
    .impeccable/critique/2026-08-11T06-03-15Z__ok-widgets-library-library-file-notes-workspace-py.md
documentation:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/029-file-notes-disk-authority.md
  - backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
  - backlog/decisions/035-file-notes-session-git-index-controls.md
priority: medium
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the remaining low-severity findings from the File Notes Impeccable critique so navigation language, keyboard guidance, warning semantics, and the commit-review focus transition are consistent and trustworthy without changing local-file or Session Git authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Linked-root warnings use the existing warning semantic token while retaining explicit warning text
- [x] #2 File Notes uses one consistent navigator-return phrase across the workspace and Session Git surfaces
- [x] #3 Keyboard guidance uses the repository's standard grouped hint grammar instead of pipe-separated telemetry
- [x] #4 The medium-width commit-review transition is stress-repeated and either fixed with a deterministic focus regression or documented as non-reproducible with repeat evidence
- [x] #5 Focused mounted tests and targeted static checks pass without changing disk, replica, staging, commit, or push behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; conform to `backlog/decisions/011-chatbook-workbench-ui-system.md`, `backlog/decisions/029-file-notes-disk-authority.md`, `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`, and `backlog/decisions/035-file-notes-session-git-index-controls.md`.
Reason: this is a scoped semantic-color, navigation-copy, guidance-contract, and focus-verification pass within existing UI, disk, and Session Git authority.

1. Characterize current dev and stress-repeat the medium-width commit-review footer transition before changing focus behavior.
2. Add mounted regressions for warning-state class and semantic token use, one navigator-return phrase, the already-current grouped key-guide grammar, and stable review focus.
3. Apply warning styling and navigation-copy changes only; change focus logic only if the repeated test reproduces a deterministic defect.
4. Run focused workspace and Git panel tests, mutation-check the new guards, and run targeted Ruff, Python compilation, CSS integrity, and diff checks.
5. Record dissolved and non-reproduced findings honestly, complete task documentation, and close TASK-15483 only when every criterion is evidenced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a linked-root warning state class that uses the existing $warning token plus bold text while retaining the explicit Warning label, so color reinforces rather than carries the state. Standardized both editor and Session Git return controls on Back to navigator. Current dev already used the approved middle-dot key-guide grammar, so that critique observation was treated as dissolved and pinned with a mounted regression instead of duplicating a change.

The reported medium-width commit-review focus failure reproduced only when the test rendered review before Textual completed its one-time screen autofocus; the late autofocus selected the scroll owner. After settling mount, 25 consecutive mounted form-to-review transitions preserved focus on Edit message. The existing footer test now models that lifecycle, and the new stress regression fails when the production review-focus handoff is removed. No runtime focus logic or disk, replica, staging, commit, or push authority changed.

Verification: 52 File Notes workspace tests passed; 5 focused Session Git tests passed; 100 shared non-obscuring focus-contract tests passed; 9 CSS integrity tests passed. Warning-state, navigation-copy, and review-focus guards were mutation-checked and failed when their production behavior was removed. Targeted Ruff, Python compilation, and `git diff --check` passed. Only existing dependency, SQLite privacy, and pytest-asyncio warnings were reported.

ADR required: no. The implementation conforms to ADR-011, ADR-029, ADR-031, and ADR-035. Modified the File Notes workspace plus its workspace and Session Git mounted regression suites. No new lessons entry was warranted; the autofocus distinction is recorded in the regression comment and this task note.

Qodo follow-up identified that a clean scan or reconcile result did not clear a warning retained from the preceding result. Both service-result adoption paths now replace the warning state unconditionally, and regressions cover warning-to-clean transitions in the unmounted state path and the mounted root-status surface. The focused state regression and direct mounted warning regression pass; targeted Ruff, Python compilation, and diff checks pass.
<!-- SECTION:NOTES:END -->
