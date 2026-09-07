---
id: TASK-15509
title: Honor semantic focus theming in File Notes Git
status: Done
assignee: []
created_date: '2026-08-11 23:14'
updated_date: '2026-08-11 23:23'
labels: []
dependencies: []
documentation:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
modified_files:
  - tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py
  - tldw_chatbook/css/components/_agentic_terminal.tcss
  - tldw_chatbook/css/tldw_cli_modular.tcss
  - Tests/UI/test_library_file_notes_git.py
  - backlog/tasks/task-15509 - Honor-semantic-focus-theming-in-File-Notes-Git.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure File Notes Session Git focus styling inherits the application design system and active theme instead of masking it with a panel-local fixed color.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The File Notes Git panel no longer defines a local hard-coded focus background or foreground palette.
- [x] #2 Focused Git rows, buttons, and workflow regions resolve through the shared semantic focus tokens and honor theme-level overrides.
- [x] #3 Focus remains content-safe, visibly distinct, and keyboard usable at compact and desktop sizes.
- [x] #4 Focused tests validate computed or painted focus behavior across representative themes, and focused static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the panel-local focus-token override and the shared theme variable resolution contract.

2. Add mounted painted-output tests proving a representative theme override reaches focused Git rows and controls at desktop and compact sizes.

3. Remove the local palette duplication while preserving the panel's content-safe focus selectors.

4. Run focused regression tests, Ruff, mutation verification, and scoped diff review.

5. Complete acceptance criteria and record implementation notes and verification evidence.

ADR required: no

ADR path: N/A

Reason: This is a local conformance fix that applies the existing focus-token contract from ADR-011; it does not introduce a new theme boundary or interaction architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the Session Git panel's local fixed focus background and foreground declarations. Isolated widget mounts now use Textual's theme-responsive primary-background, text, and primary tokens; production app-tier selectors preserve the same content-safe row, button, and workflow focus treatment. Regenerated the modular CSS bundle. Added mounted tests that switch between textual-dark and agentic_terminal at 120x40 and 40x20 and verify computed row and button backgrounds, workflow outlines, non-obscuring row height, and intact button copy. Updated the retained Git-mode test to assert the labeled search row hides while its Input remains mounted. Verification: 6 focused focus/navigation tests passed; Ruff check passed; CSS generation passed; diff check passed. The behavior assertion was mutation-verified when routing the app-tier rules back through the fixed ds-focus-bg value produced the expected agentic-theme failure. Scope note: the global ds-focus-bg declaration itself is compile-time fixed, so this bounded fix uses Textual's shared active-theme tokens rather than changing the global focus contract for unrelated screens. ADR check: no new ADR; ADR-011 remains the governing workbench focus contract. No reusable lesson was added because the variable-resolution constraint is documented here and did not justify a repository-wide rule.
<!-- SECTION:NOTES:END -->
