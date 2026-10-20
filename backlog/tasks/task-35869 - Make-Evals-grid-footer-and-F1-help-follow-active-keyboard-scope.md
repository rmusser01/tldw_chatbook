---
id: TASK-35869
title: Make Evals grid footer and F1 help follow active keyboard scope
status: In Progress
created_date: 2026-10-20 19:25
labels:
- evals
- textual-patterns
- ux
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md
assignee:
- '@codex'
updated_date: 2026-10-20 21:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evals must advertise results-grid commands only where keyboard dispatch can reach them, and make their scope discoverable through F1 when narrow layouts hide footer detail.

### Evidence and bounded scope

evals_screen.py:867-927 registers l/b/s/e by selected run-group kind without focus context. Their actual bindings belong to ResultsGrid at UI/Evals/results_grid.py:455-460. A profile-free Textual harness using the actual extracted registration method and bindings advertised all four commands; l invoked the grid action with grid focus and did nothing after focus moved to the rail. The actual grid was absent from the rail binding chain. F1 at evals_screen.py:2491-2528 lists only mode/F6/F1 keys. This verifies dispatch mechanics, not a complete mounted Evals journey.

Book basis: Chapter 8 printed pp.154-158 / PDF pp.163-167 (focus/binding scope), and printed pp.161-162 / PDF pp.170-171 (dynamic actions).

### Existing ownership and decisions

TASK-32887 completed base Evals orientation/help; TASK-31223 is the completed Library focus-scoped precedent; TASK-1064 has different remaining work. TASK-35678 is the merged Settings-only consolidation. No global action registry or Console work belongs here.

Related tasks: [TASK-32887](./task-32887%20-%20Lab-Evals-skeletal-F1-help-leaked-binding-identifiers-undiscoverable-mode-strip-keys.md); [TASK-31223](./task-31223%20-%20Footer-never-advertise-keys-the-focused-widget-will-swallow.md); [TASK-1064](./task-1064%20-%20Register-of-issues-identified-but-not-fixed-during-the-Evals-settings-session.md); [TASK-35678](./task-35678%20-%20Share-Settings-action-availability-across-shortcuts-buttons-and-help.md).

ADR required: no new ADR expected.
ADR paths: [031-tui-keybinding-and-footer-hint-conventions](../decisions/031-tui-keybinding-and-footer-hint-conventions.md); [152-roleplay-mode-keys-single-letters-nav-layer-reserved](../decisions/152-roleplay-mode-keys-single-letters-nav-layer-reserved.md).
Reason: No new ADR is expected for truthful presentation of existing widget-owned actions. Preserve binding ownership, reserved shell keys and explanatory paths rather than promoting grid keys globally.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Footer grid hints reflect actual reachability for grid/table controls, Catalog rail, Inspector and focused text entry, and update on focus, selection and readiness changes.
- [ ] #2 F1 exposes supported grid commands with an honest current scope or explicit focus requirement, including narrow layouts, without contradicting the footer.
- [ ] #3 Lens, baseline, sort and export still dispatch in their intended grid context; printable characters remain text inside editors and no global binding is added.
- [ ] #4 Availability/help projection adds no configuration or database reads to key dispatch or ordinary focus changes and does not duplicate category/action lists.
- [ ] #5 A mounted Evals test exercises real run-group selection, F6/Tab focus changes, command dispatch and compact/wide footer/F1 behavior; existing base-help tests remain valid.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/152-roleplay-mode-keys-single-letters-nav-layer-reserved.md
Reason: Project truthful scope for existing widget-owned Evals actions, preserving global keys and ownership.

- Add mounted actual-Evals tests selecting a real run group, moving focus via F6/Tab among grid/rail/inspector/text controls and exercising the real grid actions. Cover compact/wide footer/F1 and action dispatch/typing.
- Record expected stale hints/missing-help failures on original code.
- Derive supported grid hints from mounted/focused/readiness context; cache only existing selection classification if needed to avoid focus-time DB calls. Project the same facts into F1 with honest scope requirements, preserving widget binding ownership.
- Run new tests and affected existing Evals help/grid/export checks plus relevant static/performance guards; record exact evidence.
- Update docs, self-review and commit owned files; no global registry or new shell binding.

Detailed plan: Docs/superpowers/plans/2026-10-20-textual-book-followup-fixes.md (Task 5).
<!-- SECTION:PLAN:END -->
