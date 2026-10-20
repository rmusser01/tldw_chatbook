---
id: TASK-35867
title: Keep remaining Console input refreshes out of user drafts
status: In Progress
created_date: 2026-10-20 19:24
labels:
- settings
- textual-patterns
- correctness
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md
assignee:
- '@codex'
updated_date: 2026-10-20 20:23
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Programmatic refreshes of the five agent-budget inputs, tool-result display limit and two side-chat strings must not become user-owned draft entries. Untouched inputs must continue following later backing-state refreshes while genuine edits remain staged.

### Evidence and bounded scope

settings_screen.py:28781-28875 and :35317-35353 guard queued Input.Changed messages with temporary synchronous flags. One full mounted private-profile diagnostic staged only paste_collapse_threshold=120, changed the eight backing values and called the real sync method. All eight untouched values appeared in draft.values. A second backing-value change left every displayed input pinned to the first refresh. Only the original paste threshold was dirty; this is demonstrated draft pollution and stale display, not a claim of new dirty keys or observed save/data loss.

Book basis: Chapter 7 printed pp.144-145 / PDF pp.153-154 (message prevention); chapter 9 printed pp.170-171 / PDF pp.179-180 (editable state under refresh/recompose).

### Existing ownership and decisions

The merged TASK-35676/35677 pilot covers paste controls, TASK-35680 covers max_parallel_runs, and TASK-35678 consolidates action availability. TASK-32766 qualifies budget/runtime limits. Those completed scopes do not repair these eight remaining inputs; this is one bounded event-origin correction, not another widget extraction.

Related tasks: [TASK-35676](./task-35676%20-%20Keep-Console-Behavior-form-refreshes-from-staging-user-edits.md); [TASK-35677](./task-35677%20-%20Give-Settings-composer-paste-controls-local-widget-ownership.md); [TASK-35678](./task-35678%20-%20Share-Settings-action-availability-across-shortcuts-buttons-and-help.md); [TASK-35680](./task-35680%20-%20Keep-parallel-run-form-refreshes-out-of-Settings-drafts.md); [TASK-32766](./task-32766%20-%20Align-Console-budget-Settings-with-runtime-limits-and-verify-keyboard-workflows.md).

ADR required: no new ADR expected.
ADR paths: [033-settings-commit-models-three-honestly-labeled](../decisions/033-settings-commit-models-three-honestly-labeled.md); [161-component-pattern-library](../decisions/161-component-pattern-library.md).
Reason: No new ADR is expected: the screen keeps draft, validation and Save/Revert authority under the existing commit model; no runtime budget, service or persistent-state boundary changes.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Value-changing sync of all five budget fields, tool_result_display_chars, sidechat_model and sidechat_prompt_template adds no untouched draft values/originals or new staged-result message and preserves unrelated dirty or invalid drafts.
- [ ] #2 Repeated refreshes show the latest backing values for untouched fields; the first genuine edit afterward, including a value equal to a prior display, is still staged.
- [ ] #3 Mount/category return, Save and confirmed Revert do not turn initialization or stale queued messages into edits, and do not discard legitimate queued user edits.
- [ ] #4 Ordinary updates preserve the mounted Input instances, active focus, cursor and selection without recomposing the enclosing detail pane.
- [ ] #5 Existing budget ceilings/normalization, display limits, side-chat strings, real-writer Save/Revert and validation behavior remain intact; compact/wide private-profile tests cover refresh, actual typing and persistence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; backlog/decisions/161-component-pattern-library.md
Reason: Preserve the current Settings draft, validation and Save/Revert writer ownership; no new runtime or persistent boundary.

- Add mounted private-profile tests matching the audit: only paste threshold 120 staged, first eight backing values changed, then changed again; expect draft exactly `{'paste_collapse_threshold': 120}` and every untouched display follows the second refresh. Add queued-stale/genuine typing, mount/return and compact/wide real-writer Save/Revert coverage.
- Record expected draft pollution/stale display failures before product changes.
- Use native publication prevention for programmatic assignments and correct stale/no-op event handling for initialization; remove obsolete temporary sync flags. Preserve invalid text and the first genuine edit after refresh, including a previous display value.
- Run new regressions, affected budget/side-chat tests and existing paste/parallel origin regressions; test Input identity/focus/selection, size/import guards and scoped static checks.
- Update owner documentation, self-review and commit owned files; do not extract another section or alter runtime budgets.

Detailed plan: Docs/superpowers/plans/2026-10-20-textual-book-followup-fixes.md (Task 3).
<!-- SECTION:PLAN:END -->
