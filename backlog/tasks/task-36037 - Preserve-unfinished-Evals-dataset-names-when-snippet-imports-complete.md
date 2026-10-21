---
id: TASK-36037
title: Preserve unfinished Evals dataset names when snippet imports complete
status: Done
created_date: 2026-10-21 16:13
labels:
- evals
- textual-patterns
- drafts
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-21-textual-book-reanalysis.md
assignee:
- '@codex'
updated_date: 2026-10-21 19:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Importing snippets must update the dataset contents without discarding an unfinished dataset-name edit or moving the user away from an active name field. The draft must remain uncommitted until explicit Rename succeeds.

### Evidence and scope

Verified on latest dev df24de7ad71b95992f63a77fc5a19726e840849d. DatasetRenameRow.compose() at snippet_editor.py:565-571 seeds Input from the persisted name; SnippetEditor.compose():626-658 recreates that row; successful _apply_import_content():818-854 refreshes the entire editor with recompose=True. The real mounted Evals harness, real SQLite and existing private-profile helper reproduced all TXT/CSV/JSON formats: a settled keyboard-typed Stored dataset name-draft (focused, cursor25) became Stored dataset name in a different Input, focus moved to evals-rail-toggle-runs, cursor19. Each import first verified initial and imported text persisted and the success notification appeared; the stored dataset name remained unchanged. Three intended draft-retention assertions failed in 82.24s. The file-picker callback was driven directly through the existing test helper, not a native picker journey.

Book basis: Chapter 9 printed pp.170-171 / PDF pp.179-180 explains child replacement and Input state reset under recomposition; printed p.180 / PDF p.189 explains focus loss.

### Existing ownership and decisions

All related tasks are Done. TASK-1484 supplies dataset Rename and validation; TASK-1482 supplies authoring. TASK-1610 protects dirty bench editors from run completion, a different widget/trigger. TASK-15475 scoped screen selection updates, and TASK-35869 fixed grid keyboard/footer/F1 scope. Existing snippet-import tests do not combine import completion with an unfinished name draft; no duplicate open owner was found.

Related tasks: [TASK-1484](./task-1484%20-%20Rename-affordances-for-Evals-benches-and-datasets.md); [TASK-1482](./task-1482%20-%20Bench-authoring-targets-probes-and-top-K-editable-with-Duplicate-and-Delete.md); [TASK-1610](./task-1610%20-%20Run-completion-must-not-destroy-a-dirty-bench-editor.md); [TASK-15475](./task-15475%20-%20Panel-scoped-updates-for-Settings%2C-Evals%2C-and-speech-settings%3B-dedupe-mount-resume-double-work.md); [TASK-35869](./task-35869%20-%20Make-Evals-grid-footer-and-F1-help-follow-active-keyboard-scope.md).

ADR required: no.
ADR paths: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md); [150-design-token-system-and-design-language](../decisions/150-design-token-system-and-design-language.md); [161-component-pattern-library](../decisions/161-component-pattern-library.md).
Reason: This is routine preservation of existing dataset-editor state within its current owner. Keep EvalsDB storage, explicit Rename validation, import policy and existing screen/view-model boundaries. No new root state, persistence mechanism or application structure is required; token/component decisions govern any incidental UI change.

[Latest-dev book audit and evidence](../../Docs/superpowers/qa/2026-10-21-textual-book-reanalysis.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Successful TXT, CSV and JSON snippet imports preserve the exact unfinished dataset-name edit, including invalid or whitespace-containing text, without implicitly renaming the stored dataset.
- [x] #2 Imported snippets, counts, groups and warnings refresh correctly while the name draft remains intact; ordinary imports without a draft keep their behavior.
- [x] #3 When the name field is focused at import completion, its editing focus, exact caret position and selection range are preserved; the update does not move focus to the rail.
- [x] #4 Explicit Rename retains existing empty-name/collision validation and updates the stored name, heading and catalog; cancelled or failed imports retain the name draft. Mounted private-profile real-SQLite regressions cover those outcomes through actual import-worker completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR paths: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: preserve existing widget-local name state and explicit Rename; no storage, import policy, view-model, provider or application boundary change.

1. Add mounted private-profile regressions using real SQLite and actual import-worker completion. Before product edits, observe loss of the exact draft, focused caret and selection for TXT/CSV/JSON imports, including invalid or whitespace text.
2. Refresh only read-only imported snippet results, leaving the live name field and its editing state intact. Keep current IDs, summary/groups/warnings, empty state, scroll behavior and reachable Import control.
3. Verify explicit Rename validation and stored name/heading/catalog updates, cancelled/failed import preservation and ordinary no-draft imports. Update Evals architecture documentation and run targeted controls and scoped static checks.
4. Obtain independent task review. Controller admits selected regressions into required CI, checks combined integration, then completes notes and criteria from actual evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Snippet imports now recompose only a read-only result child from the existing combined-list return value. The live name Input, Rename row, heading and Import button survive, preserving exact invalid/whitespace draft text, focus, caret and selection. Explicit Rename remains the only writer and retains validation/catalog updates; no extra database read or draft store was added.

Changed snippet_editor.py, new test_evals_snippet_import_drafts.py and Evals architecture documentation. No stylesheet, schema, dependency, import policy or view-model boundary change. Shared CI admits every core TXT/CSV/JSON case and retains all prior coverage/limits.

ADR required: no. Existing ADR-033/150/161 govern this bounded correction.

Evidence: final six mounted private-profile regressions pass in 108.70s, two token/component governance controls pass, and independent Superpowers spec/quality review is Approved with no Critical/Important findings. Shared CI contracts: 121 pass; six new gated cases collect. Both changed Python files pass Ruff lint/format, UTF8/compile and whitespace checks. Six existing controls stopped before their bodies in source-profile fixture admission and are not counted as passing. No full suite, production native-terminal or OS-picker claim; hosted results are tracked on the PR.

[Qualification and structured evidence](../../Docs/superpowers/qa/2026-10-21-textual-bulk-and-import-fixes.md).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
