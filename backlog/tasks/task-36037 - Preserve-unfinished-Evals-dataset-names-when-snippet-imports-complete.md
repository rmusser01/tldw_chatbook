---
id: TASK-36037
title: Preserve unfinished Evals dataset names when snippet imports complete
status: To Do
created_date: 2026-10-21 16:13
labels:
- evals
- textual-patterns
- drafts
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-21-textual-book-reanalysis.md
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
- [ ] #1 Successful TXT, CSV and JSON snippet imports preserve the exact unfinished dataset-name edit, including invalid or whitespace-containing text, without implicitly renaming the stored dataset.
- [ ] #2 Imported snippets, counts, groups and warnings refresh correctly while the name draft remains intact; ordinary imports without a draft keep their behavior.
- [ ] #3 When the name field is focused at import completion, its editing focus, exact caret position and selection range are preserved; the update does not move focus to the rail.
- [ ] #4 Explicit Rename retains existing empty-name/collision validation and updates the stored name, heading and catalog; cancelled or failed imports retain the name draft. Mounted private-profile real-SQLite regressions cover those outcomes through actual import-worker completion.
<!-- AC:END -->
