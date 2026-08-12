---
id: TASK-203
title: 'Library Prompts: multi-select + bulk actions in the list'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-12 22:21'
updated_date: '2026-08-12 20:40'
labels:
  - ux
  - library
  - prompts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Second-pass UX review (2026-07-12): the prompts list has no multi-select, so
bulk delete and export are impossible; a growing library will want them
(mirrors the media multi-select backlog item task-159). Selection must support
curating a batch across searches and pages. Bulk tagging is explicitly outside
this task; Prompt collections already provide the current organization surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Prompt list offers a visible selection mode with checked rows, the total selected count, the count selected on the current page, Select page, Clear all, and Done controls
- [ ] #2 Selection persists across Prompt searches, pages, sort orders, collection scopes, and an Export-canvas round trip, then clears explicitly on Done, successful delete, editor/create entry, source change, or Library exit
- [ ] #3 Export selected uses the existing local Chatbook export flow and includes exactly the selected active Prompt/Recipe IDs; a missing selected item aborts the Prompt-bearing archive rather than producing a partial export
- [ ] #4 Bulk delete validates every selected Prompt/Recipe against its captured version and soft-deletes the complete selection atomically; any missing, changed, invalid, or failed item leaves every item undeleted and preserves the selection
- [ ] #5 Successful single and bulk deletes share one mutation path and leave one in-place Undo/Dismiss receipt; Undo restores the complete receipt atomically or restores nothing and keeps the receipt available
- [ ] #6 Selection and selected export remain local-only; delete and restore are policy-gated exactly once per batch; blocking export, delete, and restore work remains off the UI thread; new or modified selection/delete/restore diagnostics never contain Prompt content, names, IDs, versions, selection payloads, exception messages, or tracebacks; and selected export retains ADR-057's sanitized Prompt collection/scope diagnostics
- [ ] #7 Keyboard focus, literal labels, disabled explanations, loading/error recovery, and every selection/bulk action remain reachable and readable at both 64x24 and 120x40 with exactly the existing scroll ownership
<!-- AC:END -->

## Related Decisions

- [ADR-055: One reversibility rule for Library destructive actions](../decisions/055-library-destructive-action-reversibility-rule.md)
- [ADR-057: Portable Chatbook Prompt Records](../decisions/057-portable-chatbook-prompt-records.md)
- [ADR-060: Atomic local Prompt batch mutations](../decisions/060-atomic-local-prompt-batch-mutations.md)
