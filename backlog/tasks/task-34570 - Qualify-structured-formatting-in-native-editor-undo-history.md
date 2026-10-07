---
id: TASK-34570
title: Qualify structured formatting in native editor undo history
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-07 02:16'
updated_date: '2026-10-07 02:48'
labels:
  - notes
  - structured-editing
dependencies: []
documentation:
  - >-
    /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editors.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove explicit formatting uses the existing editor history before integrating parser output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Real native typing, formatting, typing, undo, undo and redo preserve the correct document states and earlier history.
- [ ] #2 Identity and editability guards refuse stale/read-only results; no-op output creates no edit.
- [ ] #3 Selection, caret, focus, scroll and composition behavior are qualified without a custom snapshot undo stack.
- [ ] #4 Qualification evidence names runtime versions, test commands and unsupported environments; temporary production wiring is removed before commit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md
Reason: native history qualification implements the approved reversible-edit rule without changing editor ownership.
1. Read I1 in the approved editors plan.
2. Add real native history tests and minimal guarded bridges.
3. Temporarily wire the actual Notes page for fixed replacement qualification, record the trace, then remove temporary wiring.
4. Run targeted bridge/browser checks, record limitations, review and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified native Textual8.2.8 bridge and mounted tests: 5 passing; Ruff check/format pass. Captured request/source guards, inherited disabled/read-only/no-op admission, linear caret mapping and scroll retention verified. Browser production controlled Notes gate fails WebKit26 (first Undo returns A instead of C), reproduced twice; Chromium145/Firefox146 pass. Web candidate is test-only and temporary product wiring removed. Evidence: Docs/superpowers/reviews/2026-09-27-document-editor-history.md; existing ADR194 applies, no new ADR. Keep In Progress pending review and revised web qualification approval.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
