---
id: TASK-34570
title: Qualify structured formatting in native editor undo history
status: In Progress
assignee:
- '@codex'
created_date: 2026-10-07 02:16
updated_date: 2026-10-07 09:14
labels:
- notes
- structured-editing
dependencies: []
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editors.md
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
Monaco qualification approved by human option1 on 2026-10-07. ADR required: yes; existing governing ADR194(C)/ADR065(S), amended before any new runtime/asset contract if needed. Continue I1 using installed Monaco only for structured source modes with bundled/local assets in Next and extension, unchanged Markdown ownership. Read task-11-monaco-context.md and exact review constraints. Qualify actual temporary Notes-page integration/native keyboard history in all three engines plus source representation/currentness/composition/focus/caret/model isolation; no custom undo or source CDN. Remove temporary product wiring before explicit commits. Append dated candidate evidence/report and self-review, keep task In Progress for independent review. Do not rerun unchanged C bridge tests.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Qualified Textual8.2.8 bridge: 5 passing mounted tests, Ruff check/format pass. Browser gate fails WebKit26 first Undo D -> A. Original candidate repeated twice; one authorized synchronous blur/focus boundary on both sides of insertText also fails, preserving guards and selection. No additional native experiment after failure; temporary product wiring removed. Evidence: Docs/superpowers/reviews/2026-09-27-document-editor-history.md. Existing ADR194 applies; In Progress pending review/revised web approval.
2026-10-07 I1 Monaco option-1 candidate qualified in S real Notes page: 36 Chromium/Firefox/WebKit controls plus 3 strengthened viewport controls pass, exact LF/CRLF native history; unsupported representations retained read-only. Temporary product wiring removed; installed local assets exercised in Next and WXT MV3 asset/CSP smoke. C bridge/code/tests unchanged and not rerun. Existing textarea remains unqualified; actual OS IME, extension Notes/history and production loader integration remain limits. ADR194/existing ADR065 experiment extension apply. See Docs/superpowers/reviews/2026-09-27-document-editor-history.md and task-11-report dated Monaco section for commands, failures, static evidence and self-review. Keep In Progress/no AC closure pending controller review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
