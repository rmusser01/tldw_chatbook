---
id: TASK-32715
title: Preserve RAG query and evidence focus during resize
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 06:32'
updated_date: '2026-09-17 06:54'
labels:
  - library
  - search-rag
  - verification
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keyboard users need a reliable way to return from generated answers and evidence to their query, including across window resizing. Resolve the prior native setup ambiguity with observed focus and painted-control evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keyboard users can return from evidence controls to a visible, editable RAG query before and after wide/compact resize transitions in both themes.
- [x] #2 Navigation preserves query, scope and generated answer without extra retrieval or provider calls; current control identity and settled paint distinguish product failures from stale test references.
- [x] #3 The guide and audit record the supported return route and any bounded repairs, with targeted and private native evidence and explicit provider/retrieval limits.
- [x] #4 Resizing retains and reveals the current query or evidence action; deferred resize callbacks respect a newer focus choice inside or outside the panel.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no initially; reassess after reproduction if a repair changes boundaries.
ADR path: N/A; existing backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md and backlog/decisions/150-design-token-system-and-design-language.md govern existing focus/paint behavior.
Reason: bounded review and routine focus repair under established interaction rules, with no new keyboard grammar or architecture planned.
1. Trace real key events, focused IDs/current instances, panel scroll and paint across evidence-to-query return and 170x48/80x24 resize. Reproduce before changing behavior; read existing focus and route contracts.
2. If a product defect is demonstrated, refine acceptance criteria and add a focused regression before the smallest repair. Otherwise retain existing UI and document verified behavior.
3. Run targeted tests/static checks, independent review and a private native journey with real local keyword retrieval and controlled answers. Verify captures, persistence and shutdown; update guide/audit/task and commit locally. No full suite, push or merge.
Allocation: maximum 32714 across 310 refs/27 worktrees; CLI offered 32715, assigned 32715.
Confirmed cause: LibraryScreen._transition_library_notes_presentation applies Notes semantic restoration on Search/RAG, whose query/evidence controls have no Notes role, falling back to the rail. Reuse the retained-control exemption already used by Prompts/Import; let the Search/RAG panel reveal its currently focused descendant after resize. Verify width/height transitions, reverse-Tab editing, and delayed callbacks after newer focus. Existing ADR-031/ADR-150 apply; no new ADR, binding or visual token.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resizing Search/RAG no longer transfers query/evidence focus to the rail. LibraryScreen exempts retained RAG controls from Notes semantic restoration, and LibrarySearchRagPanel reveals the current focused descendant after layout changes. Delayed callbacks respect newer focus inside/outside the panel. Shift+Tab returns to an editable RAG query without resubmitting; the guide distinguishes this route from keyword submission in the rail.

Validation: 98 targeted tests pass, including 12 new regressions; initial failures reproduce the focus defect with current mounted control identity. Four private native journeys pass 16 resize transitions in both themes; all 8 captures were inspected. Real local keyword retrieval and controlled replies yield exactly four retrievals/four answer calls. Source/default files are unchanged, ten private databases pass quick_check, and normal exit/PID absence are verified. New files pass Ruff lint/format, changed ranges pass format checks, production adds no lint diagnostics, and independent review has no remaining finding. No full suite, push or merge.

Files: two production focus paths, Tests/UI/test_library_rag_query_return.py, search/RAG guide, workflow audit and Docs/superpowers/qa/2026-09-17-rag-query-return/. Prior answer-reading QA now links the diagnosed resize defect. A testing lesson records why the delayed-callback regression must not hold unrelated Textual focus scrolling.

ADR required: no. Existing backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md and backlog/decisions/150-design-token-system-and-design-language.md govern this routine repair; no bindings, tokens or service boundaries change. Controlled provider output, semantic retrieval, factual grounding and very narrow single-pane layout remain outside qualification. Task ID rechecked across 310 refs/27 worktrees with no other owner.
<!-- SECTION:NOTES:END -->
