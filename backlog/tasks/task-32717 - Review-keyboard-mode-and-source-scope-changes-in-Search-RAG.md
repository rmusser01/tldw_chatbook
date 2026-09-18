---
id: TASK-32717
title: Review keyboard mode and source-scope changes in Search RAG
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 14:31'
updated_date: '2026-09-17 14:43'
labels:
  - library
  - search-rag
  - verification
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the bounded Library design-system review so mode and source changes keep keyboard operation visible and preserve the existing retrieval and answer contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mode and source toggles remain keyboard-operable with visible focus and truthful labels in both themes at wide and compact terminal sizes.
- [x] #2 Mode changes discard stale retrieval and answers; scope changes filter evidence and affect subsequent runs without duplicate service calls or loss of query/history.
- [x] #3 Targeted tests, private native evidence, static checks and review qualify the bounded behavior; documentation records any repairs and remaining limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-031 and ADR-150 apply. Reason: routine review and repair within existing keyboard, canvas and retrieval contracts; reassess if those boundaries change.
1. Trace mode/scope handlers and existing regression tests. Probe focused keyboard toggles with production styles and reproduce any defects before repair.
2. Make the smallest repair and verify stale-work, scope filtering, focus and service-call behavior in targeted tests.
3. Run private native journeys, inspect captures, complete targeted static checks and independent review, update guide/audit/evidence, and commit locally. No full suite, push or merge.
Allocation: all reachable object paths and 27 worktrees report maximum 32716.
Confirmed defect: all eight keyboard mode/scope cases lose the focused toggle to the panel during canvas recompose. LibrarySearchRagPanel already inherits PostRecomposeCallback; use its existing same-ID focus preservation in sync_state, as other Library canvases do. Eight initial cases now pass; add delayed newer-focus and in-flight retrieval/answer coverage. No state, token, binding or service-boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the bounded Search/RAG mode and source-scope review and repaired keyboard focus loss during panel recomposition. LibrarySearchRagPanel now uses its inherited same-ID focus helper before sync; repeated Enter stays on the mode/source toggle, while newer attached focus choices take precedence. Existing mode reset and source filtering/in-flight behavior is preserved.

Validation: 149 targeted tests pass (144 related checks plus five existing mode/scope regressions), including sixteen new production-styled keyboard and race cases. Eight pre-fix cases reproduce focus falling onto the panel. Four private native journeys across dark/light and 170x48/80x24 pass with eight real keyword searches and four controlled answers. All twelve captures were inspected. Deselecting every available source blocks Run and hides evidence; reselecting restores the same rows without another search. Ten private databases are healthy, source/default files are unchanged, no conversation messages were added, query history persists, and normal exit/PID absence are verified.

Files: LibrarySearchRagPanel, Tests/UI/test_library_rag_mode_scope_keyboard.py, Search/RAG guide, Library audit and Docs/superpowers/qa/2026-09-17-rag-mode-scope/. New test/runner lint and format checks and changed production range formatting pass; five production lint diagnostics predate this change, with no additions. Independent code/test/runner review found no actionable issue. No new lesson: this applies the established canvas recompose/focus pattern.

ADR required: no; existing backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md and backlog/decisions/150-design-token-system-and-design-language.md apply. No tokens, bindings, persistence or service boundaries change. ID checked across 307 refs and 27 worktrees with no other owner. Real-provider behavior, semantic retrieval, factual grounding and very narrow single-pane layouts remain outside qualification. No full suite, push or merge.
<!-- SECTION:NOTES:END -->
