---
id: TASK-32716
title: Review keyboard replay and clearing of recent searches
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 07:05'
updated_date: '2026-09-17 07:24'
labels:
  - library
  - search-rag
  - verification
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keyboard users must be able to revisit submitted searches and clear their history with visible focus and truthful current-mode/source behavior. Continue the bounded Library design-system review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keyboard users can expand Recent searches and replay a query under the current mode and source selection in both themes at wide and compact sizes.
- [x] #2 Replay synchronizes visible query fields, makes exactly the expected service calls, and leaves a visible usable focus target; clearing history preserves query/results/answer without new calls.
- [x] #3 Targeted tests and private native evidence establish the supported behavior; guide and audit describe limits and any routine repair.
- [x] #4 Answer arrival leaves the current Recent searches control visible; delayed reveal respects a newer focus choice inside or outside the panel.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no initially; routine UI review/repair under existing ADR-031 and ADR-150. ADR path: N/A; reassess if storage or interaction boundaries change.
1. Trace existing history recording, replay and clear owners and mounted focus/paint. History stores query text only; replay retains current mode and scope.
2. Add production-styled keyboard journeys for both themes and 170x48/80x24, proving producer-created history, query synchronization, call counts, focus and clear behavior. Reproduce defects before the smallest repair, with acceptance criteria updated first if needed.
3. Verify related targeted tests/static checks, independent review and private native keyword/controlled-answer journeys. Update guide, audit and task with evidence; commit locally. No full suite, push or merge.
Allocation: all reachable object paths plus 27 worktrees report maximum 32715.
Confirmed defect: all eight replay cases run the correct query/current mode/current scope, but the rail input retains beta while the canvas shows the replayed alpha. The handler writes state before Input.Changed, causing its equality guard to skip sibling synchronization. Use the existing _patch_sibling_library_search_input helper directly in the replay handler. Four keyboard clear/focus cases already pass; no focus repair is indicated.
After the query synchronization fix, 14/16 pass; both wide RAG replay cases retain CollapsibleTitle focus but its y=48 region is outside paint when the generated answer expands above it. Answer arrival refreshes layout without the post-refresh focused-control reveal used by resize. Reuse the panel current-focus reveal after answer refresh; verify a delayed callback respects newer focus inside/outside the panel. No captured-focus restoration or new binding.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed keyboard review of Recent searches and repaired two defects: replay now synchronizes the rail input through the existing sibling-input helper, and answer arrival uses the panel current-focus reveal after layout so the focused history heading stays visible. The reveal respects newer focus and detached panels. Clear history already retained its heading focus and required no change.

Validation: 148 targeted checks pass (135 related RAG checks plus 13 existing history/state checks), including 18 new production-styled keyboard/race cases. Red evidence separately reproduces stale rail text and answer-arrival focus leaving the viewport. Eight private native journeys pass across Search/RAG, dark/light and 170x48/80x24, totaling 24 real keyword searches and four controlled answers. All 16 captures were inspected. Cleared history remains persisted after shutdown; query/results/answer survive clear with no calls. Source/default files are unchanged, ten private databases are healthy, no conversation messages were added, and normal exit/PID absence are verified.

Files: LibraryRagSearchController, Tests/UI/test_library_rag_history_keyboard.py, Search/RAG guide, Library audit and Docs/superpowers/qa/2026-09-17-rag-history-replay/. Guide clarifies keyboard access, current-mode/source replay and the existing ten-entry/200-character limits. New files pass Ruff lint/format, changed ranges pass formatting and production adds no lint diagnostics. Independent review found no code/test issue; one ambiguous native-search-count sentence was clarified.

ADR required: no; existing backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md and backlog/decisions/150-design-token-system-and-design-language.md apply. No bindings, tokens, persistence or service contracts change. Existing current-focus/selective-callback lessons suffice. Task ID checked across 310 refs/27 worktrees with no other owner. Provider behavior, semantic retrieval, factual grounding and very narrow single-pane layouts remain outside qualification. No full suite, push or merge.
<!-- SECTION:NOTES:END -->
