---
id: TASK-32598
title: Keep Library section toggles visibly focused above the rail scroll cue
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 02:48'
updated_date: '2026-09-15 03:13'
labels:
  - library
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-library-workflow-audit.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 80x24, keyboard Tab from Search/RAG focuses Create underneath the opaque scroll-for-more row. Its glyph and focus cue disappear while Enter still collapses the section. Reproduced on baseline 2939afda63 in both themes with production styles and in a native private-profile app. The focused toggle region is (22,21,3,1), its painted crop is three spaces, and the cue covers (2,21,24,1). Task-32219 established the useful fold cue; the repair should retain its discoverability while keeping keyboard targets visible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 80x24 in textual-dark and textual-light, every rail section toggle reached by Tab paints its label or glyph and an unambiguous focus cue before it can be activated.
- [x] #2 Tab and Shift+Tab bring focused controls above the docked fold cue, including Create, Import/Export and Details Diagnostics, without changing the selected destination.
- [x] #3 Enter activates the visibly focused section and preserves usable focus through 120-to-80-to-120 resizing.
- [x] #4 The fold cue still truthfully indicates additional content, and verification includes compositor paint and a native keyboard reproduction rather than region containment alone.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the covered section toggle with production styles and assert its actual painted glyph/focus state. Trace Textual visibility and dock-aware scrolling.
2. Apply a rail-local focus reveal using existing scroll geometry; preserve the docked cue, section behavior and token styling.
3. Verify forward/reverse traversal, activation, and 120-to-80-to-120 resizing in both themes; run the targeted rail tests and native private-profile check.
4. Record evidence, self-review, and close the task with verified acceptance criteria.

ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md; backlog/decisions/086-library-adaptive-reader-shell.md
Reason: routine bug repair restoring visible focus within the existing rail and reader contracts; no new boundary or UX structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Library section toggles now scroll above the docked fold cue as focus arrives. Textual counted the covered toggle as fully visible, so LibraryRail now uses its existing dock-aware scroll operation for current focus events while the cue is present. The cue, route, focus styling and section actions remain intact.

Verification: both new regression cases failed before the fix and passed afterward; 29 targeted rail/governance cases passed, and the strengthened deep-Diagnostics resize file passed all four cases. Both themes were visually inspected. A private native profile proved visible Create focus, Enter collapse and retained focus after widening; exit 0. Ruff check/format and whitespace checks passed. The two existing query-budget failures remain unchanged (23 per three resize frames; 5 per Tab) under TASK-32599, and the known startup RichLog error remains outside this task.

Changed: Widgets/Library/library_rail.py (focus handler plus existing lint cleanup), Tests/UI/test_library_rail_focus_visibility.py, the testing-evidence lesson, and Docs/superpowers/qa/2026-09-14-library-rail-focus. No CSS rebuild was needed. No new ADR: existing ADR-150, ADR-161 and ADR-086 govern the repair. Self-review confirmed stale focus events are ignored and no service, permission or data boundary changed.
<!-- SECTION:NOTES:END -->
