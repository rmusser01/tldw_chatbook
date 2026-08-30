---
id: TASK-24529
title: Await pruning Console selection menu remounts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-30 01:28'
labels:
  - console
  - textual
  - reliability
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-29-console-selection-menu-remount-race-design.md
documentation:
  - Docs/superpowers/plans/2026-08-29-console-selection-stability.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent app-fatal duplicate selection-menu IDs when a completed Console text selection replaces a menu whose removal is already pending.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A completed text selection waits for every previously attached Console selection menu to detach, including a menu already marked for pruning, before mounting its replacement
- [x] #2 Ordinary fire-and-forget menu dismissal retains its current non-pruning behavior without duplicate removal work
- [x] #3 Immediate no-yield replacement leaves exactly one new non-pruning menu mounted, the previous menu detached, and the app running without `DuplicateIds`
- [x] #4 Settled consecutive drags, menu placement, menu actions, feedback, focus, and dismissal behavior remain unchanged under focused Console tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a deterministic already-pruning no-yield remount regression.
2. Replace the remount boundary with the public awaited screen query while preserving ordinary dismissal.
3. Run focused Console selection-menu and dismissal verification.
4. Complete task evidence and self-review.

Detailed plan: Docs/superpowers/plans/2026-08-29-console-selection-stability.md
ADR required: no
ADR path: N/A
Reason: existing Textual lifecycle ordering only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the remount-boundary fix by awaiting the public unfiltered screen menu query while leaving ordinary filtered fire-and-forget dismissal unchanged. Added a deterministic no-yield pruning regression and clarified the settled regression docstring. Modified Tests/UI/test_console_selection_menu.py and tldw_chatbook/Widgets/Console/console_transcript.py. TDD evidence: regression failed before the fix with DuplicateIds; focused verification passed 4 tests. Static evidence: git diff --check passed; Ruff retained the exact inherited three-file formatter baseline, with no bulk formatting. Independent specification and code-quality reviews approved the final implementation. Full suite not run under the repository targeted-test policy. ADR required: no. Lessons learned: none; the existing approved lifecycle design already records the generalizable race.
<!-- SECTION:NOTES:END -->
