---
id: TASK-28002
title: Library media viewer - Escape cannot leave the viewer after in-item search
status: Done
assignee:
  - '@claude'
created_date: '2026-09-02 04:10'
updated_date: '2026-09-02 05:02'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-verified 2026-09-02 live on dev tip (worktree media-ux-fixes @ b7e89b6de, tmux scratch-profile run). Still broken, and worse than first filed: after typing in the Reader's "Search content..." input and pressing Enter (match header pins, footer reads "esc close find"), the ENTIRE keyboard goes dead - Escape x2, plain letters, Down and Tab all produce zero screen change. The app is alive (a mouse click on "Next" advances the match), and one mouse click restores focus, after which a single Escape closes the find bar correctly. Hypothesis (trace before fixing): the post-submit recompose/focus handoff drops focus onto a non-focusable or unmounted widget, so key events have no target. Note the code moved since first filing - action_library_media_viewer_back now has an explicit find-controls branch (library_screen.py ~41283); the defect is upstream of the Escape gating (focus loss on submit), not in it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Enter in the in-item search input, the keyboard stays live (no focus black hole); Escape closes the find bar as the footer advertises
- [x] #2 Footer key hints match actual behavior while the find bar is open
- [x] #3 A regression test pins submit-then-Escape and submit-then-typing
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Root-cause the post-submit focus loss (trace Input.Submitted handler for the Reader find input and the recompose it triggers)\n2. Pinning test that reproduces submit-then-keyboard-dead\n3. Minimal fix at the focus handoff\n4. Full media viewer test files green
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: LibraryMediaContentSearchControls.sync_query_state recomposed on activity flips (refresh(recompose=True)), destroying the focused Input on the FIRST submit; screen focus became None and every check_action gate reading self.focused failed - total keyboard deadlock. Fix: all three children (Input, status Static, Prev/Next toolbar) are now persistent and display-gated (task-22207 banner idiom); sync_query_state patches in place and never recomposes. Input identity and focus survive both activity flips. Tests: new pinning test test_activity_flip_preserves_focused_search_input; updated the two contract tests that pinned the old recompose (hides/reveals instead of removes/mounts) and one screen-level absence assertion in t22209. Files: Widgets/Library/library_media_content.py, Tests/Library/test_library_media_content.py, Tests/UI/test_library_media_reader_match_nav_t22209.py. Note: t22208 no-change traversal failure is pre-existing on clean baseline (verified by revert-run).
<!-- SECTION:NOTES:END -->
