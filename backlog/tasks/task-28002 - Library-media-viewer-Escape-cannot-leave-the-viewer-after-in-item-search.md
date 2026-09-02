---
id: TASK-28002
title: Library media viewer - Escape cannot leave the viewer after in-item search
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
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
- [ ] #1 After Enter in the in-item search input, the keyboard stays live (no focus black hole); Escape closes the find bar as the footer advertises
- [ ] #2 Footer key hints match actual behavior while the find bar is open
- [ ] #3 A regression test pins submit-then-Escape and submit-then-typing
<!-- AC:END -->
