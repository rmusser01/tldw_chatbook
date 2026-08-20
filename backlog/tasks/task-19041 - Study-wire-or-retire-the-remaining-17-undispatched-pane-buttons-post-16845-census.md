---
id: TASK-19041
title: >-
  Study: wire or retire the remaining 17 undispatched pane buttons
  (post-16845 census)
status: To Do
assignee: []
created_date: '2026-08-20 08:40'
labels:
  - ui
  - dead-code
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16845 removed four undispatched Study buttons (add-child / create-course /
generate-guide / add-milestone) plus the Course form, and explicitly deferred the
rest of the surface as "materially larger" (see its Implementation Notes'
out-of-scope list). Fresh census at dev `1bf7f234e`: `UI/Study_Window.py` composes
48 buttons (42 literal ids + 6 f-string `review-rating-*`), of which 24 distinct
ids reach a handler (21 `@on(Button.Pressed, ...)` targets incl. the six rating
ids, plus `on_button_pressed`'s `study-back-to-workspace-button`,
`study-switch-global-button`, and the seven `view-*` sidebar ids). That leaves
**18 composed button instances across 17 distinct ids with no dispatch anywhere**
(whole-tree grep per id returns only `Study_Window.py` and tests):

`add-sibling-btn`, `delete-node-btn`, `edit-node-btn`, `import-notes-btn`,
`export-md-btn` (composed TWICE — Mindmaps pane :451 and Course pane :512, a
duplicate-id wrinkle of its own), `generate-mindmap-btn`, `add-module-btn`,
`export-pdf-btn`, `export-scorm-btn`, `add-concept-btn`,
`generate-questions-btn`, `save-guide-btn`, `mark-complete-btn`,
`set-dependencies-btn`, `import-course-btn`, `export-path-btn`,
`generate-suggestions-btn`.

`on_button_pressed` early-returns for every one of them, so each press is a
silent no-op — the exact UX shape 16195's review called worse than "does
nothing" implies. 16845's per-pane evidence already established the backing is
placeholder-grade (write-only or nonexistent schema; static trees nothing
populates), so per the owner's stability ruling the default expectation is
honest removal / honest empty state per pane, not speculative wiring to
write-only sinks. No existing backlog task covers these ids (grepped
backlog/tasks at dev).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No composed Study control silently swallows a press: each of the 17 ids either reaches a real handler or is removed with per-affordance evidence (16195/16845 pattern, including controls that exist solely to feed a removed button)
- [ ] #2 Wire-vs-remove is decided per pane on schema/service evidence, preferring durable removal or an honest empty state over wiring to write-only sinks (owner ruling: stability over quick wins)
- [ ] #3 The duplicate `export-md-btn` id no longer composes twice
- [ ] #4 Study suites stay green and pinning tests forbid removed affordances from returning
<!-- AC:END -->
