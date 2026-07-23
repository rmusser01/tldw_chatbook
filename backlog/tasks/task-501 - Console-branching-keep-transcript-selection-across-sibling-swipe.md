---
id: TASK-501
title: >-
  Console branching: keep transcript selection across sibling swipe for
  back-to-back comparison
status: To Do
assignee: []
created_date: '2026-07-23'
labels:
  - console
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With Console branching (Phase A, PR #799), pressing `<` / `>` to navigate sibling branches moves the active leaf and re-syncs the transcript, which clears the selected message (pre-existing precedent from PR #359: `selected_message_id` is dropped whenever the message no longer occupies a transcript slot, shared with the "continue" action). Consequence: the action row disappears after each swipe, so comparing branches back-to-back requires re-selecting the message and re-opening the action row before each `<` / `>`. Since rapid branch comparison is the primary reason this feature exists, the swipe path should keep the selection anchored (e.g. re-select the new active node at the same turn position) so repeated `<` / `>` works without a re-click.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a `<` / `>` sibling swipe, the transcript keeps a selection at the swiped turn so the `<` / `>` action row stays available
- [ ] #2 Repeated `<` / `>` presses navigate siblings without an intervening row re-click
- [ ] #3 The "continue"/other-action selection-clear behavior is not regressed
- [ ] #4 Verified in the live TUI
<!-- AC:END -->
