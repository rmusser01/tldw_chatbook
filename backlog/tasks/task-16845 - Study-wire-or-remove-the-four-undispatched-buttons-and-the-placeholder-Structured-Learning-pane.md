---
id: TASK-16845
title: 'Study: wire or remove the four undispatched buttons and the placeholder Structured Learning pane'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - ui
  - dead-code
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16195 (PR #1681) removed the orphaned "Add Topic" affordance and TASK-16196
(PR #1688) deleted the legacy Study event-handler module whose table was the only thing
that ever named these buttons. Both reviews confirmed — and it still holds at dev
`ee741cf10` — that **four more Study buttons compose live with no handler anywhere**:

- `UI/Study_Window.py:440` — `Button("Add Child", id="add-child-btn")`
- `UI/Study_Window.py:524` — `Button("Create Course", id="create-course-btn")`
- `UI/Study_Window.py:612` — `Button("Generate from Topic", id="generate-guide-btn")`
- `UI/Study_Window.py:671` — `Button("Add Milestone", id="add-milestone-btn")`

Zero `@on(Button.Pressed, ...)` decorators reference any of the four (re-grepped at
HEAD), and `StudyWindow.on_button_pressed` early-returns unless the id is one of two
sidebar ids or starts with `view-` — so each press is a **silent no-op**: a user can fill
in the adjacent inputs, click, and get no signal at all. Same shape as the removed
add-topic button, which 16195's review called worse UX than "does nothing" implies.

The same review raised the coherence question this task should settle alongside: the
Structured Learning pane's residual chrome (`#topic-tree` rooted at a static "Learning
Paths" node that can never gain children; `#topic-content`, a disabled TextArea whose
placeholder promises "Select a topic from the tree..." with nothing selectable and topics
write-only in the DB — no row-level read method exists) reads as broken rather than
intentionally empty. Per affordance: wire it to the live Study rebuild
(`UI/Study_Modules/` is flashcards + quizzes only today), remove it with 16195's
per-affordance evidence pattern, or replace the pane with honest empty-state copy /
gate the section until its backing feature exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 No composed Study control silently swallows a press: every remaining button has a real handler or is removed (per-button evidence, 16195-style)
- [ ] #2 The Structured Learning pane either gains a real read path or presents an honest, intentional empty state (no dead tree + disabled content pane promising interaction)
- [ ] #3 Study suites stay green and the pinning tests forbid the removed affordances from returning
<!-- AC:END -->
