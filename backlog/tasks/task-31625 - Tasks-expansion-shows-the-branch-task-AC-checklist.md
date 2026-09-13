---
id: TASK-31625
title: Tasks expansion shows the branch task's AC checklist
status: To Do
assignee: []
created_date: '2026-09-04 23:10'
labels:
  - console
  - inspector
  - backlog-integration
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner decision 5 of the Environment redesign spec
(`Docs/superpowers/specs/2026-09-04-console-inspector-environment-redesign-design.md`)
asked the Tasks section's expansion to show the branch task's acceptance-
criteria checklist above the full task list. TASK-31450 shipped it
count-only: the collapsed line reads `3/6 ACs · <title>` and the expansion
shows the task list with no per-criterion detail, so a user still has to open
the task file to see which criteria are outstanding — the question the panel
exists to answer without leaving the app.

The gap is recorded as a post-implementation ruling against decision 5 in the
spec. Completing it means parsing the AC block per criterion (text plus tick
state) and rendering it inside the rail's fixed ~34-column body without
letting a long task push the rest of the section off-screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Expanding the branch-task row lists its acceptance criteria individually, each showing whether it is done
- [ ] #2 Criterion text is fitted to the rail's column budget rather than wrapping or overflowing the section
- [ ] #3 The criteria list is height-capped so a task with many criteria never pushes the rest of the Tasks section out of view
- [ ] #4 A task file with a malformed or absent AC block degrades to today's count-only line instead of erroring or emptying the section
<!-- AC:END -->
