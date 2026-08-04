---
id: TASK-2222
title: >-
  Library ingest: "Select this folder" action in the Browse dialog
status: To Do
assignee: []
created_date: '2026-08-04 05:00'
labels:
  - library
  - ingest
  - ux
priority: medium
dependencies: []
---

## Description (the why)

Owner ruling (2026-08-04): folder import must be pickable, not
type-only. In the Browse dialog, "Open" on a directory descends into it
(correct), and a dedicated "Select current folder" action returns the
directory being viewed as the ingest source (the vendored fspicker's
SelectDirectory variant is the reference).

## Acceptance Criteria (the what)

- [ ] The ingest Browse dialog offers a visible "Select current folder"
      action (button and/or binding) that returns the directory being
      viewed; Open keeps descending.
- [ ] Choosing it fills the path field with the directory and triggers
      pre-flight, exactly like typing the path.
- [ ] File selection behavior is unchanged.
