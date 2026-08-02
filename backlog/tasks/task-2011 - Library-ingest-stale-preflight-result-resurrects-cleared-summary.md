---
id: TASK-2011
title: >-
  Library ingest stale pre-flight result resurrects a cleared summary
status: To Do
assignee: []
created_date: '2026-08-02 21:30'
labels:
  - library
  - ingest
  - ux
  - uat
priority: high
dependencies: []
---

## Description (the why)

`_do_submit_ingest` clears `form.preflight` on purpose (its comment promises
a late result "cannot repopulate what was cleared"), but pre-flight runs as
`@work(thread=True)` whose cancellation is cooperative, and
`_apply_library_ingest_preflight_result` applies unconditionally via
`call_from_thread` (`library_screen.py:12396`). A pre-flight that finishes
after the submit resurrects the cleared summary: observed twice live as
"Enter a file path to start." rendered directly above "1 plain text file ·
277 B" with an empty path field. Found in the 2026-08-02 ingest UAT
(critique snapshot 2026-08-02T21-04-04Z).

## Acceptance Criteria (the what)

- [ ] A pre-flight result whose analysis started before a submit/clear never
      repopulates the summary after the clear.
- [ ] A pre-flight result for the current (uncleared, unsuperseded) trigger
      still applies normally.
