---
id: TASK-31627
title: Unauthenticated gh is detected and classed MISSING_TOOL, not ERROR
status: To Do
assignee: []
created_date: '2026-09-04 23:10'
labels:
  - console
  - inspector
  - git
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Environment panel's PR/CI tier shells out to `gh`. An installed-but-
unauthenticated `gh` (and an expired token, which is the common case: `gh`
auth lapses silently) exits non-zero, and
`Workspaces/environment_status.py::gather_pr_env` classifies that as `ERROR`.
Two consequences, both wrong:

- `ERROR` feeds the net tier's consecutive-failure counter, so three polls
  pause the tier over a condition that is not transient and will never clear
  by retrying.
- When the tier had previously succeeded, the ERROR path keeps the last good
  PR row with a stale marker — so an auth expiry leaves a PR row on screen
  that is quietly frozen, with no cue that the panel has stopped being able
  to look.

`MISSING_TOOL` is the honest classification: the tool cannot answer, the rows
hide the way they do when `gh` is absent, and nothing accumulates toward a
backoff that cannot help.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An installed but unauthenticated `gh` yields `MISSING_TOOL`, not `ERROR`, and the PR/checks rows hide exactly as they do when `gh` is not installed
- [ ] #2 An auth failure does not increment the net tier's failure counter and cannot pause the tier
- [ ] #3 An auth failure after a successful fetch does not leave a stale PR row standing with no cue
- [ ] #4 Genuine transient failures (timeout, unparseable output) still classify as `ERROR` and still back off — proven by a negative-control test
<!-- AC:END -->
