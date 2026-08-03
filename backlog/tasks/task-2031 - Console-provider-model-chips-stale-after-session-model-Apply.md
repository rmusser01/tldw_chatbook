---
id: TASK-2031
title: 'Console provider/model chips stale after session model Apply'
status: To Do
assignee: []
created_date: '2026-08-03 00:45'
labels:
  - console
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-1980 live UAT. Changing provider/model via the Provider chip's
session model modal and pressing Apply updates the SESSION (the next run
uses the new provider — verified against a local stub endpoint) but the
status chips keep showing the old provider/model until a session/tab switch
forces a refresh. The user watches "Provider: Anthropic" while the run is
actually served by Custom — the chips' whole purpose (PR #1153) inverted.

The left-rail Model section DOES show the new values immediately; only the
status chip row misses the poke after Apply.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Apply in the session model modal, the Provider/Model chips reflect the new values without switching sessions
- [ ] #2 A test pins the chip refresh on the Apply path
<!-- AC:END -->
