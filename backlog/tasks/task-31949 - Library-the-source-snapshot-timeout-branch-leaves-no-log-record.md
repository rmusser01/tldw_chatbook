---
id: TASK-31949
title: Library - the source-snapshot timeout branch leaves no log record
status: To Do
assignee: []
created_date: '2026-09-07 08:25'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR G ruled no new logger in that module, so the Library source-snapshot deadline branch returns silently: a report of 'the snapshot never arrived' cannot be separated from a fetch error in the logs, and the branch is diagnosable only from the UI. Reusing the existing warning call with a timeout marker respects the no-new-logger ruling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A source-snapshot timeout leaves a log record naming the deadline and the source
- [ ] #2 No new logger object is introduced
<!-- AC:END -->
