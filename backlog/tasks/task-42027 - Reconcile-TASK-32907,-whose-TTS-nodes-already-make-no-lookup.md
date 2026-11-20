---
id: TASK-42027
title: "Reconcile TASK-32907, whose TTS nodes already make no lookup"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- backlog
priority: low
references:
- Tests/TTS
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41708 (round 17). TASK-32907's two TTS nodes already use MockTransport and make no DNS lookup; under a temporary private profile they pass 2 of 2, so its AC#1 and AC#2 appear to hold.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TASK-32907 is closed with evidence for each criterion, or rescoped with a dated note naming what remains
<!-- AC:END -->
