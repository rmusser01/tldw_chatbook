---
id: TASK-32584
title: >-
  Library Notes: no end-to-end parse, execute, re-parse, classify test for
  import repeat detection
status: To Do
assignee: []
created_date: '2026-09-14 22:47'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Raised by the wave-4 task-32541 review and ruled a rider. Repeat detection is tested in pieces — the parser, the executor and the classifier each have their own coverage — but nothing drives the full loop a user actually runs: import a folder, then review the same folder again and assert how each source is classified. That is the loop the task-32541 defect lived in (structured sources were re-created as New every time), and it is the loop the receipt-reader rider describes another way to break.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One test imports a folder for real, then re-reviews the same folder and asserts the outcome group of every source
- [ ] #2 It covers a .md file, a multi-record .csv and a .yaml, since those are the kinds that classify differently
- [ ] #3 An edited file moves to Changed repeat and an untouched one to Unchanged repeat in the same run
<!-- AC:END -->
