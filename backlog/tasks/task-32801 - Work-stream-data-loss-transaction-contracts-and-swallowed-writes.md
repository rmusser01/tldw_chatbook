---
id: TASK-32801
title: 'Work stream: data loss, transaction contracts and swallowed writes'
status: Done
assignee: []
created_date: '2026-09-18 16:46'
updated_date: '2026-09-18 21:52'
labels:
  - core-review
  - review-data
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The store template lets a write method assume an outer transaction. Two stores have a method that does DML on a held legacy-isolation connection anyway; in Media a shipped caller opens no transaction, so every later write on that connection is rolled back at close. Around it sit swallowed write failures, a connection leak and a lock-order inversion. This stream closes the transaction contract and makes failed writes audible.

This is a work-stream parent from the core-runtime code review of 2026-09-17 (`qa/core-code-review-2026-09-17/report.md`, all 29 slices, 887,855 lines). Its child tasks are the individual units of work; close this one when they are all closed. Findings are quoted in each child with the file and line they were verified at, and each slice's full evidence is in `qa/core-code-review-2026-09-17/slices/`.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All five subtasks Done. Shipped on PR #2710 (branch fix/core-review-crashes): .1 and .2 bare-DML durability, .3 audible write failures, .4 handle leak plus lock-order inversion plus cross-thread store access, .5 held-connection template drift. Carried forward and NOT in this stream: the two ChaChaNotes readers that use the raw connection as a context manager and so commit a caller-owned transaction (console_chat_store.py :12752 and :12802), and MediaDatabase(check_integrity_on_startup=True) always raising because the method does not exist.
<!-- SECTION:NOTES:END -->
