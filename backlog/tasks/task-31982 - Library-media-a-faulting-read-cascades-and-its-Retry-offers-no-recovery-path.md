---
id: TASK-31978
title: 'Library media: a faulting read cascades and its Retry offers no recovery path'
status: To Do
assignee: []
created_date: '2026-09-07 22:48'
labels:
  - library
  - media
  - robustness
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #6 P1, adjudicated from Assessment A's undo P0. The undo handler itself is correct (it counts rows actually restored and increments the rail by exactly that, `library_screen.py:24537-24552`; the N-of-M receipt counts genuine restore exceptions), and Assessment B got a clean undo on the same code, so the catastrophic case required a real media-DB fault. The deterministic residue: when the media read faults, the failure disables independent controls that did not need that connection (Export, Select, Review these, the facet counts, the whole Trash view); the fault-state Retry repeats one sentence with no attempt counter and no next step; and a restore that returns a non-Mapping is counted as neither success nor failure, so the receipt count can drift. Related: task-31972 (selection reconcile id-shape), task-31942 (analysis-save commit).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A media read failure is scoped to the failing surface; Export, Select, Review, facet counts and Trash remain usable when their own reads succeed
- [ ] #2 The fault-state Retry either reconnects or, when it cannot, states the recovery action (e.g. Reopen Chatbook to reconnect to the media database) instead of repeating the same sentence
- [ ] #3 The bulk-undo receipt count reflects rows the write layer actually committed, including a restore that returns an unexpected shape
- [ ] #4 A test reproduces a faulting media read and asserts the independent controls stay live
<!-- AC:END -->
