---
id: TASK-25713
title: Console shows no status while a reply is pending
status: To Do
assignee: []
created_date: '2026-08-31 05:07'
labels:
  - console
  - ux-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An assistant row mounts as an empty bordered block with no spinner, elapsed timer, or state label. Against a provider that answers in 0.8s, Console showed this blank block for over 30 seconds before any card appeared. A pending reply, a stalled run, and a silently failed run are visually identical, which is the highest-frequency moment in the product.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pending assistant row always shows a live state label distinguishing generating, waiting on an action, and failed
- [ ] #2 Elapsed time is visible while a reply is pending
- [ ] #3 A run that ends without content renders an explicit outcome instead of an empty block
<!-- AC:END -->
