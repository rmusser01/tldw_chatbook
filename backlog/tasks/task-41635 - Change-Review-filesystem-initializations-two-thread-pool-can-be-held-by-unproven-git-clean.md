---
id: TASK-41635
title: "Change Review filesystem initialization's two-thread pool can be held by unproven git cleanups"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workspaces
- console
priority: medium
references:
- tldw_chatbook/Workspaces/change_review_consent.py
dependencies:
- TASK-41307
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** Change Review runs filesystem initialization on a fixed daemon pool (`Workspaces/change_review_consent.py`, `worker_count: int = 2`), so that a slow root never creates an unbounded thread population. A git cleanup whose return was never proven stays retained (ADR-134/285), and TASK-41307's Release frees only the tool-worker slot. Two such cleanups can hold both pool threads, and every later Change Review initialization, for any root, then waits behind them with nothing on screen to say why.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A retained unproven git cleanup cannot block Change Review initialization for other roots, or the UI states why initialization is waiting and what releases it
- [ ] #2 A test holds two unproven cleanups and starts a third initialization, asserting the chosen behaviour
<!-- AC:END -->
