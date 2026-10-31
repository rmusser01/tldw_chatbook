---
id: TASK-36960
title: >-
  Dev: chat_screen.py is one line over its screen-size ratchet row since
  9688ac373f
status: To Do
assignee: []
created_date: '2026-10-26 01:53'
labels:
  - ci
  - architecture
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while rebasing TASK-33620.15.4 onto dev 7a5e2513c5 (2026-10-25). Commit 9688ac373f (perf: defer unused Console marker and token-rate imports, 2026-10-25 00:18) grew tldw_chatbook/UI/Screens/chat_screen.py from 24336 to 24345 lines, and the ChatScreen row in Tests/Architecture/test_screen_size_ratchet.py stayed at 24344. On a detached worktree of dev 7a5e2513c5 the test fails: 'chat_screen.py grew to 24345 lines (budget 24344, +1)'. The ratchet is not in the PR fast lane (derived-artifacts.yml) and the full-tree runs (test.yml on main pushes or dispatch, nightly-deep.yml on schedule from main) do not gate dev merges, so it merged red and every branch rebased onto dev inherits the +1. Still red on dev e554f25812 (same +1). Branches must not absorb it into their own row: that hides which change owns the line.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests/Architecture/test_screen_size_ratchet.py passes on dev
- [ ] #2 The ChatScreen row is the exact measurement, with a dated comment naming 9688ac373f's growth (owner rule: a fix over a size ratchet raises the row) or the extra line is removed
<!-- AC:END -->
