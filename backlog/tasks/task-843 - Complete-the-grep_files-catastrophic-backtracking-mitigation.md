---
id: TASK-843
title: Complete the grep_files catastrophic-backtracking mitigation
status: To Do
assignee: []
created_date: '2026-07-27 02:36'
labels:
  - tools
  - security
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
grep_files bounds the regex input via a line-length cap, a total-lines-scanned cap and a 20s per-tool timeout. Those make the worst case finite and small but do not eliminate it: Python's re has no timeout, and _call_with_timeout abandons the worker thread rather than killing it, so a pathological pattern keeps burning CPU after the agent reports failure. A complete fix needs a regex engine supporting timeouts or a killable subprocess. grep_files carries the reads risk tag and floors to ask, which is why the partial mitigation was accepted. Filed from the PR #953 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pathological regex cannot consume CPU after its tool call has returned,Ordinary searches are not measurably slower,The chosen approach is documented with its trade-offs
<!-- AC:END -->
