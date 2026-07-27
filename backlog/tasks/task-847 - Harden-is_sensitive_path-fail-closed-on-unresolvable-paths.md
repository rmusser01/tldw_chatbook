---
id: TASK-847
title: Harden is_sensitive_path fail-closed on unresolvable paths
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
is_sensitive_path's docstring promises it fails CLOSED for a path it cannot resolve, but _resolved() catches only OSError and RuntimeError. A path containing a NUL byte raises ValueError instead of returning True. Unreachable today because validate_path rejects such a path first and the file tools carry an outer catch-all, but the comment overstates the guarantee. Filed from the PR #953 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 is_sensitive_path returns True (refused) for a path that cannot be resolved for any reason including ValueError,A regression test covers the NUL-byte case,The docstring's fail-closed claim matches the code
<!-- AC:END -->
