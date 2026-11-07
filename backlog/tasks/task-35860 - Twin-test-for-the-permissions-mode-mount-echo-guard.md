---
id: TASK-35860
title: Twin test for the permissions-mode mount-echo guard
status: To Do
created_date: 2026-10-21 04:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-35151's review (PR follows) copied MCPRail's one-shot mount-echo guard onto the permissions profile Select (mcp_permissions_mode.py) — the guard is shape-identical to the rail's battle-tested one, but only pinned indirectly via the handoff test. If the burn-to-sentinel logic regressed into a standing sentinel, a user's default→X→default round trip would silently lose its final dispatch and no permissions test would fail. Add the twin of test_scope_a_b_a_dispatches_three_changes_and_mount_echo_zero for the permissions Select.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Twin test pins one-shot burn semantics on the permissions profile Select
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
