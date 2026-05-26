---
id: TASK-70.4
title: Add Sync v2 conflict review and recovery plan
status: To Do
labels:
- sync
- sync-v2
- conflicts
- recovery
priority: high
parent_task_id: TASK-70
documentation:
- Docs/superpowers/specs/2026-05-26-chatbook-sync-v2-completion-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Sync v2 conflict and partial-failure states actionable enough for manual Notes and Chat sync users to inspect, recover, retry, or defer without losing local-first control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Conflicts display domain, item label, cause, local summary, remote summary, and safe recovery options.
- [ ] #2 Partial push failures remain durable and visible until resolved.
- [ ] #3 Retry, keep-local, accept-remote, duplicate/fork, and defer-later states are modeled or explicitly marked unavailable.
- [ ] #4 Tests cover conflict persistence, user-facing mapping, and no cursor advancement after failed apply.
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
