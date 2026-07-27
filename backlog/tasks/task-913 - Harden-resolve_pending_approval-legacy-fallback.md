---
id: TASK-913
title: 'Harden resolve_pending_approval legacy round_id fallback'
status: To Do
assignee: []
created_date: '2026-07-27 03:55'
labels: [console, approvals, hardening]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
resolve_pending_approval's round_id=None fallback (production-unreachable; kept for legacy direct-call tests) scans _pending_approval_rounds.values() unlocked while a worker thread's finally can pop concurrently, and resolves by active session. Its twin resolve_pending_skill_script fails closed on a missing request_id. Make the fallback fail closed (or snapshot with list()) and migrate the legacy tests to pass round ids.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No unlocked live-dict iteration remains in the fallback path.
- [ ] #2 round_id=None either fails closed like resolve_pending_skill_script or is removed with tests migrated.
<!-- AC:END -->
