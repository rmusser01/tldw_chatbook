---
id: TASK-913
title: 'Harden resolve_pending_approval legacy round_id fallback'
status: In Progress
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
- [x] #1 No unlocked live-dict iteration remains in the fallback path.
- [ ] #2 round_id=None either fails closed like resolve_pending_skill_script or is removed with tests migrated.
<!-- AC:END -->

## Implementation Notes

AC #1 done as part of the Qodo-wave PR2 restack (F3b): `resolve_pending_approval`'s
`round_id=None` legacy fallback now takes its `_pending_approval_rounds.values()`
scan under the new `_approval_state_lock` (`with lock: round_states =
list(self._pending_approval_rounds.values())`), snapshotting before iterating
rather than walking the live dict a worker thread's `request_mcp_approvals`
can concurrently register/pop entries in. The `round_id is not None` branch's
`.get()` was guarded too, for consistency with every other access to this map.

AC #2 (the fail-closed-vs-remove behavioral decision for the `round_id=None`
fallback itself) was explicitly out of scope for the fix this pass authorized
(a locking/thread-safety hardening pass, not a behavioral redesign of a
production-unreachable legacy path) -- changing it would also require
migrating every direct-call test that currently relies on the "resolves
whichever round belongs to the active session" fallback, which risks
destabilizing a fully-reviewed, live-smoked branch for a change nobody asked
for in this pass. Left unchecked and the task left in `In Progress` (not
`Done`) rather than silently closing an unimplemented AC. Recommend either a
follow-up task scoped to AC #2 alone, or re-scoping this task's AC list down
to #1 (already satisfied) if AC #2 is not actually wanted.

