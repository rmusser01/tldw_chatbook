---
id: TASK-33430
title: Scoped direct sibling messaging
status: Done
assignee:
  - '@codex'
created_date: '2026-09-29 18:10'
updated_date: '2026-09-29 19:28'
labels:
  - agents
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow live sibling agents to exchange bounded untrusted messages within their exact parent and work chain, without acquiring each other's permissions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Children can discover and message only live siblings under the exact current parent, coordinator and chain; self, foreign, terminal and revoked targets refuse.
- [x] #2 Peer delivery reuses bounded steering and shares the child reporting allowance; delivery receipts distinguish queued from consumed.
- [x] #3 Message bodies never appear in step summaries or run logs; catalog anti-forgery, cancellation and existing budgets remain enforced.
- [x] #4 Targeted runtime and coordinator tests prove delivery, drain ordering, queue limits, ownership replacement and privacy.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace exact child attachment, message allowance, steering drain and private runtime projections.
2. Add failing focused tests for scoped peers, bounds, drain ordering and privacy.
3. Bind child peer capability through the existing sender and coordinator; expose reserved runtime tools and project source metadata only.
4. Run targeted coordinator/runtime/integration tests and focused Ruff checks.
ADR required: yes
ADR path: backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md
Reason: Accepted ADR-199 amends ADR-136 for scoped direct sibling authority and steering transport.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented child-only list_peer_agents and send_to_peer through an exact attached PeerMessenger capability. Discovery and delivery require a known immutable chain plus matching parent, coordinator and inbox owner; self, foreign, terminal, pruned, copied and revoked capabilities refuse. Peer sends reuse the existing steering drain and atomically share the report sender lifetime allowance; full recipient queues spend no allowance. Cancellation revokes message authority before child settlement.

Added generated source/message/target metadata with queued receipts and consumed drain records. Peer bodies are untrusted provider context and are omitted from step, display and run-log projections. Fixed the shared fenced-model log projection so message-tool arguments cannot leak through model records. Reserved both names in runtime/catalog discovery and Library skill collision checks; restored pending peer calls refuse.

Changed Agent models/runtime/service, fleet coordinator/message tools/sender allowance, catalog and Library name guard; added focused peer tests and the Console user-guide paragraph. Architecture: backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md (amends ADR-136).

Validation: 96 targeted checks passed using the main checkout .venv Python with private basetemp/logs: test_fleet_peers, test_fleet_messages, test_fleet_coordinator, restored pending message-call cases, and Library shadow-name drift guard. New test file Ruff lint and formatting plus fleet_message_tools formatting passed; git diff --check passed. Separate steering sweep: 41 passed; three older service harness cases and two teardown errors hit documented RecoveryRequired/raw_source_selection_changed before child execution. No full suite run.

Task remains In Progress with AC unchecked for root review as requested; no commits or Git metadata changes.

Root integration revalidation: repaired only the three config-aware steering service nodes to retain their bound bootstrap source. The full affected steering mailbox module now passes all 22 tests (/private/tmp/agent-burndown-steering-profile.log). The earlier profile failures are resolved without changing production admission gates.

Final disposition 2026-09-29: independent read-only implementation review approved the scoped repair with no actionable findings. The targeted acceptance checks and changed-line static checks recorded above pass; inherited whole-file lint/format debt remains outside this correctness task. All acceptance criteria are checked and this task is Done. No full-suite or live-provider qualification is claimed. This disposition supersedes earlier pending-review notes.
<!-- SECTION:NOTES:END -->
