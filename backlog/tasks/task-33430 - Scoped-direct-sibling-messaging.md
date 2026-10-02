---
id: TASK-33430
title: Scoped direct sibling messaging
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-29 18:10'
updated_date: '2026-10-02 18:29'
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
- [ ] #1 Children can discover and message only live siblings under the exact current parent, coordinator and chain; self, foreign, terminal and revoked targets refuse.
- [x] #2 Peer delivery reuses bounded steering and shares the child reporting allowance; delivery receipts distinguish queued from consumed.
- [x] #3 Message bodies never appear in step summaries or run logs; catalog anti-forgery, cancellation and existing budgets remain enforced.
- [ ] #4 Targeted runtime and coordinator tests prove delivery, drain ordering, queue limits, ownership replacement and privacy.
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

CI addendum 2026-10-01: retain exact runtime tool-name equality while adding the two implemented peer tools to its expected inventory; run the affected install-skill and peer checks. ADR required: no; direct qualification of ADR-199.

Qodo addendum 2026-10-01: document peer admission inputs/receipts/refusals and validate tool arguments through a strict installed Pydantic model reusing _validate_text. Preserve exact shapes, original fixed refusal codes, Unicode/control/length checks, quotas and privacy. Strengthen the existing invalid-argument regression to prove refusal before capability invocation; run peer and startup guards. ADR required: no; direct boundary qualification under ADR-199.

Expanded-hooks child authority qualification. 1. Reproduce an otherwise eligible child losing list/send peers through pass-only SubagentStart prospective planning; pin empty/narrow controls. 2. Include the existing peer capability in the same prospective runtime plan used by hook narrowing; do not bypass selected hook IDs or alter tokens, quotas, catalogs or permissions. 3. Qualify actual threaded private delivery and shared report allowance with hooks, refusal/narrowing and retained-resume admission. ADR required: no new ADR. ADR path: backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md and backlog/decisions/163-expanded-console-hook-runtime.md. Reason: preserve existing peers under the shared hook constraint boundary; no new capability owner.
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

October 1 CI/integration closeout: preserved the exact runtime inventory assertion and added the existing list_peer_agents/send_to_peer names. The affected messaging/peer/installer/schema selection passes 145 cases in 58.50s, including existing real writer contention and revocation coverage. Independent latest-dev runtime/routing/recovery review approves the integration. Final static scan covers 75 changed Python files with zero added-line/new-file findings; ten new files pass full Ruff/format and whitespace is clean. Rebased onto dev 31d4f9b76492120706ba8e3ad7d355f1aa4e0273. Existing ADR-199 governs; no new architectural decision. Criteria remain satisfied and status is Done. PR #2918 remote checks will rerun on publication.

October 1 Qodo review: inspect the public send contract and strict tool argument boundary, complete Google-style peer API docs and use the installed Pydantic boundary pattern without weakening exact shape, no-coercion, content-free refusal or existing allowance validation. Existing ADR-199; no new contract or dependency.

October 1 Qodo follow-up complete: PeerMessenger list/send now document receipts, exact inputs and refusal conditions. The private strict Pydantic peer-argument model reuses the existing exact-type/nonblank/control/Unicode/length validator and preserves fixed error codes; exact shape and all capability/allowance custody checks remain intact. The strengthened existing invalid-argument test proves malformed payloads never reach the capability (RED: 6 failed / 2 passed, GREEN: 8 passed). The affected peer/inventory/fallback selection passes 88 cases in 49.27s. Boot imports 679/686 and UI-ready census 1031/1033 pass, proving no startup-budget increase. Independent peer privacy, actual delivery, shared final-slot allowance and lazy-import selection passes 12; final reviewer approves all reviewed changes with 22 focused checks. Final 76 changed Python/10new full-file lint+format qualification and diagnostic guard pass. Existing ADR-199; no new dependency or architecture. Done.
<!-- SECTION:NOTES:END -->
