---
id: TASK-34214
title: Console shared automatic allowances and chat-start attempts
status: Done
assignee:
  - '@codex'
created_date: '2026-10-02 23:24'
updated_date: '2026-10-03 06:27'
labels:
  - console
  - agents
dependencies:
  - TASK-34213
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep agent-created chats within the originating automatic-work allowance while preserving each conversation's run ownership and durable execution fences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing and migrated manual chains retain their finite limits and conversation ownership; automatic descendants refer directly to an immutable canonical allowance root.
- [x] #2 Counters, deadlines, pause/review state, settlement uncertainty, and recovery apply across every member of a shared allowance, including concurrent reservations and late callbacks.
- [x] #3 Durable exact-target chat-start attempts prepare, accept once, abort only before acceptance, and retain accepted or uncertain charges without replay after restart.
- [x] #4 Automatic execution context validates native chat-start attempts and refuses stale runtime owners without fabricating wake claims or cross-conversation run parents.
- [x] #5 Targeted SQLite, migration, race, deadline, and automatic-context checks pass with the existing manual and fleet-wake behavior preserved.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/211-console-chat-destinations-and-bounded-starts.md
Reason: Implements the approved shared automatic-allowance ownership and durable chat-start authority contract.

Detailed plan: Docs/superpowers/plans/2026-10-02-console-chat-destinations-and-starts.md, Task 1.
Spec: Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md.

1. Capture targeted baseline, current schema and formatter evidence.
2. Add real-SQLite failing shared-allowance and native-attempt tests.
3. Implement AgentRunsDB migration, direct immutable root membership and root-wide admission/settlement/recovery.
4. Implement native start-attempt preparation/acceptance/abort/completion and accepted automatic context, preserving run conversation scope.
5. Verify races, deadlines, migration, uncertain accounting and existing wake/manual behavior; update implementation notes only after completion.

Allocation provenance: CLI assigned TASK-33803. A fresh scan of all locally available Git refs and 114 worktrees found IDs through 33803; this task was renamed to TASK-33804 before references were added.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the shared-allowance and native chat-start foundation under [ADR-211](../decisions/211-console-chat-destinations-and-bounded-starts.md). Existing local chain/run ownership guards remain intact; target primaries retain no cross-conversation parent.

AgentRunsDB v19 adds direct immutable allowance-root membership, indexed root-wide accounting, and body-free exact-target native attempts. Preparation reserves one generation atomically with target membership; acceptance consumes it once, abort refunds only prepared uncommitted work, and completion has no survivor delivery side effects. Wakes and starts check both active-attempt tables. Admission/refusal observations, deadlines, pause/review state, usage settlement and startup recovery resolve the canonical root. AutomaticWorkContext selects the explicit wake/chat_start reader and keeps its acceptance latch unset until the coordinator marks both fences complete.

Both runtime and standalone agent_runs_v18_to_v19_chat_starts.sql upgrades preserve historical roots, limits, reservations and wake records; SQLite foreign_key_check passes. Tests exercise standalone membership guards before runtime opening, shared resource balances, concurrent last-slot and accept/abort races, recursive lineage, SQL immutability/scope, uncertainty/overage, original clocks/deadlines and stale-owner recovery.

Changed DB/AgentRuns_DB.py, DB/automatic_work.py, Agents/automatic_work_budget.py, Agents/automatic_work_runtime.py, the v19 migration, new Tests/DB/test_automatic_chat_starts.py and five existing DB/Chat budget, deadline, migration, wake and lineage test files.

TDD evidence: initial missing-interface RED 9 failed; descendant settlement/context RED 4 failed; focused GREEN 31 passed. Final focused closure: 100 passed in 35.07s across the six changed test files. Scoped Ruff E9/F63/F7/F82, new-file formatting, inherited formatter-debt ratchet and git diff --check passed. No full test sweep or live dispatch was run; this foundation exposes no tool/UI start route. Self review preserved accepted ADR bodies and existing ownership boundaries.

Independent Task 1 review: spec compliant and quality Approved; no blocking findings. Root accounting, local run ownership, native CAS, recovery and both migration paths verified. Live source ownership, shared capacity, conversation receipt and post-receipt acceptance latch are explicit the feature implementation task obligations. ADR-211 applies. Final scoped evidence: 100 tests passed plus lint, new-file formatting and formatter ratchet.

Canonical report/review and exact verification logs: Docs/superpowers/qa/2026-10-02-console-chat-starts/README.md. the feature implementation task task review now confirms live ownership, shared physical capacity, the second conversation receipt and acceptance-latch order. Both task gates and the final whole-branch review plus its single scoped fix review are approved. Final fix 9cb68f456e preserves the foundation ownership contracts; complete evidence and qualifications are in the canonical QA record. The named branch remains for the user integration decision.
<!-- SECTION:NOTES:END -->

Current-dev publication identity: 34214 replaces unmerged 33804. The landed TASK33802 census keeps its identity; the design/foundation/feature chain moved together to preserve dependency ordering. Historical QA retains original identifiers and bytes. Current evidence: Docs/superpowers/qa/2026-10-03-console-chat-starts-dev-integration/README.md.

Current-dev integration note: the historical implementation notes above describe the reviewed source revision and its original schemas. TASK34215.2 qualifies the integrated branch with AgentRuns22 and ChaChaNotes76, current native continuation/maintenance owners and new QA. Historical verification artifacts retain their original schema names and bytes.
