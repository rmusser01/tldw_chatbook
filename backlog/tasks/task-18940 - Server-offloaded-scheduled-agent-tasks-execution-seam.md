---
id: TASK-18940
title: 'Server-offloaded scheduled agent tasks: execution seam and result pass-back'
status: To Do
assignee: []
created_date: '2026-08-19 11:05'
updated_date: '2026-08-19 11:05'
labels:
  - scheduling
  - agents
  - server
  - architecture
dependencies:
  - task-18937
  - task-18938
  - task-18939
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The strategic follow-up from the TASK-18936 parity audit, carrying the owner's stated long-term direction: **tldw_server is to be treated the same as hermes's hosted gateway — scheduled work is offloaded to the server to execute, and results are passed back to the client if so desired.** This closes the one capability where hermes runs scheduled agent work today and chatbook cannot: `agent_task` automations are fully modeled locally (ADR-018: families, lifecycle, health, preview, approval policies) but execution is `execution_unavailable` because ADR-018 deferred execution to server integration that never landed.

Scope: define and implement the first server-execution seam for `agent_task` automations — the client creates/previews a definition (already built), the server owns execution when the account is server-scoped (`owner_id="server:<user_id>"`), and completed results flow back through the existing sync/notification channels into the client's workbench (and optionally Console, reusing the existing handoff machinery). Local execution remains the offline fallback for local-owner definitions — this task adds the server path, it does not remove the local-first one. Hermes-precedent capabilities to fold into the contract: durable execution-audit history (the `AutomationAuditEvent` model exists), per-task model selection passed with the definition, and continuity/missed-fire reconciliation when the client reconnects (consumes TASK-18937's accounting).

This is architecture-first work: an ADR defining the client↔server execution contract (request shape, claim/lease semantics or server-side queue ownership, result-delivery channel, approval policy for server-side tool use, failure/timeout semantics, and what "passed back if so desired" means concretely — notification vs Console handoff vs workbench result row) must precede implementation. Sequenced after 18937–18939 so missed-fire, run-now, and timeout semantics exist as shared vocabulary in the contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ADR (amending/superseding the relevant ADR-018 clause) defines the server-execution contract: definition upload, execution ownership, result-delivery channel(s), approval policy for server-side tool use, failure/timeout semantics, and reconnect reconciliation — drafted and accepted before implementation begins
- [ ] #2 A server-scoped `agent_task` definition can be created/previewed locally, submitted for server execution, and its `health`/lifecycle transitions honestly (no more permanent `execution_unavailable` for server owners)
- [ ] #3 Completed server executions deliver results back to the client through at least one concrete channel (workbench result row, notification, or Console handoff — per the ADR), with the delivery visible in the UI and durable
- [ ] #4 Execution-audit history is durable end-to-end (client-visible audit trail of server executions, reusing `AutomationAuditEvent`)
- [ ] #5 Local-owner definitions keep today's local behavior unchanged — the local-first path is not regressed (pinned by tests)
- [ ] #6 Missed-fire/run-now/timeout semantics (18937–18939) reconcile correctly across a client reconnect (e.g. occurrences the server ran while the client was away are reported once, not re-derived or duplicated)
- [ ] #7 Per-task model selection rides the definition payload (hermes parity), bounded to the providers the server account can use
- [ ] #8 Live verification against a real tldw_server instance (per lessons-live-verification) — at minimum one real server-executed automation with result pass-back observed end-to-end; what was and was not verified live is recorded honestly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.
ADR path: backlog/decisions/071-server-offloaded-scheduled-agent-tasks.md (to be drafted before implementation; amends ADR-018's "execution remains execution_unavailable until server-side automation execution is integrated" clause).
Reason: cross-system service contract (client↔server execution ownership, result delivery, approval policy for server-side tool use) — squarely in ADR-required territory, and the owner has stated the long-term direction this task exists to realize.

1. Draft ADR-071: execution contract, result-delivery channels, approval policy, reconciliation semantics
2. Server client + service layer: definition submission, execution status polling/push, result retrieval
3. Client UI: health/lifecycle honesty for server owners; result row/notification/Console handoff per ADR
4. Audit trail wiring (AutomationAuditEvent end-to-end)
5. Reconciliation with 18937–18939 semantics; per-task model payload
6. Live verification against a real server; docs (schedules.md is a stub — this task should also give it its real content)
<!-- SECTION:PLAN:END -->
