---
id: TASK-32036
title: Persist causal automatic-work chains and atomic budget reservations
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 05:21'
updated_date: '2026-09-12 03:36'
labels:
  - agents
  - console
dependencies:
  - TASK-32019
  - TASK-32021
references:
  - backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md
  - backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Automatic wake generations need one durable allowance so new runs, old survivors, retries, and restarts cannot create fresh unattended budgets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Accepted explicit user work establishes an immutable chain identity inherited by automatic turns and children; old survivors keep their original chain after new manual work.
- [x] #2 Durable uniquely keyed reservations enforce generation, child-launch, provider-call, and estimated-token admission atomically across concurrent callers.
- [x] #3 Acceptance, pre-start failure, repeated completion, cancellation, and unknown usage follow the conservative commit/refund rules in ADR-134 without changing recorded billing.
- [x] #4 Reopening the database preserves counts, limits, deadline, and uncertain reservations; legacy or ambiguous chains cannot silently receive a new automatic allowance.
- [x] #5 Real SQLite migration/reopen, contention, duplicate-reservation, rollback, and mixed-lineage tests pass.
- [x] #6 Wake batch claims and typed attempt transitions atomically bind one chain and owner; restart uncertainty retains allowance and blocks replay without invalidating live owners on DB handle reopen.
- [x] #7 Automatic-work acceptance and reservation writes commit with FULL synchronization before granting dispatch authority, including reopened and alternate-thread connections.
- [x] #8 Wake lineage is taken from coordinator-selected durable result rows and bound to the exact live authorization; mixed chains are delivered separately and stale tokens cannot attribute later work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md and backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md
Reason: Direct implementation of accepted chain, reservation, and attempt contracts.
1. Add failing real SQLite tests for immutable lineage, finite shared reservations, duplicate/rollback/unknown-usage handling, and FULL-synchronized commits.
2. Add the next guarded AgentRunsDB migration, typed budget results, and atomic reservation/attempt APIs; explicit startup recovery must not run on DB handle reopen.
3. Add failed-claim and acceptance/abort/complete/recovery contention tests and implement the durable fence without external side-effect claims.
4. Pass immutable accepted-manual chain context through controller, bridge, service, children, and the plain-provider path; preserve old survivor lineage.
5. Verify targeted DB migrations/accounting and real controller/fleet lineage tests, static checks, self-review, and documentation. Runtime budget admission and scheduler integration remain TASK-32037.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented schema v17 and its standalone v16 upgrade under ADR-134/135. The typed ledger at db.automatic_work uses the existing AgentRunsDB connection owner with FULL-synchronized reservation and wake-attempt transactions. Exact ownership, unique source claims, conservative settlement/recovery, and immutable limits preserve allowance without altering provider billing.

Accepted manual turns establish durable conversation-scoped chains off the event loop; bridge/service children preserve ancestry and plain wakes carry private chain context. Pending wakes separate chains and reject stale authorization tokens. Storage failure prevents dispatch and clears stream ownership.

Review regressions fixed token overage across resource types, legacy parent/attachment scope, and elapsed-time loss across handles and simulated process replacements. The fixed deadline and process clock anchor survive refused admission without retaining attempted work. No runtime reservation enforcement or replay-safety claim is made here; dispatch integration remains TASK-32037.

Verification: 1,043 targeted DB/service/controller/fleet/provider tests passed; two local HTTP lifecycle cases passed separately with loopback access. The final clock corrections passed 60 focused ledger/migration/lineage tests, and independent review passed 20 transition/deadline cases plus a real SQLite residue probe. New modules pass Ruff lint/format; edited existing production files add no lint findings, and their diagnostic-call ASTs are unchanged. No full suite or external provider calls. Simulated process identity and real SQLite reopen were tested, not killed-process/power-loss durability.

Documentation: updated both fleet implementation plans, ADR-134 and ADR-135 implementation records, the agent-runs user guide, the orchestration review ledger, and the testing-evidence lesson. API placement on db.automatic_work replaces the plan's proposed flat DB method names while retaining the same storage boundary. ADR paths: backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md; backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md.

PR #2631 integration preserves upstream schema versions 13–15 and assigns this workstream versions 16–18 under ADR-131/135. Runtime and standalone migrations preserve indexed steps, spawn identities and activity receipts. Final review corrected parent-stuck wake eligibility to use the canonical terminal parent set; the five-state regression failed before the fix, and 71 affected DB/dispatch/recovery cases passed afterward. Lazy ledger construction avoids startup work; index-plan census pins all three retained automatic-work indexes to real queries.
<!-- SECTION:NOTES:END -->
