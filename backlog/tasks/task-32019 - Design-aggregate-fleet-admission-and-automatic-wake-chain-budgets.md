---
id: TASK-32019
title: Design aggregate fleet admission and automatic wake chain budgets
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 03:03'
updated_date: '2026-09-08 05:29'
labels:
  - agents
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Per-conversation concurrency and per-run budgets do not bound total background work across conversations or repeated automatically spawned wake generations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Record measured cross-conversation and repeated-wake behavior with deterministic probes.
- [x] #2 Specify aggregate admission and wake-chain budget ownership, defaults, reset rules, and visible exhaustion behavior in an ADR before implementation.
- [x] #3 Preserve manual work priority and prevent budget resets from bypassing the chosen aggregate bound.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed context/ADR review and four deterministic baseline probes in Tests/Chat/test_fleet_budget_boundary_probes.py. The user selected conservative caps. Follow Docs/superpowers/plans/2026-09-08-fleet-admission-and-wake-budgets.md for implementation task sequencing. Validate the baseline, headless/safety contracts, scoped static checks, and document the distinction between accepted design and unenforced runtime limits. ADR required: yes. ADR path: backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md. Reason: runtime ownership, automatic-chain budgets, reset/recovery policy, and manual capacity reserves.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the aggregate-admission design after the user selected conservative defaults. Four deterministic local probes measured eight simultaneous children across four conversations, six successive machine-origin wakes without a user send, one actual child worker after its handle was marked terminal, and four tool workers still alive after four timeout returns. Probes use the real service/controller boundaries with gated fake providers; the wake probe injects completion events and does not claim model-generated recursion.

ADR required: yes. ADR path: backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md. The decision specifies app-owned physical-work leases, manual reserves, finite automatic-chain generations/calls/tokens/time, atomic in-flight reservations, immutable causal identity, conservative unknown-usage handling, restart/reset rules, and visible paused results. Defaults are six child slots (two manual reserves), eight tool slots (two reserves), three automatic wakes, six automatic child launches, 32 calls, 500000 budget tokens, an 8192-token per-call output ceiling, and 900 seconds per automatic chain. Limits are design decisions and are not yet enforced in production.

Implementation is split into dependency-ordered atomic Backlog records. Plan: Docs/superpowers/plans/2026-09-08-fleet-admission-and-wake-budgets.md. Evidence: backlog/docs/agent-fleet-budget-baseline-2026-09-08.json and Tests/Chat/test_fleet_budget_boundary_probes.py. Updated the orchestration ledger and ADR index. Self-review covered retained cleanup, inline bypass, uncertain outcomes, mixed old/new chains, approval waits, and plain-provider wakes.

Final verification: 168 targeted tests passed in 46.70 seconds after repairing the separately tracked legacy headless test setup. New probe lint/format, changed-range format, no-added-lint comparison, and scoped whitespace checks pass. No full suite or live provider used; production runtime and diagnostic sites unchanged in this design pass. Tests characterize the current gaps, not an implemented admission guarantee. Changes remain uncommitted.
<!-- SECTION:NOTES:END -->
