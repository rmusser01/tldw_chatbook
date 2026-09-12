---
id: TASK-32020
title: Design incremental fleet completion delivery and wake fairness
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 03:03'
updated_date: '2026-09-08 06:39'
labels:
  - agents
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Completion delivery waits for the entire conversation fleet to drain, and one pending approval in an active wake occupies the global wake serialization slot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document a deterministic slow-sibling and blocked-approval scenario with delivery latency.
- [x] #2 Choose an incremental completion and fairness policy without duplicating results or changing approval authority.
- [x] #3 Separate completion notification timing from the last-child usage reconciliation contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md
Reason: Changes the completion event contract and automatic scheduling policy in ADR-129/134.
1. Reproduce slow-sibling and blocked-wake delays with deterministic bridge/controller gates.
2. Separate individual completion notification from the final-drain usage event.
3. Specify bounded per-conversation wake fairness, manual reserves, and chain-coalescing tradeoffs.
4. Record delivery/crash decisions, implementation dependencies, and verification scenarios.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the incremental notification and fairness design in backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md, amending ADR-129/134. Individual durable terminal settlement becomes the notification signal; final-drain usage reconciliation remains unchanged. The chosen scheduler uses a fixed 250 ms coalescing window, one causal chain per batch, round-robin eligible conversations, and at most two wakes subject to the existing primary cap and one manual reserve. Approval waits retain ownership and never resolve themselves.

Evidence: Tests/Chat/fleet_delivery_probe.py measured a ready child held for 353 ms by its sibling and zero second-conversation rows during a 350 ms real approval hold. Results and limitations are in backlog/docs/agent-fleet-delivery-baseline-2026-09-08.json. Three opt-in probes passed; the final 163-test regression run includes existing wake/approval contracts. Probe lint/format passes. Self-reviewed event ordering, mixed chains, fair scheduling, and both delivery paths.

This is a completed design task, not an implemented scheduler. TASK-32037 acceptance criteria and Docs/superpowers/plans/2026-09-08-fleet-delivery-and-recovery.md now carry the runtime work. Review ledger and related ADR index updated. No full suite or live provider test.
<!-- SECTION:NOTES:END -->
