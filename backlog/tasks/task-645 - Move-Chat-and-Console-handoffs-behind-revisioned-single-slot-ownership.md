---
id: TASK-645
title: Move Chat and Console handoffs behind revisioned single-slot ownership
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 13:37'
updated_date: '2026-07-26 21:35'
labels:
  - architecture
  - state
  - reliability
dependencies:
  - TASK-644
references:
  - backlog/decisions/026-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-application-session-state-ownership-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the raw Chat and Console pending application fields with typed, memory-only, revisioned single-slot handoff ownership while preserving current retry, replacement, and consume-once behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A PendingHandoffStore owns typed Chat and Console channels, normalizes and structurally detaches staged values including nested mappings, and remains memory-only
- [x] #2 A claim is exclusive, and acknowledge or release affects only the exact claimed revision so a newer replacement cannot be cleared
- [x] #3 The pending_chat_handoff, pending_console_launch, and pending_console_prompt_insert application fields are migrated to the owner
- [x] #4 Success, terminal rejection, and transient failure semantics preserve current mount, setup, tab-creation, and retry behavior
- [x] #5 Failure or cancellation after creating an exact ephemeral Chat handoff tab closes that tab before releasing; cleanup failure terminally acknowledges with bounded recovery so retry cannot create duplicate partial tabs
- [x] #6 Deterministic concurrency, off-owner mutation, cancellation/rollback injection, privacy-redaction, mounted-flow, static, and ownership-guard tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/026-application-session-state-ownership.md
Reason: ADR-026 defines revisioned single-slot delivery, replacement, settlement, privacy, and thread affinity.
Full plan: Docs/superpowers/plans/2026-07-26-task-645-chat-console-handoffs.md

1. Add the typed revisioned PendingHandoffStore with detached stage and claim values.
2. Migrate Chat and Console producers.
3. Migrate Console launch and prompt consumers with setup/readiness release and outcome settlement.
4. Make Chat delivery transactional with exact ephemeral-tab rollback.
5. Add deterministic concurrency, cancellation, privacy, mounted-flow, and ownership guards, then keep TASK-645 In Progress until the shared TASK-646 release gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-026 PendingHandoffStore with typed, detached, revisioned, memory-only Chat and Console channels. Producers and consumers now use exclusive claim plus exact-revision acknowledge/release semantics, preserving single-slot replacement and retry behavior. Chat delivery performs bounded rollback of the exact ephemeral tab on failure/cancellation and terminally settles cleanup failures to prevent duplicate partial tabs. Raw application pending Chat/Console fields were removed and guarded structurally.

ADR required: yes
ADR path: backlog/decisions/026-application-session-state-ownership.md
Reason: ADR-026 governs revisioned single-slot delivery, settlement, privacy, and owner-thread affinity.

Verification: deterministic replacement, cancellation, rollback, cleanup failure, redaction, actual mounted production-app flows, and ownership guards were included in the authorized integrated suite: 473 passed with one pre-existing requests dependency warning in 259.60s. Scoped compileall, Ruff, format, and git diff checks passed. Application behavior was exercised through normal TldwCli instances and real screens only.
<!-- SECTION:NOTES:END -->
