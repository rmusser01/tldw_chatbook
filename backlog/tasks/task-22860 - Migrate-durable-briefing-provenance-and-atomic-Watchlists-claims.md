---
id: TASK-22860
title: Migrate durable briefing provenance and atomic Watchlists claims
status: Done
assignee:
  - '@codex'
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 08:09'
labels:
  - watchlists
  - briefings
  - database
  - migration
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - >-
    Docs/superpowers/plans/2026-08-27-watchlists-agent-boundary-and-provenance.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve ordered briefing evidence independently of mutable source and item rows, and enforce database-backed single-active claims for source checks and briefing generation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A versioned, transactional Subscriptions migration stores ordered briefing-item snapshots including item/source identity, featured/cited positions, sanitized URLs, dates, and a provenance format version.
- [x] #2 Existing junction rows migrate as `legacy_best_effort` without inventing selection or citation order; newly completed briefings write `ordered_snapshot` provenance before publishing `complete`.
- [x] #3 Completed briefing provenance remains readable and in original order after referenced source/item edits or deletion, while nullable live links may still expose current supplemental state.
- [x] #4 Partial unique indexes enforce at most one queued/running source-check receipt per source and one generating briefing per collection across threads/processes.
- [x] #5 Migration reconciliation deterministically keeps the newest active receipt and terminalizes older duplicates with fixed, non-sensitive recovery state.
- [x] #6 Owner-level accept/transition APIs resolve uniqueness races to the winning durable receipt and release claims on every terminal transition.
- [x] #7 Migration rollback, legacy upgrade, deletion survival, duplicate reconciliation, idempotent reopen, and the complete Subscriptions migration suite pass against temporary databases only.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build a genuine historical v1 Subscriptions fixture and add RED tests for transactional upgrade, rollback, duplicate reconciliation, idempotent reopen, read-only rejection, and deletion-surviving ordered provenance.
2. Implement the inline transactional v1->v2 migration in Subscriptions_DB.py, including schema-version bootstrap correctness, sanitized legacy snapshots, deterministic active-receipt reconciliation, and partial unique indexes.
3. Add DB-owned atomic accept/transition primitives that return the durable winner on uniqueness races and release claims for every terminal state.
4. Move briefing provenance plus successful publication behind one DB transaction; preserve first-seen citation order and prove injected failures leave neither partial snapshots nor false completion.
5. Run the complete temporary-database Subscriptions migration/readiness and briefing-selection/service surfaces, Ruff, diff checks, self-review, and independent review.

ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: ADR-032's approved addendum already owns durable Watchlists receipt/provenance authority and the Console/external boundary; this task directly implements that existing database/runtime contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the inline Subscriptions v1-to-v2 migration under one explicit BEGIN IMMEDIATE. Fresh databases create v2 directly with one schema_version row; historical v1 upgrades rebuild immutable ordered briefing provenance, copy sanitized legacy snapshots (including normalized effective dates), reconcile duplicate active receipts deterministically, create partial unique indexes, and update the version last. Injected failures restore the exact v1 table/index/version state.

Added DB-owned source-run and briefing claim acceptance/start/terminal APIs. They resolve only exact unique-claim conflicts under the same reserved write transaction, re-raise other integrity errors, release claims on terminal states, and prevent late overwrite. Scope and scheduler honor acquisition ownership; losers observe the durable winner with bounded monotonic backoff and cannot execute, record failure, or mutate winner/source accounting on timeout or cancellation.

Successful briefing publication now snapshots selection and first-seen citation order before a guarded final complete update in the same transaction. Provenance remains ordered and readable after live source/item edits or deletion.

ADR required: yes; existing backlog/decisions/032-local-agent-tool-permission-boundary.md applies. No new ADR, dependency, or split migration asset. Fresh controller verification: 277 targeted migration/briefing/local/scope/scheduler tests passed; Ruff and git diff --check passed. Independent review approved with no findings. The known Requests dependency-version warning remains; no full repository suite ran. All verification databases were temporary/in-memory; the user Subscriptions database was not opened or mutated.
<!-- SECTION:NOTES:END -->
