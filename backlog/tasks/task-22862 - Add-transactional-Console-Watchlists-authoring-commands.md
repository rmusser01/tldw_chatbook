---
id: TASK-22862
title: Add transactional Console Watchlists authoring commands
status: Done
assignee:
  - '@codex'
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 22:44'
labels:
  - watchlists
  - console
  - tools
  - ux
dependencies:
  - TASK-22859
references:
  - >-
    Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - >-
    Docs/superpowers/plans/2026-08-27-console-watchlists-commands-and-operations.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user create multiple sources, create a collection, and update collection membership through approval-gated Console domain commands with explicit collision and partial-result semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `watchlists_create_sources` validates 1–50 rows before writing, rejects URL userinfo and unsafe values, preserves input order, and reports `created`, `existing`, or `invalid` with canonical IDs without echoing queries/fragments.
- [x] #2 Exact configured-source identity is outer-whitespace-trimmed only, and a database-owner write-intent batch prevents Console, UI, or OPML callers from racing duplicate lookup/insert.
- [x] #3 Mixed source outcomes return `partial_success` plus `follow_on_confirmation_required`; no dependent collection mutation occurs until the user explicitly confirms the reduced source set.
- [x] #4 `watchlists_create_collection` implements explicit `conflict`, `return_existing`, and `auto_suffix` policies; returning an existing collection never mutates it.
- [x] #5 New collection creation and up to 100 validated memberships commit atomically and do not implicitly schedule, check, or generate a briefing.
- [x] #6 `watchlists_update_collection_sources` rejects overlapping add/remove sets and missing/ambiguous IDs, then applies all validated membership changes or none.
- [x] #7 All three commands are Console-only, carry mutation approval effects/tags and definitive-after-start execution ownership, reject server mode before storage access, and have concurrency/rollback/redaction/provider-schema/runtime coverage; after approved execution begins, timeout or cancellation cannot return while a mutation can still commit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add race and validation RED tests, then implement one BEGIN IMMEDIATE database-owner exact-source batch used by Console, direct UI, and OPML.
2. Add collision-policy and atomic-membership RED tests, then implement explicit conflict/return-existing/auto-suffix creation and all-or-nothing membership updates.
3. Build the synchronous WatchlistsCommandService with exact schemas, direct Console-worker calls into application-owned synchronous mutation seams, a fixed allowlisted in-transaction result projection (no caller callback), definitive commit/rollback outcomes, redaction, server-mode refusal, and no implicit follow-on work.
4. Register the three mutation descriptors as Console-only with code-owned mutation effects/tags, definitive-after-start execution ownership, and sanitized destination presentation; carry the policy and native call identity through the catalog/runtime and approval payload without name lists, retain approved rows as disabled per-call keyed finishing cards until BaseException-safe real tool/run completion, treat finishing as status rather than pending approval with a valid keyboard focus target, prove pre-start cancellation remains interruptible, and prove read-only bindings, external MCP, and ordinary bounded tools retain their respective contracts.
5. Run complete task-targeted tests, Ruff, diff checks, self-review, and independent review.

ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: ADR-032 and its approved addendum define the Console-only mutation and approval boundary. Its TASK-22862 execution-ownership amendment records why short SQLite mutations return definitively from the Console tool worker instead of crossing a non-cancellable timeout bridge; no duplicate ADR is required.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented three approval-gated, Console-only Watchlists authoring commands over shared domain owners. Exact source batches use a database-owned `BEGIN IMMEDIATE` path with fixed safe result projection; collection collision and membership policies are explicit and atomic; partial source results require confirmation before follow-on work.

ADR-032 was amended to define definitive-after-start execution ownership. Native call IDs now flow through local approval review, approved mutations remain visible as disabled finishing status until the real keyed terminal, abnormal run exits sweep stale rows, and keyboard/badge semantics distinguish finishing from pending approval. External MCP publication and read-only bindings remain fail-closed.

Independent review completed six rounds, including deliberate mutation checks for approval-scope precedence and the no-call-ID fallback. Final controller verification: 126 Console approval/local-review tests and 193 command/database/domain tests passed; Ruff and diff checks passed. The existing Requests compatibility warning and two pre-existing shutdown-thread `QueueThreadViolation` warnings remain unchanged.

No new generalized lesson was added; the durable execution and transaction ownership decisions are captured in ADR-032 and the implementation plan.
<!-- SECTION:NOTES:END -->
