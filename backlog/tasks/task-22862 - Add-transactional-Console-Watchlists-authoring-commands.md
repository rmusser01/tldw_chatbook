---
id: TASK-22862
title: Add transactional Console Watchlists authoring commands
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 19:29'
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
- [ ] #1 `watchlists_create_sources` validates 1–50 rows before writing, rejects URL userinfo and unsafe values, preserves input order, and reports `created`, `existing`, or `invalid` with canonical IDs without echoing queries/fragments.
- [ ] #2 Exact configured-source identity is outer-whitespace-trimmed only, and a database-owner write-intent batch prevents Console, UI, or OPML callers from racing duplicate lookup/insert.
- [ ] #3 Mixed source outcomes return `partial_success` plus `follow_on_confirmation_required`; no dependent collection mutation occurs until the user explicitly confirms the reduced source set.
- [ ] #4 `watchlists_create_collection` implements explicit `conflict`, `return_existing`, and `auto_suffix` policies; returning an existing collection never mutates it.
- [ ] #5 New collection creation and up to 100 validated memberships commit atomically and do not implicitly schedule, check, or generate a briefing.
- [ ] #6 `watchlists_update_collection_sources` rejects overlapping add/remove sets and missing/ambiguous IDs, then applies all validated membership changes or none.
- [ ] #7 All three commands are Console-only, carry mutation approval effects/tags and definitive-after-start execution ownership, reject server mode before storage access, and have concurrency/rollback/redaction/provider-schema/runtime coverage; after approved execution begins, timeout or cancellation cannot return while a mutation can still commit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add race and validation RED tests, then implement one BEGIN IMMEDIATE database-owner exact-source batch used by Console, direct UI, and OPML.
2. Add collision-policy and atomic-membership RED tests, then implement explicit conflict/return-existing/auto-suffix creation and all-or-nothing membership updates.
3. Build the synchronous WatchlistsCommandService with exact schemas, direct Console-worker calls into application-owned synchronous mutation seams, in-transaction result materialization, definitive commit/rollback outcomes, redaction, server-mode refusal, and no implicit follow-on work.
4. Register the three mutation descriptors as Console-only with code-owned mutation effects/tags, definitive-after-start execution ownership, and sanitized destination presentation; carry the policy through the catalog/runtime without name lists, prove pre-start cancellation remains interruptible, and prove read-only bindings, external MCP, and ordinary bounded tools retain their respective contracts.
5. Run complete task-targeted tests, Ruff, diff checks, self-review, and independent review.

ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: ADR-032 and its approved addendum define the Console-only mutation and approval boundary. Its TASK-22862 execution-ownership amendment records why short SQLite mutations return definitively from the Console tool worker instead of crossing a non-cancellable timeout bridge; no duplicate ADR is required.
<!-- SECTION:PLAN:END -->
