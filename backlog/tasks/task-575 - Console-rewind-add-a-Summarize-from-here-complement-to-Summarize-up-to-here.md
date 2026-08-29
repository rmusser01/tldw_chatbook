---
id: TASK-575
title: 'Console /rewind: add a Summarize-from-here complement to Summarize-up-to-here'
status: In Progress
assignee: []
created_date: '2026-07-25'
updated_date: '2026-08-29 04:04'
labels:
  - console
  - chat
  - rewind
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred v1 scope cut from the `/rewind` menu (SP2, PR #844, decision D2: v1 = Restore + Summarize-up-to-here only). Up-to-here compresses the OLDEST turns and keeps the recent tail verbatim; the complementary gesture — keep the early framing verbatim and compress a recent tangent — has no affordance. Design decision needed first: exact semantics of "from here" (compress from the selected turn through the current leaf into a summary that sits at the boundary), how it composes with an existing up-to-here boundary (two boundaries vs replace), and whether the existing single `(context_summary, summary_boundary_message_id)` conversation-field pair can represent it or a second pair/shape is required. Reuses the SP2 machinery: session-provider summarization through the gateway, editable Internal_Prompts entry, render-derived banner (never a tree node), and the id-anchored leak rule (compaction only when the boundary row is in the payload).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Semantics for Summarize-from-here (including interaction with an existing up-to-here boundary) are decided and written down before implementation
- [ ] #2 The /rewind menu offers the new option and the resulting compaction follows the same leak rule as up-to-here (pre-boundary sends never receive a summary of turns they precede)
- [ ] #3 The summary banner renders derived-only (never a tree node) and survives resume
<!-- AC:END -->

## Design Decision

- Approved design: [Console `/rewind` Summarize-from-here](../../Docs/superpowers/specs/2026-08-28-console-rewind-summarize-from-here-design.md)
- ADR required: yes
- ADR path: [ADR-052](../decisions/052-console-conversation-memory-and-compaction-policy.md), amended 2026-08-28
- Reason: TASK-575 extends the durable memory scope, atomic replacement, provider-context projection, and long-lived `/rewind` UX governed by ADR-052.
- Design review resolution: branch-local select/reset events replace record-global deactivation; exact CAS, legacy-baseline overrides, monotonic event ordering, migration foreign-key auditing, manual-prefix parity, canonical idle progress, and range-to-prefix automatic planning are specified and independently re-reviewed.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: [ADR-052](../decisions/052-console-conversation-memory-and-compaction-policy.md)
Reason: The task changes durable memory scope, branch selection, atomic replacement, provider request projection, and the long-lived `/rewind` UX.

Detailed plan: [Console `/rewind` Summarize-from-here implementation](../../Docs/superpowers/plans/2026-08-28-console-rewind-summarize-from-here-implementation.md)

1. Add the local-only memory-scope and append-mostly branch-selection schema with deterministic backfill, deletion constraints, and migration foreign-key auditing.
2. Implement one typed effective-memory selector and exact repository CAS for select, reset, undo, and reset-all without record-global branch mutation.
3. Unify complete durable-unit grouping and exact one-call manual prefix/range planning with canonical idle progress accounting.
4. Route both manual directions through the existing bounded auxiliary service and project effective prefix/range memory through every preview/direct/agent request path without leaks.
5. Add ordered range-to-prefix automatic compaction plus Context & memory lifecycle controls.
6. Add the `/rewind` choice, exclusive guarded worker flow, derived banner, and restart/branch lifecycle restoration.
7. Run focused migration, repository, planner, provider, controller, and mounted UI verification; complete static/privacy/self-review and record exact evidence before marking the task Done.
<!-- SECTION:PLAN:END -->
