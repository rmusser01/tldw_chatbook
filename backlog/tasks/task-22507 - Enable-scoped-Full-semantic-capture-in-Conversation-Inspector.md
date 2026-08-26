---
id: TASK-22507
title: Enable scoped Full semantic capture in Conversation Inspector
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-26 14:34'
updated_date: '2026-08-26 14:34'
labels:
  - console
  - privacy
  - ui
  - db
  - transparency
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-26-console-full-semantic-capture-design.md
  - backlog/decisions/089-console-full-semantic-capture-policy.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users deliberately retain complete semantic provider exchanges for one eligible send, one conversation, or all Console conversations so injected context, tool traffic, and provider-specific payload content can be diagnosed without weakening the default privacy boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Conversation Inspector exposes Safe and Full capture detail for the next eligible send, the inspected conversation, and the global default, with deterministic precedence and active-run freezing.
- [ ] #2 Full capture retains semantic provider inputs and outputs, including Anthropic system content, project and workspace instructions, RAG context, tool schemas, tool calls, and tool results, while structured credentials remain excluded and binary data remains bounded stubs.
- [ ] #3 Capture policy changes use scope-appropriate confirmation, visible lifecycle states, immutable inspected-conversation targeting, and fail closed without changing an admitted run.
- [ ] #4 Each persisted exchange records queryable capture detail, historical exchanges remain backward compatible as Safe, and existing capture size and truncation limits remain enforced.
- [ ] #5 Users can delete stored Full captures for one idle conversation without deleting Safe captures, messages, usage, exports, backups, or changing capture policy; deleted in-memory captures cannot be re-persisted.
- [ ] #6 Capture detail and export profile remain distinct, with confirmed Full clipboard and filesystem exports and accurate per-call provenance in the Exchange view.
- [ ] #7 Targeted automated tests cover policy precedence and consumption, provider and injected-context capture, persistence migration and purge, concurrency and ephemeral behavior, export safety, and production-shaped 80x24 keyboard and focus behavior.
- [ ] #8 The governing privacy and storage ADR, user documentation, and implementation notes describe retention, deletion, provider-boundary caveats, and the default Safe behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write and independently review the approved Full semantic capture specification and ADR.
2. Add the smallest capture-policy model and persistence metadata needed for Safe/Full precedence, run admission freezing, and scoped purge.
3. Thread frozen capture detail through the existing provider gateway and exchange-capture pipeline without a parallel trace subsystem.
4. Add the compact Inspector policy flow, per-call provenance, distinct export profiles, and scope-specific confirmations.
5. Add targeted migration, policy, provider, purge, export, concurrency, ephemeral, and production-shaped Textual tests.
6. Update privacy and Console documentation, verify the focused gates, and record implementation notes.

ADR required: yes
ADR path: backlog/decisions/089-console-full-semantic-capture-policy.md
Reason: This task changes persisted privacy metadata, project-instruction capture guarantees, provider/runtime capture boundaries, deletion semantics, and a cross-module UI/storage contract.
<!-- SECTION:PLAN:END -->
