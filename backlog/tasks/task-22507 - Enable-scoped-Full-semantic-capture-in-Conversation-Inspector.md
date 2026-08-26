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
Let users deliberately retain complete semantic provider exchanges for one eligible send, one conversation, or all Console conversations from the Inspector or live Trace screen so injected context, tool traffic, and provider-specific payload content can be diagnosed without weakening the default privacy boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Conversation Inspector and live Trace screen expose one shared Safe/Full capture flow for the next eligible send, the inspected conversation, and the global default, with deterministic precedence and active-run freezing; imported Traces remain read-only.
- [ ] #2 Full capture retains semantic provider inputs and outputs, including Anthropic system content, project and workspace instructions, RAG context, tool schemas, tool calls, and tool results, while structured credentials remain excluded, request/response binary data becomes bounded stubs, and in-memory/compressed/decompression limits remain enforced.
- [ ] #3 Capture policy changes use scope-appropriate confirmation, visible lifecycle and Capture Off/resume states, immutable inspected-conversation targeting, honest partial-write recovery, and fail closed without changing an admitted run.
- [ ] #4 Each persisted exchange records consistent queryable capture detail, historical exchanges remain backward compatible as Safe, and corrupt provenance mismatches fail closed.
- [ ] #5 Users can delete stored Full captures across every branch and soft-deleted message of one quiescent conversation without deleting Safe captures, messages, usage, exports, backups, or changing capture policy; deleted captures cannot be re-persisted or exported from a stale Inspector.
- [ ] #6 Capture detail and export profile remain distinct, with confirmed Full clipboard and filesystem exports and accurate per-call provenance in the Exchange view.
- [ ] #7 Targeted automated tests cover policy precedence and consumption, provider and injected-context capture, persistence migration and purge, concurrency and ephemeral behavior, export safety, and production-shaped 80x24 keyboard and focus behavior.
- [ ] #8 The governing privacy and storage ADR, user documentation, and implementation notes describe retention, compression-not-encryption, logical deletion and WAL/free-page limits, provider-boundary caveats, and the default Safe behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write and independently review the approved Full semantic capture specification and ADR.
2. During implementation planning, create atomic Backlog children for capture/persistence, runtime/provider threading, scoped purge, and shared Inspector/Trace/Settings UX; do not expose Full before its privacy dependencies exist.
3. Add the smallest capture-policy model and persistence metadata needed for Safe/Full precedence, run admission freezing, and scoped purge.
4. Thread frozen capture detail through the existing provider gateway and exchange-capture pipeline without a parallel trace subsystem.
5. Add the shared policy flow, per-call provenance, distinct export profiles, scope-specific confirmations, and targeted migration/provider/purge/export/Textual tests.
6. Update privacy and Console documentation, verify the focused gates, and record implementation notes.

ADR required: yes
ADR path: backlog/decisions/089-console-full-semantic-capture-policy.md
Reason: This task changes persisted privacy metadata, project-instruction capture guarantees, provider/runtime capture boundaries, deletion semantics, and a cross-module UI/storage contract.
<!-- SECTION:PLAN:END -->
