---
id: TASK-22507
title: Enable scoped Full semantic capture in Conversation Inspector
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-26 14:34'
updated_date: '2026-08-26 15:33'
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
1. Treat Docs/superpowers/specs/2026-08-26-console-full-semantic-capture-design.md and ADR-089 as the approved contract.
2. Execute TASK-22507.1: add Safe-first capture construction, bounds, provenance, schema migration, and local policy persistence without exposing Full in the UI.
3. Execute TASK-22507.2 after TASK-22507.1: resolve and consume scoped policy at admission, freeze it on provider signals, and cover direct/retry/tool/fleet/Anthropic/llama.cpp paths.
4. Execute TASK-22507.3 after TASK-22507.1 and TASK-22507.2: add conversation-wide Full-capture count/purge under quiescence with staged cache replacement and capture-revision fences.
5. Execute TASK-22507.4 after TASK-22507.1, TASK-22507.2, and TASK-22507.3: expose the shared Inspector/live Trace/F9 flow, governed per-call export, responsive styling, documentation, and production-shaped verification.
6. Follow Docs/superpowers/plans/2026-08-26-console-full-semantic-capture.md task-by-task, close each child only with its focused evidence, then run the final integration gate and close this parent.

ADR required: yes
ADR path: backlog/decisions/089-console-full-semantic-capture-policy.md
Reason: ADR-089 governs the persisted privacy metadata, provider/runtime capture boundary, logical deletion semantics, and shared UI/storage contract.
<!-- SECTION:PLAN:END -->
