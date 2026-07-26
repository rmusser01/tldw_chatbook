---
id: TASK-645
title: Move Chat and Console handoffs behind revisioned single-slot ownership
status: To Do
assignee:
  - '@codex'
created_date: '2026-07-26 13:37'
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
- [ ] #1 A PendingHandoffStore owns typed Chat and Console channels, normalizes and structurally detaches staged values including nested mappings, and remains memory-only
- [ ] #2 A claim is exclusive, and acknowledge or release affects only the exact claimed revision so a newer replacement cannot be cleared
- [ ] #3 The pending_chat_handoff, pending_console_launch, and pending_console_prompt_insert application fields are migrated to the owner
- [ ] #4 Success, terminal rejection, and transient failure semantics preserve current mount, setup, tab-creation, and retry behavior
- [ ] #5 Deterministic concurrency, privacy-redaction, mounted-flow, static, and ownership-guard tests pass
<!-- AC:END -->
