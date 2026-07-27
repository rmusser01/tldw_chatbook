---
id: TASK-648
title: Move provider selection to Settings Console sessions and a typed handoff
status: To Do
assignee: []
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 00:10'
labels:
  - architecture
  - state
  - providers
  - console
dependencies:
  - TASK-647
references:
  - backlog/decisions/006-provider-aware-generation-settings.md
  - backlog/decisions/026-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the boot-time root provider cache so persisted defaults, active Console sessions, and away-from-Console provider commands have explicit non-overlapping owners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Settings save updates the durable default without overwriting an active Console session whose provider source is user.
- [ ] #2 An active Chat provider command changes that exact session, while an away-from-Console command stages a typed, memory-only, single-slot intent with revisioned claim, acknowledge, and release behavior.
- [ ] #3 Show-current resolves the active session or persisted default; invalid providers terminate the intent and transient readiness failures release it for retry.
- [ ] #4 The root provider descriptor, watcher, and legacy model-select path are removed, and provider resolution accepts explicit inputs without an application surrogate.
- [ ] #5 Focused protocol, privacy, static, formatting, compile, and normal production TldwCli integration checks pass.
<!-- AC:END -->
