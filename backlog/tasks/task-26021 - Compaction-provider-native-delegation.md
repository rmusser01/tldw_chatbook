---
id: TASK-26021
title: 'Compaction: provider-native delegation'
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
labels:
  - console
  - context
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Compaction always costs a client-side auxiliary call. Verified on origin/dev: a named grep for native_compaction and server compact across Chat/ returns zero, so every compaction is a full round trip that the client pays for and waits on. Some providers now expose server-side compaction that avoids resending the history. Hermes opts into Codex thread/compact/start and the OpenAI Responses equivalent where available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Where a provider exposes native compaction, it can be used instead of the client-side auxiliary call
- [ ] #2 Provider-native compaction produces a memory record with the same shape, provenance and admission semantics as the local path
- [ ] #3 Providers without native support continue to use the existing path with no behavior change
- [ ] #4 The choice is visible: the user can tell which path produced a given summary
- [ ] #5 Native compaction failure falls back to the local path rather than failing the send
- [ ] #6 Opt-in by config, defaulting to the existing local path
<!-- AC:END -->
