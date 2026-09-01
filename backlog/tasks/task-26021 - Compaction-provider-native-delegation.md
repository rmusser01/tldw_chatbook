---
id: TASK-26021
title: 'Compaction: provider-native delegation'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:45'
updated_date: '2026-09-01 17:27'
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
- [x] #1 Where a provider exposes native compaction, it can be used instead of the client-side auxiliary call
- [x] #2 Provider-native compaction produces a memory record with the same shape, provenance and admission semantics as the local path
- [x] #3 Providers without native support continue to use the existing path with no behavior change
- [x] #4 The choice is visible: the user can tell which path produced a given summary
- [x] #5 Native compaction failure falls back to the local path rather than failing the send
- [x] #6 Opt-in by config, defaulting to the existing local path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: 4 delegation tests (off-default, native commit path+marker, failure fallback, incapable provider)\n2. _summary_completion helper: probe gateway capability, native try under the 26016 timeout, ANY failure falls back local; returns (completion, engine)\n3. Both compact()/summarize_manual() call sites use the helper; records gain a compaction_engine:provider_native marker only when native produced the committed summary\n4. Config [console] compaction_native_delegation (default false, fail-closed is-True coercion)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Honest scope: NO provider chatbook bridges exposes server-side compaction today (hermes uses the Codex-backend thread/compact/start, not a public endpoint), so the deliverable is the fully-tested delegation SEAM the gateway flips on later: ConsoleCompactionService._summary_completion probes duck-typed gateway capabilities (supports_native_compaction(resolution) + complete_native_compaction(request)), tries native under the same 26016 timeout, and ANY native failure — timeout included — falls back to the local auxiliary call (AC#5); only outer cancellation propagates. Native replaces ONLY the completion step: validation (envelope/output-cap/projection), admission fences, and the record write stay on the existing path (AC#2), and the committed record's selected_units_json gains {'kind':'compaction_engine','engine':'provider_native'} only when native actually produced the summary — a fallback commit never claims it (AC#4, pinned). Both compact() and summarize_manual() route through the helper. Config [console] compaction_native_delegation, default false, coerced 'is True' fail-closed; with no capable gateway the ON state is behavior-identical (AC#3). 4 new tests via a fake capability-advertising gateway; compaction suite 142 passed. When a real provider capability lands, the gateway implements the two methods and this task's tests already pin the contract.
<!-- SECTION:NOTES:END -->
