---
id: TASK-23112
title: Add lossless chunk-row encoding for streamed trace events
status: To Do
assignee: []
created_date: '2026-08-28 14:38'
labels:
  - console
  - storage
  - performance
  - tracing
dependencies: []
references:
  - >-
    https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/session/src/chunk-rows.ts
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an optional physical encoding for future token-level Console trace events that packs consecutive compatible stream-delta events into compact storage rows while preserving the canonical logical event sequence exactly. This keeps token-level replay and diagnostics feasible without creating one database row and repeated envelope fields per streamed chunk. The encoding is a persistence optimization only: callers and the trace viewer continue to consume ordinary canonical events. This is follow-up work after the reference-backed semantic trace ledger from TASK-23026; it is not required to close that task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Encoding followed by decoding reproduces every canonical event exactly including type payload sequence timestamp turn call and block identity
- [ ] #2 Only consecutive compatible same-kind delta runs are packed and incompatible mixed or structural events preserve their original boundaries and order
- [ ] #3 Packing across persistence flush boundaries remains correct without requiring callers to coordinate batch shapes
- [ ] #4 Malformed or unsupported packed rows fail closed with a specific corruption or format diagnostic rather than yielding partial history
- [ ] #5 Existing unencoded traces remain readable and the physical encoding does not create a second logical event model
- [ ] #6 Targeted tests cover text reasoning and tool-call deltas Unicode empty chunks timestamp gaps interrupted streams short runs and mixed event kinds
- [ ] #7 A representative streaming benchmark reports database bytes row count encode cost and decode cost before and after the feature
<!-- AC:END -->
