---
id: TASK-24206
title: Add lossless chunk-row encoding for streamed trace events
status: Done
assignee:
  - '@codex'
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
  - Docs/superpowers/specs/2026-09-02-console-trace-chunk-row-encoding-design.md
  - Docs/superpowers/plans/2026-09-02-console-trace-chunk-row-encoding.md
  - backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an optional physical encoding for future token-level Console trace events that packs consecutive compatible stream-delta events into compact storage rows while preserving the canonical logical event sequence exactly. This keeps token-level replay and diagnostics feasible without creating one database row and repeated envelope fields per streamed chunk. The encoding is a persistence optimization only: callers and the trace viewer continue to consume ordinary canonical events. This is follow-up work after the reference-backed semantic trace ledger defined by ADR-097; it is not part of that implementation program.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Encoding followed by decoding reproduces every canonical event exactly including type payload sequence timestamp turn call and block identity
- [x] #2 Only consecutive compatible same-kind delta runs are packed and incompatible mixed or structural events preserve their original boundaries and order
- [x] #3 Packing across persistence flush boundaries remains correct without requiring callers to coordinate batch shapes
- [x] #4 Malformed or unsupported packed rows fail closed with a specific corruption or format diagnostic rather than yielding partial history
- [x] #5 Existing unencoded traces remain readable and the physical encoding does not create a second logical event model
- [x] #6 Targeted tests cover text reasoning and tool-call deltas Unicode empty chunks timestamp gaps interrupted streams short runs and mixed event kinds
- [x] #7 A representative streaming benchmark reports database bytes row count encode cost and decode cost before and after the feature
<!-- AC:END -->

## Renumbering provenance

This task was originally filed as `TASK-23112`. During the final rebase for
PR #2200, add-commit provenance showed that the boot-import-closure repair
reached `dev` first in `d4bd5ff91e`; this later semantic-trace follow-up arrived
in `7cf89de6c0`. Under the TASK-19601 older-arrival owner rule, the boot repair
keeps `TASK-23112` and this task moves to `TASK-24206`.

## Implementation Plan

1. Define one strict versioned physical-row codec beneath the canonical trace-event boundary; exact stream-delta shapes may pack, while structural, unknown, or extended events remain verbatim.
2. Implement lossless text, reasoning, and tool-call delta run packing plus fail-closed row validation and exact decoding.
3. Prove batch-boundary independence, Unicode and empty payloads, timestamp gaps, interrupted streams, short runs, mixed event kinds, and corrupt/unsupported row diagnostics with focused tests.
4. Add a representative SQLite benchmark that reports logical/physical rows, allocated bytes, and encode/decode cost for encoded and unencoded storage.
5. Document the format, benchmark result, and integration boundary; run focused tests, lint, compilation, and diff validation before closeout.

ADR required: no
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: ADR-097 already defines chunk-row encoding as a physical codec rather than a parallel event vocabulary. This task adds no database schema, migration, ownership, privacy, or runtime-boundary decision.

## Implementation Notes

- Added a strict, stateless v1 physical codec for consecutive text, reasoning,
  and tool-call stream deltas. Exact canonical events decode unchanged; short,
  mixed, extended, and existing unencoded events remain verbatim.
- Claimed packed rows validate exact shape, arity, identity, and signed-64-bit
  arithmetic. Unsupported versions and corrupt rows fail with distinct
  diagnostics before any partial history is returned.
- Added focused codec and SQLite benchmark coverage plus user/design
  documentation. The 6,000-event reference run reduced 6,000 rows to 3,
  JSON bytes from 1,564,655 to 182,547, and allocated SQLite bytes from
  1,724,416 to 192,512; encode/decode timings are reported but not gated.
- Verification: 23 focused codec/benchmark checks and 113 codec/repository/
  service checks passed; Ruff, Python compilation, and whitespace validation
  passed. The full repository suite was not run per the targeted-test policy.
- ADR required: no. ADR-097 already owns the physical-codec boundary; no
  schema, migration, ownership, privacy, or runtime boundary changed.
